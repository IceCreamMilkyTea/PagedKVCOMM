from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    def _sanitize_free_list(self):
        """Remove stale/duplicate entries from free list and repair used set."""
        cleaned: deque[int] = deque()
        seen: set[int] = set()
        for block_id in self.free_block_ids:
            if block_id in seen:
                continue
            seen.add(block_id)
            block = self.blocks[block_id]
            if block.ref_count == 0:
                cleaned.append(block_id)
            else:
                self.used_block_ids.add(block_id)
        self.free_block_ids = cleaned
        self.used_block_ids = {
            block_id
            for block_id in self.used_block_ids
            if self.blocks[block_id].ref_count > 0
        }

    def _pop_free_block_id(self) -> int:
        self._sanitize_free_list()
        if not self.free_block_ids:
            raise RuntimeError("No free KV-cache blocks available after sanitization")
        return self.free_block_ids.popleft()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        if block.ref_count != 0:
            raise RuntimeError(
                f"Attempted to allocate non-free block {block_id} with ref_count={block.ref_count}"
            )
        block.reset()
        if block_id in self.free_block_ids:
            self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.discard(block_id)
        if block_id not in self.free_block_ids:
            self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        self._sanitize_free_list()
        prefilled = len(seq.block_table)
        needed = max(0, seq.num_blocks - prefilled)
        return len(self.free_block_ids) >= needed

    def allocate(self, seq: Sequence):
        prefilled = len(seq.block_table)
        if prefilled > seq.num_blocks:
            raise ValueError(
                f"prefilled blocks exceed sequence blocks: {prefilled}>{seq.num_blocks}"
            )

        # If caller pre-injected prefix blocks, continue allocation from tail.
        h = -1
        if prefilled > 0:
            for i in range(prefilled):
                token_ids = seq.block(i)
                h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1

        cache_miss = prefilled > 0
        for i in range(prefilled, seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self._pop_free_block_id()
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        self._sanitize_free_list()
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            # Pre-injected custom prefix blocks may not be registered in hash map.
            block_id = self._pop_free_block_id()
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # Reused cached blocks may already carry a finalized hash. Only
            # finalize if this block has been treated as mutable (hash == -1).
            if last_block.hash == -1:
                token_ids = seq.block(seq.num_blocks-1)
                prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                h = self.compute_hash(token_ids, prefix)
                last_block.update(h, token_ids)
                self.hash_to_block_id[h] = last_block.block_id
        else:
            # Partial block is being appended to; ensure it is marked mutable.
            if last_block.hash != -1:
                last_block.hash = -1
