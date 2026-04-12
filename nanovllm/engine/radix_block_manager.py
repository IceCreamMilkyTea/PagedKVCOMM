"""Radix tree-based block manager for KV cache with KVCOMM near-miss support."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


class RadixTreeNode:
    __slots__ = (
        "children",
        "parent",
        "token_ids",
        "block_ids",
        "num_tokens",
        "ref_count",
        "last_access",
        "context_key",
    )

    _access_counter = 0

    def __init__(self):
        self.children: Dict[int, RadixTreeNode] = {}
        self.parent: Optional[RadixTreeNode] = None
        self.token_ids: List[int] = []
        self.block_ids: List[int] = []
        self.num_tokens: int = 0
        self.ref_count: int = 0
        self.context_key: Optional[str] = None
        RadixTreeNode._access_counter += 1
        self.last_access: int = RadixTreeNode._access_counter

    def touch(self):
        RadixTreeNode._access_counter += 1
        self.last_access = RadixTreeNode._access_counter


class RadixTree:
    def __init__(self, block_size: int):
        self.root = RadixTreeNode()
        self.block_size = block_size
        self._node_count = 0

    def snapshot(self, token_preview: int = 16) -> Dict[str, object]:
        nodes: List[Dict[str, object]] = []
        leaf_count = 0
        next_id = 0

        def visit(node: RadixTreeNode, parent_id: Optional[int], depth: int) -> int:
            nonlocal next_id, leaf_count
            node_id = next_id
            next_id += 1
            if not node.children:
                leaf_count += 1
            nodes.append(
                {
                    "node_id": node_id,
                    "parent_id": parent_id,
                    "depth": depth,
                    "token_len": len(node.token_ids),
                    "num_tokens": int(node.num_tokens),
                    "block_ids_len": len(node.block_ids),
                    "block_ids_preview": list(node.block_ids[:token_preview]),
                    "token_preview": list(node.token_ids[:token_preview]),
                    "ref_count": int(node.ref_count),
                    "last_access": int(node.last_access),
                    "context_key": node.context_key,
                    "child_first_tokens": sorted(
                        int(tok) for tok in node.children.keys()
                    ),
                }
            )
            for tok in sorted(node.children.keys()):
                visit(node.children[tok], node_id, depth + 1)
            return node_id

        visit(self.root, None, 0)
        return {
            "block_size": int(self.block_size),
            "node_count": len(nodes),
            "tracked_node_count": int(self._node_count),
            "leaf_count": int(leaf_count),
            "nodes": nodes,
        }

    def insert(
        self,
        token_ids: List[int],
        block_ids: List[int],
        context_key: Optional[str] = None,
    ) -> RadixTreeNode:
        node = self.root
        i = 0
        while i < len(token_ids):
            tok = token_ids[i]
            if tok in node.children:
                child = node.children[tok]
                j = 0
                while j < len(child.token_ids) and i < len(token_ids):
                    if child.token_ids[j] != token_ids[i]:
                        self._split_node(child, j)
                        break
                    j += 1
                    i += 1
                if j == len(child.token_ids):
                    node = child
                    node.touch()
                    continue
                remaining = token_ids[i:]
                start_block = i // self.block_size
                new_child = RadixTreeNode()
                new_child.token_ids = remaining
                new_child.block_ids = block_ids[start_block:]
                new_child.num_tokens = len(remaining)
                new_child.parent = child
                new_child.ref_count = 1
                new_child.context_key = context_key
                child.children[remaining[0]] = new_child
                self._node_count += 1
                return new_child
            remaining = token_ids[i:]
            start_block = i // self.block_size
            new_child = RadixTreeNode()
            new_child.token_ids = remaining
            new_child.block_ids = block_ids[start_block:]
            new_child.num_tokens = len(remaining)
            new_child.parent = node
            new_child.ref_count = 1
            new_child.context_key = context_key
            node.children[remaining[0]] = new_child
            self._node_count += 1
            return new_child
        node.touch()
        node.ref_count += 1
        if context_key:
            node.context_key = context_key
        return node

    def _split_node(self, node: RadixTreeNode, pos: int):
        new_parent = RadixTreeNode()
        new_parent.token_ids = node.token_ids[:pos]
        new_parent.block_ids = node.block_ids[
            : ((pos + self.block_size - 1) // self.block_size)
        ]
        new_parent.num_tokens = pos
        new_parent.parent = node.parent
        new_parent.ref_count = node.ref_count
        new_parent.context_key = node.context_key

        node.token_ids = node.token_ids[pos:]
        remaining_start = pos // self.block_size
        node.block_ids = node.block_ids[remaining_start:]
        node.num_tokens = len(node.token_ids)

        if node.parent:
            first_tok = new_parent.token_ids[0]
            node.parent.children[first_tok] = new_parent
        node.parent = new_parent
        if node.token_ids:
            new_parent.children[node.token_ids[0]] = node
        self._node_count += 1

    def match_prefix(self, token_ids: List[int]) -> Tuple[int, List[int]]:
        node = self.root
        matched = 0
        matched_blocks = []
        i = 0

        while i < len(token_ids):
            tok = token_ids[i]
            if tok not in node.children:
                break
            child = node.children[tok]
            j = 0
            while j < len(child.token_ids) and i < len(token_ids):
                if child.token_ids[j] != token_ids[i]:
                    full_blocks = j // self.block_size
                    matched += full_blocks * self.block_size
                    matched_blocks.extend(child.block_ids[:full_blocks])
                    return matched, matched_blocks
                j += 1
                i += 1
            if j == len(child.token_ids):
                matched += child.num_tokens
                matched_blocks.extend(child.block_ids)
                node = child
                node.touch()
            else:
                break

        return matched, matched_blocks

    def match_near_miss(
        self,
        token_ids: List[int],
        context_key: str,
        min_match_ratio: float = 0.5,
    ) -> Optional[Tuple[RadixTreeNode, int, List[int], str]]:
        matched_len, _ = self.match_prefix(token_ids)
        if matched_len == len(token_ids):
            return None

        candidates = []
        self._collect_near_miss_candidates(
            self.root, token_ids, 0, [], candidates, context_key
        )
        if not candidates:
            return None

        node, match_len, blocks, orig_ctx = max(candidates, key=lambda item: item[1])
        if match_len < len(token_ids) * min_match_ratio:
            return None
        return node, match_len, blocks, orig_ctx

    def _collect_near_miss_candidates(
        self,
        node: RadixTreeNode,
        token_ids: List[int],
        pos: int,
        acc_blocks: List[int],
        candidates: list,
        target_context: str,
        max_candidates: int = 5,
    ):
        if len(candidates) >= max_candidates:
            return

        for _, child in node.children.items():
            if pos >= len(token_ids):
                break
            j = 0
            new_pos = pos
            matching = True
            while j < len(child.token_ids) and new_pos < len(token_ids):
                if child.token_ids[j] != token_ids[new_pos]:
                    matching = False
                    break
                j += 1
                new_pos += 1

            if matching and j == len(child.token_ids):
                new_blocks = acc_blocks + child.block_ids
                if child.context_key and child.context_key != target_context:
                    candidates.append(
                        (child, new_pos, new_blocks, child.context_key)
                    )
                self._collect_near_miss_candidates(
                    child,
                    token_ids,
                    new_pos,
                    new_blocks,
                    candidates,
                    target_context,
                    max_candidates,
                )

    def evict_lru(self, num_blocks_needed: int) -> List[int]:
        freed = []
        leaves = self._get_evictable_leaves()

        while len(freed) < num_blocks_needed and leaves:
            leaf = min(leaves, key=lambda node: node.last_access)
            leaves.remove(leaf)
            freed.extend(leaf.block_ids)

            if leaf.parent:
                to_remove = None
                for key, value in leaf.parent.children.items():
                    if value is leaf:
                        to_remove = key
                        break
                if to_remove is not None:
                    del leaf.parent.children[to_remove]
                self._node_count -= 1

                parent = leaf.parent
                if len(parent.children) == 1 and parent.ref_count == 0 and parent.parent:
                    self._merge_with_child(parent)

        return freed

    def _get_evictable_leaves(self) -> List[RadixTreeNode]:
        leaves = []
        self._collect_leaves(self.root, leaves)
        return [node for node in leaves if node.ref_count == 0]

    def _collect_leaves(self, node: RadixTreeNode, result: list):
        if not node.children:
            if node is not self.root:
                result.append(node)
        else:
            for child in node.children.values():
                self._collect_leaves(child, result)

    def _merge_with_child(self, node: RadixTreeNode):
        if len(node.children) != 1:
            return
        child = next(iter(node.children.values()))
        child.token_ids = node.token_ids + child.token_ids
        child.block_ids = node.block_ids + child.block_ids
        child.num_tokens = len(child.token_ids)
        child.parent = node.parent
        if node.parent:
            for key, value in node.parent.children.items():
                if value is node:
                    node.parent.children[key] = child
                    break
        self._node_count -= 1


class RadixBlockManager(BlockManager):
    """BlockManager extended with radix tree-based prefix caching."""

    def __init__(self, num_blocks: int, block_size: int):
        super().__init__(num_blocks, block_size)
        self.radix_tree = RadixTree(block_size)
        self._near_miss_enabled = True

    def get_radix_tree_snapshot(self, token_preview: int = 16) -> Dict[str, object]:
        tree_snapshot = self.radix_tree.snapshot(token_preview=token_preview)
        tree_snapshot.update(
            {
                "free_block_count": len(self.free_block_ids),
                "used_block_count": len(self.used_block_ids),
                "hash_entry_count": len(self.hash_to_block_id),
            }
        )
        return tree_snapshot

    def allocate(
        self,
        seq: Sequence,
        prefilled_block_table: Optional[List[int]] = None,
        prefilled_num_cached: int = 0,
    ):
        if prefilled_block_table is not None:
            assert not seq.block_table, "Sequence already has a block table"
            for bid in prefilled_block_table:
                block = self.blocks[bid]
                block.ref_count += 1
                if bid in self.free_block_ids:
                    self.free_block_ids.remove(bid)
                    self.used_block_ids.add(bid)
                seq.block_table.append(bid)
            seq.num_cached_tokens = prefilled_num_cached

            h = -1
            for i, bid in enumerate(seq.block_table):
                token_ids_block = seq.block(i) if i < seq.num_blocks else []
                if len(token_ids_block) == self.block_size:
                    h = self.compute_hash(token_ids_block, h)
                    block = self.blocks[bid]
                    block.update(h, token_ids_block)
                    self.hash_to_block_id[h] = bid
                else:
                    h = -1

            remaining_blocks_needed = seq.num_blocks - len(seq.block_table)
            for _ in range(remaining_blocks_needed):
                if not self.free_block_ids:
                    raise RuntimeError(
                        "Not enough free blocks for remaining allocation"
                    )
                bid = self.free_block_ids[0]
                self._allocate_block(bid)
                seq.block_table.append(bid)
            return

        token_ids = seq.token_ids[: seq.num_prompt_tokens]
        matched_len, matched_blocks = self.radix_tree.match_prefix(token_ids)

        if matched_len > 0 and matched_blocks:
            page_aligned_len = (matched_len // self.block_size) * self.block_size
            cache_hit_blocks = page_aligned_len // self.block_size
            cache_hit_blocks = min(
                cache_hit_blocks,
                len(matched_blocks),
                seq.num_blocks,
            )

            for i in range(cache_hit_blocks):
                bid = matched_blocks[i]
                block = self.blocks[bid]
                if bid in self.used_block_ids:
                    block.ref_count += 1
                elif bid in self.free_block_ids:
                    self._allocate_block(bid)
                else:
                    block.ref_count += 1
                    self.used_block_ids.add(bid)
                seq.block_table.append(bid)
            seq.num_cached_tokens = cache_hit_blocks * self.block_size

            h = -1
            for i in range(cache_hit_blocks):
                token_ids_block = seq.block(i)
                if len(token_ids_block) == self.block_size:
                    h = self.compute_hash(token_ids_block, h)
                    block = self.blocks[seq.block_table[i]]
                    if block.hash == -1:
                        block.update(h, token_ids_block)
                        self.hash_to_block_id[h] = seq.block_table[i]
                else:
                    h = -1

            for _ in range(cache_hit_blocks, seq.num_blocks):
                if not self.free_block_ids:
                    raise RuntimeError("Not enough free blocks")
                bid = self.free_block_ids[0]
                self._allocate_block(bid)
                seq.block_table.append(bid)
        else:
            assert not seq.block_table
            h = -1
            cache_miss = False
            for i in range(seq.num_blocks):
                token_ids_block = seq.block(i)
                h = (
                    self.compute_hash(token_ids_block, h)
                    if len(token_ids_block) == self.block_size
                    else -1
                )
                block_id = self.hash_to_block_id.get(h, -1)
                if block_id == -1 or self.blocks[block_id].token_ids != token_ids_block:
                    cache_miss = True
                if cache_miss:
                    block_id = self.free_block_ids[0]
                    block = self._allocate_block(block_id)
                else:
                    seq.num_cached_tokens += self.block_size
                    if block_id in self.used_block_ids:
                        block = self.blocks[block_id]
                        block.ref_count += 1
                    else:
                        block = self._allocate_block(block_id)
                if h != -1:
                    block.update(h, token_ids_block)
                    self.hash_to_block_id[h] = block_id
                seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        if seq.block_table and seq.num_prompt_tokens > 0:
            prompt_blocks = (
                seq.num_prompt_tokens + self.block_size - 1
            ) // self.block_size
            prompt_block_ids = list(seq.block_table[:prompt_blocks])
            prompt_tokens = list(seq.token_ids[: seq.num_prompt_tokens])
            self.radix_tree.insert(prompt_tokens, prompt_block_ids)

        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]

        if len(seq) % self.block_size == 1:
            if last_block.hash == -1 and last_block.token_ids:
                prefix = (
                    self.blocks[block_table[-2]].hash
                    if len(block_table) > 1
                    else -1
                )
                token_ids = seq.block(seq.num_blocks - 2)
                h = self.compute_hash(token_ids, prefix)
                last_block.update(h, token_ids)
                self.hash_to_block_id[h] = last_block.block_id
            elif last_block.hash == -1:
                blk_idx = len(block_table) - 1
                if blk_idx > 0:
                    token_ids = seq.block(blk_idx)
                    if len(token_ids) == self.block_size:
                        prefix = (
                            self.blocks[block_table[blk_idx - 1]].hash
                            if blk_idx > 0
                            else -1
                        )
                        h = self.compute_hash(token_ids, prefix)
                        last_block.update(h, token_ids)
                        self.hash_to_block_id[h] = last_block.block_id

            if not self.free_block_ids:
                raise RuntimeError("No free blocks for append")
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)

        elif len(seq) % self.block_size == 0:
            if last_block.hash == -1:
                token_ids = seq.block(seq.num_blocks - 1)
                prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                h = self.compute_hash(token_ids, prefix)
                last_block.update(h, token_ids)
                self.hash_to_block_id[h] = last_block.block_id

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def try_near_miss_match(
        self,
        token_ids: List[int],
        context_key: str,
        min_match_ratio: float = 0.5,
    ) -> Optional[Tuple[int, List[int], str]]:
        if not self._near_miss_enabled:
            return None
        result = self.radix_tree.match_near_miss(
            token_ids,
            context_key,
            min_match_ratio,
        )
        if result is None:
            return None
        _, matched_tokens, block_ids, orig_ctx = result
        return matched_tokens, block_ids, orig_ctx
