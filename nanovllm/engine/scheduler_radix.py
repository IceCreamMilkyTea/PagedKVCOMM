from collections import deque
from typing import List, Optional

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.radix_block_manager import RadixBlockManager


class RadixScheduler:
    """Scheduler using RadixBlockManager for radix tree-based prefix caching."""

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = RadixBlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self._prefilled_tables: dict[int, tuple[list[int], int]] = {}

    def is_finished(self):
        return not self.waiting and not self.running

    def add(
        self,
        seq: Sequence,
        prefilled_block_table: Optional[List[int]] = None,
        prefilled_num_cached: int = 0,
    ):
        if prefilled_block_table is not None:
            self._prefilled_tables[seq.seq_id] = (
                prefilled_block_table,
                prefilled_num_cached,
            )
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if (
                num_batched_tokens + len(seq) > self.max_num_batched_tokens
                or not self.block_manager.can_allocate(seq)
            ):
                break

            prefilled = self._prefilled_tables.pop(seq.seq_id, None)
            if prefilled is not None:
                pf_table, pf_cached = prefilled
                self.block_manager.allocate(
                    seq,
                    prefilled_block_table=pf_table,
                    prefilled_num_cached=pf_cached,
                )
            else:
                self.block_manager.allocate(seq)

            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            uncached_tokens = len(seq) - seq.num_cached_tokens
            if uncached_tokens > 0:
                num_seqs += 1
                num_batched_tokens += uncached_tokens
                scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (
                (not seq.ignore_eos and token_id == self.eos)
                or seq.num_completion_tokens == seq.max_tokens
            ):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
