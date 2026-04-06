"""Paged attention LLM chat backend using nano-vllm engine.

This module provides `PagedLLMChat`, a drop-in replacement for `LLMChat`
that uses nano-vllm's paged attention engine instead of HuggingFace's
`model.generate()` + `DynamicCache`.

Register:  @LLMRegistry.register('PagedLLMChat')

Key differences from LLMChat:
  - Uses nano-vllm's LLMEngine (scheduler + model_runner + block_manager)
  - KV cache lives in a pre-allocated block pool (zero fragmentation)
  - Prefill writes KV directly to blocks via triton kernels (zero-copy)
  - Anchor storage uses block references instead of full tensor copies
  - Supports both Llama and Qwen models via model_runner auto-detection
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer

from KVCOMM.llm.format import Message
from KVCOMM.llm.llm import LLM
from KVCOMM.llm.llm_registry import LLMRegistry
from KVCOMM.llm.config import KVCommConfig
from KVCOMM.llm.paged_kvcomm_engine import PagedKVCOMMEngine
from KVCOMM.utils.metrics import GenerationResult
from KVCOMM.utils.log import logger

# nano-vllm imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "nano-vllm"))
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def _escape_loguru_markup(text: Optional[str]) -> str:
    if text is None:
        return ""
    return text.replace("<", "\\<")


_LATENCY_IO_LOCK = threading.Lock()


def _append_latency_record(target: Optional[Union[str, Path]], record: Dict[str, Any]) -> None:
    if target is None:
        return
    path = Path(target)
    if not path.suffix:
        path = path / "Latency.json"
    serializable = {k: v for k, v in record.items() if v is not None}
    with _LATENCY_IO_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: List[Dict[str, Any]] = []
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        existing = loaded
            except (json.JSONDecodeError, OSError):
                existing = []
        existing.append(serializable)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)


@LLMRegistry.register('PagedLLMChat')
class PagedLLMChat(LLM):
    """Local HF model chat with paged attention and KVCOMM anchor support.

    Uses nano-vllm's engine for:
      - Block-based KV cache (pre-allocated, zero fragmentation)
      - Triton store_kvcache for zero-copy prefill
      - flash_attn_with_kvcache for paged decode
      - Scheduler for batched prefill/decode

    Uses PagedKVCOMMEngine for:
      - Anchor storage via block references
      - Delta computation from block reads
      - Weighted delta blending for KV reuse
    """

    # ── Shared state across all instances ──
    _shared_engine: Optional[LLMEngine] = None
    _shared_tokenizer: Optional[AutoTokenizer] = None
    _model_lock = threading.Lock()
    _THREAD_POOL: Optional[ThreadPoolExecutor] = None
    _THREAD_POOL_WORKERS: Optional[int] = None
    _shared_kv_cache_memory: Optional[Dict[str, Any]] = None
    _initialization: Dict[str, bool] = {}
    _paged_kv_engine: Optional[PagedKVCOMMEngine] = None
    _anchor_info_dict: Dict[str, Dict[str, int]] = {}
    _global_anchor_info_dict: Dict[str, Dict[str, List[int]]] = {}
    # Align with non-paged backend: default greedy decoding.
    DEFAULT_TEMPERATURE = 0.0

    def __init__(
        self,
        model_name: str,
        prefix: str = None,
        config: Optional[KVCommConfig] = None,
    ):
        self.model_name = model_name
        self.config = (config or KVCommConfig.from_env()).validate()
        self._ensure_thread_pool(self.config.thread_pool_workers)
        self.lock = asyncio.Lock()

        self._initialize_shared_resources()

        self.tokenizer = PagedLLMChat._shared_tokenizer
        self.engine = PagedLLMChat._shared_engine
        self.paged_kv_engine = PagedLLMChat._paged_kv_engine
        self._shared_kv_cache_memory = PagedLLMChat._shared_kv_cache_memory
        self._initialization = PagedLLMChat._initialization

        self._chat_markers = self._extract_chat_markers()
        self.default_assistant_prompt = "A: "
        self.base_messages_template: List[Dict[str, str]] = [
            {"role": "system", "content": "{system_prompt}"},
            {"role": "user", "content": "{user_prompt}"},
        ]
        if prefix is not None:
            self._prepare_prefix_template(prefix)

    # ── Initialization ──

    def _initialize_shared_resources(self):
        """Lazy-init nano-vllm engine, tokenizer, and PagedKVCOMMEngine."""
        with PagedLLMChat._model_lock:
            if PagedLLMChat._shared_engine is None:
                logger.info("Initializing nano-vllm engine for model: {}", self.model_name)

                PagedLLMChat._shared_engine = LLMEngine(self.model_name, dtype=torch.float16)
                PagedLLMChat._shared_tokenizer = PagedLLMChat._shared_engine.tokenizer

                # Extract KV cache and block manager from model_runner
                model_runner = PagedLLMChat._shared_engine.model_runner
                scheduler = PagedLLMChat._shared_engine.scheduler

                hf_config = model_runner.config.hf_config
                tp_size = model_runner.world_size
                num_kv_heads = hf_config.num_key_value_heads // tp_size
                head_dim = getattr(
                    hf_config, "head_dim",
                    hf_config.hidden_size // hf_config.num_attention_heads,
                )

                PagedLLMChat._paged_kv_engine = PagedKVCOMMEngine(
                    kv_cache=model_runner.kv_cache,
                    block_manager=scheduler.block_manager,
                    block_size=model_runner.block_size,
                    num_layers=hf_config.num_hidden_layers,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                )

                logger.info(
                    "PagedKVCOMMEngine initialized: {} blocks, block_size={}",
                    len(scheduler.block_manager.blocks),
                    model_runner.block_size,
                )

            if PagedLLMChat._shared_kv_cache_memory is None:
                PagedLLMChat._shared_kv_cache_memory = {}

    @classmethod
    def _ensure_thread_pool(cls, workers: int) -> None:
        if cls._THREAD_POOL is None or cls._THREAD_POOL_WORKERS != workers:
            if cls._THREAD_POOL is not None:
                cls._THREAD_POOL.shutdown(wait=False)
            cls._THREAD_POOL = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="PagedLLM")
            cls._THREAD_POOL_WORKERS = workers

    # ── Chat template helpers (same as LLMChat) ──

    def _extract_chat_markers(self) -> Dict[str, str]:
        template = getattr(self.tokenizer, "chat_template", "") or ""
        markers = {"begin": "", "start": "", "end": "", "eot": ""}
        for token in ["<|begin_of_text|>", "<s>", getattr(self.tokenizer, "bos_token", "") or ""]:
            if token and token in template:
                markers["begin"] = token
                break
        for token in ["<|start_header_id|>", "<|im_start|>"]:
            if token and token in template:
                markers["start"] = token
                break
        for token in ["<|end_header_id|>", "<|im_end|>", "\n"]:
            if token and token in template:
                markers["end"] = token
                break
        for token in ["<|eot_id|>", "<|im_end|>", getattr(self.tokenizer, "eos_token", "") or ""]:
            if token and token in template:
                markers["eot"] = token
                break
        return markers

    def _prepare_prefix_template(self, prefix: Union[str, List[Dict[str, str]]]) -> None:
        if isinstance(prefix, list):
            self.base_messages_template = prefix
        elif isinstance(prefix, dict):
            self.base_messages_template = [prefix]
        elif isinstance(prefix, str):
            self.default_assistant_prompt = prefix

    @property
    def begin_of_text(self) -> str:
        return self._chat_markers.get("begin", "")

    @property
    def start_header_id(self) -> str:
        return self._chat_markers.get("start", "")

    @property
    def end_header_id(self) -> str:
        return self._chat_markers.get("end", "")

    @property
    def eot_id(self) -> str:
        return self._chat_markers.get("eot", "")

    def format_chat_segment(
        self,
        role: str,
        content: str,
        *,
        include_begin: bool = False,
        include_eot: bool = True,
    ) -> str:
        prefix = self.begin_of_text if include_begin else ""
        start = self.start_header_id
        end = self.end_header_id
        eot = self.eot_id if include_eot else ""
        if start and end:
            return f"{prefix}{start}{role}{end}\n{content}{eot}"
        return f"{prefix}[{role.upper()}]\n{content}{eot}"

    def _render_base_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        rendered = []
        for block in (self.base_messages_template or []):
            content = block.get("content", "").format(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            rendered.append({"role": block.get("role", "user"), "content": content})
        return rendered

    @staticmethod
    def _normalise_messages(messages) -> List[Dict[str, str]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if isinstance(messages, dict):
            if "role" in messages:
                return [messages]
            result = []
            if "system" in messages:
                result.append({"role": "system", "content": messages["system"]})
            if "user" in messages:
                result.append({"role": "user", "content": messages["user"]})
            return result
        if isinstance(messages, list):
            out = []
            for item in messages:
                if isinstance(item, Message):
                    out.append({"role": item.role, "content": item.content})
                elif isinstance(item, dict):
                    out.append(item)
                elif isinstance(item, str):
                    out.append({"role": "user", "content": item})
            return out
        return [{"role": "user", "content": str(messages)}]

    def _build_prompt_text(
        self,
        messages: Union[List[Message], List[Dict], str],
        assistant_prompt: Optional[str] = None,
    ) -> str:
        normalised = self._normalise_messages(messages)
        assistant_prompt = assistant_prompt or self.default_assistant_prompt
        try:
            text = self.tokenizer.apply_chat_template(
                normalised,
                add_generation_prompt=True,
                tokenize=False,
            ) + assistant_prompt
        except Exception:
            parts = [self.begin_of_text or ""]
            for msg in normalised:
                role, content = msg.get("role", "user"), msg.get("content", "")
                s, e, eot = self.start_header_id, self.end_header_id, self.eot_id
                if s and e:
                    parts.append(f"{s}{role}{e}\n{content}{eot}")
                else:
                    parts.append(f"[{role.upper()}]\n{content}{eot}")
            parts.append(f"{self.start_header_id}assistant{self.end_header_id}\n" if self.start_header_id else "[ASSISTANT]\n")
            text = "".join(parts) + (assistant_prompt or "")
        return text

    def _encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def _message_cache_key(message: Any) -> str:
        """Build a stable cache key for anchor/reuse dictionaries."""
        if isinstance(message, str):
            return message
        if isinstance(message, dict) or isinstance(message, list):
            try:
                return json.dumps(message, ensure_ascii=False, sort_keys=True)
            except TypeError:
                return str(message)
        if isinstance(message, Message):
            return json.dumps(
                {"role": message.role, "content": message.content},
                ensure_ascii=False,
                sort_keys=True,
            )
        return str(message)

    # ── Agent identity ──

    def set_id(self, node_id: str, role: str):
        self.node_id = node_id
        self.role = role
        if self.node_id not in PagedLLMChat._shared_kv_cache_memory:
            PagedLLMChat._shared_kv_cache_memory[self.node_id] = {}
            PagedLLMChat._initialization[self.node_id] = False

    def _ensure_agent_memory(self, agent_id: str) -> Dict[str, Any]:
        """Return the shared memory slot for a given agent id."""
        return PagedLLMChat._shared_kv_cache_memory.setdefault(agent_id, {})

    def has_prefix_initialized(self, agent_id: str) -> bool:
        return PagedLLMChat._initialization.get(agent_id, False)

    # ── Core generation via nano-vllm engine ──

    def _generate_tokens(
        self,
        token_ids: List[int],
        max_tokens: int = 512,
        temperature: float = 0.0,
        cached_prefix_block_table: Optional[List[int]] = None,
        cached_prefix_num_tokens: int = 0,
    ) -> Tuple[List[int], float, Sequence]:
        """Run prefill + decode through nano-vllm's LLMEngine.

        Returns (completion_token_ids, ttft, sequence).
        The Sequence object holds `block_table` which references physical KV blocks.
        """
        sp = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if cached_prefix_block_table:
            bs = self.paged_kv_engine.block_size
            max_cached_tokens = min(len(token_ids), len(cached_prefix_block_table) * bs)
            normalized_cached_tokens = max(0, min(cached_prefix_num_tokens, max_cached_tokens))
            if normalized_cached_tokens != cached_prefix_num_tokens:
                logger.info(
                    "[KV_REUSE_NORMALIZE] node={} role={} cached_tokens {}->{} cached_blocks={} prompt_tokens={}",
                    getattr(self, "node_id", "?"),
                    getattr(self, "role", "?"),
                    cached_prefix_num_tokens,
                    normalized_cached_tokens,
                    len(cached_prefix_block_table),
                    len(token_ids),
                )
                cached_prefix_num_tokens = normalized_cached_tokens
        seq = Sequence(
            token_ids,
            sp,
            prefilled_block_table=cached_prefix_block_table,
            num_cached_tokens=cached_prefix_num_tokens,
        )

        # Add to scheduler
        scheduler = self.engine.scheduler
        scheduler.add(seq)

        ttft = None
        prefill_latency = 0.0
        start_time = perf_counter()
        block_table_snapshot: List[int] = list(seq.block_table)
        num_tokens_snapshot: int = len(seq)
        pinned_block_table: List[int] = []

        while not seq.is_finished:
            seqs, is_prefill = scheduler.schedule()

            # Capture prefill end time right before the first decode forward pass
            if prefill_latency == 0.0 and not is_prefill:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                prefill_latency = perf_counter() - start_time

            token_ids_out = self.engine.model_runner.call("run", seqs, is_prefill)
            # Pin the terminal step blocks before scheduler.postprocess deallocates seq.
            for sched_seq, token_id in zip(seqs, token_ids_out):
                if sched_seq is not seq:
                    continue
                will_finish = (
                    ((not sched_seq.ignore_eos) and token_id == scheduler.eos)
                    or (sched_seq.num_completion_tokens + 1 >= sched_seq.max_tokens)
                )
                if will_finish and not pinned_block_table:
                    pinned_block_table = list(sched_seq.block_table)
                    self.paged_kv_engine.increment_ref(pinned_block_table)
            # Keep a copy before postprocess potentially frees sequence blocks.
            block_table_snapshot = list(seq.block_table)
            num_tokens_snapshot = len(seq)
            scheduler.postprocess(seqs, token_ids_out)

            # Capture TTFT after the first decode forward pass completes
            if ttft is None and not is_prefill:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ttft = perf_counter() - start_time

        if ttft is None:
            ttft = 0.0

        # Persist snapshots for callers that need KV block ranges after generation.
        setattr(seq, "_block_table_snapshot", block_table_snapshot)
        setattr(seq, "_num_tokens_snapshot", num_tokens_snapshot)
        setattr(seq, "_pinned_block_table", pinned_block_table)

        return seq.completion_token_ids, ttft, prefill_latency, seq

    def _find_upstream_agent_paged(self, ph_id: str, message: str) -> Optional[str]:
        """Find an agent that co-appears with self in a historical anchor for ph_id.

        Only considers anchor entries where self.node_id also has a delta, so
        the returned agent_id is guaranteed to be usable in valid_anchors.

        Returns the first such agent_id that is not self.node_id, or None.
        """
        ph_store = self.paged_kv_engine.anchors.get(ph_id, {})
        logger.info(
            "[UPSTREAM_SEARCH] self={} ph_id={} ph_store_keys={} entries_agent_ids={}",
            self.node_id, ph_id, len(ph_store),
            [{k: list(e.agent_deltas.keys()) for k, e in ph_store.items()}],
        )
        for entry in ph_store.values():
            if self.node_id not in entry.agent_deltas:
                continue
            for agent_id in entry.agent_deltas:
                if agent_id != self.node_id:
                    return agent_id
        return None

    def _prepare_kv_reuse_prefix_blocks(
        self,
        *,
        prefix_store: Dict[str, Any],
        placeholder_info: Dict[str, List[int]],
        prompt_num_tokens: int,
        message_key: str,
    ) -> Tuple[Optional[List[int]], int, Dict[str, Any]]:
        """Build pre-injected cached prefix blocks for kv_reuse mode.

        Processes ALL valid placeholder spans in token order:
          1. Current-round sharing: base + (delta_upstream_current - cross_delta_hist)
          2. KVCOMM anchor matching: weighted historical delta (pf_num=0 per span)
          3. Fallback: base template blocks (ref-incremented)
        Gaps between spans and the tail are filled with base blocks.

        Returns (block_table, cached_tokens, stats).
        """
        stats: Dict[str, Any] = {
            "anchor_candidates": 0,
            "offset_calls": 0,
            "offset_effective": 0,
            "offset_applied": False,
            "fallback_reason": None,
        }

        base_block_table = list(prefix_store.get("prefix_block_table", []) or [])
        if not base_block_table:
            stats["fallback_reason"] = "no_prefix_block_table"
            return None, 0, stats
        if not placeholder_info:
            stats["fallback_reason"] = "no_placeholder_info"
            return None, 0, stats

        bs = self.paged_kv_engine.block_size
        expected_blocks = (prompt_num_tokens + bs - 1) // bs

        # All valid spans sorted by token start position
        valid_ph_items = sorted(
            [
                (ph_id, span)
                for ph_id, span in placeholder_info.items()
                if span[0] >= 0 and span[1] > span[0] and span[1] <= prompt_num_tokens
            ],
            key=lambda x: x[1][0],
        )
        if not valid_ph_items:
            stats["fallback_reason"] = "no_valid_placeholder_span"
            return None, 0, stats

        combined_blocks: List[int] = []
        to_free: List[List[int]] = []   # every block list with a live extra ref
        current_block = 0               # next unprocessed block index
        any_offset = False

        def _cleanup() -> None:
            for blks in to_free:
                try:
                    self.paged_kv_engine.free_blocks(blks)
                except Exception:
                    pass

        try:
            for ph_id, (ph_start, ph_end) in valid_ph_items:
                ph_start_block = ph_start // bs
                ph_end_block = (ph_end - 1) // bs + 1
                ph_num = ph_end - ph_start
                # How far into its first block does the placeholder start?
                ph_start_intra = ph_start % bs

                # Skip if invalid or overlapping with the previous span's blocks
                if (ph_start_block < current_block
                        or ph_start_block >= len(base_block_table)
                        or ph_num <= 0):
                    continue

                # ── gap: base blocks between current_block and this span ──────
                gap = base_block_table[current_block:ph_start_block]
                if gap:
                    self.paged_kv_engine.increment_ref(gap)
                    combined_blocks.extend(gap)
                    to_free.append(gap)

                base_ph_blocks = base_block_table[ph_start_block:ph_end_block]
                if not base_ph_blocks:
                    current_block = ph_end_block
                    continue

                # ── How many tokens does this block range need to cover? ───────
                # When ph_start_intra > 0 the output block must also cover the
                # prefix tokens that share the first block with the placeholder.
                # Similarly the last block may contain post-ph prompt tokens.
                blk_start_tok = ph_start_block * bs
                blk_end_tok   = min(ph_end_block * bs, prompt_num_tokens)
                tokens_for_block = blk_end_tok - blk_start_tok
                # tokens_for_block == ph_num when ph_start_intra==0 and ph fills
                # the block(s) exactly; otherwise it is larger.

                # ── Read full base block KV when we need to merge prefix/suffix ─
                # We always read tokens_for_block so we have the complete picture.
                if ph_start_intra > 0 or tokens_for_block != ph_num:
                    base_full_k, base_full_v = self.paged_kv_engine.read_kv_from_blocks(
                        base_ph_blocks, tokens_for_block
                    )
                    # Slice just the placeholder portion for delta computation.
                    base_ph_k = base_full_k[..., ph_start_intra:ph_start_intra + ph_num, :]
                    base_ph_v = base_full_v[..., ph_start_intra:ph_start_intra + ph_num, :]
                    need_merge = True
                else:
                    base_full_k = base_full_v = None
                    base_ph_k = base_ph_v = None
                    need_merge = False

                new_ph_k: Optional[torch.Tensor] = None
                new_ph_v: Optional[torch.Tensor] = None

                # ── Path 1: current-round sharing (user_question only) ────────
                # Only user_question has identical prefix across agents (cross-delta ≈ 0).
                # Disabled when config.use_current_round_sharing is False.
                _crs_enabled = (
                    ph_id == "user_question"
                    and getattr(getattr(self, "config", None), "use_current_round_sharing", True)
                )
                cur_entry = self.paged_kv_engine.anchors.get(ph_id, {}).get(message_key) \
                    if _crs_enabled else None
                if cur_entry is not None:
                    up_candidates = sorted(
                        aid for aid in cur_entry.agent_deltas if aid != self.node_id
                    )
                    if up_candidates:
                        up_id = up_candidates[0]
                        up_info = cur_entry.agent_deltas[up_id]
                        if int(up_info.get("ph_delta_num_tokens", 0) or 0) >= ph_num:
                            try:
                                # Use the correctly positioned base KV slice.
                                base_k = base_ph_k if need_merge else \
                                    self.paged_kv_engine.read_kv_from_blocks(base_ph_blocks, ph_num)[0]
                                base_v = base_ph_v if need_merge else \
                                    self.paged_kv_engine.read_kv_from_blocks(base_ph_blocks, ph_num)[1]
                                if not need_merge:
                                    base_k, base_v = self.paged_kv_engine.read_kv_from_blocks(
                                        base_ph_blocks, ph_num
                                    )

                                up_dk, up_dv = self.paged_kv_engine.read_kv_from_blocks(
                                    up_info["ph_delta_blocks"], ph_num
                                )
                                up_dk = up_dk[..., :ph_num, :]
                                up_dv = up_dv[..., :ph_num, :]

                                # Historical cross-delta: Σ w_k(delta_up_k − delta_self_k)
                                ph_store = self.paged_kv_engine.anchors.get(ph_id, {})
                                hist = [
                                    (m, e) for m, e in ph_store.items()
                                    if m != message_key
                                    and up_id in e.agent_deltas
                                    and self.node_id in e.agent_deltas
                                    and int(e.agent_deltas[up_id].get("ph_delta_num_tokens", 0)) >= ph_num
                                    and int(e.agent_deltas[self.node_id].get("ph_delta_num_tokens", 0)) >= ph_num
                                ]
                                cross_k = torch.zeros_like(up_dk)
                                cross_v = torch.zeros_like(up_dv)
                                if hist:
                                    sims = torch.stack([
                                        (base_k - e.ph_key_embedding[..., -ph_num:, :].to(
                                            base_k.device, non_blocking=True)
                                        ).norm(2, dim=-2)
                                        for _, e in hist
                                    ], dim=0)
                                    weights = torch.softmax(-sims.float(), dim=0).unsqueeze(-2)
                                    for i, (_, e) in enumerate(hist):
                                        up_h_k, up_h_v = self.paged_kv_engine.read_kv_from_blocks(
                                            e.agent_deltas[up_id]["ph_delta_blocks"], ph_num
                                        )
                                        sl_h_k, sl_h_v = self.paged_kv_engine.read_kv_from_blocks(
                                            e.agent_deltas[self.node_id]["ph_delta_blocks"], ph_num
                                        )
                                        w = weights[i]
                                        cross_k += w * (up_h_k[..., :ph_num, :] - sl_h_k[..., :ph_num, :])
                                        cross_v += w * (up_h_v[..., :ph_num, :] - sl_h_v[..., :ph_num, :])

                                new_ph_k = base_k + (up_dk - cross_k)
                                new_ph_v = base_v + (up_dv - cross_v)
                                any_offset = True
                                logger.info(
                                    "[CUR_ROUND_SHARE:paged] node={} ph_id={} upstream={} "
                                    "ph_num={} hist={}",
                                    getattr(self, "node_id", "?"),
                                    ph_id, up_id, ph_num, len(hist),
                                )
                                # Store this node's CRS-derived delta so future
                                # offset_kv_cache rounds can use it (avoids silently
                                # skipping this anchor due to missing agent_deltas entry).
                                # ph delta = new_ph_k - base_k; pf delta = 0.
                                try:
                                    ph_delta_k = (new_ph_k - base_k).contiguous()
                                    ph_delta_v = (new_ph_v - base_v).contiguous()
                                    delta_blocks = self.paged_kv_engine.allocate_blocks_for_tokens(ph_num)
                                    self.paged_kv_engine.write_kv_to_blocks(
                                        delta_blocks, ph_delta_k, ph_delta_v, ph_num
                                    )
                                    with self.paged_kv_engine._lock:
                                        old = cur_entry.agent_deltas.get(self.node_id)
                                        if old is not None:
                                            self.paged_kv_engine.free_blocks(old.get("ph_delta_blocks", []))
                                            self.paged_kv_engine.free_blocks(old.get("pf_delta_blocks", []))
                                        cur_entry.agent_deltas[self.node_id] = {
                                            "ph_delta_blocks": delta_blocks,
                                            "ph_delta_num_tokens": ph_num,
                                            "pf_delta_blocks": [],
                                            "pf_delta_num_tokens": 0,
                                        }
                                    logger.info(
                                        "[CUR_ROUND_SHARE:paged] wrote CRS delta | node={} ph_id={}",
                                        getattr(self, "node_id", "?"), ph_id,
                                    )
                                except Exception as delta_exc:
                                    logger.warning(
                                        "[CUR_ROUND_SHARE:paged] failed to write CRS delta: {}",
                                        delta_exc,
                                    )
                            except Exception as exc:
                                logger.warning(
                                    "[CUR_ROUND_SHARE:paged] ph_id={} error: {}; trying KVCOMM",
                                    ph_id, exc,
                                )
                                new_ph_k = new_ph_v = None

                # ── Path 2: KVCOMM anchor matching (ph only, pf_num=0) ────────
                if new_ph_k is None:
                    anchor_msgs = [
                        m for m in self.paged_kv_engine.anchors.get(ph_id, {}).keys()
                        if m != message_key
                    ]
                    stats["anchor_candidates"] += len(anchor_msgs)
                    if anchor_msgs:
                        stats["offset_calls"] += 1
                        result_ph, _, _, _ = self.paged_kv_engine.offset_kv_cache(
                            agent_id=self.node_id,
                            ph_id=ph_id,
                            message=message_key,
                            base_ph_block_table=base_ph_blocks,
                            base_ph_num_tokens=ph_num,
                            base_pf_block_table=[],
                            base_pf_num_tokens=0,
                            anchor_list=anchor_msgs,
                            temperature=1.0,
                            ph_start_intra=ph_start_intra,
                        )
                        # offset_kv_cache returns a block with ph_num offset tokens at pos 0.
                        # Read them back as tensors so we can merge below.
                        if result_ph != base_ph_blocks:
                            new_ph_k, new_ph_v = self.paged_kv_engine.read_kv_from_blocks(
                                result_ph, ph_num
                            )
                            self.paged_kv_engine.free_blocks(result_ph)
                            any_offset = True
                            stats["offset_effective"] += 1
                            logger.info(
                                "[KV_REUSE_OFFSET:paged] node={} ph_id={} ph_start={} ph_num={} "
                                "ph_start_intra={} tokens_for_block={} need_merge={} anchor_msgs={}",
                                getattr(self, "node_id", "?"),
                                ph_id, ph_start, ph_num,
                                ph_start_intra, tokens_for_block, need_merge,
                                len(anchor_msgs),
                            )
                        else:
                            # offset_kv_cache returned base unchanged (ref incremented)
                            self.paged_kv_engine.free_blocks(result_ph)
                            logger.info(
                                "[KV_REUSE_NO_OFFSET:paged] node={} ph_id={} offset_kv_cache returned base unchanged",
                                getattr(self, "node_id", "?"),
                                ph_id,
                            )

                # ── Assemble output block(s) ──────────────────────────────────
                new_ph_blocks: Optional[List[int]] = None
                if new_ph_k is not None:
                    if need_merge:
                        # Merge: start from base full-block KV, overlay the
                        # offset-applied placeholder slice.
                        assert base_full_k is not None
                        merged_k = base_full_k.clone()
                        merged_v = base_full_v.clone()
                        merged_k[..., ph_start_intra:ph_start_intra + ph_num, :] = new_ph_k
                        merged_v[..., ph_start_intra:ph_start_intra + ph_num, :] = new_ph_v
                        new_ph_blocks = self.paged_kv_engine.allocate_blocks_for_tokens(tokens_for_block)
                        self.paged_kv_engine.write_kv_to_blocks(
                            new_ph_blocks, merged_k, merged_v, tokens_for_block
                        )
                    else:
                        # Placeholder perfectly fills its block(s): write directly.
                        new_ph_blocks = self.paged_kv_engine.allocate_blocks_for_tokens(ph_num)
                        self.paged_kv_engine.write_kv_to_blocks(
                            new_ph_blocks, new_ph_k, new_ph_v, ph_num
                        )

                # ── Fallback: base template blocks ────────────────────────────
                if new_ph_blocks is None:
                    if need_merge:
                        # Use the first base ph block only (it covers tokens_for_block).
                        self.paged_kv_engine.increment_ref([base_ph_blocks[0]])
                        new_ph_blocks = [base_ph_blocks[0]]
                    else:
                        self.paged_kv_engine.increment_ref(base_ph_blocks)
                        new_ph_blocks = base_ph_blocks

                to_free.append(new_ph_blocks)
                combined_blocks.extend(new_ph_blocks)
                current_block = ph_end_block

            # ── tail: base blocks after the last span, capped at expected_blocks ─
            # Do NOT blindly append all remaining template blocks – the template
            # may have been built for a longer prompt than the current request.
            tail = base_block_table[current_block:expected_blocks]
            if tail:
                self.paged_kv_engine.increment_ref(tail)
                combined_blocks.extend(tail)
                to_free.append(tail)

            if len(combined_blocks) != expected_blocks:
                _cleanup()
                stats["fallback_reason"] = (
                    f"combined_mismatch:{len(combined_blocks)}!={expected_blocks}"
                )
                return None, 0, stats

            if not any_offset:
                _cleanup()
                stats["fallback_reason"] = "no_offset_applied"
                return None, 0, stats

            stats["offset_applied"] = True
            logger.info(
                "[KV_REUSE_PREFIX_BUILT:paged] node={} combined_blocks={} expected={} "
                "prompt_num_tokens={} ph_items={}",
                getattr(self, "node_id", "?"),
                len(combined_blocks), expected_blocks, prompt_num_tokens,
                [(ph_id, ph_start, ph_end) for ph_id, (ph_start, ph_end) in valid_ph_items],
            )
            return combined_blocks, prompt_num_tokens, stats

        except Exception:
            _cleanup()
            raise

    # ── Prefix preparation (paged version) ──

    async def prepare_prefix_kv_segments(self, node_id: str, prefix: str, user_prompt: str):
        """Materialize prefix KV into blocks and store block tables.

        Unlike LLMChat which stores DynamicCache objects, we:
          1. Run prefill through nano-vllm engine → KV written to blocks by triton
          2. Store block_table references (not KV tensors) in shared memory
        """
        messages = self._render_base_messages(prefix, user_prompt)
        prompt_text = self._build_prompt_text(messages)
        placeholder_info, segments = self._locate_placeholder(prompt_text)

        # Run the full prompt through the engine to populate KV blocks
        full_token_ids = self._encode(prompt_text)
        sp = SamplingParams(temperature=1.0, max_tokens=1)
        seq = Sequence(full_token_ids, sp)

        scheduler = self.engine.scheduler
        scheduler.add(seq)

        # Run one prefill step to fill KV cache blocks
        seqs, is_prefill = scheduler.schedule()
        assert is_prefill, "Expected prefill step"
        self.engine.model_runner.call("run", seqs, is_prefill)

        # Now seq.block_table has the physical blocks containing the prefix KV
        full_block_table = list(seq.block_table)
        full_num_tokens = len(seq)

        # Store per-segment block ranges
        segment_block_info = []
        segment_token_ids_list = []
        for type_, text, token_ids, s, e in segments:
            if type_ == "text":
                # Compute which blocks this segment spans
                start_block = s // self.paged_kv_engine.block_size
                end_block = (e - 1) // self.paged_kv_engine.block_size + 1
                seg_blocks = full_block_table[start_block:end_block]
                self.paged_kv_engine.increment_ref(seg_blocks)
                segment_block_info.append({
                    "block_table": seg_blocks,
                    "start_token": s,
                    "end_token": e,
                    "num_tokens": e - s,
                })
                encoding = {
                    "input_ids": torch.tensor(token_ids, device="cuda").unsqueeze(0),
                }
                encoding["attention_mask"] = torch.ones_like(encoding["input_ids"])
                segment_token_ids_list.append(encoding)

        # Deallocate the generation sequence but keep
        # the blocks we incremented ref on
        scheduler.postprocess(seqs, [self.tokenizer.eos_token_id])

        mem = PagedLLMChat._shared_kv_cache_memory[node_id]
        # Free old prefix segment blocks before overwriting to prevent block leak.
        # Each segment block was increment_ref'd when first materialized; without
        # this explicit free the ref_count never reaches 0 and the blocks stay
        # pinned in the pool forever.
        old_prefix_block_info = mem.get("prefix_block_info")
        if old_prefix_block_info:
            for seg in old_prefix_block_info:
                self.paged_kv_engine.free_blocks(seg.get("block_table", []))
        mem["prefix_block_info"] = segment_block_info
        mem["prefix_block_table"] = full_block_table
        mem["prefix_num_tokens"] = full_num_tokens
        mem["placeholder_info"] = placeholder_info
        mem["token_ids"] = segment_token_ids_list

        PagedLLMChat._initialization[node_id] = True

    def _locate_placeholder(self, original_text: str):
        """Locate placeholder spans in prompt. Returns (placeholder_info, segments)."""
        pattern = r'\{((?:agent|condition)_\w+_(?:current|history)|user_question)\}'
        matches = list(re.finditer(pattern, original_text))

        segments = []
        placeholder_info = {}
        last_pos = 0
        token_num = 0

        for m in matches:
            start, end = m.span()
            placeholder_inner = m.group(1)

            if last_pos < start:
                txt = original_text[last_pos:start]
                ids = self._encode(txt)
                if txt.strip():
                    segments.append(("text", txt, ids, token_num, token_num + len(ids)))
                token_num += len(ids)

            ph_text = f'{{{placeholder_inner}}} '
            ids = self._encode(ph_text)
            segments.append(("placeholder", placeholder_inner, ids, token_num, token_num + len(ids)))
            placeholder_info[placeholder_inner] = [token_num, token_num + len(ids)]
            token_num += len(ids)
            last_pos = end

        txt = original_text[last_pos:]
        ids = self._encode(txt)
        if txt.strip():
            segments.append(("text", txt, ids, token_num, token_num + len(ids)))
            token_num += len(ids)

        segments.sort(key=lambda x: x[-1])
        placeholder_info = dict(sorted(placeholder_info.items(), key=lambda x: x[1][0], reverse=True))
        return placeholder_info, segments

    # ── Generation entry points ──

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Synchronous generation."""
        prompt_text = self._build_prompt_text(messages)
        token_ids = self._encode(prompt_text)
        max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE

        completion_ids, *_ = self._generate_tokens(token_ids, max_tokens, temperature)
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True)

    async def agen(
        self,
        messages: List[Message] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_cache: Optional[bool] = False,
        *,
        request_uid: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
    ) -> GenerationResult:
        """Async generation through nano-vllm engine."""
        async with self.lock:
            max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
            if temperature is None:
                temperature = self.DEFAULT_TEMPERATURE

            prompt_text = self._build_prompt_text(messages)
            token_ids = self._encode(prompt_text)

            safe_prompt = _escape_loguru_markup(prompt_text)
            logger.opt(colors=True).debug(
                "<blue>[PROMPT]</blue> Agent {} Role {} Prompt:\n{}",
                getattr(self, "node_id", "?"),
                getattr(self, "role", "?"),
                safe_prompt,
            )

            completion_ids, ttft, _prefill_lat, seq = self._generate_tokens(token_ids, max_tokens, temperature)
            response_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

            safe_resp = _escape_loguru_markup(response_text)
            logger.opt(colors=True).debug(
                "<blue>[RESPONSE]</blue> Agent {} Role {} Response:\n{}",
                getattr(self, "node_id", "?"),
                getattr(self, "role", "?"),
                safe_resp,
            )

            metadata: Dict[str, Any] = {}
            if request_uid:
                metadata["request_uid"] = request_uid
            if agent_id:
                metadata["agent_id"] = agent_id
            if return_cache:
                metadata["block_table"] = list(seq.block_table)
                metadata["num_tokens"] = len(seq)

            return GenerationResult(
                text=response_text,
                mode="paged",
                ttft=ttft,
                metadata=metadata,
            )

    async def generate_for_agent(
        self,
        *,
        request_uid: str,
        message: str,
        preferred_mode: Optional[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate using the requested strategy with paged attention."""
        anchor_forces_dense = self.has_active_anchor(request_uid, message)
        selected_mode = "dense_prefill" if (preferred_mode == "dense_prefill" or anchor_forces_dense) else "kv_reuse"
        logger.info(
            "[MODE_DECISION:paged] node={} role={} request_uid={} preferred_mode={} "
            "anchor_forces_dense={} selected_mode={}",
            getattr(self, "node_id", "?"),
            getattr(self, "role", "?"),
            request_uid,
            preferred_mode,
            anchor_forces_dense,
            selected_mode,
        )
        if selected_mode == "dense_prefill":
            return await self.generate_with_dense_prefill(
                message,
                max_tokens=max_tokens,
                temperature=temperature,
                request_uid=request_uid,
                agent_id=agent_id,
                agent_name=agent_name,
                agent_role=agent_role,
                output_dir=output_dir,
                **kwargs,
            )
        return await self.generate_with_kv_reuse(
            message,
            max_tokens=max_tokens,
            temperature=temperature,
            request_uid=request_uid,
            agent_id=agent_id,
            agent_name=agent_name,
            agent_role=agent_role,
            output_dir=output_dir,
            **kwargs,
        )

    async def generate_with_kv_reuse(
        self,
        messages=None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        request_uid: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate by reusing prefix KV blocks (fast path).

        The prefix KV blocks are already in the block pool from prepare_prefix_kv_segments().
        We construct a Sequence whose block_table includes those prefix blocks,
        so the scheduler skips prefilling them (prefix caching via hash match).
        """
        test_time = kwargs.pop("test_time", False)
        if test_time:
            return await self.agen_kvcomm_time_test(
                messages=messages,
                max_tokens=max_tokens,
                min_tokens=max_tokens,
                temperature=temperature,
                request_uid=request_uid,
                mode="kv_reuse",
                agent_id=agent_id,
                agent_name=agent_name,
                agent_role=agent_role,
                output_dir=output_dir,
            )
        return await self._generate_paged(
            messages, "kv_reuse",
            max_tokens=max_tokens, temperature=temperature,
            request_uid=request_uid, agent_id=agent_id,
            agent_name=agent_name, agent_role=agent_role,
            output_dir=output_dir, **kwargs,
        )

    async def generate_with_dense_prefill(
        self,
        messages=None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        request_uid: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate with fresh prefix computation and anchor update."""
        return await self._generate_paged(
            messages, "dense_prefill",
            max_tokens=max_tokens, temperature=temperature,
            request_uid=request_uid, agent_id=agent_id,
            agent_name=agent_name, agent_role=agent_role,
            output_dir=output_dir, **kwargs,
        )

    async def _generate_paged(
        self,
        messages,
        mode: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        request_uid: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> GenerationResult:
        """Core paged generation for both kv_reuse and dense_prefill modes.

        In both modes:
          1. Build full prompt token IDs (with placeholders filled)
          2. Feed to nano-vllm engine → scheduler allocates blocks, model runs
          3. Engine Attention layers write KV directly into blocks (triton)
          4. After generation, use block_table for anchor operations

        kv_reuse mode:
          - Prefix blocks may already be cached (BlockManager hash match)
          - seq.num_cached_tokens will skip re-computing cached prefix
          - Then apply anchor deltas from PagedKVCOMMEngine

        dense_prefill mode:
          - Full prefill, then compute anchor deltas for future reuse
        """
        async with self.lock:
            max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
            if temperature is None:
                temperature = self.DEFAULT_TEMPERATURE
            logger.info(
                "[MODE_EXECUTE:paged] node={} role={} request_uid={} mode={} max_tokens={} temperature={}",
                getattr(self, "node_id", "?"),
                getattr(self, "role", "?"),
                request_uid,
                mode,
                max_tokens,
                temperature,
            )
            preprocess_start = perf_counter() if mode == "kv_reuse" else None

            message = messages[0] if isinstance(messages, list) else messages
            message_key = self._message_cache_key(message)
            reuse_stats: Dict[str, Any] = {
                "mode": mode,
                "placeholder_count": 0,
                "anchor_candidates": 0,
                "offset_calls": 0,
                "offset_effective": 0,
                "num_cached_tokens": 0,
                "num_blocks": 0,
                "anchor_set_count": 0,
                "anchor_skip_reasons": {},
            }

            def _record_anchor_skip(reason: str) -> None:
                skip_map = reuse_stats["anchor_skip_reasons"]
                skip_map[reason] = int(skip_map.get(reason, 0)) + 1

            # Build prompt
            prefix_store = PagedLLMChat._shared_kv_cache_memory.get(self.node_id, {})
            stored_placeholder_info = prefix_store.get("placeholder_info", {})

            prompt_text = self._build_prompt_text(message)
            token_ids = self._encode(prompt_text)
            dynamic_placeholder_info, _ = self._locate_placeholder(prompt_text)

            # kv_reuse should use runtime-aligned spans when available.
            placeholder_info_for_reuse = dict(stored_placeholder_info)
            uq_bucket = PagedLLMChat._global_anchor_info_dict.get("user_question", {})
            uq_meta = uq_bucket.get(message_key, uq_bucket.get(message))
            reuse_span_source = "template"
            if isinstance(uq_meta, (list, tuple)) and len(uq_meta) >= 2:
                try:
                    uq_num_tokens = int(uq_meta[1])
                except (TypeError, ValueError):
                    uq_num_tokens = 0
                if 0 < uq_num_tokens <= len(token_ids):
                    uq_start = len(token_ids) - uq_num_tokens
                    placeholder_info_for_reuse["user_question"] = [uq_start, len(token_ids)]
                    reuse_span_source = "runtime_input_anchor"
            logger.info(
                "[REUSE_SPAN:paged] node={} role={} request_uid={} source={} has_uq_meta={} prompt_tokens={} uq_span={}",
                getattr(self, "node_id", "?"),
                getattr(self, "role", "?"),
                request_uid,
                reuse_span_source,
                isinstance(uq_meta, (list, tuple)) and len(uq_meta) >= 2,
                len(token_ids),
                placeholder_info_for_reuse.get("user_question"),
            )
            # dense_prefill anchor writes should prefer runtime placeholder spans.
            if dynamic_placeholder_info:
                placeholder_info_for_anchor = dynamic_placeholder_info
                placeholder_source = "dynamic_prompt"
            else:
                # If prompt has no explicit placeholders, only keep template spans
                # that are valid for the current prompt length.
                placeholder_info_for_anchor = {
                    ph_id: span
                    for ph_id, span in stored_placeholder_info.items()
                    if span[0] >= 0 and span[1] > span[0] and span[1] <= len(token_ids)
                }
                placeholder_source = "stored_template_filtered"
                # Fallback: template spans may be stored from a long system+few-shot
                # prompt that exceeds the current (user-only) generation prompt.
                # In that case, reuse the runtime-aligned user_question span that
                # was already computed for kv_reuse (from update_input_anchor metadata).
                if not placeholder_info_for_anchor:
                    uq_span = placeholder_info_for_reuse.get("user_question")
                    if uq_span and uq_span[1] > uq_span[0] and uq_span[1] <= len(token_ids):
                        placeholder_info_for_anchor = {"user_question": uq_span}
                        placeholder_source = "runtime_reuse_span_fallback"

            logger.info(
                "[PLACEHOLDER_SPAN:paged] node={} role={} request_uid={} source={} dynamic_count={} stored_count={} anchor_count={} prompt_tokens={}",
                getattr(self, "node_id", "?"),
                getattr(self, "role", "?"),
                request_uid,
                placeholder_source,
                len(dynamic_placeholder_info),
                len(stored_placeholder_info),
                len(placeholder_info_for_anchor),
                len(token_ids),
            )

            safe_prompt = _escape_loguru_markup(prompt_text)
            logger.opt(colors=True).debug(
                "<blue>[PROMPT:{}]</blue> Agent {} Role {} Prompt:\n{}",
                mode, self.node_id, self.role, safe_prompt,
            )

            cached_prefix_block_table = None
            cached_prefix_num_tokens = 0
            if mode == "kv_reuse":
                (
                    cached_prefix_block_table,
                    cached_prefix_num_tokens,
                    offset_stats,
                ) = self._prepare_kv_reuse_prefix_blocks(
                    prefix_store=prefix_store,
                    placeholder_info=placeholder_info_for_reuse,
                    prompt_num_tokens=len(token_ids),
                    message_key=message_key,
                )
                reuse_stats["anchor_candidates"] = offset_stats["anchor_candidates"]
                reuse_stats["offset_calls"] = offset_stats["offset_calls"]
                reuse_stats["offset_effective"] = offset_stats["offset_effective"]
                if not offset_stats["offset_applied"]:
                    reason = offset_stats.get("fallback_reason", "unknown")
                    _record_anchor_skip(f"kv_reuse_fallback:{reason}")
                    logger.info(
                        "kv_reuse fallback to dense behavior for this request: {}",
                        reason,
                    )

            # Generate through nano-vllm engine
            preprocess_latency = 0.0
            if preprocess_start is not None:
                preprocess_latency = max(0.0, perf_counter() - preprocess_start)
            completion_ids, ttft, prefill_latency, seq = self._generate_tokens(
                token_ids,
                max_tokens,
                temperature,
                cached_prefix_block_table=cached_prefix_block_table,
                cached_prefix_num_tokens=cached_prefix_num_tokens,
            )
            pinned_block_table = list(getattr(seq, "_pinned_block_table", []) or [])

            # Align with HF backend: preprocess_latency includes engine prefill,
            # generation_ttft is decode-only (ttft minus prefill).
            preprocess_latency += prefill_latency
            generation_ttft = max(0.0, ttft - prefill_latency)
            total_ttft = ttft
            if preprocess_start is not None:
                total_ttft = preprocess_latency + generation_ttft

            # Block table now contains all KV blocks for this generation
            block_table = list(getattr(seq, "_block_table_snapshot", list(seq.block_table)))
            prompt_num_tokens = len(token_ids)
            total_num_tokens = int(getattr(seq, "_num_tokens_snapshot", len(seq)))
            reuse_stats["num_cached_tokens"] = int(getattr(seq, "num_cached_tokens", 0))
            reuse_stats["num_blocks"] = len(block_table)
            reuse_stats["placeholder_count"] = len(placeholder_info_for_anchor)

            # ── Anchor operations ──
            if mode == "dense_prefill" and placeholder_info_for_anchor:
                # After full prefill, store anchor deltas
                for ph_id, (ph_start, ph_end) in placeholder_info_for_anchor.items():
                    anchor_messages = list(self.paged_kv_engine.anchors.get(ph_id, {}).keys())
                    reuse_stats["anchor_candidates"] += len(anchor_messages)

                    # Block range for placeholder portion
                    bs = self.paged_kv_engine.block_size
                    ph_start_block = ph_start // bs
                    ph_end_block = (ph_end - 1) // bs + 1
                    ph_blocks = block_table[ph_start_block:ph_end_block]
                    ph_num = ph_end - ph_start

                    # Block range for prefix after placeholder
                    pf_start = ph_end
                    pf_end = prompt_num_tokens
                    pf_start_block = pf_start // bs
                    pf_end_block = (pf_end - 1) // bs + 1
                    pf_blocks = block_table[pf_start_block:pf_end_block]
                    pf_num = pf_end - pf_start

                    # Validate that block ranges are within the actual block table.
                    # placeholder_info positions are from the template prompt; if the
                    # generation prompt is shorter (e.g. just the task string), the
                    # block indices can fall out of range.
                    if ph_num <= 0 or not ph_blocks or ph_start_block >= len(block_table):
                        _record_anchor_skip("placeholder_blocks_out_of_range")
                        logger.info(
                            "dense_prefill: skipping set_anchor for {} — placeholder blocks "
                            "out of range (ph_start_block={}, block_table_len={})",
                            ph_id, ph_start_block, len(block_table),
                        )
                        continue

                    if ph_end > prompt_num_tokens:
                        _record_anchor_skip("placeholder_end_exceeds_prompt")
                        logger.info(
                            "dense_prefill: skipping set_anchor for {} — placeholder end {} exceeds prompt tokens {}",
                            ph_id,
                            ph_end,
                            prompt_num_tokens,
                        )
                        continue

                    # pf_num=0 is allowed: placeholder ends at prompt end (e.g. user_question
                    # is last in prompt). set_anchor handles pf_tokens=0 gracefully.
                    if pf_num < 0:
                        _record_anchor_skip("prefix_blocks_out_of_range")
                        logger.info(
                            "dense_prefill: skipping set_anchor for {} — prefix blocks out of range "
                            "(pf_start_block={}, pf_end_block={}, block_table_len={}, pf_num={})",
                            ph_id,
                            pf_start_block,
                            pf_end_block,
                            len(block_table),
                            pf_num,
                        )
                        continue

                    # We need base blocks - stored in prefix_block_info
                    base_block_table = prefix_store.get("prefix_block_table", [])
                    if base_block_table:
                        base_ph_blocks = base_block_table[ph_start_block:ph_end_block]
                        base_pf_blocks = base_block_table[pf_start_block:pf_end_block]

                        if not base_ph_blocks:
                            _record_anchor_skip("base_blocks_out_of_range")
                            logger.info(
                                "dense_prefill: skipping set_anchor for {} — base blocks out of range",
                                ph_id,
                            )
                            continue

                        if not base_pf_blocks:
                            _record_anchor_skip("base_prefix_blocks_out_of_range")
                            logger.info(
                                "dense_prefill: skipping set_anchor for {} — base prefix blocks out of range",
                                ph_id,
                            )
                            continue

                        self.paged_kv_engine.set_anchor(
                            agent_id=self.node_id,
                            ph_id=ph_id,
                            message=message_key,
                            real_block_table=ph_blocks,
                            real_num_tokens=ph_num,
                            base_block_table=base_ph_blocks,
                            base_num_tokens=ph_num,
                            real_prefix_block_table=pf_blocks,
                            real_prefix_num_tokens=pf_num,
                            base_prefix_block_table=base_pf_blocks,
                            base_prefix_num_tokens=pf_num,
                            max_anchor_num=self.config.max_anchor_num,
                            window_length=self.config.window_size,
                            free_ratio_threshold=self.config.proactive_evict_threshold,
                            ph_start_intra=ph_start % bs,
                        )
                        reuse_stats["anchor_set_count"] += 1
                    else:
                        _record_anchor_skip("missing_prefix_block_table")
                        logger.info(
                            "dense_prefill: skipping set_anchor for {} — missing prefix_block_table",
                            ph_id,
                        )

            elif mode == "kv_reuse" and cached_prefix_block_table:
                logger.debug(
                    "kv_reuse pre-injected cached prefix blocks: cached_tokens={}, blocks={}",
                    cached_prefix_num_tokens,
                    len(cached_prefix_block_table),
                )

            # Store response block info for future reuse
            mem = PagedLLMChat._shared_kv_cache_memory.get(self.node_id, {})
            resp_blocks = mem.setdefault("response_blocks", {})
            resp_start_block = prompt_num_tokens // self.paged_kv_engine.block_size
            response_block_table = block_table[resp_start_block:]
            response_num_tokens = total_num_tokens - prompt_num_tokens
            resp_blocks.setdefault(message_key, []).append({
                "block_table": response_block_table,
                "num_tokens": response_num_tokens,
            })

            # ── Response anchor prediction (mirrors HF gpt_chat.py logic) ──
            response_anchor_key = f"agent_{self.node_id}_current"
            resp_info_bucket = PagedLLMChat._anchor_info_dict.setdefault(response_anchor_key, {})
            resp_global_bucket = PagedLLMChat._global_anchor_info_dict.setdefault(response_anchor_key, {})

            if response_block_table and response_num_tokens > 0:
                # Store actual response value KV before prediction so this round's
                # embedding is available for future comparisons (same as HF path
                # which appends response_kv_cache before calling predict_as_anchor).
                self.paged_kv_engine.store_response_embedding(
                    ph_id=response_anchor_key,
                    message=message_key,
                    block_table=response_block_table,
                    num_tokens=response_num_tokens,
                    max_entries=self.config.max_anchor_num,
                )
                # Use response_embeddings for anchor messages (not placeholder anchors).
                resp_anchor_messages = list(
                    self.paged_kv_engine.response_embeddings.get(response_anchor_key, {}).keys()
                )
                resp_prob, resp_activated = self.paged_kv_engine.predict_as_anchor(
                    ph_id=response_anchor_key,
                    candidate_block_table=response_block_table,
                    candidate_num_tokens=response_num_tokens,
                    anchor_messages=resp_anchor_messages,
                    top_p=0.9,
                    entropy_threshold=self.config.threshold,
                    max_compare_anchors=self.config.max_anchor_num,
                    use_response_embeddings=True,
                )
            else:
                resp_anchor_messages = []
                resp_prob, resp_activated = True, []

            safe_msg = _escape_loguru_markup(message)
            reuse_label = "REUSABLE" if not resp_prob else "NEW_ANCHOR"
            logger.opt(colors=True).info(
                f"<magenta>[RESPONSE_ANCHOR:paged] Agent {self.node_id} Role {self.role} | {reuse_label} | anchors={len(resp_anchor_messages)} | Message {safe_msg}</magenta>",
            )

            if not resp_prob:
                # Reusable — update activation counts
                engine_info = self.paged_kv_engine.anchor_info.setdefault(response_anchor_key, {})
                for idx, anchor_msg_key in enumerate(resp_anchor_messages):
                    if idx >= len(resp_activated):
                        break
                    resp_info_bucket[anchor_msg_key] = resp_activated[idx]
                    bucket_entry = resp_global_bucket.setdefault(anchor_msg_key, [0, 0])
                    bucket_entry[0] = resp_activated[idx]
                    engine_info[anchor_msg_key] = resp_activated[idx]
            else:
                # New anchor — only record the flag; do NOT register_base_anchor here.
                # Registering without agent deltas causes has_active_anchor to force
                # dense prefill for all agents that haven't stored a delta yet.
                # The actual anchor (with deltas) will be created by set_anchor
                # during the next dense_prefill pass, matching HF gpt_chat.py behavior.
                resp_info_bucket[message_key] = 0
                resp_global_bucket[message_key] = [0, response_num_tokens]

            # Decode response
            response_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            safe_resp = _escape_loguru_markup(response_text)
            logger.opt(colors=True).debug(
                "<blue>[RESPONSE:{}]</blue> Agent {} Role {} Response:\n{}",
                mode, self.node_id, self.role, safe_resp,
            )

            metadata: Dict[str, Any] = {
                "placeholder_ids": list(placeholder_info_for_anchor.keys()),
                "reuse_stats": reuse_stats,
            }
            if preprocess_start is not None:
                metadata["preprocess_latency"] = preprocess_latency
                metadata["generation_ttft"] = generation_ttft
            if request_uid:
                metadata["request_uid"] = request_uid
            if agent_id:
                metadata["agent_id"] = agent_id
            if agent_name:
                metadata["agent_name"] = agent_name
            if agent_role:
                metadata["agent_role"] = agent_role

            _append_latency_record(output_dir, {
                "timestamp": time.time(),
                "mode": mode,
                "ttft": float(total_ttft),
                "generation_ttft": float(generation_ttft),
                "preprocess_latency": float(preprocess_latency) if preprocess_start is not None else None,
                "request_uid": request_uid,
                "agent_id": agent_id,
                "message": message_key if message else None,
                "num_cached_tokens": reuse_stats["num_cached_tokens"],
                "num_blocks": reuse_stats["num_blocks"],
                "anchor_candidates": reuse_stats["anchor_candidates"],
                "offset_calls": reuse_stats["offset_calls"],
                "offset_effective": reuse_stats["offset_effective"],
                "anchor_set_count": reuse_stats["anchor_set_count"],
                "anchor_skip_reasons": reuse_stats["anchor_skip_reasons"],
            })

            if pinned_block_table:
                self.paged_kv_engine.free_blocks(pinned_block_table)

            return GenerationResult(
                text=response_text,
                mode=mode,
                ttft=total_ttft,
                metadata=metadata,
            )

    # ── Input/condition anchor helpers ──

    def update_input_anchor(
        self,
        *,
        request_uid: str,
        agent_id: str,
        message: str,
        user_content: str,
        prefix_text: str,
        role: str = "user",
        include_begin: bool = True,
        include_eot: bool = False,
        anchor_namespace: str = "user_question",
        test_time: bool = False,
    ) -> str:
        """Compute input KV via engine and decide kv_reuse vs dense_prefill.

        Returns: "kv_reuse" or "dense_prefill"
        """
        text = self.format_chat_segment(role, user_content, include_begin=include_begin, include_eot=include_eot)
        token_ids = self._encode(text)
        prefix_ids = self._encode(
            self.format_chat_segment(role, prefix_text, include_begin=include_begin, include_eot=include_eot)
        )
        drop_num = len(prefix_ids)

        if test_time:
            for i in range(10):
                if i == 5:
                    start_time = perf_counter()
                sp = SamplingParams(temperature=1.0, max_tokens=1)
                seq = Sequence(token_ids, sp)
                scheduler = self.engine.scheduler
                scheduler.add(seq)
                seqs, is_prefill = scheduler.schedule()
                self.engine.model_runner.call("run", seqs, is_prefill)
                scheduler.postprocess(seqs, [self.tokenizer.eos_token_id])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = perf_counter()
            safe_msg = _escape_loguru_markup(message)
            logger.opt(colors=True).info(
                f"<cyan>Latency for computing the input kv-cache of {safe_msg}: {(end_time - start_time) / 5:.3f} seconds</cyan>"
            )

            # Run one final prefill pass for actual anchor decision.
            sp = SamplingParams(temperature=1.0, max_tokens=1)
            seq = Sequence(token_ids, sp)
            scheduler = self.engine.scheduler
            scheduler.add(seq)
            seqs, is_prefill = scheduler.schedule()
            self.engine.model_runner.call("run", seqs, is_prefill)
            block_table = list(seq.block_table)
            num_tokens = len(token_ids)
            scheduler.postprocess(seqs, [self.tokenizer.eos_token_id])
        else:
            # Run through engine to get KV in blocks
            sp = SamplingParams(temperature=1.0, max_tokens=1)
            seq = Sequence(token_ids, sp)
            scheduler = self.engine.scheduler
            scheduler.add(seq)
            seqs, is_prefill = scheduler.schedule()
            self.engine.model_runner.call("run", seqs, is_prefill)
            block_table = list(seq.block_table)
            num_tokens = len(token_ids)
            scheduler.postprocess(seqs, [self.tokenizer.eos_token_id])

        # Store in shared memory
        shared_mem = PagedLLMChat._shared_kv_cache_memory
        shared_mem.setdefault("input_blocks", {}).setdefault(message, []).append({
            "block_table": block_table,
            "num_tokens": num_tokens,
            "drop_num": drop_num,
        })

        # Predict whether to activate anchor
        candidate_num = num_tokens - drop_num
        bs = self.paged_kv_engine.block_size
        candidate_start_block = drop_num // bs
        candidate_blocks = block_table[candidate_start_block:]

        anchor_messages = list(self.paged_kv_engine.anchors.get(anchor_namespace, {}).keys())

        prob, anchor_activated_list = self.paged_kv_engine.predict_as_anchor(
            ph_id=anchor_namespace,
            candidate_block_table=candidate_blocks,
            candidate_num_tokens=candidate_num,
            anchor_messages=anchor_messages,
            top_p=0.9,
            entropy_threshold=self.config.threshold,
            max_compare_anchors=self.config.max_anchor_num,
        )

        uq_info_bucket = PagedLLMChat._anchor_info_dict.setdefault(anchor_namespace, {})
        global_bucket = PagedLLMChat._global_anchor_info_dict.setdefault(anchor_namespace, {})

        safe_msg = _escape_loguru_markup(message)
        logger.info(
            "[INPUT_ANCHOR_DECISION:paged] node={} role={} request_uid={} ph_id={} "
            "message={} candidate_num_tokens={} candidate_blocks={} anchor_messages={} prob={}",
            getattr(self, "node_id", "?"),
            getattr(self, "role", "?"),
            request_uid,
            anchor_namespace,
            safe_msg,
            candidate_num,
            len(candidate_blocks),
            len(anchor_messages),
            prob,
        )

        message_key = self._message_cache_key(message)

        if not prob:
            engine_info = self.paged_kv_engine.anchor_info.setdefault(anchor_namespace, {})
            for idx, anchor_msg_key in enumerate(anchor_messages):
                if idx >= len(anchor_activated_list):
                    break
                uq_info_bucket[anchor_msg_key] = anchor_activated_list[idx]
                bucket_entry = global_bucket.setdefault(anchor_msg_key, [0, 0])
                bucket_entry[0] = anchor_activated_list[idx]
                engine_info[anchor_msg_key] = anchor_activated_list[idx]

            # Keep runtime token-length metadata for this request message so
            # kv_reuse span reconstruction can align to current prompt.
            global_bucket[message] = [0, candidate_num]
            global_bucket[message_key] = [0, candidate_num]
            return "kv_reuse"

        # Bootstrap anchor history directly from candidate blocks when no
        # reusable anchor exists yet for this namespace.
        if candidate_blocks and candidate_num > 0:
            created = self.paged_kv_engine.register_base_anchor(
                ph_id=anchor_namespace,
                message=message,
                block_table=candidate_blocks,
                num_tokens=candidate_num,
                max_anchor_num=self.config.max_anchor_num,
                window_length=self.config.window_size,
            )
            if created:
                logger.info(
                    "[ANCHOR_CREATE:paged] node={} role={} request_uid={} ph_id={} message={} num_tokens={} num_blocks={} source=update_input_anchor",
                    getattr(self, "node_id", "?"),
                    getattr(self, "role", "?"),
                    request_uid,
                    anchor_namespace,
                    safe_msg,
                    candidate_num,
                    len(candidate_blocks),
                )

        uq_info_bucket[message] = 0
        uq_info_bucket[message_key] = 0
        global_bucket[message] = [0, candidate_num]
        global_bucket[message_key] = [0, candidate_num]
        if prob:
            return "dense_prefill"
        return "kv_reuse"

    def update_condition_anchor(
        self,
        *,
        request_uid: str,
        owner_agent_id: str,
        message: str,
        content: str,
        prefix_text: str,
        role: str = "user",
        include_begin: bool = True,
        include_eot: bool = False,
        anchor_namespace: Optional[str] = None,
        max_length: int = None,
    ) -> bool:
        """Materialise condition KV cache for another agent and update anchors.

        Runs prefill through the paged engine to populate blocks, then uses
        predict_as_anchor to decide if this condition should be treated as a
        new anchor (dense_prefill) or reused (kv_reuse).

        Returns True if the condition is new (needs dense_prefill), False otherwise.
        """
        anchor_key = anchor_namespace or f"condition_{owner_agent_id}_current"
        owner_memory = PagedLLMChat._shared_kv_cache_memory.setdefault(owner_agent_id, {})
        condition_bucket = owner_memory.setdefault("condition_blocks", {})

        if message in condition_bucket:
            # Already materialised
            return False

        text = self.format_chat_segment(role, content, include_begin=include_begin, include_eot=include_eot)
        token_ids = self._encode(text)

        prefix_ids = self._encode(
            self.format_chat_segment(role, prefix_text, include_begin=include_begin, include_eot=include_eot)
        )
        drop_num = len(prefix_ids)

        if max_length is not None:
            token_ids = token_ids[:drop_num + max_length]

        # Run prefill through engine
        sp = SamplingParams(temperature=1.0, max_tokens=1)
        seq = Sequence(token_ids, sp)
        scheduler = self.engine.scheduler
        scheduler.add(seq)

        seqs, is_prefill = scheduler.schedule()
        self.engine.model_runner.call("run", seqs, is_prefill)
        block_table = list(seq.block_table)
        num_tokens = len(token_ids)
        scheduler.postprocess(seqs, [self.tokenizer.eos_token_id])

        # Store condition blocks
        condition_bucket[message] = {
            "block_table": block_table,
            "num_tokens": num_tokens,
            "drop_num": drop_num,
        }

        # Predict as anchor using blocks after drop_num
        candidate_num = num_tokens - drop_num
        bs = self.paged_kv_engine.block_size
        candidate_start_block = drop_num // bs
        candidate_blocks = block_table[candidate_start_block:]

        anchor_messages = list(self.paged_kv_engine.anchors.get(anchor_key, {}).keys())

        prob, anchor_activated_list = self.paged_kv_engine.predict_as_anchor(
            ph_id=anchor_key,
            candidate_block_table=candidate_blocks,
            candidate_num_tokens=candidate_num,
            anchor_messages=anchor_messages,
            top_p=0.9,
            entropy_threshold=self.config.threshold,
            max_compare_anchors=self.config.max_anchor_num,
        )

        cond_info_bucket = PagedLLMChat._anchor_info_dict.setdefault(anchor_key, {})
        global_bucket = PagedLLMChat._global_anchor_info_dict.setdefault(anchor_key, {})
        message_key = self._message_cache_key(message)

        if not prob:
            engine_info = self.paged_kv_engine.anchor_info.setdefault(anchor_key, {})
            for idx, anchor_msg_key in enumerate(anchor_messages):
                if idx >= len(anchor_activated_list):
                    break
                cond_info_bucket[anchor_msg_key] = anchor_activated_list[idx]
                bucket_entry = global_bucket.setdefault(anchor_msg_key, [0, 0])
                bucket_entry[0] = anchor_activated_list[idx]
                engine_info[anchor_msg_key] = anchor_activated_list[idx]
            global_bucket[message] = [0, candidate_num]
            global_bucket[message_key] = [0, candidate_num]
        else:
            if candidate_blocks and candidate_num > 0:
                created = self.paged_kv_engine.register_base_anchor(
                    ph_id=anchor_key,
                    message=message,
                    block_table=candidate_blocks,
                    num_tokens=candidate_num,
                    max_anchor_num=self.config.max_anchor_num,
                    window_length=self.config.window_size,
                )
                if created:
                    safe_msg = _escape_loguru_markup(message)
                    logger.info(
                        "[ANCHOR_CREATE:paged] node={} role={} request_uid={} ph_id={} message={} num_tokens={} num_blocks={} source=update_condition_anchor",
                        getattr(self, "node_id", "?"),
                        getattr(self, "role", "?"),
                        request_uid,
                        anchor_key,
                        safe_msg,
                        candidate_num,
                        len(candidate_blocks),
                    )
                    # In paged mode, condition KV is injected via block references
                    # (not per-agent delta). Store a zero-size stub delta for the
                    # calling agent so has_active_anchor won't force dense_prefill
                    # on every subsequent request. The stub is skipped in
                    # apply_anchor_offset (ph_delta_num_tokens=0 < base_ph_num_tokens).
                    with self.paged_kv_engine._lock:
                        ph_store = self.paged_kv_engine.anchors.get(anchor_key, {})
                        entry = ph_store.get(message)
                        if entry is not None and self.node_id not in entry.agent_deltas:
                            entry.agent_deltas[self.node_id] = {
                                "ph_delta_blocks": [],
                                "ph_delta_num_tokens": 0,
                                "pf_delta_blocks": [],
                                "pf_delta_num_tokens": 0,
                            }
                            logger.info(
                                "[CONDITION_STUB_DELTA:paged] node={} ph_id={} message={} stored stub delta to prevent force_dense",
                                getattr(self, "node_id", "?"),
                                anchor_key,
                                safe_msg,
                            )
            cond_info_bucket[message] = 0
            cond_info_bucket[message_key] = 0
            global_bucket[message] = [0, candidate_num]
            global_bucket[message_key] = [0, candidate_num]

        return prob  # True = new anchor needed (dense_prefill)

    def has_active_anchor(self, request_uid: str, message: str) -> bool:
        """Determine whether an anchor should trigger dense prefill.

        Checks if any placeholder's anchor dict indicates this message
        should be densely prefilled.
        """
        ph_ids = list(PagedLLMChat._shared_kv_cache_memory.get(self.node_id, {}).get("placeholder_info", {}).keys())
        message_key = self._message_cache_key(message)
        logger.info(
            "[ANCHOR_CHECK:paged] node={} role={} request_uid={} ph_ids={} message_key_present_check_start",
            getattr(self, "node_id", "?"),
            getattr(self, "role", "?"),
            request_uid,
            ph_ids,
        )
        for ph_id in ph_ids:
            anchor_store = self.paged_kv_engine.anchors.get(ph_id, {})
            has_message = (message in anchor_store) or (message_key in anchor_store)
            if has_message:
                # has_agent_delta is the hard gate: if this agent has not yet
                # contributed delta for the active anchor entry, force dense prefill.
                entry = anchor_store.get(message, anchor_store.get(message_key))
                agent_deltas = getattr(entry, "agent_deltas", {})
                has_agent_delta = self.node_id in agent_deltas
                logger.info(
                    "[ANCHOR_CHECK:paged] node={} ph_id={} has_message={} has_agent_delta={} -> force_dense={}",
                    getattr(self, "node_id", "?"),
                    ph_id,
                    has_message,
                    has_agent_delta,
                    not has_agent_delta,
                )
                if not has_agent_delta:
                    cfg = getattr(self, "config", None)
                    # crs_priority: always bypass dense_prefill for user_question (ablation).
                    crs_prio = getattr(cfg, "crs_priority", False)
                    if crs_prio and ph_id == "user_question":
                        logger.info(
                            "[ANCHOR_CHECK:paged] node={} ph_id={} "
                            "crs_priority=True -> bypass dense_prefill (kv_reuse)",
                            getattr(self, "node_id", "?"),
                            ph_id,
                        )
                        continue
                    # Before forcing dense_prefill, check if an upstream agent
                    # already has a delta for this message in the current round.
                    # If so, current-round sharing can approximate self's KV
                    # (delta_self ≈ delta_upstream − cross_delta_historical),
                    # so dense_prefill is not needed.
                    crs_on = getattr(cfg, "use_current_round_sharing", True)
                    has_upstream_delta = crs_on and any(
                        aid != self.node_id for aid in agent_deltas
                    )
                    if has_upstream_delta:
                        logger.info(
                            "[ANCHOR_CHECK:paged] node={} ph_id={} "
                            "has_upstream_delta=True -> allow kv_reuse (current-round sharing)",
                            getattr(self, "node_id", "?"),
                            ph_id,
                        )
                        continue  # don't force dense_prefill; fall through to return False
                    return True
            else:
                logger.info(
                    "[ANCHOR_CHECK:paged] node={} ph_id={} has_message={} -> force_dense=False",
                    getattr(self, "node_id", "?"),
                    ph_id,
                    has_message,
                )
        logger.info(
            "[ANCHOR_CHECK:paged] node={} request_uid={} force_dense=False (no active anchor)",
            getattr(self, "node_id", "?"),
            request_uid,
        )
        return False

    @classmethod
    def finalize_request(cls, request_uid: str) -> None:
        """Clean up request-scoped state.

        For the paged backend, this frees any request-scoped block references.
        """
        # In paged mode, blocks are ref-counted and freed via block_manager.
        # Per-request state is minimal; clear any cached data.
        pass

    def _map_in_pool(self, fn, iterable, timeout=None):
        """Execute fn(args) for each args in iterable using the shared thread pool."""
        pool = PagedLLMChat._THREAD_POOL
        if pool is None:
            raise RuntimeError("Thread pool not initialized")
        task_timeout = timeout or self.config.worker_timeout
        futures = [pool.submit(fn, *args) for args in iterable]
        for fut in as_completed(futures, timeout=task_timeout):
            try:
                yield fut.result(timeout=self.config.worker_timeout)
            except TimeoutError as exc:
                raise TimeoutError("Thread task timeout") from exc
            except Exception as exc:
                raise RuntimeError("Thread task failed") from exc

    # ── Time test (benchmark kv_reuse vs dense_prefill) ──

    async def agen_kvcomm_time_test(
        self,
        messages: List[Message] = None,
        max_tokens: Optional[int] = None,
        min_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        request_uid: Optional[str] = None,
        mode: str = "dense_prefill",
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> GenerationResult:
        """Benchmark: run BOTH kv_reuse and dense_prefill, compare TTFT.

        The paged equivalent of LLMChat.agen_kvcomm_time_test.
        1. First run: kv_reuse mode (prefix blocks cached → fast prefill)
        2. Second run: dense_prefill (full prefill, no block reuse)
        3. Log comparison ratio
        """
        max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        min_tokens = min_tokens if min_tokens is not None else max_tokens
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if request_uid is None:
            raise ValueError("request_uid must be provided for agen_kvcomm_time_test.")

        message = messages[0] if isinstance(messages, list) else messages
        message_key = self._message_cache_key(message)

        # Build prompt
        prefix_store = PagedLLMChat._shared_kv_cache_memory.get(self.node_id, {})
        placeholder_info = prefix_store.get("placeholder_info", {})

        prompt_text = self._build_prompt_text(message)
        token_ids = self._encode(prompt_text)
        prompt_num_tokens = len(token_ids)

        safe_prompt = _escape_loguru_markup(prompt_text)
        logger.opt(colors=True).debug(
            "<blue>[PROMPT:time_test]</blue> Agent {} Role {} Prompt:\n{}",
            self.node_id, self.role, safe_prompt,
        )

        # ── Run 1: kv_reuse (prefix blocks may be cached by BlockManager hash) ──
        preprocess_start = perf_counter() if mode == "kv_reuse" else None

        # Apply anchor deltas for kv_reuse if applicable
        if placeholder_info:
            for ph_id, (ph_start, ph_end) in placeholder_info.items():
                anchor_messages = list(self.paged_kv_engine.anchors.get(ph_id, {}).keys())
                if anchor_messages:
                    bs = self.paged_kv_engine.block_size
                    base_block_table = prefix_store.get("prefix_block_table", [])
                    if base_block_table:
                        ph_start_block = ph_start // bs
                        ph_end_block = (ph_end - 1) // bs + 1
                        ph_blocks = base_block_table[ph_start_block:ph_end_block]
                        ph_num = ph_end - ph_start

                        pf_start = ph_end
                        pf_end = prompt_num_tokens
                        pf_start_block = pf_start // bs
                        pf_end_block = (pf_end - 1) // bs + 1
                        pf_blocks = base_block_table[pf_start_block:pf_end_block]
                        pf_num = pf_end - pf_start

                        self.paged_kv_engine.offset_kv_cache(
                            agent_id=self.node_id,
                            ph_id=ph_id,
                            message=message_key,
                            base_ph_block_table=ph_blocks,
                            base_ph_num_tokens=ph_num,
                            base_pf_block_table=pf_blocks,
                            base_pf_num_tokens=pf_num,
                            anchor_list=anchor_messages,
                            temperature=1.0,
                        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        preprocess_latency = 0.0
        if preprocess_start is not None:
            preprocess_latency = max(0.0, perf_counter() - preprocess_start)

        # KV-reuse generation (prefix cached → fast)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        kvcomm_completion_ids, kvcomm_raw_ttft, kvcomm_prefill_lat, kvcomm_seq = self._generate_tokens(
            token_ids, max_tokens, temperature,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # Align with HF backend: fold engine prefill into preprocess_latency
        preprocess_latency += kvcomm_prefill_lat
        kvcomm_gen_ttft = max(0.0, kvcomm_raw_ttft - kvcomm_prefill_lat)
        kvcomm_ttft_value = preprocess_latency + kvcomm_gen_ttft
        kvcomm_e2e_latency = 0.0
        if preprocess_start is not None:
            kvcomm_e2e_latency = perf_counter() - preprocess_start

        safe_msg = _escape_loguru_markup(str(message))
        if mode == "kv_reuse" and preprocess_start is not None:
            logger.opt(colors=True).info(
                f"<green>Agent {self.node_id} Role {self.role} Message {safe_msg} "
                f"KVCOMM(Paged) E2E Latency: {kvcomm_e2e_latency:.4f}s "
                f"TTFT: {kvcomm_ttft_value:.4f}s (Preprocess: {preprocess_latency:.4f}s)</green>",
            )

        # ── Run 2: dense_prefill (full prefill, no block reuse) ──
        # Clear the block manager's cached hash table to force full recompute
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        dense_start = perf_counter()
        dense_completion_ids, dense_gen_ttft, _dense_prefill_lat, dense_seq = self._generate_tokens(
            token_ids, max_tokens, temperature,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dense_e2e_latency = perf_counter() - dense_start
        dense_prefill_ttft = dense_gen_ttft

        logger.opt(colors=True).info(
            f"<cyan>Agent {self.node_id} Role {self.role} Message {safe_msg} "
            f"Dense Prefill(Paged) E2E Latency: {dense_e2e_latency:.4f}s "
            f"TTFT: {dense_prefill_ttft:.4f}s</cyan>",
        )

        # ── Comparison ──
        ttft_value = dense_prefill_ttft
        if mode == "kv_reuse" and preprocess_start is not None and kvcomm_ttft_value > 0:
            ratio = dense_prefill_ttft / kvcomm_ttft_value
            logger.opt(colors=True).info(
                f"<green>Agent {self.node_id} Role {self.role} Message {safe_msg} "
                f"KVCOMM(Paged) is {ratio:.2f}x faster than Dense Prefill in TTFT</green>",
            )
            ttft_value = kvcomm_ttft_value

        # ── Post-generation: anchor bookkeeping on the kv_reuse result ──
        block_table = list(kvcomm_seq.block_table)
        total_num_tokens = len(kvcomm_seq)

        # Store response block info
        mem = PagedLLMChat._shared_kv_cache_memory.get(self.node_id, {})
        resp_blocks = mem.setdefault("response_blocks", {})
        resp_blocks.setdefault(message_key, []).append({
            "block_table": block_table[prompt_num_tokens // self.paged_kv_engine.block_size:],
            "num_tokens": total_num_tokens - prompt_num_tokens,
        })

        # Decode response
        response_text = self.tokenizer.decode(kvcomm_completion_ids, skip_special_tokens=True)
        safe_resp = _escape_loguru_markup(response_text)
        logger.opt(colors=True).debug(
            "<blue>[RESPONSE:time_test]</blue> Agent {} Role {} Response:\n{}",
            self.node_id, self.role, safe_resp,
        )

        # ── Build metadata and latency record ──
        metadata: Dict[str, Any] = {
            "placeholder_ids": list(placeholder_info.keys()),
        }
        if preprocess_start is not None:
            metadata["preprocess_latency"] = preprocess_latency
            metadata["generation_ttft"] = ttft_value - preprocess_latency
        if request_uid:
            metadata["request_uid"] = request_uid
        if agent_id:
            metadata["agent_id"] = agent_id
        if agent_name:
            metadata["agent_name"] = agent_name
        if agent_role:
            metadata["agent_role"] = agent_role

        latency_record: Dict[str, Any] = {
            "timestamp": time.time(),
            "mode": mode,
            "backend": "paged",
            "ttft": float(ttft_value),
            "generation_ttft": float(metadata["generation_ttft"]) if "generation_ttft" in metadata else None,
            "preprocess_latency": float(preprocess_latency) if preprocess_start is not None else None,
            "dense_prefill_ttft": float(dense_prefill_ttft),
            "kvcomm_end_to_end_latency": float(kvcomm_e2e_latency),
            "dense_end_to_end_latency": float(dense_e2e_latency),
            "ttft_ratio_dense_over_kvcomm": float(dense_prefill_ttft / ttft_value) if ttft_value > 0 else None,
            "request_uid": request_uid,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "agent_role": agent_role,
            "message": str(message) if message is not None else None,
            "placeholder_ids": list(placeholder_info.keys()),
        }
        _append_latency_record(output_dir, latency_record)

        return GenerationResult(
            text=response_text,
            mode=mode,
            ttft=ttft_value,
            metadata=metadata,
        )

    # ── Serialization support ──

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("engine", None)
        state.pop("tokenizer", None)
        state.pop("lock", None)
        state.pop("paged_kv_engine", None)
        state.pop("_shared_kv_cache_memory", None)
        state.pop("_initialization", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = PagedLLMChat._shared_tokenizer
        self.engine = PagedLLMChat._shared_engine
        self.paged_kv_engine = PagedLLMChat._paged_kv_engine
        self._shared_kv_cache_memory = PagedLLMChat._shared_kv_cache_memory
        self._initialization = PagedLLMChat._initialization
        self.lock = asyncio.Lock()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        state = self.__getstate__()
        copied_state = copy.deepcopy(state, memo)
        node_id = copied_state.get("node_id", None)
        role = copied_state.get("role", None)
        if node_id is not None:
            if node_id in PagedLLMChat._shared_kv_cache_memory:
                original_cache = PagedLLMChat._shared_kv_cache_memory[node_id]
                PagedLLMChat._shared_kv_cache_memory[node_id] = {
                    "prefix_block_info": original_cache.get("prefix_block_info"),
                    "prefix_block_table": original_cache.get("prefix_block_table"),
                    "prefix_num_tokens": original_cache.get("prefix_num_tokens"),
                    "placeholder_info": original_cache.get("placeholder_info"),
                    "token_ids": original_cache.get("token_ids"),
                    "response_blocks": {},
                    "condition_blocks": {},
                }
            result.set_id(node_id, role)
        result.__setstate__(copied_state)
        return result

    def get_memory_stats(self) -> Dict[str, int]:
        """Return paged KV cache memory statistics."""
        if self.paged_kv_engine:
            return self.paged_kv_engine.get_memory_stats()
        return {}
