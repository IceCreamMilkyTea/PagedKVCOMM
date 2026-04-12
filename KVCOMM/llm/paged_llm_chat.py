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
import inspect
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
_RADIX_DEBUG_IO_LOCK = threading.Lock()
_RADIX_TREE_DEBUG_IO_LOCK = threading.Lock()


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


def _resolve_radix_debug_root(target: Optional[Union[str, Path]]) -> Optional[Path]:
    if target is None:
        return None
    path = Path(target)
    if path.suffix:
        path = path.parent
    return path / "debug" / "radix_traces"


def _resolve_radix_tree_debug_root(target: Optional[Union[str, Path]]) -> Optional[Path]:
    if target is None:
        return None
    path = Path(target)
    if path.suffix:
        path = path.parent
    return path / "debug" / "radix_tree"


def _load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
            return loaded if isinstance(loaded, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


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
    _radix_tree_dump_counters: Dict[str, int] = {}
    _shared_model_name: Optional[str] = None
    _shared_backend: str = "paged"
    # Align with non-paged backend: default greedy decoding.
    DEFAULT_TEMPERATURE = 0.0

    def __init__(
        self,
        model_name: str,
        prefix: str = None,
        config: Optional[KVCommConfig] = None,
        paged_backend: str = "paged",
    ):
        self.model_name = model_name
        self.paged_backend = paged_backend or "paged"
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
            needs_reinit = (
                PagedLLMChat._shared_engine is None
                or PagedLLMChat._shared_model_name != self.model_name
                or PagedLLMChat._shared_backend != self.paged_backend
            )
            if needs_reinit:
                if PagedLLMChat._shared_engine is not None:
                    try:
                        PagedLLMChat._shared_engine.exit()
                    except Exception:
                        logger.exception("Failed to exit previous nano-vllm engine cleanly")

                logger.info(
                    "Initializing nano-vllm engine for model: {} backend={}",
                    self.model_name,
                    self.paged_backend,
                )

                PagedLLMChat._shared_engine = LLMEngine(
                    self.model_name,
                    paged_backend=self.paged_backend,
                )
                PagedLLMChat._shared_tokenizer = PagedLLMChat._shared_engine.tokenizer
                PagedLLMChat._shared_model_name = self.model_name
                PagedLLMChat._shared_backend = self.paged_backend

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

    @classmethod
    def _next_radix_tree_dump_index(cls, root: Path) -> int:
        key = str(root)
        with _RADIX_TREE_DEBUG_IO_LOCK:
            next_index = int(cls._radix_tree_dump_counters.get(key, 0)) + 1
            cls._radix_tree_dump_counters[key] = next_index
        return next_index

    @classmethod
    def dump_shared_radix_tree(
        cls,
        target: Optional[Union[str, Path]],
        *,
        label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        token_preview: int = 16,
    ) -> Optional[str]:
        if cls._shared_backend != "radix":
            return None
        root = _resolve_radix_tree_debug_root(target)
        if root is None:
            return None

        engine = cls._shared_engine
        scheduler = getattr(engine, "scheduler", None) if engine is not None else None
        block_manager = getattr(scheduler, "block_manager", None) if scheduler is not None else None

        if label is None:
            label = f"after_call_{cls._next_radix_tree_dump_index(root)}"

        snapshot: Dict[str, Any] = {
            "timestamp": time.time(),
            "label": label,
            "metadata": cls._json_ready(metadata or {}),
            "scheduler_type": type(scheduler).__name__ if scheduler is not None else None,
            "block_manager_type": type(block_manager).__name__ if block_manager is not None else None,
        }
        if block_manager is not None and hasattr(block_manager, "get_radix_tree_snapshot"):
            snapshot["available"] = True
            snapshot.update(
                cls._json_ready(block_manager.get_radix_tree_snapshot(token_preview=token_preview))
            )
        else:
            snapshot["available"] = False
            snapshot["reason"] = "scheduler_has_no_radix_tree"

        index_payload = {
            "timestamp": snapshot["timestamp"],
            "label": label,
            "path": str(root / f"{label}.json"),
            "available": snapshot["available"],
            "scheduler_type": snapshot["scheduler_type"],
            "block_manager_type": snapshot["block_manager_type"],
            "node_count": snapshot.get("node_count"),
            "leaf_count": snapshot.get("leaf_count"),
            "metadata": snapshot["metadata"],
        }

        with _RADIX_TREE_DEBUG_IO_LOCK:
            root.mkdir(parents=True, exist_ok=True)
            with open(root / f"{label}.json", "w", encoding="utf-8") as handle:
                json.dump(snapshot, handle, ensure_ascii=False, indent=2)
            with open(root / "index.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(index_payload, ensure_ascii=False) + "\n")

        return str(root / f"{label}.json")

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

    def _resolve_request_message(self, messages: Any) -> Optional[str]:
        if isinstance(messages, str):
            return messages
        normalised = self._normalise_messages(messages)
        for item in reversed(normalised):
            if item.get("role") == "user":
                return item.get("content", "")
        if normalised:
            return normalised[-1].get("content", "")
        return None

    def _consume_pending_generation_context(self) -> Tuple[Optional[List[Dict[str, str]]], Dict[str, str]]:
        node_id = getattr(self, "node_id", None)
        if node_id is None:
            return None, {}
        memory = self._ensure_agent_memory(node_id)
        pending_messages = memory.pop("pending_full_message", None)
        placeholder_values = memory.pop("pending_placeholder_values", None) or {}
        return pending_messages, placeholder_values

    def _resolve_generation_payload(
        self,
        messages: Any,
        request_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        pending_messages, placeholder_values = self._consume_pending_generation_context()
        resolved_messages = self._normalise_messages(
            pending_messages if pending_messages is not None else messages
        )
        resolved_request_message = request_message or self._resolve_request_message(messages)
        if resolved_request_message and placeholder_values.get("user_question") is None:
            placeholder_values["user_question"] = resolved_request_message
        return {
            "messages": resolved_messages,
            "request_message": resolved_request_message,
            "message_key_source": (
                resolved_request_message if resolved_request_message is not None else messages
            ),
            "placeholder_values": {
                key: value
                for key, value in placeholder_values.items()
                if isinstance(value, str) and value
            },
            "prompt_source": "pending_full_message" if pending_messages is not None else "call_input",
        }

    def _locate_runtime_placeholders(
        self,
        prompt_text: str,
        placeholder_values: Dict[str, str],
        stored_placeholder_info: Dict[str, List[int]],
    ) -> Dict[str, List[int]]:
        runtime_info: Dict[str, List[int]] = {}
        if not prompt_text or not placeholder_values:
            return runtime_info

        for ph_id, value in placeholder_values.items():
            if not value:
                continue
            occurrences: List[Tuple[int, int]] = []
            start = prompt_text.find(value)
            while start != -1:
                end = start + len(value)
                token_start = len(self._encode(prompt_text[:start]))
                token_len = len(self._encode(value))
                if token_len > 0:
                    occurrences.append((token_start, token_start + token_len))
                start = prompt_text.find(value, start + 1)

            if not occurrences:
                continue

            target_start = stored_placeholder_info.get(ph_id, [occurrences[0][0], 0])[0]
            best_start, best_end = min(
                occurrences,
                key=lambda span: abs(span[0] - target_start),
            )
            runtime_info[ph_id] = [best_start, best_end]

        return dict(sorted(runtime_info.items(), key=lambda x: x[1][0], reverse=True))

    @staticmethod
    def _truncate_debug_text(text: Any, limit: int = 240) -> str:
        if text is None:
            return ""
        value = str(text)
        if len(value) <= limit:
            return value
        return value[:limit] + "..."

    @staticmethod
    def _token_block_span(start: int, end: int, block_size: int) -> List[int]:
        start = max(0, int(start))
        end = max(start, int(end))
        start_block = start // block_size
        if end <= start:
            return [start_block, start_block]
        return [start_block, ((end - 1) // block_size) + 1]

    @staticmethod
    def _block_slice_layout(start_token: int, num_tokens: int, block_size: int) -> Dict[str, int]:
        start_token = max(0, int(start_token))
        num_tokens = max(0, int(num_tokens))
        start_block = start_token // block_size
        end_token = start_token + num_tokens
        end_block = start_block if num_tokens <= 0 else ((end_token - 1) // block_size) + 1
        return {
            "start_block": start_block,
            "end_block": end_block,
            "block_offset": start_token - start_block * block_size,
            "num_tokens": num_tokens,
        }

    @staticmethod
    def _prompt_block_count(prompt_num_tokens: int, block_size: int) -> int:
        prompt_num_tokens = max(0, int(prompt_num_tokens))
        block_size = max(1, int(block_size))
        if prompt_num_tokens <= 0:
            return 0
        return ((prompt_num_tokens - 1) // block_size) + 1

    @staticmethod
    def _prompt_block_table_snapshot(
        block_table: List[int],
        prompt_num_tokens: int,
        block_size: int,
    ) -> List[int]:
        prompt_block_count = PagedLLMChat._prompt_block_count(prompt_num_tokens, block_size)
        return list(block_table[:prompt_block_count])

    @staticmethod
    def _slice_fits_block_capacity(
        block_table: List[int],
        start_offset: int,
        num_tokens: int,
        block_size: int,
    ) -> bool:
        return start_offset + num_tokens <= len(block_table) * block_size

    @staticmethod
    def _json_ready(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): PagedLLMChat._json_ready(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [PagedLLMChat._json_ready(v) for v in value]
        if isinstance(value, set):
            return sorted(PagedLLMChat._json_ready(v) for v in value)
        return value

    def _record_radix_debug(
        self,
        *,
        output_dir: Optional[Union[str, Path]],
        request_uid: Optional[str],
        stage: str,
        payload: Dict[str, Any],
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
    ) -> None:
        if self.paged_backend != "radix":
            return
        root = _resolve_radix_debug_root(output_dir)
        if root is None or not request_uid:
            return

        request_dir = root / "requests"
        index_path = root / "index.jsonl"
        summary_path = root / "summary.json"
        request_path = request_dir / f"{request_uid}.json"
        agent_id = str(agent_id or getattr(self, "node_id", "?"))
        agent_name = agent_name or getattr(self, "agent_name", None) or self.__class__.__name__
        agent_role = agent_role or getattr(self, "role", None)

        event = {
            "timestamp": time.time(),
            "request_uid": request_uid,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "agent_role": agent_role,
            "stage": stage,
            "payload": self._json_ready(payload),
        }

        with _RADIX_DEBUG_IO_LOCK:
            request_dir.mkdir(parents=True, exist_ok=True)
            with open(index_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

            request_exists = request_path.exists()
            request_data = _load_json_dict(request_path)
            if not request_data:
                request_data = {"request_uid": request_uid, "agents": {}}
            if payload.get("request_message"):
                request_data["task"] = payload.get("request_message")
            request_data["last_stage"] = stage
            request_data["last_timestamp"] = event["timestamp"]
            agent_bucket = request_data.setdefault("agents", {}).setdefault(
                agent_id,
                {
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "agent_role": agent_role,
                    "events": [],
                },
            )
            agent_bucket["agent_name"] = agent_name
            agent_bucket["agent_role"] = agent_role
            agent_bucket.setdefault("events", []).append(event)
            with open(request_path, "w", encoding="utf-8") as handle:
                json.dump(request_data, handle, ensure_ascii=False, indent=2)

            summary = _load_json_dict(summary_path)
            if not summary:
                summary = {
                    "request_count": 0,
                    "event_count": 0,
                    "stage_counts": {},
                    "fallback_reasons": {},
                    "combined_blocks_mismatch_count": 0,
                    "prompt_only_runtime_exceeds_template_count": 0,
                    "agent_stats": {},
                }
            if not request_exists:
                summary["request_count"] = int(summary.get("request_count", 0)) + 1
            summary["event_count"] = int(summary.get("event_count", 0)) + 1
            stage_counts = summary.setdefault("stage_counts", {})
            stage_counts[stage] = int(stage_counts.get(stage, 0)) + 1

            if stage == "generation_result":
                agent_stats = summary.setdefault("agent_stats", {}).setdefault(
                    agent_id,
                    {
                        "agent_id": agent_id,
                        "agent_name": agent_name,
                        "agent_role": agent_role,
                        "calls": 0,
                        "kv_reuse_calls": 0,
                        "radix_hit_calls": 0,
                        "applied_reuse_calls": 0,
                        "effective_reuse_calls": 0,
                        "prompt_only_runtime_exceeds_template_calls": 0,
                    },
                )
                agent_stats["agent_name"] = agent_name
                agent_stats["agent_role"] = agent_role
                agent_stats["calls"] = int(agent_stats.get("calls", 0)) + 1
                if payload.get("mode") == "kv_reuse":
                    agent_stats["kv_reuse_calls"] = int(agent_stats.get("kv_reuse_calls", 0)) + 1
                if int(payload.get("num_cached_tokens", 0) or 0) > 0:
                    agent_stats["radix_hit_calls"] = int(agent_stats.get("radix_hit_calls", 0)) + 1
                if payload.get("applied_reuse_hit"):
                    agent_stats["applied_reuse_calls"] = int(agent_stats.get("applied_reuse_calls", 0)) + 1
                if payload.get("effective_reuse_hit"):
                    agent_stats["effective_reuse_calls"] = int(agent_stats.get("effective_reuse_calls", 0)) + 1
                if payload.get("prompt_only_runtime_exceeds_template"):
                    agent_stats["prompt_only_runtime_exceeds_template_calls"] = int(
                        agent_stats.get("prompt_only_runtime_exceeds_template_calls", 0)
                    ) + 1
                    summary["prompt_only_runtime_exceeds_template_count"] = int(
                        summary.get("prompt_only_runtime_exceeds_template_count", 0)
                    ) + 1

                fallback_reasons = summary.setdefault("fallback_reasons", {})
                for reason, count in (payload.get("anchor_skip_reasons") or {}).items():
                    fallback_reasons[reason] = int(fallback_reasons.get(reason, 0)) + int(count)
                    if "combined_blocks_mismatch" in reason:
                        summary["combined_blocks_mismatch_count"] = int(
                            summary.get("combined_blocks_mismatch_count", 0)
                        ) + int(count)

            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)

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
        scheduler = self.engine.scheduler
        scheduler_add_params = set(inspect.signature(scheduler.add).parameters)
        supports_prefilled_add = "prefilled_block_table" in scheduler_add_params

        seq = Sequence(
            token_ids,
            sp,
            prefilled_block_table=None if supports_prefilled_add else cached_prefix_block_table,
            num_cached_tokens=0 if supports_prefilled_add else cached_prefix_num_tokens,
        )

        # Add to scheduler
        if supports_prefilled_add and cached_prefix_block_table is not None:
            scheduler.add(
                seq,
                prefilled_block_table=cached_prefix_block_table,
                prefilled_num_cached=cached_prefix_num_tokens,
            )
        else:
            scheduler.add(seq)

        ttft = None
        start_time = perf_counter()
        block_table_snapshot: List[int] = list(seq.block_table)
        num_tokens_snapshot: int = len(seq)
        num_cached_tokens_snapshot: int = int(seq.num_cached_tokens)
        prompt_block_count_snapshot: int = self._prompt_block_count(
            len(token_ids), self.paged_kv_engine.block_size
        )
        prompt_block_table_snapshot: List[int] = self._prompt_block_table_snapshot(
            block_table_snapshot,
            len(token_ids),
            self.paged_kv_engine.block_size,
        )
        pinned_block_table: List[int] = []

        while not seq.is_finished:
            seqs, is_prefill = scheduler.schedule()
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
            num_cached_tokens_snapshot = max(
                num_cached_tokens_snapshot,
                int(seq.num_cached_tokens),
            )
            prompt_block_table_snapshot = self._prompt_block_table_snapshot(
                block_table_snapshot,
                len(token_ids),
                self.paged_kv_engine.block_size,
            )
            scheduler.postprocess(seqs, token_ids_out)

            if ttft is None and not is_prefill:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ttft = perf_counter() - start_time

        if ttft is None:
            ttft = 0.0

        # Persist snapshots for callers that need KV block ranges after generation.
        setattr(seq, "_block_table_snapshot", block_table_snapshot)
        setattr(seq, "_prompt_block_table_snapshot", prompt_block_table_snapshot)
        setattr(seq, "_prompt_block_count_snapshot", prompt_block_count_snapshot)
        setattr(seq, "_num_tokens_snapshot", num_tokens_snapshot)
        setattr(seq, "_num_cached_tokens_snapshot", num_cached_tokens_snapshot)
        setattr(seq, "_pinned_block_table", pinned_block_table)

        return seq.completion_token_ids, ttft, seq

    def _prepare_kv_reuse_prefix_blocks_legacy(
        self,
        *,
        prefix_store: Dict[str, Any],
        placeholder_info: Dict[str, List[int]],
        prompt_num_tokens: int,
        message_key: str,
    ) -> Tuple[Optional[List[int]], int, Dict[str, Any]]:
        """Original PagedKVCOMM offset path preserved for the paged backend."""
        stats: Dict[str, Any] = {
            "anchor_candidates": 0,
            "offset_calls": 0,
            "offset_effective": 0,
            "offset_applied": False,
            "fallback_reason": None,
            "selected_placeholder_id": None,
            "pre_token_span": None,
            "placeholder_token_span": None,
            "suffix_token_span": None,
            "pre_block_span": None,
            "placeholder_block_span": None,
            "suffix_block_span": None,
            "base_block_table_len": 0,
            "base_ph_blocks_len": 0,
            "base_pf_blocks_len": 0,
            "new_ph_blocks_len": 0,
            "new_pf_blocks_len": 0,
            "expected_blocks": 0,
            "combined_blocks": 0,
        }

        base_block_table = list(prefix_store.get("prefix_block_table", []) or [])
        stats["base_block_table_len"] = len(base_block_table)
        if not base_block_table:
            stats["fallback_reason"] = "no_prefix_block_table"
            return None, 0, stats
        if not placeholder_info:
            stats["fallback_reason"] = "no_placeholder_info"
            return None, 0, stats

        valid_ph_items = [
            (ph_id, span)
            for ph_id, span in sorted(placeholder_info.items(), key=lambda x: x[1][0])
            if span[0] >= 0 and span[1] > span[0] and span[1] <= prompt_num_tokens
        ]
        if not valid_ph_items:
            stats["fallback_reason"] = "no_valid_placeholder_span"
            return None, 0, stats

        ph_items = [item for item in valid_ph_items if item[0] == "user_question"]
        if ph_items:
            ph_id, (ph_start, ph_end) = ph_items[0]
        else:
            ph_id, (ph_start, ph_end) = valid_ph_items[-1]
        anchor_messages = list(self.paged_kv_engine.anchors.get(ph_id, {}).keys())
        stats["selected_placeholder_id"] = ph_id
        stats["anchor_candidates"] = len(anchor_messages)
        if not anchor_messages:
            stats["fallback_reason"] = "no_anchor_messages"
            return None, 0, stats

        bs = self.paged_kv_engine.block_size
        ph_start_block = ph_start // bs
        ph_end_block = (ph_end - 1) // bs + 1
        pf_start = ph_end
        pf_end = prompt_num_tokens

        stats["pre_token_span"] = [0, ph_start]
        stats["placeholder_token_span"] = [ph_start, ph_end]
        stats["suffix_token_span"] = [pf_start, prompt_num_tokens]
        stats["pre_block_span"] = [0, ph_start_block]
        stats["placeholder_block_span"] = [ph_start_block, ph_end_block]

        if ph_start_block >= len(base_block_table):
            stats["fallback_reason"] = "placeholder_out_of_base_range"
            return None, 0, stats

        base_ph_blocks = base_block_table[ph_start_block:ph_end_block]
        ph_num = ph_end - ph_start
        pf_num = max(0, pf_end - pf_start)
        if pf_num > 0:
            pf_start_block = pf_start // bs
            pf_end_block = (pf_end - 1) // bs + 1
            base_pf_blocks = base_block_table[pf_start_block:pf_end_block]
            stats["suffix_block_span"] = [pf_start_block, pf_end_block]
        else:
            base_pf_blocks = []
            stats["suffix_block_span"] = [pf_start // bs, pf_start // bs]

        stats["base_ph_blocks_len"] = len(base_ph_blocks)
        stats["base_pf_blocks_len"] = len(base_pf_blocks)

        if ph_num <= 0 or not base_ph_blocks:
            stats["fallback_reason"] = "invalid_placeholder_block_span"
            return None, 0, stats

        stats["offset_calls"] += 1
        new_ph_blocks, _, new_pf_blocks, _ = self.paged_kv_engine.offset_kv_cache(
            agent_id=self.node_id,
            ph_id=ph_id,
            message=message_key,
            base_ph_block_table=base_ph_blocks,
            base_ph_num_tokens=ph_num,
            base_pf_block_table=base_pf_blocks,
            base_pf_num_tokens=pf_num,
            anchor_list=anchor_messages,
            temperature=1.0,
        )

        stats["new_ph_blocks_len"] = len(new_ph_blocks)
        stats["new_pf_blocks_len"] = len(new_pf_blocks)
        if new_ph_blocks != base_ph_blocks or new_pf_blocks != base_pf_blocks:
            stats["offset_effective"] += 1

        pre_blocks = base_block_table[:ph_start_block]
        if pre_blocks:
            self.paged_kv_engine.increment_ref(pre_blocks)

        combined_blocks = pre_blocks + list(new_ph_blocks) + list(new_pf_blocks)
        expected_blocks = (prompt_num_tokens + bs - 1) // bs
        stats["expected_blocks"] = expected_blocks
        stats["combined_blocks"] = len(combined_blocks)
        if len(combined_blocks) != expected_blocks:
            if pre_blocks:
                self.paged_kv_engine.free_blocks(pre_blocks)
            self.paged_kv_engine.free_blocks(list(new_ph_blocks))
            self.paged_kv_engine.free_blocks(list(new_pf_blocks))
            stats["fallback_reason"] = (
                f"combined_blocks_mismatch:{len(combined_blocks)}!={expected_blocks}"
            )
            return None, 0, stats

        stats["offset_applied"] = True
        return combined_blocks, prompt_num_tokens, stats

    def _prepare_kv_reuse_prefix_blocks(
        self,
        *,
        prefix_store: Dict[str, Any],
        placeholder_info: Dict[str, List[int]],
        prompt_num_tokens: int,
        message_key: str,
    ) -> Tuple[Optional[List[int]], int, Dict[str, Any]]:
        """Build pre-injected cached prefix blocks for kv_reuse mode.

        Returns (block_table, cached_tokens, stats).
        """
        if self.paged_backend != "radix":
            return self._prepare_kv_reuse_prefix_blocks_legacy(
                prefix_store=prefix_store,
                placeholder_info=placeholder_info,
                prompt_num_tokens=prompt_num_tokens,
                message_key=message_key,
            )

        stats: Dict[str, Any] = {
            "anchor_candidates": 0,
            "offset_calls": 0,
            "offset_effective": 0,
            "offset_applied": False,
            "fallback_reason": None,
            "selected_placeholder_id": None,
            "pre_token_span": None,
            "placeholder_token_span": None,
            "suffix_token_span": None,
            "pre_block_span": None,
            "placeholder_block_span": None,
            "suffix_block_span": None,
            "base_block_table_len": 0,
            "base_ph_blocks_len": 0,
            "base_pf_blocks_len": 0,
            "new_ph_blocks_len": 0,
            "new_pf_blocks_len": 0,
            "expected_blocks": 0,
            "combined_blocks": 0,
        }

        base_block_table = list(prefix_store.get("prefix_block_table", []) or [])
        stats["base_block_table_len"] = len(base_block_table)
        if not base_block_table:
            stats["fallback_reason"] = "no_prefix_block_table"
            return None, 0, stats
        if not placeholder_info:
            stats["fallback_reason"] = "no_placeholder_info"
            return None, 0, stats

        stored_placeholder_info = prefix_store.get("placeholder_info", {}) or {}
        runtime_span = placeholder_info.get("user_question")
        template_span = stored_placeholder_info.get("user_question")
        if not runtime_span:
            legacy_blocks, legacy_tokens, legacy_stats = self._prepare_kv_reuse_prefix_blocks_legacy(
                prefix_store=prefix_store,
                placeholder_info=placeholder_info,
                prompt_num_tokens=prompt_num_tokens,
                message_key=message_key,
            )
            legacy_stats["fallback_reason"] = legacy_stats.get("fallback_reason") or "no_user_question_placeholder"
            return legacy_blocks, legacy_tokens, legacy_stats
        if not template_span:
            legacy_blocks, legacy_tokens, legacy_stats = self._prepare_kv_reuse_prefix_blocks_legacy(
                prefix_store=prefix_store,
                placeholder_info=placeholder_info,
                prompt_num_tokens=prompt_num_tokens,
                message_key=message_key,
            )
            legacy_stats["fallback_reason"] = legacy_stats.get("fallback_reason") or "no_template_user_question_placeholder"
            return legacy_blocks, legacy_tokens, legacy_stats

        ph_id = "user_question"
        ph_start, ph_end = runtime_span
        if ph_start < 0 or ph_end <= ph_start or ph_end > prompt_num_tokens:
            stats["fallback_reason"] = "no_valid_user_question_span"
            return None, 0, stats

        bs = self.paged_kv_engine.block_size
        expected_blocks = (prompt_num_tokens + bs - 1) // bs
        stats["expected_blocks"] = expected_blocks
        if expected_blocks > len(base_block_table):
            stats["fallback_reason"] = "runtime_prompt_exceeds_template_blocks"
            return None, 0, stats

        anchor_messages = list(self.paged_kv_engine.anchors.get(ph_id, {}).keys())
        stats["selected_placeholder_id"] = ph_id
        stats["anchor_candidates"] = len(anchor_messages)
        if not anchor_messages:
            stats["fallback_reason"] = "no_anchor_messages"
            return None, 0, stats

        runtime_ph = self._block_slice_layout(ph_start, ph_end - ph_start, bs)
        pf_start = ph_end
        pf_num = max(0, prompt_num_tokens - pf_start)
        runtime_pf = self._block_slice_layout(pf_start, pf_num, bs)
        base_ph_start, base_ph_end = template_span
        base_pf_start = base_ph_end
        base_ph = self._block_slice_layout(base_ph_start, runtime_ph["num_tokens"], bs)
        base_pf = self._block_slice_layout(base_pf_start, runtime_pf["num_tokens"], bs)

        stats["pre_token_span"] = [0, ph_start]
        stats["placeholder_token_span"] = [ph_start, ph_end]
        stats["suffix_token_span"] = [pf_start, prompt_num_tokens]
        stats["pre_block_span"] = [0, runtime_ph["start_block"]]
        stats["placeholder_block_span"] = [runtime_ph["start_block"], runtime_ph["end_block"]]
        stats["suffix_block_span"] = [runtime_pf["start_block"], runtime_pf["end_block"]]

        if runtime_ph["start_block"] >= len(base_block_table):
            stats["fallback_reason"] = "placeholder_out_of_base_range"
            return None, 0, stats

        pre_blocks = base_block_table[:runtime_ph["start_block"]]
        base_tail_block_table = base_block_table[runtime_ph["start_block"]:]
        base_tail_num_tokens = int(prefix_store.get("prefix_num_tokens", 0)) - runtime_ph["start_block"] * bs
        base_pre_num_tokens = ph_start - runtime_ph["start_block"] * bs
        if base_tail_num_tokens <= 0 or base_pre_num_tokens < 0:
            stats["fallback_reason"] = "invalid_base_tail_span"
            return None, 0, stats

        base_ph_blocks = base_block_table[base_ph["start_block"]:base_ph["end_block"]]
        base_pf_blocks = base_block_table[base_pf["start_block"]:base_pf["end_block"]]
        stats["base_ph_blocks_len"] = len(base_ph_blocks)
        stats["base_pf_blocks_len"] = len(base_pf_blocks)
        if not base_ph_blocks:
            stats["fallback_reason"] = "base_blocks_out_of_range"
            return None, 0, stats
        if runtime_pf["num_tokens"] > 0 and not base_pf_blocks:
            stats["fallback_reason"] = "base_prefix_blocks_out_of_range"
            return None, 0, stats
        if not self._slice_fits_block_capacity(
            base_ph_blocks,
            base_ph["block_offset"],
            runtime_ph["num_tokens"],
            bs,
        ):
            stats["fallback_reason"] = "runtime_span_exceeds_template_blocks"
            return None, 0, stats
        if runtime_pf["num_tokens"] > 0 and not self._slice_fits_block_capacity(
            base_pf_blocks,
            base_pf["block_offset"],
            runtime_pf["num_tokens"],
            bs,
        ):
            stats["fallback_reason"] = "runtime_prompt_exceeds_template_blocks"
            return None, 0, stats

        input_entries = self._shared_kv_cache_memory.get("input_blocks", {}).get(message_key, [])
        if not input_entries:
            legacy_blocks, legacy_tokens, legacy_stats = self._prepare_kv_reuse_prefix_blocks_legacy(
                prefix_store=prefix_store,
                placeholder_info=placeholder_info,
                prompt_num_tokens=prompt_num_tokens,
                message_key=message_key,
            )
            legacy_stats["fallback_reason"] = legacy_stats.get("fallback_reason") or "no_query_input_blocks"
            return legacy_blocks, legacy_tokens, legacy_stats
        input_entry = input_entries[-1]
        query_block_table = list(input_entry.get("block_table", []) or [])
        query_num_tokens = int(input_entry.get("num_tokens", 0) or 0)
        query_drop_num = int(input_entry.get("drop_num", 0) or 0)
        query_start_block = query_drop_num // bs
        query_block_offset = query_drop_num - query_start_block * bs
        query_blocks = query_block_table[query_start_block:]
        query_num_tokens = max(0, query_num_tokens - query_drop_num)
        if not query_blocks or query_num_tokens <= 0:
            legacy_blocks, legacy_tokens, legacy_stats = self._prepare_kv_reuse_prefix_blocks_legacy(
                prefix_store=prefix_store,
                placeholder_info=placeholder_info,
                prompt_num_tokens=prompt_num_tokens,
                message_key=message_key,
            )
            legacy_stats["fallback_reason"] = legacy_stats.get("fallback_reason") or "invalid_query_input_blocks"
            return legacy_blocks, legacy_tokens, legacy_stats

        stats["offset_calls"] += 1
        if pre_blocks:
            self.paged_kv_engine.increment_ref(pre_blocks)
        new_tail_blocks, new_tail_num_tokens = self.paged_kv_engine.offset_kv_cache_tail(
            agent_id=self.node_id,
            ph_id=ph_id,
            message=message_key,
            base_tail_block_table=base_tail_block_table,
            base_tail_num_tokens=base_tail_num_tokens,
            base_pre_num_tokens=base_pre_num_tokens,
            base_ph_block_table=base_ph_blocks,
            base_ph_num_tokens=runtime_ph["num_tokens"],
            base_ph_block_offset=base_ph["block_offset"],
            base_pf_block_table=base_pf_blocks,
            base_pf_num_tokens=runtime_pf["num_tokens"],
            base_pf_block_offset=base_pf["block_offset"],
            runtime_ph_num_tokens=runtime_ph["num_tokens"],
            runtime_pf_num_tokens=runtime_pf["num_tokens"],
            query_block_table=query_blocks,
            query_num_tokens=query_num_tokens,
            query_block_offset=query_block_offset,
            anchor_list=anchor_messages,
            temperature=1.0,
        )
        if not new_tail_blocks or new_tail_num_tokens <= 0:
            if pre_blocks:
                self.paged_kv_engine.free_blocks(pre_blocks)
            stats["fallback_reason"] = "no_valid_anchor_deltas"
            return None, 0, stats

        combined_blocks = pre_blocks + list(new_tail_blocks)
        stats["new_ph_blocks_len"] = len(new_tail_blocks)
        stats["new_pf_blocks_len"] = 0
        stats["combined_blocks"] = len(combined_blocks)
        if len(combined_blocks) != expected_blocks:
            if pre_blocks:
                self.paged_kv_engine.free_blocks(pre_blocks)
            self.paged_kv_engine.free_blocks(list(new_tail_blocks))
            stats["fallback_reason"] = (
                f"combined_blocks_mismatch:{len(combined_blocks)}!={expected_blocks}"
            )
            return None, 0, stats

        stats["offset_effective"] = 1
        stats["offset_applied"] = True
        actual_cached_tokens = min(
            prompt_num_tokens,
            runtime_ph["start_block"] * bs + new_tail_num_tokens,
        )
        logger.info(
            "[KV_REUSE_CACHED] node={} actual_cached={} prompt_tokens={} ph=[{},{}] pf=[{},{}]",
            getattr(self, "node_id", "?"), actual_cached_tokens, prompt_num_tokens,
            ph_start, ph_end, pf_start, pf_start + runtime_pf["num_tokens"],
        )
        return combined_blocks, actual_cached_tokens, stats

    # ── Prefix preparation (paged version) ──

    async def prepare_prefix_kv_segments(
        self,
        node_id: str,
        prefix: str,
        user_prompt: str,
        *,
        request_uid: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
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
        mem["prefix_block_info"] = segment_block_info
        mem["prefix_block_table"] = full_block_table
        mem["prefix_num_tokens"] = full_num_tokens
        mem["placeholder_info"] = placeholder_info
        mem["token_ids"] = segment_token_ids_list

        PagedLLMChat._initialization[node_id] = True

        bs = self.paged_kv_engine.block_size
        segment_debug = []
        for type_, text, token_ids, start_token, end_token in segments:
            block_span = self._token_block_span(start_token, end_token, bs)
            segment_debug.append(
                {
                    "type": type_,
                    "label": text if type_ == "placeholder" else self._truncate_debug_text(text, 160),
                    "token_span": [start_token, end_token],
                    "token_count": len(token_ids),
                    "block_span": block_span,
                    "block_count": max(0, block_span[1] - block_span[0]),
                }
            )
        self._record_radix_debug(
            output_dir=output_dir,
            request_uid=request_uid,
            stage="prepare_prefix",
            agent_id=node_id,
            agent_role=getattr(self, "role", None),
            payload={
                "request_message": user_prompt,
                "system_prompt": prefix,
                "user_prompt": user_prompt,
                "prompt_text": prompt_text,
                "prompt_tokens": len(full_token_ids),
                "placeholder_info": placeholder_info,
                "prefix_block_table_len": len(full_block_table),
                "prefix_num_tokens": full_num_tokens,
                "segments": segment_debug,
            },
        )

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

        completion_ids, ttft, seq = self._generate_tokens(token_ids, max_tokens, temperature)
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
        output_dir: Optional[Union[str, Path]] = None,
        radix_debug_output_dir: Optional[Union[str, Path]] = None,
    ) -> GenerationResult:
        """Async generation through nano-vllm engine."""
        async with self.lock:
            max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
            if temperature is None:
                temperature = self.DEFAULT_TEMPERATURE

            prompt_text = self._build_prompt_text(messages)
            token_ids = self._encode(prompt_text)
            prompt_num_tokens = len(token_ids)
            request_message = self._resolve_request_message(messages)
            prompt_source = "call_input"

            safe_prompt = _escape_loguru_markup(prompt_text)
            logger.opt(colors=True).debug(
                "<blue>[PROMPT]</blue> Agent {} Role {} Prompt:\n{}",
                getattr(self, "node_id", "?"),
                getattr(self, "role", "?"),
                safe_prompt,
            )

            completion_ids, ttft, seq = self._generate_tokens(token_ids, max_tokens, temperature)
            block_table = list(getattr(seq, "_block_table_snapshot", list(seq.block_table)))
            prompt_block_table = list(
                getattr(
                    seq,
                    "_prompt_block_table_snapshot",
                    self._prompt_block_table_snapshot(
                        block_table,
                        prompt_num_tokens,
                        self.paged_kv_engine.block_size,
                    ),
                )
            )
            prompt_block_count = int(
                getattr(seq, "_prompt_block_count_snapshot", len(prompt_block_table))
            )
            full_block_count = len(block_table)
            num_cached_tokens = int(
                getattr(seq, "_num_cached_tokens_snapshot", getattr(seq, "num_cached_tokens", 0))
            )
            cached_block_count = num_cached_tokens // self.paged_kv_engine.block_size

            # Free pinned blocks that _generate_tokens incremented for snapshot
            pinned = getattr(seq, "_pinned_block_table", [])
            if pinned:
                self.paged_kv_engine.free_blocks(pinned)
            response_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

            safe_resp = _escape_loguru_markup(response_text)
            logger.opt(colors=True).debug(
                "<blue>[RESPONSE]</blue> Agent {} Role {} Response:\n{}",
                getattr(self, "node_id", "?"),
                getattr(self, "role", "?"),
                safe_resp,
            )

            tree_snapshot_file = self.dump_shared_radix_tree(
                radix_debug_output_dir,
                metadata={
                    "request_uid": request_uid,
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "agent_role": agent_role,
                    "mode": "paged",
                    "prompt_tokens": prompt_num_tokens,
                    "num_cached_tokens": num_cached_tokens,
                },
            )
            reuse_stats: Dict[str, Any] = {
                "prompt_tokens": prompt_num_tokens,
                "prompt_block_count": prompt_block_count,
                "full_block_count": full_block_count,
                "num_blocks": full_block_count,
                "num_cached_tokens": num_cached_tokens,
                "cached_block_count": cached_block_count,
                "anchor_candidates": 0,
                "offset_calls": 0,
                "offset_effective": 0,
                "offset_applied": False,
                "applied_reuse_hit": bool(num_cached_tokens > 0),
                "effective_reuse_hit": bool(num_cached_tokens > 0),
                "anchor_set_count": 0,
                "anchor_skip_reasons": {},
                "tree_snapshot_file": tree_snapshot_file,
            }

            metadata: Dict[str, Any] = {
                "reuse_stats": reuse_stats,
                "prompt_source": prompt_source,
                "prompt_tokens": prompt_num_tokens,
                "prompt_block_count": prompt_block_count,
                "full_block_count": full_block_count,
                "input_messages": self._normalise_messages(messages),
                "prompt_text": prompt_text,
                "tree_snapshot_file": tree_snapshot_file,
            }
            if request_uid:
                metadata["request_uid"] = request_uid
            if agent_id:
                metadata["agent_id"] = agent_id
            if agent_name:
                metadata["agent_name"] = agent_name
            if agent_role:
                metadata["agent_role"] = agent_role
            if return_cache:
                metadata["block_table"] = list(seq.block_table)
                metadata["num_tokens"] = len(seq)

            _append_latency_record(output_dir, {
                "timestamp": time.time(),
                "mode": "paged",
                "ttft": float(ttft),
                "request_uid": request_uid,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_role": agent_role,
                "request_message": request_message,
                "prompt_source": prompt_source,
                "prompt_tokens": prompt_num_tokens,
                "prompt_block_count": prompt_block_count,
                "full_block_count": full_block_count,
                "num_blocks": full_block_count,
                "num_cached_tokens": num_cached_tokens,
                "cached_block_count": cached_block_count,
                "tree_snapshot_file": tree_snapshot_file,
            })

            self._record_radix_debug(
                output_dir=radix_debug_output_dir,
                request_uid=request_uid,
                stage="generation_result",
                agent_id=agent_id,
                agent_name=agent_name,
                agent_role=agent_role,
                payload={
                    "request_message": request_message,
                    "mode": "paged",
                    "prompt_source": prompt_source,
                    "prompt_tokens": prompt_num_tokens,
                    "prompt_block_count": prompt_block_count,
                    "full_block_count": full_block_count,
                    "num_cached_tokens": num_cached_tokens,
                    "applied_reuse_hit": bool(num_cached_tokens > 0),
                    "effective_reuse_hit": bool(num_cached_tokens > 0),
                    "prompt_only_runtime_exceeds_template": False,
                    "anchor_skip_reasons": {},
                    "tree_snapshot_file": tree_snapshot_file,
                },
            )

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
        radix_debug_output_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate using the requested strategy with paged attention."""
        resolved_payload = self._resolve_generation_payload(
            message,
            request_message=message,
        )
        anchor_forces_dense = self.has_active_anchor(request_uid, message)
        selected_mode = (
            "dense_prefill"
            if (preferred_mode == "dense_prefill" or anchor_forces_dense)
            else "kv_reuse"
        )
        logger.info(
            "[MODE_DECISION:paged] node={} role={} request_uid={} preferred_mode={} "
            "anchor_forces_dense={} selected_mode={} prompt_source={}",
            getattr(self, "node_id", "?"),
            getattr(self, "role", "?"),
            request_uid,
            preferred_mode,
            anchor_forces_dense,
            selected_mode,
            resolved_payload["prompt_source"],
        )
        if selected_mode == "dense_prefill":
            return await self.generate_with_dense_prefill(
                resolved_payload["messages"],
                max_tokens=max_tokens,
                temperature=temperature,
                request_uid=request_uid,
                agent_id=agent_id,
                agent_name=agent_name,
                agent_role=agent_role,
                output_dir=output_dir,
                radix_debug_output_dir=radix_debug_output_dir,
                request_message=message,
                prompt_source=resolved_payload["prompt_source"],
                placeholder_values=resolved_payload["placeholder_values"],
                **kwargs,
            )
        return await self.generate_with_kv_reuse(
            resolved_payload["messages"],
            max_tokens=max_tokens,
            temperature=temperature,
            request_uid=request_uid,
            agent_id=agent_id,
            agent_name=agent_name,
            agent_role=agent_role,
            output_dir=output_dir,
            radix_debug_output_dir=radix_debug_output_dir,
            request_message=message,
            prompt_source=resolved_payload["prompt_source"],
            placeholder_values=resolved_payload["placeholder_values"],
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
        radix_debug_output_dir: Optional[Union[str, Path]] = None,
        request_message: Optional[str] = None,
        prompt_source: str = "call_input",
        placeholder_values: Optional[Dict[str, str]] = None,
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
                request_message=request_message,
                prompt_source=prompt_source,
                placeholder_values=placeholder_values,
            )
        return await self._generate_paged(
            messages, "kv_reuse",
            max_tokens=max_tokens, temperature=temperature,
            request_uid=request_uid, agent_id=agent_id,
            agent_name=agent_name, agent_role=agent_role,
            output_dir=output_dir,
            radix_debug_output_dir=radix_debug_output_dir,
            request_message=request_message,
            prompt_source=prompt_source,
            placeholder_values=placeholder_values,
            **kwargs,
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
        radix_debug_output_dir: Optional[Union[str, Path]] = None,
        request_message: Optional[str] = None,
        prompt_source: str = "call_input",
        placeholder_values: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate with fresh prefix computation and anchor update."""
        return await self._generate_paged(
            messages, "dense_prefill",
            max_tokens=max_tokens, temperature=temperature,
            request_uid=request_uid, agent_id=agent_id,
            agent_name=agent_name, agent_role=agent_role,
            output_dir=output_dir,
            radix_debug_output_dir=radix_debug_output_dir,
            request_message=request_message,
            prompt_source=prompt_source,
            placeholder_values=placeholder_values,
            **kwargs,
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
        radix_debug_output_dir: Optional[Union[str, Path]] = None,
        request_message: Optional[str] = None,
        prompt_source: str = "call_input",
        placeholder_values: Optional[Dict[str, str]] = None,
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
            preprocess_start = perf_counter()

            resolved_messages = self._normalise_messages(messages)
            message_key_source = (
                request_message if request_message is not None else messages
            )
            message_key = self._message_cache_key(message_key_source)
            reuse_stats: Dict[str, Any] = {
                "mode": mode,
                "placeholder_count": 0,
                "anchor_candidates": 0,
                "offset_calls": 0,
                "offset_effective": 0,
                "offset_applied": False,
                "applied_reuse_hit": False,
                "effective_reuse_hit": False,
                "num_cached_tokens": 0,
                "num_blocks": 0,
                "anchor_set_count": 0,
                "prompt_tokens": 0,
                "prompt_source": prompt_source,
                "anchor_skip_reasons": {},
            }

            def _record_anchor_skip(reason: str) -> None:
                skip_map = reuse_stats["anchor_skip_reasons"]
                skip_map[reason] = int(skip_map.get(reason, 0)) + 1

            # Build prompt
            prefix_store = PagedLLMChat._shared_kv_cache_memory.get(self.node_id, {})
            stored_placeholder_info = prefix_store.get("placeholder_info", {})

            prompt_text = self._build_prompt_text(resolved_messages)
            token_ids = self._encode(prompt_text)
            reuse_stats["prompt_tokens"] = len(token_ids)
            dynamic_placeholder_info, _ = self._locate_placeholder(prompt_text)
            runtime_placeholder_info = self._locate_runtime_placeholders(
                prompt_text,
                placeholder_values or {},
                stored_placeholder_info,
            )

            # kv_reuse should use runtime-aligned spans when available.
            placeholder_info_for_reuse = dict(stored_placeholder_info)
            if runtime_placeholder_info:
                placeholder_info_for_reuse.update(runtime_placeholder_info)
            uq_bucket = PagedLLMChat._global_anchor_info_dict.get("user_question", {})
            request_lookup_value = request_message if request_message is not None else message_key_source
            uq_meta = uq_bucket.get(message_key, uq_bucket.get(request_lookup_value))
            reuse_span_source = "template"
            if "user_question" in runtime_placeholder_info:
                reuse_span_source = "runtime_placeholder_value"
            elif isinstance(uq_meta, (list, tuple)) and len(uq_meta) >= 2:
                try:
                    uq_num_tokens = int(uq_meta[1])
                except (TypeError, ValueError):
                    uq_num_tokens = 0
                if 0 < uq_num_tokens <= len(token_ids):
                    question_text = (placeholder_values or {}).get("user_question", "")
                    question_pos = prompt_text.find(question_text) if question_text else -1
                    if question_pos >= 0:
                        uq_start = len(self._encode(prompt_text[:question_pos]))
                    else:
                        uq_start = len(token_ids) - uq_num_tokens
                    uq_end = min(len(token_ids), uq_start + uq_num_tokens)
                    placeholder_info_for_reuse["user_question"] = [uq_start, uq_end]
                    reuse_span_source = "runtime_input_anchor"
            logger.info(
                "[REUSE_SPAN:paged] node={} role={} request_uid={} source={} has_uq_meta={} prompt_tokens={} uq_span={} prompt_source={}",
                getattr(self, "node_id", "?"),
                getattr(self, "role", "?"),
                request_uid,
                reuse_span_source,
                isinstance(uq_meta, (list, tuple)) and len(uq_meta) >= 2,
                len(token_ids),
                placeholder_info_for_reuse.get("user_question"),
                prompt_source,
            )
            # dense_prefill anchor writes should prefer runtime placeholder spans.
            if dynamic_placeholder_info:
                placeholder_info_for_anchor = dynamic_placeholder_info
                placeholder_source = "dynamic_prompt"
            elif runtime_placeholder_info:
                placeholder_info_for_anchor = runtime_placeholder_info
                placeholder_source = "runtime_resolved_values"
            else:
                # If prompt has no explicit placeholders, only keep template spans
                # that are valid for the current prompt length.
                placeholder_info_for_anchor = {
                    ph_id: span
                    for ph_id, span in stored_placeholder_info.items()
                    if span[0] >= 0 and span[1] > span[0] and span[1] <= len(token_ids)
                }
                placeholder_source = "stored_template_filtered"

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
            offset_stats: Dict[str, Any] = {}
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
                reuse_stats["offset_applied"] = bool(offset_stats.get("offset_applied"))
                if not offset_stats["offset_applied"]:
                    reason = offset_stats.get("fallback_reason", "unknown")
                    _record_anchor_skip(f"kv_reuse_fallback:{reason}")
                    logger.info(
                        "kv_reuse fallback to dense behavior for this request: {}",
                        reason,
                    )
                self._record_radix_debug(
                    output_dir=radix_debug_output_dir,
                    request_uid=request_uid,
                    stage="kv_reuse_prepare",
                    agent_id=agent_id,
                    agent_name=agent_name,
                    agent_role=agent_role,
                    payload={
                        "request_message": request_message,
                        "mode": mode,
                        "prompt_source": prompt_source,
                        "prompt_tokens": len(token_ids),
                        "reuse_span_source": reuse_span_source,
                        "placeholder_source": placeholder_source,
                        "stored_placeholder_info": stored_placeholder_info,
                        "runtime_placeholder_info": runtime_placeholder_info,
                        "placeholder_info_for_reuse": placeholder_info_for_reuse,
                        "selected_placeholder_id": offset_stats.get("selected_placeholder_id"),
                        "pre_token_span": offset_stats.get("pre_token_span"),
                        "placeholder_token_span": offset_stats.get("placeholder_token_span"),
                        "suffix_token_span": offset_stats.get("suffix_token_span"),
                        "pre_block_span": offset_stats.get("pre_block_span"),
                        "placeholder_block_span": offset_stats.get("placeholder_block_span"),
                        "suffix_block_span": offset_stats.get("suffix_block_span"),
                        "base_block_table_len": offset_stats.get("base_block_table_len"),
                        "base_ph_blocks_len": offset_stats.get("base_ph_blocks_len"),
                        "base_pf_blocks_len": offset_stats.get("base_pf_blocks_len"),
                        "new_ph_blocks_len": offset_stats.get("new_ph_blocks_len"),
                        "new_pf_blocks_len": offset_stats.get("new_pf_blocks_len"),
                        "combined_blocks": offset_stats.get("combined_blocks"),
                        "expected_blocks": offset_stats.get("expected_blocks"),
                        "anchor_candidates": offset_stats.get("anchor_candidates"),
                        "offset_calls": offset_stats.get("offset_calls"),
                        "offset_effective": offset_stats.get("offset_effective"),
                        "offset_applied": offset_stats.get("offset_applied"),
                        "fallback_reason": offset_stats.get("fallback_reason"),
                    },
                )

            # Generate through nano-vllm engine
            preprocess_latency = perf_counter() - preprocess_start
            completion_ids, ttft, seq = self._generate_tokens(
                token_ids,
                max_tokens,
                temperature,
                cached_prefix_block_table=cached_prefix_block_table,
                cached_prefix_num_tokens=cached_prefix_num_tokens,
            )
            pinned_block_table = list(getattr(seq, "_pinned_block_table", []) or [])

            total_ttft = ttft + preprocess_latency

            # Block table now contains all KV blocks for this generation
            block_table = list(getattr(seq, "_block_table_snapshot", list(seq.block_table)))
            prompt_block_table = list(
                getattr(
                    seq,
                    "_prompt_block_table_snapshot",
                    self._prompt_block_table_snapshot(
                        block_table,
                        len(token_ids),
                        self.paged_kv_engine.block_size,
                    ),
                )
            )
            prompt_num_tokens = len(token_ids)
            total_num_tokens = int(getattr(seq, "_num_tokens_snapshot", len(seq)))
            prompt_block_count = int(
                getattr(seq, "_prompt_block_count_snapshot", len(prompt_block_table))
            )
            full_block_count = len(block_table)
            reuse_stats["num_cached_tokens"] = int(
                getattr(seq, "_num_cached_tokens_snapshot", getattr(seq, "num_cached_tokens", 0))
            )
            reuse_stats["num_blocks"] = full_block_count
            reuse_stats["prompt_block_count"] = prompt_block_count
            reuse_stats["full_block_count"] = full_block_count
            reuse_stats["prompt_only_runtime_exceeds_template"] = False
            reuse_stats["placeholder_count"] = len(placeholder_info_for_anchor)
            reuse_stats["applied_reuse_hit"] = bool(
                reuse_stats["offset_applied"] or reuse_stats["num_cached_tokens"] > 0
            )
            reuse_stats["effective_reuse_hit"] = bool(
                reuse_stats["num_cached_tokens"] > 0 or reuse_stats["offset_effective"] > 0
            )

            # ── Anchor operations ──
            if mode == "dense_prefill" and placeholder_info_for_anchor:
                # After full prefill, store anchor deltas
                for ph_id, (ph_start, ph_end) in placeholder_info_for_anchor.items():
                    if self.paged_backend != "radix":
                        anchor_messages = list(self.paged_kv_engine.anchors.get(ph_id, {}).keys())
                        reuse_stats["anchor_candidates"] += len(anchor_messages)

                        bs = self.paged_kv_engine.block_size
                        ph_start_block = ph_start // bs
                        ph_end_block = (ph_end - 1) // bs + 1
                        ph_blocks = prompt_block_table[ph_start_block:ph_end_block]
                        ph_num = ph_end - ph_start

                        pf_start = ph_end
                        pf_end = prompt_num_tokens
                        pf_start_block = pf_start // bs
                        pf_end_block = (pf_end - 1) // bs + 1 if pf_end > pf_start else pf_start_block
                        pf_blocks = prompt_block_table[pf_start_block:pf_end_block]
                        pf_num = pf_end - pf_start

                        if ph_num <= 0 or not ph_blocks or ph_start_block >= len(prompt_block_table):
                            _record_anchor_skip("placeholder_blocks_out_of_range")
                            continue
                        if ph_end > prompt_num_tokens:
                            _record_anchor_skip("placeholder_end_exceeds_prompt")
                            continue
                        if pf_num <= 0 or not pf_blocks:
                            _record_anchor_skip("prefix_blocks_out_of_range")
                            continue

                        base_block_table = prefix_store.get("prefix_block_table", [])
                        if not base_block_table:
                            _record_anchor_skip("missing_prefix_block_table")
                            continue

                        base_ph_blocks = base_block_table[ph_start_block:ph_end_block]
                        base_pf_blocks = base_block_table[pf_start_block:pf_end_block]
                        if not base_ph_blocks:
                            _record_anchor_skip("base_blocks_out_of_range")
                            continue
                        if not base_pf_blocks:
                            _record_anchor_skip("base_prefix_blocks_out_of_range")
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
                        )
                        reuse_stats["anchor_set_count"] += 1
                        continue

                    if ph_id != "user_question":
                        _record_anchor_skip("non_user_question_reuse_disabled")
                        continue

                    anchor_messages = list(self.paged_kv_engine.anchors.get(ph_id, {}).keys())
                    reuse_stats["anchor_candidates"] += len(anchor_messages)

                    # Block range for placeholder portion
                    bs = self.paged_kv_engine.block_size
                    ph_num = ph_end - ph_start
                    runtime_ph = self._block_slice_layout(ph_start, ph_num, bs)
                    ph_start_block = runtime_ph["start_block"]
                    ph_end_block = runtime_ph["end_block"]
                    ph_blocks = prompt_block_table[ph_start_block:ph_end_block]

                    # Block range for prefix after placeholder
                    pf_start = ph_end
                    pf_end = prompt_num_tokens
                    pf_num = pf_end - pf_start
                    runtime_pf = self._block_slice_layout(pf_start, pf_num, bs)
                    pf_start_block = runtime_pf["start_block"]
                    pf_end_block = runtime_pf["end_block"]
                    pf_blocks = prompt_block_table[pf_start_block:pf_end_block]

                    # Validate that block ranges are within the actual block table.
                    # placeholder_info positions are from the template prompt; if the
                    # generation prompt is shorter (e.g. just the task string), the
                    # block indices can fall out of range.
                    if ph_num <= 0 or not ph_blocks or ph_start_block >= len(prompt_block_table):
                        _record_anchor_skip("placeholder_blocks_out_of_range")
                        logger.info(
                            "dense_prefill: skipping set_anchor for {} — placeholder blocks "
                            "out of range (ph_start_block={}, block_table_len={})",
                            ph_id, ph_start_block, len(prompt_block_table),
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

                    if pf_num <= 0 or not pf_blocks:
                        _record_anchor_skip("prefix_blocks_out_of_range")
                        logger.info(
                            "dense_prefill: skipping set_anchor for {} — prefix blocks out of range "
                            "(pf_start_block={}, pf_end_block={}, block_table_len={}, pf_num={})",
                            ph_id,
                            pf_start_block,
                            pf_end_block,
                            len(prompt_block_table),
                            pf_num,
                        )
                        continue

                    # We need base blocks - stored in prefix_block_info
                    base_block_table = prefix_store.get("prefix_block_table", [])
                    template_span = stored_placeholder_info.get(ph_id)
                    if not base_block_table:
                        _record_anchor_skip("missing_prefix_block_table")
                        logger.info(
                            "dense_prefill: skipping set_anchor for {} — missing prefix_block_table",
                            ph_id,
                        )
                        continue
                    if not template_span:
                        _record_anchor_skip("missing_template_placeholder_span")
                        continue
                    if len(prompt_block_table) > len(base_block_table):
                        reuse_stats["prompt_only_runtime_exceeds_template"] = True
                        _record_anchor_skip("runtime_span_exceeds_template_blocks")
                        logger.info(
                            "dense_prefill: skipping set_anchor for {} — runtime prompt blocks {} exceed template blocks {}",
                            ph_id,
                            len(prompt_block_table),
                            len(base_block_table),
                        )
                        continue

                    base_ph_start, base_ph_end = template_span
                    base_pf_start = base_ph_end
                    base_ph = self._block_slice_layout(base_ph_start, ph_num, bs)
                    base_pf = self._block_slice_layout(base_pf_start, pf_num, bs)
                    base_ph_blocks = base_block_table[base_ph["start_block"]:base_ph["end_block"]]
                    base_pf_blocks = base_block_table[base_pf["start_block"]:base_pf["end_block"]]

                    if not base_ph_blocks:
                        reuse_stats["prompt_only_runtime_exceeds_template"] = True
                        _record_anchor_skip("base_blocks_out_of_range")
                        logger.info(
                            "dense_prefill: skipping set_anchor for {} — base blocks out of range",
                            ph_id,
                        )
                        continue

                    if pf_num > 0 and not base_pf_blocks:
                        reuse_stats["prompt_only_runtime_exceeds_template"] = True
                        _record_anchor_skip("base_prefix_blocks_out_of_range")
                        logger.info(
                            "dense_prefill: skipping set_anchor for {} — base prefix blocks out of range",
                            ph_id,
                        )
                        continue

                    if not self._slice_fits_block_capacity(
                        base_ph_blocks,
                        base_ph["block_offset"],
                        ph_num,
                        bs,
                    ):
                        reuse_stats["prompt_only_runtime_exceeds_template"] = True
                        _record_anchor_skip("runtime_span_exceeds_template_blocks")
                        logger.info(
                            "dense_prefill: skipping set_anchor for {} — runtime placeholder span exceeds template block capacity",
                            ph_id,
                        )
                        continue

                    if pf_num > 0 and not self._slice_fits_block_capacity(
                        base_pf_blocks,
                        base_pf["block_offset"],
                        pf_num,
                        bs,
                    ):
                        reuse_stats["prompt_only_runtime_exceeds_template"] = True
                        _record_anchor_skip("runtime_prompt_exceeds_template_blocks")
                        logger.info(
                            "dense_prefill: skipping set_anchor for {} — runtime suffix span exceeds template block capacity",
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
                        real_block_offset=runtime_ph["block_offset"],
                        base_block_offset=base_ph["block_offset"],
                        real_prefix_block_offset=runtime_pf["block_offset"],
                        base_prefix_block_offset=base_pf["block_offset"],
                    )
                    reuse_stats["anchor_set_count"] += 1

            elif mode == "kv_reuse" and cached_prefix_block_table:
                logger.debug(
                    "kv_reuse pre-injected cached prefix blocks: cached_tokens={}, blocks={}",
                    cached_prefix_num_tokens,
                    len(cached_prefix_block_table),
                )

            # Store response block info for future reuse
            mem = PagedLLMChat._shared_kv_cache_memory.get(self.node_id, {})
            resp_blocks = mem.setdefault("response_blocks", {})
            resp_blocks.setdefault(message_key, []).append({
                "block_table": block_table[prompt_num_tokens // self.paged_kv_engine.block_size:],
                "num_tokens": total_num_tokens - prompt_num_tokens,
            })

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
                "prompt_source": prompt_source,
                "prompt_tokens": reuse_stats["prompt_tokens"],
                "prompt_block_count": reuse_stats["prompt_block_count"],
                "full_block_count": reuse_stats["full_block_count"],
                "input_messages": resolved_messages,
                "prompt_text": prompt_text,
                "radix_debug": {
                    "reuse_span_source": reuse_span_source,
                    "placeholder_source": placeholder_source,
                    "stored_placeholder_info": stored_placeholder_info,
                    "runtime_placeholder_info": runtime_placeholder_info,
                    "placeholder_info_for_reuse": placeholder_info_for_reuse,
                    "selected_placeholder_id": offset_stats.get("selected_placeholder_id"),
                    "pre_token_span": offset_stats.get("pre_token_span"),
                    "placeholder_token_span": offset_stats.get("placeholder_token_span"),
                    "suffix_token_span": offset_stats.get("suffix_token_span"),
                    "pre_block_span": offset_stats.get("pre_block_span"),
                    "placeholder_block_span": offset_stats.get("placeholder_block_span"),
                    "suffix_block_span": offset_stats.get("suffix_block_span"),
                    "base_block_table_len": offset_stats.get("base_block_table_len"),
                    "base_ph_blocks_len": offset_stats.get("base_ph_blocks_len"),
                    "base_pf_blocks_len": offset_stats.get("base_pf_blocks_len"),
                    "new_ph_blocks_len": offset_stats.get("new_ph_blocks_len"),
                    "new_pf_blocks_len": offset_stats.get("new_pf_blocks_len"),
                    "combined_blocks": offset_stats.get("combined_blocks"),
                    "expected_blocks": offset_stats.get("expected_blocks"),
                    "fallback_reason": offset_stats.get("fallback_reason"),
                    "prompt_only_runtime_exceeds_template": reuse_stats["prompt_only_runtime_exceeds_template"],
                },
            }
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
                "request_uid": request_uid,
                "agent_id": agent_id,
                "message": message_key if message_key_source is not None else None,
                "request_message": request_message,
                "prompt_source": prompt_source,
                "prompt_tokens": reuse_stats["prompt_tokens"],
                "prompt_block_count": reuse_stats["prompt_block_count"],
                "full_block_count": reuse_stats["full_block_count"],
                "num_cached_tokens": reuse_stats["num_cached_tokens"],
                "num_blocks": reuse_stats["num_blocks"],
                "anchor_candidates": reuse_stats["anchor_candidates"],
                "offset_calls": reuse_stats["offset_calls"],
                "offset_effective": reuse_stats["offset_effective"],
                "offset_applied": reuse_stats["offset_applied"],
                "applied_reuse_hit": reuse_stats["applied_reuse_hit"],
                "effective_reuse_hit": reuse_stats["effective_reuse_hit"],
                "prompt_only_runtime_exceeds_template": reuse_stats["prompt_only_runtime_exceeds_template"],
                "anchor_set_count": reuse_stats["anchor_set_count"],
                "anchor_skip_reasons": reuse_stats["anchor_skip_reasons"],
            })

            self._record_radix_debug(
                output_dir=radix_debug_output_dir,
                request_uid=request_uid,
                stage="generation_result",
                agent_id=agent_id,
                agent_name=agent_name,
                agent_role=agent_role,
                payload={
                    "request_message": request_message,
                    "mode": mode,
                    "prompt_source": prompt_source,
                    "prompt_tokens": reuse_stats["prompt_tokens"],
                    "prompt_block_count": reuse_stats["prompt_block_count"],
                    "full_block_count": reuse_stats["full_block_count"],
                    "reuse_span_source": reuse_span_source,
                    "placeholder_source": placeholder_source,
                    "selected_placeholder_id": offset_stats.get("selected_placeholder_id"),
                    "pre_token_span": offset_stats.get("pre_token_span"),
                    "placeholder_token_span": offset_stats.get("placeholder_token_span"),
                    "suffix_token_span": offset_stats.get("suffix_token_span"),
                    "pre_block_span": offset_stats.get("pre_block_span"),
                    "placeholder_block_span": offset_stats.get("placeholder_block_span"),
                    "suffix_block_span": offset_stats.get("suffix_block_span"),
                    "base_block_table_len": offset_stats.get("base_block_table_len"),
                    "base_ph_blocks_len": offset_stats.get("base_ph_blocks_len"),
                    "base_pf_blocks_len": offset_stats.get("base_pf_blocks_len"),
                    "new_ph_blocks_len": offset_stats.get("new_ph_blocks_len"),
                    "new_pf_blocks_len": offset_stats.get("new_pf_blocks_len"),
                    "combined_blocks": offset_stats.get("combined_blocks"),
                    "expected_blocks": offset_stats.get("expected_blocks"),
                    "fallback_reason": offset_stats.get("fallback_reason"),
                    "anchor_candidates": reuse_stats["anchor_candidates"],
                    "offset_calls": reuse_stats["offset_calls"],
                    "offset_effective": reuse_stats["offset_effective"],
                    "offset_applied": reuse_stats["offset_applied"],
                    "num_cached_tokens": reuse_stats["num_cached_tokens"],
                    "applied_reuse_hit": reuse_stats["applied_reuse_hit"],
                    "effective_reuse_hit": reuse_stats["effective_reuse_hit"],
                    "prompt_only_runtime_exceeds_template": reuse_stats["prompt_only_runtime_exceeds_template"],
                    "anchor_skip_reasons": reuse_stats["anchor_skip_reasons"],
                },
            )

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
        radix_debug_output_dir: Optional[Union[str, Path]] = None,
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
        candidate_start_offset = drop_num - candidate_start_block * bs
        candidate_blocks = block_table[candidate_start_block:]

        anchor_messages = list(self.paged_kv_engine.anchors.get(anchor_namespace, {}).keys())

        prob, anchor_activated_list = self.paged_kv_engine.predict_as_anchor(
            ph_id=anchor_namespace,
            candidate_block_table=candidate_blocks,
            candidate_num_tokens=candidate_num,
            anchor_messages=anchor_messages,
            candidate_start_offset=candidate_start_offset,
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
        created_anchor = False

        if not prob:
            for idx, anchor_msg_key in enumerate(anchor_messages):
                if idx >= len(anchor_activated_list):
                    break
                uq_info_bucket[anchor_msg_key] = anchor_activated_list[idx]
                bucket_entry = global_bucket.setdefault(anchor_msg_key, [0, 0])
                bucket_entry[0] = anchor_activated_list[idx]

            # Keep runtime token-length metadata for this request message so
            # kv_reuse span reconstruction can align to current prompt.
            global_bucket[message] = [0, candidate_num]
            global_bucket[message_key] = [0, candidate_num]
            self._record_radix_debug(
                output_dir=radix_debug_output_dir,
                request_uid=request_uid,
                stage="update_input_anchor",
                agent_id=agent_id,
                agent_role=getattr(self, "role", None),
                payload={
                    "request_message": message,
                    "anchor_namespace": anchor_namespace,
                    "message_key": message_key,
                    "candidate_num_tokens": candidate_num,
                    "drop_num": drop_num,
                    "candidate_block_count": len(candidate_blocks),
                    "block_table_len": len(block_table),
                    "anchor_candidates": len(anchor_messages),
                    "anchor_activated_list": anchor_activated_list,
                    "decision": "kv_reuse",
                    "created_base_anchor": False,
                },
            )
            return "kv_reuse"

        # Bootstrap anchor history directly from candidate blocks when no
        # reusable anchor exists yet for this namespace.
        if candidate_blocks and candidate_num > 0:
            created = self.paged_kv_engine.register_base_anchor(
                ph_id=anchor_namespace,
                message=message,
                block_table=candidate_blocks,
                num_tokens=candidate_num,
                start_offset=candidate_start_offset,
            )
            if created:
                created_anchor = True
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
        self._record_radix_debug(
            output_dir=radix_debug_output_dir,
            request_uid=request_uid,
            stage="update_input_anchor",
            agent_id=agent_id,
            agent_role=getattr(self, "role", None),
            payload={
                "request_message": message,
                "anchor_namespace": anchor_namespace,
                "message_key": message_key,
                "candidate_num_tokens": candidate_num,
                "drop_num": drop_num,
                    "candidate_block_count": len(candidate_blocks),
                    "candidate_block_offset": candidate_start_offset,
                    "block_table_len": len(block_table),
                "anchor_candidates": len(anchor_messages),
                "anchor_activated_list": anchor_activated_list,
                "decision": "dense_prefill" if prob else "kv_reuse",
                "created_base_anchor": created_anchor,
            },
        )
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
        owner_agent_role: Optional[str] = None,
        role: str = "user",
        include_begin: bool = True,
        include_eot: bool = False,
        anchor_namespace: Optional[str] = None,
        max_length: int = None,
        radix_debug_output_dir: Optional[Union[str, Path]] = None,
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
        message_key = self._message_cache_key(message)

        if message in condition_bucket:
            # Already materialised
            self._record_radix_debug(
                output_dir=radix_debug_output_dir,
                request_uid=request_uid,
                stage="update_condition_anchor",
                agent_id=owner_agent_id,
                agent_role=owner_agent_role or getattr(self, "role", None),
                payload={
                    "request_message": message,
                    "anchor_namespace": anchor_key,
                    "message_key": message_key,
                    "decision": "reuse_existing_condition",
                    "created_base_anchor": False,
                },
            )
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
        candidate_start_offset = drop_num - candidate_start_block * bs
        candidate_blocks = block_table[candidate_start_block:]

        anchor_messages = list(self.paged_kv_engine.anchors.get(anchor_key, {}).keys())

        prob, anchor_activated_list = self.paged_kv_engine.predict_as_anchor(
            ph_id=anchor_key,
            candidate_block_table=candidate_blocks,
            candidate_num_tokens=candidate_num,
            anchor_messages=anchor_messages,
            candidate_start_offset=candidate_start_offset,
            top_p=0.9,
            entropy_threshold=self.config.threshold,
            max_compare_anchors=self.config.max_anchor_num,
        )

        cond_info_bucket = PagedLLMChat._anchor_info_dict.setdefault(anchor_key, {})
        global_bucket = PagedLLMChat._global_anchor_info_dict.setdefault(anchor_key, {})
        created_anchor = False

        if not prob:
            for idx, anchor_msg_key in enumerate(anchor_messages):
                if idx >= len(anchor_activated_list):
                    break
                cond_info_bucket[anchor_msg_key] = anchor_activated_list[idx]
                bucket_entry = global_bucket.setdefault(anchor_msg_key, [0, 0])
                bucket_entry[0] = anchor_activated_list[idx]
            global_bucket[message] = [0, candidate_num]
            global_bucket[message_key] = [0, candidate_num]
        else:
            if candidate_blocks and candidate_num > 0:
                created = self.paged_kv_engine.register_base_anchor(
                    ph_id=anchor_key,
                    message=message,
                    block_table=candidate_blocks,
                    num_tokens=candidate_num,
                    start_offset=candidate_start_offset,
                )
                if created:
                    created_anchor = True
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
            cond_info_bucket[message] = 0
            cond_info_bucket[message_key] = 0
            global_bucket[message] = [0, candidate_num]
            global_bucket[message_key] = [0, candidate_num]

        self._record_radix_debug(
            output_dir=radix_debug_output_dir,
            request_uid=request_uid,
            stage="update_condition_anchor",
            agent_id=owner_agent_id,
            agent_role=owner_agent_role or getattr(self, "role", None),
            payload={
                "request_message": message,
                "anchor_namespace": anchor_key,
                "message_key": message_key,
                "candidate_num_tokens": candidate_num,
                "drop_num": drop_num,
                "candidate_block_count": len(candidate_blocks),
                "candidate_block_offset": candidate_start_offset,
                "block_table_len": len(block_table),
                "anchor_candidates": len(anchor_messages),
                "anchor_activated_list": anchor_activated_list,
                "decision": "dense_prefill" if prob else "kv_reuse",
                "created_base_anchor": created_anchor,
            },
        )

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
        mode: str = "kv_reuse",
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        radix_debug_output_dir: Optional[Union[str, Path]] = None,
        request_message: Optional[str] = None,
        prompt_source: str = "call_input",
        placeholder_values: Optional[Dict[str, str]] = None,
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

        resolved_messages = self._normalise_messages(messages)
        message_key_source = request_message if request_message is not None else messages
        message_key = self._message_cache_key(message_key_source)

        # Build prompt
        prefix_store = PagedLLMChat._shared_kv_cache_memory.get(self.node_id, {})
        stored_placeholder_info = prefix_store.get("placeholder_info", {})

        prompt_text = self._build_prompt_text(resolved_messages)
        token_ids = self._encode(prompt_text)
        prompt_num_tokens = len(token_ids)
        runtime_placeholder_info = self._locate_runtime_placeholders(
            prompt_text,
            placeholder_values or {},
            stored_placeholder_info,
        )
        placeholder_info = runtime_placeholder_info or {
            ph_id: span
            for ph_id, span in stored_placeholder_info.items()
            if span[0] >= 0 and span[1] > span[0] and span[1] <= prompt_num_tokens
        }

        safe_prompt = _escape_loguru_markup(prompt_text)
        logger.opt(colors=True).debug(
            "<blue>[PROMPT:time_test]</blue> Agent {} Role {} Prompt:\n{}",
            self.node_id, self.role, safe_prompt,
        )

        # ── Run 1: kv_reuse (prefix blocks may be cached by BlockManager hash) ──
        preprocess_start = perf_counter()

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
        preprocess_latency = perf_counter() - preprocess_start

        # KV-reuse generation (prefix cached → fast)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        kvcomm_completion_ids, kvcomm_gen_ttft, kvcomm_seq = self._generate_tokens(
            token_ids, max_tokens, temperature,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        kvcomm_ttft_value = kvcomm_gen_ttft + preprocess_latency
        kvcomm_e2e_latency = perf_counter() - preprocess_start

        safe_msg = _escape_loguru_markup(str(message_key_source))
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
        dense_completion_ids, dense_gen_ttft, dense_seq = self._generate_tokens(
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
        if kvcomm_ttft_value > 0:
            ratio = dense_prefill_ttft / kvcomm_ttft_value
            logger.opt(colors=True).info(
                f"<green>Agent {self.node_id} Role {self.role} Message {safe_msg} "
                f"KVCOMM(Paged) is {ratio:.2f}x faster than Dense Prefill in TTFT</green>",
            )
            ttft_value = kvcomm_ttft_value
        else:
            ttft_value = dense_prefill_ttft

        # ── Post-generation: anchor bookkeeping on the kv_reuse result ──
        block_table = list(
            getattr(kvcomm_seq, "_block_table_snapshot", list(kvcomm_seq.block_table))
        )
        total_num_tokens = len(kvcomm_seq)
        prompt_block_count = int(
            getattr(
                kvcomm_seq,
                "_prompt_block_count_snapshot",
                self._prompt_block_count(prompt_num_tokens, self.paged_kv_engine.block_size),
            )
        )
        full_block_count = len(block_table)

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
            "preprocess_latency": preprocess_latency,
            "generation_ttft": ttft_value - preprocess_latency,
            "prompt_source": prompt_source,
            "prompt_tokens": prompt_num_tokens,
            "prompt_block_count": prompt_block_count,
            "full_block_count": full_block_count,
            "input_messages": resolved_messages,
            "prompt_text": prompt_text,
        }
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
            "generation_ttft": float(metadata["generation_ttft"]),
            "preprocess_latency": float(preprocess_latency),
            "dense_prefill_ttft": float(dense_prefill_ttft),
            "kvcomm_end_to_end_latency": float(kvcomm_e2e_latency),
            "dense_end_to_end_latency": float(dense_e2e_latency),
            "ttft_ratio_dense_over_kvcomm": float(dense_prefill_ttft / ttft_value) if ttft_value > 0 else None,
            "request_uid": request_uid,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "agent_role": agent_role,
            "message": str(message_key_source) if message_key_source is not None else None,
            "request_message": request_message,
            "prompt_source": prompt_source,
            "prompt_tokens": prompt_num_tokens,
            "prompt_block_count": prompt_block_count,
            "full_block_count": full_block_count,
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
