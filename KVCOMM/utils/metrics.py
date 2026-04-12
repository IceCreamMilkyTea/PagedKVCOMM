from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from loguru import logger


@dataclass(slots=True)
class GenerationResult:
    """Container describing a single model generation outcome."""

    text: str
    mode: str
    ttft: float
    raw_output: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RequestMetricsRecorder:
    """Tracks per-request agent outputs, reuse rates, and mode-specific TTFT."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._requests: Dict[str, Dict[str, Any]] = {}
        self._total_calls: int = 0
        self._total_reuse_calls: int = 0
        self._total_radix_hit_calls: int = 0
        self._total_applied_reuse_calls: int = 0
        self._total_effective_reuse_calls: int = 0
        self._ttft_stats: Dict[str, Dict[str, float]] = {}
        self._trace_root: Optional[Path] = None
        self._request_trace_dir: Optional[Path] = None
        self._agent_trace_path: Optional[Path] = None

    def reset(self) -> None:
        with self._lock:
            self._requests = {}
            self._total_calls = 0
            self._total_reuse_calls = 0
            self._total_radix_hit_calls = 0
            self._total_applied_reuse_calls = 0
            self._total_effective_reuse_calls = 0
            self._ttft_stats = {}

    def configure_trace_output(self, output_dir: Optional[str | Path]) -> None:
        with self._lock:
            if output_dir is None:
                self._trace_root = None
                self._request_trace_dir = None
                self._agent_trace_path = None
                return
            root = Path(output_dir).expanduser() / "debug" / "agent_traces"
            request_dir = root / "requests"
            request_dir.mkdir(parents=True, exist_ok=True)
            for stale in request_dir.glob("*.json"):
                stale.unlink()
            agent_trace_path = root / "agent_io.jsonl"
            if agent_trace_path.exists():
                agent_trace_path.unlink()
            self._trace_root = root
            self._request_trace_dir = request_dir
            self._agent_trace_path = agent_trace_path

    @staticmethod
    def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def start_request(
        self,
        *,
        request_uid: str,
        batch_index: Optional[int],
        task: Optional[str],
        execution_mode: str,
    ) -> None:
        with self._lock:
            self._requests[request_uid] = {
                "batch_index": batch_index,
                "task": task,
                "execution_mode": execution_mode,
                "agents": [],
                "kv_reuse_count": 0,
                "radix_hit_count": 0,
                "applied_reuse_count": 0,
                "effective_reuse_count": 0,
                "total_count": 0,
            }

    def record_agent_output(
        self,
        *,
        request_uid: str,
        agent_id: str,
        agent_name: str,
        agent_role: str,
        generation: GenerationResult | None,
    ) -> None:
        if generation is None:
            return

        with self._lock:
            request_entry = self._requests.setdefault(
                request_uid,
                {
                    "batch_index": None,
                    "task": None,
                    "execution_mode": "unknown",
                    "agents": [],
                    "kv_reuse_count": 0,
                    "radix_hit_count": 0,
                    "applied_reuse_count": 0,
                    "effective_reuse_count": 0,
                    "total_count": 0,
                },
            )

            metadata = generation.metadata or {}
            agent_record = {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_role": agent_role,
                "mode": generation.mode,
                "ttft": generation.ttft,
                "text": generation.text,
            }
            if metadata:
                agent_record["metadata"] = metadata
            agents_list = request_entry["agents"]
            replaced = False
            for idx, existing in enumerate(agents_list):
                if existing["agent_id"] == agent_id:
                    agents_list[idx] = agent_record
                    replaced = True
                    break
            if not replaced:
                agents_list.append(agent_record)

            def _reuse_stats(entry: Dict[str, Any]) -> Dict[str, Any]:
                return entry.get("metadata", {}).get("reuse_stats", {})

            def _is_applied_reuse(entry: Dict[str, Any]) -> bool:
                reuse_stats = _reuse_stats(entry)
                return bool(
                    reuse_stats.get("applied_reuse_hit")
                    or reuse_stats.get("offset_applied")
                    or (reuse_stats.get("num_cached_tokens", 0) or 0) > 0
                )

            request_entry["total_count"] = len(agents_list)
            request_entry["kv_reuse_count"] = sum(
                1 for entry in agents_list if entry.get("mode") == "kv_reuse"
            )
            request_entry["radix_hit_count"] = sum(
                1
                for entry in agents_list
                if (_reuse_stats(entry).get("num_cached_tokens", 0) or 0) > 0
            )
            request_entry["applied_reuse_count"] = sum(
                1 for entry in agents_list if _is_applied_reuse(entry)
            )
            request_entry["effective_reuse_count"] = sum(
                1
                for entry in agents_list
                if _reuse_stats(entry).get("effective_reuse_hit", False)
            )

            stats = self._ttft_stats.setdefault(
                generation.mode,
                {"sum": 0.0, "count": 0.0},
            )
            stats["sum"] += generation.ttft
            stats["count"] += 1
            avg_ttft = stats["sum"] / stats["count"] if stats["count"] else 0.0

            ttft_payload = {
                "request_uid": request_uid,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_role": agent_role,
                "mode": generation.mode,
                "ttft": generation.ttft,
                "mode_avg_ttft": avg_ttft,
            }
            preprocess_latency = metadata.get("preprocess_latency")
            if preprocess_latency is not None:
                ttft_payload["preprocess_latency"] = preprocess_latency
            generation_ttft = metadata.get("generation_ttft")
            if generation_ttft is not None:
                ttft_payload["generation_ttft"] = generation_ttft
            logger.opt(colors=True).info(
                "<cyan>[TTFT:{mode}]</cyan> {}",
                json.dumps(ttft_payload, ensure_ascii=False),
                mode=generation.mode,
            )

            output_payload = {
                "request_uid": request_uid,
                "batch_index": request_entry.get("batch_index"),
                "task": request_entry.get("task"),
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_role": agent_role,
                "mode": generation.mode,
                "text": generation.text,
            }
            logger.opt(colors=True).info(
                "<green>[AGENT OUTPUT]</green> {}",
                json.dumps(output_payload, ensure_ascii=False),
            )
            if self._agent_trace_path is not None:
                trace_payload = dict(output_payload)
                if metadata:
                    trace_payload["metadata"] = metadata
                self._append_jsonl(self._agent_trace_path, trace_payload)

    def finalize_request(self, request_uid: str) -> Optional[float]:
        with self._lock:
            request_entry = self._requests.pop(request_uid, None)
            if request_entry is None:
                return None

            kv_reuse = request_entry.get("kv_reuse_count", 0)
            radix_hits = request_entry.get("radix_hit_count", 0)
            applied_reuse = request_entry.get("applied_reuse_count", 0)
            effective_reuse = request_entry.get("effective_reuse_count", 0)
            total = request_entry.get("total_count", 0)
            reuse_rate = (kv_reuse / total) if total else 0.0
            radix_hit_rate = (radix_hits / total) if total else 0.0
            applied_reuse_rate = (applied_reuse / total) if total else 0.0
            effective_reuse_rate = (effective_reuse / total) if total else 0.0

            payload = {
                "request_uid": request_uid,
                "batch_index": request_entry.get("batch_index"),
                "task": request_entry.get("task"),
                "execution_mode": request_entry.get("execution_mode"),
                "reuse_rate": reuse_rate,
                "kv_reuse_count": kv_reuse,
                "radix_hit_rate": radix_hit_rate,
                "radix_hit_count": radix_hits,
                "applied_reuse_rate": applied_reuse_rate,
                "applied_reuse_count": applied_reuse,
                "effective_reuse_rate": effective_reuse_rate,
                "effective_reuse_count": effective_reuse,
                "total_agents": total,
            }
            logger.opt(colors=True).info(
                "<magenta>[REQUEST REUSE]</magenta> {}",
                json.dumps(payload, ensure_ascii=False),
            )

            self._total_calls += total
            self._total_reuse_calls += kv_reuse
            self._total_radix_hit_calls += radix_hits
            self._total_applied_reuse_calls += applied_reuse
            self._total_effective_reuse_calls += effective_reuse
            if self._request_trace_dir is not None:
                request_payload = dict(payload)
                request_payload["agents"] = request_entry.get("agents", [])
                with open(
                    self._request_trace_dir / f"{request_uid}.json",
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(request_payload, handle, ensure_ascii=False, indent=2)
            return reuse_rate

    def log_cumulative(self, *, batch_index: Optional[int]) -> float:
        with self._lock:
            cumulative = (
                self._total_reuse_calls / self._total_calls
                if self._total_calls
                else 0.0
            )
            radix_hit_rate = (
                self._total_radix_hit_calls / self._total_calls
                if self._total_calls
                else 0.0
            )
            applied_reuse_rate = (
                self._total_applied_reuse_calls / self._total_calls
                if self._total_calls
                else 0.0
            )
            effective_reuse_rate = (
                self._total_effective_reuse_calls / self._total_calls
                if self._total_calls
                else 0.0
            )

            payload = {
                "batch_index": batch_index,
                "cumulative_reuse_rate": cumulative,
                "kv_reuse_calls": self._total_reuse_calls,
                "radix_hit_calls": self._total_radix_hit_calls,
                "applied_reuse_calls": self._total_applied_reuse_calls,
                "effective_reuse_calls": self._total_effective_reuse_calls,
                "cumulative_radix_hit_rate": radix_hit_rate,
                "cumulative_applied_reuse_rate": applied_reuse_rate,
                "cumulative_effective_reuse_rate": effective_reuse_rate,
                "total_agent_calls": self._total_calls,
            }
            logger.opt(colors=True).info(
                "<yellow>[CUMULATIVE REUSE]</yellow> {}",
                json.dumps(payload, ensure_ascii=False),
            )
            return cumulative

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            total_ttft = sum(stats["sum"] for stats in self._ttft_stats.values())
            total_ttft_count = sum(stats["count"] for stats in self._ttft_stats.values())
            avg_ttft = total_ttft / total_ttft_count if total_ttft_count else 0.0
            mode_stats = {
                mode: {
                    "count": int(stats["count"]),
                    "avg_ttft": (stats["sum"] / stats["count"]) if stats["count"] else 0.0,
                    "ttft_sum": stats["sum"],
                }
                for mode, stats in self._ttft_stats.items()
            }
            total_calls = self._total_calls
            return {
                "total_agent_calls": total_calls,
                "kv_reuse_calls": self._total_reuse_calls,
                "kv_reuse_ratio": (self._total_reuse_calls / total_calls) if total_calls else 0.0,
                "radix_hit_calls": self._total_radix_hit_calls,
                "radix_hit_ratio": (self._total_radix_hit_calls / total_calls) if total_calls else 0.0,
                "applied_reuse_calls": self._total_applied_reuse_calls,
                "applied_reuse_ratio": (self._total_applied_reuse_calls / total_calls) if total_calls else 0.0,
                "effective_reuse_calls": self._total_effective_reuse_calls,
                "effective_reuse_ratio": (self._total_effective_reuse_calls / total_calls) if total_calls else 0.0,
                "total_ttft": total_ttft,
                "total_ttft_sum": total_ttft,
                "avg_total_ttft": avg_ttft,
                "mode_stats": mode_stats,
                "trace_dir": str(self._trace_root) if self._trace_root is not None else None,
                "agent_trace_file": str(self._agent_trace_path) if self._agent_trace_path is not None else None,
            }


metrics_recorder = RequestMetricsRecorder()
