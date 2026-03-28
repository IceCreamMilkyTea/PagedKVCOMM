"""Shared utilities for benchmark experiments.

Provides:
  - method → (execution_mode, env var) mapping
  - GPU peak memory tracking
  - Latency.json TTFT statistics extraction
  - Unified summary saving
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# ─── Method setup ────────────────────────────────────────────────────────────

def setup_method(method: str) -> str:
    """Configure env vars for the chosen method, return execution_mode string.

    Args:
        method: One of ``"dense"``, ``"kvcomm"``, ``"paged_kvcomm"``.

    Returns:
        ``"default"`` for dense, ``"allow_kv_reuse"`` for both kvcomm variants.
    """
    if method == "dense":
        os.environ["KVCOMM_PAGED"] = "0"
        return "default"
    elif method == "kvcomm":
        os.environ["KVCOMM_PAGED"] = "0"
        return "allow_kv_reuse"
    elif method == "paged_kvcomm":
        os.environ["KVCOMM_PAGED"] = "1"
        return "allow_kv_reuse"
    else:
        raise ValueError(f"Unknown method: {method!r}. Choose from dense / kvcomm / paged_kvcomm")


# ─── GPU memory helpers ─────────────────────────────────────────────────────

def reset_memory_tracking(device: int = 0) -> None:
    """Reset peak memory stats so we only capture inference memory."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def get_peak_memory_mb(device: int = 0) -> float:
    """Return peak GPU memory allocated (MB) since last reset."""
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return 0.0


def get_current_memory_mb(device: int = 0) -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(device) / (1024 ** 2)
    return 0.0


# ─── Latency / TTFT helpers ────────────────────────────────────────────────

def collect_ttft_stats(output_dir: str) -> Dict[str, Any]:
    """Read Latency.json from output_dir and compute TTFT statistics.

    Returns dict with keys: count, avg_ms, median_ms, p90_ms, p99_ms, std_ms.
    Returns empty dict if Latency.json doesn't exist or is empty.
    """
    latency_path = Path(output_dir) / "Latency.json"
    if not latency_path.exists():
        return {}
    try:
        with open(latency_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(records, list) or not records:
        return {}

    ttft_values = []
    for rec in records:
        ttft = rec.get("ttft")
        if ttft is not None:
            ttft_values.append(float(ttft) * 1000)  # convert s → ms

    if not ttft_values:
        return {}

    arr = np.array(ttft_values)
    return {
        "count": len(arr),
        "avg_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def clear_latency_json(output_dir: str) -> None:
    """Remove Latency.json so each run starts fresh."""
    latency_path = Path(output_dir) / "Latency.json"
    if latency_path.exists():
        latency_path.unlink()


# ─── Summary saving ──────────────────────────────────────────────────────────

def save_benchmark_summary(
    output_dir: str,
    *,
    method: str,
    benchmark: str,
    accuracy: Optional[float] = None,
    solved: Optional[int] = None,
    total: Optional[int] = None,
    elapsed_s: Optional[float] = None,
    peak_memory_mb: Optional[float] = None,
    ttft_stats: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a unified benchmark_summary.json to output_dir."""
    summary: Dict[str, Any] = {
        "method": method,
        "benchmark": benchmark,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if accuracy is not None:
        summary["accuracy"] = accuracy
    if solved is not None:
        summary["solved"] = solved
    if total is not None:
        summary["total"] = total
    if elapsed_s is not None:
        summary["elapsed_s"] = round(elapsed_s, 2)
    if peak_memory_mb is not None:
        summary["peak_memory_mb"] = round(peak_memory_mb, 1)
    if ttft_stats:
        summary["ttft"] = ttft_stats
    if extra:
        summary.update(extra)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "benchmark_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return path


def print_summary(summary_path: Path) -> None:
    """Print a nicely formatted summary from benchmark_summary.json."""
    with open(summary_path, "r", encoding="utf-8") as f:
        s = json.load(f)
    print("\n" + "=" * 60)
    print(f"  Benchmark Summary: {s.get('benchmark', '?')} / {s.get('method', '?')}")
    print("=" * 60)
    if "accuracy" in s:
        print(f"  Accuracy:       {s['accuracy']:.4f}  ({s.get('solved', '?')}/{s.get('total', '?')})")
    if "elapsed_s" in s:
        print(f"  Wall time:      {s['elapsed_s']:.1f}s")
    if "peak_memory_mb" in s:
        print(f"  Peak GPU mem:   {s['peak_memory_mb']:.0f} MB")
    if "ttft" in s:
        t = s["ttft"]
        print(f"  TTFT avg:       {t.get('avg_ms', 0):.1f} ms  (median {t.get('median_ms', 0):.1f}, p90 {t.get('p90_ms', 0):.1f}, p99 {t.get('p99_ms', 0):.1f})")
    print("=" * 60 + "\n")
