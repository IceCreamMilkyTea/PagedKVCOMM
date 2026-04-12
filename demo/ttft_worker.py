"""
TTFT benchmark worker subprocess.

Runs N synthetic requests through a KVCOMM graph (one method at a time)
and emits JSON progress events to stdout.

Events (one JSON per line on stdout):
  {"type":"method_start", "method":"dense", "total":100}
  {"type":"request_done", "method":"dense", "idx":0, "prompt":"...",
   "ttft_avg":0.05, "elapsed":1.2, "throughput":0.83}
  {"type":"method_done", "method":"dense", "avg_ttft":0.05, "throughput":12.3}
  {"type":"error", "method":"dense", "message":"..."}

Usage:
  python demo/ttft_worker.py --method dense --samples 50 \
      --agents 5 --backend hf --model /path/to/model
"""

import argparse
import asyncio
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Keep stdout for JSON events only; redirect prints to stderr
_real_stdout = sys.stdout
sys.stdout = sys.stderr

SEED = int(os.getenv("SEED", 42))
random.seed(SEED)


def emit(event: dict):
    _real_stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    _real_stdout.flush()


def _activate_repo_root(repo_root: str | None) -> Path:
    global REPO_ROOT
    if repo_root:
        REPO_ROOT = Path(repo_root).expanduser().resolve()
    repo_root_str = str(REPO_ROOT)
    sys.path = [repo_root_str] + [p for p in sys.path if p != repo_root_str]
    return REPO_ROOT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True,
                   choices=[
                       "dense",
                       "kvcomm",
                       "paged_dense",
                       "paged_kvcomm",
                       "radix_dense",
                       "paged_radix_kvcomm",
                   ])
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--agents", type=int, default=5,
                   help="Number of CopyMachine agents")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--backend", type=str, required=True,
                   choices=["hf", "paged", "radix"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--nccl-port", type=int, default=None)
    p.add_argument("--in-length", type=int, default=512)
    p.add_argument("--out-length", type=int, default=512)
    p.add_argument("--repo-root", type=str, default="")
    return p.parse_args()


def _make_random_token_sequence(length: int) -> str:
    symbols = random.choices(["\u0394", "\u03a9"], k=length)
    return " ".join(symbols)


async def run_benchmark(args):
    """Run N requests sequentially and emit events."""
    _activate_repo_root(args.repo_root or None)
    # Set env before imports
    os.environ["IN_LENGTH"] = str(args.in_length)
    os.environ["OUT_LENGTH"] = str(args.out_length)
    os.environ["KVCOMM_PAGED"] = "1" if args.backend in ("paged", "radix") else "0"
    if args.backend in ("paged", "radix"):
        os.environ["KVCOMM_PAGED_BACKEND"] = "radix" if args.backend == "radix" else "paged"
    else:
        os.environ.pop("KVCOMM_PAGED_BACKEND", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.nccl_port is not None:
        os.environ["NANOVLLM_NCCL_PORT"] = str(args.nccl_port)

    from KVCOMM.graph.graph import Graph
    from KVCOMM.llm.config import KVCommConfig
    from KVCOMM.utils.metrics import metrics_recorder

    method = args.method
    is_kv_reuse = method in ("kvcomm", "paged_kvcomm", "paged_radix_kvcomm")
    execution_mode = "allow_kv_reuse" if is_kv_reuse else "default"

    # Build topology
    N = args.agents
    agent_names = ["CopyMachine"] * N
    fixed_spatial = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
    fixed_temporal = [[1 for _ in range(N)] for _ in range(N)]

    kv_config = KVCommConfig.from_env()
    if is_kv_reuse:
        kv_config = kv_config.apply_overrides(
            threshold=1.0,
            max_anchor_num=20,
            window_size=5,
        )

    graph = Graph(
        domain="COPY",
        llm_name=args.model,
        agent_names=agent_names,
        kv_config=kv_config,
        fixed_spatial_masks=fixed_spatial,
        fixed_temporal_masks=fixed_temporal,
    )

    # Prepare data
    data = [{"task": _make_random_token_sequence(1000)}
            for _ in range(args.samples)]

    emit({"type": "method_start", "method": method, "total": args.samples})

    all_ttfts = []
    t_inference_start = None  # set after first request completes (warmup)

    for i, input_dict in enumerate(data):
        realized = copy.deepcopy(graph)
        realized.spatial_logits = graph.spatial_logits
        realized.temporal_logits = graph.temporal_logits

        req_kwargs = {"mode": execution_mode, "test_time": False}
        if is_kv_reuse:
            req_kwargs["prefix"] = "The task is:\n\n"

        # Snapshot cumulative TTFT stats BEFORE this request
        with metrics_recorder._lock:
            pre_sum = sum(s["sum"] for s in metrics_recorder._ttft_stats.values())
            pre_count = sum(s["count"] for s in metrics_recorder._ttft_stats.values())

        await realized.arun(input_dict, **req_kwargs)

        # Mark inference start after first request (excludes model loading warmup)
        if t_inference_start is None:
            t_inference_start = time.perf_counter()

        # Snapshot cumulative TTFT stats AFTER this request
        with metrics_recorder._lock:
            post_sum = sum(s["sum"] for s in metrics_recorder._ttft_stats.values())
            post_count = sum(s["count"] for s in metrics_recorder._ttft_stats.values())

        inference_elapsed = time.perf_counter() - t_inference_start
        # throughput counts from request index 1 onward (request 0 is warmup)
        throughput = i / inference_elapsed if (i > 0 and inference_elapsed > 0) else 0

        task_text = input_dict["task"]
        prompt_snippet = task_text[:150]

        # Compute this request's avg TTFT from the delta
        delta_count = post_count - pre_count
        delta_sum = post_sum - pre_sum
        avg_ttft = delta_sum / delta_count if delta_count > 0 else 0.0
        all_ttfts.append(avg_ttft)

        emit({
            "type": "request_done",
            "method": method,
            "idx": i,
            "prompt": prompt_snippet,
            "ttft_avg": avg_ttft,
            "elapsed": round(inference_elapsed, 2),
            "throughput": round(throughput, 2),
        })

    total_inference = time.perf_counter() - (t_inference_start or time.perf_counter())
    # Exclude warmup request (index 0) from throughput
    steady_count = max(args.samples - 1, 1)
    overall_avg = sum(all_ttfts) / len(all_ttfts) if all_ttfts else 0
    overall_throughput = steady_count / total_inference if total_inference > 0 else 0

    emit({
        "type": "method_done",
        "method": method,
        "avg_ttft": round(overall_avg, 4),
        "throughput": round(overall_throughput, 2),
        "elapsed": round(total_inference, 2),
        "samples": args.samples,
    })


def main():
    args = parse_args()
    try:
        asyncio.run(run_benchmark(args))
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        emit({"type": "error", "method": args.method, "message": str(e)})


if __name__ == "__main__":
    main()
