"""
Worker subprocess for live inference.

Runs one or more (backend x mode) combinations sequentially in a single
process, printing JSON events to stdout.  When multiple modes share the
same backend the model is loaded once and reused.

Usage:
  python demo/live_worker.py \
    --question "..." \
    --tasks '[{"mode_key":"kvcomm_dense","execution_mode":"default"},
              {"mode_key":"kvcomm_reuse","execution_mode":"allow_kv_reuse"}]' \
    --backend hf

Events (one JSON per line on stdout):
  {"type": "mode_start", "mode_key": "...", "execution_mode": "..."}
  {"type": "agent_output", "mode_key": "...", "agent_id": "...", ...}
  {"type": "mode_done",  "mode_key": "...", "answers": [...]}
  {"type": "error",      "mode_key": "...", "message": "..."}
"""

import argparse
import asyncio
import base64
import copy
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_real_stdout = sys.stdout
sys.stdout = sys.stderr

_current_mode_key = {"value": ""}
DEMO_AGENT_ROLES = [
    "Math Solver",
    "Mathematical Analyst",
    "Programming Expert",
]


def emit(event: dict):
    """Write a single JSON event line to the real stdout."""
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
    parser = argparse.ArgumentParser()
    q_group = parser.add_mutually_exclusive_group(required=True)
    q_group.add_argument("--question", type=str)
    q_group.add_argument("--question-b64", type=str,
                         help="Base64-encoded question (avoids shell quoting)")
    parser.add_argument("--tasks", type=str, required=True,
                        help='JSON list of {mode_key, execution_mode}')
    parser.add_argument("--backend", type=str, required=True,
                        choices=["hf", "paged", "radix"])
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index to use (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--nccl-port", type=int, default=None,
                        help="NCCL port for nano-vllm dist.init (avoids port conflict)")
    parser.add_argument("--repo-root", type=str, default="")
    args = parser.parse_args()
    if args.question is None:
        args.question = base64.b64decode(args.question_b64).decode("utf-8")
    return args


def patch_metrics_recorder():
    """Monkey-patch metrics_recorder to emit JSON events for the live demo."""
    from KVCOMM.utils.metrics import metrics_recorder

    original_record = metrics_recorder.record_agent_output.__func__
    original_finalize = metrics_recorder.finalize_request.__func__

    def patched_record(self, *, request_uid, agent_id, agent_name,
                       agent_role, generation):
        original_record(self, request_uid=request_uid, agent_id=agent_id,
                        agent_name=agent_name, agent_role=agent_role,
                        generation=generation)
        if generation is not None:
            emit({
                "type": "agent_output",
                "mode_key": _current_mode_key["value"],
                "request_uid": request_uid,
                "agent_id": str(agent_id),
                "agent_name": agent_name,
                "agent_role": agent_role,
                "mode": generation.mode,
                "ttft": generation.ttft,
                "text": generation.text,
            })

    metrics_recorder.record_agent_output = patched_record.__get__(
        metrics_recorder, type(metrics_recorder)
    )

    def patched_finalize(self, request_uid):
        request_entry = self._requests.get(request_uid, {})
        reuse_rate = original_finalize(self, request_uid)
        emit({
            "type": "request_reuse",
            "mode_key": _current_mode_key["value"],
            "request_uid": request_uid,
            "execution_mode": request_entry.get("execution_mode"),
            "reuse_rate": 0.0 if reuse_rate is None else reuse_rate,
            "kv_reuse_count": request_entry.get("kv_reuse_count", 0),
            "total_agents": request_entry.get("total_count", 0),
            "task": request_entry.get("task"),
        })
        return reuse_rate

    metrics_recorder.finalize_request = patched_finalize.__get__(
        metrics_recorder, type(metrics_recorder)
    )


async def run_single_task(question: str, execution_mode: str, mode_key: str,
                          model_name: str):
    """Run one mode and emit events."""
    from KVCOMM.graph.graph import Graph
    from KVCOMM.llm.config import KVCommConfig
    from KVCOMM.experiments.run_gsm8k import get_kwargs

    _current_mode_key["value"] = mode_key

    emit({
        "type": "mode_start",
        "mode_key": mode_key,
        "execution_mode": execution_mode,
    })

    agent_names = ["MathSolver"] * len(DEMO_AGENT_ROLES)
    kwargs = get_kwargs("FullConnected", len(agent_names))
    kwargs["node_kwargs"] = [{"role": role} for role in DEMO_AGENT_ROLES]

    kv_config = KVCommConfig.from_env()
    if execution_mode == "allow_kv_reuse":
        kv_config = kv_config.apply_overrides()

    graph = Graph(
        domain="gsm8k",
        llm_name=model_name,
        agent_names=agent_names,
        decision_method="FinalRefer",
        kv_config=kv_config,
        **kwargs,
    )

    input_dict = {"task": question, "_batch_index": 0}
    mode_kwargs = {}
    if execution_mode == "allow_kv_reuse":
        mode_kwargs["prefix"] = "Q:\n"
        if "radix" in mode_key:
            output_subdir = "live_demo_radix"
        else:
            output_subdir = "live_demo"
        mode_kwargs["output_dir"] = str(
            REPO_ROOT / "KVCOMM" / "result" / output_subdir
        )

    result = await graph.arun(
        input_dict, 1, mode=execution_mode, **mode_kwargs
    )

    answers = result.get("answers", [])
    clean_answers = [a if isinstance(a, str) else str(a) for a in answers]

    emit({
        "type": "mode_done",
        "mode_key": mode_key,
        "execution_mode": execution_mode,
        "answers": clean_answers,
    })


async def run_all_tasks(question: str, tasks: list, model_name: str):
    """Run all tasks sequentially (model stays loaded across tasks)."""
    for task in tasks:
        try:
            await run_single_task(
                question,
                task["execution_mode"],
                task["mode_key"],
                model_name,
            )
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            emit({
                "type": "error",
                "mode_key": task["mode_key"],
                "message": str(e),
            })


def main():
    args = parse_args()
    _activate_repo_root(args.repo_root or None)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.nccl_port is not None:
        os.environ["NANOVLLM_NCCL_PORT"] = str(args.nccl_port)

    if args.backend in ("paged", "radix"):
        os.environ["KVCOMM_PAGED"] = "1"
        os.environ["KVCOMM_PAGED_BACKEND"] = "radix" if args.backend == "radix" else "paged"
    else:
        os.environ["KVCOMM_PAGED"] = "0"
        os.environ.pop("KVCOMM_PAGED_BACKEND", None)

    tasks = json.loads(args.tasks)

    model_name = args.model_path or os.environ.get(
        "KVCOMM_MODEL",
        "/home/users/yf199/workspace/hf_cache/Llama-3.1-8B-Instruct",
    )

    patch_metrics_recorder()
    asyncio.run(run_all_tasks(args.question, tasks, model_name))


if __name__ == "__main__":
    main()
