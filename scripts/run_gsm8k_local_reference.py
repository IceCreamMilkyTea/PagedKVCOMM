#!/usr/bin/env python3
"""
Streamlined launcher for GSM8K experiments with --use-local-reference flag.

This script simplifies running full GSM8K dataset experiments with various configurations,
particularly focusing on local reference (cross-agent KV offset) experiments.

Usage examples:
  # Run with local reference enabled (paged backend)
  python scripts/run_gsm8k_local_reference.py --backend paged --use-local-ref

  # Run without local reference for baseline comparison
  python scripts/run_gsm8k_local_reference.py --backend paged

  # Run with HuggingFace backend (non-paged)
  python scripts/run_gsm8k_local_reference.py --backend hf --use-local-ref

  # Run on debug dataset (15 samples)
  python scripts/run_gsm8k_local_reference.py --backend paged --use-local-ref --debug

  # Custom model and configuration
  python scripts/run_gsm8k_local_reference.py \
      --backend paged \
      --use-local-ref \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --mode FullConnected \
      --num-agents 3 \
      --batch-size 1
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GSM8K experiments with local reference support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Experiment configuration ──
    parser.add_argument(
        "--backend",
        type=str,
        choices=["paged", "hf"],
        default="paged",
        help="Backend: 'paged' (nano-vllm) or 'hf' (HuggingFace)",
    )
    parser.add_argument(
        "--use-local-ref",
        action="store_true",
        help="Enable --use-local-reference for cross-agent KV offset",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug dataset (15 samples) instead of full dataset",
    )

    # ── Model and topology ──
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or path (default: env KVCOMM_MODEL or Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="FullConnected",
        choices=["DirectAnswer", "FullConnected", "Random", "Chain", "Debate", "Layered", "Star"],
        help="Communication topology among agents",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=3,
        help="Number of MathSolver agents",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for parallel processing",
    )

    # ── KV cache tuning ──
    parser.add_argument(
        "--kv-threshold",
        type=float,
        default=None,
        help="Similarity threshold for KV cache reuse (default: 0.99)",
    )
    parser.add_argument(
        "--kv-max-anchor-num",
        type=int,
        default=None,
        help="Maximum number of anchors per placeholder (default: 20)",
    )
    parser.add_argument(
        "--kv-window-size",
        type=int,
        default=None,
        help="Window size for anchor eviction (default: 5)",
    )
    parser.add_argument(
        "--kv-thread-workers",
        type=int,
        default=None,
        help="Thread pool workers for KV operations",
    )
    parser.add_argument(
        "--kv-worker-timeout",
        type=float,
        default=None,
        help="Worker timeout in seconds",
    )

    # ── Output ──
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: KVCOMM/result/gsm8k_local_ref_<backend>_<timestamp>)",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=None,
        help="Python binary to use (default: current interpreter)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Determine Python binary ──
    python_bin = args.python_bin or sys.executable

    # ── Determine model path ──
    if args.model:
        model_name = args.model
    else:
        model_name = os.environ.get(
            "KVCOMM_MODEL",
            "meta-llama/Llama-3.1-8B-Instruct",
        )

    # ── Determine dataset ──
    if args.debug:
        dataset_path = str(REPO_ROOT / "datasets" / "gsm8k" / "gsm8k_debug15.jsonl")
    else:
        dataset_path = str(REPO_ROOT / "datasets" / "gsm8k" / "gsm8k.jsonl")

    # ── Determine output directory ──
    if args.output_dir:
        output_dir = args.output_dir
    else:
        import time
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        local_ref_tag = "local_ref" if args.use_local_ref else "baseline"
        output_dir = str(
            REPO_ROOT / "KVCOMM" / "result" / f"gsm8k_{local_ref_tag}_{args.backend}_{timestamp}"
        )

    # ── Build command ──
    cmd = [
        python_bin,
        str(REPO_ROOT / "KVCOMM" / "experiments" / "run_gsm8k.py"),
        "--dataset_json", dataset_path,
        "--llm_name", model_name,
        "--mode", args.mode,
        "--batch_size", str(args.batch_size),
        "--domain", "gsm8k",
        "--agent_names", "MathSolver",
        "--agent_nums", str(args.num_agents),
        "--decision_method", "FinalRefer",
        "--execution_mode", "allow_kv_reuse",
        "--output_dir", output_dir,
        "--prefix", "Q:\n",
    ]

    # Add backend flag
    if args.backend == "paged":
        cmd.extend(["--use-flash-attention"])

    # Add local reference flag
    if args.use_local_ref:
        cmd.append("--use-local-reference")

    # Add KV tuning parameters
    if args.kv_threshold is not None:
        cmd.extend(["--kv-threshold", str(args.kv_threshold)])
    if args.kv_max_anchor_num is not None:
        cmd.extend(["--kv-max-anchor-num", str(args.kv_max_anchor_num)])
    if args.kv_window_size is not None:
        cmd.extend(["--kv-window-size", str(args.kv_window_size)])
    if args.kv_thread_workers is not None:
        cmd.extend(["--kv-thread-workers", str(args.kv_thread_workers)])
    if args.kv_worker_timeout is not None:
        cmd.extend(["--kv-worker-timeout", str(args.kv_worker_timeout)])

    # ── Set environment for paged backend ──
    env = os.environ.copy()
    if args.backend == "paged":
        env["KVCOMM_PAGED"] = "1"
    else:
        env["KVCOMM_PAGED"] = "0"

    # ── Print configuration ──
    print("=" * 80)
    print("GSM8K Experiment Configuration")
    print("=" * 80)
    print(f"Backend:          {args.backend}")
    print(f"Local Reference:  {args.use_local_ref}")
    print(f"Model:            {model_name}")
    print(f"Mode:             {args.mode}")
    print(f"Num Agents:       {args.num_agents}")
    print(f"Dataset:          {dataset_path}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Output Dir:       {output_dir}")
    if args.kv_threshold is not None:
        print(f"KV Threshold:     {args.kv_threshold}")
    if args.kv_max_anchor_num is not None:
        print(f"Max Anchor Num:   {args.kv_max_anchor_num}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)
    print()

    # ── Execute ──
    try:
        subprocess.run(cmd, env=env, check=True)
        print("\n" + "=" * 80)
        print(f"✓ Experiment completed successfully!")
        print(f"  Results saved to: {output_dir}")
        print("=" * 80)
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 80)
        print(f"✗ Experiment failed with exit code {e.returncode}")
        print("=" * 80)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("✗ Experiment interrupted by user")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
