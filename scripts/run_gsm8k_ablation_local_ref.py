#!/usr/bin/env python3
"""
Batch runner for GSM8K local-reference ablation experiments.

This script runs multiple experiments to compare:
  1. Baseline: KV reuse without local reference
  2. Local-ref: KV reuse with cross-agent local reference

Both experiments are run on both backends (paged and HF) for comprehensive comparison.

Usage:
  # Run all ablation experiments (4 total: 2 backends × 2 modes)
  python scripts/run_gsm8k_ablation_local_ref.py

  # Run only paged backend experiments
  python scripts/run_gsm8k_ablation_local_ref.py --backends paged

  # Run on debug dataset
  python scripts/run_gsm8k_ablation_local_ref.py --debug

  # Custom configuration
  python scripts/run_gsm8k_ablation_local_ref.py \
      --backends paged hf \
      --num-agents 3 \
      --batch-size 1
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GSM8K ablation experiments for local reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["paged", "hf"],
        default=["paged", "hf"],
        help="Backends to test (default: both)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug dataset (15 samples) instead of full dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or path",
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
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root directory for all experiments",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline experiments (only run local-ref)",
    )
    parser.add_argument(
        "--skip-local-ref",
        action="store_true",
        help="Skip local-ref experiments (only run baseline)",
    )

    return parser.parse_args()


def run_experiment(
    backend: str,
    use_local_ref: bool,
    output_dir: str,
    model: str,
    mode: str,
    num_agents: int,
    batch_size: int,
    debug: bool,
) -> bool:
    """Run a single experiment configuration."""
    launcher_script = REPO_ROOT / "scripts" / "run_gsm8k_local_reference.py"

    cmd = [
        sys.executable,
        str(launcher_script),
        "--backend", backend,
        "--mode", mode,
        "--num-agents", str(num_agents),
        "--batch-size", str(batch_size),
        "--output-dir", output_dir,
    ]

    if use_local_ref:
        cmd.append("--use-local-ref")
    if debug:
        cmd.append("--debug")
    if model:
        cmd.extend(["--model", model])

    print(f"\n{'='*80}")
    print(f"Running: {backend} {'with' if use_local_ref else 'without'} local reference")
    print(f"{'='*80}")

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
        sys.exit(1)


def main():
    args = parse_args()

    # ── Determine experiment timestamp ──
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")

    # ── Determine output root ──
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = REPO_ROOT / "KVCOMM" / "result" / f"gsm8k_ablation_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    # ── Determine model ──
    model = args.model or os.environ.get("KVCOMM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

    # ── Build experiment matrix ──
    experiments = []
    for backend in args.backends:
        if not args.skip_baseline:
            experiments.append({
                "name": f"{backend}_baseline",
                "backend": backend,
                "use_local_ref": False,
                "output_dir": str(output_root / f"{backend}_baseline"),
            })
        if not args.skip_local_ref:
            experiments.append({
                "name": f"{backend}_local_ref",
                "backend": backend,
                "use_local_ref": True,
                "output_dir": str(output_root / f"{backend}_local_ref"),
            })

    # ── Print experiment plan ──
    print("=" * 80)
    print("GSM8K Local Reference Ablation Study")
    print("=" * 80)
    print(f"Timestamp:        {timestamp}")
    print(f"Output Root:      {output_root}")
    print(f"Model:            {model}")
    print(f"Mode:             {args.mode}")
    print(f"Num Agents:       {args.num_agents}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Debug Mode:       {args.debug}")
    print(f"Total Experiments: {len(experiments)}")
    print()
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['name']}")
    print("=" * 80)
    print()

    # ── Run experiments ──
    results = []
    start_time = time.time()

    for i, exp in enumerate(experiments, 1):
        exp_start = time.time()
        print(f"\n{'#'*80}")
        print(f"# Experiment {i}/{len(experiments)}: {exp['name']}")
        print(f"{'#'*80}")

        success = run_experiment(
            backend=exp["backend"],
            use_local_ref=exp["use_local_ref"],
            output_dir=exp["output_dir"],
            model=model,
            mode=args.mode,
            num_agents=args.num_agents,
            batch_size=args.batch_size,
            debug=args.debug,
        )

        exp_duration = time.time() - exp_start
        results.append({
            "name": exp["name"],
            "success": success,
            "duration": exp_duration,
            "output_dir": exp["output_dir"],
        })

        if success:
            print(f"\n✓ {exp['name']} completed in {exp_duration:.1f}s")
        else:
            print(f"\n✗ {exp['name']} failed after {exp_duration:.1f}s")

    # ── Summary ──
    total_duration = time.time() - start_time
    successful = sum(1 for r in results if r["success"])

    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    print(f"Total Time:     {total_duration:.1f}s ({total_duration/60:.1f}m)")
    print(f"Successful:     {successful}/{len(experiments)}")
    print(f"Results Root:   {output_root}")
    print()
    print("Individual Results:")
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['name']:25s} {r['duration']:6.1f}s → {r['output_dir']}")
    print("=" * 80)

    # ── Generate comparison report ──
    report_path = output_root / "ablation_report.txt"
    with open(report_path, "w") as f:
        f.write("GSM8K Local Reference Ablation Study\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp:        {timestamp}\n")
        f.write(f"Model:            {model}\n")
        f.write(f"Mode:             {args.mode}\n")
        f.write(f"Num Agents:       {args.num_agents}\n")
        f.write(f"Batch Size:       {args.batch_size}\n")
        f.write(f"Debug Mode:       {args.debug}\n")
        f.write(f"Total Duration:   {total_duration:.1f}s\n")
        f.write(f"Successful:       {successful}/{len(experiments)}\n\n")
        f.write("Experiment Results:\n")
        f.write("-" * 80 + "\n")
        for r in results:
            status = "SUCCESS" if r["success"] else "FAILURE"
            f.write(f"{r['name']:25s} {status:8s} {r['duration']:6.1f}s\n")
            f.write(f"  Output: {r['output_dir']}\n\n")

    print(f"\nReport written to: {report_path}")

    # ── Exit code ──
    if successful < len(experiments):
        print(f"\n⚠ {len(experiments) - successful} experiment(s) failed")
        sys.exit(1)
    else:
        print("\n✓ All experiments completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
