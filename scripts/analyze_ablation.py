#!/usr/bin/env python3
"""
Ablation Study Analysis Script
Compare performance across three setups:
1. Baseline: gpt_chat without Flash Attention
2. With FA: gpt_chat with Flash Attention
3. Paged: nano-vllm with Flash Attention + Paged KV Cache
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import statistics

def load_results(result_dir: Path) -> Dict[str, Any]:
    """Load results from a JSON file."""
    json_files = list(result_dir.glob("*.json"))
    if not json_files:
        return None

    # Use the most recent file
    json_file = max(json_files, key=lambda p: p.stat().st_mtime)
    with open(json_file, 'r') as f:
        return json.load(f)

def compute_metrics(results: List[Dict]) -> Dict[str, float]:
    """Compute key metrics from results."""
    if not results:
        return None

    metrics = {
        "total_tests": len(results),
        "total_solved": sum(1 for r in results if r.get("Solved", False)),
        "accuracy": results[-1]["Accuracy"] if results else 0,
    }

    # Extract TTFT from metrics (if available in logs)
    ttfts = []
    for r in results:
        if "ttft" in r:
            ttfts.append(r["ttft"])

    if ttfts:
        metrics["avg_ttft"] = statistics.mean(ttfts)
        metrics["min_ttft"] = min(ttfts)
        metrics["max_ttft"] = max(ttfts)

    return metrics

def main():
    repo_root = Path(__file__).parent.parent
    result_base = repo_root / "KVCOMM" / "result"

    experiments = {
        "Baseline (gpt_chat, no FA)": result_base / "ablation_1_baseline",
        "With Flash Attention": result_base / "ablation_2_with_fa",
        "Paged (FA + Paged KV)": result_base / "ablation_3_paged_fa",
    }

    print("=" * 80)
    print("KVCOMM Ablation Study: Flash Attention + Paged KV Cache")
    print("=" * 80)
    print()

    all_metrics = {}

    for exp_name, result_dir in experiments.items():
        print(f"Analyzing: {exp_name}")
        print(f"  Path: {result_dir}")

        if not result_dir.exists():
            print(f"  ❌ Directory not found")
            print()
            continue

        results = load_results(result_dir)
        if results is None:
            print(f"  ❌ No results found")
            print()
            continue

        metrics = compute_metrics(results)
        all_metrics[exp_name] = metrics

        print(f"  ✓ Tests: {metrics['total_tests']}")
        print(f"  ✓ Solved: {metrics['total_solved']}/{metrics['total_tests']}")
        print(f"  ✓ Accuracy: {metrics['accuracy']:.4f}")

        if "avg_ttft" in metrics:
            print(f"  ✓ Avg TTFT: {metrics['avg_ttft']:.4f}s")
            print(f"  ✓ Min TTFT: {metrics['min_ttft']:.4f}s")
            print(f"  ✓ Max TTFT: {metrics['max_ttft']:.4f}s")

        print()

    # Comparison Table
    print("=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print()
    print(f"{'Metric':<30} {'Baseline':<20} {'With FA':<20} {'Paged':<20}")
    print("-" * 90)

    if all_metrics:
        # Accuracy
        baseline_acc = all_metrics.get("Baseline (gpt_chat, no FA)", {}).get("accuracy", 0)
        with_fa_acc = all_metrics.get("With Flash Attention", {}).get("accuracy", 0)
        paged_acc = all_metrics.get("Paged (FA + Paged KV)", {}).get("accuracy", 0)

        print(f"{'Accuracy':<30} {baseline_acc:<20.4f} {with_fa_acc:<20.4f} {paged_acc:<20.4f}")

        # Speedup Analysis
        if "avg_ttft" in all_metrics.get("Baseline (gpt_chat, no FA)", {}):
            baseline_ttft = all_metrics["Baseline (gpt_chat, no FA)"]["avg_ttft"]
            with_fa_ttft = all_metrics["With Flash Attention"]["avg_ttft"]
            paged_ttft = all_metrics["Paged (FA + Paged KV)"]["avg_ttft"]

            fa_speedup = baseline_ttft / with_fa_ttft
            paged_speedup = baseline_ttft / paged_ttft

            print(f"{'Avg TTFT (s)':<30} {baseline_ttft:<20.4f} {with_fa_ttft:<20.4f} {paged_ttft:<20.4f}")
            print(f"{'Speedup vs Baseline':<30} {1.0:<20.2f}x {fa_speedup:<20.2f}x {paged_speedup:<20.2f}x")

    print()
    print("=" * 80)
    print("INSIGHTS")
    print("=" * 80)
    print()
    print("1. Flash Attention Impact: Compare With FA vs Baseline")
    print("   - Shows pure Flash Attention overhead/benefit in gpt_chat backend")
    print()
    print("2. Paged KV Impact: Compare Paged vs With FA")
    print("   - Shows additional benefits from Paged KV Cache on top of Flash Attention")
    print()
    print("3. Combined Effect: Paged vs Baseline")
    print("   - Shows total performance gain from Flash Attention + Paged KV Cache")
    print()

if __name__ == "__main__":
    main()
