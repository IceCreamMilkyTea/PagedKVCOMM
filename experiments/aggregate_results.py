"""Aggregate benchmark_summary.json files and produce a comparison report.

Usage:
    python experiments/aggregate_results.py --results_dir result

Scans for all benchmark_summary.json files under results_dir and outputs
a unified comparison table.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def collect_summaries(results_dir: Path) -> List[Dict[str, Any]]:
    summaries = []
    for p in sorted(results_dir.rglob("benchmark_summary.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_path"] = str(p.relative_to(results_dir))
            summaries.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return summaries


def _group_by_benchmark(summaries: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    """Returns {benchmark: {method: summary}}."""
    grouped: Dict[str, Dict[str, Dict]] = {}
    for s in summaries:
        b = s.get("benchmark", "?")
        m = s.get("method", "?")
        grouped.setdefault(b, {})[m] = s
    return grouped


def print_report(summaries: List[Dict]):
    grouped = _group_by_benchmark(summaries)
    methods = ["dense", "kvcomm", "paged_kvcomm"]

    print("\n" + "=" * 90)
    print("  COMPARISON REPORT: Dense vs KVCOMM vs Paged KVCOMM")
    print("=" * 90)

    for benchmark, by_method in sorted(grouped.items()):
        print(f"\n[{benchmark}]")
        header = f"  {'Metric':<20}"
        for m in methods:
            header += f" | {m:>18}"
        print(header)
        print("  " + "-" * (20 + 3 * 21))

        # Accuracy row
        has_acc = any(by_method.get(m, {}).get("accuracy") is not None for m in methods)
        if has_acc:
            line = f"  {'Accuracy':<20}"
            for m in methods:
                s = by_method.get(m, {})
                acc = s.get("accuracy")
                if acc is not None:
                    line += f" | {acc:>18.4f}"
                else:
                    line += f" | {'N/A':>18}"
            print(line)

        # Peak memory row
        has_mem = any(by_method.get(m, {}).get("peak_memory_mb") is not None for m in methods)
        if has_mem:
            line = f"  {'Peak Mem (MB)':<20}"
            for m in methods:
                s = by_method.get(m, {})
                mem = s.get("peak_memory_mb")
                if mem is not None:
                    line += f" | {mem:>18.0f}"
                else:
                    line += f" | {'N/A':>18}"
            print(line)

        # TTFT avg row
        has_ttft = any(by_method.get(m, {}).get("ttft", {}).get("avg_ms") is not None for m in methods)
        if has_ttft:
            line = f"  {'Avg TTFT (ms)':<20}"
            for m in methods:
                s = by_method.get(m, {})
                avg = s.get("ttft", {}).get("avg_ms")
                if avg is not None:
                    line += f" | {avg:>18.1f}"
                else:
                    line += f" | {'N/A':>18}"
            print(line)

            line = f"  {'P90 TTFT (ms)':<20}"
            for m in methods:
                s = by_method.get(m, {})
                p90 = s.get("ttft", {}).get("p90_ms")
                if p90 is not None:
                    line += f" | {p90:>18.1f}"
                else:
                    line += f" | {'N/A':>18}"
            print(line)

        # Wall time row
        has_time = any(by_method.get(m, {}).get("elapsed_s") is not None for m in methods)
        if has_time:
            line = f"  {'Wall Time (s)':<20}"
            for m in methods:
                s = by_method.get(m, {})
                t = s.get("elapsed_s")
                if t is not None:
                    line += f" | {t:>18.1f}"
                else:
                    line += f" | {'N/A':>18}"
            print(line)

    print("\n" + "=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Aggregate benchmark results")
    parser.add_argument("--results_dir", type=str, default="result")
    args = parser.parse_args()
    results_dir = Path(args.results_dir).expanduser()

    summaries = collect_summaries(results_dir)
    if not summaries:
        print(f"No benchmark_summary.json found under {results_dir}")
        return

    print_report(summaries)

    report_path = results_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"Raw data saved to: {report_path}")


if __name__ == "__main__":
    main()
