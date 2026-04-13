#!/usr/bin/env python3
"""
Generate LaTeX tables from experiment_summary.csv for Overleaf.
Only includes full GSM8K experiments (1319 samples).
"""

import csv
import sys
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent.parent / "result" / "experiment_summary.csv"


def load_data(csv_path: str) -> list[dict]:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    # Only keep full experiments
    return [r for r in rows if int(r["num_samples"]) >= 1319]


def fmt_pct(val: str, digits: int = 2) -> str:
    """Format a decimal string as percentage."""
    if not val or val == "N/A":
        return "--"
    return f"{float(val)*100:.{digits}f}"


def fmt_ms(val: str, digits: int = 1) -> str:
    """Format seconds string as milliseconds."""
    if not val or val == "N/A":
        return "--"
    return f"{float(val)*1000:.{digits}f}"


def fmt_float(val: str, digits: int = 4) -> str:
    if not val or val == "N/A":
        return "--"
    return f"{float(val):.{digits}f}"


# Short readable names for experiments
SHORT_NAMES = {
    "ablation_1_baseline_rtx_pro":                     "HF, CRS Off",
    "crs_priority_ablation_off_20260405_195800":       "HF, CRS Off (v2)",
    "ablation_2_with_fa_rtx_pro":                      "HF + FA, CRS Off",
    "ablation_3_paged_fa_rtx_pro":                     "Paged, CRS Off",
    "crs_ablation_kv_reuse_crs_20260403_014748":       "Paged, CRS Off (v2)",
    "paged_kvcomm_20260403_121751":                    "Paged, CRS Off (v3)",
    "hf_crs_ablation_crs_rtx_pro_20260404_161443":    "HF, CRS On (old)",
    "crs_priority_ablation_on_20260405_195800":        "HF, CRS On",
    "crs_priority_ablation_priority_20260405_195800":  "HF, CRS Priority",
    "hf_crs_ablation_crs_rtx_pro_20260405_131449":    "HF, CRS On (v2)",
    "gsm8k_baseline":                                  "Paged (early)",
    "gsm8k_local_ref":                                 "Paged + LocalRef",
    "gsm8k_nano":                                      "Paged (nano)",
    "gsm8k_nano_rtx_pro":                              "Paged (nano, RTX)",
    "lr_ablation_baseline_20260401_093104":            "LR Ablation: Base",
    "lr_ablation_consistency_20260401_093104":          "LR Ablation: Consist.",
    "lr_ablation_no_check_20260401_093104":            "LR Ablation: NoCheck",
}


def get_short_name(exp: dict) -> str:
    name = exp["experiment"]
    return SHORT_NAMES.get(name, name)


def deduplicate(rows: list[dict]) -> list[dict]:
    """Remove duplicate experiments (same accuracy + same reuse rate = same run)."""
    seen = set()
    result = []
    for r in rows:
        key = (r["final_accuracy"], r["cumulative_reuse_rate"], r["backend"])
        if key in seen:
            continue
        seen.add(key)
        result.append(r)
    return result


def table_main_accuracy(rows: list[dict]) -> str:
    """Table 1: Main accuracy + reuse overview."""
    rows = sorted(rows, key=lambda r: float(r["final_accuracy"]), reverse=True)

    lines = []
    lines.append(r"""
\begin{table*}[htbp]
\centering
\caption{GSM8K Accuracy and KV Reuse Summary (Full Dataset, 1319 samples)}
\label{tab:accuracy_summary}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l c c c c c c c c}
\toprule
\textbf{Experiment} & \textbf{Backend} & \textbf{FA} & \textbf{CRS} & \textbf{Overall Acc.} & \textbf{Node 1 Acc.} & \textbf{Node 2 Acc.} & \textbf{Node 3 Acc.} & \textbf{KV Reuse Rate} \\
\midrule""")

    for r in rows:
        name = get_short_name(r)
        backend = r["backend"].upper()
        fa = r["flash_attention"]
        fa_str = r"\cmark" if fa == "True" else r"\xmark"
        crs = r["crs_status"]
        acc = fmt_pct(r["final_accuracy"])
        n1 = fmt_pct(r.get("node1_acc", ""))
        n2 = fmt_pct(r.get("node2_acc", ""))
        n3 = fmt_pct(r.get("node3_acc", ""))
        reuse = fmt_pct(r.get("cumulative_reuse_rate", ""))

        lines.append(
            f"  {name} & {backend} & {fa_str} & {crs} & {acc}\\% & {n1}\\% & {n2}\\% & {n3}\\% & {reuse}\\% \\\\"
        )

    lines.append(r"""\bottomrule
\end{tabular}%
}
\end{table*}""")
    return "\n".join(lines)


def table_latency(rows: list[dict]) -> str:
    """Table 2: Latency comparison (TTFT, generation TTFT, preprocess) in ms."""
    rows = sorted(rows, key=lambda r: float(r["final_accuracy"]), reverse=True)

    lines = []
    lines.append(r"""
\begin{table*}[htbp]
\centering
\caption{Latency Breakdown by Mode (milliseconds, averaged over all agents)}
\label{tab:latency}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l c | r r r | r r r}
\toprule
 & & \multicolumn{3}{c|}{\textbf{Dense Prefill}} & \multicolumn{3}{c}{\textbf{KV Reuse}} \\
\textbf{Experiment} & \textbf{Backend} & \textbf{TTFT} & \textbf{Gen TTFT} & \textbf{Preproc.} & \textbf{TTFT} & \textbf{Gen TTFT} & \textbf{Preproc.} \\
\midrule""")

    for r in rows:
        name = get_short_name(r)
        backend = r["backend"].upper()
        dp_ttft = fmt_ms(r.get("dense_prefill_avg_ttft", ""))
        dp_gen = fmt_ms(r.get("dense_prefill_avg_gen_ttft", ""))
        dp_pp = fmt_ms(r.get("dense_prefill_avg_preprocess", ""))
        kr_ttft = fmt_ms(r.get("kv_reuse_avg_ttft", ""))
        kr_gen = fmt_ms(r.get("kv_reuse_avg_gen_ttft", ""))
        kr_pp = fmt_ms(r.get("kv_reuse_avg_preprocess", ""))

        lines.append(
            f"  {name} & {backend} & {dp_ttft} & {dp_gen} & {dp_pp} & {kr_ttft} & {kr_gen} & {kr_pp} \\\\"
        )

    lines.append(r"""\bottomrule
\end{tabular}%
}
\end{table*}""")
    return "\n".join(lines)


def table_per_agent_mode(rows: list[dict]) -> str:
    """Table 3: Per-agent KV reuse ratio + accuracy by mode."""
    # Only rows that have per-agent data
    rows = [r for r in rows if r.get("agent1_reuse_ratio")]
    rows = sorted(rows, key=lambda r: float(r["final_accuracy"]), reverse=True)

    if not rows:
        return "% No per-agent mode data available"

    lines = []
    lines.append(r"""
\begin{table*}[htbp]
\centering
\caption{Per-Agent KV Reuse Ratio and Accuracy by Mode}
\label{tab:per_agent_mode}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l | c c c | c c c | c c c}
\toprule
 & \multicolumn{3}{c|}{\textbf{KV Reuse Ratio (\%)}} & \multicolumn{3}{c|}{\textbf{Dense Prefill Acc. (\%)}} & \multicolumn{3}{c}{\textbf{KV Reuse Acc. (\%)}} \\
\textbf{Experiment} & \textbf{Agent 1} & \textbf{Agent 2} & \textbf{Agent 3} & \textbf{Agent 1} & \textbf{Agent 2} & \textbf{Agent 3} & \textbf{Agent 1} & \textbf{Agent 2} & \textbf{Agent 3} \\
\midrule""")

    for r in rows:
        name = get_short_name(r)
        rr1 = fmt_pct(r.get("agent1_reuse_ratio", ""))
        rr2 = fmt_pct(r.get("agent2_reuse_ratio", ""))
        rr3 = fmt_pct(r.get("agent3_reuse_ratio", ""))
        da1 = fmt_pct(r.get("agent1_dense_acc", ""))
        da2 = fmt_pct(r.get("agent2_dense_acc", ""))
        da3 = fmt_pct(r.get("agent3_dense_acc", ""))
        ra1 = fmt_pct(r.get("agent1_reuse_acc", ""))
        ra2 = fmt_pct(r.get("agent2_reuse_acc", ""))
        ra3 = fmt_pct(r.get("agent3_reuse_acc", ""))

        lines.append(
            f"  {name} & {rr1} & {rr2} & {rr3} & {da1} & {da2} & {da3} & {ra1} & {ra2} & {ra3} \\\\"
        )

    lines.append(r"""\bottomrule
\end{tabular}%
}
\end{table*}""")
    return "\n".join(lines)


def table_crs(rows: list[dict]) -> str:
    """Table 4: CRS-specific metrics."""
    rows = [r for r in rows if r.get("crs_applied_count") and r["crs_applied_count"]]
    if not rows:
        return "% No CRS data available"

    rows = sorted(rows, key=lambda r: float(r["final_accuracy"]), reverse=True)

    lines = []
    lines.append(r"""
\begin{table*}[htbp]
\centering
\caption{Cross-Reference Sharing (CRS) Statistics}
\label{tab:crs}
\begin{tabular}{l c c c c c}
\toprule
\textbf{Experiment} & \textbf{CRS Mode} & \textbf{CRS Count} & \textbf{CRS Ratio (\%)} & \textbf{KV Reuse (\%)} & \textbf{Overall Acc. (\%)} \\
\midrule""")

    for r in rows:
        name = get_short_name(r)
        crs_mode = r["crs_status"]
        crs_count = r.get("crs_applied_count", "0")
        crs_ratio = fmt_pct(r.get("crs_ratio", ""))
        reuse = fmt_pct(r.get("cumulative_reuse_rate", ""))
        acc = fmt_pct(r["final_accuracy"])

        lines.append(
            f"  {name} & {crs_mode} & {crs_count} & {crs_ratio} & {reuse} & {acc} \\\\"
        )

    lines.append(r"""\bottomrule
\end{tabular}
\end{table*}""")
    return "\n".join(lines)


def table_per_agent_latency(rows: list[dict]) -> str:
    """Table 5: Per-agent latency breakdown for key experiments."""
    # Only show a few representative experiments
    key_exps = [
        "ablation_1_baseline_rtx_pro",
        "crs_priority_ablation_off_20260405_195800",
        "crs_priority_ablation_on_20260405_195800",
        "ablation_3_paged_fa_rtx_pro",
    ]
    rows = [r for r in rows if r["experiment"] in key_exps]
    rows = sorted(rows, key=lambda r: float(r["final_accuracy"]), reverse=True)

    if not rows:
        return "% No per-agent latency data for key experiments"

    lines = []
    lines.append(r"""
\begin{table*}[htbp]
\centering
\caption{Per-Agent Latency (ms): KV Reuse Mode -- Key Experiments}
\label{tab:per_agent_latency}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l | r r r | r r r | r r r | r r r}
\toprule
 & \multicolumn{3}{c|}{\textbf{Agent 0 (FinalRefer)}} & \multicolumn{3}{c|}{\textbf{Agent 1}} & \multicolumn{3}{c|}{\textbf{Agent 2}} & \multicolumn{3}{c}{\textbf{Agent 3}} \\
\textbf{Experiment} & \textbf{TTFT} & \textbf{Gen} & \textbf{Pre} & \textbf{TTFT} & \textbf{Gen} & \textbf{Pre} & \textbf{TTFT} & \textbf{Gen} & \textbf{Pre} & \textbf{TTFT} & \textbf{Gen} & \textbf{Pre} \\
\midrule""")

    for r in rows:
        name = get_short_name(r)
        cols = []
        for aid in ["0", "1", "2", "3"]:
            ttft = fmt_ms(r.get(f"agent{aid}_kv_reuse_avg_ttft", ""))
            gen = fmt_ms(r.get(f"agent{aid}_kv_reuse_avg_gen_ttft", ""))
            pre = fmt_ms(r.get(f"agent{aid}_kv_reuse_avg_preprocess", ""))
            cols.extend([ttft, gen, pre])

        lines.append(f"  {name} & " + " & ".join(cols) + r" \\")

    lines.append(r"""\bottomrule
\end{tabular}%
}
\end{table*}""")
    return "\n".join(lines)


def table_experiment_log(rows: list[dict]) -> str:
    """Table 6: Experiment run log with job IDs, GPU, runs, time."""
    rows = sorted(rows, key=lambda r: r.get("start_time", ""), reverse=False)

    lines = []
    lines.append(r"""
\begin{table*}[htbp]
\centering
\caption{Experiment Run Log: All GSM8K Experiments with Job Metadata}
\label{tab:experiment_log}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l c c c r r c c l}
\toprule
\textbf{Experiment} & \textbf{Backend} & \textbf{CRS} & \textbf{GPU} & \textbf{Runs} & \textbf{Samples} & \textbf{Acc. (\%)} & \textbf{Reuse (\%)} & \textbf{Slurm Job IDs} \\
\midrule""")

    for r in rows:
        name = get_short_name(r)
        backend = r["backend"].upper()
        crs = r["crs_status"]
        gpu = r.get("gpu", "N/A")
        if gpu == "N/A":
            gpu = "--"
        elif "RTX PRO" in gpu:
            gpu = "RTX PRO"
        elif "A6000" in gpu:
            gpu = "A6000"
        num_runs = r.get("num_runs", "?")
        samples = r["num_samples"]
        acc = fmt_pct(r["final_accuracy"])
        reuse = fmt_pct(r.get("cumulative_reuse_rate", ""))
        job_ids = r.get("slurm_job_ids", "")
        if isinstance(job_ids, list):
            job_ids = ", ".join(job_ids)
        if not job_ids:
            job_ids = "--"

        lines.append(
            f"  {name} & {backend} & {crs} & {gpu} & {num_runs} & {samples} & {acc} & {reuse} & {job_ids} \\\\"
        )

    lines.append(r"""\bottomrule
\end{tabular}%
}
\end{table*}""")
    return "\n".join(lines)


def table_time_log(rows: list[dict]) -> str:
    """Table 7: Experiment time log."""
    rows = sorted(rows, key=lambda r: r.get("start_time", ""))

    lines = []
    lines.append(r"""
\begin{table*}[htbp]
\centering
\caption{Experiment Timeline}
\label{tab:timeline}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l r c c c}
\toprule
\textbf{Experiment} & \textbf{Runs} & \textbf{Log Files} & \textbf{Start} & \textbf{End} \\
\midrule""")

    for r in rows:
        name = get_short_name(r)
        num_runs = r.get("num_runs", "?")
        n_logs = r.get("num_log_files", "?")
        start = r.get("start_time", "--")
        end = r.get("end_time", "--")
        # Shorten timestamps
        if start and len(start) > 10:
            start = start[:16]  # YYYY-MM-DD HH:MM
        if end and len(end) > 10:
            end = end[:16]

        lines.append(
            f"  {name} & {num_runs} & {n_logs} & {start} & {end} \\\\"
        )

    lines.append(r"""\bottomrule
\end{tabular}%
}
\end{table*}""")
    return "\n".join(lines)


def main():
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}", file=sys.stderr)
        print("Run summarize_experiments.py first.", file=sys.stderr)
        sys.exit(1)

    rows = load_data(str(CSV_PATH))
    deduped = deduplicate(rows)

    output_path = CSV_PATH.parent / "latex_tables.tex"

    preamble = r"""%% Auto-generated LaTeX tables for KVCOMM GSM8K experiments
%% Generated by: KVCOMM/scripts/generate_latex_tables.py
%%
%% Required packages:
%%   \usepackage{booktabs}
%%   \usepackage{graphicx}   % for \resizebox
%%   \usepackage{amssymb}    % for \checkmark
%%   \newcommand{\cmark}{\checkmark}
%%   \newcommand{\xmark}{$\times$}
"""

    tables = [
        preamble,
        "% ════════════════════════════════════════════════════════════════",
        "% Table 1: Accuracy + KV Reuse Summary (deduplicated)",
        "% ════════════════════════════════════════════════════════════════",
        table_main_accuracy(deduped),
        "",
        "% ════════════════════════════════════════════════════════════════",
        "% Table 2: Latency Breakdown (deduplicated)",
        "% ════════════════════════════════════════════════════════════════",
        table_latency(deduped),
        "",
        "% ════════════════════════════════════════════════════════════════",
        "% Table 3: Per-Agent Mode Accuracy (deduplicated)",
        "% ════════════════════════════════════════════════════════════════",
        table_per_agent_mode(deduped),
        "",
        "% ════════════════════════════════════════════════════════════════",
        "% Table 4: CRS Statistics",
        "% ════════════════════════════════════════════════════════════════",
        table_crs(deduped),
        "",
        "% ════════════════════════════════════════════════════════════════",
        "% Table 5: Per-Agent Latency (key experiments)",
        "% ════════════════════════════════════════════════════════════════",
        table_per_agent_latency(rows),
        "",
        "% ════════════════════════════════════════════════════════════════",
        "% Table 6: Full Experiment Log with Job IDs (ALL experiments)",
        "% ════════════════════════════════════════════════════════════════",
        table_experiment_log(rows),
        "",
        "% ════════════════════════════════════════════════════════════════",
        "% Table 7: Experiment Timeline (ALL experiments)",
        "% ════════════════════════════════════════════════════════════════",
        table_time_log(rows),
        "",
        "% ════════════════════════════════════════════════════════════════",
        "% FULL TABLES (all rows, including duplicates)",
        "% ════════════════════════════════════════════════════════════════",
        "",
        "% Table A1: All experiments accuracy",
        table_main_accuracy(rows).replace("tab:accuracy_summary", "tab:accuracy_full")
                                 .replace("Accuracy and KV Reuse Summary", "All Experiments: Accuracy and KV Reuse"),
        "",
        "% Table A2: All experiments latency",
        table_latency(rows).replace("tab:latency", "tab:latency_full")
                           .replace("Latency Breakdown by Mode", "All Experiments: Latency Breakdown"),
    ]

    content = "\n".join(tables)

    with open(output_path, "w") as f:
        f.write(content)

    print(f"LaTeX tables written to: {output_path}")
    print(f"  - {len(deduped)} unique experiments (deduplicated)")
    print(f"  - {len(rows)} total experiments")
    print()
    print(content)


if __name__ == "__main__":
    main()
