#!/usr/bin/env python3
"""
Summarize all GSM8K experiments from KVCOMM/result/.

For each completed experiment (1319 samples = full GSM8K test set), extracts:
  - Experiment settings from [CONFIG] log line
  - Backend (HF / Paged) from log tags
  - Overall accuracy + per-agent (node 1,2,3) accuracy
  - Per-mode (dense_prefill / kv_reuse) avg TTFT, avg preprocess_latency, avg generation_ttft
  - KV reuse ratio (cumulative)
  - Per-agent KV reuse ratio + per-agent accuracy broken down by mode
  - CRS ratio (if CRS is on)
  - CRS-gated accuracy (accuracy when CRS was applied vs not)
  - Log file path

Usage:
    python KVCOMM/scripts/summarize_experiments.py [--min-samples 1319] [--csv output.csv]
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

RESULT_DIR = Path(__file__).resolve().parent.parent / "result"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


# ─────────────────────────────────────────────────────────────────────
# Log parsing helpers
# ─────────────────────────────────────────────────────────────────────

def find_log_files(exp_dir: Path) -> list[Path]:
    """Return all log .txt files sorted by modification time (newest last)."""
    logs_dir = exp_dir / "logs"
    if not logs_dir.is_dir():
        return []
    files = sorted(logs_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime)
    return files


def find_gsm8k_json(exp_dir: Path) -> Path | None:
    """Find the GSM8K results JSON file."""
    gsm8k_dir = exp_dir / "gsm8k_"
    if not gsm8k_dir.is_dir():
        return None
    jsons = list(gsm8k_dir.rglob("*.json"))
    if not jsons:
        return None
    return max(jsons, key=lambda p: p.stat().st_mtime)


def concat_log_lines(log_files: list[Path]) -> list[str]:
    """Concatenate all log file lines."""
    lines = []
    for f in log_files:
        with open(f, "r", errors="replace") as fh:
            lines.extend(fh.readlines())
    return lines


# ─────────────────────────────────────────────────────────────────────
# Parse experiment config from [CONFIG] line
# ─────────────────────────────────────────────────────────────────────

CONFIG_RE = re.compile(r"\[CONFIG\]\s+(.*)")

def parse_config(lines: list[str]) -> dict:
    """Extract config from [CONFIG] log line."""
    config = {}
    for line in lines:
        m = CONFIG_RE.search(line)
        if m:
            raw = m.group(1)
            # Parse key: value pairs separated by commas
            for part in raw.split(","):
                part = part.strip()
                if ":" in part:
                    k, v = part.split(":", 1)
                    config[k.strip()] = v.strip()
            break
    return config


# ─────────────────────────────────────────────────────────────────────
# Detect backend from log tags
# ─────────────────────────────────────────────────────────────────────

def detect_backend(lines: list[str]) -> str:
    for line in lines[:500]:
        if "MODE_EXECUTE:paged" in line:
            return "paged"
        if "MODE_EXECUTE:hf" in line:
            return "hf"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────
# Parse TTFT / preprocess_latency from Latency.json
# ─────────────────────────────────────────────────────────────────────

def parse_latency(exp_dir: Path) -> dict:
    """Parse Latency.json and compute per-mode, per-agent averages."""
    lat_file = exp_dir / "Latency.json"
    if not lat_file.exists():
        return {}

    with open(lat_file) as f:
        entries = json.load(f)

    if not entries:
        return {}

    result = {}

    # ── Global per-mode stats ──
    for mode in ["dense_prefill", "kv_reuse"]:
        subset = [e for e in entries if e.get("mode") == mode]
        if not subset:
            continue
        ttfts = [e["ttft"] for e in subset]
        gen_ttfts = [e["generation_ttft"] for e in subset]
        pp_lats = [e["preprocess_latency"] for e in subset if "preprocess_latency" in e]

        result[f"{mode}_count"] = len(subset)
        result[f"{mode}_avg_ttft"] = sum(ttfts) / len(ttfts)
        result[f"{mode}_avg_gen_ttft"] = sum(gen_ttfts) / len(gen_ttfts)
        if pp_lats:
            result[f"{mode}_avg_preprocess"] = sum(pp_lats) / len(pp_lats)

    # ── Per-agent per-mode stats ──
    agent_ids = sorted(set(e.get("agent_id", "?") for e in entries))
    for aid in agent_ids:
        for mode in ["dense_prefill", "kv_reuse"]:
            subset = [e for e in entries if e.get("agent_id") == aid and e.get("mode") == mode]
            if not subset:
                continue
            ttfts = [e["ttft"] for e in subset]
            gen_ttfts = [e["generation_ttft"] for e in subset]
            pp_lats = [e["preprocess_latency"] for e in subset if "preprocess_latency" in e]

            prefix = f"agent{aid}_{mode}"
            result[f"{prefix}_count"] = len(subset)
            result[f"{prefix}_avg_ttft"] = sum(ttfts) / len(ttfts)
            result[f"{prefix}_avg_gen_ttft"] = sum(gen_ttfts) / len(gen_ttfts)
            if pp_lats:
                result[f"{prefix}_avg_preprocess"] = sum(pp_lats) / len(pp_lats)

    return result


# ─────────────────────────────────────────────────────────────────────
# Parse overall + per-node accuracy from logs
# ─────────────────────────────────────────────────────────────────────

NODE_ACC_RE = re.compile(r"\[NODE_ACCURACY\]\s+(.*)")
NODE_VAL_RE = re.compile(r"node=(\d+):\s+([\d.]+)")

def parse_node_accuracy(lines: list[str]) -> dict[int, float]:
    """Get final per-node accuracy from last NODE_ACCURACY line."""
    last = {}
    for line in lines:
        m = NODE_ACC_RE.search(line)
        if m:
            last = {}
            for vm in NODE_VAL_RE.finditer(m.group(1)):
                last[int(vm.group(1))] = float(vm.group(2))
    return last


# ─────────────────────────────────────────────────────────────────────
# Parse KV reuse ratio from CUMULATIVE REUSE
# ─────────────────────────────────────────────────────────────────────

CUMUL_RE = re.compile(r"\[CUMULATIVE REUSE\]\s+(\{.*\})")

def parse_cumulative_reuse(lines: list[str]) -> dict:
    """Get final cumulative reuse stats."""
    last = {}
    for line in lines:
        m = CUMUL_RE.search(line)
        if m:
            try:
                last = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    return last


# ─────────────────────────────────────────────────────────────────────
# Parse per-agent KV reuse ratio + per-mode accuracy
# (reuse existing analyze_kv_reuse_accuracy.py logic)
# ─────────────────────────────────────────────────────────────────────

MODE_EXEC_RE = re.compile(
    r"\[MODE_EXECUTE:(?:hf|paged)\]\s+node=(\d+)\s+role=(.+?)\s+request_uid=(\w+)\s+mode=(\w+)"
)
REQ_REUSE_RE = re.compile(r"\[REQUEST REUSE\]\s+(\{.*\})")

def parse_per_agent_mode_accuracy(lines: list[str]) -> dict:
    """
    Parse per-agent mode distribution and per-mode accuracy.
    Returns dict with:
      - agent{id}_reuse_ratio: fraction of calls that used kv_reuse
      - agent{id}_dense_acc, agent{id}_reuse_acc: accuracy under each mode
    """
    uid_to_modes = defaultdict(dict)
    uid_to_batch = {}
    cumulative_accs = []

    for line in lines:
        m = MODE_EXEC_RE.search(line)
        if m:
            nid, role, uid, mode = int(m.group(1)), m.group(2), m.group(3), m.group(4)
            uid_to_modes[uid][nid] = mode
            continue

        m = REQ_REUSE_RE.search(line)
        if m:
            data = json.loads(m.group(1))
            uid_to_batch[data["request_uid"]] = data["batch_index"]
            continue

        m = NODE_ACC_RE.search(line)
        if m:
            accs = {}
            for vm in NODE_VAL_RE.finditer(m.group(1)):
                accs[int(vm.group(1))] = float(vm.group(2))
            cumulative_accs.append(accs)

    if not cumulative_accs:
        return {}

    # Build batch -> {node: mode}
    batch_modes = {}
    for uid, bidx in uid_to_batch.items():
        batch_modes[bidx] = uid_to_modes.get(uid, {})

    # Derive per-batch correctness from cumulative accuracy
    agent_nodes = [1, 2, 3]
    batch_correct = {}
    for i, accs in enumerate(cumulative_accs):
        batch_correct[i] = {}
        for nid in agent_nodes:
            if nid not in accs:
                continue
            cum_solved = accs[nid] * (i + 1)
            prev_solved = cumulative_accs[i - 1].get(nid, 0.0) * i if i > 0 else 0.0
            correct = round(cum_solved - prev_solved)
            batch_correct[i][nid] = correct >= 1

    # Compute per-agent per-mode accuracy + reuse ratio
    result = {}
    for nid in agent_nodes:
        mode_results = defaultdict(list)
        for bidx in range(len(cumulative_accs)):
            modes = batch_modes.get(bidx, {})
            correct = batch_correct.get(bidx, {})
            if nid in modes and nid in correct:
                mode_results[modes[nid]].append(correct[nid])

        dense = mode_results.get("dense_prefill", [])
        reuse = mode_results.get("kv_reuse", [])
        total = len(dense) + len(reuse)

        if total > 0:
            result[f"agent{nid}_reuse_ratio"] = len(reuse) / total

        if dense:
            result[f"agent{nid}_dense_count"] = len(dense)
            result[f"agent{nid}_dense_acc"] = sum(dense) / len(dense)
        if reuse:
            result[f"agent{nid}_reuse_count"] = len(reuse)
            result[f"agent{nid}_reuse_acc"] = sum(reuse) / len(reuse)

    return result


# ─────────────────────────────────────────────────────────────────────
# Parse CRS stats
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# Parse run metadata: restarts, timestamps, job IDs, GPU
# ─────────────────────────────────────────────────────────────────────

TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

def parse_run_metadata(lines: list[str], exp_name: str) -> dict:
    """Extract run metadata: number of restarts, start/end time, job IDs, GPU."""
    # Count CONFIG lines = number of (re)starts
    num_runs = sum(1 for line in lines if "[CONFIG]" in line)

    # Extract first and last timestamp
    first_ts = None
    last_ts = None
    for line in lines:
        m = TIMESTAMP_RE.match(line)
        if m:
            if first_ts is None:
                first_ts = m.group(1)
            last_ts = m.group(1)

    # Find slurm job IDs from KVCOMM/logs/
    job_ids = _find_slurm_job_ids(exp_name)

    # Detect GPU from slurm logs
    gpu = _detect_gpu(exp_name)

    return {
        "num_runs": num_runs,
        "start_time": first_ts or "N/A",
        "end_time": last_ts or "N/A",
        "slurm_job_ids": job_ids,
        "gpu": gpu,
    }


def _find_slurm_job_ids(exp_name: str) -> list[str]:
    """Find slurm job IDs associated with an experiment by matching log files."""
    job_ids = []

    if not LOGS_DIR.is_dir():
        return job_ids

    # Strategy 1: Direct tee'd .log files in KVCOMM/logs/ matching experiment name
    for log_file in LOGS_DIR.glob("*.log"):
        if exp_name in log_file.stem:
            try:
                with open(log_file, "r", errors="replace") as f:
                    for line in f:
                        if "Job ID:" in line:
                            parts = line.strip().split()
                            if parts:
                                jid = parts[-1]
                                if jid.isdigit():
                                    job_ids.append(jid)
                            break
            except Exception:
                pass

    # Strategy 2: Date directories (4-5/, 4-6/, etc.) - check slurm logs
    for log_dir in LOGS_DIR.iterdir():
        if not log_dir.is_dir():
            continue
        for lf in log_dir.glob("*.log"):
            fname = lf.stem
            job_id_match = re.search(r"(\d{7,})", fname)
            if not job_id_match:
                continue
            job_id = job_id_match.group(1)

            try:
                with open(lf, "r", errors="replace") as f:
                    content = f.read()
                if exp_name in content or f"result/{exp_name}" in content:
                    job_ids.append(job_id)
            except Exception:
                pass

    # Strategy 3: Check date dir tee'd .log files (not slurm numbered ones)
    # e.g., 4-3_page_ablation_rtx_pro/ablation_3_paged_fa_rtx_pro.log
    for log_dir in LOGS_DIR.iterdir():
        if not log_dir.is_dir():
            continue
        for lf in log_dir.glob("*.log"):
            if exp_name in lf.stem:
                # Found a tee'd log matching this experiment
                # Get the job ID from sibling files in same dir
                for sibling in log_dir.glob("*_*.log"):
                    jid_match = re.search(r"_(\d{7,})\.", sibling.name)
                    if jid_match:
                        job_ids.append(jid_match.group(1))
                break

    return sorted(set(job_ids))


def _detect_gpu(exp_name: str) -> str:
    """Detect GPU type from slurm logs or experiment name."""
    if "rtx_pro" in exp_name:
        return "RTX PRO 6000"
    if "a6000" in exp_name:
        return "A6000"

    # Try to detect from slurm logs
    if not LOGS_DIR.is_dir():
        return "N/A"

    for log_dir in LOGS_DIR.iterdir():
        # Check direct .log files
        if not log_dir.is_dir() and log_dir.suffix == ".log" and exp_name in log_dir.stem:
            try:
                with open(log_dir, "r", errors="replace") as f:
                    for line in f:
                        if "NVIDIA RTX PRO 6000" in line:
                            return "RTX PRO 6000"
                        if "NVIDIA RTX A6000" in line or "NVIDIA A6000" in line:
                            return "A6000"
            except Exception:
                pass
            break

        # Check date directories
        if log_dir.is_dir():
            for lf in log_dir.glob("*.log"):
                try:
                    with open(lf, "r", errors="replace") as f:
                        content = f.read(20000)
                    if exp_name not in content and f"result/{exp_name}" not in content:
                        continue
                    if "RTX PRO 6000" in content:
                        return "RTX PRO 6000"
                    if "RTX A6000" in content or "NVIDIA A6000" in content:
                        return "A6000"
                except Exception:
                    pass

    return "N/A"


CRS_APPLIED_RE = re.compile(r"\[CRS:(?:hf|paged)\] APPLIED")

def parse_crs_stats(lines: list[str], total_agent_calls: int) -> dict:
    """Count CRS applications and compute ratio."""
    crs_count = sum(1 for line in lines if CRS_APPLIED_RE.search(line))
    if crs_count == 0:
        return {}
    result = {"crs_applied_count": crs_count}
    if total_agent_calls > 0:
        result["crs_ratio"] = crs_count / total_agent_calls
    return result


# ─────────────────────────────────────────────────────────────────────
# Main: iterate all experiments
# ─────────────────────────────────────────────────────────────────────

def analyze_experiment(exp_dir: Path) -> dict | None:
    """Analyze a single experiment directory. Returns None if incomplete."""
    name = exp_dir.name

    # Check for GSM8K results
    gsm8k_file = find_gsm8k_json(exp_dir)
    if gsm8k_file is None:
        return None

    with open(gsm8k_file) as f:
        gsm8k_data = json.load(f)

    num_samples = len(gsm8k_data)
    if num_samples == 0:
        return None

    final_acc = gsm8k_data[-1].get("Accuracy", 0.0)

    # Find and parse logs
    log_files = find_log_files(exp_dir)
    if not log_files:
        return None

    lines = concat_log_lines(log_files)

    # Config
    config = parse_config(lines)
    backend = detect_backend(lines)

    # Determine CRS status from config or experiment name
    crs_priority = config.get("CRS Priority", "N/A")
    # Detect if --no-current-round-sharing was used
    has_crs_applied = any("CRS" in line and "APPLIED" in line for line in lines[:10000])
    # Check from experiment name or config
    if "no-current-round-sharing" in " ".join(lines[:100]) or "_off_" in name:
        crs_status = "OFF"
    elif crs_priority == "True":
        crs_status = "PRIORITY"
    elif has_crs_applied:
        crs_status = "ON"
    else:
        crs_status = "OFF"

    # Node accuracy
    node_accs = parse_node_accuracy(lines)

    # Cumulative reuse
    cum_reuse = parse_cumulative_reuse(lines)

    # Latency stats
    lat_stats = parse_latency(exp_dir)

    # Per-agent mode accuracy
    mode_acc = parse_per_agent_mode_accuracy(lines)

    # CRS stats
    total_agent_calls = cum_reuse.get("total_agent_calls", 0)
    crs_stats = parse_crs_stats(lines, total_agent_calls)

    # Run metadata
    run_meta = parse_run_metadata(lines, name)

    # Log file path (relative)
    log_paths = [str(f.relative_to(RESULT_DIR.parent.parent)) for f in log_files]

    return {
        "experiment": name,
        "num_samples": num_samples,
        "backend": backend,
        "execution_mode": config.get("Execution", "N/A"),
        "flash_attention": config.get("Flash Attention", "N/A"),
        "crs_status": crs_status,
        "crs_priority": crs_priority,
        "final_accuracy": final_acc,
        "node1_acc": node_accs.get(1),
        "node2_acc": node_accs.get(2),
        "node3_acc": node_accs.get(3),
        "cumulative_reuse_rate": cum_reuse.get("cumulative_reuse_rate"),
        "kv_reuse_calls": cum_reuse.get("kv_reuse_calls"),
        "total_agent_calls": total_agent_calls,
        **lat_stats,
        **mode_acc,
        **crs_stats,
        "num_runs": run_meta["num_runs"],
        "start_time": run_meta["start_time"],
        "end_time": run_meta["end_time"],
        "slurm_job_ids": run_meta["slurm_job_ids"],
        "gpu": run_meta["gpu"],
        "num_log_files": len(log_files),
        "log_paths": log_paths,
    }


def print_report(experiments: list[dict]):
    """Print a formatted report of all experiments."""

    W = 100
    print("=" * W)
    print("KVCOMM GSM8K Experiment Summary")
    print(f"Total experiments analyzed: {len(experiments)}")
    print("=" * W)

    for exp in experiments:
        print(f"\n{'━' * W}")
        print(f"  {exp['experiment']}")
        print(f"{'━' * W}")

        # Settings
        print(f"  Samples: {exp['num_samples']}  |  Backend: {exp['backend']}  |  "
              f"Execution: {exp['execution_mode']}  |  FA: {exp['flash_attention']}  |  "
              f"CRS: {exp['crs_status']}  |  CRS Priority: {exp['crs_priority']}")

        # Run metadata
        job_ids = exp.get("slurm_job_ids", [])
        job_str = ", ".join(job_ids) if job_ids else "N/A"
        gpu = exp.get("gpu", "N/A")
        num_runs = exp.get("num_runs", "?")
        start = exp.get("start_time", "N/A")
        end = exp.get("end_time", "N/A")
        n_logs = exp.get("num_log_files", "?")
        print(f"  Runs: {num_runs}  |  GPU: {gpu}  |  Job IDs: {job_str}")
        print(f"  Time: {start} -> {end}  |  Log files: {n_logs}")

        # Overall accuracy
        print(f"\n  Overall Accuracy: {exp['final_accuracy']:.4f} ({exp['final_accuracy']*100:.2f}%)")

        # Per-node accuracy
        n1 = exp.get("node1_acc")
        n2 = exp.get("node2_acc")
        n3 = exp.get("node3_acc")
        if n1 is not None:
            print(f"  Node Accuracy:  node1={n1:.4f}  node2={n2:.4f}  node3={n3:.4f}")

        # KV reuse ratio
        cr = exp.get("cumulative_reuse_rate")
        if cr is not None:
            print(f"\n  KV Reuse Rate: {cr:.4f} ({cr*100:.2f}%)  "
                  f"[{exp.get('kv_reuse_calls', 0)}/{exp.get('total_agent_calls', 0)} agent calls]")

        # Per-agent reuse ratio + mode accuracy
        has_mode_data = any(f"agent{n}_reuse_ratio" in exp for n in [1, 2, 3])
        if has_mode_data:
            print(f"\n  {'Agent':<8} {'Reuse%':>8} {'Dense#':>8} {'Dense Acc':>10} {'Reuse#':>8} {'Reuse Acc':>10}")
            print(f"  {'─' * 56}")
            for nid in [1, 2, 3]:
                rr = exp.get(f"agent{nid}_reuse_ratio")
                dc = exp.get(f"agent{nid}_dense_count", 0)
                da = exp.get(f"agent{nid}_dense_acc")
                rc = exp.get(f"agent{nid}_reuse_count", 0)
                ra = exp.get(f"agent{nid}_reuse_acc")
                rr_s = f"{rr*100:.1f}%" if rr is not None else "N/A"
                da_s = f"{da*100:.2f}%" if da is not None else "N/A"
                ra_s = f"{ra*100:.2f}%" if ra is not None else "N/A"
                print(f"  node{nid:<3} {rr_s:>8} {dc:>8} {da_s:>10} {rc:>8} {ra_s:>10}")

        # Latency summary
        dp_ttft = exp.get("dense_prefill_avg_ttft")
        kr_ttft = exp.get("kv_reuse_avg_ttft")
        if dp_ttft is not None or kr_ttft is not None:
            print(f"\n  {'Mode':<16} {'Count':>8} {'Avg TTFT':>12} {'Avg GenTTFT':>12} {'Avg Preproc':>12}")
            print(f"  {'─' * 64}")
            if dp_ttft is not None:
                dp_gen = exp.get("dense_prefill_avg_gen_ttft", 0)
                dp_pp = exp.get("dense_prefill_avg_preprocess")
                dp_pp_s = f"{dp_pp:.6f}" if dp_pp is not None else "N/A"
                print(f"  {'dense_prefill':<16} {exp.get('dense_prefill_count', 0):>8} "
                      f"{dp_ttft:>12.6f} {dp_gen:>12.6f} {dp_pp_s:>12}")
            if kr_ttft is not None:
                kr_gen = exp.get("kv_reuse_avg_gen_ttft", 0)
                kr_pp = exp.get("kv_reuse_avg_preprocess")
                kr_pp_s = f"{kr_pp:.6f}" if kr_pp is not None else "N/A"
                print(f"  {'kv_reuse':<16} {exp.get('kv_reuse_count', 0):>8} "
                      f"{kr_ttft:>12.6f} {kr_gen:>12.6f} {kr_pp_s:>12}")

        # CRS stats
        crs_count = exp.get("crs_applied_count")
        if crs_count:
            crs_ratio = exp.get("crs_ratio", 0)
            print(f"\n  CRS Applied: {crs_count} times  "
                  f"({crs_ratio*100:.2f}% of {exp.get('total_agent_calls', 0)} agent calls)")

        # Log paths
        print(f"\n  Log: {exp['log_paths'][-1] if exp['log_paths'] else 'N/A'}")


def write_csv(experiments: list[dict], csv_path: str):
    """Write experiments to CSV."""
    import csv

    # Collect all keys
    all_keys = []
    seen = set()
    priority_keys = [
        "experiment", "num_samples", "backend", "execution_mode",
        "flash_attention", "crs_status", "crs_priority",
        "final_accuracy", "node1_acc", "node2_acc", "node3_acc",
        "cumulative_reuse_rate", "kv_reuse_calls", "total_agent_calls",
        "dense_prefill_count", "dense_prefill_avg_ttft", "dense_prefill_avg_gen_ttft", "dense_prefill_avg_preprocess",
        "kv_reuse_count", "kv_reuse_avg_ttft", "kv_reuse_avg_gen_ttft", "kv_reuse_avg_preprocess",
    ]
    for k in priority_keys:
        if k not in seen:
            all_keys.append(k)
            seen.add(k)

    # Add per-agent keys
    for nid in [1, 2, 3]:
        for suffix in ["_reuse_ratio", "_dense_count", "_dense_acc", "_reuse_count", "_reuse_acc"]:
            k = f"agent{nid}{suffix}"
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # Add per-agent per-mode latency keys
    for aid in ["0", "1", "2", "3"]:
        for mode in ["dense_prefill", "kv_reuse"]:
            for suffix in ["_count", "_avg_ttft", "_avg_gen_ttft", "_avg_preprocess"]:
                k = f"agent{aid}_{mode}{suffix}"
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

    for k in ["crs_applied_count", "crs_ratio"]:
        if k not in seen:
            all_keys.append(k)
            seen.add(k)

    # Run metadata
    for k in ["num_runs", "start_time", "end_time", "gpu", "slurm_job_ids", "num_log_files"]:
        if k not in seen:
            all_keys.append(k)
            seen.add(k)

    all_keys.append("log_path")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for exp in experiments:
            row = dict(exp)
            row["log_path"] = exp["log_paths"][-1] if exp.get("log_paths") else ""
            row["slurm_job_ids"] = ",".join(exp.get("slurm_job_ids", []))
            writer.writerow(row)

    print(f"\nCSV written to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize KVCOMM GSM8K experiments")
    parser.add_argument("--min-samples", type=int, default=100,
                        help="Minimum samples to include experiment (default: 100)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Output CSV file path")
    parser.add_argument("--all", action="store_true",
                        help="Include debug/incomplete experiments too")
    args = parser.parse_args()

    if not RESULT_DIR.is_dir():
        print(f"Result directory not found: {RESULT_DIR}")
        sys.exit(1)

    experiments = []
    for exp_dir in sorted(RESULT_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        # Skip debug experiments by default
        if not args.all and ("debug" in exp_dir.name and exp_dir.name != "gsm8k_debug15_baseline"):
            # Keep debug15 experiments as they might be reference
            if "debug5" in exp_dir.name or "debug_" in exp_dir.name:
                continue

        result = analyze_experiment(exp_dir)
        if result is None:
            continue

        if result["num_samples"] < args.min_samples:
            continue

        experiments.append(result)

    if not experiments:
        print("No experiments found matching criteria.")
        sys.exit(0)

    # Sort by accuracy descending
    experiments.sort(key=lambda x: x["final_accuracy"], reverse=True)

    print_report(experiments)

    if args.csv:
        write_csv(experiments, args.csv)
    else:
        # Default CSV output
        default_csv = str(RESULT_DIR / "experiment_summary.csv")
        write_csv(experiments, default_csv)


if __name__ == "__main__":
    main()
