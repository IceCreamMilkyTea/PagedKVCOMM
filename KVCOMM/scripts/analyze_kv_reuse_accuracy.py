#!/usr/bin/env python3
"""
Analyze per-agent accuracy broken down by KV reuse vs dense prefill mode.

Parses the ablation log to extract:
1. Per-batch per-agent execution mode (from MODE_EXECUTE lines)
2. Per-batch per-node cumulative accuracy (from NODE_ACCURACY lines)
3. Derives per-batch per-node correctness by diffing cumulative stats
4. Groups by (node, mode) and computes accuracy
"""

import re
import sys
import json
from collections import defaultdict
from pathlib import Path


def parse_log(log_path: str):
    """Parse the ablation log and return per-batch per-agent mode and correctness."""

    # ── 1. Extract MODE_EXECUTE: (request_uid, node, mode) ──
    # Pattern: [MODE_EXECUTE:hf] node=1 role=Math Solver request_uid=XXX mode=dense_prefill
    mode_re = re.compile(
        r"\[MODE_EXECUTE:(?:hf|paged)\]\s+node=(\d+)\s+role=(.+?)\s+request_uid=(\w+)\s+mode=(\w+)"
    )

    # ── 2. Extract REQUEST REUSE JSON: maps request_uid -> batch_index ──
    reuse_re = re.compile(r"\[REQUEST REUSE\]\s+(\{.*\})")

    # ── 3. Extract NODE_ACCURACY: cumulative per-node accuracy ──
    # Pattern: [NODE_ACCURACY] node=1: 0.7726  node=2: 0.3730  node=3: 0.7445
    node_acc_re = re.compile(r"\[NODE_ACCURACY\]\s+(.*)")
    node_val_re = re.compile(r"node=(\d+):\s+([\d.]+)")

    # Storage
    uid_to_modes = defaultdict(dict)  # {request_uid: {node_id: mode}}
    uid_to_batch = {}                 # {request_uid: batch_index}
    uid_to_roles = defaultdict(dict)  # {request_uid: {node_id: role}}
    cumulative_accs = []              # [{node_id: cumulative_acc}, ...]

    with open(log_path, "r") as f:
        for line in f:
            # MODE_EXECUTE
            m = mode_re.search(line)
            if m:
                node_id = int(m.group(1))
                role = m.group(2)
                uid = m.group(3)
                mode = m.group(4)
                uid_to_modes[uid][node_id] = mode
                uid_to_roles[uid][node_id] = role
                continue

            # REQUEST REUSE
            m = reuse_re.search(line)
            if m:
                data = json.loads(m.group(1))
                uid_to_batch[data["request_uid"]] = data["batch_index"]
                continue

            # NODE_ACCURACY
            m = node_acc_re.search(line)
            if m:
                accs = {}
                for vm in node_val_re.finditer(m.group(1)):
                    accs[int(vm.group(1))] = float(vm.group(2))
                cumulative_accs.append(accs)

    # ── Build batch -> {node: mode} mapping ──
    batch_modes = {}  # {batch_index: {node_id: mode}}
    batch_roles = {}  # {batch_index: {node_id: role}}
    for uid, batch_idx in uid_to_batch.items():
        batch_modes[batch_idx] = uid_to_modes.get(uid, {})
        batch_roles[batch_idx] = uid_to_roles.get(uid, {})

    # ── Derive per-batch per-node correctness from cumulative accuracy ──
    # cumulative_acc[i] = node_solved[i] / (i + 1)
    # node_solved[i] = cumulative_acc[i] * (i + 1)
    # correct_at_i = node_solved[i] - node_solved[i-1]
    #              = cumulative_acc[i] * (i+1) - cumulative_acc[i-1] * i

    # Only nodes 1, 2, 3 are in NODE_ACCURACY (not node 0/FinalRefer)
    agent_nodes = [1, 2, 3]
    batch_correct = {}  # {batch_index: {node_id: bool}}

    for i, accs in enumerate(cumulative_accs):
        batch_correct[i] = {}
        for nid in agent_nodes:
            if nid not in accs:
                continue
            cum_solved = accs[nid] * (i + 1)
            if i == 0:
                prev_solved = 0.0
            else:
                prev_solved = cumulative_accs[i - 1].get(nid, 0.0) * i
            correct = round(cum_solved - prev_solved)  # should be 0 or 1
            batch_correct[i][nid] = correct >= 1

    return batch_modes, batch_roles, batch_correct, len(cumulative_accs)


def analyze(batch_modes, batch_roles, batch_correct, total_batches):
    """Group by (node, mode) and compute accuracy."""

    # {(node_id, mode): [correct_bools]}
    node_mode_results = defaultdict(list)
    # Track role names
    node_role_names = {}

    for batch_idx in range(total_batches):
        modes = batch_modes.get(batch_idx, {})
        correct = batch_correct.get(batch_idx, {})
        roles = batch_roles.get(batch_idx, {})

        for nid in [1, 2, 3]:
            if nid not in modes or nid not in correct:
                continue
            mode = modes[nid]
            node_mode_results[(nid, mode)].append(correct[nid])
            if nid not in node_role_names and nid in roles:
                node_role_names[nid] = roles[nid]

    return node_mode_results, node_role_names


def print_report(node_mode_results, node_role_names, total_batches):
    """Print the analysis report."""

    print("=" * 70)
    print("KV Reuse vs Dense Prefill: Per-Agent Accuracy Analysis")
    print(f"Total batches: {total_batches}")
    print("=" * 70)

    # Organize by node
    nodes = sorted(set(nid for nid, _ in node_mode_results.keys()))

    for nid in nodes:
        role = node_role_names.get(nid, f"Node {nid}")
        print(f"\n{'─' * 60}")
        print(f"  Node {nid}: {role}")
        print(f"{'─' * 60}")

        for mode in ["dense_prefill", "kv_reuse"]:
            key = (nid, mode)
            if key not in node_mode_results:
                print(f"  {mode:20s}:  (no data)")
                continue
            results = node_mode_results[key]
            total = len(results)
            correct = sum(results)
            acc = correct / total if total > 0 else 0
            print(f"  {mode:20s}:  {correct:4d}/{total:4d} = {acc:.4f}  ({acc*100:.2f}%)")

    # Summary table
    print(f"\n{'=' * 70}")
    print("Summary Table")
    print(f"{'=' * 70}")
    print(f"{'Node':<8} {'Role':<25} {'Dense Prefill':>15} {'KV Reuse':>15} {'Delta':>10}")
    print(f"{'─' * 73}")

    for nid in nodes:
        role = node_role_names.get(nid, f"Node {nid}")
        dp_key = (nid, "dense_prefill")
        kr_key = (nid, "kv_reuse")

        dp_results = node_mode_results.get(dp_key, [])
        kr_results = node_mode_results.get(kr_key, [])

        dp_acc = sum(dp_results) / len(dp_results) if dp_results else float("nan")
        kr_acc = sum(kr_results) / len(kr_results) if kr_results else float("nan")

        dp_str = f"{dp_acc*100:.2f}%" if dp_results else "N/A"
        kr_str = f"{kr_acc*100:.2f}%" if kr_results else "N/A"

        if dp_results and kr_results:
            delta = (kr_acc - dp_acc) * 100
            delta_str = f"{delta:+.2f}%"
        else:
            delta_str = "N/A"

        print(f"{nid:<8} {role:<25} {dp_str:>15} {kr_str:>15} {delta_str:>10}")

    # Mode distribution
    print(f"\n{'=' * 70}")
    print("Mode Distribution per Node")
    print(f"{'=' * 70}")
    for nid in nodes:
        role = node_role_names.get(nid, f"Node {nid}")
        dp_n = len(node_mode_results.get((nid, "dense_prefill"), []))
        kr_n = len(node_mode_results.get((nid, "kv_reuse"), []))
        total = dp_n + kr_n
        print(f"  Node {nid} ({role}):")
        print(f"    dense_prefill: {dp_n:4d} ({dp_n/total*100:.1f}%)")
        print(f"    kv_reuse:      {kr_n:4d} ({kr_n/total*100:.1f}%)")


def main():
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        # Default to latest ablation stage1 log
        log_path = "KVCOMM/logs/4-5_page_abl_stage1_rtx_pro/ablation_1_baseline_rtx_pro.log"

    log_path = Path(log_path)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)

    print(f"Parsing log: {log_path}")
    batch_modes, batch_roles, batch_correct, total_batches = parse_log(str(log_path))
    print(f"Parsed {total_batches} batches\n")

    node_mode_results, node_role_names = analyze(
        batch_modes, batch_roles, batch_correct, total_batches
    )
    print_report(node_mode_results, node_role_names, total_batches)


if __name__ == "__main__":
    main()
