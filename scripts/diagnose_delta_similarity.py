#!/usr/bin/env python3
"""
Diagnose cross-agent delta STABILITY across queries.

Core hypothesis (from docs/kvcomm_inner_round_improvement.md):
  For the same ph_id, given two different queries (messages) q1 and q2,
  the cross-agent offset should be stable:

    cross_q1 = delta_base→agent_i(q1) - delta_base→agent_j(q1)
    cross_q2 = delta_base→agent_i(q2) - delta_base→agent_j(q2)
    cross_q1 ≈ cross_q2  ?

  If yes → we can precompute cross-agent offset from earlier queries
  and reuse it for new queries without dense prefill.

Usage (auto-hooked into run_gsm8k.py after all batches):
    from scripts.diagnose_delta_similarity import run_diagnostic
    run_diagnostic(output_dir="diagnostic_output")
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ── helpers ────────────────────────────────────────────────────────────────


def _extract_agent_ids_hf(entry: dict) -> List[str]:
    """Return all agent node_ids that have ph_key_delta in this anchor entry."""
    suffix = "_ph_key_delta"
    return sorted(k[: -len(suffix)] for k in entry if k.endswith(suffix))


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two flattened tensors."""
    return F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()


def _l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).float().norm(2).item()


def _align_tokens(*tensors: torch.Tensor) -> List[torch.Tensor]:
    """Trim all tensors to the min token length along dim 2."""
    min_t = min(t.shape[2] for t in tensors)
    return [t[..., :min_t, :] for t in tensors]


# ── data collection ────────────────────────────────────────────────────────


def collect_hf(anchors: dict) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
    """
    Collect per-ph_id, per-message, per-agent deltas from HF anchor store.

    Returns:
        {
            ph_id: {
                message: { agent_id: ph_key_delta [L, H, T, D] }
            }
        }

    Only includes ph_ids that have ≥2 messages each with ≥2 agents
    (needed to compare cross-agent offsets across queries).
    """
    # First pass: collect everything
    raw = {}
    for ph_id, messages_dict in anchors.items():
        for message, entry in messages_dict.items():
            agents = _extract_agent_ids_hf(entry)
            if len(agents) < 2:
                continue
            raw.setdefault(ph_id, {})[message] = {
                aid: entry[f"{aid}_ph_key_delta"] for aid in agents
            }

    # Filter: need ≥2 messages per ph_id to compare across queries
    return {
        ph_id: msgs
        for ph_id, msgs in raw.items()
        if len(msgs) >= 2
    }


def collect_paged(engine) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
    """Same as collect_hf but for paged engine (reads blocks → tensors)."""
    raw = {}
    for ph_id, messages_dict in engine.anchors.items():
        for message, entry in messages_dict.items():
            if len(entry.agent_deltas) < 2:
                continue
            agent_deltas = {}
            for agent_id, delta_info in entry.agent_deltas.items():
                ph_key_delta, _ = engine.read_kv_from_blocks(
                    delta_info["ph_delta_blocks"],
                    delta_info["ph_delta_num_tokens"],
                )
                agent_deltas[agent_id] = ph_key_delta
            raw.setdefault(ph_id, {})[message] = agent_deltas

    return {
        ph_id: msgs
        for ph_id, msgs in raw.items()
        if len(msgs) >= 2
    }


# ── core analysis ──────────────────────────────────────────────────────────


def analyze_cross_offset_stability(
    data: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
) -> Tuple[dict, List[dict]]:
    """
    For each ph_id, for each agent pair, compute cross-agent offset per query
    then compare across queries.

    cross_q = delta_base→agent_i(q) - delta_base→agent_j(q)
    For pairs of queries (q1, q2): compare cross_q1 vs cross_q2.
    """
    records = []

    for ph_id, messages in data.items():
        msg_keys = sorted(messages.keys())
        if len(msg_keys) < 2:
            continue

        # Find agent pairs that appear in ≥2 messages
        # (intersection of agent sets across messages)
        all_agent_sets = [set(messages[m].keys()) for m in msg_keys]
        common_agents = all_agent_sets[0]
        for s in all_agent_sets[1:]:
            common_agents &= s

        if len(common_agents) < 2:
            # Try pairwise message combinations instead
            # Some messages might share agents even if not all do
            for m1, m2 in combinations(msg_keys, 2):
                shared = set(messages[m1].keys()) & set(messages[m2].keys())
                if len(shared) < 2:
                    continue
                shared_sorted = sorted(shared)
                for ai, aj in combinations(shared_sorted, 2):
                    rec = _compare_cross_offsets(
                        ph_id, m1, m2, ai, aj,
                        messages[m1][ai], messages[m1][aj],
                        messages[m2][ai], messages[m2][aj],
                    )
                    if rec:
                        records.append(rec)
        else:
            common_sorted = sorted(common_agents)
            for ai, aj in combinations(common_sorted, 2):
                for m1, m2 in combinations(msg_keys, 2):
                    rec = _compare_cross_offsets(
                        ph_id, m1, m2, ai, aj,
                        messages[m1][ai], messages[m1][aj],
                        messages[m2][ai], messages[m2][aj],
                    )
                    if rec:
                        records.append(rec)

    if not records:
        return {"error": "no cross-query pairs found"}, []

    all_cos = [r["cosine_cross_q1_vs_q2"] for r in records]
    all_l2 = [r["l2_cross_q1_vs_q2"] for r in records]
    all_rel = [r["relative_diff"] for r in records]

    summary = {
        "num_comparisons": len(records),
        "cosine_mean": float(np.mean(all_cos)),
        "cosine_std": float(np.std(all_cos)),
        "cosine_min": float(np.min(all_cos)),
        "cosine_max": float(np.max(all_cos)),
        "l2_mean": float(np.mean(all_l2)),
        "l2_std": float(np.std(all_l2)),
        "relative_diff_mean": float(np.mean(all_rel)),
        "relative_diff_std": float(np.std(all_rel)),
        "high_similarity_count": sum(1 for c in all_cos if c > 0.9),
        "very_high_similarity_count": sum(1 for c in all_cos if c > 0.95),
    }
    return summary, records


def analyze_crs_estimation_error(
    data: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
) -> Tuple[dict, List[dict]]:
    """
    Directly measure CRS estimation accuracy.

    For agent pair (i, j), query q_new, reference query q_old:
        estimated_delta_j(q_new) = delta_i(q_new) + (delta_j(q_old) - delta_i(q_old))
        true_delta_j(q_new) = delta_j(q_new)

    Metric: cosine(estimated, true)  ← how accurate is the CRS estimate?

    Uses each query as q_new once, averaged over all valid q_old choices.
    """
    records = []

    for ph_id, messages in data.items():
        msg_keys = sorted(messages.keys())
        if len(msg_keys) < 2:
            continue

        for q_new in msg_keys:
            agents_new = messages[q_new]
            agent_ids = sorted(agents_new.keys())
            if len(agent_ids) < 2:
                continue

            for ai, aj in combinations(agent_ids, 2):
                if ai not in agents_new or aj not in agents_new:
                    continue

                true_j = agents_new[aj]
                upstream_i_new = agents_new[ai]

                # Average CRS estimate over all valid reference queries
                estimates = []
                for q_old in msg_keys:
                    if q_old == q_new:
                        continue
                    agents_old = messages[q_old]
                    if ai not in agents_old or aj not in agents_old:
                        continue
                    di_old = agents_old[ai]
                    dj_old = agents_old[aj]

                    # Align all to same token length
                    aligned = _align_tokens(upstream_i_new, true_j, di_old, dj_old)
                    i_new_a, j_new_a, i_old_a, j_old_a = aligned

                    cross_old = j_old_a.float() - i_old_a.float()
                    estimate = i_new_a.float() + cross_old
                    estimates.append(estimate)

                if not estimates:
                    continue

                avg_estimate = torch.stack(estimates).mean(0)
                true_j_a, avg_est_a = _align_tokens(true_j.float(), avg_estimate)
                cos = _cosine(true_j_a, avg_est_a)

                records.append({
                    "ph_id": ph_id,
                    "q_new": q_new[:50],
                    "agent_upstream": ai,
                    "agent_target": aj,
                    "cosine_estimate_vs_true": cos,
                    "num_ref_queries": len(estimates),
                })

    if not records:
        return {"error": "no valid CRS estimation pairs found"}, []

    all_cos = [r["cosine_estimate_vs_true"] for r in records]
    summary = {
        "num_comparisons": len(records),
        "cosine_mean": float(np.mean(all_cos)),
        "cosine_std": float(np.std(all_cos)),
        "cosine_min": float(np.min(all_cos)),
        "cosine_max": float(np.max(all_cos)),
        "high_similarity_count": sum(1 for c in all_cos if c > 0.9),
        "very_high_similarity_count": sum(1 for c in all_cos if c > 0.95),
    }
    return summary, records


def _compare_cross_offsets(
    ph_id: str,
    msg1: str, msg2: str,
    agent_i: str, agent_j: str,
    delta_i_q1: torch.Tensor,  # delta_base→agent_i for query 1
    delta_j_q1: torch.Tensor,  # delta_base→agent_j for query 1
    delta_i_q2: torch.Tensor,  # delta_base→agent_i for query 2
    delta_j_q2: torch.Tensor,  # delta_base→agent_j for query 2
) -> Optional[dict]:
    """
    Compute cross_q1 = delta_i_q1 - delta_j_q1
           cross_q2 = delta_i_q2 - delta_j_q2
    Compare cross_q1 vs cross_q2.
    """
    # Align tokens within each query
    di_q1, dj_q1 = _align_tokens(delta_i_q1.float(), delta_j_q1.float())
    di_q2, dj_q2 = _align_tokens(delta_i_q2.float(), delta_j_q2.float())

    cross_q1 = di_q1 - dj_q1  # cross-agent offset for query 1
    cross_q2 = di_q2 - dj_q2  # cross-agent offset for query 2

    # Align cross offsets across queries (they may have different token lengths)
    cross_q1, cross_q2 = _align_tokens(cross_q1, cross_q2)

    if cross_q1.numel() == 0:
        return None

    # Global comparison
    cos_global = _cosine(cross_q1, cross_q2)
    l2_global = _l2(cross_q1, cross_q2)
    # relative diff: ||cross_q1 - cross_q2|| / (0.5 * (||cross_q1|| + ||cross_q2||))
    avg_norm = 0.5 * (cross_q1.norm(2).item() + cross_q2.norm(2).item())
    rel_diff = l2_global / max(avg_norm, 1e-8)

    # Per-layer
    num_layers = cross_q1.shape[0]
    layer_cos = []
    layer_l2 = []
    layer_rel = []
    for l in range(num_layers):
        lcos = _cosine(cross_q1[l], cross_q2[l])
        ll2 = _l2(cross_q1[l], cross_q2[l])
        lavg = 0.5 * (cross_q1[l].norm(2).item() + cross_q2[l].norm(2).item())
        lrel = ll2 / max(lavg, 1e-8)
        layer_cos.append(lcos)
        layer_l2.append(ll2)
        layer_rel.append(lrel)

    return {
        "ph_id": ph_id,
        "query_1": msg1[:50],
        "query_2": msg2[:50],
        "agent_i": agent_i,
        "agent_j": agent_j,
        "cosine_cross_q1_vs_q2": cos_global,
        "l2_cross_q1_vs_q2": l2_global,
        "relative_diff": rel_diff,
        "cross_q1_norm": cross_q1.norm(2).item(),
        "cross_q2_norm": cross_q2.norm(2).item(),
        "per_layer_cosine": layer_cos,
        "per_layer_l2": layer_l2,
        "per_layer_relative_diff": layer_rel,
    }


# ── visualization ──────────────────────────────────────────────────────────


def plot_results(records: List[dict], output_dir: Path):
    """Generate diagnostic plots for cross-offset stability."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_cos = [r["cosine_cross_q1_vs_q2"] for r in records]
    all_rel = [r["relative_diff"] for r in records]
    all_layer_cos = np.array([r["per_layer_cosine"] for r in records])
    all_layer_rel = np.array([r["per_layer_relative_diff"] for r in records])
    num_layers = all_layer_cos.shape[1]
    layers = np.arange(num_layers)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(
        "Cross-Agent Offset Stability Across Queries\n"
        r"$\mathrm{cross}_q = \Delta_{\mathrm{base} \to A_i}(q) - \Delta_{\mathrm{base} \to A_j}(q)$"
        r"  |  Compare $\mathrm{cross}_{q_1}$ vs $\mathrm{cross}_{q_2}$",
        fontsize=13,
        fontweight="bold",
    )

    # 1. Cosine similarity distribution (cross_q1 vs cross_q2)
    ax = axes[0, 0]
    ax.hist(all_cos, bins=max(15, len(all_cos) // 3), color="steelblue",
            edgecolor="white", alpha=0.8)
    ax.axvline(x=np.mean(all_cos), color="orange", linestyle="-",
               label=f"mean={np.mean(all_cos):.4f}")
    ax.axvline(x=0.9, color="green", linestyle="--", alpha=0.6, label="0.9 threshold")
    ax.set_xlabel("Cosine Similarity (cross_q1 vs cross_q2)")
    ax.set_ylabel("Count")
    ax.set_title("Cross-Offset Cosine Similarity")
    ax.legend(fontsize=8)

    # 2. Relative difference distribution
    ax = axes[0, 1]
    ax.hist(all_rel, bins=max(15, len(all_rel) // 3), color="coral",
            edgecolor="white", alpha=0.8)
    ax.axvline(x=np.mean(all_rel), color="orange", linestyle="-",
               label=f"mean={np.mean(all_rel):.4f}")
    ax.set_xlabel("Relative Diff: ||cross_q1 - cross_q2|| / avg_norm")
    ax.set_ylabel("Count")
    ax.set_title("Cross-Offset Relative Difference\n(lower = more stable)")
    ax.legend(fontsize=8)

    # 3. cross_q1 norm vs cross_q2 norm scatter
    ax = axes[0, 2]
    norms_q1 = [r["cross_q1_norm"] for r in records]
    norms_q2 = [r["cross_q2_norm"] for r in records]
    ax.scatter(norms_q1, norms_q2, alpha=0.6, s=30, color="mediumpurple")
    lim = max(max(norms_q1), max(norms_q2)) * 1.1
    ax.plot([0, lim], [0, lim], "r--", alpha=0.5, label="y=x (identical norm)")
    ax.set_xlabel("||cross_q1||")
    ax.set_ylabel("||cross_q2||")
    ax.set_title("Cross-Offset Magnitude Consistency")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    # 4. Per-layer cosine similarity
    ax = axes[1, 0]
    mean_cos = all_layer_cos.mean(axis=0)
    std_cos = all_layer_cos.std(axis=0)
    ax.plot(layers, mean_cos, "o-", color="steelblue", markersize=3)
    ax.fill_between(layers, mean_cos - std_cos, mean_cos + std_cos,
                    alpha=0.2, color="steelblue")
    ax.axhline(y=0.9, color="green", linestyle="--", alpha=0.5, label="0.9")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Per-Layer Cross-Offset Cosine (mean ± std)")
    ax.legend(fontsize=8)

    # 5. Per-layer relative difference
    ax = axes[1, 1]
    mean_rel = all_layer_rel.mean(axis=0)
    std_rel = all_layer_rel.std(axis=0)
    ax.plot(layers, mean_rel, "o-", color="coral", markersize=3)
    ax.fill_between(layers, mean_rel - std_rel, mean_rel + std_rel,
                    alpha=0.2, color="coral")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Relative Difference")
    ax.set_title("Per-Layer Cross-Offset Relative Diff (mean ± std)")

    # 6. Heatmap: per-layer cosine for each comparison
    ax = axes[1, 2]
    if len(records) <= 50:
        im = ax.imshow(all_layer_cos, aspect="auto", cmap="RdYlGn",
                       vmin=0.0, vmax=1.0)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Comparison Index")
        ax.set_title("Per-Layer Cosine Heatmap")
        plt.colorbar(im, ax=ax, shrink=0.8)
    else:
        # Too many records for heatmap, show percentile bands
        p25 = np.percentile(all_layer_cos, 25, axis=0)
        p50 = np.percentile(all_layer_cos, 50, axis=0)
        p75 = np.percentile(all_layer_cos, 75, axis=0)
        ax.plot(layers, p50, "o-", color="steelblue", markersize=3, label="median")
        ax.fill_between(layers, p25, p75, alpha=0.2, color="steelblue", label="25-75th")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Per-Layer Cosine Percentiles")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = output_dir / "cross_offset_stability.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {plot_path}")


# ── entry point ────────────────────────────────────────────────────────────


def run_diagnostic(
    output_dir: str = "diagnostic_output",
    backend: str = "auto",
):
    """Run cross-agent offset stability analysis."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = {}

    if backend in ("auto", "hf"):
        try:
            from KVCOMM.llm.kvcomm_engine import KVCOMMEngine
            hf_data = collect_hf(KVCOMMEngine.anchors)
            if hf_data:
                print(f"[HF] Found {sum(len(m) for m in hf_data.values())} messages "
                      f"across {len(hf_data)} ph_id(s) with multi-agent deltas")
                data.update(hf_data)
        except Exception as e:
            print(f"[HF] Skipped: {e}")

    if backend in ("auto", "paged"):
        try:
            from KVCOMM.llm.paged_llm_chat import PagedLLMChat
            engine = PagedLLMChat._paged_kv_engine
            if engine is not None:
                paged_data = collect_paged(engine)
                if paged_data:
                    print(f"[Paged] Found {sum(len(m) for m in paged_data.values())} messages "
                          f"across {len(paged_data)} ph_id(s) with multi-agent deltas")
                    data.update(paged_data)
        except Exception as e:
            print(f"[Paged] Skipped: {e}")

    if not data:
        print("No multi-agent anchor deltas with ≥2 queries found. Nothing to analyze.")
        print("(Need ≥2 messages per ph_id, each with ≥2 agents)")
        return

    # Analyze: CRS estimation accuracy
    crs_summary, crs_records = analyze_crs_estimation_error(data)
    if "error" not in crs_summary:
        print(f"\n{'='*65}")
        print("CRS ESTIMATION ACCURACY")
        print(f"{'='*65}")
        print(f"  Comparisons (q_new × agent pairs)   : {crs_summary['num_comparisons']}")
        print(f"  cosine(CRS_estimate, true_delta)    : {crs_summary['cosine_mean']:.4f} ± {crs_summary['cosine_std']:.4f}")
        print(f"    min={crs_summary['cosine_min']:.4f}  max={crs_summary['cosine_max']:.4f}")
        print(f"  High accuracy (cos > 0.9)           : {crs_summary['high_similarity_count']} / {crs_summary['num_comparisons']}")
        print(f"  Very high accuracy (cos > 0.95)     : {crs_summary['very_high_similarity_count']} / {crs_summary['num_comparisons']}")
        if crs_summary["cosine_mean"] > 0.9:
            print("  → CRS estimate is HIGHLY ACCURATE.")
        elif crs_summary["cosine_mean"] > 0.7:
            print("  → CRS estimate is MODERATELY ACCURATE.")
        else:
            print("  → CRS estimate is INACCURATE.")
        crs_out = out / "crs_estimation_accuracy.json"
        with open(crs_out, "w") as f:
            json.dump({"summary": crs_summary, "records": crs_records}, f, indent=2)
        print(f"  JSON saved → {crs_out}")

    # Analyze: cross-offset stability across queries
    summary, records = analyze_cross_offset_stability(data)

    if "error" in summary:
        print(f"Analysis failed: {summary['error']}")
        return

    # Print summary
    print(f"\n{'='*65}")
    print("CROSS-AGENT OFFSET STABILITY ACROSS QUERIES")
    print(f"{'='*65}")
    print(f"  Comparisons (query pairs × agent pairs) : {summary['num_comparisons']}")
    print(f"  Cosine sim (cross_q1 vs cross_q2)       : {summary['cosine_mean']:.4f} ± {summary['cosine_std']:.4f}")
    print(f"    min={summary['cosine_min']:.4f}  max={summary['cosine_max']:.4f}")
    print(f"  Relative difference                     : {summary['relative_diff_mean']:.4f} ± {summary['relative_diff_std']:.4f}")
    print(f"  High similarity (cos > 0.9)             : {summary['high_similarity_count']} / {summary['num_comparisons']}")
    print(f"  Very high similarity (cos > 0.95)       : {summary['very_high_similarity_count']} / {summary['num_comparisons']}")
    print()

    if summary["cosine_mean"] > 0.9:
        print("  → Cross-agent offset is HIGHLY STABLE across queries!")
        print("  → Inner-round optimization is very promising.")
    elif summary["cosine_mean"] > 0.7:
        print("  → Cross-agent offset shows MODERATE stability.")
        print("  → Inner-round optimization may help with careful thresholding.")
    else:
        print("  → Cross-agent offset is NOT stable across queries.")
        print("  → Inner-round optimization may not be effective.")

    # Per-record detail
    print(f"\n{'─'*65}")
    print("PER-COMPARISON DETAILS:")
    print(f"  {'ph_id':<20s}  {'agents':>10s}  {'cos':>7s}  {'rel_diff':>8s}  q1_preview / q2_preview")
    for r in records:
        print(
            f"  {r['ph_id'][:20]:<20s}  "
            f"{r['agent_i']}v{r['agent_j']:>5s}  "
            f"{r['cosine_cross_q1_vs_q2']:>7.4f}  "
            f"{r['relative_diff']:>8.4f}  "
            f"{r['query_1'][:25]} / {r['query_2'][:25]}"
        )

    # Plot
    try:
        plot_results(records, out)
    except Exception as e:
        print(f"  Plot failed: {e}")

    # Save JSON
    json_path = out / "cross_offset_stability.json"
    with open(json_path, "w") as f:
        ser = []
        for r in records:
            sr = {k: (v if not isinstance(v, list) else [float(x) for x in v])
                  for k, v in r.items()}
            ser.append(sr)
        json.dump({"summary": summary, "records": ser}, f, indent=2)
    print(f"  JSON saved → {json_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="diagnostic_output")
    parser.add_argument("--backend", default="auto", choices=["auto", "hf", "paged"])
    args = parser.parse_args()
    run_diagnostic(output_dir=args.output_dir, backend=args.backend)
