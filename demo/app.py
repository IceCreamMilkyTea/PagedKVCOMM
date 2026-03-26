"""
KVCOMM Interactive Demo — Multi-Agent Math Solving Comparison

Four execution modes displayed side-by-side:
  1. KVCOMM  (Dense Prefill)      — HuggingFace backend, standard inference
  2. KVCOMM  (KV Reuse)           — HuggingFace backend, cross-context KV reuse
  3. PagedKVCOMM (Dense Prefill)  — nano-vllm paged backend, flash attention
  4. PagedKVCOMM (KV Reuse)       — nano-vllm paged backend, KV reuse + flash attn

Usage:
  /usr/project/xtmp/yw641/envs/nano_vllm_a6000/bin/python3 demo/app.py [--port 7860] [--share]
"""

import argparse
import asyncio
import copy
import json
import math
import os
import queue
import random
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AGENT_COLORS = [
    "#4A90D9",  # blue   — Math Solver
    "#D94A4A",  # red    — Mathematical Analyst
    "#50B86C",  # green  — Programming Expert
    "#E6A23C",  # orange — FinalRefer / Inspector
    "#9B59B6",  # purple
    "#1ABC9C",  # teal
]

ROLE_ICONS = {
    "Math Solver": "\U0001f9ee",           # 🧮
    "Mathematical Analyst": "\U0001f4d0",  # 📐
    "Programming Expert": "\U0001f4bb",    # 💻
    "Inspector": "\U0001f50d",             # 🔍
    "FinalRefer": "\U0001f3af",            # 🎯
}

# The 4 demo modes
MODE_DEFS = [
    {
        "key": "kvcomm_dense",
        "label": "KVCOMM (Dense Prefill)",
        "short": "HF Dense",
        "desc": "HuggingFace DynamicCache \u2014 full dense prefill, no KV reuse",
        "result_dir": "gsm8k_debug5_kvcomm",
        "latency_dir": "gsm8k_debug5_kvcomm",
        "backend": "HuggingFace DynamicCache",
        "color": "#4A90D9",
    },
    {
        "key": "kvcomm_reuse",
        "label": "KVCOMM (KV Reuse)",
        "short": "HF KV-Reuse",
        "desc": "HuggingFace DynamicCache \u2014 cross-context anchor-based KV reuse",
        "result_dir": "gsm8k_debug5_kvcomm",
        "latency_dir": "gsm8k_debug5_kvcomm",
        "backend": "HuggingFace DynamicCache",
        "color": "#E6A23C",
    },
    {
        "key": "paged_dense",
        "label": "PagedKVCOMM (Dense Prefill)",
        "short": "Paged Dense",
        "desc": "Paged Cache + flash attention \u2014 dense prefill",
        "result_dir": "gsm8k_debug5_paged",
        "latency_dir": "gsm8k_debug5_paged",
        "backend": "Paged Cache",
        "color": "#50B86C",
    },
    {
        "key": "paged_reuse",
        "label": "PagedKVCOMM (KV Reuse)",
        "short": "Paged KV-Reuse",
        "desc": "Paged Cache + flash attention \u2014 KV reuse",
        "result_dir": "gsm8k_debug5_paged",
        "latency_dir": "gsm8k_debug5_paged",
        "backend": "Paged Cache",
        "color": "#9B59B6",
    },
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_sample_problems(n: int = 20) -> List[Dict]:
    """Load sample GSM8K problems for the dropdown."""
    for name in ["gsm8k.jsonl", "gsm8k_debug5.jsonl"]:
        path = REPO_ROOT / "datasets" / "gsm8k" / name
        if path.exists():
            break
    else:
        return []
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            obj = json.loads(line.strip())
            answer_str = obj["answer"].split("####")[-1].strip()
            problems.append({
                "question": obj["question"],
                "answer": answer_str,
                "steps": obj["answer"].split("####")[0].strip(),
            })
    return problems


def _find_latest_result_file(result_dir: str) -> Optional[Path]:
    """Find the most recently modified result JSON (excluding Latency.json)."""
    base = REPO_ROOT / "KVCOMM" / "result" / result_dir
    if not base.exists():
        return None
    candidates = [f for f in base.rglob("*.json") if f.name != "Latency.json"]
    return max(candidates, key=lambda f: f.stat().st_mtime) if candidates else None


def load_all_data() -> Dict[str, Any]:
    """
    Load result JSONs and latency JSONs for all 4 modes.
    Returns dict keyed by mode key -> {results: [...], latency: [...]}
    """
    data = {}
    for mode_def in MODE_DEFS:
        key = mode_def["key"]
        entry: Dict[str, Any] = {"results": [], "latency": [], "latency_by_question": {}}

        # Results
        fpath = _find_latest_result_file(mode_def["result_dir"])
        if fpath:
            with open(fpath, "r", encoding="utf-8") as f:
                entry["results"] = json.load(f)

        # Latency
        lat_path = REPO_ROOT / "KVCOMM" / "result" / mode_def["latency_dir"] / "Latency.json"
        if lat_path.exists():
            with open(lat_path, "r", encoding="utf-8") as f:
                all_lat = json.load(f)
            entry["latency"] = all_lat

            # Group latency by (question, request_uid)
            by_req = defaultdict(list)
            for e in all_lat:
                by_req[e.get("request_uid", "")].append(e)

            # Build question -> list of request groups
            by_question = defaultdict(list)
            for uid, agents in by_req.items():
                if agents:
                    q = agents[0].get("message", "").strip()
                    by_question[q].append(agents)
            entry["latency_by_question"] = dict(by_question)

        data[key] = entry
    return data


SAMPLE_PROBLEMS = load_sample_problems(20)
ALL_DATA = load_all_data()


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------
def get_result_for_question(mode_key: str, question: str) -> Optional[Dict]:
    """Find saved result for a question in a specific mode."""
    for item in ALL_DATA.get(mode_key, {}).get("results", []):
        if item["Question"].strip() == question.strip():
            return item
    return None


def get_latency_agents_for_question(mode_key: str, question: str) -> List[Dict]:
    """
    Get per-agent latency entries for a question.
    For 'dense' modes, pick entries with mode=dense_prefill.
    For 'reuse' modes, pick entries with mode=kv_reuse (falling back to dense_prefill).
    """
    groups = ALL_DATA.get(mode_key, {}).get("latency_by_question", {}).get(question.strip(), [])
    is_reuse_mode = "reuse" in mode_key

    if is_reuse_mode:
        # Find a request group that contains kv_reuse entries
        for group in groups:
            reuse_entries = [e for e in group if e.get("mode") == "kv_reuse"]
            if reuse_entries:
                return group  # return full group (mix of dense_prefill + kv_reuse)
        # Fallback: return last group if available
        return groups[-1] if groups else []
    else:
        # Find a request group that is all dense_prefill
        for group in groups:
            if all(e.get("mode") == "dense_prefill" for e in group):
                return group
        # Fallback: first group
        return groups[0] if groups else []


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------
def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")


def render_agent_bubble(agent_id: int, role: str, text: str,
                        token_count: int, ttft: float,
                        infer_mode: str = "dense_prefill",
                        is_streaming: bool = False) -> str:
    """Render one agent's message as a chat bubble."""
    color = AGENT_COLORS[agent_id % len(AGENT_COLORS)]
    icon = ROLE_ICONS.get(role, "\U0001f916")
    mode_badge_color = "#E6A23C" if infer_mode == "kv_reuse" else "#585b70"
    mode_label = "KV Reuse" if infer_mode == "kv_reuse" else "Dense"
    streaming_dot = '<span style="animation:blink 1s infinite;color:#E6A23C;font-weight:bold;">...</span>' if is_streaming else ""

    return f"""
    <div style="display:flex; gap:10px; margin:8px 0; animation:fadeIn .3s ease-in;">
      <div style="flex-shrink:0; width:42px; height:42px; border-radius:50%;
                  background:{color}; display:flex; align-items:center; justify-content:center;
                  font-size:20px; color:white; box-shadow:0 2px 4px rgba(0,0,0,.2);">
        {icon}
      </div>
      <div style="flex:1; background:#1e1e2e; border-radius:12px; padding:12px 16px;
                  border-left:3px solid {color}; box-shadow:0 1px 3px rgba(0,0,0,.3);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; flex-wrap:wrap; gap:4px;">
          <span style="font-weight:bold; color:{color}; font-size:13px;">
            Agent {agent_id} &mdash; {role}
          </span>
          <div style="display:flex; gap:8px; align-items:center;">
            <span style="font-size:10px; padding:2px 6px; border-radius:4px;
                         background:{mode_badge_color}33; color:{mode_badge_color}; font-weight:600;">
              {mode_label}
            </span>
            <span style="font-size:11px; color:#888;">
              {token_count} tok &bull; TTFT {ttft*1000:.1f}ms
            </span>
          </div>
        </div>
        <div style="color:#cdd6f4; font-size:13px; line-height:1.5;
                    font-family:'SF Mono',Monaco,Consolas,monospace; max-height:300px; overflow-y:auto;">
          {_esc(text)}{streaming_dot}
        </div>
      </div>
    </div>
    """


def render_progress_bar(current: int, total: int, label: str = "",
                        color: str = "#4A90D9") -> str:
    pct = min(100, int(current / max(total, 1) * 100))
    return f"""
    <div style="margin:6px 0;">
      <div style="display:flex; justify-content:space-between; font-size:11px; color:#888; margin-bottom:2px;">
        <span>{label}</span>
        <span>{current}/{total} tokens ({pct}%)</span>
      </div>
      <div style="background:#313244; border-radius:6px; height:8px; overflow:hidden;">
        <div style="background:linear-gradient(90deg,{color},{color}dd);
                    width:{pct}%; height:100%; border-radius:6px;
                    transition:width .3s ease;"></div>
      </div>
    </div>
    """


def render_topology_svg(num_agents: int = 3) -> str:
    """SVG of a FullConnected topology."""
    w, h = 180, 120
    cx, cy = w // 2, h // 2
    r = 40
    nodes = []
    for i in range(num_agents):
        angle = 2 * math.pi * i / num_agents - math.pi / 2
        nodes.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))

    svg = f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:
                x1, y1 = nodes[i]
                x2, y2 = nodes[j]
                svg += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#585b70" stroke-width="1.5" opacity=".5"/>'
    for i, (nx, ny) in enumerate(nodes):
        c = AGENT_COLORS[i % len(AGENT_COLORS)]
        svg += f'<circle cx="{nx}" cy="{ny}" r="14" fill="{c}" stroke="#1e1e2e" stroke-width="2"/>'
        svg += f'<text x="{nx}" y="{ny+4}" text-anchor="middle" fill="white" font-size="10" font-weight="bold">{i}</text>'
    svg += "</svg>"
    return svg


def render_panel_header(mode_def: Dict, status: str = "Ready",
                        num_agents: int = 3) -> str:
    status_colors = {"Ready": "#888", "Running": "#E6A23C", "Done": "#50B86C",
                     "No Data": "#585b70", "Simulated": "#585b70"}
    sc = status_colors.get(status, "#888")
    topo_svg = render_topology_svg(num_agents)
    return f"""
    <div style="background:#181825; border-radius:12px; padding:14px; margin-bottom:8px;
                border:1px solid #313244;">
      <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div style="flex:1;">
          <h3 style="margin:0; color:{mode_def['color']}; font-size:15px;">
            {mode_def['label']}
          </h3>
          <p style="margin:4px 0 0; color:#888; font-size:11px;">{mode_def['desc']}</p>
          <span style="display:inline-block; margin-top:4px; font-size:10px; padding:2px 8px;
                       border-radius:4px; background:#313244; color:#a6adc8;">
            {mode_def['backend']}
          </span>
        </div>
        <div style="display:flex; flex-direction:column; align-items:flex-end; gap:4px;">
          <span style="background:{sc}22; color:{sc}; padding:3px 10px;
                       border-radius:10px; font-size:11px; font-weight:600;">
            {status}
          </span>
          {topo_svg}
        </div>
      </div>
    </div>
    """


def render_result_badge(solved: Optional[bool], predicted: str, true_answer: str) -> str:
    if solved is None:
        return ""
    color = "#50B86C" if solved else "#D94A4A"
    text = "Correct!" if solved else "Incorrect"
    return f"""
    <div style="margin-top:10px; padding:10px; background:{color}15;
                border:1px solid {color}; border-radius:8px;
                display:flex; justify-content:space-between; align-items:center;">
      <span style="color:{color}; font-weight:bold; font-size:14px;">{text}</span>
      <span style="color:#cdd6f4; font-size:13px;">
        Predicted: {predicted} &bull; Ground Truth: {true_answer}
      </span>
    </div>
    """


# ---------------------------------------------------------------------------
# Build panel data for one mode + one question
# ---------------------------------------------------------------------------
def build_panel_data(mode_def: Dict, question: str) -> Dict:
    """Assemble per-agent messages for one mode and one question."""
    mode_key = mode_def["key"]
    result = get_result_for_question(mode_key, question)
    lat_agents = get_latency_agents_for_question(mode_key, question)

    # Build agent messages from latency data (which has per-agent detail)
    # Note: agent_id in latency data uses a global counter that doesn't reset
    # across Graph instances, so we normalize to 0-based index per panel.
    messages = []
    if lat_agents:
        for idx, entry in enumerate(lat_agents):
            messages.append({
                "agent_id": idx,
                "role": entry.get("agent_role", entry.get("agent_name", "Agent")),
                "ttft": entry.get("ttft", 0.0),
                "mode": entry.get("mode", "dense_prefill"),
                "text": "",  # we'll fill from result if available
                "token_count": 0,
            })

    # Fill response text from result
    response_text = ""
    if result and result.get("Response"):
        response_text = result["Response"][0] if isinstance(result["Response"], list) else str(result["Response"])

    # Assign text to agents
    if messages and response_text:
        # First agent gets the full response; others get a summary
        messages[0]["text"] = response_text
        messages[0]["token_count"] = len(response_text.split())
        for m in messages[1:]:
            summary = f"[Collaborated with Agent 0; contributed {m['role']} perspective to the reasoning above]"
            m["text"] = summary
            m["token_count"] = len(summary.split())
    elif not messages and response_text:
        # No latency data, but have result — create synthetic agents
        for i, role in enumerate(["Math Solver", "Mathematical Analyst", "Programming Expert"]):
            text = response_text if i == 0 else f"[Agent {i} ({role}): contributed to collaborative reasoning]"
            messages.append({
                "agent_id": i,
                "role": role,
                "ttft": 0.0,
                "mode": "dense_prefill",
                "text": text,
                "token_count": len(text.split()),
            })

    solved = result.get("Solved") if result else None
    predicted = result.get("Attempt answer", "?") if result else "?"
    true_answer = result.get("Answer", "?") if result else "?"

    return {
        "mode_def": mode_def,
        "messages": messages,
        "solved": solved,
        "predicted": predicted,
        "true_answer": true_answer,
        "response_text": response_text,
        "has_data": bool(messages) or bool(result),
    }


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------
def build_metrics_table(panels: List[Dict]) -> str:
    html = """
    <div style="background:#181825; border-radius:12px; padding:16px; margin-top:12px;
                border:1px solid #313244;">
      <h3 style="color:#cdd6f4; margin:0 0 12px; font-size:15px;">
        Performance Comparison
      </h3>
      <table style="width:100%; border-collapse:collapse; font-size:13px;">
        <thead>
          <tr style="border-bottom:2px solid #313244;">
            <th style="text-align:left; padding:8px 12px; color:#888;">Mode</th>
            <th style="text-align:center; padding:8px; color:#888;">Backend</th>
            <th style="text-align:center; padding:8px; color:#888;">Agents</th>
            <th style="text-align:center; padding:8px; color:#888;">Avg TTFT</th>
            <th style="text-align:center; padding:8px; color:#888;">KV Reuse</th>
            <th style="text-align:center; padding:8px; color:#888;">Tokens</th>
            <th style="text-align:center; padding:8px; color:#888;">Result</th>
          </tr>
        </thead>
        <tbody>
    """
    for p in panels:
        md = p["mode_def"]
        msgs = p["messages"]
        n_agents = len(msgs)
        avg_ttft = sum(m["ttft"] for m in msgs) / max(n_agents, 1) if msgs else 0
        total_tok = sum(m["token_count"] for m in msgs)
        n_reuse = sum(1 for m in msgs if m.get("mode") == "kv_reuse")
        reuse_str = f"{n_reuse}/{n_agents}" if n_reuse else "\u2014"

        if p["solved"] is True:
            result_icon = '<span style="color:#50B86C;">\u2714 Correct</span>'
        elif p["solved"] is False:
            result_icon = '<span style="color:#D94A4A;">\u2718 Wrong</span>'
        else:
            result_icon = '<span style="color:#585b70;">\u2014</span>'

        ttft_color = "#50B86C" if avg_ttft < 0.05 else "#E6A23C" if avg_ttft < 0.1 else "#D94A4A"

        html += f"""
          <tr style="border-bottom:1px solid #313244;">
            <td style="padding:8px 12px; color:{md['color']}; font-weight:600;">{md['short']}</td>
            <td style="text-align:center; padding:8px; color:#a6adc8; font-size:11px;">{md['backend']}</td>
            <td style="text-align:center; padding:8px; color:#cdd6f4;">{n_agents}</td>
            <td style="text-align:center; padding:8px; color:{ttft_color}; font-weight:600;">
              {avg_ttft*1000:.1f}ms
            </td>
            <td style="text-align:center; padding:8px; color:#cdd6f4;">{reuse_str}</td>
            <td style="text-align:center; padding:8px; color:#cdd6f4;">{total_tok}</td>
            <td style="text-align:center; padding:8px;">{result_icon}</td>
          </tr>
        """
    html += "</tbody></table></div>"
    return html


# ---------------------------------------------------------------------------
# Main output builders (instant + streaming)
# ---------------------------------------------------------------------------
def build_all_panels(question: str):
    """Build all 4 panels instantly. Returns 9-tuple (4 * (header, chat) + metrics)."""
    panels = [build_panel_data(md, question) for md in MODE_DEFS]
    outputs = []
    for p in panels:
        md = p["mode_def"]
        status = "Done" if p["has_data"] else "No Data"
        header = render_panel_header(md, status)

        chat = ""
        total_tok = 0
        for m in p["messages"]:
            chat += render_agent_bubble(
                m["agent_id"], m["role"], m["text"],
                m["token_count"], m["ttft"], m["mode"],
            )
            total_tok += m["token_count"]

        if p["messages"]:
            chat += render_progress_bar(total_tok, total_tok, "Total tokens", md["color"])

        chat += render_result_badge(p["solved"], p["predicted"], p["true_answer"])
        outputs.extend([header, chat])

    outputs.append(build_metrics_table(panels))
    return tuple(outputs)


def stream_all_panels(question: str):
    """
    Generator yielding incremental updates to simulate streaming.
    Each yield is the same 9-tuple as build_all_panels.
    Agents appear one by one across all 4 panels simultaneously.
    """
    panels = [build_panel_data(md, question) for md in MODE_DEFS]

    # Determine max number of agents across all panels
    max_agents = max((len(p["messages"]) for p in panels), default=0)
    if max_agents == 0:
        yield build_all_panels(question)
        return

    # For streaming the first agent's text word-by-word
    first_texts = []
    for p in panels:
        if p["messages"]:
            first_texts.append(p["messages"][0]["text"].split())
        else:
            first_texts.append([])
    max_words = max((len(w) for w in first_texts), default=0)
    chunk_size = max(1, max_words // 25)

    # Phase 1: Stream first agent's text word-by-word
    for step in range(0, max_words + chunk_size, chunk_size):
        outputs = []
        for pi, p in enumerate(panels):
            md = p["mode_def"]
            words = first_texts[pi]
            n_revealed = min(step, len(words))
            is_streaming = step < len(words)

            header = render_panel_header(md, "Running" if is_streaming else "Done")
            chat = ""
            if p["messages"]:
                m = p["messages"][0]
                revealed_text = " ".join(words[:n_revealed])
                chat += render_agent_bubble(
                    m["agent_id"], m["role"], revealed_text,
                    n_revealed, m["ttft"], m["mode"],
                    is_streaming=is_streaming,
                )
                chat += render_progress_bar(
                    n_revealed, len(words),
                    "Agent 0 generating...",
                    md["color"],
                )
            else:
                chat = '<p style="color:#585b70; text-align:center; padding:20px;">No data for this mode</p>'
            outputs.extend([header, chat])

        outputs.append('<div style="color:#888; text-align:center; padding:12px;">Generating...</div>')
        yield tuple(outputs)
        if step < max_words:
            time.sleep(0.06)

    # Phase 2: Reveal remaining agents one by one
    for agent_idx in range(1, max_agents):
        outputs = []
        for pi, p in enumerate(panels):
            md = p["mode_def"]
            header = render_panel_header(md, "Done")
            chat = ""
            # Show all agents up to agent_idx
            total_tok = 0
            for ai in range(min(agent_idx + 1, len(p["messages"]))):
                m = p["messages"][ai]
                chat += render_agent_bubble(
                    m["agent_id"], m["role"], m["text"],
                    m["token_count"], m["ttft"], m["mode"],
                )
                total_tok += m["token_count"]
            if p["messages"]:
                full_tok = sum(m["token_count"] for m in p["messages"])
                chat += render_progress_bar(total_tok, full_tok, "Agents completed", md["color"])
            outputs.extend([header, chat])
        outputs.append('<div style="color:#888; text-align:center; padding:12px;">Agents responding...</div>')
        yield tuple(outputs)
        time.sleep(0.4)

    # Phase 3: Final state with metrics
    yield build_all_panels(question)


# ---------------------------------------------------------------------------
# Live Inference Engine (subprocess-based, parallel backends)
# ---------------------------------------------------------------------------
# Two backends run in PARALLEL, each as a separate subprocess.
# Within each subprocess, dense + kv_reuse run sequentially (model loaded once).
#
#   ┌─ Subprocess 1 (HF,    KVCOMM_PAGED=0) ── kvcomm_dense ──→ kvcomm_reuse ─┐
#   │                                                                           │ parallel
#   └─ Subprocess 2 (Paged, KVCOMM_PAGED=1) ── paged_dense  ──→ paged_reuse ──┘
#
# If GPU memory is insufficient for two models, use --sequential mode.

import subprocess

# The 4 live inference tasks grouped by backend
LIVE_TASKS = [
    {"mode_key": "kvcomm_dense",  "backend": "hf",    "execution_mode": "default"},
    {"mode_key": "kvcomm_reuse",  "backend": "hf",    "execution_mode": "allow_kv_reuse"},
    {"mode_key": "paged_dense",   "backend": "paged", "execution_mode": "default"},
    {"mode_key": "paged_reuse",   "backend": "paged", "execution_mode": "allow_kv_reuse"},
]

WORKER_SCRIPT = str(Path(__file__).parent / "live_worker.py")
PYTHON_BIN = os.environ.get(
    "KVCOMM_PYTHON",
    "/usr/project/xtmp/yw641/envs/nano_vllm_a6000/bin/python3",
)
DEFAULT_MODEL = os.environ.get(
    "KVCOMM_MODEL",
    "/usr/project/xtmp/yw641/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct"
    "/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
)


class LiveEngine:
    """
    Runs all 4 modes via separate subprocesses, each pinned to its own GPU.

    GPU assignment (--num-gpus N):
      N >= 4: each mode gets its own GPU (full parallel)
      N == 2: each backend shares a GPU, dense→reuse sequential within
      N == 1: everything sequential on one GPU

    Paged (nano-vllm) subprocesses also get unique NCCL ports to avoid
    the "EADDRINUSE :2333" conflict.
    """

    # GPU assignment per task index (mode_key order matches LIVE_TASKS)
    # With 4 GPUs: task 0→GPU0, task 1→GPU1, task 2→GPU2, task 3→GPU3
    # With 2 GPUs: HF tasks→GPU0, Paged tasks→GPU1
    # With 1 GPU:  all→GPU0 (sequential)

    # Base NCCL port — each paged subprocess gets base + offset
    NCCL_PORT_BASE = 2333

    def __init__(self, num_gpus: int = 4):
        self._output_queue: queue.Queue = queue.Queue()
        self._running = False
        self._num_gpus = num_gpus

    def _build_cmd(self, question: str, tasks: List[Dict], backend: str,
                   gpu: Optional[int] = None, nccl_port: Optional[int] = None
                   ) -> List[str]:
        """Build the worker command line."""
        tasks_json = json.dumps(tasks)
        cmd = [
            PYTHON_BIN, WORKER_SCRIPT,
            "--question", question,
            "--tasks", tasks_json,
            "--backend", backend,
            "--model_path", DEFAULT_MODEL,
        ]
        if gpu is not None:
            cmd.extend(["--gpu", str(gpu)])
        if nccl_port is not None:
            cmd.extend(["--nccl-port", str(nccl_port)])
        return cmd

    def _run_subprocess(self, question: str, tasks: List[Dict],
                        backend: str, gpu: Optional[int] = None,
                        nccl_port: Optional[int] = None):
        """Run one subprocess. Blocks until done. Thread-safe via queue."""
        cmd = self._build_cmd(question, tasks, backend, gpu, nccl_port)

        env = os.environ.copy()
        env["KVCOMM_PAGED"] = "1" if backend == "paged" else "0"
        env["HF_HOME"] = os.environ.get(
            "HF_HOME", "/usr/project/xtmp/yw641/hf_cache"
        )
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                self._output_queue.put(event)
            except json.JSONDecodeError:
                pass

        proc.wait()
        stderr_output = proc.stderr.read()

        if proc.returncode != 0:
            for task in tasks:
                self._output_queue.put({
                    "type": "error",
                    "mode_key": task["mode_key"],
                    "message": (
                        f"Worker ({backend}, GPU {gpu}) code {proc.returncode}"
                        f": {stderr_output[-500:] if stderr_output else ''}"
                    ),
                })

    def _run_all(self, question: str):
        """Launch subprocesses with GPU assignment based on num_gpus."""
        try:
            if self._num_gpus >= 4:
                # 4 GPUs: each mode gets its own GPU, all parallel
                threads = []
                for i, task in enumerate(LIVE_TASKS):
                    nccl_port = self.NCCL_PORT_BASE + i if task["backend"] == "paged" else None
                    t = threading.Thread(
                        target=self._run_subprocess,
                        args=(question, [task], task["backend"], i, nccl_port),
                        daemon=True,
                    )
                    threads.append(t)
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

            elif self._num_gpus >= 2:
                # 2 GPUs: HF→GPU0, Paged→GPU1, dense→reuse sequential within
                hf_tasks = [t for t in LIVE_TASKS if t["backend"] == "hf"]
                paged_tasks = [t for t in LIVE_TASKS if t["backend"] == "paged"]
                threads = []
                if hf_tasks:
                    t = threading.Thread(
                        target=self._run_subprocess,
                        args=(question, hf_tasks, "hf", 0, None),
                        daemon=True,
                    )
                    threads.append(t)
                if paged_tasks:
                    t = threading.Thread(
                        target=self._run_subprocess,
                        args=(question, paged_tasks, "paged", 1, self.NCCL_PORT_BASE),
                        daemon=True,
                    )
                    threads.append(t)
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

            else:
                # 1 GPU: fully sequential
                for i, task in enumerate(LIVE_TASKS):
                    nccl_port = self.NCCL_PORT_BASE + i if task["backend"] == "paged" else None
                    self._run_subprocess(
                        question, [task], task["backend"], 0, nccl_port
                    )

        finally:
            self._output_queue.put({"type": "all_done"})
            self._running = False

    def run_inference(self, question: str):
        """Start live inference in a background thread."""
        self._running = True
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except queue.Empty:
                break

        thread = threading.Thread(
            target=self._run_all, args=(question,), daemon=True
        )
        thread.start()


# Global instance
live_engine = LiveEngine()


def live_stream_panels(question: str):
    """
    Generator that runs live inference for ALL 4 modes and yields UI updates.
    Each mode runs as a subprocess — no singleton conflicts.

    Yields the same 9-tuple as build_all_panels:
      4 x (header_html, chat_html) + metrics_html
    """
    if not question.strip():
        empty = "<p style='color:#585b70;'>Enter a question first</p>"
        yield tuple([empty] * 9)
        return

    all_mode_keys = [t["mode_key"] for t in LIVE_TASKS]

    # Live state for all 4 modes
    live_state: Dict[str, Dict] = {}
    for mode_key in all_mode_keys:
        live_state[mode_key] = {
            "status": "Waiting",
            "messages": [],
            "answers": [],
        }

    def _build_output():
        """Build the 9-tuple from current state."""
        outputs = []
        all_panel_data = []
        for md in MODE_DEFS:
            key = md["key"]
            state = live_state[key]
            status = state["status"]
            header = render_panel_header(md, status)

            chat = ""
            total_tok = 0
            for m in state["messages"]:
                chat += render_agent_bubble(
                    m["agent_id"], m["role"], m["text"],
                    m["token_count"], m["ttft"], m["mode"],
                    is_streaming=(status == "Running"),
                )
                total_tok += m["token_count"]

            if state["messages"]:
                expected_tok = max(total_tok, 1)
                label = "Generating..." if status == "Running" else "Total tokens"
                chat += render_progress_bar(total_tok, expected_tok, label, md["color"])

            if status == "Done" and state["answers"]:
                answer_text = state["answers"][0] if state["answers"] else "?"
                chat += f"""
                <div style="margin-top:10px; padding:10px; background:#4A90D915;
                            border:1px solid #4A90D9; border-radius:8px;">
                  <span style="color:#4A90D9; font-weight:bold;">Live Result</span>
                  <span style="color:#cdd6f4; font-size:13px; margin-left:8px;">
                    {_esc(str(answer_text)[:500])}
                  </span>
                </div>
                """

            if status == "Waiting":
                chat = f'<p style="color:#E6A23C; text-align:center; padding:20px;">Waiting for {md["label"]}...</p>'
            elif status == "Error":
                chat = f'<p style="color:#D94A4A; text-align:center; padding:20px;">Error: {_esc(state.get("error", "Unknown"))}</p>'

            outputs.extend([header, chat])
            all_panel_data.append({
                "mode_def": md,
                "messages": state["messages"],
                "solved": None,
                "predicted": "?",
                "true_answer": "?",
            })

        outputs.append(build_metrics_table(all_panel_data))
        return tuple(outputs)

    # Start all 4 modes
    live_engine.run_inference(question)
    yield _build_output()

    # Poll queue for updates
    while True:
        try:
            event = live_engine._output_queue.get(timeout=0.5)
        except queue.Empty:
            if not live_engine._running:
                break
            yield _build_output()  # keep UI alive
            continue

        etype = event["type"]

        if etype == "mode_start":
            mode_key = event["mode_key"]
            live_state[mode_key]["status"] = "Running"
            live_state[mode_key]["messages"] = []
            yield _build_output()

        elif etype == "agent_output":
            # Route by mode_key (each event carries its own mode_key)
            mode_key = event.get("mode_key")
            if not mode_key or mode_key not in live_state:
                continue

            # Normalize agent_id to 0-based index within this panel
            aid_int = len(live_state[mode_key]["messages"])

            live_state[mode_key]["messages"].append({
                "agent_id": aid_int,
                "role": event["agent_role"],
                "text": event["text"],
                "token_count": len(event["text"].split()),
                "ttft": event["ttft"],
                "mode": event["mode"],
            })
            yield _build_output()

        elif etype == "mode_done":
            mode_key = event["mode_key"]
            live_state[mode_key]["status"] = "Done"
            live_state[mode_key]["answers"] = event.get("answers", [])
            yield _build_output()

        elif etype == "error":
            mode_key = event.get("mode_key")
            if mode_key:
                live_state[mode_key]["status"] = "Error"
                live_state[mode_key]["error"] = event["message"]
            else:
                for mk in all_mode_keys:
                    if live_state[mk]["status"] == "Running":
                        live_state[mk]["status"] = "Error"
                        live_state[mk]["error"] = event["message"]
            yield _build_output()

        elif etype == "all_done":
            break

    yield _build_output()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
.gradio-container {
    background: #11111b !important;
    max-width: 1800px !important;
}
.panel-col {
    background: #1e1e2e !important;
    border-radius: 12px !important;
    padding: 10px !important;
    border: 1px solid #313244 !important;
    min-width: 320px !important;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: .3; }
}
footer { display: none !important; }
"""


def create_demo():
    with gr.Blocks(title="KVCOMM Interactive Demo") as demo:
        gr.HTML("""
        <div style="text-align:center; padding:20px 0 8px;">
          <h1 style="color:#cdd6f4; font-size:26px; margin:0;">
            KVCOMM: Multi-Agent Collaborative Math Solving
          </h1>
          <p style="color:#888; font-size:13px; margin:6px 0 0;">
            Compare 4 execution modes: KVCOMM vs PagedKVCOMM &times; Dense Prefill vs KV Reuse
          </p>
        </div>
        """)

        # Input row
        with gr.Row():
            with gr.Column(scale=1):
                sample_choices = [p["question"][:90] + "..." for p in SAMPLE_PROBLEMS]
                sample_dropdown = gr.Dropdown(
                    choices=sample_choices,
                    label="Select a GSM8K problem",
                    interactive=True,
                )
                question_input = gr.Textbox(
                    label="Or type your own math problem",
                    placeholder="Janet's ducks lay 16 eggs per day...",
                    lines=3,
                )
                with gr.Row():
                    run_btn = gr.Button("Run Comparison", variant="primary", size="lg")
                    stream_btn = gr.Button("Stream (Animated)", variant="secondary", size="lg")
                    live_btn = gr.Button("\u26a1 Live Inference (4 Modes Parallel)", variant="primary", size="lg")
                ground_truth = gr.HTML(
                    '<div style="color:#585b70; font-size:12px; padding:8px;">Select a problem to see ground truth</div>'
                )

        # 4 output panels in 2x2 grid
        # Row 1: KVCOMM Dense | KVCOMM KV Reuse
        with gr.Row():
            with gr.Column(scale=1, elem_classes="panel-col"):
                p1_header = gr.HTML()
                p1_chat = gr.HTML()
            with gr.Column(scale=1, elem_classes="panel-col"):
                p2_header = gr.HTML()
                p2_chat = gr.HTML()

        # Row 2: PagedKVCOMM Dense | PagedKVCOMM KV Reuse
        with gr.Row():
            with gr.Column(scale=1, elem_classes="panel-col"):
                p3_header = gr.HTML()
                p3_chat = gr.HTML()
            with gr.Column(scale=1, elem_classes="panel-col"):
                p4_header = gr.HTML()
                p4_chat = gr.HTML()

        # Metrics
        metrics_output = gr.HTML()

        all_outputs = [p1_header, p1_chat, p2_header, p2_chat,
                       p3_header, p3_chat, p4_header, p4_chat,
                       metrics_output]

        # --- Event handlers ---
        def on_select_sample(choice):
            if choice is None:
                return "", '<div style="color:#585b70; font-size:12px; padding:8px;">Select a problem</div>'
            idx = sample_choices.index(choice) if choice in sample_choices else 0
            p = SAMPLE_PROBLEMS[idx]
            gt = f"""
            <div style="background:#181825; border-radius:8px; padding:10px; margin-top:6px;
                        border:1px solid #313244;">
              <div style="color:#50B86C; font-weight:bold; font-size:13px; margin-bottom:4px;">
                Ground Truth: {p["answer"]}
              </div>
              <div style="color:#888; font-size:11px; white-space:pre-wrap;">{p["steps"]}</div>
            </div>
            """
            return p["question"], gt

        sample_dropdown.change(
            fn=on_select_sample,
            inputs=[sample_dropdown],
            outputs=[question_input, ground_truth],
        )

        def on_run(question):
            if not question.strip():
                empty = "<p style='color:#585b70;'>Enter a question first</p>"
                return tuple([empty] * 9)
            return build_all_panels(question)

        run_btn.click(fn=on_run, inputs=[question_input], outputs=all_outputs)

        def on_stream(question):
            if not question.strip():
                empty = "<p style='color:#585b70;'>Enter a question first</p>"
                yield tuple([empty] * 9)
                return
            yield from stream_all_panels(question)

        stream_btn.click(fn=on_stream, inputs=[question_input], outputs=all_outputs)

        def on_live(question):
            if not question.strip():
                empty = "<p style='color:#585b70;'>Enter a question first</p>"
                yield tuple([empty] * 9)
                return
            yield from live_stream_panels(question)

        live_btn.click(fn=on_live, inputs=[question_input], outputs=all_outputs)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    global live_engine
    parser = argparse.ArgumentParser(description="KVCOMM Interactive Demo")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument(
        "--num-gpus", type=int, default=4,
        help="Number of GPUs for live inference: "
             "4 = each mode on its own GPU (full parallel); "
             "2 = each backend on its own GPU; "
             "1 = sequential on one GPU",
    )
    args = parser.parse_args()
    live_engine = LiveEngine(num_gpus=args.num_gpus)

    for md in MODE_DEFS:
        data = ALL_DATA[md["key"]]
        n_results = len(data["results"])
        n_latency = len(data["latency"])
        print(f"  {md['label']}: {n_results} results, {n_latency} latency entries")
    print(f"  Sample problems: {len(SAMPLE_PROBLEMS)}")

    demo = create_demo()
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="gray",
            font=gr.themes.GoogleFont("Inter"),
        ),
    )


if __name__ == "__main__":
    main()
