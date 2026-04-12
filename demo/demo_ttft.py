"""
KVCOMM TTFT Race Demo — Video-style batch benchmark comparison

Six methods race sequentially (single GPU):
  1. Dense Prefill           — baseline, no KV reuse
  2. KV Reuse                — cross-context anchor-based KV reuse
  3. Paged Attention         — paged cache + flash attention
  4. PagedKVCOMM KV-Reuse    — paged cache + KV reuse
  5. Radix only              — radix prefix cache
  6. Radix KVCOMM           — revised(1) paged radix + KV reuse

Each method runs N synthetic requests.  A live progress bar, current
prompt text, throughput counter, and per-method TTFT are displayed.

Usage:
  python demo/demo_ttft.py --model /path/to/model [--samples 50] \
      [--agents 5] [--port 7861] [--share]
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CURRENT_REPO_ROOT = REPO_ROOT
REVISED1_REPO_ROOT = Path("/home/users/hz314/workspace/PagedKVCOMM-revised(1)/PagedKVCOMM-revised")

STEP_BADGES = ["\u2460", "\u2461", "\u2462", "\u2463", "\u2464", "\u2465"]
REPLAY_TOTAL_REQUESTS = 10
REPLAY_TARGET_SECONDS = 15.0

METHOD_DEFS = [
    {
        "key": "dense",
        "label": "Dense Prefill",
        "short": "Dense (HF)",
        "backend": "hf",
        "repo_root": str(CURRENT_REPO_ROOT),
        "color": "#4A90D9",
        "bar_gradient": "linear-gradient(90deg, #4A90D9, #357ABD)",
    },
    {
        "key": "kvcomm",
        "label": "KV Reuse",
        "short": "KV-Comm (HF)",
        "backend": "hf",
        "repo_root": str(CURRENT_REPO_ROOT),
        "color": "#E6A23C",
        "bar_gradient": "linear-gradient(90deg, #E6A23C, #D4911A)",
    },
    {
        "key": "paged_dense",
        "label": "Paged Attention",
        "short": "Dense (Paged)",
        "backend": "paged",
        "repo_root": str(CURRENT_REPO_ROOT),
        "color": "#50B86C",
        "bar_gradient": "linear-gradient(90deg, #50B86C, #3DA55A)",
    },
    {
        "key": "paged_kvcomm",
        "label": "Paged KV Reuse",
        "short": "KV-Comm (Paged)",
        "backend": "paged",
        "repo_root": str(CURRENT_REPO_ROOT),
        "color": "#9B59B6",
        "bar_gradient": "linear-gradient(90deg, #9B59B6, #8E44AD)",
    },
    {
        "key": "radix_dense",
        "label": "Radix only",
        "short": "Radix only",
        "backend": "radix",
        "repo_root": str(CURRENT_REPO_ROOT),
        "color": "#E57373",
        "bar_gradient": "linear-gradient(90deg, #E57373, #D65C5C)",
    },
    {
        "key": "paged_radix_kvcomm",
        "label": "Radix KVCOMM",
        "short": "Radix KVCOMM",
        "backend": "radix",
        "repo_root": str(REVISED1_REPO_ROOT),
        "color": "#EF5350",
        "bar_gradient": "linear-gradient(90deg, #EF5350, #D84343)",
    },
]

REPLAY_BENCHMARK = {
    "dense": {"ttft_ms": 1643.0, "throughput_req_min": 1.8},
    "kvcomm": {"ttft_ms": 433.0, "throughput_req_min": 4.8},
    "paged_dense": {"ttft_ms": 184.0, "throughput_req_min": 2.4},
    "paged_kvcomm": {"ttft_ms": 116.0, "throughput_req_min": 22.2},
    "radix_dense": {"ttft_ms": 239.0, "throughput_req_min": 2.4},
    "paged_radix_kvcomm": {"ttft_ms": 136.0, "throughput_req_min": 16.8},
}

WORKER_SCRIPT = str(Path(__file__).parent / "ttft_worker.py")

# Filled from CLI args
_CONFIG = {
    "python": sys.executable,
    "model": "",
    "samples": 50,
    "agents": 5,
    "in_length": 512,
    "out_length": 512,
}

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))


def _throughput_value(value_req_per_sec: float, unit: str) -> float:
    return value_req_per_sec * 60.0 if unit == "req/min" else value_req_per_sec


def _throughput_label(unit: str) -> str:
    return "req/min" if unit == "req/min" else "req/s"


def render_method_card(idx: int, mdef: Dict, state: Dict, total: int, throughput_unit: str) -> str:
    """Render one method's race column."""
    step = STEP_BADGES[idx]
    color = mdef["color"]
    grad = mdef["bar_gradient"]
    status = state["status"]

    # Progress
    done_count = state["done_count"]
    pct = min(100, done_count / max(total, 1) * 100)
    throughput = _throughput_value(state.get("throughput", 0), throughput_unit)
    throughput_label = _throughput_label(throughput_unit)
    avg_ttft = state.get("avg_ttft", 0)
    prompt = state.get("prompt", "")
    segment_count = int(state.get("segment_count", 0) or 0)

    # Status badge
    status_styles = {
        "waiting":  ("Waiting",   "#585b70", "#313244"),
        "running":  ("Running",   color,     f"{color}33"),
        "done":     ("Done",      "#50B86C", "#50B86C22"),
        "error":    ("Error",     "#D94A4A", "#D94A4A22"),
    }
    s_label, s_fg, s_bg = status_styles.get(status, status_styles["waiting"])

    # Card border
    border = f"2px solid {color}" if status == "running" else "1px solid #313244"
    glow = f"box-shadow: 0 0 18px {color}44;" if status == "running" else ""

    html = f"""
    <div style="background:#181825; border-radius:14px; padding:18px;
                border:{border}; {glow} transition:all .3s ease;
                display:flex; flex-direction:column; height:100%; min-height:360px;">
    """

    # -- Header --
    html += f"""
      <div style="display:flex; align-items:center; justify-content:space-between;
                  margin-bottom:14px;">
        <div style="display:flex; align-items:center; gap:8px;">
          <span style="font-size:22px; color:{color};">{step}</span>
          <div>
            <div style="font-weight:700; color:{color}; font-size:14px;">
              {mdef['label']}
            </div>
          </div>
        </div>
        <span style="background:{s_bg}; color:{s_fg}; padding:3px 10px;
                     border-radius:10px; font-size:11px; font-weight:600;">
          {s_label}
        </span>
      </div>
    """

    # -- Progress bar --
    if segment_count > 0:
        filled_segments = min(segment_count, int(done_count))
        segments = []
        for seg_idx in range(segment_count):
            seg_style = grad if seg_idx < filled_segments else "#313244"
            segments.append(
                f'<div style="flex:1; height:100%; background:{seg_style}; '
                'border-radius:4px; transition:all .2s ease;"></div>'
            )
        html += f"""
      <div style="margin-bottom:12px;">
        <div style="display:flex; justify-content:space-between;
                    font-size:11px; color:#888; margin-bottom:4px;">
          <span>Request: {done_count}/{total}</span>
          <span>{pct:.0f}%</span>
        </div>
        <div style="display:flex; gap:4px; height:28px;">
          {''.join(segments)}
        </div>
      </div>
        """
    else:
        html += f"""
      <div style="margin-bottom:12px;">
        <div style="display:flex; justify-content:space-between;
                    font-size:11px; color:#888; margin-bottom:4px;">
          <span>Request: {done_count}/{total}</span>
          <span>{pct:.0f}%</span>
        </div>
        <div style="background:#313244; border-radius:6px; height:28px;
                    overflow:hidden; position:relative;">
          <div style="background:{grad}; width:{pct:.1f}%;
                      height:100%; border-radius:6px;
                      transition:width .3s ease;"></div>
        </div>
      </div>
        """

    # -- Metrics row --
    if status in ("running", "done"):
        ttft_ms = avg_ttft * 1000 if avg_ttft else 0
        ttft_color = "#50B86C" if ttft_ms < 100 else "#E6A23C" if ttft_ms < 500 else "#D94A4A"
        html += f"""
      <div style="display:flex; gap:12px; margin-bottom:14px;">
        <div style="flex:1; background:#1e1e2e; border-radius:8px; padding:10px;
                    text-align:center;">
          <div style="color:#888; font-size:9px; text-transform:uppercase;
                      letter-spacing:1px; margin-bottom:4px;">Avg TTFT</div>
          <div style="color:{ttft_color}; font-size:22px; font-weight:800;
                      font-family:'SF Mono',Monaco,Consolas,monospace;">
            {ttft_ms:.0f}<span style="font-size:12px; color:#888;">ms</span>
          </div>
        </div>
        <div style="flex:1; background:#1e1e2e; border-radius:8px; padding:10px;
                    text-align:center;">
          <div style="color:#888; font-size:9px; text-transform:uppercase;
                      letter-spacing:1px; margin-bottom:4px;">Throughput</div>
          <div style="color:#cdd6f4; font-size:22px; font-weight:800;
                      font-family:'SF Mono',Monaco,Consolas,monospace;">
            {throughput:.1f}<span style="font-size:12px; color:#888;">{throughput_label}</span>
          </div>
        </div>
      </div>
        """
    else:
        html += '<div style="height:76px;"></div>'

    # -- Prompt display --
    html += '<div style="flex:1; min-height:100px;">'
    if status == "running" and prompt:
        html += f"""
        <div style="background:#1e1e2e; border-radius:8px; padding:12px;
                    border:1px solid #313244;">
          <div style="color:#888; font-size:10px; text-transform:uppercase;
                      letter-spacing:1px; margin-bottom:6px;">
            Request #{done_count}
          </div>
          <div style="color:#a6adc8; font-size:11px; line-height:1.5;
                      font-family:'SF Mono',Monaco,Consolas,monospace;
                      max-height:120px; overflow-y:auto;
                      word-break:break-all;">
            Prompt: {_esc(prompt[:300])}
          </div>
        </div>
        """
    elif status == "done":
        html += f"""
        <div style="background:#50B86C11; border-radius:8px; padding:16px;
                    border:1px solid #50B86C33; text-align:center;">
          <div style="font-size:24px; margin-bottom:4px;">\u2705</div>
          <div style="color:#50B86C; font-size:13px; font-weight:600;">
            Completed {total} requests
          </div>
          <div style="color:#888; font-size:11px; margin-top:4px;">
            {state.get('elapsed', 0):.1f}s total
          </div>
        </div>
        """
    elif status == "error":
        err = state.get("error", "Unknown")
        html += f"""
        <div style="background:#D94A4A11; border-radius:8px; padding:12px;
                    border:1px solid #D94A4A33;">
          <div style="color:#D94A4A; font-size:12px; font-weight:600;
                      margin-bottom:4px;">\u274c Error</div>
          <pre style="color:#f38ba8; font-size:10px; white-space:pre-wrap;
                      word-break:break-all; max-height:120px; overflow-y:auto;
                      margin:0;">{_esc(err[:500])}</pre>
        </div>
        """
    elif status == "waiting":
        html += """
        <div style="text-align:center; padding:30px 0; color:#585b70;
                    font-size:13px;">
          \u23f3 Waiting\u2026
        </div>
        """
    html += "</div></div>"
    return html


def render_summary(states: Dict[str, Dict], total: int, throughput_unit: str) -> str:
    """Bottom summary bar chart + winner."""
    completed = [
        (k, s) for k, s in states.items()
        if s["status"] == "done"
    ]
    if not completed:
        return ('<div style="color:#585b70; text-align:center; padding:16px;">'
                'Race in progress\u2026</div>')

    results = []
    for key, s in completed:
        mdef = next(m for m in METHOD_DEFS if m["key"] == key)
        results.append({
            "key": key,
            "avg_ttft_ms": s.get("avg_ttft", 0) * 1000,
            "throughput": s.get("throughput", 0),
            "elapsed": s.get("elapsed", 0),
            "mdef": mdef,
        })

    max_ttft = max(r["avg_ttft_ms"] for r in results) if results else 1
    baseline = results[0]["avg_ttft_ms"] if results else 1

    html = """
    <div style="background:#181825; border-radius:14px; padding:20px; margin-top:16px;
                border:1px solid #313244;">
      <h3 style="color:#cdd6f4; margin:0 0 16px; font-size:16px; text-align:center;">
        Results
      </h3>
    """

    for r in results:
        color = r["mdef"]["color"]
        grad = r["mdef"]["bar_gradient"]
        pct = max(r["avg_ttft_ms"] / max(max_ttft, 0.01) * 100, 5)
        throughput_value = _throughput_value(r["throughput"], throughput_unit)
        throughput_label = _throughput_label(throughput_unit)

        speedup_html = ""
        if baseline > 0 and r["avg_ttft_ms"] < baseline * 0.95:
            sp = baseline / r["avg_ttft_ms"]
            speedup_html = (
                f'<span style="color:#50B86C; font-weight:700; font-size:12px; '
                f'white-space:nowrap;">\u26a1 {sp:.1f}x faster</span>'
            )
        elif r["key"] == results[0]["key"]:
            speedup_html = ('<span style="color:#888; font-size:11px; '
                            'white-space:nowrap;">baseline</span>')

        html += f"""
        <div style="margin:10px 0;">
          <div style="display:grid; grid-template-columns: 280px minmax(220px, 1fr) auto auto;
                      align-items:center; gap:12px;">
            <span style="color:{color}; font-weight:600; font-size:12px;
                         min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
              {r['mdef']['label']}
            </span>
            <div style="flex:1; background:#313244; border-radius:5px;
                        height:26px; overflow:hidden; min-width:0;">
              <div style="background:{grad};
                          width:{pct:.1f}%; height:100%; border-radius:5px;
                          transition:width 1s ease;"></div>
            </div>
            <span style="color:#ffffff; font-size:11px; font-weight:700;
                         font-family:monospace; white-space:nowrap;
                         background:#1e1e2e; border:1px solid #313244;
                         border-radius:999px; padding:5px 10px;">
              {r['avg_ttft_ms']:.0f}ms \u00b7 {throughput_value:.1f} {throughput_label}
            </span>
            {speedup_html}
          </div>
        </div>
        """


    html += "</div>"
    return html


# ---------------------------------------------------------------------------
# Race Engine
# ---------------------------------------------------------------------------
class RaceEngine:
    """Runs all configured methods sequentially on single GPU via ttft_worker.py."""

    NCCL_PORT_BASE = 2333

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def _run_worker(self, method_def: Dict, nccl_port: Optional[int] = None):
        cmd = [
            _CONFIG["python"], WORKER_SCRIPT,
            "--method", method_def["key"],
            "--samples", str(_CONFIG["samples"]),
            "--agents", str(_CONFIG["agents"]),
            "--model", _CONFIG["model"],
            "--backend", method_def["backend"],
            "--repo-root", method_def.get("repo_root", str(CURRENT_REPO_ROOT)),
            "--gpu", "0",
            "--in-length", str(_CONFIG["in_length"]),
            "--out-length", str(_CONFIG["out_length"]),
        ]
        if nccl_port is not None:
            cmd.extend(["--nccl-port", str(nccl_port)])

        env = os.environ.copy()
        env["KVCOMM_PAGED"] = "1" if method_def["backend"] in ("paged", "radix") else "0"
        if method_def["backend"] in ("paged", "radix"):
            env["KVCOMM_PAGED_BACKEND"] = (
                "radix" if method_def["backend"] == "radix" else "paged"
            )
        else:
            env.pop("KVCOMM_PAGED_BACKEND", None)
        env["CUDA_VISIBLE_DEVICES"] = "0"

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=None,  # let stderr flow to terminal for real-time debug
            env=env,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._queue.put(json.loads(line))
            except json.JSONDecodeError:
                pass

        proc.wait()
        stderr_out = ""
        if proc.returncode != 0:
            self._queue.put({
                "type": "error",
                "method": method_def["key"],
                "message": (
                    f"Worker exit {proc.returncode}: "
                    f"{stderr_out[-1000:] if stderr_out else ''}"
                ),
            })

    def _run_all(self):
        try:
            for mdef in METHOD_DEFS:
                nccl_port = (
                    self.NCCL_PORT_BASE
                    if mdef["backend"] in ("paged", "radix")
                    else None
                )
                self._run_worker(mdef, nccl_port)
        finally:
            self._queue.put({"type": "all_done"})
            self._running = False

    def start(self):
        self._running = True
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        threading.Thread(target=self._run_all, daemon=True).start()


race_engine = RaceEngine()


# ---------------------------------------------------------------------------
# Gradio streaming generator
# ---------------------------------------------------------------------------
def race_stream(throughput_unit: str):
    """Yield the current race cards plus the summary panel."""
    total = _CONFIG["samples"]
    states: Dict[str, Dict] = {}
    for mdef in METHOD_DEFS:
        states[mdef["key"]] = {
            "status": "waiting",
            "done_count": 0,
            "throughput": 0,
            "avg_ttft": 0,
            "prompt": "",
            "elapsed": 0,
            "error": "",
            "segment_count": 0,
        }

    def _build():
        cards = [render_method_card(i, mdef, states[mdef["key"]], total, throughput_unit)
                 for i, mdef in enumerate(METHOD_DEFS)]
        summary = render_summary(states, total, throughput_unit)
        return tuple(cards + [summary])

    race_engine.start()
    yield _build()

    while True:
        try:
            event = race_engine._queue.get(timeout=0.3)
        except queue.Empty:
            if not race_engine.running:
                break
            yield _build()
            continue

        etype = event.get("type")
        method = event.get("method", "")

        if etype == "method_start":
            states[method]["status"] = "running"

        elif etype == "request_done":
            s = states[method]
            s["done_count"] = event["idx"] + 1
            s["throughput"] = event.get("throughput", 0)
            s["avg_ttft"] = event.get("ttft_avg", 0)
            s["prompt"] = event.get("prompt", "")
            s["elapsed"] = event.get("elapsed", 0)

        elif etype == "method_done":
            s = states[method]
            s["status"] = "done"
            s["avg_ttft"] = event.get("avg_ttft", 0)
            s["throughput"] = event.get("throughput", 0)
            s["elapsed"] = event.get("elapsed", 0)
            s["done_count"] = total

        elif etype == "error":
            states[method]["status"] = "error"
            states[method]["error"] = event.get("message", "")

        elif etype == "all_done":
            break

        yield _build()

    yield _build()


def replay_stream(throughput_unit: str):
    """Replay a fixed benchmark trace in parallel using the current UI."""
    total = REPLAY_TOTAL_REQUESTS
    states: Dict[str, Dict] = {}
    for mdef in METHOD_DEFS:
        key = mdef["key"]
        spec = REPLAY_BENCHMARK[key]
        states[key] = {
            "status": "running",
            "done_count": 0,
            "throughput": spec["throughput_req_min"] / 60.0,
            "avg_ttft": spec["ttft_ms"] / 1000.0,
            "prompt": "Replay mode: benchmark snapshot",
            "elapsed": 0.0,
            "error": "",
            "segment_count": REPLAY_TOTAL_REQUESTS,
        }

    slowest_req_min = min(
        spec["throughput_req_min"]
        for spec in REPLAY_BENCHMARK.values()
        if spec["throughput_req_min"] > 0
    )
    durations = {
        key: REPLAY_TARGET_SECONDS * slowest_req_min / max(spec["throughput_req_min"], 1e-6)
        for key, spec in REPLAY_BENCHMARK.items()
    }

    def _build():
        cards = [
            render_method_card(i, mdef, states[mdef["key"]], total, throughput_unit)
            for i, mdef in enumerate(METHOD_DEFS)
        ]
        summary = render_summary(states, total, throughput_unit)
        return tuple(cards + [summary])

    start = time.perf_counter()
    tick_seconds = 0.05
    yield _build()

    while True:
        elapsed = time.perf_counter() - start
        all_done = True
        for mdef in METHOD_DEFS:
            key = mdef["key"]
            duration = max(durations[key], tick_seconds)
            progress = min(elapsed / duration, 1.0)
            done_count = min(total, int(progress * total))
            if progress >= 1.0:
                done_count = total
            states[key]["done_count"] = done_count
            states[key]["elapsed"] = round(min(elapsed, duration), 2)
            if progress >= 1.0:
                states[key]["status"] = "done"
            else:
                states[key]["status"] = "running"
                all_done = False

        yield _build()
        if all_done:
            break
        time.sleep(tick_seconds)

    yield _build()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
.gradio-container {
    background: #11111b !important;
    max-width: 1700px !important;
}
.race-col { min-width: 260px !important; }
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
footer { display: none !important; }
"""


def create_demo():
    with gr.Blocks(
        title="KVCOMM TTFT Race",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="gray",
            font=gr.themes.GoogleFont("Inter"),
        ),
    ) as demo:

        # Start button
        with gr.Row():
            with gr.Column():
                throughput_toggle = gr.Radio(
                    choices=["req/s", "req/min"],
                    value="req/s",
                    label="Throughput Unit",
                )
                with gr.Row():
                    race_btn = gr.Button(
                        "START",
                        variant="primary", size="lg",
                    )
                    replay_btn = gr.Button(
                        "REPLAY",
                        variant="secondary", size="lg",
                    )

        # 6 race columns in a 3+3 layout
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes="race-col"):
                card_0 = gr.HTML()
            with gr.Column(scale=1, elem_classes="race-col"):
                card_1 = gr.HTML()
            with gr.Column(scale=1, elem_classes="race-col"):
                card_2 = gr.HTML()

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes="race-col"):
                card_3 = gr.HTML()
            with gr.Column(scale=1, elem_classes="race-col"):
                card_4 = gr.HTML()
            with gr.Column(scale=1, elem_classes="race-col"):
                card_5 = gr.HTML()

        summary_out = gr.HTML()
        all_outputs = [card_0, card_1, card_2, card_3, card_4, card_5, summary_out]

        race_btn.click(fn=race_stream, inputs=[throughput_toggle], outputs=all_outputs)
        replay_btn.click(fn=replay_stream, inputs=[throughput_toggle], outputs=all_outputs)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="KVCOMM TTFT Race Demo")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model")
    parser.add_argument("--python", type=str, default="",
                        help="Python binary for workers")
    parser.add_argument("--samples", type=int, default=50,
                        help="Number of synthetic requests per method")
    parser.add_argument("--agents", type=int, default=5,
                        help="Number of CopyMachine agents per request")
    parser.add_argument("--in-length", type=int, default=512,
                        help="Input token length for synthetic prompts")
    parser.add_argument("--out-length", type=int, default=512,
                        help="Output token length for synthetic prompts")
    args = parser.parse_args()

    _CONFIG["model"] = args.model
    _CONFIG["samples"] = args.samples
    _CONFIG["agents"] = args.agents
    _CONFIG["in_length"] = args.in_length
    _CONFIG["out_length"] = args.out_length
    if args.python:
        _CONFIG["python"] = args.python

    print(f"  Model:      {_CONFIG['model']}")
    print(f"  Python:     {_CONFIG['python']}")
    print(f"  Worker:     {WORKER_SCRIPT}")
    print(f"  Samples:    {_CONFIG['samples']}")
    print(f"  Agents:     {_CONFIG['agents']}")
    print(f"  In/Out Len: {_CONFIG['in_length']}/{_CONFIG['out_length']}")

    demo = create_demo()
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
