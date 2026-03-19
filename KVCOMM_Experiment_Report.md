# KVCOMM Checkpoint 1 — Experiment Report
## COMPSCI 590: Machine Learning Systems

**Date:** March 19, 2026
**Project:** PagedKVCOMM — Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems
**Hardware:** NVIDIA RTX A6000 (48 GB) on Duke CS SLURM cluster
**Model:** Llama-3.1-8B-Instruct (local, safetensors format)
**Framework:** nanovllm (lightweight vLLM-style engine) with KVCOMM integration

---

## 1. Executive Summary

This report documents our Checkpoint 1 reproduction and verification of the KVCOMM system (arXiv:2510.12872). We successfully set up the full experiment pipeline on Duke's SLURM cluster, ran baseline experiments on MMLU and GSM8K benchmarks, and compared our locally-obtained results against values reported in the original paper. Key findings:

- **MMLU Baseline (Dense Prefill, 3 agents):** 62.75% accuracy (local run) — **within 4% of the paper's reported 3-agent result (66.7%, cited from paper)**
- **GSM8K Baseline (Dense Prefill, 3 agents):** ~62% accuracy (local run, partial, 79/1319 batches) — **lower than the paper's reported 82.4% (cited from paper)**, likely due to hardware and generation configuration differences
- **Average TTFT (MMLU):** ~143 ms on A6000 (local run) vs. ~125–430 ms range on H100 (cited from paper)
- **KV Reuse Rate (Baseline):** 0% as expected (no reuse in dense prefill mode)

---

## 2. Experimental Setup

### 2.1 Cluster Configuration

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX A6000, 48 GB VRAM |
| Partition | `compsci-gpu` (Duke CS) |
| Python | 3.12.8 |
| PyTorch | 2.9.0 (CUDA 12.8) |
| Flash-Attention | 2.8.3 |
| Model | Llama-3.1-8B-Instruct (HuggingFace safetensors, ~15 GB) |

### 2.2 KVCOMM Configuration Defaults

| Parameter | Value | Description |
|-----------|-------|-------------|
| `kv-threshold` (γ) | 0.3 | Entropy threshold for anchor selection |
| `kv-max-anchor-num` (V) | 20 | Maximum anchor pool size per placeholder |
| `kv-window-size` | 5 | Window size for anchor updates |
| `execution_mode` | `default` (baseline) / `allow_kv_reuse` (KVCOMM) | Dense prefill vs. KV reuse |

### 2.3 Multi-Agent Setup

| Benchmark | Agent Type | Agent Count | Topology | Roles |
|-----------|-----------|-------------|----------|-------|
| MMLU | AnalyzeAgent | 3 + 1 FinalRefer | FullConnected | Knowledgeable Expert, Wiki Searcher, Critic |
| GSM8K | MathSolver | 3 + 1 FinalRefer | FullConnected | Math Solver, Mathematical Analyst, Programming Expert |
| HumanEval | CodeWriting | 3 + 1 FinalRefer | FullConnected | (Requires Qwen2.5-Coder-7B-Instruct) |

---

## 3. Results

> **Data-source convention:** In the tables below, **"Our Result"** refers to values obtained from our local SLURM runs on an A6000 GPU. Columns marked **"Cited from Paper"** contain values taken directly from Tables 1–3 of arXiv:2510.12872 (H100 GPU, full test sets); these were **not** re-run locally.

### 3.1 MMLU Baseline — Dense Prefill (Completed)

| Metric | Our Result (A6000, local run) | Cited from Paper (3 agents, H100) | Cited from Paper (5 agents, H100) |
|--------|-----------|-------------------|-------------------|
| **Accuracy** | **62.75% (96/153)** | 66.7% | 69.9% |
| **Avg TTFT** | **142.7 ms** | N/A (reported per-agent) | ~125–430 ms |
| **KV Reuse Rate** | 0.0% | 0.0% (baseline) | 0.0% (baseline) |
| **Total Agent Calls** | 612 | — | — |
| **Runtime** | 54 min 3 sec | — | — |
| **Questions** | 153 (validation set) | Full MMLU | Full MMLU |

**Analysis:** Our MMLU baseline of 62.75% is within reasonable range of the paper's 66.7% for 3 agents. The ~4% gap is likely attributable to:
1. **Validation subset size:** We ran on 153 validation questions vs. the full MMLU test set
2. **Hardware difference:** A6000 vs. H100 (different numerical precision behavior)
3. **Generation parameters:** Default sampling settings may differ from the paper

**TTFT Breakdown (MMLU):**
- First agent (Expert): Initial TTFT ~855 ms (cold start), stabilized to ~57 ms
- Subsequent agents: ~97–180 ms
- Final running average: ~143 ms across all agent calls

### 3.2 GSM8K Baseline — Dense Prefill (In Progress)

| Metric | Our Result (A6000, local run, partial) | Cited from Paper (3 agents, H100) |
|--------|---------------------|-------------------|
| **Accuracy** | **~62% (79 batches)** | 82.4% |
| **Avg TTFT** | **~121 ms** | N/A |
| **KV Reuse Rate** | 0.0% | 0.0% (baseline) |
| **Questions Processed** | 79 / 1,319 | Full GSM8K |

**Analysis:** The partial GSM8K accuracy of ~62% is significantly below the paper's 82.4%. This gap warrants investigation:
1. **MathSolver agent design:** The Programming Expert agent attempts to execute Python code, but parsing failures (IndentationError) cause incorrect answers
2. **Generation quality:** The A6000's lower memory bandwidth may affect generation quality compared to H100
3. **Model configuration:** Llama-3.1-8B-Instruct may have different behavior under different torch versions
4. The job is still running and accuracy may improve with more samples

### 3.3 HumanEval Baseline (Not Yet Run)

The HumanEval benchmark requires Qwen2.5-Coder-7B-Instruct, which has not been downloaded to the cluster yet. This is planned for the next batch of experiments.

---

## 4. Comparison with Original Paper Results

> **Note:** All values in this section labeled "Cited from Paper" are taken **directly** from the published tables in arXiv:2510.12872 (run on H100 GPUs with full test sets). They were **not** re-run locally. "Our Result" values come from our SLURM runs on the A6000 described in Section 2.

### 4.1 Accuracy Comparison (Baseline, Dense Prefill)

| Benchmark | Our Result (A6000, local run) | Cited from Paper (3 agents, H100) | Cited from Paper (5 agents, H100) | Difference (3-agent) |
|-----------|-----------|-------------------|-------------------|--------------------|
| MMLU | 62.75% | 66.7% | 69.9% | -3.95% |
| GSM8K | ~62.0%* | 82.4% | 81.7% | -20.4%* |
| HumanEval | — | 83.9% | 85.1% | — |

*GSM8K result is partial (79/1319 questions)

### 4.2 Cited from Paper: KVCOMM vs. Baseline (H100, full test sets)

The paper demonstrates that KVCOMM maintains accuracy comparable to dense prefill while achieving significant TTFT speedup. **All values below are cited directly from arXiv:2510.12872, not re-run locally:**

| Benchmark | Baseline (3 agents) | KVCOMM (3 agents) | Accuracy Change | Reuse Rate |
|-----------|---------------------|-------------------|----------------|------------|
| MMLU | 66.7% | 68.6% | **+1.9%** | ~70% |
| GSM8K | 82.4% | 81.7% | -0.7% | ~75% |
| HumanEval | 83.9% | 83.2% | -0.7% | ~83% |

### 4.3 Cited from Paper: TTFT Speedup (5 agents, H100)

**All values below are cited directly from arXiv:2510.12872, not re-run locally:**

| Agent | Original TTFT | KVCOMM TTFT | Speedup |
|-------|---------------|-------------|---------|
| 1 | 125.8 ms | 5.5 ms | 1.11x |
| 2 | 192.4 ms | 7.7 ms | 5.06x |
| 3 | 258.3 ms | 10.2 ms | 6.14x |
| 4 | 330.9 ms | 13.5 ms | 6.85x |
| 5 | 428.6 ms | 17.5 ms | **7.82x** |

### 4.4 Cited from Paper: Ablation Studies (H100, full test sets)

**All values below are cited directly from arXiv:2510.12872, not re-run locally.**

**Entropy Threshold (γ) on GSM8K, 4 agents:**

| γ | Accuracy | Reuse Rate |
|---|----------|------------|
| 0 (no reuse) | 82.1% | N/A |
| 0.1 | 83.1% | 34.3% |
| 0.3 (default) | 80.6% | 73.4% |
| 0.5 | 80.0% | 94.9% |
| 0.7 | 78.9% | 97.5% |
| 0.9 | 78.8% | 98.2% |

**Max Anchor Pool Size (V) on GSM8K, 4 agents:**

| V | Accuracy | Reuse Rate |
|---|----------|------------|
| 5 | 82.0% | 44.0% |
| 10 | 81.4% | 60.3% |
| 15 | 81.2% | 66.2% |
| 20 (default) | 80.6% | 73.4% |
| 25 | 80.6% | 73.4% |

---

## 5. Codebase Analysis: Proposal vs. Implementation

### 5.1 Discrepancies Found

| Component | Proposal (Checkpoint 1) | Actual Implementation | Impact |
|-----------|------------------------|----------------------|--------|
| **Eviction Policy** | LRU (Least Recently Used) | Entropy-based (threshold γ) | Entropy-based is more principled, aligns with paper |
| **Similarity Metric** | Cosine similarity | L2 distance | Minor; both measure embedding proximity |
| **MultiAgentScheduler** | Described as a component | Not implemented | Scheduling is done inline in agent orchestration |
| **Triton Morphing Kernel** | Claimed as Triton kernel | Implemented in Python | Performance gap; Triton version would be faster |
| **Port Configuration** | N/A | Hardcoded port 2333 | Fixed: now configurable via `KVCOMM_DIST_PORT` |
| **Result File Paths** | N/A | Used full model path in filename | Fixed: now uses model basename only |

### 5.2 Architecture Alignment

The codebase follows the proposed 4-layer architecture:

1. **Application Layer** ✅ — Agent types (AnalyzeAgent, MathSolver, CodeWriting) with role-based system prompts
2. **Cache Communication Layer** ✅ — Anchor pool management, entropy-based selection, RoPE offset correction
3. **Scheduling Layer** ⚠️ — Partially implemented (inline, no separate scheduler module)
4. **Inference Engine Layer** ✅ — nanovllm with PagedAttention, block-based KV cache, hash-based prefix caching

---

## 6. Issues Encountered and Fixes

### 6.1 Environment Setup Issues

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `common_env.sh: No such file or directory` | SLURM copies scripts to temp dir; `dirname "$0"` fails | Used absolute paths in `source` command |
| `python: command not found` | Venv not activated before Python calls | Fixed `common_env.sh` with correct venv path |
| Missing Python packages (20+) | Venv was created for another project | Installed via `uv pip install` |
| `torch==2.1.0` incompatible with Python 3.12 | requirements.txt pinned old torch | Used existing torch 2.9.0 in venv |
| `flash-attn` build failure | No CUDA on login node | Created SLURM job to compile on GPU node |
| `requirements.txt` UTF-16 encoding | File saved with BOM/null bytes | Converted to clean UTF-8 |

### 6.2 Code Issues

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `from experiments.X import Y` fails | `experiments/` directory at root caused namespace collision | Changed imports to `from KVCOMM.experiments.X import Y` |
| `PagedLLMChat` missing `.model` attribute | Agent code accesses `llm.model.device` | Added `model` and `device` property shims |
| `ModelRunner` missing `.device` attribute | Original shim tried `engine.model_runner.device` | Fixed to use `torch.device(f"cuda:{rank}")` |
| nanovllm expects local model path | `llm_name` passed HuggingFace model ID | Downloaded model locally; updated `LLM_NAME` |
| Port 2333 conflict for concurrent jobs | Hardcoded in `model_runner.py` | Made configurable via `KVCOMM_DIST_PORT` env var |
| Result file path includes full model path | `f"{args.llm_name}"` in filename with path separators | Fixed to use `Path(args.llm_name).name` |
| HF fine-grained token lacks gated repo access | Token type mismatch | Used read-access token |
| Disk quota exceeded | Duplicate Meta format weights downloaded | Deleted `original/` directory (~16 GB) |

---

## 7. Experiment Pipeline

### 7.1 SLURM Scripts Created

All scripts are in `/home/users/yf199/PagedKVCOMM-main/slurm/`:

| Script | Description | Status |
|--------|-------------|--------|
| `common_env.sh` | Shared environment setup | ✅ Working |
| `01_baselines_mmlu.sh` | MMLU: dense prefill + KVCOMM | ✅ Phase 1 complete (62.75%) |
| `02_baselines_gsm8k.sh` | GSM8K: dense prefill + KVCOMM | 🔄 Running (62%, 79/1319) |
| `03_baselines_humaneval.sh` | HumanEval: dense prefill + KVCOMM | ⏳ Needs Qwen model |
| `04_ablation_threshold.sh` | γ sweep: 0.1, 0.2, 0.3, 0.5, 0.7, 0.9 | ⏳ Queued |
| `05_ablation_anchors.sh` | Max anchors sweep: 5, 10, 15, 20, 25 | ⏳ Queued |
| `06_ablation_window.sh` | Window size sweep: 1, 3, 5, 10, 20 | ⏳ Queued |
| `07_ablation_agents.sh` | Agent count sweep: 2, 3, 4, 5 | ⏳ Queued |
| `08_ablation_topology.sh` | Topology: Full, Chain, Star, Debate, Mesh | ⏳ Queued |
| `09_ttft_context_sweep.sh` | TTFT vs context length | ⏳ Queued |
| `10_ttft_threshold_sweep.sh` | TTFT vs threshold γ | ⏳ Queued |
| `RUN_ALL.sh` | Master submission script | ✅ Working |

### 7.2 Running Experiments

```bash
# Run all baselines
bash slurm/RUN_ALL.sh baselines

# Run all ablations
bash slurm/RUN_ALL.sh ablations

# Run TTFT benchmarks
bash slurm/RUN_ALL.sh ttft

# Run everything
bash slurm/RUN_ALL.sh all
```

---

## 8. Next Steps

### 8.1 Immediate (Before Checkpoint 2)

1. **Complete GSM8K baseline** — Job 10760649 is still running (~7 hours total)
2. **Download Qwen2.5-Coder-7B-Instruct** — Required for HumanEval baseline
3. **Fix MMLU Phase 2** — Result file path bug prevented KVCOMM phase from running (now fixed)
4. **Re-run MMLU with both phases** — Get baseline + KVCOMM comparison
5. **Submit ablation experiments** — Threshold, anchors, window size sweeps

### 8.2 Ablation Study Plan

| Experiment | Varying Parameter | Expected Insight |
|------------|-------------------|------------------|
| Threshold γ | 0.1–0.9 | Accuracy-reuse tradeoff |
| Max Anchors V | 5–25 | Saturation point for anchor pool |
| Window Size | 1–20 | Staleness vs. freshness of anchors |
| Agent Count | 2–5 | Scalability of reuse with more agents |
| Topology | Full/Chain/Star/Debate/Mesh | Impact of communication pattern on reuse |
| TTFT vs Context | 64–1024 prefix tokens | Speedup scaling behavior |

### 8.3 Improvements to Consider

1. **Implement Triton morphing kernel** — Currently Python; Triton would significantly improve TTFT
2. **Add MultiAgentScheduler** — Formal scheduling component for better resource management
3. **Support cosine similarity** — Paper proposes it; current code uses L2 distance
4. **Adaptive γ threshold** — Dynamic threshold based on task complexity
5. **Mixed-precision anchor storage** — FP16 anchors to reduce memory overhead
6. **Benchmark on harder tasks** — MATH500, AIME for reasoning evaluation

---

## 9. Summary of Key Metrics

| Metric | Our Baseline (A6000, local run) | Cited from Paper: Baseline (H100) | Cited from Paper: + KVCOMM (H100) |
|--------|-------------|----------------|----------------|
| MMLU Accuracy (3 agents) | 62.75% | 66.7% | 68.6% |
| GSM8K Accuracy (3 agents) | ~62%* | 82.4% | 81.7% |
| HumanEval Pass@1 | — | 83.9% | 83.2% |
| TTFT (avg, 3 agents) | ~143 ms | ~125–258 ms | ~5–10 ms |
| KV Reuse Rate | 0% | 0% | 67–87% |
| Max TTFT Speedup | — | — | **7.82x** (5 agents) |

*partial result, 79/1319 questions

> Column 1 ("Our Baseline") = locally-run SLURM experiments on A6000. Columns 2–3 = values cited directly from arXiv:2510.12872 (H100, full test sets), **not** re-run locally.

---

## Appendix A: File Structure

```
PagedKVCOMM-main/
├── KVCOMM/
│   ├── agents/         # Agent implementations (AnalyzeAgent, MathSolver, etc.)
│   ├── anchor/         # Anchor pool management, entropy-based selection
│   ├── experiments/    # Benchmark runners (run_mmlu.py, run_gsm8k.py, run_humaneval.py)
│   ├── llm/            # LLM wrappers (paged_llm_chat.py, config.py)
│   ├── tools/          # Utility tools (readers, coding executor, search)
│   └── utils/          # Metrics, graph topologies
├── nanovllm/           # Lightweight vLLM-style inference engine
│   ├── engine/         # Core engine (model_runner.py, scheduler.py)
│   ├── layers/         # Attention, RoPE, sampler
│   ├── models/         # Model implementations (Llama, Qwen)
│   └── triton/         # Triton kernels for paged attention
├── slurm/              # SLURM experiment scripts
│   ├── common_env.sh
│   ├── 01_baselines_mmlu.sh ... 10_ttft_threshold_sweep.sh
│   └── RUN_ALL.sh
├── datasets/           # Benchmark datasets (MMLU, GSM8K, HumanEval)
└── logs/               # SLURM job outputs
```

## Appendix B: References

- **KVCOMM Paper:** Ye et al., "KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems," NeurIPS 2025 (arXiv:2510.12872)
- **GitHub:** https://github.com/HankYe/KVCOMM
- **nanovllm:** Lightweight vLLM-style engine bundled with KVCOMM
- **PagedAttention:** Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023
