# GSM8K Local Reference Experiments

This directory contains scripts for running GSM8K experiments with the `--use-local-reference` flag, which enables cross-agent KV cache offsetting for improved inner-round communication.

## Quick Start

### 1. Single Experiment with Local Reference (Paged Backend)

```bash
# Quick launch with defaults
./scripts/quick_run_local_ref.sh

# Or use the Python launcher for more control
python scripts/run_gsm8k_local_reference.py --backend paged --use-local-ref
```

### 2. Debug Mode (15 samples)

```bash
./scripts/quick_run_local_ref.sh debug

# Or
python scripts/run_gsm8k_local_reference.py --backend paged --use-local-ref --debug
```

### 3. Baseline Comparison (Without Local Reference)

```bash
./scripts/quick_run_local_ref.sh baseline

# Or
python scripts/run_gsm8k_local_reference.py --backend paged
```

### 4. Full Ablation Study (Compare All Configurations)

```bash
# Run 4 experiments: paged/hf × baseline/local-ref
python scripts/run_gsm8k_ablation_local_ref.py

# Run only paged backend experiments
python scripts/run_gsm8k_ablation_local_ref.py --backends paged

# Debug mode (faster testing)
python scripts/run_gsm8k_ablation_local_ref.py --debug
```

## Scripts Overview

### `run_gsm8k_local_reference.py`

Main launcher for individual experiments. Supports:
- **Backends**: `paged` (nano-vllm) or `hf` (HuggingFace)
- **Local reference**: Enable with `--use-local-ref`
- **Topologies**: FullConnected, Chain, Star, Debate, etc.
- **KV tuning**: threshold, max-anchor-num, window-size, etc.

**Example usage:**
```bash
# Paged backend with local reference
python scripts/run_gsm8k_local_reference.py \
    --backend paged \
    --use-local-ref \
    --num-agents 3 \
    --batch-size 1

# HF backend baseline
python scripts/run_gsm8k_local_reference.py \
    --backend hf \
    --num-agents 3

# Custom KV cache configuration
python scripts/run_gsm8k_local_reference.py \
    --backend paged \
    --use-local-ref \
    --kv-threshold 0.95 \
    --kv-max-anchor-num 30 \
    --kv-window-size 10
```

### `run_gsm8k_ablation_local_ref.py`

Batch runner for ablation studies. Automatically runs multiple configurations:

**Default ablation matrix:**
- Paged backend + Baseline
- Paged backend + Local reference
- HF backend + Baseline
- HF backend + Local reference

**Example usage:**
```bash
# Full ablation (4 experiments)
python scripts/run_gsm8k_ablation_local_ref.py

# Only paged backend
python scripts/run_gsm8k_ablation_local_ref.py --backends paged

# Skip baseline experiments
python scripts/run_gsm8k_ablation_local_ref.py --skip-baseline

# Debug mode (15 samples per experiment)
python scripts/run_gsm8k_ablation_local_ref.py --debug
```

**Output structure:**
```
KVCOMM/result/gsm8k_ablation_<timestamp>/
├── ablation_report.txt          # Summary of all experiments
├── paged_baseline/              # Results for paged without local-ref
├── paged_local_ref/             # Results for paged with local-ref
├── hf_baseline/                 # Results for HF without local-ref
└── hf_local_ref/                # Results for HF with local-ref
```

### `quick_run_local_ref.sh`

Shell script for quick launches with common configurations.

**Usage:**
```bash
# Default: paged + local-ref + full dataset
./scripts/quick_run_local_ref.sh

# Debug mode (15 samples)
./scripts/quick_run_local_ref.sh debug

# Baseline (no local reference)
./scripts/quick_run_local_ref.sh baseline

# HuggingFace backend
./scripts/quick_run_local_ref.sh hf

# Combine options
./scripts/quick_run_local_ref.sh hf debug
```

## Configuration Options

### Backend Selection

- **`paged`**: Uses nano-vllm with paged attention and flash attention
  - Efficient memory management with block-based KV cache
  - Zero-copy KV reuse across agents
  - Set `KVCOMM_PAGED=1` automatically

- **`hf`**: Uses HuggingFace Transformers
  - Standard dense prefill
  - Tensor-based KV cache storage
  - Set `KVCOMM_PAGED=0` automatically

### Local Reference Flag

- **`--use-local-reference`**: Enable cross-agent KV offsetting
  - Uses upstream agent's KV cache as local reference
  - Computes weighted delta from similar historical anchors
  - Applies delta to local reference for improved inner-round communication

### KV Cache Tuning

- **`--kv-threshold FLOAT`**: Similarity threshold (default: 0.99)
  - Higher values → more selective anchor matching
  - Lower values → more anchor reuse

- **`--kv-max-anchor-num INT`**: Max anchors per placeholder (default: 20)
  - Limits memory usage
  - Triggers LRU eviction when exceeded

- **`--kv-window-size INT`**: Eviction window size (default: 5)
  - Number of least-recent anchors to consider for eviction

- **`--kv-thread-workers INT`**: Thread pool size for parallel KV operations
- **`--kv-worker-timeout FLOAT`**: Worker timeout in seconds

## Output Structure

Each experiment creates a timestamped directory:

```
KVCOMM/result/gsm8k_<mode>_<backend>_<timestamp>/
├── logs/
│   └── log.txt                          # Detailed execution logs
├── gsm8k_<model>_<timestamp>.json       # Main results file
├── delta_diagnostic/                    # Delta similarity analysis (if enabled)
│   ├── delta_stats.json
│   └── similarity_heatmap.png
└── metrics/                             # Per-request metrics
    ├── request_<uid>_metrics.json
    └── ...
```

**Main results JSON format:**
```json
[
  {
    "Question": "...",
    "Answer": 42,
    "Step": "...",
    "Response": ["..."],
    "Attempt answer": "42",
    "Solved": true,
    "Total solved": 150,
    "Total executed": 200,
    "Accuracy": 0.75
  },
  ...
]
```

## Environment Variables

- **`KVCOMM_MODEL`**: Default model path
- **`KVCOMM_PYTHON`**: Python binary to use
- **`KVCOMM_PAGED`**: Set automatically by scripts (1=paged, 0=hf)
- **`SEED`**: Random seed (default: 42)

## Examples

### Example 1: Quick Test on Debug Dataset

```bash
# Test local reference on 15 samples
python scripts/run_gsm8k_local_reference.py \
    --backend paged \
    --use-local-ref \
    --debug
```

### Example 2: Full Dataset Comparison

```bash
# Run baseline
python scripts/run_gsm8k_local_reference.py \
    --backend paged \
    --output-dir results/baseline

# Run with local reference
python scripts/run_gsm8k_local_reference.py \
    --backend paged \
    --use-local-ref \
    --output-dir results/local_ref
```

### Example 3: Full Ablation Study

```bash
# Run all 4 configurations
python scripts/run_gsm8k_ablation_local_ref.py \
    --num-agents 3 \
    --batch-size 1 \
    --output-root results/ablation_study
```

### Example 4: Custom Topology and KV Tuning

```bash
python scripts/run_gsm8k_local_reference.py \
    --backend paged \
    --use-local-ref \
    --mode Chain \
    --num-agents 5 \
    --kv-threshold 0.95 \
    --kv-max-anchor-num 30
```

## Troubleshooting

### Out of Memory (OOM)

- Reduce `--kv-max-anchor-num` (default: 20)
- Reduce `--num-agents`
- Use `--batch-size 1`
- Free unused anchors more aggressively with smaller `--kv-window-size`

### Slow Performance

- Ensure Flash Attention is enabled (automatic for paged backend)
- Increase `--kv-thread-workers` for parallel KV operations
- Reduce `--kv-threshold` if too few anchors match
- Check GPU utilization and adjust batch size

### No Anchor Reuse

- Lower `--kv-threshold` (try 0.90-0.95)
- Check that prompts have consistent structure
- Verify `--prefix` matches across runs (default: "Q:\n")

## Performance Metrics

The experiments automatically collect:
- **Accuracy**: Correctness on GSM8K dataset
- **TTFT** (Time to First Token): Prefill latency
- **Generation latency**: Decode throughput
- **KV reuse rate**: Percentage of tokens using cached KV
- **Anchor match rate**: Similarity-based anchor selection
- **Delta similarity**: Cross-agent KV offset effectiveness (diagnostic)

Check `logs/log.txt` for detailed per-request metrics.
