#!/bin/bash
#SBATCH --job-name=paged-smoke
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/paged_smoke_%j.out
#SBATCH --error=logs/paged_smoke_%j.err

# Smoke test: run paged_kvcomm on a tiny GSM8K subset.
# Usage:  sbatch scripts/smoke_paged.sh
#    or:  sbatch scripts/smoke_paged.sh --model Qwen/Qwen3.5-0.8B
#    or:  sbatch scripts/smoke_paged.sh --samples 3

set -uo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

mkdir -p logs

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/result/smoke_$(date +%Y%m%d_%H%M%S)}"
SAMPLES="${SAMPLES:-2}"
USE_LOCAL_REF=""
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2";       shift 2 ;;
        --results_dir) RESULTS_DIR="$2"; shift 2 ;;
        --samples)     SAMPLES="$2";     shift 2 ;;
        --use-local-reference) USE_LOCAL_REF="--use-local-reference"; shift ;;
        *) echo "Unknown argument: $1"; shift ;;
    esac
done

if ! [[ "$SAMPLES" =~ ^[0-9]+$ ]] || [[ "$SAMPLES" -lt 1 ]]; then
    echo "Invalid --samples value: ${SAMPLES} (must be integer >= 1)"
    exit 2
fi

echo "=== Smoke Test: paged_kvcomm × ${SAMPLES} samples ==="
echo "Model:       ${MODEL}"
echo "Results dir: ${RESULTS_DIR}"
echo "Samples:     ${SAMPLES}"
echo "Host:        $(hostname)"
echo "Start time:  $(date)"

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
nvidia-smi || true

# Build a tiny dataset using the first N distinct samples.
MINI_DATASET="/tmp/gsm8k_smoke_${SAMPLES}sample_${SLURM_JOB_ID:-$$}.jsonl"
head -n "$SAMPLES" "$PROJECT_ROOT/datasets/gsm8k/gsm8k.jsonl" > "$MINI_DATASET"
echo "Using mini dataset: ${MINI_DATASET} ($(wc -l < "$MINI_DATASET") lines)"

python experiments/run_gsm8k.py \
    --llm_name "${MODEL}" \
    --method paged_kvcomm \
    --agent_nums 3 \
    --dataset_json "${MINI_DATASET}" \
    --output_dir "${RESULTS_DIR}/gsm8k/paged_kvcomm" \
    ${USE_LOCAL_REF}

EXIT_CODE=$?

rm -f "$MINI_DATASET"

echo
echo "Exit code: ${EXIT_CODE}"
echo "End time:  $(date)"
exit "${EXIT_CODE}"
