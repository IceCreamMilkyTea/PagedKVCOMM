#!/bin/bash
#SBATCH --job-name=mmlu-bench
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/mmlu_%j.out
#SBATCH --error=logs/mmlu_%j.err

set -uo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

mkdir -p logs

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/result/$(date +%Y%m%d_%H%M%S)}"
USE_LOCAL_REF=""
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2";       shift 2 ;;
        --results_dir) RESULTS_DIR="$2"; shift 2 ;;
        --use-local-reference) USE_LOCAL_REF="--use-local-reference"; shift ;;
        *) echo "Unknown argument: $1"; shift ;;
    esac
done

echo "=== MMLU Benchmark ==="
echo "Model:       ${MODEL}"
echo "Results dir: ${RESULTS_DIR}"
echo "Host:        $(hostname)"
echo "Start time:  $(date)"

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
nvidia-smi || true

METHODS=("dense" "kvcomm" "paged_kvcomm")
FAILED=0

for METHOD in "${METHODS[@]}"; do
    echo
    echo "[MMLU / ${METHOD}] Running..."
    python experiments/run_mmlu.py \
        --llm_name "${MODEL}" \
        --method "${METHOD}" \
        --agent_nums 5 \
        --output_dir "${RESULTS_DIR}/mmlu/${METHOD}" \
        ${USE_LOCAL_REF} \
    || { echo "[MMLU / ${METHOD}] FAILED"; FAILED=$((FAILED+1)); }
done

echo
echo "MMLU done. (${FAILED} failures)"
echo "End time: $(date)"
exit "${FAILED}"
