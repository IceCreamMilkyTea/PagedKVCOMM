#!/bin/bash
# submit_all.sh — Submit all benchmark jobs to Slurm in parallel.
# Usage:
#   cd /path/to/PagedKVCOMM
#   bash scripts/submit_all.sh [--model MODEL_NAME] [--results_dir DIR]
#
# All 4 jobs share the same RESULTS_DIR so aggregate_results.py can
# collect from a single root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/result/$(date +%Y%m%d_%H%M%S)}"

USE_LOCAL_REF=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2";       shift 2 ;;
        --results_dir) RESULTS_DIR="$2"; shift 2 ;;
        --use-local-reference) USE_LOCAL_REF="--use-local-reference"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "${PROJECT_ROOT}/logs"

echo "Submitting all benchmarks"
echo "  Model:       ${MODEL}"
echo "  Results dir: ${RESULTS_DIR}"
echo ""

export MODEL RESULTS_DIR

JOB_GSM8K=$(sbatch --chdir="$PROJECT_ROOT" "$SCRIPT_DIR/run_gsm8k.sh" ${USE_LOCAL_REF} 2>&1)
echo "GSM8K:     ${JOB_GSM8K}"

JOB_MMLU=$(sbatch --chdir="$PROJECT_ROOT" "$SCRIPT_DIR/run_mmlu.sh" ${USE_LOCAL_REF} 2>&1)
echo "MMLU:      ${JOB_MMLU}"

JOB_HUMANEVAL=$(sbatch --chdir="$PROJECT_ROOT" "$SCRIPT_DIR/run_humaneval.sh" ${USE_LOCAL_REF} 2>&1)
echo "HumanEval: ${JOB_HUMANEVAL}"

JOB_TTFT=$(sbatch --chdir="$PROJECT_ROOT" "$SCRIPT_DIR/run_ttft.sh" ${USE_LOCAL_REF} 2>&1)
echo "TTFT:      ${JOB_TTFT}"

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "After completion, run:"
echo "  python experiments/aggregate_results.py --results_dir ${RESULTS_DIR}"
