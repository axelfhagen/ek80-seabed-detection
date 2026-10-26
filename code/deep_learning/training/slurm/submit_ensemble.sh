#!/bin/bash
# Submit a two-phase ensemble training + evaluation pipeline.
#
# Usage:
#   ./submit_ensemble.sh [N_MEMBERS] [EXTRA_TRAIN_ARGS...]
#
# Examples:
#   ./submit_ensemble.sh                          # 10 members, default config
#   ./submit_ensemble.sh 5                        # 5 members
#   ./submit_ensemble.sh 10 --loss-type focal      # 10 members, focal loss
#   ./submit_ensemble.sh 3 --encoder resnet50      # 3 members, resnet50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

N_MEMBERS="${1:-10}"
shift 2>/dev/null || true
EXTRA_ARGS="$*"

ENSEMBLE_ID="$(date +%Y%m%d_%H%M%S)"

echo "============================================================"
echo "Deep Ensemble Submission"
echo "  Members:     ${N_MEMBERS}"
echo "  Ensemble ID: ${ENSEMBLE_ID}"
echo "  Extra args:  ${EXTRA_ARGS:-<none>}"
echo "============================================================"
echo ""

read -p "Submit ${N_MEMBERS} training jobs + 1 evaluation job? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Phase 1: training array job
PHASE1_JOB=$(sbatch \
    --array="1-${N_MEMBERS}" \
    --export="ALL,ENSEMBLE_ID=${ENSEMBLE_ID},EXTRA_ARGS=${EXTRA_ARGS}" \
    --parsable \
    "${SCRIPT_DIR}/ensemble_train.sh")

echo "Phase 1 submitted: job ${PHASE1_JOB} (${N_MEMBERS} members)"

# Phase 2: evaluation (runs after all training completes)
PHASE2_JOB=$(sbatch \
    --dependency="afterok:${PHASE1_JOB}" \
    --export="ALL,ENSEMBLE_ID=${ENSEMBLE_ID}" \
    --parsable \
    "${SCRIPT_DIR}/ensemble_eval.sh")

echo "Phase 2 submitted: job ${PHASE2_JOB} (depends on ${PHASE1_JOB})"

echo ""
echo "Monitor: squeue -u $(whoami)"
echo "Checkpoints: code/checkpoints/exp_ensemble_${ENSEMBLE_ID}/"
echo "Results:     code/ensemble_results/ensemble_${ENSEMBLE_ID}/"
