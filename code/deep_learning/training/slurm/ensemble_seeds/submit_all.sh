#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Submit the seed-diverse ResNet-34 U-Net deep ensemble:
#   5 training jobs (3 on share-ie-idi, 2 on share-ie-imf)
#   1 dependent eval job (afterok on all 5 trainers)
#
# Idempotent against partially-submitted runs: if you re-run after a member
# crashed, just re-sbatch that one trainer manually and re-submit eval with
# the new --dependency string.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

TRAINERS=(
    run_ensemble_resnet34_seed42.sh
    run_ensemble_resnet34_seed1.sh
    run_ensemble_resnet34_seed7.sh
    run_ensemble_resnet34_seed123.sh
    run_ensemble_resnet34_seed2025.sh
)

echo "═══════════════════════════════════════════════════════════════════"
echo "Deep ensemble submission — ResNet-34 U-Net, 5 seeds"
echo "═══════════════════════════════════════════════════════════════════"

JOB_IDS=()
for script in "${TRAINERS[@]}"; do
    OUT="$(sbatch "$script")"
    # sbatch prints: "Submitted batch job 12345"
    JID="$(echo "$OUT" | awk '{print $NF}')"
    JOB_IDS+=("$JID")
    printf "  %-45s → %s\n" "$script" "$JID"
done

DEP="$(IFS=:; echo "${JOB_IDS[*]}")"
echo
echo "Submitting eval with --dependency=afterok:${DEP}"
EVAL_OUT="$(sbatch --dependency="afterok:${DEP}" run_ensemble_eval.sh)"
EVAL_JID="$(echo "$EVAL_OUT" | awk '{print $NF}')"
printf "  %-45s → %s (waits for trainers)\n" "run_ensemble_eval.sh" "$EVAL_JID"

echo
echo "═══════════════════════════════════════════════════════════════════"
echo "Submitted ${#JOB_IDS[@]} trainers + 1 dependent eval."
echo "Trainer IDs: ${JOB_IDS[*]}"
echo "Eval ID:     ${EVAL_JID}"
echo "Monitor:     squeue -u \$USER"
echo "═══════════════════════════════════════════════════════════════════"
