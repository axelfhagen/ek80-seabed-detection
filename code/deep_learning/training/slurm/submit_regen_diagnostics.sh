#!/bin/bash
# Discover checkpoint dirs whose diagnostics never completed and submit a Slurm
# array (one dir per task) to regenerate them with the fixed diagnostics code.
#
# Usage:
#   ./submit_regen_diagnostics.sh [include_regex] [split] [concurrency]
#
#   include_regex  egrep pattern applied to dir names (default: standard
#                  survey-split sweeps; excludes single-survey, jun2025, and
#                  all_surveys runs whose val set is not the survey split).
#   split          train/val split passed to the CLI (default: survey).
#   concurrency    max simultaneous array tasks (default: 6).
#
# A checkpoint dir qualifies if it has a *_best.pth and fewer than 10 plots in
# its diagnostics/ folder. The manifest is written next to this script; pass
# --list as the first arg to preview the selection without submitting.
set -euo pipefail

PROJECT=/cluster/work/axelfh/Master
SLURM_DIR="${PROJECT}/code/deep_learning/training/slurm"
CKPT_PARENT="${PROJECT}/code/checkpoints"
MANIFEST="${SLURM_DIR}/diag_regen_manifest.txt"

LIST_ONLY=0
if [ "${1:-}" = "--list" ]; then LIST_ONLY=1; shift; fi

# Default scope: standard survey-split sweeps. Single-survey (exp_survey_*),
# jun2025, and all_surveys runs trained on a different split and are excluded.
INCLUDE_REGEX="${1:-^exp_(arch_sweep|loss|ds|finetune|cosine|earlystop|resnet50|full_finetune|line_mask)}"
SPLIT="${2:-survey}"
CONCURRENCY="${3:-6}"

: > "$MANIFEST"
n=0
shopt -s nullglob
for d in "$CKPT_PARENT"/exp_*; do
    [ -d "$d" ] || continue
    base=$(basename "$d")
    echo "$base" | grep -Eq "$INCLUDE_REGEX" || continue
    ckpts=("$d"/*_best.pth)
    [ "${#ckpts[@]}" -gt 0 ] || continue
    pngs=("$d"/diagnostics/*.png)
    [ "${#pngs[@]}" -lt 10 ] || continue
    echo "$d" >> "$MANIFEST"
    n=$((n+1))
done
shopt -u nullglob

if [ "$n" -eq 0 ]; then
    echo "No checkpoint dirs matched (regex='${INCLUDE_REGEX}'). Nothing to do."
    rm -f "$MANIFEST"
    exit 0
fi

echo "Matched ${n} checkpoint dir(s) for regeneration (split=${SPLIT}):"
nl -ba "$MANIFEST"
echo
echo "Manifest: ${MANIFEST}"

if [ "$LIST_ONLY" -eq 1 ]; then
    echo "(--list given — not submitting)"
    exit 0
fi

echo "Submitting array 1-${n}%${CONCURRENCY} ..."
sbatch --array=1-${n}%${CONCURRENCY} \
    "${SLURM_DIR}/run_regen_diagnostics.sh" "$MANIFEST" "$SPLIT"
