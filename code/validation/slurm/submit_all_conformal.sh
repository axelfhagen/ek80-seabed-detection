#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive overnight conformal-prediction batch.
#
# Submits every conformal analysis as an INDEPENDENT Slurm job, so a failure in
# one does not block the others. Settings are bumped up relative to the per-
# script defaults for richer visualisations. All jobs use `gpu:1` (any GPU) so
# they schedule quickly even when the A100 queue is busy with training.
#
# Coverage of the batch (each writes into its own Figures/conformal/<sub>/ dir):
#   1. base         — naive survey-split 3-method baseline (the under-coverage
#                     motivation)                              Figures/conformal/
#   2. v2           — recalibrated exchangeable-split, micro-averaged
#                                                Figures/conformal/v2_recalibrated/
#   3. scores a10   — 5 conformity-score formulations @ alpha=0.10
#                                                       Figures/conformal/scores/
#   4. scores a05   — same @ alpha=0.05 (stricter)  Figures/conformal/scores_alpha05/
#   5. adaptive     — sigma-threshold sweep + Mondrian (backscatter, slope)
#                                                     Figures/conformal/adaptive/
#   6. sigma_cmp    — global vs local transition-width sigma
#                                                 Figures/conformal/sigma_compare/
#   7. exch         — cross-survey exchangeability tests (all surveys)
#                                                Figures/conformal/exchangeability/
#   8. exch c-vs-t  — focused calibration-vs-test deployment figures (depends 7)
#   9. labelless    — repeated-crossing label-free band (pairwise)
#                                                    Figures/conformal/labelless/
#  10. labelless ad — label-free adaptive per-ping band
#                                           Figures/conformal/labelless_adaptive/
#
# Submit:  bash code/validation/slurm/submit_all_conformal.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SLURM_DIR"
FIG=/cluster/work/axelfh/Master/Figures/conformal

submit() { echo "+ sbatch $*"; sbatch "$@"; }

# 1. Naive survey-split baseline (heavy: 526 files, generous RAM/time).
submit run_conformal_base.sh

# 2. Recalibrated exchangeable-split comparison.
submit run_conformal_v2.sh

# 3. Five conformity scores @ 90% target.
submit --job-name=conformal_scores_a10 run_conformal_scores.sh \
    --n-calib 120 --n-test 120 --max-plot-files 8 --alpha 0.10
# 4. Same @ 95% target (separate output dir).
submit --job-name=conformal_scores_a05 run_conformal_scores.sh \
    --n-calib 120 --n-test 120 --max-plot-files 8 --alpha 0.05 \
    --output-dir "${FIG}/scores_alpha05"

# 5. Adaptive sigma sweep + Mondrian.
submit run_conformal_adaptive.sh \
    --n-calib 120 --n-test 120 --n-bins 6 --max-plot-files 12

# 6. Global vs local sigma.
submit run_sigma_comparison.sh --n-viz 16

# 7. Cross-survey exchangeability (all surveys, 30 files each).
EXCH_ID=$(sbatch --parsable --mem=64G --time=06:00:00 \
    run_conformal_exchangeability.sh --max-files-per-survey 30)
echo "+ exchangeability main job: ${EXCH_ID}"
# 8. Focused calibration-vs-test deployment figures, from the cache above.
submit --dependency="afterok:${EXCH_ID}" \
    run_conformal_exchangeability.sh --from-cache --calib-vs-test

# 9. Label-less conformal from repeated ferry crossings (pairwise).
submit --time=08:00:00 run_labelless_conformal.sh

# 10. Label-less, adaptive per-ping band.
submit --time=08:00:00 run_labelless_conformal.sh \
    --adaptive --sigma-source transition \
    --out-dir "${FIG}/labelless_adaptive"

echo ""
echo "All conformal jobs submitted. Track with:  squeue -u $USER"
