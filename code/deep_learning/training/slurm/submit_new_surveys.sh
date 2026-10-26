#!/bin/bash
# Submit per-survey training jobs and the all-surveys combined run.
# Usage: bash submit_new_surveys.sh [--skip-all]
#
# Submits 7 per-survey jobs (array, max 3 concurrent) + 1 all-surveys job.
# Pass --skip-all to only submit the per-survey array.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Per-survey runs (9 surveys, max 3 concurrent):"
echo "  1=dec_2023  2=jan_2024  3=apr_2024  4=jul_2024  5=sep_2024"
echo "  6=oct_2024  7=jan_2025  8=feb_2025  9=jun_2025"
echo ""

sbatch --array=1-9%3 "${SCRIPT_DIR}/run_per_survey.sh"

if [[ "${1:-}" != "--skip-all" ]]; then
    echo ""
    echo "All-surveys combined run (24h time limit):"
    sbatch "${SCRIPT_DIR}/run_all_surveys.sh"
fi

echo ""
echo "Monitor with: squeue -u \$USER"
