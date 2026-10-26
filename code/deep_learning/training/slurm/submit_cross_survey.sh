#!/bin/bash
# Submit the cross-survey generalization experiment as a SLURM array job.
# Usage: ./submit_cross_survey.sh

set -euo pipefail

PROJECT=/cluster/work/axelfh/Master
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Show survey list
echo "Cross-survey generalization experiment"
echo "======================================="
python "$PROJECT/code/deep_learning/training/survey_groups.py"
echo ""
echo "Will submit 7 jobs (one per train survey), each evaluating on all others."
echo ""

read -p "Submit 7 array jobs? [y/N] " confirm
if [[ "$confirm" != [yY] ]]; then
    echo "Aborted."
    exit 0
fi

sbatch --array=1-7 "$SCRIPT_DIR/launch_cross_survey.sh"
echo "Submitted! Monitor with: squeue -u \$USER"
