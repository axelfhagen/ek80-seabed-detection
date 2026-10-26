#!/bin/bash
# Submit an experiment grid as a Slurm array job.
# Usage: ./submit_sweep.sh <grid_name> [max_concurrent]
#
# Examples:
#   ./submit_sweep.sh loss_sweep       # up to 10 concurrent jobs
#   ./submit_sweep.sh lr_sweep 5       # up to 5 concurrent jobs

set -euo pipefail

GRID="${1:?Usage: submit_sweep.sh <grid_name> [max_concurrent]}"
MAX_CONCURRENT="${2:-10}"
PROJECT=/cluster/work/axelfh/Master
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

N=$(python "$PROJECT/code/deep_learning/training/slurm/experiment_grid.py" \
    --grid "$GRID" --count)

echo "Grid '$GRID': $N configs, max $MAX_CONCURRENT concurrent"
echo ""
python "$PROJECT/code/deep_learning/training/slurm/experiment_grid.py" \
    --grid "$GRID" --list
echo ""

read -p "Submit $N jobs? [y/N] " confirm
if [[ "$confirm" != [yY] ]]; then
    echo "Aborted."
    exit 0
fi

sbatch --array="1-${N}%${MAX_CONCURRENT}" \
    "$SCRIPT_DIR/launch_sweep.sh" "$GRID"
