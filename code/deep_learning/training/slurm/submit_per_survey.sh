#!/bin/bash
# Submit each survey as an independent SLURM job (no array).
# Usage: bash submit_per_survey.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SURVEYS=(dec_2023 jan_2024 apr_2024 jul_2024 sep_2024 oct_2024 jan_2025 feb_2025 jun_2025)

for i in "${!SURVEYS[@]}"; do
    TASK_ID=$((i + 1))
    echo "Submitting survey ${SURVEYS[$i]} (task_id=$TASK_ID)..."
    sbatch --export=ALL,FORCE_TASK_ID=$TASK_ID "$SCRIPT_DIR/run_per_survey.sh"
done
