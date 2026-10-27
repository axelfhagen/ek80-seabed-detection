#!/bin/bash
# Submit one SLURM job per day-folder that contains .raw files.
# Usage: bash submit_convert_all.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/cluster/work/axelfh}"
RAW_ROOT="${DATA_ROOT}/Data_raw/EK80"

count=0
for day_dir in "${RAW_ROOT}"/*/D*/; do
    [ -d "$day_dir" ] || continue
    n_raw=$(find "$day_dir" -maxdepth 1 -name "*.raw" | wc -l)
    [ "$n_raw" -eq 0 ] && continue

    day_name=$(basename "$day_dir")
    sbatch --job-name="r2z_${day_name}" \
           --export=RAW_DIR="$day_dir" \
           "${SCRIPT_DIR}/convert_raw_to_zarr.sh"
    echo "  Submitted ${day_name} (${n_raw} raw files)"
    count=$((count + 1))
done

echo ""
echo "Submitted ${count} jobs total."
