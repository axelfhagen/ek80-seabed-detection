#!/bin/bash
# Submit one job per raw file for the 3 failed June 2025 dates.
# One process per file avoids memory accumulation across files.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="/cluster/work/axelfh/Data_raw/EK80/D202506"
ZARR_OUT="/cluster/work/axelfh/Master/Datasets/echopype_zarr"

count=0
for day in D20250624 D20250625 D20250626; do
    for raw_file in "${DATA_ROOT}/${day}"/*.raw; do
        [ -f "$raw_file" ] || continue
        stem=$(basename "$raw_file" .raw)

        if [ -d "${ZARR_OUT}/${stem}.zarr" ]; then
            echo "  SKIP ${stem} (zarr exists)"
            continue
        fi

        sbatch --job-name="r2z_${stem}" \
               --export=RAW_FILE="$raw_file" \
               "${SCRIPT_DIR}/convert_single_raw.sh"
        echo "  Submitted ${stem}"
        count=$((count + 1))
    done
done

echo ""
echo "Submitted ${count} jobs total."
