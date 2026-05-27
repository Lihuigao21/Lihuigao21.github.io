#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "${script_dir}/.." && pwd)
cd "${root_dir}"

list_file=${1:-}
outdir=${2:-paper_analysis/mudot_$(date +%Y%m%d_%H%M%S)}
dt_fs=${DT_FS:-0.25}
max_cm=${MAX_CM:-4000}
window=${WINDOW:-gaussian}
window_width_ps=${WINDOW_WIDTH_PS:-2.0}
block_size=${BLOCK_SIZE:-10}

mkdir -p "${outdir}"

if [ -n "${list_file}" ]; then
  mapfile -t dipoles < "${list_file}"
else
  mapfile -t dipoles < <(
    find paper_production \
      paper_production_parallel_b01 \
      paper_production_parallel_b02 \
      paper_production_parallel_b03 \
      paper_production_parallel_b04 \
      -maxdepth 2 -name 'traj_*_dipoles.csv' 2>/dev/null | sort
  )
fi

if [ "${#dipoles[@]}" -eq 0 ]; then
  echo "No dipole files found."
  exit 1
fi

printf "%s\n" "${dipoles[@]}" > "${outdir}/dipole_file_list.txt"
echo "Using ${#dipoles[@]} dipole files."
echo "Output directory: ${outdir}"

python scripts/compute_ir_spectrum_mudot.py "${dipoles[@]}" \
  "${outdir}/spcf_mudot_${window}${window_width_ps}ps_0_${max_cm}.csv" \
  --dt-fs "${dt_fs}" \
  --window "${window}" \
  --window-width-ps "${window_width_ps}" \
  --max-cm "${max_cm}" \
  --acf-output "${outdir}/mudot_acf.csv" \
  --block-size "${block_size}" \
  --block-output "${outdir}/spcf_mudot_${window}${window_width_ps}ps_blocks_0_${max_cm}.csv" \
  --plot "${outdir}/spcf_mudot_${window}${window_width_ps}ps_block_sem_0_${max_cm}.png"

echo "Analysis finished."
