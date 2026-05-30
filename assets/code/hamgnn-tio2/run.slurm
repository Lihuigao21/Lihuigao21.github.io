#!/bin/bash
#SBATCH -p chu
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH -t 12:00:00
#SBATCH --job-name=TiO2_nvt_cmp_200fs
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
ulimit -s unlimited
export UCX_TLS=dc,self
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

source "${HAMGNN_ENV:-/path/to/load_hamgnn_openmx.sh}"

JOB_DIR=${HAMGNN_TIO2_WORKDIR:-$PWD}

echo "=== TiO2 NVT probe and OpenMX/HamGNN band comparison ==="
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Job name: ${SLURM_JOB_NAME:-NA}"
echo "Node list: ${SLURM_JOB_NODELIST:-NA}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-NA}"
echo "Started at: $(date)"

cd "$JOB_DIR"
python "$JOB_DIR/prepare_probe_inputs.py" --stage initial --job-dir "$JOB_DIR"

echo "=== OpenMX NVT_VS: 200 steps, 1 fs, explicit new 300 K initial velocities ==="
cd "$JOB_DIR/nvt"
mpirun -np ${SLURM_NTASKS:-32} openmx TiO2_probe_nvt.dat > TiO2_probe_nvt.std 2> TiO2_probe_nvt.stderr

echo "=== Prepare final NVT frame for static SCF ==="
cd "$JOB_DIR"
python "$JOB_DIR/prepare_probe_inputs.py" --stage final --job-dir "$JOB_DIR"
python "$JOB_DIR/geometry_nearest.py" \
  --query-poscar "$JOB_DIR/scf/POSCAR" \
  --out-json "$JOB_DIR/geometry_nearest_train.json" \
  > "$JOB_DIR/geometry_nearest_train.stdout" 2> "$JOB_DIR/geometry_nearest_train.stderr"

echo "=== OpenMX static SCF for final NVT frame ==="
cd "$JOB_DIR/scf"
mpirun -np ${SLURM_NTASKS:-32} openmx openmx.dat > openmx.std 2> openmx.stderr
mpirun -np ${SLURM_NTASKS:-32} openmx_postprocess openmx.dat > postprocess.stdout 2> postprocess.stderr

echo "=== Generate single-structure HamGNN graph ==="
cd "$JOB_DIR"
python "$JOB_DIR/run_graph_data_safe.py" "$JOB_DIR/graph_data_gen.yaml" > "$JOB_DIR/graph_data_gen.stdout" 2> "$JOB_DIR/graph_data_gen.stderr"

echo "=== HamGNN checkpoint prediction for the probe graph ==="
python "$JOB_DIR/run_hamgnn_no_tb.py" --config "$JOB_DIR/config_test_one.yaml" > "$JOB_DIR/hamgnn_test.stdout" 2> "$JOB_DIR/hamgnn_test.stderr"

echo "=== Plot OpenMX vs HamGNN band comparison ==="
python "$JOB_DIR/plot_band_compare.py" --job-dir "$JOB_DIR" --nk 80 > "$JOB_DIR/band_compare.stdout" 2> "$JOB_DIR/band_compare.stderr"

echo "Finished at: $(date)"
echo "=== Done ==="
