#!/bin/bash
#SBATCH -p chu
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH -t 12:00:00
#SBATCH --job-name=TiO2_nve500
#SBATCH --output=/data/home/lhgao/codex_jobs/hamgnn_tio2_nve500_nac_compare_20260529_1430/slurm-nve-%j.out
#SBATCH --error=/data/home/lhgao/codex_jobs/hamgnn_tio2_nve500_nac_compare_20260529_1430/slurm-nve-%j.err

set -euo pipefail
ulimit -s unlimited
export UCX_TLS=dc,self
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

source /data/home/lhgao/software/load_hamgnn_openmx_4090.sh

JOB_DIR=/data/home/lhgao/codex_jobs/hamgnn_tio2_nve500_nac_compare_20260529_1430

echo "=== TiO2 500 fs NVE preparation ==="
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Node list: ${SLURM_JOB_NODELIST:-NA}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-NA}"
echo "Started at: $(date)"

cd "$JOB_DIR"
python "$JOB_DIR/prepare_nve500_nac_inputs.py" --stage initial --job-dir "$JOB_DIR"

echo "=== OpenMX NVE: 501 frames, 1 fs, starting from 200 fs NVT endpoint ==="
cd "$JOB_DIR/nve"
mpirun -np ${SLURM_NTASKS:-32} openmx TiO2_nve500.dat > TiO2_nve500.std 2> TiO2_nve500.stderr

echo "=== Prepare sampled SCF frames: 0-500 fs every 5 fs ==="
cd "$JOB_DIR"
python "$JOB_DIR/prepare_nve500_nac_inputs.py" --stage scf --job-dir "$JOB_DIR"

echo "Finished at: $(date)"
echo "=== NVE done ==="
