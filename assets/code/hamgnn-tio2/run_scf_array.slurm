#!/bin/bash
#SBATCH -p chu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH -t 04:00:00
#SBATCH --job-name=TiO2_scf500
#SBATCH --array=0-100%10
#SBATCH --output=/data/home/lhgao/codex_jobs/hamgnn_tio2_nve500_nac_compare_20260529_1430/slurm-scf-%A_%a.out
#SBATCH --error=/data/home/lhgao/codex_jobs/hamgnn_tio2_nve500_nac_compare_20260529_1430/slurm-scf-%A_%a.err

set -euo pipefail
ulimit -s unlimited
export UCX_TLS=dc,self
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

source /data/home/lhgao/software/load_hamgnn_openmx_4090.sh

JOB_DIR=/data/home/lhgao/codex_jobs/hamgnn_tio2_nve500_nac_compare_20260529_1430
FRAME_LIST="$JOB_DIR/scf_frame_dirs.txt"
LINE_NO=$((SLURM_ARRAY_TASK_ID + 1))
FRAME_DIR=$(sed -n "${LINE_NO}p" "$FRAME_LIST")

echo "=== TiO2 sampled-frame OpenMX SCF ==="
echo "Array job: ${SLURM_ARRAY_JOB_ID:-NA}_${SLURM_ARRAY_TASK_ID:-NA}"
echo "Node list: ${SLURM_JOB_NODELIST:-NA}"
echo "Frame dir: $FRAME_DIR"
echo "Started at: $(date)"

cd "$FRAME_DIR"
mpirun -np ${SLURM_NTASKS:-16} openmx openmx.dat > openmx.std 2> openmx.stderr
mpirun -np ${SLURM_NTASKS:-16} openmx_postprocess openmx.dat > postprocess.stdout 2> postprocess.stderr

echo "Finished at: $(date)"
echo "=== SCF frame done ==="
