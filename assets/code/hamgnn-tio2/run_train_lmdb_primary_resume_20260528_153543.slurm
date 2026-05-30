#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 36:00:00
#SBATCH --job-name=TiO2_HG_resume2
#SBATCH --output=train-%j.out
#SBATCH --error=train-%j.err

set -euo pipefail
ulimit -s unlimited
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TMPDIR=/tmp/${USER}/hamgnn_tio2_resume_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

source "${HAMGNN_ENV:-/path/to/load_hamgnn_openmx.sh}"

JOB_DIR=${HAMGNN_TIO2_TRAIN_DIR:-$PWD}
GRAPH_DATA=${GRAPH_DATA_LMDB:-$JOB_DIR/graph_data/graph_data.lmdb}
CKPT=${HAMGNN_PRETRAINED_CKPT:-$JOB_DIR/checkpoints/seed.ckpt}

echo "=== TiO2 HamGNN LMDB primary resume training, no-op logger ==="
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Job name: ${SLURM_JOB_NAME:-NA}"
echo "Node list: ${SLURM_JOB_NODELIST:-NA}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NA}"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "TMPDIR=$TMPDIR"
echo "Started at: $(date)"
nvidia-smi || true
test -d "$GRAPH_DATA"
test -s "$CKPT"

cd "$JOB_DIR"
python "$JOB_DIR/run_hamgnn_no_tb.py" --config "$JOB_DIR/config_resume.yaml"

echo "Finished at: $(date)"
echo "=== Done ==="
