#!/bin/bash
#SBATCH -p 8-4090
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 36:00:00
#SBATCH --job-name=TiO2_HG_resume2
#SBATCH --output=/data/home/lhgao/codex_jobs/hamgnn_tio2_train_lmdb_primary_resume_20260528_153543/train-%j.out
#SBATCH --error=/data/home/lhgao/codex_jobs/hamgnn_tio2_train_lmdb_primary_resume_20260528_153543/train-%j.err

set -euo pipefail
ulimit -s unlimited
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TMPDIR=/tmp/${USER}/hamgnn_tio2_resume_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

source /data/home/lhgao/software/load_hamgnn_openmx_4090.sh

JOB_DIR=/data/home/lhgao/codex_jobs/hamgnn_tio2_train_lmdb_primary_resume_20260528_153543
GRAPH_DATA=/data/home/lhgao/codex_jobs/hamgnn_tio2_npz_to_lmdb_20260528_080312/graph_data/graph_data.lmdb
CKPT=/data/home/lhgao/codex_jobs/hamgnn_tio2_train_lmdb_primary_20260528_083744/train_output/version_0/checkpoints/epoch=31-val_loss=0.000000.ckpt

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
