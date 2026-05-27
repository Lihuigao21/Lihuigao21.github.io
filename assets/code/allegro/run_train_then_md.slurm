#!/bin/bash
#SBATCH -p 8-4090
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -J allegro_formal_md

set -euo pipefail

ulimit -s unlimited
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

ENV_PY=/data/home/lhgao/.conda/envs/allergo/bin/python
NEQUIP_TRAIN=/data/home/lhgao/.conda/envs/allergo/bin/nequip-train
NEQUIP_COMPILE=/data/home/lhgao/.conda/envs/allergo/bin/nequip-compile
NEQUIP_PACKAGE=/data/home/lhgao/.conda/envs/allergo/bin/nequip-package
DATASET=/data/home/lhgao/allegro-main/test/MAPBI3/nve5_encut600_stride15_discard100.extxyz
TB_DIR=outputs/tensorboard_logs/lightning_logs/formal_nve970_20260527
MD_DIR=mlpes_md_nvt330_5ps

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of processors: $SLURM_NTASKS"
echo "Task is running on the following nodes:"
echo "$SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo "============================================================"

echo "[1/4] Formal Allegro training"
srun "$NEQUIP_TRAIN" -cn formal_nve970.yaml

echo "[2/4] Locate best checkpoint"
BEST_CKPT=$(ls -t "$TB_DIR"/checkpoints/best_epoch_*.ckpt | head -n 1)
echo "BEST_CKPT=$BEST_CKPT"

echo "[3/4] Package and compile for ASE"
mkdir -p "$MD_DIR"
"$NEQUIP_PACKAGE" build "$BEST_CKPT" "$MD_DIR/formal_nve970.nequip.zip"
"$NEQUIP_COMPILE" "$BEST_CKPT" "$MD_DIR/formal_nve970_ase_cuda.nequip.pth" \
    --device cuda \
    --mode torchscript \
    --target ase \
    --data-path "$DATASET" \
    --num-frames static \
    --num-nodes static

echo "[4/4] Run ASE/MLPES NVT dynamics smoke"
srun "$ENV_PY" run_mlpes_md.py \
    --model "$MD_DIR/formal_nve970_ase_cuda.nequip.pth" \
    --dataset "$DATASET" \
    --output-dir "$MD_DIR" \
    --start-index -1 \
    --temperature-K 330 \
    --steps 5000 \
    --timestep-fs 1.0 \
    --log-interval 10 \
    --device cuda

echo "Done."
