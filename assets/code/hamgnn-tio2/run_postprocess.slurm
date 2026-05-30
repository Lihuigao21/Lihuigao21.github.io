#!/bin/bash
#SBATCH -p chu
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH -t 08:00:00
#SBATCH --job-name=TiO2_nac500
#SBATCH --output=/data/home/lhgao/codex_jobs/hamgnn_tio2_nve500_nac_compare_20260529_1430/slurm-post-%j.out
#SBATCH --error=/data/home/lhgao/codex_jobs/hamgnn_tio2_nve500_nac_compare_20260529_1430/slurm-post-%j.err

set -euo pipefail
ulimit -s unlimited
export UCX_TLS=dc,self
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

source /data/home/lhgao/software/load_hamgnn_openmx_4090.sh

JOB_DIR=/data/home/lhgao/codex_jobs/hamgnn_tio2_nve500_nac_compare_20260529_1430

echo "=== TiO2 500 fs HamGNN/NAC postprocess ==="
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Node list: ${SLURM_JOB_NODELIST:-NA}"
echo "Started at: $(date)"

cd "$JOB_DIR"
expected=$(wc -l < "$JOB_DIR/scf_frame_dirs.txt")
scfout_count=$(find "$JOB_DIR/scf" -maxdepth 2 -name openmx.scfout | wc -l)
post_count=$(find "$JOB_DIR/scf" -maxdepth 2 -name postprocess.stdout | wc -l)
echo "Expected frames: $expected"
echo "openmx.scfout count: $scfout_count"
echo "postprocess.stdout count: $post_count"
if [ "$scfout_count" -ne "$expected" ] || [ "$post_count" -ne "$expected" ]; then
  echo "Not all SCF/postprocess outputs are present."
  exit 2
fi

echo "=== Generate multi-frame HamGNN graph ==="
python "$JOB_DIR/run_graph_data_safe.py" "$JOB_DIR/graph_data_gen.yaml" > "$JOB_DIR/graph_data_gen.stdout" 2> "$JOB_DIR/graph_data_gen.stderr"

echo "=== HamGNN checkpoint prediction for sampled NVE frames ==="
python "$JOB_DIR/run_hamgnn_no_tb.py" --config "$JOB_DIR/config_test_nve500.yaml" > "$JOB_DIR/hamgnn_test.stdout" 2> "$JOB_DIR/hamgnn_test.stderr"

echo "=== Compute Gamma-point NAC proxy comparison ==="
python "$JOB_DIR/compute_nac_compare.py" --job-dir "$JOB_DIR" > "$JOB_DIR/nac_compare.stdout" 2> "$JOB_DIR/nac_compare.stderr"

echo "Finished at: $(date)"
echo "=== Done ==="
