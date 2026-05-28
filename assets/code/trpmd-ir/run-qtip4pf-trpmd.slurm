#!/bin/bash
#SBATCH -p chu
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH -t 144:00:00
#SBATCH -J qtip_trp_001
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail
ulimit -s unlimited

export PATH="/data/home/lhgao/.conda/envs/deepmd/bin:${PATH}"
export PYTHONNOUSERSITE=1

ROOT="/data/home/lhgao/codex_jobs/trpmd_qtip4pf_fig7"
DRIVER="${ROOT}/driver_src/driver.x"
IPI_EXE="/data/home/lhgao/.conda/envs/deepmd/bin/i-pi"
RUN_TAG="production_trpmd_lambda_0p5_traj_001"
NCLIENTS="32"
SOCKET="qtip4pf_${SLURM_JOB_ID}_${RUN_TAG}"
IPI_INPUT="input.${SLURM_JOB_ID}.xml"
IPI_LOG="ipi.${SLURM_JOB_ID}.log"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Job name: ${SLURM_JOB_NAME}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Run dir: $(pwd)"
echo "Socket: ${SOCKET}"
echo "NCLIENTS: ${NCLIENTS}"
echo "Driver: ${DRIVER}"

if [ ! -x "${DRIVER}" ]; then
  echo "Missing executable driver: ${DRIVER}"
  exit 1
fi

python "${ROOT}/scripts/render_input.py" input.template.xml "${IPI_INPUT}" "${SOCKET}"

"${IPI_EXE}" "${IPI_INPUT}" > "${IPI_LOG}" 2>&1 &
ipi_pid=$!

for attempt in $(seq 1 90); do
  if [ -S "/tmp/ipi_${SOCKET}" ]; then
    echo "i-PI unix socket is ready after ${attempt} checks."
    break
  fi
  if ! kill -0 "${ipi_pid}" 2>/dev/null; then
    echo "i-PI exited before opening the socket."
    cat "${IPI_LOG}" || true
    exit 1
  fi
  sleep 2
  if [ "${attempt}" -eq 90 ]; then
    echo "Timed out waiting for i-PI unix socket."
    cat "${IPI_LOG}" || true
    exit 1
  fi
done

set +e
pids=()
labels=()
for client in $(seq 1 "${NCLIENTS}"); do
  label=$(printf "%03d" "${client}")
  srun --exclusive --mpi=pmi2 -N 1 -n 1 "${DRIVER}" -u -h "${SOCKET}" -m qtip4pf     > "driver.${SLURM_JOB_ID}.${label}.log" 2>&1 &
  pids+=("$!")
  labels+=("${label}")
  sleep 0.1
done

wait "${ipi_pid}"
ipi_status=$?

driver_bad=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  label="${labels[$idx]}"
  wait "${pid}"
  status=$?
  if [ "${status}" -ne 0 ]; then
    if grep -q "EXIT" "driver.${SLURM_JOB_ID}.${label}.log" 2>/dev/null; then
      echo "Driver ${label} reported i-PI EXIT; treating as normal shutdown."
    elif [ "${ipi_status}" -eq 0 ]; then
      echo "Driver ${label} exited with status ${status} after successful i-PI completion."
    else
      echo "Driver ${label} failed with status ${status}."
      driver_bad=1
    fi
  fi
done
set -e

echo "i-PI status: ${ipi_status}"
echo "Driver aggregate status: ${driver_bad}"

if [ "${ipi_status}" -ne 0 ] || [ "${driver_bad}" -ne 0 ]; then
  echo "===== i-PI log tail ====="
  tail -120 "${IPI_LOG}" || true
  echo "===== driver log tails ====="
  tail -40 driver.${SLURM_JOB_ID}.*.log 2>/dev/null || true
  exit 1
fi

echo "Run finished successfully."
