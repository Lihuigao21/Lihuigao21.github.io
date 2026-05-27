#!/usr/bin/env python3
"""Prepare corrected NVT->NVE branch calculations."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


RUN_SLURM = """#!/bin/bash
#SBATCH -p {partition}
#SBATCH -N 1
#SBATCH --ntasks-per-node {ntasks_per_node}
#SBATCH --exclusive
#SBATCH -t {time_limit}
#SBATCH --array=0-{array_max}%{array_concurrency}
#SBATCH -J ma_nve600

export I_MPI_PMI_LIBRARY=/opt/gridview/slurm/lib/libpmi2.so
ulimit -s unlimited

module purge
module load compiler/intel/2021.3.0
module load mpi/intelmpi/2021.3.0
export UCX_TLS=dc,self

ROOT_DIR="$(pwd)"
JOB_DIR="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" branches.list)"

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Job dir: $JOB_DIR"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "============================================================"

cd "$ROOT_DIR/$JOB_DIR" || exit 2
export PATH=/public/home/gaolihui/vasp:$PATH
srun --mpi=pmi2 vasp_std
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--incar", type=Path, required=True)
    parser.add_argument("--kpoints", type=Path, required=True)
    parser.add_argument("--potcar", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[3000, 3500, 4000, 4500, 5000])
    parser.add_argument("--partition", default="queue1-1")
    parser.add_argument("--array-concurrency", type=int, default=5)
    parser.add_argument("--ntasks-per-node", type=int, default=32)
    parser.add_argument("--time-limit", default="144:00:00")
    args = parser.parse_args()

    if args.outdir.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {args.outdir}")

    args.outdir.mkdir(parents=True)
    branch_names = []
    manifest = {
        "source_root": str(args.source_root),
        "steps": args.steps,
        "incar": str(args.incar),
        "kpoints": str(args.kpoints),
        "potcar": str(args.potcar),
        "partition": args.partition,
        "array_concurrency": args.array_concurrency,
        "ntasks_per_node": args.ntasks_per_node,
        "note": "POSCAR_step files contain positions only; INCAR TEBEG initializes velocities for NVE.",
    }

    for i, step in enumerate(args.steps, start=1):
        source_poscar = args.source_root / f"POSCAR_step{step}"
        if not source_poscar.exists():
            raise FileNotFoundError(source_poscar)
        branch = f"NVE{i:02d}_from_step{step}"
        branch_dir = args.outdir / branch
        branch_dir.mkdir()
        shutil.copyfile(source_poscar, branch_dir / "POSCAR")
        shutil.copyfile(args.incar, branch_dir / "INCAR")
        shutil.copyfile(args.kpoints, branch_dir / "KPOINTS")
        shutil.copyfile(args.potcar, branch_dir / "POTCAR")
        metadata = {
            "branch": branch,
            "source_poscar": str(source_poscar),
            "source_nvt_step": step,
            "encut_eV": 600,
            "ensemble": "NVE",
            "initial_velocity_policy": "VASP initialized from TEBEG=330 because POSCAR_step has no velocities",
        }
        (branch_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        branch_names.append(branch)

    (args.outdir / "branches.list").write_text("\n".join(branch_names) + "\n")
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.outdir / "run.slurm").write_text(
        RUN_SLURM.format(
            partition=args.partition,
            ntasks_per_node=args.ntasks_per_node,
            time_limit=args.time_limit,
            array_max=len(branch_names) - 1,
            array_concurrency=args.array_concurrency,
        )
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
