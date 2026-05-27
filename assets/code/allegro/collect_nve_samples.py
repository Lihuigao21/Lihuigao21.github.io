#!/usr/bin/env python3
"""Collect sampled frames from corrected NVE branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ase.io import read, write


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--discard", type=int, default=100)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    branches = [line.strip() for line in (args.job_root / "branches.list").read_text().splitlines() if line.strip()]
    all_atoms = []
    status = []
    for branch in branches:
        branch_dir = args.job_root / branch
        outcar = branch_dir / "OUTCAR"
        meta_path = branch_dir / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        entry = {"branch": branch, "outcar": str(outcar), "exists": outcar.exists(), "sampled": 0}
        if not outcar.exists():
            if not args.allow_missing:
                raise FileNotFoundError(outcar)
            status.append(entry)
            continue
        atoms_list = read(outcar, index=f"{args.discard}::{args.stride}")
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        for local_i, atoms in enumerate(atoms_list):
            atoms.info["nve_branch"] = branch
            atoms.info["source_nvt_step"] = meta.get("source_nvt_step")
            atoms.info["nve_sample_local_index"] = local_i
            atoms.info["nve_discard"] = args.discard
            atoms.info["nve_stride"] = args.stride
            atoms.info["dft_encut_eV"] = 600
            atoms.info["dft_sampling"] = "NVT_to_NVE_stride"
            all_atoms.append(atoms)
        entry["sampled"] = len(atoms_list)
        status.append(entry)

    if all_atoms:
        write(args.output, all_atoms, format="extxyz")
    report = {
        "job_root": str(args.job_root),
        "output": str(args.output),
        "discard": args.discard,
        "stride": args.stride,
        "branches": len(branches),
        "frames": len(all_atoms),
        "status": status,
    }
    report_path = args.output.with_suffix(".collect_report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "status"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
