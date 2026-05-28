#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


AU_DIPOLE_TO_DEBYE = 2.541746473


def read_one(path: Path) -> np.ndarray:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not rows:
        raise ValueError(f"No dipole rows found in {path}")
    return np.asarray(rows, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Average i-PI q-TIP4P/F bead dipole extras and write a CSV."
    )
    parser.add_argument("dipole_files", nargs="+", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--unit", choices=("au", "debye"), default="debye")
    args = parser.parse_args()

    arrays = [read_one(path) for path in sorted(args.dipole_files)]
    nmin = min(len(arr) for arr in arrays)
    if nmin == 0:
        raise ValueError("No common frames in dipole files")
    avg = np.mean([arr[:nmin] for arr in arrays], axis=0)
    if args.unit == "debye":
        avg = avg * AU_DIPOLE_TO_DEBYE

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "mux_D", "muy_D", "muz_D"])
        for iframe, row in enumerate(avg):
            writer.writerow([iframe, *row])
    print(f"Wrote {nmin} frames from {len(arrays)} bead files to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
