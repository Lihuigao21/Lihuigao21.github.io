#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze all completed q-TIP4P/F runs for one method.")
    parser.add_argument("method_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dt-fs", type=float, required=True)
    parser.add_argument("--max-cm", type=float, default=4200.0)
    parser.add_argument("--window", default="gaussian")
    parser.add_argument("--window-width-ps", type=float, default=2.0)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--skip-frames", type=int, default=0)
    parser.add_argument("--skip-ps", type=float, default=0.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    parser_script = root / "scripts" / "parse_qtip4pf_dipoles.py"
    spectrum_script = root / "scripts" / "compute_ir_spectrum_mudot.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csvs = []
    for traj in sorted(args.method_dir.glob("traj_*")):
        dipoles = sorted(traj.glob("simulation.dip*"))
        if not dipoles:
            print(f"Skipping {traj}: no simulation.dip* files")
            continue
        out_csv = args.output_dir / f"{traj.name}_dipoles.csv"
        subprocess.run(
            ["python", str(parser_script), *map(str, dipoles), str(out_csv)],
            check=True,
        )
        csvs.append(out_csv)

    if not csvs:
        raise SystemExit("No trajectory dipole CSVs were produced.")

    label = args.method_dir.name
    subprocess.run(
        [
            "python",
            str(spectrum_script),
            *map(str, csvs),
            str(args.output_dir / f"{label}_mudot_spectrum.csv"),
            "--dt-fs",
            str(args.dt_fs),
            "--window",
            args.window,
            "--window-width-ps",
            str(args.window_width_ps),
            "--max-cm",
            str(args.max_cm),
            "--acf-output",
            str(args.output_dir / f"{label}_mudot_acf.csv"),
            "--block-size",
            str(args.block_size),
            "--skip-frames",
            str(args.skip_frames),
            "--skip-ps",
            str(args.skip_ps),
            "--block-output",
            str(args.output_dir / f"{label}_mudot_blocks.csv"),
            "--plot",
            str(args.output_dir / f"{label}_mudot_spectrum.png"),
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
