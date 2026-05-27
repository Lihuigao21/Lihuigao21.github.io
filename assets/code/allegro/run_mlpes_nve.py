#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read, write
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from nequip.ase import NequIPCalculator


def min_pair_distance(atoms) -> float:
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances))


def max_force_norm(atoms) -> float:
    return float(np.linalg.norm(atoms.get_forces(), axis=1).max())


def finite_or_raise(step: int, atoms) -> None:
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    if not math.isfinite(energy) or not np.isfinite(forces).all():
        raise RuntimeError(f"non-finite energy/forces at step {step}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-index", type=int, default=-1)
    parser.add_argument("--temperature-K", type=float, default=330.0)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--timestep-fs", type=float, default=0.25)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--abort-min-distance-A", type=float, default=0.5)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "nve_log.csv"
    traj_path = out_dir / "trajectory.extxyz"
    summary_path = out_dir / "nve_summary.json"
    initial_path = out_dir / "initial_structure.extxyz"

    atoms = read(args.dataset, index=args.start_index)
    write(initial_path, atoms)
    atoms.calc = NequIPCalculator.from_compiled_model(
        args.model,
        device=args.device,
        chemical_symbols=["H", "Pb", "C", "I", "N"],
    )

    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_K, force_temp=True)
    Stationary(atoms)
    ZeroRotation(atoms)
    dynamics = VelocityVerlet(atoms, timestep=args.timestep_fs * units.fs)

    records = []
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("step,time_fs,potential_eV,kinetic_eV,total_eV,temperature_K,max_force_eV_A,min_pair_A\n")

        def sample(step: int) -> None:
            finite_or_raise(step, atoms)
            potential = float(atoms.get_potential_energy())
            kinetic = float(atoms.get_kinetic_energy())
            min_dist = min_pair_distance(atoms)
            if min_dist < args.abort_min_distance_A:
                raise RuntimeError(f"minimum pair distance {min_dist:.4f} A below threshold at step {step}")
            row = {
                "step": step,
                "time_fs": step * args.timestep_fs,
                "potential_eV": potential,
                "kinetic_eV": kinetic,
                "total_eV": potential + kinetic,
                "temperature_K": float(atoms.get_temperature()),
                "max_force_eV_A": max_force_norm(atoms),
                "min_pair_A": min_dist,
            }
            records.append(row)
            handle.write(
                f"{row['step']},{row['time_fs']:.6f},{row['potential_eV']:.12f},"
                f"{row['kinetic_eV']:.12f},{row['total_eV']:.12f},"
                f"{row['temperature_K']:.6f},{row['max_force_eV_A']:.12f},"
                f"{row['min_pair_A']:.12f}\n"
            )
            handle.flush()
            write(traj_path, atoms, append=True)

        sample(0)
        for step in range(args.log_interval, args.steps + 1, args.log_interval):
            dynamics.run(args.log_interval)
            sample(step)

    def stats(key: str) -> dict:
        values = np.array([row[key] for row in records], dtype=float)
        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "first": float(values[0]),
            "last": float(values[-1]),
        }

    total0 = records[0]["total_eV"]
    total_last = records[-1]["total_eV"]
    total_span = max(row["total_eV"] for row in records) - min(row["total_eV"] for row in records)
    time_ps = args.steps * args.timestep_fs / 1000.0
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "start_index": args.start_index,
        "ensemble": "NVE",
        "steps": args.steps,
        "timestep_fs": args.timestep_fs,
        "time_ps": time_ps,
        "temperature_initial_K": args.temperature_K,
        "frames_written": len(records),
        "total_energy_drift_eV": float(total_last - total0),
        "total_energy_drift_meV_per_atom_per_ps": float((total_last - total0) * 1000.0 / len(atoms) / time_ps),
        "total_energy_span_eV": float(total_span),
        "potential_eV": stats("potential_eV"),
        "kinetic_eV": stats("kinetic_eV"),
        "total_eV": stats("total_eV"),
        "temperature_K": stats("temperature_K"),
        "max_force_eV_A": stats("max_force_eV_A"),
        "min_pair_A": stats("min_pair_A"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
