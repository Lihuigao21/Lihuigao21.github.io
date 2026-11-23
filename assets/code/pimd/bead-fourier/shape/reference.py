"""Independent quantum and finite-action equilibria for the quadratic–quartic family."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import json
import itertools
import math
import os
from pathlib import Path
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.special import roots_hermitenorm


HERE = Path(__file__).resolve().parent
METRICS = ("x2", "x4", "energy")


def uniform_grid(box: float, max_dx: float) -> tuple[np.ndarray, float]:
    half_steps = math.ceil(box / max_dx)
    return np.linspace(-box, box, 2 * half_steps + 1), box / half_steps


@lru_cache(maxsize=16)
def gaussian_rule(rank: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    if rank == 0:
        return np.zeros((1, 0)), np.ones(1)
    abscissae, weights = roots_hermitenorm(order)
    weights = weights / math.sqrt(2 * math.pi)
    nodes = np.array(list(itertools.product(abscissae, repeat=rank)))
    products = np.prod(np.array(list(itertools.product(weights, repeat=rank))), axis=1)
    return nodes, products


def write_exclusive(path: Path, content: dict) -> None:
    payload = json.dumps(content, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def bulk_quantum(shape: float, reduced_beta: float, *, box: float, dx: float, states: int) -> dict:
    grid, dx = uniform_grid(box, dx)
    x = grid[1:-1]
    squared = x * x
    potential = (1 - shape) * squared / 2 + shape * squared**2 / 4
    eigenvalues, wavefunctions = eigh_tridiagonal(
        dx**-2 + potential,
        np.full(x.size - 1, -0.5 / dx**2),
        select="i",
        select_range=(0, states - 1),
    )
    gap = float(eigenvalues[1] - eigenvalues[0])
    beta = reduced_beta / gap
    weights = np.exp(-beta * (eigenvalues - eigenvalues[0]))
    weights /= np.sum(weights)
    density = wavefunctions**2 @ weights
    x2 = float(density @ squared)
    x4 = float(density @ squared**2)
    mean_curvature = 1.0 - shape + 3.0 * shape * x2
    curvature_cv = 3.0 * shape * math.sqrt(max(0.0, x4 - x2**2)) / mean_curvature
    return {
        "shape": shape,
        "reduced_beta": reduced_beta,
        "beta": beta,
        "gap": gap,
        "mean_curvature": mean_curvature,
        "curvature_cv": curvature_cv,
        "x2": x2,
        "x4": x4,
        "energy": float(weights @ eigenvalues),
        "virial_energy": (1 - shape) * x2 + 0.75 * shape * x4,
        "dx": dx,
        "box": box,
        "states": states,
        "last_state_weight": float(weights[-1]),
        "normalization_error": float(abs(np.sum(density) - 1)),
    }


def relative_max(first: dict, second: dict, keys=METRICS) -> float:
    return float(max(abs(first[name] / second[name] - 1) for name in keys))


@lru_cache(maxsize=5)
def gauss_legendre(points: int) -> tuple[np.ndarray, np.ndarray]:
    if points == 0:
        return np.array([0.0]), np.array([1.0])
    x, w = leggauss(points)
    return (x + 1.0) / 2.0, w / 2.0


def bridge_basis(epsilon: float, modes: int, sites: np.ndarray) -> np.ndarray:
    kpi = np.pi * np.arange(1, modes + 1)
    return math.sqrt(2 * epsilon) * np.sin(sites[:, None] * kpi) / kpi


def path_kernel(x: np.ndarray, y: np.ndarray, shape: float, epsilon: float, modes: int, points: int, order: int) -> tuple[np.ndarray, int]:
    c2, c4 = (1 - shape) / 2, shape / 4
    if points == 0:
        x2, y2 = x * x, y * y
        p2 = (x2 + y2) / 2
        p4 = (x2**2 + y2**2) / 2
        weight = np.exp(-epsilon * (c2 * p2 + c4 * p4))
        return np.column_stack((weight, weight * p2, weight * p4)), 0

    sites, weights = gauss_legendre(points)
    bridge = bridge_basis(epsilon, modes, sites)
    u, singular, _ = np.linalg.svd(bridge, full_matrices=False)
    rank = int(np.sum(singular > 1e-13 * singular[0]))
    bridge = u[:, :rank] * singular[:rank]
    linear_path = x[:, None] * (1 - sites) + y[:, None] * sites

    # Integrate a positive harmonic reference exactly. The residual
    # GH integral is smoother at the largest epsilon and remains exact
    # in the GH order limit. The quartic residual exponent is bounded.
    c2_ref = c2 + math.sqrt(c4 / epsilon)
    metric = np.eye(rank) + 2 * epsilon * c2_ref * (bridge.T @ (weights[:, None] * bridge))
    covariance = np.linalg.inv(metric)
    negative_gradient = -2 * epsilon * c2_ref * (linear_path @ (weights[:, None] * bridge))
    shifted_mean = negative_gradient @ covariance
    conditioned_path = linear_path + shifted_mean @ bridge.T
    normalizing_log = (
        -epsilon * c2_ref * (linear_path**2 @ weights)
        + 0.5 * np.sum(shifted_mean * negative_gradient, axis=1)
        - 0.5 * np.linalg.slogdet(metric)[1]
    )
    nodes, gh_weights = gaussian_rule(rank, order)
    fluctuation = nodes @ np.linalg.cholesky(covariance).T @ bridge.T
    chunk_length = max(1, 2**18 // nodes.shape[0])
    result = np.empty((x.size, 3))
    for start in range(0, x.size, chunk_length):
        chunk = slice(start, start + chunk_length)
        second = np.zeros((conditioned_path[chunk].shape[0], nodes.shape[0]))
        fourth = np.zeros_like(second)
        for i, weight in enumerate(weights):
            squared = (conditioned_path[chunk, i, None] + fluctuation[:, i]) ** 2
            second += weight * squared
            fourth += weight * squared**2
        probability = np.exp(epsilon * ((c2_ref - c2) * second - c4 * fourth)) * gh_weights
        result[chunk, 0] = np.sum(probability, axis=1)
        result[chunk, 1] = np.einsum("ij,ij->i", probability, second, optimize=False)
        result[chunk, 2] = np.einsum("ij,ij->i", probability, fourth, optimize=False)
    return result * np.exp(normalizing_log)[:, None], rank


def format_moments(x2: float, x4: float, shape: float) -> dict:
    return {"x2": float(x2), "x4": float(x4), "energy": (1 - shape) * float(x2) + 0.75 * shape * float(x4)}


def transfer(shape: float, beta: float, beads: int, modes: int, points: int, *, box: float, dx: float, order: int) -> dict:
    q, dx = uniform_grid(box, dx)
    endpoint = q.size - 1
    i, j = np.triu_indices(q.size)
    keep = i + j <= endpoint
    i, j = i[keep], j[keep]
    epsilon = beta / beads
    prefactor = dx / math.sqrt(2 * math.pi * epsilon) * np.exp(-(q[i] - q[j])**2 / (2 * epsilon))
    entries, rank = path_kernel(q[i], q[j], shape, epsilon, modes, points, order)

    def matrix(column: int) -> np.ndarray:
        full = np.empty((q.size, q.size))
        data = prefactor * entries[:, column]
        full[i, j] = full[j, i] = full[endpoint - i, endpoint - j] = full[endpoint - j, endpoint - i] = data
        return full

    kernel = matrix(0)
    values, vectors = eigh(kernel, check_finite=False)
    normalized = values / values[-1]
    z = np.sum(normalized**beads)
    density = (vectors * vectors) @ (normalized**beads / z)
    insertion_weight = ((vectors * normalized**(beads - 1)) @ vectors.T) / (z * values[-1])
    return {
        "bead": format_moments(density @ q**2, density @ q**4, shape),
        "path": format_moments(np.sum(insertion_weight * matrix(1)), np.sum(insertion_weight * matrix(2)), shape),
        "normalization_error": float(max(abs(density.sum() - 1), abs(np.sum(insertion_weight * kernel) - 1))),
        "dx": dx,
        "box": box,
        "hermite_order": order if rank else 0,
        "bridge_rank": rank,
        "smallest_relative_eigenvalue": float(normalized[0]),
    }


def harmonic_blocks(beta: float, beads: int, modes: int, points: int) -> dict:
    sites, weights = gauss_legendre(points)
    epsilon = beta / beads
    basis = bridge_basis(epsilon, modes, sites)
    bead_variance = 0.0
    path_variance = np.zeros(sites.size)
    for theta in 2 * np.pi * np.arange(beads) / beads:
        design = np.column_stack((1 - sites + sites * np.exp(1j * theta), basis))
        free = np.eye(modes + 1)
        free[0, 0] = 4 * np.sin(theta / 2)**2 / epsilon
        precision = free + epsilon * (design.conj().T @ (weights[:, None] * design))
        covariance = np.linalg.inv(precision)
        bead_variance += covariance[0, 0].real / beads
        path_variance += np.einsum("ij,jk,ik->i", design, covariance, design.conj()).real / beads
    return {
        "bead": format_moments(bead_variance, 3 * bead_variance**2, 0.0),
        "path": format_moments(path_variance @ weights, 3 * path_variance**2 @ weights, 0.0),
        "method": "exact finite-action harmonic Fourier blocks",
    }


def build_slice(job: dict) -> tuple[Path, dict]:
    shape = job["shape"]
    reduced_beta = job["reduced_beta"]
    start = time.perf_counter()
    quantum = bulk_quantum(shape, reduced_beta, box=10, dx=0.0025, states=80)
    quantum_controls = {
        "dx": relative_max(bulk_quantum(shape, reduced_beta, box=10, dx=0.005, states=80), quantum, (*METRICS, "gap")),
        "box": relative_max(bulk_quantum(shape, reduced_beta, box=8.5, dx=0.0025, states=80), quantum, (*METRICS, "gap")),
        "states": relative_max(bulk_quantum(shape, reduced_beta, box=10, dx=0.0025, states=60), quantum, (*METRICS, "gap")),
        "virial": abs(quantum["virial_energy"] / quantum["energy"] - 1),
    }
    beta = quantum["beta"]
    if shape == 0:
        exact = 0.5 / np.tanh(beta / 2)
        quantum_controls["analytic_harmonic"] = max(abs(quantum["gap"] - 1), abs(quantum["x2"] / exact - 1), abs(quantum["x4"] / (3 * exact**2) - 1), abs(quantum["energy"] / exact - 1))

    rows = []
    tolerance = job["tolerance"]
    for scheme in job["schemes"]:
        modes, points = scheme["modes"], scheme["points"]
        for beads in job["beads"]:
            row = {"shape": shape, "reduced_beta": reduced_beta, "beta": beta, "beads": beads, "modes": modes, "points": points}
            if shape == 0:
                selected = harmonic_blocks(beta, beads, modes, points)
                change = 2e-12
                if beads in (2, 8, 16, 32):
                    control = transfer(shape, beta, beads, modes, points, box=8.5, dx=0.085, order=12)
                    change = max(relative_max(control[readout], selected[readout]) for readout in ("bead", "path"))
                    selected["normalization_error"] = control["normalization_error"]
                    selected["transfer_control"] = {key: control[key] for key in ("dx", "box", "hermite_order", "bridge_rank")}
            else:
                low_box = 7.5 if reduced_beta <= 0.5 else 7.0
                levels = [(low_box, 0.125, 10), (low_box + 1.0, 0.085, 16), (low_box + 1.5, 0.065, 24), (low_box + 2.0, 0.05, 36), (low_box + 2.5, 0.045, 52)]
                changes = []
                smaller_result = None
                for box, dx, order in levels:
                    selected = transfer(shape, beta, beads, modes, points, box=box, dx=dx, order=order)
                    if smaller_result is not None:
                        change = max(relative_max(selected[readout], smaller_result[readout]) for readout in ("bead", "path"))
                        changes.append(change)
                        if change < tolerance / 8:
                            break
                    smaller_result = selected
                if (shape, reduced_beta, beads, modes, points) == (1, 16, 1, 1, 4):
                    selected = transfer(shape, beta, beads, modes, points, box=10.5, dx=0.035, order=144)
                    controls = [
                        transfer(shape, beta, beads, modes, points, box=box, dx=dx, order=order)
                        for box, dx, order in ((10.5, 0.035, 96), (10.5, 0.035, 112), (10.5, 0.047, 144), (9.5, 0.035, 144))
                    ]
                    change = max(relative_max(item[readout], selected[readout]) for item in controls for readout in ("bead", "path"))
                    changes.append(change)
                selected["refinement_path"] = changes
            row.update(selected, relative_numerical_change=change)
            row["accepted"] = change < tolerance and row.get("normalization_error", 0.0) < 1e-10
            rows.append(row)

    accepted = max(quantum_controls.values()) < tolerance and quantum["normalization_error"] < 1e-10 and all(row["accepted"] for row in rows)
    receipt = {
        "provenance": job["provenance"],
        "quantum": quantum,
        "rows": rows,
        "validation": {"accepted": accepted, "quantum_controls": quantum_controls, "max_transfer_relative_change": max(row["relative_numerical_change"] for row in rows)},
        "wall_seconds": time.perf_counter() - start,
    }
    output = job["output"]
    write_exclusive(output, receipt)
    return output, receipt


def cached_slice(job: dict) -> tuple[Path, dict] | None:
    path = job["output"]
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {(n, scheme["modes"], scheme["points"]) for n in job["beads"] for scheme in job["schemes"]}
    data = value["rows"]
    obtained = {(row["beads"], row["modes"], row["points"]) for row in data}
    if value["provenance"] != job["provenance"] or expected != obtained or len(expected) != len(data):
        raise ValueError(f"Stored finite-action slice has a different identity: {path}")
    return path, value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slice-directory", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--anchor", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if args.slice_directory is None:
        args.slice_directory = args.output.parent / (args.output.stem + "_slices")
    args.slice_directory.mkdir(parents=True, exist_ok=True)
    provenance = {"code": Path(__file__).name, "protocol": args.protocol.name}
    jobs = []
    for beta in sorted(protocol["reduced_beta"], key=lambda value: value != 8):
        beads = [n for n in protocol["bead_grid"] if n <= protocol["max_beads_by_reduced_beta"][str(beta)]]
        for shape in sorted(protocol["shape"], key=lambda value: value not in (0, 1)):
            if args.anchor and (beta != 8 or shape not in (0, 1)):
                continue
            jobs.append({
                "shape": shape,
                "reduced_beta": beta,
                "beads": beads,
                "schemes": protocol["schemes"],
                "tolerance": protocol["acceptance"]["numerical_reference_relative_tolerance"],
                "output": args.slice_directory / f"shape_{shape}_reduced_beta_{beta}.json",
                "provenance": provenance,
            })
    start = time.perf_counter()
    completed = []
    remaining = []
    for job in jobs:
        cached = cached_slice(job)
        if cached is None:
            remaining.append(job)
        else:
            completed.append(cached)
    # Windows subprocesses spawn afresh. Keep each worker's BLAS serial.
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build_slice, job) for job in remaining]
        for future in as_completed(futures):
            path, result = future.result()
            completed.append((path, result))
            val = result["validation"]
            print(f"{path.name}: {len(result['rows'])} cells, accepted={val['accepted']}, max relative refinement {val['max_transfer_relative_change']:.3g}, {result['wall_seconds']:.1f}s", flush=True)

    results = [data for _, data in completed]
    quantum = sorted((data["quantum"] for data in results), key=lambda row: (row["reduced_beta"], row["shape"]))
    rows = sorted((row for data in results for row in data["rows"]), key=lambda row: (row["reduced_beta"], row["shape"], row["modes"], row["points"], row["beads"]))
    valid = len(results) == len(jobs) and all(data["validation"]["accepted"] for data in results)
    receipt = {
        "provenance": provenance,
        "quantum": quantum,
        "rows": rows,
        "validation": {
            "accepted": valid,
            "max_transfer_relative_change": max(data["validation"]["max_transfer_relative_change"] for data in results),
            "max_quantum_relative_change": max(max(data["validation"]["quantum_controls"].values()) for data in results),
            "slices": [{"file": path.name, **data["validation"]} for path, data in sorted(completed)],
        },
        "wall_seconds": time.perf_counter() - start,
    }
    write_exclusive(args.output, receipt)
    print(f"{len(rows)} finite-action cells, acceptance={valid}, {receipt['wall_seconds']:.1f}s", flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
