"""Test first-cumulant free-bridge tail averaging at fixed BF quadrature."""

from __future__ import annotations

from functools import lru_cache
import argparse
import hashlib
import json
from math import ceil, pi, sqrt
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import eigh
from scipy.special import roots_hermitenorm, roots_legendre


WORK = Path(__file__).resolve().parent
PROTOCOL_PATH = WORK / "protocol.json"
OBSERVABLES = ("x2", "energy")
VARIANTS = ("plain", "partial_average")
SPECTRAL_SIZES = (40, 64, 96)
FINITE_ACTION_GRIDS = {
    "fine": (6.5, 0.05, 96),
    "dx": (6.5, 0.075, 96),
    "box": (6.0, 0.05, 96),
    "hermite_order": (6.5, 0.05, 64),
}


def observables(shape: float, x2: float, x4: float) -> dict[str, float]:
    return {
        "x2": float(x2),
        "x4": float(x4),
        "energy": float((1 - shape) * x2 + 0.75 * shape * x4),
    }


def continuum_reference(shape: float, reduced_beta: float, size: int) -> dict:
    # Four extra states retain all terms when projecting x^4 into the basis.
    off_diagonal = np.sqrt(np.arange(1, size + 4) / 2)
    x_op = np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    x2_full = x_op @ x_op
    x4_full = x2_full @ x2_full
    x2_proj = x2_full[:size, :size]
    x4_proj = x4_full[:size, :size]
    ham = np.diag(np.arange(size) + 0.5) + shape * (x4_proj / 4 - x2_proj / 2)
    energies, states = eigh(ham, check_finite=False)
    excitation = energies - energies[0]
    gap = excitation[1]
    beta = reduced_beta / gap
    boltzmann = np.exp(-beta * excitation)
    weights = boltzmann / boltzmann.sum()
    row = observables(
        shape,
        weights @ np.sum(states * (x2_proj @ states), axis=0),
        weights @ np.sum(states * (x4_proj @ states), axis=0),
    )
    return {
        "shape": shape,
        "beta": float(beta),
        "reduced_beta": reduced_beta,
        "gap": float(gap),
        "x2": row["x2"],
        "x4": row["x4"],
        "energy": float(weights @ energies),
        "virial_energy": row["energy"],
        "last_relative_boltzmann_weight": float(boltzmann[-1]),
        "basis_size": size,
    }


@lru_cache(maxsize=4)
def bridge_geometry(points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u, weights = roots_legendre(points)
    u = (u + 1) / 2
    weights /= 2
    retained_amplitude = sqrt(2) * np.sin(pi * u) / pi
    tail_variance_per_epsilon = u * (1 - u) - retained_amplitude**2
    if not np.all(tail_variance_per_epsilon > 0):
        raise ArithmeticError("A retained bridge variance exceeded the free bridge variance.")
    return u, weights, retained_amplitude, tail_variance_per_epsilon


@lru_cache(maxsize=4)
def standard_normal_hermite(order: int) -> tuple[np.ndarray, np.ndarray]:
    z, weights = roots_hermitenorm(order)
    return z, weights / sqrt(2 * pi)


def segment_weights(
    left: np.ndarray,
    right: np.ndarray,
    shape: float,
    epsilon: float,
    points: int,
    order: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    u, weights, retained_amplitude, omitted = bridge_geometry(points)
    sigma2 = epsilon * omitted
    z, gh_weights = standard_normal_hermite(order)
    offsets = (sqrt(epsilon) * z[:, None]) * retained_amplitude
    quadratic = (1 - shape) / 2
    quartic = shape / 4
    tail_stiffening = 1.5 * shape * sigma2
    constant = float(weights @ (quadratic * sigma2 + 3 * quartic * sigma2**2))
    plain = np.empty_like(left)
    averaged = np.empty_like(left)

    # Accumulate the physical action and tail terms at the same force nodes.
    # Bounded chunks keep both exponent arrays and temporaries small.
    for start in range(0, left.size, 1600):
        sl = slice(start, start + 1600)
        x = left[sl, None]
        difference = right[sl, None] - x
        energy = np.zeros((x.size, order))
        stiffening = np.zeros_like(energy)
        for j, site in enumerate(u):
            squared = (x + site * difference + offsets[:, j]) ** 2
            energy += weights[j] * (quadratic + quartic * squared) * squared
            stiffening += weights[j] * tail_stiffening[j] * squared
        plain[sl] = np.exp(-epsilon * energy) @ gh_weights
        averaged[sl] = np.exp(-epsilon * (energy + stiffening + constant)) @ gh_weights
    return plain, averaged, constant


def transfer_trace(
    shape: float,
    beta: float,
    beads: int,
    points: int,
    box: float,
    max_dx: float,
    order: int,
) -> dict:
    epsilon = beta / beads
    half_steps = ceil(box / max_dx)
    grid = np.linspace(-box, box, 2 * half_steps + 1)
    dx = box / half_steps
    end = grid.size - 1
    i, j = np.triu_indices(grid.size)
    use = i + j <= end
    i, j = i[use], j[use]
    left, right = grid[i], grid[j]
    prefactor = dx / sqrt(2 * pi * epsilon) * np.exp(-0.5 * (left - right)**2 / epsilon)
    plain, averaged, constant = segment_weights(left, right, shape, epsilon, points, order)

    result = {}
    for variant, rows in zip(VARIANTS, (plain, averaged)):
        transfer = np.empty((grid.size, grid.size))
        transfer[i, j] = transfer[j, i] = transfer[end - i, end - j] = transfer[end - j, end - i] = (
            prefactor * rows
        )
        spectrum, eigenvectors = eigh(transfer, check_finite=False)
        if spectrum[-1] <= 0 or abs(spectrum[0]) > spectrum[-1] * (1 + 1e-12):
            raise ArithmeticError("The positive transfer kernel lost its principal eigenvalue.")
        relative = spectrum / spectrum[-1]
        occupancy = relative**beads
        partition = occupancy.sum()
        density = (eigenvectors * eigenvectors) @ (occupancy / partition)
        bead = observables(shape, density @ grid**2, density @ grid**4)
        result[variant] = {
            "bead": bead,
            "log_partition": float(beads * np.log(spectrum[-1]) + np.log(partition)),
            "density_normalization": float(density.sum()),
            "minimum_density_mass": float(density.min()),
            "edge_density_mass": float(density[0] + density[-1]),
            "minimum_relative_eigenvalue": float(relative[0]),
        }
    return {
        "box": box,
        "dx": dx,
        "grid_size": int(grid.size),
        "hermite_order": order,
        "tail_variance_sites": (epsilon * bridge_geometry(points)[3]).tolist(),
        "coordinate_independent_action_energy": constant,
        **result,
    }


def force_and_covariance_controls(points_set, shapes) -> dict:
    maximum_force_error = 0.0
    tail_checks = []
    high_modes = np.arange(2, 50001)
    for points in points_set:
        u, _, _, tail = bridge_geometry(points)
        explicit_sum = 2 / pi**2 * np.sum((np.sin(pi * u[:, None] * high_modes) / high_modes)**2, axis=1)
        series_residual = tail - explicit_sum
        truncation_bound = 2 / (pi**2 * high_modes[-1])
        if np.any(series_residual < -1e-12) or np.any(series_residual > truncation_bound + 1e-12):
            raise ArithmeticError("Explicit free-bridge summation failed to bound the analytic tail.")
        tail_checks.append({
            "points": points,
            "highest_mode": int(high_modes[-1]),
            "maximum_residual_per_epsilon": float(np.max(series_residual)),
            "analytic_upper_bound_per_epsilon": float(truncation_bound),
        })
        coordinates = np.linspace(-2.1, 2.3, 19)[:, None]
        step = 1e-5
        for shape in shapes:
            sigma2 = 2.0 * tail
            # Delta V contains a single coordinate-dependent quadratic term.
            d_v = 0.5 * (1 - shape) * sigma2 + 0.75 * shape * sigma2**2
            plus = d_v + 1.5 * shape * sigma2 * (coordinates + step)**2
            minus = d_v + 1.5 * shape * sigma2 * (coordinates - step)**2
            max_error = np.max(abs((plus - minus) / (2 * step) - 3 * shape * sigma2 * coordinates))
            maximum_force_error = max(maximum_force_error, float(max_error))
    if maximum_force_error > 2e-9:
        raise ArithmeticError("Numerical differentiation failed to recover the tail force.")
    return {
        "covariance_series": tail_checks,
        "central_difference_step": step,
        "maximum_coordinate_derivative_error": maximum_force_error,
    }


def compare(variant_result, quantum) -> dict:
    return {key: variant_result["bead"][key] / quantum[key] - 1 for key in OBSERVABLES}


def reductions(row: dict) -> dict:
    improvement = row["absolute_error_reduction"]
    numerical = row["maximum_relative_finite_action_control_change"]
    return {
        key: "reduced" if improvement[key] > 2 * numerical + 1e-12
        else "increased" if improvement[key] < -2 * numerical - 1e-12
        else "unchanged within numerical reference resolution" for key in OBSERVABLES
    }


def run(protocol: dict) -> dict:
    if protocol["mass"] != 1 or protocol["hbar"] != 1:
        raise ValueError("This test is expressed in m=hbar=1 units.")
    mechanism = protocol["mechanism_control"]
    reduced_beta = mechanism["reduced_beta"]
    if mechanism["modes"] != 1:
        raise ValueError("The retained bridge Gaussian quadrature uses exactly one mode.")
    target = protocol["acceptance"]["numerical_reference_relative_tolerance"]
    start = perf_counter()
    shapes = protocol["shape"]
    continuum = []
    rows = []
    spectral_change = 0.0
    transfer_change = 0.0
    harmonic_constant_check = 0.0
    harmonic_readout_check = 0.0
    for shape in shapes:
        spectral = [continuum_reference(shape, reduced_beta, size) for size in SPECTRAL_SIZES]
        exact = spectral[-1]
        difference = max(
            abs(earlier[key] / exact[key] - 1)
            for earlier in spectral[:-1] for key in ("gap", "x2", "x4", "energy")
        )
        virial_difference = abs(exact["virial_energy"] / exact["energy"] - 1)
        spectral_change = max(spectral_change, difference, virial_difference)
        continuum.append({
            **exact,
            "spectral_resolution": spectral,
            "maximum_coarser_spectral_relative_change": difference,
            "spectral_virial_relative_difference": virial_difference,
        })
        for beads in mechanism["beads"]:
            for points in mechanism["points"]:
                transfers = {
                    control: transfer_trace(shape, exact["beta"], beads, points, box, dx, order)
                    for control, (box, dx, order) in FINITE_ACTION_GRIDS.items()
                }
                fine = transfers["fine"]
                delta = {
                    key: fine["partial_average"]["bead"][key] - fine["plain"]["bead"][key]
                    for key in (*OBSERVABLES, "x4")
                }
                error = {variant: compare(fine[variant], exact) for variant in VARIANTS}
                changes = {
                    control: max(abs(row[variant]["bead"][key] / fine[variant]["bead"][key] - 1)
                                 for variant in VARIANTS for key in (*OBSERVABLES, "x4"))
                    for control, row in transfers.items() if control != "fine"
                }
                max_change = max(changes.values())
                transfer_change = max(transfer_change, max_change)
                if shape == 0.0:
                    shift = fine["partial_average"]["log_partition"] - fine["plain"]["log_partition"]
                    constant = fine["coordinate_independent_action_energy"]
                    harmonic_constant_check = max(harmonic_constant_check, abs(shift + exact["beta"] * constant))
                    harmonic_readout_check = max(harmonic_readout_check, abs(delta["x2"]), abs(delta["energy"]))
                benefit = {
                    key: abs(error["plain"][key]) - abs(error["partial_average"][key])
                    for key in OBSERVABLES
                }
                row = {
                    "shape": shape, "beta": exact["beta"], "reduced_beta": reduced_beta,
                    "beads": beads, "modes": 1, "points": points,
                    "signed_relative_error": error,
                    "delta_partial_average_minus_plain": delta,
                    "absolute_error_reduction": benefit,
                    "maximum_relative_finite_action_control_change": max_change,
                    "finite_action_relative_controls": changes,
                    "finite_action_coarser_records": {k: v for k, v in transfers.items() if k != "fine"},
                    **fine,
                }
                row["absolute_error_result"] = reductions(row)
                rows.append(row)
                print(f"s={shape:g} P={beads} L={points} E plain={error['plain']['energy']:+.5f} "
                      f"PA={error['partial_average']['energy']:+.5f} refine={max_change:.3g}", flush=True)

    if max(spectral_change, transfer_change) > target:
        raise ArithmeticError(f"Numerical control exceeds tolerance: {spectral_change}, {transfer_change}.")
    if max(harmonic_constant_check, harmonic_readout_check) > 1e-10:
        raise ArithmeticError("The normalized harmonic law changed under constant averaging.")
    windows = [{
        "beads": beads,
        "points": points,
        "shape": [row["shape"] for row in rows if (row["beads"], row["points"]) == (beads, points)],
        "absolute_energy_error_reduction": [
            row["absolute_error_reduction"]["energy"]
            for row in rows if (row["beads"], row["points"]) == (beads, points)
        ],
        "energy_crosses_exact_value_on_average": [
            row["shape"] for row in rows if (row["beads"], row["points"]) == (beads, points)
            and row["signed_relative_error"]["plain"]["energy"]
            * row["signed_relative_error"]["partial_average"]["energy"] < -1e-12
        ],
    } for beads in mechanism["beads"] for points in mechanism["points"]]
    return {
        "equations": {
            "action": "V(u) + 0.5(1-shape)*sigma^2 + 0.25*shape*(6*x_K(u)^2*sigma^2 + 3*sigma^4)",
            "sigma2": "(beta/P)*(u*(1-u) - 2*sin(pi*u)^2/pi^2)",
            "energy": "physical endpoint virial energy (1-shape)<x^2> + 0.75*shape*<x^4>",
            "interpretation": "Retains only the first potential cumulant over omitted free bridge modes. This is a diagnostic change to the original finite action.",
        },
        "quantum": continuum,
        "rows": rows,
        "cross_shape_energy_cuts": windows,
        "controls": {
            "frozen_numerical_relative_tolerance": target,
            "maximum_continuum_resolution_or_virial_relative_difference": spectral_change,
            "maximum_transfer_relative_difference": transfer_change,
            "maximum_harmonic_bead_observable_absolute_change": harmonic_readout_check,
            "maximum_harmonic_log_partition_shift_residual": harmonic_constant_check,
            **force_and_covariance_controls(mechanism["points"], shapes),
        },
        "wall_seconds": perf_counter() - start,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--anchor", action="store_true", help="run only s=0 and s=1 at P=8")
    args = parser.parse_args()
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    config = json.loads(protocol_bytes)
    if args.anchor:
        config["shape"] = [0, 1]
        config["mechanism_control"]["beads"] = [8]
    artifact = run(config)
    artifact["protocol_sha256"] = hashlib.sha256(protocol_bytes).hexdigest()
    artifact["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
