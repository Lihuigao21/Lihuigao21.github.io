"""Reproduce the algebraic and prescribed-electronic checks in MQC VI.

Requires numpy and matplotlib. Run from the website repository:
    python assets/code/mqc/cps_dual_kernel_checks.py

This is deterministic quadrature on CP^1, not a nuclear-dynamics benchmark.
The plotted intervals are analytic bounds, not sampled trajectories.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TOLERANCE = 1.0e-12
GAMMAS = (-0.2, 0.0, (np.sqrt(3.0) - 1.0) / 2.0, 0.5, 1.0)
TIMES = (0.0, 0.7, 1.4, 2.1)


def projective_quadrature(npolar: int, nphase: int):
    """Normalized Haar measure dmu = d(cos theta) dphi / (4 pi)."""
    costheta, weight = np.polynomial.legendre.leggauss(npolar)
    phi = 2.0 * np.pi * np.arange(nphase) / nphase
    up = np.broadcast_to(np.sqrt((1.0 + costheta[:, None]) / 2.0),
                         (npolar, nphase))
    down = np.sqrt((1.0 - costheta[:, None]) / 2.0) * np.exp(1j * phi)
    vectors = np.stack((up, down), axis=-1).reshape(-1, 2)
    measure = np.broadcast_to(weight[:, None] / (2.0 * nphase),
                              (npolar, nphase)).reshape(-1)
    return vectors, measure


def outer(vectors):
    return vectors[:, :, None] * vectors[:, None, :].conj()


def symbol(matrix, vectors, radius, offset):
    return radius * np.einsum("ni,ij,nj->n", vectors.conj(), matrix,
                              vectors) - offset * np.trace(matrix)


def evaluate(npolar, nphase):
    vectors, measure = projective_quadrature(npolar, nphase)
    identity = np.eye(2)
    hamiltonian = np.array([[0.35, 0.23 + 0.17j],
                            [0.23 - 0.17j, 0.90]])
    psi = np.array([np.sqrt(0.7), np.sqrt(0.3) * np.exp(0.4j)])
    initial_density = np.outer(psi, psi.conj())
    a = np.array([[0.2, 0.3 + 0.4j], [0.3 - 0.4j, -0.7]])
    b = np.array([[1.1, -0.2 + 0.1j], [-0.2 - 0.1j, 0.8]])
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    rows = []
    diagnostics = []
    for gamma in GAMMAS:
        radius = 1.0 + 2.0 * gamma
        dual_radius = 3.0 / radius
        dual_offset = (1.0 - gamma) / radius
        kernel = radius * outer(vectors) - gamma * identity
        dual = dual_radius * outer(vectors) - dual_offset * identity
        f_initial = symbol(initial_density, vectors, radius, gamma).real
        initial_weight = 2.0 * f_initial
        trace_integral = 2.0 * np.sum(
            measure * symbol(a, vectors, radius, gamma)
            * symbol(b, vectors, dual_radius, dual_offset))
        trace_error = float(abs(trace_integral - np.trace(a @ b)))
        trace_one_error = float(max(
            np.max(abs(np.trace(kernel, axis1=1, axis2=2) - 1.0)),
            np.max(abs(np.trace(dual, axis1=1, axis2=2) - 1.0))))
        weight_error = float(abs(np.sum(measure * initial_weight) - 1.0))
        max_density_error = 0.0
        max_wrong_readout_error = 0.0
        for time in TIMES:
            unitary = ((eigenvectors * np.exp(-1j * eigenvalues * time))
                       @ eigenvectors.conj().T)
            evolved_vectors = vectors @ unitary.T
            evolved_dual = (dual_radius * outer(evolved_vectors)
                            - dual_offset * identity)
            estimated = np.einsum("n,n,nij->ij", measure, initial_weight,
                                  evolved_dual)
            exact = unitary @ initial_density @ unitary.conj().T
            density_error = float(np.max(abs(estimated - exact)))
            max_density_error = max(max_density_error, density_error)
            population = float(estimated[0, 0].real)
            exact_population = float(exact[0, 0].real)
            wrong_symbol = (radius * abs(evolved_vectors[:, 0]) ** 2
                            - gamma)
            wrong_population = float(
                np.sum(measure * initial_weight * wrong_symbol))
            max_wrong_readout_error = max(
                max_wrong_readout_error,
                abs(wrong_population - exact_population))
            rows.append(dict(gamma=float(gamma), time=time,
                             exact_population=exact_population,
                             dual_population=population,
                             same_kernel_population=wrong_population,
                             density_max_abs_error=density_error))
        diagnostics.append(dict(
            gamma=float(gamma), radius_squared=radius,
            dual_gamma=dual_offset, trace_pair_error=trace_error,
            trace_one_error=trace_one_error,
            initial_weight_error=weight_error,
            max_density_error=max_density_error,
            same_kernel_max_population_error=max_wrong_readout_error))
    return rows, diagnostics


def plot_intervals(output_dir):
    fig, ax = plt.subplots(figsize=(8.0, 4.2), layout="constrained")
    ax.axvspan(0.0, 1.0, color="#e9e9e3",
               label="Adiabatic energy interval")
    labels = [r"$\gamma=-0.2$", r"$\gamma=0\ (Q)$",
              r"$\gamma=(\sqrt{3}-1)/2\ (W)$",
              r"$\gamma=0.5\ (\mathrm{MMST\ offset})$",
              r"$\gamma=1\ (P)$"]
    bounds = []
    for index, gamma in enumerate(GAMMAS):
        lower, upper = -gamma, 1.0 + gamma
        color = "#315b75" if index == 2 else "#68685e"
        ax.plot([lower, upper], [index, index], color=color,
                linewidth=3, marker="|", markersize=14)
        bounds.append(dict(gamma=float(gamma), minimum=float(lower),
                           maximum=float(upper)))
    ax.set_yticks(range(len(GAMMAS)), labels)
    ax.invert_yaxis()
    ax.set_xlim(-1.12, 2.12)
    ax.set_xlabel(r"Electronic potential / gap $\Delta$"
                  r"  ($E_1=0,\ E_2=\Delta$)")
    ax.set_title("Scalar CPS potential bounds at one nuclear geometry")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=9)
    for suffix in ("png", "svg"):
        destination = output_dir / f"cps-potential-intervals.{suffix}"
        fig.savefig(destination, dpi=180)
        if suffix == "svg":
            lines = destination.read_bytes().splitlines()
            destination.write_bytes(b"\n".join(line.rstrip() for line in lines) + b"\n")
    plt.close(fig)
    return bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=(
        Path(__file__).resolve().parents[2] / "img" / "mqc-series"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, diagnostics = evaluate(6, 12)
    refined_rows, refined_diagnostics = evaluate(10, 20)
    refinement_error = max(abs(a["dual_population"] - b["dual_population"])
                           for a, b in zip(rows, refined_rows))
    for item in diagnostics + refined_diagnostics:
        for key in ("trace_pair_error", "trace_one_error",
                    "initial_weight_error", "max_density_error"):
            if item[key] > TOLERANCE:
                raise RuntimeError(f"{key} exceeds declared tolerance: {item}")
    if refinement_error > TOLERANCE:
        raise RuntimeError("Quadrature refinement exceeds tolerance")
    filename = args.output_dir / "cps-dual-kernel-checks.csv"
    with filename.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    bounds = plot_intervals(args.output_dir)
    report = dict(
        source="assets/code/mqc/cps_dual_kernel_checks.py",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        method="deterministic Gauss-Legendre x uniform azimuth quadrature",
        measure="normalized Haar measure on CP^1; trace identity includes F=2",
        hamiltonian=[[0.35, [0.23, 0.17]], [[0.23, -0.17], 0.90]],
        initial_state={"population_0": 0.7, "population_1": 0.3,
                       "relative_phase": 0.4},
        units="dimensionless; hbar=1",
        quadrature={"primary": [6, 12], "refined": [10, 20]},
        times=list(TIMES), tolerance=TOLERANCE,
        max_refinement_difference=float(refinement_error),
        checks=diagnostics, refined_checks=refined_diagnostics,
        potential_bounds=bounds,
        figure_contract={"type": "analytic intervals, not sampled curves",
                         "physical_energy_interval": [0, 1],
                         "visual_review_required": True},
        versions={"numpy": np.__version__, "matplotlib": matplotlib.__version__},
        scope="static duality and prescribed electronic evolution only",
        passed=True)
    with (args.output_dir / "cps-dual-kernel-checks.json").open(
            "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({
        "passed": True,
        "maximum_density_error": max(x["max_density_error"]
                                     for x in diagnostics),
        "maximum_trace_pair_error": max(x["trace_pair_error"]
                                        for x in diagnostics),
        "refinement_difference": float(refinement_error)}, indent=2))


if __name__ == "__main__":
    main()
