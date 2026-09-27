"""Gaussian HEOM versus coordinate sinc DVR: harmonic control, NOT SAC.

Run with python -m toymodel.benchmarks.heom_harmonic --output NEW_DIRECTORY.
H_total = p²/2 + omega² x²/2 + H_s + g sqrt(2 omega) Q x (hbar=m=1).
The factorized initial bath is the free harmonic Gibbs state. All reduced
density matrices are in the fixed input electronic basis, without repair.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import eigh

from toymodel.DVRmethods.DVRBASE import DVRBASE
from toymodel.methods.heom import BosonicHEOM, harmonic_mode_exponents
from toymodel.benchmarks.curve_quality import curve_smoothness_metrics, curve_smoothness_gate


# Freeze both physical case and acceptance gates before inspecting results.
CONFIG = {
    "label": "harmonic Gaussian bath control; not original SAC",
    "omega": 0.9, "g": 0.19, "beta": 2.2, "mass": 1.0, "hbar": 1.0,
    "H_sz": 0.31, "H_sx": 0.12, "Q_sx": 0.8, "Q_sz": 0.2,
    "initial_electronic_state": 0,
    "initial_bath": "factorized free harmonic Gibbs, not interacting equilibrium",
    "tmax": 6.0, "ntime": 301, "depths": [2, 4, 8, 10],
    "dvr_cells": [[64, 8.0], [96, 8.0], [120, 10.0]],
    "kinetic_operator": "sinc", "grid_convention": "cell-centered", "CAP": False,
    "thermal_factor_weight_cutoff": 1e-16,
    "gates": {"maxrho": 1e-7, "grid_box": 1e-8, "tier": 1e-7,
              "trace": 1e-10, "hermiticity": 1e-10, "negativity": 1e-10,
              "thermal_discarded_weight": 1e-13, "eigenvector_orthogonality": 1e-10},
    "curve_window_fraction": 0.02,
    "curve_gates": {"max_rms_fraction": 0.002, "max_peak_fraction": 0.01,
                    "max_spike_fraction": 0.01, "max_endpoint_roughness_fraction": 0.01},
}


def _operators(config):
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sz = np.diag([1.0, -1.0])
    return config["H_sz"] * sz + config["H_sx"] * sx, config["Q_sx"] * sx + config["Q_sz"] * sz


class HarmonicCoupledModel:
    nstate = 2

    def __init__(self, config):
        self.config = config
        self.H, self.Q = _operators(config)

    def V(self, x):
        x = np.asarray(x, dtype=float)
        omega, g = self.config["omega"], self.config["g"]
        return (0.5 * omega**2 * x[..., None, None]**2 * np.eye(2)
                + self.H + g * np.sqrt(2 * omega) * x[..., None, None] * self.Q)


def _dvr(n, bound, times, config):
    dvr = DVRBASE(model=HarmonicCoupledModel(config), Nxgrid=n,
                  xbound=(-bound, bound), m=1.0, basis_ordering="state-major",
                  grid_convention="cell-centered", kinetic_operator="sinc")
    dvr.construct_grid()
    dvr.construct_T()
    dvr.construct_V()
    dvr.construct_H()
    free = dvr.Tn + np.diag(0.5 * config["omega"]**2 * dvr.xgrid**2)
    bath_e, bath_u = eigh(free, driver="evd")
    weights = np.exp(-config["beta"] * (bath_e - bath_e[0]))
    weights /= weights.sum()
    keep = weights > config["thermal_factor_weight_cutoff"]
    factors = np.zeros((2 * n, int(keep.sum())), dtype=complex)
    state = config["initial_electronic_state"]
    factors[state * n:(state + 1) * n] = bath_u[:, keep] * np.sqrt(weights[keep])[None, :]
    energies, vectors = eigh(dvr.H, driver="evd")
    coefficients = vectors.conj().T @ factors
    rho = np.empty((len(times), 2, 2), dtype=complex)
    for it, t in enumerate(times):
        factor_t = (vectors @ (np.exp(-1j * energies * t)[:, None] * coefficients)).reshape(2, n, -1)
        rho[it] = np.einsum("ixk,jxk->ij", factor_t, factor_t.conj())
    diagnostics = {
        "n": n, "bound": bound, "dx": dvr.dx, "thermal_factors": int(keep.sum()),
        "thermal_discarded_weight": float(weights[~keep].sum()),
        "eigenvector_orthogonality": float(np.max(abs(vectors.conj().T @ vectors - np.eye(2 * n)))),
        "free_ground_energy_error": float(abs(bath_e[0] - config["omega"] / 2)),
    }
    return rho, diagnostics


def _density_audit(rho):
    return {"finite": bool(np.isfinite(rho).all()),
            "trace": float(np.max(abs(np.trace(rho, axis1=1, axis2=2) - 1))),
            "hermiticity": float(np.max(abs(rho - rho.conj().transpose(0, 2, 1)))),
            "min_eigenvalue": float(np.linalg.eigvalsh(rho).min())}


def _provenance():
    paths = [Path(__file__), Path(inspect.getfile(BosonicHEOM)),
             Path(inspect.getfile(DVRBASE)), Path(inspect.getfile(curve_smoothness_metrics))]
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def run(output: Path):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {"task": "20260927-heom-sac-dvr", "case": "harmonic-control", "role": "validation",
                "attempt": output.parent.name if output.name == "results" else output.name,
                "host": platform.node(), "started_utc": datetime.now(timezone.utc).isoformat(),
                "output": str(output), "state": "running", "config": CONFIG,
                "command": f"python -m toymodel.benchmarks.heom_harmonic --output {output}",
                "provenance": _provenance(), "expected_outputs": ["config.json", "run.json", "data.npz", "metrics.json", "harmonic-control.png"]}
    config_text = json.dumps(CONFIG, indent=2, sort_keys=True)
    manifest["config_sha256"] = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    (output / "config.json").write_text(config_text, encoding="utf-8")
    (output / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    times = np.linspace(0, CONFIG["tmax"], CONFIG["ntime"])
    series, cell_diagnostics = {}, {}
    for n, bound in CONFIG["dvr_cells"]:
        label = f"DVR_N{n}_L{bound:g}"
        series[label], cell_diagnostics[label] = _dvr(n, bound, times, CONFIG)
    h, q = _operators(CONFIG)
    heom_diagnostics = {}
    for depth in CONFIG["depths"]:
        solver = BosonicHEOM(h, [q], harmonic_mode_exponents(CONFIG["omega"], CONFIG["g"], CONFIG["beta"]), depth)
        label = f"HEOM_L{depth}"
        series[label] = solver.propagate(np.diag([1.0, 0.0]), times).density_matrices
        heom_diagnostics[label] = {"depth": depth, "nados": solver.nados, "generator_nnz": solver.generator.nnz}
    np.savez_compressed(output / "data.npz", times=times, **series)
    difference = lambda a, b: float(np.max(abs(series[a] - series[b])))
    comparisons = {"grid": difference("DVR_N64_L8", "DVR_N96_L8"),
                   "box": difference("DVR_N96_L8", "DVR_N120_L10"),
                   "tier_8_to_10": difference("HEOM_L8", "HEOM_L10"),
                   "heom_to_dvr": {str(d): difference(f"HEOM_L{d}", "DVR_N120_L10") for d in CONFIG["depths"]}}
    density = {label: _density_audit(rho) for label, rho in series.items()}
    curve_quality = {}
    for label, rho in series.items():
        for name, values in (("P0", rho[:, 0, 0].real), ("P1", rho[:, 1, 1].real),
                             ("Re_rho01", rho[:, 0, 1].real), ("Im_rho01", rho[:, 0, 1].imag)):
            metric = curve_smoothness_metrics(times, values, window_fraction=CONFIG["curve_window_fraction"])
            curve_quality[f"{label}_{name}"] = dict(metric, passed=curve_smoothness_gate(metric, **CONFIG["curve_gates"]))
    gates = CONFIG["gates"]
    passed = {
        "dvr_grid": comparisons["grid"] < gates["grid_box"], "dvr_box": comparisons["box"] < gates["grid_box"],
        "heom_tier": comparisons["tier_8_to_10"] < gates["tier"],
        "heom_dvr": comparisons["heom_to_dvr"]["10"] < gates["maxrho"],
        "density": all(m["finite"] and m["trace"] < gates["trace"] and m["hermiticity"] < gates["hermiticity"]
                       and m["min_eigenvalue"] > -gates["negativity"] for m in density.values()),
        "dvr_diagonalization": all(m["thermal_discarded_weight"] < gates["thermal_discarded_weight"]
                                   and m["eigenvector_orthogonality"] < gates["eigenvector_orthogonality"]
                                   for m in cell_diagnostics.values()),
        "curve_quality": all(m["passed"] for m in curve_quality.values()),
    }
    # Plot only physical populations/coherences, whose quality is audited above.
    # Tier errors remain quantitative table data, avoiding unresolved tiny lines.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for label, style, width in (("DVR_N120_L10", "-", 2.5), ("HEOM_L2", ":", 1.8),
                                ("HEOM_L8", "--", 1.7), ("HEOM_L10", "-.", 1.4)):
        rho = series[label]
        axes[0].plot(times, rho[:, 1, 1].real, style, lw=width, label=label)
        axes[1].plot(times, rho[:, 0, 1].real, style, lw=width, label=label + " Re")
    axes[0].set_ylabel("Electronic population P1")
    axes[1].set_ylabel("Re electronic coherence rho01")
    for ax in axes:
        ax.set_xlabel("Time (hbar = 1)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Gaussian harmonic-bath control: standard HEOM versus coordinate DVR (NOT SAC)", fontsize=11)
    fig.savefig(output / "harmonic-control.png", dpi=180)
    plt.close(fig)
    metrics = {"comparisons": comparisons, "density_audits": density, "dvr_diagnostics": cell_diagnostics,
               "heom_diagnostics": heom_diagnostics, "curve_quality": curve_quality, "gates": passed,
               "numerical_gates_passed": all(passed.values()), "visual_audit": "pending independent rendered-image inspection",
               "limitations": "Gaussian linear harmonic control only; no claim about original SAC or general nonlinear baths.",
               "elapsed_seconds": time.perf_counter() - started}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    manifest.update(state="completed" if all(passed.values()) else "failed_numerical_gate",
                    finished_utc=datetime.now(timezone.utc).isoformat(), numerical_gates_passed=all(passed.values()))
    (output / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "comparisons": comparisons, "gates": passed,
                      "elapsed_seconds": metrics["elapsed_seconds"]}, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are never overwritten")
    args = parser.parse_args()
    result = run(args.output)
    if not result["numerical_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
