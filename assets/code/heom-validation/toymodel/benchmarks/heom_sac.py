"""Original SAC: explicit nuclear-bath block hierarchy versus wavefunction DVR.

This is NOT a Gaussian HEOM mapping of SAC.  The nonlinear nuclear operators
are retained exactly in a finite bath basis.  Pure-state factor propagation
stores all joint amplitudes and offers no general bath-compression claim.
Run with ``python -m toymodel.benchmarks.heom_sac --output <new-directory>``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from toymodel.DVRmethods.DVRBASE import DVRBASE
from toymodel.model.tully_model import TullySimpleAvoidedCrossing
from toymodel.methods.finite_bath_hierarchy import FiniteBathHierarchy
from toymodel.benchmarks.curve_quality import (
    curve_smoothness_gate, curve_smoothness_metrics,
)


# Frozen before inspection; all quantities are in atomic units, hbar=1.
CONFIG = {
    "model": {"a": 0.01, "b": 1.6, "c": 0.005, "d": 1.0, "mass": 2000.0},
    "initial": {"x0": -5.0, "p0": 15.0, "sigma": 0.75, "diabatic_state": 0},
    "tmax": 1200.0, "ntime": 301,
    "grid_convention": "cell-centered", "kinetic_operator": "sinc", "CAP": False,
    "cells": [[192, 12.0, 128], [192, 12.0, 160], [192, 12.0, 192],
              [256, 12.0, 256], [320, 15.0, 320]],
    "gates": {"same_grid_rho": 1e-9, "grid_rho": 1e-5, "box_rho": 1e-5,
              "bath_rho": 1e-5, "trace": 1e-9, "hermiticity": 1e-10,
              "negative_eigenvalue": 1e-9, "energy_drift": 1e-9,
              "boundary_probability": 1e-7, "initial_projection_loss": 1e-8,
              "hamiltonian_reconstruction": 1e-11,
              "spectral_orthogonality": 1e-11, "spectral_residual": 1e-11,
              "output_cadence_rho": 1e-10},
    "curve_gates": {"max_rms_fraction": 0.002, "max_peak_fraction": 0.01,
                    "max_spike_fraction": 0.01,
                    "max_endpoint_roughness_fraction": 0.01},
    "curve_window_fraction": 0.02,
}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    import scipy
    import toymodel.methods.finite_bath_hierarchy as finite_module
    import inspect
    paths = [Path(__file__), Path(finite_module.__file__),
             Path(inspect.getfile(DVRBASE)), Path(inspect.getfile(TullySimpleAvoidedCrossing)),
             Path(inspect.getfile(curve_smoothness_metrics))]
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "source_sha256": {str(p): _hash(p) for p in paths}}


def reduced(psi):
    """psi[t, electronic, bath]; rho_ij=sum_b psi_ib psi_jb* (no renormalization)."""
    return np.einsum("tib,tjb->tij", psi, psi.conj())


def density_audit(rhos):
    return {
        "finite": bool(np.isfinite(rhos).all()),
        "max_trace_defect": float(np.max(np.abs(np.trace(rhos, axis1=1, axis2=2) - 1))),
        "max_hermiticity_defect": float(np.max(np.abs(rhos - rhos.conj().transpose(0, 2, 1)))),
        "min_eigenvalue": float(np.linalg.eigvalsh(rhos).min()),
    }


def rho_difference(a, b):
    delta = a - b
    return {"max_element_error": float(np.abs(delta).max()),
            "max_trace_distance": float((0.5 * np.abs(np.linalg.eigvalsh(delta)).sum(axis=-1)).max())}


def run_cell(n, bound, nb, config):
    start = time.perf_counter()
    model = TullySimpleAvoidedCrossing(**config["model"])
    dvr = DVRBASE(model=model, Nxgrid=n, xbound=(-bound, bound),
                  m=config["model"]["mass"], basis_ordering="state-major",
                  grid_convention=config["grid_convention"],
                  kinetic_operator=config["kinetic_operator"], is_CAP=False)
    dvr.build()
    x, dx = dvr.xgrid, dvr.dx
    initial = config["initial"]
    nuclear = np.exp(1j * initial["p0"] * x - (x-initial["x0"])**2/(4*initial["sigma"]**2))
    # Euclidean DVR coefficients contain sqrt(dx); this is initial preparation only.
    nuclear *= np.sqrt(dx)
    nuclear /= np.linalg.norm(nuclear)
    psi0 = np.zeros((2, n), dtype=complex)
    psi0[initial["diabatic_state"]] = nuclear
    times = np.linspace(0, config["tmax"], config["ntime"])

    energy, eigvec = eigh(dvr.H, driver="evd")
    coeff = eigvec.conj().T @ psi0.ravel()
    waves = (np.exp(-1j * times[:, None] * energy) * coeff) @ eigvec.T
    dvr_rho = reduced(waves.reshape(-1, 2, n))
    orthogonality = float(np.max(np.abs(eigvec.conj().T @ eigvec - np.eye(len(energy)))))
    spectral_residual = float(np.max(np.abs(dvr.H @ eigvec - eigvec * energy)))
    dvr_energies = np.einsum("ti,ij,tj->t", waves.conj(), dvr.H, waves).real

    bath_energy, bath_vectors = eigh(dvr.Tn, driver="evd")
    u = bath_vectors[:, :nb]
    v = model.V(x)
    z = np.diag([1.0, -1.0])
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    b_z = u.conj().T @ (v[:, 0, 0, None] * u)
    b_x = u.conj().T @ (v[:, 0, 1, None] * u)
    solver = FiniteBathHierarchy(np.zeros((2, 2)), np.diag(bath_energy[:nb]),
                                 couplings=[(z, b_z), (sx, b_x)])
    factors = (psi0 @ u.conj()).reshape(2*nb, 1)
    transform = np.kron(np.eye(2), u)
    reconstruction = float(np.max(np.abs(solver.joint_hamiltonian - transform.conj().T @ dvr.H @ transform)))
    result = solver.evolve_factors(factors, times, store_joint=True, check_normalized=False)
    joint = result.joint_factors[:, :, 0]
    position = (joint.reshape(-1, 2, nb) @ u.T)
    expected_rho = reduced(position)
    if np.max(np.abs(result.density_matrices - expected_rho)) > 1e-10:
        raise RuntimeError("Bath partial trace disagrees with position reconstruction.")
    energies = np.einsum("ti,ij,tj->t", joint.conj(), solver.joint_hamiltonian, joint).real
    edge = np.abs(x) > bound - 2.0
    boundary = float(np.sum(np.abs(position[:, :, edge])**2, axis=(1, 2)).max())
    projection_loss = float(1-np.vdot(factors, factors).real)
    metrics = {
        "grid_points": n, "box_half_width": bound, "bath_states": nb,
        "initial_projection_loss": projection_loss,
        "hamiltonian_reconstruction_error": reconstruction,
        "hierarchy_vs_same_grid_dvr": rho_difference(result.density_matrices, dvr_rho),
        "hierarchy_density": density_audit(result.density_matrices),
        "dvr_density": density_audit(dvr_rho),
        "dvr_eigenvector_orthogonality": orthogonality,
        "dvr_eigenvalue_residual": spectral_residual,
        "dvr_energy_drift": float(np.max(np.abs(dvr_energies-dvr_energies[0]))),
        "dvr_boundary_probability": float(np.sum(np.abs(waves.reshape(-1,2,n)[:,:,edge])**2, axis=(1,2)).max()),
        "energy_drift": float(np.max(np.abs(energies-energies[0]))),
        "boundary_probability": boundary,
        "elapsed_seconds": time.perf_counter()-start,
        "method": "explicit finite-bath block hierarchy; exact rank-one factors; not Gaussian HEOM",
        "DVR_propagation": "spectral diagonalization, scipy.linalg.eigh driver=evd",
        "hierarchy_propagation": "Krylov matrix exponential, no time-step splitting",
    }
    return times, dvr_rho, result.density_matrices, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    frozen = {"config": CONFIG, "provenance": provenance()}
    config_file = out / "frozen_inputs.json"
    if config_file.exists():
        if json.loads(config_file.read_text(encoding="utf-8")) != frozen:
            raise RuntimeError("Resume source/config identity mismatch; use a new attempt.")
    else:
        config_file.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    identity = _hash(config_file)
    series, metrics = {}, {}
    for n, bound, nb in CONFIG["cells"]:
        key = f"n{n}_b{bound:g}_k{nb}"
        data_file, receipt_file = out/f"{key}.npz", out/f"{key}.json"
        reused = False
        if data_file.exists() and receipt_file.exists():
            receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
            if receipt.get("identity") == identity and receipt.get("data_sha256") == _hash(data_file):
                with np.load(data_file) as stored:
                    t, dr, hr = (stored[k].copy() for k in ("times", "dvr", "hierarchy"))
                if t.shape == (CONFIG["ntime"],) and dr.shape == hr.shape == (len(t),2,2) and all(np.isfinite(a).all() for a in (t,dr,hr)):
                    reused = True
        if not reused:
            t, dr, hr, cell_metrics = run_cell(n, bound, nb, CONFIG)
            np.savez_compressed(data_file, times=t, dvr=dr, hierarchy=hr)
            receipt = {"identity": identity, "data_sha256": _hash(data_file), "metrics": cell_metrics}
            receipt_file.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        series[key], metrics[key] = (t, dr, hr), receipt["metrics"]
        print(json.dumps({"cell": key, "reused": reused, "seconds": metrics[key]["elapsed_seconds"],
                          "max_rho_error": metrics[key]["hierarchy_vs_same_grid_dvr"]["max_element_error"]}), flush=True)

    main_key, fine_key, wide_key = "n192_b12_k192", "n256_b12_k256", "n320_b15_k320"
    convergence = {
        "grid": rho_difference(series[main_key][1], series[fine_key][1]),
        "box": rho_difference(series[fine_key][1], series[wide_key][1]),
        "bath128_to160": rho_difference(series["n192_b12_k128"][2], series["n192_b12_k160"][2]),
        "bath160_to192": rho_difference(series["n192_b12_k160"][2], series[main_key][2]),
    }
    # Independent denser output schedule tests the Krylov interval operation;
    # eigensolution itself has no integration timestep to refine.
    refined_config = dict(CONFIG, ntime=2*CONFIG["ntime"]-1)
    _, _, refined, refinement_metrics = run_cell(192, 12.0, 192, refined_config)
    np.savez_compressed(out/"output_cadence_refinement.npz",
                        times=np.linspace(0, CONFIG["tmax"], refined_config["ntime"]),
                        hierarchy=refined)
    convergence["output_cadence"] = rho_difference(refined[::2], series[main_key][2])
    curves = {}
    for label, rho in [("dvr", series[fine_key][1]), ("hierarchy", series[fine_key][2])]:
        for name, values in [("P0",rho[:,0,0].real),("P1",rho[:,1,1].real),
                             ("Re_rho01",rho[:,0,1].real),("Im_rho01",rho[:,0,1].imag)]:
            metric = curve_smoothness_metrics(t, values, window_fraction=CONFIG["curve_window_fraction"])
            curves[f"{label}_{name}"] = dict(metric, passed=curve_smoothness_gate(metric, **CONFIG["curve_gates"]))
    g = CONFIG["gates"]
    selected = [metrics[k] for k in (main_key, fine_key, wide_key)]
    gates = {
        "same_grid": all(m["hierarchy_vs_same_grid_dvr"]["max_element_error"] < g["same_grid_rho"] for m in selected),
        "grid": convergence["grid"]["max_element_error"] < g["grid_rho"],
        "box": convergence["box"]["max_element_error"] < g["box_rho"],
        "bath160_to192": convergence["bath160_to192"]["max_element_error"] < g["bath_rho"],
        "projection160": abs(metrics["n192_b12_k160"]["initial_projection_loss"]) < g["initial_projection_loss"],
        "trace": all(m["hierarchy_density"]["max_trace_defect"] < g["trace"] for m in selected),
        "dvr_trace": all(m["dvr_density"]["max_trace_defect"] < g["trace"] for m in selected),
        "hermiticity": all(m["hierarchy_density"]["max_hermiticity_defect"] < g["hermiticity"] for m in selected),
        "positivity": all(m["hierarchy_density"]["min_eigenvalue"] > -g["negative_eigenvalue"] for m in selected),
        "energy": all(m["energy_drift"] < g["energy_drift"] for m in selected),
        "dvr_energy": all(m["dvr_energy_drift"] < g["energy_drift"] for m in selected),
        "dvr_eigenbasis": all(m["dvr_eigenvector_orthogonality"] < g["spectral_orthogonality"] and m["dvr_eigenvalue_residual"] < g["spectral_residual"] for m in selected),
        "boundary": all(m["boundary_probability"] < g["boundary_probability"] for m in selected),
        "dvr_boundary": all(m["dvr_boundary_probability"] < g["boundary_probability"] for m in selected),
        "hamiltonian": all(m["hamiltonian_reconstruction_error"] < g["hamiltonian_reconstruction"] for m in selected),
        "output_cadence": convergence["output_cadence"]["max_element_error"] < g["output_cadence_rho"],
        "curve_quality": all(m["passed"] for m in curves.values()),
    }
    summary = {"identity": identity, "metrics": metrics, "convergence": convergence,
               "output_refinement_metrics": refinement_metrics, "curve_quality": curves,
               "gates": gates, "numerical_gates_passed": all(gates.values()),
               "visual_audit": "pending", "scope": "finite-time original Tully SAC; explicit bath, not Gaussian HEOM"}
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(out, series[fine_key], convergence)
    print(json.dumps({"gates": gates, "numerical_gates_passed": all(gates.values())}), flush=True)


def plot(out, chosen, convergence):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t, dvr, hierarchy = chosen
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for i in range(2):
        ax[0,0].plot(t, dvr[:,i,i].real, label=f"DVR P{i}")
        ax[0,0].plot(t[::10], hierarchy[::10,i,i].real, "o", ms=3, fillstyle="none", label=f"Bath hierarchy P{i}")
    for part, fn in [("Re", np.real),("Im", np.imag)]:
        ax[0,1].plot(t, fn(dvr[:,0,1]), label=f"DVR {part} rho01")
        ax[0,1].plot(t[::10], fn(hierarchy[::10,0,1]), "o", ms=3, fillstyle="none", label=f"Hierarchy {part}")
    ax[1,0].scatter(t, np.max(np.abs(hierarchy-dvr), axis=(1,2)), color="black", s=5)
    ax[1,0].set_ylabel("Max reduced-density element error")
    ax[1,1].axis("off")
    text = ("Original Tully SAC; atomic units\n"
            "A=0.01, B=1.6, C=0.005, D=1, m=2000\n"
            "Initial: diabatic state 0; q=-5, p=15, sigma=0.75\n"
            "Sinc DVR: N=256, box [-12,12], no CAP\n\n"
            "Explicit nuclear-bath block hierarchy\n"
            "Full nuclear information retained; NOT Gaussian HEOM\n"
            "No wavefunction/density renormalization in propagation\n\n"
            f"Grid refinement max error: {convergence['grid']['max_element_error']:.2e}\n"
            f"Box refinement max error: {convergence['box']['max_element_error']:.2e}\n"
            f"Bath 160 to 192 max error: {convergence['bath160_to192']['max_element_error']:.2e}")
    ax[1,1].text(0, 1, text, va="top", fontsize=10, linespacing=1.5)
    ax[0,0].set_ylabel("Fixed-diabatic population")
    ax[0,1].set_ylabel("Fixed-diabatic coherence")
    for a in (ax[0,0], ax[0,1], ax[1,0]):
        a.set_xlabel("Time / a.u.")
        a.grid(alpha=0.2)
    ax[0,0].legend(fontsize=8)
    ax[0,1].legend(fontsize=8)
    fig.suptitle("1D SAC: exact finite nuclear bath versus DVR")
    fig.savefig(out/"sac_comparison.png", dpi=180)
    fig.savefig(out/"sac_comparison.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
