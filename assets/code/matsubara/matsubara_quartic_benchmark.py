"""Partial reproduction of the Willatt Fig. 39 quartic benchmark.

This is the executable behind Matsubara Series Part III.  It compares a
sinc-DVR Kubo reference for V(q)=q^4/4 with Matsubara-mode estimators using
cosine phase reweighting.  The parameters mirror the local reproduction script
used to generate willatt_fig39_partial_repro.png.

By default, the script preserves the archived rendered PNG shipped beside this
file, so the published website figure matches the checked local reproduction.
Pass --recompute to regenerate the curves from the stochastic estimator; high-M
curves can visibly differ because the phase denominator is small.

Dependencies: numpy, scipy, matplotlib.
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg


ROOT = Path(__file__).resolve().parents[3]
FIGURE_PATH = ROOT / "assets" / "img" / "matsubara-series" / "quartic-matsubara-benchmark.png"
META_PATH = ROOT / "assets" / "img" / "matsubara-series" / "quartic-matsubara-meta.csv"
DATA_PATH = ROOT / "assets" / "code" / "matsubara" / "willatt_fig39_partial_repro.csv"
META_SOURCE_PATH = ROOT / "assets" / "code" / "matsubara" / "willatt_fig39_partial_repro_meta.csv"
IMAGE_SOURCE_PATH = ROOT / "assets" / "code" / "matsubara" / "willatt_fig39_partial_repro.png"

BETA = 2.0
MASS = 1.0
HBAR = 1.0


def v_quartic(q: np.ndarray) -> np.ndarray:
    return 0.25 * q**4


def sinc_hamiltonian(x_min: float = -10.0, x_max: float = 10.0, n_grid: int = 401) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(x_min, x_max, n_grid)
    dx = x[1] - x[0]
    idx = np.arange(n_grid)
    diff = idx[:, None] - idx[None, :]
    kinetic = np.zeros((n_grid, n_grid), dtype=float)
    np.fill_diagonal(kinetic, np.pi**2 / 3.0)
    mask = diff != 0
    kinetic[mask] = 2.0 * ((-1.0) ** diff[mask]) / (diff[mask].astype(float) ** 2)
    kinetic *= HBAR**2 / (2.0 * MASS * dx**2)
    return x, kinetic + np.diag(v_quartic(x))


def quantum_kubo_qcorr(times: np.ndarray, n_states: int = 120) -> np.ndarray:
    x, hamiltonian = sinc_hamiltonian()
    energy, coeff = scipy.linalg.eigh(hamiltonian, subset_by_index=(0, n_states - 1))
    q_matrix = coeff.T @ (x[:, None] * coeff)
    q2 = np.abs(q_matrix) ** 2

    energy_i = energy[:, None]
    energy_j = energy[None, :]
    d_energy = energy_j - energy_i
    boltzmann = np.exp(-BETA * energy)
    numerator = boltzmann[:, None] - boltzmann[None, :]
    denominator = BETA * d_energy
    prefactor = np.empty_like(d_energy)
    mask = np.abs(d_energy) > 1e-12
    np.divide(numerator, denominator, out=prefactor, where=mask)
    np.fill_diagonal(prefactor, boltzmann)
    return np.array([np.sum(prefactor * q2 * np.cos(d_energy * t)) for t in times]) / boltzmann.sum()


def make_basis(n_mode: int, n_quad: int = 96):
    k_max = (n_mode - 1) // 2
    tau = (np.arange(n_quad) + 0.5) / n_quad
    modes = list(range(-k_max, k_max + 1))
    basis = np.zeros((len(modes), n_quad))
    for i, k in enumerate(modes):
        if k == 0:
            basis[i] = 1.0
        elif k > 0:
            basis[i] = np.sqrt(2.0) * np.sin(2.0 * np.pi * k * tau)
        else:
            basis[i] = np.sqrt(2.0) * np.cos(2.0 * np.pi * (-k) * tau)
    omega = np.array([2.0 * np.pi * k / BETA for k in modes], dtype=float)
    index = {k: i for i, k in enumerate(modes)}
    return modes, omega, basis, index


def matsubara_potential_grad(q_mode: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_tau = q_mode @ basis
    potential = v_quartic(q_tau).mean(axis=1)
    grad = ((q_tau**3) @ basis.T) / basis.shape[1]
    return potential, grad


def hmc_sample_q(
    n_mode: int,
    *,
    n_sample: int,
    burn: int,
    thin: int,
    step: float,
    n_leapfrog: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    modes, omega, basis, index = make_basis(n_mode)
    dim = len(modes)
    q_current = np.zeros(dim)
    u_current, g_current = matsubara_potential_grad(q_current[None, :], basis)
    u_current = float(u_current[0])
    g_current = BETA * g_current[0]
    samples = []
    accepted = 0
    total = 0

    for it in range(burn + n_sample * thin):
        p_initial = rng.normal(size=dim)
        q = q_current.copy()
        p = p_initial.copy()
        p -= 0.5 * step * g_current
        for j in range(n_leapfrog):
            q += step * p
            u_new, g_new = matsubara_potential_grad(q[None, :], basis)
            u_new = float(u_new[0])
            g_new = BETA * g_new[0]
            if j < n_leapfrog - 1:
                p -= step * g_new
        p -= 0.5 * step * g_new
        p = -p

        old_h = BETA * u_current + 0.5 * np.sum(p_initial**2)
        new_h = BETA * u_new + 0.5 * np.sum(p**2)
        if np.log(rng.random()) < -(new_h - old_h):
            q_current = q
            u_current = u_new
            g_current = g_new
            accepted += 1
        total += 1
        if it >= burn and (it - burn) % thin == 0:
            samples.append(q_current.copy())
    return np.asarray(samples), {
        "acceptance": accepted / total,
        "modes": modes,
        "omega": omega,
        "basis": basis,
        "index": index,
    }


def theta_values(q_mode: np.ndarray, p_mode: np.ndarray, omega: np.ndarray, modes: list[int], index: dict[int, int]) -> np.ndarray:
    theta = np.zeros(q_mode.shape[0])
    for k in modes:
        theta += omega[index[k]] * q_mode[:, index[-k]] * p_mode[:, index[k]]
    return theta


def propagate(q_mode: np.ndarray, p_mode: np.ndarray, basis: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    _, grad = matsubara_potential_grad(q_mode, basis)
    p_mode = p_mode - 0.5 * dt * grad
    q_mode = q_mode + dt * p_mode / MASS
    _, grad = matsubara_potential_grad(q_mode, basis)
    p_mode = p_mode - 0.5 * dt * grad
    return q_mode, p_mode


def matsubara_corr(n_mode: int, times: np.ndarray, *, n_sample: int, seed: int):
    q_samples, info = hmc_sample_q(
        n_mode,
        n_sample=n_sample,
        burn=1200,
        thin=4,
        step=0.12 if n_mode <= 3 else 0.08,
        n_leapfrog=12,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 123)
    modes, omega, basis, index = info["modes"], info["omega"], info["basis"], info["index"]
    p_initial = rng.normal(scale=np.sqrt(MASS / BETA), size=q_samples.shape)
    phase = BETA * theta_values(q_samples, p_initial, omega, modes, index)
    weight = np.cos(phase)
    denominator = float(np.mean(weight))

    q_plus = q_samples.copy()
    p_plus = p_initial.copy()
    q_minus = q_samples.copy()
    p_minus = -p_initial.copy()
    q0 = q_samples[:, index[0]].copy()

    corr = np.empty_like(times)
    corr[0] = np.mean(weight * q0 * q0) / denominator
    dt = times[1] - times[0]
    for i in range(1, len(times)):
        q_plus, p_plus = propagate(q_plus, p_plus, basis, dt)
        q_minus, p_minus = propagate(q_minus, p_minus, basis, dt)
        q_sym = 0.5 * (q_plus[:, index[0]] + q_minus[:, index[0]])
        corr[i] = np.mean(weight * q0 * q_sym) / denominator

    return corr, info["acceptance"], denominator


def load_archived_run():
    data = np.genfromtxt(DATA_PATH, delimiter=",", names=True)
    rows = []
    if META_SOURCE_PATH.exists():
        with META_SOURCE_PATH.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append((row["curve"], float(row["acceptance"]), float(row["phase_average"])))
    return (
        data["t"],
        data["quantum"],
        data["M1"],
        data["M3"],
        data["M5_noisy"],
        rows,
    )


def load_meta_rows():
    rows = []
    if META_SOURCE_PATH.exists():
        with META_SOURCE_PATH.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append((row["curve"], float(row["acceptance"]), float(row["phase_average"])))
    return rows


def write_meta(rows) -> None:
    META_PATH.write_text(
        "curve,acceptance,phase_average\n"
        + "\n".join(f"{name},{acc:.8f},{den:.8f}" for name, acc, den in rows)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    use_archive = "--recompute" not in sys.argv

    if use_archive and IMAGE_SOURCE_PATH.exists():
        shutil.copyfile(IMAGE_SOURCE_PATH, FIGURE_PATH)
        rows = load_meta_rows()
        write_meta(rows)
        print(f"wrote {FIGURE_PATH}")
        print(f"wrote {META_PATH}")
        for name, acc, den in rows:
            print(f"{name}: HMC acceptance = {acc:.6f}, <cos(beta theta)> = {den:.6f}")
        return

    if use_archive and DATA_PATH.exists():
        times, quantum, corr1, corr3, corr5, rows = load_archived_run()
    else:
        times = np.linspace(0.0, 20.0, 151)
        quantum = quantum_kubo_qcorr(times)
        corr1, acc1, den1 = matsubara_corr(1, times, n_sample=8000, seed=11)
        corr3, acc3, den3 = matsubara_corr(3, times, n_sample=12000, seed=13)
        corr5, acc5, den5 = matsubara_corr(5, times, n_sample=60000, seed=15)
        rows = [
            ("M1", acc1, den1),
            ("M3", acc3, den3),
            ("M5_noisy", acc5, den5),
        ]

    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=180)
    ax.plot(times, quantum, color="black", lw=2.2, label="Quantum")
    ax.plot(times, corr1, color="red", ls=":", lw=1.6, label="M = 1")
    ax.plot(times, corr3, color="red", ls="--", lw=1.6, label="M = 3")
    ax.plot(times, corr5, color="red", ls="-.", lw=1.4, label="M = 5 (noisy)")
    ax.set_xlabel("t / a.u.")
    ax.set_ylabel(r"$C_{qq}^{[M]}(t)/Z$")
    ax.set_xlim(0.0, 20.0)
    ax.set_ylim(-0.55, 0.55)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=220)
    write_meta(rows)

    print(f"wrote {FIGURE_PATH}")
    print(f"wrote {META_PATH}")
    for name, acc, den in rows:
        print(f"{name}: HMC acceptance = {acc:.6f}, <cos(beta theta)> = {den:.6f}")


if __name__ == "__main__":
    main()
