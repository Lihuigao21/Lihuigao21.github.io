"""Excited-ground population-resolved Kubo correlation in a two-state DVR model.

This is the cleaned implementation behind DVR Series Part VII.  It builds a
two-state Tully simple avoided-crossing Hamiltonian, constructs local adiabatic
projectors Pg and Pe, Kubo-dresses the excited-channel flux operator Pe F Pe,
and correlates it with either Pg(t) or dPg(t)/dt. The integrated dPg(t)/dt
signal is scaled by 2*pi*hbar*beta so it can be compared with a
flux-conditioned ground-state population.

The default grid is intentionally modest so the script runs as a smoke test on
a laptop while still covering the default flux surface. Increase --n-half and
reduce --dx to approach the production figures shown in the article.

Dependencies: numpy, scipy, matplotlib.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "assets" / "img" / "dvr-series" / "dvr6-population-kubo"


def diag(values: np.ndarray) -> np.ndarray:
    return np.diagflat(values)


def make_2state_block(a11: np.ndarray, a12: np.ndarray, a22: np.ndarray) -> np.ndarray:
    return np.block([[a11, a12], [a12, a22]])


def sinc_kinetic(n_grid: int, mass: float, dx: float) -> np.ndarray:
    """Sinc DVR kinetic matrix for hbar = 1."""
    sign = np.resize([1.0, -1.0], n_grid)
    kinetic = np.outer(sign, sign) / (2.0 * mass * dx**2)
    ij = np.subtract.outer(np.arange(n_grid), np.arange(n_grid)).astype(float)
    denom = ij**2 / 2.0 + np.eye(n_grid) * (3.0 / np.pi**2)
    kinetic /= denom
    return 0.5 * (kinetic + kinetic.T)


def tully_sac_potential(
    x: np.ndarray,
    *,
    a: float = 0.01,
    b: float = 1.6,
    c: float = 0.002,
    d: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tully simple avoided crossing in a diabatic basis."""
    temp = np.exp(-b * np.abs(x))
    v11 = a * (1.0 - temp) * np.sign(x)
    v22 = -v11
    v12 = c * np.exp(-d * x**2)
    return v11, v22, v12


def local_adiabatic_projectors(
    v11: np.ndarray,
    v22: np.ndarray,
    v12: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Pg and Pe in state-major DVR ordering [state 0 grid, state 1 grid]."""
    n_grid = len(v11)
    pg11 = np.zeros(n_grid)
    pg22 = np.zeros(n_grid)
    pg12 = np.zeros(n_grid)
    pe11 = np.zeros(n_grid)
    pe22 = np.zeros(n_grid)
    pe12 = np.zeros(n_grid)

    for j, elements in enumerate(zip(v11, v12, v22)):
        values, vectors = np.linalg.eigh(np.array([[elements[0], elements[1]], [elements[1], elements[2]]]))
        del values
        ground = vectors[:, 0]
        excited = vectors[:, 1]
        pg11[j], pg22[j], pg12[j] = ground[0] ** 2, ground[1] ** 2, ground[0] * ground[1]
        pe11[j], pe22[j], pe12[j] = excited[0] ** 2, excited[1] ** 2, excited[0] * excited[1]

    pg = make_2state_block(diag(pg11), diag(pg12), diag(pg22))
    pe = make_2state_block(diag(pe11), diag(pe12), diag(pe22))

    identity = np.eye(2 * n_grid)
    if not np.allclose(pg + pe, identity, atol=1e-10):
        raise RuntimeError("adiabatic projectors are not complete")
    return pg, pe


def side_operator(x: np.ndarray, surface: float) -> np.ndarray:
    h = diag((x >= surface).astype(float))
    zero = h * 0.0
    return make_2state_block(h, zero, h)


def flux_operator(hamiltonian: np.ndarray, operator: np.ndarray, hbar: float = 1.0) -> np.ndarray:
    return 1j / hbar * (hamiltonian @ operator - operator @ hamiltonian)


def kubo_kernel(energy: np.ndarray, beta: float) -> np.ndarray:
    """Analytic Kubo imaginary-time kernel in the energy basis."""
    energy_i = energy[:, None]
    energy_j = energy[None, :]
    gap = energy_j - energy_i
    numerator = np.exp(-beta * energy_i) - np.exp(-beta * energy_j)
    kernel = np.zeros_like(gap)
    mask = np.abs(gap) > 1e-12
    kernel[mask] = numerator[mask] / (beta * gap[mask])
    np.fill_diagonal(kernel, np.exp(-beta * energy))
    return kernel


def kubo_dress_operator(hamiltonian: np.ndarray, operator: np.ndarray, beta: float) -> np.ndarray:
    energy, coeff = scipy.linalg.eigh(hamiltonian)
    operator_e = coeff.conj().T @ operator @ coeff
    dressed_e = kubo_kernel(energy, beta) * operator_e
    return coeff @ dressed_e @ coeff.conj().T


def cap_profile(x: np.ndarray, x0: float, width: float) -> np.ndarray:
    return 2.0 / (1.0 + np.exp((x0 - np.abs(x)) / width))


def schur_correlation(hamiltonian_abs: np.ndarray, a_operator: np.ndarray, b_operator: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Compute Tr[A exp(+i H_abs^dag t) B exp(-i H_abs t)] with a Schur form."""
    t_schur, z_schur = scipy.linalg.schur(hamiltonian_abs, output="complex")
    helper_a = z_schur.conj().T @ a_operator @ z_schur.conj()
    helper_b = z_schur.T @ b_operator @ z_schur

    values = []
    for t in time:
        propagator = scipy.linalg.expm(-1j * float(t) * t_schur)
        values.append(np.einsum("qk,kp,pn,nq->", helper_a, propagator.conj(), helper_b, propagator, optimize=True))
    return np.asarray(values)


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=complex)
    for i in range(1, len(x)):
        out[i] = out[i - 1] + 0.5 * (y[i - 1] + y[i]) * (x[i] - x[i - 1])
    return out


def run(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n_grid = 2 * args.n_half
    x = (np.arange(-args.n_half, args.n_half) + 0.5) * args.dx
    beta = 315.7781721496283 * 1000.0 / args.temperature

    kinetic = sinc_kinetic(n_grid, args.mass, args.dx)
    v11, v22, v12 = tully_sac_potential(x, c=args.coupling)
    hamiltonian = make_2state_block(kinetic + diag(v11), diag(v12), kinetic + diag(v22))
    pg, pe = local_adiabatic_projectors(v11, v22, v12)

    side = side_operator(x, args.flux_surface)
    flux = flux_operator(hamiltonian, side)
    excited_flux = pe @ flux @ pe
    dressed_excited_flux = kubo_dress_operator(hamiltonian, excited_flux, beta)

    ground_flux = flux_operator(hamiltonian, pg)
    cap = cap_profile(x, args.cap_x0, args.cap_width)
    cap_full = make_2state_block(diag(cap), diag(cap) * 0.0, diag(cap))
    hamiltonian_abs = hamiltonian - 1j * args.cap_eta * cap_full

    time = np.linspace(0.0, args.t_max, args.n_time)
    c_pop = schur_correlation(hamiltonian_abs, dressed_excited_flux, pg, time)
    c_flux = schur_correlation(hamiltonian_abs, dressed_excited_flux, ground_flux, time)
    c_integrated = cumulative_trapezoid(c_flux, time)
    return time, c_pop, c_integrated, beta


def plot_results(time: np.ndarray, c_pop: np.ndarray, c_integrated: np.ndarray, beta: float, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10})

    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    ax.plot(time, np.real(c_pop), label="direct Re Cpop")
    ax.set_xlabel("t")
    ax.set_ylabel("Cpop")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "excited-ground-cpop-demo.png", dpi=180)
    plt.close(fig)

    scaled_population = 2.0 * np.pi * beta * np.real(c_integrated)

    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    ax.plot(time, scaled_population, label="scaled Kubo population")
    ax.set_xlabel("t")
    ax.set_ylabel("flux-conditioned ground-state population")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "excited-ground-scaled-pop-demo.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    ax.plot(time, np.real(c_integrated), label="unscaled integrated Re Cflux")
    ax.set_xlabel("t")
    ax.set_ylabel("integrated signal")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "excited-ground-integral-demo.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-half", type=int, default=90, help="Half the number of DVR grid points.")
    parser.add_argument("--dx", type=float, default=0.12, help="DVR grid spacing.")
    parser.add_argument("--mass", type=float, default=2000.0)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--coupling", type=float, default=0.002)
    parser.add_argument("--flux-surface", type=float, default=-10.0)
    parser.add_argument("--cap-eta", type=float, default=0.05)
    parser.add_argument("--cap-x0", type=float, default=8.0)
    parser.add_argument("--cap-width", type=float, default=1.0)
    parser.add_argument("--t-max", type=float, default=4000.0)
    parser.add_argument("--n-time", type=int, default=21)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    time, c_pop, c_integrated, beta = run(args)
    plot_results(time, c_pop, c_integrated, beta, args.output_dir)
    print(f"wrote figures to {args.output_dir}")
    print(f"final Re Cpop = {np.real(c_pop[-1]):.8e}")
    print(f"final integrated signal = {np.real(c_integrated[-1]):.8e}")
    print(f"final scaled population = {2.0 * np.pi * beta * np.real(c_integrated[-1]):.8e}")


if __name__ == "__main__":
    main()
