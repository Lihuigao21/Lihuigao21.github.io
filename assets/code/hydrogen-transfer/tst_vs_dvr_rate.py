"""Compare one-dimensional transition-state theory with a sinc-DVR benchmark.

The model is an Eckart (Pöschl–Teller) barrier,

    V(x) = V0 / cosh(a x)^2.

For a one-dimensional scattering coordinate it is convenient to compare the
thermal *flux numerator* k Q_R.  Dividing both results by the same reactant
partition function Q_R would give the corresponding rate constants, so their
ratio is the transmission coefficient k_DVR / k_TST.

The quantum benchmark uses a sinc-DVR Hamiltonian in a large box and evaluates
the Boltzmann flux–side correlation function.  Its pre-recurrence plateau is
checked against the analytic transmission probability of this particular
barrier.  The analytic result is a validation aid, not an input to the DVR
calculation.

Run:
    python assets/code/hydrogen-transfer/tst_vs_dvr_rate.py

Outputs:
    assets/data/hydrogen-transfer/tst-vs-dvr-rate.csv
    assets/img/hydrogen-transfer/tst-vs-dvr-rate.png
    assets/img/hydrogen-transfer/tst-vs-dvr-rate-mobile.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import scipy.linalg


HARTREE_TO_KCAL_MOL = 627.509474
KB_AU = 3.166811563e-6
AU_TIME_TO_FS = 0.024188843
AU_TIME_TO_S = AU_TIME_TO_FS * 1.0e-15

MASS = 1836.152673
V0 = 0.010
A = 1.50
BOX = (-35.0, 35.0)
N_GRID = 701
N_STATES = 300
TEMPERATURES = np.array([250.0, 300.0, 400.0, 500.0])


def potential(x: np.ndarray) -> np.ndarray:
    """Eckart barrier in Hartree."""
    return V0 / np.cosh(A * x) ** 2


def sinc_kinetic(n_grid: int, bound: tuple[float, float], mass: float):
    """Return the sinc-DVR kinetic matrix, grid, and spacing (atomic units)."""
    x = np.linspace(bound[0], bound[1], n_grid)
    dx = (bound[1] - bound[0]) / (n_grid - 1)
    offsets = np.arange(1, n_grid)
    first_row = np.concatenate(
        ([np.pi**2 / 3.0], 2.0 * (-1.0) ** offsets / offsets**2)
    )
    kinetic = scipy.linalg.toeplitz(first_row / (2.0 * mass * dx**2))
    return kinetic, x, dx


def dvr_spectrum():
    """Diagonalize the boxed sinc-DVR Hamiltonian below the chosen cutoff."""
    kinetic, x, _ = sinc_kinetic(N_GRID, BOX, MASS)
    hamiltonian = kinetic + np.diag(potential(x))
    energy, vectors = scipy.linalg.eigh(
        hamiltonian, subset_by_index=(0, N_STATES - 1)
    )
    side = np.diag((x > 0.0).astype(float))
    side_energy = vectors.T @ side @ vectors
    return x, energy, side_energy


def flux_side_curve(
    energy: np.ndarray,
    side_energy: np.ndarray,
    temperature: float,
    time: np.ndarray,
) -> np.ndarray:
    """Return Re Tr[exp(-beta H) F h(t)] for F = i[H,h]."""
    beta = 1.0 / (KB_AU * temperature)
    delta = energy[:, None] - energy[None, :]
    amplitude = (
        np.exp(-beta * energy[:, None])
        * 1j
        * delta
        * np.abs(side_energy) ** 2
    )
    curve = np.empty_like(time)
    for index, instant in enumerate(time):
        curve[index] = np.real(np.sum(amplitude * np.exp(-1j * delta * instant)))
    return curve


def eckart_transmission(energy: np.ndarray) -> np.ndarray:
    """Exact transmission probability for V0 sech^2(a x), with hbar = 1."""
    energy = np.asarray(energy)
    k = np.sqrt(2.0 * MASS * np.maximum(energy, 0.0))
    numerator = np.sinh(np.pi * k / A) ** 2
    barrier_term = np.cosh(
        0.5 * np.pi * np.sqrt(8.0 * MASS * V0 / A**2 - 1.0)
    ) ** 2
    return numerator / (numerator + barrier_term)


def exact_flux(temperature: float) -> float:
    """Thermal quantum flux numerator from the analytic transmission curve."""
    beta = 1.0 / (KB_AU * temperature)
    integrand = lambda energy: np.exp(-beta * energy) * eckart_transmission(energy)
    integral, _ = scipy.integrate.quad(integrand, 0.0, V0 + 40.0 / beta, epsabs=1e-15)
    return integral / (2.0 * np.pi)


def tst_flux(temperature: float) -> float:
    """Classical transition-state flux numerator k_TST Q_R in atomic units."""
    beta = 1.0 / (KB_AU * temperature)
    return np.exp(-beta * V0) / (2.0 * np.pi * beta)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    data_path = root / "assets/data/hydrogen-transfer/tst-vs-dvr-rate.csv"
    figure_path = root / "assets/img/hydrogen-transfer/tst-vs-dvr-rate.png"
    mobile_figure_path = (
        root / "assets/img/hydrogen-transfer/tst-vs-dvr-rate-mobile.png"
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    x, energy, side_energy = dvr_spectrum()
    time = np.linspace(0.0, 12000.0, 301)
    plateau_window = (time >= 1800.0) & (time <= 4200.0)

    rows = []
    curves = {}
    for temperature in TEMPERATURES:
        curve = flux_side_curve(energy, side_energy, temperature, time)
        dvr_value = float(np.mean(curve[plateau_window]))
        dvr_std = float(np.std(curve[plateau_window]))
        tst_value = tst_flux(temperature)
        exact_value = exact_flux(temperature)
        curves[temperature] = curve
        rows.append(
            (
                temperature,
                tst_value,
                dvr_value,
                dvr_std,
                exact_value,
                tst_value / AU_TIME_TO_S,
                dvr_value / AU_TIME_TO_S,
                dvr_value / tst_value,
                exact_value / tst_value,
            )
        )

    header = (
        "temperature_K,tst_flux_au,dvr_flux_au,dvr_plateau_std_au,"
        "analytic_flux_au,tst_rate_s-1,dvr_rate_s-1,kappa_dvr,kappa_analytic"
    )
    with data_path.open("w", encoding="utf-8", newline="\n") as handle:
        np.savetxt(handle, np.asarray(rows), delimiter=",", header=header, comments="")

    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.15))
    ax = axes[0]
    ax.plot(x, potential(x) * HARTREE_TO_KCAL_MOL, color="#22324b", lw=2.2)
    ax.axvline(0.0, color="#bf5b42", lw=1.2, ls="--")
    ax.set_xlim(-4.0, 4.0)
    ax.set_xlabel(r"reaction coordinate $x$ (bohr)")
    ax.set_ylabel(r"$V(x)$ (kcal mol$^{-1}$)")
    ax.set_title("One-dimensional barrier")
    ax.text(0.05, 0.90, r"$V_0=6.28$ kcal mol$^{-1}$", transform=ax.transAxes)

    ax = axes[1]
    reference_temperature = 300.0
    exact_reference = exact_flux(reference_temperature)
    ax.plot(
        time * AU_TIME_TO_FS,
        curves[reference_temperature] / exact_reference,
        color="#22324b",
        lw=1.8,
    )
    ax.axhline(1.0, color="#bf5b42", lw=1.1, ls="--")
    ax.axvspan(
        time[plateau_window][0] * AU_TIME_TO_FS,
        time[plateau_window][-1] * AU_TIME_TO_FS,
        color="#bf5b42",
        alpha=0.13,
        label="averaging window",
    )
    ax.set_xlim(0.0, 210.0)
    ax.set_ylim(0.0, 1.18)
    ax.set_xlabel("correlation time (fs)")
    ax.set_ylabel(r"$C_{fs}(t)/F_{\rm exact}$")
    ax.set_title("DVR rate plateau at 300 K")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[2]
    temps = np.asarray([row[0] for row in rows])
    kappa_dvr = np.asarray([row[7] for row in rows])
    kappa_exact = np.asarray([row[8] for row in rows])
    ax.plot(temps, kappa_exact, color="#22324b", lw=2.0, label="analytic transmission")
    ax.scatter(temps, kappa_dvr, color="#bf5b42", s=48, zorder=3, label="sinc-DVR plateau")
    ax.axhline(1.0, color="#7a8491", lw=1.0, ls="--", label="TST")
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel(r"transmission factor $\kappa=k/k_{\rm TST}$")
    ax.set_title("The barrier top is not the whole rate")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(bottom=0.0)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.18)

    figure.tight_layout()
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")

    # The three panels would be unreadably small if the wide figure were merely
    # scaled to a phone screen, so the same data are also rendered vertically.
    mobile_figure, axes = plt.subplots(3, 1, figsize=(6.0, 11.8))
    ax = axes[0]
    ax.plot(x, potential(x) * HARTREE_TO_KCAL_MOL, color="#22324b", lw=2.2)
    ax.axvline(0.0, color="#bf5b42", lw=1.2, ls="--")
    ax.set_xlim(-4.0, 4.0)
    ax.set_xlabel(r"reaction coordinate $x$ (bohr)")
    ax.set_ylabel(r"$V(x)$ (kcal mol$^{-1}$)")
    ax.set_title("One-dimensional barrier")
    ax.text(0.05, 0.90, r"$V_0=6.28$ kcal mol$^{-1}$", transform=ax.transAxes)

    ax = axes[1]
    ax.plot(
        time * AU_TIME_TO_FS,
        curves[reference_temperature] / exact_reference,
        color="#22324b",
        lw=1.8,
    )
    ax.axhline(1.0, color="#bf5b42", lw=1.1, ls="--")
    ax.axvspan(
        time[plateau_window][0] * AU_TIME_TO_FS,
        time[plateau_window][-1] * AU_TIME_TO_FS,
        color="#bf5b42",
        alpha=0.13,
        label="averaging window",
    )
    ax.set_xlim(0.0, 210.0)
    ax.set_ylim(0.0, 1.18)
    ax.set_xlabel("correlation time (fs)")
    ax.set_ylabel(r"$C_{fs}(t)/F_{\rm exact}$")
    ax.set_title("DVR rate plateau at 300 K")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[2]
    ax.plot(temps, kappa_exact, color="#22324b", lw=2.0, label="analytic transmission")
    ax.scatter(temps, kappa_dvr, color="#bf5b42", s=48, zorder=3, label="sinc-DVR plateau")
    ax.axhline(1.0, color="#7a8491", lw=1.0, ls="--", label="TST")
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel(r"transmission factor $\kappa=k/k_{\rm TST}$")
    ax.set_title("The barrier top is not the whole rate")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(bottom=0.0)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.18)

    mobile_figure.tight_layout(h_pad=2.4)
    mobile_figure.savefig(mobile_figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    plt.close(mobile_figure)

    print(header)
    for row in rows:
        print(",".join(f"{value:.8e}" for value in row))
    print(f"saved {data_path.relative_to(root).as_posix()}")
    print(f"saved {figure_path.relative_to(root).as_posix()}")
    print(f"saved {mobile_figure_path.relative_to(root).as_posix()}")


if __name__ == "__main__":
    main()
