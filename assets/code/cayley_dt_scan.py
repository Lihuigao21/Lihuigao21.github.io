"""Reproduce the ring-polymer dt scan used in the Cayley note.

The script compares two choices for the free ring-polymer propagation step:

1. exact normal-mode propagation,
2. the Cayley approximation to the same harmonic substep.

It saves two figures:

- exact-dt-scan-recomputed.png
- cayley-dt-scan-recomputed.png

Dependencies: numpy, matplotlib.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def normal_mode_matrix(n_beads: int) -> np.ndarray:
    """Return an orthonormal real normal-mode transform matrix."""
    if n_beads % 2 != 0:
        raise ValueError("This simple real basis assumes an even bead number.")

    bead_index = np.arange(n_beads)
    rows = [np.ones(n_beads) / np.sqrt(n_beads)]

    for mode in range(1, n_beads // 2):
        angle = 2.0 * np.pi * mode * bead_index / n_beads
        rows.append(np.sqrt(2.0 / n_beads) * np.cos(angle))
        rows.append(np.sqrt(2.0 / n_beads) * np.sin(angle))

    rows.append(((-1.0) ** bead_index) / np.sqrt(n_beads))
    return np.vstack(rows)


def ring_polymer_frequencies(n_beads: int, beta: float = 1.0, hbar: float = 1.0) -> np.ndarray:
    """Return the normal-mode frequencies in the same order as normal_mode_matrix."""
    omega_p = n_beads / (beta * hbar)
    frequencies = [0.0]

    for mode in range(1, n_beads // 2):
        omega = 2.0 * omega_p * np.sin(mode * np.pi / n_beads)
        frequencies.extend([omega, omega])

    frequencies.append(2.0 * omega_p)
    return np.array(frequencies)


def exact_free_matrix(omega: float, dt: float, mass: float = 1.0) -> np.ndarray:
    """Exact free propagation matrix acting on the vector [p, q]."""
    if omega == 0.0:
        return np.array([[1.0, 0.0], [dt / mass, 1.0]])

    c = np.cos(omega * dt)
    s = np.sin(omega * dt)
    return np.array(
        [
            [c, -mass * omega * s],
            [s / (mass * omega), c],
        ]
    )


def cayley_free_matrix(omega: float, dt: float, mass: float = 1.0) -> np.ndarray:
    """Cayley free propagation matrix acting on the vector [p, q]."""
    if omega == 0.0:
        return np.array([[1.0, 0.0], [dt / mass, 1.0]])

    alpha_sq = (0.5 * dt * omega) ** 2
    denominator = 1.0 + alpha_sq
    return np.array(
        [
            [1.0 - alpha_sq, -dt * mass * omega**2],
            [dt / mass, 1.0 - alpha_sq],
        ]
    ) / denominator


def external_potential(q: np.ndarray, lambd: float = 1.0, mass: float = 1.0) -> float:
    """Harmonic external potential energy."""
    return float(np.sum(0.5 * mass * lambd**2 * q**2))


def external_force(q: np.ndarray, lambd: float = 1.0, mass: float = 1.0) -> np.ndarray:
    """Force from the harmonic external potential."""
    return -mass * lambd**2 * q


def total_energy(
    p: np.ndarray,
    q: np.ndarray,
    transform: np.ndarray,
    frequencies: np.ndarray,
    lambd: float = 1.0,
    mass: float = 1.0,
) -> float:
    """Compute kinetic + external potential + internal ring-polymer energy."""
    q_nm = transform @ q
    kinetic = np.sum(0.5 * p**2 / mass)
    external = external_potential(q, lambd=lambd, mass=mass)
    internal = np.sum(0.5 * mass * frequencies**2 * q_nm**2)
    return float((kinetic + external + internal) / len(q))


def integrate_one_step(
    p: np.ndarray,
    q: np.ndarray,
    dt: float,
    transform: np.ndarray,
    frequencies: np.ndarray,
    method: str,
    lambd: float = 1.0,
    mass: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """One velocity-Verlet-like ring-polymer timestep."""
    p = p + 0.5 * dt * external_force(q, lambd=lambd, mass=mass)

    p_nm = transform @ p
    q_nm = transform @ q

    for idx, omega in enumerate(frequencies):
        if method == "exact":
            matrix = exact_free_matrix(omega, dt, mass=mass)
        elif method == "cayley":
            matrix = cayley_free_matrix(omega, dt, mass=mass)
        else:
            raise ValueError(f"Unknown method: {method}")

        p_nm[idx], q_nm[idx] = matrix @ np.array([p_nm[idx], q_nm[idx]])

    p = transform.T @ p_nm
    q = transform.T @ q_nm
    p = p + 0.5 * dt * external_force(q, lambd=lambd, mass=mass)
    return p, q


def scan_dt(
    method: str,
    dt_values: np.ndarray,
    n_steps: int = 300,
    n_beads: int = 6,
    seed: int = 7,
    blowup_ratio: float = 1.0e6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return final/initial energy ratios for a dt scan."""
    rng = np.random.default_rng(seed)
    transform = normal_mode_matrix(n_beads)
    frequencies = ring_polymer_frequencies(n_beads)

    q_initial = -np.ones(n_beads)
    p_initial = rng.normal(loc=0.0, scale=np.sqrt(n_beads), size=n_beads)

    ratios = []
    for dt in dt_values:
        p = p_initial.copy()
        q = q_initial.copy()
        e0 = total_energy(p, q, transform, frequencies)
        ratio = 1.0

        for _ in range(n_steps):
            p, q = integrate_one_step(p, q, dt, transform, frequencies, method)
            ratio = total_energy(p, q, transform, frequencies) / e0
            if ratio > blowup_ratio:
                break

        ratios.append(ratio)

    return dt_values, np.array(ratios)


def resonance_timesteps(n_beads: int = 6, dt_max: float = 1.0) -> list[float]:
    """Predicted exact-propagation eigenvalue collision timesteps."""
    frequencies = ring_polymer_frequencies(n_beads)
    positive_unique = sorted({round(float(w), 12) for w in frequencies if w > 0.0})

    values = []
    for omega in positive_unique:
        multiple = 1
        while True:
            dt = np.pi * multiple / omega
            if dt_max >= dt:
                values.append(float(dt))
                multiple += 1
            else:
                break

    return sorted(set(round(value, 8) for value in values))


def plot_scan(method: str, filename: str) -> None:
    """Run one scan and save the figure."""
    dt_values = np.linspace(0.001, 1.0, 2000)
    dt_values, ratios = scan_dt(method, dt_values)
    resonances = resonance_timesteps()

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(dt_values, ratios, label="E_final / E_initial")

    for idx, dt in enumerate(resonances):
        ax.axvline(
            dt,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label="predicted resonance" if idx == 0 else None,
        )

    ax.set_yscale("log")
    ax.set_ylim(1.0, 1.0e6)
    ax.set_xlabel("dt")
    ax.set_ylabel("E_final / E_initial")
    ax.set_title(f"{method.capitalize()} free ring-polymer propagation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def main() -> None:
    plot_scan("exact", "exact-dt-scan-recomputed.png")
    plot_scan("cayley", "cayley-dt-scan-recomputed.png")


if __name__ == "__main__":
    main()
