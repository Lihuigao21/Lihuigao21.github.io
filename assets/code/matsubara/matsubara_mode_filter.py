"""Fourier-mode filtering demo for Matsubara paths.

This script makes the result used in Matsubara Series Part II.  It constructs
a periodic imaginary-time path from many Fourier modes, truncates it to the
lowest Matsubara modes, and reports how much spectral power is retained.

Dependencies: numpy, matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
FIGURE_PATH = ROOT / "assets" / "img" / "matsubara-series" / "matsubara-mode-filter.png"


def make_path(n_modes: int = 50, n_grid: int = 512, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tau = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    path = np.zeros_like(tau)
    coefficients = []

    for k in range(1, n_modes + 1):
        envelope = 1.0 / (k ** 0.85)
        amp_sin = rng.normal(scale=envelope)
        amp_cos = rng.normal(scale=envelope)
        path += amp_sin * np.sin(2.0 * np.pi * k * tau)
        path += amp_cos * np.cos(2.0 * np.pi * k * tau)
        coefficients.append((k, amp_sin, amp_cos))

    path += 0.4
    return tau, path, np.asarray(coefficients)


def reconstruct_low_modes(tau: np.ndarray, coefficients: np.ndarray, m_keep: int) -> np.ndarray:
    k_max = (m_keep - 1) // 2
    low = np.full_like(tau, 0.4)
    for k, amp_sin, amp_cos in coefficients:
        if k <= k_max:
            low += amp_sin * np.sin(2.0 * np.pi * k * tau)
            low += amp_cos * np.cos(2.0 * np.pi * k * tau)
    return low


def main() -> None:
    n_modes = 50
    m_keep = 11
    tau, full_path, coefficients = make_path(n_modes=n_modes)
    low_path = reconstruct_low_modes(tau, coefficients, m_keep=m_keep)

    power = coefficients[:, 1] ** 2 + coefficients[:, 2] ** 2
    retained = int((m_keep - 1) // 2)
    retained_fraction = float(power[:retained].sum() / power.sum())
    rms_removed = float(np.sqrt(np.mean((full_path - low_path) ** 2)))

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), constrained_layout=True)

    axes[0].plot(tau, full_path, color="#1f77b4", lw=1.1, label=f"all modes (N={2 * n_modes + 1})")
    axes[0].plot(tau, low_path, color="#ff7f0e", lw=2.0, label=f"low modes only (M={m_keep})")
    axes[0].set_xlabel(r"imaginary time $\tau/\beta$")
    axes[0].set_ylabel(r"$q(\tau)$")
    axes[0].set_title("Low-mode truncation smooths the path")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].bar(coefficients[:, 0], power, color="#9aa6b2", width=0.85, label="discarded")
    axes[1].bar(coefficients[:retained, 0], power[:retained], color="#315f7d", width=0.85, label="retained")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Fourier mode index k")
    axes[1].set_ylabel("mode power")
    axes[1].set_title("Matsubara subspace as a low-pass filter")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(frameon=False)

    fig.savefig(FIGURE_PATH, dpi=190)
    print(f"wrote {FIGURE_PATH}")
    print(f"retained power fraction = {retained_fraction:.6f}")
    print(f"RMS removed high-frequency component = {rms_removed:.6f}")


if __name__ == "__main__":
    main()
