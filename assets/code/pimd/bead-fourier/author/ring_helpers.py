# SPDX-License-Identifier: GPL-3.0-only
"""Ring normal-mode and split-Rhat helpers extracted from ToyModel."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _positive_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar.")
    return result


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def beta_per_bead(beta: float, beads: int) -> float:
    r"""Return :math:`\beta_P=\beta/P` for a ring polymer."""

    beta = _positive_scalar(beta, "beta")
    beads = _positive_integer(beads, "beads")
    return beta / beads


def ring_polymer_frequency(
    beta: float, beads: int, *, hbar: float = 1.0
) -> float:
    r"""Return :math:`\omega_P=P/(\beta\hbar)`."""

    beta = _positive_scalar(beta, "beta")
    beads = _positive_integer(beads, "beads")
    hbar = _positive_scalar(hbar, "hbar")
    return beads / (beta * hbar)


def real_normal_mode_matrix(beads: int) -> np.ndarray:
    r"""Return a real orthogonal Fourier matrix for a cyclic ring polymer.

    Columns are ordered as centroid, cosine/sine pairs with increasing
    Fourier index, and the Nyquist column last when beads is even.
    Bead coordinates and mode coordinates obey q = C @ Q and Q = C.T @ q.
    """

    beads = _positive_integer(beads, "beads")
    bead_indices = np.arange(beads, dtype=float)
    columns = [np.full(beads, 1.0 / np.sqrt(beads))]
    for index in range(1, (beads - 1) // 2 + 1):
        angle = 2.0 * np.pi * index * bead_indices / beads
        columns.append(np.sqrt(2.0 / beads) * np.cos(angle))
        columns.append(np.sqrt(2.0 / beads) * np.sin(angle))
    if beads % 2 == 0:
        columns.append((-1.0) ** bead_indices / np.sqrt(beads))
    return np.column_stack(columns)


def real_normal_mode_frequencies(
    beta: float, beads: int, *, hbar: float = 1.0
) -> np.ndarray:
    """Return frequencies aligned with the real normal-mode matrix."""

    beads = _positive_integer(beads, "beads")
    omega_p = ring_polymer_frequency(beta, beads, hbar=hbar)
    frequencies = [0.0]
    for index in range(1, (beads - 1) // 2 + 1):
        frequency = 2.0 * omega_p * np.sin(np.pi * index / beads)
        frequencies.extend((frequency, frequency))
    if beads % 2 == 0:
        frequencies.append(2.0 * omega_p)
    return np.asarray(frequencies, dtype=float)


def split_rhat(chains: ArrayLike) -> float:
    """Return the conventional split-Rhat diagnostic for scalar chains.

    ``chains`` has shape ``(n_chains, n_draws)``.  If ``n_draws`` is odd, the
    central draw is omitted before the first and last halves are formed.
    Constant split chains are rejected because their Rhat is undefined.
    """

    values = np.asarray(chains, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 4:
        raise ValueError(
            "chains must have shape (at least 2 chains, at least 4 draws)."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("chains must be finite.")
    half = values.shape[1] // 2
    split = np.concatenate((values[:, :half], values[:, -half:]), axis=0)
    within = float(np.mean(np.var(split, axis=1, ddof=1)))
    if within <= 0.0 or not np.isfinite(within):
        raise ValueError("split-Rhat is undefined for zero within-chain variance.")
    between = half * float(np.var(np.mean(split, axis=1), ddof=1))
    variance = ((half - 1.0) / half) * within + between / half
    return float(np.sqrt(variance / within))
