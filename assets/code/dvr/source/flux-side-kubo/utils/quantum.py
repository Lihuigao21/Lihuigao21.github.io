# Copyright 2025 Zhe Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import scipy
from typing import Callable, Tuple
import scipy.linalg
import scipy.integrate

import numpy as np
from typing import Callable, Any


def eval_V_on_grid_general(
    V_func: Callable[[np.ndarray], np.ndarray],
    xx: np.ndarray,
    nstates: int,
    *,
    ndim: int = 1,
    grid_to_X: Callable[[float], np.ndarray] | None = None,
) -> np.ndarray:
    """
    Evaluate a general model potential on a 1D DVR grid and return Vgrid with shape (nstates,nstates,ndvr).

    Parameters
    ----------
    V_func: callable
        Function V(X) returning a potential matrix for a nuclear configuration X.
        Supported outputs per call:
          - (nstates, nstates)
          - (nstates,) or scalar for nstates==1 (will be promoted)
    xx: (ndvr,) array
        1D DVR grid.
    nstates: int
        Number of electronic states.
    ndim: int
        Nuclear dimension. Default 1. For ndim>1, provide grid_to_X.
    grid_to_X: callable or None
        Maps a grid point x (float) -> X (ndim,) array. If None and ndim==1, uses X=[x].

    Returns
    -------
    Vgrid: (nstates,nstates,ndvr) float64
    """
    ndvr = xx.shape[0]
    Vgrid = np.zeros((nstates, nstates, ndvr), dtype=np.float64)

    if grid_to_X is None:
        if ndim != 1:
            raise ValueError(
                "ndim != 1 requires grid_to_X to map grid coordinate to X."
            )
        grid_to_X = lambda x: np.array([x], dtype=np.float64)

    for i, x in enumerate(xx):
        X = grid_to_X(float(x))
        Vij = np.asarray(V_func(X), dtype=np.float64)

        # Normalize Vij to (nstates,nstates)
        if Vij.ndim == 0:
            if nstates != 1:
                raise ValueError("V returned scalar but nstates != 1.")
            Vij = Vij.reshape(1, 1)
        elif Vij.ndim == 1:
            # allow diagonal-only specification
            if Vij.shape[0] == nstates:
                Vij = np.diag(Vij)
            else:
                raise ValueError(
                    f"V returned shape {Vij.shape}, cannot promote to ({nstates},{nstates})."
                )
        elif Vij.ndim == 2:
            if Vij.shape != (nstates, nstates):
                raise ValueError(
                    f"V returned shape {Vij.shape}, expected ({nstates},{nstates})."
                )
        else:
            raise ValueError(
                f"V returned ndim={Vij.ndim}, expected 0/1/2 per grid point."
            )

        Vgrid[:, :, i] = Vij

    # Optional: enforce Hermiticity (recommended for numerical stability)
    Vgrid = 0.5 * (Vgrid + np.swapaxes(Vgrid, 0, 1))
    return Vgrid


def build_sinc_kin_1d(ndvr: int, bound: Tuple[float, float], mass: float):
    dx = (bound[1] - bound[0]) / (ndvr - 1)
    xx = np.linspace(bound[0], bound[1], ndvr)

    neq = np.ones(ndvr - 1)
    neq[0::2] = -1
    kin = np.concatenate(
        [np.array([np.pi**2 / 3.0]), 2.0 * neq / (np.arange(1, ndvr) ** 2)]
    )
    T = scipy.linalg.toeplitz(kin / (mass * dx**2 * 2.0))
    return T, xx, dx


def solve_sinc_dvr_multistate_dense(
    V_func: Callable[[np.ndarray], np.ndarray],
    *,
    nstates: int,
    cutoff: int = 10,
    N_grids: int = 1025,
    bound: Tuple[float, float] = (-10.0, 10.0),
    mass: float = 2000.0,
    ndim: int = 1,
    grid_to_X: Callable[[float], np.ndarray] | None = None,
):
    ndvr = N_grids
    if ndvr < 2:
        raise ValueError("N_grids must be >= 2.")

    T, xx, dx = build_sinc_kin_1d(ndvr, bound, mass)
    Vgrid = eval_V_on_grid_general(
        V_func, xx, nstates, ndim=ndim, grid_to_X=grid_to_X
    )  # (n,n,ndvr)

    dim = ndvr * nstates
    H = np.zeros((dim, dim), dtype=np.float64)

    # kinetic blocks (T on diagonal blocks)
    for a in range(nstates):
        ia = slice(a * ndvr, (a + 1) * ndvr)
        H[ia, ia] += T

    # potential blocks: grid-diagonal
    for a in range(nstates):
        ia = slice(a * ndvr, (a + 1) * ndvr)
        for b in range(nstates):
            ib = slice(b * ndvr, (b + 1) * ndvr)
            H[ia, ib] += np.diag(Vgrid[a, b, :])

    ene, coef = scipy.linalg.eigh(H, subset_by_index=(0, cutoff - 1))

    # normalize total wavefunction: sum_a ∫ |psi_a(x)|^2 dx = 1
    Ninterv = ndvr - 1
    n = 1
    while n < Ninterv:
        n <<= 1
    integ = scipy.integrate.romb if n == Ninterv else scipy.integrate.simpson

    for k in range(coef.shape[1]):
        psi = coef[:, k].reshape(nstates, ndvr)
        dens = np.sum(psi**2, axis=0)
        norm2 = integ(dens, dx=dx)
        coef[:, k] /= np.sqrt(norm2)

    return ene, coef, xx


def get_Kubo_correlation(
    time: np.ndarray,
    energy: np.ndarray,
    A_matrix: np.ndarray,
    B_matrix: np.ndarray,
    beta: float = 1.0,
) -> np.ndarray:
    """
    Compute the quantum Kubo correlation function for operators A and B.

    Mathematically, it is defined as:
        K_AB(t) = (1 / Z / beta) ∫_0^beta dλ
                  Tr{e^(-(beta-λ) H) A e^(-λ H) B(t)}

    Args:
        time (np.ndarray): Array of time points.
        energy (np.ndarray): Array of energy eigenvalues.
        A_matrix (np.ndarray): Matrix elements of operator A in the energy basis <i|A|j>.
        B_matrix (np.ndarray): Matrix elements of operator B in the energy basis <i|B|j>.

    Returns:
        np.ndarray: Kubo correlation function evaluated at each time point.
    """
    # This is equivalent to:
    # K_AB(t) = (1 / Z / beta) * sum_{i,j} (e^(-beta E_j) - e^(-beta E_i)) / (E_i - E_j) *
    #               <i|A|j><j|B|i> * exp(-i (E_i - E_j) t)

    energy_diff = energy[:, np.newaxis] - energy[np.newaxis, :]  # E_i - E_j

    with np.errstate(divide="ignore", invalid="ignore"):
        # (e^(-beta E_j) - e^(-beta E_i)) / (E_i - E_j) / beta
        prefactor = (
            (np.exp(-beta * energy[:, np.newaxis]) - np.exp(-beta * energy)).T
            / energy_diff
            / beta
        )
    np.fill_diagonal(
        prefactor, np.exp(-beta * energy)
    )  # when i=j, limit -> beta * e^(-beta E_i) / beta

    # exp(-i (E_i - E_j) t)
    time_factor = np.exp(
        -1j * energy_diff[:, :, np.newaxis] * time[np.newaxis, np.newaxis, :]
    )

    # Z (partition function)
    partition = np.exp(-beta * energy).sum()

    kubo = (
        np.einsum("ij, ij, ji, ijt -> t", prefactor, A_matrix, B_matrix, time_factor)
        / partition
    )
    return kubo
