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


def solve_sinc_dvr(
    potential: Callable[[np.ndarray], np.ndarray],
    N_states: int = 10,
    N_grids: int = 1,
    bound: Tuple[float, float] = (-2, 2),
    mass: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the time-independent Schrödinger equation using the DVR method.

    Args:
        potential (Callable[[np.ndarray], np.ndarray]): Potential energy function V(x).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - ene: Array of energy eigenvalues.
            - coef: Matrix of eigenfunctions (each column corresponds to an eigenfunction).
            - xx: Grid points used (for convenience).
    """
    dx = float(bound[1] - bound[0]) / (N_grids - 1)
    xx = np.linspace(bound[0], bound[1], num=N_grids)

    # Sinc DVR
    # See Eq. 2.1 in https://doi.org/10.1063/1.462100
    neq = np.ones(N_grids - 1)
    neq[0::2] = -1
    kin = np.concatenate(
        [np.array([np.pi**2 / 3]), 2 * neq / (np.arange(1, N_grids) ** 2)]
    )
    kin = scipy.linalg.toeplitz(kin / (mass * dx**2 * 2))

    # The potential matrix is diagonal
    H = kin + np.diag(potential(xx))  # (N_grids, N_grids)
    # Solve Hc = ec (shift-invert mode)
    # We only need the lowest N_states eigenvalues/eigenvectors
    ene, coef = scipy.linalg.eigh(H, subset_by_index=(0, N_states - 1))

    # Normalize the eigenfunctions
    Ninterv = N_grids - 1
    n = 1
    while n < Ninterv:
        n <<= 1
    if n == Ninterv:
        fun = scipy.integrate.romb
    else:
        fun = scipy.integrate.simpson
    coef = np.divide(coef, np.sqrt(fun(coef.T**2, dx=dx)))

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
