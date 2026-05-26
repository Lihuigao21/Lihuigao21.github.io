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
from numba import jit
from typing import Callable, List, Tuple


@jit(nopython=True)
def get_bead_action(
    potential: Callable[[float], float],
    x_plus: float,
    x_minus: float,
    x_center: float,
    mass: float,
    beta_N: float,
) -> float:
    """
    Compute the actions.

    Every time we only move one bead, so the action only
    depends on the two neighboring beads and the current bead.

    S = 0.5 * m / beta_N * [ ... (x_{i+1} - x_i)^2 + (x_{i-1} - x_i)^2 ... ]
        + [ ... + V(x_i) + ...] * beta_N
    """
    S = (
        0.5 * mass / beta_N * ((x_plus - x_center) ** 2 + (x_minus - x_center) ** 2)
        + potential(x_center) * beta_N
    )
    return S


@jit(nopython=True)
def run_pimc_sweep(
    path: np.ndarray,
    N_beads: int,
    step_size: float,
    potential: Callable[[float], float],
    mass: float,
    beta: float,
) -> Tuple[np.ndarray, float]:
    """
    Perform a single Monte Carlo sweep over all beads in the path.

    Args:
        path (np.ndarray): Current path (positions of all beads).
        potential (Callable[[float], float]): Potential energy function V(x).

    Returns:
        Tuple[np.ndarray, float]:
            - Updated path as a NumPy array.
            - Acceptance rate for the sweep as a float.
    """
    n_accepted = 0
    current_path = path.copy()
    beta_N = beta / N_beads

    # Iterate through beads in a random order for one sweep
    for bead_index in np.random.permutation(N_beads):
        
        # Get neighbors with periodic boundary conditions
        prev_bead = (bead_index - 1 + N_beads) % N_beads
        next_bead = (bead_index + 1) % N_beads

        # Propose a move for the current bead
        current_pos = current_path[bead_index]
        proposed_pos = current_pos + (np.random.rand() - 0.5) * 2 * step_size

        # Calculate action for old and new positions
        old_action = get_bead_action(
            potential, current_path[next_bead], current_path[prev_bead], current_pos, mass, beta_N
        )
        new_action = get_bead_action(
            potential, current_path[next_bead], current_path[prev_bead], proposed_pos, mass, beta_N
        )
        
        delta_action = new_action - old_action

        # Metropolis acceptance criterion
        if delta_action < 0 or np.random.rand() < np.exp(-delta_action):
            current_path[bead_index] = proposed_pos
            n_accepted += 1

    acceptance_rate = n_accepted / N_beads
    return current_path, acceptance_rate


@jit(nopython=True)
def pimc_sample(
    N_beads: int,
    step_size: float,
    potential: Callable[[float], float],
    mass: float,
    beta: float,
    callback: Callable[[np.ndarray], float],
    n_iterations: int = 10000,
    n_sweeps: int = 2000,
    burn_in: int = 1000,
) -> List[float]:
    """
    Perform Path Integral Monte Carlo (PIMC) sampling.

    Args:
        potential (Callable[[float], float]): Potential energy function V(x).
        callback (Callable[[np.ndarray], float]): Function to process the path after burn-in.
        burn_in (int, optional): Number of sweeps to discard as burn-in.

    Returns:
        List[float]: List of callback results for each sampled path after burn-in.
    """
    ret = []
    for _ in range(n_iterations):
        path = np.zeros(N_beads)
        for i in range(n_sweeps):
            path, _ = run_pimc_sweep(path, N_beads, step_size, potential, mass, beta)
            if i >= burn_in:
                ret.append(callback(path))
    return ret
