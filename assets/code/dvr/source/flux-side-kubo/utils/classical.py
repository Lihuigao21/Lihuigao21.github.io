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
def run_metropolis_sampling(
    n_samples: int,
    potential: Callable[[float], float],
    step_size: float = 0.4,
    beta: float = 2.0,
    burn_in: int = 5000,
    star_x: float = 0.0,
) -> List[float]:
    """
    Perform Metropolis sampling for a 1D position distribution.

    Samples positions according to P(x) ~ exp(-beta * V(x)).

    Args:
        potential (Callable[[float], float]): Potential energy function V(x).

    Returns:
        List[float]: List of sampled positions after burn-in.
    """
    samples_x = []
    accepted_moves = 0
    x_current = star_x  # Starting position

    for i in range(n_samples + burn_in):
        # Propose a new position by random displacement
        x_proposed = x_current + np.random.uniform(-step_size, step_size)

        # Compute change in potential energy
        delta_V = potential(x_proposed) - potential(x_current)

        # Accept move if energy decreases or with Boltzmann probability
        if delta_V < 0 or np.random.rand() < np.exp(-beta * delta_V):
            x_current = x_proposed
            if i >= burn_in:
                accepted_moves += 1

        # Store sample after burn-in period
        if i >= burn_in:
            samples_x.append(x_current)
    print("Acceptance rate: ", 100 * accepted_moves / n_samples, "%")
    return samples_x


@jit(nopython=True)
def run_verlet_step(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: float,
    dV: Callable[[np.ndarray], np.ndarray],
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Velocity Verlet algorithm for NVE (constant energy) simulation.

    Args:
        dV (Callable[[np.ndarray], np.ndarray]): Function returning the derivative of potential energy w.r.t. position.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Updated positions and velocities.
    """
    d_vel = -dV(pos) / mass
    pos = pos + vel * dt + 0.5 * d_vel * (dt**2)
    d_vel_new = -dV(pos) / mass
    vel = vel + (d_vel + d_vel_new) * dt / 2
    return pos, vel


@jit(nopython=True)
def run_nve_simulation(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: float,
    dV: Callable[[np.ndarray], np.ndarray],
    dt: float,
    n_steps: int,
) -> np.ndarray:
    """
    Run an NVE simulation using the velocity Verlet algorithm.

    Args:
        pos (np.ndarray): Initial positions of particles.
        vel (np.ndarray): Initial velocities of particles.
        dV (Callable[[np.ndarray], np.ndarray]): Function returning the derivative of potential energy w.r.t. position.

    Returns:
        np.ndarray: Trajectory of positions over time (shape: [n_steps, len(pos)]).
    """
    trajectory = np.zeros((n_steps, len(pos)))
    trajectory[0] = pos
    for step in range(1, n_steps):
        pos, vel = run_verlet_step(pos, vel, mass, dV, dt)
        trajectory[step] = pos
    return trajectory
