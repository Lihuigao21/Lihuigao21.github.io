"""Small SU(N) diagnostics used by the spin-mapping foundations note.

The script constructs the generalized Gell-Mann basis S_i = lambda_i / 2,
checks the normalization tr(S_i S_j) = delta_ij / 2, checks the quadratic
Casimir sum_i S_i^2 = (N^2 - 1) I / (2N), and prints the W-representation
spin-mapping zero-point-energy parameter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SUNCheck:
    n: int
    generators: int
    pure_state_dof: int
    bloch_components: int
    casimir_target: float
    casimir_error: float
    orthogonality_error: float
    gamma_w: float
    r2_w: float


def su_n_generators(n: int) -> list[np.ndarray]:
    """Return generalized Gell-Mann generators with tr(S_i S_j)=delta_ij/2."""
    if n < 2:
        raise ValueError("n must be at least 2")

    generators: list[np.ndarray] = []

    for j in range(n):
        for k in range(j + 1, n):
            symmetric = np.zeros((n, n), dtype=complex)
            symmetric[j, k] = symmetric[k, j] = 0.5
            generators.append(symmetric)

            antisymmetric = np.zeros((n, n), dtype=complex)
            antisymmetric[j, k] = -0.5j
            antisymmetric[k, j] = 0.5j
            generators.append(antisymmetric)

    for ell in range(1, n):
        diagonal = np.zeros((n, n), dtype=complex)
        coeff = 1.0 / math.sqrt(2.0 * ell * (ell + 1.0))
        diagonal[:ell, :ell] += np.eye(ell) * coeff
        diagonal[ell, ell] = -ell * coeff
        generators.append(diagonal)

    return generators


def check_su_n(n: int) -> SUNCheck:
    generators = su_n_generators(n)
    identity = np.eye(n, dtype=complex)

    gram = np.array(
        [[np.trace(a @ b).real for b in generators] for a in generators]
    )
    orthogonality_error = float(
        np.max(np.abs(gram - 0.5 * np.eye(len(generators))))
    )

    casimir = sum((g @ g for g in generators), start=np.zeros((n, n), dtype=complex))
    casimir_target = (n * n - 1.0) / (2.0 * n)
    casimir_error = float(np.max(np.abs(casimir - casimir_target * identity)))

    gamma_w = (2.0 / n) * (math.sqrt(n + 1.0) - 1.0)
    r2_w = 2.0 * math.sqrt(n + 1.0)

    return SUNCheck(
        n=n,
        generators=len(generators),
        pure_state_dof=2 * n - 2,
        bloch_components=n * n - 1,
        casimir_target=casimir_target,
        casimir_error=casimir_error,
        orthogonality_error=orthogonality_error,
        gamma_w=gamma_w,
        r2_w=r2_w,
    )


def main() -> None:
    rows = [check_su_n(n) for n in (2, 3, 7, 8)]
    print(
        "N  generators  pure_dof  bloch_components  Casimir       "
        "gamma_W      R_W^2        max_errors"
    )
    for row in rows:
        print(
            f"{row.n:<2d} {row.generators:<10d} {row.pure_state_dof:<9d} "
            f"{row.bloch_components:<17d} {row.casimir_target:<12.8f} "
            f"{row.gamma_w:<12.8f} {row.r2_w:<12.8f} "
            f"{max(row.casimir_error, row.orthogonality_error):.2e}"
        )


if __name__ == "__main__":
    main()
