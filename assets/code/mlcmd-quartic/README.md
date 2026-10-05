# One-dimensional direct and delta ML-CMD

This compact attachment contains the final training labels, independent
reference and validation labels, trained conservative networks, and complete
time series for a harmonic oscillator and a quartic single well. It is an
independent one-dimensional implementation of the force-learning idea, not
the original authors' software or a reproduction of their liquid benchmarks.

All quantities use dimensionless model units: mass = 1, hbar = 1, beta = 2.
The potentials are V(q) = q^2/2 and V(q) = q^2/2 + 0.1 q^4.

## Run from the website repository

Use Python 3.10 or later with NumPy, SciPy and Matplotlib:

```sh
python -m pip install numpy scipy matplotlib
python assets/code/mlcmd-quartic/reproduce.py --output mlcmd-output
```

The same script can be called from any working directory. It locates its
input files beside itself and imports only the accompanying learning module
and the declared Python dependencies. A ToyModel installation is not needed.

The default command does **not** sample paths, train networks, propagate
trajectories, or solve a quantum Hamiltonian. It:

- verifies the checksum of every supplied code/data file;
- loads the trained networks and checks force = minus energy derivative with
  an independent finite difference over the plotted interval;
- recomputes errors against independent labels, including centroid positions
  that were absent from training;
- verifies the saved ML-CMD/CMD and CMD/quantum curve differences;
- checks that all integration times are present;
- redraws learning diagnostics and the four-method correlations, writing six
  PNGs and `verification.json` to the requested output directory.

For each model, the outputs are `*_potentials.png` (one potential panel),
`*_force_validation.png` (force correction and residuals at the 40 new
validation positions), and `*_correlations.png` (all four correlations,
ML-CMD minus CMD, and CMD minus quantum). The light band in the first
correlation panel is the range of four independently
sampled reference-CMD curves. It is **not** a simultaneous confidence band.
Learning plots display q in [-3, 3]; the supplied tables and quantitative
checks retain discrete support points spanning [-8, 8]. Unseen-position
validation is confined to the 40 central midpoints. No curve smoothing is applied.

## Files and scientific meaning

For each model, the following files are supplied:

| File suffix | Contents |
|---|---|
| `_training.csv` | 51 fixed-centroid conditional mean forces, their sampling standard errors, and the coordinate mask. Both ML schemes used these same labels. |
| `_reference.csv` | Independently sampled mean forces on the same 51 coordinates, used to construct the reference CMD potential. |
| `_validation.csv` | Independently sampled forces on a 91-point grid, including 40 coordinates absent from training. |
| `_direct.json`, `_delta.json` | All parameters and normalization constants for the trained energy networks. |
| `_curves.csv` | All 801 integration times, t = 0 to 8 in increments of 0.01, for quantum Kubo, reference CMD, direct ML-CMD and delta ML-CMD. |
| `_cmd_chains.csv` | Four reference-CMD curves constructed from the four independent chains, on the same full time grid. |

`metadata.json` records model parameters, sampling/learning settings, quantum
grid choices, frozen numerical thresholds and the scalar checks reproduced by
this attachment. `checksums.json` records SHA-256 checksums of the public files.

Labels are constrained-PIMC **conditional mean forces**, not instantaneous
projected forces drawn from an unconstrained PIMD trajectory. Each centroid
has independent chains. Standard errors account for autocorrelation and
between-chain dispersion; this compact attachment does not include the raw
force and ring-width traces needed to independently recalculate those errors.

The production bead counts are 32 for the harmonic oscillator and 64 for the
quartic model; their stricter checks used 64 and 128, respectively. Refined
centroid-grid labels use the production bead count with independent random
streams. More beads test imaginary-time discretization; they do not remove
the real-time CMD approximation.

The direct network learns the centroid potential. The delta network learns
its difference from the bare PES. Both use the same 12-unit tanh energy
network, initialization, normalized force loss and evaluation budget. Forces
are analytic energy derivatives. The potential is defined only within its
declared training domain; no clipping or extrapolation is permitted.

Some stored fits reached the 2000-evaluation budget. Their independent force
and trajectory comparisons are fixed-budget predictive checks, **not** a
claim of optimizer convergence or a globally optimal fit. The harmonic
delta target is zero; that analytic simplification is not evidence of a
general advantage on anharmonic problems.

## Optional new fit

```sh
python assets/code/mlcmd-quartic/reproduce.py --output mlcmd-refit --model quartic --refit
```

This performs new fits to the published labels and writes separate
`*_refitted.json` files. The plots and saved correlations still describe the
supplied models. Refitting does not regenerate trajectories. Numerical
libraries may produce different optimized parameters, and a new fit needs
its own predictive validation. The published verification used the default
load-only command; this optional branch was not run as a new scientific fit.

## What this reproduces—and what it does not

The attachment reproduces the learned potentials/forces from saved network
parameters, their independent label errors, the stored four-method plots,
and the quoted curve differences. The PMF curve in the learning plot is the
analytic integral of the same cubic force spline, with its energy zero at
q = 0; its derivative remains consistent with the interpolated force.

It does not rerun the original constrained PIMC, the canonical phase-space
quadrature and Verlet propagation, or the finite-box quantum DVR calculation.
Those calculations and their convergence controls belong to the source
project. The quantum reference uses the particle-in-a-box sine DVR: N
interior points give spacing L/(N+1). Enlarging the box at fixed spacing uses
that formula, not the periodic L/N spacing.

Agreement between an ML-CMD curve and reference CMD tests replacement of the
centroid mean force. The difference between reference CMD and the quantum
Kubo result tests the dynamical approximation. A learned curve being closer
to the quantum result would not, by itself, establish more faithful learning
of CMD.
