# PIMD IX author benchmarks

The scripts reproduce equilibrium energies for a one-dimensional harmonic
oscillator with \(m=\hbar=\omega=1\). They use finite-action Gaussian integrals,
the ToyModel normal-mode sampler and an exact covariance calculation for its
linear OBABO update.

## Run the checks

Use Python 3.10 or later. From the website repository root, install the two
Python requirements and run the deterministic and sampler checks.

```sh
python -m pip install -r assets/code/pimd/bead-fourier/author/requirements.txt
python -B assets/code/pimd/bead-fourier/author/reproduce.py --check
python -B assets/code/pimd/bead-fourier/author/reproduce.py --smoke
```

`--check` recomputes the finite-action energies behind 70 London vector points
and 25 Ivanov raster entries. It checks four stationary covariance references
and reconstructs the cost projections from the recorded timing trials.
`--smoke` checks force-site accounting and finite readouts for four short
ToyModel runs. Both commands run without writing files.

To generate fresh trajectories and wall timings, use a new output directory.
The first command covers seven sampler cases with 32 chains, 50,000
equilibration steps and 250,000 production steps per chain. The second command
checks the saved trajectory results before timing 13 configurations across
five interleaved repeats. Wall times depend on the hardware and thread
settings. Use one BLAS thread for a serial comparison. The timer uses a fixed
CPU affinity when the optional `psutil` package and the operating system
permit it.

```sh
python -B assets/code/pimd/bead-fourier/author/reproduce.py author-run --workers 2
python -B assets/code/pimd/bead-fourier/author/reproduce.py author-run --timing
```

## Evidence and interpretation

- [`london_fig1.csv`](london_fig1.csv) pairs calibrated vector point centers and
  drawn bar limits with the calculated equilibrium energies at \(\beta=8\).
  All 70 calculated values lie within the drawn bars. The bar limits are read
  directly from the graphic and carry no assigned confidence level in this
  bundle. The BF action uses 20 trapezoid panels per segment with reuse of
  shared periodic endpoints. The \(K=0\) BF column still integrates the
  line between neighboring beads. [London and Momeni, Fig. 1](https://arxiv.org/pdf/2510.09961v2#page=6)
- [`ivanov_fig5.csv`](ivanov_fig5.csv) records the raster-read intervals at
  \(\beta\hbar\omega=15.8\), alongside the analytic segment-action endpoint
  energies. The source discussion and low-mode points support the endpoint
  attribution. Overlapping marks share a readable interval and have blank
  center fields. The 5-bead, 2-mode result \(0.493921\hbar\omega\) is below the
  combined interval \(0.49517\)–\(0.50288\,\hbar\omega\) for the overlapping
  5-bead, 2–4-mode marks. The 5-bead, 4-mode result meets the \(0.3\%\)
  discretization gate. [Ivanov et al., Fig. 5 and Sec. IV.A](https://doi.org/10.1103/PhysRevE.67.066710)
- [`efficiency.json`](efficiency.json) contains measured step-time trials,
  stationary variances, sampling checks and the resulting cost projections.
  In the London endpoint comparison, \(5.28\) s for the 4-bead, 2-mode BF
  model and \(9.96\) s for ordinary 32-bead PIMD are projected wall times from
  an equilibrated ensemble to reach a \(1\%\) relative standard error. Each
  projection multiplies the predicted number of 32-chain batch steps by the
  measured median wall time per batch step. The calculations use the common
  \(1\%\) discretization gate and separately account for time correlations.
  Both London methods use the reported centroid time constant of 50. The
  scaled Fourier amplitudes use twice their free frequency as friction. The
  Ivanov timing comparison uses matched PILE settings. The authors used
  Nosé–Hoover chains for that case. [London and Momeni, Sec. III.2](https://arxiv.org/html/2510.09961v2#S3)

[`protocol.json`](protocol.json) specifies the full calculation. The
ToyModel-derived Python code is distributed under the
[GNU GPL version 3](LICENSE).
