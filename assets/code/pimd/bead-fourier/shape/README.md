# Bead–Fourier PIMD across a quadratic–quartic potential family

The comparison uses \(V_s(x)=(1-s)x^2/2+s x^4/4\), with \(m=\hbar=1\). The inverse temperature is set by \(B=\beta(E_1-E_0)\), using the gap of each potential. The discretization criterion applies to both \(\langle x^2\rangle\) and the virial energy \((1-s)\langle x^2\rangle+3s\langle x^4\rangle/4\). It requires agreement within the target tolerance at the selected bead count and at every finer bead count tested, with at least one finer test. The two readouts average the same position functions either over the beads or over the action quadrature sites.

- [protocol.json](protocol.json) gives the potential, temperature, bead grid, sampler controls, and convergence criteria.
- [reference.json](reference.json) contains 18 continuum references and 1,260 finite-action results. Each finite-action row identifies \((s,B,P,K,L)\), the two readouts `bead` and `path`, and the numerical refinement change. \(K=L=0\) denotes primitive PIMD. The number of force sites per path step is \(P\) for primitive PIMD and \(PL\) for BF-PIMD.
- [figure_data.json](figure_data.json) contains the plotted data and selected comparisons. `figures` gives the exact marker or bar values and PNG checksums. `matched_costs` includes both readouts at 1%, 2%, and 5% tolerances. `tail.rows` gives the signed endpoint errors for the plain and tail-averaged actions. `timing.measurements` retains repeat timings and projected costs by observable. `timing.comparisons` gives the BF-to-primitive ratios.

The tail calculation replaces the potential at each site by its Gaussian expectation over the omitted free-bridge modes. It holds \(P,K,L\) fixed and supplies a first-cumulant test of the missing fluctuations. The reference and cutoff checks remain separate from finite-time MD sampling.

The timing measurements use the matched endpoint configurations. For each observable, the projected number of batch steps is \((\sigma/(0.01\,|\mu|))^2 / (N_{\rm eff}/n_{\rm step})\), with aggregate ESS from the 48 chains. Multiply this by the median 48-chain batch step time to estimate wall time or by \(48\) times the sites per path step to estimate total force sites. Each reported joint estimate uses the more demanding of the two observables. Single-path step timings are also provided as a separate measurement. Wall times depend on the host and the implementation of the polynomial forces.

## Run from the website repository root

Python 3.10+, NumPy, and SciPy are required. Matplotlib is used for the figures. The sampler reuses the shared [ToyModel kernel](../author/bead_fourier.py), its [normal-mode helpers](../author/ring_helpers.py), and the separately extracted [autocorrelation and ESS functions](sampler_statistics.py). ToyModel code is distributed under the [GPLv3](../author/LICENSE). All paths below are relative to the repository root. Choose a new output name for each rerun.

```bash
python -m pip install numpy scipy matplotlib
python -c "from pathlib import Path; Path('articles/bf-pimd-reproduction').mkdir(parents=True, exist_ok=True)"
python assets/code/pimd/bead-fourier/shape/reference.py --anchor --output articles/bf-pimd-reproduction/reference-anchors.json
python assets/code/pimd/bead-fourier/shape/tail_control.py articles/bf-pimd-reproduction/tail-anchors.json --anchor
python assets/code/pimd/bead-fourier/shape/select_configs.py assets/code/pimd/bead-fourier/shape/reference.json articles/bf-pimd-reproduction/selected.json
python assets/code/pimd/bead-fourier/shape/run_shape_md.py articles/bf-pimd-reproduction/smoke --smoke
python assets/code/pimd/bead-fourier/shape/plot_shape.py articles/bf-pimd-reproduction/figures
```

The anchor commands check all 160 harmonic and quartic finite-action cells at \(B=8\), four tail-control cases, and four ToyModel sampling and force-accounting cases. Selection from the released reference gives 34 sampler configurations. The plotting script checks its numerical series against the released plot data. Remove `--anchor` to generate the complete deterministic datasets. The generated `reference.py` receipt includes the refined Gauss–Hermite result for the cold one-bead quartic case.

To repeat the sampler benchmark, run `run_shape_md.py` in a new results directory with the public reference and the generated selection. Use the same directory and arguments with `--timing` after the equilibrium checks pass. Single-threaded BLAS settings and the cadence in `protocol.json` help match the original timing comparison. CPU affinity is applied when supported by `psutil` and the host.
