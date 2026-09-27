# HEOM validation: two distinct comparisons

This self-contained source bundle accompanies **HEOM: Theory, Boundaries, and DVR Tests**. Python 3.10 or later is required. The original and publication checks used Python 3.10.9 and the library versions in `requirements.txt`.

## Run

Download and extract `heom-validation.zip`, then run from the extracted `heom-validation` directory:

```sh
python -m pip install -r requirements.txt
python run.py harmonic --output results/harmonic
python run.py sac --output results/sac
```

Use a fresh output directory for the harmonic case. The SAC runner can resume validated completed cells from its existing output directory when source, configuration, and data checksums still match. It recomputes the output-cadence check and final figure. Outputs include density trajectories, configurations, source fingerprints, numerical checks, and figures. Both complete protocols are small local calculations; timing depends on the numerical backend and hardware.

From a website checkout, the same entry points are:

```sh
python assets/code/heom-validation/run.py harmonic --output articles/heom-publication/code-qa/harmonic
python assets/code/heom-validation/run.py sac --output articles/heom-publication/code-qa/sac
```

The wrapper imports only the bundled `toymodel` subset and verifies its SHA-256 checksums. No installation of the original repository or source-repository `PYTHONPATH` is needed. It defaults numerical libraries to one thread, disables bytecode output, and places the Matplotlib cache beside the chosen output directories. Explicit environment settings take precedence.

Generated run receipts record their actual execution paths, and the harmonic receipt also records the executing host. These are local provenance data: inspect and sanitize receipts before publishing them. The download contains source files only, with no private execution paths or raw trajectories.

## What the two protocols establish

### `harmonic`: standard Gaussian HEOM versus coordinate DVR

The model has a thermal harmonic mode with linear bath coupling, frequency 0.9, coupling 0.19, inverse temperature 2.2, system Hamiltonian `0.31 sigma_z + 0.12 sigma_x`, and system coupling operator `0.8 sigma_x + 0.2 sigma_z`. Units have hbar and nuclear mass equal to one. A factorized preparation starts in electronic state 0. The time interval is 0 to 6 with 301 output points.

The runner compares HEOM depths 2, 4, 8, and 10 against sinc coordinate DVR grids of 64/96 points on [-8, 8] and 120 points on [-10, 10]. The depth-10 maximum reduced-density element error in the accepted run was approximately `9.55e-12`. This is a genuine Gaussian HEOM validation on a discrete harmonic bath, with depth and grid/box controls; it is not an original SAC calculation or a continuum-relaxation test.

### `sac`: explicit full nuclear bath versus coordinate DVR

The original Tully simple avoided crossing uses `A=0.01, B=1.6, C=0.005, D=1, m=2000` in atomic units. The initial nuclear packet has position -5, momentum 15, position standard deviation 0.75, and fixed diabatic electronic state 0. The time interval is 0 to 1200 with 301 output points, plus a 601-point output-cadence control.

The two propagation routes retain the same electron-nuclear Hamiltonian. One works in coordinate DVR and the other uses a nuclear kinetic-energy eigenbasis. The latter retains all joint amplitudes when the complete bath basis is used. It is **not standard HEOM, not a non-Gaussian cumulant HEOM implementation, and not bath compression**. Same-grid agreement validates the representation and propagation implementation, not a new reduced-bath approximation.

The accepted 256-point full-bath comparison had a maximum density-element difference of approximately `6.22e-15`; grid refinement gave `4.24e-6`, box refinement `6.55e-15`, and 160-to-192 bath-state refinement `3.20e-6`. The 128-state truncation is an intentionally inadequate control and should remain visibly inaccurate. All reported populations and coherences are in the fixed diabatic basis at finite times, not asymptotic adiabatic scattering probabilities.

## Source and numerical provenance

`source-provenance.json` identifies each module and its raw-byte SHA-256 hash. Seven modules are exact copies of the accepted, frozen calculation snapshots. Two supporting import dependencies, `model/modelbase.py` and `utils/constant.py`, were not included in those original snapshots and are copied from the source repository at publication time; this limitation is explicitly recorded in the manifest. Their numerical behavior is checked by rerunning both full protocols from this bundle. Original source bytes, including line endings, are preserved.

The small package initializers and `run.py` are publication additions. They isolate imports and enforce the existing numerical pass/fail result; no scientific algorithm, physical parameter, or acceptance threshold is altered. `SHA256SUMS` covers the published payload excluding the checksum file itself and the ZIP archive. The ZIP contains the same payload in a single `heom-validation/` directory.

Numerical last digits may vary across BLAS/LAPACK implementations. Inspect `metrics.json` or `summary.json` for all frozen acceptance gates. Their automatic visual-audit field is intentionally pending: inspect the generated figures at their exported resolution before describing a new run as visually verified. The article's published figures were separately inspected.

## Files

- `toymodel/methods/heom.py`: Gaussian bosonic HEOM generator and propagation.
- `toymodel/methods/finite_bath_hierarchy.py`: explicit finite bath dynamics, including full joint-factor propagation.
- `toymodel/benchmarks/heom_harmonic.py`: harmonic HEOM/DVR protocol and figure.
- `toymodel/benchmarks/heom_sac.py`: original SAC basis-comparison protocol and figure.
- `toymodel/benchmarks/curve_quality.py`: declared continuous-curve diagnostics.
- `toymodel/DVRmethods/DVRBASE.py`: DVR grid and Hamiltonian construction.
- `toymodel/model/tully_model.py`, `model/modelbase.py`, and `utils/constant.py`: model and supporting definitions.

The full protocol specifications and thresholds are in each runner's `CONFIG` dictionary. Gaussian/nonstationary scope is a physical modeling condition, not a threshold that can be relaxed to make SAC into standard HEOM.
