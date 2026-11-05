# Hessian and harmonic-Wigner teaching example

The molecular data contain the optimized 30-atom PSiNaD tutorial motor and
84 internal vibrational modes calculated with GFN2-xTB 6.7.1. Coordinates are
in angstrom, masses in atomic mass units, and frequencies are wavenumbers
in inverse centimeters. Each mass-weighted mode has unit Euclidean norm.
Source hashes and public source URLs are included in the JSON.

The Python example and interactive figures use the exact harmonic thermal
Gaussian. They are not a replay of SHARC 4.1 finite-temperature proposals.
The xTB model is a low-cost proposal model, not a CASSCF thermal ensemble.
The examples check sampling and reconstruction, not molecular dynamics accuracy.

Place `sample_harmonic.py` and `motor-normal-modes.json` together. With NumPy and
Matplotlib available, run:

```sh
python sample_harmonic.py --output wigner-example --samples 50000 --seed 20261105
```

The chosen output directory receives the checks, raw standardized draws,
analytic plot arrays, and SVG/PNG illustrations. The published check files
use 50 K, 50,000 independent draws, and the seed shown above. Curve-quality
limits are declared in the script before calculation. No curves are smoothed.

The molecular browser illustration uses a separate explicit reproducible seed
for demonstration. Moving a normal-coordinate slider is not a time trajectory.
