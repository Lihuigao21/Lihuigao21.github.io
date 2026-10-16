# Open Quantum Systems II: TCL2 and NZ2

This bundle accompanies the twelve population examples. All quantities use
hbar = omega_ref = 1, beta = 2. Black solid is the full quantum reference;
orange dashed is TCL2; blue dotted is NZ2. Negative populations are retained.

## Replot the published examples

Install NumPy, SciPy and Matplotlib in your own environment. From the extracted
bundle, run:

    python plot_gallery.py --output output/gallery

The script verifies each CSV checksum, checks every continuous series against
the benchmark's frozen smoothness thresholds, and exports twelve PNG files
and curve-quality.json. It never modifies the input data. The CSV uses 17
significant digits and has been verified to reconstruct the original floating
point arrays exactly. Every saved point is plotted; no filtering,
interpolation, clipping, density renormalization or population repair is used.

## Verify the numerical implementation

    python runner.py audit

This compact control checks the full-bath double commutator, kernel formulas,
analytic pure-dephasing references, zero coupling and the fixed-time
fourth-order difference of TCL2 and NZ2. It writes diagnostics under records/.
The runner and config are byte-for-byte copies of the frozen scientific source;
their original Windows line endings are intentionally preserved.

To regenerate the wider benchmark, the original entry point is:

    python runner.py run

That command is substantially more expensive than replotting: it executes
42 model/coupling cells with production and numerical-validation roles.
It resumes only completed outputs with matching source/configuration identity
and data checksums, stopping on incompatible existing files.
The full matrix is not required to reproduce the published gallery from CSV.
The comparison table is supplied as comparison.csv.

## Models and observables

The four gallery rows use H = Delta*sigma_z/2 + HB + g*sigma_x*B,
HB = (p^2+q^2)/2 + kappa*q^4 and B = q/sqrt(<q^2>_beta).
Each starts from the bare upper electronic state and an independent bath Gibbs
state. The fixed-basis upper population is Re rho[0,0]; it is not a hopping
active-state frequency or a local adiabatic projection.

The rows are harmonic resonance (kappa=0, Delta=1), harmonic detuning
(kappa=0, Delta=1.6), weak quartic (kappa=.01) and strong quartic (kappa=.5).
In the quartic rows Delta equals the bath's lowest transition energy.
Each row contains g=.05, .2 and .6.

The full comparison.csv also checks initially lower and coherent electronic
states and takes errors over t<=5, 15 or 30. It includes two additional
pure-dephasing controls. An acceptable single population does not certify
positivity or full-density accuracy for these other initial preparations.
The min_eigenvalue field is a density-positivity diagnostic, not a numerical
integration error.

The reference is spectral propagation of the complete joint Hamiltonian,
followed by a bath trace. Gallery basis size: 40; independent basis control:
64. DOP853 integrates the approximate equations with rtol=2e-11,
atol=2e-14 and maximum internal step .05. The CSV output interval .0125
is a saved-time spacing, not the integration timestep. The spectral correlation
has discarded positive pair weight below 2e-13.

The largest numerical-control density differences in the full study were
1.22e-10 (integration) and 2.25e-8 (basis), compared with a 2e-6 gate.
The 0.01 absolute density-accuracy gate and -1e-7 positivity threshold concern
the method, not numerical convergence. No arbitrary-initial-state complete
positivity theorem is claimed.

## Provenance and scope

gallery.json identifies source runner/configuration bytes, CSV files,
source-array fingerprints and numerical conventions. data/ contains only
the population arrays used by the twelve figures. curve-quality.json records
per-series finite-value, abscissa, duplicate-point, high-frequency-residual,
endpoint and spike checks. It does not replace visual review or integration
and basis convergence.

Tested scientific versions: NumPy 2.2.6 and SciPy 1.15.3. A current Matplotlib
is needed for the plotter's constrained layout. These are deterministic
calculations without Monte Carlo uncertainty or random seeds.

The interpretation concerns finite undamped baths and specified finite
observation windows. It is not a validation of material RPMD correlations,
long-time thermalization or a particular nonadiabatic trajectory workflow.
