# PIMD VIII: linear spectra in a quartic potential

Source-only reproduction package for the model

\[
H=p^2/2+q^2/2+\alpha q^4,\qquad A=q,
\]

with mass, Planck constant and harmonic frequency set to one, inverse
temperature beta=8, and alpha=0, 0.01, 0.5. The physical observable stays linear.
The two auxiliary readouts measure within-ring variance and its contribution
to the centroid force; they do not replace the physical observable.

## Requirements and first check

Use Python 3.10 or later, NumPy, SciPy, Matplotlib, and a GCC-compatible C++17
compiler with OpenMP. The scripts use `Path.write_text(..., newline=...)`, so
Python 3.10 is the minimum.
Install the Python requirements in your preferred environment:

```sh
python -m pip install -r requirements.txt
```

From the unpacked package directory, run:

```sh
python reproduce.py --smoke --output quartic-smoke
```

Place `g++` on PATH or add `--cxx /path/to/g++`. On Windows, use a MinGW-w64
g++ installation with OpenMP and pass its executable to `--cxx` if needed.
The loader discovers its runtime DLL directory from that compiler. Linux/GCC
and Windows/MinGW build commands are provided below; the actual public-entry
validation was performed on Windows/MinGW. Other platforms require a working
GCC/OpenMP toolchain and their own validation.

The launcher copies sources and the protocol into the explicitly chosen output
directory, compiles there, and runs only small implementation checks:

1. Exact quantum reference for alpha=0.5 at basis sizes 80, 120 and 180.
2. Independent same-noise NumPy/native controls, harmonic analytic controls,
   noise and chunk-restart checks.
3. The original small RPMD/TRPMD runner: 128 paths each, T=20, P=64.
4. Initial/final array-shape and centroid checks, unique trajectory RNG states,
   repeated-seed reproducibility and distinct independent-batch initial rings.

Smoke does not reproduce production sampling or establish a converged spectrum.
It writes no calculated files into this source package. Reusing an output
directory is permitted only for the same source identity; a different or
unidentified nonempty directory is refused. Never point it at an archived
production directory.

## Full reproduction, explicitly requested

The following commands generate fresh results; the source ZIP does not contain
precomputed trajectory arrays or native binaries:

```sh
python reproduce.py --stage exact --output quartic-results
python reproduce.py --stage main --output quartic-results
python reproduce.py --stage refine --output quartic-results
python reproduce.py --stage extension --output quartic-results
python reproduce.py --stage analyze --output quartic-results
python reproduce.py --stage validate --output quartic-results
```

`--stage all` runs those six stages in that order. `--stage build` only stages
and compiles. Existing validated batches with identical source, protocol and
seed identity are reused by the original restart checks. Editing a source or
configuration requires a fresh output directory.

The exact stage uses all three alpha values and basis sizes 80, 120, 180.
The main stage has ten ensembles: RPMD and TRPMD lambda=1 for the harmonic
control; lambda=0, 0.5, 1, 2 for each quartic potential. Each ensemble has
16 independent batches of 512 paths, T=200, native dt=0.025 and output dt=0.05.
The refine stage adds strong-quartic P=128 and dt=0.0125 comparisons plus the
weak-quartic internal-mass diagnostic (mu=0.5). The extension stage adds an
independent 8192-path pool to each strong-quartic RPMD discretization case;
analysis combines these pools without discarding the original batches.

The validation stage reads all 18 ensemble outputs, rebuilds static/sampling
and curve checks, performs paired batch bootstrap comparisons, and runs a
separate 1024-path output-cadence check. It therefore requires the preceding
stages; it is not a read-only inspection command. No new production was run
when preparing this public source package.

Analysis writes the six figures, absolute spectra, comparison metrics and
curve-quality records to `quartic-results/results/`. The bootstrap and final
validation record are written by the validation stage. No manuscript or
website generator is required.

## Calculation conventions

- Independent canonical rings are sampled by exact rejection of harmonic-ring
  Gaussian proposals with weight `exp(-beta*alpha*mean(q**4))`. Initial momenta
  are independently sampled. No correlated MCMC chain is used.
- The quartic kick is split around the exact linear harmonic/spring Langevin
  transition. Physical masses are used in production. The explicit auxiliary
  mass diagnostic changes only the k>0 masses; velocities remain dq/dt.
- Standard friction is `gamma[k]=2*lambda*omega_s[k]`, with gamma[0]=0.
- The known symmetry mean(q)=0 is used. Correlations average all saved time
  origins within each independent path and then average independent batches.
- Spectra use `(2/pi)*integral C(t)*exp(-0.05*t)*cos(omega*t) dt`. They are
  absolute: no unit-area normalization, peak-height normalization, negative-tail
  clipping or smoothing of plotted data is applied. The common Lorentz width
  is observation resolution, not a physical lifetime.
- Exact spectra come from projected physical q and q^4 operators and thermal
  Kubo transition weights. Projected powers are formed before basis truncation.
  Their analytic line transforms use the same eta; the saved exact time grid
  extends to T=240, while nuclear production stops at T=200.
- Nonlinear dynamics approximation error, bead/step error and finite-path
  statistical uncertainty remain separate. A smooth curve alone is not a
  quantum-accuracy certificate.

The frozen numerical parameters and acceptance thresholds are in
`inputs/protocol.json`. Exact-reference thresholds are also recorded in
`work/exact_1d.py`. The independent native checks have their own implementation
thresholds in `work/native_smoke_1d.py`.

## Native build and ABI

The launcher performs the platform-appropriate build in the output directory.
Equivalent direct commands, issued there after staging, are:

```sh
# Linux with GCC/OpenMP
g++ -O3 -std=c++17 -fopenmp -shared -fPIC work/native_engine.cpp -o work/libnative_engine.so

# Windows with MinGW-w64/OpenMP
g++ -O3 -std=c++17 -fopenmp -shared -static-libgcc -static-libstdc++ work/native_engine.cpp -o work/native_engine.dll
```

The 1D C ABI is retained from production:

```c
native_run_1d(n,p,nsteps,stride,dt,alpha,beta,internal_mass,
              coeff,q,v,seed,states,output,diagnostics,obs,threads)
```

Float arrays are C-contiguous double precision. `q,v` have shape `(n,P)`;
`coeff` has shape `(P/2+1,7)` with columns
`F00,F01,F10,F11,L00,L10,L11`. Persistent states are uint64 `(n,)`.
`output` has shape `(n,1+nsteps/stride)` and includes the initial centroid.
`diagnostics` has ten columns: initial/final ring Hamiltonian, maximum sampled
relative energy change, maximum absolute coordinate, initial/final bead q^2,
initial/final bead q^4, and initial/final centroid. Optional `obs` has shape
`(n,1+nsteps/stride,2)` and stores
`mean(q**2)-qc**2` and `-4*alpha*(mean(q**3)-qc**3)`.
Return zero means finite execution; negative means invalid input; positive
counts nonfinite paths. Thermostatted energy exchange is not an NVE drift
criterion. Saved checkpoints retain source/configuration and stable seed
identity.

## Source adaptation

The production sampling, seed families, propagation algorithm, Kubo reference,
correlation/spectral estimators, analysis and extension policy are retained.
Public adaptations are limited to portable native-library loading and naming,
portable C symbol export, removal of unused two-dimensional routines and
private design-history metadata, and this staging/build/smoke launcher.
The one-dimensional native implementation is otherwise unchanged. The package
does not depend on ToyModel or any private repository module.
