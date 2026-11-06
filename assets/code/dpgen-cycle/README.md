# Committee-disagreement teaching example

This dimensionless example illustrates the selection idea in DP-GEN. It is not
HamGNN, molecular dynamics, an electronic-structure calculation, or an
active-learning convergence benchmark.

Run with Python 3.9+ and NumPy 2.0+ / Matplotlib:

```bash
python assets/code/dpgen-cycle/committee_demo.py --output demo-output
```

The reference is `E(q)=0.12q^4-0.18q^2+0.08sin(3q)` and `F=-dE/dq`.
Thirteen shared points in `[-0.9,0.9]` supply both energies and forces.
Four independent, recorded random seeds choose 24 Gaussian features each.
Every model uses the same ridge parameter and energy/force loss weights.
Predicted forces are analytic negative derivatives of each model's energy.

The optional common tilt `b(q)=0.4q` shifts every member's force by `-0.4`.
This leaves their population-standard-deviation spread unchanged. It is a
controlled counterexample to interpreting agreement as an accuracy guarantee.
The actual error can increase or decrease at an individual point.

`committee-models.json` contains the reproducible trained parameters.
`numerical-check.json` records numerical controls and the fixed seeds;
`curve-quality.json` records per-series rendering-resolution checks. No curves
are smoothed. Absolute-error derivative cusps have a declared signed-parent
exception. The static SVG/PNG and interactive widget use the same models.

The editable lower/upper thresholds classify a synthetic uniform grid of 601
points. They are standard-style teaching triage, not the molecular project's
multi-observable ranking configuration. Grid points are not accepted labels.
