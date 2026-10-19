Graphene tutorial: inputs, existing results, and analysis

Reference: NAMDinMomentumSpace commit 689c9521b634d1b1fe291c529703d1dd657763f1
Tutorial: tutorials/Training_202406_WSZhu.pptx, slides 21-35, figures on slide 34.
The PNG author figures are original embedded bytes. Rights remain with their
authors. Reproduced with attribution for discussion of this scientific example;
no separate license for the slide illustrations is asserted here.

inp and INICON retain the exact bytes of the completed tutorial-parameter run.
INICON is transcribed from slide 31. No new dynamics is run by these attachments.
electronic-energy.csv: all 2001 electronic times, three-sample means.
SH denotes active-state populations; PSI denotes continuous amplitudes.
Eref = -4.5302 eV. The PSI energy is not divided by its norm.
phonon-energy.csv: all 1001 phonon times, signed increments for each 2 fs bin,
their cumulative sums, and the cumulative total over six branches.
PHPROP already averages the three starting samples; no extra division is used.

Compact check (Python 3 + NumPy, from the website checkout):
  python assets/code/hefeinamd_graphene_check.py
Or download the script and both CSVs, then pass --data-dir DIRECTORY.
Its tolerances concern CSV rounding, not physical convergence.

The archived hefeinamd_graphene_postprocess.py is the unchanged original run
postprocessor, not a standalone calculation from the compact CSVs. --help runs
without the large data. To rerun its full-output analysis, place it under a
separate run/scripts directory. The script treats run/ as its output boundary.
Supply --work-dir pointing to the full original outputs:
  inp, INICON, BASSEL, SHPROP.194, SHPROP.20, SHPROP.14,
  PSICT.194, PSICT.20, PSICT.14, and PHPROP.1 through PHPROP.6.
PHPROP files include all original q columns. These large files are not included.
Supply --author-root pointing to the pinned author checkout, including:
  scripts/postnamd.py
  scripts/namdplt.py
  example/perturbo/bands/bandplt_pert.py
Supply --reference-dir containing the linked author figures under their original
names image76.png and image77.png, and REFERENCE.json with slide provenance.
Supply --validation-json completed-validation.json. This published subset binds
the original output hashes and preserves the failed additional norm diagnostic.
Choose --output run/results/NEW_DIRECTORY, which must not already exist.
The script checks exact input/source/image hashes before writing output.
Python 3.9+, NumPy, and Matplotlib are required. Optional --h5-levels additionally
requires h5py and the EPC HDF5 files. Without it, marker energies use BASSEL.
The two original author plotting functions are selected without modification.
The separate estimator illustration uses the same complete CSV time series.

Scope: full tutorial parameters and visual comparison, not certified pointwise
reference agreement, exact Perturbo 2.1.0 release identity, or parameter
convergence. The existing EPR came from Perturbo's official example outputs.
This is not an independently regenerated SCF/DFPT dataset.
