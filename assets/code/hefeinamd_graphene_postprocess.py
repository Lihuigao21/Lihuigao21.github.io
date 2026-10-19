"""Prepare slide34-type figures from completed, original-parameter NAMD output.

Original author files are read-only. The author plot functions are selected
unchanged from their pinned AST, avoiding unrelated main()/SciPy/HDF5 imports.
No reference curve is digitized or generated. No data are normalized or thinned.
"""
from pathlib import Path
import argparse
import ast
import datetime
import hashlib
import inspect
import json
import os
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUTHOR_COMMIT = "689c9521b634d1b1fe291c529703d1dd657763f1"
EXPECTED_SHAS = {
    "scripts/postnamd.py": "92b12e58ff2e8fd17753f4bb2b0896707e07bea592677abaaec09fb237b13d16",
    "scripts/namdplt.py": "914975a2afef06c94962da2b2f2a86cd15192845a8dc7173f7e17e08e3a85cc1",
    "example/perturbo/bands/bandplt_pert.py": "5530debb3096b94a7cf999ca92e30c94f523ffd05015830eb8d8350205bb6858",
}
INPUT_SHAS = {
    "inp": "4741c38ecfe1275c04d3815c2219fa956d8144738a6630f7282ae5adfc852489",
    "INICON": "6917d48621b1de197f71b58bc80658cc43aeba6bd558539df87d35cd9f529def",
}
REFERENCE_SHAS = {
    "image76.png": "9d3496dc9a1870fd83d750029fc1d8f5e613c42b5486e7e2c4a3441179e3dfa8",
    "image77.png": "528ddeccb72d276a21679b3b311896fc834f114a2525a3b03de914d61ef610ee",
}
STARTS = (194, 20, 14)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def identity(path):
    s = path.stat()
    return (s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino)


def source_record(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))


def author_sources(author):
    records = {}
    for rel, expected in EXPECTED_SHAS.items():
        path = author / rel
        assert digest(path) == expected, "Author source differs: " + rel
        records[rel] = source_record(path)
    text = (author / "example/perturbo/bands/bandplt_pert.py").read_text()
    values = [ast.literal_eval(node.value) for node in ast.parse(text).body
              if isinstance(node, ast.Assign)
              and any(isinstance(x, ast.Name) and x.id == "Efermi" for x in node.targets)]
    assert values == [-4.5302]
    return records, float(values[0])


def author_plot_functions(author, output, np_module, mpl, plt):
    """Compile only the two unmodified pinned author function definitions."""
    path = author / "scripts/namdplt.py"
    source = ast.parse(path.read_text(), filename=str(path))
    wanted = {"plot_tdprop", "plot_tdphen"}
    nodes = [x for x in source.body if isinstance(x, ast.FunctionDef) and x.name in wanted]
    assert {x.name for x in nodes} == wanted
    namespace = dict(np=np_module, mpl=mpl, plt=plt, os=os, fig_path=str(output))
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    assert str(inspect.signature(namespace["plot_tdprop"])) == "(shp, Eref=0.0, lplot=1, ksen=None, figname='tdshp.png')"
    assert str(inspect.signature(namespace["plot_tdphen"])) == "(php, figname='TDPHEN.png')"
    return namespace["plot_tdprop"], namespace["plot_tdphen"]


def read_electronic(work, prefix):
    paths = [work / ("%s.%d" % (prefix, start)) for start in STARTS]
    expected = {p.name for p in paths}
    assert {p.name for p in work.glob(prefix + ".*")} == expected
    snapshots = [identity(p) for p in paths]
    # Exact postnamd.readshp definition: array of full files, arithmetic mean.
    arrays = np.array([np.loadtxt(str(p)) for p in paths])
    assert arrays.shape == (3, 2001, 276), (prefix, arrays.shape)
    assert np.isfinite(arrays).all()
    assert np.array_equal(arrays[:, :, 0], np.tile(np.arange(2001), (3, 1)))
    mean = np.average(arrays, axis=0)
    records = []
    for p, a, initial_identity in zip(paths, arrays, snapshots):
        record = source_record(p)
        assert initial_identity == identity(p), "Output changed during reading"
        pop = a[:, 2:]
        record.update(rows=2001, columns=276, spacing_fs=1,
                      max_norm_error=float(np.max(np.abs(pop.sum(axis=1) - 1.0))),
                      min_population=float(pop.min()), max_population=float(pop.max()),
                      absolute_energy_initial_eV=float(a[0, 1]),
                      absolute_energy_final_eV=float(a[-1, 1]))
        records.append(record)
    return mean, arrays, records


def read_phonon_columns(work):
    """Validate full rows, retain only columns actually used by plot_tdphen."""
    assert {p.name for p in work.glob("PHPROP.*")} == {"PHPROP.%d" % i for i in range(1, 7)}
    columns = []
    records = []
    for mode in range(1, 7):
        p = work / ("PHPROP.%d" % mode)
        initial_identity = identity(p)
        h = hashlib.sha256()
        rows = []
        with p.open("rb") as f:
            for raw in f:
                h.update(raw)
                line = raw.decode("ascii").strip()
                if not line or line.startswith("#"):
                    continue
                values = np.fromstring(line, sep=" ")
                assert values.shape == (8102,), (p.name, len(rows), values.shape)
                assert np.isfinite(values).all(), (p.name, len(rows))
                rows.append(values[:2].copy())
        assert initial_identity == identity(p), "Output changed during reading"
        a = np.array(rows)
        assert a.shape == (1001, 2)
        assert np.array_equal(a[:, 0], np.arange(0, 2001, 2))
        columns.append(a)
        records.append(dict(path=str(p), bytes=p.stat().st_size,
                            sha256=h.hexdigest(), mode=mode, rows=1001,
                            original_columns=8102, retained_observable_columns=[0, 1],
                            spacing_fs=2, all_original_values_finite=True))
    # plot_tdphen only reads [:,:,0] and [:,:,1]; q-resolved data are unchanged.
    return np.array(columns), records


def basis_levels(work, h5_levels):
    path = work / "BASSEL"
    before = identity(path)
    basis = np.loadtxt(str(path), skiprows=1)
    assert basis.shape == (274, 3) and np.isfinite(basis).all()
    assert np.array_equal(basis[:, :2], basis[:, :2].astype(int))
    source = source_record(path)
    assert before == identity(path)
    if not h5_levels:
        return basis[:, 2].copy(), dict(
            method="Original BASSEL third column, in exact output basis order",
            display_level_precision="NAMD outputBAS writes F20.8 eV; at most5e-9 eV rounding in marker energies. Mean energy comes directly from SHPROP/PSICT and is not reconstructed.",
            source=source)
    import h5py
    energies = []
    metadata = []
    for part in range(1, 15):
        p = work / ("h5files/graphene_ephmat_p%d.h5" % part)
        with h5py.File(p, "r") as h:
            # Small metadata only; no g arrays read.
            a = h["el_ph_band_info/el_band_eV"][()]
        assert a.shape == ((600 if part < 13 else 450), 2)
        assert np.isfinite(a).all()
        energies.append(a)
        metadata.append(dict(path=str(p), shape=list(a.shape),
                             dataset="el_ph_band_info/el_band_eV"))
    whole = np.vstack(energies)
    indices = basis[:, :2].astype(int) - 1
    levels = whole[indices[:, 0], indices[:, 1]]
    return levels, dict(method="Same indexed H5 band-energy metadata as postnamd.ek_selected",
                        basis_order=source, datasets_read=metadata,
                        max_BASSEL_rounding_difference_eV=float(np.max(np.abs(levels - basis[:, 2]))))


def shifted_energy_facts(mean, samples, reference, levels):
    norms = mean[:, 2:].sum(axis=1)
    return dict(
        samples=3, time_points=2001, spacing_fs=1,
        average_definition="postnamd.readshp: arithmetic average of the three raw files, before plotting",
        absolute_energy_initial_eV=float(mean[0, 1]),
        absolute_energy_final_eV=float(mean[-1, 1]),
        energy_above_reference_initial_eV=float(mean[0, 1] - reference),
        energy_above_reference_final_eV=float(mean[-1, 1] - reference),
        raw_energy_change_eV=float(mean[-1, 1] - mean[0, 1]),
        mean_population_sum_initial=float(norms[0]),
        mean_population_sum_final=float(norms[-1]),
        max_mean_norm_error=float(np.max(np.abs(norms - 1))),
        max_single_sample_norm_error=float(np.max(np.abs(samples[:, :, 2:].sum(axis=2) - 1))),
        max_energy_column_vs_selected_levels_error_eV=float(
            np.max(np.abs(mean[:, 1] - mean[:, 2:] @ levels))),
        energy_reference_eV=reference, data_renormalized=False,
        nonunit_norm_note="Displayed offset follows author plot_tdprop: raw energy column minus fixed Eref. An unnormalized PSI energy is not divided by its norm.")


def comparison_montage(output, references, plt, gate_passed):
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0))
    pairs = [("image76.png", "TDEN.png"), ("image77.png", "TDPHEN.png")]
    for row, (refname, resultname) in enumerate(pairs):
        for col, path in enumerate((references / refname, output / resultname)):
            axes[row, col].imshow(plt.imread(path))
            axes[row, col].set_axis_off()
        axes[row, 0].set_title("Author tutorial, slide 34: original image", fontsize=12, pad=8)
        axes[row, 1].set_title("Current run: " + resultname.removesuffix(".png"), fontsize=12, pad=8)
    status = "passed" if gate_passed else "FAILED (original outputs retained)"
    fig.suptitle("Graphene tutorial: direct visual comparison", fontsize=16)
    fig.text(.5, .025,
             "Reference images only: no digitized curve, numerical tolerance or fitted agreement.  "
             "Additional 1e-5 norm diagnostic: " + status,
             ha="center", fontsize=9, wrap=True)
    fig.subplots_adjust(top=.92, bottom=.07, wspace=.04, hspace=.16)
    fig.savefig(output / "tutorial-comparison.png", dpi=220, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory inside this registered task; never overwritten")
    parser.add_argument("--author-root", type=Path, default=ROOT / "author-github")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "records/tutorial-slides")
    parser.add_argument("--validation-json", type=Path,
                        help="Defaults to work-dir/../results/namd-validation.json")
    parser.add_argument("--h5-levels", action="store_true",
                        help="Read exact small H5 band metadata for marker levels; never reads g arrays")
    args = parser.parse_args()
    work = args.work_dir.resolve(strict=True)
    output = args.output.resolve()
    author = args.author_root.resolve(strict=True)
    references = args.reference_dir.resolve(strict=True)
    assert output.is_relative_to(ROOT) and not output.exists(), "Task-local new output only"
    sources, eref = author_sources(author)
    for name, expected in INPUT_SHAS.items():
        assert digest(work / name) == expected, "Original tutorial input changed"
    for name, expected in REFERENCE_SHAS.items():
        assert digest(references / name) == expected, "Reference image changed"
    validation_path = args.validation_json or work.parent / "results/namd-validation.json"
    validation = json.loads(validation_path.read_text())
    assert validation["state"] == "completed" and validation["complete_finite_original_outputs"] is True
    # A failed norm gate is deliberately retained; it does not authorize parameter repair.
    gate_passed = bool(validation["numerical_gate_passed"])
    shp, shps, sh_files = read_electronic(work, "SHPROP")
    psi, psis, psi_files = read_electronic(work, "PSICT")
    php, ph_files = read_phonon_columns(work)
    levels, level_source = basis_levels(work, args.h5_levels)
    validated = {x["file"]: x for x in validation["files"]}
    for x in sh_files + psi_files + ph_files:
        name = Path(x["path"]).name
        assert validated[name]["sha256"] == x["sha256"], "Output differs from completed validation"
    # Exact plot_tdphen observable: time accumulation, then mode stacking.
    increments = php[:, :, 1].T
    by_mode = np.cumsum(increments, axis=0)
    stacked = np.cumsum(np.cumsum(increments, axis=0), axis=1)
    facts = dict(
        recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_work_dir=str(work), author_commit=AUTHOR_COMMIT, author_sources=sources,
        script_source=source_record(Path(__file__).resolve()),
        original_inputs={name: source_record(work / name) for name in INPUT_SHAS},
        output_validation=source_record(validation_path), original_norm_gate_passed=gate_passed,
        normalization_threshold_source="Additional diagnostic for the original-parameter run; not an author tutorial tolerance.",
        output_files=sh_files + psi_files + ph_files, selected_level_source=level_source,
        energy_reference=dict(value_eV=eref, source="Pinned example/perturbo/bands/bandplt_pert.py: Efermi assignment",
                              absolute_energy_preserved=True,
                              caution="namdplt.main defaults Eref=0; this invocation explicitly uses the author's graphene band reference. The embedded reference figure supplies no independent numerical-zero metadata."),
        SHPROP=shifted_energy_facts(shp, shps, eref, levels),
        PSICT=shifted_energy_facts(psi, psis, eref, levels),
        phonons=dict(
            time_points=1001, spacing_fs=2, modes=6,
            definition="namdplt.plot_tdphen: phen=php[:,:,1].T; np.cumsum(np.cumsum(phen,axis=0),axis=1)",
            final_net_energy_by_mode_eV=by_mode[-1].tolist(),
            final_stacked_total_net_energy_eV=float(stacked[-1, -1]),
            last_bin_increment_by_mode_eV=increments[-1].tolist(),
            largest_signed_net_contribution_mode_1based=int(np.argmax(by_mode[-1]) + 1),
            cumulative_three_sample_mean=True, extra_sample_average_applied=False,
            energy_conservation_gate_imposed=False),
        reference=dict(
            slide=34, images={name: source_record(references / name) for name in REFERENCE_SHAS},
            provenance=source_record(references / "REFERENCE.json"),
            data_available="Original embedded PNG images only",
            raw_numeric_reference_available=False, digitization_performed=False,
            numerical_agreement_or_reproduction_certified=False,
            comparison_scope="Same observable definitions and visual trend only"),
        all_original_time_points_used=True, interpolation_applied=False,
        renormalization_applied=False, changes_to_author_files=False,
        exact_Perturbo_release_certified=False,
        version_boundary="Official historical ec230 commit + unedited author patch; not a certified2.1.0 release.",
    )
    # No files are written before complete input/identity checks pass.
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=False)
    plot_tdprop, plot_tdphen = author_plot_functions(author, output, np, mpl, plt)
    plot_tdprop(shp, Eref=eref, lplot=2, ksen=levels, figname="TDEN.png")
    plot_tdphen(php, figname="TDPHEN.png")
    comparison_montage(output, references, plt, gate_passed)
    # Compact curves preserve all original time points and both energy conventions.
    np.savetxt(output / "electronic-energy.csv",
               np.column_stack((shp[:, 0], shp[:, 1], shp[:, 1] - eref,
                                psi[:, 1], psi[:, 1] - eref,
                                shp[:, 2:].sum(axis=1), psi[:, 2:].sum(axis=1))),
               delimiter=",", header="time_fs,SH_absolute_eV,SH_minus_Eref_eV,PSI_absolute_eV,PSI_minus_Eref_eV,SH_norm,PSI_norm",
               comments="", fmt="%.12e")
    np.savetxt(output / "phonon-energy.csv",
               np.column_stack((php[0, :, 0], increments, by_mode, stacked[:, -1])),
               delimiter=",", header="time_fs," + ",".join(
                   ["increment_mode%d_eV" % i for i in range(1, 7)] +
                   ["cumulative_mode%d_eV" % i for i in range(1, 7)] + ["cumulative_total_eV"]),
               comments="", fmt="%.12e")
    facts["deliverables"] = [source_record(p) for p in sorted(output.iterdir()) if p.is_file()]
    with (output / "facts.json").open("x", encoding="utf-8", newline="\n") as f:
        json.dump(facts, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(output=str(output), norm_gate_passed=gate_passed,
                          SH_final_above_reference_eV=facts["SHPROP"]["energy_above_reference_final_eV"],
                          PH_final_net_energy_eV=facts["phonons"]["final_stacked_total_net_energy_eV"],
                          reference_numeric_agreement_certified=False)))


if __name__ == "__main__":
    main()
