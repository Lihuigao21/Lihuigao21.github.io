"""Check the published graphene tutorial curves; no dynamics is performed.

Run from the website checkout with Python 3 and NumPy:
    python assets/code/hefeinamd_graphene_check.py
Use --data-dir when checking downloaded CSVs in another directory.
The checks establish internal data consistency, not tutorial agreement or
convergence of the physical calculation.
"""
from pathlib import Path
import argparse
import json
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(data_dir):
    electronic = np.genfromtxt(
        data_dir / "electronic-energy.csv", delimiter=",", names=True
    )
    phonon = np.genfromtxt(
        data_dir / "phonon-energy.csv", delimiter=",", names=True
    )
    require(electronic.shape == (2001,), "Expected 2001 electronic times")
    require(phonon.shape == (1001,), "Expected 1001 phonon times")
    for data in (electronic, phonon):
        for name in data.dtype.names:
            require(np.isfinite(data[name]).all(), "Nonfinite column: " + name)
    require(np.array_equal(electronic["time_fs"], np.arange(2001)),
            "Electronic times must be 0:1:2000 fs")
    require(np.array_equal(phonon["time_fs"], np.arange(0, 2001, 2)),
            "Phonon times must be 0:2:2000 fs")
    for prefix in ("SH", "PSI"):
        require(np.allclose(
            electronic[prefix + "_minus_Eref_eV"],
            electronic[prefix + "_absolute_eV"] + 4.5302,
            rtol=0, atol=2e-12
        ), "Energy-reference offset changed: " + prefix)
    accumulated = []
    for branch in range(1, 7):
        values = np.cumsum(phonon["increment_mode%d_eV" % branch])
        saved = phonon["cumulative_mode%d_eV" % branch]
        require(np.allclose(values, saved, rtol=0, atol=2e-11),
                "Phonon cumulative sum differs: branch %d" % branch)
        accumulated.append(saved)
    require(np.allclose(
        np.sum(accumulated, axis=0), phonon["cumulative_total_eV"],
        rtol=0, atol=2e-11
    ), "Total phonon energy differs from the branch sum")
    sh = electronic["SH_minus_Eref_eV"]
    psi = electronic["PSI_minus_Eref_eV"]
    loss = float(sh[0] - sh[-1])
    ph_gain = float(phonon["cumulative_total_eV"][-1])
    return {
        "electronic_time_points": len(electronic),
        "phonon_time_points": len(phonon),
        "SH_initial_eV": float(sh[0]),
        "SH_final_eV": float(sh[-1]),
        "PSI_final_eV": float(psi[-1]),
        "SH_band_energy_loss_eV": loss,
        "phonon_final_net_eV": ph_gain,
        "phonon_branch_5_final_eV": float(accumulated[4][-1]),
        "phonon_branch_6_final_eV": float(accumulated[5][-1]),
        "phonon_minus_SH_loss_eV": ph_gain - loss,
        "max_mean_PSI_norm_deviation": float(
            np.max(np.abs(electronic["PSI_norm"] - 1))
        ),
        "note": (
            "Norm column is the three-sample mean. The reported 2.5818e-5 "
            "maximum over individual samples requires the original files. "
            "Checks use CSV rounding tolerances, not physical error bounds."
        ),
        "reference_pointwise_agreement_certified": False,
        "parameter_convergence_certified": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data/hefeinamd-graphene"
    )
    args = parser.parse_args()
    print(json.dumps(check(args.data_dir), indent=2, allow_nan=False))
