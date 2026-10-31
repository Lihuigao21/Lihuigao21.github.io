"""Inspect the article's molecular label using only the Python standard library.

Run: python assets/code/molecular-excited-states/inspect_motor_label.py
An optional first argument selects another JSON with the same schema.
This inspects retained data; it does not run an electronic-structure calculation.
"""
from pathlib import Path
import json
import math
import sys


def inspect(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    numbers = data["atomic_numbers"]
    natom = len(numbers)

    def vector_array(name, array):
        if len(array) != natom or any(len(row) != 3 for row in array):
            raise ValueError(name + " must have one Cartesian vector per atom")
        if not all(math.isfinite(x) for row in array for x in row):
            raise ValueError(name + " contains a nonfinite number")

    vector_array("positions", data["positions"])
    for state, array in enumerate(data["forces"]):
        vector_array("force " + str(state), array)
    vector_array("NAC", data["nac_01"])
    vector_array("gap-scaled coupling", data["G_01"])
    energies = data["energies"]
    if len(energies) != 2 or not all(math.isfinite(x) for x in energies):
        raise ValueError("Expected two finite energies")
    gap = energies[1] - energies[0]
    ref = data["reference"]
    weights = ref["state_average_weights"]
    if len(weights) != 2 or abs(sum(weights) - 1) > 1e-12:
        raise ValueError("State-average weights must sum to one")
    electrons = sum(numbers) - ref["charge"]
    if 2 * ref["inactive_doubly_occupied_spatial_orbitals"] + ref["active_electrons"] != electrons:
        raise ValueError("Active and inactive electron counts disagree")
    mean = sum(w * e for w, e in zip(weights, energies))
    residual = max(abs(g - gap * d) for grow, drow in zip(data["G_01"], data["nac_01"]) for g, d in zip(grow, drow))
    if residual > 1e-11:
        raise ValueError("G01 does not equal (E1-E0) d01")
    conversion = data["constants"]["hartree_to_eV"]
    if abs(gap * conversion - data["derived"]["gap_eV"]) > 1e-10:
        raise ValueError("Retained excitation gap is inconsistent")
    if abs(mean - data["derived"]["state_average_energy_hartree"]) > 1e-10:
        raise ValueError("Retained state-average energy is inconsistent")
    force_conversion = conversion / data["constants"]["bohr_to_angstrom"]
    return {
        "atoms": natom,
        "electrons": electrons,
        "active_electrons": ref["active_electrons"],
        "active_spatial_orbitals": ref["active_spatial_orbitals"],
        "energies_hartree": energies,
        "gap_eV": gap * conversion,
        "state_average_energy_hartree": mean,
        "NAC_norm_inverse_bohr": math.sqrt(sum(x*x for row in data["nac_01"] for x in row)),
        "max_G01_identity_residual_hartree_per_bohr": residual,
        "max_atomic_force_eV_per_angstrom": [
            max(math.sqrt(sum(x*x for x in row)) for row in forces) * force_conversion
            for forces in data["forces"]
        ],
        "scope": "One local two-state label, not an electronic-structure or dynamics accuracy test",
    }


if __name__ == "__main__":
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("motor-example.json")
    print(json.dumps(inspect(source), indent=2, allow_nan=False))
