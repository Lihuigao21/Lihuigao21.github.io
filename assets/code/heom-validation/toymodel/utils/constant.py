"""
Provide constants for toymodel.
"""
__all__ = [
    "Bohr_to_m",
    "Hartree_to_eV",
    "Hartree_to_kJmol",
    "au_to_m_per_s",
    "me",
    "amu_to_au",
    "kB",
    "hbar",
    "h",
    "fs_to_au",
    "s_to_au",
]

Bohr_to_m = 5.29177210903e-11

# Hartree to eV
Hartree_to_eV = 27.211386245988

# Hartree to kJ/mol
Hartree_to_kJmol = 2625.49962


au_to_m_per_s = 2.18769126364e6  # ≈ α·c ≈ 2.1877e6 m/s


me = 1.0  # a.u.


amu_to_au = 1822.888486209  # 1 amu ≈ 1822.89 me


kB = 3.166811563e-6  # in Hartree/K


hbar = 1.0  # a.u.
h = 2 * 3.141592653589793  # a.u.


fs_to_au = 41.3413745758  # 1 fs ≈ 41.341 a.u.
s_to_au = 4.134137457575e16  # 1 s ≈ 4.13e16 a.u
