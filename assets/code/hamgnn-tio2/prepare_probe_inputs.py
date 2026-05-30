#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import json
import os
from pathlib import Path
import random


SPECIES_ORDER = ["Ti", "O"]
SPECIES_VALENCE = {"Ti": (6.0, 6.0), "O": (3.0, 3.0)}
ATOMIC_MASS_KG = {"Ti": 47.867 * 1.66053906660e-27, "O": 15.999 * 1.66053906660e-27}
MD_STEPS = 200
MD_TEMP_K = 300.0
VELOCITY_SEED = 20260529
OPENMX_DFT_DATA = os.environ.get("OPENMX_DFT_DATA", "/path/to/openmx/DFT_DATA19")
HAMGNN_ROOT = os.environ.get("HAMGNN_ROOT", "/path/to/HamGNN")
HAMGNN_CKPT = os.environ.get("HAMGNN_CKPT", "./checkpoints/hamgnn.ckpt")


def read_poscar(path: Path):
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    scale = float(lines[1].split()[0])
    lattice = [[float(x) * scale for x in lines[i].split()[:3]] for i in range(2, 5)]
    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    coord_mode = lines[7].lower()
    direct = coord_mode.startswith("d")
    atoms = []
    idx = 8
    for sp, count in zip(species, counts):
        for _ in range(count):
            vals = [float(x) for x in lines[idx].split()[:3]]
            if direct:
                xyz = [
                    vals[0] * lattice[0][0] + vals[1] * lattice[1][0] + vals[2] * lattice[2][0],
                    vals[0] * lattice[0][1] + vals[1] * lattice[1][1] + vals[2] * lattice[2][1],
                    vals[0] * lattice[0][2] + vals[1] * lattice[1][2] + vals[2] * lattice[2][2],
                ]
            else:
                xyz = vals
            atoms.append((sp, xyz))
            idx += 1
    return lattice, atoms


def cell_lengths(lattice):
    return [lattice[0][0], lattice[1][1], lattice[2][2]]


def wrap_xyz(xyz, lengths):
    return [xyz[i] % lengths[i] for i in range(3)]


def write_poscar(path: Path, title: str, lattice, atoms):
    lengths = cell_lengths(lattice)
    grouped = {sp: [] for sp in SPECIES_ORDER}
    for sp, xyz in atoms:
        grouped[sp].append(wrap_xyz(xyz, lengths))
    with path.open("w", encoding="utf-8") as out:
        out.write(f"{title}\n1.0\n")
        for row in lattice:
            out.write(f"  {row[0]:16.10f} {row[1]:16.10f} {row[2]:16.10f}\n")
        out.write("  " + "  ".join(SPECIES_ORDER) + "\n")
        out.write("  " + "  ".join(str(len(grouped[sp])) for sp in SPECIES_ORDER) + "\n")
        out.write("Direct\n")
        for sp in SPECIES_ORDER:
            for xyz in grouped[sp]:
                frac = [xyz[i] / lengths[i] for i in range(3)]
                out.write(f"  {frac[0]:16.12f} {frac[1]:16.12f} {frac[2]:16.12f}\n")


def make_init_velocity_block(atoms):
    rng = random.Random(VELOCITY_SEED)
    velocities = []
    total_mass = 0.0
    momentum = [0.0, 0.0, 0.0]
    for sp, _xyz in atoms:
        mass = ATOMIC_MASS_KG[sp]
        sigma = math.sqrt(1.380649e-23 * MD_TEMP_K / mass)
        vec = [rng.gauss(0.0, sigma) for _ in range(3)]
        velocities.append([sp, vec])
        total_mass += mass
        for axis in range(3):
            momentum[axis] += mass * vec[axis]

    v_cm = [p / total_mass for p in momentum]
    kinetic = 0.0
    for sp, vec in velocities:
        mass = ATOMIC_MASS_KG[sp]
        for axis in range(3):
            vec[axis] -= v_cm[axis]
        kinetic += 0.5 * mass * sum(v * v for v in vec)

    target = 1.5 * len(velocities) * 1.380649e-23 * MD_TEMP_K
    scale = math.sqrt(target / kinetic)
    lines = ["<MD.Init.Velocity"]
    for idx, (_sp, vec) in enumerate(velocities, start=1):
        vx, vy, vz = [v * scale for v in vec]
        lines.append(f"{idx:4d}  {vx:16.8f} {vy:16.8f} {vz:16.8f}")
    lines.append("MD.Init.Velocity>")
    return "\n".join(lines)


def write_openmx_dat(path: Path, system_name: str, lattice, atoms, *, md: bool):
    lengths = cell_lengths(lattice)
    grouped = []
    for sp in SPECIES_ORDER:
        for atom_sp, xyz in atoms:
            if atom_sp == sp:
                grouped.append((sp, wrap_xyz(xyz, lengths)))

    if md:
        hs_fileout = "off"
        kgrid = "1 1 1"
        max_iter = 40
        criterion = "1.0e-5"
        init_velocity = make_init_velocity_block(grouped)
        md_block = f"""MD.Type                          NVT_VS
MD.maxIter                       {MD_STEPS}
MD.TimeStep                      1.0
<MD.TempControl
  1
  {MD_STEPS}  1  {MD_TEMP_K:.1f}  0.0
MD.TempControl>
{init_velocity}
"""
    else:
        hs_fileout = "on"
        kgrid = "2 2 2"
        max_iter = 80
        criterion = "1.0e-6"
        md_block = """MD.Type                          Nomd
MD.maxIter                       1
MD.TimeStep                      0.5
MD.Opt.criterion                 1.0e-4
"""

    with path.open("w", encoding="utf-8") as out:
        out.write(f"""System.CurrrentDirectory         ./
System.Name                      {system_name}
DATA.PATH                        {OPENMX_DFT_DATA}
level.of.stdout                  1
level.of.fileout                 1
HS.fileout                       {hs_fileout}

scf.XcType                       GGA-PBE
scf.SpinPolarization             off
scf.ElectronicTemperature        300.0
scf.energycutoff                 200.0
scf.maxIter                      {max_iter}
scf.EigenvalueSolver             Band
scf.Kgrid                        {kgrid}
scf.Mixing.Type                  rmm-diis
scf.Init.Mixing.Weight           0.10
scf.Min.Mixing.Weight            0.001
scf.Max.Mixing.Weight            0.400
scf.Mixing.History               7
scf.Mixing.StartPulay            5
scf.criterion                    {criterion}

{md_block}
MO.fileout                       off
Dos.fileout                      off

Species.Number       2
<Definition.of.Atomic.Species
Ti   Ti7.0-s3p2d1       Ti_PBE19
O   O6.0-s2p2d1       O_PBE19
Definition.of.Atomic.Species>

Atoms.Number          {len(grouped)}
Atoms.SpeciesAndCoordinates.Unit   Ang
<Atoms.SpeciesAndCoordinates           # Unit=Ang.
""")
        for idx, (sp, xyz) in enumerate(grouped, start=1):
            up, dn = SPECIES_VALENCE[sp]
            out.write(f"{idx:4d}  {sp:2s}  {xyz[0]:13.8f} {xyz[1]:13.8f} {xyz[2]:13.8f}  {up:5.2f}  {dn:5.2f}\n")
        out.write("""Atoms.SpeciesAndCoordinates>
Atoms.UnitVectors.Unit             Ang
<Atoms.UnitVectors                     # unit=Ang.
""")
        for row in lattice:
            out.write(f"  {row[0]:16.10f} {row[1]:16.10f} {row[2]:16.10f}\n")
        out.write("Atoms.UnitVectors>\n")


def read_md2(path: Path):
    lines = [line.rstrip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    natoms = int(lines[0].split()[0])
    atoms = []
    for raw in lines[2:2 + natoms]:
        parts = raw.split()
        atoms.append((parts[1], [float(parts[2]), float(parts[3]), float(parts[4])]))
    return atoms


def write_graph_config(path: Path, job_dir: Path):
    path.write_text(
        f"""nao_max: 26
graph_data_save_path: "{job_dir}/graph_data"
read_openmx_path: "{HAMGNN_ROOT}/DFT_interfaces/openmx/openmx_postprocess/read_openmx"
max_SCF_skip: 200
scfout_paths: "{job_dir}/scf"
dat_file_name: "openmx.dat"
std_file_name: "openmx.std"
scfout_file_name: "openmx.scfout"
soc_switch: false
""",
        encoding="utf-8",
    )


def write_hamgnn_config(path: Path, job_dir: Path):
    path.write_text(
        f"""dataset_params:
  batch_size: 1
  split_file: null
  test_ratio: 1.0
  train_ratio: 0.0
  val_ratio: 0.0
  num_workers: 0
  preload: 0
  data_format: npz
  graph_data_path: {job_dir}/graph_data/graph_data.npz

losses_metrics:
  losses:
    - loss_weight: 27.211
      metric: mae
      prediction: hamiltonian
      target: hamiltonian
  metrics:
    - metric: mae
      prediction: hamiltonian
      target: hamiltonian

optim_params:
  lr: 0.005
  lr_decay: 0.5
  lr_patience: 5
  gradient_clip_val: 0.0
  max_epochs: 100
  min_epochs: 30
  stop_patience: 15

output_nets:
  output_module: HamGNN_out
  HamGNN_out:
    ham_only: true
    ham_type: openmx
    nao_max: 26
    add_H0: true
    symmetrize: true
    calculate_band_energy: false
    num_k: 5
    band_num_control: 8
    k_path: null
    soc_switch: false
    nonlinearity_type: gate
    spin_constrained: false
    collinear_spin: false
    minMagneticMoment: 0.5

profiler_params:
  progress_bar_refresh_rat: 10
  train_dir: {job_dir}/hamgnn_eval

representation_nets:
  HamGNN_pre:
    legacy_edge_update: false
    cutoff: 26.0
    cutoff_func: cos
    edge_sh_normalization: component
    edge_sh_normalize: true
    irreps_edge_sh: 0e + 1o + 2e + 3o + 4e + 5o
    irreps_node_features: 64x0e+64x0o+32x1o+16x1e+12x2o+25x2e+18x3o+9x3e+4x4o+9x4e+4x5o+4x5e+2x6e
    num_layers: 3
    num_radial: 64
    num_types: 96
    rbf_func: bessel
    set_features: true
    radial_MLP: [64, 64]
    use_corr_prod: false
    correlation: 2
    num_hidden_features: 16
    use_kan: false
    radius_scale: 1.01
    build_internal_graph: false
    use_gradient_checkpointing: false

setup:
  GNN_Net: HamGNNpre
  accelerator: null
  ignore_warnings: true
  checkpoint_path: {HAMGNN_CKPT}
  load_from_checkpoint: true
  resume: false
  num_gpus: 0
  precision: 32
  property: hamiltonian
  stage: test
""",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["initial", "final"], required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--poscar", default=os.environ.get("OPENMX_POSCAR", "./POSCAR"))
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    nvt_dir = job_dir / "nvt"
    scf_dir = job_dir / "scf"
    graph_dir = job_dir / "graph_data"
    for d in [nvt_dir, scf_dir, graph_dir, job_dir / "plots"]:
        d.mkdir(parents=True, exist_ok=True)

    lattice, initial_atoms = read_poscar(Path(args.poscar))
    (job_dir / "cell_lattice.json").write_text(json.dumps(lattice, indent=2), encoding="utf-8")

    if args.stage == "initial":
        write_poscar(nvt_dir / "POSCAR_start", "TiO2 OpenMX NVT probe start", lattice, initial_atoms)
        write_openmx_dat(nvt_dir / "TiO2_probe_nvt.dat", "TiO2_probe_nvt", lattice, initial_atoms, md=True)
        write_graph_config(job_dir / "graph_data_gen.yaml", job_dir)
        write_hamgnn_config(job_dir / "config_test_one.yaml", job_dir)
    else:
        md2 = nvt_dir / "TiO2_probe_nvt.md2"
        final_atoms = read_md2(md2)
        write_poscar(scf_dir / "POSCAR", "TiO2 OpenMX NVT probe final", lattice, final_atoms)
        write_openmx_dat(scf_dir / "openmx.dat", "openmx", lattice, final_atoms, md=False)


if __name__ == "__main__":
    main()
