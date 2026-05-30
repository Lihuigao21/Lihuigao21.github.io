#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


SPECIES_ORDER = ["Ti", "O"]
SPECIES_VALENCE = {"Ti": (6.0, 6.0), "O": (3.0, 3.0)}
NVE_STEPS = 501
SAMPLE_START_FS = 0
SAMPLE_END_FS = 500
SAMPLE_STRIDE_FS = 5
OPENMX_DFT_DATA = os.environ.get("OPENMX_DFT_DATA", "/path/to/openmx/DFT_DATA19")
HAMGNN_ROOT = os.environ.get("HAMGNN_ROOT", "/path/to/HamGNN")
HAMGNN_CKPT = os.environ.get("HAMGNN_CKPT", "./checkpoints/hamgnn.ckpt")


def read_poscar(path: Path):
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    scale = float(lines[1].split()[0])
    lattice = [[float(x) * scale for x in lines[i].split()[:3]] for i in range(2, 5)]
    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    mode_idx = 7
    if lines[mode_idx].lower().startswith("s"):
        mode_idx += 1
    direct = lines[mode_idx].lower().startswith("d")
    atoms = []
    idx = mode_idx + 1
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


def read_md_frames(path: Path):
    raw = [line.rstrip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    frames = []
    idx = 0
    while idx < len(raw):
        try:
            natoms = int(raw[idx].split()[0])
        except Exception:
            idx += 1
            continue
        header = raw[idx + 1]
        time_fs = None
        if "time=" in header:
            try:
                time_fs = float(header.split("time=", 1)[1].split("(fs)", 1)[0].strip())
            except Exception:
                time_fs = None
        atoms = []
        velocities = []
        for line in raw[idx + 2: idx + 2 + natoms]:
            parts = line.split()
            sp = parts[0]
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
            vel = [float(parts[7]), float(parts[8]), float(parts[9])] if len(parts) >= 10 else [0.0, 0.0, 0.0]
            atoms.append((sp, xyz))
            velocities.append(vel)
        frames.append({"time_fs": time_fs, "atoms": atoms, "velocities": velocities})
        idx += natoms + 2
    if not frames:
        raise ValueError(f"No MD frames parsed from {path}")
    return frames


def ordered_atoms(atoms):
    grouped = []
    for sp in SPECIES_ORDER:
        for atom_sp, xyz in atoms:
            if atom_sp == sp:
                grouped.append((sp, xyz))
    return grouped


def write_openmx_dat(path: Path, system_name: str, lattice, atoms, *, md_type: str, velocities=None):
    lengths = cell_lengths(lattice)
    grouped = [(sp, wrap_xyz(xyz, lengths)) for sp, xyz in ordered_atoms(atoms)]
    if md_type == "nve":
        hs_fileout = "off"
        kgrid = "1 1 1"
        max_iter = 40
        criterion = "1.0e-5"
        mo_block = "MO.fileout                       off\n"
        if velocities is None:
            raise ValueError("NVE requires initial velocities")
        velocity_lines = ["<MD.Init.Velocity"]
        for idx, vel in enumerate(velocities, start=1):
            velocity_lines.append(f"{idx:4d}  {vel[0]:16.8f} {vel[1]:16.8f} {vel[2]:16.8f}")
        velocity_lines.append("MD.Init.Velocity>")
        md_block = f"""MD.Type                          NVE
MD.maxIter                       {NVE_STEPS}
MD.TimeStep                      1.0
{chr(10).join(velocity_lines)}
"""
    else:
        hs_fileout = "on"
        kgrid = "2 2 2"
        max_iter = 100
        criterion = "1.0e-6"
        mo_block = "MO.fileout                       off\n"
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
Dos.fileout                      off
{mo_block}
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


def write_graph_config(path: Path, job_dir: Path):
    path.write_text(
        f"""nao_max: 26
graph_data_save_path: "{job_dir}/graph_data"
read_openmx_path: "{HAMGNN_ROOT}/DFT_interfaces/openmx/openmx_postprocess/read_openmx"
max_SCF_skip: 200
scfout_paths: "{job_dir}/scf/frame_*"
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
    radius_type: openmx
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


def nearest_frame_indices(frames):
    out = []
    for target in range(SAMPLE_START_FS, SAMPLE_END_FS + 1, SAMPLE_STRIDE_FS):
        idx = min(range(len(frames)), key=lambda i: abs((frames[i]["time_fs"] or 0.0) - target))
        if out and idx <= out[-1]:
            raise ValueError(f"Non-increasing sampled frame index at target {target} fs")
        out.append(idx)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["initial", "scf"], required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--poscar", default=os.environ.get("OPENMX_POSCAR", "./POSCAR"))
    parser.add_argument("--source-md", default=os.environ.get("SOURCE_MD", "./nvt/TiO2_probe_nvt.md"))
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    nve_dir = job_dir / "nve"
    scf_root = job_dir / "scf"
    for directory in [nve_dir, scf_root, job_dir / "graph_data", job_dir / "plots"]:
        directory.mkdir(parents=True, exist_ok=True)

    lattice, _initial_atoms = read_poscar(Path(args.poscar))
    (job_dir / "cell_lattice.json").write_text(json.dumps(lattice, indent=2), encoding="utf-8")

    if args.stage == "initial":
        source_frames = read_md_frames(Path(args.source_md))
        last = source_frames[-1]
        write_openmx_dat(
            nve_dir / "TiO2_nve500.dat",
            "TiO2_nve500",
            lattice,
            last["atoms"],
            md_type="nve",
            velocities=last["velocities"],
        )
        write_graph_config(job_dir / "graph_data_gen.yaml", job_dir)
        write_hamgnn_config(job_dir / "config_test_nve500.yaml", job_dir)
        (job_dir / "nve_start_info.json").write_text(
            json.dumps(
                {
                    "source_md": args.source_md,
                    "source_time_fs": last["time_fs"],
                    "nve_steps": NVE_STEPS,
                    "sample_start_fs": SAMPLE_START_FS,
                    "sample_end_fs": SAMPLE_END_FS,
                    "sample_stride_fs": SAMPLE_STRIDE_FS,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        frames = read_md_frames(nve_dir / "TiO2_nve500.md")
        indices = nearest_frame_indices(frames)
        selected = []
        frame_dirs = []
        for sample_idx, frame_idx in enumerate(indices):
            frame = frames[frame_idx]
            frame_dir = scf_root / f"frame_{sample_idx:04d}_md{frame_idx:04d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            write_openmx_dat(frame_dir / "openmx.dat", "openmx", lattice, frame["atoms"], md_type="scf")
            selected.append(
                {
                    "sample_index": sample_idx,
                    "frame_index": frame_idx,
                    "time_fs": frame["time_fs"],
                    "directory": str(frame_dir),
                }
            )
            frame_dirs.append(str(frame_dir))
        (job_dir / "selected_frames.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
        (job_dir / "scf_frame_dirs.txt").write_text("\n".join(frame_dirs) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
