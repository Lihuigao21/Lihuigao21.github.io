#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure

from DFT_interfaces.openmx.utils import (
    au2ang,
    au2ev,
    basis_def_26,
    kpoints_generator,
    num_valence_openmx,
)


NAO_MAX = 26
K_PATH = [
    [0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.5],
    [0.0, 0.0, 0.5],
]
K_LABELS = ["G", "X", "M", "G", "Z", "R", "A", "Z"]


def split_hamiltonian(data, flat_hamiltonian):
    n_on = len(data.Hon)
    n_off = len(data.Hoff)
    hon = flat_hamiltonian[:n_on].reshape(-1, NAO_MAX, NAO_MAX)
    hoff = flat_hamiltonian[n_on:n_on + n_off].reshape(-1, NAO_MAX, NAO_MAX)
    if len(flat_hamiltonian) != n_on + n_off:
        raise ValueError(f"Unexpected H length {len(flat_hamiltonian)} != {n_on + n_off}")
    return hon, hoff


def compute_bands(data, flat_hamiltonian, nk):
    son = data.Son.numpy().reshape(-1, NAO_MAX, NAO_MAX)
    soff = data.Soff.numpy().reshape(-1, NAO_MAX, NAO_MAX)
    hon, hoff = split_hamiltonian(data, flat_hamiltonian)
    latt = data.cell.numpy().reshape(3, 3)
    pos = data.pos.numpy() * au2ang
    nbr_shift = data.nbr_shift.numpy()
    edge_index = data.edge_index.numpy()
    species = data.z.numpy()

    basis_definition = np.zeros((99, NAO_MAX))
    for atomic_number in basis_def_26:
        basis_definition[atomic_number][basis_def_26[atomic_number]] = 1
    orb_mask = basis_definition[species].reshape(-1)
    orb_mask = orb_mask[:, None] * orb_mask[None, :]

    struct = Structure(
        lattice=latt * au2ang,
        species=[Element.from_Z(int(k)).symbol for k in species],
        coords=pos,
        coords_are_cartesian=True,
    )

    kpts = kpoints_generator(dim_k=3, lat=latt)
    k_vec, k_dist, k_node, lat_per_inv, node_index = kpts.k_path(K_PATH, nk)
    k_vec = k_vec.dot(lat_per_inv[np.newaxis, :, :]).reshape(-1, 3)

    natoms = len(struct)
    na = np.arange(natoms)
    eigen = []
    for ik in range(nk):
        hk = np.zeros((natoms, natoms, NAO_MAX, NAO_MAX), dtype=np.complex64)
        sk = np.zeros((natoms, natoms, NAO_MAX, NAO_MAX), dtype=np.complex64)
        hk[na, na, :, :] += hon[na, :, :]
        sk[na, na, :, :] += son[na, :, :]

        coeff = np.exp(2j * np.pi * np.sum(nbr_shift * k_vec[ik][None, :], axis=-1))
        for iedge in range(len(hoff)):
            src = edge_index[0, iedge]
            dst = edge_index[1, iedge]
            hk[src, dst] += coeff[iedge, None, None] * hoff[iedge]
            sk[src, dst] += coeff[iedge, None, None] * soff[iedge]

        hk = np.swapaxes(hk, -2, -3).reshape(natoms * NAO_MAX, natoms * NAO_MAX)
        sk = np.swapaxes(sk, -2, -3).reshape(natoms * NAO_MAX, natoms * NAO_MAX)
        hk = hk[orb_mask > 0]
        sk = sk[orb_mask > 0]
        norbs = int(math.sqrt(hk.size))
        hk = hk.reshape(norbs, norbs)
        sk = sk.reshape(norbs, norbs)

        sk_t = torch.complex(torch.tensor(sk.real), torch.tensor(sk.imag)).unsqueeze(0)
        hk_t = torch.complex(torch.tensor(hk.real), torch.tensor(hk.imag)).unsqueeze(0)
        chol = torch.linalg.cholesky(sk_t)
        chol_h = torch.transpose(chol.conj(), dim0=-1, dim1=-2)
        hs = torch.bmm(torch.bmm(torch.linalg.inv(chol), hk_t), torch.linalg.inv(chol_h))
        vals, _ = torch.linalg.eigh(hs)
        eigen.append(vals.squeeze(0).cpu().numpy())

    eigen = np.swapaxes(np.array(eigen), 0, 1) * au2ev
    num_val = np.zeros((99,), dtype=int)
    for atomic_number in num_valence_openmx:
        num_val[atomic_number] = num_valence_openmx[atomic_number]
    num_electrons = int(np.sum(num_val[species]))
    vbm_band = math.ceil(num_electrons / 2) - 1
    cbm_band = vbm_band + 1
    vbm = float(np.max(eigen[vbm_band]))
    cbm = float(np.min(eigen[cbm_band]))
    return {
        "eigen_raw_eV": eigen,
        "eigen_shifted_eV": eigen - vbm,
        "vbm_eV": vbm,
        "cbm_eV": cbm,
        "gap_eV": cbm - vbm,
        "vbm_band_index": vbm_band,
        "cbm_band_index": cbm_band,
        "k_dist": k_dist,
        "k_node": k_node,
        "node_index": node_index,
        "norbs": int(eigen.shape[0]),
    }


def plot_compare(openmx, hamgnn, out_png: Path, out_pdf: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    k_dist = openmx["k_dist"]
    for i, band in enumerate(openmx["eigen_shifted_eV"]):
        ax.plot(k_dist, band, color="black", linewidth=0.85, alpha=0.72, label="OpenMX" if i == 0 else None)
    for i, band in enumerate(hamgnn["eigen_shifted_eV"]):
        ax.plot(k_dist, band, color="#d62728", linewidth=0.75, alpha=0.62, linestyle="--", label="HamGNN" if i == 0 else None)
    for node in openmx["k_node"]:
        ax.axvline(node, color="0.65", linewidth=0.55)
    ax.axhline(0.0, color="0.25", linewidth=0.7, linestyle=":")
    ax.set_xlim(float(openmx["k_node"][0]), float(openmx["k_node"][-1]))
    ax.set_xticks(openmx["k_node"])
    ax.set_xticklabels(K_LABELS)
    ax.set_ylim(-3.0, 3.0)
    ax.set_ylabel("Energy relative to VBM (eV)")
    ax.set_xlabel("k path")
    ax.set_title(
        f"TiO2 NVT probe band comparison | OpenMX gap {openmx['gap_eV']:.3f} eV, "
        f"HamGNN gap {hamgnn['gap_eV']:.3f} eV"
    )
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--nk", type=int, default=80)
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    graph_path = job_dir / "graph_data" / "graph_data.npz"
    pred_path = job_dir / "hamgnn_eval" / "noop" / "prediction_hamiltonian.npy"
    target_path = job_dir / "hamgnn_eval" / "noop" / "target_hamiltonian.npy"
    plot_dir = job_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    graph_dict = np.load(graph_path, allow_pickle=True)["graph"].item()
    data = next(iter(graph_dict.values()))
    pred = np.load(pred_path)
    target = np.load(target_path)

    openmx = compute_bands(data, target, args.nk)
    hamgnn = compute_bands(data, pred, args.nk)

    np.savez(
        plot_dir / "band_compare_data.npz",
        k_dist=openmx["k_dist"],
        k_node=openmx["k_node"],
        openmx_eigen_shifted_eV=openmx["eigen_shifted_eV"],
        hamgnn_eigen_shifted_eV=hamgnn["eigen_shifted_eV"],
        openmx_eigen_raw_eV=openmx["eigen_raw_eV"],
        hamgnn_eigen_raw_eV=hamgnn["eigen_raw_eV"],
    )
    summary = {
        "nk": args.nk,
        "k_labels": K_LABELS,
        "openmx_gap_eV": openmx["gap_eV"],
        "hamgnn_gap_eV": hamgnn["gap_eV"],
        "gap_abs_error_eV": abs(hamgnn["gap_eV"] - openmx["gap_eV"]),
        "openmx_vbm_eV": openmx["vbm_eV"],
        "hamgnn_vbm_eV": hamgnn["vbm_eV"],
        "vbm_band_index": openmx["vbm_band_index"],
        "cbm_band_index": openmx["cbm_band_index"],
        "norbs": openmx["norbs"],
        "prediction_hamiltonian_path": str(pred_path),
        "target_hamiltonian_path": str(target_path),
    }
    (plot_dir / "band_compare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_compare(openmx, hamgnn, plot_dir / "band_compare.png", plot_dir / "band_compare.pdf")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
