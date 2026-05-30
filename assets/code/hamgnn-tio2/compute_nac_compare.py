#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from DFT_interfaces.openmx.utils import basis_def_26, num_valence_openmx


NAO_MAX = 26


def graph_sort_key(key):
    text = str(key)
    nums = re.findall(r"\d+", text)
    return (int(nums[-1]) if nums else 10**9, text)


def load_graph_dataset(path: Path):
    graph_data = np.load(path, allow_pickle=True)["graph"].item()
    keys = sorted(graph_data.keys(), key=graph_sort_key)
    return [graph_data[k] for k in keys]


def h_rows_from_data(data):
    return np.concatenate([data.Hon.numpy(), data.Hoff.numpy()], axis=0)


def build_gamma(data, h_rows):
    son = data.Son.numpy().reshape(-1, NAO_MAX, NAO_MAX)
    soff = data.Soff.numpy().reshape(-1, NAO_MAX, NAO_MAX)
    hon = h_rows[: len(data.Hon)].reshape(-1, NAO_MAX, NAO_MAX)
    hoff = h_rows[len(data.Hon): len(data.Hon) + len(data.Hoff)].reshape(-1, NAO_MAX, NAO_MAX)
    edge_index = data.edge_index.numpy()
    species = data.z.numpy()
    natoms = len(species)

    h = np.zeros((natoms, natoms, NAO_MAX, NAO_MAX), dtype=np.complex128)
    s = np.zeros((natoms, natoms, NAO_MAX, NAO_MAX), dtype=np.complex128)
    na = np.arange(natoms)
    h[na, na, :, :] += hon[na, :, :]
    s[na, na, :, :] += son[na, :, :]
    for edge in range(len(hoff)):
        src = edge_index[0, edge]
        dst = edge_index[1, edge]
        h[src, dst, :, :] += hoff[edge]
        s[src, dst, :, :] += soff[edge]

    basis_definition = np.zeros((99, NAO_MAX))
    for atomic_number in basis_def_26:
        basis_definition[atomic_number][basis_def_26[atomic_number]] = 1
    orb_mask = basis_definition[species].reshape(-1)
    orb_mask = orb_mask[:, None] * orb_mask[None, :]

    h = np.swapaxes(h, -2, -3).reshape(natoms * NAO_MAX, natoms * NAO_MAX)
    s = np.swapaxes(s, -2, -3).reshape(natoms * NAO_MAX, natoms * NAO_MAX)
    h = h[orb_mask > 0]
    s = s[orb_mask > 0]
    norbs = int(math.sqrt(h.size))
    h = h.reshape(norbs, norbs)
    s = s.reshape(norbs, norbs)
    h = 0.5 * (h + h.conj().T)
    s = 0.5 * (s + s.conj().T)
    return h, s


def split_prediction_rows(pred_rows, dataset):
    out = []
    cursor = 0
    for data in dataset:
        nrows = len(data.Hon) + len(data.Hoff)
        out.append(pred_rows[cursor: cursor + nrows])
        cursor += nrows
    if cursor != len(pred_rows):
        raise ValueError(f"Prediction row mismatch: used {cursor}, total {len(pred_rows)}")
    return out


def electron_band_indices(data):
    species = data.z.numpy()
    num_val = np.zeros((99,), dtype=int)
    for atomic_number in num_valence_openmx:
        num_val[atomic_number] = num_valence_openmx[atomic_number]
    num_electrons = int(np.sum(num_val[species]))
    vbm = math.ceil(num_electrons / 2) - 1
    cbm = vbm + 1
    return vbm, cbm


def pair_nac(h0, s0, h1, s1, band_indices, dt_fs):
    hmid = 0.5 * (h0 + h1)
    smid = 0.5 * (s0 + s1)
    energies, coeffs = eigh(hmid, smid, check_finite=False)
    dh = (h1 - h0) / dt_fs
    ds = (s1 - s0) / dt_fs
    nb = len(band_indices)
    mat = np.zeros((nb, nb), dtype=np.complex128)
    for a, i in enumerate(band_indices):
        ci = coeffs[:, i]
        for b, j in enumerate(band_indices):
            if i == j:
                continue
            cj = coeffs[:, j]
            denom = energies[j] - energies[i]
            if abs(denom) < 1.0e-10:
                continue
            op = dh - energies[j] * ds
            mat[a, b] = ci.conj().T @ op @ cj / denom
    return mat, energies


def pearson_abs(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) < 1.0e-20 or np.std(y) < 1.0e-20:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def percentile_dict(values, prefix):
    arr = np.asarray(values, dtype=float)
    return {
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_p99": float(np.percentile(arr, 99)),
        f"{prefix}_max": float(np.max(arr)),
    }


def load_frame_info(job_dir: Path, n_graphs: int):
    frame_info = json.loads((job_dir / "selected_frames.json").read_text(encoding="utf-8"))
    failed_dirs = []
    graph_stdout = job_dir / "graph_data_gen.stdout"
    if graph_stdout.exists():
        for line in graph_stdout.read_text(encoding="utf-8", errors="replace").splitlines():
            marker = " is not read successfully!"
            if marker in line:
                failed_dirs.append(str(Path(line.split(marker, 1)[0]).parent))

    if failed_dirs:
        failed_set = set(failed_dirs)
        frame_info = [row for row in frame_info if row["directory"] not in failed_set]

    if len(frame_info) != n_graphs:
        raise ValueError(
            f"Frame/graph mismatch after filtering: {len(frame_info)} frames, "
            f"{n_graphs} graphs, failed_dirs={failed_dirs}"
        )
    return frame_info, failed_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--band-pad", type=int, default=3)
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    dataset = load_graph_dataset(job_dir / "graph_data" / "graph_data.npz")
    pred_rows = np.load(job_dir / "hamgnn_eval" / "noop" / "prediction_hamiltonian.npy")
    pred_by_graph = split_prediction_rows(pred_rows, dataset)
    frame_info, failed_dirs = load_frame_info(job_dir, len(dataset))

    vbm, cbm = electron_band_indices(dataset[0])
    band_indices = list(range(vbm - args.band_pad, cbm + args.band_pad + 1))
    rel_labels = [idx - vbm for idx in band_indices]

    dft_hs = [build_gamma(data, h_rows_from_data(data)) for data in dataset]
    ml_hs = [(build_gamma(data, pred_by_graph[i])[0], dft_hs[i][1]) for i, data in enumerate(dataset)]

    rows = []
    extrema = []
    dft_all = []
    ml_all = []
    vbm_cbm_dft = []
    vbm_cbm_ml = []
    mats_dft = []
    mats_ml = []

    vbm_local = band_indices.index(vbm)
    cbm_local = band_indices.index(cbm)
    offdiag = ~np.eye(len(band_indices), dtype=bool)

    for pair_idx in range(len(dataset) - 1):
        dt_fs = float(frame_info[pair_idx + 1]["time_fs"] - frame_info[pair_idx]["time_fs"])
        if dt_fs <= 0:
            raise ValueError(f"Non-positive dt at pair {pair_idx}: {dt_fs}")
        dft_mat, dft_e = pair_nac(*dft_hs[pair_idx], *dft_hs[pair_idx + 1], band_indices, dt_fs)
        ml_mat, ml_e = pair_nac(*ml_hs[pair_idx], *ml_hs[pair_idx + 1], band_indices, dt_fs)
        dft_abs = np.abs(dft_mat)
        ml_abs = np.abs(ml_mat)
        err_abs = np.abs(ml_abs - dft_abs)
        mats_dft.append(dft_abs)
        mats_ml.append(ml_abs)
        dft_all.extend(dft_abs[offdiag].tolist())
        ml_all.extend(ml_abs[offdiag].tolist())
        for a, rel_i in enumerate(rel_labels):
            for b, rel_j in enumerate(rel_labels):
                if a == b:
                    continue
                extrema.append(
                    {
                        "abs_error_1_per_fs": float(err_abs[a, b]),
                        "dft_abs_1_per_fs": float(dft_abs[a, b]),
                        "ml_abs_1_per_fs": float(ml_abs[a, b]),
                        "pair_index": pair_idx,
                        "time_i_fs": frame_info[pair_idx]["time_fs"],
                        "time_j_fs": frame_info[pair_idx + 1]["time_fs"],
                        "band_i_rel_vbm": rel_i,
                        "band_j_rel_vbm": rel_j,
                    }
                )
        d_vc = float(dft_abs[vbm_local, cbm_local])
        m_vc = float(ml_abs[vbm_local, cbm_local])
        vbm_cbm_dft.append(d_vc)
        vbm_cbm_ml.append(m_vc)
        rows.append({
            "pair_index": pair_idx,
            "frame_i": frame_info[pair_idx]["frame_index"],
            "frame_j": frame_info[pair_idx + 1]["frame_index"],
            "time_i_fs": frame_info[pair_idx]["time_fs"],
            "time_j_fs": frame_info[pair_idx + 1]["time_fs"],
            "dt_fs": dt_fs,
            "dft_abs_vbm_cbm_1_per_fs": d_vc,
            "ml_abs_vbm_cbm_1_per_fs": m_vc,
            "abs_error_vbm_cbm_1_per_fs": abs(m_vc - d_vc),
            "dft_mean_abs_window_1_per_fs": float(np.mean(dft_abs[offdiag])),
            "ml_mean_abs_window_1_per_fs": float(np.mean(ml_abs[offdiag])),
            "dft_gap_mid_Ha": float(dft_e[cbm] - dft_e[vbm]),
            "ml_gap_mid_Ha": float(ml_e[cbm] - ml_e[vbm]),
        })

    dft_all = np.asarray(dft_all)
    ml_all = np.asarray(ml_all)
    abs_diff = np.abs(ml_all - dft_all)
    vbm_cbm_dft = np.asarray(vbm_cbm_dft)
    vbm_cbm_ml = np.asarray(vbm_cbm_ml)
    extrema = sorted(extrema, key=lambda row: row["abs_error_1_per_fs"], reverse=True)

    summary = {
        "definition": "Gamma-point finite-difference nonadiabatic-coupling proxy: C_i^H (dH/dt - E_j dS/dt) C_j / (E_j - E_i). H is OpenMX self-consistent target or HamGNN prediction; S is OpenMX overlap.",
        "n_frames": len(dataset),
        "n_pairs": len(rows),
        "dropped_frame_dirs": failed_dirs,
        "time_range_fs": [frame_info[0]["time_fs"], frame_info[-1]["time_fs"]],
        "dt_fs_values": sorted({float(row["dt_fs"]) for row in rows}),
        "band_indices": band_indices,
        "relative_to_vbm_labels": rel_labels,
        "vbm_band_index": vbm,
        "cbm_band_index": cbm,
        "window_mean_abs_dft_1_per_fs": float(np.mean(dft_all)),
        "window_mean_abs_ml_1_per_fs": float(np.mean(ml_all)),
        "window_mae_abs_1_per_fs": float(np.mean(abs_diff)),
        "window_rmse_abs_1_per_fs": float(np.sqrt(np.mean((ml_all - dft_all) ** 2))),
        "window_pearson_abs": pearson_abs(dft_all, ml_all),
        **percentile_dict(abs_diff, "window_abs_error_1_per_fs"),
        "vbm_cbm_mean_abs_dft_1_per_fs": float(np.mean(vbm_cbm_dft)),
        "vbm_cbm_mean_abs_ml_1_per_fs": float(np.mean(vbm_cbm_ml)),
        "vbm_cbm_mae_abs_1_per_fs": float(np.mean(np.abs(vbm_cbm_ml - vbm_cbm_dft))),
        "vbm_cbm_max_abs_dft_1_per_fs": float(np.max(vbm_cbm_dft)),
        "vbm_cbm_max_abs_ml_1_per_fs": float(np.max(vbm_cbm_ml)),
        "top_abs_error_pairs": extrema[:12],
    }

    out_dir = job_dir / "nac_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "nac_compare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "nac_pairs.csv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    avg_dft = np.mean(np.stack(mats_dft), axis=0)
    avg_ml = np.mean(np.stack(mats_ml), axis=0)
    vmax = max(float(np.max(avg_dft)), float(np.max(avg_ml)), 1.0e-12)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), dpi=180)
    for ax, mat, title in [
        (axes[0], avg_dft, "OpenMX SCF |NAC|"),
        (axes[1], avg_ml, "HamGNN ML |NAC|"),
        (axes[2], np.abs(avg_ml - avg_dft), "|ML-DFT|"),
    ]:
        im = ax.imshow(mat, origin="lower", vmin=0.0, vmax=vmax if ax is not axes[2] else None, cmap="magma")
        ax.set_title(title)
        ax.set_xticks(range(len(rel_labels)))
        ax.set_yticks(range(len(rel_labels)))
        ax.set_xticklabels(rel_labels)
        ax.set_yticklabels(rel_labels)
        ax.set_xlabel("band rel. VBM")
    axes[0].set_ylabel("band rel. VBM")
    fig.colorbar(im, ax=axes, shrink=0.82, label="1/fs")
    fig.savefig(out_dir / "nac_window_heatmaps.png", bbox_inches="tight")
    fig.savefig(out_dir / "nac_window_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)

    times = [row["time_i_fs"] for row in rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=180)
    ax.plot(times, vbm_cbm_dft, linewidth=1.5, label="OpenMX SCF")
    ax.plot(times, vbm_cbm_ml, linewidth=1.5, label="HamGNN ML")
    ax.set_xlabel("NVE time (fs)")
    ax.set_ylabel("|NAC(VBM, CBM)| (1/fs)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "nac_vbm_cbm_timeseries.png")
    fig.savefig(out_dir / "nac_vbm_cbm_timeseries.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.3), dpi=180)
    ax.scatter(dft_all, ml_all, s=10, alpha=0.55)
    lim = max(float(np.percentile(dft_all, 99.5)), float(np.percentile(ml_all, 99.5)), 1.0e-12)
    ax.plot([0, lim], [0, lim], color="black", linewidth=1.0, linestyle="--")
    ax.set_xlim(-0.02 * lim, 1.05 * lim)
    ax.set_ylim(-0.02 * lim, 1.05 * lim)
    ax.set_xlabel("OpenMX SCF |NAC| (1/fs)")
    ax.set_ylabel("HamGNN ML |NAC| (1/fs)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "nac_abs_scatter_p995.png")
    fig.savefig(out_dir / "nac_abs_scatter_p995.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=180)
    ax.hist(abs_diff, bins=80, color="#4c78a8", alpha=0.85)
    ax.set_xlabel("|ML-DFT| |NAC| error (1/fs)")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "nac_abs_error_hist.png")
    fig.savefig(out_dir / "nac_abs_error_hist.pdf")
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
