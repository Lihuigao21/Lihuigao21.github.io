"""Reproduce the non-RPMD three-state Morse Spin-MInt benchmark.

This follows the three-state Morse model used in Cook, Rampton, and Hele,
J. Chem. Phys. 164, 144112 (2026), Fig. 6 and SI Fig. S.14.  The calculation
uses the local SpinMInt implementation directly, with focused electronic
action-angle initial conditions and a one-dimensional nuclear Wigner
distribution.

The default settings are intentionally light enough for a quick local check.
For a closer literature run, use ``--ntraj 10000 --dt 1 --tmax 3000``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _ensure_repo_imports():
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        src = candidate / "src"
        if (src / "toymodel").exists():
            src_str = str(src)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
            return candidate
    raise RuntimeError("Could not locate repository root containing src/toymodel.")


REPO_ROOT = _ensure_repo_imports()

from toymodel.methods import SpinMInt  # noqa: E402
from toymodel.model import ThreeStateMorse  # noqa: E402
from toymodel.utils.su_n import sw_parameters  # noqa: E402


MODEL_PARAMETERS = {
    "A": {
        "R0": 2.1,
        "params": {
            "d1": 0.02,
            "alpha1": 0.40,
            "r1": 4.0,
            "c1": 0.02,
            "d2": 0.02,
            "alpha2": 0.65,
            "r2": 4.5,
            "c2": 0.0,
            "d3": 0.003,
            "alpha3": 0.65,
            "r3": 6.0,
            "c3": 0.02,
            "a12": 0.005,
            "a13": 0.005,
            "a23": 0.0,
            "alpha12": 32.0,
            "alpha13": 32.0,
            "alpha23": 0.0,
            "r12": 3.40,
            "r13": 4.97,
            "r23": 0.0,
        },
    },
    "B": {
        "R0": 3.3,
        "params": {
            "d1": 0.02,
            "alpha1": 0.65,
            "r1": 4.5,
            "c1": 0.0,
            "d2": 0.01,
            "alpha2": 0.40,
            "r2": 4.0,
            "c2": 0.01,
            "d3": 0.003,
            "alpha3": 0.65,
            "r3": 4.4,
            "c3": 0.02,
            "a12": 0.005,
            "a13": 0.005,
            "a23": 0.0,
            "alpha12": 32.0,
            "alpha13": 32.0,
            "alpha23": 0.0,
            "r12": 3.66,
            "r13": 3.34,
            "r23": 0.0,
        },
    },
    "C": {
        "R0": 2.9,
        "params": {
            "d1": 0.003,
            "alpha1": 0.65,
            "r1": 5.0,
            "c1": 0.0,
            "d2": 0.004,
            "alpha2": 0.60,
            "r2": 4.0,
            "c2": 0.01,
            "d3": 0.003,
            "alpha3": 0.65,
            "r3": 6.0,
            "c3": 0.006,
            "a12": 0.002,
            "a13": 0.0,
            "a23": 0.002,
            "alpha12": 16.0,
            "alpha13": 0.0,
            "alpha23": 16.0,
            "r12": 3.40,
            "r13": 0.0,
            "r23": 4.80,
        },
    },
}


@dataclass
class FixedDistribution:
    q: float
    p: float
    mass: float
    T: float = 300.0
    Ntraj: int = 1

    def sample(self):
        return float(self.q), float(self.p)


def make_model(model_label: str) -> tuple[ThreeStateMorse, float]:
    entry = MODEL_PARAMETERS[model_label]
    return (
        ThreeStateMorse(representation="diabatic", **entry["params"]),
        float(entry["R0"]),
    )


def sample_initial_conditions(args, nstate: int):
    rng = np.random.default_rng(args.seed)
    q_sigma = np.sqrt(1.0 / (2.0 * args.mass * args.omega))
    p_sigma = np.sqrt(args.mass * args.omega / 2.0)
    q0 = rng.normal(args.R0, q_sigma, size=args.ntraj)
    p0 = rng.normal(args.P0, p_sigma, size=args.ntraj)

    gamma, _ = sw_parameters(nstate, "W")
    action = np.zeros(nstate, dtype=float)
    action[args.start_state] = 1.0
    amplitude = np.sqrt(2.0 * action + gamma)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(args.ntraj, nstate))

    # Literature SI Eq. S.46-S.47 uses q=amp*sin(theta), p=amp*cos(theta).
    mapping_x = amplitude[None, :] * np.sin(theta)
    mapping_p = amplitude[None, :] * np.cos(theta)
    return q0, p0, mapping_x, mapping_p


def run_trajectory(model, args, q0, p0, mapping_x, mapping_p):
    method = SpinMInt(
        model=model,
        distribution=FixedDistribution(q=q0, p=p0, mass=args.mass),
        Nstep=args.nstep,
        dt=args.dt,
        start_state=args.start_state,
        mapping_x=mapping_x,
        mapping_p=mapping_p,
        population_estimator="action",
        is_record=False,
        verbose=False,
    )
    method.initialize()
    pops = np.empty((args.nstep + 1, model.nstate), dtype=float)
    q_path = np.empty(args.nstep + 1, dtype=float)
    p_path = np.empty(args.nstep + 1, dtype=float)

    pops[0] = method._population_from_density(method.rho)
    q_path[0] = method.q
    p_path[0] = method.p
    for istep in range(1, args.nstep + 1):
        method.step()
        pops[istep] = method.poplist[-1]
        q_path[istep] = method.q
        p_path[istep] = method.p
    return pops, q_path, p_path


def run_ensemble(model, args):
    q0, p0, mapping_x, mapping_p = sample_initial_conditions(args, model.nstate)
    pop_sum = np.zeros((args.nstep + 1, model.nstate), dtype=float)
    pop_square_sum = np.zeros_like(pop_sum)
    final_rows = []
    progress_stride = max(1, args.ntraj // 10)

    for itraj in range(args.ntraj):
        pops, q_path, p_path = run_trajectory(
            model,
            args,
            q0[itraj],
            p0[itraj],
            mapping_x[itraj],
            mapping_p[itraj],
        )
        pop_sum += pops
        pop_square_sum += pops * pops
        final_rows.append(
            {
                "traj": itraj,
                "q0": q0[itraj],
                "p0": p0[itraj],
                "q_final": q_path[-1],
                "p_final": p_path[-1],
                "pop1_final": pops[-1, 0],
                "pop2_final": pops[-1, 1],
                "pop3_final": pops[-1, 2],
            }
        )
        if args.progress and ((itraj + 1) % progress_stride == 0 or itraj + 1 == args.ntraj):
            print(f"completed {itraj + 1}/{args.ntraj} trajectories", flush=True)

    mean = pop_sum / args.ntraj
    if args.ntraj > 1:
        variance = np.maximum(pop_square_sum / args.ntraj - mean * mean, 0.0)
        stderr = np.sqrt(variance / args.ntraj)
    else:
        stderr = np.zeros_like(mean)
    return mean, stderr, final_rows


def write_population_csv(path, time, mean, stderr):
    fields = [
        "time",
        "pop1",
        "pop2",
        "pop3",
        "pop1_stderr",
        "pop2_stderr",
        "pop3_stderr",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, t in enumerate(time):
            writer.writerow(
                {
                    "time": t,
                    "pop1": mean[idx, 0],
                    "pop2": mean[idx, 1],
                    "pop3": mean[idx, 2],
                    "pop1_stderr": stderr[idx, 0],
                    "pop2_stderr": stderr[idx, 1],
                    "pop3_stderr": stderr[idx, 2],
                }
            )


def write_final_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_reproduction(path, model, args, time, mean, stderr):
    colors = ["#2eae66", "#ed9072", "#6d63ff"]
    labels = ["state 1", "state 2", "state 3"]
    r_grid = np.linspace(2.0, 12.0, 1600)
    v_grid = model.V(r_grid)

    fig, axes = plt.subplots(2, 1, figsize=(4.6, 6.0), sharex=False)

    for istate, (color, label) in enumerate(zip(colors, labels)):
        axes[0].plot(r_grid, v_grid[:, istate, istate], color=color, lw=2.0, label=label)
    axes[0].annotate(
        "",
        xy=(args.R0, min(0.047, np.max(np.diagonal(v_grid, axis1=1, axis2=2)))),
        xytext=(args.R0, 0.0),
        arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.2},
    )
    axes[0].set_xlim(2.0, 12.0)
    axes[0].set_ylim(0.0, 0.05)
    axes[0].set_ylabel("Potential (a.u.)")
    axes[0].set_title(f"Model {args.model}")
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)

    for istate, (color, label) in enumerate(zip(colors, labels)):
        axes[1].plot(time, mean[:, istate], color=color, lw=2.0, label=label)
        if args.show_stderr and args.ntraj > 1:
            axes[1].fill_between(
                time,
                mean[:, istate] - stderr[:, istate],
                mean[:, istate] + stderr[:, istate],
                color=color,
                alpha=0.16,
                linewidth=0.0,
            )
    axes[1].set_xlim(0.0, args.nstep * args.dt)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("Population")
    axes[1].legend(frameon=False, loc="upper right", fontsize=8)

    for ax in axes:
        ax.tick_params(direction="in")
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    fig.suptitle(
        f"Spin-MInt, dt={args.dt:g}, ntraj={args.ntraj}",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


PAPER_PANELS = {
    "main_fig6": {
        "page_index": 11,
        "crop": (280, 1130, 620, 1628),
        "label": "Literature Fig. 6, Model A",
    },
    "si_figs14_dt10": {
        "page_index": 26,
        "crop": (190, 175, 620, 875),
        "label": "Literature Fig. S.14, Model A, dt=10",
    },
}


def crop_paper_model_a(paper_pdf, out_path, panel):
    import fitz
    from PIL import Image

    spec = PAPER_PANELS[panel]
    page_index = spec["page_index"]
    doc = fitz.open(paper_pdf)
    page = doc[page_index]
    zoom = 2.5
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    crop = image.crop(spec["crop"])
    crop.save(out_path)
    return out_path, spec["label"]


def make_comparison(path, paper_crop_path, reproduction_path, paper_label):
    from PIL import Image, ImageDraw, ImageFont

    paper = Image.open(paper_crop_path).convert("RGB")
    repro = Image.open(reproduction_path).convert("RGB")
    target_height = max(paper.height, repro.height)

    def scale_to_height(image, height):
        if image.height == height:
            return image
        width = int(round(image.width * height / image.height))
        return image.resize((width, height), Image.Resampling.LANCZOS)

    paper = scale_to_height(paper, target_height)
    repro = scale_to_height(repro, target_height)
    pad = 24
    title_h = 42
    width = paper.width + repro.width + 3 * pad
    height = target_height + title_h + 2 * pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((pad, pad), paper_label, fill="black", font=font)
    draw.text((paper.width + 2 * pad, pad), "Local Spin-MInt reproduction", fill="black", font=font)
    canvas.paste(paper, (pad, pad + title_h))
    canvas.paste(repro, (paper.width + 2 * pad, pad + title_h))
    canvas.save(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_PARAMETERS), default="A")
    parser.add_argument("--ntraj", type=int, default=300)
    parser.add_argument("--tmax", type=float, default=3000.0)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--mass", type=float, default=20000.0)
    parser.add_argument("--omega", type=float, default=0.005)
    parser.add_argument("--P0", type=float, default=0.0)
    parser.add_argument("--start-state", type=int, default=0)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--outdir", default=str(REPO_ROOT / "output" / "spin_mint_three_state_morse"))
    parser.add_argument("--paper-pdf", default=None)
    parser.add_argument("--paper-panel", choices=sorted(PAPER_PANELS), default="main_fig6")
    parser.add_argument("--show-stderr", action="store_true")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    _, R0 = make_model(args.model)
    args.R0 = R0
    args.nstep = int(round(args.tmax / args.dt))
    if abs(args.nstep * args.dt - args.tmax) > 1.0e-10:
        raise ValueError("tmax must be an integer multiple of dt.")
    if args.ntraj < 1:
        raise ValueError("ntraj must be positive.")
    if args.start_state not in (0, 1, 2):
        raise ValueError("start-state must be 0, 1, or 2.")
    return args


def main():
    args = parse_args()
    model, _ = make_model(args.model)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    time = np.arange(args.nstep + 1, dtype=float) * args.dt
    mean, stderr, final_rows = run_ensemble(model, args)

    stem = f"spin_mint_tsm_model_{args.model}_dt{args.dt:g}_ntraj{args.ntraj}"
    population_csv = outdir / f"{stem}_populations.csv"
    final_csv = outdir / f"{stem}_final_trajectories.csv"
    figure_path = outdir / f"{stem}.png"
    write_population_csv(population_csv, time, mean, stderr)
    write_final_csv(final_csv, final_rows)
    plot_reproduction(figure_path, model, args, time, mean, stderr)

    comparison_path = None
    if args.paper_pdf:
        paper_crop = outdir / f"paper_{args.paper_panel}_model_A_crop.png"
        _, paper_label = crop_paper_model_a(args.paper_pdf, paper_crop, args.paper_panel)
        comparison_path = outdir / f"{stem}_comparison.png"
        make_comparison(comparison_path, paper_crop, figure_path, paper_label)

    print("wrote", outdir)
    print("population_csv", population_csv)
    print("figure", figure_path)
    if comparison_path is not None:
        print("comparison", comparison_path)
    print(
        "final_pop",
        " ".join(f"P{i + 1}={value:.6f}" for i, value in enumerate(mean[-1])),
    )


if __name__ == "__main__":
    main()
