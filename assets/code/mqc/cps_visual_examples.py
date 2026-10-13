"""Reproduce the visual teaching examples in MQC VI and VII.

Run from any directory: python path/to/cps_visual_examples.py
Dependencies: NumPy and Matplotlib. Output defaults to assets/img/mqc-series.
These are analytic geometry and deterministic quadrature checks, not a
nonadiabatic benchmark or a test of approximate nuclear dynamics.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# Declared before any computed result is inspected.
GATES = {
    "algebra_max_abs": 2e-13,
    "quadrature_refinement_max_abs": 2e-13,
    "curve_high_frequency_rms": 1e-4,
    "curve_high_frequency_max": 2e-4,
    "curve_refinement_max_abs": 1e-4,
    "endpoint_slope_jump_normalized": 2e-3,
}
BLUE, ORANGE, GRAY = "#316583", "#b46c35", "#777777"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.titlesize": 14, "axes.labelsize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "svg.fonttype": "none", "svg.hashsalt": "cps-visual-examples",
})
SIGMA = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]],
                  [[1, 0], [0, -1]]], dtype=complex)
IDENTITY = np.eye(2)
RHO = np.array([[.9, .3], [.3, .1]], dtype=complex)
RVEC = np.array([.6, 0., .8])
GAMMA_W = (np.sqrt(3.)-1)/2


def save(fig, name, output):
    fig.savefig(output/(name+".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output/(name+".svg"), bbox_inches="tight",
                metadata={"Date": None})
    # Explicit LF is part of the public asset format.
    svg = output/(name+".svg")
    svg.write_bytes(b"\n".join(line.rstrip() for line in
                              svg.read_bytes().replace(b"\r\n", b"\n").splitlines())+b"\n")
    plt.close(fig)


def curve_quality(x, y):
    x, y = np.asarray(x), np.asarray(y)
    scale = max(float(np.ptp(y)), 1.)
    residual = y[1:-1]-(y[:-2]+y[2:])/2
    slopes = np.diff(y)
    endpoint_jump = max(abs(slopes[1]-slopes[0]),
                        abs(slopes[-1]-slopes[-2]))/scale
    refinement = np.max(np.abs(y-np.interp(x, x[::2], y[::2])))
    result = {
        "finite": bool(np.isfinite(x).all() and np.isfinite(y).all()),
        "strictly_increasing_x": bool((np.diff(x)>0).all()),
        "duplicate_x": int(len(x)-len(np.unique(x))),
        "high_frequency_residual_rms_normalized": float(np.sqrt(np.mean(residual**2))/scale),
        "high_frequency_residual_max_normalized": float(np.max(np.abs(residual))/scale),
        "endpoint_slope_jump_normalized": float(endpoint_jump),
        "unexpected_isolated_spikes": int(np.sum(np.abs(residual)/scale>GATES["curve_high_frequency_max"])),
        "coarse_fine_interpolation_max_abs": float(refinement),
        "audit_definition": "Residual to the adjacent-point linear interpolant; no data smoothing applied.",
    }
    result["passed"] = bool(result["finite"] and result["strictly_increasing_x"]
        and not result["duplicate_x"] and not result["unexpected_isolated_spikes"]
        and result["high_frequency_residual_rms_normalized"]<GATES["curve_high_frequency_rms"]
        and result["high_frequency_residual_max_normalized"]<GATES["curve_high_frequency_max"]
        and endpoint_jump<GATES["endpoint_slope_jump_normalized"]
        and refinement<GATES["curve_refinement_max_abs"])
    return result


def sphere_grid(nz_count, phi_count):
    nz, weights = np.polynomial.legendre.leggauss(nz_count)
    phi = 2*np.pi*np.arange(phi_count)/phi_count
    z, p = np.meshgrid(nz, phi, indexing="ij")
    n = np.stack([np.sqrt(1-z*z)*np.cos(p),
                  np.sqrt(1-z*z)*np.sin(p), z], axis=-1).reshape(-1, 3)
    # dmu = d(nz) dphi / (4*pi), normalized to one.
    w = np.repeat(weights/(2*phi_count), phi_count)
    pi = (IDENTITY+np.einsum("ka,aij->kij", n, SIGMA))/2
    return n, pi, w


def reconstruct(gamma, nz_count=12, phi_count=24, matched=True):
    n, pi, weights = sphere_grid(nz_count, phi_count)
    s = 1+2*gamma
    kernel = s*pi-gamma*IDENTITY
    dual = 3/s*pi-(1-gamma)/s*IDENTITY if matched else kernel
    symbol = np.einsum("ij,kji->k", RHO, kernel).real
    rebuilt = 2*np.einsum("k,k,kij->ij", weights, symbol, dual)
    return rebuilt


def su2_figure(output, metrics):
    angle = np.linspace(0, 4*np.pi, 1441)
    u = np.stack([np.cos(angle/2), np.sin(angle/2)], axis=-1)
    n = np.stack([np.sin(angle), np.zeros_like(angle), np.cos(angle)], axis=-1)
    metrics["spinor_c0"] = curve_quality(angle, u[:, 0])
    metrics["spinor_c1"] = curve_quality(angle, u[:, 1])
    metrics["bloch_nx"] = curve_quality(angle, n[:, 0])
    metrics["bloch_nz"] = curve_quality(angle, n[:, 2])
    fig = plt.figure(figsize=(6.8, 10.2))
    fig.suptitle(r"$U_y(\alpha)=e^{-i\alpha\sigma_y/2}$"+"\n"+
                 r"$u\sim e^{i\chi}u$: common phase is removed in the Bloch ray", fontsize=14)
    ax = fig.add_subplot(211)
    ax.plot(u[:, 0], u[:, 1], color=BLUE, lw=2)
    ax.scatter([1, -1], [0, 0], color=[BLUE, ORANGE], s=70, zorder=5)
    ax.annotate(r"$u(0)=u(4\pi)$", (1, 0), xytext=(.05, .23))
    ax.annotate(r"$u(2\pi)=-u(0)$", (-1, 0), xytext=(-.7, -.35))
    ax.annotate("", xy=(.5, .866), xytext=(.72, .7),
                arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 2})
    ax.set(xlim=(-1.25, 1.25), ylim=(-1.2, 1.2), aspect="equal",
           xlabel=r"Real amplitude $c_0$", ylabel=r"Real amplitude $c_1$",
           title="A. Real spinor orbit: a slice of state space")
    ax.axhline(0, color="#dddddd", lw=.7)
    ax.axvline(0, color="#dddddd", lw=.7)
    ax = fig.add_subplot(212, projection="3d")
    phi, theta = np.meshgrid(np.linspace(0, 2*np.pi, 25), np.linspace(0, np.pi, 13))
    ax.plot_wireframe(np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi),
                      np.cos(theta), color="#bbbbbb", linewidth=.45, alpha=.6)
    ax.plot(n[:721, 0], n[:721, 1], n[:721, 2], color=BLUE, lw=3)
    ax.scatter([0], [0], [1], s=65, color=BLUE)
    ax.scatter([0], [0], [-1], s=65, color=ORANGE)
    ax.text2D(.5, .92, "0, 2π, 4π: same ray (blue)", transform=ax.transAxes, ha="center", fontsize=11)
    fig.text(.5, .035, "π: opposite Bloch direction (orange)", ha="center", fontsize=11)
    ax.set(xlabel=r"$n_x$", ylabel=r"$n_y$", zlabel=r"$n_z$",
           title="B. Bloch sphere: the orbit of rays")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-55)
    fig.subplots_adjust(hspace=.4, top=.86, bottom=.12)
    save(fig, "cps-su2-projective-orbit", output)
    return float(np.max(np.abs(n[:, 0]-2*u[:, 0]*u[:, 1])))


def symbol_figure(output):
    phi = np.linspace(-np.pi, np.pi, 361)
    nz = np.linspace(-1, 1, 241)
    p, z = np.meshgrid(phi, nz)
    dot = .6*np.sqrt(1-z*z)*np.cos(p)+.8*z
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.6), sharex=True, sharey=True)
    for ax, gamma, label in zip(axes, [0, GAMMA_W, .5],
                               ["Q: γ = 0", "W: γ = (√3 − 1)/2", "γ = 1/2"]):
        values = (1+(1+2*gamma)*dot)/2
        im = ax.pcolormesh(phi/np.pi, nz, values, cmap="RdBu_r", vmin=-.5, vmax=1.5,
                           shading="auto", rasterized=True)
        ax.set(ylabel=r"$n_z=\cos\theta$", title=label)
        if gamma>0:
            ax.contour(phi/np.pi, nz, values, levels=[0], colors="#333333", linewidths=1)
        fig.colorbar(im, ax=ax, label=r"$f_{\rho,\gamma}$", pad=.025)
    axes[-1].set_xlabel(r"Azimuth $\phi/\pi$")
    fig.suptitle("One pure density matrix, three scalar symbols\nBloch vector r = (0.6, 0, 0.8); black contour: fρ = 0", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, "cps-density-to-symbol", output)


def box(ax, xy, width, height, text, color=BLUE, size=13):
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=.02,rounding_size=.025",
                          edgecolor=color, facecolor="#fafafa", linewidth=1.5)
    ax.add_patch(patch)
    ax.text(xy[0]+width/2, xy[1]+height/2, text, ha="center", va="center", fontsize=size)


def arrow(ax, start, end, color=BLUE, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=17,
                                color=color, linewidth=1.6,
                                linestyle="--" if dashed else "-"))


def inverse_diagram(output):
    fig, ax = plt.subplots(figsize=(6.8, 8.5))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    ax.set_title("Forward and inverse mapping are different operations", pad=20)
    box(ax, (.12, .77), .76, .18, "Quantum operator ρ\nPopulations and coherences in one matrix", size=13)
    arrow(ax, (.2, .74), (.2, .64))
    ax.text(.27, .69, r"$f_\rho(\mathbf{n})=\mathrm{Tr}(\rho K_\gamma)$", va="center", fontsize=13)
    box(ax, (.12, .42), .76, .19, "Scalar function on every Bloch direction\nIts values can be signed;\nthey are not individual physical populations", size=12)
    arrow(ax, (.2, .39), (.2, .24))
    ax.text(.27, .32, "Sphere quadrature with the matched dual\n"+
            r"$\int d\mu=1,\quad d\mu=d\phi\,dn_z/(4\pi)$", va="center", fontsize=11.5)
    box(ax, (.12, .025), .76, .185,
        r"$\rho_{\rm rebuilt}=2\int f_\rho(\mathbf{n})\bar K_\gamma(\mathbf{n})\,d\mu$"
        "\nThe whole weighted integral recovers the matrix", size=12)
    fig.tight_layout()
    save(fig, "cps-forward-inverse-map", output)


def reconstruction_figure(output, metrics):
    gammas = np.linspace(-.25, .7, 401)
    matched = np.array([reconstruct(g).real for g in gammas])
    wrong = np.array([reconstruct(g, matched=False).real for g in gammas])
    analytic_wrong = np.array([IDENTITY/2+(1+2*g)**2/3*(RHO.real-IDENTITY/2) for g in gammas])
    error = float(np.max(np.abs(matched-RHO.real)))
    control_error = float(np.max(np.abs(wrong-analytic_wrong)))
    refined = np.array([reconstruct(g, 24, 48).real for g in gammas[::20]])
    refinement = float(np.max(np.abs(refined-matched[::20])))
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 7.3), sharex=True)
    for ax, ij, title in zip(axes, [(0, 0), (0, 1)],
                             ["A. Population", "B. Real coherence"]):
        y = matched[:, ij[0], ij[1]]
        yy = wrong[:, ij[0], ij[1]]
        key = "rho00" if ij==(0, 0) else "rho01_real"
        metrics[key+"_matched"] = curve_quality(gammas, y)
        metrics[key+"_same_kernel"] = curve_quality(gammas, yy)
        ax.plot(gammas, yy, color=ORANGE, label="Same kernel on both sides", lw=2)
        ax.plot(gammas, y, color=BLUE, label="Matched dual = target", lw=2)
        ax.axvline(GAMMA_W, color=GRAY, ls=":", lw=1.3, label="W" if ij==(0, 0) else None)
        ax.set(ylabel=r"$\rho_{00}$" if ij==(0, 0) else r"$\mathrm{Re}\,\rho_{01}$", title=title)
        ax.grid(alpha=.17)
    axes[0].legend(fontsize=11, loc="upper left")
    axes[-1].set_xlabel(r"Representation parameter $\gamma$")
    axes[-1].set_xticks([-.25, 0, GAMMA_W, .5, .7], ["−0.25", "0 (Q)", "W", "1/2", "0.7"])
    fig.suptitle("Reconstruct ρ = [[0.9, 0.3], [0.3, 0.1]]", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96))
    save(fig, "cps-dual-reconstruction-sweep", output)
    return error, control_error, refinement


def generator_figure(output, metrics):
    nz = np.linspace(-1, 1, 801)
    fig, ax = plt.subplots(figsize=(7.1, 4.8))
    matrix = -np.ones_like(nz)
    ax.plot(nz, matrix, color="#222222", lw=2, ls="--", label="Mapped matrix generator: −1")
    metrics["matrix_generator"] = curve_quality(nz, matrix)
    for gamma, label, color in [(0, "Scalar Q", GRAY), (GAMMA_W, "Scalar W", BLUE), (.5, "Scalar γ = 1/2", ORANGE)]:
        values = -(1+2*gamma)**2*nz**2
        ax.plot(nz, values, color=color, lw=2, label=label)
        metrics[label] = curve_quality(nz, values)
    ax.set(xlabel=r"Bloch direction $n_z$", ylabel=r"Generator acting on $b=Psn_z$",
           title="The nuclear force needs an operator product")
    ax.legend(fontsize=10.5, loc="lower center")
    ax.grid(alpha=.16)
    fig.tight_layout()
    save(fig, "cps-coupled-generator-example", output)


def feedback_diagram(output):
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 8.6))
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    ax = axes[0]
    ax.set_title("A. A common prescribed nuclear path", pad=15)
    box(ax, (.2, .77), .6, .17, "External R(t) → one common U(t)", size=13)
    for x, label in [(.05, "1"), (.375, "2"), (.7, "3")]:
        center = x+.125
        arrow(ax, (.5, .74), (center, .59))
        box(ax, (x, .38), .25, .17, "Sample "+label+"\nElectronic u"+label+"(t)", size=11)
        arrow(ax, (center, .35), (center, .24), color=GRAY)
    box(ax, (.05, .035), .9, .175,
        "Weighted reconstruction\nCommon U can leave the integral: ρ(t) = Uρ(0)U†", size=12)
    ax = axes[1]
    ax.set_title("B. Each sample drives its own nuclei", pad=15)
    for x, label in [(.05, "1"), (.375, "2"), (.7, "3")]:
        box(ax, (x, .74), .25, .17, "Nuclei "+label+"\nR"+label+"(t), P"+label+"(t)", color=ORANGE, size=11)
        box(ax, (x, .31), .25, .17, "Electronic\nu"+label+"(t)", size=11)
        arrow(ax, (x+.055, .71), (x+.055, .51))
        arrow(ax, (x+.195, .51), (x+.195, .71), color=ORANGE, dashed=True)
        arrow(ax, (x+.125, .28), (x+.125, .215), color=GRAY)
    box(ax, (.05, .045), .9, .145,
        "Different U₁, U₂, U₃ stay inside the integral\nThe common-unitary proof no longer applies", size=12)
    fig.text(.5, .016, "Blue: nuclei → electronics.  Orange: force feedback.  Gray: reconstruction.",
             ha="center", fontsize=10)
    fig.subplots_adjust(hspace=.35, top=.93, bottom=.065)
    save(fig, "cps-prescribed-path-feedback", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[2]/"img"/"mqc-series")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {}
    orbit_error = su2_figure(args.output, metrics)
    symbol_figure(args.output)
    inverse_diagram(args.output)
    reconstruction_error, control_error, refinement = reconstruction_figure(args.output, metrics)
    generator_figure(args.output, metrics)
    feedback_diagram(args.output)
    _, pi, weights = sphere_grid(12, 24)
    moment_error = float(np.max(np.abs(np.einsum("k,kij->ij", weights, pi)-IDENTITY/2)))
    passed = bool(max(orbit_error, reconstruction_error, control_error, moment_error)<GATES["algebra_max_abs"]
                  and refinement<GATES["quadrature_refinement_max_abs"]
                  and all(x["passed"] for x in metrics.values()))
    names = ["cps-su2-projective-orbit", "cps-density-to-symbol", "cps-forward-inverse-map",
             "cps-dual-reconstruction-sweep", "cps-coupled-generator-example", "cps-prescribed-path-feedback"]
    receipt = {
        "passed": passed, "scope": "Analytic geometry and deterministic electronic quadrature only",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gates": GATES, "rho_real": RHO.real.tolist(), "rho_imag": RHO.imag.tolist(),
        "sphere_quadrature": {"rule": "Gauss-Legendre nz times periodic uniform azimuth",
                              "base": [12, 24], "refined": [24, 48], "normalized_measure": True},
        "orbit_identity_max_abs": orbit_error, "first_moment_max_abs": moment_error,
        "matched_reconstruction_max_abs": reconstruction_error,
        "same_kernel_analytic_control_max_abs": control_error,
        "quadrature_refinement_max_abs": refinement,
        "curve_quality": metrics,
        "figures": {name:{ext:hashlib.sha256((args.output/(name+"."+ext)).read_bytes()).hexdigest()
                         for ext in ["png", "svg"]} for name in names},
    }
    (args.output/"cps-visual-examples.json").write_text(json.dumps(receipt, indent=2)+"\n",
                                                   encoding="utf-8", newline="\n")
    print(json.dumps({k:v for k,v in receipt.items() if k not in ["curve_quality", "figures"]}, indent=2))
    if not passed:
        raise SystemExit("A declared numerical gate failed; inspect the JSON receipt.")


if __name__ == "__main__":
    main()
