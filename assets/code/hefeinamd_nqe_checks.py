"""Analytic diagnostics for nuclear fluctuations in EPC dynamics.

Run from any directory:
    python path/to/assets/code/hefeinamd_nqe_checks.py

Requires NumPy, SciPy, and Matplotlib. The figures compare analytic distributions,
coherence factors, and bare resonant spectral weights. They are not a material
simulation, an implementation of HefeiNAMD hopping, or a lifetime benchmark.

For a real mass-weighted harmonic mode, sigma^2 = ell^2 (2 n + 1).
The sampling figure uses x = Q/sigma. Its coherence panel shows the signed real
coherence, J0(z), rather than its absolute value. The transition figure compares
weights in a weak-coupling, energy-resolved single-channel limit only.
"""

from argparse import ArgumentParser
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.special import j0


GAUSSIAN = "#325b73"
FIXED = "#b56d45"
MUTED = "#777777"


def save_svg(figure, path):
    figure.savefig(path, metadata={"Date": None}, bbox_inches="tight")
    plt.close(figure)
    text = path.read_text(encoding="utf-8")
    path.write_text("\n".join(s.rstrip() for s in text.splitlines()) + "\n",
                    encoding="utf-8", newline="\n")


def average_phase(function):
    return quad(function, 0.0, 2.0 * np.pi, epsabs=2e-12)[0] / (2.0 * np.pi)


def gaussian_average(function):
    return quad(lambda x: function(x) * np.exp(-0.5*x*x) / np.sqrt(2*np.pi),
                -np.inf, np.inf, epsabs=2e-12)[0]


def diagnostics():
    """Check formulas by quadrature, without random-sampling uncertainty."""
    fixed = {str(k): average_phase(lambda phi: (np.sqrt(2)*np.cos(phi))**k)
             for k in (0, 2, 4)}
    gaussian = {str(k): gaussian_average(lambda x: x**k) for k in (0, 2, 4)}
    # J/(n+1/2) is an exponential variable u; x = sqrt(2u) cos(phi).
    wigner_polar = {}
    for k in (0, 2, 4):
        radial = quad(lambda u: np.exp(-u) * (2*u)**(k/2),
                      0.0, np.inf, epsabs=2e-12)[0]
        angular = average_phase(lambda phi: np.cos(phi)**k)
        wigner_polar[str(k)] = radial * angular

    characteristic_errors = []
    for z in (0.1, 1.0, 3.0, 5.0):
        fixed_value = average_phase(lambda phi: np.cos(z*np.cos(phi)))
        gaussian_value = gaussian_average(lambda x: np.cos(z*x/np.sqrt(2)))
        characteristic_errors.extend([abs(fixed_value-j0(z)),
                                       abs(gaussian_value-np.exp(-z*z/4))])

    # Linear coupling g sigma_x (b+b^dagger), hbar=omega=Omega=1.
    # This is O(g^2) time-dependent perturbation theory, not an exact TDSE run.
    t = 1e-5
    resonant = 1.0  # |integral exp[i(Omega-omega)s] ds|^2 / t^2
    counterrotating = float(np.sinc(t/np.pi)**2)
    short_time = []
    for n in (0.0, 0.4, 3.0):
        quantum = (n+1)*resonant + n*counterrotating
        symmetric = (n+0.5)*(resonant+counterrotating)
        target = 2*n+1
        short_time.append({"n": n, "t_omega": t,
                           "quantum_P_over_g2_t2": quantum,
                           "phase_averaged_P_over_g2_t2": symmetric,
                           "short_time_limit": target})

    k_b_ev = 8.617333262145e-5
    x = 0.160 / (k_b_ev * 300.0)
    n = 1.0 / np.expm1(x)
    ratio = (n+0.5)/(n+1)
    checks = {
        "fixed_moments": max(abs(fixed[str(k)]-v)
                             for k, v in ((0, 1), (2, 1), (4, 1.5))) < 1e-10,
        "gaussian_moments": max(abs(gaussian[str(k)]-v)
                                for k, v in ((0, 1), (2, 1), (4, 3))) < 1e-10,
        "wigner_polar_moments": max(abs(wigner_polar[str(k)]-v)
                                    for k, v in ((0, 1), (2, 1), (4, 3))) < 1e-10,
        "coherence_characteristic_functions": max(characteristic_errors) < 1e-10,
        "common_short_time_coefficient": max(
            abs(item[key]-item["short_time_limit"])
            for item in short_time
            for key in ("quantum_P_over_g2_t2", "phase_averaged_P_over_g2_t2")) < 1e-8,
        "emission_ratio_identity": abs(ratio-0.5*(1+np.exp(-x))) < 1e-14,
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    result = {
        "scope": "Analytic single-mode diagnostics; not a HefeiNAMD lifetime test.",
        "coordinate": "x=Q/sigma, sigma^2=ell^2(2n+1)",
        "normalized_moments_by_order": {
            "fixed_amplitude_uniform_phase": fixed,
            "thermal_gaussian": gaussian,
            "wigner_exponential_action_uniform_phase": wigner_polar,
        },
        "maximum_characteristic_function_error": max(characteristic_errors),
        "short_time_perturbation_check": short_time,
        "160_meV_300_K": {"hbar_omega_over_kBT": x, "bose_occupation": n,
                         "symmetric_over_quantum_emission_weight": ratio},
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not result["passed"]:
        raise AssertionError(json.dumps(result, indent=2))
    return result


def style_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.17, linewidth=0.6)
    axis.set_axisbelow(True)


def sampling_figure(path):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.19, top=0.86, wspace=0.29)
    ax = axes[0]
    x = np.linspace(-3.5, 3.5, 1001)
    ax.plot(x, np.exp(-x*x/2)/np.sqrt(2*np.pi), color=GAUSSIAN,
            label="Gaussian Wigner", linewidth=2)
    bound = np.sqrt(2)
    inside = np.linspace(-bound+1e-4, bound-1e-4, 2001)
    ax.plot(inside, 1/(np.pi*np.sqrt(2-inside**2)), color=FIXED,
            label="Fixed amplitude", linewidth=2)
    for sign in (-1, 1):
        ax.axvline(sign*bound, color=FIXED, linestyle=":", linewidth=1)
    ax.set(xlim=(-3.5, 3.5), ylim=(0, 1.05), xlabel=r"Normal coordinate $x=Q/\sigma$",
           ylabel=r"Probability density $p(x)$", title="(a) Equal variance, different distributions")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.95, fontsize=9)
    ax.text(0.98, 0.96, r"Both: $\langle x^2\rangle=1$" + "\n" +
            r"Fixed: $\langle x^4\rangle=1.5$" + "\n" +
            r"Gaussian: $\langle x^4\rangle=3$", ha="right", va="top",
            transform=ax.transAxes, fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95})
    ax.text(0.5, -0.24, r"Dotted guides: $x=\pm\sqrt{2}$; integrable divergences",
            ha="center", transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax = axes[1]
    z = np.linspace(0, 9, 901)
    ax.plot(z, np.exp(-z*z/4), color=GAUSSIAN, label=r"Gaussian: $e^{-z^2/4}$", linewidth=2)
    ax.plot(z, j0(z), color=FIXED, label=r"Fixed amplitude: $J_0(z)$", linewidth=2)
    ax.axhline(0, color=MUTED, linewidth=0.7)
    ax.set(xlim=(0, 9), ylim=(-0.48, 1.05), xlabel=r"Accumulated-phase amplitude $z$",
           ylabel=r"Real coherence factor $L$", title="(b) Same initial curvature, different coherence")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.text(0.5, -0.24, r"Both begin $1-z^2/4$; fourth-order terms differ",
            ha="center", transform=ax.transAxes, fontsize=8.5, color=MUTED)
    for ax in axes:
        style_axis(ax)
    save_svg(fig, path)


def transition_figure(path, reference):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.23, top=0.85, wspace=0.30)
    x = np.linspace(0, 8, 801)
    emission = 1/(1+np.exp(-x))
    absorption = 1-emission
    ax = axes[0]
    ax.plot(x, emission, color=GAUSSIAN, linewidth=2, label=r"Quantum emission: $(n+1)/(2n+1)$")
    ax.plot(x, absorption, color=GAUSSIAN, linestyle="--", linewidth=2,
            label=r"Quantum absorption: $n/(2n+1)$")
    ax.axhline(0.5, color=FIXED, linewidth=2, linestyle=":",
               label="Symmetric: 1/2 per channel")
    ax.set(xlim=(0, 8), ylim=(0, 1.04), xlabel=r"$x=\hbar\omega/(k_B T)$",
           ylabel="Fraction of total bare spectral weight",
           title="(a) Symmetrization redistributes spectral weight")
    ax.legend(loc="center right", frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.95, fontsize=8.5)
    ax = axes[1]
    ratio = 0.5*(1+np.exp(-x))
    ax.plot(x, ratio, color=FIXED, linewidth=2)
    point = reference["160_meV_300_K"]
    px = point["hbar_omega_over_kBT"]
    py = point["symmetric_over_quantum_emission_weight"]
    ax.scatter([px], [py], color=GAUSSIAN, s=28, zorder=3)
    ax.annotate("160 meV, 300 K\n" + f"n = {point['bose_occupation']:.5f}\nR = {py:.3f}",
                xy=(px, py), xytext=(3.1, 0.69), fontsize=9,
                arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1)
    ax.set(xlim=(0, 8), ylim=(0.47, 1.04), xlabel=r"$x=\hbar\omega/(k_B T)$",
           ylabel=r"Emission-weight ratio $R$",
           title=r"(b) $R=(n+1/2)/(n+1)$")
    for ax in axes:
        style_axis(ax)
    fig.text(0.5, 0.025,
             "Weak-coupling, energy-resolved channel comparison; not a carrier-cooling lifetime benchmark.",
             ha="center", fontsize=9, color=MUTED)
    save_svg(fig, path)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        help="Write both figures and nqe-check.json into this directory.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output_dir if args.output_dir else root / "assets/img/hefeinamd-series"
    summary = (output / "nqe-check.json" if args.output_dir else
               root / "assets/data/hefeinamd-nqes/nqe-check.json")
    output.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 10, "axes.labelsize": 10,
                         "svg.fonttype": "path", "svg.hashsalt": "hefeinamd-nqe-checks"})
    result = diagnostics()
    sampling_figure(output / "nqe-sampling.svg")
    transition_figure(output / "nqe-transition-weights.svg", result)
    summary.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n",
                       encoding="utf-8", newline="\n")
    print(f"Passed {len(result['checks'])} analytic checks; wrote two SVG figures and nqe-check.json.")


if __name__ == "__main__":
    main()
