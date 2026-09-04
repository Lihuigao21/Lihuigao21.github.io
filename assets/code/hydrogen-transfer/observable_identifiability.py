#!/usr/bin/env python3
"""Generate the synthetic identifiability diagnostic for Hydrogen Transfer I.

The calculation is intentionally a toy forward model, not a fit to experimental
data.  Two mechanisms are calibrated to the same effective H and D rates at
300 K, then compared away from that calibration point and in the time domain.

Dependencies: numpy, matplotlib.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "assets" / "img" / "hydrogen-transfer"
FIGURE_PATH = OUTPUT_DIR / "observable-identifiability.png"
DATA_PATH = OUTPUT_DIR / "observable-identifiability.csv"

R = 8.314462618e-3  # kJ mol^-1 K^-1
T_REF = 300.0


def anchored_arrhenius(temperature, rate_at_reference, activation_energy):
    """Arrhenius rate anchored to a specified value at T_REF."""
    temperature = np.asarray(temperature, dtype=float)
    exponent = -(activation_energy / R) * (1.0 / temperature - 1.0 / T_REF)
    return rate_at_reference * np.exp(exponent)


def serial_effective_rate(rate_1, rate_2):
    """Inverse mean first-passage time for two irreversible serial steps."""
    return rate_1 * rate_2 / (rate_1 + rate_2)


def sequential_product_population(time, rate_1, rate_2):
    """Product population for R -> I -> P with P_R(0)=1."""
    if np.isclose(rate_1, rate_2):
        return 1.0 - np.exp(-rate_1 * time) * (1.0 + rate_1 * time)
    remaining = (
        rate_2 * np.exp(-rate_1 * time)
        - rate_1 * np.exp(-rate_2 * time)
    ) / (rate_2 - rate_1)
    return 1.0 - remaining


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    temperature = np.linspace(240.0, 360.0, 241)

    # Model A: a single effective transfer step.
    rate_a_h = anchored_arrhenius(temperature, 100.0, 35.0)
    rate_a_d = anchored_arrhenius(temperature, 20.0, 39.0)

    # Model B: an isotope-insensitive gate followed by isotope-sensitive transfer.
    # The reference values give k_eff,H = 100 s^-1 and k_eff,D = 20 s^-1.
    rate_gate = anchored_arrhenius(temperature, 125.0, 45.0)
    rate_transfer_h = anchored_arrhenius(temperature, 500.0, 15.0)
    rate_transfer_d = anchored_arrhenius(temperature, 500.0 / 21.0, 19.0)
    rate_b_h = serial_effective_rate(rate_gate, rate_transfer_h)
    rate_b_d = serial_effective_rate(rate_gate, rate_transfer_d)

    kie_a = rate_a_h / rate_a_d
    kie_b = rate_b_h / rate_b_d

    time = np.linspace(0.0, 0.040, 401)
    product_a = 1.0 - np.exp(-100.0 * time)
    product_b = sequential_product_population(time, 125.0, 500.0)

    table = np.column_stack(
        [temperature, rate_a_h, rate_a_d, rate_b_h, rate_b_d, kie_a, kie_b]
    )
    np.savetxt(
        DATA_PATH,
        table,
        delimiter=",",
        header="temperature_K,model_A_kH_s-1,model_A_kD_s-1,model_B_kH_s-1,model_B_kD_s-1,model_A_KIE,model_B_KIE",
        comments="",
        fmt="%.8g",
    )

    colors = {"a": "#1f5a94", "b": "#b34a3c", "ref": "#555555"}
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), constrained_layout=True)

    ax = axes[0]
    ax.semilogy(temperature, rate_a_h, color=colors["a"], lw=2.2, label="A: one effective step")
    ax.semilogy(temperature, rate_b_h, color=colors["b"], lw=2.2, ls="--", label="B: gate + transfer")
    ax.scatter([T_REF], [100.0], s=35, color=colors["ref"], zorder=4)
    ax.axvline(T_REF, color=colors["ref"], lw=0.9, ls=":")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Effective $k_H$ (s$^{-1}$)")
    ax.set_title("(a) Same rate at 300 K")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(temperature, kie_a, color=colors["a"], lw=2.2, label="A: one effective step")
    ax.plot(temperature, kie_b, color=colors["b"], lw=2.2, ls="--", label="B: gate + transfer")
    ax.scatter([T_REF], [5.0], s=35, color=colors["ref"], zorder=4)
    ax.axvline(T_REF, color=colors["ref"], lw=0.9, ls=":")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"KIE $=k_H/k_D$")
    ax.set_title("(b) Same KIE at 300 K")

    ax = axes[2]
    ax.plot(1.0e3 * time, product_a, color=colors["a"], lw=2.2, label="A: single exponential")
    ax.plot(1.0e3 * time, product_b, color=colors["b"], lw=2.2, ls="--", label="B: resolved intermediate")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Product population")
    ax.set_ylim(0.0, 1.03)
    ax.set_title("(c) Time resolution separates them")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    for ax in axes:
        ax.grid(alpha=0.18, lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")

    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    print(f"wrote {FIGURE_PATH.relative_to(ROOT)}")
    print(f"wrote {DATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
