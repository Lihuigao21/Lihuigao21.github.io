"""Check a single diagonal harmonic density-matrix segment in unit oscillator units."""

from argparse import ArgumentParser
from math import pi, sin, sqrt, tanh


def link_coefficients(epsilon):
    """Return C in rho(x, x; epsilon) proportional to exp(-C * x**2)."""
    node = (1 - 1 / sqrt(3)) / 2
    midpoint_variance = epsilon / 4
    sine_variance = 2 * epsilon / pi**2
    constant = epsilon / 2

    # Write the conditional exponent as -A*z**2 - D*x*z - constant*x**2.
    # Integrating over the added coordinate z gives C = constant - D**2/(4*A).
    midpoint_a = 1 / (2 * midpoint_variance) + epsilon / 4
    midpoint_d = epsilon / 2
    sine_free_a = 1 / (2 * sine_variance)
    sampled_sine = sin(pi * node)
    two_node_a = sine_free_a + epsilon * sampled_sine**2 / 2
    two_node_d = epsilon * sampled_sine
    exact_sine_a = sine_free_a + epsilon / 4
    exact_sine_d = 2 * epsilon / pi
    return {
        "Exact bridge": tanh(epsilon / 2),
        "One midpoint, primitive half-links": constant - midpoint_d**2 / (4 * midpoint_a),
        "One sine, two Gauss-Legendre nodes": constant - two_node_d**2 / (4 * two_node_a),
        "One sine, analytic integral": constant - exact_sine_d**2 / (4 * exact_sine_a),
    }


def area_variance_fractions():
    """Return the retained fractions of the free variance of the segment mean."""
    node = (1 - 1 / sqrt(3)) / 2
    return {
        "Midpoint tent": 12 / 16,
        "One sine, two Gauss-Legendre nodes": 24 * sin(pi * node)**2 / pi**2,
        "One sine, analytic integral": 96 / pi**4,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, default=1.0)
    args = parser.parse_args()
    if not args.epsilon > 0:
        parser.error("--epsilon must be positive")
    for label, coefficient in link_coefficients(args.epsilon).items():
        print(f"{label:38s} C = {coefficient:.9f}")
    for label, fraction in area_variance_fractions().items():
        print(f"{label:38s} free area variance = {100 * fraction:.3f}%")


if __name__ == "__main__":
    main()
