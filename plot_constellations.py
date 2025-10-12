#!/usr/bin/env python3
"""
Constellation plotter using QAM and PSK modulations.
Shows constellation points with binary labels.
"""

import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from ofdm_based_systems.constellation.models import QAMConstellationMapper, PSKConstellationMapper

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
sys.path.insert(0, src_dir)


def binary_representation(index: int, bits_per_symbol: int) -> str:
    """Return the binary representation of an index with correct padding."""
    return format(index, f"0{bits_per_symbol}b")


def plot_constellation(ax, constellation_mapper, title, no_power=False):
    """Plot a constellation with binary word labels."""
    constellation = constellation_mapper.constellation
    bits_per_symbol = constellation_mapper.bits_per_symbol

    # Calculate constellation power (average energy)
    avg_power = np.mean(np.abs(constellation) ** 2)

    # Calculate power of each point for color mapping
    powers = np.abs(constellation) ** 2

    if no_power:
        # Simple plot without power coloring
        ax.scatter(constellation.real, constellation.imag, color="blue", s=50, zorder=2)
    else:
        # Plot constellation points with color based on power
        scatter = ax.scatter(
            constellation.real,
            constellation.imag,
            c=powers,
            cmap="viridis",  # Color map: blue (low) to yellow (high)
            s=50,
            zorder=2,
        )

        # Add a colorbar to show power scale
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label("Symbol Power")

    # Draw unit circle for reference (especially useful for PSK)
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    ax.plot(circle_x, circle_y, "k--", alpha=0.3, zorder=1)

    # Add labels with binary representation
    for i, point in enumerate(constellation):
        # Get original binary index from coded index
        binary_index = i
        binary_str = binary_representation(binary_index, bits_per_symbol)

        # Position the label slightly offset from the point
        ax.annotate(
            binary_str,
            (point.real, point.imag),
            xytext=(5, 5),  # Small offset for readability
            textcoords="offset points",
            fontsize=8,
        )

    # Set plot limits with some padding
    max_val = max(np.max(np.abs(constellation.real)), np.max(np.abs(constellation.imag)))
    ax.set_xlim(-max_val * 1.2, max_val * 1.2)
    ax.set_ylim(-max_val * 1.2, max_val * 1.2)

    # Add grid, title, and labels with power information
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{title}\nAverage Power: {avg_power:.4f}")
    ax.set_xlabel("In-Phase")
    ax.set_ylabel("Quadrature")
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax.axvline(x=0, color="k", linestyle="-", alpha=0.3)
    ax.set_aspect("equal")

    # Calculate individual point powers
    powers = np.abs(constellation) ** 2

    # Calculate Peak-to-Average Power Ratio (PAPR)
    peak_power = np.max(powers)
    papr = peak_power / avg_power
    papr_db = 10 * np.log10(papr)

    # Add textbox with power statistics
    power_info = (
        f"Min Power: {np.min(powers):.4f}\n"
        f"Max Power: {peak_power:.4f}\n"
        f"Avg Power: {avg_power:.4f}\n"
        f"Std Dev: {np.std(powers):.4f}\n"
        f"PAPR: {papr:.2f} ({papr_db:.2f} dB)"
    )

    # Place the textbox in the lower right corner
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.95,
        0.05,
        power_info,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot QAM and PSK constellations with binary labels."
    )
    parser.add_argument(
        "order",
        type=int,
        help="Constellation order (must be a power of 2, QAM also requires perfect square)",
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Output file to save the plot (e.g., constellation.png)"
    )
    parser.add_argument(
        "--no-power", "-np", action="store_true", help="Disable power analysis coloring"
    )

    args = parser.parse_args()

    # Validate constellation order
    order = args.order

    # Create figure with two subplots for QAM and PSK
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    try:
        # Create QAM constellation mapper
        qam_mapper = QAMConstellationMapper(order)
        plot_constellation(
            ax1, qam_mapper, f"{qam_mapper.constellation_name} Constellation", args.no_power
        )
    except ValueError as e:
        ax1.text(
            0.5,
            0.5,
            f"Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax1.transAxes,
            color="red",
            fontsize=10,
        )
        ax1.set_title(f"{order}-QAM Constellation (Invalid)")

    try:
        # Create PSK constellation mapper
        psk_mapper = PSKConstellationMapper(order)
        plot_constellation(
            ax2, psk_mapper, f"{psk_mapper.constellation_name} Constellation", args.no_power
        )
    except ValueError as e:
        ax2.text(
            0.5,
            0.5,
            f"Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            color="red",
            fontsize=10,
        )
        ax2.set_title(f"{order}-PSK Constellation (Invalid)")

    fig.suptitle(f"Digital Modulation Constellations (Order = {order})")
    fig.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"Saved constellation plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
