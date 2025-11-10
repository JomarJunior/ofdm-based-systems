"""Waterfilling Diagram Visualization.

This script creates a clear visualization of the waterfilling power allocation
algorithm showing the "water container" analogy with channel floor and water level.
"""

import matplotlib.pyplot as plt
import numpy as np

from ofdm_based_systems.power_allocation.models import (
    UniformPowerAllocation,
    WaterfillingPowerAllocation,
)


def plot_waterfilling_diagram():
    """Generate waterfilling diagram with channel floor and power allocation."""

    # Configuration from simulation settings
    num_subcarriers = 64
    total_power = 1.0
    snr_db = 20.0
    noise_power = 10 ** (-snr_db / 10)

    # Create frequency-selective channel (multipath)
    # Simulate realistic multipath: h(t) = δ(t) + 0.7·δ(t-1) + 0.4·δ(t-2) + 0.2·δ(t-3)
    channel_impulse = np.array([1.0, 0.7, 0.4, 0.2], dtype=np.complex128)
    channel_response = np.fft.fft(channel_impulse, n=num_subcarriers)
    channel_gains = np.abs(channel_response) ** 2

    print("=" * 80)
    print("Waterfilling Diagram Visualization")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  - Number of subcarriers: {num_subcarriers}")
    print(f"  - Total power budget: {total_power}")
    print(f"  - SNR: {snr_db} dB")
    print(f"  - Noise power: {noise_power:.6f}")
    print(f"  - Channel: Multipath (4 taps)")

    # Calculate power allocations
    uniform_allocator = UniformPowerAllocation(
        total_power=total_power, num_subcarriers=num_subcarriers
    )
    uniform_power = uniform_allocator.allocate()

    waterfilling_allocator = WaterfillingPowerAllocation(
        total_power=total_power, channel_gains=channel_gains, noise_power=noise_power
    )
    waterfilling_power = waterfilling_allocator.allocate()

    # Calculate floor (inverse SNR: N₀/|H[k]|²)
    floor = noise_power / channel_gains

    # Calculate water level
    water_level = waterfilling_power + floor
    allocated = waterfilling_power > 1e-10
    mean_water_level = np.mean(water_level[allocated]) if np.any(allocated) else 0

    print(f"\nWaterfilling Results:")
    print(f"  - Mean water level (μ): {mean_water_level:.6f}")
    print(f"  - Min floor level: {np.min(floor):.6f}")
    print(f"  - Max floor level: {np.max(floor):.6f}")
    print(f"  - Subcarriers with zero power: {np.sum(waterfilling_power < 1e-10)}")

    # Create the visualization
    create_waterfilling_diagram(
        channel_gains,
        floor,
        waterfilling_power,
        uniform_power,
        mean_water_level,
        num_subcarriers,
        noise_power,
    )

    print(f"\n{'=' * 80}")
    print("Visualization saved to 'waterfilling_diagram.png'")
    print("=" * 80)


def create_waterfilling_diagram(
    channel_gains,
    floor,
    waterfilling_power,
    uniform_power,
    water_level,
    num_subcarriers,
    noise_power,
):
    """Create detailed waterfilling diagram visualization.

    Args:
        channel_gains: Channel power gains |H[k]|²
        floor: Inverse SNR (N₀/|H[k]|²) - the container floor
        waterfilling_power: Waterfilling power allocation
        uniform_power: Uniform power allocation for comparison
        water_level: Constant water level (μ)
        num_subcarriers: Number of subcarriers
        noise_power: Noise power
    """

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    subcarriers = np.arange(num_subcarriers)

    # Main waterfilling diagram (large, top subplot)
    ax_main = fig.add_subplot(gs[0, :])

    # Plot the floor (inverse channel quality)
    bars_floor = ax_main.bar(
        subcarriers,
        floor,
        width=0.8,
        color="#8B4513",  # Saddle brown
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
        label="Floor: N₀/|H[k]|²",
    )

    # Plot the water (power allocation) on top of floor
    bars_water = ax_main.bar(
        subcarriers,
        waterfilling_power,
        bottom=floor,
        width=0.8,
        color="#4169E1",  # Royal blue (water color)
        edgecolor="navy",
        linewidth=0.5,
        alpha=0.7,
        label="Water: Allocated Power P[k]",
    )

    # Draw water level line
    ax_main.axhline(
        y=water_level,
        color="#00BFFF",  # Deep sky blue
        linestyle="--",
        linewidth=3,
        label=f"Water Level μ = {water_level:.4f}",
        zorder=10,
    )

    # Annotate some specific subcarriers
    # Find best and worst channels
    best_idx = np.argmax(channel_gains)
    worst_idx = np.argmin(channel_gains)

    # Annotate best channel
    ax_main.annotate(
        f"Best Channel\nP[{best_idx}]={waterfilling_power[best_idx]:.4f}",
        xy=(best_idx, floor[best_idx] + waterfilling_power[best_idx] / 2),
        xytext=(best_idx + 5, water_level + 0.005),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7),
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
    )

    # Annotate worst channel
    ax_main.annotate(
        f"Worst Channel\nP[{worst_idx}]={waterfilling_power[worst_idx]:.4f}",
        xy=(worst_idx, floor[worst_idx] + waterfilling_power[worst_idx] / 2),
        xytext=(worst_idx - 10, water_level + 0.005),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.7),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )

    ax_main.set_xlabel("Subcarrier Index", fontsize=12, fontweight="bold")
    ax_main.set_ylabel("Power Level", fontsize=12, fontweight="bold")
    ax_main.set_title(
        'Waterfilling Power Allocation - "Water Container" Analogy\n'
        "Floor represents channel quality (higher floor = worse channel)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax_main.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax_main.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    ax_main.set_xlim(-1, num_subcarriers)

    # Add text box with explanation
    textstr = (
        "Waterfilling Principle:\n"
        "• Water seeks constant level\n"
        "• Deep areas (good channels) hold more water\n"
        "• Shallow areas (bad channels) hold less water\n"
        f"• Total water volume = {np.sum(waterfilling_power):.4f}"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax_main.text(
        0.02,
        0.97,
        textstr,
        transform=ax_main.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    # Subplot 1: Channel Gains (inverted to show floor concept)
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.bar(
        subcarriers,
        1.0 / channel_gains,  # Inverted to show as "depth"
        width=0.8,
        color="#CD853F",  # Peru (earth color)
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
    )
    ax1.set_xlabel("Subcarrier Index", fontsize=10)
    ax1.set_ylabel("Inverse Channel Gain (1/|H[k]|²)", fontsize=10)
    ax1.set_title(
        "Channel Quality (Inverted)\nHigher = Worse Channel", fontsize=11, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Power Allocation Comparison
    ax2 = fig.add_subplot(gs[1, 1])
    width = 0.4
    ax2.bar(
        subcarriers - width / 2,
        uniform_power,
        width,
        label="Uniform",
        color="orange",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.7,
    )
    ax2.bar(
        subcarriers + width / 2,
        waterfilling_power,
        width,
        label="Waterfilling",
        color="green",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.7,
    )
    ax2.set_xlabel("Subcarrier Index", fontsize=10)
    ax2.set_ylabel("Allocated Power", fontsize=10)
    ax2.set_title("Power Allocation Comparison", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Channel Gains (normal view)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.bar(
        subcarriers,
        channel_gains,
        width=0.8,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
    )
    ax3.set_xlabel("Subcarrier Index", fontsize=10)
    ax3.set_ylabel("Channel Power Gain |H[k]|²", fontsize=10)
    ax3.set_title(
        "Channel Frequency Response\nHigher = Better Channel", fontsize=11, fontweight="bold"
    )
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Power vs Channel Gain Scatter
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.scatter(
        channel_gains,
        waterfilling_power,
        s=100,
        c=subcarriers,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.7,
    )
    # Fit and plot trend line
    z = np.polyfit(channel_gains, waterfilling_power, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(np.min(channel_gains), np.max(channel_gains), 100)
    ax4.plot(x_trend, p(x_trend), "r--", linewidth=2, label=f"Trend: P ∝ |H|²")

    ax4.set_xlabel("Channel Gain |H[k]|²", fontsize=10)
    ax4.set_ylabel("Allocated Power P[k]", fontsize=10)
    ax4.set_title(
        "Power vs Channel Quality\nShows Positive Correlation", fontsize=11, fontweight="bold"
    )
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Add colorbar for scatter plot
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=num_subcarriers - 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax4)
    cbar.set_label("Subcarrier Index", fontsize=9)

    # Overall title
    fig.suptitle(
        f"Waterfilling Power Allocation Visualization\n"
        f"{num_subcarriers} Subcarriers | Total Power: {np.sum(waterfilling_power):.4f} | SNR: {-10*np.log10(noise_power):.1f} dB",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Save figure
    plt.savefig("waterfilling_diagram.png", dpi=300, bbox_inches="tight")
    print("\n✓ High-resolution diagram saved to 'waterfilling_diagram.png'")

    # Also create a simplified version for presentations
    create_simple_diagram(subcarriers, floor, waterfilling_power, water_level, channel_gains)


def create_simple_diagram(subcarriers, floor, waterfilling_power, water_level, channel_gains):
    """Create a simplified, clean diagram suitable for presentations."""

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot floor
    ax.bar(
        subcarriers,
        floor,
        width=0.9,
        color="#8B4513",
        edgecolor="black",
        linewidth=1,
        alpha=0.9,
        label="Channel Floor: N₀/|H[k]|²",
    )

    # Plot water
    ax.bar(
        subcarriers,
        waterfilling_power,
        bottom=floor,
        width=0.9,
        color="#4169E1",
        edgecolor="navy",
        linewidth=1,
        alpha=0.8,
        label="Allocated Power: P[k]",
    )

    # Water level line
    ax.axhline(
        y=water_level,
        color="cyan",
        linestyle="--",
        linewidth=3,
        label=f"Water Level: μ = {water_level:.4f}",
    )

    ax.set_xlabel("Subcarrier Index", fontsize=14, fontweight="bold")
    ax.set_ylabel("Power Level", fontsize=14, fontweight="bold")
    ax.set_title(
        'Waterfilling Power Allocation\n"Pouring Water" into Frequency-Selective Channel',
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="upper right", fontsize=12, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=1)

    # Annotate extremes
    best_idx = np.argmax(channel_gains)
    worst_idx = np.argmin(channel_gains)

    ax.annotate(
        "Best Channel",
        xy=(best_idx, water_level),
        xytext=(best_idx, water_level + 0.01),
        fontsize=11,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
    )

    ax.annotate(
        "Worst Channel",
        xy=(worst_idx, water_level),
        xytext=(worst_idx, water_level + 0.01),
        fontsize=11,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )

    plt.tight_layout()
    plt.savefig("waterfilling_diagram_simple.png", dpi=300, bbox_inches="tight")
    print("✓ Simple diagram saved to 'waterfilling_diagram_simple.png'")
    plt.close()


if __name__ == "__main__":
    plot_waterfilling_diagram()
