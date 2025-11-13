"""Demonstration of Waterfilling Power Allocation.

This script demonstrates the waterfilling power allocation algorithm
and compares it with uniform power allocation.
"""

import matplotlib.pyplot as plt
import numpy as np

from ofdm_based_systems.power_allocation.models import (
    UniformPowerAllocation,
    WaterfillingPowerAllocation,
    calculate_capacity,
    compare_allocations,
)


def demonstrate_waterfilling():
    """Demonstrate waterfilling algorithm with visualization."""

    # Setup: Frequency-selective channel
    print("=" * 70)
    print("Waterfilling Power Allocation Demonstration")
    print("=" * 70)

    # Channel parameters
    num_subcarriers = 64
    total_power = 1.0
    snr_db = 20.0
    noise_power = 10 ** (-snr_db / 10)

    # Create frequency-selective channel
    # Simulate multipath: h(t) = δ(t) + 0.7·δ(t-1) + 0.4·δ(t-2) + 0.2·δ(t-3)
    channel_impulse = np.array([1.0, 0.7, 0.4, 0.2], dtype=np.complex128)
    channel_response = np.fft.fft(channel_impulse, n=num_subcarriers)
    channel_gains = np.abs(channel_response) ** 2

    print(f"\nChannel Configuration:")
    print(f"  - Number of subcarriers: {num_subcarriers}")
    print(f"  - Total power budget: {total_power}")
    print(f"  - SNR: {snr_db} dB")
    print(f"  - Noise power: {noise_power:.6f}")
    print(f"  - Channel: Multipath (4 taps)")

    # Uniform allocation
    uniform_allocator = UniformPowerAllocation(
        total_power=total_power, num_subcarriers=num_subcarriers
    )
    uniform_power = uniform_allocator.allocate()

    # Waterfilling allocation
    waterfilling_allocator = WaterfillingPowerAllocation(
        total_power=total_power, channel_gains=channel_gains, noise_power=noise_power
    )
    waterfilling_power = waterfilling_allocator.allocate()

    # Calculate capacities
    uniform_capacity = calculate_capacity(uniform_power, channel_gains, noise_power)
    waterfilling_capacity = calculate_capacity(waterfilling_power, channel_gains, noise_power)

    # Compare
    comparison = compare_allocations(uniform_power, waterfilling_power, channel_gains, noise_power)

    print(f"\nPower Allocation Results:")
    print(f"\nUniform Allocation:")
    print(f"  - Power per subcarrier: {uniform_power[0]:.6f}")
    print(f"  - Channel capacity: {uniform_capacity:.4f} bits/channel use")

    print(f"\nWaterfilling Allocation:")
    print(f"  - Min power: {np.min(waterfilling_power):.6f}")
    print(f"  - Max power: {np.max(waterfilling_power):.6f}")
    print(f"  - Subcarriers with zero power: {np.sum(waterfilling_power < 1e-10)}")
    print(f"  - Channel capacity: {waterfilling_capacity:.4f} bits/channel use")

    print(f"\nCapacity Improvement:")
    print(f"  - Absolute gain: {comparison['capacity_gain']:.4f} bits/channel use")
    print(f"  - Percentage gain: {comparison['capacity_gain_percent']:.2f}%")

    print(f"\nWater Level Property:")
    # Calculate water level for allocated subcarriers
    floor = noise_power / channel_gains
    water_levels = waterfilling_power + floor
    allocated = waterfilling_power > 1e-10
    if np.any(allocated):
        water_level_mean = np.mean(water_levels[allocated])
        water_level_std = np.std(water_levels[allocated])
        print(f"  - Mean water level: {water_level_mean:.6f}")
        print(f"  - Std deviation: {water_level_std:.9f} (should be ~0)")

    # Print detailed allocation
    print(f"\nDetailed Power Allocation:")
    print(f"{'Subcarrier':<12} {'Ch. Gain':<12} {'Uniform':<12} {'Waterfilling':<15} {'Ratio':<10}")
    print("-" * 70)
    for i in range(num_subcarriers):
        ratio = waterfilling_power[i] / uniform_power[i] if uniform_power[i] > 0 else 0
        print(
            f"{i:<12} {channel_gains[i]:<12.4f} {uniform_power[i]:<12.6f} "
            f"{waterfilling_power[i]:<15.6f} {ratio:<10.2f}x"
        )

    print("\n" + "=" * 70)
    print("Visualization saved to 'waterfilling_demo.png'")
    print("=" * 70)

    # Create visualization
    create_visualization(channel_gains, uniform_power, waterfilling_power, floor, num_subcarriers)


def create_visualization(channel_gains, uniform_power, waterfilling_power, floor, num_subcarriers):
    """Create visualization of waterfilling algorithm."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Waterfilling Power Allocation Demonstration", fontsize=16, fontweight="bold")

    subcarriers = np.arange(num_subcarriers)

    # 1. Channel Gains
    ax = axes[0, 0]
    ax.bar(subcarriers, channel_gains, alpha=0.7, color="steelblue", edgecolor="black")
    ax.set_xlabel("Subcarrier Index")
    ax.set_ylabel("Channel Power Gain |H[k]|²")
    ax.set_title("Channel Frequency Response")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(subcarriers)

    # 2. Power Allocation Comparison
    ax = axes[0, 1]
    width = 0.35
    ax.bar(
        subcarriers - width / 2,
        uniform_power,
        width,
        label="Uniform",
        alpha=0.7,
        color="orange",
        edgecolor="black",
    )
    ax.bar(
        subcarriers + width / 2,
        waterfilling_power,
        width,
        label="Waterfilling",
        alpha=0.7,
        color="green",
        edgecolor="black",
    )
    ax.set_xlabel("Subcarrier Index")
    ax.set_ylabel("Allocated Power")
    ax.set_title("Power Allocation Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(subcarriers)

    # 3. Waterfilling Visualization (Water Container)
    ax = axes[1, 0]
    # Plot floor (inverse SNR)
    ax.bar(
        subcarriers, floor, alpha=0.5, color="brown", edgecolor="black", label="Floor (N₀/|H[k]|²)"
    )
    # Plot water (power allocation) on top of floor
    ax.bar(
        subcarriers,
        waterfilling_power,
        bottom=floor,
        alpha=0.7,
        color="cyan",
        edgecolor="black",
        label="Water (Power)",
    )
    # Draw water level line
    water_level = waterfilling_power + floor
    allocated = waterfilling_power > 1e-10
    if np.any(allocated):
        mean_water_level = np.mean(water_level[allocated])
        ax.axhline(
            y=mean_water_level,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Water Level (μ={mean_water_level:.3f})",
        )
    ax.set_xlabel("Subcarrier Index")
    ax.set_ylabel("Level")
    ax.set_title("Waterfilling Container Analogy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(subcarriers)

    # 4. SNR per Subcarrier
    ax = axes[1, 1]
    uniform_snr = uniform_power * channel_gains / floor[0] * channel_gains
    waterfilling_snr = waterfilling_power * channel_gains / floor[0] * channel_gains
    ax.bar(
        subcarriers - width / 2,
        10 * np.log10(uniform_snr + 1e-10),
        width,
        label="Uniform",
        alpha=0.7,
        color="orange",
        edgecolor="black",
    )
    ax.bar(
        subcarriers + width / 2,
        10 * np.log10(waterfilling_snr + 1e-10),
        width,
        label="Waterfilling",
        alpha=0.7,
        color="green",
        edgecolor="black",
    )
    ax.set_xlabel("Subcarrier Index")
    ax.set_ylabel("SNR (dB)")
    ax.set_title("SNR per Subcarrier")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(subcarriers)

    plt.tight_layout()
    plt.savefig("waterfilling_demo.png", dpi=150, bbox_inches="tight")
    print("\nVisualization saved to 'waterfilling_demo.png'")
    plt.close()


if __name__ == "__main__":
    demonstrate_waterfilling()
