"""Demonstration of Adaptive Modulation with Capacity-Based Constellation Selection.

This script demonstrates the adaptive modulation feature where constellation
orders are selected per-subcarrier based on channel capacity. It compares
FIXED mode (traditional single-order modulation) with CAPACITY_BASED mode
(adaptive per-subcarrier constellation selection).
"""

import matplotlib.pyplot as plt
import numpy as np

from ofdm_based_systems.configuration.enums import (
    AdaptiveModulationMode,
    ChannelType,
    ConstellationType,
    EqualizationMethod,
    ModulationType,
    NoiseType,
    PowerAllocationType,
    PrefixType,
)
from ofdm_based_systems.configuration.models import SimulationSettings
from ofdm_based_systems.simulation.models import Simulation
from ofdm_based_systems.utils.visualization import (
    plot_adaptive_constellation_diagram,
    plot_constellation_order_distribution,
    plot_water_level_diagram,
)


def run_fixed_modulation(
    num_subcarriers: int, num_ofdm_symbols: int, snr_db: float, channel_path: str
):
    """Run simulation with fixed constellation order (traditional approach)."""
    print("\n" + "=" * 70)
    print("FIXED MODULATION MODE (Traditional)")
    print("=" * 70)

    # For FIXED mode: Calculate total constellation symbols needed
    # Each OFDM symbol requires num_subcarriers constellation symbols
    constellation_order = 64
    total_constellation_symbols = num_subcarriers * num_ofdm_symbols

    settings = SimulationSettings(
        num_bands=num_subcarriers,
        signal_noise_ratios=[snr_db],
        channel_model_path=channel_path,
        channel_type=ChannelType.CUSTOM,
        noise_type=NoiseType.AWGN,
        num_bits=None,
        num_symbols=total_constellation_symbols,
        constellation_order=constellation_order,
        constellation_type=ConstellationType.QAM,
        prefix_type=PrefixType.CYCLIC,
        prefix_length_ratio=1.0,
        equalization_method=EqualizationMethod.MMSE,
        modulation_type=ModulationType.OFDM,
        power_allocation_type=PowerAllocationType.UNIFORM,
        adaptive_modulation_mode=AdaptiveModulationMode.FIXED,
        min_constellation_order=4,
        max_constellation_order=2048,
        capacity_scaling_factor=1.0,
    )

    simulations = Simulation.create_from_simulation_settings(settings)
    # Run the first simulation (single SNR value)
    results = simulations[0].run()

    print(f"\nFixed Mode Results:")
    print(f"  - Constellation Order: 16-QAM (all subcarriers)")
    print(f"  - Total Bits: {results['total_bits']}")
    print(f"  - Bit Error Rate: {results['bit_error_rate']:.6f}")
    print(f"  - Transmission Time: {results['transmission_time_ms']:.2f} ms")
    print(f"  - Bitrate: {results['bitrate_mbps']:.2f} Mbps")

    return results


def run_adaptive_modulation(
    num_subcarriers: int,
    num_ofdm_symbols: int,
    snr_db: float,
    channel_path: str,
    min_order: int = 4,
    max_order: int = 2048,
    scaling_factor: float = 1.0,
):
    """Run simulation with adaptive constellation order selection."""
    print("\n" + "=" * 70)
    print("ADAPTIVE MODULATION MODE (Capacity-Based)")
    print("=" * 70)

    settings = SimulationSettings(
        num_bands=num_subcarriers,
        signal_noise_ratios=[snr_db],
        channel_model_path=channel_path,
        channel_type=ChannelType.CUSTOM,
        noise_type=NoiseType.AWGN,
        num_bits=None,
        num_symbols=num_ofdm_symbols,
        constellation_order=16,  # Default, will be overridden per subcarrier
        constellation_type=ConstellationType.QAM,
        prefix_type=PrefixType.CYCLIC,
        prefix_length_ratio=1.0,
        equalization_method=EqualizationMethod.MMSE,
        modulation_type=ModulationType.OFDM,
        power_allocation_type=PowerAllocationType.WATERFILLING,
        adaptive_modulation_mode=AdaptiveModulationMode.CAPACITY_BASED,
        min_constellation_order=min_order,
        max_constellation_order=max_order,
        capacity_scaling_factor=scaling_factor,
    )

    simulations = Simulation.create_from_simulation_settings(settings)
    # Run the first simulation (single SNR value)
    results = simulations[0].run()

    # Analyze constellation order distribution
    orders = np.array(results["constellation_order_per_subcarrier"], dtype=np.int64)
    active_orders = orders[orders > 0]
    unique_orders = np.unique(active_orders)

    print(f"\nAdaptive Mode Results:")
    print(f"  - Min Order: {min_order}")
    print(f"  - Max Order: {max_order}")
    print(f"  - Capacity Scaling Factor: {scaling_factor}")
    print(f"  - Active Subcarriers: {len(active_orders)}/{num_subcarriers}")
    print(f"  - Unique Orders Used: {unique_orders}")
    print(f"  - Average Order: {np.mean(active_orders):.1f}")
    print(f"  - Total Bits: {results['total_bits']}")
    print(f"  - Bit Error Rate: {results['bit_error_rate']:.6f}")
    print(f"  - Transmission Time: {results['transmission_time_ms']:.2f} ms")
    print(f"  - Bitrate: {results['bitrate_mbps']:.2f} Mbps")
    if results.get("water_level"):
        print(f"  - Water Level (μ): {results['water_level']:.6f}")

    return results


def visualize_constellation_diagram(results, num_subcarriers: int, filename: str):
    """Create constellation diagram for a simulation result."""
    print(f"\nGenerating constellation diagram: {filename}")

    # Check if received_symbols are available
    if "received_symbols" not in results:
        print("  Warning: No received symbols available for plotting")
        return

    received_symbols = results["received_symbols"]
    orders = np.array(results.get("constellation_order_per_subcarrier", []), dtype=np.int64)

    # If no orders available (fixed mode), create uniform array
    if len(orders) == 0:
        constellation_order = results.get("constellation_order", 16)
        orders = np.full(num_subcarriers, constellation_order, dtype=np.int64)

    # Calculate PAPR
    signal_power = np.abs(received_symbols) ** 2
    peak_power = np.max(signal_power)
    avg_power = np.mean(signal_power)
    papr_db = 10 * np.log10(peak_power / avg_power) if avg_power > 0 else 0

    # Create plot
    fig = plot_adaptive_constellation_diagram(
        received_symbols=received_symbols,
        constellation_orders=orders,
        num_subcarriers=num_subcarriers,
        ber=results["bit_error_rate"],
        snr_db=results.get("snr_db", 20.0),
        papr_db=papr_db,
        figsize=(14, 6),
        title_prefix=results.get("title", "OFDM Simulation"),
    )

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {filename}")


def visualize_comparison(fixed_results, adaptive_results, num_subcarriers: int):
    """Create comparison visualizations."""
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Extract data
    orders = np.array(adaptive_results["constellation_order_per_subcarrier"], dtype=np.int64)
    power_allocation = np.array(adaptive_results["allocated_power"])

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))

    # Subplot 1: Constellation Order Distribution
    ax1 = plt.subplot(2, 2, 1)
    active_orders = orders[orders > 0]
    unique_orders, counts = np.unique(active_orders, return_counts=True)

    from matplotlib import cm

    colors = cm.viridis(np.linspace(0, 1, len(unique_orders)))  # type: ignore
    bars = ax1.bar(
        range(len(unique_orders)), counts, color=colors, edgecolor="black", linewidth=1.5
    )

    ax1.set_xlabel("Constellation Order (M-QAM)", fontsize=12)
    ax1.set_ylabel("Number of Subcarriers", fontsize=12)
    ax1.set_title("Adaptive Modulation - Order Distribution", fontsize=14, fontweight="bold")
    ax1.set_xticks(range(len(unique_orders)))
    ax1.set_xticklabels([f"{int(order)}" for order in unique_orders])
    ax1.grid(True, axis="y", alpha=0.3, linestyle="--")

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Subplot 2: Power Allocation per Subcarrier
    ax2 = plt.subplot(2, 2, 2)
    subcarrier_indices = np.arange(num_subcarriers)
    ax2.bar(subcarrier_indices, power_allocation, color="steelblue", alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Subcarrier Index", fontsize=12)
    ax2.set_ylabel("Allocated Power", fontsize=12)
    ax2.set_title("Waterfilling Power Allocation", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--")

    if adaptive_results.get("water_level"):
        ax2.axhline(
            adaptive_results["water_level"],
            color="darkblue",
            linestyle="--",
            linewidth=2,
            label=f'Water Level (μ={adaptive_results["water_level"]:.4f})',
        )
        ax2.legend()

    # Subplot 3: Constellation Orders per Subcarrier
    ax3 = plt.subplot(2, 2, 3)
    # Color-code by order
    max_order_value = np.max(orders) if np.max(orders) > 0 else 1
    normalized_orders = orders / max_order_value
    scatter = ax3.scatter(
        subcarrier_indices,
        orders,
        c=normalized_orders,
        cmap="viridis",
        s=100,
        edgecolor="black",
        linewidth=1,
    )
    ax3.set_xlabel("Subcarrier Index", fontsize=12)
    ax3.set_ylabel("Constellation Order (M)", fontsize=12)
    ax3.set_title("Per-Subcarrier Constellation Orders", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3, linestyle="--")
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label("Normalized Order", fontsize=10)

    # Subplot 4: Performance Comparison
    ax4 = plt.subplot(2, 2, 4)
    modes = ["Fixed\n(16-QAM)", "Adaptive\n(Capacity-Based)"]
    bitrates = [fixed_results["bitrate_mbps"], adaptive_results["bitrate_mbps"]]
    bers = [fixed_results["bit_error_rate"], adaptive_results["bit_error_rate"]]

    x = np.arange(len(modes))
    width = 0.35

    ax4_twin = ax4.twinx()

    bars1 = ax4.bar(
        x - width / 2, bitrates, width, label="Bitrate (Mbps)", color="skyblue", edgecolor="black"
    )
    bars2 = ax4_twin.bar(x + width / 2, bers, width, label="BER", color="salmon", edgecolor="black")

    ax4.set_xlabel("Modulation Mode", fontsize=12)
    ax4.set_ylabel("Bitrate (Mbps)", fontsize=12, color="skyblue")
    ax4_twin.set_ylabel("Bit Error Rate", fontsize=12, color="salmon")
    ax4.set_title("Performance Comparison", fontsize=14, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(modes)
    ax4.tick_params(axis="y", labelcolor="skyblue")
    ax4_twin.tick_params(axis="y", labelcolor="salmon")

    # Add value labels on bars
    for bar, val in zip(bars1, bitrates):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    for bar, val in zip(bars2, bers):
        height = bar.get_height()
        ax4_twin.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.6f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig("adaptive_modulation_comparison.png", dpi=150, bbox_inches="tight")
    print("\n✓ Visualization saved as 'adaptive_modulation_comparison.png'")

    # plt.show()


def main():
    """Main demonstration function."""
    print("\n" + "=" * 70)
    print("ADAPTIVE MODULATION DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo compares traditional fixed modulation with adaptive")
    print("capacity-based modulation over a frequency-selective channel.")

    # Configuration
    num_subcarriers = 64
    num_ofdm_symbols = (
        10000  # Number of OFDM symbols (must result in total symbols divisible by num_subcarriers)
    )
    snr_db = 20.0
    channel_path = "config/channel_models/Lin-Phoong_P2.npy"

    print(f"\nConfiguration:")
    print(f"  - Number of Subcarriers: {num_subcarriers}")
    print(f"  - Number of OFDM Symbols: {num_ofdm_symbols}")
    print(f"  - SNR: {snr_db} dB")
    print(f"  - Channel: Lin-Phoong P2")
    print(f"  - Power Allocation: Waterfilling")
    print(f"  - Equalization: MMSE")

    # Run fixed modulation
    fixed_results = run_fixed_modulation(num_subcarriers, num_ofdm_symbols, snr_db, channel_path)

    # Run adaptive modulation with different configurations
    print("\n" + "=" * 70)
    print("TRYING DIFFERENT ADAPTIVE CONFIGURATIONS")
    print("=" * 70)

    # Conservative (scaling_factor = 0.7)
    print("\n--- Configuration 1: Conservative (scaling_factor = 0.7) ---")
    adaptive_conservative = run_adaptive_modulation(
        num_subcarriers,
        num_ofdm_symbols,
        snr_db,
        channel_path,
        min_order=4,
        max_order=2048,
        scaling_factor=2.5,
    )

    # Balanced (scaling_factor = 0.85)
    print("\n--- Configuration 2: Balanced (scaling_factor = 0.85) ---")
    adaptive_balanced = run_adaptive_modulation(
        num_subcarriers,
        num_ofdm_symbols,
        snr_db,
        channel_path,
        min_order=4,
        max_order=2048,
        scaling_factor=2**2,
    )

    # Aggressive (scaling_factor = 1.0)
    print("\n--- Configuration 3: Aggressive (scaling_factor = 1.0) ---")
    adaptive_aggressive = run_adaptive_modulation(
        num_subcarriers,
        num_ofdm_symbols,
        snr_db,
        channel_path,
        min_order=4,
        max_order=2048,
        scaling_factor=2**4,
    )

    # Generate constellation diagrams for each configuration
    print("\n" + "=" * 70)
    print("GENERATING CONSTELLATION DIAGRAMS")
    print("=" * 70)

    visualize_constellation_diagram(fixed_results, num_subcarriers, "constellation_fixed.png")
    visualize_constellation_diagram(
        adaptive_conservative, num_subcarriers, "constellation_adaptive_conservative.png"
    )
    visualize_constellation_diagram(
        adaptive_balanced, num_subcarriers, "constellation_adaptive_balanced.png"
    )
    visualize_constellation_diagram(
        adaptive_aggressive, num_subcarriers, "constellation_adaptive_aggressive.png"
    )

    # Visualize the balanced configuration comparison
    visualize_comparison(fixed_results, adaptive_balanced, num_subcarriers)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nBitrate Comparison:")
    print(f"  Fixed (16-QAM):             {fixed_results['bitrate_mbps']:.2f} Mbps")
    print(f"  Adaptive (Conservative):    {adaptive_conservative['bitrate_mbps']:.2f} Mbps")
    print(f"  Adaptive (Balanced):        {adaptive_balanced['bitrate_mbps']:.2f} Mbps")
    print(f"  Adaptive (Aggressive):      {adaptive_aggressive['bitrate_mbps']:.2f} Mbps")

    print("\nBER Comparison:")
    print(f"  Fixed (16-QAM):             {fixed_results['bit_error_rate']:.6f}")
    print(f"  Adaptive (Conservative):    {adaptive_conservative['bit_error_rate']:.6f}")
    print(f"  Adaptive (Balanced):        {adaptive_balanced['bit_error_rate']:.6f}")
    print(f"  Adaptive (Aggressive):      {adaptive_aggressive['bit_error_rate']:.6f}")

    print("\nTransmission Time Comparison:")
    print(f"  Fixed (16-QAM):             {fixed_results['transmission_time_ms']:.2f} ms")
    print(f"  Adaptive (Conservative):    {adaptive_conservative['transmission_time_ms']:.2f} ms")
    print(f"  Adaptive (Balanced):        {adaptive_balanced['transmission_time_ms']:.2f} ms")
    print(f"  Adaptive (Aggressive):      {adaptive_aggressive['transmission_time_ms']:.2f} ms")

    print("\nTotal Bits Comparison:")
    print(f"  Fixed (16-QAM):             {fixed_results['total_bits']}")
    print(f"  Adaptive (Conservative):    {adaptive_conservative['total_bits']}")
    print(f"  Adaptive (Balanced):        {adaptive_balanced['total_bits']}")
    print(f"  Adaptive (Aggressive):      {adaptive_aggressive['total_bits']}")

    print("\nTotal sent symbols comparison:")
    print(f"  Fixed (16-QAM):             {len(fixed_results['received_symbols'])}")
    print(f"  Adaptive (Conservative):    {len(adaptive_conservative['received_symbols'])}")
    print(f"  Adaptive (Balanced):        {len(adaptive_balanced['received_symbols'])}")
    print(f"  Adaptive (Aggressive):      {len(adaptive_aggressive['received_symbols'])}")

    print("\nKey Observations:")
    print("  • Adaptive modulation adjusts constellation orders based on channel quality")
    print("  • Higher scaling factors use higher-order modulations (higher bitrate, more errors)")
    print("  • Lower scaling factors are more conservative (lower bitrate, fewer errors)")
    print("  • Waterfilling allocates more power to better subcarriers")
    print("  • Combined with adaptive modulation, this optimizes spectral efficiency")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nGenerated Files:")
    print("  • constellation_fixed.png - Fixed modulation constellation diagram")
    print("  • constellation_adaptive_conservative.png - Conservative adaptive (0.7 scaling)")
    print("  • constellation_adaptive_balanced.png - Balanced adaptive (0.85 scaling)")
    print("  • constellation_adaptive_aggressive.png - Aggressive adaptive (1.0 scaling)")
    print("  • adaptive_modulation_comparison.png - Performance comparison")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
