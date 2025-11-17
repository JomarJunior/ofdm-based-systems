"""Simple demo showing how to plot constellation diagrams for adaptive modulation.

This script demonstrates how to:
1. Run an adaptive modulation simulation
2. Extract received symbols and constellation orders
3. Create constellation diagrams with the plot_adaptive_constellation_diagram utility
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
from ofdm_based_systems.utils.visualization import plot_adaptive_constellation_diagram


def main():
    """Run adaptive modulation and plot constellation diagram."""

    print("\n" + "=" * 70)
    print("ADAPTIVE MODULATION - CONSTELLATION DIAGRAM DEMO")
    print("=" * 70)

    # Configuration
    num_subcarriers = 64
    num_ofdm_symbols = 100
    snr_db = 20.0

    # Create adaptive modulation configuration
    settings = SimulationSettings(
        num_bands=num_subcarriers,
        signal_noise_ratios=[snr_db],
        channel_model_path="config/channel_models/Lin-Phoong_P1.npy",
        channel_type=ChannelType.CUSTOM,
        noise_type=NoiseType.AWGN,
        num_bits=None,
        num_symbols=num_ofdm_symbols,
        constellation_order=16,
        constellation_type=ConstellationType.QAM,
        prefix_type=PrefixType.CYCLIC,
        prefix_length_ratio=0.25,
        modulation_type=ModulationType.OFDM,
        equalization_method=EqualizationMethod.MMSE,
        power_allocation_type=PowerAllocationType.WATERFILLING,
        # Adaptive modulation settings
        adaptive_modulation_mode=AdaptiveModulationMode.CAPACITY_BASED,
        min_constellation_order=4,
        max_constellation_order=256,
        capacity_scaling_factor=0.85,
    )

    print("\nRunning simulation...")
    simulations = Simulation.create_from_simulation_settings(settings)
    results = simulations[0].run()

    print("\n" + "=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)
    print(f"BER: {results['bit_error_rate']:.6f}")
    print(f"Bitrate: {results['bitrate_mbps']:.2f} Mbps")
    print(f"Total Bits: {results['total_bits']}")

    # Extract data for plotting
    received_symbols = results["received_symbols"]
    constellation_orders = np.array(results["constellation_order_per_subcarrier"], dtype=np.int64)

    # Print constellation order statistics
    active_orders = constellation_orders[constellation_orders > 0]
    unique_orders = np.unique(active_orders)
    print(f"\nConstellation Orders Used: {unique_orders.tolist()}")
    print(f"Active Subcarriers: {len(active_orders)}/{num_subcarriers}")
    print(f"Average Order: {np.mean(active_orders):.1f}")

    # Calculate PAPR for the plot
    signal_power = np.abs(received_symbols) ** 2
    peak_power = np.max(signal_power)
    avg_power = np.mean(signal_power)
    papr_db = 10 * np.log10(peak_power / avg_power) if avg_power > 0 else 0

    print("\n" + "=" * 70)
    print("GENERATING CONSTELLATION DIAGRAM")
    print("=" * 70)

    # Create constellation diagram
    fig = plot_adaptive_constellation_diagram(
        received_symbols=received_symbols,
        constellation_orders=constellation_orders,
        num_subcarriers=num_subcarriers,
        ber=results["bit_error_rate"],
        snr_db=snr_db,
        papr_db=papr_db,
        figsize=(14, 6),
        title_prefix="Adaptive Modulation Example",
    )

    # Save the plot
    filename = "adaptive_constellation_example.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\n✓ Constellation diagram saved as: {filename}")

    # Display the plot
    plt.show()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nThe constellation diagram shows:")
    print("  • Left plot: Received symbols color-coded by constellation order")
    print("    - Each color represents a different modulation order (4-QAM, 16-QAM, etc.)")
    print("    - 'X' markers show ideal constellation points for each order")
    print("    - Scattered points show actual received symbols")
    print("  • Right plot: Distribution of constellation orders across subcarriers")
    print("\nKey insights:")
    print("  • Subcarriers with better channel quality use higher-order modulations")
    print("  • Symbol clustering around ideal points indicates good channel quality")
    print("  • Spread/noise shows channel impairments and noise effects")


if __name__ == "__main__":
    main()
