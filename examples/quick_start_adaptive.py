"""Quick Start - Simple Adaptive Modulation Example

This is a minimal example showing how to run adaptive modulation.
For a comprehensive demo with visualizations, see adaptive_modulation_demo.py
"""

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


def run_simple_example():
    """Run a simple adaptive modulation example."""

    print("\n" + "=" * 70)
    print("SIMPLE ADAPTIVE MODULATION EXAMPLE")
    print("=" * 70)

    # Create configuration
    settings = SimulationSettings(
        # Basic parameters
        num_bands=64,  # Number of subcarriers
        signal_noise_ratios=[20.0],  # SNR in dB
        # Channel configuration
        channel_model_path="config/channel_models/Lin-Phoong_P1.npy",
        channel_type=ChannelType.CUSTOM,
        noise_type=NoiseType.AWGN,
        # Data configuration
        num_bits=None,  # Use num_symbols instead
        num_symbols=100,  # Number of OFDM symbols
        # Modulation configuration
        constellation_order=16,  # Base order (for reference)
        constellation_type=ConstellationType.QAM,
        # OFDM configuration
        prefix_type=PrefixType.CYCLIC,
        prefix_length_ratio=0.25,
        modulation_type=ModulationType.OFDM,
        equalization_method=EqualizationMethod.MMSE,
        # Power allocation
        power_allocation_type=PowerAllocationType.WATERFILLING,
        # ADAPTIVE MODULATION SETTINGS
        adaptive_modulation_mode=AdaptiveModulationMode.CAPACITY_BASED,  # Enable adaptive mode
        min_constellation_order=4,  # Minimum: 4-QAM
        max_constellation_order=256,  # Maximum: 256-QAM
        capacity_scaling_factor=0.85,  # Scaling factor (0.7=conservative, 1.0=aggressive)
    )

    # Create and run simulation
    print("\nCreating simulation...")
    simulations = Simulation.create_from_simulation_settings(settings)

    print("Running simulation...")
    results = simulations[0].run()  # Run first (and only) SNR value

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Mode: {results['adaptive_modulation_mode']}")
    print(f"Total Bits: {results['total_bits']}")
    print(f"Bit Error Rate: {results['bit_error_rate']:.6f}")
    print(f"Transmission Time: {results['transmission_time_ms']:.2f} ms")
    print(f"Bitrate: {results['bitrate_mbps']:.2f} Mbps")

    if "constellation_order_per_subcarrier" in results:
        import numpy as np

        orders = np.array(results["constellation_order_per_subcarrier"])
        active = orders[orders > 0]
        if len(active) > 0:
            print(f"\nAdaptive Modulation Details:")
            print(f"  Active Subcarriers: {len(active)}/{len(orders)}")
            print(f"  Orders Used: {sorted(set(active.tolist()))}")
            print(f"  Average Order: {active.mean():.1f}")
            if results.get("water_level"):
                print(f"  Water Level: {results['water_level']:.6f}")

    print("\n" + "=" * 70)
    print("✓ Simulation Complete!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    # Run the example
    results = run_simple_example()

    print("\nTo run with different configurations, modify the settings above:")
    print("  - Try adaptive_modulation_mode=AdaptiveModulationMode.FIXED for traditional mode")
    print("  - Adjust min/max_constellation_order for different order ranges")
    print("  - Change capacity_scaling_factor (0.5-1.5) to control aggressiveness")
    print("  - Use different SNR values: signal_noise_ratios=[10, 15, 20, 25, 30]")
    print("\nFor a comprehensive demo with visualizations:")
    print("  python examples/adaptive_modulation_demo.py")
