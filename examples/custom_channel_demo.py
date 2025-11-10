"""Example: Using Custom Channel Models from Files.

This example demonstrates how to load and use custom channel impulse responses
from numpy files in OFDM simulations.
"""

from pathlib import Path

import numpy as np

from ofdm_based_systems.configuration.enums import (
    ConstellationType,
    EqualizationMethod,
    ModulationType,
    NoiseType,
    PowerAllocationType,
    PrefixType,
)
from ofdm_based_systems.configuration.models import SimulationSettings
from ofdm_based_systems.simulation.models import Simulation


def demonstrate_custom_channels():
    """Demonstrate loading and using custom channel models."""

    print("=" * 80)
    print("OFDM Simulation: Custom Channel Models Demo")
    print("=" * 80)

    # List available channel models
    channel_dir = Path("config/channel_models")
    channel_files = sorted(channel_dir.glob("*.npy"))

    print(f"\nAvailable channel models in {channel_dir}:")
    for i, filepath in enumerate(channel_files, 1):
        # Load and inspect
        channel = np.load(filepath)
        print(f"  {i}. {filepath.name}")
        print(f"     - Taps: {len(channel)}")
        print(f"     - Energy: {np.sum(np.abs(channel)**2):.6f}")

    print("\n" + "─" * 80)
    print("Testing with different channel models...")
    print("─" * 80)

    # Test with a few different channels
    test_channels = [
        "flat_fading.npy",
        "two_ray.npy",
        "default_multipath.npy",
        "severe_multipath.npy",
    ]

    results = []

    for channel_file in test_channels:
        channel_path = channel_dir / channel_file
        if not channel_path.exists():
            continue

        print(f"\n{'=' * 80}")
        print(f"Testing: {channel_file}")
        print("=" * 80)

        # Load channel
        channel_impulse = np.load(channel_path)

        # Create simulation
        sim = Simulation(
            num_symbols=10240,  # 10240 symbols = 64 * 160
            num_subcarriers=64,
            constellation_order=16,
            constellation_scheme=ConstellationType.QAM,
            modulator_type=ModulationType.OFDM,
            prefix_scheme=PrefixType.CYCLIC,
            prefix_length_ratio=1.0,
            equalizator_type=EqualizationMethod.MMSE,
            snr_db=20.0,
            noise_scheme=NoiseType.AWGN,
            power_allocation_type=PowerAllocationType.UNIFORM,
            channel_impulse_response=channel_impulse,
            verbose=False,
        )

        # Run simulation
        result = sim.run()

        print(f"\nResults:")
        print(f"  Channel: {channel_file}")
        print(f"  Channel taps: {len(channel_impulse)}")
        print(f"  BER: {result['bit_error_rate']:.6e}")
        print(f"  PAPR: {result['papr_db']:.2f} dB")
        print(f"  Bit Errors: {result['bit_errors']}/{result['total_bits']}")

        results.append(
            {
                "channel": channel_file,
                "taps": len(channel_impulse),
                "ber": result["bit_error_rate"],
                "papr": result["papr_db"],
            }
        )

    # Summary comparison
    print("\n" + "=" * 80)
    print("Summary: BER Comparison Across Channel Models (SNR=20dB)")
    print("=" * 80)
    print(f"{'Channel':<30} {'Taps':<8} {'BER':<15} {'PAPR (dB)':<10}")
    print("─" * 80)

    for r in results:
        print(f"{r['channel']:<30} {r['taps']:<8} {r['ber']:<15.6e} {r['papr']:<10.2f}")

    print("=" * 80)


def demonstrate_config_file_loading():
    """Demonstrate loading channel from configuration file."""

    print("\n\n" + "=" * 80)
    print("Loading Channel from Configuration File")
    print("=" * 80)

    # Load configuration with custom channel
    config_path = "config/simulation_settings_custom_channel.json"

    print(f"\nLoading configuration from: {config_path}")

    try:
        settings = SimulationSettings.from_json(config_path)

        print(f"\nConfiguration loaded:")
        print(f"  Channel Type: {settings.channel_type}")
        print(f"  Channel Path: {settings.channel_model_path}")
        print(f"  Power Allocation: {settings.power_allocation_type}")
        print(f"  Modulation: {settings.modulation_type}")

        # Create simulations (will automatically load channel)
        print(f"\nCreating simulations...")
        simulations = Simulation.create_from_simulation_settings(settings)

        print(f"  ✓ Created {len(simulations)} simulation(s)")
        print(f"  ✓ Channel automatically loaded from file")

        # Run one simulation as example
        print(f"\nRunning simulation with SNR = {simulations[0].snr_db} dB...")
        result = simulations[0].run()

        print(f"\nResults:")
        print(f"  BER: {result['bit_error_rate']:.6e}")
        print(f"  PAPR: {result['papr_db']:.2f} dB")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    print("=" * 80)


if __name__ == "__main__":
    # Demonstrate programmatic channel loading
    demonstrate_custom_channels()

    # Demonstrate config file loading
    demonstrate_config_file_loading()
    demonstrate_config_file_loading()
