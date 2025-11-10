"""Example: Running OFDM Simulation with Different Configurations.

This example demonstrates the improved, configurable simulation structure
comparing Uniform vs Waterfilling power allocation strategies.
"""

from ofdm_based_systems.configuration.enums import (
    ConstellationType,
    EqualizationMethod,
    ModulationType,
    NoiseType,
    PowerAllocationType,
    PrefixType,
)
from ofdm_based_systems.simulation.models import Simulation


def run_comparison_example():
    """Compare Uniform vs Waterfilling power allocation."""

    print("=" * 80)
    print("OFDM Simulation: Uniform vs Waterfilling Power Allocation Comparison")
    print("=" * 80)

    # Common configuration
    config = {
        "num_symbols": 10240,  # Must be divisible by num_subcarriers (64 * 160 = 10240)
        "num_subcarriers": 64,
        "constellation_order": 16,
        "constellation_scheme": ConstellationType.QAM,
        "modulator_type": ModulationType.OFDM,
        "prefix_scheme": PrefixType.CYCLIC,
        "prefix_length_ratio": 1.0,
        "equalizator_type": EqualizationMethod.MMSE,
        "snr_db": 20.0,
        "noise_scheme": NoiseType.AWGN,
        "verbose": False,  # Disable verbose logging for cleaner output
    }

    # Run with Uniform power allocation
    print("\n" + "─" * 80)
    print("Running simulation with UNIFORM power allocation...")
    print("─" * 80)

    sim_uniform = Simulation(**config, power_allocation_type=PowerAllocationType.UNIFORM)
    result_uniform = sim_uniform.run()

    print(f"\nResults (Uniform):")
    print(f"  BER: {result_uniform['bit_error_rate']:.6e}")
    print(f"  PAPR: {result_uniform['papr_db']:.2f} dB")
    print(f"  Bit Errors: {result_uniform['bit_errors']}/{result_uniform['total_bits']}")

    # Run with Waterfilling power allocation
    print("\n" + "─" * 80)
    print("Running simulation with WATERFILLING power allocation...")
    print("─" * 80)

    sim_waterfilling = Simulation(**config, power_allocation_type=PowerAllocationType.WATERFILLING)
    result_waterfilling = sim_waterfilling.run()

    print(f"\nResults (Waterfilling):")
    print(f"  BER: {result_waterfilling['bit_error_rate']:.6e}")
    print(f"  PAPR: {result_waterfilling['papr_db']:.2f} dB")
    print(f"  Bit Errors: {result_waterfilling['bit_errors']}/{result_waterfilling['total_bits']}")
    if "channel_capacity" in result_waterfilling:
        print(f"  Channel Capacity: {result_waterfilling['channel_capacity']:.4f} bits/channel use")

    # Comparison
    print("\n" + "=" * 80)
    print("Comparison Summary")
    print("=" * 80)

    ber_improvement = (
        (result_uniform["bit_error_rate"] - result_waterfilling["bit_error_rate"])
        / result_uniform["bit_error_rate"]
        * 100
    )

    print(f"  BER Improvement: {ber_improvement:+.2f}%")
    print(f"  Uniform BER:      {result_uniform['bit_error_rate']:.6e}")
    print(f"  Waterfilling BER: {result_waterfilling['bit_error_rate']:.6e}")

    if ber_improvement > 0:
        print(f"\n  ✓ Waterfilling provides better performance!")
    else:
        print(f"\n  ✗ Uniform performs better (channel may be flat)")

    print("=" * 80 + "\n")


def run_programmatic_example():
    """Example of programmatically creating and running simulations."""

    print("=" * 80)
    print("OFDM Simulation: Programmatic Configuration Example")
    print("=" * 80)

    # Create a simulation with specific parameters
    sim = Simulation(
        num_bits=80000,  # 80,000 bits
        num_subcarriers=128,  # More subcarriers
        constellation_order=64,  # 64-QAM
        constellation_scheme=ConstellationType.QAM,
        modulator_type=ModulationType.SC_OFDM,  # Single-Carrier OFDM
        prefix_scheme=PrefixType.CYCLIC,
        prefix_length_ratio=1.0,
        equalizator_type=EqualizationMethod.ZF,  # Zero-Forcing
        snr_db=25.0,
        noise_scheme=NoiseType.AWGN,
        power_allocation_type=PowerAllocationType.WATERFILLING,
        verbose=True,  # Enable detailed logging
    )

    print("\nRunning SC-OFDM simulation with 64-QAM and Waterfilling...\n")
    result = sim.run()

    print("\n" + "=" * 80)
    print("Simulation Complete")
    print("=" * 80)
    print(
        f"Configuration: {result['constellation_order']}-{result['constellation_scheme']}, "
        f"{result['modulator_type']}, {result['equalizator_type']}"
    )
    print(f"Power Allocation: {result['power_allocation_type']}")
    print(f"SNR: {result['snr_db']} dB")
    print(f"BER: {result['bit_error_rate']:.6e}")
    print(f"PAPR: {result['papr_db']:.2f} dB")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Run comparison example
    run_comparison_example()

    # Run programmatic example
    # Uncomment to run:
    # run_programmatic_example()
