"""Test to demonstrate the normalization fix improving BER performance."""

from pathlib import Path

from ofdm_based_systems.configuration.models import Settings, SimulationSettings
from ofdm_based_systems.main import ResultsManager, SimulationRunner


def main():
    """Run test simulation showing improved BER after normalization fix."""
    # Load configuration
    settings = Settings.from_json(file_path="config/settings.json")
    simulation_settings = SimulationSettings.from_json(
        file_path="config/simulation_settings_test.json"
    )

    # Extract channel name from configuration
    channel_name = "default"
    if (
        simulation_settings.channel_type.value == "CUSTOM"
        and simulation_settings.channel_model_path
    ):
        channel_name = Path(simulation_settings.channel_model_path).stem
    elif simulation_settings.channel_type.value == "FLAT":
        channel_name = "flat"

    print(f"\n{'='*80}")
    print(f"  Testing Normalization Fix")
    print(f"{'='*80}")
    print(f"  Channel: {channel_name}")
    print(
        f"  Configuration: {simulation_settings.constellation_order}-{simulation_settings.constellation_type.value}"
    )
    print(f"  Equalization: {simulation_settings.equalization_method.value}")
    print(f"  Power Allocation: {simulation_settings.power_allocation_type.value}")
    print(f"\n  Expected Behavior:")
    print(f"    - Symbols are normalized to unit average power before demapping")
    print(f"    - BER should be significantly improved (from ~25% to realistic values)")
    print(f"    - At high SNR (30dB), BER should approach 0")
    print(f"{'='*80}\n")

    # Initialize results manager with channel-specific directory
    results_manager = ResultsManager(
        results_dir="results",
        images_dir="images",
        channel_name=channel_name,
    )

    # Create and run simulation runner
    runner = SimulationRunner(settings, simulation_settings, results_manager)
    results = runner.run_all()

    # Process and save results
    runner.process_results(results)

    # Analyze results
    print(f"\n{'='*80}")
    print(f"  Results Analysis")
    print(f"{'='*80}")

    for i, result in enumerate(results, 1):
        snr = result.get("snr_db", 0)
        ber = result.get("bit_error_rate", 0)
        bit_errors = result.get("bit_errors", 0)
        total_bits = result.get("total_bits", 1)

        print(f"\n  Simulation {i}:")
        print(f"    SNR: {snr:.1f} dB")
        print(f"    BER: {ber:.6f} ({ber*100:.2f}%)")
        print(f"    Bit Errors: {bit_errors:,} / {total_bits:,}")

        # Provide interpretation
        if snr >= 25:
            if ber < 0.001:
                print(f"    ✓ EXCELLENT: BER < 0.1% at high SNR (normalization working!)")
            elif ber < 0.05:
                print(f"    ✓ GOOD: BER < 5% at high SNR")
            else:
                print(f"    ⚠ WARNING: BER still high at {snr}dB, check normalization")
        elif snr >= 15:
            if ber < 0.1:
                print(f"    ✓ GOOD: Reasonable BER at medium SNR")
            else:
                print(f"    ⚠ WARNING: BER high at {snr}dB")

    print(f"\n{'='*80}")
    print("\n✓ Normalization test completed!\n")


if __name__ == "__main__":
    main()
    main()
