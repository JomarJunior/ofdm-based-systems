"""Test script to demonstrate new filename pattern with different configurations."""

from pathlib import Path

from ofdm_based_systems.configuration.models import Settings, SimulationSettings
from ofdm_based_systems.main import ResultsManager, SimulationRunner


def test_configuration(config_file: str):
    """Run test simulation with given configuration."""
    # Load configuration
    settings = Settings.from_json(file_path="config/settings.json")
    simulation_settings = SimulationSettings.from_json(file_path=config_file)

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
    print(f"  Testing Configuration: {config_file}")
    print(f"  Channel: {channel_name}")
    print(f"  Equalization: {simulation_settings.equalization_method}")
    print(f"  Power Allocation: {simulation_settings.power_allocation_type}")
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

    # List generated files
    print(f"\n{'='*80}")
    print(f"  Generated Files in images/{channel_name}/")
    print(f"{'='*80}")

    image_dir = Path(f"images/{channel_name}")
    if image_dir.exists():
        files = sorted(image_dir.glob("*.png"))
        for file in files:
            print(f"  ✓ {file.name}")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Test with MMSE equalization and Waterfilling
    test_configuration("config/simulation_settings_test2.json")
    test_configuration("config/simulation_settings_test2.json")
