"""Test script to demonstrate new filename pattern and channel-specific directories."""

from pathlib import Path

from ofdm_based_systems.configuration.models import Settings, SimulationSettings
from ofdm_based_systems.main import ResultsManager, SimulationRunner


def main():
    """Run test simulation with new naming convention."""
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
        # Extract channel name from path (e.g., "config/channel_models/severe_multipath.npy" -> "severe_multipath")
        channel_name = Path(simulation_settings.channel_model_path).stem
    elif simulation_settings.channel_type.value == "FLAT":
        channel_name = "flat"

    print(f"\n{'='*80}")
    print(f"  Testing New Filename Pattern")
    print(f"{'='*80}")
    print(f"  Channel Name: {channel_name}")
    print(f"  Images Directory: images/{channel_name}/")
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
    print(f"  Generated Files")
    print(f"{'='*80}")

    image_dir = Path(f"images/{channel_name}")
    if image_dir.exists():
        files = sorted(image_dir.glob("*.png"))
        for file in files:
            print(f"  ✓ {file}")

    print(f"\n{'='*80}")
    print("\n✓ Test completed successfully!\n")


if __name__ == "__main__":
    main()
    main()
