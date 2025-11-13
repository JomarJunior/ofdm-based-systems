"""Main entry point for OFDM simulation system.

This module provides a clean, modular interface for running OFDM simulations
with proper separation of concerns and comprehensive result management.
"""

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from ofdm_based_systems.configuration.models import Settings, SimulationSettings
from ofdm_based_systems.simulation.models import Simulation


class ResultsManager:
    """Manages simulation results including CSV storage and visualization."""

    def __init__(
        self,
        results_dir: str = "results",
        images_dir: str = "images",
        channel_name: str = "default",
        doc_figures_dir: Union[str, Path, None] = "docs/figures",
    ):
        """Initialize results manager.

        Args:
            results_dir: Directory for storing CSV results
            images_dir: Base directory for storing plot images
            channel_name: Name of the channel (used to create subdirectory)
            doc_figures_dir: Directory for mirroring plots for documentation
        """
        self.results_dir = Path(results_dir)
        self.channel_name = channel_name
        # Create channel-specific subdirectory for images
        self.images_dir = Path(images_dir) / channel_name
        self.csv_path = self.results_dir / "ber_results.csv"
        self.doc_figures_dir: Optional[Path] = Path(doc_figures_dir) if doc_figures_dir else None
        self.doc_channel_dir: Optional[Path] = None

        # Create directories if they don't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        if self.doc_figures_dir:
            self.doc_figures_dir.mkdir(parents=True, exist_ok=True)
            self.doc_channel_dir = self.doc_figures_dir / self.channel_name
            self.doc_channel_dir.mkdir(parents=True, exist_ok=True)

    def _mirror_to_docs(self, source_path: Path) -> Optional[Path]:
        """Copy a generated plot to the documentation figures directory."""

        if not self.doc_channel_dir or not source_path.exists():
            return None

        try:
            relative_path = source_path.relative_to(self.images_dir)
        except ValueError:
            relative_path = source_path.name

        destination = self.doc_channel_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        return destination

    def update_ber_csv(self, simulation_name: str, snr_db: float, bit_error_rate: float) -> None:
        """Update or append BER results to CSV file.

        Args:
            simulation_name: Name of the simulation
            snr_db: Signal-to-noise ratio in dB
            bit_error_rate: Bit error rate value
        """
        new_data = {
            "simulation_name": simulation_name,
            "snr_db": snr_db,
            "bit_error_rate": bit_error_rate,
        }

        if self.csv_path.exists():
            df = pd.read_csv(self.csv_path)
        else:
            df = pd.DataFrame(columns=["simulation_name", "snr_db", "bit_error_rate"])

        # Check if row exists
        existing_row = df[(df["simulation_name"] == simulation_name) & (df["snr_db"] == snr_db)]

        if not existing_row.empty:
            # Update existing row
            df.loc[
                (df["simulation_name"] == simulation_name) & (df["snr_db"] == snr_db),
                "bit_error_rate",
            ] = bit_error_rate
        else:
            # Append new row
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

        df.to_csv(self.csv_path, index=False)

    def save_constellation_plot(
        self,
        image: Image.Image,
        prefix_type: str,
        modulation_type: str,
        equalization_method: str,
        constellation_order: int,
        constellation_type: str,
        power_allocation: str,
        snr_db: float,
    ) -> Path:
        """Save constellation plot image with structured filename.

        Args:
            image: PIL Image of constellation plot
            prefix_type: Prefix scheme (e.g., "CP", "ZP", "NONE")
            modulation_type: Modulation type (e.g., "OFDM", "SC-OFDM")
            equalization_method: Equalization method (e.g., "ZF", "MMSE", "NONE")
            constellation_order: Constellation order (e.g., 4, 16, 64)
            constellation_type: Constellation type (e.g., "QAM", "PSK")
            power_allocation: Power allocation type (e.g., "WF", "UNIFORM")
            snr_db: Signal-to-noise ratio in dB

        Returns:
            Path to saved image file

        Example filename: "CP-OFDM-ZF-64QAM-WF-SNR30_0dB.png"
        """
        # Format SNR with underscore for decimal point (e.g., 30.5 -> "30_5")
        snr_str = f"{snr_db:.1f}".replace(".", "_")

        # Build filename components
        filename = (
            f"{prefix_type}-{modulation_type}-{equalization_method}-"
            f"{constellation_order}{constellation_type}-{power_allocation}-"
            f"SNR{snr_str}dB.png"
        )

        filepath = self.images_dir / filename
        image.save(filepath)
        self._mirror_to_docs(filepath)
        return filepath

    def plot_ber_vs_snr(self, results: List[Dict[str, Any]]) -> Path:
        """Generate and save BER vs SNR plot.

        Args:
            results: List of simulation result dictionaries

        Returns:
            Path to saved plot file
        """
        bers = [r["bit_error_rate"] for r in results if "bit_error_rate" in r]
        snrs = [r["snr_db"] for r in results if "snr_db" in r]

        if not bers or not snrs:
            print("Warning: No BER or SNR data to plot")
            return self.images_dir / "ber_vs_snr.png"

        # Extract configuration for filename from first result
        if results:
            result = results[0]
            prefix_type = result.get("prefix_acronym", "NONE")
            modulation_type = result.get("modulator_type", "OFDM")
            equalization_method = result.get("equalizator_type", "NONE")
            constellation_order = result.get("constellation_order", 16)
            constellation_type = result.get("constellation_scheme", "QAM")
            power_allocation = result.get("power_allocation_acronym", "UNIFORM")

            output_filename = (
                f"{prefix_type}-{modulation_type}-{equalization_method}-"
                f"{constellation_order}{constellation_type}-{power_allocation}-"
                f"BER_vs_SNR.png"
            )
        else:
            output_filename = "ber_vs_snr.png"

        plt.figure(figsize=(10, 6))
        plt.semilogy(snrs, bers, marker="o", linestyle="-", label="BER vs SNR", color="blue")
        plt.xlabel("SNR (dB)", fontsize=12)
        plt.ylabel("Bit Error Rate (BER)", fontsize=12)
        plt.title("BER vs SNR Performance", fontsize=14, fontweight="bold")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        plt.legend(fontsize=11)
        plt.tight_layout()

        filepath = self.images_dir / output_filename
        plt.savefig(filepath, dpi=150)
        plt.close()
        self._mirror_to_docs(filepath)

        return filepath


class SimulationRunner:
    """Orchestrates the execution of OFDM simulations."""

    def __init__(
        self,
        settings: Settings,
        simulation_settings: SimulationSettings,
        results_manager: ResultsManager,
    ):
        """Initialize simulation runner.

        Args:
            settings: General project settings
            simulation_settings: Simulation-specific settings
            results_manager: Manager for handling results
        """
        self.settings = settings
        self.simulation_settings = simulation_settings
        self.results_manager = results_manager

    def run_all(self) -> List[Dict[str, Any]]:
        """Run all simulations defined in settings.

        Returns:
            List of result dictionaries from all simulations
        """
        print("=" * 80)
        print(f"  {self.settings.project_name} v{self.settings.version}")
        print("=" * 80)
        print(f"\n{self.simulation_settings}\n")

        # Create simulations (one per SNR value)
        simulations = Simulation.create_from_simulation_settings(self.simulation_settings)
        print(f"Created {len(simulations)} simulation(s) to run\n")

        # Run simulations
        results = []
        for i, sim in enumerate(simulations, start=1):
            print(f"\n{'#' * 80}")
            print(f"  Running Simulation {i}/{len(simulations)} (SNR = {sim.snr_db} dB)")
            print(f"{'#' * 80}\n")

            result = sim.run()
            results.append(result)

            print(f"\n  ✓ Simulation {i} completed")
            print(f"    BER: {result['bit_error_rate']:.6e}")
            print(f"    Bit Errors: {result['bit_errors']}/{result['total_bits']}")
            print(f"    PAPR: {result['papr_db']:.2f} dB")
            if "channel_capacity" in result:
                print(f"    Channel Capacity: {result.get('channel_capacity', 'N/A')}")

        return results

    def process_results(self, results: List[Dict[str, Any]]) -> None:
        """Process and save simulation results.

        Args:
            results: List of result dictionaries
        """
        if not results:
            print("Warning: No results to process")
            return

        print(f"\n{'=' * 80}")
        print("  Processing Results")
        print("=" * 80)

        # Save constellation plots
        saved_images = []
        for result in results:
            if "constellation_plot" in result:
                image: Image.Image = result["constellation_plot"]

                # Extract configuration for filename
                prefix_type = result.get("prefix_acronym", "NONE")
                modulation_type = result.get("modulator_type", "OFDM")
                equalization_method = result.get("equalizator_type", "NONE")
                constellation_order = result.get("constellation_order", 16)
                constellation_type = result.get("constellation_scheme", "QAM")
                power_allocation = result.get("power_allocation_acronym", "UNIFORM")
                snr_db = result.get("snr_db", 0.0)

                filepath = self.results_manager.save_constellation_plot(
                    image=image,
                    prefix_type=prefix_type,
                    modulation_type=modulation_type,
                    equalization_method=equalization_method,
                    constellation_order=constellation_order,
                    constellation_type=constellation_type,
                    power_allocation=power_allocation,
                    snr_db=snr_db,
                )
                saved_images.append(filepath)
                image.close()

        print(f"  ✓ Saved {len(saved_images)} constellation plot(s)")
        if saved_images and self.results_manager.doc_channel_dir:
            print(
                f"  -> Mirrored constellation plot(s) to "
                f"{self.results_manager.doc_channel_dir}"
            )

        # Save BER data to CSV
        simulation_name = (
            results[0].get("title", "unknown").replace(" ", "_") if results else "unknown"
        )

        for result in results:
            if "bit_error_rate" in result and "snr_db" in result:
                self.results_manager.update_ber_csv(
                    simulation_name=simulation_name,
                    snr_db=result["snr_db"],
                    bit_error_rate=result["bit_error_rate"],
                )

        print(f"  ✓ Updated BER results CSV: {self.results_manager.csv_path}")

        # Generate BER vs SNR plot
        plot_path = self.results_manager.plot_ber_vs_snr(results)
        print(f"  ✓ Generated BER vs SNR plot: {plot_path}")
        if self.results_manager.doc_channel_dir:
            print(
                f"  -> Mirrored BER plot to "
                f"{self.results_manager.doc_channel_dir / plot_path.name}"
            )

        # Print summary statistics
        print(f"\n{'=' * 80}")
        print("  Summary Statistics")
        print("=" * 80)

        bers = [r["bit_error_rate"] for r in results]
        snrs = [r["snr_db"] for r in results]
        paprs = [r["papr_db"] for r in results]

        print(f"  SNR Range: {min(snrs):.1f} dB to {max(snrs):.1f} dB")
        print(f"  BER Range: {min(bers):.6e} to {max(bers):.6e}")
        print(f"  Average PAPR: {sum(paprs) / len(paprs):.2f} dB")

        if any("channel_capacity" in r for r in results):
            capacities = [r.get("channel_capacity") for r in results if "channel_capacity" in r]
            if capacities:
                print(
                    f"  Channel Capacity Range: {min(capacities):.2f} to {max(capacities):.2f} bits/channel use"
                )

        print("=" * 80)


def main():
    """Main entry point for OFDM simulation system."""
    try:
        # Load configuration
        settings = Settings.from_json(file_path="config/settings.json")
        simulation_settings = SimulationSettings.from_json(
            file_path="config/simulation_settings.json"
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

        print("\n✓ All simulations completed successfully!\n")

    except FileNotFoundError as e:
        print(f"Error: Configuration file not found - {e}")
        return 1
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
