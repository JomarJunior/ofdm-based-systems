"""
This script is a worker that will be running in a separate process to update the BER vs SNR plot.
It listens for a specific file to be updated, then re-plots the graph and saves it to a file.
"""

import os
import time
import argparse
import matplotlib.pyplot as plt
import pandas as pd


def update_ber_vs_snr_plot(plot_path: str):
    last_modified_time = 0

    while True:
        try:
            current_modified_time = os.path.getmtime(plot_path)
            if current_modified_time != last_modified_time:
                last_modified_time = current_modified_time

                # Load the BER results
                ber_results = pd.read_csv(plot_path)

                # Plot BER vs SNR
                plt.figure()

                # Check if the results file has data (more than just headers)
                if len(ber_results) > 0:
                    for simulation, group in ber_results.groupby("simulation_name"):
                        plt.semilogy(
                            group["snr_db"], group["bit_error_rate"], marker="o", label=simulation
                        )

                    plt.ylim(1e-6, 1)
                    plt.xlim(min(ber_results["snr_db"]), max(ber_results["snr_db"]))
                    plt.legend()
                else:
                    # Handle empty results file by creating an empty plot with default limits
                    plt.text(
                        0.5,
                        0.5,
                        "No data available",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=plt.gca().transAxes,
                    )
                    plt.ylim(1e-6, 1)
                    plt.xlim(0, 30)  # Default x-axis range for empty plot

                plt.title("BER vs SNR")
                plt.xlabel("SNR (dB)")
                plt.ylabel("Bit Error Rate (BER)")
                plt.grid(True, which="both", linestyle="--", linewidth=0.5)
                plt.savefig("results/ber_vs_snr_plot.png")
                plt.close()
                print("Updated BER vs SNR plot.")

            time.sleep(1)  # Check every 1 second
        except Exception as e:
            print(f"Error updating BER vs SNR plot: {e}")
            time.sleep(1)  # Wait before retrying


def clear_results_file(plot_path: str):
    """Clear the results file by creating a new one with just headers."""
    try:
        if os.path.exists(plot_path):
            # Create a new file with just the headers
            with open(plot_path, "w") as f:
                f.write("simulation_name,snr_db,bit_error_rate\n")
            print(f"Results file {plot_path} has been cleared.")
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            with open(plot_path, "w") as f:
                f.write("simulation_name,snr_db,bit_error_rate\n")
            print(f"Results file {plot_path} has been created.")
    except Exception as e:
        print(f"Error clearing results file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BER vs SNR Plot Updater")
    parser.add_argument(
        "--mode",
        choices=["update", "clear"],
        default="update",
        help="Choose between updating the plot continuously or clearing the results file",
    )
    parser.add_argument(
        "--file", default="results/ber_results.csv", help="Path to the BER results CSV file"
    )

    args = parser.parse_args()

    if args.mode == "update":
        print(f"Starting update loop for file: {args.file}")
        update_ber_vs_snr_plot(args.file)
    elif args.mode == "clear":
        clear_results_file(args.file)
