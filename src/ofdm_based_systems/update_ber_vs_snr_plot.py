"""
This script is a worker that will be running in a separate process to update the BER vs SNR plot.
It listens for a specific file to be updated, then re-plots the graph and saves it to a file.
"""

import os
import time
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
                for simulation, group in ber_results.groupby("simulation_name"):
                    plt.semilogy(
                        group["snr_db"], group["bit_error_rate"], marker="o", label=simulation
                    )

                plt.title("BER vs SNR")
                plt.xlabel("SNR (dB)")
                plt.ylabel("Bit Error Rate (BER)")
                plt.grid(True, which="both", linestyle="--", linewidth=0.5)
                plt.legend()
                plt.ylim(1e-6, 1)
                plt.xlim(min(ber_results["snr_db"]), max(ber_results["snr_db"]))
                plt.savefig("results/ber_vs_snr_plot.png")
                plt.close()
                print("Updated BER vs SNR plot.")

            time.sleep(1)  # Check every 1 second1
        except Exception as e:
            print(f"Error updating BER vs SNR plot: {e}")
            time.sleep(1)  # Wait before retrying


if __name__ == "__main__":
    plot_file_path = "results/ber_results.csv"
    update_ber_vs_snr_plot(plot_file_path)
