import os
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image
from ofdm_based_systems.configuration.models import Settings, SimulationSettings
from ofdm_based_systems.simulation.models import Simulation


def update_results_csv(new_data: dict, file_path: str = "results/ber_results.csv"):
    """
    Format of the file:
    simulation_name, snr_db, bit_error_rate
    ...
    This function updates the row if the simulation_name and snr_db already exists in the file or appends a new row otherwise.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=["simulation_name", "snr_db", "bit_error_rate"])

    # Check if the simulation_name already exists
    existing_row = df[df["simulation_name"] == new_data["simulation_name"]]
    existing_row = existing_row[existing_row["snr_db"] == new_data["snr_db"]]

    if not existing_row.empty:
        # Update the existing row
        df.loc[
            df["simulation_name"] == new_data["simulation_name"], ["snr_db", "bit_error_rate"]
        ] = [
            new_data["snr_db"],
            new_data["bit_error_rate"],
        ]
    else:
        # Append a new row
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

    df.to_csv(file_path, index=False)


def main():
    settings = Settings.from_json(file_path="config/settings.json")
    print(settings)
    simulation_settings = SimulationSettings.from_json(file_path="config/simulation_settings.json")
    print(simulation_settings)

    simulations = Simulation.create_from_simulation_settings(simulation_settings)

    results = []
    for sim in simulations:
        result = sim.run()
        results.append(result)

    for i, result in enumerate(results):
        print(f"Result {i + 1}:")
        for key, value in result.items():
            print(f"  {key}: {value}")

    # Save images from results
    os.makedirs("images", exist_ok=True)

    for i, result in enumerate(results):
        if "constellation_plot" in result:
            image: Image.Image = result["constellation_plot"]
            title = f"{result.get('title', 'unknown').replace(' ', '_')}"
            subtitle = f"{result.get('subtitle', 'unknown').replace(' ', '_')}"
            filename = f"constellation_{title}_{subtitle}.png"
            image.save(f"images/{filename}")
            image.close()

    # Plot BER vs SNR for all simulations
    plt.figure(figsize=(10, 6))
    bers = [result["bit_error_rate"] for result in results if "bit_error_rate" in result]
    snrs = [result["snr_db"] for result in results if "snr_db" in result]

    # Save results to CSV
    title = results[0].get("title", "unknown").replace(" ", "_") if results else "unknown"
    to_save = [
        {"simulation_name": title, "snr_db": snr_db, "bit_error_rate": ber}
        for snr_db, ber in zip(snrs, bers)
    ]
    for data in to_save:
        update_results_csv(data)

    print(f"bers: {bers}")
    print(f"snrs: {snrs}")
    plt.semilogy(snrs, bers, marker="o", linestyle="-", label="BER vs SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER vs SNR")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/ber_vs_snr.png")
    plt.close()


if __name__ == "__main__":
    main()
