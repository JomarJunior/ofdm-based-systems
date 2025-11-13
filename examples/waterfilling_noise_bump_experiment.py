"""Water-filling robustness experiment with colored noise bumps.

Generates BER data and constellation plots for three scenarios:
1. Baseline CP-OFDM with uniform power allocation under a +3 dB high-band noise bump.
2. CP-OFDM with water-filling under the same +3 dB noise bump.
3. CP-OFDM with water-filling under a stronger +6 dB noise bump.

Outputs BER curves, per-SNR constellation diagrams, and CSV summaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from ofdm_based_systems.bits_generation.models import RandomBitsGenerator
from ofdm_based_systems.channel.models import ChannelModel
from ofdm_based_systems.constellation.models import QAMConstellationMapper
from ofdm_based_systems.equalization.models import MMSEEqualizator
from ofdm_based_systems.modulation.models import OFDMModulator
from ofdm_based_systems.noise.models import NoNoiseModel
from ofdm_based_systems.power_allocation.models import (
    UniformPowerAllocation,
    WaterfillingPowerAllocation,
)
from ofdm_based_systems.prefix.models import CyclicPrefixScheme
from ofdm_based_systems.serial_parallel.models import SerialToParallelConverter
from ofdm_based_systems.simulation.models import read_bits_from_stream


@dataclass
class Scenario:
    key: str
    name: str
    short_prefix: str
    power_allocation: str  # "UNIFORM" or "WATERFILLING"
    noise_bump_db: float


def create_noise_profile(num_subcarriers: int, bump_db: float) -> np.ndarray:
    """Create a piecewise noise profile that adds a bump to the top quarter band."""
    profile = np.ones(num_subcarriers, dtype=np.float64)
    if bump_db <= 0:
        return profile

    bump_factor = 10 ** (bump_db / 10)
    start = int(0.75 * num_subcarriers)
    profile[start:] = bump_factor
    return profile


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    np.random.seed(42)

    # Simulation constants
    channel_path = Path("config/channel_models/Lin-Phoong_P2.npy")
    channel_impulse = np.load(channel_path)

    snr_values = [0, 5, 10, 15, 20, 25, 30]
    num_subcarriers = 64
    constellation_order = 64
    bits_per_symbol = int(np.log2(constellation_order))
    num_ofdm_symbols = 2048  # 2048 OFDM blocks × 64 subcarriers × 6 bits ≈ 786k bits
    total_qam_symbols = num_ofdm_symbols * num_subcarriers
    total_bits = total_qam_symbols * bits_per_symbol

    prefix_length = int(1.0 * (len(channel_impulse) - 1))
    prefix_scheme = CyclicPrefixScheme(prefix_length=prefix_length)
    serial_parallel = SerialToParallelConverter()
    mapper = QAMConstellationMapper(order=constellation_order)

    channel_model = ChannelModel(
        impulse_response=channel_impulse, snr_db=0.0, noise_model=NoNoiseModel()
    )
    channel_frequency_response = np.fft.fft(channel_impulse, num_subcarriers)
    channel_gains = np.abs(channel_frequency_response) ** 2

    base_output_dir = Path("images/Lin-Phoong_P2/CP-OFDM-Waterfilling-Study")
    results_dir = Path("results")

    scenarios: List[Scenario] = [
        Scenario(
            key="uniform",
            name="Baseline (Uniform Power, +3 dB bump)",
            short_prefix="CP-OFDM-UNIFORM",
            power_allocation="UNIFORM",
            noise_bump_db=3.0,
        ),
        Scenario(
            key="wf_plus3",
            name="Water-filling (+3 dB noise bump)",
            short_prefix="CP-OFDM-WF-3dB",
            power_allocation="WATERFILLING",
            noise_bump_db=3.0,
        ),
        Scenario(
            key="wf_plus6",
            name="Water-filling (+6 dB noise bump)",
            short_prefix="CP-OFDM-WF-6dB",
            power_allocation="WATERFILLING",
            noise_bump_db=6.0,
        ),
    ]

    scenario_results: Dict[str, List[float]] = {}

    for scenario in scenarios:
        scenario_dir = base_output_dir / scenario.short_prefix
        ensure_dir(scenario_dir)

        noise_profile = create_noise_profile(num_subcarriers, scenario.noise_bump_db)
        ber_values: List[float] = []

        for snr_db in snr_values:
            bits_generator = RandomBitsGenerator()
            bits_stream = bits_generator.generate_bits(total_bits)
            bits_list = read_bits_from_stream(bits_stream)

            symbols = mapper.encode(bits_stream)
            bits_stream.close()

            parallel_data = serial_parallel.to_parallel(symbols, num_subcarriers)

            noise_power = 10 ** (-snr_db / 10)

            if scenario.power_allocation == "WATERFILLING":
                effective_gains = channel_gains / noise_profile
                allocator = WaterfillingPowerAllocation(
                    total_power=1.0,
                    channel_gains=effective_gains,
                    noise_power=noise_power,
                )
            else:
                allocator = UniformPowerAllocation(total_power=1.0, num_subcarriers=num_subcarriers)

            power_allocation = allocator.allocate()
            if scenario.power_allocation == "WATERFILLING":
                power_floor = 1e-4
                power_allocation = np.maximum(power_allocation, power_floor)
                power_allocation = power_allocation / np.sum(power_allocation)
            parallel_data = parallel_data * np.sqrt(power_allocation)

            equalizator = MMSEEqualizator(channel_frequency_response=channel_frequency_response, snr_db=snr_db)
            modulator = OFDMModulator(
                num_subcarriers=num_subcarriers,
                prefix_scheme=prefix_scheme,
                equalizator=equalizator,
            )

            modulated_signal = modulator.modulate(parallel_data)
            serial_signal = serial_parallel.to_serial(modulated_signal)
            received_signal = channel_model.transmit(serial_signal)
            received_parallel = serial_parallel.to_parallel(
                received_signal, num_subcarriers + prefix_length
            )
            demodulated_data = modulator.demodulate(received_parallel)

            # Inject colored noise after equalization
            noise_variances = noise_power * noise_profile
            noise_std = np.sqrt(noise_variances / 2.0)[np.newaxis, :]
            noise_matrix = (
                np.random.normal(size=demodulated_data.shape)
                + 1j * np.random.normal(size=demodulated_data.shape)
            ) * noise_std
            demodulated_data_noisy = demodulated_data + noise_matrix

            # Compensate power allocation
            power_sqrt = np.sqrt(power_allocation)
            power_sqrt_safe = power_sqrt.copy()
            power_sqrt_safe[power_sqrt_safe < 1e-10] = 1.0
            demodulated_data_noisy = demodulated_data_noisy / power_sqrt_safe

            demodulated_serial = serial_parallel.to_serial(demodulated_data_noisy)
            avg_power = np.mean(np.abs(demodulated_serial) ** 2)
            if avg_power > 1e-12:
                demodulated_serial = demodulated_serial / np.sqrt(avg_power)

            received_bits_stream = mapper.decode(demodulated_serial)
            received_bits_list = read_bits_from_stream(received_bits_stream)
            received_bits_stream.close()

            bit_errors = sum(b1 != b2 for b1, b2 in zip(bits_list, received_bits_list))
            ber = bit_errors / total_bits
            ber_values.append(ber)

            # Save constellation scatter plot
            ideal_points = mapper.constellation
            plt.figure(figsize=(6, 6))
            plt.scatter(
                demodulated_serial.real,
                demodulated_serial.imag,
                color="tab:blue",
                alpha=0.15,
                s=6,
                label="Received Symbols",
            )
            plt.scatter(
                ideal_points.real,
                ideal_points.imag,
                color="tab:red",
                marker="o",
                s=30,
                label="Ideal Constellation",
            )
            plt.title(
                f"{scenario.name}\nSNR = {snr_db} dB | BER = {ber:.3e}"
            )
            plt.xlabel("In-Phase")
            plt.ylabel("Quadrature")
            plt.axhline(0, color="gray", linewidth=0.5)
            plt.axvline(0, color="gray", linewidth=0.5)
            plt.grid(True, linestyle=":", alpha=0.4)
            plt.legend(loc="upper right", fontsize=8)
            plt.xlim([-2.2, 2.2])
            plt.ylim([-2.2, 2.2])
            plt.gca().set_aspect("equal", adjustable="box")

            filename = scenario_dir / f"{scenario.short_prefix}-SNR{snr_db:02d}dB.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=200)
            plt.close()

        scenario_results[scenario.name] = ber_values

        # Write CSV summary for scenario
        csv_path = results_dir / f"ber_{scenario.key}_waterfilling_study.csv"
        with csv_path.open("w", encoding="utf-8") as fp:
            fp.write("snr_db,bit_error_rate\n")
            for snr_db, ber in zip(snr_values, ber_values):
                fp.write(f"{snr_db},{ber}\n")

    # Plot BER comparison
    ensure_dir(base_output_dir)
    plt.figure(figsize=(7, 5))
    for scenario in scenarios:
        ber_values = scenario_results[scenario.name]
        plt.semilogy(
            snr_values,
            ber_values,
            marker="o",
            linewidth=2,
            label=scenario.name,
        )

    plt.title("CP-OFDM BER vs. SNR with Colored Noise Bumps")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend()
    comparison_path = base_output_dir / "CP-OFDM-waterfilling-ber-comparison.png"
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=250)
    plt.close()


if __name__ == "__main__":
    main()
