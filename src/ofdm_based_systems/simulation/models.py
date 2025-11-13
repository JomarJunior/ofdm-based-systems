import os
from io import BytesIO
from typing import Any, BinaryIO, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ofdm_based_systems.bits_generation.models import IGenerator, RandomBitsGenerator
from ofdm_based_systems.channel.models import ChannelModel
from ofdm_based_systems.configuration.enums import (
    ConstellationType,
    EqualizationMethod,
    ModulationType,
    NoiseType,
    PowerAllocationType,
    PrefixType,
)
from ofdm_based_systems.configuration.models import SimulationSettings
from ofdm_based_systems.constellation.models import (
    IConstellationMapper,
    PSKConstellationMapper,
    QAMConstellationMapper,
)
from ofdm_based_systems.equalization.models import (
    MMSEEqualizator,
    NoEqualizator,
    ZeroForcingEqualizator,
)
from ofdm_based_systems.modulation.models import (
    IModulator,
    OFDMModulator,
    SingleCarrierOFDMModulator,
)
from ofdm_based_systems.noise.models import AWGNoiseModel, NoNoiseModel
from ofdm_based_systems.power_allocation.models import (
    UniformPowerAllocation,
    WaterfillingPowerAllocation,
)
from ofdm_based_systems.prefix.models import (
    CyclicPrefixScheme,
    NoPrefixScheme,
    ZeroPaddingPrefixScheme,
)
from ofdm_based_systems.serial_parallel.models import SerialToParallelConverter


def read_bits_from_stream(stream: BinaryIO) -> List[int]:
    """Reads bits from a binary stream and returns them as a list of integers (0s and 1s)."""
    byte = stream.read(1)
    bits = []
    while byte:
        byte_value = ord(byte)
        for i in range(8):
            bits.append((byte_value >> (7 - i)) & 1)
        byte = stream.read(1)
    stream.seek(0)  # Reset stream position to the beginning
    return bits


class Simulation:
    CONSTELLATION_SCHEME_MAPPERS = {
        ConstellationType.QAM: QAMConstellationMapper,
        ConstellationType.PSK: PSKConstellationMapper,
    }

    MODULATOR_SCHEME_MAPPERS = {
        ModulationType.OFDM: OFDMModulator,
        ModulationType.SC_OFDM: SingleCarrierOFDMModulator,
    }

    PREFIX_SCHEME_MAPPERS = {
        PrefixType.NONE: NoPrefixScheme,
        PrefixType.CYCLIC: CyclicPrefixScheme,
        PrefixType.ZERO: ZeroPaddingPrefixScheme,
    }

    EQUALIZATOR_SCHEME_MAPPERS = {
        EqualizationMethod.NONE: NoEqualizator,
        EqualizationMethod.ZF: ZeroForcingEqualizator,
        EqualizationMethod.MMSE: MMSEEqualizator,
    }

    NOISE_SCHEME_MAPPERS = {
        NoiseType.AWGN: AWGNoiseModel,
        NoiseType.NONE: NoNoiseModel,
    }

    POWER_ALLOCATION_MAPPERS = {
        PowerAllocationType.UNIFORM: UniformPowerAllocation,
        PowerAllocationType.WATERFILLING: WaterfillingPowerAllocation,
    }

    def __init__(
        self,
        num_bits: Optional[int] = None,
        num_symbols: Optional[int] = None,
        num_subcarriers: int = 64,
        constellation_order: int = 16,
        constellation_scheme: ConstellationType = ConstellationType.QAM,
        modulator_type: ModulationType = ModulationType.OFDM,
        prefix_scheme: PrefixType = PrefixType.CYCLIC,
        prefix_length_ratio: float = 1.0,  # should stay at 1.0
        equalizator_type: EqualizationMethod = EqualizationMethod.MMSE,
        snr_db: float = 20.0,
        noise_scheme: NoiseType = NoiseType.AWGN,
        power_allocation_type: PowerAllocationType = PowerAllocationType.UNIFORM,
        channel_impulse_response: Optional[NDArray[np.complex128]] = None,
        verbose: bool = True,
    ):
        if num_bits is None and num_symbols is None:
            raise ValueError("Either num_bits or num_symbols must be provided.")
        if num_bits is not None and num_symbols is not None:
            raise ValueError("Only one of num_bits or num_symbols should be provided.")

        self.num_bits = num_bits
        self.num_symbols = num_symbols
        self.num_subcarriers = num_subcarriers
        self.constellation_order = constellation_order
        self.constellation_scheme = constellation_scheme
        self.modulator_type = modulator_type
        self.prefix_scheme = prefix_scheme
        self.prefix_length_ratio = prefix_length_ratio
        self.equalizator_type = equalizator_type
        self.snr_db = snr_db
        self.noise_scheme = noise_scheme
        self.power_allocation_type = power_allocation_type
        self.channel_impulse_response = channel_impulse_response
        self.verbose = verbose

    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    @classmethod
    def create_from_simulation_settings(
        cls, simulation_settings: SimulationSettings
    ) -> List["Simulation"]:
        """Create multiple simulations from settings.

        Args:
            simulation_settings: Configuration settings

        Returns:
            List of Simulation instances, one per SNR value
        """
        # Load channel impulse response if custom channel is specified
        channel_impulse_response = None
        if simulation_settings.channel_type.value == "CUSTOM":
            channel_path = simulation_settings.channel_model_path
            if not channel_path:
                raise ValueError("channel_model_path must be specified when channel_type is CUSTOM")

            # Handle relative paths from project root
            if not channel_path.startswith("/"):
                # Assume path is relative to current working directory
                channel_path = os.path.abspath(channel_path)

            if not os.path.exists(channel_path):
                raise FileNotFoundError(f"Channel model file not found: {channel_path}")

            try:
                channel_impulse_response = np.load(channel_path)
                print(f"✓ Loaded custom channel impulse response from: {channel_path}")
                print(f"  Channel length: {len(channel_impulse_response)} taps")
                print(f"  Channel dtype: {channel_impulse_response.dtype}")
            except Exception as e:
                raise ValueError(f"Failed to load channel model from {channel_path}: {e}")

        simulations = []
        for snr in simulation_settings.signal_noise_ratios:
            simulation = cls(
                num_bits=simulation_settings.num_bits,
                num_symbols=simulation_settings.num_symbols,
                num_subcarriers=simulation_settings.num_bands,
                constellation_order=simulation_settings.constellation_order,
                constellation_scheme=simulation_settings.constellation_type,
                modulator_type=simulation_settings.modulation_type,
                prefix_scheme=simulation_settings.prefix_type,
                prefix_length_ratio=simulation_settings.prefix_length_ratio,
                equalizator_type=simulation_settings.equalization_method,
                snr_db=snr,
                noise_scheme=simulation_settings.noise_type,
                power_allocation_type=simulation_settings.power_allocation_type,
                channel_impulse_response=channel_impulse_response,
            )
            simulations.append(simulation)
        return simulations

    def run(self) -> Dict[str, Any]:
        results = {}
        print("=" * 50)
        print("Starting OFDM-based System Simulation")
        print("=" * 50)
        # Bits Generation
        bits_generator: IGenerator = RandomBitsGenerator()
        # Constellation Mapping
        constellation_mapper: IConstellationMapper = self.CONSTELLATION_SCHEME_MAPPERS.get(
            self.constellation_scheme, QAMConstellationMapper
        )(order=self.constellation_order)
        # Serial to Parallel Conversion
        serial_to_parallel_converter: SerialToParallelConverter = SerialToParallelConverter()

        # Noise Model
        noise_model = self.NOISE_SCHEME_MAPPERS.get(self.noise_scheme, AWGNoiseModel)()

        # Channel Model
        if self.channel_impulse_response is not None:
            # Use custom channel from configuration
            channel_impulse_response: NDArray[np.complex128] = self.channel_impulse_response
            self._log(
                f"Using custom channel impulse response ({len(channel_impulse_response)} taps)"
            )
        else:
            # Use default multipath channel (backward compatibility)
            channel_impulse_response = np.array(
                [
                    (7.767824138452235072e-01 + 4.560896742466611919e-01j),
                    (-6.669848996328063551e-02 + 2.839935704583463338e-01j),
                    (1.398968327715586490e-01 - 1.591963958343969865e-01j),
                    (2.229949514514480494e-02 + 2.409945439452868821e-01j),
                ],
                dtype=np.complex128,
            )
            self._log("Using default multipath channel (4 taps)")

        channel: ChannelModel = ChannelModel(
            impulse_response=channel_impulse_response, snr_db=self.snr_db, noise_model=noise_model
        )
        prefix_length = int(self.prefix_length_ratio * channel.order)
        if self.prefix_scheme == PrefixType.NONE:
            prefix_length = 0
        print(f"Using prefix length: {prefix_length}")

        # Prefix Scheme
        prefix_scheme = self.PREFIX_SCHEME_MAPPERS.get(self.prefix_scheme, NoPrefixScheme)(
            prefix_length=prefix_length
        )
        # Equalizator
        equalizator = self.EQUALIZATOR_SCHEME_MAPPERS.get(self.equalizator_type, NoEqualizator)(
            channel_frequency_response=np.fft.fft(channel_impulse_response, self.num_subcarriers),
            snr_db=self.snr_db,
        )

        # Modulator
        modulator: IModulator = self.MODULATOR_SCHEME_MAPPERS.get(
            self.modulator_type, OFDMModulator
        )(
            num_subcarriers=self.num_subcarriers,
            prefix_scheme=prefix_scheme,
            equalizator=equalizator,
        )

        results.update(
            {
                "num_bits": self.num_bits,
                "num_symbols": self.num_symbols,
                "num_subcarriers": self.num_subcarriers,
                "constellation_order": self.constellation_order,
                "constellation_scheme": self.constellation_scheme.name,
                "modulator_type": self.modulator_type.name,
                "prefix_scheme": self.prefix_scheme.name,
                "prefix_acronym": prefix_scheme.acronym,
                "equalizator_type": self.equalizator_type.name,
                "snr_db": self.snr_db,
                "noise_scheme": self.noise_scheme.name,
                "power_allocation_type": self.power_allocation_type.name,
                "power_allocation_acronym": (
                    "WF"
                    if self.power_allocation_type == PowerAllocationType.WATERFILLING
                    else "UNIFORM"
                ),
                "title": (
                    f"{prefix_scheme.acronym}-"
                    f"{self.modulator_type.name}-{self.equalizator_type.name}"
                ),
                "subtitle": (
                    f"{self.constellation_order}{self.constellation_scheme.name}-"
                    f"SNR{self.snr_db}dB-{self.power_allocation_type.name}"
                ),
            }
        )

        print("=" * 50)
        print("Generating bits stream...")
        print("=" * 50)

        total_bits = self.num_bits
        if self.num_symbols is not None:
            total_bits = self.num_symbols * int(np.log2(self.constellation_order))

        if total_bits is None:
            raise ValueError("Total bits could not be determined.")

        print(f"Generating {total_bits} random bits...")
        bits = bits_generator.generate_bits(total_bits)

        # Read bits from the binary stream just to display
        bits_list = read_bits_from_stream(bits)
        print(f"Generated Bits Length: {len(bits_list)}")
        # Do not use bits_list further; use bits directly for mapping

        print("=" * 50)
        print("Mapping bits to constellation symbols...")
        print("=" * 50)
        symbols: NDArray[np.complex128] = constellation_mapper.encode(bits)

        print(f"Mapped Symbols Length: {len(symbols)}")

        print("=" * 50)
        print("Serial to Parallel Conversion...")
        print("=" * 50)
        parallel_data = serial_to_parallel_converter.to_parallel(
            symbols, self.num_subcarriers  # type: ignore
        )

        print(f"Parallel Data Shape: {parallel_data.shape}")

        # Power Allocation
        print("=" * 50)
        print(f"Applying Power Allocation ({self.power_allocation_type.name})...")
        print("=" * 50)

        channel_frequency_response = np.fft.fft(channel_impulse_response, self.num_subcarriers)
        channel_gains = np.abs(channel_frequency_response) ** 2
        noise_power = 10 ** (-self.snr_db / 10)

        if self.power_allocation_type == PowerAllocationType.WATERFILLING:
            power_allocator = WaterfillingPowerAllocation(
                total_power=1.0,
                channel_gains=channel_gains,
                noise_power=noise_power,
            )
        else:
            power_allocator = UniformPowerAllocation(
                total_power=1.0,
                num_subcarriers=self.num_subcarriers,
            )

        power_allocation = power_allocator.allocate()
        self._log(
            f"Power allocation computed: min={power_allocation.min():.6f}, max={power_allocation.max():.6f}"
        )

        # Apply power allocation to parallel data
        parallel_data = parallel_data * np.sqrt(power_allocation)
        results["allocated_power"] = power_allocation.tolist()

        print("=" * 50)
        print("OFDM-based Modulation")
        print("=" * 50)
        modulated_signal = modulator.modulate(parallel_data)

        print(f"Modulated Signal Shape: {modulated_signal.shape}")

        # Calculate PAPR
        power_signal = np.abs(modulated_signal) ** 2
        peak_power = np.max(power_signal)
        avg_power = np.mean(power_signal)
        papr_db = 10 * np.log10(peak_power / avg_power) if avg_power > 0 else float("inf")
        print(f"PAPR: {papr_db:.2f} dB")
        results.update({"papr_db": papr_db})

        print("=" * 50)
        print("Parallel to Serial Conversion...")
        print("=" * 50)
        serial_signal = serial_to_parallel_converter.to_serial(modulated_signal)
        print(f"Serial Signal Shape: {serial_signal.shape}")

        print("=" * 50)
        print("Transmitting signal through the channel...")
        print("=" * 50)

        received_signal = channel.transmit(serial_signal)

        print(f"Received Signal Shape: {received_signal.shape}")

        print("=" * 50)
        print("Serial to Parallel Conversion of Received Signal...")
        print("=" * 50)

        received_parallel_signal = serial_to_parallel_converter.to_parallel(
            received_signal, self.num_subcarriers + prefix_scheme.prefix_length  # type: ignore
        )
        print(f"Received Parallel Signal Shape: {received_parallel_signal.shape}")

        print("=" * 50)
        print("OFDM-based Demodulation")
        print("=" * 50)
        demodulated_data = modulator.demodulate(received_parallel_signal)
        print(f"Demodulated Data Shape: {demodulated_data.shape}")

        # Compensate for power allocation
        print("=" * 50)
        print("Compensating for Power Allocation...")
        print("=" * 50)
        power_sqrt = np.sqrt(power_allocation)
        power_sqrt[power_sqrt < 1e-10] = 1.0  # Avoid division by near-zero
        demodulated_data = demodulated_data / power_sqrt
        self._log("Power allocation compensation applied")

        print("=" * 50)
        print("Parallel to Serial Conversion of Demodulated Data...")
        print("=" * 50)
        demodulated_serial_data = serial_to_parallel_converter.to_serial(demodulated_data)
        print(f"Demodulated Serial Data Shape: {demodulated_serial_data.shape}")

        # Normalize symbols to constellation scale
        print("=" * 50)
        print("Normalizing symbols for constellation demapping...")
        print("=" * 50)
        # Calculate current average power of received symbols
        current_avg_power = np.mean(np.abs(demodulated_serial_data) ** 2)
        # Constellation has unit average power, so normalize to unit power
        if current_avg_power > 1e-10:  # Avoid division by zero
            normalization_factor = np.sqrt(current_avg_power)
            demodulated_serial_data = demodulated_serial_data / normalization_factor
            self._log(f"Normalized symbols: avg power {current_avg_power:.6f} -> 1.0")
        else:
            self._log("Warning: Received signal has near-zero power, skipping normalization")

        print("=" * 50)
        print("Constellation Demapping...")
        print("=" * 50)
        received_bits = constellation_mapper.decode(demodulated_serial_data)

        received_bits_list = read_bits_from_stream(received_bits)
        print(f"Received Bits Length: {len(received_bits_list)}")

        # Calculate Bit Error Rate (BER)
        bit_errors = sum(b1 != b2 for b1, b2 in zip(bits_list, received_bits_list))
        ber = bit_errors / total_bits if total_bits > 0 else 0.0
        print("=" * 50)
        print(f"Bit Errors: {bit_errors} out of {total_bits} bits")
        print(f"Bit Error Rate (BER): {ber:.6f}")
        print("=" * 50)
        results.update(
            {
                "bit_errors": bit_errors,
                "total_bits": total_bits,
                "bit_error_rate": ber,
            }
        )

        # Generate plot image and store in results

        # Clear previous plots
        plt.clf()
        plt.cla()

        plt.figure(figsize=(8, 8))

        # Plot received Symbols
        plt.scatter(
            demodulated_serial_data.real,
            demodulated_serial_data.imag,
            color="blue",
            marker=".",
            alpha=0.1,
            label="Received Symbols",
        )

        # Plot the constellation diagram
        ideal_points = constellation_mapper.constellation
        plt.scatter(
            ideal_points.real,
            ideal_points.imag,
            color="red",
            marker="o",
            label="Ideal Constellation Points",
        )

        # Set plot attributes
        plt.title(f"{results['title']}")
        plt.xlabel("In-Phase")
        plt.ylabel("Quadrature")
        plt.axhline(0, color="black", lw=0.5)
        plt.axvline(0, color="black", lw=0.5)
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.axis("equal")

        # Add text box with BER, SNR, and PAPR
        textstr = f"BER: {ber:.6f}\n" f"SNR: {self.snr_db} dB\n" f"PAPR: {papr_db:.2f} dB"
        plt.gcf().text(0.15, 0.75, textstr, fontsize=10, bbox=dict(facecolor="white", alpha=0.5))
        plt.tight_layout()

        # Save the plot to a BytesIO object
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format="png")
        img_buffer.seek(0)

        # Load the image from the BytesIO object
        image = Image.open(img_buffer)
        results["constellation_plot"] = image

        # Close the plot to free memory
        plt.close()

        # Close the bits stream
        bits.close()

        print("Simulation completed.")

        return results
