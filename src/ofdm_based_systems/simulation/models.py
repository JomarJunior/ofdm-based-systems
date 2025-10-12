import re
from typing import BinaryIO, List, Optional
import numpy as np

from numpy.typing import NDArray
from ofdm_based_systems.bits_generation.models import IGenerator, RandomBitsGenerator
from ofdm_based_systems.channel.models import ChannelModel
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
        "QAM": QAMConstellationMapper,
        "PSK": PSKConstellationMapper,
    }

    MODULATOR_SCHEME_MAPPERS = {
        "OFDM": OFDMModulator,
        "SC-OFDM": SingleCarrierOFDMModulator,
    }

    PREFIX_SCHEME_MAPPERS = {
        "None": NoPrefixScheme,
        "CP": CyclicPrefixScheme,
        "ZP": ZeroPaddingPrefixScheme,
    }

    EQUALIZATOR_SCHEME_MAPPERS = {
        "None": NoEqualizator,
        "ZF": ZeroForcingEqualizator,
        "MMSE": MMSEEqualizator,
    }

    NOISE_SCHEME_MAPPERS = {
        "AWGN": AWGNoiseModel,
        "None": NoNoiseModel,
    }

    def __init__(
        self,
        num_bits: Optional[int],
        num_symbols: Optional[int],
        num_subcarriers: int,
        constellation_order: int = 64,
        constellation_scheme: str = "QAM",
        modulator_type: str = "OFDM",
        prefix_scheme: str = "ZP",
        equalizator_type: str = "MMSE",
        snr_db: float = 0.0,
        noise_scheme: str = "AWGN",
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
        self.equalizator_type = equalizator_type
        self.snr_db = snr_db
        self.noise_scheme = noise_scheme

    def run(self):
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
        channel_impulse_response: NDArray[np.complex128] = np.array(
            [
                (7.767824138452235072e-01 + 4.560896742466611919e-01j),
                (-6.669848996328063551e-02 + 2.839935704583463338e-01j),
                (1.398968327715586490e-01 - 1.591963958343969865e-01j),
                (2.229949514514480494e-02 + 2.409945439452868821e-01j),
            ],
            dtype=np.complex128,
        )
        channel: ChannelModel = ChannelModel(
            impulse_response=channel_impulse_response, snr_db=self.snr_db, noise_model=noise_model
        )

        # Prefix Scheme
        prefix_scheme = self.PREFIX_SCHEME_MAPPERS.get(self.prefix_scheme, NoPrefixScheme)(
            prefix_length=channel.order if self.prefix_scheme != "None" else 0
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

        print("=" * 50)
        print("OFDM-based Modulation")
        print("=" * 50)
        modulated_signal = modulator.modulate(parallel_data)

        print(f"Modulated Signal Shape: {modulated_signal.shape}")

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

        print("=" * 50)
        print("Parallel to Serial Conversion of Demodulated Data...")
        print("=" * 50)
        demodulated_serial_data = serial_to_parallel_converter.to_serial(demodulated_data)
        print(f"Demodulated Serial Data Shape: {demodulated_serial_data.shape}")

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

        print("Simulation completed.")
