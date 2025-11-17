import math
from abc import ABC, abstractmethod
from io import BytesIO
from queue import Queue
from typing import BinaryIO, Tuple

import numpy as np
from numpy.random import PCG64, Generator
from numpy.typing import NDArray


class IGenerator(ABC):
    @abstractmethod
    def generate_bits(self, num_bits: int) -> BinaryIO:
        pass


class RandomBitsGenerator(IGenerator):
    """
    Generates random bits using a specified random number generator.
    Uses numpy's Generator for random number generation.
    """

    def __init__(self, generator: Generator = Generator(PCG64())):
        self.generator = generator

    def generate_bits(self, num_bits: int) -> BinaryIO:
        """
        Generates a binary stream of random bits (0s and 1s).
        :param num_bits: Number of bits to generate.
        :return: The binary stream containing the generated bits.
        """
        # Calculate the number of bytes needed to store num_bits
        num_bytes = math.ceil(num_bits / 8)

        # Generate random bytes
        random_bytes = self.generator.bytes(num_bytes)

        # If num_bits is not a multiple of 8, we need to mask the last byte
        # to ensure we only have num_bits of random data.
        bits_to_keep = num_bits % 8
        if bits_to_keep > 0:
            # Create a mask to zero out the unused bits in the last byte
            # e.g., if we need 3 bits, mask is 0b11100000
            mask = (0xFF << (8 - bits_to_keep)) & 0xFF

            # Get the last byte, apply the mask, and keep the original part
            last_byte = random_bytes[-1] & mask
            random_bytes = random_bytes[:-1] + bytes([last_byte])

        bits_stream = BytesIO(random_bytes)
        # The stream is already at the beginning, but seek(0) is good practice
        # if you were to perform other operations before returning.
        bits_stream.seek(0)
        return bits_stream


class AdaptiveBitsGenerator(IGenerator):
    """Generate bits for adaptive modulation with per-subcarrier constellation orders.

    This generator creates the exact number of bits needed for adaptive modulation,
    where different subcarriers may use different constellation orders.

    Attributes:
        bits_per_subcarrier: Number of bits allocated to each subcarrier
        num_ofdm_symbols: Number of OFDM symbols to generate bits for
        generator: Random number generator instance
    """

    def __init__(
        self,
        bits_per_subcarrier: NDArray[np.int64],
        num_ofdm_symbols: int,
        generator: Generator = Generator(PCG64()),
    ):
        """Initialize adaptive bits generator.

        Args:
            bits_per_subcarrier: Array of bits per subcarrier (one value per subcarrier)
            num_ofdm_symbols: Number of OFDM symbols to generate bits for
            generator: Random number generator (default: PCG64-based Generator)

        Raises:
            ValueError: If bits_per_subcarrier is empty or num_ofdm_symbols is not positive
        """
        if len(bits_per_subcarrier) == 0:
            raise ValueError("bits_per_subcarrier cannot be empty")
        if num_ofdm_symbols <= 0:
            raise ValueError(f"num_ofdm_symbols must be positive, got {num_ofdm_symbols}")

        self.bits_per_subcarrier = np.array(bits_per_subcarrier, dtype=np.int64)
        self.num_ofdm_symbols = num_ofdm_symbols
        self.generator = generator

    def generate_bits(self, num_bits: int = 0) -> BinaryIO:
        """Generate random bits for adaptive modulation.

        Args:
            num_bits: Ignored (determined by bits_per_subcarrier and num_ofdm_symbols)

        Returns:
            Binary stream containing exactly the required number of bits
        """
        # Calculate total bits needed
        total_bits = self.get_total_bits()

        # Generate random bits
        num_bytes = math.ceil(total_bits / 8)
        random_bytes = self.generator.bytes(num_bytes)

        # Mask unused bits in last byte if needed
        bits_to_keep = total_bits % 8
        if bits_to_keep > 0:
            mask = (0xFF << (8 - bits_to_keep)) & 0xFF
            last_byte = random_bytes[-1] & mask
            random_bytes = random_bytes[:-1] + bytes([last_byte])

        bits_stream = BytesIO(random_bytes)
        bits_stream.seek(0)
        return bits_stream

    def get_total_bits(self) -> int:
        """Calculate total number of bits that will be generated.

        Returns:
            Total bits = sum(bits_per_subcarrier) * num_ofdm_symbols
        """
        return int(np.sum(self.bits_per_subcarrier) * self.num_ofdm_symbols)

    @staticmethod
    def calculate_requirements(
        constellation_orders: NDArray[np.int64], num_ofdm_symbols: int
    ) -> Tuple[int, NDArray[np.int64]]:
        """Calculate bit requirements for adaptive modulation.

        Static helper method to determine how many bits are needed given
        constellation orders per subcarrier.

        Args:
            constellation_orders: Array of constellation orders per subcarrier
            num_ofdm_symbols: Number of OFDM symbols to transmit

        Returns:
            Tuple of (total_bits_needed, bits_per_subcarrier)

        Example:
            >>> orders = np.array([4, 16, 64, 0])  # QPSK, 16-QAM, 64-QAM, no transmission
            >>> total, bits_per = AdaptiveBitsGenerator.calculate_requirements(orders, 1000)
            >>> print(total)  # (2 + 4 + 6 + 0) * 1000 = 12000
            12000
            >>> print(bits_per)
            [2 4 6 0]
        """
        # Calculate bits per subcarrier (log2 of order)
        bits_per_subcarrier = np.array(
            [int(np.log2(order)) if order > 0 else 0 for order in constellation_orders],
            dtype=np.int64,
        )

        # Calculate total bits
        total_bits = int(np.sum(bits_per_subcarrier) * num_ofdm_symbols)

        return total_bits, bits_per_subcarrier
