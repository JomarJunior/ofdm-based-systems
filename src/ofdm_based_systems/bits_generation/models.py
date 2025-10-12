from abc import ABC, abstractmethod
from io import BytesIO
import math
from queue import Queue
from typing import BinaryIO

from numpy.random import Generator, PCG64


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
