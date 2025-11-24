from abc import ABC, abstractmethod
from functools import cached_property
from io import BytesIO
from typing import BinaryIO, Dict, List, Tuple, Type, Union

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


class ISymbolClassifier(ABC):
    @abstractmethod
    def classify(
        self, constellation: NDArray[np.complex128], symbols: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        pass


class NNClassifier(ISymbolClassifier):
    def classify(
        self, constellation: NDArray[np.complex128], symbols: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        distances = np.abs(
            symbols[:, np.newaxis] - constellation[np.newaxis, :]
        )  # Broadcasted distance calculation
        nearest_indices = np.argmin(distances, axis=1)  # Indices of nearest constellation points
        return constellation[nearest_indices]


class IWordCoder(ABC):
    def __init__(self, bits_per_word: int):
        self.bits_per_word = bits_per_word

    @abstractmethod
    def encode(self, word: int) -> int:
        pass

    @abstractmethod
    def decode(self, coded_word: int) -> int:
        pass

    @abstractmethod
    def reorder_constellation(
        self, constellation: NDArray[np.complex128], constellation_name: str
    ) -> NDArray[np.complex128]:
        pass


class NoWordCoder(IWordCoder):
    @property
    def size(self) -> int:
        return 1 << self.bits_per_word  # 2^bits_per_word

    def encode(self, word: int) -> int:
        if not 0 <= word < self.size:
            raise ValueError(f"Word must be in range [0, {self.size})")
        return word

    def decode(self, coded_word: int) -> int:
        if not 0 <= coded_word < self.size:
            raise ValueError(f"Coded word must be in range [0, {self.size})")
        return coded_word

    def reorder_constellation(
        self, constellation: NDArray[np.complex128], constellation_name: str
    ) -> NDArray[np.complex128]:
        return constellation  # No reordering needed


class GrayWordCoder(IWordCoder):
    @property
    def size(self) -> int:
        return 1 << self.bits_per_word  # 2^bits_per_word

    @cached_property
    def gray_table(self) -> Dict[int, int]:
        return {i: i ^ (i >> 1) for i in range(self.size)}

    @cached_property
    def inverse_gray_table(self) -> Dict[int, int]:
        gray_table = self.gray_table
        return {v: k for k, v in gray_table.items()}

    def encode(self, word: int) -> int:
        if not 0 <= word < self.size:
            raise ValueError(f"Word must be in range [0, {self.size})")
        return self.gray_table[word]

    def decode(self, coded_word: int) -> int:
        if not 0 <= coded_word < self.size:
            raise ValueError(f"Gray word must be in range [0, {self.size})")
        return self.inverse_gray_table[coded_word]

    def reorder_constellation(
        self, constellation: NDArray[np.complex128], constellation_name: str
    ) -> NDArray[np.complex128]:
        """Reorder constellation points in a zig-zag manner to match Gray coding."""
        if QAMConstellationMapper.__name__ == constellation_name:
            reordered = np.zeros_like(constellation)
            row_length = int(np.sqrt(len(constellation)))
            for i in range(int(row_length)):
                start_index = i * int(row_length)
                end_index = (i + 1) * int(row_length)
                row = constellation[start_index:end_index]
                if i % 2 == 1:
                    row = row[::-1]  # Reverse for odd rows
                reordered[start_index:end_index] = row
            return reordered
        return constellation  # Default: no reordering


class IConstellationMapper(ABC):
    constellation: NDArray[np.complex128]
    constellation_map: Dict[Tuple[float, float], int]  # Maps constellation points to their indices

    def __init__(
        self,
        order: int,
        word_coder: Type[IWordCoder] = GrayWordCoder,
        classifier: Type[ISymbolClassifier] = NNClassifier,
    ):
        self.order = order
        self.word_coder = word_coder(bits_per_word=self.bits_per_symbol)
        self.classifier = classifier()

    @property
    @abstractmethod
    def constellation_name(self) -> str:
        pass

    @property
    @abstractmethod
    def bits_per_symbol(self) -> int:
        pass

    @abstractmethod
    def encode(self, bits: BinaryIO) -> NDArray[np.complex128]:
        pass

    @abstractmethod
    def decode(self, symbols: NDArray[np.complex128] | np.complex128) -> BinaryIO:
        pass

    @classmethod
    @abstractmethod
    def calculate_bit_loading_order(cls, ser: float, snr: float) -> int:
        pass


class QAMConstellationMapper(IConstellationMapper):
    """
    Maps bits to QAM constellation symbols and vice versa.

    Supports square QAM constellations with Gray coding.
    Normalizes constellation to unit average energy.
    """

    def __init__(
        self,
        order: int,
        word_coder: Type[IWordCoder] = GrayWordCoder,
        classifier: Type[ISymbolClassifier] = NNClassifier,
    ):
        super().__init__(order, word_coder, classifier)
        self.validate_order()
        self.constellation, self.constellation_map = self.generate_constellation()

    @property
    def constellation_name(self) -> str:
        return f"{self.order}-QAM"

    @property
    def bits_per_symbol(self) -> int:
        return int(np.log2(self.order))

    def validate_order(self) -> None:
        if int(np.sqrt(self.order)) ** 2 != self.order:
            raise ValueError("Order must be a perfect square (e.g., 4, 16, 64).")

    def generate_constellation(
        self,
    ) -> Tuple[NDArray[np.complex128], Dict[Tuple[float, float], int]]:
        """Generate normalized square QAM constellation."""
        points_per_side = int(np.sqrt(self.order))  # How many points on each axis

        # For this many points, levels are spaced by 2, centered around 0
        in_phase_levels = np.arange(-points_per_side + 1, points_per_side, 2)
        quadrature_levels = np.arange(-points_per_side + 1, points_per_side, 2)
        # The energy will be normalized later

        # Generate constellation points in natural binary order first
        binary_ordered_points = []

        # top to bottom (positive to negative imaginary)
        for q_level in quadrature_levels[::-1]:
            # left to right (negative to positive real)
            for i_level in in_phase_levels:
                binary_ordered_points.append(complex(i_level, q_level))

        # Now reorder according to the WordCoder (e.g., Gray coding)
        # We want constellation[<coded_index>] = point_at_binary_position
        constellation = np.zeros(self.order, dtype=np.complex128)
        for binary_index in range(self.order):
            coded_index = self.word_coder.encode(binary_index)
            constellation[binary_index] = binary_ordered_points[coded_index]

        constellation = self.word_coder.reorder_constellation(
            constellation, QAMConstellationMapper.__name__
        )

        # Normalize constellation to unit average power
        average_power = np.mean(np.abs(constellation) ** 2)
        constellation /= np.sqrt(average_power)

        # Use (real, imag) tuples as keys instead of complex numbers for better reliability
        return constellation, {
            (float(point.real), float(point.imag)): i for i, point in enumerate(constellation)
        }

    def encode(self, bits: Union[BinaryIO, List[int]]) -> NDArray[np.complex128]:
        """Encodes a stream of bits into QAM symbols."""
        # Read the bytes from the binary stream
        bits_list: List[int] = []
        if isinstance(bits, list):
            bits_list = bits
        else:
            bytes_data = bits.read()
            # Extract individual bits from bytes
            bits_list = []
            for byte in bytes_data:
                # Extract 8 bits from each byte, from MSB to LSB
                for i in range(7, -1, -1):
                    bits_list.append((byte >> i) & 1)

            if len(bits_list) % self.bits_per_symbol != 0:
                # Pad with zeros if bits are not a multiple of bits_per_symbol
                bits_list += [0] * (self.bits_per_symbol - len(bits_list) % self.bits_per_symbol)

        # Reshape bits into groups corresponding to symbols
        bit_chunks = np.array(bits_list).reshape(-1, self.bits_per_symbol)

        # Convert each chunk of bits to its corresponding integer index
        binary_indices = bit_chunks.dot(1 << np.arange(self.bits_per_symbol - 1, -1, -1))

        # Map indices to constellation symbols
        symbols = self.constellation[binary_indices]

        # Return the symbols as a 1D array
        return symbols

    def decode(self, symbols: NDArray[np.complex128] | np.complex128) -> BinaryIO:
        """Decodes QAM symbols back into a stream of bits."""
        if np.isscalar(symbols):
            symbols = np.array([symbols], dtype=np.complex128)
        else:
            symbols = np.asarray(symbols, dtype=np.complex128)

        # Classify received symbols to nearest constellation points
        classified_symbols = self.classifier.classify(self.constellation, symbols)

        # Map classified symbols back to their indices using tuple keys
        binary_indices = np.array(
            [
                self.constellation_map[(float(sym.real), float(sym.imag))]
                for sym in classified_symbols
            ]
        )

        # Convert binary indices back to bits
        bits_list = []
        for bi in binary_indices:
            bits = [(bi >> i) & 1 for i in range(self.bits_per_symbol - 1, -1, -1)]
            bits_list.extend(bits)

        # Pack bits into bytes (8 bits per byte)
        packed_bytes = []
        for i in range(0, len(bits_list), 8):
            if i + 8 <= len(bits_list):
                # We have a full byte
                byte_value = 0
                for j in range(8):
                    byte_value = (byte_value << 1) | bits_list[i + j]
                packed_bytes.append(byte_value)
            else:
                # For the last partial byte (if any), pad with zeros
                byte_value = 0
                remaining = len(bits_list) - i
                for j in range(remaining):
                    byte_value = (byte_value << 1) | bits_list[i + j]
                # Shift left to align with MSB
                byte_value = byte_value << (8 - remaining)
                packed_bytes.append(byte_value)

        # Return bits as a bytes object
        return BytesIO(bytes(packed_bytes))

    @classmethod
    def calculate_bit_loading_order(cls, ser: float, snr: float) -> int:
        print("=" * 20)
        # Calculate the inverse Q-function value for SER/2
        q_inv = norm.isf(ser / 4)
        print(f"Q-inv value for SER={ser}: {q_inv}")
        # Calculate the gap SNR using the formula for QAM
        gamma = (1 / 3) * (q_inv**2)
        print(f"Gap SNR (gamma) for SER={ser}: {gamma}")
        # Determine the maximum bits per symbol that can be supported
        bits_per_symbol = int(np.round(np.log2(1 + (snr / gamma))))
        print(f"Calculated bits per symbol for SER={ser}, SNR={snr}: {bits_per_symbol}")

        # bits_per_symbol must be even for square QAM
        if bits_per_symbol % 2 != 0:
            print("Bits per symbol is not even, reducing by 1 to make it even.")
            bits_per_symbol -= 1

        # if bits_per_symbol is negative or zero, return 0
        if bits_per_symbol <= 0:
            print("Bits per symbol is less than or equal to 0, returning 0.")
            return 0

        print(f"Final bits per symbol for QAM: {bits_per_symbol}")
        return 2**bits_per_symbol


class PSKConstellationMapper(IConstellationMapper):
    """
    Maps bits to PSK constellation symbols and vice versa.

    Supports M-ary Phase Shift Keying with Gray coding.
    All constellation points have unit amplitude and are equally spaced in phase.
    """

    def __init__(
        self,
        order: int,
        word_coder: Type[IWordCoder] = GrayWordCoder,
        classifier: Type[ISymbolClassifier] = NNClassifier,
    ):
        super().__init__(order, word_coder, classifier)
        self.validate_order()
        self.constellation, self.constellation_map = self.generate_constellation()

    @property
    def constellation_name(self) -> str:
        return f"{self.order}-PSK"

    @property
    def bits_per_symbol(self) -> int:
        return int(np.log2(self.order))

    def validate_order(self) -> None:
        """Validates that the constellation order is a power of 2."""
        bits_per_symbol = np.log2(self.order)
        if bits_per_symbol != int(bits_per_symbol) or self.order < 2:
            raise ValueError("PSK order must be a power of 2 (e.g., 2, 4, 8, 16).")

    def generate_constellation(
        self,
    ) -> Tuple[NDArray[np.complex128], Dict[Tuple[float, float], int]]:
        """Generate normalized PSK constellation with points on the unit circle."""
        # Generate constellation points in natural binary order first
        angles = 2 * np.pi * np.arange(self.order) / self.order
        binary_ordered_points = np.exp(1j * angles)

        # Now reorder according to the WordCoder (e.g., Gray coding)
        # We want constellation[<coded_index>] = point_at_binary_position
        constellation = np.zeros(self.order, dtype=np.complex128)
        for binary_index in range(self.order):
            coded_index = self.word_coder.encode(binary_index)
            constellation[coded_index] = binary_ordered_points[binary_index]

        constellation = self.word_coder.reorder_constellation(
            constellation, PSKConstellationMapper.__name__
        )

        # No need to normalize PSK as all points already have unit amplitude

        # Use (real, imag) tuples as keys instead of complex numbers for better reliability
        return constellation, {
            (float(point.real), float(point.imag)): i for i, point in enumerate(constellation)
        }

    def encode(self, bits: Union[BinaryIO, List[int]]) -> NDArray[np.complex128]:
        """Encodes a stream of bits into PSK symbols."""
        if isinstance(bits, list):
            bits_list: List[int] = bits
        else:
            # Read the bytes from the binary stream
            bytes_data = bits.read()

            # Extract individual bits from bytes
            bits_list = []
            for byte in bytes_data:
                # Extract 8 bits from each byte, from MSB to LSB
                for i in range(7, -1, -1):
                    bits_list.append((byte >> i) & 1)

            if len(bits_list) % self.bits_per_symbol != 0:
                # Pad with zeros if bits are not a multiple of bits_per_symbol
                bits_list += [0] * (self.bits_per_symbol - len(bits_list) % self.bits_per_symbol)

        # Reshape bits into groups corresponding to symbols
        bit_chunks = np.array(bits_list).reshape(-1, self.bits_per_symbol)

        # Convert each chunk of bits to its corresponding integer index
        binary_indices = bit_chunks.dot(1 << np.arange(self.bits_per_symbol - 1, -1, -1))

        # Map binary indices to constellation symbols
        symbols = self.constellation[binary_indices]

        # Return the symbols as a 1D array
        return symbols

    def decode(self, symbols: NDArray[np.complex128] | np.complex128) -> BinaryIO:
        """Decodes PSK symbols back into a stream of bits."""
        if np.isscalar(symbols):
            symbols = np.array([symbols], dtype=np.complex128)
        else:
            symbols = np.asarray(symbols, dtype=np.complex128)

        # Classify received symbols to nearest constellation points
        classified_symbols = self.classifier.classify(self.constellation, symbols)

        # Map classified symbols back to their indices using tuple keys
        binary_indices = np.array(
            [
                self.constellation_map[(float(sym.real), float(sym.imag))]
                for sym in classified_symbols
            ]
        )

        # Convert binary indices back to bits
        bits_list = []
        for bi in binary_indices:
            bits = [(bi >> i) & 1 for i in range(self.bits_per_symbol - 1, -1, -1)]
            bits_list.extend(bits)

        # Pack bits into bytes (8 bits per byte)
        packed_bytes = []
        for i in range(0, len(bits_list), 8):
            if i + 8 <= len(bits_list):
                # We have a full byte
                byte_value = 0
                for j in range(8):
                    byte_value = (byte_value << 1) | bits_list[i + j]
                packed_bytes.append(byte_value)
            else:
                # For the last partial byte (if any), pad with zeros
                byte_value = 0
                remaining = len(bits_list) - i
                for j in range(remaining):
                    byte_value = (byte_value << 1) | bits_list[i + j]
                # Shift left to align with MSB
                byte_value = byte_value << (8 - remaining)
                packed_bytes.append(byte_value)

        # Return bits as a bytes object
        return BytesIO(bytes(packed_bytes))

    @classmethod
    def calculate_bit_loading_order(cls, ser: float, snr: float) -> int:
        # Calculate the inverse Q-function value for SER/2
        q_inv = norm.isf(ser / 2)
        # Calculate the gap SNR using the formula for PSK
        gamma_star = (q_inv**2) / (2 * (np.pi**2))
        gamma = (np.sqrt(snr * gamma_star)) / (1 - np.sqrt(gamma_star / (snr + 1e-10)))

        # Determine the maximum bits per symbol that can be supported
        bits_per_symbol = int(np.floor(np.log2(1 + snr / (gamma + 1e-10)) + 1e-10))

        # bits_per_symbol must be at least 1 for PSK
        if bits_per_symbol <= 0:
            return 0

        return 2**bits_per_symbol
