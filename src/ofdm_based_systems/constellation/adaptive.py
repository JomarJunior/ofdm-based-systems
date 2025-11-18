"""Adaptive constellation mapping for per-subcarrier modulation orders.

This module implements adaptive modulation where different subcarriers can use
different constellation orders based on their channel quality and allocated power.
"""

from io import BytesIO
from typing import BinaryIO, Dict, List, Tuple, Type, Union

import numpy as np
from numpy.typing import NDArray

from ofdm_based_systems.constellation.models import IConstellationMapper


class AdaptiveConstellationMapper(IConstellationMapper):
    """Adaptive constellation mapper supporting per-subcarrier modulation orders.

    This mapper allows different subcarriers to use different constellation orders,
    enabling capacity-optimized transmission over frequency-selective channels.
    Subcarriers with better channel conditions can use higher-order modulations
    while weaker subcarriers use lower orders or are disabled.

    The mapper groups subcarriers by constellation order for efficient encoding/decoding
    and supports both QAM and PSK modulation schemes through the base_mapper_class parameter.

    Attributes:
        constellation_orders: Array of constellation orders per subcarrier
        base_mapper_class: Class to instantiate for each unique order (QAM or PSK)
        num_subcarriers: Total number of subcarriers
        mappers: Dictionary grouping subcarriers by order with their mapper instances
        bits_per_subcarrier: Number of bits allocated to each subcarrier
        constellation: Composite constellation (all unique constellation points)
    """

    def __init__(
        self,
        constellation_orders: NDArray[np.int64],
        base_mapper_class: Type[IConstellationMapper],
        num_subcarriers: int,
    ):
        """Initialize adaptive constellation mapper.

        Args:
            constellation_orders: Array of constellation orders (one per subcarrier).
                Must be powers of 2. Zero indicates no transmission on that subcarrier.
            base_mapper_class: Constellation mapper class to use (e.g., QAMConstellationMapper, PSKConstellationMapper)
            num_subcarriers: Total number of subcarriers in the system

        Raises:
            ValueError: If constellation_orders length doesn't match num_subcarriers
        """
        if len(constellation_orders) != num_subcarriers:
            raise ValueError(
                f"constellation_orders length ({len(constellation_orders)}) "
                f"must match num_subcarriers ({num_subcarriers})"
            )

        self.constellation_orders = np.array(constellation_orders, dtype=np.int64)
        self.base_mapper_class = base_mapper_class
        self.num_subcarriers = num_subcarriers

        # Group subcarriers by constellation order
        # Key: order, Value: (list of subcarrier indices, mapper instance)
        self.mappers: Dict[int, Tuple[List[int], IConstellationMapper]] = {}

        unique_orders = np.unique(constellation_orders)
        for order in unique_orders:
            if order > 0:  # Skip zero-order (no transmission)
                indices = np.where(constellation_orders == order)[0].tolist()
                mapper = base_mapper_class(order=int(order))
                self.mappers[int(order)] = (indices, mapper)

        # Calculate bits per subcarrier
        self.bits_per_subcarrier = np.array(
            [int(np.log2(order)) if order > 0 else 0 for order in constellation_orders],
            dtype=np.int64,
        )

        # Build composite constellation from all unique constellation points
        all_points = []
        for order in unique_orders:
            if order > 0:
                mapper = self.mappers[int(order)][1]
                all_points.extend(mapper.constellation.tolist())
        self.constellation = np.unique(np.array(all_points, dtype=np.complex128))

        # Build constellation map (not really used in adaptive mode, but required by interface)
        self.constellation_map = {
            (float(point.real), float(point.imag)): i for i, point in enumerate(self.constellation)
        }

    @property
    def order(self) -> int:
        """Return the maximum constellation order used."""
        return int(np.max(self.constellation_orders))

    @property
    def constellation_name(self) -> str:
        """Return name describing the adaptive constellation configuration."""
        unique_orders = np.unique(self.constellation_orders[self.constellation_orders > 0])
        if len(unique_orders) == 0:
            return "No-Transmission"
        elif len(unique_orders) == 1:
            return f"{int(unique_orders[0])}-{self.base_mapper_class.__name__.replace('ConstellationMapper', '')}"
        else:
            return f"Adaptive-{int(unique_orders.min())}-to-{int(unique_orders.max())}-{self.base_mapper_class.__name__.replace('ConstellationMapper', '')}"

    @property
    def bits_per_symbol(self) -> int:
        """Return the maximum bits per symbol across all subcarriers."""
        return int(np.max(self.bits_per_subcarrier))

    def get_bits_per_subcarrier(self) -> NDArray[np.int64]:
        """Get the number of bits allocated to each subcarrier.

        Returns:
            Array of bits per subcarrier (length = num_subcarriers)
        """
        return self.bits_per_subcarrier

    def get_constellation_orders(self) -> NDArray[np.int64]:
        """Get the constellation order for each subcarrier.

        Returns:
            Array of constellation orders (length = num_subcarriers)
        """
        return self.constellation_orders

    def encode(self, bits: Union[BinaryIO, List[int]]) -> NDArray[np.complex128]:
        """Encode bits into symbols with per-subcarrier constellation orders.

        Distributes the input bits across subcarriers according to their allocated
        bit counts, then maps each subcarrier's bits to the appropriate constellation.

        Args:
            bits: Binary stream or list of bits to encode. Length must be multiple of sum(bits_per_subcarrier)

        Returns:
            Array of complex symbols (length = multiple of num_subcarriers)

        Raises:
            ValueError: If bits length is not compatible with bit allocation

        Example:
            If bits_per_subcarrier = [2, 4, 2, 0] (orders [4, 16, 4, 0]):
            - Input bits [0,1,1,0,1,1,0,1] will be split as:
              - Subcarrier 0: [0,1] -> QPSK symbol
              - Subcarrier 1: [1,0,1,1] -> 16-QAM symbol
              - Subcarrier 2: [0,1] -> QPSK symbol
              - Subcarrier 3: (no bits, zero symbol)
        """
        # Convert bits to list if BinaryIO
        bits_list: List[int] = []
        if isinstance(bits, list):
            bits_list = bits
        else:
            bytes_data = bits.read()
            for byte in bytes_data:
                for i in range(7, -1, -1):
                    bits_list.append((byte >> i) & 1)

        bits_per_symbol = int(np.sum(self.bits_per_subcarrier))

        if bits_per_symbol == 0:
            raise ValueError("No active subcarriers (all orders are zero)")

        if len(bits_list) % bits_per_symbol != 0:
            raise ValueError(
                f"Bits length ({len(bits_list)}) must be multiple of "
                f"bits_per_symbol ({bits_per_symbol})"
            )

        num_symbols = len(bits_list) // bits_per_symbol
        symbols = np.zeros((num_symbols, self.num_subcarriers), dtype=np.complex128)

        # Process each OFDM symbol
        for symbol_idx in range(num_symbols):
            bit_offset = symbol_idx * bits_per_symbol
            subcarrier_bit_offset = 0

            # Distribute bits to each subcarrier
            for subcarrier_idx in range(self.num_subcarriers):
                num_bits = self.bits_per_subcarrier[subcarrier_idx]

                if num_bits > 0:
                    # Extract bits for this subcarrier
                    start_bit = bit_offset + subcarrier_bit_offset
                    end_bit = start_bit + num_bits
                    subcarrier_bits = bits_list[start_bit:end_bit]

                    # Get the appropriate mapper and encode
                    order = int(self.constellation_orders[subcarrier_idx])
                    _, mapper = self.mappers[order]
                    symbols[symbol_idx, subcarrier_idx] = mapper.encode(subcarrier_bits)[0]  # type: ignore

                    subcarrier_bit_offset += num_bits
                # else: keep zero symbol for inactive subcarriers

        # Flatten to 1D array
        return symbols.flatten()

    def decode(self, symbols: Union[NDArray[np.complex128], np.complex128]) -> BinaryIO:
        """Decode symbols into bits using per-subcarrier constellation orders.

        Demaps each symbol to bits using the appropriate constellation mapper for
        that subcarrier, then concatenates all bits.

        Args:
            symbols: Array of complex symbols or single symbol (length = multiple of num_subcarriers)

        Returns:
            Decoded bits as BinaryIO stream

        Raises:
            ValueError: If symbols length is not multiple of num_subcarriers
        """
        # Handle scalar input
        if np.isscalar(symbols):
            symbols = np.array([symbols], dtype=np.complex128)
        else:
            symbols = np.asarray(symbols, dtype=np.complex128)

        if len(symbols) % self.num_subcarriers != 0:
            raise ValueError(
                f"Symbols length ({len(symbols)}) must be multiple of "
                f"num_subcarriers ({self.num_subcarriers})"
            )

        # Reshape to (num_ofdm_symbols, num_subcarriers)
        symbols_2d = symbols.reshape(-1, self.num_subcarriers)
        num_ofdm_symbols = symbols_2d.shape[0]

        all_bits = []

        # Process each OFDM symbol
        for symbol_idx in range(num_ofdm_symbols):
            for subcarrier_idx in range(self.num_subcarriers):
                num_bits = self.bits_per_subcarrier[subcarrier_idx]

                if num_bits > 0:
                    # Get the appropriate mapper and decode
                    order = int(self.constellation_orders[subcarrier_idx])
                    _, mapper = self.mappers[order]
                    symbol = symbols_2d[symbol_idx, subcarrier_idx]

                    # Decode to bytes then extract bits
                    decoded_stream = mapper.decode(np.array([symbol]))
                    decoded_bytes = decoded_stream.read()
                    # Convert bytes to bits (take only the required number of bits)
                    all_bits.extend(
                        [(int(byte) >> (7 - i)) & 1 for byte in decoded_bytes for i in range(8)][
                            :num_bits
                        ]
                    )

        # Convert bits to bytes
        result_bytes = bytearray()
        for i in range(0, len(all_bits), 8):
            byte_bits = all_bits[i : i + 8]
            if len(byte_bits) == 8:
                byte_value = sum(bit << (7 - j) for j, bit in enumerate(byte_bits))
                result_bytes.append(byte_value)

        return BytesIO(bytes(result_bytes))
    
    def calculate_bit_loading_order(self, ser: float, snr: float) -> int:
        raise NotImplementedError("This method is not implemented in AdaptiveConstellationMapper.")


def calculate_constellation_orders(
    capacity: NDArray[np.float64],
    min_order: int,
    max_order: int,
    scaling_factor: float,
    base_mapper_class: Type[IConstellationMapper],
) -> NDArray[np.int64]:
    """Calculate constellation orders based on per-subcarrier capacity.

    Uses Shannon capacity to determine appropriate constellation order for each
    subcarrier, enforcing min/max bounds and ensuring orders are powers of 2.

    Args:
        capacity: Shannon capacity per subcarrier (bits per channel use)
        min_order: Minimum allowed constellation order (power of 2)
        max_order: Maximum allowed constellation order (power of 2)
        scaling_factor: Multiplier for capacity (for empirical BER adjustment)
        base_mapper_class: Constellation mapper class (QAM requires even bits, PSK allows any)

    Returns:
        Array of constellation orders (powers of 2) per subcarrier

    Algorithm:
        1. Scale capacity by scaling_factor
        2. Clip to log2(min_order) and log2(max_order) range
        3. For QAM: round down to nearest even number (QAM requires even bits)
        4. For PSK: round down to nearest integer
        5. Convert to orders: 2^bits
        6. Set very low capacity subcarriers to 0 (no transmission)

    Example:
        capacity = [8.5, 6.2, 3.1, 1.5] bits/symbol
        min_order = 4, max_order = 256, scaling = 1.0
        For QAM: bits = [8, 6, 2, 0] -> orders = [256, 64, 4, 0]
    """
    from ofdm_based_systems.constellation.models import QAMConstellationMapper

    # Scale capacity
    bits_per_symbol = capacity * scaling_factor

    # Clip to valid range
    min_bits = np.log2(min_order)
    max_bits = np.log2(max_order)
    bits_per_symbol = np.clip(bits_per_symbol, 0, max_bits)

    # Force even bits for QAM (requires I and Q symmetry)
    if base_mapper_class == QAMConstellationMapper:
        bits_per_symbol = bits_per_symbol // 2 * 2
    else:
        # PSK can use any number of bits
        bits_per_symbol = np.floor(bits_per_symbol)

    # Set subcarriers below minimum to zero (no transmission)
    bits_per_symbol = np.where(bits_per_symbol < min_bits, 0, bits_per_symbol)

    # Convert to constellation orders
    orders = np.where(bits_per_symbol > 0, 2**bits_per_symbol, 0).astype(np.int64)

    return orders