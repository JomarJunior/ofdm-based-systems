"""Integration tests for adaptive modulation functionality.

Tests the adaptive modulation components including capacity-based
constellation order selection and adaptive bits generation.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from ofdm_based_systems.bits_generation.models import AdaptiveBitsGenerator
from ofdm_based_systems.constellation.adaptive import (
    AdaptiveConstellationMapper,
    calculate_constellation_orders,
)
from ofdm_based_systems.constellation.models import (
    PSKConstellationMapper,
    QAMConstellationMapper,
)


class TestAdaptiveConstellationMapper:
    """Test adaptive constellation mapper functionality."""

    def test_adaptive_mapper_initialization(self):
        """Test that adaptive mapper initializes correctly with per-subcarrier orders."""
        num_subcarriers = 8
        constellation_orders = np.array([4, 4, 16, 16, 64, 64, 256, 256], dtype=np.int64)

        mapper = AdaptiveConstellationMapper(
            constellation_orders=constellation_orders,
            base_mapper_class=QAMConstellationMapper,
            num_subcarriers=num_subcarriers,
        )

        assert mapper.num_subcarriers == num_subcarriers
        assert_array_equal(mapper.constellation_orders, constellation_orders)
        # Verify mapper can return constellation properties
        assert hasattr(mapper, "constellation")
        assert hasattr(mapper, "bits_per_symbol")

    def test_adaptive_mapper_bits_per_subcarrier(self):
        """Test that bits per subcarrier are calculated correctly."""
        constellation_orders = np.array([4, 16, 64, 256, 4, 16, 64, 256], dtype=np.int64)

        mapper = AdaptiveConstellationMapper(
            constellation_orders=constellation_orders,
            base_mapper_class=QAMConstellationMapper,
            num_subcarriers=8,
        )

        expected_bits = np.array([2, 4, 6, 8, 2, 4, 6, 8], dtype=np.int64)
        assert_array_equal(mapper.get_bits_per_subcarrier(), expected_bits)

    def test_adaptive_mapper_encode(self):
        """Test that adaptive mapper can encode bits."""
        constellation_orders = np.array([4, 16, 64, 16, 4, 16, 64, 16], dtype=np.int64)
        num_subcarriers = len(constellation_orders)

        mapper = AdaptiveConstellationMapper(
            constellation_orders=constellation_orders,
            base_mapper_class=QAMConstellationMapper,
            num_subcarriers=num_subcarriers,
        )

        # Generate bits: sum of bits per subcarrier
        bits_per_subcarrier = mapper.get_bits_per_subcarrier()
        total_bits = int(np.sum(bits_per_subcarrier))
        bits_array = np.random.randint(0, 2, total_bits, dtype=np.int8)

        # Encode
        from io import BytesIO

        bits_stream = BytesIO()
        bits_stream.write(bytes(bits_array))
        bits_stream.seek(0)

        symbols = mapper.encode(bits_stream)

        # Should return symbols
        assert len(symbols) > 0
        assert symbols.dtype == np.complex128

    def test_adaptive_mapper_with_zero_orders(self):
        """Test that mapper handles subcarriers with zero order (inactive)."""
        # Some subcarriers have zero order (will be skipped)
        constellation_orders = np.array([4, 0, 16, 0, 64, 0, 256, 0], dtype=np.int64)

        mapper = AdaptiveConstellationMapper(
            constellation_orders=constellation_orders,
            base_mapper_class=QAMConstellationMapper,
            num_subcarriers=8,
        )

        bits_per_subcarrier = mapper.get_bits_per_subcarrier()
        expected = np.array([2, 0, 4, 0, 6, 0, 8, 0], dtype=np.int64)
        assert_array_equal(bits_per_subcarrier, expected)


class TestCalculateConstellationOrders:
    """Test constellation order calculation from capacity."""

    def test_calculate_orders_basic(self):
        """Test basic constellation order calculation."""
        # Capacities that should map to specific orders
        capacity = np.array([2.0, 4.0, 6.0, 8.0])  # bits/symbol

        orders = calculate_constellation_orders(
            capacity=capacity,
            min_order=4,
            max_order=256,
            scaling_factor=1.0,
            base_mapper_class=QAMConstellationMapper,
        )

        # Expected: 4-QAM, 16-QAM, 64-QAM, 256-QAM
        expected = np.array([4, 16, 64, 256], dtype=np.int64)
        assert_array_equal(orders, expected)

    def test_calculate_orders_with_scaling(self):
        """Test that scaling factor affects order selection."""
        capacity = np.array([4.0, 4.0, 4.0, 4.0])

        # Without scaling: should give 16-QAM
        orders_no_scale = calculate_constellation_orders(
            capacity=capacity,
            min_order=4,
            max_order=256,
            scaling_factor=1.0,
            base_mapper_class=QAMConstellationMapper,
        )
        assert_array_equal(orders_no_scale, np.array([16, 16, 16, 16], dtype=np.int64))

        # With conservative scaling: should give lower order
        orders_conservative = calculate_constellation_orders(
            capacity=capacity,
            min_order=4,
            max_order=256,
            scaling_factor=0.7,
            base_mapper_class=QAMConstellationMapper,
        )
        # 4.0 * 0.7 = 2.8 bits -> rounds to 2 bits -> 4-QAM
        assert_array_equal(orders_conservative, np.array([4, 4, 4, 4], dtype=np.int64))

    def test_calculate_orders_clamping(self):
        """Test that orders are clamped to min/max bounds."""
        # Very low and very high capacities
        capacity = np.array([0.5, 1.0, 10.0, 12.0])

        orders = calculate_constellation_orders(
            capacity=capacity,
            min_order=4,
            max_order=256,
            scaling_factor=1.0,
            base_mapper_class=QAMConstellationMapper,
        )

        # All should be within [4, 256] or 0 for inactive
        active_orders = orders[orders > 0]
        assert np.all(active_orders >= 4)
        assert np.all(active_orders <= 256)
        # High capacities clamped to max
        assert orders[2] == 256
        assert orders[3] == 256

    def test_calculate_orders_negative_capacity(self):
        """Test that negative capacity results in zero order (inactive subcarrier)."""
        capacity = np.array([2.0, -1.0, 4.0, -0.5])

        orders = calculate_constellation_orders(
            capacity=capacity,
            min_order=4,
            max_order=256,
            scaling_factor=1.0,
            base_mapper_class=QAMConstellationMapper,
        )

        # Negative capacity -> order 0
        assert orders[1] == 0
        assert orders[3] == 0
        # Positive capacities work normally
        assert orders[0] == 4
        assert orders[2] == 16


class TestAdaptiveBitsGenerator:
    """Test adaptive bits generator functionality."""

    def test_adaptive_bits_generator_initialization(self):
        """Test that adaptive bits generator initializes correctly."""
        bits_per_subcarrier = np.array([2, 4, 6, 8], dtype=np.int64)
        num_ofdm_symbols = 10

        generator = AdaptiveBitsGenerator(
            bits_per_subcarrier=bits_per_subcarrier, num_ofdm_symbols=num_ofdm_symbols
        )

        expected_total = (2 + 4 + 6 + 8) * 10  # 200 bits
        assert generator.get_total_bits() == expected_total

    def test_adaptive_bits_generator_generates_correct_amount(self):
        """Test that generator creates exact number of bits."""
        bits_per_subcarrier = np.array([2, 2, 4, 4, 6, 6], dtype=np.int64)
        num_ofdm_symbols = 5

        generator = AdaptiveBitsGenerator(
            bits_per_subcarrier=bits_per_subcarrier, num_ofdm_symbols=num_ofdm_symbols
        )

        bits = generator.generate_bits()
        from ofdm_based_systems.simulation.models import read_bits_from_stream

        bits_list = read_bits_from_stream(bits)

        expected_total = (2 + 2 + 4 + 4 + 6 + 6) * 5  # 120 bits
        assert len(bits_list) == expected_total

    def test_adaptive_bits_generator_calculate_requirements(self):
        """Test static helper for calculating bit requirements."""
        constellation_orders = np.array([4, 16, 64, 256], dtype=np.int64)
        num_ofdm_symbols = 8

        # Returns (total_bits, bits_per_subcarrier)
        total_bits, bits_per_subcarrier = AdaptiveBitsGenerator.calculate_requirements(
            constellation_orders, num_ofdm_symbols
        )

        expected_bits_per_subcarrier = np.array([2, 4, 6, 8], dtype=np.int64)
        expected_total = (2 + 4 + 6 + 8) * 8  # 160 bits

        assert total_bits == expected_total
        assert_array_equal(bits_per_subcarrier, expected_bits_per_subcarrier)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
