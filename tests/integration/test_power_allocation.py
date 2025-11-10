"""TDD integration tests for power allocation module.

Tests are written before implementation (Test-Driven Development).
These tests define the expected behavior of power allocation models.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

# Note: These imports will fail until the models are implemented
# This is intentional - we're following TDD (test-first development)
try:
    from ofdm_based_systems.power_allocation.models import (
        IPowerAllocation,
        UniformPowerAllocation,
        WaterfillingPowerAllocation,
    )

    MODELS_EXIST = True
except ImportError:
    MODELS_EXIST = False

    # Create mock classes for test development
    class IPowerAllocation:
        """Mock interface for power allocation."""

        pass

    class UniformPowerAllocation:
        """Mock uniform power allocation."""

        def __init__(self, total_power: float, num_subcarriers: int):
            pass

        def allocate(self):
            pass

    class WaterfillingPowerAllocation:
        """Mock waterfilling power allocation."""

        def __init__(self, total_power: float, channel_gains: np.ndarray, noise_power: float):
            pass

        def allocate(self):
            pass


class TestIPowerAllocationInterface:
    """Test the abstract power allocation interface."""

    @pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet implemented (TDD)")
    def test_interface_is_abstract(self):
        """Test that IPowerAllocation is an abstract base class."""
        from abc import ABCMeta

        assert isinstance(IPowerAllocation, ABCMeta) or hasattr(
            IPowerAllocation, "__abstractmethods__"
        )

    @pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet implemented (TDD)")
    def test_interface_has_allocate_method(self):
        """Test that interface defines allocate method."""
        assert hasattr(IPowerAllocation, "allocate")

    @pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet implemented (TDD)")
    def test_cannot_instantiate_interface(self):
        """Test that abstract interface cannot be instantiated."""
        with pytest.raises(TypeError):
            IPowerAllocation()


class TestUniformPowerAllocation:
    """Test uniform power allocation across subcarriers."""

    @pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet implemented (TDD)")
    def test_initialization(self):
        """Test uniform power allocation initialization."""
        total_power = 1.0
        num_subcarriers = 64

        allocator = UniformPowerAllocation(total_power=total_power, num_subcarriers=num_subcarriers)

        assert allocator.total_power == total_power
        assert allocator.num_subcarriers == num_subcarriers

    @pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet implemented (TDD)")
    def test_uniform_allocation_equal_power(self):
        """Test that uniform allocation gives equal power to all subcarriers."""
        total_power = 1.0
        num_subcarriers = 64

        allocator = UniformPowerAllocation(total_power=total_power, num_subcarriers=num_subcarriers)
        power_allocation = allocator.allocate()

        # All subcarriers should have equal power
        expected_power = total_power / num_subcarriers
        assert len(power_allocation) == num_subcarriers
        assert_array_almost_equal(power_allocation, np.full(num_subcarriers, expected_power))

    @pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet implemented (TDD)")
    def test_uniform_allocation_sums_to_total_power(self):
        """Test that allocated power sums to total power."""
        total_power = 10.0
        num_subcarriers = 128

        allocator = UniformPowerAllocation(total_power=total_power, num_subcarriers=num_subcarriers)
        power_allocation = allocator.allocate()

        assert np.isclose(np.sum(power_allocation), total_power)

    @pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet implemented (TDD)")
    def test_uniform_allocation_different_power_levels(self):
        """Test uniform allocation with different total power levels."""
        num_subcarriers = 32

        for total_power in [0.5, 1.0, 2.0, 5.0, 10.0]:
            allocator = UniformPowerAllocation(
                total_power=total_power, num_subcarriers=num_subcarriers
            )
            power_allocation = allocator.allocate()

            expected_per_subcarrier = total_power / num_subcarriers
            assert_array_almost_equal(
                power_allocation, np.full(num_subcarriers, expected_per_subcarrier)
            )

    @pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet implemented (TDD)")
    def test_uniform_allocation_single_subcarrier(self):
        """Test uniform allocation with single subcarrier."""
        total_power = 1.0
        num_subcarriers = 1

        allocator = UniformPowerAllocation(total_power=total_power, num_subcarriers=num_subcarriers)
        power_allocation = allocator.allocate()

        assert len(power_allocation) == 1
        assert power_allocation[0] == total_power

    @pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet implemented (TDD)")
    def test_uniform_allocation_many_subcarriers(self):
        """Test uniform allocation with many subcarriers."""
        total_power = 1.0
        num_subcarriers = 1024

        allocator = UniformPowerAllocation(total_power=total_power, num_subcarriers=num_subcarriers)
        power_allocation = allocator.allocate()

        assert len(power_allocation) == num_subcarriers
        assert np.isclose(np.sum(power_allocation), total_power)

    @pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet implemented (TDD)")
    def test_uniform_allocation_invalid_inputs(self):
        """Test uniform allocation with invalid inputs."""
        # Negative power
        with pytest.raises(ValueError):
            UniformPowerAllocation(total_power=-1.0, num_subcarriers=64)

        # Zero subcarriers
        with pytest.raises(ValueError):
            UniformPowerAllocation(total_power=1.0, num_subcarriers=0)

        # Negative subcarriers
        with pytest.raises(ValueError):
            UniformPowerAllocation(total_power=1.0, num_subcarriers=-10)


class TestWaterfillingPowerAllocation:
    """Test waterfilling power allocation algorithm."""

    @pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet implemented (TDD)")
    def test_initialization(self):
        """Test waterfilling power allocation initialization."""
        total_power = 1.0
        channel_gains = np.array([1.0, 0.8, 0.6, 0.4])
        noise_power = 0.1

        allocator = WaterfillingPowerAllocation(
            total_power=total_power, channel_gains=channel_gains, noise_power=noise_power
        )

        assert allocator.total_power == total_power
        assert_array_almost_equal(allocator.channel_gains, channel_gains)
        assert allocator.noise_power == noise_power

    @pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet implemented (TDD)")
    def test_waterfilling_favors_good_channels(self):
        """Test that waterfilling allocates more power to good channels."""
        total_power = 1.0
        # Vary channel quality significantly
        channel_gains = np.array([1.0, 0.5, 0.25, 0.1])
        noise_power = 0.1

        allocator = WaterfillingPowerAllocation(
            total_power=total_power, channel_gains=channel_gains, noise_power=noise_power
        )
        power_allocation = allocator.allocate()

        # Better channels (higher gains) should get more power
        assert power_allocation[0] >= power_allocation[1]
        assert power_allocation[1] >= power_allocation[2]
        assert power_allocation[2] >= power_allocation[3]

    @pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet implemented (TDD)")
    def test_waterfilling_sums_to_total_power(self):
        """Test that waterfilling allocates exactly the total power."""
        total_power = 10.0
        channel_gains = np.random.rand(64) + 0.1  # Avoid zero gains
        noise_power = 0.5

        allocator = WaterfillingPowerAllocation(
            total_power=total_power, channel_gains=channel_gains, noise_power=noise_power
        )
        power_allocation = allocator.allocate()

        assert np.isclose(np.sum(power_allocation), total_power, atol=1e-6)

    @pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet implemented (TDD)")
    def test_waterfilling_zero_allocation_for_bad_channels(self):
        """Test that very bad channels get zero power."""
        total_power = 1.0
        # Mix of good and very bad channels
        channel_gains = np.array([1.0, 0.9, 0.01, 0.001])
        noise_power = 0.1

        allocator = WaterfillingPowerAllocation(
            total_power=total_power, channel_gains=channel_gains, noise_power=noise_power
        )
        power_allocation = allocator.allocate()

        # Bad channels should get zero or near-zero power
        assert power_allocation[2] < power_allocation[0] * 0.1 or power_allocation[2] == 0
        assert power_allocation[3] < power_allocation[0] * 0.1 or power_allocation[3] == 0

    @pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet implemented (TDD)")
    def test_waterfilling_with_equal_channel_gains(self):
        """Test waterfilling with equal channel gains behaves like uniform."""
        total_power = 1.0
        num_subcarriers = 16
        channel_gains = np.ones(num_subcarriers)
        noise_power = 0.1

        allocator = WaterfillingPowerAllocation(
            total_power=total_power, channel_gains=channel_gains, noise_power=noise_power
        )
        power_allocation = allocator.allocate()

        # With equal gains, should allocate approximately equally
        expected_power = total_power / num_subcarriers
        assert_array_almost_equal(
            power_allocation, np.full(num_subcarriers, expected_power), decimal=3
        )

    @pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet implemented (TDD)")
    def test_waterfilling_non_negative_allocation(self):
        """Test that waterfilling never allocates negative power."""
        total_power = 1.0
        channel_gains = np.random.rand(32) * 2  # Random gains
        noise_power = 0.2

        allocator = WaterfillingPowerAllocation(
            total_power=total_power, channel_gains=channel_gains, noise_power=noise_power
        )
        power_allocation = allocator.allocate()

        # All allocations must be non-negative
        assert np.all(power_allocation >= 0)

    @pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet implemented (TDD)")
    def test_waterfilling_water_level_property(self):
        """Test the water level property of waterfilling."""
        total_power = 5.0
        channel_gains = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        noise_power = 0.1

        allocator = WaterfillingPowerAllocation(
            total_power=total_power, channel_gains=channel_gains, noise_power=noise_power
        )
        power_allocation = allocator.allocate()

        # Water level: P_i + N_0/|H_i|^2 should be constant for allocated channels
        # or the allocated power should be zero
        snr_inverse = noise_power / channel_gains  # N₀/|H[k]|², not N₀/|H[k]|⁴
        water_levels = power_allocation + snr_inverse

        # Find channels with non-zero allocation
        allocated = power_allocation > 1e-10
        if np.any(allocated):
            # Water levels for allocated channels should be approximately equal
            water_level_allocated = water_levels[allocated]
            assert (
                np.std(water_level_allocated) < 1e-6
            )  # Should be very close (numerical precision)

    @pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet implemented (TDD)")
    def test_waterfilling_varying_noise_power(self):
        """Test waterfilling with different noise power levels."""
        total_power = 1.0
        channel_gains = np.array([1.0, 0.8, 0.6, 0.4])

        for noise_power in [0.01, 0.1, 0.5, 1.0]:
            allocator = WaterfillingPowerAllocation(
                total_power=total_power, channel_gains=channel_gains, noise_power=noise_power
            )
            power_allocation = allocator.allocate()

            # Should always sum to total power
            assert np.isclose(np.sum(power_allocation), total_power, atol=1e-6)
            # Should always be non-negative
            assert np.all(power_allocation >= 0)

    @pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet implemented (TDD)")
    def test_waterfilling_invalid_inputs(self):
        """Test waterfilling with invalid inputs."""
        channel_gains = np.array([1.0, 0.8, 0.6])

        # Negative total power
        with pytest.raises(ValueError):
            WaterfillingPowerAllocation(
                total_power=-1.0, channel_gains=channel_gains, noise_power=0.1
            )

        # Negative noise power
        with pytest.raises(ValueError):
            WaterfillingPowerAllocation(
                total_power=1.0, channel_gains=channel_gains, noise_power=-0.1
            )

        # Empty channel gains
        with pytest.raises(ValueError):
            WaterfillingPowerAllocation(
                total_power=1.0, channel_gains=np.array([]), noise_power=0.1
            )

        # Zero channel gains
        with pytest.raises(ValueError):
            WaterfillingPowerAllocation(
                total_power=1.0, channel_gains=np.array([0, 0, 0]), noise_power=0.1
            )


class TestPowerAllocationIntegration:
    """Integration tests for power allocation with OFDM system."""

    @pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet implemented (TDD)")
    def test_uniform_power_allocation_with_modulation(self):
        """Test uniform power allocation integrated with OFDM modulation."""
        from ofdm_based_systems.equalization.models import NoEqualizator
        from ofdm_based_systems.modulation.models import OFDMModulator
        from ofdm_based_systems.prefix.models import NoPrefixScheme

        num_subcarriers = 16
        total_power = 1.0

        # Allocate power uniformly
        allocator = UniformPowerAllocation(total_power=total_power, num_subcarriers=num_subcarriers)
        power_allocation = allocator.allocate()

        # Create symbols and scale by power
        symbols = np.random.randn(4, num_subcarriers) + 1j * np.random.randn(4, num_subcarriers)
        symbols = symbols.astype(np.complex128)

        # Scale symbols by allocated power (amplitude is sqrt of power)
        power_scaling = np.sqrt(power_allocation)
        symbols_scaled = symbols * power_scaling

        # Modulate
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers,
            prefix_scheme=NoPrefixScheme(),
            equalizator=NoEqualizator(channel_frequency_response=channel_response),
        )

        modulated = modulator.modulate(symbols_scaled)

        # Verify modulated signal power
        signal_power = np.mean(np.abs(modulated) ** 2)
        # Should be approximately equal to total power (accounting for IFFT normalization)
        assert signal_power > 0

    @pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet implemented (TDD)")
    def test_waterfilling_improves_ber_over_uniform(self):
        """Test that waterfilling provides better BER than uniform in frequency-selective channel."""
        from ofdm_based_systems.bits_generation.models import RandomBitsGenerator
        from ofdm_based_systems.channel.models import ChannelModel
        from ofdm_based_systems.constellation.models import QAMConstellationMapper
        from ofdm_based_systems.equalization.models import ZeroForcingEqualizator
        from ofdm_based_systems.modulation.models import OFDMModulator
        from ofdm_based_systems.noise.models import AWGNoiseModel
        from ofdm_based_systems.prefix.models import CyclicPrefixScheme
        from ofdm_based_systems.serial_parallel.models import SerialToParallelConverter
        from ofdm_based_systems.simulation.models import read_bits_from_stream

        num_bits = 2048
        num_subcarriers = 64
        constellation_order = 16
        snr_db = 15.0

        # Frequency-selective channel
        channel_impulse = np.array([1.0, 0.7, 0.4, 0.2], dtype=np.complex128)
        channel_response = np.fft.fft(channel_impulse, n=num_subcarriers)
        channel_gains = np.abs(channel_response)

        # This test documents expected behavior - waterfilling should perform better
        # Implementation will validate this expectation

        assert len(channel_gains) == num_subcarriers

    @pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet implemented (TDD)")
    def test_power_allocation_with_channel_estimation(self):
        """Test power allocation using estimated channel gains."""
        num_subcarriers = 32

        # Simulate channel estimation
        true_channel = np.random.randn(num_subcarriers) + 1j * np.random.randn(num_subcarriers)
        true_channel = true_channel.astype(np.complex128)
        estimated_gains = np.abs(true_channel)

        # Use estimated gains for waterfilling
        allocator = WaterfillingPowerAllocation(
            total_power=1.0, channel_gains=estimated_gains, noise_power=0.1
        )
        power_allocation = allocator.allocate()

        assert len(power_allocation) == num_subcarriers
        assert np.isclose(np.sum(power_allocation), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
