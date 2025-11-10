"""
Unit tests for modulation models.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

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
from ofdm_based_systems.prefix.models import (
    CyclicPrefixScheme,
    NoPrefixScheme,
    ZeroPaddingPrefixScheme,
)


class TestOFDMModulatorInitialization:
    """Test OFDMModulator initialization."""

    def test_init_with_basic_parameters(self):
        """Test initialization with basic parameters."""
        num_subcarriers = 64
        prefix_scheme = NoPrefixScheme()
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        assert modulator.num_subcarriers == num_subcarriers
        assert modulator.prefix_scheme is prefix_scheme
        assert modulator.equalizator is equalizator

    def test_init_with_cyclic_prefix(self):
        """Test initialization with cyclic prefix scheme."""
        num_subcarriers = 128
        prefix_length = 16
        prefix_scheme = CyclicPrefixScheme(prefix_length=prefix_length)
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        assert modulator.prefix_scheme.prefix_length == prefix_length

    def test_init_with_zero_forcing_equalizator(self):
        """Test initialization with ZF equalizer."""
        num_subcarriers = 64
        prefix_scheme = NoPrefixScheme()
        channel_response = np.random.randn(num_subcarriers) + 1j * np.random.randn(num_subcarriers)
        channel_response = channel_response.astype(np.complex128)
        equalizator = ZeroForcingEqualizator(channel_frequency_response=channel_response)

        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        assert isinstance(modulator.equalizator, ZeroForcingEqualizator)


class TestOFDMModulatorModulate:
    """Test OFDMModulator modulate method."""

    def test_modulate_basic_symbols(self):
        """Test modulation with basic symbols."""
        num_subcarriers = 4
        prefix_scheme = NoPrefixScheme()
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        # Single OFDM symbol
        symbols = np.array([[1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j]], dtype=np.complex128)

        modulated = modulator.modulate(symbols)

        # Should return time domain signal
        assert modulated.shape == symbols.shape

    def test_modulate_multiple_ofdm_symbols(self):
        """Test modulation with multiple OFDM symbols."""
        num_subcarriers = 8
        num_symbols = 5
        prefix_scheme = NoPrefixScheme()
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        symbols = np.random.randn(num_symbols, num_subcarriers) + 1j * np.random.randn(
            num_symbols, num_subcarriers
        )
        symbols = symbols.astype(np.complex128)

        modulated = modulator.modulate(symbols)

        assert modulated.shape == (num_symbols, num_subcarriers)

    def test_modulate_with_cyclic_prefix(self):
        """Test modulation with cyclic prefix."""
        num_subcarriers = 8
        prefix_length = 2
        prefix_scheme = CyclicPrefixScheme(prefix_length=prefix_length)
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        symbols = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=np.complex128)
        modulated = modulator.modulate(symbols)

        # Should have prefix added
        assert modulated.shape[1] == num_subcarriers + prefix_length

    def test_modulate_wrong_number_of_subcarriers_raises_error(self):
        """Test that wrong number of subcarriers raises ValueError."""
        num_subcarriers = 8
        prefix_scheme = NoPrefixScheme()
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        # Wrong number of subcarriers
        symbols = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.complex128)

        with pytest.raises(ValueError, match="Number of symbols must be"):
            modulator.modulate(symbols)

    def test_modulate_ifft_orthogonality(self):
        """Test that IFFT produces orthogonal time domain signals."""
        num_subcarriers = 16
        prefix_scheme = NoPrefixScheme()
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        # Create symbols on different subcarriers
        symbols1 = np.zeros((1, num_subcarriers), dtype=np.complex128)
        symbols1[0, 0] = 1.0

        symbols2 = np.zeros((1, num_subcarriers), dtype=np.complex128)
        symbols2[0, 1] = 1.0

        modulated1 = modulator.modulate(symbols1)
        modulated2 = modulator.modulate(symbols2)

        # Orthogonality check
        correlation = np.sum(modulated1 * np.conj(modulated2))
        assert np.abs(correlation) < 1e-10

    def test_modulate_power_preservation(self):
        """Test that modulation preserves signal power (with ortho norm)."""
        num_subcarriers = 64
        prefix_scheme = NoPrefixScheme()
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        symbols = np.random.randn(10, num_subcarriers) + 1j * np.random.randn(10, num_subcarriers)
        symbols = symbols.astype(np.complex128)

        input_power = np.mean(np.abs(symbols) ** 2)
        modulated = modulator.modulate(symbols)
        output_power = np.mean(np.abs(modulated) ** 2)

        # With ortho normalization, power should be preserved
        assert np.isclose(input_power, output_power, rtol=1e-5)


class TestOFDMModulatorDemodulate:
    """Test OFDMModulator demodulate method."""

    def test_demodulate_basic_signal(self):
        """Test demodulation of basic time domain signal."""
        num_subcarriers = 4
        prefix_scheme = NoPrefixScheme()
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        # Time domain signal
        time_signal = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.complex128)

        demodulated = modulator.demodulate(time_signal)

        assert demodulated.shape == (1, num_subcarriers)

    def test_demodulate_with_cyclic_prefix(self):
        """Test demodulation with cyclic prefix removal."""
        num_subcarriers = 8
        prefix_length = 2
        prefix_scheme = CyclicPrefixScheme(prefix_length=prefix_length)
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        # Time domain signal with prefix
        time_signal = np.random.randn(3, num_subcarriers + prefix_length) + 1j * np.random.randn(
            3, num_subcarriers + prefix_length
        )
        time_signal = time_signal.astype(np.complex128)

        demodulated = modulator.demodulate(time_signal)

        # Should remove prefix and return frequency domain
        assert demodulated.shape == (3, num_subcarriers)

    def test_demodulate_applies_equalization(self):
        """Test that demodulation applies equalization."""
        num_subcarriers = 4
        prefix_scheme = NoPrefixScheme()
        channel_response = np.array([2.0, 0.5, 1.0, 4.0], dtype=np.complex128)
        equalizator = ZeroForcingEqualizator(channel_frequency_response=channel_response)

        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        # Simulate received signal affected by channel
        freq_domain = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.complex128)
        time_domain = np.fft.ifft(freq_domain, axis=1, norm="ortho")
        received = np.fft.fft(time_domain, axis=1, norm="ortho") * channel_response
        time_received = np.fft.ifft(received, axis=1, norm="ortho")

        demodulated = modulator.demodulate(time_received)

        # Should equalize and recover original
        assert_array_almost_equal(demodulated, freq_domain, decimal=10)


class TestOFDMModulatorRoundTrip:
    """Test OFDM modulation-demodulation round trip."""

    def test_modulate_demodulate_perfect_channel(self):
        """Test modulate-demodulate round trip with perfect channel."""
        num_subcarriers = 16
        prefix_scheme = NoPrefixScheme()
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        original_symbols = np.random.randn(5, num_subcarriers) + 1j * np.random.randn(
            5, num_subcarriers
        )
        original_symbols = original_symbols.astype(np.complex128)

        modulated = modulator.modulate(original_symbols)
        demodulated = modulator.demodulate(modulated)

        # Should recover original symbols
        assert_array_almost_equal(demodulated, original_symbols, decimal=10)

    def test_modulate_demodulate_with_cyclic_prefix(self):
        """Test round trip with cyclic prefix."""
        num_subcarriers = 32
        prefix_length = 8
        prefix_scheme = CyclicPrefixScheme(prefix_length=prefix_length)
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        original_symbols = np.random.randn(10, num_subcarriers) + 1j * np.random.randn(
            10, num_subcarriers
        )
        original_symbols = original_symbols.astype(np.complex128)

        modulated = modulator.modulate(original_symbols)
        demodulated = modulator.demodulate(modulated)

        assert_array_almost_equal(demodulated, original_symbols, decimal=10)

    def test_modulate_demodulate_with_channel_equalization(self):
        """Test round trip with channel and equalization."""
        num_subcarriers = 8
        prefix_scheme = NoPrefixScheme()
        channel_response = np.array(
            [1.0, 0.8, 0.6, 0.9, 0.7, 0.95, 0.85, 0.75], dtype=np.complex128
        )
        equalizator = ZeroForcingEqualizator(channel_frequency_response=channel_response)

        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        original_symbols = np.random.randn(3, num_subcarriers) + 1j * np.random.randn(
            3, num_subcarriers
        )
        original_symbols = original_symbols.astype(np.complex128)

        # Modulate
        modulated = modulator.modulate(original_symbols)

        # Apply channel in frequency domain
        freq_domain = np.fft.fft(modulated, axis=1, norm="ortho")
        freq_domain *= channel_response
        time_with_channel = np.fft.ifft(freq_domain, axis=1, norm="ortho")

        # Demodulate (includes equalization)
        demodulated = modulator.demodulate(time_with_channel)

        # Should recover original symbols
        assert_array_almost_equal(demodulated, original_symbols, decimal=10)


class TestSingleCarrierOFDMModulatorInitialization:
    """Test SingleCarrierOFDMModulator initialization."""

    def test_init_with_basic_parameters(self):
        """Test initialization with basic parameters."""
        num_subcarriers = 64
        prefix_scheme = NoPrefixScheme()
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = SingleCarrierOFDMModulator(
            prefix_scheme=prefix_scheme, equalizator=equalizator, num_subcarriers=num_subcarriers
        )

        assert modulator.num_subcarriers == num_subcarriers
        assert modulator.prefix_scheme is prefix_scheme
        assert modulator.equalizator is equalizator

    def test_init_with_cyclic_prefix(self):
        """Test initialization with cyclic prefix scheme."""
        num_subcarriers = 128
        prefix_length = 16
        prefix_scheme = CyclicPrefixScheme(prefix_length=prefix_length)
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = SingleCarrierOFDMModulator(
            prefix_scheme=prefix_scheme, equalizator=equalizator, num_subcarriers=num_subcarriers
        )

        assert modulator.prefix_scheme.prefix_length == prefix_length


class TestSingleCarrierOFDMModulatorModulate:
    """Test SingleCarrierOFDMModulator modulate method."""

    def test_modulate_basic_symbols(self):
        """Test SC-OFDM modulation with basic symbols."""
        num_subcarriers = 8
        prefix_scheme = NoPrefixScheme()
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = SingleCarrierOFDMModulator(
            prefix_scheme=prefix_scheme, equalizator=equalizator, num_subcarriers=num_subcarriers
        )

        # Time domain symbols (already modulated)
        symbols = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=np.complex128)

        modulated = modulator.modulate(symbols)

        # Without prefix, should be same shape
        assert modulated.shape == symbols.shape

    def test_modulate_with_cyclic_prefix(self):
        """Test SC-OFDM modulation adds cyclic prefix."""
        num_subcarriers = 8
        prefix_length = 2
        prefix_scheme = CyclicPrefixScheme(prefix_length=prefix_length)
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = SingleCarrierOFDMModulator(
            prefix_scheme=prefix_scheme, equalizator=equalizator, num_subcarriers=num_subcarriers
        )

        symbols = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=np.complex128)
        modulated = modulator.modulate(symbols)

        # Should add prefix
        assert modulated.shape[1] == num_subcarriers + prefix_length

        # Check cyclic prefix is correctly added
        expected_prefix = symbols[0, -prefix_length:]
        actual_prefix = modulated[0, :prefix_length]
        assert_array_equal(actual_prefix, expected_prefix)

    def test_modulate_multiple_blocks(self):
        """Test SC-OFDM modulation with multiple blocks."""
        num_subcarriers = 16
        num_blocks = 5
        prefix_scheme = NoPrefixScheme()
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = SingleCarrierOFDMModulator(
            prefix_scheme=prefix_scheme, equalizator=equalizator, num_subcarriers=num_subcarriers
        )

        symbols = np.random.randn(num_blocks, num_subcarriers) + 1j * np.random.randn(
            num_blocks, num_subcarriers
        )
        symbols = symbols.astype(np.complex128)

        modulated = modulator.modulate(symbols)

        assert modulated.shape == (num_blocks, num_subcarriers)


class TestSingleCarrierOFDMModulatorDemodulate:
    """Test SingleCarrierOFDMModulator demodulate method."""

    def test_demodulate_basic_signal(self):
        """Test SC-OFDM demodulation of basic signal."""
        num_subcarriers = 8
        prefix_scheme = NoPrefixScheme()
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = SingleCarrierOFDMModulator(
            prefix_scheme=prefix_scheme, equalizator=equalizator, num_subcarriers=num_subcarriers
        )

        # Received time domain signal
        received = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=np.complex128)

        demodulated = modulator.demodulate(received)

        # Should return time domain after FFT-equalize-IFFT
        assert demodulated.shape == received.shape

    def test_demodulate_with_cyclic_prefix(self):
        """Test SC-OFDM demodulation with cyclic prefix removal."""
        num_subcarriers = 8
        prefix_length = 2
        prefix_scheme = CyclicPrefixScheme(prefix_length=prefix_length)
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = SingleCarrierOFDMModulator(
            prefix_scheme=prefix_scheme, equalizator=equalizator, num_subcarriers=num_subcarriers
        )

        # Received signal with prefix
        received = np.random.randn(3, num_subcarriers + prefix_length) + 1j * np.random.randn(
            3, num_subcarriers + prefix_length
        )
        received = received.astype(np.complex128)

        demodulated = modulator.demodulate(received)

        # Should remove prefix and return time domain
        assert demodulated.shape == (3, num_subcarriers)

    def test_demodulate_applies_frequency_domain_equalization(self):
        """Test that SC-OFDM demodulation applies FDE."""
        num_subcarriers = 4
        prefix_scheme = NoPrefixScheme()
        channel_response = np.array([2.0, 0.5, 1.0, 4.0], dtype=np.complex128)
        equalizator = ZeroForcingEqualizator(channel_frequency_response=channel_response)

        modulator = SingleCarrierOFDMModulator(
            prefix_scheme=prefix_scheme, equalizator=equalizator, num_subcarriers=num_subcarriers
        )

        # Simulate received signal
        original_time = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.complex128)

        # Apply channel in frequency domain
        freq = np.fft.fft(original_time, axis=1, norm="ortho")
        freq_with_channel = freq * channel_response
        time_with_channel = np.fft.ifft(freq_with_channel, axis=1, norm="ortho")

        # Demodulate (includes FDE)
        demodulated = modulator.demodulate(time_with_channel)

        # Should recover original time domain signal
        assert_array_almost_equal(demodulated, original_time, decimal=10)


class TestSingleCarrierOFDMModulatorRoundTrip:
    """Test SC-OFDM modulation-demodulation round trip."""

    def test_modulate_demodulate_perfect_channel(self):
        """Test SC-OFDM round trip with perfect channel."""
        num_subcarriers = 16
        prefix_scheme = NoPrefixScheme()
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = SingleCarrierOFDMModulator(
            prefix_scheme=prefix_scheme, equalizator=equalizator, num_subcarriers=num_subcarriers
        )

        original_symbols = np.random.randn(5, num_subcarriers) + 1j * np.random.randn(
            5, num_subcarriers
        )
        original_symbols = original_symbols.astype(np.complex128)

        modulated = modulator.modulate(original_symbols)
        demodulated = modulator.demodulate(modulated)

        # Should recover original symbols
        assert_array_almost_equal(demodulated, original_symbols, decimal=10)

    def test_modulate_demodulate_with_cyclic_prefix(self):
        """Test SC-OFDM round trip with cyclic prefix."""
        num_subcarriers = 32
        prefix_length = 8
        prefix_scheme = CyclicPrefixScheme(prefix_length=prefix_length)
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = SingleCarrierOFDMModulator(
            prefix_scheme=prefix_scheme, equalizator=equalizator, num_subcarriers=num_subcarriers
        )

        original_symbols = np.random.randn(10, num_subcarriers) + 1j * np.random.randn(
            10, num_subcarriers
        )
        original_symbols = original_symbols.astype(np.complex128)

        modulated = modulator.modulate(original_symbols)
        demodulated = modulator.demodulate(modulated)

        assert_array_almost_equal(demodulated, original_symbols, decimal=10)

    def test_modulate_demodulate_with_channel_equalization(self):
        """Test SC-OFDM round trip with channel and FDE."""
        num_subcarriers = 8
        prefix_scheme = NoPrefixScheme()
        channel_response = np.array(
            [1.0, 0.8, 0.6, 0.9, 0.7, 0.95, 0.85, 0.75], dtype=np.complex128
        )
        equalizator = ZeroForcingEqualizator(channel_frequency_response=channel_response)

        modulator = SingleCarrierOFDMModulator(
            prefix_scheme=prefix_scheme, equalizator=equalizator, num_subcarriers=num_subcarriers
        )

        original_symbols = np.random.randn(3, num_subcarriers) + 1j * np.random.randn(
            3, num_subcarriers
        )
        original_symbols = original_symbols.astype(np.complex128)

        # Modulate
        modulated = modulator.modulate(original_symbols)

        # Apply channel in frequency domain
        freq_domain = np.fft.fft(modulated, axis=1, norm="ortho")
        freq_domain *= channel_response
        time_with_channel = np.fft.ifft(freq_domain, axis=1, norm="ortho")

        # Demodulate (includes FDE)
        demodulated = modulator.demodulate(time_with_channel)

        # Should recover original symbols
        assert_array_almost_equal(demodulated, original_symbols, decimal=10)


class TestOFDMvsSCOFDMComparison:
    """Compare OFDM and SC-OFDM behavior."""

    def test_both_recover_signal_in_perfect_channel(self):
        """Test both modulators recover signal in perfect channel."""
        num_subcarriers = 16
        prefix_scheme = NoPrefixScheme()
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        ofdm = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        sc_ofdm = SingleCarrierOFDMModulator(
            prefix_scheme=prefix_scheme, equalizator=equalizator, num_subcarriers=num_subcarriers
        )

        # For OFDM: frequency domain symbols
        freq_symbols = np.random.randn(3, num_subcarriers) + 1j * np.random.randn(
            3, num_subcarriers
        )
        freq_symbols = freq_symbols.astype(np.complex128)

        # For SC-OFDM: time domain symbols (constellation mapped)
        time_symbols = np.random.randn(3, num_subcarriers) + 1j * np.random.randn(
            3, num_subcarriers
        )
        time_symbols = time_symbols.astype(np.complex128)

        # OFDM round trip
        ofdm_mod = ofdm.modulate(freq_symbols)
        ofdm_demod = ofdm.demodulate(ofdm_mod)

        # SC-OFDM round trip
        sc_mod = sc_ofdm.modulate(time_symbols)
        sc_demod = sc_ofdm.demodulate(sc_mod)

        # Both should recover
        assert_array_almost_equal(ofdm_demod, freq_symbols, decimal=10)
        assert_array_almost_equal(sc_demod, time_symbols, decimal=10)

    def test_different_signal_domains(self):
        """Test that OFDM works in freq domain, SC-OFDM in time domain."""
        num_subcarriers = 8
        prefix_scheme = NoPrefixScheme()
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        ofdm = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        sc_ofdm = SingleCarrierOFDMModulator(
            prefix_scheme=prefix_scheme, equalizator=equalizator, num_subcarriers=num_subcarriers
        )

        # Single impulse in frequency domain for OFDM
        ofdm_input = np.zeros((1, num_subcarriers), dtype=np.complex128)
        ofdm_input[0, 0] = 1.0

        # Single impulse in time domain for SC-OFDM
        sc_input = np.zeros((1, num_subcarriers), dtype=np.complex128)
        sc_input[0, 0] = 1.0

        ofdm_output = ofdm.modulate(ofdm_input)
        sc_output = sc_ofdm.modulate(sc_input)

        # OFDM modulate (IFFT) converts impulse to constant
        # SC-OFDM keeps impulse in time domain
        assert not np.allclose(ofdm_output, sc_output)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_subcarrier_ofdm(self):
        """Test OFDM with single subcarrier."""
        num_subcarriers = 1
        prefix_scheme = NoPrefixScheme()
        channel_response = np.array([1.0], dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        symbols = np.array([[1.0 + 1.0j]], dtype=np.complex128)
        modulated = modulator.modulate(symbols)
        demodulated = modulator.demodulate(modulated)

        assert_array_almost_equal(demodulated, symbols, decimal=10)

    def test_large_number_of_subcarriers(self):
        """Test OFDM with large number of subcarriers."""
        num_subcarriers = 2048
        prefix_scheme = NoPrefixScheme()
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        symbols = np.random.randn(2, num_subcarriers) + 1j * np.random.randn(2, num_subcarriers)
        symbols = symbols.astype(np.complex128)

        modulated = modulator.modulate(symbols)
        demodulated = modulator.demodulate(modulated)

        assert_array_almost_equal(demodulated, symbols, decimal=10)

    def test_zero_symbols(self):
        """Test modulation with all-zero symbols."""
        num_subcarriers = 8
        prefix_scheme = NoPrefixScheme()
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        symbols = np.zeros((3, num_subcarriers), dtype=np.complex128)

        modulated = modulator.modulate(symbols)
        demodulated = modulator.demodulate(modulated)

        assert_array_almost_equal(demodulated, symbols, decimal=10)

    def test_pure_real_symbols(self):
        """Test OFDM with pure real symbols."""
        num_subcarriers = 16
        prefix_scheme = NoPrefixScheme()
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        symbols = np.random.randn(5, num_subcarriers).astype(np.complex128)

        modulated = modulator.modulate(symbols)
        demodulated = modulator.demodulate(modulated)

        assert_array_almost_equal(demodulated, symbols, decimal=10)

    def test_very_long_cyclic_prefix(self):
        """Test OFDM with very long cyclic prefix."""
        num_subcarriers = 16
        prefix_length = 12  # Long prefix
        prefix_scheme = CyclicPrefixScheme(prefix_length=prefix_length)
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        symbols = np.random.randn(2, num_subcarriers) + 1j * np.random.randn(2, num_subcarriers)
        symbols = symbols.astype(np.complex128)

        modulated = modulator.modulate(symbols)
        assert modulated.shape[1] == num_subcarriers + prefix_length

        demodulated = modulator.demodulate(modulated)
        assert_array_almost_equal(demodulated, symbols, decimal=10)


class TestIntegrationScenarios:
    """Integration tests with realistic scenarios."""

    def test_ofdm_with_mmse_equalization(self):
        """Test OFDM with MMSE equalization in noisy channel."""
        num_subcarriers = 64
        prefix_scheme = CyclicPrefixScheme(prefix_length=16)

        # Frequency selective channel
        impulse_response = np.array([1.0, 0.5, 0.3, 0.2], dtype=np.complex128)
        channel_response = np.fft.fft(impulse_response, n=num_subcarriers)

        equalizator = MMSEEqualizator(channel_frequency_response=channel_response, snr_db=20.0)

        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        symbols = np.random.randn(10, num_subcarriers) + 1j * np.random.randn(10, num_subcarriers)
        symbols = symbols.astype(np.complex128)

        # Modulate
        modulated = modulator.modulate(symbols)

        # Apply channel
        time_no_prefix = np.array([prefix_scheme.remove_prefix(row) for row in modulated])
        freq = np.fft.fft(time_no_prefix, axis=1, norm="ortho")
        freq *= channel_response
        time_with_channel = np.fft.ifft(freq, axis=1, norm="ortho")

        # Add cyclic prefix back
        time_with_prefix = np.array([prefix_scheme.add_prefix(row) for row in time_with_channel])

        # Demodulate
        demodulated = modulator.demodulate(time_with_prefix)

        # Should approximately recover (MMSE doesn't perfectly invert)
        assert demodulated.shape == symbols.shape
        # Check correlation is high
        correlation = np.abs(np.sum(symbols * np.conj(demodulated))) / (
            np.sqrt(np.sum(np.abs(symbols) ** 2)) * np.sqrt(np.sum(np.abs(demodulated) ** 2))
        )
        assert correlation > 0.9

    def test_sc_ofdm_frequency_domain_equalization(self):
        """Test SC-OFDM FDE in frequency selective channel."""
        num_subcarriers = 32
        prefix_length = 8
        prefix_scheme = CyclicPrefixScheme(prefix_length=prefix_length)

        # Frequency selective channel
        channel_response = np.random.rand(num_subcarriers) + 1j * np.random.rand(num_subcarriers)
        channel_response = channel_response.astype(np.complex128)
        channel_response /= np.abs(channel_response).mean()  # Normalize

        equalizator = ZeroForcingEqualizator(channel_frequency_response=channel_response)

        modulator = SingleCarrierOFDMModulator(
            prefix_scheme=prefix_scheme, equalizator=equalizator, num_subcarriers=num_subcarriers
        )

        symbols = np.random.randn(5, num_subcarriers) + 1j * np.random.randn(5, num_subcarriers)
        symbols = symbols.astype(np.complex128)

        # Modulate
        modulated = modulator.modulate(symbols)

        # Apply channel
        time_no_prefix = np.array([prefix_scheme.remove_prefix(row) for row in modulated])
        freq = np.fft.fft(time_no_prefix, axis=1, norm="ortho")
        freq *= channel_response
        time_with_channel = np.fft.ifft(freq, axis=1, norm="ortho")

        # Add prefix back
        time_with_prefix = np.array([prefix_scheme.add_prefix(row) for row in time_with_channel])

        # Demodulate
        demodulated = modulator.demodulate(time_with_prefix)

        # Should recover original
        assert_array_almost_equal(demodulated, symbols, decimal=9)
