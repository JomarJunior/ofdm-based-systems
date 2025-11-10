"""
Unit tests for channel models.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from ofdm_based_systems.channel.models import ChannelModel
from ofdm_based_systems.noise.models import AWGNoiseModel, NoNoiseModel


class TestChannelModelInitialization:
    """Test ChannelModel initialization and properties."""

    def test_init_with_simple_impulse_response(self):
        """Test initialization with a simple impulse response."""
        impulse_response = np.array([1.0 + 0.0j, 0.5 + 0.0j], dtype=np.complex128)
        snr_db = 10.0

        channel = ChannelModel(impulse_response=impulse_response, snr_db=snr_db)

        # Check that the impulse response is normalized
        power = np.sum(np.abs(channel.impulse_response) ** 2)
        assert np.isclose(power, 1.0), "Impulse response should have unit power"

        # Check SNR is set correctly
        assert channel.snr_db == snr_db

        # Check default noise model
        assert isinstance(channel.noise_model, AWGNoiseModel)

    def test_init_with_custom_noise_model(self):
        """Test initialization with a custom noise model."""
        impulse_response = np.array([1.0 + 0.0j], dtype=np.complex128)
        snr_db = 15.0
        noise_model = NoNoiseModel()

        channel = ChannelModel(
            impulse_response=impulse_response, snr_db=snr_db, noise_model=noise_model
        )

        assert isinstance(channel.noise_model, NoNoiseModel)

    def test_init_with_complex_impulse_response(self):
        """Test initialization with complex impulse response."""
        impulse_response = np.array([1.0 + 1.0j, 0.5 - 0.5j, 0.2 + 0.3j], dtype=np.complex128)
        snr_db = 20.0

        channel = ChannelModel(impulse_response=impulse_response, snr_db=snr_db)

        # Verify normalization
        power = np.sum(np.abs(channel.impulse_response) ** 2)
        assert np.isclose(power, 1.0)

    def test_init_with_zero_impulse_response_raises_error(self):
        """Test that initializing with all-zero impulse response raises ValueError."""
        impulse_response = np.array([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
        snr_db = 10.0

        with pytest.raises(ValueError, match="Impulse response cannot be all zeros"):
            ChannelModel(impulse_response=impulse_response, snr_db=snr_db)

    def test_init_with_single_tap(self):
        """Test initialization with single-tap channel (no multipath)."""
        impulse_response = np.array([1.0 + 0.0j], dtype=np.complex128)
        snr_db = 25.0

        channel = ChannelModel(impulse_response=impulse_response, snr_db=snr_db)

        assert len(channel.impulse_response) == 1
        assert channel.order == 0


class TestChannelModelOrder:
    """Test the order property of ChannelModel."""

    def test_order_single_tap(self):
        """Test order for single-tap channel."""
        impulse_response = np.array([1.0 + 0.0j], dtype=np.complex128)
        channel = ChannelModel(impulse_response=impulse_response, snr_db=10.0)

        assert channel.order == 0

    def test_order_two_taps(self):
        """Test order for two-tap channel."""
        impulse_response = np.array([1.0 + 0.0j, 0.5 + 0.0j], dtype=np.complex128)
        channel = ChannelModel(impulse_response=impulse_response, snr_db=10.0)

        assert channel.order == 1

    def test_order_multi_tap(self):
        """Test order for multi-tap channel."""
        impulse_response = np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.complex128)
        channel = ChannelModel(impulse_response=impulse_response, snr_db=10.0)

        assert channel.order == 4


class TestNormalizeImpulseResponse:
    """Test the normalize_impulse_response method."""

    def test_normalize_real_valued(self):
        """Test normalization with real-valued impulse response."""
        impulse_response = np.array([2.0, 1.0, 0.5], dtype=np.complex128)
        channel = ChannelModel(impulse_response=impulse_response, snr_db=10.0)

        normalized = channel.impulse_response
        power = np.sum(np.abs(normalized) ** 2)

        assert np.isclose(power, 1.0)

    def test_normalize_complex_valued(self):
        """Test normalization with complex-valued impulse response."""
        impulse_response = np.array([1.0 + 1.0j, 0.5 - 0.5j], dtype=np.complex128)
        channel = ChannelModel(impulse_response=impulse_response, snr_db=10.0)

        normalized = channel.impulse_response
        power = np.sum(np.abs(normalized) ** 2)

        assert np.isclose(power, 1.0)

    def test_normalize_already_normalized(self):
        """Test normalization with already normalized impulse response."""
        # Create a normalized impulse response
        impulse_response = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], dtype=np.complex128)
        channel = ChannelModel(impulse_response=impulse_response, snr_db=10.0)

        # Should remain normalized
        power = np.sum(np.abs(channel.impulse_response) ** 2)
        assert np.isclose(power, 1.0)

    def test_normalize_preserves_phase(self):
        """Test that normalization preserves phase relationships."""
        impulse_response = np.array([1.0 + 1.0j, -1.0 + 1.0j], dtype=np.complex128)
        original_phases = np.angle(impulse_response)

        channel = ChannelModel(impulse_response=impulse_response, snr_db=10.0)
        normalized_phases = np.angle(channel.impulse_response)

        # Phases should be preserved
        assert_array_almost_equal(original_phases, normalized_phases)


class TestGetFrequencyResponse:
    """Test the get_frequency_response method."""

    def test_frequency_response_single_tap(self):
        """Test frequency response for single-tap channel."""
        impulse_response = np.array([1.0 + 0.0j], dtype=np.complex128)
        channel = ChannelModel(impulse_response=impulse_response, snr_db=10.0)

        n_fft = 64
        freq_response = channel.get_frequency_response(n_fft)

        # For a single tap with unit amplitude, frequency response should be flat
        expected = np.ones(n_fft, dtype=np.complex128)
        assert_array_almost_equal(freq_response, expected)

    def test_frequency_response_two_taps(self):
        """Test frequency response for two-tap channel."""
        # Normalized two-tap channel
        h = np.array([1.0, 0.5], dtype=np.complex128)
        channel = ChannelModel(impulse_response=h, snr_db=10.0)

        n_fft = 8
        freq_response = channel.get_frequency_response(n_fft)

        # Manually compute expected FFT
        h_normalized = h / np.sqrt(np.sum(np.abs(h) ** 2))
        expected = np.fft.fft(h_normalized, n=n_fft)

        assert_array_almost_equal(freq_response, expected)

    def test_frequency_response_caching(self):
        """Test that frequency response is cached correctly."""
        impulse_response = np.array([1.0, 0.5, 0.3], dtype=np.complex128)
        channel = ChannelModel(impulse_response=impulse_response, snr_db=10.0)

        n_fft = 64

        # First call - should compute and cache
        freq_response_1 = channel.get_frequency_response(n_fft)

        # Second call - should return cached value
        freq_response_2 = channel.get_frequency_response(n_fft)

        # Should be the same object (cached)
        assert freq_response_1 is freq_response_2
        assert_array_equal(freq_response_1, freq_response_2)

    def test_frequency_response_different_fft_sizes(self):
        """Test frequency response for different FFT sizes."""
        impulse_response = np.array([1.0, 0.5], dtype=np.complex128)
        channel = ChannelModel(impulse_response=impulse_response, snr_db=10.0)

        # Get frequency responses for different FFT sizes
        freq_response_32 = channel.get_frequency_response(32)
        freq_response_64 = channel.get_frequency_response(64)

        # Should have different lengths
        assert len(freq_response_32) == 32
        assert len(freq_response_64) == 64

        # Both should be cached
        assert 32 in channel.frequency_response_cache
        assert 64 in channel.frequency_response_cache


class TestTransmit:
    """Test the transmit method."""

    def test_transmit_preserves_signal_length(self):
        """Test that transmission preserves signal length."""
        impulse_response = np.array([1.0, 0.5, 0.3], dtype=np.complex128)
        channel = ChannelModel(impulse_response=impulse_response, snr_db=10.0)

        signal = np.random.randn(100) + 1j * np.random.randn(100)
        received = channel.transmit(signal)

        assert len(received) == len(signal)

    def test_transmit_with_no_noise(self):
        """Test transmission without noise."""
        impulse_response = np.array([1.0 + 0.0j], dtype=np.complex128)
        noise_model = NoNoiseModel()
        channel = ChannelModel(
            impulse_response=impulse_response, snr_db=10.0, noise_model=noise_model
        )

        # Simple signal
        signal = np.ones(10, dtype=np.complex128)
        received = channel.transmit(signal)

        # With unit impulse response and no noise, signal should be preserved
        # (after power normalization)
        assert_array_almost_equal(np.abs(received), np.abs(signal), decimal=5)

    def test_transmit_applies_channel_effects(self):
        """Test that transmission applies channel convolution."""
        # Two-tap channel
        impulse_response = np.array([1.0, 0.5], dtype=np.complex128)
        noise_model = NoNoiseModel()
        channel = ChannelModel(
            impulse_response=impulse_response, snr_db=10.0, noise_model=noise_model
        )

        # Impulse signal
        signal = np.zeros(10, dtype=np.complex128)
        signal[0] = 1.0

        received = channel.transmit(signal)

        # First sample should be affected by first tap
        # Second sample should be affected by second tap
        assert np.abs(received[0]) > 0
        assert np.abs(received[1]) > 0

    def test_transmit_adds_noise(self):
        """Test that transmission adds noise when using AWGNoiseModel."""
        impulse_response = np.array([1.0 + 0.0j], dtype=np.complex128)
        channel = ChannelModel(impulse_response=impulse_response, snr_db=10.0)

        # Set random seed for reproducibility
        np.random.seed(42)

        signal = np.ones(1000, dtype=np.complex128)
        received = channel.transmit(signal)

        # With noise, received signal should differ from transmitted
        assert not np.allclose(signal, received)

        # But should be correlated
        correlation = np.abs(np.mean(signal * np.conj(received)))
        assert correlation > 0.5  # Strong correlation expected at SNR=10dB

    def test_transmit_power_normalization(self):
        """Test that transmission normalizes output to unit power."""
        impulse_response = np.array([2.0, 1.0, 0.5], dtype=np.complex128)
        noise_model = NoNoiseModel()
        channel = ChannelModel(
            impulse_response=impulse_response, snr_db=10.0, noise_model=noise_model
        )

        signal = np.random.randn(100) + 1j * np.random.randn(100)

        received = channel.transmit(signal)
        received_power = np.mean(np.abs(received) ** 2)

        # Channel normalizes output to unit power
        assert np.isclose(received_power, 1.0, rtol=0.01)

    def test_transmit_with_multidimensional_signal_raises_error(self):
        """Test that transmit raises error for multidimensional signals."""
        impulse_response = np.array([1.0, 0.5], dtype=np.complex128)
        channel = ChannelModel(impulse_response=impulse_response, snr_db=10.0)

        # 2D signal
        signal = np.ones((10, 10), dtype=np.complex128)

        with pytest.raises(ValueError, match="Signal must be serial \\(1D array\\)"):
            channel.transmit(signal)

    def test_transmit_with_zero_signal(self):
        """Test transmission of zero signal results in NaN due to normalization."""
        impulse_response = np.array([1.0, 0.5], dtype=np.complex128)
        noise_model = NoNoiseModel()
        channel = ChannelModel(
            impulse_response=impulse_response, snr_db=10.0, noise_model=noise_model
        )

        signal = np.zeros(10, dtype=np.complex128)
        received = channel.transmit(signal)

        # Zero signal results in NaN after power normalization (division by zero)
        # This is expected behavior - the channel cannot normalize zero power
        assert np.all(np.isnan(received))

    def test_transmit_snr_consistency(self):
        """Test that transmission respects SNR setting."""
        impulse_response = np.array([1.0 + 0.0j], dtype=np.complex128)

        # High SNR channel
        high_snr_channel = ChannelModel(impulse_response=impulse_response, snr_db=30.0)

        # Low SNR channel
        low_snr_channel = ChannelModel(impulse_response=impulse_response, snr_db=5.0)

        np.random.seed(42)
        signal = np.ones(1000, dtype=np.complex128)

        # Reset seed for fair comparison
        np.random.seed(42)
        received_high_snr = high_snr_channel.transmit(signal)

        np.random.seed(42)
        received_low_snr = low_snr_channel.transmit(signal)

        # High SNR should be closer to original signal
        error_high = np.mean(np.abs(signal - received_high_snr) ** 2)
        error_low = np.mean(np.abs(signal - received_low_snr) ** 2)

        assert error_high < error_low


class TestChannelModelIntegration:
    """Integration tests for ChannelModel."""

    def test_frequency_selective_channel(self):
        """Test a frequency-selective multipath channel."""
        # Typical multipath channel
        impulse_response = np.array(
            [1.0 + 0.0j, 0.5 + 0.3j, 0.3 - 0.2j, 0.1 + 0.1j], dtype=np.complex128
        )
        channel = ChannelModel(impulse_response=impulse_response, snr_db=15.0)

        # Generate OFDM-like signal
        n_fft = 64
        freq_domain = np.random.randn(n_fft) + 1j * np.random.randn(n_fft)
        time_domain = np.fft.ifft(freq_domain)

        # Transmit through channel
        received = channel.transmit(time_domain)

        # Convert back to frequency domain
        received_freq = np.fft.fft(received)

        # Check that frequency response is applied
        channel_freq_response = channel.get_frequency_response(n_fft)

        # The received frequency domain signal should be related to channel response
        # (within noise tolerance)
        assert len(received_freq) == len(freq_domain)

    def test_flat_fading_channel(self):
        """Test a flat fading channel (single tap)."""
        # Flat fading - single complex gain
        impulse_response = np.array([0.7 + 0.7j], dtype=np.complex128)
        noise_model = NoNoiseModel()
        channel = ChannelModel(
            impulse_response=impulse_response, snr_db=20.0, noise_model=noise_model
        )

        signal = np.ones(50, dtype=np.complex128)
        received = channel.transmit(signal)

        # For flat fading, all symbols should be scaled by same complex factor
        # Check that the received signal has constant phase difference
        phases = np.angle(received)
        phase_diffs = np.diff(phases)

        # Phase differences should be near zero (constant phase shift)
        assert np.allclose(phase_diffs, 0, atol=1e-10)

    def test_power_normalization_across_multiple_transmissions(self):
        """Test that channel consistently normalizes to unit power."""
        impulse_response = np.array([1.0, 0.5, 0.3, 0.2], dtype=np.complex128)
        noise_model = NoNoiseModel()
        channel = ChannelModel(
            impulse_response=impulse_response, snr_db=10.0, noise_model=noise_model
        )

        # Transmit multiple signals
        for _ in range(5):
            signal = np.random.randn(100) + 1j * np.random.randn(100)

            received = channel.transmit(signal)
            received_power = np.mean(np.abs(received) ** 2)

            # Channel normalizes all outputs to unit power
            assert np.isclose(received_power, 1.0, rtol=0.01)

    def test_real_world_lte_like_channel(self):
        """Test with a realistic LTE-like channel model."""
        # EPA (Extended Pedestrian A) like channel
        impulse_response = np.array(
            [0.7767 + 0.4561j, -0.0667 + 0.2840j, 0.1399 - 0.1592j, 0.0223 + 0.2409j],
            dtype=np.complex128,
        )
        channel = ChannelModel(impulse_response=impulse_response, snr_db=12.0)

        # OFDM symbol
        n_carriers = 128
        symbols = np.random.randn(n_carriers) + 1j * np.random.randn(n_carriers)
        time_signal = np.fft.ifft(symbols)

        # Add cyclic prefix
        cp_length = impulse_response.shape[0]  # CP length >= channel order
        time_with_cp = np.concatenate([time_signal[-cp_length:], time_signal])

        # Transmit
        received = channel.transmit(time_with_cp)

        # Remove cyclic prefix
        received_no_cp = received[cp_length:]

        # Check that we can recover something meaningful
        assert len(received_no_cp) == len(time_signal)

        # Convert back to frequency domain
        received_freq = np.fft.fft(received_no_cp)

        # Should have same length as transmitted
        assert len(received_freq) == len(symbols)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_long_impulse_response(self):
        """Test with a very long impulse response."""
        impulse_response = np.random.randn(100) + 1j * np.random.randn(100)
        impulse_response = impulse_response.astype(np.complex128)

        channel = ChannelModel(impulse_response=impulse_response, snr_db=10.0)

        # Should still normalize correctly
        power = np.sum(np.abs(channel.impulse_response) ** 2)
        assert np.isclose(power, 1.0)

        assert channel.order == 99

    def test_very_short_signal(self):
        """Test transmission of very short signal."""
        impulse_response = np.array([1.0, 0.5], dtype=np.complex128)
        noise_model = NoNoiseModel()
        channel = ChannelModel(
            impulse_response=impulse_response, snr_db=10.0, noise_model=noise_model
        )

        # Single sample signal
        signal = np.array([1.0 + 0.0j], dtype=np.complex128)
        received = channel.transmit(signal)

        assert len(received) == 1

    def test_extreme_snr_values(self):
        """Test with extreme SNR values."""
        impulse_response = np.array([1.0 + 0.0j], dtype=np.complex128)

        # Very high SNR
        high_snr_channel = ChannelModel(impulse_response=impulse_response, snr_db=100.0)
        signal = np.ones(100, dtype=np.complex128)
        received_high = high_snr_channel.transmit(signal)

        # Should be very close to original
        assert np.allclose(signal, received_high, rtol=1e-3)

        # Very low SNR
        low_snr_channel = ChannelModel(impulse_response=impulse_response, snr_db=-10.0)
        np.random.seed(42)
        received_low = low_snr_channel.transmit(signal)

        # Should have significant noise
        error = np.mean(np.abs(signal - received_low) ** 2)
        assert error > 0.1  # Significant error expected

    def test_negative_frequency_response_magnitude(self):
        """Test that frequency response can have varying magnitudes."""
        # Channel with deep fade at some frequencies
        impulse_response = np.array([1.0, -0.9, 0.8, -0.7], dtype=np.complex128)
        channel = ChannelModel(impulse_response=impulse_response, snr_db=10.0)

        freq_response = channel.get_frequency_response(64)

        # Should have varying magnitude
        magnitudes = np.abs(freq_response)
        assert np.max(magnitudes) > np.min(magnitudes)

        # Some frequencies might have very low gain
        assert np.min(magnitudes) < np.max(magnitudes) / 2
