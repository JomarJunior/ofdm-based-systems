"""
Unit tests for equalization models.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from ofdm_based_systems.equalization.models import (
    IEqualizator,
    MMSEEqualizator,
    NoEqualizator,
    ZeroForcingEqualizator,
)


class TestZeroForcingEqualizatorInitialization:
    """Test ZeroForcingEqualizator initialization."""

    def test_init_with_simple_channel(self):
        """Test initialization with simple channel response."""
        channel_response = np.array([1.0 + 0.0j, 0.8 + 0.2j], dtype=np.complex128)
        snr_db = 15.0

        equalizer = ZeroForcingEqualizator(
            channel_frequency_response=channel_response, snr_db=snr_db
        )

        assert_array_equal(equalizer.channel_frequency_response, channel_response)
        assert equalizer.snr_db == snr_db

    def test_init_without_snr(self):
        """Test initialization without SNR (optional parameter)."""
        channel_response = np.array([1.0 + 0.0j], dtype=np.complex128)

        equalizer = ZeroForcingEqualizator(channel_frequency_response=channel_response)

        assert equalizer.snr_db is None

    def test_init_with_complex_channel(self):
        """Test initialization with complex channel response."""
        channel_response = np.array(
            [0.9 + 0.3j, 0.5 - 0.5j, 0.7 + 0.1j, 0.3 + 0.8j], dtype=np.complex128
        )

        equalizer = ZeroForcingEqualizator(channel_frequency_response=channel_response, snr_db=20.0)

        assert len(equalizer.channel_frequency_response) == 4


class TestZeroForcingEqualizatorEqualize:
    """Test ZeroForcingEqualizator equalize method."""

    def test_equalize_perfect_channel(self):
        """Test equalization with perfect channel (all ones)."""
        channel_response = np.ones(4, dtype=np.complex128)
        received_symbols = np.array(
            [1.0 + 1.0j, 2.0 - 1.0j, 0.5 + 0.5j, -1.0 + 0.0j], dtype=np.complex128
        )

        equalizer = ZeroForcingEqualizator(channel_frequency_response=channel_response)
        equalized = equalizer.equalize(received_symbols)

        # With perfect channel, output should equal input
        assert_array_almost_equal(equalized, received_symbols)

    def test_equalize_real_valued_channel(self):
        """Test equalization with real-valued channel gains."""
        channel_response = np.array([2.0, 0.5, 1.0, 4.0], dtype=np.complex128)
        received_symbols = np.array([2.0, 1.0, 3.0, 8.0], dtype=np.complex128)

        equalizer = ZeroForcingEqualizator(channel_frequency_response=channel_response)
        equalized = equalizer.equalize(received_symbols)

        # ZF divides by channel response
        expected = received_symbols / channel_response
        assert_array_almost_equal(equalized, expected)

    def test_equalize_complex_channel(self):
        """Test equalization with complex channel response."""
        channel_response = np.array([1.0 + 1.0j, 0.5 - 0.5j], dtype=np.complex128)
        received_symbols = np.array([2.0 + 2.0j, 1.0 - 1.0j], dtype=np.complex128)

        equalizer = ZeroForcingEqualizator(channel_frequency_response=channel_response)
        equalized = equalizer.equalize(received_symbols)

        expected = received_symbols / channel_response
        assert_array_almost_equal(equalized, expected)

    def test_equalize_with_zero_channel_gain(self):
        """Test equalization handles zero channel gain with epsilon."""
        channel_response = np.array([1.0, 0.0, 0.5], dtype=np.complex128)
        received_symbols = np.array([1.0, 2.0, 1.5], dtype=np.complex128)

        equalizer = ZeroForcingEqualizator(channel_frequency_response=channel_response)
        equalized = equalizer.equalize(received_symbols)

        # Should not raise error and should use epsilon for zero gain
        assert not np.any(np.isinf(equalized))
        assert not np.any(np.isnan(equalized))
        # First and third should be equalized normally
        assert np.isclose(equalized[0], 1.0)
        assert np.isclose(equalized[2], 3.0)

    def test_equalize_shape_mismatch_raises_error(self):
        """Test that shape mismatch raises ValueError."""
        channel_response = np.array([1.0, 0.5], dtype=np.complex128)
        received_symbols = np.array([1.0, 0.5, 0.3], dtype=np.complex128)

        equalizer = ZeroForcingEqualizator(channel_frequency_response=channel_response)

        with pytest.raises(ValueError, match="must have the same shape"):
            equalizer.equalize(received_symbols)

    def test_equalize_single_symbol(self):
        """Test equalization with single symbol."""
        channel_response = np.array([0.8 + 0.6j], dtype=np.complex128)
        received_symbols = np.array([1.6 + 1.2j], dtype=np.complex128)

        equalizer = ZeroForcingEqualizator(channel_frequency_response=channel_response)
        equalized = equalizer.equalize(received_symbols)

        expected = received_symbols / channel_response
        assert_array_almost_equal(equalized, expected)

    def test_equalize_noise_enhancement(self):
        """Test that ZF equalizer amplifies noise in weak channels."""
        # Weak channel gain
        channel_response = np.array([0.1 + 0.0j], dtype=np.complex128)
        received_symbols = np.array([0.1 + 0.01j], dtype=np.complex128)  # Signal + noise

        equalizer = ZeroForcingEqualizator(channel_frequency_response=channel_response)
        equalized = equalizer.equalize(received_symbols)

        # ZF should amplify both signal and noise
        assert np.abs(equalized[0]) > np.abs(received_symbols[0])

    def test_equalize_multiple_subcarriers(self):
        """Test equalization across multiple OFDM subcarriers."""
        n_subcarriers = 64
        channel_response = np.random.rand(n_subcarriers) + 1j * np.random.rand(n_subcarriers)
        channel_response = channel_response.astype(np.complex128)
        received_symbols = np.random.rand(n_subcarriers) + 1j * np.random.rand(n_subcarriers)
        received_symbols = received_symbols.astype(np.complex128)

        equalizer = ZeroForcingEqualizator(channel_frequency_response=channel_response)
        equalized = equalizer.equalize(received_symbols)

        assert len(equalized) == n_subcarriers
        # Check that equalization is correctly applied
        expected = received_symbols / np.where(channel_response == 0, 1e-10, channel_response)
        assert_array_almost_equal(equalized, expected)


class TestMMSEEqualizatorInitialization:
    """Test MMSEEqualizator initialization."""

    def test_init_with_snr(self):
        """Test initialization with SNR provided."""
        channel_response = np.array([1.0 + 0.0j, 0.5 + 0.5j], dtype=np.complex128)
        snr_db = 20.0

        equalizer = MMSEEqualizator(channel_frequency_response=channel_response, snr_db=snr_db)

        assert equalizer.snr_db == snr_db
        assert_array_equal(equalizer.channel_frequency_response, channel_response)

    def test_init_without_snr(self):
        """Test initialization without SNR."""
        channel_response = np.array([1.0 + 0.0j], dtype=np.complex128)

        equalizer = MMSEEqualizator(channel_frequency_response=channel_response)

        assert equalizer.snr_db is None


class TestMMSEEqualizatorCalculateNoiseVariance:
    """Test MMSEEqualizator calculate_noise_variance method."""

    def test_calculate_noise_variance_with_snr(self):
        """Test noise variance calculation with given SNR."""
        channel_response = np.array([1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
        snr_db = 10.0

        equalizer = MMSEEqualizator(channel_frequency_response=channel_response, snr_db=snr_db)

        received_signal = np.array([1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
        noise_var = equalizer.calculate_noise_variance(received_signal)

        # Noise variance should be positive
        assert noise_var > 0
        # For SNR=10dB, noise variance should be reasonable
        assert isinstance(noise_var, float)

    def test_calculate_noise_variance_without_snr_raises_error(self):
        """Test that calculation without SNR raises ValueError."""
        channel_response = np.array([1.0 + 0.0j], dtype=np.complex128)
        equalizer = MMSEEqualizator(channel_frequency_response=channel_response)

        received_signal = np.array([1.0 + 0.0j], dtype=np.complex128)

        with pytest.raises(ValueError, match="SNR in dB must be provided"):
            equalizer.calculate_noise_variance(received_signal)

    def test_calculate_noise_variance_zero_channel_gain(self):
        """Test noise variance calculation with zero channel gain."""
        channel_response = np.array([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
        snr_db = 10.0

        equalizer = MMSEEqualizator(channel_frequency_response=channel_response, snr_db=snr_db)

        received_signal = np.array([1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
        noise_var = equalizer.calculate_noise_variance(received_signal)

        # Should return infinity for zero channel gain
        assert noise_var == float("inf")

    def test_calculate_noise_variance_high_snr(self):
        """Test noise variance decreases with high SNR."""
        channel_response = np.array([1.0 + 0.0j], dtype=np.complex128)

        high_snr_equalizer = MMSEEqualizator(
            channel_frequency_response=channel_response, snr_db=30.0
        )
        low_snr_equalizer = MMSEEqualizator(channel_frequency_response=channel_response, snr_db=5.0)

        received_signal = np.array([1.0 + 0.0j], dtype=np.complex128)

        high_snr_noise_var = high_snr_equalizer.calculate_noise_variance(received_signal)
        low_snr_noise_var = low_snr_equalizer.calculate_noise_variance(received_signal)

        # Higher SNR should result in lower noise variance
        assert high_snr_noise_var < low_snr_noise_var

    def test_calculate_noise_variance_with_complex_signal(self):
        """Test noise variance calculation with complex signal."""
        channel_response = np.array([0.8 + 0.6j, 0.5 - 0.5j], dtype=np.complex128)
        snr_db = 15.0

        equalizer = MMSEEqualizator(channel_frequency_response=channel_response, snr_db=snr_db)

        received_signal = np.array([1.0 + 1.0j, 0.5 - 0.5j], dtype=np.complex128)
        noise_var = equalizer.calculate_noise_variance(received_signal)

        assert noise_var > 0
        assert np.isfinite(noise_var)


class TestMMSEEqualizatorEqualize:
    """Test MMSEEqualizator equalize method."""

    def test_equalize_perfect_channel_high_snr(self):
        """Test MMSE equalization with perfect channel and high SNR."""
        channel_response = np.ones(4, dtype=np.complex128)
        received_symbols = np.array(
            [1.0 + 1.0j, 2.0 - 1.0j, 0.5 + 0.5j, -1.0 + 0.0j], dtype=np.complex128
        )

        equalizer = MMSEEqualizator(
            channel_frequency_response=channel_response, snr_db=50.0  # Very high SNR
        )
        equalized = equalizer.equalize(received_symbols)

        # With perfect channel and high SNR, MMSE should approximate ZF
        assert_array_almost_equal(equalized, received_symbols, decimal=2)

    def test_equalize_real_channel(self):
        """Test MMSE equalization with real-valued channel."""
        channel_response = np.array([2.0, 0.5, 1.0], dtype=np.complex128)
        received_symbols = np.array([2.0, 1.0, 3.0], dtype=np.complex128)

        equalizer = MMSEEqualizator(channel_frequency_response=channel_response, snr_db=10.0)
        equalized = equalizer.equalize(received_symbols)

        # Should apply MMSE filter
        assert len(equalized) == len(received_symbols)
        assert not np.any(np.isinf(equalized))
        assert not np.any(np.isnan(equalized))

    def test_equalize_complex_channel(self):
        """Test MMSE equalization with complex channel response."""
        channel_response = np.array([1.0 + 1.0j, 0.5 - 0.5j, 0.8 + 0.2j], dtype=np.complex128)
        received_symbols = np.array([2.0 + 2.0j, 1.0 - 1.0j, 1.6 + 0.4j], dtype=np.complex128)

        equalizer = MMSEEqualizator(channel_frequency_response=channel_response, snr_db=15.0)
        equalized = equalizer.equalize(received_symbols)

        assert len(equalized) == len(received_symbols)
        # MMSE should produce finite values
        assert np.all(np.isfinite(equalized))

    def test_equalize_shape_mismatch_raises_error(self):
        """Test that shape mismatch raises ValueError."""
        channel_response = np.array([1.0, 0.5], dtype=np.complex128)
        received_symbols = np.array([1.0, 0.5, 0.3], dtype=np.complex128)

        equalizer = MMSEEqualizator(channel_frequency_response=channel_response, snr_db=10.0)

        with pytest.raises(ValueError, match="must have the same shape"):
            equalizer.equalize(received_symbols)

    def test_equalize_weak_channel_vs_zf(self):
        """Test that MMSE is more robust than ZF for weak channels."""
        # Very weak channel gain
        channel_response = np.array([0.1 + 0.0j], dtype=np.complex128)
        received_symbols = np.array([0.1 + 0.01j], dtype=np.complex128)

        zf_equalizer = ZeroForcingEqualizator(channel_frequency_response=channel_response)
        mmse_equalizer = MMSEEqualizator(channel_frequency_response=channel_response, snr_db=10.0)

        zf_output = zf_equalizer.equalize(received_symbols)
        mmse_output = mmse_equalizer.equalize(received_symbols)

        # MMSE should have lower magnitude (less noise amplification)
        assert np.abs(mmse_output[0]) < np.abs(zf_output[0])

    def test_equalize_low_snr_behavior(self):
        """Test MMSE behavior at low SNR."""
        channel_response = np.array([0.5 + 0.5j], dtype=np.complex128)
        received_symbols = np.array([1.0 + 1.0j], dtype=np.complex128)

        equalizer = MMSEEqualizator(
            channel_frequency_response=channel_response, snr_db=0.0  # Very low SNR (0 dB)
        )
        equalized = equalizer.equalize(received_symbols)

        # At low SNR, MMSE should be more conservative
        assert np.isfinite(equalized[0])

    def test_equalize_high_snr_approaches_zf(self):
        """Test that MMSE approaches ZF at high SNR."""
        channel_response = np.array([0.8 + 0.6j, 0.5 - 0.5j], dtype=np.complex128)
        received_symbols = np.array([1.6 + 1.2j, 1.0 - 1.0j], dtype=np.complex128)

        zf_equalizer = ZeroForcingEqualizator(channel_frequency_response=channel_response)
        mmse_equalizer = MMSEEqualizator(
            channel_frequency_response=channel_response, snr_db=100.0  # Very high SNR
        )

        zf_output = zf_equalizer.equalize(received_symbols)
        mmse_output = mmse_equalizer.equalize(received_symbols)

        # At very high SNR, MMSE should approximate ZF
        assert_array_almost_equal(mmse_output, zf_output, decimal=1)

    def test_equalize_multiple_subcarriers(self):
        """Test MMSE equalization across multiple OFDM subcarriers."""
        n_subcarriers = 64
        channel_response = np.random.rand(n_subcarriers) + 1j * np.random.rand(n_subcarriers)
        channel_response = channel_response.astype(np.complex128)
        received_symbols = np.random.rand(n_subcarriers) + 1j * np.random.rand(n_subcarriers)
        received_symbols = received_symbols.astype(np.complex128)

        equalizer = MMSEEqualizator(channel_frequency_response=channel_response, snr_db=15.0)
        equalized = equalizer.equalize(received_symbols)

        assert len(equalized) == n_subcarriers
        assert np.all(np.isfinite(equalized))


class TestNoEqualizator:
    """Test NoEqualizator (pass-through equalizer)."""

    def test_init(self):
        """Test NoEqualizator initialization."""
        channel_response = np.array([1.0 + 0.0j, 0.5 + 0.5j], dtype=np.complex128)

        equalizer = NoEqualizator(channel_frequency_response=channel_response)

        assert_array_equal(equalizer.channel_frequency_response, channel_response)

    def test_equalize_returns_input_unchanged(self):
        """Test that NoEqualizator returns input unchanged."""
        channel_response = np.array([0.5 + 0.5j], dtype=np.complex128)
        received_symbols = np.array([1.0 + 1.0j, 2.0 - 1.0j, 0.5 + 0.5j], dtype=np.complex128)

        equalizer = NoEqualizator(channel_frequency_response=channel_response)
        equalized = equalizer.equalize(received_symbols)

        # Should return exactly the same array
        assert_array_equal(equalized, received_symbols)
        assert equalized is received_symbols  # Should be the same object

    def test_equalize_with_zeros(self):
        """Test NoEqualizator with zero symbols."""
        channel_response = np.array([1.0], dtype=np.complex128)
        received_symbols = np.zeros(10, dtype=np.complex128)

        equalizer = NoEqualizator(channel_frequency_response=channel_response)
        equalized = equalizer.equalize(received_symbols)

        assert_array_equal(equalized, received_symbols)

    def test_equalize_with_complex_symbols(self):
        """Test NoEqualizator preserves complex symbols."""
        channel_response = np.array([0.8 + 0.6j], dtype=np.complex128)
        received_symbols = np.array(
            [1.0 + 2.0j, -1.0 - 1.0j, 0.5 - 0.8j, 3.0 + 0.1j], dtype=np.complex128
        )

        equalizer = NoEqualizator(channel_frequency_response=channel_response)
        equalized = equalizer.equalize(received_symbols)

        assert_array_equal(equalized, received_symbols)


class TestEqualizatorComparison:
    """Compare behavior of different equalizers."""

    def test_zf_vs_mmse_perfect_channel(self):
        """Compare ZF and MMSE with perfect channel."""
        channel_response = np.ones(4, dtype=np.complex128)
        received_symbols = np.array([1.0 + 1.0j, 2.0, 0.5 - 0.5j, -1.0 + 1.0j], dtype=np.complex128)

        zf = ZeroForcingEqualizator(channel_frequency_response=channel_response)
        mmse = MMSEEqualizator(channel_frequency_response=channel_response, snr_db=30.0)

        zf_output = zf.equalize(received_symbols)
        mmse_output = mmse.equalize(received_symbols)

        # With perfect channel, both should give similar results
        assert_array_almost_equal(zf_output, mmse_output, decimal=1)

    def test_zf_vs_mmse_weak_channel(self):
        """Compare ZF and MMSE with weak channel."""
        # Weak channel with deep fade
        channel_response = np.array([0.1 + 0.0j, 0.9 + 0.0j], dtype=np.complex128)
        received_symbols = np.array([0.1 + 0.01j, 0.9 + 0.05j], dtype=np.complex128)

        zf = ZeroForcingEqualizator(channel_frequency_response=channel_response)
        mmse = MMSEEqualizator(channel_frequency_response=channel_response, snr_db=10.0)

        zf_output = zf.equalize(received_symbols)
        mmse_output = mmse.equalize(received_symbols)

        # MMSE should amplify less for weak subcarrier
        assert np.abs(mmse_output[0]) < np.abs(zf_output[0])
        # Both should be similar for strong subcarrier
        assert np.abs(mmse_output[1] - zf_output[1]) < 0.5

    def test_no_vs_zf_comparison(self):
        """Compare NoEqualizator with ZF equalizer."""
        channel_response = np.array([0.5 + 0.5j], dtype=np.complex128)
        received_symbols = np.array([1.0 + 1.0j], dtype=np.complex128)

        no_eq = NoEqualizator(channel_frequency_response=channel_response)
        zf = ZeroForcingEqualizator(channel_frequency_response=channel_response)

        no_eq_output = no_eq.equalize(received_symbols)
        zf_output = zf.equalize(received_symbols)

        # NoEqualizator should return input unchanged
        assert_array_equal(no_eq_output, received_symbols)
        # ZF should equalize
        assert not np.allclose(zf_output, received_symbols)


class TestEqualizatorIntegration:
    """Integration tests for equalizers in realistic scenarios."""

    def test_frequency_selective_channel_equalization(self):
        """Test equalization in frequency-selective OFDM channel."""
        # Simulate frequency response of multipath channel
        n_subcarriers = 64
        impulse_response = np.array([1.0, 0.5, 0.3, 0.2], dtype=np.complex128)
        channel_response = np.fft.fft(impulse_response, n=n_subcarriers)

        # Simulate received OFDM symbols
        transmitted = np.random.randn(n_subcarriers) + 1j * np.random.randn(n_subcarriers)
        transmitted = transmitted.astype(np.complex128)
        received = transmitted * channel_response

        # Test ZF equalization
        zf = ZeroForcingEqualizator(channel_frequency_response=channel_response)
        zf_equalized = zf.equalize(received)

        # Should recover transmitted symbols
        assert_array_almost_equal(zf_equalized, transmitted, decimal=10)

    def test_mmse_equalization_with_noise(self):
        """Test MMSE equalization with noisy channel."""
        channel_response = np.array([0.8 + 0.6j, 0.5 - 0.5j, 0.9 + 0.1j], dtype=np.complex128)
        transmitted = np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j], dtype=np.complex128)

        # Add channel effect and noise
        received = transmitted * channel_response
        noise = 0.1 * (np.random.randn(3) + 1j * np.random.randn(3))
        received_noisy = received + noise

        mmse = MMSEEqualizator(channel_frequency_response=channel_response, snr_db=20.0)
        equalized = mmse.equalize(received_noisy)

        # Should be close to transmitted symbols
        assert len(equalized) == len(transmitted)
        # Rough check - should have correct order of magnitude
        for i in range(len(transmitted)):
            assert np.abs(equalized[i] - transmitted[i]) < 2.0

    def test_flat_fading_channel(self):
        """Test equalization in flat fading channel."""
        # Flat fading - single complex gain for all subcarriers
        flat_gain = 0.7 + 0.7j
        channel_response = np.full(8, flat_gain, dtype=np.complex128)
        transmitted = np.array(
            [1.0, -1.0, 1.0j, -1.0j, 1.0 + 1.0j, -1.0 - 1.0j, 0.5, -0.5], dtype=np.complex128
        )
        received = transmitted * channel_response

        zf = ZeroForcingEqualizator(channel_frequency_response=channel_response)
        equalized = zf.equalize(received)

        # Should perfectly recover in flat fading
        assert_array_almost_equal(equalized, transmitted)

    def test_deep_fade_handling(self):
        """Test equalizer behavior with deep channel fades."""
        # Channel with deep fade at some subcarriers
        channel_response = np.array([1.0, 0.01, 0.05, 0.9], dtype=np.complex128)
        transmitted = np.ones(4, dtype=np.complex128)
        received = transmitted * channel_response

        # Add small noise to show amplification effect
        noise = 0.001 * (np.random.randn(4) + 1j * np.random.randn(4))
        received_noisy = received + noise

        zf = ZeroForcingEqualizator(channel_frequency_response=channel_response)
        mmse = MMSEEqualizator(channel_frequency_response=channel_response, snr_db=15.0)

        zf_output = zf.equalize(received_noisy)
        mmse_output = mmse.equalize(received_noisy)

        # Both should recover transmitted signal, but with different noise handling
        # ZF perfectly equalizes the channel but amplifies noise at deep fades
        # The equalization gain at deep fade (0.01) is 1/0.01 = 100
        assert np.abs(1.0 / channel_response[1]) > 50  # Large equalization gain

        # MMSE should be more conservative due to noise regularization
        # MMSE output magnitude should be smaller than ZF due to regularization
        assert np.abs(mmse_output[1]) < np.abs(zf_output[1])


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_subcarrier_equalization(self):
        """Test equalization with single subcarrier."""
        channel_response = np.array([0.5 + 0.5j], dtype=np.complex128)
        received = np.array([1.0 + 1.0j], dtype=np.complex128)

        zf = ZeroForcingEqualizator(channel_frequency_response=channel_response)
        mmse = MMSEEqualizator(channel_frequency_response=channel_response, snr_db=20.0)

        zf_output = zf.equalize(received)
        mmse_output = mmse.equalize(received)

        assert len(zf_output) == 1
        assert len(mmse_output) == 1

    def test_very_large_number_of_subcarriers(self):
        """Test equalization with large number of subcarriers."""
        n_subcarriers = 2048
        channel_response = np.random.rand(n_subcarriers) + 1j * np.random.rand(n_subcarriers)
        channel_response = channel_response.astype(np.complex128)
        received = np.random.rand(n_subcarriers) + 1j * np.random.rand(n_subcarriers)
        received = received.astype(np.complex128)

        zf = ZeroForcingEqualizator(channel_frequency_response=channel_response)
        equalized = zf.equalize(received)

        assert len(equalized) == n_subcarriers

    def test_pure_real_channel_and_symbols(self):
        """Test equalization with pure real values."""
        channel_response = np.array([2.0, 0.5, 1.0], dtype=np.complex128)
        received = np.array([4.0, 1.0, 3.0], dtype=np.complex128)

        zf = ZeroForcingEqualizator(channel_frequency_response=channel_response)
        equalized = zf.equalize(received)

        expected = np.array([2.0, 2.0, 3.0], dtype=np.complex128)
        assert_array_almost_equal(equalized, expected)

    def test_pure_imaginary_channel(self):
        """Test equalization with pure imaginary channel."""
        channel_response = np.array([1.0j, 0.5j], dtype=np.complex128)
        received = np.array([1.0j, 0.5j], dtype=np.complex128)

        zf = ZeroForcingEqualizator(channel_frequency_response=channel_response)
        equalized = zf.equalize(received)

        expected = np.array([1.0, 1.0], dtype=np.complex128)
        assert_array_almost_equal(equalized, expected)

    def test_extreme_snr_values(self):
        """Test MMSE with extreme SNR values."""
        channel_response = np.array([0.8 + 0.6j], dtype=np.complex128)
        received = np.array([1.6 + 1.2j], dtype=np.complex128)

        # Very high SNR
        mmse_high = MMSEEqualizator(channel_frequency_response=channel_response, snr_db=100.0)
        high_output = mmse_high.equalize(received)

        # Very low SNR
        mmse_low = MMSEEqualizator(channel_frequency_response=channel_response, snr_db=-20.0)
        low_output = mmse_low.equalize(received)

        # High SNR should give larger output (more aggressive equalization)
        assert np.abs(high_output[0]) > np.abs(low_output[0])

    def test_all_zeros_channel_with_epsilon(self):
        """Test ZF handling of all-zero channel gains."""
        channel_response = np.zeros(3, dtype=np.complex128)
        received = np.array([1.0, 2.0, 3.0], dtype=np.complex128)

        zf = ZeroForcingEqualizator(channel_frequency_response=channel_response)
        equalized = zf.equalize(received)

        # Should not be infinite or NaN due to epsilon
        assert np.all(np.isfinite(equalized))

    def test_channel_with_phase_only(self):
        """Test equalization with phase-only channel (constant magnitude)."""
        # Phase shifts only, unit magnitude
        phases = np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        channel_response = np.exp(1j * phases).astype(np.complex128)
        transmitted = np.ones(4, dtype=np.complex128)
        received = transmitted * channel_response

        zf = ZeroForcingEqualizator(channel_frequency_response=channel_response)
        equalized = zf.equalize(received)

        # Should recover transmitted (all ones)
        assert_array_almost_equal(np.abs(equalized), np.ones(4))
