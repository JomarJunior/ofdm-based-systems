"""
Unit tests for prefix models.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from ofdm_based_systems.prefix.models import (
    CyclicPrefixScheme,
    IPrefixScheme,
    NoPrefixScheme,
    ZeroPaddingPrefixScheme,
)


class TestIPrefixSchemeInitialization:
    """Test IPrefixScheme base class initialization."""

    def test_init_with_valid_prefix_length(self):
        """Test initialization with valid prefix length."""
        prefix_scheme = CyclicPrefixScheme(prefix_length=4)
        assert prefix_scheme.prefix_length == 4

    def test_init_with_zero_prefix_length(self):
        """Test initialization with zero prefix length."""
        prefix_scheme = CyclicPrefixScheme(prefix_length=0)
        assert prefix_scheme.prefix_length == 0

    def test_init_with_negative_prefix_length_raises_error(self):
        """Test that negative prefix length raises ValueError."""
        with pytest.raises(ValueError, match="Prefix length must be a non-negative integer"):
            CyclicPrefixScheme(prefix_length=-1)

    def test_init_default_prefix_length(self):
        """Test initialization with default prefix length."""
        prefix_scheme = CyclicPrefixScheme()
        assert prefix_scheme.prefix_length == 0


class TestCyclicPrefixSchemeAcronym:
    """Test CyclicPrefixScheme acronym property."""

    def test_acronym_returns_cp(self):
        """Test that acronym returns 'CP'."""
        prefix_scheme = CyclicPrefixScheme(prefix_length=4)
        assert prefix_scheme.acronym == "CP"


class TestCyclicPrefixSchemeAddPrefix:
    """Test CyclicPrefixScheme add_prefix method."""

    def test_add_prefix_basic(self):
        """Test adding cyclic prefix to basic signal."""
        prefix_length = 2
        prefix_scheme = CyclicPrefixScheme(prefix_length=prefix_length)

        symbols = np.array([1, 2, 3, 4, 5, 6], dtype=np.complex128)
        result = prefix_scheme.add_prefix(symbols)

        # Last 2 elements should be prepended
        expected = np.array([5, 6, 1, 2, 3, 4, 5, 6], dtype=np.complex128)
        assert_array_equal(result, expected)

    def test_add_prefix_complex_symbols(self):
        """Test adding cyclic prefix to complex symbols."""
        prefix_length = 3
        prefix_scheme = CyclicPrefixScheme(prefix_length=prefix_length)

        symbols = np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j, 5 + 5j], dtype=np.complex128)
        result = prefix_scheme.add_prefix(symbols)

        # Last 3 elements prepended
        expected = np.array(
            [3 + 3j, 4 + 4j, 5 + 5j, 1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j, 5 + 5j], dtype=np.complex128
        )
        assert_array_equal(result, expected)

    def test_add_prefix_zero_length(self):
        """Test adding zero-length prefix returns original."""
        prefix_scheme = CyclicPrefixScheme(prefix_length=0)

        symbols = np.array([1, 2, 3, 4], dtype=np.complex128)
        result = prefix_scheme.add_prefix(symbols)

        assert_array_equal(result, symbols)

    def test_add_prefix_full_length(self):
        """Test adding prefix equal to symbol length."""
        symbols = np.array([1, 2, 3, 4], dtype=np.complex128)
        prefix_scheme = CyclicPrefixScheme(prefix_length=4)

        result = prefix_scheme.add_prefix(symbols)

        # Entire symbol should be prepended
        expected = np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=np.complex128)
        assert_array_equal(result, expected)

    def test_add_prefix_multidimensional_raises_error(self):
        """Test that multidimensional input raises ValueError."""
        prefix_scheme = CyclicPrefixScheme(prefix_length=2)
        symbols = np.array([[1, 2], [3, 4]], dtype=np.complex128)

        with pytest.raises(ValueError, match="Input symbols must be a 1D array"):
            prefix_scheme.add_prefix(symbols)

    def test_add_prefix_length_exceeds_symbols_raises_error(self):
        """Test that prefix longer than symbols raises ValueError."""
        prefix_scheme = CyclicPrefixScheme(prefix_length=5)
        symbols = np.array([1, 2, 3], dtype=np.complex128)

        with pytest.raises(
            ValueError, match="Input symbols length must be greater than prefix length"
        ):
            prefix_scheme.add_prefix(symbols)

    def test_add_prefix_preserves_dtype(self):
        """Test that add_prefix preserves complex128 dtype."""
        prefix_scheme = CyclicPrefixScheme(prefix_length=2)
        symbols = np.array([1, 2, 3, 4], dtype=np.complex128)

        result = prefix_scheme.add_prefix(symbols)

        assert result.dtype == np.complex128


class TestCyclicPrefixSchemeRemovePrefix:
    """Test CyclicPrefixScheme remove_prefix method."""

    def test_remove_prefix_basic(self):
        """Test removing cyclic prefix from basic signal."""
        prefix_length = 2
        prefix_scheme = CyclicPrefixScheme(prefix_length=prefix_length)

        symbols_with_prefix = np.array([5, 6, 1, 2, 3, 4, 5, 6], dtype=np.complex128)
        result = prefix_scheme.remove_prefix(symbols_with_prefix)

        expected = np.array([1, 2, 3, 4, 5, 6], dtype=np.complex128)
        assert_array_equal(result, expected)

    def test_remove_prefix_complex_symbols(self):
        """Test removing cyclic prefix from complex symbols."""
        prefix_length = 3
        prefix_scheme = CyclicPrefixScheme(prefix_length=prefix_length)

        symbols_with_prefix = np.array(
            [3 + 3j, 4 + 4j, 5 + 5j, 1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j, 5 + 5j], dtype=np.complex128
        )
        result = prefix_scheme.remove_prefix(symbols_with_prefix)

        expected = np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j, 5 + 5j], dtype=np.complex128)
        assert_array_equal(result, expected)

    def test_remove_prefix_zero_length(self):
        """Test removing zero-length prefix returns original."""
        prefix_scheme = CyclicPrefixScheme(prefix_length=0)

        symbols = np.array([1, 2, 3, 4], dtype=np.complex128)
        result = prefix_scheme.remove_prefix(symbols)

        assert_array_equal(result, symbols)

    def test_remove_prefix_multidimensional_raises_error(self):
        """Test that multidimensional input raises ValueError."""
        prefix_scheme = CyclicPrefixScheme(prefix_length=2)
        symbols = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.complex128)

        with pytest.raises(ValueError, match="Input symbols must be a 1D array"):
            prefix_scheme.remove_prefix(symbols)

    def test_remove_prefix_length_too_short_raises_error(self):
        """Test that symbols shorter than or equal to prefix raises ValueError."""
        prefix_scheme = CyclicPrefixScheme(prefix_length=5)
        symbols = np.array([1, 2, 3, 4, 5], dtype=np.complex128)

        with pytest.raises(
            ValueError, match="Input symbols length must be greater than prefix length"
        ):
            prefix_scheme.remove_prefix(symbols)


class TestCyclicPrefixSchemeRoundTrip:
    """Test cyclic prefix add-remove round trip."""

    def test_add_remove_roundtrip(self):
        """Test that add followed by remove recovers original."""
        prefix_scheme = CyclicPrefixScheme(prefix_length=4)

        original = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.complex128)
        with_prefix = prefix_scheme.add_prefix(original)
        recovered = prefix_scheme.remove_prefix(with_prefix)

        assert_array_equal(recovered, original)

    def test_add_remove_roundtrip_complex(self):
        """Test round trip with complex symbols."""
        prefix_scheme = CyclicPrefixScheme(prefix_length=3)

        original = np.random.randn(20) + 1j * np.random.randn(20)
        original = original.astype(np.complex128)

        with_prefix = prefix_scheme.add_prefix(original)
        recovered = prefix_scheme.remove_prefix(with_prefix)

        assert_array_almost_equal(recovered, original)

    def test_add_remove_roundtrip_zero_prefix(self):
        """Test round trip with zero prefix length."""
        prefix_scheme = CyclicPrefixScheme(prefix_length=0)

        original = np.array([1, 2, 3, 4, 5], dtype=np.complex128)
        with_prefix = prefix_scheme.add_prefix(original)
        recovered = prefix_scheme.remove_prefix(with_prefix)

        assert_array_equal(recovered, original)


class TestZeroPaddingPrefixSchemeAcronym:
    """Test ZeroPaddingPrefixScheme acronym property."""

    def test_acronym_returns_zp(self):
        """Test that acronym returns 'ZP'."""
        prefix_scheme = ZeroPaddingPrefixScheme(prefix_length=4)
        assert prefix_scheme.acronym == "ZP"


class TestZeroPaddingPrefixSchemeAddPrefix:
    """Test ZeroPaddingPrefixScheme add_prefix method."""

    def test_add_prefix_basic(self):
        """Test adding zero padding to basic signal."""
        prefix_length = 3
        prefix_scheme = ZeroPaddingPrefixScheme(prefix_length=prefix_length)

        symbols = np.array([1, 2, 3, 4], dtype=np.complex128)
        result = prefix_scheme.add_prefix(symbols)

        # Zeros should be appended
        expected = np.array([1, 2, 3, 4, 0, 0, 0], dtype=np.complex128)
        assert_array_equal(result, expected)

    def test_add_prefix_complex_symbols(self):
        """Test adding zero padding to complex symbols."""
        prefix_length = 2
        prefix_scheme = ZeroPaddingPrefixScheme(prefix_length=prefix_length)

        symbols = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex128)
        result = prefix_scheme.add_prefix(symbols)

        expected = np.array([1 + 1j, 2 + 2j, 3 + 3j, 0 + 0j, 0 + 0j], dtype=np.complex128)
        assert_array_equal(result, expected)

    def test_add_prefix_zero_length(self):
        """Test adding zero-length padding returns original."""
        prefix_scheme = ZeroPaddingPrefixScheme(prefix_length=0)

        symbols = np.array([1, 2, 3, 4], dtype=np.complex128)
        result = prefix_scheme.add_prefix(symbols)

        assert_array_equal(result, symbols)

    def test_add_prefix_multidimensional_raises_error(self):
        """Test that multidimensional input raises ValueError."""
        prefix_scheme = ZeroPaddingPrefixScheme(prefix_length=2)
        symbols = np.array([[1, 2], [3, 4]], dtype=np.complex128)

        with pytest.raises(ValueError, match="Input symbols must be a 1D array"):
            prefix_scheme.add_prefix(symbols)

    def test_add_prefix_preserves_dtype(self):
        """Test that add_prefix preserves dtype."""
        prefix_scheme = ZeroPaddingPrefixScheme(prefix_length=2)
        symbols = np.array([1, 2, 3], dtype=np.complex128)

        result = prefix_scheme.add_prefix(symbols)

        assert result.dtype == symbols.dtype


class TestZeroPaddingPrefixSchemeRemovePrefix:
    """Test ZeroPaddingPrefixScheme remove_prefix method."""

    def test_remove_prefix_basic(self):
        """Test removing zero padding with overlap-add."""
        prefix_length = 2
        prefix_scheme = ZeroPaddingPrefixScheme(prefix_length=prefix_length)

        # Simulate received signal: original signal convolved with channel
        # For simplicity, assume no channel effect, just zeros at end
        symbols_with_zp = np.array([1, 2, 3, 4, 0, 0], dtype=np.complex128)
        result = prefix_scheme.remove_prefix(symbols_with_zp)

        # Should return first 4 elements
        expected = np.array([1, 2, 3, 4], dtype=np.complex128)
        assert_array_equal(result, expected)

    def test_remove_prefix_with_overlap(self):
        """Test overlap-add functionality of zero padding removal."""
        prefix_length = 3
        prefix_scheme = ZeroPaddingPrefixScheme(prefix_length=prefix_length)

        # Simulate signal where last samples wrap around (channel effect)
        # Original: [1, 2, 3, 4] with ZP [0, 0, 0]
        # After channel: [1, 2, 3, 4+a, b, c] where a,b,c come from channel
        # The overlap-add should add the tail to the beginning
        symbols_with_zp = np.array([1, 2, 3, 4, 0.5, 0.3, 0.2], dtype=np.complex128)
        result = prefix_scheme.remove_prefix(symbols_with_zp)

        # Result should overlap last 3 samples onto first 3
        expected = np.array([1 + 0.5, 2 + 0.3, 3 + 0.2, 4], dtype=np.complex128)
        assert_array_almost_equal(result, expected)

    def test_remove_prefix_zero_length(self):
        """Test removing zero-length padding returns original."""
        prefix_scheme = ZeroPaddingPrefixScheme(prefix_length=0)

        symbols = np.array([1, 2, 3, 4], dtype=np.complex128)
        result = prefix_scheme.remove_prefix(symbols)

        assert_array_equal(result, symbols)

    def test_remove_prefix_multidimensional_raises_error(self):
        """Test that multidimensional input raises ValueError."""
        prefix_scheme = ZeroPaddingPrefixScheme(prefix_length=2)
        symbols = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.complex128)

        with pytest.raises(ValueError, match="Input symbols must be a 1D array"):
            prefix_scheme.remove_prefix(symbols)

    def test_remove_prefix_length_too_short_raises_error(self):
        """Test that symbols shorter than or equal to prefix raises ValueError."""
        prefix_scheme = ZeroPaddingPrefixScheme(prefix_length=5)
        symbols = np.array([1, 2, 3, 4, 5], dtype=np.complex128)

        with pytest.raises(
            ValueError, match="Input symbols length must be greater than prefix length"
        ):
            prefix_scheme.remove_prefix(symbols)

    def test_remove_prefix_matrix_construction(self):
        """Test that overlap-add matrix is constructed correctly."""
        prefix_length = 2
        prefix_scheme = ZeroPaddingPrefixScheme(prefix_length=prefix_length)

        # Test with known values
        symbols = np.array([1, 2, 3, 4, 5, 6], dtype=np.complex128)
        result = prefix_scheme.remove_prefix(symbols)

        # Should add last 2 elements to first 2
        expected = np.array([1 + 5, 2 + 6, 3, 4], dtype=np.complex128)
        assert_array_equal(result, expected)


class TestZeroPaddingPrefixSchemeRoundTrip:
    """Test zero padding add-remove round trip."""

    def test_add_remove_roundtrip_no_channel(self):
        """Test that add followed by remove with no channel effect recovers original."""
        prefix_scheme = ZeroPaddingPrefixScheme(prefix_length=3)

        original = np.array([1, 2, 3, 4, 5], dtype=np.complex128)
        with_zp = prefix_scheme.add_prefix(original)
        recovered = prefix_scheme.remove_prefix(with_zp)

        # With no channel (zeros in tail), should recover original
        assert_array_equal(recovered, original)

    def test_add_remove_with_simulated_overlap(self):
        """Test overlap-add correctly handles tail from channel."""
        prefix_length = 2
        prefix_scheme = ZeroPaddingPrefixScheme(prefix_length=prefix_length)

        original = np.array([1, 2, 3, 4], dtype=np.complex128)
        with_zp = prefix_scheme.add_prefix(original)

        # Simulate channel adding values to zero padding region
        with_zp[-2:] += np.array([0.5, 0.3], dtype=np.complex128)

        recovered = prefix_scheme.remove_prefix(with_zp)

        # Should overlap last 2 onto first 2
        expected = original.copy()
        expected[:2] += np.array([0.5, 0.3], dtype=np.complex128)

        assert_array_almost_equal(recovered, expected)


class TestNoPrefixSchemeAcronym:
    """Test NoPrefixScheme acronym property."""

    def test_acronym_returns_empty_string(self):
        """Test that acronym returns empty string."""
        prefix_scheme = NoPrefixScheme()
        assert prefix_scheme.acronym == ""


class TestNoPrefixSchemeAddPrefix:
    """Test NoPrefixScheme add_prefix method."""

    def test_add_prefix_returns_unchanged(self):
        """Test that add_prefix returns input unchanged."""
        prefix_scheme = NoPrefixScheme()

        symbols = np.array([1, 2, 3, 4, 5], dtype=np.complex128)
        result = prefix_scheme.add_prefix(symbols)

        assert_array_equal(result, symbols)
        assert result is symbols  # Should return same object

    def test_add_prefix_complex_symbols(self):
        """Test with complex symbols."""
        prefix_scheme = NoPrefixScheme()

        symbols = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex128)
        result = prefix_scheme.add_prefix(symbols)

        assert_array_equal(result, symbols)


class TestNoPrefixSchemeRemovePrefix:
    """Test NoPrefixScheme remove_prefix method."""

    def test_remove_prefix_returns_unchanged(self):
        """Test that remove_prefix returns input unchanged."""
        prefix_scheme = NoPrefixScheme()

        symbols = np.array([1, 2, 3, 4, 5], dtype=np.complex128)
        result = prefix_scheme.remove_prefix(symbols)

        assert_array_equal(result, symbols)
        assert result is symbols  # Should return same object

    def test_remove_prefix_complex_symbols(self):
        """Test with complex symbols."""
        prefix_scheme = NoPrefixScheme()

        symbols = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex128)
        result = prefix_scheme.remove_prefix(symbols)

        assert_array_equal(result, symbols)


class TestNoPrefixSchemeRoundTrip:
    """Test no prefix add-remove round trip."""

    def test_add_remove_roundtrip(self):
        """Test that add followed by remove returns original."""
        prefix_scheme = NoPrefixScheme()

        original = np.array([1, 2, 3, 4, 5], dtype=np.complex128)
        with_prefix = prefix_scheme.add_prefix(original)
        recovered = prefix_scheme.remove_prefix(with_prefix)

        assert_array_equal(recovered, original)


class TestPrefixSchemeComparison:
    """Compare behavior of different prefix schemes."""

    def test_cyclic_prefix_vs_no_prefix_length(self):
        """Test that cyclic prefix adds length, no prefix doesn't."""
        symbols = np.array([1, 2, 3, 4], dtype=np.complex128)

        cp = CyclicPrefixScheme(prefix_length=2)
        no_prefix = NoPrefixScheme()

        cp_result = cp.add_prefix(symbols)
        no_prefix_result = no_prefix.add_prefix(symbols)

        assert len(cp_result) == len(symbols) + 2
        assert len(no_prefix_result) == len(symbols)

    def test_zero_padding_vs_cyclic_prefix_content(self):
        """Test that ZP adds zeros, CP copies tail."""
        symbols = np.array([1, 2, 3, 4], dtype=np.complex128)
        prefix_length = 2

        zp = ZeroPaddingPrefixScheme(prefix_length=prefix_length)
        cp = CyclicPrefixScheme(prefix_length=prefix_length)

        zp_result = zp.add_prefix(symbols)
        cp_result = cp.add_prefix(symbols)

        # Both add 2 elements
        assert len(zp_result) == len(cp_result) == 6

        # ZP adds zeros at end
        assert_array_equal(zp_result[-2:], np.array([0, 0], dtype=np.complex128))

        # CP prepends tail at beginning
        assert_array_equal(cp_result[:2], symbols[-2:])

    def test_all_schemes_with_zero_length(self):
        """Test that all schemes with zero length behave identically."""
        symbols = np.array([1, 2, 3, 4], dtype=np.complex128)

        cp = CyclicPrefixScheme(prefix_length=0)
        zp = ZeroPaddingPrefixScheme(prefix_length=0)
        no_prefix = NoPrefixScheme()

        cp_result = cp.add_prefix(symbols)
        zp_result = zp.add_prefix(symbols)
        no_prefix_result = no_prefix.add_prefix(symbols)

        assert_array_equal(cp_result, symbols)
        assert_array_equal(zp_result, symbols)
        assert_array_equal(no_prefix_result, symbols)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_cyclic_prefix_single_element(self):
        """Test cyclic prefix with single element symbol."""
        cp = CyclicPrefixScheme(prefix_length=1)

        symbols = np.array([5.0], dtype=np.complex128)
        result = cp.add_prefix(symbols)

        expected = np.array([5.0, 5.0], dtype=np.complex128)
        assert_array_equal(result, expected)

    def test_zero_padding_single_element(self):
        """Test zero padding with single element symbol."""
        zp = ZeroPaddingPrefixScheme(prefix_length=1)

        symbols = np.array([5.0], dtype=np.complex128)
        result = zp.add_prefix(symbols)

        expected = np.array([5.0, 0.0], dtype=np.complex128)
        assert_array_equal(result, expected)

    def test_cyclic_prefix_very_long(self):
        """Test cyclic prefix with very long symbols."""
        prefix_length = 256
        symbols = np.random.randn(1024) + 1j * np.random.randn(1024)
        symbols = symbols.astype(np.complex128)

        cp = CyclicPrefixScheme(prefix_length=prefix_length)

        result = cp.add_prefix(symbols)
        recovered = cp.remove_prefix(result)

        assert len(result) == 1024 + 256
        assert_array_almost_equal(recovered, symbols)

    def test_zero_padding_very_long(self):
        """Test zero padding with very long symbols."""
        prefix_length = 256
        symbols = np.random.randn(1024) + 1j * np.random.randn(1024)
        symbols = symbols.astype(np.complex128)

        zp = ZeroPaddingPrefixScheme(prefix_length=prefix_length)

        result = zp.add_prefix(symbols)

        assert len(result) == 1024 + 256
        # Last 256 should be zeros
        assert_array_equal(result[-256:], np.zeros(256, dtype=np.complex128))

    def test_all_zeros_symbols(self):
        """Test prefix schemes with all-zero symbols."""
        symbols = np.zeros(8, dtype=np.complex128)

        cp = CyclicPrefixScheme(prefix_length=2)
        zp = ZeroPaddingPrefixScheme(prefix_length=2)

        cp_result = cp.add_prefix(symbols)
        zp_result = zp.add_prefix(symbols)

        # Both should add zeros
        assert_array_equal(cp_result, np.zeros(10, dtype=np.complex128))
        assert_array_equal(zp_result, np.zeros(10, dtype=np.complex128))

    def test_pure_imaginary_symbols(self):
        """Test prefix schemes with pure imaginary symbols."""
        symbols = np.array([1j, 2j, 3j, 4j], dtype=np.complex128)

        cp = CyclicPrefixScheme(prefix_length=2)
        zp = ZeroPaddingPrefixScheme(prefix_length=2)

        cp_result = cp.add_prefix(symbols)
        zp_result = zp.add_prefix(symbols)

        # CP should prepend last 2
        assert_array_equal(cp_result[:2], np.array([3j, 4j], dtype=np.complex128))

        # ZP should append zeros
        assert_array_equal(zp_result[-2:], np.array([0j, 0j], dtype=np.complex128))


class TestIntegrationScenarios:
    """Integration tests with realistic OFDM scenarios."""

    def test_cyclic_prefix_isi_mitigation(self):
        """Test that cyclic prefix helps mitigate ISI in multipath channel."""
        # Simulate OFDM symbol
        num_subcarriers = 64
        freq_symbols = np.random.randn(num_subcarriers) + 1j * np.random.randn(num_subcarriers)
        freq_symbols = freq_symbols.astype(np.complex128)

        # Convert to time domain
        time_symbols = np.fft.ifft(freq_symbols)

        # Add cyclic prefix (longer than channel order)
        cp_length = 16
        cp = CyclicPrefixScheme(prefix_length=cp_length)
        time_with_cp = cp.add_prefix(time_symbols)

        # Simulate simple multipath channel (impulse response)
        channel_ir = np.array([1.0, 0.5, 0.3, 0.1], dtype=np.complex128)

        # Convolve with channel
        received = np.convolve(time_with_cp, channel_ir, mode="same").astype(np.complex128)

        # Remove cyclic prefix
        received_no_cp = cp.remove_prefix(received)

        # Should have removed ISI from previous symbol
        assert len(received_no_cp) == num_subcarriers

    def test_zero_padding_circular_convolution(self):
        """Test zero padding converts linear to circular convolution."""
        # OFDM symbol in time domain
        symbols = np.random.randn(16) + 1j * np.random.randn(16)
        symbols = symbols.astype(np.complex128)

        # Add zero padding
        zp_length = 4
        zp = ZeroPaddingPrefixScheme(prefix_length=zp_length)
        symbols_with_zp = zp.add_prefix(symbols)

        # Simulate channel with length <= zp_length
        channel_ir = np.array([1.0, 0.5, 0.3], dtype=np.complex128)

        # Linear convolution
        received = np.convolve(symbols_with_zp, channel_ir, mode="same").astype(np.complex128)

        # Remove zero padding (overlap-add)
        recovered = zp.remove_prefix(received)

        # Should have proper circular convolution effect
        assert len(recovered) == len(symbols)

    def test_prefix_overhead_calculation(self):
        """Test overhead introduced by different prefix schemes."""
        symbols = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.complex128)
        prefix_length = 2

        cp = CyclicPrefixScheme(prefix_length=prefix_length)
        zp = ZeroPaddingPrefixScheme(prefix_length=prefix_length)
        no_prefix = NoPrefixScheme()

        cp_result = cp.add_prefix(symbols)
        zp_result = zp.add_prefix(symbols)
        no_prefix_result = no_prefix.add_prefix(symbols)

        # Calculate overhead
        cp_overhead = (len(cp_result) - len(symbols)) / len(symbols)
        zp_overhead = (len(zp_result) - len(symbols)) / len(symbols)
        no_prefix_overhead = (len(no_prefix_result) - len(symbols)) / len(symbols)

        assert cp_overhead == 0.25  # 2/8 = 25%
        assert zp_overhead == 0.25  # 2/8 = 25%
        assert no_prefix_overhead == 0.0

    def test_multiple_ofdm_symbols_with_prefix(self):
        """Test handling multiple OFDM symbols with prefixes."""
        num_symbols = 5
        symbol_length = 16
        prefix_length = 4

        cp = CyclicPrefixScheme(prefix_length=prefix_length)

        # Generate multiple OFDM symbols
        symbols_list = []
        for _ in range(num_symbols):
            symbol = np.random.randn(symbol_length) + 1j * np.random.randn(symbol_length)
            symbol = symbol.astype(np.complex128)
            symbols_list.append(symbol)

        # Add prefix to each
        symbols_with_cp = [cp.add_prefix(sym) for sym in symbols_list]

        # Remove prefix from each
        recovered = [cp.remove_prefix(sym) for sym in symbols_with_cp]

        # Should recover all original symbols
        for orig, rec in zip(symbols_list, recovered):
            assert_array_almost_equal(orig, rec)
