"""
Unit tests for serial_parallel models.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from ofdm_based_systems.serial_parallel.models import SerialToParallelConverter


class TestSerialToParallelConverterToParallel:
    """Test SerialToParallelConverter.to_parallel method."""

    def test_to_parallel_basic(self):
        """Test basic serial to parallel conversion."""
        data = np.array([1, 2, 3, 4, 5, 6], dtype=np.complex128)
        num_streams = 2

        result = SerialToParallelConverter.to_parallel(data, num_streams)

        expected = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.complex128)
        assert_array_equal(result, expected)

    def test_to_parallel_single_stream(self):
        """Test conversion with single stream (column vector)."""
        data = np.array([1, 2, 3, 4], dtype=np.complex128)
        num_streams = 1

        result = SerialToParallelConverter.to_parallel(data, num_streams)

        expected = np.array([[1], [2], [3], [4]], dtype=np.complex128)
        assert_array_equal(result, expected)

    def test_to_parallel_all_streams(self):
        """Test conversion where num_streams equals data length (single row)."""
        data = np.array([1, 2, 3, 4], dtype=np.complex128)
        num_streams = 4

        result = SerialToParallelConverter.to_parallel(data, num_streams)

        expected = np.array([[1, 2, 3, 4]], dtype=np.complex128)
        assert_array_equal(result, expected)

    def test_to_parallel_three_streams(self):
        """Test conversion with three streams."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.complex128)
        num_streams = 3

        result = SerialToParallelConverter.to_parallel(data, num_streams)

        expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.complex128)
        assert_array_equal(result, expected)

    def test_to_parallel_complex_values(self):
        """Test conversion with complex values."""
        data = np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=np.complex128)
        num_streams = 2

        result = SerialToParallelConverter.to_parallel(data, num_streams)

        expected = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=np.complex128)
        assert_array_equal(result, expected)

    def test_to_parallel_large_data(self):
        """Test conversion with large data array."""
        data_size = 1024
        num_streams = 64
        data = np.arange(data_size, dtype=np.complex128)

        result = SerialToParallelConverter.to_parallel(data, num_streams)

        assert result.shape == (data_size // num_streams, num_streams)
        # Verify first and last elements
        assert result[0, 0] == 0
        assert result[-1, -1] == data_size - 1

    def test_to_parallel_preserves_dtype(self):
        """Test that conversion preserves complex128 dtype."""
        data = np.array([1, 2, 3, 4], dtype=np.complex128)
        num_streams = 2

        result = SerialToParallelConverter.to_parallel(data, num_streams)

        assert result.dtype == np.complex128

    def test_to_parallel_multidimensional_raises_error(self):
        """Test that multidimensional input raises ValueError."""
        data = np.array([[1, 2], [3, 4]], dtype=np.complex128)
        num_streams = 2

        with pytest.raises(ValueError, match="Input data must be a 1D array"):
            SerialToParallelConverter.to_parallel(data, num_streams)

    def test_to_parallel_zero_streams_raises_error(self):
        """Test that zero streams raises ValueError."""
        data = np.array([1, 2, 3, 4], dtype=np.complex128)
        num_streams = 0

        with pytest.raises(ValueError, match="Number of streams must be a positive integer"):
            SerialToParallelConverter.to_parallel(data, num_streams)

    def test_to_parallel_negative_streams_raises_error(self):
        """Test that negative streams raises ValueError."""
        data = np.array([1, 2, 3, 4], dtype=np.complex128)
        num_streams = -2

        with pytest.raises(ValueError, match="Number of streams must be a positive integer"):
            SerialToParallelConverter.to_parallel(data, num_streams)

    def test_to_parallel_indivisible_length_raises_error(self):
        """Test that data length not divisible by num_streams raises ValueError."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.complex128)
        num_streams = 2

        with pytest.raises(
            ValueError, match="Length of data must be divisible by number of streams"
        ):
            SerialToParallelConverter.to_parallel(data, num_streams)

    def test_to_parallel_ordering(self):
        """Test that elements are ordered correctly in parallel form."""
        data = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.complex128)
        num_streams = 4

        result = SerialToParallelConverter.to_parallel(data, num_streams)

        # First row should have elements 0-3
        assert_array_equal(result[0, :], np.array([0, 1, 2, 3], dtype=np.complex128))
        # Second row should have elements 4-7
        assert_array_equal(result[1, :], np.array([4, 5, 6, 7], dtype=np.complex128))


class TestSerialToParallelConverterToSerial:
    """Test SerialToParallelConverter.to_serial method."""

    def test_to_serial_basic(self):
        """Test basic parallel to serial conversion."""
        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.complex128)

        result = SerialToParallelConverter.to_serial(data)

        expected = np.array([1, 2, 3, 4, 5, 6], dtype=np.complex128)
        assert_array_equal(result, expected)

    def test_to_serial_single_column(self):
        """Test conversion from single column."""
        data = np.array([[1], [2], [3], [4]], dtype=np.complex128)

        result = SerialToParallelConverter.to_serial(data)

        expected = np.array([1, 2, 3, 4], dtype=np.complex128)
        assert_array_equal(result, expected)

    def test_to_serial_single_row(self):
        """Test conversion from single row."""
        data = np.array([[1, 2, 3, 4]], dtype=np.complex128)

        result = SerialToParallelConverter.to_serial(data)

        expected = np.array([1, 2, 3, 4], dtype=np.complex128)
        assert_array_equal(result, expected)

    def test_to_serial_complex_values(self):
        """Test conversion with complex values."""
        data = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=np.complex128)

        result = SerialToParallelConverter.to_serial(data)

        expected = np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=np.complex128)
        assert_array_equal(result, expected)

    def test_to_serial_large_data(self):
        """Test conversion with large data array."""
        rows = 16
        cols = 64
        data = np.arange(rows * cols, dtype=np.complex128).reshape(rows, cols)

        result = SerialToParallelConverter.to_serial(data)

        assert result.shape == (rows * cols,)
        assert result[0] == 0
        assert result[-1] == rows * cols - 1

    def test_to_serial_preserves_dtype(self):
        """Test that conversion preserves complex128 dtype."""
        data = np.array([[1, 2], [3, 4]], dtype=np.complex128)

        result = SerialToParallelConverter.to_serial(data)

        assert result.dtype == np.complex128

    def test_to_serial_one_dimensional_raises_error(self):
        """Test that 1D input raises ValueError."""
        data = np.array([1, 2, 3, 4], dtype=np.complex128)

        with pytest.raises(ValueError, match="Input data must be a 2D array"):
            SerialToParallelConverter.to_serial(data)

    def test_to_serial_three_dimensional_raises_error(self):
        """Test that 3D input raises ValueError."""
        data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.complex128)

        with pytest.raises(ValueError, match="Input data must be a 2D array"):
            SerialToParallelConverter.to_serial(data)

    def test_to_serial_ordering(self):
        """Test that elements are ordered correctly in serial form (row-major)."""
        data = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.complex128)

        result = SerialToParallelConverter.to_serial(data)

        # Should flatten in row-major order (C-style)
        expected = np.array([0, 1, 2, 3, 4, 5], dtype=np.complex128)
        assert_array_equal(result, expected)


class TestSerialToParallelConverterRoundTrip:
    """Test round-trip conversions."""

    def test_serial_to_parallel_to_serial(self):
        """Test that serial->parallel->serial recovers original."""
        original = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.complex128)
        num_streams = 4

        parallel = SerialToParallelConverter.to_parallel(original, num_streams)
        recovered = SerialToParallelConverter.to_serial(parallel)

        assert_array_equal(recovered, original)

    def test_parallel_to_serial_to_parallel(self):
        """Test that parallel->serial->parallel recovers original."""
        original = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.complex128)
        num_streams = 3

        serial = SerialToParallelConverter.to_serial(original)
        recovered = SerialToParallelConverter.to_parallel(serial, num_streams)

        assert_array_equal(recovered, original)

    def test_roundtrip_single_stream(self):
        """Test round trip with single stream."""
        original = np.array([1, 2, 3, 4, 5], dtype=np.complex128)
        num_streams = 1

        parallel = SerialToParallelConverter.to_parallel(original, num_streams)
        recovered = SerialToParallelConverter.to_serial(parallel)

        assert_array_equal(recovered, original)

    def test_roundtrip_complex_random(self):
        """Test round trip with random complex data."""
        np.random.seed(42)
        original = np.random.randn(100) + 1j * np.random.randn(100)
        original = original.astype(np.complex128)
        num_streams = 10

        parallel = SerialToParallelConverter.to_parallel(original, num_streams)
        recovered = SerialToParallelConverter.to_serial(parallel)

        assert_array_almost_equal(recovered, original)

    def test_roundtrip_all_streams(self):
        """Test round trip where num_streams equals data length."""
        original = np.array([1, 2, 3, 4, 5], dtype=np.complex128)
        num_streams = 5

        parallel = SerialToParallelConverter.to_parallel(original, num_streams)
        assert parallel.shape == (1, 5)

        recovered = SerialToParallelConverter.to_serial(parallel)
        assert_array_equal(recovered, original)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_to_parallel_two_elements_one_stream(self):
        """Test minimal conversion with two elements, one stream."""
        data = np.array([1, 2], dtype=np.complex128)
        num_streams = 1

        result = SerialToParallelConverter.to_parallel(data, num_streams)

        expected = np.array([[1], [2]], dtype=np.complex128)
        assert_array_equal(result, expected)

    def test_to_parallel_two_elements_two_streams(self):
        """Test minimal conversion with two elements, two streams."""
        data = np.array([1, 2], dtype=np.complex128)
        num_streams = 2

        result = SerialToParallelConverter.to_parallel(data, num_streams)

        expected = np.array([[1, 2]], dtype=np.complex128)
        assert_array_equal(result, expected)

    def test_to_serial_minimal(self):
        """Test minimal serial conversion."""
        data = np.array([[1]], dtype=np.complex128)

        result = SerialToParallelConverter.to_serial(data)

        expected = np.array([1], dtype=np.complex128)
        assert_array_equal(result, expected)

    def test_all_zeros(self):
        """Test with all-zero data."""
        data = np.zeros(8, dtype=np.complex128)
        num_streams = 4

        parallel = SerialToParallelConverter.to_parallel(data, num_streams)
        serial = SerialToParallelConverter.to_serial(parallel)

        assert_array_equal(parallel, np.zeros((2, 4), dtype=np.complex128))
        assert_array_equal(serial, data)

    def test_pure_imaginary(self):
        """Test with pure imaginary values."""
        data = np.array([1j, 2j, 3j, 4j, 5j, 6j], dtype=np.complex128)
        num_streams = 3

        parallel = SerialToParallelConverter.to_parallel(data, num_streams)
        serial = SerialToParallelConverter.to_serial(parallel)

        expected_parallel = np.array([[1j, 2j, 3j], [4j, 5j, 6j]], dtype=np.complex128)
        assert_array_equal(parallel, expected_parallel)
        assert_array_equal(serial, data)

    def test_very_large_array(self):
        """Test with very large array."""
        size = 10000
        num_streams = 100
        data = np.arange(size, dtype=np.complex128)

        parallel = SerialToParallelConverter.to_parallel(data, num_streams)
        serial = SerialToParallelConverter.to_serial(parallel)

        assert parallel.shape == (size // num_streams, num_streams)
        assert_array_equal(serial, data)


class TestIntegrationScenarios:
    """Integration tests with realistic OFDM scenarios."""

    def test_ofdm_symbol_parallelization(self):
        """Test parallelizing OFDM symbols for multiple subcarriers."""
        # Simulate QAM symbols for multiple OFDM symbols
        num_subcarriers = 64
        num_ofdm_symbols = 10
        total_symbols = num_subcarriers * num_ofdm_symbols

        symbols = np.random.randn(total_symbols) + 1j * np.random.randn(total_symbols)
        symbols = symbols.astype(np.complex128)

        # Convert to parallel: each row is one OFDM symbol with 64 subcarriers
        parallel_symbols = SerialToParallelConverter.to_parallel(symbols, num_subcarriers)

        assert parallel_symbols.shape == (num_ofdm_symbols, num_subcarriers)

        # Verify we can recover original
        recovered = SerialToParallelConverter.to_serial(parallel_symbols)
        assert_array_almost_equal(recovered, symbols)

    def test_mimo_stream_separation(self):
        """Test separating data into MIMO spatial streams."""
        # Simulate data for 4 spatial streams
        num_spatial_streams = 4
        symbols_per_stream = 256
        total_symbols = num_spatial_streams * symbols_per_stream

        data = np.random.randn(total_symbols) + 1j * np.random.randn(total_symbols)
        data = data.astype(np.complex128)

        # Convert to parallel: each column is one spatial stream
        parallel_data = SerialToParallelConverter.to_parallel(data, num_spatial_streams)

        assert parallel_data.shape == (symbols_per_stream, num_spatial_streams)

        # Each spatial stream should have correct length
        for stream_idx in range(num_spatial_streams):
            stream = parallel_data[:, stream_idx]
            assert len(stream) == symbols_per_stream

    def test_frequency_domain_processing(self):
        """Test organizing frequency domain samples for IFFT processing."""
        # Simulate frequency domain samples
        num_subcarriers = 128
        num_blocks = 8
        freq_samples = np.random.randn(num_blocks * num_subcarriers) + 1j * np.random.randn(
            num_blocks * num_subcarriers
        )
        freq_samples = freq_samples.astype(np.complex128)

        # Organize into blocks for IFFT
        blocks = SerialToParallelConverter.to_parallel(freq_samples, num_subcarriers)

        # Should have one block per row
        assert blocks.shape == (num_blocks, num_subcarriers)

        # Simulate IFFT on each block (placeholder)
        time_blocks = np.fft.ifft(blocks, axis=1)

        # Flatten back to serial
        time_serial = SerialToParallelConverter.to_serial(time_blocks)
        assert len(time_serial) == num_blocks * num_subcarriers

    def test_interleaving_pattern(self):
        """Test that parallelization maintains correct interleaving."""
        # Create data with known pattern
        data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.complex128)
        num_streams = 4

        parallel = SerialToParallelConverter.to_parallel(data, num_streams)

        # Verify pattern:
        # Row 0: [0, 1, 2, 3]
        # Row 1: [4, 5, 6, 7]
        # Row 2: [8, 9, 10, 11]
        assert parallel[0, 0] == 0
        assert parallel[0, 1] == 1
        assert parallel[1, 0] == 4
        assert parallel[2, 3] == 11

        # Verify serialization maintains order
        serial = SerialToParallelConverter.to_serial(parallel)
        assert_array_equal(serial, data)

    def test_block_processing_pipeline(self):
        """Test a complete block processing pipeline."""
        # Simulate a stream of coded bits -> symbols -> OFDM blocks
        num_subcarriers = 64
        num_blocks = 20

        # Generate random symbols
        symbols = np.random.randn(num_blocks * num_subcarriers) + 1j * np.random.randn(
            num_blocks * num_subcarriers
        )
        symbols = symbols.astype(np.complex128)

        # Step 1: Organize into OFDM blocks
        blocks = SerialToParallelConverter.to_parallel(symbols, num_subcarriers)
        assert blocks.shape == (num_blocks, num_subcarriers)

        # Step 2: Process each block (e.g., add pilot symbols, IFFT, etc.)
        # Here we just multiply by a scaling factor as placeholder
        processed_blocks = blocks * 0.5

        # Step 3: Serialize for transmission
        tx_symbols = SerialToParallelConverter.to_serial(processed_blocks)
        assert len(tx_symbols) == num_blocks * num_subcarriers

        # Step 4: At receiver, reorganize into blocks
        rx_blocks = SerialToParallelConverter.to_parallel(tx_symbols, num_subcarriers)

        # Step 5: Verify correct recovery
        assert_array_almost_equal(rx_blocks, processed_blocks)
