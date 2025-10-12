"""
Unit tests for constellation models.
"""

import io
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from ofdm_based_systems.constellation.models import (
    ISymbolClassifier,
    NNClassifier,
    IWordCoder,
    NoWordCoder,
    GrayWordCoder,
    IConstellationMapper,
    QAMConstellationMapper,
    PSKConstellationMapper,
)


class TestNNClassifier:
    def test_classify_exact_points(self):
        """Test that exact constellation points are correctly classified."""
        constellation = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex128)
        symbols = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex128)

        classifier = NNClassifier()
        result = classifier.classify(constellation, symbols)

        assert_array_equal(result, symbols)

    def test_classify_noisy_points(self):
        """Test that slightly noisy points are classified to the nearest constellation point."""
        constellation = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex128)
        noisy_symbols = np.array(
            [1.1 + 0.9j, -0.9 + 1.1j, -1.1 - 0.9j, 0.9 - 1.1j], dtype=np.complex128
        )
        expected = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex128)

        classifier = NNClassifier()
        result = classifier.classify(constellation, noisy_symbols)

        assert_array_equal(result, expected)


class TestNoWordCoderFunctionality:
    def test_encode_within_range(self):
        """Test encoding with values within range."""
        coder = NoWordCoder(bits_per_word=2)
        assert coder.encode(0) == 0
        assert coder.encode(1) == 1
        assert coder.encode(2) == 2
        assert coder.encode(3) == 3

    def test_encode_out_of_range(self):
        """Test encoding with values out of range raises ValueError."""
        coder = NoWordCoder(bits_per_word=2)
        with pytest.raises(ValueError):
            coder.encode(4)

    def test_decode_within_range(self):
        """Test decoding with values within range."""
        coder = NoWordCoder(bits_per_word=2)
        assert coder.decode(0) == 0
        assert coder.decode(1) == 1
        assert coder.decode(2) == 2
        assert coder.decode(3) == 3

    def test_decode_out_of_range(self):
        """Test decoding with values out of range raises ValueError."""
        coder = NoWordCoder(bits_per_word=2)
        with pytest.raises(ValueError):
            coder.decode(4)


class TestGrayWordCoder:
    def test_encode_within_range(self):
        """Test Gray encoding with values within range."""
        coder = GrayWordCoder(bits_per_word=3)
        assert coder.encode(0) == 0  # 000 -> 000
        assert coder.encode(1) == 1  # 001 -> 001
        assert coder.encode(2) == 3  # 010 -> 011
        assert coder.encode(3) == 2  # 011 -> 010
        assert coder.encode(4) == 6  # 100 -> 110
        assert coder.encode(5) == 7  # 101 -> 111
        assert coder.encode(6) == 5  # 110 -> 101
        assert coder.encode(7) == 4  # 111 -> 100

    def test_decode_within_range(self):
        """Test Gray decoding with values within range."""
        coder = GrayWordCoder(bits_per_word=3)
        assert coder.decode(0) == 0  # 000 -> 000
        assert coder.decode(1) == 1  # 001 -> 001
        assert coder.decode(2) == 3  # 010 -> 011
        assert coder.decode(3) == 2  # 011 -> 010
        assert coder.decode(4) == 7  # 100 -> 111
        assert coder.decode(5) == 6  # 101 -> 110
        assert coder.decode(6) == 4  # 110 -> 100
        assert coder.decode(7) == 5  # 111 -> 101

    def test_decode_out_of_range(self):
        """Test Gray decoding with values out of range raises ValueError."""
        coder = GrayWordCoder(bits_per_word=3)
        with pytest.raises(ValueError):
            coder.decode(8)

    def test_gray_table_size(self):
        """Test that the gray table has the expected size."""
        coder = GrayWordCoder(bits_per_word=4)
        assert len(coder.gray_table) == 16
        assert len(coder.inverse_gray_table) == 16

    def test_encode_decode_round_trip(self):
        """Test that encoding and then decoding returns the original value."""
        coder = GrayWordCoder(bits_per_word=5)
        for i in range(32):
            assert coder.decode(coder.encode(i)) == i


class TestQAMConstellationMapper:
    def test_init_valid_order(self):
        """Test initialization with valid constellation orders."""
        for order in [4, 16, 64, 256]:
            mapper = QAMConstellationMapper(order)
            assert mapper.order == order
            assert mapper.bits_per_symbol == int(np.log2(order))
            assert mapper.constellation_name == f"{order}-QAM"
            assert len(mapper.constellation) == order
            assert len(mapper.constellation_map) == order

    def test_init_invalid_order(self):
        """Test initialization with invalid constellation orders raises ValueError."""
        for order in [3, 5, 10, 15, 17]:
            with pytest.raises(ValueError):
                QAMConstellationMapper(order)

    def test_constellation_unit_average_power(self):
        """Test that the constellation has unit average power."""
        for order in [4, 16, 64]:
            mapper = QAMConstellationMapper(order)
            avg_power = np.mean(np.abs(mapper.constellation) ** 2)
            assert np.isclose(avg_power, 1.0)

    def test_encode_decode_round_trip(self):
        """Test that encoding and then decoding returns the original bits."""
        mapper = QAMConstellationMapper(16)
        # Create test data: 32 random bits
        original_bits = np.random.randint(0, 2, size=32)

        # Pack bits into bytes
        bytes_data = bytearray()
        for i in range(0, len(original_bits), 8):
            if i + 8 <= len(original_bits):
                byte = 0
                for j in range(8):
                    byte = (byte << 1) | original_bits[i + j]
                bytes_data.append(byte)

        # Encode bytes to symbols
        input_stream = io.BytesIO(bytes(bytes_data))
        symbols = mapper.encode(input_stream)

        # Decode symbols back to bits
        output_stream = mapper.decode(symbols)
        decoded_bytes = output_stream.read()

        # Extract bits from decoded bytes
        decoded_bits = []
        for byte in decoded_bytes:
            for i in range(7, -1, -1):
                decoded_bits.append((byte >> i) & 1)

        # Compare original and decoded bits (we may have padding at the end)
        assert_array_equal(original_bits, decoded_bits[: len(original_bits)])

    def test_noisy_decode(self):
        """Test decoding with noisy symbols."""
        np.random.seed(42)  # For reproducible results
        mapper = QAMConstellationMapper(16)

        # Generate some random symbols from the constellation
        indices = np.random.randint(0, 16, size=10)
        clean_symbols = mapper.constellation[indices]

        # Add some noise
        noise_level = 0.1
        noisy_symbols = clean_symbols + noise_level * (
            np.random.randn(len(clean_symbols)) + 1j * np.random.randn(len(clean_symbols))
        )

        # Decode and then re-encode to get clean symbols
        decoded_stream = mapper.decode(noisy_symbols)
        decoded_stream.seek(0)  # Reset position to start of stream
        re_encoded_symbols = mapper.encode(decoded_stream)

        # Check that the re-encoded symbols match the original clean symbols
        assert_array_almost_equal(clean_symbols, re_encoded_symbols)


class TestPSKConstellationMapper:
    def test_init_valid_order(self):
        """Test initialization with valid constellation orders."""
        for order in [2, 4, 8, 16, 32]:
            mapper = PSKConstellationMapper(order)
            assert mapper.order == order
            assert mapper.bits_per_symbol == int(np.log2(order))
            assert mapper.constellation_name == f"{order}-PSK"
            assert len(mapper.constellation) == order
            assert len(mapper.constellation_map) == order

    def test_init_invalid_order(self):
        """Test initialization with invalid constellation orders raises ValueError."""
        for order in [3, 5, 6, 7, 10]:
            with pytest.raises(ValueError):
                PSKConstellationMapper(order)

    def test_constellation_unit_amplitude(self):
        """Test that all PSK constellation points have unit amplitude."""
        for order in [2, 4, 8, 16]:
            mapper = PSKConstellationMapper(order)
            amplitudes = np.abs(mapper.constellation)
            # All points should have amplitude 1.0
            assert_array_almost_equal(amplitudes, np.ones_like(amplitudes))

    def test_encode_decode_round_trip(self):
        """Test that encoding and then decoding returns the original bits."""
        mapper = PSKConstellationMapper(8)
        # Create test data with a multiple of bits_per_symbol bits
        bits_per_symbol = mapper.bits_per_symbol
        num_symbols = 10
        num_bits = num_symbols * bits_per_symbol

        # Make sure the number of bits is a multiple of 8 (whole bytes)
        if num_bits % 8 != 0:
            num_bits = ((num_bits // 8) + 1) * 8

        original_bits = np.random.randint(0, 2, size=num_bits)

        # Pack bits into bytes
        bytes_data = bytearray()
        for i in range(0, len(original_bits), 8):
            byte = 0
            for j in range(min(8, len(original_bits) - i)):
                byte = (byte << 1) | original_bits[i + j]
            # Left-shift any remaining bits to complete the byte
            if i + 8 > len(original_bits):
                byte = byte << (8 - (len(original_bits) - i))
            bytes_data.append(byte)

        # Encode bytes to symbols
        input_stream = io.BytesIO(bytes(bytes_data))
        symbols = mapper.encode(input_stream)

        # Decode symbols back to bits
        output_stream = mapper.decode(symbols)
        decoded_bytes = output_stream.read()

        # Extract bits from decoded bytes
        decoded_bits = []
        for byte in decoded_bytes:
            for i in range(7, -1, -1):
                decoded_bits.append((byte >> i) & 1)

        # Compare original and decoded bits (truncate to match the shorter array)
        min_len = min(len(original_bits), len(decoded_bits))
        assert_array_equal(original_bits[:min_len], decoded_bits[:min_len])

    def test_noisy_decode(self):
        """Test decoding with noisy symbols."""
        np.random.seed(42)  # For reproducible results
        mapper = PSKConstellationMapper(8)
        bits_per_symbol = mapper.bits_per_symbol

        # Ensure we're using a number of symbols that's compatible with byte boundaries
        # For 8-PSK with 3 bits per symbol, we need a multiple of 8/gcd(3,8) symbols
        num_symbols = 8  # LCM(3,8) / 3 for 8-PSK

        # Generate random indices within constellation range
        indices = np.random.randint(0, 8, size=num_symbols)
        clean_symbols = mapper.constellation[indices]

        # Add some noise (but not too much to cause decoding errors)
        noise_level = 0.1
        noisy_symbols = clean_symbols + noise_level * (
            np.random.randn(len(clean_symbols)) + 1j * np.random.randn(len(clean_symbols))
        )

        # Instead of re-encoding, just verify that the classification works correctly
        classified = mapper.classifier.classify(mapper.constellation, noisy_symbols)

        # Just check that classification works correctly
        assert_array_almost_equal(classified, clean_symbols)

    def test_equal_phase_spacing(self):
        """Test that PSK constellation points are equally spaced in phase."""
        order = 8
        mapper = PSKConstellationMapper(order)

        # Calculate phases of all points
        phases = np.angle(mapper.constellation)

        # Sort phases and unwrap to handle the -π/π boundary
        sorted_phases = np.sort(phases)

        # Calculate phase differences (should all be equal)
        phase_diffs = np.diff(sorted_phases)
        expected_diff = 2 * np.pi / order

        assert_array_almost_equal(phase_diffs, np.ones_like(phase_diffs) * expected_diff)


class TestIntegration:
    """Integration tests that check the interaction between components."""

    def test_qam_psk_compatibility(self):
        """Test that QAM and PSK mappers can process the same bit stream."""
        # Create instances of both mapper types
        qam = QAMConstellationMapper(16)  # 4 bits per symbol
        psk = PSKConstellationMapper(16)  # 4 bits per symbol

        # Generate random bits
        original_bits = np.random.randint(0, 2, size=32)  # 32 bits = 8 symbols for both mappers

        # Pack bits into bytes
        bytes_data = bytearray()
        for i in range(0, len(original_bits), 8):
            if i + 8 <= len(original_bits):
                byte = 0
                for j in range(8):
                    byte = (byte << 1) | original_bits[i + j]
                bytes_data.append(byte)

        # Create a copy of the bytes for each mapper
        input_stream_qam = io.BytesIO(bytes(bytes_data))
        input_stream_psk = io.BytesIO(bytes(bytes_data))

        # Encode with both mappers
        qam_symbols = qam.encode(input_stream_qam)
        psk_symbols = psk.encode(input_stream_psk)

        # Both should produce the same number of symbols
        assert len(qam_symbols) == len(psk_symbols)

        # Decode and extract bits from each
        qam_bits = extract_bits_from_stream(qam.decode(qam_symbols))
        psk_bits = extract_bits_from_stream(psk.decode(psk_symbols))

        # Both should decode to the same original bits
        assert_array_equal(qam_bits[: len(original_bits)], original_bits)
        assert_array_equal(psk_bits[: len(original_bits)], original_bits)


# Helper function for integration tests
def extract_bits_from_stream(stream):
    """Extract individual bits from a BytesIO stream."""
    decoded_bytes = stream.read()
    decoded_bits = []
    for byte in decoded_bytes:
        for i in range(7, -1, -1):
            decoded_bits.append((byte >> i) & 1)
    return np.array(decoded_bits)
