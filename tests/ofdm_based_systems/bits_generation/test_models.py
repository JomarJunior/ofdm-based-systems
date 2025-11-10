"""Comprehensive unit tests for bits_generation.models module."""

import math
from io import BytesIO
from unittest.mock import Mock, patch

import pytest
from numpy.random import Generator, PCG64

from ofdm_based_systems.bits_generation.models import IGenerator, RandomBitsGenerator


class TestIGenerator:
    """Test IGenerator abstract base class."""

    def test_igenerator_is_abstract(self):
        """Test that IGenerator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            IGenerator()

    def test_igenerator_requires_generate_bits_implementation(self):
        """Test that subclasses must implement generate_bits method."""
        class IncompleteGenerator(IGenerator):
            pass

        with pytest.raises(TypeError):
            IncompleteGenerator()

    def test_igenerator_can_be_subclassed(self):
        """Test that IGenerator can be properly subclassed."""
        class ConcreteGenerator(IGenerator):
            def generate_bits(self, num_bits: int) -> BytesIO:
                return BytesIO(b"\x00")

        generator = ConcreteGenerator()
        assert isinstance(generator, IGenerator)
        result = generator.generate_bits(8)
        assert isinstance(result, BytesIO)

    def test_igenerator_abstract_method_signature(self):
        """Test that abstract method has correct signature."""
        import inspect
        method = IGenerator.generate_bits
        assert hasattr(method, '__isabstractmethod__')
        assert method.__isabstractmethod__ is True
        
        # Check method signature
        sig = inspect.signature(IGenerator.generate_bits)
        params = list(sig.parameters.keys())
        assert 'num_bits' in params


class TestRandomBitsGeneratorInitialization:
    """Test RandomBitsGenerator initialization."""

    def test_init_with_default_generator(self):
        """Test initialization with default numpy Generator."""
        generator = RandomBitsGenerator()
        assert isinstance(generator.generator, Generator)

    def test_init_with_custom_generator(self):
        """Test initialization with custom numpy Generator."""
        custom_gen = Generator(PCG64(seed=42))
        generator = RandomBitsGenerator(generator=custom_gen)
        assert generator.generator is custom_gen

    def test_init_with_seeded_generator(self):
        """Test initialization with seeded generator for reproducibility."""
        gen1 = RandomBitsGenerator(generator=Generator(PCG64(seed=12345)))
        gen2 = RandomBitsGenerator(generator=Generator(PCG64(seed=12345)))
        
        # Same seed should produce same results
        bits1 = gen1.generate_bits(16)
        bits2 = gen2.generate_bits(16)
        assert bits1.read() == bits2.read()


class TestRandomBitsGeneratorGenerateBits:
    """Test RandomBitsGenerator.generate_bits method."""

    def test_generate_bits_returns_bytesio(self):
        """Test that generate_bits returns a BytesIO object."""
        generator = RandomBitsGenerator()
        result = generator.generate_bits(8)
        assert isinstance(result, BytesIO)

    def test_generate_bits_stream_position(self):
        """Test that returned stream is positioned at the beginning."""
        generator = RandomBitsGenerator()
        result = generator.generate_bits(8)
        assert result.tell() == 0

    def test_generate_bits_exact_byte_count(self):
        """Test generating bits that align exactly with byte boundaries."""
        generator = RandomBitsGenerator()
        
        # Test multiples of 8 bits
        for num_bits in [8, 16, 24, 32, 64, 128]:
            result = generator.generate_bits(num_bits)
            data = result.read()
            expected_bytes = num_bits // 8
            assert len(data) == expected_bytes, f"Failed for {num_bits} bits"

    def test_generate_bits_non_byte_aligned(self):
        """Test generating bits that don't align with byte boundaries."""
        generator = RandomBitsGenerator()
        
        # Test non-multiples of 8 bits
        for num_bits in [1, 3, 5, 7, 9, 15, 17, 25]:
            result = generator.generate_bits(num_bits)
            data = result.read()
            expected_bytes = math.ceil(num_bits / 8)
            assert len(data) == expected_bytes, f"Failed for {num_bits} bits"

    def test_generate_bits_single_bit(self):
        """Test generating a single bit."""
        generator = RandomBitsGenerator()
        result = generator.generate_bits(1)
        data = result.read()
        assert len(data) == 1
        # Only the MSB should potentially be set, rest should be 0
        assert data[0] & 0x7F == 0

    def test_generate_bits_seven_bits(self):
        """Test generating exactly 7 bits."""
        generator = RandomBitsGenerator()
        result = generator.generate_bits(7)
        data = result.read()
        assert len(data) == 1
        # Only the LSB should be 0
        assert data[0] & 0x01 == 0

    def test_generate_bits_masking_for_non_byte_aligned(self):
        """Test that unused bits are properly masked to zero."""
        generator = RandomBitsGenerator(generator=Generator(PCG64(seed=42)))
        
        # Generate 10 bits (1 byte + 2 bits)
        result = generator.generate_bits(10)
        data = result.read()
        
        assert len(data) == 2
        # Last 6 bits of the second byte should be 0
        # Mask: 11000000 = 0xC0
        last_byte = data[1]
        assert last_byte & 0x3F == 0

    def test_generate_bits_various_sizes(self):
        """Test generating bits for various sizes."""
        generator = RandomBitsGenerator()
        
        test_cases = [
            (1, 1),    # 1 bit -> 1 byte
            (2, 1),    # 2 bits -> 1 byte
            (8, 1),    # 8 bits -> 1 byte
            (9, 2),    # 9 bits -> 2 bytes
            (16, 2),   # 16 bits -> 2 bytes
            (17, 3),   # 17 bits -> 3 bytes
            (100, 13), # 100 bits -> 13 bytes
            (1000, 125), # 1000 bits -> 125 bytes
        ]
        
        for num_bits, expected_bytes in test_cases:
            result = generator.generate_bits(num_bits)
            data = result.read()
            assert len(data) == expected_bytes, f"Failed for {num_bits} bits"

    def test_generate_bits_zero_bits(self):
        """Test generating zero bits."""
        generator = RandomBitsGenerator()
        result = generator.generate_bits(0)
        data = result.read()
        assert len(data) == 0

    def test_generate_bits_large_number(self):
        """Test generating a large number of bits."""
        generator = RandomBitsGenerator()
        num_bits = 10000
        result = generator.generate_bits(num_bits)
        data = result.read()
        expected_bytes = math.ceil(num_bits / 8)
        assert len(data) == expected_bytes

    def test_generate_bits_reproducibility_with_seed(self):
        """Test that same seed produces same bit sequence."""
        seed = 999
        gen1 = RandomBitsGenerator(generator=Generator(PCG64(seed=seed)))
        gen2 = RandomBitsGenerator(generator=Generator(PCG64(seed=seed)))
        
        for num_bits in [8, 15, 32, 100]:
            bits1 = gen1.generate_bits(num_bits).read()
            bits2 = gen2.generate_bits(num_bits).read()
            assert bits1 == bits2, f"Reproducibility failed for {num_bits} bits"

    def test_generate_bits_randomness(self):
        """Test that different calls produce different random bits."""
        generator = RandomBitsGenerator()
        
        # Generate multiple sequences
        sequences = [generator.generate_bits(64).read() for _ in range(10)]
        
        # Check that not all sequences are identical
        # (statistically extremely unlikely with proper RNG)
        unique_sequences = set(sequences)
        assert len(unique_sequences) > 1, "Generator appears to not be random"


class TestRandomBitsGeneratorByteCalculation:
    """Test byte calculation logic in RandomBitsGenerator."""

    def test_byte_calculation_formula(self):
        """Test that byte calculation follows ceil(num_bits / 8) formula."""
        generator = RandomBitsGenerator()
        
        for num_bits in range(0, 100):
            result = generator.generate_bits(num_bits)
            actual_bytes = len(result.read())
            expected_bytes = math.ceil(num_bits / 8) if num_bits > 0 else 0
            assert actual_bytes == expected_bytes, \
                f"Byte calculation failed for {num_bits} bits"


class TestRandomBitsGeneratorMaskingLogic:
    """Test bit masking logic for non-byte-aligned generation."""

    def test_masking_preserves_requested_bits(self):
        """Test that masking correctly preserves only requested bits."""
        # Use seeded generator for deterministic testing
        generator = RandomBitsGenerator(generator=Generator(PCG64(seed=777)))
        
        # Generate full byte then partial byte
        full_result = generator.generate_bits(8)
        full_byte = full_result.read()[0]
        
        # Reset generator with same seed
        generator = RandomBitsGenerator(generator=Generator(PCG64(seed=777)))
        
        # Generate same byte but request fewer bits
        for bits_to_keep in range(1, 8):
            generator_test = RandomBitsGenerator(generator=Generator(PCG64(seed=777)))
            partial_result = generator_test.generate_bits(bits_to_keep)
            partial_byte = partial_result.read()[0]
            
            # Create expected mask
            mask = (0xFF << (8 - bits_to_keep)) & 0xFF
            expected_byte = full_byte & mask
            
            assert partial_byte == expected_byte, \
                f"Masking failed for {bits_to_keep} bits"

    def test_masking_zeros_unused_bits(self):
        """Test that unused bits in last byte are always zero."""
        generator = RandomBitsGenerator()
        
        for num_bits in range(1, 64):
            bits_in_last_byte = num_bits % 8
            if bits_in_last_byte == 0:
                continue  # Skip byte-aligned cases
            
            result = generator.generate_bits(num_bits)
            data = result.read()
            last_byte = data[-1]
            
            # Calculate which bits should be zero
            num_zero_bits = 8 - bits_in_last_byte
            zero_mask = (1 << num_zero_bits) - 1  # Mask for bits that should be 0
            
            assert last_byte & zero_mask == 0, \
                f"Unused bits not zero for {num_bits} bits"


class TestRandomBitsGeneratorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_generate_one_million_bits(self):
        """Test generating a very large number of bits."""
        generator = RandomBitsGenerator()
        num_bits = 1_000_000
        result = generator.generate_bits(num_bits)
        data = result.read()
        expected_bytes = math.ceil(num_bits / 8)
        assert len(data) == expected_bytes

    def test_consecutive_calls_independent(self):
        """Test that consecutive calls produce independent results."""
        generator = RandomBitsGenerator()
        
        results = []
        for _ in range(5):
            result = generator.generate_bits(32)
            results.append(result.read())
        
        # Check that results are different (statistically)
        unique_results = set(results)
        assert len(unique_results) > 1

    def test_stream_seekable(self):
        """Test that returned BytesIO stream is seekable."""
        generator = RandomBitsGenerator()
        result = generator.generate_bits(16)
        
        # Read first byte
        first_byte = result.read(1)
        
        # Seek back to start
        result.seek(0)
        
        # Read again and verify
        first_byte_again = result.read(1)
        assert first_byte == first_byte_again

    def test_stream_readable_multiple_times(self):
        """Test that stream can be read multiple times after seeking."""
        generator = RandomBitsGenerator()
        result = generator.generate_bits(24)
        
        data1 = result.read()
        result.seek(0)
        data2 = result.read()
        
        assert data1 == data2
        assert len(data1) == 3


class TestRandomBitsGeneratorIntegration:
    """Integration tests for RandomBitsGenerator."""

    def test_multiple_generators_independent(self):
        """Test that multiple generator instances are independent."""
        gen1 = RandomBitsGenerator()
        gen2 = RandomBitsGenerator()
        
        bits1 = gen1.generate_bits(32).read()
        bits2 = gen2.generate_bits(32).read()
        
        # Different unseeded generators should produce different results
        # (statistically extremely unlikely to be the same)
        assert bits1 != bits2

    def test_generator_state_persists(self):
        """Test that generator state persists across calls."""
        seed = 12345
        gen1 = RandomBitsGenerator(generator=Generator(PCG64(seed=seed)))
        gen2 = RandomBitsGenerator(generator=Generator(PCG64(seed=seed)))
        
        # First call
        bits1_call1 = gen1.generate_bits(16).read()
        bits2_call1 = gen2.generate_bits(16).read()
        assert bits1_call1 == bits2_call1
        
        # Second call - should still match because state advances identically
        bits1_call2 = gen1.generate_bits(16).read()
        bits2_call2 = gen2.generate_bits(16).read()
        assert bits1_call2 == bits2_call2
        
        # But first and second calls should be different
        assert bits1_call1 != bits1_call2

    def test_interface_compliance(self):
        """Test that RandomBitsGenerator properly implements IGenerator."""
        generator = RandomBitsGenerator()
        assert isinstance(generator, IGenerator)
        assert hasattr(generator, 'generate_bits')
        assert callable(generator.generate_bits)


class TestRandomBitsGeneratorMockingScenarios:
    """Test scenarios using mocked numpy Generator."""

    def test_with_mocked_generator(self):
        """Test RandomBitsGenerator with a mocked numpy Generator."""
        mock_generator = Mock(spec=Generator)
        mock_generator.bytes.return_value = b'\xFF\xFF'
        
        generator = RandomBitsGenerator(generator=mock_generator)
        result = generator.generate_bits(16)
        
        mock_generator.bytes.assert_called_once_with(2)
        assert result.read() == b'\xFF\xFF'

    def test_bytes_called_with_correct_size(self):
        """Test that Generator.bytes is called with correct byte count."""
        mock_generator = Mock(spec=Generator)
        mock_generator.bytes.return_value = b'\x00\x00\x00'
        
        generator = RandomBitsGenerator(generator=mock_generator)
        
        # Request 20 bits (needs 3 bytes)
        generator.generate_bits(20)
        
        mock_generator.bytes.assert_called_once_with(3)

    def test_masking_applied_correctly(self):
        """Test that masking is applied correctly to generator output."""
        mock_generator = Mock(spec=Generator)
        # Return byte with all bits set
        mock_generator.bytes.return_value = b'\xFF'
        
        generator = RandomBitsGenerator(generator=mock_generator)
        
        # Request 5 bits (should mask to 11111000 = 0xF8)
        result = generator.generate_bits(5)
        data = result.read()
        
        assert data[0] == 0xF8


class TestRandomBitsGeneratorBitPatterns:
    """Test various bit patterns and their handling."""

    def test_all_zeros_possible(self):
        """Test that generator can produce all zeros (statistically rare)."""
        mock_generator = Mock(spec=Generator)
        mock_generator.bytes.return_value = b'\x00\x00'
        
        generator = RandomBitsGenerator(generator=mock_generator)
        result = generator.generate_bits(16)
        
        assert result.read() == b'\x00\x00'

    def test_all_ones_possible(self):
        """Test that generator can produce all ones (statistically rare)."""
        mock_generator = Mock(spec=Generator)
        mock_generator.bytes.return_value = b'\xFF\xFF'
        
        generator = RandomBitsGenerator(generator=mock_generator)
        result = generator.generate_bits(16)
        
        assert result.read() == b'\xFF\xFF'

    def test_mixed_pattern(self):
        """Test generator with mixed bit pattern."""
        mock_generator = Mock(spec=Generator)
        mock_generator.bytes.return_value = b'\xAA\x55'  # 10101010 01010101
        
        generator = RandomBitsGenerator(generator=mock_generator)
        result = generator.generate_bits(16)
        
        assert result.read() == b'\xAA\x55'
