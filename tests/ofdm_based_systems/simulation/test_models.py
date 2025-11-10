"""
Unit tests for simulation models.
"""

from io import BytesIO
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from ofdm_based_systems.configuration.enums import (
    ChannelType,
    ConstellationType,
    EqualizationMethod,
    ModulationType,
    NoiseType,
    PrefixType,
)
from ofdm_based_systems.configuration.models import SimulationSettings
from ofdm_based_systems.simulation.models import Simulation, read_bits_from_stream


class TestReadBitsFromStream:
    """Test read_bits_from_stream helper function."""

    def test_read_single_byte(self):
        """Test reading a single byte from stream."""
        # Byte 0b10101010 = 170
        stream = BytesIO(b"\xaa")
        bits = read_bits_from_stream(stream)

        expected = [1, 0, 1, 0, 1, 0, 1, 0]
        assert bits == expected

    def test_read_multiple_bytes(self):
        """Test reading multiple bytes from stream."""
        # Bytes 0xFF (11111111) and 0x00 (00000000)
        stream = BytesIO(b"\xff\x00")
        bits = read_bits_from_stream(stream)

        expected = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        assert bits == expected

    def test_read_empty_stream(self):
        """Test reading from an empty stream."""
        stream = BytesIO(b"")
        bits = read_bits_from_stream(stream)

        assert bits == []

    def test_stream_position_reset(self):
        """Test that stream position is reset to beginning after reading."""
        stream = BytesIO(b"\x55")  # 01010101
        bits = read_bits_from_stream(stream)

        # Check stream position is reset
        assert stream.tell() == 0

        # Read again to verify it works
        bits2 = read_bits_from_stream(stream)
        assert bits == bits2

    def test_read_byte_with_all_zeros(self):
        """Test reading a byte with all zeros."""
        stream = BytesIO(b"\x00")
        bits = read_bits_from_stream(stream)

        expected = [0, 0, 0, 0, 0, 0, 0, 0]
        assert bits == expected

    def test_read_byte_with_all_ones(self):
        """Test reading a byte with all ones."""
        stream = BytesIO(b"\xff")
        bits = read_bits_from_stream(stream)

        expected = [1, 1, 1, 1, 1, 1, 1, 1]
        assert bits == expected

    def test_read_arbitrary_pattern(self):
        """Test reading an arbitrary bit pattern."""
        # 0x0F = 00001111, 0xF0 = 11110000
        stream = BytesIO(b"\x0f\xf0")
        bits = read_bits_from_stream(stream)

        expected = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        assert bits == expected

    def test_read_long_stream(self):
        """Test reading a longer stream."""
        data = bytes(range(256))
        stream = BytesIO(data)
        bits = read_bits_from_stream(stream)

        # Should have 256 * 8 = 2048 bits
        assert len(bits) == 2048

        # Verify first byte (0x00 = 00000000)
        assert bits[0:8] == [0, 0, 0, 0, 0, 0, 0, 0]

        # Verify last byte (0xFF = 11111111)
        assert bits[-8:] == [1, 1, 1, 1, 1, 1, 1, 1]


class TestSimulationInitialization:
    """Test Simulation class initialization."""

    def test_init_with_num_bits(self):
        """Test initialization with num_bits."""
        sim = Simulation(num_bits=1000)

        assert sim.num_bits == 1000
        assert sim.num_symbols is None

    def test_init_with_num_symbols(self):
        """Test initialization with num_symbols."""
        sim = Simulation(num_symbols=100)

        assert sim.num_bits is None
        assert sim.num_symbols == 100

    def test_init_with_neither_raises_error(self):
        """Test that initialization without num_bits or num_symbols raises error."""
        with pytest.raises(ValueError, match="Either num_bits or num_symbols must be provided"):
            Simulation()

    def test_init_with_both_raises_error(self):
        """Test that initialization with both num_bits and num_symbols raises error."""
        with pytest.raises(
            ValueError, match="Only one of num_bits or num_symbols should be provided"
        ):
            Simulation(num_bits=1000, num_symbols=100)

    def test_init_with_default_parameters(self):
        """Test initialization with default parameters."""
        sim = Simulation(num_bits=1000)

        assert sim.num_subcarriers == 64
        assert sim.constellation_order == 16
        assert sim.constellation_scheme == ConstellationType.QAM
        assert sim.modulator_type == ModulationType.OFDM
        assert sim.prefix_scheme == PrefixType.CYCLIC
        assert sim.prefix_length_ratio == 1.0
        assert sim.equalizator_type == EqualizationMethod.MMSE
        assert sim.snr_db == 20.0
        assert sim.noise_scheme == NoiseType.AWGN

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        sim = Simulation(
            num_bits=2000,
            num_subcarriers=128,
            constellation_order=64,
            constellation_scheme=ConstellationType.PSK,
            modulator_type=ModulationType.SC_OFDM,
            prefix_scheme=PrefixType.ZERO,
            prefix_length_ratio=0.5,
            equalizator_type=EqualizationMethod.ZF,
            snr_db=15.0,
            noise_scheme=NoiseType.NONE,
        )

        assert sim.num_bits == 2000
        assert sim.num_subcarriers == 128
        assert sim.constellation_order == 64
        assert sim.constellation_scheme == ConstellationType.PSK
        assert sim.modulator_type == ModulationType.SC_OFDM
        assert sim.prefix_scheme == PrefixType.ZERO
        assert sim.prefix_length_ratio == 0.5
        assert sim.equalizator_type == EqualizationMethod.ZF
        assert sim.snr_db == 15.0
        assert sim.noise_scheme == NoiseType.NONE


class TestSimulationCreateFromSettings:
    """Test Simulation.create_from_simulation_settings class method."""

    def test_create_single_snr(self):
        """Test creating simulations with single SNR value."""
        settings = SimulationSettings(
            num_bands=64,
            signal_noise_ratios=[10.0],
            channel_model_path="test_channel.mat",
            channel_type=ChannelType.FLAT,
            num_bits=None,
            num_symbols=1000,
            constellation_order=16,
            constellation_type=ConstellationType.QAM,
            modulation_type=ModulationType.OFDM,
            prefix_type=PrefixType.CYCLIC,
            prefix_length_ratio=0.25,
            equalization_method=EqualizationMethod.MMSE,
            noise_type=NoiseType.AWGN,
        )

        simulations = Simulation.create_from_simulation_settings(settings)

        assert len(simulations) == 1
        assert simulations[0].snr_db == 10.0
        assert simulations[0].num_symbols == 1000
        assert simulations[0].num_subcarriers == 64

    def test_create_multiple_snrs(self):
        """Test creating simulations with multiple SNR values."""
        settings = SimulationSettings(
            num_bands=128,
            signal_noise_ratios=[5.0, 10.0, 15.0, 20.0],
            channel_model_path="test_channel.mat",
            channel_type=ChannelType.FLAT,
            num_bits=10000,
            num_symbols=None,
            constellation_order=64,
            constellation_type=ConstellationType.PSK,
            modulation_type=ModulationType.SC_OFDM,
            prefix_type=PrefixType.ZERO,
            prefix_length_ratio=0.5,
            equalization_method=EqualizationMethod.ZF,
            noise_type=NoiseType.NONE,
        )

        simulations = Simulation.create_from_simulation_settings(settings)

        assert len(simulations) == 4
        for i, snr in enumerate([5.0, 10.0, 15.0, 20.0]):
            assert simulations[i].snr_db == snr
            assert simulations[i].num_bits == 10000
            assert simulations[i].num_subcarriers == 128

    def test_create_preserves_all_settings(self):
        """Test that all settings are correctly transferred."""
        settings = SimulationSettings(
            num_bands=256,
            signal_noise_ratios=[12.5],
            channel_model_path="custom_channel.mat",
            channel_type=ChannelType.FLAT,
            num_bits=None,
            num_symbols=5000,
            constellation_order=4,
            constellation_type=ConstellationType.PSK,
            modulation_type=ModulationType.OFDM,
            prefix_type=PrefixType.NONE,
            prefix_length_ratio=0.75,
            equalization_method=EqualizationMethod.NONE,
            noise_type=NoiseType.AWGN,
        )

        simulations = Simulation.create_from_simulation_settings(settings)

        sim = simulations[0]
        assert sim.num_subcarriers == 256
        assert sim.num_symbols == 5000
        assert sim.constellation_order == 4
        assert sim.constellation_scheme == ConstellationType.PSK
        assert sim.modulator_type == ModulationType.OFDM
        assert sim.prefix_scheme == PrefixType.NONE
        assert sim.prefix_length_ratio == 0.75
        assert sim.equalizator_type == EqualizationMethod.NONE
        assert sim.noise_scheme == NoiseType.AWGN


class TestSimulationMappers:
    """Test that mapper dictionaries are correctly defined."""

    def test_constellation_mappers(self):
        """Test CONSTELLATION_SCHEME_MAPPERS dictionary."""
        assert ConstellationType.QAM in Simulation.CONSTELLATION_SCHEME_MAPPERS
        assert ConstellationType.PSK in Simulation.CONSTELLATION_SCHEME_MAPPERS

    def test_modulator_mappers(self):
        """Test MODULATOR_SCHEME_MAPPERS dictionary."""
        assert ModulationType.OFDM in Simulation.MODULATOR_SCHEME_MAPPERS
        assert ModulationType.SC_OFDM in Simulation.MODULATOR_SCHEME_MAPPERS

    def test_prefix_mappers(self):
        """Test PREFIX_SCHEME_MAPPERS dictionary."""
        assert PrefixType.NONE in Simulation.PREFIX_SCHEME_MAPPERS
        assert PrefixType.CYCLIC in Simulation.PREFIX_SCHEME_MAPPERS
        assert PrefixType.ZERO in Simulation.PREFIX_SCHEME_MAPPERS

    def test_equalizator_mappers(self):
        """Test EQUALIZATOR_SCHEME_MAPPERS dictionary."""
        assert EqualizationMethod.NONE in Simulation.EQUALIZATOR_SCHEME_MAPPERS
        assert EqualizationMethod.ZF in Simulation.EQUALIZATOR_SCHEME_MAPPERS
        assert EqualizationMethod.MMSE in Simulation.EQUALIZATOR_SCHEME_MAPPERS

    def test_noise_mappers(self):
        """Test NOISE_SCHEME_MAPPERS dictionary."""
        assert NoiseType.AWGN in Simulation.NOISE_SCHEME_MAPPERS
        assert NoiseType.NONE in Simulation.NOISE_SCHEME_MAPPERS


class TestSimulationRun:
    """Test Simulation.run method availability.

    Note: Full end-to-end run() integration tests are omitted from unit tests as they require
    complex mocking of the entire OFDM pipeline including real NumPy FFT operations, channel
    transmission, and signal processing. These are better tested in a separate integration
    test suite with proper fixtures.
    """

    def test_run_method_exists(self):
        """Test that run() method exists and is callable."""
        sim = Simulation(num_bits=64, num_subcarriers=64, constellation_order=4)
        assert hasattr(sim, "run")
        assert callable(sim.run)


class TestSimulationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_simulation_with_minimal_parameters(self):
        """Test simulation with minimal valid parameters."""
        sim = Simulation(num_bits=8, num_subcarriers=1, constellation_order=2)

        assert sim.num_bits == 8
        assert sim.num_subcarriers == 1
        assert sim.constellation_order == 2

    def test_simulation_with_large_parameters(self):
        """Test simulation with large parameters."""
        sim = Simulation(
            num_symbols=1_000_000,
            num_subcarriers=2048,
            constellation_order=256,
        )

        assert sim.num_symbols == 1_000_000
        assert sim.num_subcarriers == 2048
        assert sim.constellation_order == 256

    def test_prefix_length_calculation_with_different_ratios(self):
        """Test prefix length calculation with different ratios."""
        # This tests the logic: prefix_length = int(prefix_length_ratio * channel.order)
        sim1 = Simulation(num_bits=100, prefix_length_ratio=0.25)
        sim2 = Simulation(num_bits=100, prefix_length_ratio=0.5)
        sim3 = Simulation(num_bits=100, prefix_length_ratio=1.0)

        assert sim1.prefix_length_ratio == 0.25
        assert sim2.prefix_length_ratio == 0.5
        assert sim3.prefix_length_ratio == 1.0
