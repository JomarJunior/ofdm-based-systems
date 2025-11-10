"""Comprehensive end-to-end integration tests for OFDM-based systems.

Tests the complete transmission pipeline from bits to decoded bits,
validating the interaction between all system components.
"""

import math
from io import BytesIO

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from ofdm_based_systems.bits_generation.models import RandomBitsGenerator
from ofdm_based_systems.channel.models import ChannelModel
from ofdm_based_systems.constellation.models import (
    PSKConstellationMapper,
    QAMConstellationMapper,
)
from ofdm_based_systems.equalization.models import (
    MMSEEqualizator,
    NoEqualizator,
    ZeroForcingEqualizator,
)
from ofdm_based_systems.modulation.models import (
    OFDMModulator,
    SingleCarrierOFDMModulator,
)
from ofdm_based_systems.noise.models import AWGNoiseModel, NoNoiseModel
from ofdm_based_systems.prefix.models import (
    CyclicPrefixScheme,
    NoPrefixScheme,
    ZeroPaddingPrefixScheme,
)
from ofdm_based_systems.serial_parallel.models import SerialToParallelConverter
from ofdm_based_systems.simulation.models import Simulation, read_bits_from_stream


class TestEndToEndOFDMPipeline:
    """Test complete OFDM transmission and reception pipeline."""

    def test_perfect_channel_qam_ofdm_no_prefix(self):
        """Test end-to-end with perfect channel, QAM, OFDM, no prefix."""
        # Configuration
        num_bits = 256
        num_subcarriers = 16
        constellation_order = 16

        # Components
        bits_generator = RandomBitsGenerator()
        constellation_mapper = QAMConstellationMapper(order=constellation_order)
        converter = SerialToParallelConverter()

        # Perfect channel (identity)
        channel_impulse = np.array([1.0 + 0.0j], dtype=np.complex128)
        noise_model = NoNoiseModel()
        channel = ChannelModel(
            impulse_response=channel_impulse, snr_db=100.0, noise_model=noise_model
        )

        # Modulation components
        prefix_scheme = NoPrefixScheme()
        channel_response = np.fft.fft(channel_impulse, n=num_subcarriers)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)
        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        # Transmitter
        bits = bits_generator.generate_bits(num_bits)
        original_bits = read_bits_from_stream(bits)

        symbols = constellation_mapper.encode(bits)
        parallel_symbols = converter.to_parallel(symbols, num_subcarriers)
        modulated = modulator.modulate(parallel_symbols)
        serial_signal = converter.to_serial(modulated)

        # Channel
        received = channel.transmit(serial_signal)

        # Receiver
        received_parallel = converter.to_parallel(received, num_subcarriers)
        demodulated = modulator.demodulate(received_parallel)
        demodulated_serial = converter.to_serial(demodulated)

        decoded_bits_stream = constellation_mapper.decode(demodulated_serial)
        decoded_bits = read_bits_from_stream(decoded_bits_stream)

        # Validation
        # With perfect channel, we should get exact bits back (or very close)
        errors = sum(b1 != b2 for b1, b2 in zip(original_bits, decoded_bits))
        ber = errors / len(original_bits)

        assert ber < 0.01, f"BER too high for perfect channel: {ber}"

    def test_end_to_end_with_cyclic_prefix(self):
        """Test end-to-end with cyclic prefix and channel."""
        num_bits = 512
        num_subcarriers = 32
        constellation_order = 4
        prefix_length = 8

        # Components
        bits_generator = RandomBitsGenerator()
        constellation_mapper = QAMConstellationMapper(order=constellation_order)
        converter = SerialToParallelConverter()

        # Multipath channel
        channel_impulse = np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.complex128)
        noise_model = NoNoiseModel()
        channel = ChannelModel(
            impulse_response=channel_impulse, snr_db=100.0, noise_model=noise_model
        )

        # Modulation components
        prefix_scheme = CyclicPrefixScheme(prefix_length=prefix_length)
        channel_response = np.fft.fft(channel_impulse, n=num_subcarriers)
        equalizator = ZeroForcingEqualizator(
            channel_frequency_response=channel_response, snr_db=100.0
        )
        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        # Transmitter
        bits = bits_generator.generate_bits(num_bits)
        original_bits = read_bits_from_stream(bits)

        symbols = constellation_mapper.encode(bits)
        parallel_symbols = converter.to_parallel(symbols, num_subcarriers)
        modulated = modulator.modulate(parallel_symbols)
        serial_signal = converter.to_serial(modulated)

        # Channel
        received = channel.transmit(serial_signal)

        # Receiver (need to account for prefix in parallel conversion)
        received_parallel = converter.to_parallel(received, num_subcarriers + prefix_length)
        demodulated = modulator.demodulate(received_parallel)
        demodulated_serial = converter.to_serial(demodulated)

        decoded_bits_stream = constellation_mapper.decode(demodulated_serial)
        decoded_bits = read_bits_from_stream(decoded_bits_stream)

        # Validation - with CP and equalization, should handle multipath well
        errors = sum(b1 != b2 for b1, b2 in zip(original_bits[: len(decoded_bits)], decoded_bits))
        ber = errors / min(len(original_bits), len(decoded_bits))

        assert ber < 0.05, f"BER too high with CP and equalization: {ber}"

    def test_end_to_end_psk_sc_ofdm(self):
        """Test end-to-end with PSK and SC-OFDM."""
        num_bits = 256
        num_subcarriers = 16
        constellation_order = 4  # QPSK

        # Components
        bits_generator = RandomBitsGenerator()
        constellation_mapper = PSKConstellationMapper(order=constellation_order)
        converter = SerialToParallelConverter()

        # Simple channel
        channel_impulse = np.array([1.0, 0.3], dtype=np.complex128)
        noise_model = NoNoiseModel()
        channel = ChannelModel(
            impulse_response=channel_impulse, snr_db=100.0, noise_model=noise_model
        )

        # SC-OFDM modulation
        prefix_scheme = NoPrefixScheme()
        channel_response = np.fft.fft(channel_impulse, n=num_subcarriers)
        equalizator = ZeroForcingEqualizator(
            channel_frequency_response=channel_response, snr_db=100.0
        )
        modulator = SingleCarrierOFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        # Transmitter
        bits = bits_generator.generate_bits(num_bits)
        original_bits = read_bits_from_stream(bits)

        symbols = constellation_mapper.encode(bits)
        parallel_symbols = converter.to_parallel(symbols, num_subcarriers)
        modulated = modulator.modulate(parallel_symbols)
        serial_signal = converter.to_serial(modulated)

        # Channel
        received = channel.transmit(serial_signal)

        # Receiver
        received_parallel = converter.to_parallel(received, num_subcarriers)
        demodulated = modulator.demodulate(received_parallel)
        demodulated_serial = converter.to_serial(demodulated)

        decoded_bits_stream = constellation_mapper.decode(demodulated_serial)
        decoded_bits = read_bits_from_stream(decoded_bits_stream)

        # Validation
        errors = sum(b1 != b2 for b1, b2 in zip(original_bits[: len(decoded_bits)], decoded_bits))
        ber = errors / min(len(original_bits), len(decoded_bits))

        assert ber < 0.05, f"BER too high for PSK SC-OFDM: {ber}"

    def test_end_to_end_with_awgn_noise(self):
        """Test end-to-end with AWGN noise at moderate SNR."""
        num_bits = 1024
        num_subcarriers = 64
        constellation_order = 16
        snr_db = 20.0

        # Components
        bits_generator = RandomBitsGenerator()
        constellation_mapper = QAMConstellationMapper(order=constellation_order)
        converter = SerialToParallelConverter()

        # Channel with noise
        channel_impulse = np.array([1.0, 0.4, 0.2], dtype=np.complex128)
        noise_model = AWGNoiseModel()
        channel = ChannelModel(
            impulse_response=channel_impulse, snr_db=snr_db, noise_model=noise_model
        )

        # Modulation with MMSE equalization
        prefix_scheme = CyclicPrefixScheme(prefix_length=16)
        channel_response = np.fft.fft(channel_impulse, n=num_subcarriers)
        equalizator = MMSEEqualizator(channel_frequency_response=channel_response, snr_db=snr_db)
        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        # Transmitter
        bits = bits_generator.generate_bits(num_bits)
        original_bits = read_bits_from_stream(bits)

        symbols = constellation_mapper.encode(bits)
        parallel_symbols = converter.to_parallel(symbols, num_subcarriers)
        modulated = modulator.modulate(parallel_symbols)
        serial_signal = converter.to_serial(modulated)

        # Channel
        received = channel.transmit(serial_signal)

        # Receiver
        received_parallel = converter.to_parallel(received, num_subcarriers + 16)
        demodulated = modulator.demodulate(received_parallel)
        demodulated_serial = converter.to_serial(demodulated)

        decoded_bits_stream = constellation_mapper.decode(demodulated_serial)
        decoded_bits = read_bits_from_stream(decoded_bits_stream)

        # Validation - with noise, we expect some errors but not too many at SNR=20dB
        errors = sum(b1 != b2 for b1, b2 in zip(original_bits[: len(decoded_bits)], decoded_bits))
        ber = errors / min(len(original_bits), len(decoded_bits))

        # At 20dB SNR with 16-QAM, BER should be reasonably low
        assert ber < 0.1, f"BER too high at SNR={snr_db}dB: {ber}"

    def test_end_to_end_zero_padding(self):
        """Test end-to-end with zero padding prefix."""
        num_bits = 512
        num_subcarriers = 32
        constellation_order = 16  # Must be perfect square for QAM
        prefix_length = 8

        # Components
        bits_generator = RandomBitsGenerator()
        constellation_mapper = QAMConstellationMapper(order=constellation_order)
        converter = SerialToParallelConverter()

        # Channel
        channel_impulse = np.array([1.0, 0.6, 0.3, 0.1], dtype=np.complex128)
        noise_model = NoNoiseModel()
        channel = ChannelModel(
            impulse_response=channel_impulse, snr_db=100.0, noise_model=noise_model
        )

        # Modulation with zero padding
        prefix_scheme = ZeroPaddingPrefixScheme(prefix_length=prefix_length)
        channel_response = np.fft.fft(channel_impulse, n=num_subcarriers)
        equalizator = ZeroForcingEqualizator(
            channel_frequency_response=channel_response, snr_db=100.0
        )
        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        # Transmitter
        bits = bits_generator.generate_bits(num_bits)
        original_bits = read_bits_from_stream(bits)

        symbols = constellation_mapper.encode(bits)
        parallel_symbols = converter.to_parallel(symbols, num_subcarriers)
        modulated = modulator.modulate(parallel_symbols)
        serial_signal = converter.to_serial(modulated)

        # Channel
        received = channel.transmit(serial_signal)

        # Receiver
        received_parallel = converter.to_parallel(received, num_subcarriers + prefix_length)
        demodulated = modulator.demodulate(received_parallel)
        demodulated_serial = converter.to_serial(demodulated)

        decoded_bits_stream = constellation_mapper.decode(demodulated_serial)
        decoded_bits = read_bits_from_stream(decoded_bits_stream)

        # Validation
        errors = sum(b1 != b2 for b1, b2 in zip(original_bits[: len(decoded_bits)], decoded_bits))
        ber = errors / min(len(original_bits), len(decoded_bits))

        assert ber < 0.05, f"BER too high with zero padding: {ber}"


class TestCrossModuleIntegration:
    """Test interactions between specific module pairs."""

    def test_bits_generation_to_constellation(self):
        """Test bit generation directly feeding constellation mapper."""
        num_bits = 1000
        constellation_order = 16

        generator = RandomBitsGenerator()
        mapper = QAMConstellationMapper(order=constellation_order)

        # Generate and map
        bits = generator.generate_bits(num_bits)
        symbols = mapper.encode(bits)

        # Should produce correct number of symbols
        bits_per_symbol = int(np.log2(constellation_order))
        expected_symbols = num_bits // bits_per_symbol

        assert len(symbols) == expected_symbols
        assert symbols.dtype == np.complex128

    def test_constellation_to_serial_parallel(self):
        """Test constellation mapper output feeding serial/parallel converter."""
        num_bits = 512
        constellation_order = 4
        num_subcarriers = 16

        mapper = QAMConstellationMapper(order=constellation_order)
        converter = SerialToParallelConverter()

        # Generate symbols
        bits = BytesIO(b"\x00" * (num_bits // 8))
        symbols = mapper.encode(bits)

        # Convert to parallel
        parallel = converter.to_parallel(symbols, num_subcarriers)

        # Should create proper blocks
        expected_blocks = len(symbols) // num_subcarriers
        assert parallel.shape[0] == expected_blocks
        assert parallel.shape[1] == num_subcarriers

    def test_modulation_with_prefix_schemes(self):
        """Test modulator working with different prefix schemes."""
        num_subcarriers = 8
        channel_response = np.ones(num_subcarriers, dtype=np.complex128)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)

        symbols = np.random.randn(5, num_subcarriers) + 1j * np.random.randn(5, num_subcarriers)
        symbols = symbols.astype(np.complex128)

        # Test with different prefixes
        for prefix_scheme in [
            NoPrefixScheme(),
            CyclicPrefixScheme(prefix_length=2),
            ZeroPaddingPrefixScheme(prefix_length=3),
        ]:
            modulator = OFDMModulator(
                num_subcarriers=num_subcarriers,
                prefix_scheme=prefix_scheme,
                equalizator=equalizator,
            )

            modulated = modulator.modulate(symbols)
            demodulated = modulator.demodulate(modulated)

            # Should preserve shape
            assert demodulated.shape == symbols.shape
            # Should preserve approximately the values
            assert_array_almost_equal(demodulated, symbols, decimal=10)

    def test_channel_with_noise_models(self):
        """Test channel model with different noise models."""
        signal = np.array([1.0, 0.5, 0.25, 0.125], dtype=np.complex128)
        channel_impulse = np.array([1.0, 0.3], dtype=np.complex128)

        # Test with no noise
        channel_no_noise = ChannelModel(
            impulse_response=channel_impulse, snr_db=100.0, noise_model=NoNoiseModel()
        )
        received_no_noise = channel_no_noise.transmit(signal)

        # Test with AWGN
        channel_awgn = ChannelModel(
            impulse_response=channel_impulse, snr_db=10.0, noise_model=AWGNoiseModel()
        )
        received_awgn = channel_awgn.transmit(signal)

        # Both should produce same-length output
        assert len(received_no_noise) == len(signal)
        assert len(received_awgn) == len(signal)

        # AWGN version should be different from no-noise version
        assert not np.allclose(received_no_noise, received_awgn)

    def test_equalization_integration(self):
        """Test different equalizers in realistic scenario."""
        num_subcarriers = 16

        # Frequency selective channel
        channel_impulse = np.array([1.0, 0.5, 0.3, 0.2], dtype=np.complex128)
        channel_response = np.fft.fft(channel_impulse, n=num_subcarriers)

        symbols = np.random.randn(3, num_subcarriers) + 1j * np.random.randn(3, num_subcarriers)
        symbols = symbols.astype(np.complex128)

        # Test with different equalizers
        for EqualizerClass in [NoEqualizator, ZeroForcingEqualizator, MMSEEqualizator]:
            equalizator = EqualizerClass(channel_frequency_response=channel_response, snr_db=20.0)
            modulator = OFDMModulator(
                num_subcarriers=num_subcarriers,
                prefix_scheme=NoPrefixScheme(),
                equalizator=equalizator,
            )

            # Modulate and demodulate
            modulated = modulator.modulate(symbols)
            demodulated = modulator.demodulate(modulated)

            # All should produce valid output
            assert demodulated.shape == symbols.shape
            assert not np.any(np.isnan(demodulated))
            assert not np.any(np.isinf(demodulated))


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_varying_snr_ber_relationship(self):
        """Test that BER decreases as SNR increases."""
        num_bits = 2048
        num_subcarriers = 64
        constellation_order = 16

        bers = []
        snr_values = [5.0, 10.0, 15.0, 20.0, 25.0]

        for snr_db in snr_values:
            # Setup pipeline
            bits_generator = RandomBitsGenerator()
            constellation_mapper = QAMConstellationMapper(order=constellation_order)
            converter = SerialToParallelConverter()

            channel_impulse = np.array([1.0, 0.4, 0.2], dtype=np.complex128)
            noise_model = AWGNoiseModel()
            channel = ChannelModel(
                impulse_response=channel_impulse, snr_db=snr_db, noise_model=noise_model
            )

            prefix_scheme = CyclicPrefixScheme(prefix_length=16)
            channel_response = np.fft.fft(channel_impulse, n=num_subcarriers)
            equalizator = MMSEEqualizator(
                channel_frequency_response=channel_response, snr_db=snr_db
            )
            modulator = OFDMModulator(
                num_subcarriers=num_subcarriers,
                prefix_scheme=prefix_scheme,
                equalizator=equalizator,
            )

            # Run transmission
            bits = bits_generator.generate_bits(num_bits)
            original_bits = read_bits_from_stream(bits)

            symbols = constellation_mapper.encode(bits)
            parallel_symbols = converter.to_parallel(symbols, num_subcarriers)
            modulated = modulator.modulate(parallel_symbols)
            serial_signal = converter.to_serial(modulated)
            received = channel.transmit(serial_signal)
            received_parallel = converter.to_parallel(received, num_subcarriers + 16)
            demodulated = modulator.demodulate(received_parallel)
            demodulated_serial = converter.to_serial(demodulated)
            decoded_bits_stream = constellation_mapper.decode(demodulated_serial)
            decoded_bits = read_bits_from_stream(decoded_bits_stream)

            # Calculate BER
            errors = sum(
                b1 != b2 for b1, b2 in zip(original_bits[: len(decoded_bits)], decoded_bits)
            )
            ber = errors / min(len(original_bits), len(decoded_bits))
            bers.append(ber)

        # BER should generally decrease with increasing SNR
        # Allow some noise in the relationship
        assert bers[0] > bers[-1], "BER should decrease with increasing SNR"

    def test_large_constellation_order(self):
        """Test system with large constellation order (256-QAM)."""
        num_bits = 2048
        num_subcarriers = 64
        constellation_order = 256  # 256-QAM

        bits_generator = RandomBitsGenerator()
        constellation_mapper = QAMConstellationMapper(order=constellation_order)
        converter = SerialToParallelConverter()

        # Need high SNR for 256-QAM
        channel_impulse = np.array([1.0], dtype=np.complex128)
        noise_model = AWGNoiseModel()
        channel = ChannelModel(
            impulse_response=channel_impulse, snr_db=35.0, noise_model=noise_model
        )

        prefix_scheme = NoPrefixScheme()
        channel_response = np.fft.fft(channel_impulse, n=num_subcarriers)
        equalizator = NoEqualizator(channel_frequency_response=channel_response)
        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        # Transmission
        bits = bits_generator.generate_bits(num_bits)
        original_bits = read_bits_from_stream(bits)

        symbols = constellation_mapper.encode(bits)
        parallel_symbols = converter.to_parallel(symbols, num_subcarriers)
        modulated = modulator.modulate(parallel_symbols)
        serial_signal = converter.to_serial(modulated)
        received = channel.transmit(serial_signal)
        received_parallel = converter.to_parallel(received, num_subcarriers)
        demodulated = modulator.demodulate(received_parallel)
        demodulated_serial = converter.to_serial(demodulated)
        decoded_bits_stream = constellation_mapper.decode(demodulated_serial)
        decoded_bits = read_bits_from_stream(decoded_bits_stream)

        # Should handle 256-QAM
        assert len(decoded_bits) > 0
        errors = sum(b1 != b2 for b1, b2 in zip(original_bits[: len(decoded_bits)], decoded_bits))
        ber = errors / min(len(original_bits), len(decoded_bits))

        # With high SNR and perfect channel, should have low BER
        assert ber < 0.1

    def test_many_subcarriers(self):
        """Test system with many subcarriers (like real OFDM systems)."""
        num_bits = 4096
        num_subcarriers = 256  # Many subcarriers
        constellation_order = 16

        bits_generator = RandomBitsGenerator()
        constellation_mapper = QAMConstellationMapper(order=constellation_order)
        converter = SerialToParallelConverter()

        channel_impulse = np.array([1.0, 0.3, 0.1], dtype=np.complex128)
        noise_model = NoNoiseModel()
        channel = ChannelModel(
            impulse_response=channel_impulse, snr_db=100.0, noise_model=noise_model
        )

        prefix_scheme = CyclicPrefixScheme(prefix_length=64)
        channel_response = np.fft.fft(channel_impulse, n=num_subcarriers)
        equalizator = ZeroForcingEqualizator(
            channel_frequency_response=channel_response, snr_db=100.0
        )
        modulator = OFDMModulator(
            num_subcarriers=num_subcarriers, prefix_scheme=prefix_scheme, equalizator=equalizator
        )

        # Transmission
        bits = bits_generator.generate_bits(num_bits)
        original_bits = read_bits_from_stream(bits)

        symbols = constellation_mapper.encode(bits)
        parallel_symbols = converter.to_parallel(symbols, num_subcarriers)
        modulated = modulator.modulate(parallel_symbols)
        serial_signal = converter.to_serial(modulated)
        received = channel.transmit(serial_signal)
        received_parallel = converter.to_parallel(received, num_subcarriers + 64)
        demodulated = modulator.demodulate(received_parallel)
        demodulated_serial = converter.to_serial(demodulated)
        decoded_bits_stream = constellation_mapper.decode(demodulated_serial)
        decoded_bits = read_bits_from_stream(decoded_bits_stream)

        # Should handle many subcarriers
        errors = sum(b1 != b2 for b1, b2 in zip(original_bits[: len(decoded_bits)], decoded_bits))
        ber = errors / min(len(original_bits), len(decoded_bits))

        assert ber < 0.01


class TestSimulationClassIntegration:
    """Test the Simulation class which orchestrates the entire system."""

    def test_simulation_run_basic(self):
        """Test basic simulation run."""
        sim = Simulation(num_bits=512, num_subcarriers=32, constellation_order=4, snr_db=20.0)

        results = sim.run()

        # Verify results structure
        assert "bit_errors" in results
        assert "total_bits" in results
        assert "bit_error_rate" in results
        assert "papr_db" in results
        assert "constellation_plot" in results

        # Verify results validity
        assert results["total_bits"] == 512
        assert 0 <= results["bit_error_rate"] <= 1
        assert results["bit_errors"] >= 0
        assert not np.isnan(results["papr_db"])
        assert not np.isinf(results["papr_db"])

    def test_simulation_run_with_different_configurations(self):
        """Test simulation with various configurations."""
        # Test with num_bits
        sim1 = Simulation(num_bits=256, num_subcarriers=64, constellation_order=4, snr_db=15.0)
        results1 = sim1.run()
        assert "bit_error_rate" in results1
        assert 0 <= results1["bit_error_rate"] <= 1

        # Test with different constellation order
        sim2 = Simulation(num_bits=512, num_subcarriers=64, constellation_order=16, snr_db=20.0)
        results2 = sim2.run()
        assert "bit_error_rate" in results2
        assert 0 <= results2["bit_error_rate"] <= 1

        # Test with num_symbols (must be multiple of num_subcarriers)
        sim3 = Simulation(
            num_symbols=128,  # 128 symbols with 64-QAM = 768 bits (divisible by 64)
            num_subcarriers=64,
            constellation_order=64,
            snr_db=25.0,
        )
        results3 = sim3.run()
        assert "bit_error_rate" in results3
        assert 0 <= results3["bit_error_rate"] <= 1

    def test_simulation_reproducibility(self):
        """Test that simulations with same parameters produce consistent results."""
        # Run twice - results should be similar (not identical due to randomness)
        sim1 = Simulation(num_bits=512, num_subcarriers=32, constellation_order=16, snr_db=20.0)
        results1 = sim1.run()

        sim2 = Simulation(num_bits=512, num_subcarriers=32, constellation_order=16, snr_db=20.0)
        results2 = sim2.run()

        # BER should be in similar range (within reasonable tolerance)
        ber_diff = abs(results1["bit_error_rate"] - results2["bit_error_rate"])
        assert ber_diff < 0.5, "BER should be relatively consistent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
