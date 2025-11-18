import os
from io import BytesIO
from typing import Any, BinaryIO, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ofdm_based_systems.bits_generation.models import (
    AdaptiveBitsGenerator,
    IGenerator,
    RandomBitsGenerator,
)
from ofdm_based_systems.channel.models import ChannelModel
from ofdm_based_systems.configuration.enums import (
    AdaptiveModulationMode,
    ConstellationType,
    EqualizationMethod,
    ModulationType,
    NoiseType,
    PowerAllocationType,
    PrefixType,
)
from ofdm_based_systems.configuration.models import SimulationSettings
from ofdm_based_systems.constellation.adaptive import (
    AdaptiveConstellationMapper,
    calculate_constellation_orders,
)
from ofdm_based_systems.constellation.models import (
    IConstellationMapper,
    PSKConstellationMapper,
    QAMConstellationMapper,
)
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
from ofdm_based_systems.noise.models import AWGNoiseModel, NoNoiseModel
from ofdm_based_systems.power_allocation.models import (
    UniformPowerAllocation,
    WaterfillingPowerAllocation,
    calculate_capacity_per_subcarrier,
)
from ofdm_based_systems.prefix.models import (
    CyclicPrefixScheme,
    NoPrefixScheme,
    ZeroPaddingPrefixScheme,
)
from ofdm_based_systems.serial_parallel.models import SerialToParallelConverter


def read_bits_from_stream(stream: BinaryIO) -> List[int]:
    """Reads bits from a binary stream and returns them as a list of integers (0s and 1s)."""
    byte = stream.read(1)
    bits = []
    while byte:
        byte_value = ord(byte)
        for i in range(8):
            bits.append((byte_value >> (7 - i)) & 1)
        byte = stream.read(1)
    stream.seek(0)  # Reset stream position to the beginning
    return bits


class Simulation:
    CONSTELLATION_SCHEME_MAPPERS = {
        ConstellationType.QAM: QAMConstellationMapper,
        ConstellationType.PSK: PSKConstellationMapper,
    }

    MODULATOR_SCHEME_MAPPERS = {
        ModulationType.OFDM: OFDMModulator,
        ModulationType.SC_OFDM: SingleCarrierOFDMModulator,
    }

    PREFIX_SCHEME_MAPPERS = {
        PrefixType.NONE: NoPrefixScheme,
        PrefixType.CYCLIC: CyclicPrefixScheme,
        PrefixType.ZERO: ZeroPaddingPrefixScheme,
    }

    EQUALIZATOR_SCHEME_MAPPERS = {
        EqualizationMethod.NONE: NoEqualizator,
        EqualizationMethod.ZF: ZeroForcingEqualizator,
        EqualizationMethod.MMSE: MMSEEqualizator,
    }

    NOISE_SCHEME_MAPPERS = {
        NoiseType.AWGN: AWGNoiseModel,
        NoiseType.NONE: NoNoiseModel,
    }

    POWER_ALLOCATION_MAPPERS = {
        PowerAllocationType.UNIFORM: UniformPowerAllocation,
        PowerAllocationType.WATERFILLING: WaterfillingPowerAllocation,
    }

    def __init__(
        self,
        num_bits: Optional[int] = None,
        num_symbols: Optional[int] = None,
        num_subcarriers: int = 64,
        constellation_order: int = 16,
        constellation_scheme: ConstellationType = ConstellationType.QAM,
        modulator_type: ModulationType = ModulationType.OFDM,
        prefix_scheme: PrefixType = PrefixType.CYCLIC,
        prefix_length_ratio: float = 1.0,  # should stay at 1.0
        equalizator_type: EqualizationMethod = EqualizationMethod.MMSE,
        snr_db: float = 20.0,
        noise_scheme: NoiseType = NoiseType.AWGN,
        power_allocation_type: PowerAllocationType = PowerAllocationType.UNIFORM,
        adaptive_modulation_mode: AdaptiveModulationMode = AdaptiveModulationMode.FIXED,
        min_constellation_order: int = 4,
        max_constellation_order: int = 256,
        desired_symbol_error_rate: float = 1e-3,
        channel_impulse_response: Optional[NDArray[np.complex128]] = None,
        verbose: bool = True,
    ):
        if num_bits is None and num_symbols is None:
            raise ValueError("Either num_bits or num_symbols must be provided.")
        if num_bits is not None and num_symbols is not None:
            raise ValueError("Only one of num_bits or num_symbols should be provided.")

        self.num_bits = num_bits
        self.num_symbols = num_symbols
        self.num_subcarriers = num_subcarriers
        self.constellation_order = constellation_order
        self.constellation_scheme = constellation_scheme
        self.modulator_type = modulator_type
        self.prefix_scheme = prefix_scheme
        self.prefix_length_ratio = prefix_length_ratio
        self.equalizator_type = equalizator_type
        self.snr_db = snr_db
        self.noise_scheme = noise_scheme
        self.power_allocation_type = power_allocation_type
        self.adaptive_modulation_mode = adaptive_modulation_mode
        self.min_constellation_order = min_constellation_order
        self.max_constellation_order = max_constellation_order
        self.desired_symbol_error_rate = desired_symbol_error_rate
        self.channel_impulse_response = channel_impulse_response
        self.verbose = verbose

    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    @classmethod
    def create_from_simulation_settings(
        cls, simulation_settings: SimulationSettings
    ) -> List["Simulation"]:
        """Create multiple simulations from settings.

        Args:
            simulation_settings: Configuration settings

        Returns:
            List of Simulation instances, one per SNR value
        """
        # Load channel impulse response if custom channel is specified
        channel_impulse_response = None
        if simulation_settings.channel_type.value == "CUSTOM":
            channel_path = simulation_settings.channel_model_path
            if not channel_path:
                raise ValueError("channel_model_path must be specified when channel_type is CUSTOM")

            # Handle relative paths from project root
            if not channel_path.startswith("/"):
                # Assume path is relative to current working directory
                channel_path = os.path.abspath(channel_path)

            if not os.path.exists(channel_path):
                raise FileNotFoundError(f"Channel model file not found: {channel_path}")

            try:
                channel_impulse_response = np.load(channel_path)
                print(f"✓ Loaded custom channel impulse response from: {channel_path}")
                print(f"  Channel length: {len(channel_impulse_response)} taps")
                print(f"  Channel dtype: {channel_impulse_response.dtype}")
            except Exception as e:
                raise ValueError(f"Failed to load channel model from {channel_path}: {e}")

        simulations = []
        for snr in simulation_settings.signal_noise_ratios:
            simulation = cls(
                num_bits=simulation_settings.num_bits,
                num_symbols=simulation_settings.num_symbols,
                num_subcarriers=simulation_settings.num_bands,
                constellation_order=simulation_settings.constellation_order,
                constellation_scheme=simulation_settings.constellation_type,
                modulator_type=simulation_settings.modulation_type,
                prefix_scheme=simulation_settings.prefix_type,
                prefix_length_ratio=simulation_settings.prefix_length_ratio,
                equalizator_type=simulation_settings.equalization_method,
                snr_db=snr,
                noise_scheme=simulation_settings.noise_type,
                power_allocation_type=simulation_settings.power_allocation_type,
                adaptive_modulation_mode=simulation_settings.adaptive_modulation_mode,
                min_constellation_order=simulation_settings.min_constellation_order,
                max_constellation_order=simulation_settings.max_constellation_order,
                desired_symbol_error_rate=simulation_settings.desired_symbol_error_rate,
                channel_impulse_response=channel_impulse_response,
            )
            simulations.append(simulation)
        return simulations

    def run(self) -> Dict[str, Any]:
        results = {}
        import time

        print("=" * 50)
        print("Starting OFDM-based System Simulation")
        print("=" * 50)

        # Serial to Parallel Conversion
        serial_to_parallel_converter: SerialToParallelConverter = SerialToParallelConverter()

        # Noise Model
        noise_model = self.NOISE_SCHEME_MAPPERS.get(self.noise_scheme, AWGNoiseModel)()

        # Channel Model
        if self.channel_impulse_response is not None:
            # Use custom channel from configuration
            channel_impulse_response: NDArray[np.complex128] = self.channel_impulse_response
            self._log(
                f"Using custom channel impulse response ({len(channel_impulse_response)} taps)"
            )
        else:
            # Use default multipath channel (backward compatibility)
            channel_impulse_response = np.array(
                [
                    (7.767824138452235072e-01 + 4.560896742466611919e-01j),
                    (-6.669848996328063551e-02 + 2.839935704583463338e-01j),
                    (1.398968327715586490e-01 - 1.591963958343969865e-01j),
                    (2.229949514514480494e-02 + 2.409945439452868821e-01j),
                ],
                dtype=np.complex128,
            )
            self._log("Using default multipath channel (4 taps)")

        channel: ChannelModel = ChannelModel(
            impulse_response=channel_impulse_response, snr_db=self.snr_db, noise_model=noise_model
        )
        prefix_length = int(self.prefix_length_ratio * channel.order)
        if self.prefix_scheme == PrefixType.NONE:
            prefix_length = 0
        print(f"Using prefix length: {prefix_length}")

        print(f"Signal to noise ratio: {self.snr_db} dB")

        # Prefix Scheme
        prefix_scheme = self.PREFIX_SCHEME_MAPPERS.get(self.prefix_scheme, NoPrefixScheme)(
            prefix_length=prefix_length
        )
        # Equalizator
        equalizator = self.EQUALIZATOR_SCHEME_MAPPERS.get(self.equalizator_type, NoEqualizator)(
            channel_frequency_response=np.fft.fft(channel_impulse_response, self.num_subcarriers),
            snr_db=self.snr_db,
        )

        # Modulator
        modulator: IModulator = self.MODULATOR_SCHEME_MAPPERS.get(
            self.modulator_type, OFDMModulator
        )(
            num_subcarriers=self.num_subcarriers,
            prefix_scheme=prefix_scheme,
            equalizator=equalizator,
        )

        # Pre-calculate channel parameters for power allocation and adaptive modulation
        channel_frequency_response = np.fft.fft(channel_impulse_response, self.num_subcarriers)
        channel_gains = np.abs(channel_frequency_response) ** 2
        noise_power = 10 ** (-self.snr_db / 10)
        print(f"Channel gains (|H|^2): {channel_gains}")
        print(f"Noise power: {noise_power:.6f}")

        # Determine constellation mapper and bit requirements based on adaptive mode
        water_level: Optional[float] = None
        constellation_orders: NDArray[np.int64]
        power_allocation: NDArray[np.float64] = np.array([])  # Will be assigned in either branch

        if self.adaptive_modulation_mode == AdaptiveModulationMode.CAPACITY_BASED:
            print("=" * 50)
            print("Configuring Adaptive Modulation (CAPACITY_BASED)")
            print("=" * 50)

            # Calculate power allocation first (needed for capacity calculation)
            if self.power_allocation_type == PowerAllocationType.WATERFILLING:
                power_allocator = WaterfillingPowerAllocation(
                    total_power=self.num_subcarriers,
                    channel_gains=channel_gains,
                    noise_power=noise_power,
                )
            else:
                power_allocator = UniformPowerAllocation(
                    total_power=self.num_subcarriers,
                    num_subcarriers=self.num_subcarriers,
                )

            power_allocation = power_allocator.allocate()

            # Calculate water level for waterfilling
            if self.power_allocation_type == PowerAllocationType.WATERFILLING:
                floor = noise_power / channel_gains
                water_level_array = power_allocation + floor
                water_level = float(np.mean(water_level_array[power_allocation > 1e-10]))

            # Calculate capacity per subcarrier
            # capacity = calculate_capacity_per_subcarrier(
            #     power_allocation, channel_gains, noise_power
            # )

            # # Determine constellation orders based on capacity
            # base_mapper_class = self.CONSTELLATION_SCHEME_MAPPERS.get(
            #     self.constellation_scheme, QAMConstellationMapper
            # )
            # constellation_orders = calculate_constellation_orders(
            #     capacity=capacity,
            #     min_order=self.min_constellation_order,
            #     max_order=self.max_constellation_order,
            #     scaling_factor=self.capacity_scaling_factor,
            #     base_mapper_class=base_mapper_class,
            # )

            # Calculate capacity per subcarrier using gap function method
            base_mapper_class: IConstellationMapper = self.CONSTELLATION_SCHEME_MAPPERS.get(
                self.constellation_scheme, QAMConstellationMapper
            )
            def calculate_snr_per_subcarrier(subcarrier_power, channel_gain):
                print("*"*20)
                print(f"SNR: {subcarrier_power * channel_gain / noise_power}")
                print("*"*20)
                return subcarrier_power * channel_gain / noise_power

            constellation_orders = [
                base_mapper_class.calculate_bit_loading_order(
                    ser=self.desired_symbol_error_rate, snr=calculate_snr_per_subcarrier(p_alloc, h_gain)
                ) for p_alloc, h_gain in zip(power_allocation, channel_gains)
            ]
            constellation_orders = np.array(constellation_orders, dtype=np.int64)
            print("Adaptive Constellation Orders per Subcarrier:")
            print(constellation_orders)


            # print(f"  Capacity range: {capacity.min():.2f} - {capacity.max():.2f} bits/symbol")
            print(
                f"  Constellation orders: min={constellation_orders[constellation_orders>0].min() if np.any(constellation_orders>0) else 0}, "
                f"max={constellation_orders.max()}, "
                f"mean={constellation_orders[constellation_orders>0].mean() if np.any(constellation_orders>0) else 0:.1f}"
            )
            print(
                f"  Active subcarriers: {np.sum(constellation_orders > 0)}/{self.num_subcarriers}"
            )
            if water_level is not None:
                print(f"  Water level (μ): {water_level:.6f}")

            # Create adaptive constellation mapper
            constellation_mapper: IConstellationMapper = AdaptiveConstellationMapper(
                constellation_orders=constellation_orders,
                base_mapper_class=base_mapper_class,
                num_subcarriers=self.num_subcarriers,
            )

            # Calculate bit requirements
            bits_per_subcarrier = constellation_mapper.get_bits_per_subcarrier()

            # Determine number of OFDM symbols
            if self.num_symbols is not None:
                num_ofdm_symbols = self.num_symbols
            elif self.num_bits is not None:
                bits_per_ofdm_symbol = int(np.sum(bits_per_subcarrier))
                if bits_per_ofdm_symbol == 0:
                    raise ValueError("All subcarriers have zero order - cannot transmit data")
                num_ofdm_symbols = self.num_bits // bits_per_ofdm_symbol
            else:
                raise ValueError("Either num_bits or num_symbols must be specified")

            # Create adaptive bits generator
            bits_generator: IGenerator = AdaptiveBitsGenerator(
                bits_per_subcarrier=bits_per_subcarrier,
                num_ofdm_symbols=num_ofdm_symbols,
            )

            total_bits = bits_generator.get_total_bits()

        else:
            # Fixed constellation mode (original behavior)
            constellation_orders = np.full(
                self.num_subcarriers, self.constellation_order, dtype=np.int64
            )
            constellation_mapper = self.CONSTELLATION_SCHEME_MAPPERS.get(
                self.constellation_scheme, QAMConstellationMapper
            )(order=self.constellation_order)
            bits_generator = RandomBitsGenerator()

            # Calculate total bits
            total_bits = self.num_bits
            if self.num_symbols is not None:
                total_bits = self.num_symbols * int(np.log2(self.constellation_order))
                print(self.constellation_order)

        results.update(
            {
                "num_bits": self.num_bits,
                "num_symbols": self.num_symbols,
                "num_subcarriers": self.num_subcarriers,
                "constellation_order": self.constellation_order,
                "constellation_scheme": self.constellation_scheme.name,
                "modulator_type": self.modulator_type.name,
                "prefix_scheme": self.prefix_scheme.name,
                "prefix_acronym": prefix_scheme.acronym,
                "equalizator_type": self.equalizator_type.name,
                "snr_db": self.snr_db,
                "noise_scheme": self.noise_scheme.name,
                "power_allocation_type": self.power_allocation_type.name,
                "power_allocation_acronym": (
                    "WF"
                    if self.power_allocation_type == PowerAllocationType.WATERFILLING
                    else "UNIFORM"
                ),
                "adaptive_modulation_mode": self.adaptive_modulation_mode.name,
                "constellation_order_per_subcarrier": constellation_orders.tolist(),
                "water_level": water_level,
                "title": (
                    f"{prefix_scheme.acronym}-"
                    f"{self.modulator_type.name}-{self.equalizator_type.name}"
                ),
                "subtitle": (
                    f"{self.constellation_order}{self.constellation_scheme.name}-"
                    f"SNR{self.snr_db}dB-{self.power_allocation_type.name}"
                ),
            }
        )

        print("=" * 50)
        print("Generating bits stream...")
        print("=" * 50)

        if total_bits is None:
            raise ValueError("Total bits could not be determined.")

        print(f"Generating {total_bits} random bits...")
        bits = bits_generator.generate_bits(total_bits)

        # Read bits from the binary stream just to display
        bits_list = read_bits_from_stream(bits)
        print(f"Generated Bits Length: {len(bits_list)}")
        # Do not use bits_list further; use bits directly for mapping

        print("=" * 50)
        print("Mapping bits to constellation symbols...")
        print("=" * 50)
        symbols: NDArray[np.complex128] = constellation_mapper.encode(bits)

        print(f"Mapped Symbols Length: {len(symbols)}")

        print("=" * 50)
        print("Serial to Parallel Conversion...")
        print("=" * 50)
        parallel_data = serial_to_parallel_converter.to_parallel(
            symbols, self.num_subcarriers  # type: ignore
        )

        print(f"Parallel Data Shape: {parallel_data.shape}")

        # Power Allocation
        print("=" * 50)
        print(f"Applying Power Allocation ({self.power_allocation_type.name})...")
        print("=" * 50)

        # Calculate power allocation for fixed mode (adaptive mode already calculated it above)
        if self.adaptive_modulation_mode == AdaptiveModulationMode.FIXED:
            if self.power_allocation_type == PowerAllocationType.WATERFILLING:
                power_allocator = WaterfillingPowerAllocation(
                    total_power=1.0,
                    channel_gains=channel_gains,
                    noise_power=noise_power,
                )
                power_allocation = power_allocator.allocate()

                # Calculate water level for results
                floor = noise_power / channel_gains
                water_level_array = power_allocation + floor
                water_level = float(np.mean(water_level_array[power_allocation > 1e-10]))
            else:
                power_allocator = UniformPowerAllocation(
                    total_power=1.0,
                    num_subcarriers=self.num_subcarriers,
                )
                power_allocation = power_allocator.allocate()

        self._log(
            f"Power allocation computed: min={power_allocation.min():.6f}, max={power_allocation.max():.6f}"
        )

        # Apply power allocation to parallel data
        # parallel_data = parallel_data * np.sqrt(power_allocation)
        results["allocated_power"] = power_allocation.tolist()

        print("=" * 50)
        print("OFDM-based Modulation")
        print("=" * 50)
        modulated_signal = modulator.modulate(parallel_data)

        print(f"Modulated Signal Shape: {modulated_signal.shape}")

        # Calculate PAPR
        power_signal = np.abs(modulated_signal) ** 2
        peak_power = np.max(power_signal)
        avg_power = np.mean(power_signal)
        papr_db = 10 * np.log10(peak_power / avg_power) if avg_power > 0 else float("inf")
        print(f"PAPR: {papr_db:.2f} dB")
        results.update({"papr_db": papr_db})

        print("=" * 50)
        print("Parallel to Serial Conversion...")
        print("=" * 50)
        serial_signal = serial_to_parallel_converter.to_serial(modulated_signal)
        print(f"Serial Signal Shape: {serial_signal.shape}")

        print("=" * 50)
        print("Transmitting signal through the channel...")
        print("=" * 50)

        # Start timing
        start_time = time.perf_counter()
        received_signal = channel.transmit(serial_signal)

        print(f"Received Signal Shape: {received_signal.shape}")

        print("=" * 50)
        print("Serial to Parallel Conversion of Received Signal...")
        print("=" * 50)

        received_parallel_signal = serial_to_parallel_converter.to_parallel(
            received_signal, self.num_subcarriers + prefix_scheme.prefix_length  # type: ignore
        )
        print(f"Received Parallel Signal Shape: {received_parallel_signal.shape}")

        print("=" * 50)
        print("OFDM-based Demodulation")
        print("=" * 50)
        demodulated_data = modulator.demodulate(received_parallel_signal)
        print(f"Demodulated Data Shape: {demodulated_data.shape}")

        # Normalize demodulated data regardless of power allocation
        # print("=" * 50)
        # print("Normalizing Demodulated Data...")
        # print("=" * 50)
        # for sc_idx in range(self.num_subcarriers):
        #     power = np.mean(np.abs(demodulated_data[:, sc_idx]) ** 2)
        #     if power > 1e-10:
        #         demodulated_data[:, sc_idx] /= np.sqrt(power)

        # self._log("Power allocation compensation applied")

        print("=" * 50)
        print("Parallel to Serial Conversion of Demodulated Data...")
        print("=" * 50)
        demodulated_serial_data = serial_to_parallel_converter.to_serial(demodulated_data)
        print(f"Demodulated Serial Data Shape: {demodulated_serial_data.shape}")

        # Normalize symbols to constellation scale
        # print("=" * 50)
        # print("Normalizing symbols for constellation demapping...")
        # print("=" * 50)
        # # Calculate current average power of received symbols
        # current_avg_power = np.mean(np.abs(demodulated_serial_data) ** 2)
        # # Constellation has unit average power, so normalize to unit power
        # if current_avg_power > 1e-10:  # Avoid division by zero
        #     normalization_factor = np.sqrt(current_avg_power)
        #     demodulated_serial_data = demodulated_serial_data / normalization_factor
        #     self._log(f"Normalized symbols: avg power {current_avg_power:.6f} -> 1.0")
        # else:
        #     self._log("Warning: Received signal has near-zero power, skipping normalization")

        print("=" * 50)
        print("Constellation Demapping...")
        print("=" * 50)
        received_bits = constellation_mapper.decode(demodulated_serial_data)

        received_bits_list = read_bits_from_stream(received_bits)
        print(f"Received Bits Length: {len(received_bits_list)}")

        # Calculate Bit Error Rate (BER)
        bit_errors = sum(b1 != b2 for b1, b2 in zip(bits_list, received_bits_list))
        ber = bit_errors / total_bits if total_bits > 0 else 0.0
        print("=" * 50)
        print(f"Bit Errors: {bit_errors} out of {total_bits} bits")
        print(f"Bit Error Rate (BER): {ber:.6f}")

        # Calculate Symbol Error Rate (SER) for information
        recoded_symbols = constellation_mapper.encode(received_bits)
        symbol_errors = np.sum(symbols != recoded_symbols)
        ser = symbol_errors / len(symbols) if len(symbols) > 0 else 0.0
        print(f"Symbol Errors: {symbol_errors} out of {len(symbols)} symbols")
        print(f"Symbol Error Rate (SER): {ser:.6f}")
        print("=" * 50)

        results.update(
            {
                "bit_errors": bit_errors,
                "symbol_errors": symbol_errors,
                "total_bits": total_bits,
                "bit_error_rate": ber,
                "symbol_error_rate": ser,
                "received_symbols": demodulated_serial_data,  # Store for visualization
            }
        )

        # Generate plot image and store in results

        # Clear previous plots
        plt.clf()
        plt.cla()

        # Determine if we should create adaptive visualization
        is_adaptive = self.adaptive_modulation_mode == AdaptiveModulationMode.CAPACITY_BASED

        if is_adaptive:
            # Create figure with two subplots for adaptive mode
            fig = plt.figure(figsize=(16, 8))

            # Subplot 1: Constellation diagram
            ax1 = plt.subplot(1, 2, 1)

            # Plot received symbols (all in one color for simplicity)
            ax1.scatter(
                demodulated_serial_data.real,
                demodulated_serial_data.imag,
                color="blue",
                marker=".",
                alpha=0.1,
                label="Received Symbols",
            )

            # Plot ideal constellation points (combined from all orders)
            ideal_points = constellation_mapper.constellation
            ax1.scatter(
                ideal_points.real,
                ideal_points.imag,
                color="red",
                marker="o",
                s=50,
                label="Ideal Constellation Points",
            )

            # Set plot attributes
            ax1.set_title("Constellation Diagram (Adaptive Modulation)")
            ax1.set_xlabel("In-Phase")
            ax1.set_ylabel("Quadrature")
            ax1.axhline(0, color="black", lw=0.5)
            ax1.axvline(0, color="black", lw=0.5)
            ax1.legend(loc="upper right")
            ax1.grid(True)
            ax1.set_xlim(-1.5, 1.5)
            ax1.set_ylim(-1.5, 1.5)
            ax1.set_aspect("equal")

            # Add text box with BER, SNR, and PAPR
            textstr = f"BER: {ber:.6f}\nSNR: {self.snr_db} dB\nPAPR: {papr_db:.2f} dB"
            ax1.text(
                0.05,
                0.95,
                textstr,
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8),
            )

            # Subplot 2: Constellation order distribution
            ax2 = plt.subplot(1, 2, 2)

            # Count constellation orders
            unique_orders, counts = np.unique(
                constellation_orders[constellation_orders > 0], return_counts=True
            )

            # Create bar chart
            from matplotlib import cm

            colors = cm.viridis(np.linspace(0, 1, len(unique_orders)))  # type: ignore
            bars = ax2.bar(range(len(unique_orders)), counts, color=colors, edgecolor="black")

            # Set labels and title
            ax2.set_xlabel("Constellation Order (M-QAM/PSK)")
            ax2.set_ylabel("Number of Subcarriers")
            ax2.set_title("Constellation Order Distribution")
            ax2.set_xticks(range(len(unique_orders)))
            ax2.set_xticklabels([f"{int(order)}" for order in unique_orders])
            ax2.grid(True, axis="y", alpha=0.3)

            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(count)}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            # Add statistics text box
            active_subcarriers = np.sum(constellation_orders > 0)
            inactive_subcarriers = np.sum(constellation_orders == 0)
            avg_order = (
                np.mean(constellation_orders[constellation_orders > 0])
                if active_subcarriers > 0
                else 0
            )

            stats_text = (
                f"Total Subcarriers: {self.num_subcarriers}\n"
                f"Active: {active_subcarriers}\n"
                f"Inactive: {inactive_subcarriers}\n"
                f"Avg Order: {avg_order:.1f}"
            )

            ax2.text(
                0.98,
                0.98,
                stats_text,
                transform=ax2.transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(facecolor="white", alpha=0.8),
            )

            plt.tight_layout()

        else:
            # Original single plot for fixed mode
            plt.figure(figsize=(8, 8))

            # Plot received Symbols
            plt.scatter(
                demodulated_serial_data.real,
                demodulated_serial_data.imag,
                color="blue",
                marker=".",
                alpha=0.1,
                label="Received Symbols",
            )

            # Plot the constellation diagram
            ideal_points = constellation_mapper.constellation
            plt.scatter(
                ideal_points.real,
                ideal_points.imag,
                color="red",
                marker="o",
                label="Ideal Constellation Points",
            )

            # Set plot attributes
            plt.title(f"{results['title']}")
            plt.xlabel("In-Phase")
            plt.ylabel("Quadrature")
            plt.axhline(0, color="black", lw=0.5)
            plt.axvline(0, color="black", lw=0.5)
            plt.legend(loc="upper right")
            plt.grid(True)
            plt.xlim([-1.5, 1.5])
            plt.ylim([-1.5, 1.5])
            plt.axis("equal")

            # Add text box with BER, SNR, and PAPR
            textstr = f"BER: {ber:.6f}\n" f"SNR: {self.snr_db} dB\n" f"PAPR: {papr_db:.2f} dB"
            plt.gcf().text(
                0.15, 0.75, textstr, fontsize=10, bbox=dict(facecolor="white", alpha=0.5)
            )
            plt.tight_layout()

        # Save the plot to a BytesIO object
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format="png")
        img_buffer.seek(0)

        # Load the image from the BytesIO object
        image = Image.open(img_buffer)
        results["constellation_plot"] = image

        # Close the plot to free memory
        plt.close()

        # Close the bits stream
        bits.close()

        # Calculate transmission timing metrics
        end_time = time.perf_counter()
        transmission_time_ms = (end_time - start_time) * 1000
        bitrate_mbps = (total_bits) / 1e6

        results["transmission_time_ms"] = transmission_time_ms
        results["bitrate_mbps"] = bitrate_mbps

        print("=" * 50)
        print(f"Transmission time: {transmission_time_ms:.2f} ms")
        print(f"Bitrate: {bitrate_mbps:.2f} Mbps")
        print("Simulation completed.")
        print("=" * 50)

        return results
