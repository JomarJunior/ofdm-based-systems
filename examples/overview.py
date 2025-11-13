from datetime import datetime
from typing import BinaryIO, Dict, List, Optional, Union

import numpy as np

from ofdm_based_systems.channel.models import ChannelModel
from ofdm_based_systems.constellation.models import (
    PSKConstellationMapper,
    QAMConstellationMapper,
)
from ofdm_based_systems.equalization.models import (
    MMSEEqualizator,
)
from ofdm_based_systems.modulation.models import OFDMModulator
from ofdm_based_systems.power_allocation.models import (
    WaterfillingPowerAllocation,
    calculate_capacity_per_subcarrier,
)
from ofdm_based_systems.prefix.models import CyclicPrefixScheme
from ofdm_based_systems.simulation.models import SerialToParallelConverter

# 01100101 -> 8bits -> 1Byte

# Constelacao 4 -> 00, 01, 10, 11
#                  s0, s1, s2, s3
#                  1hz, 4hz, 8hz, 16hz
# s1, s2, s1, s1
# 01, 10, 01, 01 -> 01100101

# Constelacao maior: 64
# num_bits = log2(64) -> 8
# sx = 01100101
# recebido = 10011110


def read_bits_from_stream(bit_stream: Union[BinaryIO, List[int]]) -> List[int]:
    """Reads bits from a binary stream or list of bits and returns them as a list of integers (0s and 1s)."""
    if isinstance(bit_stream, list):
        return bit_stream
    bits = []
    byte = bit_stream.read(1)
    while byte:
        byte_value = ord(byte)
        for i in range(8):
            bits.append((byte_value >> (7 - i)) & 1)
        byte = bit_stream.read(1)
    return bits


subcarriers = 64

# Channel creation
snr = 15.0  # in dB
snr_linear = 10 ** (snr / 10)
print(f"SNR (dB): {snr}")
print(f"SNR (linear scale): {snr_linear}")
channel_impulse_response = np.load("config/channel_models/Lin-Phoong_P2.npy")
print(f"Channel Impulse Response shape: {channel_impulse_response.shape}")
channel_model = ChannelModel(
    impulse_response=channel_impulse_response,
    snr_db=snr,
)

# power allocation
power_allocator = WaterfillingPowerAllocation(
    total_power=1.0,
    channel_gains=(np.abs(channel_model.get_frequency_response(n_fft=subcarriers)) ** 2),
    noise_power=(1 / snr_linear),
)

allocated_powers = power_allocator.allocate()
print(f"Allocated Powers: {allocated_powers}")
# snr per subcarrier
capacity_per_subcarrier = calculate_capacity_per_subcarrier(
    power_allocation=allocated_powers,
    channel_gains=(np.abs(channel_model.get_frequency_response(n_fft=subcarriers)) ** 2),
    noise_power=(1 / snr_linear),
)

print(f"Capacity per Subcarrier: {np.int64(capacity_per_subcarrier)}")

# qam must enforce bits to be even numbers
original_capacity_per_subcarrier = capacity_per_subcarrier.copy().astype(np.int64)
capacity_per_subcarrier = np.array([int(c // 2 * 2) for c in capacity_per_subcarrier])


order_per_subcarrier = [int(2**c) if c > 0 else 0 for c in capacity_per_subcarrier]

# increase order empirically based on ber
order_per_subcarrier = [o * 2**2 for o in order_per_subcarrier]

if sum(order_per_subcarrier) == 0:
    print("All subcarriers have zero order, adjusting all to minimum order of 4.")
    order_per_subcarrier = [int(2**4 * o) for o in original_capacity_per_subcarrier]

print(f"Order per Subcarrier: {order_per_subcarrier}")

unique_orders = set(order_per_subcarrier)

constellation_generators: Dict[int, Optional[QAMConstellationMapper]] = {}

for order in unique_orders:
    if order > 0:
        constellation_generators[order] = QAMConstellationMapper(order=order)
    else:
        constellation_generators[order] = None  # No transmission on this subcarrier

# constellation_generators -> Dict[order: int, generator: QAMConstellationMapper]
# Big picture:
# - we start with a number of symbols to generate
num_symbols = 100000
# We need to generate bits according to the order of the constellation per subcarrier
# The symbols are a single complex number, regardless of how much bits they represent.
# The list of symbols to transmit should follow the order of the subcarriers and their respective constellation orders.
symbols = np.array([], dtype=np.complex128)
symbols_colors = []
colors_order_map = {
    0: "black",
    2: "red",
    4: "blue",
    16: "green",
    64: "orange",
    256: "purple",
    1024: "brown",
    2048: "pink",
}
generated_bits = []
while len(symbols) < num_symbols:
    for idx, order in enumerate(order_per_subcarrier):
        constellation: Optional[QAMConstellationMapper] = constellation_generators[order]
        if constellation is not None:
            bits_per_symbol = int(np.log2(order))
            # generate random bits
            bits = np.random.randint(0, 2, bits_per_symbol).tolist()
            generated_bits.extend(bits)
            # map bits to symbol
            symbol = constellation.encode(bits)
            # add waterfilling power allocation
            symbol = symbol * np.sqrt(allocated_powers[idx])
            symbols = np.append(symbols, symbol)
            symbols_colors.append(colors_order_map[order])
        else:
            # To maintain alignment, we can append a zero symbol (no transmission)
            symbols = np.append(symbols, 0.0 + 0.0j)
            symbols_colors.append(colors_order_map[order])

print(f"Total bits generated: {len(generated_bits)}")
print(f"Total symbols generated: {len(symbols)}")
print(f"Symbols: {symbols}")

# plot symbols in complex plane
from matplotlib import pyplot as plt

plt.scatter(symbols.real, symbols.imag, c=symbols_colors, alpha=0.5)
plt.title("Transmitted Symbols in Complex Plane")
plt.xlabel("In-Phase")
plt.ylabel("Quadrature")
plt.grid(True)
plt.savefig("waterfilling_transmitted_symbols.png")
plt.clf()

# =============
# Input: mapped_bits -> symbols to be transmitted
# =============
mapped_bits = symbols

started_ofdm_transmition = datetime.now()
print(f"Starting OFDM transmission at {started_ofdm_transmition}")

# ofdm
ofdm_modulator: OFDMModulator = OFDMModulator(
    num_subcarriers=subcarriers,
    prefix_scheme=CyclicPrefixScheme(prefix_length=channel_impulse_response.shape[0] - 1),
    equalizator=MMSEEqualizator(
        channel_frequency_response=channel_model.get_frequency_response(n_fft=subcarriers),
        snr_db=snr,
    ),
)

# serial to parallel
serial_to_parallel: SerialToParallelConverter = SerialToParallelConverter()

mapped_bits = serial_to_parallel.to_parallel(mapped_bits, ofdm_modulator.num_subcarriers)
print(f"Mapped bits after serial to parallel shape: {mapped_bits.shape}")

mapped_bits = ofdm_modulator.modulate(mapped_bits)
print(f"OFDM modulated symbols shape: {mapped_bits.shape}")

# parallel to serial
mapped_bits = serial_to_parallel.to_serial(mapped_bits)
print(f"Mapped bits after parallel to serial shape: {mapped_bits.shape}")

# transmit through channel
received_symbols = channel_model.transmit(mapped_bits)
print(f"Received symbols shape: {received_symbols.shape}")

# serial to parallel
received_symbols = serial_to_parallel.to_parallel(
    received_symbols, ofdm_modulator.num_subcarriers + ofdm_modulator.prefix_scheme.prefix_length
)
print(f"Received symbols after serial to parallel shape: {received_symbols.shape}")

# ofdm demodulation
received_symbols = ofdm_modulator.demodulate(received_symbols)
print(f"OFDM demodulated symbols shape: {received_symbols.shape}")

# normalize per subcarrier
for idx in range(received_symbols.shape[1]):
    power = np.mean(np.abs(received_symbols[:, idx]) ** 2)
    if power > 0:
        received_symbols[:, idx] = received_symbols[:, idx] / np.sqrt(power)


# parallel to serial
received_symbols = serial_to_parallel.to_serial(received_symbols)
print(f"Received symbols after parallel to serial shape: {received_symbols.shape}")

ended_ofdm_transmition = datetime.now()
print(f"Ended OFDM transmission at {ended_ofdm_transmition}")

# ===========
# Outputs: received_symbols -> symbols received after channel
# ===========

print("Decoding mapped bits for verification...")

# demap according to subcarrier order
# received_symbols is a 1D array of complex symbols
tx = symbols
rx = received_symbols
print(f"Transmitted symbols shape: {tx.shape}")
print(f"Received symbols shape: {rx.shape}")

symbol_errors = 0
original_bits = []
symbols_without_zeros = []
colors_without_zeros = []
for idx, symbol in enumerate(rx):
    constellation: Optional[QAMConstellationMapper] = constellation_generators[
        order_per_subcarrier[idx % subcarriers]
    ]
    if constellation is not None:
        symbols_without_zeros.append(symbol)
        colors_without_zeros.append(symbols_colors[idx])
        decoded_symbol = constellation.decode(symbol)
        original_bits.extend(
            read_bits_from_stream(decoded_symbol)[0 : int(np.log2(constellation.order))]
        )

print(f"Decoded total bits: {len(original_bits)}")

# remove 0 symbols

# calculate bit errors
bit_errors = sum(b1 != b2 for b1, b2 in zip(generated_bits[: len(original_bits)], original_bits))
print(f"Bit Errors: {bit_errors} out of {len(original_bits)} bits")
ber = bit_errors / len(original_bits)
print(f"Bit Error Rate (BER): {ber}")

# plot received symbols in complex plane
plt.scatter(
    np.array(symbols_without_zeros).real,
    np.array(symbols_without_zeros).imag,
    c=colors_without_zeros,
    alpha=0.1,
)
plt.title("Received Symbols in Complex Plane")
plt.xlabel("In-Phase")
plt.ylabel("Quadrature")

# add the constellation points for reference
for order, constellation in constellation_generators.items():
    if constellation is not None:
        points = constellation.constellation  # assuming uniform power for reference
        plt.scatter(
            points.real,
            points.imag,
            label=f"Constellation Order {order}",
            marker="o",
            s=100,
            edgecolors="k",
        )

plt.grid(True)
plt.savefig("waterfilling_received_symbols.png")
plt.clf()

tx_time_ms = (ended_ofdm_transmition - started_ofdm_transmition).total_seconds() * 1000
bits_per_second = len(generated_bits) / (tx_time_ms / 1000)
print("OFDM transmission and reception simulation completed.")
print("=" * 20)
print(f"Total transmission time: {tx_time_ms} ms")
print(f"Total bits transmitted: {len(generated_bits)}")
print(f"Throughput: {bits_per_second / 1e6:.0f} mbits/second")
print(f"Total transmission time: {tx_time_ms} ms")
print(f"Total bits transmitted: {len(generated_bits)}")
print(f"Throughput: {bits_per_second / 1e6:.0f} mbits/second")
