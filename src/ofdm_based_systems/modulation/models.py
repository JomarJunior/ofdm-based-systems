from abc import ABC, abstractmethod

from numpy.typing import NDArray
import numpy as np
from ofdm_based_systems.equalization.models import IEqualizator
from ofdm_based_systems.prefix.models import IPrefixScheme


class IModulator(ABC):
    @abstractmethod
    def modulate(self, symbols: NDArray[np.complex128]) -> NDArray[np.complex128]:
        pass

    @abstractmethod
    def demodulate(self, symbols: NDArray[np.complex128]) -> NDArray[np.complex128]:
        pass


class OFDMModulator(IModulator):
    def __init__(
        self, num_subcarriers: int, prefix_scheme: IPrefixScheme, equalizator: IEqualizator
    ):
        self.num_subcarriers = num_subcarriers
        self.prefix_scheme = prefix_scheme
        self.equalizator = equalizator

    def modulate(self, symbols: NDArray[np.complex128]) -> NDArray[np.complex128]:
        if symbols.shape[1] != self.num_subcarriers:
            raise ValueError(f"Number of symbols must be {self.num_subcarriers}")

        # Perform IFFT to convert frequency domain symbols to time domain
        time_domain_signal = np.fft.ifft(symbols, n=self.num_subcarriers, axis=1, norm="ortho")

        # Add prefix to each row
        time_domain_signal_with_prefix = np.array(
            [self.prefix_scheme.add_prefix(row) for row in time_domain_signal]
        )

        return time_domain_signal_with_prefix

    def demodulate(self, symbols: NDArray[np.complex128]) -> NDArray[np.complex128]:
        # Remove prefix from each row
        time_domain_signal = np.array([self.prefix_scheme.remove_prefix(row) for row in symbols])

        # Perform FFT to convert time domain signal back to frequency domain
        frequency_domain_symbols = np.fft.fft(
            time_domain_signal, n=self.num_subcarriers, axis=1, norm="ortho"
        )

        # Equalize the received symbols by row
        frequency_domain_symbols = np.array(
            [self.equalizator.equalize(row) for row in frequency_domain_symbols]
        )

        return frequency_domain_symbols


class SingleCarrierOFDMModulator(IModulator):
    def __init__(
        self, prefix_scheme: IPrefixScheme, equalizator: IEqualizator, num_subcarriers: int
    ):
        self.prefix_scheme = prefix_scheme
        self.equalizator = equalizator
        self.num_subcarriers = num_subcarriers

    def modulate(self, symbols: NDArray[np.complex128]) -> NDArray[np.complex128]:
        # Add prefix to each row
        time_domain_signal_with_prefix = np.array(
            [self.prefix_scheme.add_prefix(row) for row in symbols]
        )

        return time_domain_signal_with_prefix

    def demodulate(self, symbols: NDArray[np.complex128]) -> NDArray[np.complex128]:
        # Remove prefix from each row
        time_domain_signal = np.array([self.prefix_scheme.remove_prefix(row) for row in symbols])

        # Perform FFT to convert time domain signal back to frequency domain
        frequency_domain_symbols = np.fft.fft(
            time_domain_signal, n=self.num_subcarriers, axis=1, norm="ortho"
        )

        # Equalize the received symbols by row
        frequency_domain_symbols = np.array(
            [self.equalizator.equalize(row) for row in frequency_domain_symbols]
        )

        # Convert back to time domain using IFFT
        result = np.fft.ifft(frequency_domain_symbols, n=self.num_subcarriers, axis=1, norm="ortho")

        return result
