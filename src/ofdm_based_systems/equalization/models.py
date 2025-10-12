from abc import ABC, abstractmethod
from typing import Optional

from numpy.typing import NDArray
import numpy as np


class IEqualizator(ABC):
    def __init__(
        self,
        channel_frequency_response: NDArray[np.complex128],
        snr_db: Optional[float] = None,
    ):
        self.channel_frequency_response = channel_frequency_response
        self.snr_db = snr_db

    @abstractmethod
    def equalize(self, received_symbols: NDArray[np.complex128]) -> NDArray[np.complex128]:
        pass


class ZeroForcingEqualizator(IEqualizator):
    def equalize(
        self,
        received_symbols: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        if received_symbols.shape != self.channel_frequency_response.shape:
            raise ValueError(
                "Received symbols and channel frequency response must have the same shape."
            )

        # Avoid division by zero by adding a small epsilon where the channel response is zero
        epsilon = 1e-10
        h = np.where(self.channel_frequency_response == 0, epsilon, self.channel_frequency_response)
        return received_symbols / h


class MMSEEqualizator(IEqualizator):
    def calculate_noise_variance(self, received_signal: NDArray[np.complex128]) -> float:
        if self.snr_db is None:
            raise ValueError("SNR in dB must be provided to calculate noise variance.")

        signal_power = np.mean(np.abs(received_signal) ** 2)
        snr_linear = 10 ** (self.snr_db / 10)
        channel_gain = np.mean(np.abs(self.channel_frequency_response) ** 2)
        noise_power = signal_power / snr_linear
        if channel_gain == 0:
            return float("inf")  # Infinite noise variance if channel gain is zero
        return float(noise_power / channel_gain)

    def equalize(self, received_symbols: NDArray[np.complex128]) -> NDArray[np.complex128]:
        noise_variance = self.calculate_noise_variance(received_symbols)

        if received_symbols.shape != self.channel_frequency_response.shape:
            raise ValueError(
                "Received symbols and channel frequency response must have the same shape."
            )

        h_conj = np.conj(self.channel_frequency_response)
        h_abs_squared = np.abs(self.channel_frequency_response) ** 2
        mmse_filter = h_conj / (h_abs_squared + noise_variance)

        return received_symbols * mmse_filter


class NoEqualizator(IEqualizator):
    def equalize(self, received_symbols: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return received_symbols
