import numpy as np
from numpy.typing import NDArray

from ofdm_based_systems.noise.models import AWGNoiseModel, INoiseModel


class ChannelModel:
    def __init__(
        self,
        impulse_response: NDArray[np.complex128],
        snr_db: float,
        noise_model: INoiseModel = AWGNoiseModel(),
    ):
        self.impulse_response: NDArray[np.complex128] = self.normalize_impulse_response(
            impulse_response
        )
        self.snr_db: float = snr_db
        self.noise_model: INoiseModel = noise_model
        self.frequency_response_cache: dict[int, NDArray[np.complex128]] = {}

    @property
    def order(self) -> int:
        """Get the order of the channel (length of impulse response - 1)."""
        return len(self.impulse_response) - 1

    def get_frequency_response(self, n_fft: int) -> NDArray[np.complex128]:
        """Get the frequency response of the channel for a given FFT size."""
        if n_fft not in self.frequency_response_cache:
            self.frequency_response_cache[n_fft] = np.fft.fft(self.impulse_response, n=n_fft)
        return self.frequency_response_cache[n_fft]

    def get_gains(self, n_fft: int) -> NDArray[np.float64]:
        """Get the channel gains (magnitude squared of frequency response) for a given FFT size."""
        frequency_response = self.get_frequency_response(n_fft)
        return np.abs(frequency_response) ** 2

    def normalize_impulse_response(
        self, impulse_response: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        """Normalize the impulse response to have unit power."""
        power = np.sum(np.abs(impulse_response) ** 2)
        if power == 0:
            raise ValueError("Impulse response cannot be all zeros.")
        return impulse_response / np.sqrt(power)

    def transmit(self, signal: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Transmit a signal through the channel, applying convolution and adding noise."""
        if signal.ndim != 1:
            raise ValueError("Signal must be serial (1D array)")

        # Convolve signal with channel impulse response
        convolved_signal = np.convolve(signal, self.impulse_response, mode="full")

        # Remove the extra samples added by convolution
        convolved_signal = convolved_signal[: signal.shape[0]]

        # Add noise to the convolved signal
        noisy_signal = self.noise_model.add_noise(
            convolved_signal.astype(np.complex128), self.snr_db
        )

        return noisy_signal
