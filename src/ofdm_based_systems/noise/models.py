from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class INoiseModel(ABC):
    @abstractmethod
    def add_noise(self, signal: NDArray[np.complex128], snr_db: float) -> NDArray[np.complex128]:
        pass


class AWGNoiseModel(INoiseModel):
    def add_noise(self, signal: NDArray[np.complex128], snr_db: float) -> NDArray[np.complex128]:
        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate complex Gaussian noise
        noise = np.sqrt(noise_power / 2) * (
            np.random.normal(size=signal.shape) + 1j * np.random.normal(size=signal.shape)
        )
        return signal + noise


class NoNoiseModel(INoiseModel):
    def add_noise(self, signal: NDArray[np.complex128], snr_db: float) -> NDArray[np.complex128]:
        return signal  # No noise added
