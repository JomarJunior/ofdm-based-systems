from numpy.typing import NDArray
import numpy as np


class SerialToParallelConverter:
    @staticmethod
    def to_parallel(data: NDArray[np.complex128], num_streams: int) -> NDArray[np.complex128]:
        print(f"Converting to parallel with {num_streams} streams.")
        if data.ndim != 1:
            raise ValueError("Input data must be a 1D array.")
        if num_streams <= 0:
            raise ValueError("Number of streams must be a positive integer.")
        if len(data) % num_streams != 0:
            raise ValueError("Length of data must be divisible by number of streams.")

        return data.reshape(-1, num_streams)

    @staticmethod
    def to_serial(data: NDArray[np.complex128]) -> NDArray[np.complex128]:
        if data.ndim != 2:
            raise ValueError("Input data must be a 2D array.")
        return data.flatten()
