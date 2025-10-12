from abc import ABC, abstractmethod

from numpy.typing import NDArray
import numpy as np


class IPrefixScheme(ABC):
    prefix_length: int

    def __init__(self, prefix_length: int = 0):
        if prefix_length < 0:
            raise ValueError("Prefix length must be a non-negative integer.")
        self.prefix_length = prefix_length

    @property
    @abstractmethod
    def prefix_acronym(self) -> str:
        pass

    @abstractmethod
    def add_prefix(self, symbols: NDArray[np.complex128]) -> NDArray[np.complex128]:
        pass

    @abstractmethod
    def remove_prefix(self, symbols: NDArray[np.complex128]) -> NDArray[np.complex128]:
        pass


class CyclicPrefixScheme(IPrefixScheme):
    @property
    def prefix_acronym(self) -> str:
        return "CP"

    def add_prefix(self, symbols: NDArray[np.complex128]) -> NDArray[np.complex128]:
        if symbols.ndim != 1:
            raise ValueError("Input symbols must be a 1D array.")
        if len(symbols) < self.prefix_length:
            raise ValueError("Input symbols length must be greater than prefix length.")

        prefix = symbols[-self.prefix_length :]
        return np.concatenate((prefix, symbols))

    def remove_prefix(self, symbols: NDArray[np.complex128]) -> NDArray[np.complex128]:
        if symbols.ndim != 1:
            raise ValueError("Input symbols must be a 1D array.")
        if len(symbols) <= self.prefix_length:
            raise ValueError("Input symbols length must be greater than prefix length.")

        return symbols[self.prefix_length :]


class ZeroPaddingPrefixScheme(IPrefixScheme):
    @property
    def prefix_acronym(self) -> str:
        return "ZP"

    def add_prefix(self, symbols: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Adds a zero-padding prefix to the input symbols.
        Although this is not properly a "prefix", the method is named for consistency.
        """
        if symbols.ndim != 1:
            raise ValueError("Input symbols must be a 1D array.")

        prefix = np.zeros(self.prefix_length, dtype=symbols.dtype)
        return np.concatenate((symbols, prefix))

    def remove_prefix(self, symbols: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        The removal of the zero-padding "prefix" is not done exactly by removing the prefix,
        but by overlapping the prefix length slice from the end to the beginning and then
        adding the two parts together. We achieve this result by creating, first, an identity
        matrix of the size of the original symbols (without prefix), then appending another
        smaler identity matrix (of size prefix_length) to the right of it, padding zeros to
        the bottom of that smaller identity matrix to make it the same size as the original.
        Finally, we multiply this matrix by the symbols with prefix, effectively summing the
        overlapping parts.
        """
        if symbols.ndim != 1:
            raise ValueError("Input symbols must be a 1D array.")
        if len(symbols) <= self.prefix_length:
            raise ValueError("Input symbols length must be greater than prefix length.")

        original_length = len(symbols) - self.prefix_length
        identity_main = np.eye(original_length, dtype=symbols.dtype)
        identity_prefix = np.eye(self.prefix_length, dtype=symbols.dtype)
        overlap_matrix = np.vstack(
            (
                identity_prefix,
                np.zeros(
                    (original_length - self.prefix_length, self.prefix_length),
                    dtype=symbols.dtype,
                ),
            )
        )
        transformation_matrix = np.hstack((identity_main, overlap_matrix))

        return transformation_matrix @ symbols  # Matrix-vector multiplication


class NoPrefixScheme(IPrefixScheme):
    @property
    def prefix_acronym(self) -> str:
        return ""

    def add_prefix(self, symbols: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return symbols

    def remove_prefix(self, symbols: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return symbols
