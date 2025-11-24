"""Power allocation strategies for OFDM systems.

This module implements various power allocation algorithms including
uniform distribution and waterfilling for capacity maximization.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class IPowerAllocation(ABC):
    """Abstract base class for power allocation strategies.

    All power allocation implementations must inherit from this class
    and implement the allocate() method.
    """

    @abstractmethod
    def allocate(self) -> NDArray[np.float64]:
        """Allocate power across subcarriers.

        Returns:
            NDArray[np.float64]: Power allocation array where element i
                represents the power allocated to subcarrier i.
        """
        pass


class UniformPowerAllocation(IPowerAllocation):
    """Uniform power allocation across all subcarriers.

    This is the simplest power allocation strategy where all subcarriers
    receive equal power regardless of channel conditions. It serves as
    a baseline for comparison with more sophisticated strategies.

    Attributes:
        total_power: Total power budget to be distributed
        num_subcarriers: Number of OFDM subcarriers
    """

    def __init__(self, total_power: float, num_subcarriers: int):
        """Initialize uniform power allocator.

        Args:
            total_power: Total power budget (must be non-negative)
            num_subcarriers: Number of OFDM subcarriers (must be positive)

        Raises:
            ValueError: If total_power is negative or num_subcarriers is not positive
        """
        if total_power < 0:
            raise ValueError(f"Total power must be non-negative, got {total_power}")
        if num_subcarriers <= 0:
            raise ValueError(f"Number of subcarriers must be positive, got {num_subcarriers}")

        self.total_power = total_power
        self.num_subcarriers = num_subcarriers

    def allocate(self) -> NDArray[np.float64]:
        """Allocate equal power to all subcarriers.

        Returns:
            Power allocation array of length num_subcarriers where each
            element equals total_power / num_subcarriers.
        """
        power_per_subcarrier = self.total_power / self.num_subcarriers
        return np.full(self.num_subcarriers, power_per_subcarrier, dtype=np.float64)


class WaterfillingPowerAllocation(IPowerAllocation):
    """Waterfilling power allocation for capacity maximization.

    Implements the waterfilling algorithm which optimally allocates power
    across frequency-selective channels to maximize Shannon capacity.
    The algorithm allocates more power to subcarriers with better channel
    conditions (higher SNR) and may allocate zero power to very poor subcarriers.

    The name comes from the analogy of pouring water into a container with
    an uneven floor - the water level is constant, but deeper areas (better
    channels) contain more water (power).

    Mathematical formulation:
        P[k] = max(0, μ - N₀/|H[k]|²)

    Where:
        - P[k]: Power allocated to subcarrier k
        - μ: Water level (Lagrange multiplier)
        - N₀: Noise power
        - |H[k]|²: Channel power gain for subcarrier k

    Attributes:
        total_power: Total power budget
        channel_gains: Channel power gains |H[k]|² for each subcarrier
        noise_power: Noise power (N₀)
        tolerance: Convergence tolerance for binary search
        num_subcarriers: Number of subcarriers
    """

    def __init__(
        self,
        total_power: float,
        channel_gains: NDArray[np.float64],
        noise_power: float,
        tolerance: float = 1e-8,
    ):
        """Initialize waterfilling power allocator.

        Args:
            total_power: Total power budget (must be non-negative)
            channel_gains: Channel power gains |H[k]|² for each subcarrier
                (must be all positive)
            noise_power: Noise power N₀ (must be non-negative)
            tolerance: Convergence tolerance for binary search (default: 1e-8)

        Raises:
            ValueError: If inputs are invalid (negative values, empty array,
                zero/negative channel gains)
        """
        # Input validation
        if total_power < 0:
            raise ValueError(f"Total power must be non-negative, got {total_power}")
        if noise_power < 0:
            raise ValueError(f"Noise power must be non-negative, got {noise_power}")
        if len(channel_gains) == 0:
            raise ValueError("Channel gains array cannot be empty")
        if np.any(channel_gains <= 0):
            raise ValueError(
                "All channel gains must be positive, "
                f"got min={np.min(channel_gains)}, max={np.max(channel_gains)}"
            )

        self.total_power = total_power
        self.channel_gains = np.array(channel_gains, dtype=np.float64)
        self.noise_power = noise_power
        self.tolerance = tolerance
        self.num_subcarriers = len(channel_gains)

    def allocate(self) -> NDArray[np.float64]:
        """Allocate power using waterfilling algorithm.

        Uses binary search to find the optimal water level that satisfies
        the total power constraint while maximizing channel capacity.

        Returns:
            Power allocation array where element k represents the power
            allocated to subcarrier k. Some elements may be zero if the
            corresponding channel quality is too poor.

        Algorithm:
            1. Calculate inverse SNR (floor levels): N₀/|H[k]|²
            2. Binary search for water level μ
            3. For each μ, compute P[k] = max(0, μ - N₀/|H[k]|²)
            4. Adjust μ until sum(P) ≈ P_total
            5. Normalize for exact power constraint
        """
        # @book: pg ~603
        # Calculate inverse SNR - the "floor" of the container
        # Higher floor = worse channel = less power allocated
        floor = self.noise_power / (self.channel_gains * len(self.channel_gains))

        # Binary search for optimal water level
        water_level = self._find_water_level(floor)

        # Calculate power allocation using the water level
        # Power is the "depth of water" above the floor
        power = np.maximum(0, water_level - floor)

        # Normalize to satisfy exact power constraint
        # (handles numerical precision issues)
        power_sum = np.sum(power)
        if power_sum > 0:
            power = power * (self.total_power / power_sum)

        return power

    def _find_water_level(self, floor: NDArray[np.float64]) -> float:
        """Find optimal water level using binary search.

        The water level μ is chosen such that the total allocated power
        equals the power budget: sum(max(0, μ - floor[k])) = P_total

        Args:
            floor: Inverse SNR array (N₀/|H[k]|²) for each subcarrier

        Returns:
            Optimal water level μ that satisfies the power constraint

        Algorithm:
            - Lower bound: μ_min = 0 (no power allocated)
            - Upper bound: μ_max = P_total + max(floor) (all power to one subcarrier)
            - Binary search between bounds until convergence
        """
        # Initialize search bounds
        mu_min = 0.0
        mu_max = self.total_power + np.max(floor)

        # Binary search for water level
        max_iterations = 100  # Prevent infinite loops
        mu = (mu_min + mu_max) / 2  # Initialize mu

        for iteration in range(max_iterations):
            # Midpoint of current search interval
            mu = (mu_min + mu_max) / 2

            # Calculate power allocation with current water level
            power = np.maximum(0, mu - floor)
            power_sum = np.sum(power)

            # Check if we've converged to the solution
            if np.abs(power_sum - self.total_power) < self.tolerance:
                return mu

            # Adjust search bounds based on total power
            if power_sum < self.total_power:
                # Allocated power too low - raise water level
                mu_min = mu
            else:
                # Allocated power too high - lower water level
                mu_max = mu

        # Return best estimate if max iterations reached
        # (should rarely happen with reasonable tolerance)
        return mu


def calculate_capacity(
    power_allocation: NDArray[np.float64],
    channel_gains: NDArray[np.float64],
    noise_power: float,
) -> float:
    """Calculate Shannon capacity for given power allocation.

    Computes the achievable data rate using the Shannon-Hartley theorem:
        C = sum(log₂(1 + P[k]·|H[k]|²/N₀))

    Args:
        power_allocation: Power allocated to each subcarrier
        channel_gains: Channel power gains |H[k]|²
        noise_power: Noise power N₀

    Returns:
        Channel capacity in bits per channel use (bps/Hz)

    Example:
        >>> power = np.array([0.5, 0.3, 0.2])
        >>> gains = np.array([1.0, 0.8, 0.6])
        >>> noise = 0.1
        >>> capacity = calculate_capacity(power, gains, noise)
        >>> print(f"Capacity: {capacity:.2f} bits/channel use")
    """
    # Calculate SNR for each subcarrier
    snr = power_allocation * channel_gains / noise_power

    # Shannon capacity formula with small epsilon to avoid log(1)
    # log₂(1 + SNR) gives capacity in bits per channel use
    capacity = np.sum(np.log2(1 + snr + 1e-12))

    return capacity


def calculate_capacity_per_subcarrier(
    power_allocation: NDArray[np.float64],
    channel_gains: NDArray[np.float64],
    noise_power: float,
) -> NDArray[np.float64]:
    """Calculate Shannon capacity per subcarrier for given power allocation.

    Computes the achievable data rate for each subcarrier using the
    Shannon-Hartley theorem:
        C[k] = log₂(1 + P[k]·|H[k]|²/N₀)
    Args:
        power_allocation: Power allocated to each subcarrier
        channel_gains: Channel power gains |H[k]|²
        noise_power: Noise power N₀
    Returns:
        Capacity per subcarrier in bits per channel use (bps/Hz)
    Example:
        >>> power = np.array([0.5, 0.3, 0.2])
        >>> gains = np.array([1.0, 0.8, 0.6])
        >>> noise = 0.1
        >>> capacity_per_subcarrier = calculate_capacity_per_subcarrier(power, gains, noise)
        >>> print(f"Capacity per subcarrier: {capacity_per_subcarrier}")
    """
    # Calculate SNR for each subcarrier
    snr = power_allocation * channel_gains / noise_power

    # Shannon capacity formula with small epsilon to avoid log(1)
    # log₂(1 + SNR) gives capacity in bits per channel use
    capacity_per_subcarrier = np.log2(1 + snr + 1e-12)

    return capacity_per_subcarrier


def compare_allocations(
    uniform: NDArray[np.float64],
    waterfilling: NDArray[np.float64],
    channel_gains: NDArray[np.float64],
    noise_power: float,
) -> dict:
    """Compare performance of uniform vs waterfilling allocation.

    Calculates and compares the channel capacity achieved by uniform
    and waterfilling power allocation strategies.

    Args:
        uniform: Uniform power allocation array
        waterfilling: Waterfilling power allocation array
        channel_gains: Channel power gains |H[k]|²
        noise_power: Noise power N₀

    Returns:
        Dictionary containing:
            - uniform_capacity: Capacity with uniform allocation
            - waterfilling_capacity: Capacity with waterfilling
            - capacity_gain: Absolute capacity improvement
            - capacity_gain_percent: Percentage improvement

    Example:
        >>> results = compare_allocations(uniform, waterfilling, gains, noise)
        >>> print(f"Waterfilling improves capacity by {results['capacity_gain_percent']:.1f}%")
    """
    cap_uniform = calculate_capacity(uniform, channel_gains, noise_power)
    cap_waterfilling = calculate_capacity(waterfilling, channel_gains, noise_power)

    return {
        "uniform_capacity": cap_uniform,
        "waterfilling_capacity": cap_waterfilling,
        "capacity_gain": cap_waterfilling - cap_uniform,
        "capacity_gain_percent": (
            100 * (cap_waterfilling - cap_uniform) / cap_uniform if cap_uniform > 0 else 0
        ),
    }
