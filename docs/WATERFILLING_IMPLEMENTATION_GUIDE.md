# Waterfilling Power Allocation - Implementation Guide

## Table of Contents

1. [Theoretical Background](#theoretical-background)
2. [Algorithm Overview](#algorithm-overview)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [Integration with OFDM System](#integration-with-ofdm-system)
5. [Testing Strategy](#testing-strategy)
6. [Implementation Code](#implementation-code)

## Theoretical Background

### What is Waterfilling?

**Waterfilling** (also called water-filling or water-pouring) is an optimal power allocation algorithm that maximizes the channel capacity of a frequency-selective channel under a total power constraint.

### The Water Analogy

Imagine pouring water into a container with an uneven floor:

- The **floor** represents the inverse channel quality (noise/gain ratio)
- The **water** represents the allocated power
- The **water level** is constant across all subcarriers
- Deep areas (good channels) receive more water (power)
- Shallow areas (poor channels) may receive no water (zero power)

### Mathematical Foundation

For an OFDM system with N subcarriers:

**Channel Model**:

- Channel frequency response: `H[k]` for subcarrier k
- Channel power gain: `|H[k]|²`
- Noise power: `N₀`

**Capacity Maximization**:

```
Maximize: C = Σ log₂(1 + P[k]·|H[k]|²/N₀)
Subject to: Σ P[k] = P_total
            P[k] ≥ 0
```

**Solution (Waterfilling)**:

```
P[k] = max(0, μ - N₀/|H[k]|²)
```

Where:

- `P[k]` = power allocated to subcarrier k
- `μ` = water level (Lagrange multiplier)
- `N₀` = noise power
- `|H[k]|²` = channel power gain

**Key Properties**:

1. **Optimality**: Maximizes Shannon capacity
2. **Channel-aware**: Allocates more power to better subcarriers
3. **Threshold**: Subcarriers with `|H[k]|²/N₀ < 1/μ` get zero power
4. **Power constraint**: `Σ P[k] = P_total`

### Signal Transmission Context

In practical OFDM transmission:

1. **Channel Estimation Phase**:
   - Send pilot symbols
   - Estimate channel frequency response `H[k]`
   - Calculate channel gains `|H[k]|²`

2. **Power Allocation Phase**:
   - Apply waterfilling algorithm
   - Compute power allocation `P[k]` for each subcarrier
   - Calculate scaling factors: `A[k] = √P[k]`

3. **Transmission Phase**:
   - Generate data symbols: `S[k]`
   - Scale by power: `S'[k] = A[k] · S[k]`
   - IFFT and transmit

4. **Reception Phase**:
   - Receive and FFT
   - Equalize with knowledge of power allocation
   - Decode symbols

### Benefits in OFDM Systems

1. **Increased Capacity**: Achieves Shannon capacity limit
2. **Better BER**: Concentrates power where it's most effective
3. **Robustness**: Avoids wasting power on deep fades
4. **Flexibility**: Adapts to channel conditions

### Trade-offs

**Advantages**:

- Optimal capacity
- Adapts to channel conditions
- Better performance than uniform allocation

**Disadvantages**:

- Requires accurate channel state information (CSI)
- Higher computational complexity
- Needs feedback channel for adaptive systems
- Some subcarriers may be unused

## Algorithm Overview

### Waterfilling Algorithm Steps

```python
Input:
  - P_total: Total power budget
  - H: Channel frequency response (complex array)
  - N0: Noise power
  
Output:
  - P: Power allocation array (length N)

1. Calculate channel gains:
   G[k] = |H[k]|²

2. Calculate inverse SNR (floor levels):
   floor[k] = N0 / G[k]

3. Initialize water level search:
   μ_min = 0
   μ_max = P_total + max(floor)

4. Binary search for water level μ:
   while not converged:
       μ = (μ_min + μ_max) / 2
       
       # Calculate power allocation
       for k in range(N):
           P[k] = max(0, μ - floor[k])
       
       # Check power constraint
       P_sum = sum(P)
       
       if P_sum < P_total:
           μ_min = μ  # Water level too low
       else:
           μ_max = μ  # Water level too high
   
5. Normalize to exact power:
   P = P * (P_total / sum(P))

6. Return P
```

### Convergence Criteria

The binary search converges when:

```python
|sum(P) - P_total| < tolerance
```

Typical tolerance: `1e-8 * P_total`

### Computational Complexity

- **Time**: O(N log(1/ε)) where N = subcarriers, ε = tolerance
- **Space**: O(N)
- Very efficient for typical OFDM systems (64-2048 subcarriers)

## Step-by-Step Implementation

### Step 1: Create Abstract Base Class

**File**: `src/ofdm_based_systems/power_allocation/models.py`

```python
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

class IPowerAllocation(ABC):
    """Abstract base class for power allocation strategies."""
    
    @abstractmethod
    def allocate(self) -> NDArray[np.float64]:
        """Allocate power across subcarriers.
        
        Returns:
            NDArray[np.float64]: Power allocation array
        """
        pass
```

**Purpose**: Define interface for all power allocation strategies

**Testing**: Verify it's abstract and cannot be instantiated

### Step 2: Implement Uniform Power Allocation (Baseline)

```python
class UniformPowerAllocation(IPowerAllocation):
    """Uniform power allocation across all subcarriers."""
    
    def __init__(self, total_power: float, num_subcarriers: int):
        """Initialize uniform power allocator.
        
        Args:
            total_power: Total power budget
            num_subcarriers: Number of OFDM subcarriers
            
        Raises:
            ValueError: If total_power < 0 or num_subcarriers <= 0
        """
        if total_power < 0:
            raise ValueError("Total power must be non-negative")
        if num_subcarriers <= 0:
            raise ValueError("Number of subcarriers must be positive")
            
        self.total_power = total_power
        self.num_subcarriers = num_subcarriers
    
    def allocate(self) -> NDArray[np.float64]:
        """Allocate equal power to all subcarriers.
        
        Returns:
            Power allocation array of length num_subcarriers
        """
        power_per_subcarrier = self.total_power / self.num_subcarriers
        return np.full(self.num_subcarriers, power_per_subcarrier, dtype=np.float64)
```

**Purpose**: Simple baseline for comparison with waterfilling

**Testing**:

- Equal power to all subcarriers
- Power sums to total
- Edge cases (1 subcarrier, 1000 subcarriers)

### Step 3: Implement Waterfilling Algorithm

```python
class WaterfillingPowerAllocation(IPowerAllocation):
    """Waterfilling power allocation for capacity maximization."""
    
    def __init__(
        self,
        total_power: float,
        channel_gains: NDArray[np.float64],
        noise_power: float,
        tolerance: float = 1e-8
    ):
        """Initialize waterfilling power allocator.
        
        Args:
            total_power: Total power budget
            channel_gains: Channel power gains |H[k]|² for each subcarrier
            noise_power: Noise power (N₀)
            tolerance: Convergence tolerance for binary search
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validation
        if total_power < 0:
            raise ValueError("Total power must be non-negative")
        if noise_power < 0:
            raise ValueError("Noise power must be non-negative")
        if len(channel_gains) == 0:
            raise ValueError("Channel gains array cannot be empty")
        if np.any(channel_gains <= 0):
            raise ValueError("All channel gains must be positive")
            
        self.total_power = total_power
        self.channel_gains = np.array(channel_gains, dtype=np.float64)
        self.noise_power = noise_power
        self.tolerance = tolerance
        self.num_subcarriers = len(channel_gains)
    
    def allocate(self) -> NDArray[np.float64]:
        """Allocate power using waterfilling algorithm.
        
        Returns:
            Power allocation array maximizing channel capacity
        """
        # Calculate inverse SNR (floor levels)
        floor = self.noise_power / self.channel_gains
        
        # Binary search for water level
        water_level = self._find_water_level(floor)
        
        # Calculate power allocation
        power = np.maximum(0, water_level - floor)
        
        # Normalize to exact total power (handle numerical errors)
        power_sum = np.sum(power)
        if power_sum > 0:
            power = power * (self.total_power / power_sum)
        
        return power
    
    def _find_water_level(self, floor: NDArray[np.float64]) -> float:
        """Find water level using binary search.
        
        Args:
            floor: Inverse SNR array (N₀/|H[k]|²)
            
        Returns:
            Optimal water level μ
        """
        # Initialize search bounds
        mu_min = 0.0
        mu_max = self.total_power + np.max(floor)
        
        # Binary search
        max_iterations = 100
        for _ in range(max_iterations):
            mu = (mu_min + mu_max) / 2
            
            # Calculate power with current water level
            power = np.maximum(0, mu - floor)
            power_sum = np.sum(power)
            
            # Check convergence
            if np.abs(power_sum - self.total_power) < self.tolerance:
                return mu
            
            # Adjust search bounds
            if power_sum < self.total_power:
                mu_min = mu  # Water level too low
            else:
                mu_max = mu  # Water level too high
        
        # Return best estimate if max iterations reached
        return mu
```

**Purpose**: Optimal power allocation for capacity maximization

**Key Features**:

- Binary search for water level
- Handles numerical precision
- Efficient O(N log(1/ε)) complexity

### Step 4: Add Utility Functions

```python
def calculate_capacity(
    power_allocation: NDArray[np.float64],
    channel_gains: NDArray[np.float64],
    noise_power: float
) -> float:
    """Calculate Shannon capacity for given power allocation.
    
    Args:
        power_allocation: Power per subcarrier
        channel_gains: Channel power gains |H[k]|²
        noise_power: Noise power
        
    Returns:
        Channel capacity in bits per channel use
    """
    snr = power_allocation * channel_gains / noise_power
    # Avoid log(1 + 0) issues
    capacity = np.sum(np.log2(1 + snr + 1e-12))
    return capacity


def compare_allocations(
    uniform: NDArray[np.float64],
    waterfilling: NDArray[np.float64],
    channel_gains: NDArray[np.float64],
    noise_power: float
) -> dict:
    """Compare uniform vs waterfilling performance.
    
    Returns:
        Dictionary with capacity and gain metrics
    """
    cap_uniform = calculate_capacity(uniform, channel_gains, noise_power)
    cap_waterfilling = calculate_capacity(waterfilling, channel_gains, noise_power)
    
    return {
        "uniform_capacity": cap_uniform,
        "waterfilling_capacity": cap_waterfilling,
        "capacity_gain": cap_waterfilling - cap_uniform,
        "capacity_gain_percent": 100 * (cap_waterfilling - cap_uniform) / cap_uniform
    }
```

## Integration with OFDM System

### Integration Points

The power allocation module integrates with existing OFDM system at several points:

#### 1. **Simulation Setup** (Before Transmission)

```python
# File: src/ofdm_based_systems/simulation/models.py

class Simulation:
    def __init__(self, ..., power_allocation_scheme: str = "uniform"):
        self.power_allocation_scheme = power_allocation_scheme
        # ... existing code
    
    def run(self):
        # ... existing code
        
        # Get channel response
        channel_response = np.fft.fft(self.channel_impulse, n=self.num_subcarriers)
        channel_gains = np.abs(channel_response) ** 2
        
        # Allocate power based on scheme
        if self.power_allocation_scheme == "waterfilling":
            allocator = WaterfillingPowerAllocation(
                total_power=1.0,  # Normalized power
                channel_gains=channel_gains,
                noise_power=10 ** (-self.snr_db / 10)  # Convert SNR to noise power
            )
        else:  # uniform
            allocator = UniformPowerAllocation(
                total_power=1.0,
                num_subcarriers=self.num_subcarriers
            )
        
        power_allocation = allocator.allocate()
        
        # Scale symbols by power
        power_scaling = np.sqrt(power_allocation)  # Amplitude scaling
        symbols_scaled = symbols * power_scaling
        
        # Continue with modulation...
```

#### 2. **Symbol Scaling** (Transmitter)

```python
# After constellation mapping and parallel conversion
parallel_symbols = converter.to_parallel(symbols, num_subcarriers)

# Scale each subcarrier by its allocated power
for block_idx in range(parallel_symbols.shape[0]):
    parallel_symbols[block_idx, :] *= np.sqrt(power_allocation)

# Continue with IFFT modulation
modulated = modulator.modulate(parallel_symbols)
```

#### 3. **Receiver Processing** (Optional: Power-Aware Equalization)

```python
# After demodulation
demodulated = modulator.demodulate(received_parallel)

# If using power-aware equalization, incorporate power allocation
# into the equalization weights (for MMSE equalizer)
```

### Modified Simulation Flow

```
1. Initialize Simulation
2. Estimate Channel → Get channel_gains |H[k]|²
3. Calculate Noise Power → from SNR
4. Apply Power Allocation:
   - Waterfilling → WaterfillingPowerAllocation.allocate()
   - Uniform → UniformPowerAllocation.allocate()
5. Generate Bits
6. Constellation Mapping
7. Serial → Parallel
8. Scale Symbols by √P[k]  ← NEW STEP
9. OFDM Modulation (IFFT)
10. Add Prefix
11. Parallel → Serial
12. Channel Transmission
13. Serial → Parallel
14. Remove Prefix
15. OFDM Demodulation (FFT)
16. Equalization (consider power in weights)
17. Parallel → Serial
18. Constellation Demapping
19. Calculate BER
20. Return Results (include capacity)
```

## Testing Strategy

### Test Categories

#### 1. **Unit Tests** (Basic Functionality)

**Test File**: `tests/ofdm_based_systems/power_allocation/test_models.py`

```python
class TestWaterfillingBasics:
    """Basic waterfilling functionality tests."""
    
    def test_initialization_valid_inputs(self):
        """Test initialization with valid parameters."""
        
    def test_initialization_invalid_inputs(self):
        """Test validation of inputs."""
        # Negative power
        # Negative noise
        # Empty channel gains
        # Zero channel gains
    
    def test_allocate_returns_correct_shape(self):
        """Test output array has correct shape."""
    
    def test_power_conservation(self):
        """Test allocated power sums to total power."""
    
    def test_non_negative_allocation(self):
        """Test all power values are non-negative."""
```

#### 2. **Algorithm Tests** (Waterfilling Properties)

```python
class TestWaterfillingProperties:
    """Test waterfilling algorithm properties."""
    
    def test_favors_good_channels(self):
        """Better channels should receive more power."""
        # channel_gains = [1.0, 0.5, 0.25]
        # Assert: P[0] >= P[1] >= P[2]
    
    def test_zero_power_to_bad_channels(self):
        """Very poor channels should get zero power."""
        # Mix good and very bad channels
        # Assert some get zero power
    
    def test_water_level_property(self):
        """Test water level is constant for allocated channels."""
        # For P[k] > 0: P[k] + N₀/|H[k]|² ≈ constant
    
    def test_equal_channels_uniform_allocation(self):
        """Equal channel gains should give uniform allocation."""
    
    def test_single_good_channel(self):
        """All power should go to single good channel."""
        # channel_gains = [1.0, 0.001, 0.001, ...]
        # Assert: P[0] ≈ P_total
```

#### 3. **Convergence Tests**

```python
class TestWaterfillingConvergence:
    """Test algorithm convergence."""
    
    def test_convergence_uniform_channels(self):
        """Should converge quickly for uniform channels."""
    
    def test_convergence_diverse_channels(self):
        """Should converge for diverse channel gains."""
    
    def test_tolerance_accuracy(self):
        """Test different tolerance levels."""
    
    def test_many_subcarriers(self):
        """Test convergence with 1024+ subcarriers."""
```

#### 4. **Edge Cases**

```python
class TestWaterfillingEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_subcarrier(self):
        """Single subcarrier gets all power."""
    
    def test_all_channels_identical(self):
        """Identical channels → uniform allocation."""
    
    def test_high_snr(self):
        """High SNR behavior (all channels allocated)."""
    
    def test_low_snr(self):
        """Low SNR behavior (only best channels allocated)."""
    
    def test_zero_total_power(self):
        """Zero power budget."""
    
    def test_very_high_noise(self):
        """Very high noise power."""
```

#### 5. **Integration Tests** (Already Created)

**Test File**: `tests/integration/test_power_allocation.py`

```python
class TestPowerAllocationIntegration:
    """Integration with OFDM system."""
    
    def test_with_real_ofdm_channel(self):
        """Test with actual OFDM channel model."""
    
    def test_waterfilling_improves_ber(self):
        """Waterfilling should improve BER over uniform."""
    
    def test_capacity_improvement(self):
        """Waterfilling should increase capacity."""
    
    def test_with_channel_estimation_errors(self):
        """Test robustness to CSI errors."""
```

#### 6. **Performance Tests**

```python
class TestWaterfillingPerformance:
    """Performance and optimization tests."""
    
    def test_execution_time(self):
        """Should be fast enough for real-time."""
        # < 1ms for 64 subcarriers
        # < 10ms for 2048 subcarriers
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
    
    def test_memory_usage(self):
        """Should not use excessive memory."""
```

### Test Cases Summary

**Critical Test Cases**:

1. ✅ **Power Conservation**: `sum(P) == P_total`
2. ✅ **Non-negativity**: `all(P >= 0)`
3. ✅ **Channel Priority**: Better channels get more power
4. ✅ **Water Level**: Constant for allocated subcarriers
5. ✅ **Equal Channels**: Behaves like uniform allocation
6. ✅ **Capacity**: Achieves higher capacity than uniform
7. ✅ **BER Improvement**: Better BER than uniform in frequency-selective channels
8. ✅ **Convergence**: Reliable convergence in reasonable iterations
9. ✅ **Edge Cases**: Single subcarrier, zero power, extreme SNR
10. ✅ **Integration**: Works with existing OFDM pipeline

### Example Test Implementation

```python
def test_waterfilling_water_level_property():
    """Test the water level property."""
    total_power = 5.0
    channel_gains = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    noise_power = 0.1
    
    allocator = WaterfillingPowerAllocation(
        total_power=total_power,
        channel_gains=channel_gains,
        noise_power=noise_power
    )
    power_allocation = allocator.allocate()
    
    # Calculate water level for each subcarrier
    water_levels = power_allocation + noise_power / channel_gains
    
    # Find allocated subcarriers (power > threshold)
    allocated = power_allocation > 1e-10
    
    if np.any(allocated):
        # Water level should be constant for allocated subcarriers
        water_level_allocated = water_levels[allocated]
        std_dev = np.std(water_level_allocated)
        
        assert std_dev < 0.01, f"Water level not constant: std = {std_dev}"
```

## Implementation Code

### Complete Implementation

Here's the complete implementation ready to paste into your codebase:

```python
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
            raise ValueError(
                f"Number of subcarriers must be positive, got {num_subcarriers}"
            )

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
        # Calculate inverse SNR - the "floor" of the container
        # Higher floor = worse channel = less power allocated
        floor = self.noise_power / self.channel_gains

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
```

### Usage Example

```python
# Example: Using waterfilling in OFDM system

# 1. Setup channel
channel_impulse = np.array([1.0, 0.7, 0.4, 0.2], dtype=np.complex128)
num_subcarriers = 64
channel_response = np.fft.fft(channel_impulse, n=num_subcarriers)
channel_gains = np.abs(channel_response) ** 2

# 2. Setup parameters
total_power = 1.0  # Normalized
snr_db = 20.0
noise_power = 10 ** (-snr_db / 10)

# 3. Allocate power
allocator = WaterfillingPowerAllocation(
    total_power=total_power,
    channel_gains=channel_gains,
    noise_power=noise_power
)
power_allocation = allocator.allocate()

# 4. Scale symbols
symbols = generate_symbols()  # Your constellation mapper
power_scaling = np.sqrt(power_allocation)
symbols_scaled = symbols * power_scaling

# 5. Transmit (rest of OFDM pipeline)
```

## Conclusion

This implementation provides:

- ✅ Optimal power allocation for OFDM
- ✅ Efficient O(N log(1/ε)) algorithm
- ✅ Comprehensive validation
- ✅ Clear documentation
- ✅ Ready for integration
- ✅ Full test coverage planned

The waterfilling algorithm will significantly improve system performance in frequency-selective channels by intelligently distributing power based on channel quality.
