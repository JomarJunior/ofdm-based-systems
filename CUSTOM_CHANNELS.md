# Custom Channel Models Feature

## Overview

The OFDM simulation system now supports loading custom channel impulse responses from numpy files. This allows you to test your system with various channel conditions without modifying code.

## Quick Start

### 1. Generate Sample Channel Models

Run the channel generator script to create sample channel models:

```bash
python examples/generate_channel_models.py
```

This creates 6 channel models in `config/channel_models/`:
- `flat_fading.npy` - Single tap (frequency-flat)
- `two_ray.npy` - Simple 2-tap multipath
- `default_multipath.npy` - Default 4-tap channel (same as hardcoded)
- `severe_multipath.npy` - 8-tap severe multipath
- `rayleigh_fading.npy` - 6-tap Rayleigh fading
- `Lin-Phoong_P1.npy` - Reference channel from literature

### 2. Configure Your Simulation

Update your JSON configuration file:

```json
{
    "num_bands": 64,
    "signal_noise_ratios": [10, 15, 20, 25, 30],
    "channel_type": "CUSTOM",
    "channel_model_path": "config/channel_models/severe_multipath.npy",
    "num_symbols": 100000,
    "constellation_order": 16,
    "constellation_type": "QAM",
    "noise_type": "AWGN",
    "prefix_type": "CYCLIC",
    "modulation_type": "OFDM",
    "equalization_method": "MMSE",
    "power_allocation_type": "UNIFORM"
}
```

**Key fields:**
- `channel_type`: Set to `"CUSTOM"` to load from file, or `"FLAT"` to use hardcoded default
- `channel_model_path`: Relative or absolute path to `.npy` file

### 3. Run Your Simulation

The channel is automatically loaded when you create simulations from settings:

```python
from ofdm_based_systems.configuration.models import SimulationSettings
from ofdm_based_systems.simulation.models import Simulation

# Load configuration (includes channel loading)
settings = SimulationSettings.from_json("config/simulation_settings_custom_channel.json")

# Create simulations (channel automatically loaded)
simulations = Simulation.create_from_simulation_settings(settings)

# Run simulations
for sim in simulations:
    result = sim.run()
    print(f"SNR: {result['snr_db']} dB, BER: {result['bit_error_rate']:.6e}")
```

## Channel File Format

### Creating Your Own Channel Files

Channel impulse responses must be saved as numpy arrays with `complex128` dtype:

```python
import numpy as np

# Define your channel impulse response
channel = np.array([
    (0.9 + 0.1j),
    (0.5 - 0.2j),
    (0.3 + 0.15j),
], dtype=np.complex128)

# Normalize to unit energy (optional but recommended)
channel = channel / np.sqrt(np.sum(np.abs(channel) ** 2))

# Save to file
np.save('config/channel_models/my_channel.npy', channel)
```

### Loading and Inspecting Channels

```python
import numpy as np

# Load channel
channel = np.load('config/channel_models/my_channel.npy')

print(f"Number of taps: {len(channel)}")
print(f"Channel energy: {np.sum(np.abs(channel)**2):.6f}")
print(f"Channel dtype: {channel.dtype}")
```

## Programmatic Usage

### Option 1: Load from Configuration File

```python
from ofdm_based_systems.configuration.models import SimulationSettings
from ofdm_based_systems.simulation.models import Simulation

settings = SimulationSettings.from_json("config/simulation_settings.json")
simulations = Simulation.create_from_simulation_settings(settings)

# Channel is automatically loaded based on settings
for sim in simulations:
    result = sim.run()
```

### Option 2: Pass Channel Directly

```python
import numpy as np
from ofdm_based_systems.simulation.models import Simulation
from ofdm_based_systems.configuration.enums import *

# Load your channel
channel = np.load('config/channel_models/my_channel.npy')

# Create simulation with custom channel
sim = Simulation(
    num_symbols=10240,
    num_subcarriers=64,
    constellation_order=16,
    constellation_scheme=ConstellationType.QAM,
    modulator_type=ModulationType.OFDM,
    prefix_scheme=PrefixType.CYCLIC,
    prefix_length_ratio=1.0,
    equalizator_type=EqualizationMethod.MMSE,
    snr_db=20.0,
    noise_scheme=NoiseType.AWGN,
    power_allocation_type=PowerAllocationType.UNIFORM,
    channel_impulse_response=channel,  # Pass channel directly
    verbose=True
)

result = sim.run()
```

## Channel Types

### Flat Fading (1 tap)
- **File**: `flat_fading.npy`
- **Description**: No frequency selectivity, simple AWGN channel
- **Use case**: Baseline testing, minimal ISI

### Two-Ray (2 taps)
- **File**: `two_ray.npy`
- **Description**: Direct path + one reflection
- **Use case**: Simple multipath testing

### Default Multipath (4 taps)
- **File**: `default_multipath.npy`
- **Description**: Same as hardcoded default channel
- **Use case**: Backward compatibility testing

### Severe Multipath (8 taps)
- **File**: `severe_multipath.npy`
- **Description**: Strong frequency selectivity with significant ISI
- **Use case**: Stress testing, worst-case scenarios

### Rayleigh Fading (6 taps)
- **File**: `rayleigh_fading.npy`
- **Description**: Exponential power delay profile
- **Use case**: Statistical channel modeling

### Lin-Phoong P1 (4 taps)
- **File**: `Lin-Phoong_P1.npy`
- **Description**: Reference channel from literature
- **Use case**: Validation against published results

## Examples

### Example 1: Compare Channels

See `examples/custom_channel_demo.py` for a complete example:

```python
python examples/custom_channel_demo.py
```

This script:
1. Lists all available channel models
2. Runs simulations with different channels
3. Compares BER performance
4. Demonstrates config file loading

### Example 2: Batch Testing

Test your algorithm across multiple channel conditions:

```python
import numpy as np
from pathlib import Path
from ofdm_based_systems.simulation.models import Simulation

channel_dir = Path("config/channel_models")

results = {}
for channel_file in channel_dir.glob("*.npy"):
    channel = np.load(channel_file)
    
    sim = Simulation(
        num_symbols=10240,
        num_subcarriers=64,
        constellation_order=16,
        # ... other params ...
        channel_impulse_response=channel,
        verbose=False
    )
    
    result = sim.run()
    results[channel_file.name] = result['bit_error_rate']

# Compare results
for name, ber in sorted(results.items(), key=lambda x: x[1]):
    print(f"{name:30s} BER: {ber:.6e}")
```

## Configuration Options

### Using Default Channel

To use the hardcoded default channel (backward compatible):

```json
{
    "channel_type": "FLAT",
    "channel_model_path": "",
    ...
}
```

Or simply omit the `channel_impulse_response` parameter when creating `Simulation` objects programmatically.

### Using Custom Channel

```json
{
    "channel_type": "CUSTOM",
    "channel_model_path": "config/channel_models/my_channel.npy",
    ...
}
```

**Path resolution:**
- Relative paths are resolved from the current working directory
- Absolute paths are used as-is
- File must exist and be readable
- Must be a valid `.npy` file with `complex128` data

## Validation and Error Handling

The system performs comprehensive validation:

1. **File Existence Check**
   ```
   FileNotFoundError: Channel model file not found: <path>
   ```

2. **Loading Validation**
   ```
   ValueError: Failed to load channel model from <path>: <error>
   ```

3. **Automatic Type Conversion**
   - Channel data is automatically converted to `np.complex128` if needed

4. **Informative Logging**
   ```
   ✓ Loaded custom channel impulse response from: <path>
     Channel length: 8 taps
     Channel dtype: complex128
   ```

## Performance Considerations

### Channel Length and Prefix

The cyclic prefix length is automatically adjusted based on channel length:

```python
prefix_length = int(prefix_length_ratio * channel.order)
```

For longer channels:
- Increase `prefix_length_ratio` (default: 1.0)
- Expect higher overhead but better ISI mitigation

### Memory Usage

Channel files are small (typically < 1KB):
- 4 taps: ~128 bytes
- 8 taps: ~256 bytes
- 16 taps: ~512 bytes

Multiple simulations share the same loaded channel instance (efficient).

## Advanced Topics

### Creating Statistical Channels

Generate channels with specific statistical properties:

```python
import numpy as np

def generate_rayleigh_channel(num_taps, seed=None):
    """Generate Rayleigh fading channel with exponential PDP."""
    if seed is not None:
        np.random.seed(seed)
    
    # Complex Gaussian samples
    channel = (np.random.randn(num_taps) + 1j * np.random.randn(num_taps)) / np.sqrt(2)
    
    # Exponential power delay profile
    pdp = np.exp(-np.arange(num_taps) / 2.0)
    channel = channel * np.sqrt(pdp)
    
    # Normalize
    channel = channel / np.sqrt(np.sum(np.abs(channel) ** 2))
    
    return channel.astype(np.complex128)

# Generate and save
channel = generate_rayleigh_channel(num_taps=10, seed=42)
np.save('config/channel_models/custom_rayleigh.npy', channel)
```

### Frequency Response Analysis

Analyze your channel in the frequency domain:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load channel
channel = np.load('config/channel_models/severe_multipath.npy')

# Compute frequency response
num_subcarriers = 64
freq_response = np.fft.fft(channel, n=num_subcarriers)
channel_gains = np.abs(freq_response) ** 2

# Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.stem(np.abs(channel))
plt.title('Time Domain (Impulse Response)')
plt.xlabel('Tap Index')
plt.ylabel('Magnitude')

plt.subplot(1, 2, 2)
plt.plot(10 * np.log10(channel_gains))
plt.title('Frequency Domain (Channel Gain)')
plt.xlabel('Subcarrier Index')
plt.ylabel('Gain (dB)')
plt.grid(True)
plt.tight_layout()
plt.show()
```

## Backward Compatibility

The feature is **100% backward compatible**:

- Existing code without `channel_impulse_response` parameter works unchanged
- Default hardcoded channel is used when no custom channel is specified
- All 38 integration tests pass without modification

## Summary

### Benefits
✅ Load channels from files without code changes  
✅ Easy testing with multiple channel conditions  
✅ Reproducible results with saved channels  
✅ Share channel models between projects  
✅ Validate against published channel models  
✅ Backward compatible with existing code  

### Files Created
- `config/channel_models/*.npy` - 6 sample channel models
- `examples/generate_channel_models.py` - Channel generator
- `examples/custom_channel_demo.py` - Usage examples
- `config/simulation_settings_custom_channel.json` - Example config
- `CUSTOM_CHANNELS.md` - This documentation

### Quick Reference

**Generate channels:**
```bash
python examples/generate_channel_models.py
```

**Run demo:**
```bash
python examples/custom_channel_demo.py
```

**Use in config:**
```json
{
    "channel_type": "CUSTOM",
    "channel_model_path": "config/channel_models/severe_multipath.npy"
}
```

**Use programmatically:**
```python
channel = np.load('path/to/channel.npy')
sim = Simulation(..., channel_impulse_response=channel)
```

---

**Last Updated:** November 10, 2025  
**Feature Version:** 2.1.0  
**Status:** Production Ready ✅
