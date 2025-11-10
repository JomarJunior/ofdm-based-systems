# Enhanced Image Organization and Naming System

## Overview

The OFDM simulation system now features an enhanced image organization and naming system that:
1. **Organizes images by channel** - Images are saved in channel-specific subdirectories
2. **Uses structured filenames** - Filenames follow a consistent pattern that includes all configuration parameters
3. **Supports easy comparison** - Multiple configurations can coexist in the same channel directory

## Changes Summary

### 1. Channel-Specific Directories

Images are now saved in subdirectories based on the channel name:
```
images/
├── Lin-Phoong_P1/          # For Lin-Phoong P1 channel
│   ├── CP-OFDM-MMSE-64QAM-UNIFORM-SNR20_0dB.png
│   └── CP-OFDM-MMSE-64QAM-UNIFORM-BER_vs_SNR.png
├── severe_multipath/       # For severe multipath channel
│   ├── CP-OFDM-ZF-16QAM-WF-SNR20_0dB.png
│   └── CP-OFDM-ZF-16QAM-WF-SNR30_0dB.png
└── default/                # For default channel (backward compatible)
    └── ...
```

### 2. Structured Filename Pattern

Filenames now follow the pattern:
```
{PREFIX}-{MODULATION}-{EQUALIZATION}-{ORDER}{CONSTELLATION}-{POWER_ALLOCATION}-{METRIC}.png
```

**Components:**
- **PREFIX**: Prefix scheme acronym
  - `CP` - Cyclic Prefix
  - `ZP` - Zero Padding
  - `NONE` - No Prefix
  
- **MODULATION**: Modulation type
  - `OFDM` - OFDM
  - `SC-OFDM` - Single Carrier OFDM
  
- **EQUALIZATION**: Equalization method
  - `ZF` - Zero Forcing
  - `MMSE` - Minimum Mean Square Error
  - `NONE` - No Equalization
  
- **ORDER**: Constellation order (e.g., 4, 16, 64)

- **CONSTELLATION**: Constellation type
  - `QAM` - Quadrature Amplitude Modulation
  - `PSK` - Phase Shift Keying
  
- **POWER_ALLOCATION**: Power allocation strategy
  - `WF` - Waterfilling
  - `UNIFORM` - Uniform Power Allocation
  
- **METRIC**: Specific metric or SNR value
  - `SNRxx_xdB` - Constellation plot at specific SNR (decimal point replaced with underscore)
  - `BER_vs_SNR` - BER vs SNR curve

**Examples:**
```
CP-OFDM-ZF-64QAM-WF-SNR30_0dB.png       # Constellation at SNR=30.0 dB
CP-OFDM-ZF-64QAM-WF-BER_vs_SNR.png      # BER curve
ZP-SC-OFDM-MMSE-16QAM-UNIFORM-SNR25_5dB.png  # SNR=25.5 dB with decimal
```

## Modified Files

### `src/ofdm_based_systems/main.py`

#### `ResultsManager` class changes:

**1. Constructor now accepts channel_name:**
```python
def __init__(
    self,
    results_dir: str = "results",
    images_dir: str = "images",
    channel_name: str = "default",
):
    """Initialize results manager.
    
    Args:
        results_dir: Directory for storing CSV results
        images_dir: Base directory for storing plot images
        channel_name: Name of the channel (used to create subdirectory)
    """
    self.results_dir = Path(results_dir)
    self.channel_name = channel_name
    # Create channel-specific subdirectory for images
    self.images_dir = Path(images_dir) / channel_name
    # ... rest of initialization
```

**2. `save_constellation_plot()` method signature updated:**
```python
def save_constellation_plot(
    self,
    image: Image.Image,
    prefix_type: str,
    modulation_type: str,
    equalization_method: str,
    constellation_order: int,
    constellation_type: str,
    power_allocation: str,
    snr_db: float,
) -> Path:
    """Save constellation plot with structured filename.
    
    Example filename: "CP-OFDM-ZF-64QAM-WF-SNR30_0dB.png"
    """
    # Format SNR with underscore for decimal point (e.g., 30.5 -> "30_5")
    snr_str = f"{snr_db:.1f}".replace(".", "_")
    
    # Build filename
    filename = (
        f"{prefix_type}-{modulation_type}-{equalization_method}-"
        f"{constellation_order}{constellation_type}-{power_allocation}-"
        f"SNR{snr_str}dB.png"
    )
    # ... save logic
```

**3. `plot_ber_vs_snr()` method updated:**
```python
def plot_ber_vs_snr(self, results: List[Dict[str, Any]]) -> Path:
    """Generate BER vs SNR plot with structured filename.
    
    Filename pattern: "CP-OFDM-ZF-64QAM-WF-BER_vs_SNR.png"
    """
    # Extract configuration from first result
    result = results[0]
    prefix_type = result.get("prefix_acronym", "NONE")
    # ... extract other parameters
    
    output_filename = (
        f"{prefix_type}-{modulation_type}-{equalization_method}-"
        f"{constellation_order}{constellation_type}-{power_allocation}-"
        f"BER_vs_SNR.png"
    )
    # ... plotting logic
```

**4. `main()` function extracts channel name:**
```python
def main():
    """Main entry point for OFDM simulation system."""
    # Load configuration
    settings = Settings.from_json(file_path="config/settings.json")
    simulation_settings = SimulationSettings.from_json(
        file_path="config/simulation_settings.json"
    )
    
    # Extract channel name from configuration
    channel_name = "default"
    if simulation_settings.channel_type.value == "CUSTOM" and simulation_settings.channel_model_path:
        # Extract from path: "config/channel_models/severe_multipath.npy" -> "severe_multipath"
        channel_name = Path(simulation_settings.channel_model_path).stem
    elif simulation_settings.channel_type.value == "FLAT":
        channel_name = "flat"
    
    # Initialize with channel-specific directory
    results_manager = ResultsManager(
        results_dir="results",
        images_dir="images",
        channel_name=channel_name,
    )
    # ... rest of main
```

### `src/ofdm_based_systems/simulation/models.py`

**Enhanced results dictionary to include naming components:**
```python
results.update(
    {
        # ... existing fields
        "prefix_acronym": prefix_scheme.acronym,  # NEW: for filename
        "power_allocation_acronym": "WF" if self.power_allocation_type == PowerAllocationType.WATERFILLING else "UNIFORM",  # NEW: for filename
        # ... rest of fields
    }
)
```

## Usage Examples

### Example 1: Single Configuration
```python
from ofdm_based_systems.configuration.models import Settings, SimulationSettings
from ofdm_based_systems.main import ResultsManager, SimulationRunner

settings = Settings.from_json("config/settings.json")
simulation_settings = SimulationSettings.from_json("config/simulation_settings.json")

# Channel name automatically extracted from configuration
runner = SimulationRunner(settings, simulation_settings, results_manager)
results = runner.run_all()
runner.process_results(results)
```

**Output structure:**
```
images/severe_multipath/
├── CP-OFDM-ZF-16QAM-WF-SNR20_0dB.png
├── CP-OFDM-ZF-16QAM-WF-SNR30_0dB.png
└── CP-OFDM-ZF-16QAM-WF-BER_vs_SNR.png
```

### Example 2: Comparing Different Equalizers

Run with ZF equalization:
```bash
# Create config with equalization_method: "ZF"
python -m ofdm_based_systems.main
```

Run with MMSE equalization:
```bash
# Create config with equalization_method: "MMSE"
python -m ofdm_based_systems.main
```

**Both sets of images coexist in the same channel directory:**
```
images/severe_multipath/
├── CP-OFDM-ZF-16QAM-WF-SNR20_0dB.png       # Zero Forcing
├── CP-OFDM-ZF-16QAM-WF-SNR30_0dB.png
├── CP-OFDM-MMSE-16QAM-WF-SNR20_0dB.png     # MMSE
└── CP-OFDM-MMSE-16QAM-WF-SNR30_0dB.png
```

### Example 3: Different Channels

```bash
# Run with severe_multipath channel
python -m ofdm_based_systems.main

# Change channel_model_path to "config/channel_models/flat_fading.npy"
python -m ofdm_based_systems.main
```

**Results organized by channel:**
```
images/
├── severe_multipath/
│   └── CP-OFDM-ZF-16QAM-WF-SNR30_0dB.png
└── flat_fading/
    └── CP-OFDM-ZF-16QAM-WF-SNR30_0dB.png
```

## Benefits

1. **Better Organization**: Images are grouped by channel, making it easy to compare results for the same channel across different configurations

2. **Self-Documenting Filenames**: Each filename contains all the parameters used, eliminating confusion about which configuration produced which result

3. **Easy Comparison**: Multiple configurations can be run for the same channel, with results clearly identifiable by filename

4. **Automated**: Channel name extraction and filename generation are fully automatic based on configuration

5. **Backward Compatible**: Default channel name is used when not specified, maintaining compatibility with existing code

## Migration Notes

### For Existing Code

If you have existing scripts that create `ResultsManager` without the `channel_name` parameter, they will still work using "default" as the channel name:

```python
# Old code - still works
results_manager = ResultsManager(results_dir="results", images_dir="images")
# Images will be saved to: images/default/
```

### For New Code

It's recommended to extract and pass the channel name:

```python
from pathlib import Path

# Extract channel name from configuration
channel_name = "default"
if simulation_settings.channel_type.value == "CUSTOM":
    channel_name = Path(simulation_settings.channel_model_path).stem

# Create results manager with channel name
results_manager = ResultsManager(
    results_dir="results",
    images_dir="images",
    channel_name=channel_name,
)
```

## Testing

All integration tests pass with the new changes (38/38 tests passing). The changes are fully backward compatible.

Run tests with:
```bash
pytest tests/integration/ -v
```

## Example Output

When running a simulation, you'll see:
```
================================================================================
  Processing Results
================================================================================
  ✓ Saved 2 constellation plot(s)
  ✓ Updated BER results CSV: results/ber_results.csv
  ✓ Generated BER vs SNR plot: images/severe_multipath/CP-OFDM-ZF-16QAM-WF-BER_vs_SNR.png

================================================================================
  Summary Statistics
================================================================================
  SNR Range: 20.0 dB to 30.0 dB
  BER Range: 2.498926e-01 to 2.632959e-01
  Average PAPR: 10.75 dB
================================================================================
```

Files generated:
```
images/severe_multipath/
├── CP-OFDM-ZF-16QAM-WF-SNR20_0dB.png
├── CP-OFDM-ZF-16QAM-WF-SNR30_0dB.png
└── CP-OFDM-ZF-16QAM-WF-BER_vs_SNR.png
```
