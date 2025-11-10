"""Generate sample channel impulse response files for testing.

This script creates various channel models that can be used in OFDM simulations.
"""

from pathlib import Path

import numpy as np


def create_channel_models():
    """Create various channel impulse response models."""

    # Create output directory
    output_dir = Path("config/channel_models")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating channel impulse response models...")
    print("=" * 80)

    # 1. Default multipath channel (4 taps) - Same as hardcoded
    default_channel = np.array(
        [
            (7.767824138452235072e-01 + 4.560896742466611919e-01j),
            (-6.669848996328063551e-02 + 2.839935704583463338e-01j),
            (1.398968327715586490e-01 - 1.591963958343969865e-01j),
            (2.229949514514480494e-02 + 2.409945439452868821e-01j),
        ],
        dtype=np.complex128,
    )
    filepath = output_dir / "default_multipath.npy"
    np.save(filepath, default_channel)
    print(f"✓ Created: {filepath}")
    print(f"  - Taps: {len(default_channel)}")
    print(f"  - Type: Multipath (frequency-selective)")
    print(f"  - Description: Default 4-tap channel model")

    # 2. Flat fading channel (1 tap)
    flat_channel = np.array([1.0 + 0.0j], dtype=np.complex128)
    filepath = output_dir / "flat_fading.npy"
    np.save(filepath, flat_channel)
    print(f"\n✓ Created: {filepath}")
    print(f"  - Taps: {len(flat_channel)}")
    print(f"  - Type: Flat fading (frequency-flat)")
    print(f"  - Description: Single tap, no ISI")

    # 3. Severe multipath channel (8 taps)
    severe_multipath = np.array(
        [
            (0.8 + 0.3j),
            (0.5 - 0.2j),
            (0.3 + 0.4j),
            (-0.2 + 0.3j),
            (0.15 - 0.25j),
            (0.1 + 0.1j),
            (-0.08 + 0.12j),
            (0.05 - 0.05j),
        ],
        dtype=np.complex128,
    )
    # Normalize to unit energy
    severe_multipath = severe_multipath / np.sqrt(np.sum(np.abs(severe_multipath) ** 2))
    filepath = output_dir / "severe_multipath.npy"
    np.save(filepath, severe_multipath)
    print(f"\n✓ Created: {filepath}")
    print(f"  - Taps: {len(severe_multipath)}")
    print(f"  - Type: Severe multipath (frequency-selective)")
    print(f"  - Description: 8-tap channel with significant ISI")

    # 4. Rayleigh fading channel (random, 6 taps)
    np.random.seed(42)  # For reproducibility
    num_taps = 6
    rayleigh_channel = (np.random.randn(num_taps) + 1j * np.random.randn(num_taps)) / np.sqrt(2)
    # Apply exponential power delay profile
    delay_profile = np.exp(-np.arange(num_taps) / 2.0)
    rayleigh_channel = rayleigh_channel * np.sqrt(delay_profile)
    # Normalize
    rayleigh_channel = rayleigh_channel / np.sqrt(np.sum(np.abs(rayleigh_channel) ** 2))
    filepath = output_dir / "rayleigh_fading.npy"
    np.save(filepath, rayleigh_channel)
    print(f"\n✓ Created: {filepath}")
    print(f"  - Taps: {len(rayleigh_channel)}")
    print(f"  - Type: Rayleigh fading (random)")
    print(f"  - Description: 6-tap channel with exponential power delay profile")

    # 5. Two-ray channel (2 taps) - Simple multipath
    two_ray = np.array(
        [
            (1.0 + 0.0j),
            (0.5 - 0.3j),
        ],
        dtype=np.complex128,
    )
    two_ray = two_ray / np.sqrt(np.sum(np.abs(two_ray) ** 2))
    filepath = output_dir / "two_ray.npy"
    np.save(filepath, two_ray)
    print(f"\n✓ Created: {filepath}")
    print(f"  - Taps: {len(two_ray)}")
    print(f"  - Type: Two-ray multipath")
    print(f"  - Description: Simple 2-tap channel (direct + reflection)")

    # 6. Lin-Phoong P1 channel (from IEEE paper, if exists)
    # This is a placeholder - you would load actual channel data
    lin_phoong = np.array(
        [
            (0.9 + 0.1j),
            (0.3 - 0.2j),
            (-0.1 + 0.3j),
            (0.05 + 0.05j),
        ],
        dtype=np.complex128,
    )
    lin_phoong = lin_phoong / np.sqrt(np.sum(np.abs(lin_phoong) ** 2))
    filepath = output_dir / "Lin-Phoong_P1.npy"
    np.save(filepath, lin_phoong)
    print(f"\n✓ Created: {filepath}")
    print(f"  - Taps: {len(lin_phoong)}")
    print(f"  - Type: Lin-Phoong P1 (reference)")
    print(f"  - Description: 4-tap reference channel from literature")

    print("\n" + "=" * 80)
    print(f"All channel models saved to: {output_dir.absolute()}")
    print("=" * 80)

    # Print usage example
    print("\nUsage in configuration JSON:")
    print("{")
    print('    "channel_type": "CUSTOM",')
    print('    "channel_model_path": "config/channel_models/default_multipath.npy",')
    print("    ...")
    print("}")

    print("\nOr use FLAT channel type to use default hardcoded channel:")
    print("{")
    print('    "channel_type": "FLAT",')
    print('    "channel_model_path": "",')
    print("    ...")
    print("}")


if __name__ == "__main__":
    create_channel_models()
    create_channel_models()
