"""
Visualization utilities for OFDM simulation results.

This module provides plotting functions for constellation diagrams,
constellation order distributions, and other visualization needs.
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.figure import Figure
from numpy.typing import NDArray


def plot_constellation_order_distribution(
    constellation_orders: NDArray[np.int64],
    num_subcarriers: int,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Constellation Order Distribution",
) -> Figure:
    """
    Create a bar chart showing the distribution of constellation orders across subcarriers.

    Args:
        constellation_orders: Array of constellation orders per subcarrier (M in M-QAM/PSK)
        num_subcarriers: Total number of subcarriers
        figsize: Figure size as (width, height) in inches
        title: Plot title

    Returns:
        Matplotlib Figure object containing the bar chart
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Count constellation orders (exclude zeros)
    active_orders = constellation_orders[constellation_orders > 0]
    unique_orders, counts = np.unique(active_orders, return_counts=True)

    if len(unique_orders) == 0:
        # No active subcarriers
        ax.text(
            0.5,
            0.5,
            "No active subcarriers",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_title(title)
        return fig

    # Create bar chart with color gradient
    colors = cm.viridis(np.linspace(0, 1, len(unique_orders)))  # type: ignore
    bars = ax.bar(range(len(unique_orders)), counts, color=colors, edgecolor="black", linewidth=1.5)

    # Set labels and title
    ax.set_xlabel("Constellation Order (M-QAM/PSK)", fontsize=12)
    ax.set_ylabel("Number of Subcarriers", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(unique_orders)))
    ax.set_xticklabels([f"{int(order)}" for order in unique_orders])
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Calculate and display statistics
    active_subcarriers = len(active_orders)
    inactive_subcarriers = num_subcarriers - active_subcarriers
    avg_order = float(np.mean(active_orders))
    median_order = float(np.median(active_orders))

    stats_text = (
        f"Total Subcarriers: {num_subcarriers}\n"
        f"Active: {active_subcarriers} ({100*active_subcarriers/num_subcarriers:.1f}%)\n"
        f"Inactive: {inactive_subcarriers} ({100*inactive_subcarriers/num_subcarriers:.1f}%)\n"
        f"Avg Order: {avg_order:.1f}\n"
        f"Median Order: {median_order:.0f}"
    )

    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"),
    )

    plt.tight_layout()
    return fig


def plot_combined_constellation_analysis(
    constellation_orders: NDArray[np.int64],
    demodulated_symbols: NDArray[np.complex128],
    ideal_constellation_points: NDArray[np.complex128],
    num_subcarriers: int,
    ber: float,
    snr_db: float,
    papr_db: float,
    figsize: Tuple[float, float] = (16, 8),
    title_prefix: str = "Adaptive Modulation",
) -> Figure:
    """
    Create a combined figure with constellation diagram and order distribution.

    Args:
        constellation_orders: Array of constellation orders per subcarrier
        demodulated_symbols: Received/demodulated complex symbols
        ideal_constellation_points: Ideal constellation points for reference
        num_subcarriers: Total number of subcarriers
        ber: Bit error rate
        snr_db: Signal-to-noise ratio in dB
        papr_db: Peak-to-average power ratio in dB
        figsize: Figure size as (width, height) in inches
        title_prefix: Prefix for subplot titles

    Returns:
        Matplotlib Figure object containing both subplots
    """
    fig = plt.figure(figsize=figsize)

    # Subplot 1: Constellation diagram
    ax1 = plt.subplot(1, 2, 1)

    # Plot received symbols
    ax1.scatter(
        demodulated_symbols.real,
        demodulated_symbols.imag,
        color="blue",
        marker=".",
        alpha=0.1,
        s=10,
        label="Received Symbols",
    )

    # Plot ideal constellation points
    ax1.scatter(
        ideal_constellation_points.real,
        ideal_constellation_points.imag,
        color="red",
        marker="o",
        s=50,
        edgecolor="black",
        linewidth=1,
        label="Ideal Points",
        zorder=5,
    )

    # Set plot attributes
    ax1.set_title(f"{title_prefix} - Constellation Diagram", fontsize=14, fontweight="bold")
    ax1.set_xlabel("In-Phase", fontsize=12)
    ax1.set_ylabel("Quadrature", fontsize=12)
    ax1.axhline(0, color="black", lw=0.5)
    ax1.axvline(0, color="black", lw=0.5)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect("equal")

    # Add performance metrics text box
    metrics_text = f"BER: {ber:.6f}\nSNR: {snr_db} dB\nPAPR: {papr_db:.2f} dB"
    ax1.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"),
    )

    # Subplot 2: Constellation order distribution
    ax2 = plt.subplot(1, 2, 2)

    # Count constellation orders
    active_orders = constellation_orders[constellation_orders > 0]

    if len(active_orders) == 0:
        ax2.text(
            0.5,
            0.5,
            "No active subcarriers",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax2.transAxes,
        )
        ax2.set_title(f"{title_prefix} - Order Distribution", fontsize=14, fontweight="bold")
    else:
        unique_orders, counts = np.unique(active_orders, return_counts=True)

        # Create bar chart
        colors = cm.viridis(np.linspace(0, 1, len(unique_orders)))  # type: ignore
        bars = ax2.bar(
            range(len(unique_orders)), counts, color=colors, edgecolor="black", linewidth=1.5
        )

        # Set labels and title
        ax2.set_xlabel("Constellation Order (M-QAM/PSK)", fontsize=12)
        ax2.set_ylabel("Number of Subcarriers", fontsize=12)
        ax2.set_title(f"{title_prefix} - Order Distribution", fontsize=14, fontweight="bold")
        ax2.set_xticks(range(len(unique_orders)))
        ax2.set_xticklabels([f"{int(order)}" for order in unique_orders])
        ax2.grid(True, axis="y", alpha=0.3, linestyle="--")

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # Add statistics text box
        active_subcarriers = len(active_orders)
        inactive_subcarriers = num_subcarriers - active_subcarriers
        avg_order = float(np.mean(active_orders))
        median_order = float(np.median(active_orders))

        stats_text = (
            f"Total: {num_subcarriers}\n"
            f"Active: {active_subcarriers}\n"
            f"Inactive: {inactive_subcarriers}\n"
            f"Avg: {avg_order:.1f}\n"
            f"Median: {median_order:.0f}"
        )

        ax2.text(
            0.98,
            0.98,
            stats_text,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"),
        )

    plt.tight_layout()
    return fig


def plot_water_level_diagram(
    power_allocation: NDArray[np.float64],
    channel_gains: NDArray[np.float64],
    noise_power: float,
    water_level: Optional[float] = None,
    figsize: Tuple[float, float] = (12, 6),
    title: str = "Waterfilling Power Allocation",
) -> Figure:
    """
    Create a waterfilling diagram showing power allocation across subcarriers.

    Args:
        power_allocation: Allocated power per subcarrier
        channel_gains: Channel gain (|H|^2) per subcarrier
        noise_power: Noise power level
        water_level: Water level (mu) value if available
        figsize: Figure size as (width, height) in inches
        title: Plot title

    Returns:
        Matplotlib Figure object containing the waterfilling diagram
    """
    fig, ax = plt.subplots(figsize=figsize)

    num_subcarriers = len(power_allocation)
    subcarrier_indices = np.arange(num_subcarriers)

    # Calculate noise floor for each subcarrier
    noise_floor = noise_power / channel_gains

    # Plot noise floor
    ax.bar(
        subcarrier_indices,
        noise_floor,
        color="lightcoral",
        alpha=0.6,
        label="Noise Floor (N₀/|H|²)",
        edgecolor="black",
        linewidth=0.5,
    )

    # Plot allocated power on top of noise floor
    ax.bar(
        subcarrier_indices,
        power_allocation,
        bottom=noise_floor,
        color="steelblue",
        alpha=0.7,
        label="Allocated Power",
        edgecolor="black",
        linewidth=0.5,
    )

    # Plot water level if provided
    if water_level is not None:
        ax.axhline(
            water_level,
            color="darkblue",
            linestyle="--",
            linewidth=2,
            label=f"Water Level (μ={water_level:.4f})",
        )

    # Set labels and title
    ax.set_xlabel("Subcarrier Index", fontsize=12)
    ax.set_ylabel("Power Level", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add statistics
    active_subcarriers = np.sum(power_allocation > 1e-10)
    total_power = float(np.sum(power_allocation))
    avg_power = (
        float(np.mean(power_allocation[power_allocation > 1e-10])) if active_subcarriers > 0 else 0
    )

    stats_text = (
        f"Total Power: {total_power:.4f}\n"
        f"Active Subcarriers: {active_subcarriers}/{num_subcarriers}\n"
        f"Avg Power (active): {avg_power:.4f}"
    )

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"),
    )

    plt.tight_layout()
    return fig


def plot_adaptive_constellation_diagram(
    received_symbols: NDArray[np.complex128],
    constellation_orders: NDArray[np.int64],
    constellation_title: str,
    num_subcarriers: int,
    ber: float,
    ser: float,
    snr_db: float,
    papr_db: float,
    figsize: Tuple[float, float] = (14, 6),
    title_prefix: str = "Adaptive Modulation",
) -> Figure:
    """
    Create a constellation diagram showing received symbols grouped by constellation order.

    For adaptive modulation, this shows received symbols color-coded by their constellation
    order, along with ideal constellation points for each order used.

    Args:
        received_symbols: Complex-valued received symbols (shape: num_ofdm_symbols * num_subcarriers)
        constellation_orders: Array of constellation orders per subcarrier
        num_subcarriers: Number of subcarriers
        ber: Bit error rate
        snr_db: Signal-to-noise ratio in dB
        papr_db: Peak-to-average power ratio in dB
        figsize: Figure size as (width, height) in inches
        title_prefix: Prefix for plot titles

    Returns:
        Matplotlib Figure object with constellation diagram
    """
    from ofdm_based_systems.constellation.models import (
        PSKConstellationMapper,
        QAMConstellationMapper,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Get unique constellation orders (excluding zeros)
    unique_orders = np.unique(constellation_orders[constellation_orders > 0])

    # Normalize received symbols for visualization
    max_magnitude = np.max(np.abs(received_symbols)) if len(received_symbols) > 0 else 1.0
    if max_magnitude > 0:
        normalized_symbols = received_symbols / max_magnitude
    else:
        normalized_symbols = received_symbols

    # Reshape symbols to (num_ofdm_symbols, num_subcarriers)
    num_total_symbols = len(received_symbols)
    num_ofdm_symbols = num_total_symbols // num_subcarriers

    if num_ofdm_symbols > 0:
        symbols_matrix = normalized_symbols.reshape(num_ofdm_symbols, num_subcarriers)
    else:
        symbols_matrix = normalized_symbols.reshape(-1, num_subcarriers)

    # Plot 1: Received symbols grouped by constellation order
    colors = cm.viridis(np.linspace(0, 1, len(unique_orders)))  # type: ignore

    for idx, order in enumerate(unique_orders):
        # Find subcarriers with this order
        subcarrier_mask = constellation_orders == order

        # Extract symbols from those subcarriers
        symbols_for_order = symbols_matrix[:, subcarrier_mask].flatten()

        # Plot received symbols
        ax1.scatter(
            symbols_for_order.real,
            symbols_for_order.imag,
            alpha=0.3,
            s=20,
            c=[colors[idx]],
            label=f"{int(order)}-{constellation_title} ({np.sum(subcarrier_mask)} subcarriers)",
        )

    # Add ideal constellation points for each order
    constellation_type = "QAM" if "QAM" in constellation_title.upper() else "PSK"

    if constellation_type == "QAM":
        for idx, order in enumerate(unique_orders):
            # Create mapper for this order
            mapper = QAMConstellationMapper(order=int(order))
            ideal_points = mapper.constellation

            # Normalize ideal points
            if max_magnitude > 0:
                ideal_normalized = ideal_points / max_magnitude
            else:
                ideal_normalized = ideal_points

            # Plot ideal points
            ax1.scatter(
                ideal_normalized.real,
                ideal_normalized.imag,
                marker="x",
                s=100,
                linewidths=2,
            )

            from ofdm_based_systems.constellation.models import (
                PSKConstellationMapper,
            )

    if constellation_type == "PSK":
        for idx, order in enumerate(unique_orders):
            mapper = PSKConstellationMapper(order=int(order))
            ideal_points = mapper.constellation

            # Normalize ideal points
            if max_magnitude > 0:
                ideal_normalized = ideal_points / max_magnitude
            else:
                ideal_normalized = ideal_points

            # Plot ideal points
            ax1.scatter(
                ideal_normalized.real,
                ideal_normalized.imag,
                marker="x",
                s=100,
                linewidths=2,
            )

    if constellation_type not in ["QAM", "PSK"]:
        raise ValueError(f"Unsupported constellation type: {constellation_type}")

    ax1.set_title(f"{title_prefix} - Constellation Diagram", fontsize=12, fontweight="bold")
    ax1.set_xlabel("In-Phase (I)", fontsize=11)
    ax1.set_ylabel("Quadrature (Q)", fontsize=11)
    ax1.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax1.axvline(0, color="black", lw=0.5, alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    # Add statistics text box
    textstr = f"BER: {ber:.6f}\nSER: {ser:.6f}\nSNR: {snr_db} dB\nPAPR: {papr_db:.2f} dB"
    ax1.text(
        0.98,
        0.02,
        textstr,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"),
    )

    # Plot 2: Constellation order distribution
    active_orders = constellation_orders[constellation_orders > 0]
    unique_orders_dist, counts = np.unique(active_orders, return_counts=True)

    colors_dist = cm.viridis(np.linspace(0, 1, len(unique_orders_dist)))  # type: ignore
    bars = ax2.bar(
        range(len(unique_orders_dist)),
        counts,
        color=colors_dist,
        edgecolor="black",
        linewidth=1.5,
    )

    ax2.set_xlabel("Constellation Order (M-QAM)", fontsize=11)
    ax2.set_ylabel("Number of Subcarriers", fontsize=11)
    ax2.set_title("Constellation Order Distribution", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(len(unique_orders_dist)))
    ax2.set_xticklabels([f"{int(order)}" for order in unique_orders_dist])
    ax2.grid(True, axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Add statistics text box
    active_subcarriers = len(active_orders)
    inactive_subcarriers = num_subcarriers - active_subcarriers
    avg_order = np.mean(active_orders) if active_subcarriers > 0 else 0
    median_order = np.median(active_orders) if active_subcarriers > 0 else 0

    stats_text = (
        f"Active: {active_subcarriers}/{num_subcarriers}\n"
        f"Inactive: {inactive_subcarriers}\n"
        f"Avg Order: {avg_order:.1f}\n"
        f"Median Order: {median_order:.0f}"
    )

    ax2.text(
        0.98,
        0.98,
        stats_text,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"),
    )

    plt.tight_layout()
    return fig
    plt.tight_layout()
    return fig
    plt.tight_layout()
    return fig
