"""This script takes in a channel impulse response series and plots the time and frequency response."""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    h = np.array([0.5, 0, 0, 0.3, 0.1])  # Example channel impulse response
    h = h / np.linalg.norm(h)  # Normalize the impulse response
    N = 512
    H = np.fft.fft(h, N)  # Compute the frequency response
    freq = np.linspace(0, 2 * np.pi, N)

    # Plot time response
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(h)
    plt.title("Time Response")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    # Plot frequency response
    plt.subplot(2, 1, 2)
    plt.plot(freq, np.abs(H))
    plt.title("Frequency Response")
    plt.xlabel("Frequency (radians)")
    plt.ylabel("Magnitude")

    plt.tight_layout()
    plt.savefig("channel_response.png")
