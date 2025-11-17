- Must include the Gap function for calculating the constellation order based on desired erro probability.
- Each constellation type has its own Gap function implementation.

$$
R = \frac{1}{2}(log_2(1 + \frac{SNR}{\Gamma}))
$$

Where:

- \( $R$ \) is the achievable rate (bits/symbol)
- \( $SNR$ \) is the signal-to-noise ratio per subcarrier
- \( $\Gamma$ \) is the Gap function value for the desired error probability and constellation type

$$
Q(x) = \int^\infty_x \frac{e^{\frac{-u^3}{2}}}{\sqrt{2\pi}}du
$$

Where:

- \( $Q(x)$ \) is the Q-function representing the tail probability of the standard normal distribution
- \( $u$ \) is the integration variable

## QAM Constellation Gap Function

$$
    \Gamma = \frac{1}{3}\left[Q^{-1}\frac{SER}{4}\right]^2
$$

## PSK Constellation Gap Function

$$
\Gama^* = \left(\right)
$$

## Time Variant Channels / Doppler
