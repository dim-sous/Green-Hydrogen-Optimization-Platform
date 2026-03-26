"""Renewable power source model (stochastic generation + CSV loading).

Placeholder — will be replaced with real data in future versions.
Generates realistic power profiles scaled for a single AEL cell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from config.parameters import PowerSourceParams


class PowerSource:
    """Renewable power source with forecast capability."""

    def __init__(self, params: PowerSourceParams | None = None, seed: int = 42):
        self.p = params or PowerSourceParams()
        self.rng = np.random.default_rng(seed)
        self._power_profile: np.ndarray | None = None
        self._time: np.ndarray | None = None

    def _speed_to_power(self, v: float) -> float:
        """Convert source speed to power via cubic power curve [W]."""
        x = np.clip((v - self.p.v_ci_ms) / (self.p.v_r_ms - self.p.v_ci_ms), 0.0, 1.0)
        return self.p.P_rated_w * x**3

    def generate_profile(
        self, duration: float, dt: float, mean_v: float = 8.0, k_shape: float = 2.0,
    ) -> None:
        """Generate stochastic power profile with Weibull marginal distribution.

        Uses the inverse-CDF method on autocorrelated uniform samples to
        produce temporally correlated speeds that follow a Weibull
        distribution.

        Args:
            duration: Simulation duration [s]
            dt: Time step [s]
            mean_v: Mean source speed [m/s]
            k_shape: Weibull shape parameter [-]
        """
        from scipy.special import gamma
        from scipy.stats import norm

        n_steps = int(duration / dt) + 1
        self._time = np.linspace(0, duration, n_steps)

        # Weibull scale from desired mean: E[V] = lambda * Gamma(1 + 1/k)
        lam = mean_v / gamma(1.0 + 1.0 / k_shape)

        # Autocorrelated Gaussian noise (first-order AR filter)
        tau = 10.0  # autocorrelation time-steps
        alpha = 1.0 - dt / (tau * dt + dt)
        white = self.rng.normal(0, 1, n_steps)
        z = np.zeros(n_steps)
        z[0] = white[0]
        for i in range(1, n_steps):
            z[i] = alpha * z[i - 1] + (1 - alpha) * white[i]

        # Gaussian CDF -> uniform -> inverse Weibull CDF
        u = norm.cdf(z)
        u = np.clip(u, 1e-12, 1.0 - 1e-12)
        speed = lam * (-np.log(1.0 - u)) ** (1.0 / k_shape)

        self._power_profile = np.array([self._speed_to_power(v) for v in speed])

    def load_csv(self, path: str | Path) -> None:
        """Load power profile from CSV file."""
        df = pd.read_csv(path)
        self._power_profile = df["power_kW"].values * 1000.0
        if "timestamp" in df.columns:
            self._time = np.arange(len(self._power_profile)) * 30.0

    def P_avail(self, step_idx: int) -> float:
        """Return available power at time step [W]."""
        if self._power_profile is None:
            return self.p.P_rated_w * 0.8
        idx = np.clip(step_idx, 0, len(self._power_profile) - 1)
        return float(self._power_profile[idx])

    def P_forecast(self, step_idx: int, N: int, sigma_frac: float = 0.05) -> np.ndarray:
        """Return N-step power forecast with growing uncertainty [W].

        Args:
            step_idx: current time step index
            N: forecast horizon length
            sigma_frac: base noise fraction of true power

        Returns:
            forecast: array of shape (N,) with predicted P_avail [W]
        """
        forecast = np.zeros(N)
        for k in range(N):
            true_p = self.P_avail(step_idx + k + 1)
            sigma = sigma_frac * true_p + 0.02 * self.p.P_rated_w * ((k + 1) / N)
            forecast[k] = max(0.0, true_p + self.rng.normal(0, sigma))
        return forecast
