"""Renewable power source model (wind turbine + CSV loading)."""

import numpy as np
import pandas as pd
from pathlib import Path
from config.parameters import PowerSourceParams


class PowerSource:
    """Wind-based renewable power source with forecast capability."""

    def __init__(self, params: PowerSourceParams | None = None, seed: int = 42):
        self.p = params or PowerSourceParams()
        self.rng = np.random.default_rng(seed)
        self._power_profile: np.ndarray | None = None
        self._time: np.ndarray | None = None

    def _wind_power(self, v: float) -> float:
        x = np.clip((v - self.p.v_ci) / (self.p.v_r - self.p.v_ci), 0.0, 1.0)
        return self.p.P_rated * x**3

    def generate_wind_profile(
        self, duration: float, dt: float, mean_v: float = 8.0, sigma_v: float = 1.5,
    ) -> None:
        """Generate turbulent wind speed profile (Kaimal-like filtered noise)."""
        n_steps = int(duration / dt) + 1
        self._time = np.linspace(0, duration, n_steps)

        white = self.rng.normal(0, 1, n_steps)
        tau = 10.0
        alpha = 1.0 - dt / (tau * dt + dt)
        filtered = np.zeros(n_steps)
        filtered[0] = white[0]
        for i in range(1, n_steps):
            filtered[i] = alpha * filtered[i - 1] + (1 - alpha) * white[i]

        wind_speed = np.maximum(mean_v + sigma_v * filtered, 0.0)
        self._power_profile = np.array([self._wind_power(v) for v in wind_speed])

    def load_csv(self, path: str | Path) -> None:
        df = pd.read_csv(path)
        self._power_profile = df["power_kW"].values * 1000.0
        if "timestamp" in df.columns:
            self._time = np.arange(len(self._power_profile)) * 30.0

    def P_avail(self, step_idx: int) -> float:
        if self._power_profile is None:
            return self.p.P_rated * 0.8
        idx = np.clip(step_idx, 0, len(self._power_profile) - 1)
        return float(self._power_profile[idx])

    def P_forecast(self, step_idx: int, N: int, sigma_frac: float = 0.05) -> np.ndarray:
        forecast = np.zeros(N)
        for k in range(N):
            true_p = self.P_avail(step_idx + k + 1)
            sigma = sigma_frac * true_p + 0.02 * self.p.P_rated * ((k + 1) / N)
            forecast[k] = max(0.0, true_p + self.rng.normal(0, sigma))
        return forecast
