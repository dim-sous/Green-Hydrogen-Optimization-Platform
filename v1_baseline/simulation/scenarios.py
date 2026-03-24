"""Predefined test scenarios for AEL simulation."""

import numpy as np
from data.power_source import PowerSource
from config.parameters import PowerSourceParams, ThermalParams


def steady_wind(dt: float = 30.0, seed: int = 42) -> dict:
    """Constant 80% rated power for 2 hours."""
    duration = 2 * 3600.0
    n_steps = int(duration / dt)
    P_rated = PowerSourceParams().P_rated
    T_amb = ThermalParams().T_amb
    return {
        "name": "steady_wind",
        "duration": duration, "dt": dt, "n_steps": n_steps,
        "power_profile": np.full(n_steps, 0.8 * P_rated),
        "T_amb_profile": np.full(n_steps, T_amb),
    }


def ramp_up_down(dt: float = 30.0, seed: int = 42) -> dict:
    """Linear ramp 20% -> 100% -> 20% over 3 hours."""
    duration = 3 * 3600.0
    n_steps = int(duration / dt)
    P_rated = PowerSourceParams().P_rated
    T_amb = ThermalParams().T_amb
    t = np.linspace(0, duration, n_steps)
    half = duration / 2.0
    power = np.where(
        t <= half,
        P_rated * (0.2 + 0.8 * t / half),
        P_rated * (1.0 - 0.8 * (t - half) / half),
    )
    return {
        "name": "ramp_up_down",
        "duration": duration, "dt": dt, "n_steps": n_steps,
        "power_profile": power,
        "T_amb_profile": np.full(n_steps, T_amb),
    }


def turbulent_wind(dt: float = 30.0, seed: int = 42) -> dict:
    """Kaimal-spectrum turbulent wind for 6 hours."""
    duration = 6 * 3600.0
    n_steps = int(duration / dt)
    T_amb = ThermalParams().T_amb
    ps = PowerSource(seed=seed)
    ps.generate_wind_profile(duration, dt, mean_v=8.0, sigma_v=2.0)
    power = np.array([ps.P_avail(k) for k in range(n_steps)])
    return {
        "name": "turbulent_wind",
        "duration": duration, "dt": dt, "n_steps": n_steps,
        "power_profile": power,
        "T_amb_profile": np.full(n_steps, T_amb),
    }


def cold_start(dt: float = 30.0, seed: int = 42) -> dict:
    """Plant starts at ambient temperature, 2 h at 80% power."""
    duration = 2 * 3600.0
    n_steps = int(duration / dt)
    P_rated = PowerSourceParams().P_rated
    T_amb = ThermalParams().T_amb
    return {
        "name": "cold_start",
        "duration": duration, "dt": dt, "n_steps": n_steps,
        "power_profile": np.full(n_steps, 0.8 * P_rated),
        "T_amb_profile": np.full(n_steps, T_amb),
        "T0": T_amb,
    }


SCENARIOS = {
    "steady_wind": steady_wind,
    "ramp_up_down": ramp_up_down,
    "turbulent_wind": turbulent_wind,
    "cold_start": cold_start,
}
