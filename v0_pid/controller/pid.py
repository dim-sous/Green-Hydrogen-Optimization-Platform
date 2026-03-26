"""PID feedback controller for alkaline electrolyzer.

Feedback-only cooling control. Current is determined by the plant
(power equality V_cell·I = P_avail), so the PID only controls Q_cool.

Uses derivative-on-measurement with first-order low-pass filter
to avoid derivative kick and reduce noise sensitivity.

Control law:
    Q_cool = K_p · (T − T_ref) + K_i · ∫(T − T_ref) dt + D_filtered

where D_filtered applies a first-order filter to K_d · dT/dt:
    α = dt / (τ_d + dt)
    D_filtered[k] = (1 − α) · D_filtered[k−1] + α · K_d · (T[k] − T[k−1]) / dt
"""

import numpy as np
from config.parameters import PIDControlParams, ThermalParams


class PIDController:
    """PID feedback controller: temperature error -> Q_cool."""

    def __init__(
        self,
        pid: PIDControlParams | None = None,
        th: ThermalParams | None = None,
    ):
        self.pid = pid or PIDControlParams()
        th = th or ThermalParams()

        self.T_ref = th.T_ref_k
        self.Q_cool_max = th.Q_cool_max_w

        # Internal states
        self._integral_error: float = 0.0
        self._T_prev: float | None = None
        self._d_filtered: float = 0.0

    def reset(self) -> None:
        self._integral_error = 0.0
        self._T_prev = None
        self._d_filtered = 0.0

    def step(self, x: np.ndarray, d: np.ndarray, dt: float) -> np.ndarray:
        """Compute control action u = [Q_cool] from state x and disturbance d.

        Args:
            x: [T]               state [K]
            d: [P_avail, T_amb]  disturbances [W, K]
            dt: time step [s]

        Returns:
            u: [Q_cool]  control [W]
        """
        T = x[0]
        error = T - self.T_ref

        # --- Proportional term ---
        P_term = self.pid.K_p_T * error

        # --- Integral term with anti-windup ---
        self._integral_error += error * dt
        max_integral = self.Q_cool_max / max(abs(self.pid.K_i_T), 1e-6)
        self._integral_error = np.clip(
            self._integral_error, -max_integral, max_integral
        )
        I_term = self.pid.K_i_T * self._integral_error

        # --- Derivative term (derivative-on-measurement, filtered) ---
        if self._T_prev is not None and dt > 0:
            dT_dt = (T - self._T_prev) / dt
            raw_D = self.pid.K_d_T * dT_dt
            alpha = dt / (self.pid.tau_d_s + dt)
            self._d_filtered = (1.0 - alpha) * self._d_filtered + alpha * raw_D
        D_term = self._d_filtered
        self._T_prev = T

        # --- Combine and clamp ---
        Q_cool = P_term + I_term + D_term
        Q_cool = np.clip(Q_cool, 0.0, self.Q_cool_max)

        return np.array([Q_cool])
