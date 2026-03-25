"""PI baseline controller for alkaline electrolyzer.

Feedback-only cooling control. Current is determined by the plant
(power equality V_cell·I = P_avail), so the PI only controls Q_cool.
Kept as a comparison baseline for NMPC.
"""

import numpy as np
from config.parameters import PIControlParams, ThermalParams


class PIController:
    """PI feedback controller: temperature error → Q_cool."""

    def __init__(
        self,
        pi: PIControlParams | None = None,
        th: ThermalParams | None = None,
    ):
        self.pi = pi or PIControlParams()
        th = th or ThermalParams()

        self.T_ref = th.T_ref_k
        self.Q_cool_max = th.Q_cool_max_w
        self._integral_error = 0.0

    def reset(self) -> None:
        self._integral_error = 0.0

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

        # PI temperature control with anti-windup
        error = T - self.T_ref
        self._integral_error += error * dt
        self._integral_error = np.clip(
            self._integral_error,
            -self.Q_cool_max / max(abs(self.pi.K_i_T), 1e-6),
            self.Q_cool_max / max(abs(self.pi.K_i_T), 1e-6),
        )

        Q_cool = self.pi.K_p_T * error + self.pi.K_i_T * self._integral_error
        Q_cool = np.clip(Q_cool, 0.0, self.Q_cool_max)

        return np.array([Q_cool])
