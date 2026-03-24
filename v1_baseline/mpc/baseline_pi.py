"""PI baseline controller for alkaline electrolyzer."""

import numpy as np
from config.parameters import PIControlParams, ThermalParams, ElectrolyzerParams, PowerSourceParams


class PIController:
    """PI controller: feedforward current + feedback cooling."""

    def __init__(
        self,
        pi: PIControlParams | None = None,
        th: ThermalParams | None = None,
        ely: ElectrolyzerParams | None = None,
        ps: PowerSourceParams | None = None,
    ):
        self.pi = pi or PIControlParams()
        th = th or ThermalParams()
        ely = ely or ElectrolyzerParams()
        ps = ps or PowerSourceParams()

        self.T_ref = th.T_ref
        self.I_min = ely.I_min
        self.I_max = ely.I_max
        self.P_rated = ps.P_rated
        self.Q_cool_max = th.Q_cool_max
        self._integral_error = 0.0

    def reset(self) -> None:
        self._integral_error = 0.0

    def step(self, x: np.ndarray, d: np.ndarray, dt: float) -> np.ndarray:
        """Compute control action u = [I, Q_cool] from state x and disturbance d."""
        T = x[0]
        P_avail = d[0]

        # Feedforward current
        I = self.pi.I_nom * (P_avail / self.P_rated)
        I = np.clip(I, self.I_min, self.I_max)

        # PI temperature control
        error = T - self.T_ref
        self._integral_error += error * dt
        self._integral_error = np.clip(
            self._integral_error,
            -self.Q_cool_max / max(abs(self.pi.K_i_T), 1e-6),
            self.Q_cool_max / max(abs(self.pi.K_i_T), 1e-6),
        )

        Q_cool = self.pi.K_p_T * error + self.pi.K_i_T * self._integral_error
        Q_cool = np.clip(Q_cool, 0.0, self.Q_cool_max)

        return np.array([I, Q_cool])
