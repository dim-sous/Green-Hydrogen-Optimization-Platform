"""Gas separator / hydrogen-to-oxygen (HTO) crossover model."""

import numpy as np
from config.parameters import GasSeparatorParams


class GasSeparator:
    """HTO crossover dynamics in the gas separator."""

    def __init__(self, params: GasSeparatorParams | None = None):
        self.p = params or GasSeparatorParams()

    def dx_HTO_dt(self, x_HTO: float, n_dot_H2: float) -> float:
        """Rate of change of HTO mole fraction.

        Uses molar holdup n_total = p_op * V_sep / (R * T_ref) as denominator
        for correct dimensional analysis (result in 1/s).
        """
        dp = 0.5 * self.p.p_op  # O2-H2 pressure difference
        n_total = self.p.n_total  # molar holdup (mol)
        return (self.p.K_perm * dp - x_HTO * n_dot_H2) / n_total

    def step(self, x_HTO: float, n_dot_H2: float, dt: float) -> float:
        """Euler integration of HTO dynamics."""
        dx = self.dx_HTO_dt(x_HTO, n_dot_H2) * dt
        x_HTO_next = x_HTO + dx
        return np.clip(x_HTO_next, 0.0, 1.0)
