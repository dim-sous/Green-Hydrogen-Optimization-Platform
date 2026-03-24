"""Lumped thermal sub-model for alkaline electrolyzer."""

import numpy as np
from scipy.integrate import solve_ivp
from config.parameters import ThermalParams


class ThermalModel:
    """Lumped energy balance for electrolyzer stack temperature."""

    def __init__(self, params: ThermalParams | None = None):
        self.p = params or ThermalParams()

    def dTdt(self, T: float, P_el: float, n_dot_H2: float,
             Q_cool: float, T_amb: float | None = None) -> float:
        T_amb = T_amb if T_amb is not None else self.p.T_amb
        Q_loss = self.p.UA_loss * (T - T_amb)
        Q_cool = np.clip(Q_cool, 0.0, self.p.Q_cool_max)
        return (P_el - n_dot_H2 * self.p.DH_rxn - Q_loss - Q_cool) / self.p.C_th

    def step(self, T: float, P_el: float, n_dot_H2: float,
             Q_cool: float, dt: float, T_amb: float | None = None) -> float:
        """Integrate temperature one step forward (RK45)."""
        def rhs(t, y):
            return [self.dTdt(y[0], P_el, n_dot_H2, Q_cool, T_amb)]

        sol = solve_ivp(rhs, [0.0, dt], [T], method="RK45", rtol=1e-8, atol=1e-10)
        return sol.y[0, -1]
