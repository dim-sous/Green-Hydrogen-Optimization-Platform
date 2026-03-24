"""Electrochemical sub-model for alkaline electrolyzer stack (Ulleberg form)."""

import numpy as np
from config.parameters import ElectrolyzerParams


class Electrolyzer:
    """Single-stack alkaline electrolyzer electrochemical model."""

    def __init__(self, params: ElectrolyzerParams | None = None):
        self.p = params or ElectrolyzerParams()

    def V_rev(self, T: float) -> float:
        return 1.229 - 8.5e-4 * (T - 298.15)

    def V_ohm(self, T: float, I: float) -> float:
        return (self.p.r1 + self.p.r2 * T) / self.p.A * I

    def V_act(self, T: float, I: float) -> float:
        j = I / self.p.A
        return self.p.s * np.log10(
            (self.p.t1 + self.p.t2 / T + self.p.t3 / T**2) * j + 1.0
        )

    def V_cell(self, T: float, I: float) -> float:
        return self.V_rev(T) + self.V_ohm(T, I) + self.V_act(T, I)

    def V_stack(self, T: float, I: float) -> float:
        return self.p.N_cells * self.V_cell(T, I)

    def P_el(self, T: float, I: float) -> float:
        return self.V_stack(T, I) * I

    def eta_F(self, I: float) -> float:
        j = I / self.p.A
        return j**2 / (self.p.f1 + j**2) * self.p.f2

    def n_dot_H2(self, T: float, I: float) -> float:
        return self.eta_F(I) * self.p.N_cells * I / (2.0 * self.p.F_const)

    def SEC(self, T: float, I: float) -> float:
        n_h2 = self.n_dot_H2(T, I)
        if n_h2 < 1e-12:
            return 0.0
        return self.P_el(T, I) / n_h2

    def step(self, I: float, T: float, dt: float) -> tuple:
        """Compute electrochemical outputs.

        Returns:
            (V_stack, P_el, n_dot_H2, eta_F, SEC)
        """
        I = np.clip(I, self.p.I_min, self.p.I_max)
        v_stack = self.V_stack(T, I)
        p_el = v_stack * I
        eta_f = self.eta_F(I)
        n_h2 = self.n_dot_H2(T, I)
        sec = p_el / n_h2 if n_h2 > 1e-12 else 0.0
        return v_stack, p_el, n_h2, eta_f, sec
