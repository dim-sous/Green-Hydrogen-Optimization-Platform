"""Combined AEL plant model wrapping all sub-models."""

import numpy as np
from config.parameters import (
    ElectrolyzerParams,
    ThermalParams,
    GasSeparatorParams,
    PowerSourceParams,
)
from models.electrolyzer import Electrolyzer
from models.thermal import ThermalModel
from models.gas_separator import GasSeparator


class AELPlant:
    """
    Combined alkaline electrolyzer plant (NumPy, ground truth).

    State:  x = [T, x_HTO]
    Input:  u = [I, Q_cool]
    Output: y = [T, x_HTO, V_stack, n_dot_H2, SEC]
    Disturbance: d = [P_avail, T_amb]
    """

    def __init__(
        self,
        ely_params: ElectrolyzerParams | None = None,
        th_params: ThermalParams | None = None,
        gs_params: GasSeparatorParams | None = None,
    ):
        self.electrolyzer = Electrolyzer(ely_params)
        self.thermal = ThermalModel(th_params)
        self.gas_separator = GasSeparator(gs_params)

        th = th_params or ThermalParams()
        ely = ely_params or ElectrolyzerParams()

        self.T = th.T_ref
        self.x_HTO = 0.005
        self.I_min = ely.I_min
        self.I_max = ely.I_max
        self.Q_cool_max = th.Q_cool_max

    def reset(self, T0: float | None = None, x_HTO_0: float | None = None) -> None:
        self.T = T0 if T0 is not None else 353.15
        self.x_HTO = x_HTO_0 if x_HTO_0 is not None else 0.005

    def step(self, u: np.ndarray, d: np.ndarray, dt: float) -> np.ndarray:
        """Advance plant one time step.

        Args:
            u: [I, Q_cool]
            d: [P_avail, T_amb]
            dt: time step (s)
        Returns:
            y: [T, x_HTO, V_stack, n_dot_H2, SEC]
        """
        I, Q_cool = float(u[0]), float(u[1])
        P_avail, T_amb = float(d[0]), float(d[1])

        I = np.clip(I, self.I_min, self.I_max)
        Q_cool = np.clip(Q_cool, 0.0, self.Q_cool_max)

        # Enforce power constraint: reduce current if needed
        if self.electrolyzer.V_stack(self.T, I) * I > P_avail and I > self.I_min:
            I_lo, I_hi = self.I_min, I
            for _ in range(30):
                I_mid = 0.5 * (I_lo + I_hi)
                if self.electrolyzer.V_stack(self.T, I_mid) * I_mid > P_avail:
                    I_hi = I_mid
                else:
                    I_lo = I_mid
            I = I_lo

        V_stack, P_el, n_dot_H2, eta_F, SEC = self.electrolyzer.step(I, self.T, dt)
        self.T = self.thermal.step(self.T, P_el, n_dot_H2, Q_cool, dt, T_amb)
        self.x_HTO = self.gas_separator.step(self.x_HTO, n_dot_H2, dt)

        return np.array([self.T, self.x_HTO, V_stack, n_dot_H2, SEC])

    def get_state(self) -> np.ndarray:
        return np.array([self.T, self.x_HTO])

    def get_output(self) -> np.ndarray:
        I_nom = 0.5 * (self.I_min + self.I_max)
        V_stack, P_el, n_dot_H2, eta_F, SEC = self.electrolyzer.step(I_nom, self.T, 0.0)
        return np.array([self.T, self.x_HTO, V_stack, n_dot_H2, SEC])
