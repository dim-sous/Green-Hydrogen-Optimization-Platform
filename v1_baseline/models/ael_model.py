"""Alkaline electrolyzer model — CasADi symbolic DAE and NumPy plant.

Single-cell Ulleberg electrochemical model with lumped thermal dynamics.
The system is a semi-explicit index-1 DAE:

    Differential state:  x = [T]           temperature [K]
    Algebraic variable:  z = [I]           current [A]
    Control input:       u = [Q_cool]      cooling power [W]
    Disturbances:        p = [P_avail, T_amb]

    ODE:  dT/dt = (P_el − ṅ_H₂·ΔH_rxn − UA·(T−T_amb) − Q_cool) / C_th
    ALG:  0 = V_cell(T, I)·I − P_avail
"""

from __future__ import annotations

import numpy as np
import casadi as ca
from scipy.optimize import brentq

from config.parameters import ElectrolyzerParams, ThermalParams, PowerSourceParams


# ---------------------------------------------------------------------------
# Electrochemistry helpers (NumPy, used by AELPlant)
# ---------------------------------------------------------------------------

def v_rev(T: float) -> float:
    """Reversible cell voltage (Nernst) [V]."""
    return 1.229 - 8.5e-4 * (T - 298.15)


def v_ohm(T: float, I: float, ep: ElectrolyzerParams) -> float:
    """Ohmic overpotential [V]."""
    return (ep.r1_ohm_m2 + ep.r2_ohm_m2_k * T) / ep.A_m2 * I


def v_act(T: float, I: float, ep: ElectrolyzerParams) -> float:
    """Activation overpotential (Ulleberg) [V]."""
    j = I / ep.A_m2
    return ep.s_v * np.log10(
        (ep.t1 + ep.t2 / T + ep.t3 / T**2) * j + 1.0
    )


def v_cell(T: float, I: float, ep: ElectrolyzerParams) -> float:
    """Single cell voltage [V]."""
    return v_rev(T) + v_ohm(T, I, ep) + v_act(T, I, ep)


def eta_faradaic(I: float, ep: ElectrolyzerParams) -> float:
    """Faradaic efficiency [-]."""
    j = I / ep.A_m2
    return j**2 / (ep.f1 + j**2) * ep.f2


def n_dot_h2(I: float, ep: ElectrolyzerParams) -> float:
    """Hydrogen production rate [mol/s]."""
    return eta_faradaic(I, ep) * ep.N_cells * I / (2.0 * ep.F_const)


# ---------------------------------------------------------------------------
# CasADi symbolic DAE
# ---------------------------------------------------------------------------

def build_casadi_dynamics(
    ep: ElectrolyzerParams | None = None,
    tp: ThermalParams | None = None,
) -> dict:
    """Build CasADi symbolic DAE model.

    Returns dict with keys:
        x, z, u, p          — symbolic variables
        ode, alg             — DAE right-hand sides
        dae                  — dict ready for ca.integrator
        n_dot_H2, V_cell, P_el, SEC  — symbolic outputs
        h                    — CasADi Function (x, z, u, p) → y
    """
    ep = ep or ElectrolyzerParams()
    tp = tp or ThermalParams()

    # Symbolic variables
    x = ca.MX.sym("x", 1)       # [T]
    z = ca.MX.sym("z", 1)       # [I]  (algebraic)
    u = ca.MX.sym("u", 1)       # [Q_cool]
    p = ca.MX.sym("p", 2)       # [P_avail, T_amb]

    T = x[0]
    I = z[0]
    Q_cool = u[0]
    P_avail = p[0]
    T_amb = p[1]

    # --- Electrochemistry (Ulleberg) ---
    j = I / ep.A_m2
    V_rev = 1.229 - 8.5e-4 * (T - 298.15)
    V_ohm = (ep.r1_ohm_m2 + ep.r2_ohm_m2_k * T) / ep.A_m2 * I
    V_act = ep.s_v * ca.log10((ep.t1 + ep.t2 / T + ep.t3 / T**2) * j + 1.0)
    V_cell = V_rev + V_ohm + V_act
    P_el = V_cell * I

    # --- Faradaic efficiency ---
    eps = 1e-6
    eta_F = (j**2) / (ep.f1 + j**2 + eps) * ep.f2

    # --- H₂ production ---
    n_dot_H2 = eta_F * ep.N_cells * I / (2.0 * ep.F_const)
    SEC = P_el / (n_dot_H2 + eps)

    # --- Thermal ODE ---
    Q_loss = tp.UA_loss_wk * (T - T_amb)
    dTdt = (P_el - n_dot_H2 * tp.DH_rxn_jmol - Q_loss - Q_cool) / tp.C_th_jk

    # --- Algebraic constraint: power equality ---
    # V_cell(T, I) · I = P_avail
    # The NMPC enforces I ∈ [I_min, I_max] via NLP bounds.
    # For the EKF/validation, ensure P_avail is within the feasible range.
    alg = V_cell * I - P_avail

    # --- DAE dict for CasADi integrator ---
    dae_dict = {
        "x": x,
        "z": z,
        "p": ca.vertcat(u, p),   # combined parameters: [Q_cool, P_avail, T_amb]
        "ode": dTdt,
        "alg": alg,
    }

    # --- Output function ---
    y = ca.vertcat(T, V_cell, n_dot_H2, SEC, I)
    h_func = ca.Function("h", [x, z, u, p], [y],
                          ["x", "z", "u", "p"], ["y"])

    return {
        "x": x, "z": z, "u": u, "p": p,
        "ode": dTdt, "alg": alg,
        "dae": dae_dict,
        "n_dot_H2": n_dot_H2, "V_cell": V_cell, "P_el": P_el, "SEC": SEC,
        "h": h_func,
    }


def build_casadi_integrator(
    model: dict,
    dt: float,
) -> ca.Function:
    """Build a collocation integrator for the DAE.

    Args:
        model: output of build_casadi_dynamics()
        dt: integration step [s]

    Returns:
        F: CasADi Function  F(x0, z0, p) -> (x_next, z_next)
            where p = [Q_cool, P_avail, T_amb]
    """
    dae = model["dae"]
    opts = {
        "number_of_finite_elements": 1,
        "interpolation_order": 3,
        "rootfinder": "newton",
    }
    intg = ca.integrator("F_col", "collocation", dae, 0.0, dt, opts)

    # Wrap into a clean function signature
    x0 = ca.MX.sym("x0", 1)
    z0 = ca.MX.sym("z0", 1)
    p_in = ca.MX.sym("p_in", 3)   # [Q_cool, P_avail, T_amb]

    result = intg(x0=x0, z0=z0, p=p_in)
    x_next = result["xf"]
    z_next = result["zf"]

    return ca.Function("F", [x0, z0, p_in], [x_next, z_next],
                        ["x0", "z0", "p"], ["x_next", "z_next"])


# ---------------------------------------------------------------------------
# NumPy ground-truth plant
# ---------------------------------------------------------------------------

class AELPlant:
    """Single-cell alkaline electrolyzer plant (NumPy, ground truth).

    State:       x = [T]
    Control:     u = [Q_cool]
    Disturbance: d = [P_avail, T_amb]
    Output:      y = [T, V_cell, n_dot_H2, SEC, I_actual]

    The current I is determined by the power equality V_cell(T,I)·I = P_avail
    via root-finding (Brent's method). This is the algebraic constraint.
    """

    def __init__(
        self,
        ep: ElectrolyzerParams | None = None,
        tp: ThermalParams | None = None,
    ):
        self.ep = ep or ElectrolyzerParams()
        self.tp = tp or ThermalParams()

        self.T: float = self.tp.T_ref_k
        self.I_min = self.ep.I_min_a
        self.I_max = self.ep.I_max_a

    def reset(self, T0: float | None = None) -> None:
        """Reset plant state."""
        self.T = T0 if T0 is not None else self.tp.T_ref_k

    def get_state(self) -> np.ndarray:
        """Return current state vector [T]."""
        return np.array([self.T])

    def _solve_current_for_power(self, T: float, P_avail: float) -> float:
        """Solve V_cell(T, I)·I = P_avail for I via Brent's method.

        Returns I clamped to [I_min, I_max]. If P_avail is below the
        minimum feasible power, returns I_min.
        """
        P_min = v_cell(T, self.I_min, self.ep) * self.I_min
        P_max = v_cell(T, self.I_max, self.ep) * self.I_max

        if P_avail <= P_min:
            return self.I_min
        if P_avail >= P_max:
            return self.I_max

        def residual(I: float) -> float:
            return v_cell(T, I, self.ep) * I - P_avail

        return brentq(residual, self.I_min, self.I_max, xtol=1e-6)

    def _dTdt(self, T: float, I: float, Q_cool: float, T_amb: float) -> float:
        """Temperature rate of change [K/s]."""
        P_el = v_cell(T, I, self.ep) * I
        n_h2 = n_dot_h2(I, self.ep)
        Q_loss = self.tp.UA_loss_wk * (T - T_amb)
        return (P_el - n_h2 * self.tp.DH_rxn_jmol - Q_loss - Q_cool) / self.tp.C_th_jk

    def step(self, u: np.ndarray, d: np.ndarray, dt: float) -> np.ndarray:
        """Advance plant one time step (Euler integration).

        Args:
            u: [Q_cool]          control input [W]
            d: [P_avail, T_amb]  disturbances [W, K]
            dt: time step [s]

        Returns:
            y: [T, V_cell, n_dot_H2, SEC, I_actual]
        """
        Q_cool = float(np.clip(u[0], 0.0, self.tp.Q_cool_max_w))
        P_avail = float(d[0])
        T_amb = float(d[1])

        # Solve algebraic constraint for current
        I = self._solve_current_for_power(self.T, P_avail)

        # Compute outputs before state update
        V = v_cell(self.T, I, self.ep)
        P_el = V * I
        n_h2 = n_dot_h2(I, self.ep)
        sec = P_el / n_h2 if n_h2 > 1e-12 else 0.0

        # Euler integration of temperature
        self.T += self._dTdt(self.T, I, Q_cool, T_amb) * dt

        return np.array([self.T, V, n_h2, sec, I])
