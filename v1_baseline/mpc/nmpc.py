"""Multiple-shooting NMPC for single-cell alkaline electrolyzer.

Objective: maximize H₂ production while maintaining thermal safety.

The DAE (V_cell·I = P_avail) is handled by making I an explicit NLP
decision variable with the power equality as a constraint. The ODE
for temperature is integrated via explicit RK4 (fast, no implicit solver).

Decision variables per shooting node:
    T_k       differential state [K]
    I_k       algebraic variable (constrained by power equality) [A]
    Q_cool_k  control input [W]

The NMPC anticipates future power changes (P_avail forecast) and
pre-cools before power ramps to prevent thermal spikes.
"""

from __future__ import annotations

import time
import numpy as np
import casadi as ca
from scipy.optimize import brentq

from config.parameters import (
    ElectrolyzerParams,
    ThermalParams,
    NMPCParams,
)
from models.ael_model import v_cell as np_v_cell


class TrackingNMPC:
    """Multiple-shooting NMPC with explicit RK4 integrator.

    The NLP is built once at construction. At each call to step(),
    the initial state and P_avail forecast are updated as parameters,
    and IPOPT resolves the optimal Q_cool trajectory.
    """

    def __init__(
        self,
        nmpc_params: NMPCParams | None = None,
        ely_params: ElectrolyzerParams | None = None,
        th_params: ThermalParams | None = None,
    ):
        self.nmpc = nmpc_params or NMPCParams()
        self.ep = ely_params or ElectrolyzerParams()
        self.tp = th_params or ThermalParams()

        N = self.nmpc.N
        dt = self.nmpc.dt_s
        ep = self.ep
        tp = self.tp

        # --- Build symbolic ODE + algebraic functions ---
        T_sym = ca.MX.sym("T")
        I_sym = ca.MX.sym("I")
        Q_sym = ca.MX.sym("Qc")
        P_sym = ca.MX.sym("Pa")
        Ta_sym = ca.MX.sym("Ta")

        # Electrochemistry
        j = I_sym / ep.A_m2
        V_rev = 1.229 - 8.5e-4 * (T_sym - 298.15)
        V_ohm = (ep.r1_ohm_m2 + ep.r2_ohm_m2_k * T_sym) / ep.A_m2 * I_sym
        V_act = ep.s_v * ca.log10((ep.t1 + ep.t2 / T_sym + ep.t3 / T_sym**2) * j + 1.0)
        V_cell_sym = V_rev + V_ohm + V_act
        P_el_sym = V_cell_sym * I_sym

        # Faradaic efficiency and H₂ production
        eps = 1e-6
        eta_F = (j**2) / (ep.f1 + j**2 + eps) * ep.f2
        n_dot_H2_sym = eta_F * ep.N_cells * I_sym / (2.0 * ep.F_const)

        # Thermal ODE: dT/dt
        Q_loss = tp.UA_loss_wk * (T_sym - Ta_sym)
        dTdt = (P_el_sym - n_dot_H2_sym * tp.DH_rxn_jmol - Q_loss - Q_sym) / tp.C_th_jk

        # Power equality residual
        power_residual = V_cell_sym * I_sym - P_sym

        # CasADi functions
        self._f_dTdt = ca.Function("dTdt", [T_sym, I_sym, Q_sym, Ta_sym], [dTdt])
        self._f_nH2 = ca.Function("nH2", [T_sym, I_sym], [n_dot_H2_sym])
        self._f_power = ca.Function("pwr", [T_sym, I_sym, P_sym], [power_residual])

        # RK4 integrator as a CasADi function
        k1 = self._f_dTdt(T_sym, I_sym, Q_sym, Ta_sym)
        k2 = self._f_dTdt(T_sym + dt/2 * k1, I_sym, Q_sym, Ta_sym)
        k3 = self._f_dTdt(T_sym + dt/2 * k2, I_sym, Q_sym, Ta_sym)
        k4 = self._f_dTdt(T_sym + dt * k3, I_sym, Q_sym, Ta_sym)
        T_next = T_sym + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        self._F_rk4 = ca.Function("F_rk4",
                                   [T_sym, I_sym, Q_sym, Ta_sym], [T_next])

        # --- Build NLP ---
        w = []        # decision variables
        w0 = []       # initial guess
        lbw = []
        ubw = []
        g = []        # constraints
        lbg = []
        ubg = []
        J = 0.0

        # NLP parameters: [T_init, P_avail_0..N-1, T_amb, u_prev]
        n_par = 1 + N + 1 + 1
        P = ca.MX.sym("P", n_par)

        T_init = P[0]
        P_avail = P[1:1 + N]
        T_amb = P[1 + N]
        u_prev_par = P[1 + N + 1]

        T_prev = T_init
        Q_cool_prev = u_prev_par

        for k in range(N):
            # Decision: I_k (algebraic)
            Ik = ca.MX.sym(f"I_{k}")
            w.append(Ik)
            w0.append(300.0)
            lbw.append(float(ep.I_min_a))
            ubw.append(float(ep.I_max_a))

            # Decision: Q_cool_k (control)
            Qk = ca.MX.sym(f"Q_{k}")
            w.append(Qk)
            w0.append(0.0)
            lbw.append(0.0)
            ubw.append(float(tp.Q_cool_max_w))

            # Power equality constraint: V_cell(T,I)*I = P_avail
            g.append(self._f_power(T_prev, Ik, P_avail[k]))
            lbg.append(0.0)
            ubg.append(0.0)

            # H₂ production objective (maximize → negate)
            J -= self.nmpc.w_H2 * self._f_nH2(T_prev, Ik)

            # Cooling rate-of-change penalty
            J += self.nmpc.R_dQ * (Qk - Q_cool_prev)**2

            # RK4 integration
            T_next = self._F_rk4(T_prev, Ik, Qk, T_amb)

            # Decision: T_{k+1} (state at next node)
            Tk = ca.MX.sym(f"T_{k+1}")
            w.append(Tk)
            w0.append(float(tp.T_ref_k))
            lbw.append(float(tp.T_min_k))
            ubw.append(float(tp.T_max_k))

            # Soft thermal bound penalties (Christensen eq. 40):
            #   w_soft * max(0, T - T_max)^2  +  w_soft * max(0, T_min - T)^2
            J += self.nmpc.w_T_soft * ca.fmax(0, Tk - tp.T_max_k)**2
            J += self.nmpc.w_T_soft * ca.fmax(0, tp.T_min_k - Tk)**2

            # Shooting continuity: T_{k+1} = F(T_k, I_k, Q_k)
            g.append(T_next - Tk)
            lbg.append(0.0)
            ubg.append(0.0)

            T_prev = Tk
            Q_cool_prev = Qk

        # Build NLP
        w_flat = ca.vertcat(*w)
        g_flat = ca.vertcat(*g)

        nlp = {"f": J, "x": w_flat, "g": g_flat, "p": P}

        opts = {
            "ipopt.max_iter": self.nmpc.solver_max_iter,
            "ipopt.print_level": self.nmpc.solver_print_level,
            "ipopt.sb": "yes",
            "ipopt.warm_start_init_point": "yes",
            "print_time": 0,
        }
        self._solver = ca.nlpsol("nmpc", "ipopt", nlp, opts)

        self._N = N
        self._n_par = n_par
        self._w0 = np.array(w0)
        self._lbw = np.array(lbw)
        self._ubw = np.array(ubw)
        self._lbg = np.array(lbg)
        self._ubg = np.array(ubg)

        # Warm-start storage
        self._w_prev: np.ndarray | None = None
        self._lam_g_prev: np.ndarray | None = None
        self._lam_x_prev: np.ndarray | None = None
        self._u_prev: float = 0.0

        self.last_solve_time: float = 0.0
        self.last_solve_success: bool = True

    def reset(self) -> None:
        """Reset warm-start state."""
        self._w_prev = None
        self._lam_g_prev = None
        self._lam_x_prev = None
        self._u_prev = 0.0

    def step(
        self,
        x_hat: np.ndarray,
        d_forecast: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Solve the NMPC problem and return the first control action.

        Args:
            x_hat: current state estimate [T_hat] (1,)
            d_forecast: P_avail forecast, shape (N,) or (N,2)
            dt: time step [s] (unused, horizon dt fixed at construction)

        Returns:
            u: control action [Q_cool] (1,)
        """
        N = self._N
        T_hat = float(x_hat[0])

        # Parse forecast
        if d_forecast.ndim == 1:
            P_forecast = d_forecast[:N] if len(d_forecast) >= N else \
                np.pad(d_forecast, (0, N - len(d_forecast)), mode="edge")
            T_amb = self.tp.T_amb_k
        else:
            P_forecast = d_forecast[:N, 0]
            T_amb = float(d_forecast[0, 1]) if d_forecast.shape[1] > 1 else self.tp.T_amb_k
            if len(P_forecast) < N:
                P_forecast = np.pad(P_forecast, (0, N - len(P_forecast)), mode="edge")

        # Build parameter vector
        par = np.zeros(self._n_par)
        par[0] = T_hat
        par[1:1 + N] = P_forecast
        par[1 + N] = T_amb
        par[1 + N + 1] = self._u_prev

        # Initial guess (warm-start or default with I from power equality)
        if self._w_prev is not None:
            w0 = self._w_prev
        else:
            w0 = self._w0.copy()
            # Better initial I guesses from power equality
            T_g = T_hat
            for k in range(N):
                stride = 3  # [I_k, Q_k, T_{k+1}]
                try:
                    P_k = float(P_forecast[k])
                    P_min = np_v_cell(T_g, self.ep.I_min_a, self.ep) * self.ep.I_min_a
                    P_max = np_v_cell(T_g, self.ep.I_max_a, self.ep) * self.ep.I_max_a
                    if P_min < P_k < P_max:
                        I_g = brentq(
                            lambda I: np_v_cell(T_g, I, self.ep) * I - P_k,
                            self.ep.I_min_a, self.ep.I_max_a, xtol=1e-4,
                        )
                        w0[k * stride] = I_g
                except Exception:
                    pass

        # Solve
        t0 = time.perf_counter()
        try:
            sol = self._solver(
                x0=w0, p=par,
                lbx=self._lbw, ubx=self._ubw,
                lbg=self._lbg, ubg=self._ubg,
                lam_g0=self._lam_g_prev if self._lam_g_prev is not None else 0,
                lam_x0=self._lam_x_prev if self._lam_x_prev is not None else 0,
            )
            self.last_solve_success = True
        except Exception:
            self.last_solve_success = False
            self.last_solve_time = time.perf_counter() - t0
            return np.array([self._u_prev])

        self.last_solve_time = time.perf_counter() - t0

        # Extract solution
        w_opt = np.array(sol["x"]).flatten()
        self._lam_g_prev = np.array(sol["lam_g"]).flatten()
        self._lam_x_prev = np.array(sol["lam_x"]).flatten()

        # First Q_cool: layout is [I_0, Q_0, T_1, I_1, Q_1, T_2, ...]
        Q_cool_opt = float(w_opt[1])  # Q_0 is at index 1

        # Warm-start: shift solution forward by one node
        stride = 3
        w_shifted = w_opt.copy()
        for k in range(N - 1):
            w_shifted[k * stride:(k + 1) * stride] = \
                w_opt[(k + 1) * stride:(k + 2) * stride]
        self._w_prev = w_shifted

        self._u_prev = Q_cool_opt

        return np.array([Q_cool_opt])
