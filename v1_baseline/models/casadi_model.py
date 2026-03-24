"""CasADi symbolic AEL model and integrator for NMPC prediction."""

import casadi as ca
from config.parameters import (
    ElectrolyzerParams,
    ThermalParams,
    GasSeparatorParams,
)


def build_ael_model(
    ely: ElectrolyzerParams | None = None,
    th: ThermalParams | None = None,
    gs: GasSeparatorParams | None = None,
) -> dict:
    """Build CasADi symbolic AEL model.

    Returns dict with keys:
        x, u, p, xdot, y, V_cell, P_el, f, h
    """
    ely = ely or ElectrolyzerParams()
    th = th or ThermalParams()
    gs = gs or GasSeparatorParams()

    # Symbolic variables (MX for efficiency, matching Energy Storage repo)
    x = ca.MX.sym("x", 2)      # [T, x_HTO]
    u = ca.MX.sym("u", 2)      # [I, Q_cool]
    par = ca.MX.sym("p", 2)    # [P_avail, T_amb]

    T = x[0]
    x_HTO = x[1]
    I = u[0]
    Q_cool = u[1]
    P_avail = par[0]
    T_amb = par[1]

    # Current density
    j = I / ely.A

    # Electrochemistry
    V_rev = 1.229 - 8.5e-4 * (T - 298.15)
    V_ohm = (ely.r1 + ely.r2 * T) / ely.A * I
    V_act = ely.s * ca.log10((ely.t1 + ely.t2 / T + ely.t3 / T**2) * j + 1.0)
    V_cell = V_rev + V_ohm + V_act
    V_stack = ely.N_cells * V_cell
    P_el = V_stack * I

    # Faradaic efficiency (epsilon avoids division by zero at j=0)
    eps = 1e-6
    eta_F = (j**2) / (ely.f1 + j**2 + eps) * ely.f2

    # H2 production rate
    n_dot_H2 = eta_F * ely.N_cells * I / (2.0 * ely.F_const)

    # SEC (smooth)
    SEC = P_el / (n_dot_H2 + eps)

    # Thermal dynamics
    Q_loss = th.UA_loss * (T - T_amb)
    dTdt = (P_el - n_dot_H2 * th.DH_rxn - Q_loss - Q_cool) / th.C_th

    # Gas separator dynamics — use molar holdup for correct units
    dp = 0.5 * gs.p_op
    n_total = gs.n_total  # mol
    dx_HTO_dt = (gs.K_perm * dp - x_HTO * n_dot_H2) / n_total

    # Assemble
    xdot = ca.vertcat(dTdt, dx_HTO_dt)
    y = ca.vertcat(T, x_HTO, V_stack, n_dot_H2, SEC)

    f_func = ca.Function("f", [x, u, par], [xdot], ["x", "u", "p"], ["xdot"])
    h_func = ca.Function("h", [x, u, par], [y], ["x", "u", "p"], ["y"])

    return {
        "x": x, "u": u, "p": par,
        "xdot": xdot, "y": y,
        "V_cell": V_cell, "P_el": P_el,
        "f": f_func, "h": h_func,
    }


def build_integrator(model: dict, dt: float, method: str = "collocation") -> ca.Function:
    """Wrap the ODE in a CasADi integrator.

    Args:
        model: output of build_ael_model()
        dt: integration step (s)
        method: 'collocation' or 'rk4'
    Returns:
        F: CasADi Function  F(x0, u, p) -> x_next
    """
    if method == "collocation":
        x = model["x"]
        u = model["u"]
        p = model["p"]
        xdot = model["xdot"]

        dae = {"x": x, "p": ca.vertcat(u, p), "ode": xdot}
        opts = {
            "number_of_finite_elements": 1,
            "interpolation_order": 3,
        }
        # New CasADi API: pass t0, tf as positional args after dae
        intg = ca.integrator("F_col", "collocation", dae, 0.0, dt, opts)

        x0 = ca.MX.sym("x0", 2)
        u_in = ca.MX.sym("u_in", 2)
        p_in = ca.MX.sym("p_in", 2)
        result = intg(x0=x0, p=ca.vertcat(u_in, p_in))
        x_next = result["xf"]

        return ca.Function("F", [x0, u_in, p_in], [x_next], ["x0", "u", "p"], ["x_next"])

    elif method == "rk4":
        f = model["f"]

        x0 = ca.MX.sym("x0", 2)
        u_in = ca.MX.sym("u_in", 2)
        p_in = ca.MX.sym("p_in", 2)

        h = dt
        k1 = f(x0, u_in, p_in)
        k2 = f(x0 + h / 2 * k1, u_in, p_in)
        k3 = f(x0 + h / 2 * k2, u_in, p_in)
        k4 = f(x0 + h * k3, u_in, p_in)
        x_next = x0 + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return ca.Function("F_rk4", [x0, u_in, p_in], [x_next], ["x0", "u", "p"], ["x_next"])

    else:
        raise ValueError(f"Unknown method: {method}. Use 'collocation' or 'rk4'.")
