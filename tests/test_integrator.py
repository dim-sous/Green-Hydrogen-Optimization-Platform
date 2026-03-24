"""Tests for CasADi integrator methods."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "v1_baseline"))

import numpy as np
import pytest
import casadi as ca
from models.casadi_model import build_ael_model, build_integrator


class TestIntegratorConsistency:
    def test_collocation_rk4_agreement(self):
        """Both methods should agree within 1% for a single step."""
        model = build_ael_model()
        dt = 30.0

        F_col = build_integrator(model, dt, "collocation")
        F_rk4 = build_integrator(model, dt, "rk4")

        x0 = np.array([353.15, 0.005])
        u = np.array([300.0, 5000.0])
        p = np.array([100000.0, 293.15])

        x_col = np.array(F_col(x0, u, p)).flatten()
        x_rk4 = np.array(F_rk4(x0, u, p)).flatten()

        for i in range(2):
            if abs(x_col[i]) > 1e-10:
                rel = abs(x_col[i] - x_rk4[i]) / abs(x_col[i])
                assert rel < 0.01, f"State {i}: col={x_col[i]}, rk4={x_rk4[i]}, rel={rel}"

    def test_stiff_ode(self):
        """Collocation handles stiff ODE: dy/dt = -1000*y, y(0)=1."""
        y = ca.SX.sym("y")
        u_dummy = ca.SX.sym("u", 2)
        p_dummy = ca.SX.sym("p", 2)
        ydot = -1000 * y

        dae = {"x": y, "p": ca.vertcat(u_dummy, p_dummy), "ode": ydot}
        opts = {"number_of_finite_elements": 4, "interpolation_order": 3}
        intg = ca.integrator("stiff_test", "collocation", dae, 0.0, 0.01, opts)

        result = intg(x0=1.0, p=np.zeros(4))
        y_final = float(result["xf"])
        y_exact = np.exp(-10.0)
        assert abs(y_final - y_exact) < 1e-3

    def test_multiple_steps_trajectory(self):
        """Temperature rises without cooling, stays physically reasonable."""
        model = build_ael_model()
        dt = 30.0
        F_int = build_integrator(model, dt, "collocation")

        x = np.array([293.15, 0.001])
        u = np.array([300.0, 0.0])
        p = np.array([100000.0, 293.15])

        temps = [x[0]]
        for _ in range(120):
            x = np.array(F_int(x, u, p)).flatten()
            temps.append(x[0])

        assert temps[-1] > temps[0]
        assert temps[-1] < 500
        assert x[1] >= 0
