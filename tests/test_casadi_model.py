"""Tests for CasADi symbolic model consistency with NumPy plant."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "v1_baseline"))

import numpy as np
import pytest
from models.electrolyzer import Electrolyzer
from models.thermal import ThermalModel
from models.gas_separator import GasSeparator
from models.casadi_model import build_ael_model, build_integrator


@pytest.fixture
def casadi_model():
    return build_ael_model()


class TestModelConsistency:
    def test_output_at_operating_point(self, casadi_model):
        ely = Electrolyzer()
        T, x_HTO = 353.15, 0.005
        I, Q_cool = 300.0, 5000.0

        x = np.array([T, x_HTO])
        u = np.array([I, Q_cool])
        p = np.array([100000.0, 293.15])

        y_ca = np.array(casadi_model["h"](x, u, p)).flatten()
        V_np = ely.V_stack(T, I)
        n_np = ely.n_dot_H2(T, I)
        P_np = ely.P_el(T, I)
        SEC_np = P_np / n_np

        assert abs(y_ca[2] - V_np) / abs(V_np) < 0.001
        assert abs(y_ca[3] - n_np) / abs(n_np) < 0.001
        assert abs(y_ca[4] - SEC_np) / abs(SEC_np) < 0.001

    def test_derivatives_match(self, casadi_model):
        ely = Electrolyzer()
        th = ThermalModel()
        gs = GasSeparator()

        T, x_HTO = 353.15, 0.005
        I, Q_cool = 300.0, 5000.0
        T_amb = 293.15

        x = np.array([T, x_HTO])
        u = np.array([I, Q_cool])
        p = np.array([100000.0, T_amb])

        xdot_ca = np.array(casadi_model["f"](x, u, p)).flatten()

        P_el = ely.P_el(T, I)
        n_h2 = ely.n_dot_H2(T, I)
        dTdt_np = th.dTdt(T, P_el, n_h2, Q_cool, T_amb)
        dx_np = gs.dx_HTO_dt(x_HTO, n_h2)

        if abs(dTdt_np) > 1e-10:
            assert abs(xdot_ca[0] - dTdt_np) / abs(dTdt_np) < 0.001
        if abs(dx_np) > 1e-10:
            assert abs(xdot_ca[1] - dx_np) / abs(dx_np) < 0.001


class TestOpenLoopTrajectory:
    def _run_open_loop(self, method: str):
        model = build_ael_model()
        dt = 30.0
        F_int = build_integrator(model, dt, method=method)

        ely = Electrolyzer()
        th = ThermalModel()
        gs = GasSeparator()

        T_np, x_HTO_np = 353.15, 0.005
        x_ca = np.array([353.15, 0.005])
        I, Q_cool = 300.0, 5000.0
        u = np.array([I, Q_cool])
        p = np.array([100000.0, 293.15])

        for _ in range(int(3600.0 / dt)):
            P_el = ely.P_el(T_np, I)
            n_h2 = ely.n_dot_H2(T_np, I)
            T_np = th.step(T_np, P_el, n_h2, Q_cool, dt, 293.15)
            x_HTO_np = gs.step(x_HTO_np, n_h2, dt)
            x_ca = np.array(F_int(x_ca, u, p)).flatten()

        rel_T = abs(x_ca[0] - T_np) / abs(T_np)
        rel_HTO = abs(x_ca[1] - x_HTO_np) / max(abs(x_HTO_np), 1e-10)
        return rel_T, rel_HTO

    def test_collocation_matches_numpy(self):
        rel_T, rel_HTO = self._run_open_loop("collocation")
        assert rel_T < 0.001, f"Collocation T rel error: {rel_T}"
        assert rel_HTO < 0.05, f"Collocation x_HTO rel error: {rel_HTO}"

    def test_rk4_matches_numpy(self):
        rel_T, rel_HTO = self._run_open_loop("rk4")
        assert rel_T < 0.001, f"RK4 T rel error: {rel_T}"
        assert rel_HTO < 0.05, f"RK4 x_HTO rel error: {rel_HTO}"


class TestIntegratorBuilds:
    def test_collocation_builds(self):
        model = build_ael_model()
        F = build_integrator(model, 30.0, "collocation")
        assert F is not None

    def test_rk4_builds(self):
        model = build_ael_model()
        F = build_integrator(model, 30.0, "rk4")
        assert F is not None

    def test_invalid_method_raises(self):
        model = build_ael_model()
        with pytest.raises(ValueError):
            build_integrator(model, 30.0, "euler")
