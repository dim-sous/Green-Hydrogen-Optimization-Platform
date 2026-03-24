"""Tests for NMPC controller (CasADi Opti + IPOPT validation)."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "v1_baseline"))

import numpy as np
import pytest
import casadi as ca


class TestSimpleThermalNMPC:
    """Validate CasADi Opti + IPOPT on a simple 1-state thermal system."""

    def test_thermal_tracking(self):
        """NMPC tracks T_ref within ±1 K."""
        C = 1000.0
        T_ref = 350.0
        P = 5000.0
        dt = 10.0
        N = 10

        T = ca.SX.sym("T")
        Q = ca.SX.sym("Q")
        T_next = T + dt * (P - Q) / C
        F = ca.Function("F", [T, Q], [T_next])

        T_current = 340.0
        T_log = [T_current]

        for _ in range(200):
            opti = ca.Opti()
            T_var = opti.variable(N + 1)
            Q_var = opti.variable(N)

            opti.subject_to(T_var[0] == T_current)

            cost = 0
            for k in range(N):
                cost += 10.0 * (T_var[k] - T_ref) ** 2 + 1e-8 * Q_var[k] ** 2
                opti.subject_to(T_var[k + 1] == F(T_var[k], Q_var[k]))
                opti.subject_to(Q_var[k] >= 0)
                opti.subject_to(Q_var[k] <= 15000)
            cost += 100.0 * (T_var[N] - T_ref) ** 2

            opti.minimize(cost)
            opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "max_iter": 100})

            sol = opti.solve()
            Q_opt = sol.value(Q_var[0])
            T_current = float(F(T_current, Q_opt))
            T_log.append(T_current)

        assert abs(T_log[-1] - T_ref) < 1.0, f"Final T={T_log[-1]}, T_ref={T_ref}"
