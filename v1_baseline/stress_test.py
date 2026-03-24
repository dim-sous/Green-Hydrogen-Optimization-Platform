"""Stress tests for v1_baseline — validates all sub-models and closed-loop."""

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from models.electrolyzer import Electrolyzer
from models.thermal import ThermalModel
from models.gas_separator import GasSeparator
from models.ael_plant import AELPlant
from models.casadi_model import build_ael_model, build_integrator
from mpc.baseline_pi import PIController
from simulation.simulator import SimulationRunner
from simulation.scenarios import steady_wind, ramp_up_down


def test_electrolyzer_voltage_range():
    """V_cell at typical conditions should be 1.5-2.5 V."""
    ely = Electrolyzer()
    for T in (323.15, 353.15, 373.15):
        for I in (50.0, 200.0, 500.0):
            V = ely.V_cell(T, I)
            assert 1.2 < V < 3.0, f"V_cell={V} at T={T}, I={I}"
    print("  [PASS] electrolyzer voltage range")


def test_faradaic_efficiency():
    """eta_F in (0, 1) and increases with current."""
    ely = Electrolyzer()
    prev = 0.0
    for I in (50.0, 100.0, 200.0, 300.0, 500.0):
        eta = ely.eta_F(I)
        assert 0 < eta <= 1.0, f"eta_F={eta} at I={I}"
        assert eta >= prev, f"eta_F not monotonic at I={I}"
        prev = eta
    print("  [PASS] Faradaic efficiency")


def test_energy_balance():
    """P_el == V_stack * I."""
    ely = Electrolyzer()
    V, P, n, eta, sec = ely.step(300.0, 353.15, 30.0)
    assert abs(P - V * 300.0) < 1e-6
    print("  [PASS] energy balance")


def test_thermal_heating():
    """Temperature should rise without cooling."""
    th = ThermalModel()
    T = 293.15
    for _ in range(100):
        T = th.step(T, 50000.0, 0.05, 0.0, 30.0)
    assert T > 293.15 + 10, f"Expected significant heating, got T={T}"
    print("  [PASS] thermal heating")


def test_gas_separator_bounded():
    """x_HTO should remain bounded and positive."""
    gs = GasSeparator()
    x = 0.005
    for _ in range(1000):
        x = gs.step(x, 0.09, 30.0)
    assert 0 <= x <= 1.0, f"x_HTO out of bounds: {x}"
    print("  [PASS] gas separator bounded")


def test_casadi_numpy_consistency():
    """CasADi model matches NumPy at a single operating point."""
    model = build_ael_model()
    ely = Electrolyzer()

    T, I = 353.15, 300.0
    x = np.array([T, 0.005])
    u = np.array([I, 5000.0])
    p = np.array([100000.0, 293.15])

    y_ca = np.array(model["h"](x, u, p)).flatten()
    V_np = ely.V_stack(T, I)
    n_np = ely.n_dot_H2(T, I)

    rel_V = abs(y_ca[2] - V_np) / abs(V_np)
    rel_n = abs(y_ca[3] - n_np) / abs(n_np)
    assert rel_V < 0.001, f"V_stack rel error: {rel_V}"
    assert rel_n < 0.001, f"n_dot_H2 rel error: {rel_n}"
    print("  [PASS] CasADi-NumPy consistency (single point)")


def test_casadi_trajectory(method: str = "collocation"):
    """CasADi integrator matches NumPy plant over 1h open-loop."""
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
    assert rel_T < 0.001, f"{method} T rel error: {rel_T}"
    assert rel_HTO < 0.05, f"{method} x_HTO rel error: {rel_HTO}"
    print(f"  [PASS] CasADi trajectory ({method}): T err={rel_T:.2e}, HTO err={rel_HTO:.2e}")


def test_pi_steady_wind():
    """PI keeps T within bounds and no HTO violations on steady_wind."""
    plant = AELPlant()
    ctrl = PIController()
    runner = SimulationRunner(plant, ctrl)
    results = runner.run(steady_wind())

    T = results["y"][:, 0]
    x_HTO = results["y"][:, 1]
    assert np.all(T > 323.15) and np.all(T < 373.15), \
        f"T out of bounds: [{T.min():.1f}, {T.max():.1f}]"
    assert np.all(x_HTO < 0.02), f"HTO violation: max={x_HTO.max():.4f}"
    print("  [PASS] PI steady_wind: T and HTO within bounds")


def test_pi_ramp_up_down():
    """PI keeps T within bounds and no HTO violations on ramp_up_down."""
    plant = AELPlant()
    ctrl = PIController()
    runner = SimulationRunner(plant, ctrl)
    results = runner.run(ramp_up_down())

    T = results["y"][:, 0]
    x_HTO = results["y"][:, 1]
    assert np.all(T > 323.15) and np.all(T < 373.15), \
        f"T out of bounds: [{T.min():.1f}, {T.max():.1f}]"
    assert np.all(x_HTO < 0.02), f"HTO violation: max={x_HTO.max():.4f}"
    print("  [PASS] PI ramp_up_down: T and HTO within bounds")


def main():
    print("=" * 50)
    print("  v1_baseline stress tests")
    print("=" * 50)

    tests = [
        test_electrolyzer_voltage_range,
        test_faradaic_efficiency,
        test_energy_balance,
        test_thermal_heating,
        test_gas_separator_bounded,
        test_casadi_numpy_consistency,
        lambda: test_casadi_trajectory("collocation"),
        lambda: test_casadi_trajectory("rk4"),
        test_pi_steady_wind,
        test_pi_ramp_up_down,
    ]

    passed, failed = 0, 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
