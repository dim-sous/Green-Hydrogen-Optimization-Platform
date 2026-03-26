"""Validation test suite for v1_baseline NMPC + EKF.

Run:  cd v1_baseline && python stress_test.py
"""

from __future__ import annotations

import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from config.parameters import (
    ElectrolyzerParams, ThermalParams, NMPCParams, EKFParams, SimulationParams,
)
from models.ael_model import (
    AELPlant, build_casadi_dynamics, build_casadi_integrator,
    v_cell, n_dot_h2,
)
from controller.nmpc import TrackingNMPC
from estimation.ekf import ExtendedKalmanFilter
from simulation.simulator import SimulationRunner
from simulation.scenarios import SCENARIOS


def _run_test(name: str, func) -> bool:
    """Run a single test and print pass/fail."""
    try:
        func()
        print(f"  PASS  {name}")
        return True
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
        return False
    except Exception as e:
        print(f"  ERROR {name}: {e}")
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_casadi_numpy_consistency():
    """CasADi DAE integrator matches NumPy plant over 1h open-loop."""
    ep = ElectrolyzerParams()
    tp = ThermalParams()
    dt = 30.0

    model = build_casadi_dynamics(ep, tp)
    F = build_casadi_integrator(model, dt)

    plant = AELPlant(ep, tp)
    plant.reset(T0=tp.T_ref_k)

    x_ca = np.array([tp.T_ref_k])
    z_ca = np.array([300.0])

    for _ in range(120):  # 1 hour
        p_vec = np.array([100.0, 600.0, tp.T_amb_k])
        result = F(x_ca, z_ca, p_vec)
        x_ca = np.array(result[0]).flatten()
        z_ca = np.array(result[1]).flatten()
        plant.step(np.array([100.0]), np.array([600.0, tp.T_amb_k]), dt)

    rel_err = abs(x_ca[0] - plant.T) / abs(plant.T)
    assert rel_err < 0.01, f"CasADi-NumPy relative error {rel_err:.2e} exceeds 1%"


def test_plant_shutdown():
    """Plant shuts down (I=0, P_el=0) when P_avail < P_min."""
    plant = AELPlant()
    plant.reset(T0=353.15)
    y = plant.step(np.array([0.0]), np.array([0.0, 293.15]), 30.0)
    assert y[4] == 0.0, f"I should be 0 at shutdown, got {y[4]}"
    assert y[1] == 0.0, f"V_cell should be 0 at shutdown, got {y[1]}"
    assert y[2] == 0.0, f"n_dot_H2 should be 0 at shutdown, got {y[2]}"


def test_power_equality():
    """V_cell(T,I)*I = P_avail for operating points."""
    plant = AELPlant()
    for P in [100, 400, 600, 800]:
        plant.reset(T0=353.15)
        y = plant.step(np.array([100.0]), np.array([float(P), 293.15]), 0.001)
        P_el = y[1] * y[4]
        assert abs(P_el - P) / P < 1e-4, (
            f"Power equality violated at P={P}: P_el={P_el:.2f}"
        )


def test_nmpc_solves():
    """NMPC produces a valid Q_cool on a single step."""
    ep = ElectrolyzerParams()
    tp = ThermalParams()
    nmpc = TrackingNMPC(NMPCParams(), ep, tp)
    nmpc.reset()

    x_hat = np.array([tp.T_ref_k])
    d_forecast = np.full(NMPCParams().N, 600.0)
    u = nmpc.step(x_hat, d_forecast, 30.0)

    assert u.shape == (1,), f"Expected shape (1,), got {u.shape}"
    assert 0.0 <= u[0] <= tp.Q_cool_max_w, (
        f"Q_cool={u[0]:.1f} outside [0, {tp.Q_cool_max_w}]"
    )
    assert nmpc.last_solve_success, "NMPC solve failed"


def test_ekf_predict_correct():
    """EKF predict-correct cycle returns reasonable estimate."""
    ekf = ExtendedKalmanFilter(dt=30.0)
    ekf.reset(353.15)

    u = np.array([100.0])
    d = np.array([600.0, 293.15])
    T_meas = 353.5  # slightly noisy measurement

    x_hat = ekf.step(u, d, T_meas)
    assert x_hat.shape == (1,), f"Expected shape (1,), got {x_hat.shape}"
    # Should be between prediction and measurement
    assert 350.0 < x_hat[0] < 360.0, f"EKF estimate {x_hat[0]:.2f} out of range"


def test_closed_loop_power_step():
    """Full NMPC+EKF closed-loop on power_step: T within bounds."""
    ep = ElectrolyzerParams()
    tp = ThermalParams()

    plant = AELPlant(ep, tp)
    nmpc = TrackingNMPC(NMPCParams(), ep, tp)
    ekf = ExtendedKalmanFilter(EKFParams(), ep, tp, dt=SimulationParams().dt_s)

    scenario = SCENARIOS["power_step"]()
    power_profile = scenario["power_profile"]

    sim_dt = scenario["dt"]
    nmpc_dt = NMPCParams().dt_s
    stride = max(1, int(nmpc_dt / sim_dt))

    def forecast_func(step_idx: int, N: int) -> np.ndarray:
        forecasts = np.zeros(N)
        for k in range(N):
            idx = min(step_idx + (k + 1) * stride, len(power_profile) - 1)
            forecasts[k] = power_profile[idx]
        return forecasts

    from main import NMPCWithEKF

    ctrl = NMPCWithEKF(nmpc, ekf, power_forecast_func=forecast_func)
    runner = SimulationRunner(plant, ctrl)
    results = runner.run(scenario)

    T = results["y"][:, 0]
    T_max = float(np.max(T))
    T_min = float(np.min(T))

    assert T_max <= tp.T_max_k, (
        f"T_max = {T_max:.2f} K exceeds safe limit {tp.T_max_k:.2f} K"
    )
    assert T_min >= tp.T_min_k, (
        f"T_min = {T_min:.2f} K below safe limit {tp.T_min_k:.2f} K"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"{'=' * 60}")
    print(f"  v1_baseline Stress Test Suite")
    print(f"{'=' * 60}")

    tests = [
        ("CasADi-NumPy consistency", test_casadi_numpy_consistency),
        ("Plant shutdown (P_avail=0)", test_plant_shutdown),
        ("Power equality (Brent)", test_power_equality),
        ("NMPC single solve", test_nmpc_solves),
        ("EKF predict-correct", test_ekf_predict_correct),
        ("Closed-loop power_step", test_closed_loop_power_step),
    ]

    results = [_run_test(name, func) for name, func in tests]
    passed = sum(results)
    total = len(results)

    print(f"\n  {passed}/{total} tests passed")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
