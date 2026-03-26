"""Validation test suite for v0_pid PID controller.

Run:  cd v0_pid && python stress_test.py
"""

from __future__ import annotations

import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from config.parameters import PIDControlParams, ThermalParams, ElectrolyzerParams
from controller.pid import PIDController
from models.ael_model import AELPlant
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

def test_pid_step_response():
    """Fixed positive error -> Q_cool should increase as integral accumulates."""
    ctrl = PIDController()
    ctrl.reset()

    T_above = ThermalParams().T_ref_k + 5.0  # 5 K above setpoint
    dt = 30.0
    x = np.array([T_above])
    d = np.array([600.0, 293.15])

    outputs = []
    for _ in range(10):
        u = ctrl.step(x, d, dt)
        outputs.append(u[0])

    # With constant positive error, integral grows -> Q_cool should increase
    assert outputs[-1] > outputs[0], (
        f"Q_cool should increase over time with constant error: "
        f"first={outputs[0]:.2f}, last={outputs[-1]:.2f}"
    )
    # All outputs should be positive (cooling when hot)
    assert all(q >= 0 for q in outputs), "Q_cool should be non-negative"


def test_pid_derivative_response():
    """Linearly rising temperature -> D term should contribute to cooling."""
    pid_params = PIDControlParams(K_p_T=0.0, K_i_T=0.0, K_d_T=50.0, tau_d_s=5.0)
    ctrl = PIDController(pid=pid_params)
    ctrl.reset()

    T_ref = ThermalParams().T_ref_k
    dt = 30.0
    d = np.array([600.0, 293.15])

    # Feed linearly rising temperature
    outputs = []
    for k in range(20):
        T = T_ref + k * 0.5  # 0.5 K/step = 0.0167 K/s
        x = np.array([T])
        u = ctrl.step(x, d, dt)
        outputs.append(u[0])

    # After a few steps, D-only controller should produce positive output
    # (first step has no derivative, so skip it)
    assert outputs[5] > 0, (
        f"D-only controller should produce positive Q_cool for rising T: {outputs[5]:.4f}"
    )


def test_pid_anti_windup():
    """Large error for many steps -> integral should be bounded."""
    ctrl = PIDController()
    ctrl.reset()

    T_very_hot = ThermalParams().T_ref_k + 50.0  # way above setpoint
    dt = 30.0
    x = np.array([T_very_hot])
    d = np.array([600.0, 293.15])

    for _ in range(1000):
        ctrl.step(x, d, dt)

    pid = PIDControlParams()
    max_integral = ThermalParams().Q_cool_max_w / pid.K_i_T
    assert abs(ctrl._integral_error) <= max_integral + 1e-6, (
        f"Integral error {ctrl._integral_error:.1f} exceeds max {max_integral:.1f}"
    )


def test_pid_reset():
    """After reset(), all internal states should be zeroed."""
    ctrl = PIDController()
    x = np.array([ThermalParams().T_ref_k + 10.0])
    d = np.array([600.0, 293.15])

    # Accumulate state
    for _ in range(10):
        ctrl.step(x, d, 30.0)

    ctrl.reset()
    assert ctrl._integral_error == 0.0, "Integral should be 0 after reset"
    assert ctrl._T_prev is None, "_T_prev should be None after reset"
    assert ctrl._d_filtered == 0.0, "_d_filtered should be 0 after reset"


def test_pid_degrades_to_pi():
    """With K_d=0, PID should produce the same output as a pure PI."""
    pid_params = PIDControlParams(K_d_T=0.0, tau_d_s=1.0)
    ctrl_pid = PIDController(pid=pid_params)
    ctrl_pid.reset()

    # Manually compute PI output for comparison
    tp = ThermalParams()
    K_p = pid_params.K_p_T
    K_i = pid_params.K_i_T
    T_ref = tp.T_ref_k
    Q_max = tp.Q_cool_max_w

    dt = 30.0
    d = np.array([600.0, 293.15])
    integral = 0.0

    temperatures = [T_ref + 3.0, T_ref + 5.0, T_ref + 2.0, T_ref - 1.0, T_ref + 8.0]

    for T in temperatures:
        x = np.array([T])
        u_pid = ctrl_pid.step(x, d, dt)[0]

        # Manual PI
        error = T - T_ref
        integral += error * dt
        max_int = Q_max / max(abs(K_i), 1e-6)
        integral = np.clip(integral, -max_int, max_int)
        u_pi = np.clip(K_p * error + K_i * integral, 0.0, Q_max)

        assert abs(u_pid - u_pi) < 1e-10, (
            f"PID(K_d=0) != PI at T={T:.1f}: PID={u_pid:.6f}, PI={u_pi:.6f}"
        )


def test_closed_loop_power_step():
    """Full power_step scenario: temperature must stay within safe bounds."""
    tp = ThermalParams()
    plant = AELPlant()
    ctrl = PIDController()
    runner = SimulationRunner(plant, ctrl)

    scenario = SCENARIOS["power_step"]()
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
    print(f"  v0_pid Stress Test Suite")
    print(f"{'=' * 60}")

    tests = [
        ("PID step response", test_pid_step_response),
        ("PID derivative response", test_pid_derivative_response),
        ("PID anti-windup", test_pid_anti_windup),
        ("PID reset", test_pid_reset),
        ("PID degrades to PI (K_d=0)", test_pid_degrades_to_pi),
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
