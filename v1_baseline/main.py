"""v1_baseline — Single-cell AEL with NMPC + EKF.

Entry point for running simulations, validation, and saving results.
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys

VERSION_TAG = "v1_baseline"

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from config.parameters import (
    ElectrolyzerParams,
    ThermalParams,
    NMPCParams,
    EKFParams,
    SimulationParams,
)
from models.ael_model import AELPlant, build_casadi_dynamics, build_casadi_integrator
from controller.nmpc import TrackingNMPC
from estimation.ekf import ExtendedKalmanFilter
from simulation.simulator import SimulationRunner
from simulation.scenarios import SCENARIOS
from visualization.plot_results import plot_results

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT.parent / "results" / VERSION_TAG


# ---------------------------------------------------------------------------
# NMPC + EKF controller wrapper
# ---------------------------------------------------------------------------

class NMPCWithEKF:
    """Wraps EKF + NMPC into the step(x, d, dt) -> u interface.

    The EKF filters the temperature measurement, then the NMPC
    computes the optimal cooling trajectory using the P_avail forecast.
    """

    def __init__(
        self,
        nmpc: TrackingNMPC,
        ekf: ExtendedKalmanFilter,
        power_forecast_func=None,
        solve_stride: int = 10,
    ):
        self.nmpc = nmpc
        self.ekf = ekf
        self._power_forecast_func = power_forecast_func
        self._solve_stride = solve_stride  # solve NMPC every N sim steps
        self._u_prev = np.array([0.0])
        self._step_idx = 0

    def reset(self) -> None:
        self.nmpc.reset()
        self.ekf.reset(ThermalParams().T_ref_k)
        self._u_prev = np.array([125.0])
        self._step_idx = 0

    def step(self, x: np.ndarray, d: np.ndarray, dt: float) -> np.ndarray:
        """Compute NMPC control from EKF-filtered state.

        The NMPC NLP is only solved every solve_stride simulation steps
        (matching the NMPC dt). Between solves, the last control is held.
        The EKF runs every step.

        Args:
            x: plant state [T] — used as measurement (1,)
            d: [P_avail, T_amb] disturbances (2,)
            dt: time step [s]

        Returns:
            u: [Q_cool] control (1,)
        """
        T_meas = x[0]

        # EKF: predict + correct every step
        x_hat = self.ekf.step(self._u_prev, d, T_meas)

        # NMPC: solve only every solve_stride steps
        if self._step_idx % self._solve_stride == 0:
            N = self.nmpc._N
            if self._power_forecast_func is not None:
                d_forecast = self._power_forecast_func(self._step_idx, N)
            else:
                d_forecast = np.full(N, d[0])
            u = self.nmpc.step(x_hat, d_forecast, dt)
            self._u_prev = u

        self._step_idx += 1
        return self._u_prev


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: dict) -> dict:
    """Compute performance metrics from simulation results.

    Output format y = [T, V_cell, n_dot_H2, SEC, I_actual]
    """
    tp = ThermalParams()
    y = results["y"]
    u = results["u"]
    d = results["d"]
    dt = results["dt"]

    T       = y[:, 0]
    V_cell  = y[:, 1]
    n_dot   = y[:, 2]
    SEC     = y[:, 3]
    I       = y[:, 4]
    P_el    = V_cell * I
    P_avail = d[:, 0]
    Q_cool  = u[:, 0]

    H2_yield = float(np.sum(n_dot) * dt)
    producing = n_dot > 1e-10
    avg_SEC = float(np.mean(SEC[producing])) if np.any(producing) else 0.0
    avg_SEC_Wh = avg_SEC / 3600.0  # J/mol -> Wh/mol

    T_violations = int(np.sum((T > tp.T_max_k) | (T < tp.T_min_k)))

    safe_P = np.where(P_avail > 1e-6, P_avail, 1.0)
    power_util = float(np.mean(
        np.where(P_avail > 1e-6, P_el / safe_P, 0.0)
    ))

    mean_solve = float(np.mean(results["solve_times"]))
    max_solve = float(np.max(results["solve_times"]))

    return {
        "H2_yield_mol": H2_yield,
        "avg_SEC_Wh_per_mol": avg_SEC_Wh,
        "T_max_K": float(np.max(T)),
        "T_min_K": float(np.min(T)),
        "T_violations": T_violations,
        "power_utilization": power_util,
        "avg_Q_cool_W": float(np.mean(Q_cool)),
        "mean_solve_time_ms": mean_solve * 1000,
        "max_solve_time_ms": max_solve * 1000,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_casadi_dae() -> None:
    """Validate CasADi DAE integrator matches NumPy plant on 1h open-loop."""
    log.info("=== CasADi DAE vs NumPy Plant Validation ===")

    ep = ElectrolyzerParams()
    tp = ThermalParams()
    dt = 30.0

    model = build_casadi_dynamics(ep, tp)
    F = build_casadi_integrator(model, dt)

    plant = AELPlant(ep, tp)
    plant.reset(T0=tp.T_ref_k)

    # Fixed control and disturbance (P_avail within feasible range)
    Q_cool = 100.0
    P_avail = 800.0   # well within [P_min, P_max] at T_ref
    T_amb = tp.T_amb_k

    # CasADi state
    x_ca = np.array([tp.T_ref_k])
    z_ca = np.array([300.0])  # initial algebraic guess

    n_steps = int(3600.0 / dt)
    for _ in range(n_steps):
        # CasADi DAE step
        p_vec = np.array([Q_cool, P_avail, T_amb])
        result = F(x_ca, z_ca, p_vec)
        x_ca = np.array(result[0]).flatten()
        z_ca = np.array(result[1]).flatten()

        # NumPy plant step
        plant.step(np.array([Q_cool]), np.array([P_avail, T_amb]), dt)

    T_np = plant.get_state()[0]
    T_ca = x_ca[0]
    rel_T = abs(T_ca - T_np) / abs(T_np)
    status = "PASS" if rel_T < 0.01 else "FAIL"
    log.info(f"  T after 1h: CasADi={T_ca:.4f} K  NumPy={T_np:.4f} K  "
             f"rel_err={rel_T:.2e}  [{status}]")


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_nmpc_scenario(name: str) -> dict:
    """Run a scenario with NMPC + EKF controller."""
    ep = ElectrolyzerParams()
    tp = ThermalParams()

    plant = AELPlant(ep, tp)
    nmpc = TrackingNMPC(NMPCParams(), ep, tp)
    ekf = ExtendedKalmanFilter(EKFParams(), ep, tp, dt=SimulationParams().dt_s)

    # Build forecast function from scenario power profile
    scenario = SCENARIOS[name]()
    power_profile = scenario["power_profile"]

    sim_dt = scenario["dt"]
    nmpc_dt = NMPCParams().dt_s
    stride = max(1, int(nmpc_dt / sim_dt))  # 300/30 = 10

    def forecast_func(step_idx: int, N: int) -> np.ndarray:
        """Perfect forecast: sample power profile at NMPC dt intervals."""
        forecasts = np.zeros(N)
        for k in range(N):
            idx = min(step_idx + (k + 1) * stride, len(power_profile) - 1)
            forecasts[k] = power_profile[idx]
        return forecasts

    ctrl = NMPCWithEKF(nmpc, ekf, power_forecast_func=forecast_func)
    runner = SimulationRunner(plant, ctrl)

    results = runner.run(scenario)
    metrics = compute_metrics(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_results(results, metrics,
                 save_path=str(RESULTS_DIR / f"nmpc_{name}.png"),
                 controller_label="NMPC + EKF")

    with open(RESULTS_DIR / f"nmpc_{name}.json", "w") as f:
        json.dump({"version": VERSION_TAG, "controller": "NMPC",
                    "scenario": name, **metrics}, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=" * 60)
    log.info(f"  AEL-NMPC {VERSION_TAG}")
    log.info("=" * 60)

    # 1. Model validation
    validate_casadi_dae()
    log.info("")

    # 2. Run scenarios with NMPC + EKF
    for scenario_name in ["power_step"]:
        log.info(f"--- Scenario: {scenario_name} ---")

        log.info("  [NMPC + EKF]")
        nmpc_metrics = run_nmpc_scenario(scenario_name)
        for k, v in nmpc_metrics.items():
            log.info(f"    {k:25s}: {v:.4f}" if isinstance(v, float) else f"    {k:25s}: {v}")

        log.info("")

    log.info(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
