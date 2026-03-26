"""v0_pid — Single-cell AEL with PID temperature control.

Entry point for running simulations and saving results.
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys

VERSION_TAG = "v0_pid"

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from config.parameters import (
    ElectrolyzerParams,
    ThermalParams,
    SimulationParams,
)
from models.ael_model import AELPlant
from controller.pid import PIDController
from simulation.simulator import SimulationRunner
from simulation.scenarios import SCENARIOS
from visualization.plot_results import plot_results

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT.parent / "results" / VERSION_TAG


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
# Scenario runner
# ---------------------------------------------------------------------------

def run_pid_scenario(name: str) -> dict:
    """Run a scenario with PID controller."""
    plant = AELPlant()
    ctrl = PIDController()
    runner = SimulationRunner(plant, ctrl)

    scenario = SCENARIOS[name]()
    results = runner.run(scenario)
    metrics = compute_metrics(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_results(results, metrics,
                 save_path=str(RESULTS_DIR / f"pid_{name}.png"),
                 controller_label="PID Controller")

    with open(RESULTS_DIR / f"pid_{name}.json", "w") as f:
        json.dump({"version": VERSION_TAG, "controller": "PID",
                    "scenario": name, **metrics}, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=" * 60)
    log.info(f"  AEL-PID {VERSION_TAG}")
    log.info("=" * 60)

    for scenario_name in ["power_step"]:
        log.info(f"--- Scenario: {scenario_name} ---")

        log.info("  [PID Controller]")
        metrics = run_pid_scenario(scenario_name)
        for k, v in metrics.items():
            log.info(f"    {k:25s}: {v:.4f}" if isinstance(v, float) else f"    {k:25s}: {v}")

        log.info("")

    log.info(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
