"""v1_baseline — PI Control + CasADi Model Validation.

Entry point for running v1 simulations and saving results.
"""

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from config.parameters import (
    ElectrolyzerParams,
    ThermalParams,
    GasSeparatorParams,
    PIControlParams,
    PowerSourceParams,
)
from models.ael_plant import AELPlant
from models.casadi_model import build_ael_model, build_integrator
from mpc.baseline_pi import PIController
from simulation.simulator import SimulationRunner
from simulation.scenarios import SCENARIOS
from visualization.plot_results import plot_results

RESULTS_DIR = PROJECT_ROOT.parent / "results" / "v1_pi"


def compute_metrics(results: dict, T_ref: float = 353.15, x_HTO_max: float = 0.02) -> dict:
    """Compute performance metrics from simulation results."""
    y = results["y"]
    u = results["u"]
    d = results["d"]
    dt = results["dt"]

    T = y[:, 0]
    x_HTO = y[:, 1]
    V_stack = y[:, 2]
    n_dot_H2 = y[:, 3]
    SEC = y[:, 4]
    I = u[:, 0]
    P_el = V_stack * I
    P_avail = d[:, 0]

    H2_yield = float(np.sum(n_dot_H2) * dt)
    producing = n_dot_H2 > 1e-10
    avg_SEC = float(np.mean(SEC[producing])) if np.any(producing) else 0.0

    DH_HHV = 285800.0
    eta_energy = np.where(P_el > 1e-6, n_dot_H2 * DH_HHV / P_el, 0.0)
    avg_efficiency = float(np.mean(eta_energy[producing])) if np.any(producing) else 0.0

    T_rmse = float(np.sqrt(np.mean((T - T_ref) ** 2)))
    HTO_violations = int(np.sum(x_HTO > x_HTO_max))

    power_util = float(np.mean(P_el / np.maximum(P_avail, 1e-6)))
    mean_solve = float(np.mean(results["solve_times"]))
    max_solve = float(np.max(results["solve_times"]))

    return {
        "H2_yield_mol": H2_yield,
        "avg_SEC_W_per_mol_s": avg_SEC,
        "avg_efficiency": avg_efficiency,
        "T_rmse_K": T_rmse,
        "T_max_K": float(np.max(T)),
        "T_min_K": float(np.min(T)),
        "HTO_violations": HTO_violations,
        "HTO_max": float(np.max(x_HTO)),
        "power_utilization": power_util,
        "mean_solve_time_s": mean_solve,
        "max_solve_time_s": max_solve,
    }


def validate_casadi_model() -> None:
    """Validate CasADi model matches NumPy plant on 1h open-loop trajectory."""
    from models.electrolyzer import Electrolyzer
    from models.thermal import ThermalModel
    from models.gas_separator import GasSeparator

    print("=== CasADi vs NumPy Model Validation ===")

    model = build_ael_model()
    dt = 30.0

    for method in ("collocation", "rk4"):
        F_int = build_integrator(model, dt, method=method)

        ely = Electrolyzer()
        thermal = ThermalModel()
        gas_sep = GasSeparator()

        T_np, x_HTO_np = 353.15, 0.005
        x_casadi = np.array([353.15, 0.005])

        I, Q_cool = 300.0, 5000.0
        P_avail, T_amb = 100000.0, 293.15
        u = np.array([I, Q_cool])
        p = np.array([P_avail, T_amb])

        n_steps = int(3600.0 / dt)
        for _ in range(n_steps):
            P_el = ely.P_el(T_np, I)
            n_dot_H2 = ely.n_dot_H2(T_np, I)
            T_np = thermal.step(T_np, P_el, n_dot_H2, Q_cool, dt, T_amb)
            x_HTO_np = gas_sep.step(x_HTO_np, n_dot_H2, dt)

            x_casadi = np.array(F_int(x_casadi, u, p)).flatten()

        rel_T = abs(x_casadi[0] - T_np) / abs(T_np)
        rel_HTO = abs(x_casadi[1] - x_HTO_np) / max(abs(x_HTO_np), 1e-10)
        status_T = "PASS" if rel_T < 0.001 else "FAIL"
        status_HTO = "PASS" if rel_HTO < 0.05 else "FAIL"

        print(f"  {method:12s}  T: rel_err={rel_T:.2e} [{status_T}]"
              f"  x_HTO: rel_err={rel_HTO:.2e} [{status_HTO}]")


def run_scenario(name: str) -> dict:
    """Run a single scenario with PI controller and save results."""
    plant = AELPlant()
    ctrl = PIController()
    runner = SimulationRunner(plant, ctrl)

    scenario = SCENARIOS[name]()
    results = runner.run(scenario)
    metrics = compute_metrics(results)

    # Save plot
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_results(results, metrics, save_path=str(RESULTS_DIR / f"{name}.png"))

    # Save metrics
    import json
    with open(RESULTS_DIR / f"{name}.json", "w") as f:
        json.dump({"version": "v1_pi", "scenario": name, **metrics}, f, indent=2)

    return metrics


def main() -> None:
    print("=" * 60)
    print("  AEL-NMPC v1_baseline — PI Controller")
    print("=" * 60)

    # Model validation
    validate_casadi_model()
    print()

    # Run v1 scenarios
    for name in ("steady_wind", "ramp_up_down"):
        print(f"--- Scenario: {name} ---")
        metrics = run_scenario(name)
        for k, v in metrics.items():
            print(f"  {k:25s}: {v:.6f}" if isinstance(v, float) else f"  {k:25s}: {v}")
        print()

    print("Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
