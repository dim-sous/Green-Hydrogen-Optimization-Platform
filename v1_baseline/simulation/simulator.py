"""Closed-loop simulation orchestrator."""

import numpy as np
import time
from models.ael_plant import AELPlant


class SimulationRunner:
    """Run closed-loop simulation with any controller."""

    def __init__(self, plant: AELPlant, controller):
        self.plant = plant
        self.controller = controller

    def run(self, scenario: dict) -> dict:
        n_steps = scenario["n_steps"]
        dt = scenario["dt"]
        power_profile = scenario["power_profile"]
        T_amb_profile = scenario["T_amb_profile"]

        T0 = scenario.get("T0", 353.15)
        x_HTO_0 = scenario.get("x_HTO_0", 0.005)
        self.plant.reset(T0=T0, x_HTO_0=x_HTO_0)
        self.controller.reset()

        t_log = np.zeros(n_steps + 1)
        x_log = np.zeros((n_steps + 1, 2))
        u_log = np.zeros((n_steps, 2))
        y_log = np.zeros((n_steps, 5))
        d_log = np.zeros((n_steps, 2))
        solve_times = np.zeros(n_steps)

        x_log[0] = self.plant.get_state()

        for k in range(n_steps):
            t_log[k + 1] = (k + 1) * dt
            x_k = self.plant.get_state()
            d_k = np.array([power_profile[k], T_amb_profile[k]])
            d_log[k] = d_k

            t_start = time.perf_counter()
            u_k = self.controller.step(x_k, d_k, dt)
            solve_times[k] = time.perf_counter() - t_start
            u_log[k] = u_k

            y_k = self.plant.step(u_k, d_k, dt)
            y_log[k] = y_k
            x_log[k + 1] = self.plant.get_state()

        return {
            "scenario": scenario["name"], "dt": dt, "n_steps": n_steps,
            "t": t_log, "x": x_log, "u": u_log, "y": y_log, "d": d_log,
            "solve_times": solve_times,
        }
