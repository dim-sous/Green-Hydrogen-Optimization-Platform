"""Power scenario for AEL simulation.

Based on Christensen et al. (DTU) and thesis code:
piecewise-constant power with step up and step down,
designed to test anticipatory thermal control.
"""

import numpy as np
from config.parameters import ThermalParams


def power_step(dt: float = 30.0) -> dict:
    """24-hour piecewise-constant power profile.

    Structure:
        0 – 5 h:   400 W   (mid — warm-up baseline)
        5 – 10 h:  800 W   (high — thermal stress, pre-cool test)
       10 – 15 h:  400 W   (mid — cool-down)
       15 – 20 h:    0 W   (zero — shutdown, pre-heat test)
       20 – 24 h:  400 W   (mid — recovery to baseline)

    Tests:
    1. Pre-cooling before 400->800 W step (hour 5)
    2. Reduce cooling before 800->400 W drop (hour 10)
    3. Stop cooling before 400->10 W drop (hour 15)
    4. Resume cooling before 10->400 W step up (hour 20)
    """
    duration = 24 * 3600.0
    n_steps = int(duration / dt)
    T_amb = ThermalParams().T_amb_k

    t = np.arange(n_steps) * dt
    t_h = t / 3600.0

    power = np.empty(n_steps)
    for k in range(n_steps):
        h = t_h[k]
        if h < 5.0:
            power[k] = 400.0
        elif h < 10.0:
            power[k] = 800.0
        elif h < 15.0:
            power[k] = 400.0
        elif h < 20.0:
            power[k] = 0.0
        else:
            power[k] = 400.0

    return {
        "name": "power_step",
        "duration": duration, "dt": dt, "n_steps": n_steps,
        "power_profile": power,
        "T_amb_profile": np.full(n_steps, T_amb),
    }


SCENARIOS = {
    "power_step": power_step,
}
