"""Power scenario for AEL simulation.

Based on Christensen et al. (DTU) and thesis code:
piecewise-constant power with step up and step down,
designed to test anticipatory thermal control.
"""

import numpy as np
from config.parameters import ThermalParams


def power_step(dt: float = 30.0) -> dict:
    """14-hour piecewise-constant power profile.

    Structure:
        0 – 2 h:   600 W   (moderate baseline)
        2 – 5 h:   900 W   (step UP — thermal stress, pre-cool test)
        5 – 8 h:   600 W   (return to baseline)
        8 – 10 h:   50 W   (step DOWN near zero — no wind, pre-heat test)
       10 – 14 h:  600 W   (step UP back to baseline — recovery)

    Tests:
    1. Pre-cooling before 600→900 W step (hour 2)
    2. Reduce cooling before 900→600 W drop (hour 5)
    3. Pre-heating / stop cooling before 600→50 W drop (hour 8)
    4. Resume cooling before 50→600 W step up (hour 10)
    """
    duration = 14 * 3600.0
    n_steps = int(duration / dt)
    T_amb = ThermalParams().T_amb_k

    t = np.arange(n_steps) * dt
    t_h = t / 3600.0

    power = np.empty(n_steps)
    for k in range(n_steps):
        h = t_h[k]
        if h < 2.0:
            power[k] = 600.0
        elif h < 5.0:
            power[k] = 900.0
        elif h < 8.0:
            power[k] = 600.0
        elif h < 10.0:
            power[k] = 50.0
        else:
            power[k] = 600.0

    return {
        "name": "power_step",
        "duration": duration, "dt": dt, "n_steps": n_steps,
        "power_profile": power,
        "T_amb_profile": np.full(n_steps, T_amb),
    }


SCENARIOS = {
    "power_step": power_step,
}
