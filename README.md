# AEL-NMPC: Nonlinear MPC for Alkaline Electrolyzer Operation

Nonlinear Model Predictive Control for dynamic operation of an alkaline electrolyzer
under renewable power intermittency. Built with CasADi + IPOPT for nonlinear
optimization and first-principles electrochemical-thermal-mass transfer modeling.

## Quick Start

```bash
uv sync
uv run pytest tests/ -v
python -c "
from src.plant import AELPlant
from src.control.baseline_pi import PIController
from src.simulation.runner import SimulationRunner
from src.simulation.scenarios import steady_wind
from src.utils.metrics import compute_metrics

plant = AELPlant()
ctrl = PIController()
runner = SimulationRunner(plant, ctrl)
results = runner.run(steady_wind())
metrics = compute_metrics(results)
print(metrics)
"
```

## Architecture

```
Plant (NumPy) ────── u_opt ─────── NMPC (CasADi/IPOPT)
     │                                  │
     │ y_meas                           │ x̂, d̂
     └──────────── Estimator ───────────┘
                   (v3: Luenberger)
                   (v4+: MHE)
```

## Version History

| Version | Adds | Key Result |
|---------|------|------------|
| v1_baseline | Full plant model, CasADi model, PI controller | Baseline metrics established |
| v2_nmpc | Core NMPC replaces PI | Lower SEC, higher H₂ yield |
| v3_disturbance | Offset-free NMPC + Luenberger observer | Zero steady-state error |
| v4_mhe | Moving Horizon Estimation | Better state estimation under noise |
| v5_stochastic | Multi-scenario NMPC | Robust to forecast uncertainty |
| v6_multi_stack | Multi-stack coordination | Optimal load allocation |

## Key Results

| Version | Scenario | H₂ Yield (mol) | Avg SEC | T RMSE (K) | HTO Violations |
|---------|----------|-----------------|---------|------------|----------------|
| v1_pi | steady_wind | — | — | — | — |
| v1_pi | ramp_up_down | — | — | — | — |

*(Filled after each version run)*

## References

- Ulleberg, Ø. (2003). Modeling of advanced alkaline electrolyzers: a system simulation approach.
- Pannocchia, G. & Rawlings, J.B. (2003). Disturbance models for offset-free MPC.
- Zhong, W. et al. (2025). Model predictive control for electrolyzer systems.
- Biegler, L.T. (2010). Nonlinear Programming: Concepts, Algorithms, and Applications.

## License

MIT
