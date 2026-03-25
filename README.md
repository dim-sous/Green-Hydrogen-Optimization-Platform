# Alkaline Electrolyzer NMPC Platform

**Optimal control and state estimation for alkaline electrolyzers under intermittent renewable power.**

## What It Solves

Given variable wind power, how should cooling be controlled to maximize hydrogen production while keeping temperature within safe operating bounds? How can the controller anticipate power changes before they cause thermal violations?

This platform answers these questions using nonlinear model predictive control (NMPC) with a DAE formulation where current is determined by the power equality constraint V(T,I)*I = P_avail.

## Architecture

```
 Wind Power ──> AEL Plant  <──  NMPC (5 min steps, 4 h horizon)  <──  EKF
                1-state DAE      maximize H2, soft thermal bounds       1D filter
                Euler + brentq   CasADi/IPOPT, warm-started             CasADi AD
```

## Versioned Upgrades

Each version adds one capability, passes a validation gate, and is frozen before the next begins.

| Version | Focus | Key Addition |
|---------|-------|-------------|
| **v1** Baseline | Foundation | Single-cell DAE, NMPC+EKF, PI comparison |
| **v2** Full Plant | Realism | 7-state DAE, simplified control model, CD-EKF with disturbance augmentation |
| **v3** Stochastic | Robustness | Scenario-tree NMPC for wind uncertainty |
| **v4** Estimators | Comparison | EKF vs UKF vs MHE |
| **v5** Model Enrichment | Physics | 3-stage HTO, lye circulation, radiation |
| **v6** Multi-Stack | Scale | N-in-1 shared BoP, load allocation |
| **v7** Solvers | Performance | acados vs IPOPT benchmark |

## Quick Start

```bash
uv sync
uv run python v1_baseline/main.py
```

Results are saved to `results/`. Each version is independently runnable.

## Technical Stack

Python, CasADi + IPOPT (nonlinear optimization), NumPy (numerics), SciPy (root-finding), Matplotlib (visualization).

## References

- Ulleberg, O. (2003). Modeling of advanced alkaline electrolyzers.
- Christensen, A.H.D. et al. Nonlinear model predictive control for dynamic operation of an alkaline electrolyzer. DTU.
- Qiu, Y. et al. (2025). Dynamic operation and control of a multi-stack AWE system. arXiv:2501.14576.

---

*Each version contains its own `README.md` with full mathematical formulations.*
