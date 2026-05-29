# Alkaline Electrolyzer — NMPC + EKF (single cell)

A personal project exploring nonlinear model predictive control (NMPC) and
state estimation for the thermal operation of an alkaline electrolyzer (AEL)
under variable renewable power. Single-cell model, in active development.

## What it does

Given a variable available-power profile, it asks: how should cooling be
controlled to maximize hydrogen production while keeping cell temperature
within safe bounds, and can a controller use a power forecast to act before
temperature drifts rather than after?

The model is a single cell formulated as a semi-explicit index-1 DAE:
temperature is the differential state, and current is an algebraic variable
fixed by the power equality `V_cell(T, I) * I = P_avail`. Two controllers are
implemented as separate, independently runnable versions — a PID temperature
baseline (`v0_pid`) and an NMPC with a 1-D EKF (`v1_baseline`) — sharing the
same plant model and `power_step` scenario so they can be compared.

## Architecture (v1)

```
 Power profile ──> AEL plant      <──  NMPC (5 min step, 4 h horizon)  <──  EKF
                   1-state DAE          maximize H2, soft thermal bounds      1-D filter
                   Euler + brentq       CasADi / IPOPT, warm-started          CasADi AD
```

- **Plant** ([v1_baseline/models/ael_model.py](v1_baseline/models/ael_model.py)) —
  Ulleberg electrochemistry + lumped thermal ODE; current solved from the power
  equality via Brent root-finding (NumPy/SciPy). CasADi symbolic DAE + integrator
  for the controller and estimator.
- **NMPC** ([v1_baseline/controller/nmpc.py](v1_baseline/controller/nmpc.py)) —
  multiple shooting, explicit RK4 prediction, soft thermal-bound penalties;
  CasADi/IPOPT, warm-started.
- **EKF** ([v1_baseline/estimation/ekf.py](v1_baseline/estimation/ekf.py)) — 1-D
  temperature filter; trivial here (the only state is measured), present to
  establish the estimation structure for later versions.
- **PID baseline** ([v0_pid/controller/pid.py](v0_pid/controller/pid.py)) —
  feedback-only cooling, the comparison baseline in its own `v0_pid` version.

Full equations and parameter tables are in each version's README
([v0_pid/README.md](v0_pid/README.md), [v1_baseline/README.md](v1_baseline/README.md)).

## Status

| Version | Focus | State |
|---------|-------|-------|
| **v0** pid | Single-cell DAE, PID temperature control baseline | Implemented; `stress_test.py` passes |
| **v1** baseline | Single-cell DAE, NMPC + EKF | Implemented; soft-constraint tuning in progress |

Planned (not yet implemented — see [BACKLOG.md](BACKLOG.md)):
v2 full 7-state plant + CD-EKF, v3 scenario-tree NMPC for forecast uncertainty,
v4 EKF/UKF/MHE comparison, v5 model enrichment (HTO, lye circulation, radiation),
v6 multi-stack allocation, v7 solver comparison. These are a roadmap only; none
of this code exists yet.

### Known limitations
- One scenario only (`power_step`, a 14 h piecewise-constant profile). The NMPC
  weights are tuned for this scenario; on it the latest run keeps temperature
  within bounds (0 violations, T_max ≈ 372.7 K), but this has not been tested
  beyond the single profile.
- The forecast used is the exact future profile (perfect foresight), not a real
  forecast. `data/power_source.py` and the sample CSV are placeholders and are
  not wired into the closed loop.
- Each version is validated by its own `stress_test.py` (both pass). The
  top-level `tests/` directory is an orphaned older suite that imports modules
  from before the model was consolidated/renamed and no longer collects; the
  per-version `stress_test.py` files are the current tests.

## Quick start

```bash
uv sync

uv run python v0_pid/main.py        # PID baseline
uv run python v1_baseline/main.py   # NMPC + EKF
```

Each runs the `power_step` scenario and writes plots and metrics JSON to
`results/<version>/`. `v1_baseline/main.py` also runs a CasADi-DAE-vs-NumPy
consistency check before simulating.

Per-version validation:

```bash
cd v1_baseline && uv run python -m pytest stress_test.py
```

## Tech stack

Python, CasADi + IPOPT (NLP / automatic differentiation), NumPy, SciPy
(`brentq` root-finding), Matplotlib. Dependencies pinned via `uv` (`pyproject.toml`,
`uv.lock`).

## References

- Ulleberg, Ø. (2003). *Modeling of advanced alkaline electrolyzers: a system
  simulation approach.* Int. J. Hydrogen Energy.
- Christensen, A.H.D. et al. *Nonlinear model predictive control for dynamic
  operation of an alkaline electrolyzer.* DTU. (`docs/`)
- Qiu, Y. et al. (2025). *Dynamic operation and control of a multi-stack AWE
  system.* arXiv:2501.14576. (`docs/`)
