# AEL NMPC — Backlog

## Known Issues
| ID | Description | Severity | Introduced | Resolved |
|----|-------------|----------|------------|----------|
| K1 | NMPC weights are tuned only for the `power_step` scenario; not validated on any other profile | Medium | v1 | |
| K2 | Top-level `tests/` is an orphaned older suite (imports pre-consolidation module names) and does not collect; superseded by per-version `stress_test.py` | Low | v1 | |
| K3 | `comparison/compare_versions.py::print_comparison` references metric keys not produced by the version `main.py` scripts; no `results/version_comparison.csv` is generated | Low | v1 | |
| K4 | Forecast is perfect foresight (exact future profile); `data/power_source.py` + sample CSV are unused placeholders | Medium | v1 | |
| K5 | `v1_baseline/README.md` still describes the old layout (`mpc/`, in-version PI baseline) from before PI was split into `v0_pid` and `mpc/` was renamed to `controller/` | Low | v1 | |

## Completed
| Version | Item | Notes |
|---------|------|-------|
| v0 | Single-cell PID temperature-control baseline | `v0_pid/`; `stress_test.py` passes |
| v1 | Single-cell Ulleberg + lumped-thermal plant (NumPy) | `models/ael_model.py` |
| v1 | CasADi symbolic DAE + integrator | `models/ael_model.py` |
| v1 | Multiple-shooting NMPC (CasADi/IPOPT, RK4) | `controller/nmpc.py` |
| v1 | 1-D Extended Kalman Filter | `estimation/ekf.py` |
| v1 | CasADi → NumPy consistency check | `validate_casadi_dae()` in `main.py`; passes, rel_err ≈ 3e-5 |
| v1 | `power_step` scenario + closed-loop runner | `simulation/`; one scenario only |
| v0/v1 | Per-version `stress_test.py` validation suites | both pass (6 tests each) |

## Planned (not yet implemented)
| Version | Item |
|---------|------|
| v2 | Full 7-state DAE plant, simplified control model, CD-EKF + disturbance augmentation |
| v3 | Scenario-tree NMPC for wind-forecast uncertainty |
| v4 | EKF vs UKF vs MHE comparison study |
| v5 | Model enrichment (3-stage HTO, lye circulation, radiation losses) |
| v6 | Multi-stack, shared balance-of-plant, inter-stack load allocation |
| v7 | Solver comparison (acados vs IPOPT vs ESDIRK34) |
