# AEL NMPC — Backlog

## Known Issues
| ID | Description | Severity | Introduced | Resolved |
|----|-------------|----------|------------|----------|
| K1 | PI has no anticipation — reacts to temperature error, doesn't prevent it | Medium | v1 | |
| K2 | PI wastes available power — no SEC optimization | Medium | v1 | |
| K3 | No handling of forecast uncertainty | Medium | v1 | |
| K4 | No state estimation — assumes full state access | Medium | v1 | |

## Completed
| Version | Item |
|---------|------|
| v1 | Full IEHHM plant model (NumPy) |
| v1 | CasADi symbolic model + integrator |
| v1 | PI baseline controller |
| v1 | CasADi → NumPy model consistency validation |
| v1 | 2-scenario test suite (steady_wind, ramp_up_down) |
| v1 | Cross-version comparison infrastructure |
