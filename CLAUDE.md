## Current State
- Active: v1 (baseline — NMPC + EKF + PI comparison, single cell, DAE formulation)
- Next: v2 (full 7-state plant, simplified control model, CD-EKF with disturbance augmentation)
- Status: v1 NMPC soft constraint tuning in progress

## If context is unclear Re-read this file top to bottom. Ask me to confirm the current version.

You are assisting in the development of an industry-grade alkaline electrolyzer digital twin, control, and optimization platform in Python.
Version 1 (baseline) is the current working version. All further development follows an incremental engineering process toward a robust, production-grade system.
────────────────────────────────────────
BEHAVIORAL PRINCIPLES
────────────────────────────────────────
1. PROPOSE BEFORE IMPLEMENTING
Before writing any code for a new upgrade, output a short proposal:
What does this upgrade add and why does it matter?
What are the expected benefits and risks?
Any dependencies on prior upgrades or known implementation pitfalls?
For non-trivial changes, wait for confirmation. If you see a better approach or a meaningful tradeoff, surface it first.
2. ONE UPGRADE AT A TIME
Each upgrade creates a new versioned folder. Never merge multiple upgrades into a single step. Each version must be independently runnable and its changes reversible.
3. THREE-STAGE GATE — mandatory before moving to the next upgrade
A. Validation — confirm the implementation is physically and mathematically consistent
B. Evaluation — compute and log the standard metrics (see below)
C. Comparison — generate plots comparing this version to the previous one and to v1
4. PRODUCTION-GRADE CODE
Use modular architecture, type hints, docstrings, configuration files, logging, and exception handling throughout. Use explicit physical units everywhere. Prefer clarity and maintainability over cleverness.
5. AVOID OVER-ENGINEERING
Prefer the smallest implementation that meaningfully advances the system. If a proposed approach significantly increases complexity or compute time, explain the tradeoff and suggest a lighter alternative.
6. COMPUTATIONAL AWARENESS
For each upgrade, note the effect on simulation and solver time. Ensure the system remains tractable. Flag any upgrade that risks making real-time operation infeasible.
────────────────────────────────────────
CODEBASE STRUCTURE — mirrors Energy Storage Optimization Platform
────────────────────────────────────────
Each version folder follows this module layout:

vN_<name>/
├── main.py                    # Entry point with VERSION_TAG
├── README.md                  # Math-heavy: full equations, parameters, results
├── stress_test.py             # Validation test suite
├── config/parameters.py       # Frozen dataclasses, explicit units
├── models/                    # CasADi symbolic + NumPy plant
├── mpc/                       # Controller (NMPC, PI)
├── estimation/                # State estimators (EKF, MHE)
├── data/                      # Power source / disturbance generators
├── simulation/                # Closed-loop runner + scenarios
└── visualization/             # Plotting

Conventions:
- All parameters as @dataclass(frozen=True) with unit suffix names (e.g. C_th_jk, T_amb_k, P_rated_w)
- Comment style: # Description [unit]
- Type hints on all function signatures
- VERSION_TAG = "vN_<name>" in main.py
- Equations in code comments match README notation
- Root README.md: simple (architecture + results)
- Version README.md: math-heavy (full ODE/DAE system + parameter tables)
────────────────────────────────────────
VERSIONING STRUCTURE (draft roadmap)
────────────────────────────────────────
ael_nmpc/
├── v1_baseline/       Single-cell DAE, NMPC+EKF, PI comparison
├── v2_full_plant/     7-state DAE plant, simplified control model, CD-EKF + disturbance augmentation
├── v3_stochastic/     Scenario-tree NMPC for wind forecast uncertainty
├── v4_estimators/     EKF vs UKF vs MHE comparison study
├── v5_model_enrich/   3-stage HTO model (Qiu), lye circulation, radiation losses
├── v6_multi_stack/    N-in-1 shared BoP, inter-stack load allocation NMPC
├── v7_solvers/        acados vs IPOPT vs ESDIRK34 solver comparison
├── comparison/
└── results/version_comparison.csv
────────────────────────────────────────
STANDARD METRICS — compute and store after every version
────────────────────────────────────────
Production: total_H2_yield, avg_SEC
Safety: T_violations, T_max, T_min
Control: power_utilization, avg_Q_cool
Computational: avg_mpc_solve_time, max_mpc_solve_time
Store in results/version_comparison.csv.
────────────────────────────────────────
The goal is a well-tested, physically realistic, and maintainable platform that improves measurably at each step — not one that is maximally complex.
────────────────────────────────────────
