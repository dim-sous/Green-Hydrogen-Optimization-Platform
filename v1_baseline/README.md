# v1_baseline — Single-Cell AEL with NMPC + EKF

Single-cell alkaline electrolyzer baseline with **multiple-shooting NMPC** (CasADi/IPOPT), **1D Extended Kalman Filter**, and a PI comparison baseline. The system is formulated as a **semi-explicit index-1 DAE**: temperature is the differential state, current is an algebraic variable determined by the power equality constraint.

---

## 1. Model

### DAE System (Semi-Explicit Index-1)

```
Differential state:  x = [T]                                        (1 state)
Algebraic variable:  z = [I]                                        (1 algebraic)
Control input:       u = [Q_cool]                                   (1 control)
Disturbances:        p = [P_avail, T_amb]                           (2 parameters)
```

### Differential equation (energy balance)

```
dT/dt = (P_el - n_dot_H2 * DH_rxn - UA * (T - T_amb) - Q_cool) / C_th   [K/s]

where:
  P_el      = V_cell(T, I) * I                                      [W]
  n_dot_H2  = eta_F(I) * N_cells * I / (2 * F)                      [mol/s]
  DH_rxn    = 285,800                                                [J/mol]
  UA        = 0.208                                                  [W/K]
  C_th      = 10,417                                                 [J/K]
```

### Algebraic constraint (power equality)

```
0 = V_cell(T, I) * I - P_avail                                      [W]
```

Current I is not a free variable — it is determined by available power and temperature at every instant.

### Electrochemistry (Ulleberg model)

```
V_cell(T, I) = V_rev(T) + V_ohm(T, I) + V_act(T, I)               [V]

V_rev(T) = 1.229 - 8.5e-4 * (T - 298.15)                           [V]

V_ohm(T, I) = (r1 + r2 * T) / A * I                                [V]

where:
  r1  = 4.457e-5                                                     [ohm m2]
  r2  = 6.889e-9                                                     [ohm m2/K]
  A   = 0.25                                                         [m2]

V_act(T, I) = s * log10[(t1 + t2/T + t3/T^2) * j + 1]              [V]

where:
  j = I / A                                                          [A/m2]
  s = 0.185                                                          [V]
  t1 = 1.002, t2 = 8.424 [K], t3 = 247.3 [K2]
```

### Faradaic efficiency

```
eta_F(I) = j^2 / (f1 + j^2) * f2                                    [-]

where:
  f1 = 250.0                                                         [(A/m2)^2]
  f2 = 0.98                                                          [-]
```

### Parameter tables

#### Electrolyzer

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `N_cells` | 1 | - | Cells in series |
| `A_m2` | 0.25 | m2 | Cell active area |
| `r1_ohm_m2` | 4.457e-5 | ohm m2 | Ohmic resistance base |
| `r2_ohm_m2_k` | 6.889e-9 | ohm m2/K | Ohmic resistance temperature coefficient |
| `s_v` | 0.185 | V | Activation coefficient |
| `I_min_a` | 50 | A | Minimum current |
| `I_max_a` | 500 | A | Maximum current |
| `F_const` | 96,485 | C/mol | Faraday constant |

#### Thermal

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `C_th_jk` | 10,417 | J/K | Thermal capacitance |
| `UA_loss_wk` | 0.208 | W/K | Heat loss coefficient |
| `T_amb_k` | 293.15 | K | Ambient temperature (20 C) |
| `Q_cool_max_w` | 200 | W | Maximum cooling power |
| `T_min_k` | 323.15 | K | Minimum safe temperature (50 C) |
| `T_max_k` | 373.15 | K | Maximum safe temperature (100 C) |
| `DH_rxn_jmol` | 285,800 | J/mol | Reaction enthalpy |

---

## 2. Solver

### Plant simulation

| Component | Method | Library |
|-----------|--------|---------|
| Temperature ODE | Forward Euler | NumPy |
| Power equality (I from P_avail) | Brent's method root-finding | SciPy |

### NMPC prediction model

| Component | Method | Library |
|-----------|--------|---------|
| Temperature ODE | Explicit RK4 | CasADi (symbolic) |
| Power equality | Explicit NLP constraint | CasADi / IPOPT |
| NLP solver | Interior-point, warm-started | IPOPT |

### EKF prediction model

| Component | Method | Library |
|-----------|--------|---------|
| State propagation | DAE collocation integrator | CasADi |
| Jacobian A = dF/dT | Automatic differentiation | CasADi |

### CasADi DAE integrator (validation + EKF)

```python
dae = {'x': T, 'z': I, 'p': vertcat(Q_cool, P_avail, T_amb),
       'ode': dTdt, 'alg': V_cell(T,I)*I - P_avail}
```

Validation: CasADi DAE integrator matches NumPy plant to < 0.01% relative error over 1-hour trajectories.

---

## 3. Controller

### NMPC (TrackingNMPC)

**Objective**: maximize hydrogen production with soft thermal safety constraints. There is no temperature setpoint — temperature is free within bounds. Soft penalties (Christensen et al. formulation) penalize bound violations quadratically.

```
max   Sum_{k=0}^{N-1}  w_H2 * n_dot_H2(T_k, I_k)
    - Sum_{k=0}^{N-1}  R_dQ * (Q_cool_k - Q_cool_{k-1})^2
    - Sum_{k=0}^{N}    w_T_soft * max(0, T_k - T_max)^2
    - Sum_{k=0}^{N}    w_T_soft * max(0, T_min - T_k)^2

subject to:
  T_{k+1} = F_rk4(T_k, I_k, Q_cool_k, T_amb)      RK4 integration
  V_cell(T_k, I_k) * I_k = P_avail_k                power equality
  T_min <= T_k <= T_max                               thermal bounds
  I_min <= I_k <= I_max                                current limits
  0 <= Q_cool_k <= Q_cool_max                         actuator limits
  T_0 = T_hat                                         from EKF
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N` | 48 | Prediction horizon steps |
| `dt` | 300 s | Prediction step size (5 min) |
| Horizon | 4 h | Total lookahead (matches Christensen et al.) |
| `w_H2` | 1e5 | H2 production weight |
| `R_dQ` | 1e-2 | Cooling rate-of-change penalty |
| `w_T_soft` | 1e6 | Soft thermal bound penalty |
| Solver | IPOPT | Interior-point, warm-started |

**Key behavior**: The NMPC pushes temperature toward T_max (where efficiency is highest) and uses the P_avail forecast to anticipate power changes. The 4-hour horizon allows the controller to see upcoming step changes and pre-cool or reduce cooling accordingly.

### EKF (ExtendedKalmanFilter)

1D state filter — trivial in this version, establishes architecture for future unmeasured states.

```
State:        x = [T]       (1x1)
Measurement:  y = [T_meas]  (1x1)

Predict:  T_minus = F(T_hat, Q_cool, P_avail, T_amb)     P_minus = A*P*A + Q
Correct:  K = P_minus / (P_minus + R)                      T_hat = T_minus + K*(T_meas - T_minus)
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `Q_T` | 0.01 K^2 | Process noise — moderate model confidence |
| `R_T` | 0.25 K^2 | Measurement noise — realistic thermocouple (~0.5 K std) |
| `P0_T` | 1.0 K^2 | Initial uncertainty |

### PI baseline (PIController)

Feedback-only cooling control for comparison. Current is determined by the plant. The PI reacts to temperature error but cannot anticipate power changes.

```
Q_cool = K_p * (T - T_ref) + K_i * integral(T - T_ref) dt
```

---

## Architecture

```
                    P_avail forecast (48 steps, 4 h)
                           |
                           v
  +---------------------------------------------+
  |              TrackingNMPC                    |
  |  max Sum n_dot_H2(T_k, I_k)                |
  |  - w_T_soft * max(0, T - T_max)^2          |
  |  s.t. V_cell * I = P_avail                  |
  |  CasADi/IPOPT, RK4, warm-started           |
  +--------------------+------------------------+
                       | Q_cool*
                       v
  +----------+    +----------------------------+
  |   EKF    |<---|       AEL Plant            |
  |  T_hat   |    |  dT/dt = f(T, I, Q_cool)  |
  |          |    |  V*I = P_avail (brentq)    |
  +----------+    +----------------------------+
       |                    |
       | T_hat              | T_meas
       +--------------------+
```

## Test Scenario

| Scenario | Duration | Profile | Purpose |
|----------|----------|---------|---------|
| `power_step` | 14 h | 600 -> 900 -> 600 -> 50 -> 600 W | Anticipatory control: pre-cool, pre-heat, no-wind recovery |

## Module Structure

```
v1_baseline/
├── main.py                        VERSION_TAG = "v1_baseline"
├── README.md                      This file
├── config/
│   └── parameters.py              Frozen dataclasses with unit suffixes
├── models/
│   └── ael_model.py               CasADi DAE + NumPy plant (consolidated)
├── mpc/
│   ├── baseline_pi.py             PI comparison controller
│   └── nmpc.py                    TrackingNMPC (multiple shooting)
├── estimation/
│   └── ekf.py                     ExtendedKalmanFilter (1D)
├── data/
│   └── power_source.py            Wind turbine power model
├── simulation/
│   ├── simulator.py               Closed-loop runner
│   └── scenarios.py               Power step scenario
└── visualization/
    └── plot_results.py            4-panel diagnostic plots
```

## Running

```bash
uv run python v1_baseline/main.py
```
