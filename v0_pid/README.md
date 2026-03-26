# v0_pid — Single-Cell AEL with PID Temperature Control

Standalone PID feedback controller for a single alkaline electrolyzer cell.
Baseline version for comparison with NMPC (v1) and future controllers.

---

## 1. Plant Model (DAE)

Single-cell Ulleberg electrochemical model with lumped thermal dynamics.
Semi-explicit index-1 DAE:

| Symbol | Dimension | Description |
|--------|-----------|-------------|
| x = [T] | 1 | Temperature [K] |
| z = [I] | 1 | Current (algebraic) [A] |
| u = [Q_cool] | 1 | Cooling power [W] |
| p = [P_avail, T_amb] | 2 | Disturbances |

### ODE (Thermal)

$$
\frac{dT}{dt} = \frac{P_{el} - \dot{n}_{H_2} \cdot \Delta H_{rxn} - UA \cdot (T - T_{amb}) - Q_{cool}}{C_{th}}
$$

### Algebraic Constraint (Power Equality)

$$
0 = V_{cell}(T, I) \cdot I - P_{avail}
$$

### Electrochemistry (Ulleberg)

| Equation | Formula |
|----------|---------|
| V_rev(T) | 1.229 − 8.5×10⁻⁴ · (T − 298.15) [V] |
| V_ohm(T, I) | (r₁ + r₂·T) / A · I [V] |
| V_act(T, I) | s · log₁₀[(t₁ + t₂/T + t₃/T²) · j + 1] [V] |
| η_F(I) | j² / (f₁ + j²) · f₂ [-] |
| ṅ_H₂ | η_F · N_cells · I / (2F) [mol/s] |

---

## 2. PID Controller

Temperature feedback controller producing cooling power Q_cool.
Current I is determined by the plant algebraic constraint (not by the controller).

### Control Law

$$
Q_{cool} = K_p \cdot e(t) + K_i \cdot \int e(\tau)\,d\tau + D_{filtered}(t)
$$

where e(t) = T − T_ref.

### Derivative Term (Derivative-on-Measurement, Filtered)

$$
\alpha = \frac{dt}{\tau_d + dt}
$$

$$
D_{filtered}[k] = (1 - \alpha) \cdot D_{filtered}[k-1] + \alpha \cdot K_d \cdot \frac{T[k] - T[k-1]}{dt}
$$

Using derivative-on-measurement (dT/dt instead of de/dt) avoids derivative kick
on setpoint changes. The first-order low-pass filter with time constant τ_d
smooths discrete derivative noise.

### Anti-Windup

Integral error is clamped to ±Q_cool_max / K_i to prevent integrator wind-up
when the actuator saturates.

---

## 3. Parameters

### Electrolyzer (Ulleberg, single cell)

| Parameter | Value | Unit |
|-----------|-------|------|
| N_cells | 1 | - |
| A | 0.25 | m² |
| r₁ | 4.457×10⁻⁵ | Ω·m² |
| r₂ | 6.889×10⁻⁹ | Ω·m²/K |
| s | 0.185 | V |
| t₁, t₂, t₃ | 1.002, 8.424, 247.3 | -, K, K² |
| f₁, f₂ | 250.0, 0.98 | (A/m²)², - |
| I_min, I_max | 50, 500 | A |

### Thermal (lumped, ÷60 from stack)

| Parameter | Value | Unit |
|-----------|-------|------|
| C_th | 10,417 | J/K |
| UA | 0.208 | W/K |
| T_ref | 353.15 (80°C) | K |
| T_min, T_max | 323.15, 373.15 | K |
| Q_cool_max | 200 | W |
| ΔH_rxn | 285,800 | J/mol |

### PID Tuning

| Parameter | Value | Unit | Rationale |
|-----------|-------|------|-----------|
| K_p | 12.0 | W/K | Raised from 8.0 to offset slower integral build-up |
| K_i | 0.08 | W/(K·s) | Reduced from 0.17 to limit integral dominance |
| K_d | 1500.0 | W·s/K | Raised from 50 for meaningful derivative damping |
| τ_d | 10.0 | s | Moderate filtering (α = 0.75 at dt=30s) |

---

## 4. Plant Shutdown Behaviour

When P_avail < P_min (where P_min = V_cell(T, I_min) · I_min ≈ 81 W), the electrolyzer
cannot sustain even the minimum operating current. The plant shuts down:
I = 0, P_el = 0, ṅ_H₂ = 0.

The thermal ODE still runs during shutdown (convective loss cools the cell toward T_amb).
This is physically correct: you cannot draw more power than is available.

---

## 5. PID Tuning Investigation

### Problem

The initial PID gains (K_p=8, K_i=0.17, K_d=50) were inherited from the v1 PI baseline
with a small derivative term added. Simulation revealed two problems:

1. **D term was negligible**: max |D| = 0.56 W vs max |I| = 200 W. The derivative
   contributed nothing meaningful.
2. **Integral dominance caused large Q_cool jumps**: at power transitions (e.g. 50→600 W
   at hour 10), the integral needed many steps to wind up from its saturated negative
   state, producing max |dQ|/step = 26.8 W.

### Root Cause

The thermal time constant τ_th = C_th / UA = 10417 / 0.208 ≈ **50,000 s** (~14 hours).
Temperature changes very slowly — typical dT/dt ≈ 0.03 K/s during power steps. With
K_d = 50, the D term was only 50 × 0.03 = 1.5 W, lost in the noise of a 200 W system.

### Tuning Study

Systematic sweep across 11 configurations, evaluated on a 14h power_step scenario
(600/900/600/50/600 W). Key results shown below:

| Configuration | T range [K] | max |dQ| [W] | dQ std [W] | T violations |
|---------------|-------------|--------------|------------|--------------|
| K_p=8, K_i=0.17, K_d=50 (initial) | [343.3, 358.3] | 26.8 | 3.0 | 0 |
| K_p=8, K_i=0.17, K_d=2000 | [343.5, 358.1] | 24.7 | 2.7 | 0 |
| K_p=8, K_i=0.05, K_d=2000 | [342.4, 361.7] | 21.0 | 1.9 | 0 |
| **K_p=12, K_i=0.08, K_d=1500** | **[343.0, 359.8]** | **18.3** | **2.0** | **0** |
| K_p=15, K_i=0.05, K_d=2000 | [342.9, 360.7] | 23.5 | 1.8 | 0 |
| K_p=20, K_i=0.05, K_d=2000 | [343.1, 360.0] | 25.2 | 1.8 | 0 |
| K_p=8, K_i=0.17, K_d=5000 | [343.7, 357.9] | 43.7 | 2.7 | 0 |

### Selected Tuning: K_p=12, K_i=0.08, K_d=1500, τ_d=10

**Why this configuration wins:**
- Lowest max |dQ|/step: **18.3 W** (32% reduction from initial 26.8 W)
- Low dQ std: **2.0 W** (33% reduction from 3.0 W)
- Tight temperature range within bounds, zero violations
- All three PID terms now contribute meaningfully:

| Term | Initial tuning | Selected tuning |
|------|----------------|-----------------|
| P (proportional) | mean 10.2 W, max 79.1 W | mean 17.0 W, max 121.6 W |
| I (integral) | mean 141.5 W, max 200.0 W | mean 140.3 W, max 200.0 W |
| D (derivative) | mean 0.1 W, max 0.6 W | mean 2.0 W, **max 16.4 W** |

The D term is now **29× more impactful** (max 16.4 W vs 0.6 W), providing real damping
during power transitions. The reduced K_i prevents the integral from dominating during
transients, while the higher K_p compensates for slower steady-state convergence.

### Key Insight

K_d = 5000 performed **worse** (max |dQ| = 43.7 W) because excessive derivative gain
amplifies step changes in dT/dt at power transitions. The τ_d filter helps, but there
is a diminishing-returns inflection around K_d = 1500–2000 for this system.

---

## 6. Module Structure


```
v0_pid/
├── main.py                    # Entry point
├── stress_test.py             # Validation tests
├── config/parameters.py       # Frozen dataclasses
├── models/ael_model.py        # CasADi DAE + NumPy plant
├── controller/pid.py          # PID controller
├── data/power_source.py       # Stochastic power generation
├── simulation/                # Simulator + scenarios
└── visualization/             # 4-panel plots
```

---

## 7. Running

```bash
cd v0_pid
python main.py           # Run simulation, save results to results/v0_pid/
python stress_test.py    # Run validation tests
```

---

## 8. Metrics

| Metric | Description |
|--------|-------------|
| H2_yield_mol | Total hydrogen produced [mol] |
| avg_SEC_Wh_per_mol | Specific energy consumption [Wh/mol] |
| T_max_K, T_min_K | Temperature extremes [K] |
| T_violations | Steps outside [T_min, T_max] |
| power_utilization | Mean P_el / P_avail |
| avg_Q_cool_W | Average cooling power [W] |
| mean_solve_time_ms | Controller compute time [ms] |
