"""All tuneable parameters as frozen dataclasses.

Naming convention: unit suffixes on dimensional parameters (e.g. _m2, _jk, _w).
Comment style: # Description [unit]
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ElectrolyzerParams:
    """Ulleberg electrochemical model parameters for a single AEL cell."""

    N_cells: int = 1                       # Number of cells in series [-]
    A_m2: float = 0.25                     # Cell active area [m²]
    r1_ohm_m2: float = 4.45685e-5         # Ohmic resistance base [Ω·m²]
    r2_ohm_m2_k: float = 6.88874e-9       # Ohmic resistance temperature coeff [Ω·m²/K]
    s_v: float = 0.185                     # Activation overvoltage coefficient [V]
    t1: float = 1.002                      # Activation kinetic parameter [-]
    t2: float = 8.424                      # Activation kinetic parameter [K]
    t3: float = 247.3                      # Activation kinetic parameter [K²]
    f1: float = 250.0                      # Faradaic efficiency half-saturation [(A/m²)²]
    f2: float = 0.98                       # Faradaic efficiency high-current asymptote [-]
    I_min_a: float = 50.0                  # Minimum operating current [A]
    I_max_a: float = 500.0                 # Maximum operating current [A]
    F_const: float = 96485.0               # Faraday constant [C/mol]


@dataclass(frozen=True)
class ThermalParams:
    """Lumped thermal model parameters (single cell, scaled ÷60 from stack)."""

    C_th_jk: float = 10417.0              # Thermal capacitance [J/K]
    UA_loss_wk: float = 0.208             # Convective heat loss coefficient [W/K]
    T_amb_k: float = 293.15               # Nominal ambient temperature [K]
    Q_cool_max_w: float = 200.0           # Maximum active cooling power [W]
    T_ref_k: float = 353.15               # Reference operating temperature [K]
    T_min_k: float = 323.15               # Minimum safe operating temperature [K]
    T_max_k: float = 373.15               # Maximum safe operating temperature [K]
    DH_rxn_jmol: float = 285800.0         # Water-splitting reaction enthalpy [J/mol]


@dataclass(frozen=True)
class PowerSourceParams:
    """Wind turbine power source (placeholder — will be replaced with real data)."""

    P_rated_w: float = 940.0              # Rated power output [W] (V_cell*I_max at T_ref)
    v_ci_ms: float = 3.0                  # Cut-in wind speed [m/s]
    v_r_ms: float = 12.0                  # Rated wind speed [m/s]


@dataclass(frozen=True)
class PIControlParams:
    """PI baseline controller tuning (scaled ÷60 from stack)."""

    K_p_T: float = 8.0                    # Proportional gain on temperature error [W/K]
    K_i_T: float = 0.17                   # Integral gain on temperature error [W/(K·s)]
    I_nom_a: float = 300.0                # Nominal current for feedforward [A]


@dataclass(frozen=True)
class NMPCParams:
    """Multiple-shooting NMPC tuning parameters."""

    N: int = 48                            # Prediction horizon steps [-]
    dt_s: float = 300.0                    # Prediction step size [s] (5 min, matches Christensen)
    w_H2: float = 1e5                       # H₂ production rate weight [-]
    R_dQ: float = 1e-2                     # Cooling rate-of-change penalty [-]
    w_T_soft: float = 1e6                  # Soft thermal bound penalty weight [-]
    solver_max_iter: int = 300             # IPOPT maximum iterations [-]
    solver_print_level: int = 0            # IPOPT verbosity [-]


@dataclass(frozen=True)
class EKFParams:
    """Extended Kalman Filter tuning (1D: state = T)."""

    Q_T: float = 0.01                     # Process noise variance for T [K²]
    R_T: float = 0.25                     # Measurement noise variance for T [K²]
    P0_T: float = 1.0                     # Initial state error covariance [K²]


@dataclass(frozen=True)
class SimulationParams:
    """Simulation timing and reproducibility."""

    dt_s: float = 30.0                     # Simulation time step [s]
    seed: int = 42                         # Random seed [-]
