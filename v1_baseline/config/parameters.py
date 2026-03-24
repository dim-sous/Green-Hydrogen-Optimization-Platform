"""All tuneable parameters as frozen dataclasses."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ElectrolyzerParams:
    N_cells: int = 60
    A: float = 0.25           # m² cell area
    r1: float = 4.45685e-5    # Ω·m²
    r2: float = 6.88874e-9    # Ω·m²/K
    s: float = 0.185          # activation coefficient
    t1: float = 1.002
    t2: float = 8.424
    t3: float = 247.3
    f1: float = 250.0         # Faradaic efficiency param
    f2: float = 0.98          # Faradaic efficiency param
    I_min: float = 50.0       # A
    I_max: float = 500.0      # A
    F_const: float = 96485.0  # C/mol Faraday constant


@dataclass(frozen=True)
class ThermalParams:
    C_th: float = 625000.0     # J/K thermal capacitance
    UA_loss: float = 12.5      # W/K heat loss coefficient
    T_amb: float = 293.15      # K ambient temperature
    Q_cool_max: float = 15000.0  # W max cooling power
    T_ref: float = 353.15      # K reference temperature
    T_min: float = 323.15      # K min operating temp
    T_max: float = 373.15      # K max operating temp
    DH_rxn: float = 285800.0   # J/mol enthalpy of reaction


@dataclass(frozen=True)
class GasSeparatorParams:
    K_perm: float = 5.0e-10    # mol/(s·Pa) permeation coefficient
    V_sep: float = 0.05        # m³ separator volume
    p_op: float = 10.0e5       # Pa operating pressure (10 bar)
    x_HTO_max: float = 0.02    # safety limit
    R_gas: float = 8.314       # J/(mol·K) universal gas constant

    @property
    def n_total(self) -> float:
        """Molar holdup in separator at operating conditions (mol)."""
        return self.p_op * self.V_sep / (self.R_gas * 293.15)


@dataclass(frozen=True)
class PowerSourceParams:
    P_rated: float = 125000.0  # W rated power
    v_ci: float = 3.0          # m/s cut-in wind speed
    v_r: float = 12.0          # m/s rated wind speed


@dataclass(frozen=True)
class PIControlParams:
    K_p_T: float = 500.0
    K_i_T: float = 10.0
    I_nom: float = 300.0       # A nominal current


@dataclass(frozen=True)
class SimulationParams:
    dt: float = 30.0           # s simulation time step
    seed: int = 42
