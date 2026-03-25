"""Extended Kalman Filter for single-cell AEL state estimation.

1D EKF for baseline — state x = [T], measurement y = [T_meas].
Trivial in this version (measuring the only state), but establishes
the estimation architecture for future versions that add unmeasured states.

Uses the same DAE collocation integrator as the NMPC for prediction.
Jacobian A = dF/dT computed via CasADi automatic differentiation.
"""

from __future__ import annotations

import numpy as np
import casadi as ca

from config.parameters import (
    ElectrolyzerParams,
    ThermalParams,
    EKFParams,
)
from models.ael_model import build_casadi_dynamics, build_casadi_integrator


class ExtendedKalmanFilter:
    """Discrete-time Extended Kalman Filter for AEL temperature estimation.

    State:        x = [T]       (1×1)
    Measurement:  y = [T_meas]  (1×1)
    Parameters:   p = [Q_cool, P_avail, T_amb]
    """

    def __init__(
        self,
        ekf_params: EKFParams | None = None,
        ely_params: ElectrolyzerParams | None = None,
        th_params: ThermalParams | None = None,
        dt: float = 30.0,
    ):
        self.ekf = ekf_params or EKFParams()
        self.ep = ely_params or ElectrolyzerParams()
        self.tp = th_params or ThermalParams()

        # Build CasADi model and integrator
        model = build_casadi_dynamics(self.ep, self.tp)
        self._F = build_casadi_integrator(model, dt)

        # Build Jacobian function: A = dF_x/dx (scalar)
        x_sym = ca.MX.sym("x_jac", 1)
        z_sym = ca.MX.sym("z_jac", 1)
        p_sym = ca.MX.sym("p_jac", 3)

        x_next, z_next = self._F(x_sym, z_sym, p_sym)
        A_sym = ca.jacobian(x_next, x_sym)

        self._A_func = ca.Function("A_ekf", [x_sym, z_sym, p_sym], [A_sym],
                                   ["x", "z", "p"], ["A"])

        # State and covariance
        self.x_hat: float = self.tp.T_ref_k
        self.P: float = self.ekf.P0_T
        self.z_hat: float = 300.0  # algebraic variable estimate

        # Noise covariances
        self.Q = self.ekf.Q_T
        self.R = self.ekf.R_T

        # Measurement matrix (trivial: C = 1)
        self.C = 1.0

    def reset(self, x0: float | np.ndarray) -> None:
        """Initialize state estimate and covariance."""
        self.x_hat = float(x0) if np.isscalar(x0) else float(x0[0])
        self.P = self.ekf.P0_T
        self.z_hat = 300.0

    def predict(self, u: np.ndarray, d: np.ndarray) -> float:
        """Prediction step: propagate state and covariance through DAE model.

        Args:
            u: [Q_cool]          control applied at previous step [W]
            d: [P_avail, T_amb]  disturbances [W, K]

        Returns:
            x_hat_minus: predicted state (prior) [K]
        """
        Q_cool = float(u[0])
        P_avail = float(d[0])
        T_amb = float(d[1])

        p_vec = np.array([Q_cool, P_avail, T_amb])

        # Propagate state through DAE integrator
        x_next = self._F(self.x_hat, self.z_hat, p_vec)
        self.x_hat = float(x_next[0])
        self.z_hat = float(x_next[1])

        # Linearize: A = dF/dx
        A = float(self._A_func(self.x_hat, self.z_hat, p_vec))

        # Propagate covariance
        self.P = A * self.P * A + self.Q

        return self.x_hat

    def correct(self, y_meas: float | np.ndarray) -> float:
        """Measurement update step.

        Args:
            y_meas: measured temperature [K]

        Returns:
            x_hat: corrected state estimate [K]
        """
        y = float(y_meas) if np.isscalar(y_meas) else float(y_meas[0])

        # Innovation
        y_pred = self.C * self.x_hat
        innovation = y - y_pred

        # Kalman gain (scalar)
        S = self.C * self.P * self.C + self.R
        K = self.P * self.C / S

        # Update state and covariance
        self.x_hat = self.x_hat + K * innovation
        self.P = (1.0 - K * self.C) * self.P

        return self.x_hat

    def step(
        self,
        u: np.ndarray,
        d: np.ndarray,
        y_meas: float | np.ndarray,
    ) -> np.ndarray:
        """Full predict + correct cycle.

        Args:
            u: [Q_cool]          previous control [W]
            d: [P_avail, T_amb]  current disturbances [W, K]
            y_meas: measured temperature [K]

        Returns:
            x_hat: corrected state estimate as array [T_hat] (1,)
        """
        self.predict(u, d)
        self.correct(y_meas)
        return np.array([self.x_hat])
