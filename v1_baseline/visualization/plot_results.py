"""Four-panel result visualisation for AEL v1 baseline.

Panel layout (2 rows x 2 columns)
----------------------------------
  [0,0] Temperature (state) + hard bounds     |  [0,1] Power profile (disturbance + P_el)
  [1,0] Cooling power (control) + limits      |  [1,1] H₂ production rate + cumulative yield

All constraint limits (hard bounds, actuator limits) are shown on each panel.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path

from config.parameters import ThermalParams, ElectrolyzerParams


# ── Global style constants ───────────────────────────────────────────
_SUPTITLE_SIZE = 17
_TITLE_SIZE = 15
_LABEL_SIZE = 12
_TICK_SIZE = 11
_LEGEND_SIZE = 9
_LW_MAIN = 2.2
_LW_SEC = 1.6
_LW_REF = 1.0


def plot_results(
    results: dict,
    metrics: dict | None = None,
    save_path: str | None = None,
    controller_label: str = "v1 Baseline",
) -> None:
    """Generate the four-panel summary figure.

    Output format y = [T, V_cell, n_dot_H2, SEC, I_actual]
    """
    plt.rcParams.update({
        "font.size": _TICK_SIZE,
        "axes.titlesize": _TITLE_SIZE,
        "axes.labelsize": _LABEL_SIZE,
        "xtick.labelsize": _TICK_SIZE,
        "ytick.labelsize": _TICK_SIZE,
        "legend.fontsize": _LEGEND_SIZE,
    })

    tp = ThermalParams()

    t_h = results["t"][:-1] / 3600.0
    y = results["y"]
    u = results["u"]
    d = results["d"]
    dt = results["dt"]

    T       = y[:, 0]                    # [K]
    V_cell  = y[:, 1]                    # [V]
    n_dot   = y[:, 2] * 1000            # [mol/s] -> [mmol/s]
    I       = y[:, 4]                    # [A]
    P_avail = d[:, 0]                    # [W]
    P_el    = V_cell * I                 # [W]
    Q_cool  = u[:, 0]                    # [W]

    duration_h = t_h[-1] - t_h[0]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(
        f"AEL Digital Twin — {controller_label} — {results['scenario']} "
        f"({duration_h:.0f} h)",
        fontsize=_SUPTITLE_SIZE, fontweight="bold", y=0.995,
    )

    # ── [0,0] Temperature (state) ───────────────────────────────────
    ax = axes[0, 0]
    ax.axhspan(tp.T_max_k, tp.T_max_k + 20, alpha=0.08, color="red",
               label="Unsafe")
    ax.axhspan(tp.T_min_k - 20, tp.T_min_k, alpha=0.08, color="red")
    ax.axhline(tp.T_max_k, color="tab:red", lw=_LW_REF, ls="--",
               label=f"T_max = {tp.T_max_k:.0f} K")
    ax.axhline(tp.T_min_k, color="tab:red", lw=_LW_REF, ls="--",
               label=f"T_min = {tp.T_min_k:.0f} K")
    ax.plot(t_h, T, color="0.25", lw=_LW_MAIN, label="T (state)")
    if metrics:
        ax.annotate(
            f"Max = {metrics['T_max_K']:.1f} K\n"
            f"Min = {metrics['T_min_K']:.1f} K\n"
            f"Violations = {metrics['T_violations']}",
            xy=(0.02, 0.95), xycoords="axes fraction",
            fontsize=_LEGEND_SIZE + 1, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
        )
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Temperature (State, Hard Constraint)")
    ax.legend(loc="center right", fontsize=_LEGEND_SIZE - 1)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ── [0,1] Power profile (disturbance + response) ────────────────
    ax = axes[0, 1]
    ax.plot(t_h, P_avail, color="tab:orange", lw=_LW_MAIN,
            label="P_avail (disturbance)")
    ax.plot(t_h, P_el, color="tab:blue", lw=_LW_SEC, ls="--",
            alpha=0.8, label="P_el (consumed)")
    if metrics:
        ax.annotate(
            f"Utilisation = {metrics['power_utilization']:.1%}",
            xy=(0.02, 0.95), xycoords="axes fraction",
            fontsize=_LEGEND_SIZE + 1, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
        )
    ax.set_ylabel("Power [W]")
    ax.set_title("Power (Disturbance, P_el = P_avail)")
    ax.legend(loc="upper right")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ── [1,0] Cooling power (control input) ──────────────────────────
    ax = axes[1, 0]
    ax.plot(t_h, Q_cool, color="0.25", lw=_LW_MAIN, label="Q_cool (control)")
    ax.axhline(tp.Q_cool_max_w, color="tab:red", lw=_LW_REF, ls="--",
               alpha=0.7, label=f"Q_cool_max = {tp.Q_cool_max_w:.0f} W")
    ax.axhline(0.0, color="tab:red", lw=_LW_REF, ls="--",
               alpha=0.7, label="Q_cool_min = 0 W")
    if metrics:
        ax.annotate(
            f"Avg = {metrics['avg_Q_cool_W']:.0f} W",
            xy=(0.02, 0.95), xycoords="axes fraction",
            fontsize=_LEGEND_SIZE + 1, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
        )
    ax.set_ylabel("Cooling [W]")
    ax.set_xlabel("Time [h]")
    ax.set_title("Cooling Power (Control Input, Actuator Limits)")
    ax.legend(loc="upper right")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ── [1,1] H₂ production ─────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(t_h, n_dot, color="tab:blue", lw=_LW_MAIN, label="H₂ rate")
    ax.set_ylabel("H₂ rate [mmol/s]")
    ax.set_xlabel("Time [h]")
    ax.set_title("Hydrogen Production (Objective)")
    ax.legend(loc="upper left")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    cum_mol = np.cumsum(y[:, 2] * dt)
    ax2 = ax.twinx()
    ax2.plot(t_h, cum_mol, color="tab:green", lw=_LW_SEC, alpha=0.6,
             ls="--", label="Cumulative yield")
    ax2.set_ylabel("Cumulative H₂ [mol]", color="tab:green", alpha=0.6)
    ax2.tick_params(axis="y", colors="tab:green", labelsize=_TICK_SIZE)
    if metrics:
        ax.annotate(
            f"Total = {metrics['H2_yield_mol']:.1f} mol",
            xy=(0.02, 0.95), xycoords="axes fraction",
            fontsize=_LEGEND_SIZE + 1, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
        )

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
