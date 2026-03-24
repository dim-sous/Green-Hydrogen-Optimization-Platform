"""Plotting utilities for AEL simulation results."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def plot_results(results: dict, metrics: dict | None = None, save_path: str | None = None) -> None:
    """Plot simulation results in a 4-panel figure."""
    t = results["t"][:-1] / 3600.0
    y = results["y"]
    u = results["u"]
    d = results["d"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"AEL Simulation — {results['scenario']}", fontsize=14)

    # Temperature
    ax = axes[0]
    ax.plot(t, y[:, 0], "r-", label="T (K)")
    ax.axhline(353.15, color="k", ls="--", alpha=0.5, label="T_ref")
    ax.axhline(323.15, color="b", ls=":", alpha=0.5, label="T_min")
    ax.axhline(373.15, color="b", ls=":", alpha=0.5, label="T_max")
    ax.set_ylabel("Temperature (K)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # HTO
    ax = axes[1]
    ax.plot(t, y[:, 1] * 100, "g-", label="x_HTO (%)")
    ax.axhline(2.0, color="r", ls="--", label="Safety limit (2%)")
    ax.set_ylabel("HTO (vol%)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Power and H2
    ax = axes[2]
    P_el = y[:, 2] * u[:, 0]
    ax.plot(t, d[:, 0] / 1000, "b--", alpha=0.7, label="P_avail (kW)")
    ax.plot(t, P_el / 1000, "b-", label="P_el (kW)")
    ax2 = ax.twinx()
    ax2.plot(t, y[:, 3] * 1000, "m-", alpha=0.8, label="n_dot_H2 (mmol/s)")
    ax.set_ylabel("Power (kW)")
    ax2.set_ylabel("H2 rate (mmol/s)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Control inputs
    ax = axes[3]
    ax.plot(t, u[:, 0], "k-", label="I (A)")
    ax3 = ax.twinx()
    ax3.plot(t, u[:, 1] / 1000, "c-", label="Q_cool (kW)")
    ax.set_ylabel("Current (A)")
    ax3.set_ylabel("Cooling (kW)")
    ax.set_xlabel("Time (h)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
