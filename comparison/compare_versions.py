"""Cross-version metrics comparison infrastructure."""

import json
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def save_results(version: str, scenario: str, metrics: dict) -> Path:
    out_dir = RESULTS_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{scenario}.json"
    with open(path, "w") as f:
        json.dump({"version": version, "scenario": scenario, **metrics}, f, indent=2)
    return path


def load_results(version: str, scenario: str) -> dict | None:
    path = RESULTS_DIR / version / f"{scenario}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def comparison_table(
    versions: list[str] | None = None,
    scenarios: list[str] | None = None,
) -> pd.DataFrame:
    if versions is None:
        versions = sorted([d.name for d in RESULTS_DIR.iterdir() if d.is_dir()])
    rows = []
    for v in versions:
        v_dir = RESULTS_DIR / v
        if not v_dir.exists():
            continue
        for f in sorted(v_dir.glob("*.json")):
            scen = f.stem
            if scenarios and scen not in scenarios:
                continue
            with open(f) as fh:
                rows.append(json.load(fh))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "version" in df.columns and "scenario" in df.columns:
        df = df.set_index(["version", "scenario"])
    return df


def print_comparison(
    versions: list[str] | None = None,
    scenarios: list[str] | None = None,
) -> None:
    df = comparison_table(versions, scenarios)
    if df.empty:
        print("No results found.")
        return
    key_cols = [
        "H2_yield_mol", "avg_SEC_W_per_mol_s", "avg_efficiency",
        "T_rmse_K", "HTO_violations", "power_utilization", "mean_solve_time_s",
    ]
    available = [c for c in key_cols if c in df.columns]
    print(df[available].to_string(float_format="%.4f"))
