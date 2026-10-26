
"""Evaluate zero and persistence PER baselines by scenario."""

from pathlib import Path
import re

import numpy as np
import pandas as pd


SCENARIO_PATTERN = re.compile(
    r"^juelich_core_d(?P<density>\d+)_"
    r"r(?P<cam_rate>\d+)_"
    r"(?P<propagation>.+)_s\d+_w\d+$"
)


def metrics(y, prediction):
    """Regression metrics for PER prediction."""
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    mae = np.mean(np.abs(y - prediction))
    mse = np.mean((y - prediction) ** 2)
    rmse = np.sqrt(mse)

    variance = np.mean((y - y.mean()) ** 2)

    r2 = (
        1 - mse / variance
        if variance > 0
        else np.nan
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def evaluate_scenario(path: Path):
    """Evaluate a single processed scenario."""

    df = pd.read_parquet(path)

    if df.empty:
        raise ValueError(f"Empty scenario: {path}")

    match = SCENARIO_PATTERN.match(path.stem)

    if match is None:
        raise ValueError(
            f"Unrecognized scenario name: {path.stem}"
        )

    y = df["target_per"].to_numpy(dtype=float)
    persistence = df["per_t"].to_numpy(dtype=float)
    zero = np.zeros_like(y)

    zero_metrics = metrics(y, zero)
    persistence_metrics = metrics(y, persistence)

    result = {
        "scenario": path.stem,
        "density": int(match.group("density")),
        "cam_rate_hz": int(match.group("cam_rate")),
        "propagation": match.group("propagation"),
        "samples": len(df),
        "unique_links": df[
            ["run_id", "src", "dst"]
        ].drop_duplicates().shape[0],
        "mean_per": float(y.mean()),
        "loss_window_pct": float(100 * np.mean(y > 0)),
        "complete_loss_pct": float(100 * np.mean(y == 1)),
        "zero_mae": zero_metrics["mae"],
        "zero_rmse": zero_metrics["rmse"],
        "zero_r2": zero_metrics["r2"],
        "persistence_mae": persistence_metrics["mae"],
        "persistence_rmse": persistence_metrics["rmse"],
        "persistence_r2": persistence_metrics["r2"],
    }

    return result


def evaluate_all(input_dir: Path):
    """Evaluate all processed scenarios."""

    files = sorted(input_dir.glob("juelich_core_*.parquet"))

    if len(files) != 27:
        raise ValueError(
            f"Expected 27 scenarios, found {len(files)}"
        )

    results = []

    for path in files:
        result = evaluate_scenario(path)
        results.append(result)

        print(
            f"{result['scenario']}: "
            f"{result['samples']:,} samples, "
            f"mean PER={result['mean_per']:.5f}, "
            f"persistence RMSE="
            f"{result['persistence_rmse']:.5f}"
        )

    return pd.DataFrame(results)
