
"""Prepare leakage-aware, next-window PER prediction samples."""

from pathlib import Path

import pandas as pd
import zstandard as zstd
import numpy as np


BASE_FEATURES = [
    "per",
    "dist_m",
    "v_src_mps",
    "v_dst_mps",
    "cbr_tx",
    "cbr_rx",
    "cbr_global",
]

KEYS = ["run_id", "src", "dst"]


def load_windows(path: Path) -> pd.DataFrame:
    """Read a Zstandard-compressed window CSV."""
    with path.open("rb") as compressed:
        with zstd.ZstdDecompressor().stream_reader(compressed) as reader:
            return pd.read_csv(reader)



def prepare_next_window(
    df: pd.DataFrame,
    radius_only: bool = True,
) -> pd.DataFrame:
    """Create next-window PER prediction samples."""

    required = set(
        KEYS
        + BASE_FEATURES
        + [
            "window_start_s",
            "window_end_s",
            "tx",
            "n_loss",
            "within_evaluation_radius",
        ]
    )

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.sort_values(
        KEYS + ["window_start_s"]
    ).reset_index(drop=True)

    groups = df.groupby(KEYS, sort=False)

    next_start = groups["window_start_s"].shift(-1)
    next_per = groups["per"].shift(-1)
    next_tx = groups["tx"].shift(-1)
    next_loss = groups["n_loss"].shift(-1)
    next_radius = groups["within_evaluation_radius"].shift(-1)

    consecutive = (
        (next_start - df["window_end_s"]).abs() < 1e-6
    )

    valid = (
        consecutive
        & (df["tx"] > 0)
        & (next_tx > 0)
        & next_per.notna()
        & next_loss.notna()
    )

    if radius_only:
        valid &= (
            df["within_evaluation_radius"].eq(1)
            & next_radius.eq(1)
        )

    result = df.loc[
        valid,
        KEYS
        + ["window_start_s", "window_end_s"]
        + BASE_FEATURES
    ].copy()

    result = result.rename(
        columns={
            "per": "per_t",
            "window_start_s": "feature_window_start_s",
            "window_end_s": "feature_window_end_s",
        }
    )

    result["target_per"] = next_per.loc[valid].to_numpy()
    result["target_tx"] = next_tx.loc[valid].astype("int64").to_numpy()
    result["target_n_loss"] = (
        next_loss.loc[valid].astype("int64").to_numpy()
    )

    if result["target_per"].isna().any():
        raise ValueError("Missing target PER detected")

    if not result["target_per"].between(0, 1).all():
        raise ValueError("Target PER outside [0, 1]")

    if not (
        (result["target_n_loss"] >= 0)
        & (result["target_n_loss"] <= result["target_tx"])
    ).all():
        raise ValueError("Invalid target packet counts")

    calculated_per = (
        result["target_n_loss"] / result["target_tx"]
    )

    if not np.allclose(
        calculated_per,
        result["target_per"],
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError(
            "Target PER does not match target_n_loss / target_tx"
        )

    return result



def prepare_scenario(
    scenario_dir: Path,
    output_dir: Path,
    radius_only: bool = True,
) -> Path:
    """Prepare and save one scenario."""

    windows = load_windows(scenario_dir / "window.csv.zst")

    samples = prepare_next_window(
        windows,
        radius_only=radius_only,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{scenario_dir.name}.parquet"

    samples.to_parquet(output_path, index=False)

    print(
        f"{scenario_dir.name}: "
        f"{len(windows):,} windows -> "
        f"{len(samples):,} samples"
    )

    return output_path
