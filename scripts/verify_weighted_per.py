
#!/usr/bin/env python3
"""Verify target packet counts and calculate weighted PER."""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_DIR = Path("/workspace/data/processed")
OUTPUT_DIR = Path("/workspace/results/data_quality")


def main():
    files = sorted(INPUT_DIR.glob("juelich_core_*.parquet"))

    if len(files) != 27:
        raise ValueError(
            f"Expected 27 Parquet files, found {len(files)}"
        )

    results = []

    total_samples = 0
    total_tx = 0
    total_loss = 0
    total_per_sum = 0.0

    for path in files:
        df = pd.read_parquet(
            path,
            columns=[
                "target_per",
                "target_tx",
                "target_n_loss",
            ],
        )

        if df.empty:
            raise ValueError(f"Empty scenario: {path.name}")

        tx = df["target_tx"].to_numpy()
        loss = df["target_n_loss"].to_numpy()
        per = df["target_per"].to_numpy()

        if np.any(tx <= 0):
            raise ValueError(f"Invalid TX: {path.name}")

        if np.any((loss < 0) | (loss > tx)):
            raise ValueError(f"Invalid loss counts: {path.name}")

        if not np.allclose(
            per,
            loss / tx,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError(f"PER mismatch: {path.name}")

        scenario_tx = int(tx.sum())
        scenario_loss = int(loss.sum())

        result = {
            "scenario": path.stem,
            "samples": len(df),
            "total_tx": scenario_tx,
            "total_loss": scenario_loss,
            "mean_window_per": float(per.mean()),
            "packet_weighted_per": (
                scenario_loss / scenario_tx
            ),
            "loss_window_pct": float(100 * np.mean(per > 0)),
        }

        results.append(result)

        total_samples += len(df)
        total_tx += scenario_tx
        total_loss += scenario_loss
        total_per_sum += float(per.sum())

    summary = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = OUTPUT_DIR / "weighted_per_by_scenario.csv"
    summary.to_csv(output, index=False)

    print("\n=== PER verification ===")
    print("Scenarios:", len(files))
    print("Samples:", f"{total_samples:,}")
    print("Transmission opportunities:", f"{total_tx:,}")
    print("Lost opportunities:", f"{total_loss:,}")

    print(
        "Overall mean window PER:",
        f"{total_per_sum / total_samples:.6f}",
    )

    print(
        "Overall packet-weighted PER:",
        f"{total_loss / total_tx:.6f}",
    )

    print("\n=== Scenario results ===")
    print(summary.to_string(index=False))

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
