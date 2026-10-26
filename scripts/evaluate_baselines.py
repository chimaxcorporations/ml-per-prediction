
#!/usr/bin/env python3
"""Generate scenario-level PER baseline results."""

import argparse
from pathlib import Path

from per_prediction.evaluation.baselines import evaluate_all


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/workspace/data/processed"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/results/baselines"),
    )

    args = parser.parse_args()

    results = evaluate_all(args.input_dir)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = args.output_dir / "scenario_baselines.csv"

    results.to_csv(output, index=False)

    print("\n=== Scenario-level summary ===")

    print(
        results[
            [
                "density",
                "cam_rate_hz",
                "propagation",
                "samples",
                "mean_per",
                "loss_window_pct",
                "zero_rmse",
                "persistence_rmse",
                "persistence_r2",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
