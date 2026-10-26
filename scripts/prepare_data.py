
#!/usr/bin/env python3
"""Prepare next-window PER samples for VANET scenarios."""

import argparse
from pathlib import Path

from per_prediction.data.preprocess import prepare_scenario


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/datasets/vanet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/data/processed"),
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Process one scenario; default processes all.",
    )

    args = parser.parse_args()

    if args.scenario:
        scenario_dirs = [args.dataset_root / args.scenario]
    else:
        scenario_dirs = sorted(
            args.dataset_root.glob("juelich_core_*")
        )

    if not scenario_dirs:
        parser.error("No scenario directories found")

    for scenario_dir in scenario_dirs:
        if not scenario_dir.is_dir():
            parser.error(
                f"Scenario directory not found: {scenario_dir}"
            )

        prepare_scenario(
            scenario_dir,
            args.output_dir,
            radius_only=True,
        )

    print(f"\nProcessed {len(scenario_dirs)} scenarios.")


if __name__ == "__main__":
    main()
