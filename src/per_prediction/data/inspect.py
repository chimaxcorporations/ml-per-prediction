
"""Inspect the VANET PER dataset without modifying source files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_FILES = {
    "manifest.json",
    "mapping.csv",
    "opportunities.csv.zst",
    "rx_events.csv.zst",
    "tx_events.csv.zst",
    "window.csv.zst",
}


def find_scenarios(dataset_root: Path) -> list[Path]:
    """Find scenario directories in the dataset."""
    return sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and path.name.startswith("juelich_core_")
    )


def inspect_scenario(scenario_dir: Path) -> bool:
    """Inspect one scenario and return whether it passes basic checks."""
    files = {
        path.name
        for path in scenario_dir.iterdir()
        if path.is_file()
    }

    missing = EXPECTED_FILES - files
    extra = files - EXPECTED_FILES

    manifest_path = scenario_dir / "manifest.json"

    print(f"\nScenario: {scenario_dir.name}")

    if not manifest_path.is_file():
        print("  ERROR: manifest.json missing")
        return False

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    counters = manifest.get("counters", {})
    interval = manifest.get("beacon_interval_s")
    cam_rate = 1.0 / interval if interval else None

    print(f"  Status:        {manifest.get('status')}")
    print(f"  Propagation:   {manifest.get('propagation')}")
    print(f"  Window:        {manifest.get('window_s')} s")
    print(f"  CAM rate:      {cam_rate} Hz")
    print(f"  Warm-up:       {manifest.get('warmup_s')} s")
    print(f"  Sim. end:      {manifest.get('simulation_end_s')} s")
    print(f"  TX packets:    {counters.get('tx_packets')}")
    print(f"  Opportunities: {counters.get('opportunities')}")
    print(f"  Unique RX:     {counters.get('unique_rx')}")

    if missing:
        print(f"  Missing files: {sorted(missing)}")

    if extra:
        print(f"  Extra files:   {sorted(extra)}")

    valid = (
        not missing
        and manifest.get("status") == "completed"
        and manifest.get("run_id") == scenario_dir.name
    )

    print(f"  Validation:    {'PASS' if valid else 'FAIL'}")

    return valid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the VANET PER Zenodo dataset."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/datasets/vanet"),
        help="Dataset root inside Docker.",
    )

    args = parser.parse_args()
    root = args.dataset_root

    if not root.is_dir():
        parser.error(f"Dataset directory not found: {root}")

    scenarios = find_scenarios(root)

    print("=" * 70)
    print("VANET PER DATASET INSPECTION")
    print("=" * 70)
    print(f"Dataset root: {root}")
    print(f"Scenario count: {len(scenarios)}")

    passed = 0

    for scenario in scenarios:
        if inspect_scenario(scenario):
            passed += 1

    print("\n" + "=" * 70)
    print(f"Total scenarios inspected: {len(scenarios)}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(scenarios) - passed}")
    print("=" * 70)

    if len(scenarios) != 27 or passed != len(scenarios):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
