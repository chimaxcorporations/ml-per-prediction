#!/usr/bin/env python3
"""Prepare a common 10-history evaluation population for every scenario."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from per_prediction.data.temporal import prepare_temporal_scenario


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("/workspace/data/processed"))
    ap.add_argument("--output-dir", type=Path, default=Path("/workspace/data/temporal_h10"))
    ap.add_argument("--scenario", help="One source filename or stem; default: all scenarios")
    ap.add_argument("--batch-size", type=int, default=100_000)
    args = ap.parse_args()
    if args.batch_size < 10:
        ap.error("--batch-size must be at least 10")
    sources = sorted(args.input_dir.glob("juelich_core_*.parquet"))
    if args.scenario:
        sources = [p for p in sources if p.name == args.scenario or p.stem == args.scenario]
    elif len(sources) != 27:
        raise ValueError(f"Expected 27 source scenarios; found {len(sources)}")
    if not sources:
        raise ValueError("No matching source scenarios")
    for source in sources:
        destination = args.output_dir / source.name
        n = prepare_temporal_scenario(source, destination, batch_size=args.batch_size)
        print(f"{source.name}: {n:,} eligible samples -> {destination}", flush=True)


if __name__ == "__main__":
    main()
