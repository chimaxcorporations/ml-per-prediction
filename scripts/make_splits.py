#!/usr/bin/env python3
"""Deterministic, balanced, scenario-held-out split manifest."""
import argparse
import json
import re
from pathlib import Path

PATTERN = re.compile(r'^juelich_core_d(10|30|60)_r(2|5|10)_(log_distance|log_distance_nakagami|urban)_s01_w1$')
PLAN = {
    'log_distance': {'validation': [(10, 2)], 'test': [(30, 5), (60, 10)]},
    'log_distance_nakagami': {'validation': [(30, 5)], 'test': [(60, 2), (10, 10)]},
    'urban': {'validation': [(60, 10)], 'test': [(10, 5), (30, 2)]},
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', type=Path, default=Path('/workspace/data/processed'))
    ap.add_argument('--output', type=Path, default=Path('/workspace/configs/scenario_splits.json'))
    args = ap.parse_args()
    files = sorted(args.input_dir.glob('juelich_core_*.parquet'))
    if len(files) != 27:
        raise ValueError(f'Expected 27 files, found {len(files)}')
    splits = {'train': [], 'validation': [], 'test': []}
    seen = set()
    for file in files:
        m = PATTERN.fullmatch(file.stem)
        if not m:
            raise ValueError(f'Unexpected scenario name: {file.stem}')
        density, rate, prop = m.groups()
        combination = (int(density), int(rate))
        if file.stem in seen:
            raise ValueError(f'Duplicate scenario: {file.stem}')
        seen.add(file.stem)
        group = 'train'
        for candidate in ('validation', 'test'):
            if combination in PLAN[prop][candidate]:
                group = candidate
        splits[group].append(file.name)
    expected = {'train': 18, 'validation': 3, 'test': 6}
    for name, count in expected.items():
        if len(splits[name]) != count:
            raise ValueError(f'{name}: expected {count}, got {len(splits[name])}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(splits, indent=2) + '\n')
    for name, paths in splits.items():
        print(f'{name}: {len(paths)} scenarios')
        for path in paths:
            print(' ', path)
    print('Saved:', args.output)

if __name__ == '__main__':
    main()
