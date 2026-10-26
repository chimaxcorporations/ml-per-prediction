#!/usr/bin/env python3
"""Create leakage-safe, consecutive H=10 Berlin PER samples per scenario."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

BASE = ['per_t', 'dist_m', 'v_src_mps', 'v_dst_mps']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='/workspace/data/berlin/sidelink_dataframe.parquet')
    ap.add_argument('--output-dir', default='/workspace/data/berlin/temporal_h10')
    ap.add_argument('--history', type=int, default=10)
    args = ap.parse_args()
    if args.history < 1:
        ap.error('--history must be positive')
    columns = ['Scenario', 'Source', 'Destination', 'timestamp', 'Packet_error_ratio',
               'distance', 'speed_kmh_source', 'speed_kmh_destination',
               'Packet_transmission_rate_hz', 'Received Packets']
    df = pd.read_parquet(args.input, columns=columns)
    if 'timestamp' not in df.columns:
        df = df.reset_index()
    df = df.rename(columns={'Packet_error_ratio':'per_t', 'distance':'dist_m',
        'speed_kmh_source':'v_src_mps', 'speed_kmh_destination':'v_dst_mps',
        'Packet_transmission_rate_hz':'tx_count'})
    df['v_src_mps'] /= 3.6
    df['v_dst_mps'] /= 3.6
    keys = ['Scenario', 'Source', 'Destination']
    df = df.sort_values(keys + ['timestamp']).reset_index(drop=True)
    if df.duplicated(keys + ['timestamp']).any():
        raise ValueError('Duplicate scenario-link-timestamp records')
    valid = (df['per_t'].between(0,1) & (df['Received Packets'] >= 0) &
             (df['Received Packets'] <= df['tx_count']) & (df['tx_count'] > 0))
    for c in BASE:
        valid &= np.isfinite(df[c])
    print('Input rows:', len(df), 'invalid rows:', int((~valid).sum()))
    # A new segment starts at any invalid row, missing second, or link change.
    gap = df.groupby(keys, sort=False)['timestamp'].diff().dt.total_seconds()
    changed = df[keys].ne(df[keys].shift()).any(axis=1)
    start = changed | gap.ne(1) | (~valid) | (~valid.shift(fill_value=False))
    df['_segment'] = start.cumsum()
    clean = df.loc[valid].copy()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    total = 0
    for scenario, sdf in clean.groupby('Scenario', sort=True):
        parts = []
        for _, group in sdf.groupby('_segment', sort=False):
            n = len(group)
            if n <= args.history:
                continue
            arrays = {c: group[c].to_numpy(dtype=np.float32) for c in BASE}
            rows = {}
            # lag0 is latest observed time; target is immediately following second.
            for lag in range(args.history):
                for c in BASE:
                    rows[f'{c}_lag{lag}'] = arrays[c][args.history - 1 - lag:n - 1 - lag]
            rows['target_per'] = arrays['per_t'][args.history:]
            rows['target_tx'] = group['tx_count'].to_numpy(dtype=np.float32)[args.history:]
            rows['Scenario'] = np.repeat(scenario, n - args.history)
            rows['Source'] = group['Source'].to_numpy()[args.history:]
            rows['Destination'] = group['Destination'].to_numpy()[args.history:]
            rows['timestamp'] = group['timestamp'].to_numpy()[args.history:]
            parts.append(pd.DataFrame(rows))
        if not parts:
            print(f'{scenario}: no eligible samples')
            continue
        output = pd.concat(parts, ignore_index=True)
        path = outdir / f'berlin_{scenario}_h{args.history}.parquet'
        output.to_parquet(path, index=False)
        total += len(output)
        print(f'{scenario}: {len(output):,} samples -> {path}')
    print('Total eligible prediction samples:', f'{total:,}')

if __name__ == '__main__':
    main()
