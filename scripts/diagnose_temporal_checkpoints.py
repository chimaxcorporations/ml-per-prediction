#!/usr/bin/env python3
"""Inference-only diagnostics for saved common-40 GRU/LSTM/TCN/Transformer checkpoints.

Import architectures and feature ordering from scripts/run_temporal_transfer.py.
Exact aggregate and binned statistics; sampled absolute-error quantiles.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_temporal_transfer import CLASSES, FEATURES, COLS, PATTERN

MODELS = ('gru', 'lstm', 'tcn', 'transformer')
DIRECTIONS = ('juelich_to_berlin', 'berlin_to_juelich')
PER_EDGES = np.array([0, .01, .05, .1, .25, .5, .75, .9, 1.00000001])
DIST_EDGES = np.array([0, 25, 50, 100, 200, 300, 400, 500, np.inf])


class Accumulator:
    def __init__(self):
        self.n = 0
        self.sy = self.sy2 = self.sp = self.signed = 0.
        self.ae = self.se = self.w = self.wae = self.wse = 0.

    def update(self, y, p, w):
        e = p - y
        self.n += len(y)
        self.sy += float(np.sum(y))
        self.sy2 += float(np.dot(y, y))
        self.sp += float(np.sum(p))
        self.signed += float(np.sum(e))
        self.ae += float(np.abs(e).sum())
        self.se += float(np.dot(e, e))
        self.w += float(w.sum())
        self.wae += float(np.dot(w, np.abs(e)))
        self.wse += float(np.dot(w, e * e))

    def result(self):
        if not self.n:
            return None
        sst = self.sy2 - self.sy**2 / self.n
        return dict(samples=self.n, total_tx=self.w,
                    mean_true_per=self.sy/self.n, mean_pred_per=self.sp/self.n,
                    bias=self.signed/self.n, mae=self.ae/self.n,
                    rmse=(self.se/self.n)**.5,
                    r2=1-self.se/sst if sst>0 else np.nan,
                    weighted_mae=self.wae/self.w,
                    weighted_rmse=(self.wse/self.w)**.5)


def bin_label(edges, idx):
    a, b = edges[idx:idx+2]
    return f'[{a:g}, {b:g})'


def load_checkpoint(path, expected_model, expected_features):
    # Checkpoints were generated locally by the trusted training pipeline.
    # weights_only=False may execute pickle code; never use on untrusted files.
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if ckpt.get('model') != expected_model:
        raise ValueError(f'{path}: model identifier mismatch')
    if list(ckpt.get('features', [])) != list(expected_features):
        raise ValueError(f'{path}: feature order mismatch')
    mean = np.asarray(ckpt['scaler_mean'], dtype=np.float64)
    scale = np.asarray(ckpt['scaler_scale'], dtype=np.float64)
    if mean.shape != (40,) or scale.shape != (40,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError(f'{path}: invalid scaler arrays')
    model = CLASSES[expected_model]()
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    return model, mean, scale


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint-dir', type=Path, default=Path('/workspace/results/temporal_transfer_v1/models'))
    ap.add_argument('--benchmark-metrics', type=Path, default=Path('/workspace/results/temporal_transfer_v1/metrics.csv'))
    ap.add_argument('--juelich-dir', type=Path, default=Path('/workspace/data/temporal_h10'))
    ap.add_argument('--berlin-dir', type=Path, default=Path('/workspace/data/berlin/temporal_h10'))
    ap.add_argument('--output-dir', type=Path, default=Path('/workspace/results/temporal_checkpoint_diagnostics_v1'))
    ap.add_argument('--direction', choices=('both',)+DIRECTIONS, default='both')
    ap.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    ap.add_argument('--eval-batch-size', type=int, default=50000)
    ap.add_argument('--nn-batch-size', type=int, default=8192)
    ap.add_argument('--sample-fraction', type=float, default=.01)
    ap.add_argument('--seed', type=int, default=4242)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--rmse-tolerance', type=float, default=1e-5)
    args = ap.parse_args()
    if args.eval_batch_size < 1 or args.nn_batch_size < 1 or args.threads < 1 or not (0 < args.sample_fraction <= 1):
        ap.error('Invalid batch size, threads or sample fraction')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        ap.error('Output directory is not empty; choose a new directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    directions = DIRECTIONS if args.direction == 'both' else (args.direction,)
    files_j = sorted(p for p in args.juelich_dir.glob('juelich_core_*.parquet') if PATTERN.fullmatch(p.name))
    if len(files_j) != 27:
        raise ValueError(f'Expected 27 Juelich scenario files, found {len(files_j)}')
    files_b = [args.berlin_dir / f'berlin_{s}_h10.parquet' for s in ('S1','S2')]
    if not all(p.is_file() for p in files_b):
        raise FileNotFoundError('Berlin scenario files missing')
    reference = pd.read_csv(args.benchmark_metrics)
    summary_rows, binned_rows, quantile_rows, verification_rows = [], [], [], []
    for direction in directions:
        files = files_b if direction == 'juelich_to_berlin' else files_j
        models = {}
        for name in args.models:
            ckpt_path = args.checkpoint_dir / f'{direction}_{name}.pt'
            models[name] = load_checkpoint(ckpt_path, name, FEATURES)
            print('Loaded', ckpt_path, flush=True)
        overall = {name: Accumulator() for name in args.models}
        per_bins = {(name, i): Accumulator() for name in args.models for i in range(len(PER_EDGES)-1)}
        dist_bins = {(name, i): Accumulator() for name in args.models for i in range(len(DIST_EDGES)-1)}
        # Sampled quantiles only: independent Bernoulli sampling, deterministic per direction.
        rng = np.random.default_rng(args.seed + (0 if direction == DIRECTIONS[0] else 1))
        sampled_errors = {name: [] for name in args.models}
        for path in files:
            scenario = {name: Accumulator() for name in args.models}
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=args.eval_batch_size, columns=COLS):
                df = batch.to_pandas()
                X = df[FEATURES].to_numpy(dtype=np.float64)
                y = df['target_per'].to_numpy(dtype=np.float64)
                w = df['target_tx'].to_numpy(dtype=np.float64)
                d = df['dist_m_lag0'].to_numpy(dtype=np.float64)
                if not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isfinite(w).all() or (w <= 0).any() or (y < 0).any() or (y > 1).any():
                    raise ValueError(f'Invalid evaluation data in {path}')
                pb = np.searchsorted(PER_EDGES, y, side='right') - 1
                db = np.searchsorted(DIST_EDGES, d, side='right') - 1
                sample_mask = rng.random(len(y)) < args.sample_fraction
                for name, (model, mean, scale) in models.items():
                    preds = np.empty(len(y), dtype=np.float64)
                    with torch.inference_mode():
                        for start in range(0, len(y), args.nn_batch_size):
                            end = min(start + args.nn_batch_size, len(y))
                            z = np.ascontiguousarray(((X[start:end] - mean) / scale).astype(np.float32))
                            preds[start:end] = model(torch.from_numpy(z)).numpy().astype(np.float64)
                    np.clip(preds, 0, 1, out=preds)
                    scenario[name].update(y, preds, w)
                    overall[name].update(y, preds, w)
                    for i in np.unique(pb):
                        if 0 <= i < len(PER_EDGES)-1:
                            mask = pb == i
                            per_bins[name, int(i)].update(y[mask], preds[mask], w[mask])
                    for i in np.unique(db):
                        if 0 <= i < len(DIST_EDGES)-1:
                            mask = db == i
                            dist_bins[name, int(i)].update(y[mask], preds[mask], w[mask])
                    if sample_mask.any():
                        sampled_errors[name].append(np.abs(preds[sample_mask] - y[sample_mask]).astype(np.float32))
            for name in args.models:
                summary_rows.append(dict(direction=direction, level='scenario', scenario=path.stem, model=name, **scenario[name].result()))
            print(f'{direction}: evaluated {path.name} ({pf.metadata.num_rows:,} rows)', flush=True)
        for name in args.models:
            summary_rows.append(dict(direction=direction, level='overall', scenario='ALL', model=name, **overall[name].result()))
            for i in range(len(PER_EDGES)-1):
                result = per_bins[name, i].result()
                if result:
                    binned_rows.append(dict(direction=direction, model=name, variable='true_per', bin=bin_label(PER_EDGES,i), **result))
            for i in range(len(DIST_EDGES)-1):
                result = dist_bins[name, i].result()
                if result:
                    binned_rows.append(dict(direction=direction, model=name, variable='distance_m', bin=bin_label(DIST_EDGES,i), **result))
            errs = np.concatenate(sampled_errors[name]) if sampled_errors[name] else np.array([])
            if len(errs):
                quantile_rows.append(dict(direction=direction, model=name, sampled_rows=len(errs),
                    sample_fraction=args.sample_fraction, median_ae=float(np.quantile(errs,.5)),
                    q90_ae=float(np.quantile(errs,.9)), q95_ae=float(np.quantile(errs,.95)),
                    q99_ae=float(np.quantile(errs,.99))))
            ref = reference[(reference.direction == direction) & (reference.model == name) & (reference.level == 'overall')]
            if len(ref) != 1:
                raise ValueError(f'Expected one reference metric row for {direction}/{name}; found {len(ref)}')
            rmse = overall[name].result()['rmse']
            old = float(ref.iloc[0]['rmse'])
            diff = rmse - old
            verification_rows.append(dict(direction=direction, model=name, benchmark_rmse=old,
                                          reproduced_rmse=rmse, difference=diff, passed=abs(diff) <= args.rmse_tolerance))
            print(f'VALIDATION {direction} {name}: reproduced={rmse:.8f}, original={old:.8f}, delta={diff:+.3g}',flush=True)
    pd.DataFrame(summary_rows).to_csv(args.output_dir/'error_summary.csv',index=False)
    pd.DataFrame(binned_rows).to_csv(args.output_dir/'error_by_bin.csv',index=False)
    pd.DataFrame(quantile_rows).to_csv(args.output_dir/'sample_error_quantiles.csv',index=False)
    pd.DataFrame(verification_rows).to_csv(args.output_dir/'benchmark_verification.csv',index=False)
    (args.output_dir/'settings.json').write_text(json.dumps({
        'direction':args.direction,'models':args.models,'eval_batch_size':args.eval_batch_size,
        'nn_batch_size':args.nn_batch_size,'sample_fraction':args.sample_fraction,
        'seed':args.seed,'rmse_tolerance':args.rmse_tolerance,
        'quantiles':'reproducible Bernoulli sampled, not exact full-population quantiles',
        'bias_definition':'prediction minus ground truth',
        'checkpoint_warning':'Only load locally generated trusted PyTorch checkpoints'
    },indent=2)+'\n')
    checks = pd.DataFrame(verification_rows)
    print('\n=== BENCHMARK VERIFICATION ===')
    print(checks.to_string(index=False))
    print('\n=== OVERALL ERROR SUMMARY ===')
    print(pd.DataFrame(summary_rows).query("level == 'overall'").to_string(index=False))
    print('\nSaved to:', args.output_dir)
    if not checks.passed.all():
        raise SystemExit('WARNING: At least one reproduced RMSE differs from benchmark beyond tolerance. Do not interpret diagnostics until resolved.')

if __name__ == '__main__':
    main()
