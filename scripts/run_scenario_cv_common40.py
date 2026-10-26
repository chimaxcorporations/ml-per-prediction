#!/usr/bin/env python3
"""Nine-fold scenario-held-out CV for H=10 common 40-feature PER prediction.

Each fold holds out three full scenarios (one of each density, rate, and
propagation model). All 27 scenarios are evaluated exactly once. Fixed model
hyperparameters; no hyperparameter selection on held-out folds.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from per_prediction.data.temporal import feature_columns

COMMON_FEATURES = [f'{name}_lag{lag}' for lag in range(10) for name in ('per_t', 'dist_m', 'v_src_mps', 'v_dst_mps')]
FEATURES = COMMON_FEATURES
PATTERN = re.compile(r'^juelich_core_d(10|30|60)_r(2|5|10)_(log_distance|log_distance_nakagami|urban)_s\d+_w\d+\.parquet$')
DENSITIES = (10, 30, 60)
RATES = (2, 5, 10)
PROPS = ('log_distance', 'log_distance_nakagami', 'urban')


class Metrics:
    def __init__(self):
        self.n = 0
        self.y = self.y2 = self.ae = self.se = 0.0
        self.w = self.wae = self.wse = 0.0

    def update(self, target, pred, weight):
        err = target - pred
        self.n += len(target)
        self.y += float(np.sum(target))
        self.y2 += float(np.dot(target, target))
        self.ae += float(np.sum(np.abs(err)))
        self.se += float(np.dot(err, err))
        self.w += float(np.sum(weight))
        self.wae += float(np.dot(weight, np.abs(err)))
        self.wse += float(np.dot(weight, err * err))

    def result(self):
        if not self.n or self.w <= 0:
            raise ValueError('Empty metrics accumulator')
        sst = self.y2 - self.y * self.y / self.n
        return dict(samples=self.n, total_tx=self.w, mae=self.ae / self.n,
                    rmse=np.sqrt(self.se / self.n),
                    r2=1 - self.se / sst if sst > 1e-12 else np.nan,
                    weighted_mae=self.wae / self.w,
                    weighted_rmse=np.sqrt(self.wse / self.w))


def build_folds(names):
    """Two orthogonal modular constraints yield nine balanced folds."""
    folds = {i: [] for i in range(9)}
    for name in names:
        match = PATTERN.fullmatch(name)
        if not match:
            raise ValueError(f'Unexpected filename: {name}')
        density, rate, prop = match.groups()
        i = DENSITIES.index(int(density))
        j = RATES.index(int(rate))
        k = PROPS.index(prop)
        fold = (i + j) % 3 + 3 * ((i + k) % 3)
        folds[fold].append(name)
    for fold, held_out in folds.items():
        if len(held_out) != 3:
            raise ValueError(f'Fold {fold}: expected 3 scenarios, got {len(held_out)}')
        values = [PATTERN.fullmatch(x).groups() for x in held_out]
        if any(len({v[c] for v in values}) != 3 for c in range(3)):
            raise ValueError(f'Fold {fold} is not balanced')
    if len({x for fold in folds.values() for x in fold}) != 27:
        raise ValueError('Scenarios missing or duplicated across folds')
    return folds


def sample_rows(path, count, seed, batch_size):
    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows
    if n < 1:
        raise ValueError(f'Empty scenario: {path}')
    selected = np.sort(np.random.default_rng(seed).choice(n, min(n, count), replace=False))
    frames = []
    offset = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=FEATURES + ['target_per']):
        stop = offset + batch.num_rows
        left = np.searchsorted(selected, offset)
        right = np.searchsorted(selected, stop)
        if right > left:
            frame = batch.to_pandas()
            frames.append(frame.iloc[selected[left:right] - offset].copy())
        offset = stop
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', type=Path, default=Path('/workspace/data/temporal_h10'))
    ap.add_argument('--output-dir', type=Path, default=Path('/workspace/results/juelich_cv_h10_common40/classical'))
    ap.add_argument('--samples-per-scenario', type=int, default=30_000)
    ap.add_argument('--batch-size', type=int, default=100_000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save-models', action='store_true', help='Save 18 fitted models (can be large)')
    args = ap.parse_args()
    if args.samples_per_scenario < 1 or args.batch_size < 1:
        ap.error('Samples per scenario and batch size must be positive')

    names = sorted(p.name for p in args.input_dir.glob('juelich_core_*.parquet'))
    if len(names) != 27:
        raise ValueError(f'Expected 27 temporal Parquet files, found {len(names)}')
    folds = build_folds(names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'folds.json').write_text(json.dumps({str(k): v for k, v in folds.items()}, indent=2) + '\n')
    if args.save_models:
        (args.output_dir / 'models').mkdir(exist_ok=True)

    rows = []
    overall = {m: Metrics() for m in ('persistence', 'ridge', 'hist_gb')}
    for fold in range(9):
        test_names = folds[fold]
        train_names = [n for n in names if n not in set(test_names)]
        print(f'\nFold {fold + 1}/9: 24 train scenarios, 3 held out', flush=True)
        for name in test_names:
            print('  HOLD OUT:', name, flush=True)

        frames = []
        # Seed depends on scenario identity, not fold membership or iteration order.
        for name in train_names:
            scenario_index = names.index(name)
            frames.append(sample_rows(args.input_dir / name, args.samples_per_scenario,
                                      args.seed + scenario_index, args.batch_size))
        training = pd.concat(frames, ignore_index=True)
        del frames
        X = training[FEATURES].to_numpy(dtype=np.float64)
        y = training['target_per'].to_numpy(dtype=np.float64)
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError(f'Non-finite training values in fold {fold}')
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        ridge.fit(X, y)
        hist_gb = HistGradientBoostingRegressor(max_iter=150, random_state=args.seed)
        hist_gb.fit(X, y)
        del training, X, y
        if args.save_models:
            joblib.dump(ridge, args.output_dir / 'models' / f'ridge_fold{fold + 1}.joblib')
            joblib.dump(hist_gb, args.output_dir / 'models' / f'hist_gb_fold{fold + 1}.joblib')
        fold_acc = {m: Metrics() for m in overall}
        for name in test_names:
            scenario_acc = {m: Metrics() for m in overall}
            pf = pq.ParquetFile(args.input_dir / name)
            cols = FEATURES + ['target_per', 'target_tx']
            for batch in pf.iter_batches(batch_size=args.batch_size, columns=cols):
                frame = batch.to_pandas()
                X_eval = frame[FEATURES].to_numpy(dtype=np.float64)
                target = frame['target_per'].to_numpy(dtype=np.float64)
                weight = frame['target_tx'].to_numpy(dtype=np.float64)
                if not (np.isfinite(X_eval).all() and np.isfinite(target).all() and
                        np.isfinite(weight).all() and (weight > 0).all()):
                    raise ValueError(f'Invalid evaluation data: {name}')
                predictions = {
                    'persistence': frame['per_t_lag0'].to_numpy(dtype=np.float64),
                    'ridge': np.clip(ridge.predict(X_eval), 0, 1),
                    'hist_gb': np.clip(hist_gb.predict(X_eval), 0, 1),
                }
                for model, prediction in predictions.items():
                    scenario_acc[model].update(target, prediction, weight)
                    fold_acc[model].update(target, prediction, weight)
                    overall[model].update(target, prediction, weight)
            for model, acc in scenario_acc.items():
                rows.append(dict(level='scenario', fold=fold + 1, scenario=name,
                                 model=model, **acc.result()))
            print(f'  evaluated {name}: {pf.metadata.num_rows:,} rows', flush=True)
        for model, acc in fold_acc.items():
            rows.append(dict(level='fold', fold=fold + 1, scenario='__fold__',
                             model=model, **acc.result()))
        pd.DataFrame(rows).to_csv(args.output_dir / 'metrics_partial.csv', index=False)
        del ridge, hist_gb

    for model, acc in overall.items():
        rows.append(dict(level='overall', fold=0, scenario='__pooled__',
                         model=model, **acc.result()))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / 'metrics.csv', index=False)
    scenario = metrics[metrics.level == 'scenario']
    pivot = scenario.pivot(index='scenario', columns='model', values='rmse')
    gains = pd.DataFrame({
        'ridge_gain_pct': 100 * (1 - pivot['ridge'] / pivot['persistence']),
        'hist_gb_gain_pct': 100 * (1 - pivot['hist_gb'] / pivot['persistence']),
    })
    gains.to_csv(args.output_dir / 'scenario_gains.csv')
    print('\n=== OVERALL OUT-OF-FOLD RESULTS ===', flush=True)
    print(metrics[metrics.level == 'overall'].drop(columns=['level', 'fold', 'scenario']).to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print('\n=== SCENARIO-LEVEL RMSE IMPROVEMENTS ===', flush=True)
    print(gains.agg(['mean', 'median', 'std', 'min', 'max']).to_string(float_format=lambda x: f'{x:.3f}'))
    print('Ridge wins:', int((gains.ridge_gain_pct > 0).sum()), '/27')
    print('HistGB wins:', int((gains.hist_gb_gain_pct > 0).sum()), '/27')
    print('Saved:', args.output_dir / 'metrics.csv', flush=True)


if __name__ == '__main__':
    main()
