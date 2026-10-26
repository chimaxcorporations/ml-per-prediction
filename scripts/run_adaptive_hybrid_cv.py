#!/usr/bin/env python3
"""Nested-validation adaptive PER prediction, 9 scenario-held-out folds.

Copy alongside scripts/run_scenario_cv.py. Uses its fold construction, sampling,
feature definitions, and streaming metrics. No outer-test labels tune the gate.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_scenario_cv import FEATURES, Metrics, build_folds, sample_rows

THRESHOLDS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
LAGS = [f'per_t_lag{i}' for i in range(10)]


def train(names, all_names, input_dir, n, seed, batch_size):
    frames = [sample_rows(input_dir / name, n, seed + all_names.index(name), batch_size)
              for name in names]
    df = pd.concat(frames, ignore_index=True)
    X = df[FEATURES].to_numpy(dtype=np.float64)
    y = df['target_per'].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('Non-finite training data')
    model = HistGradientBoostingRegressor(max_iter=150, random_state=seed)
    model.fit(X, y)
    return model


def batches(path, batch_size):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size,
                                 columns=FEATURES + ['target_per', 'target_tx']):
        df = batch.to_pandas()
        X = df[FEATURES].to_numpy(dtype=np.float64)
        y = df['target_per'].to_numpy(dtype=np.float64)
        w = df['target_tx'].to_numpy(dtype=np.float64)
        if not (np.isfinite(X).all() and np.isfinite(y).all() and
                np.isfinite(w).all() and (w > 0).all()):
            raise ValueError(f'Invalid data in {path}')
        yield df, X, y, w


def predict_components(model, df, X):
    p = np.clip(df['per_t_lag0'].to_numpy(dtype=np.float64), 0, 1)
    h = np.clip(model.predict(X), 0, 1)
    recent_max = df[LAGS].to_numpy(dtype=np.float64).max(axis=1)
    return p, h, recent_max


def choose_gate(model, validation_names, input_dir, batch_size):
    # Select by aggregate internal-validation RMSE, not outer-test performance.
    choices = ['always_histgb'] + [f'max_per_le_{t:g}' for t in THRESHOLDS]
    sse = np.zeros(len(choices), dtype=np.float64)
    count = 0
    for name in validation_names:
        for df, X, y, _ in batches(input_dir / name, batch_size):
            p, h, recent_max = predict_components(model, df, X)
            sse[0] += np.square(y - h).sum()
            for j, threshold in enumerate(THRESHOLDS, 1):
                pred = np.where(recent_max <= threshold, p, h)
                sse[j] += np.square(y - pred).sum()
            count += len(y)
    # Favor always-HistGB on ties; conservative gating.
    best = int(np.argmin(sse))
    return choices[best], (None if best == 0 else THRESHOLDS[best - 1]), {
        choices[j]: float(np.sqrt(sse[j] / count)) for j in range(len(choices))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, default=Path('/workspace/data/temporal_h10'))
    parser.add_argument('--output-dir', type=Path, default=Path('/workspace/results/adaptive_hybrid_cv_h10'))
    parser.add_argument('--samples-per-scenario', type=int, default=30_000)
    parser.add_argument('--batch-size', type=int, default=100_000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.samples_per_scenario < 1 or args.batch_size < 1:
        parser.error('Samples per scenario and batch size must be positive')
    names = sorted(p.name for p in args.input_dir.glob('juelich_core_*.parquet'))
    if len(names) != 27:
        raise ValueError(f'Expected 27 temporal scenario files; found {len(names)}')
    for lag in LAGS:
        if lag not in FEATURES:
            raise ValueError(f'Missing history feature {lag}; available: {FEATURES}')
    folds = build_folds(names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'folds.json').write_text(json.dumps(folds, indent=2) + '\n')
    models = ('persistence', 'hist_gb', 'hybrid')
    overall = {name: Metrics() for name in models}
    rows, gates = [], []
    for fold in range(9):
        outer_test = folds[fold]
        inner_val = folds[(fold + 1) % 9]
        outer_train = [name for name in names if name not in outer_test]
        inner_train = [name for name in outer_train if name not in inner_val]
        assert len(outer_test) == len(inner_val) == 3 and len(inner_train) == 21
        print(f'\nFold {fold + 1}/9: calibrating on 3 internal-validation scenarios', flush=True)
        calibration_model = train(inner_train, names, args.input_dir,
                                  args.samples_per_scenario, args.seed, args.batch_size)
        gate_name, threshold, validation_rmse = choose_gate(
            calibration_model, inner_val, args.input_dir, args.batch_size)
        del calibration_model
        print(f'  gate={gate_name}; internal RMSE={validation_rmse[gate_name]:.6f}', flush=True)
        gates.append(dict(fold=fold + 1, gate=gate_name, threshold=threshold,
                          inner_validation_scenarios=inner_val,
                          internal_rmse=validation_rmse))
        # Retrain on all 24 outer-training scenarios; no outer-test leakage.
        final_model = train(outer_train, names, args.input_dir,
                            args.samples_per_scenario, args.seed, args.batch_size)
        fold_metrics = {name: Metrics() for name in models}
        for scenario in outer_test:
            accum = {name: Metrics() for name in models}
            for df, X, y, w in batches(args.input_dir / scenario, args.batch_size):
                p, h, recent_max = predict_components(final_model, df, X)
                hybrid = h if threshold is None else np.where(recent_max <= threshold, p, h)
                preds = {'persistence': p, 'hist_gb': h, 'hybrid': hybrid}
                for name, pred in preds.items():
                    accum[name].update(y, pred, w)
                    fold_metrics[name].update(y, pred, w)
                    overall[name].update(y, pred, w)
            for name in models:
                rows.append(dict(level='scenario', fold=fold + 1, scenario=scenario,
                                 model=name, **accum[name].result()))
            print(f'  evaluated {scenario}', flush=True)
        for name in models:
            rows.append(dict(level='fold', fold=fold + 1, scenario='__fold__',
                             model=name, **fold_metrics[name].result()))
        pd.DataFrame(rows).to_csv(args.output_dir / 'metrics_partial.csv', index=False)
        (args.output_dir / 'gates_partial.json').write_text(json.dumps(gates, indent=2) + '\n')
        del final_model
    for name in models:
        rows.append(dict(level='overall', fold=0, scenario='__pooled__',
                         model=name, **overall[name].result()))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / 'metrics.csv', index=False)
    (args.output_dir / 'gates.json').write_text(json.dumps(gates, indent=2) + '\n')
    pivot = metrics[metrics.level == 'scenario'].pivot(index='scenario', columns='model', values='rmse')
    gains = pd.DataFrame({
        'hist_gb_gain_pct': 100 * (1 - pivot.hist_gb / pivot.persistence),
        'hybrid_gain_pct': 100 * (1 - pivot.hybrid / pivot.persistence),
        'hybrid_vs_histgb_pct': 100 * (1 - pivot.hybrid / pivot.hist_gb),
    })
    gains.to_csv(args.output_dir / 'scenario_gains.csv')
    print('\n=== OVERALL OUT-OF-FOLD RESULTS ===')
    print(metrics[metrics.level == 'overall'].drop(columns=['level', 'fold', 'scenario']).to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print('\n=== SCENARIO-LEVEL RMSE IMPROVEMENTS ===')
    print(gains.agg(['mean', 'median', 'std', 'min', 'max']).to_string(float_format=lambda x: f'{x:.3f}'))
    print('HistGB wins:', int((gains.hist_gb_gain_pct > 0).sum()), '/27')
    print('Hybrid wins:', int((gains.hybrid_gain_pct > 0).sum()), '/27')
    print('Hybrid beats HistGB:', int((gains.hybrid_vs_histgb_pct > 0).sum()), '/27')
    print('Saved:', args.output_dir / 'metrics.csv')

if __name__ == '__main__':
    main()
