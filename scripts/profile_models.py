#!/usr/bin/env python3
"""Profile unquantized Ridge and HistGB H=10 for VANET PER prediction.

Uses the same features and hyperparameters as scripts/run_scenario_cv.py.
No FPGA synthesis is performed: hardware numbers are analytical estimates.
"""
import argparse
import json
import re
import sys
import time
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

PATTERN = re.compile(r'^juelich_core_d(10|30|60)_r(2|5|10)_(log_distance|log_distance_nakagami|urban)_s\d+_w\d+\.parquet$')
DENSITIES = (10, 30, 60)
RATES = (2, 5, 10)
PROPS = ('log_distance', 'log_distance_nakagami', 'urban')


def folds_for(names):
    folds = {i: [] for i in range(9)}
    for name in names:
        m = PATTERN.fullmatch(name)
        if not m:
            raise ValueError(f'Unexpected scenario name: {name}')
        d, r, p = m.groups()
        i, j, k = DENSITIES.index(int(d)), RATES.index(int(r)), PROPS.index(p)
        folds[(i+j) % 3 + 3*((i+k) % 3)].append(name)
    if any(len(v) != 3 for v in folds.values()):
        raise ValueError('Expected 27 complete factorial scenarios')
    return folds


def load_split(path, names):
    """Accept common split JSON formats: {train:[...]}, {splits:{train:[...]}}."""
    obj = json.loads(path.read_text())
    if isinstance(obj, dict) and 'splits' in obj:
        obj = obj['splits']
    if not isinstance(obj, dict) or 'train' not in obj:
        raise ValueError('Split JSON must contain a train list (or splits.train)')
    values = obj['train']
    if not isinstance(values, list):
        raise ValueError('train must be a list')
    train = [Path(str(v)).name for v in values]
    if not train or len(train) != len(set(train)) or not set(train).issubset(names):
        raise ValueError('Invalid train scenario names in split JSON')
    return sorted(train)


def sample_rows(path, count, seed, batch_size, features):
    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows
    if not n:
        raise ValueError(f'Empty Parquet file: {path}')
    selected = np.sort(np.random.default_rng(seed).choice(n, min(n, count), replace=False))
    chunks = []
    offset = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=features + ['target_per']):
        stop = offset + batch.num_rows
        left, right = np.searchsorted(selected, [offset, stop])
        if right > left:
            chunks.append(batch.to_pandas().iloc[selected[left:right] - offset].copy())
        offset = stop
    return pd.concat(chunks, ignore_index=True)


def hist_structure(model):
    # sklearn HistGradientBoostingRegressor internal predictors; read-only profiling.
    trees = [p for stage in model._predictors for p in stage]
    nodes = leaves = max_depth = 0
    comparisons_by_tree = []
    feature_uses = {}
    for predictor in trees:
        arr = predictor.nodes
        nodes += len(arr)
        is_leaf = arr['is_leaf'].astype(bool)
        leaves += int(is_leaf.sum())
        for f in arr['feature_idx'][~is_leaf]:
            key = str(int(f))
            feature_uses[key] = feature_uses.get(key, 0) + 1
        # Depth computed by traversing child indices (avoids version-specific depth API).
        stack = [(0, 0)]
        depth = 0
        while stack:
            idx, d = stack.pop()
            depth = max(depth, d)
            if not is_leaf[idx]:
                stack.append((int(arr['left'][idx]), d+1))
                stack.append((int(arr['right'][idx]), d+1))
        max_depth = max(max_depth, depth)
        comparisons_by_tree.append(depth)
    return dict(n_trees=len(trees), n_nodes=nodes, n_leaves=leaves,
                n_internal_nodes=nodes-leaves, max_depth=max_depth,
                sum_tree_max_depths=sum(comparisons_by_tree),
                feature_split_counts=feature_uses)


def benchmark(model, X, repeats, warmups, single_count):
    # CPU wall-clock measurements, not FPGA latency; per-batch includes Python overhead.
    for _ in range(warmups):
        model.predict(X)
    batch_times = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        model.predict(X)
        batch_times.append((time.perf_counter_ns() - start) / 1e6)
    one = X[:1]
    for _ in range(warmups):
        model.predict(one)
    single_times = []
    for _ in range(single_count):
        start = time.perf_counter_ns()
        model.predict(one)
        single_times.append((time.perf_counter_ns() - start) / 1e3)
    return dict(batch_size=len(X), batch_ms_median=float(np.median(batch_times)),
                batch_ms_p95=float(np.percentile(batch_times, 95)),
                per_prediction_us_amortized=float(np.median(batch_times)*1000/len(X)),
                single_call_us_median=float(np.median(single_times)),
                single_call_us_p95=float(np.percentile(single_times, 95)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', type=Path, default=Path('/workspace/data/temporal_h10'))
    ap.add_argument('--output-dir', type=Path, default=Path('/workspace/results/model_profile_h10'))
    ap.add_argument('--split-json', type=Path, default=None,
                    help='Optional existing split JSON with train scenario names; otherwise use CV fold 1 training scenarios')
    ap.add_argument('--cv-fold', type=int, choices=range(1, 10), default=1)
    ap.add_argument('--samples-per-scenario', type=int, default=30000)
    ap.add_argument('--batch-size', type=int, default=100000)
    ap.add_argument('--latency-batch-size', type=int, default=10000)
    ap.add_argument('--repeats', type=int, default=15)
    ap.add_argument('--single-count', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save-models', action='store_true')
    args = ap.parse_args()
    if min(args.samples_per_scenario, args.batch_size, args.latency_batch_size,
           args.repeats, args.single_count) < 1:
        ap.error('All sample and repetition counts must be positive')
    features = feature_columns(10)
    names = sorted(p.name for p in args.input_dir.glob('juelich_core_*.parquet'))
    if len(names) != 27:
        raise ValueError(f'Expected 27 temporal scenarios, got {len(names)}')
    if args.split_json:
        train_names = load_split(args.split_json, names)
        training_source = str(args.split_json)
    else:
        holdout = folds_for(names)[args.cv_fold-1]
        train_names = [n for n in names if n not in holdout]
        training_source = f'cv_fold_{args.cv_fold}_train_24_scenarios'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Training {len(train_names)} scenarios; {len(features)} features; source={training_source}', flush=True)
    frames = []
    for name in train_names:
        frame = sample_rows(args.input_dir/name, args.samples_per_scenario,
                            args.seed+names.index(name), args.batch_size, features)
        frames.append(frame)
        print(f'  {name}: {len(frame):,} training rows', flush=True)
    training = pd.concat(frames, ignore_index=True)
    del frames
    X = training[features].to_numpy(dtype=np.float64)
    y = training['target_per'].to_numpy(dtype=np.float64)
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError('Non-finite training data')
    models = {
        'ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        'hist_gb': HistGradientBoostingRegressor(max_iter=150, random_state=args.seed),
    }
    for name, model in models.items():
        print(f'Fitting {name} ...', flush=True)
        model.fit(X, y)
    # Keep the latency workload fixed, with exact same data for both models.
    rng = np.random.default_rng(args.seed + 12345)
    idx = rng.choice(len(X), min(len(X), args.latency_batch_size), replace=False)
    X_bench = np.ascontiguousarray(X[idx])
    profile = dict(training_source=training_source, n_train_scenarios=len(train_names),
                   training_scenarios=train_names, n_training_rows=len(y),
                   features=features, n_features=len(features),
                   training_sample_seed=args.seed,
                   numeric_representation='float64 sklearn CPU models (not FPGA implementation)',
                   models={})
    for name, model in models.items():
        path = args.output_dir / f'{name}_h10.joblib'
        joblib.dump(model, path, compress=0)
        structure = {}
        if name == 'ridge':
            scaler, estimator = model.steps[0][1], model.steps[1][1]
            structure = dict(n_coefficients=int(estimator.coef_.size), intercept_count=1,
                             scaler_mean_count=int(scaler.mean_.size),
                             scaler_scale_count=int(scaler.scale_.size),
                             # naive sequential float64 operations; compiler can fold scaling.
                             naive_mul_count_per_prediction=2*len(features),
                             naive_add_count_per_prediction=2*len(features),
                             fused_linear_mul_count_per_prediction=len(features),
                             fused_linear_add_count_per_prediction=len(features))
        else:
            structure = hist_structure(model)
        timing = benchmark(model, X_bench, args.repeats, warmups=3,
                           single_count=args.single_count)
        profile['models'][name] = dict(serialized_bytes=path.stat().st_size,
                                       structure=structure, cpu_latency=timing)
        print(f'{name}: {path.stat().st_size:,} bytes, '
              f'batch median {timing["batch_ms_median"]:.3f} ms, '
              f'single median {timing["single_call_us_median"]:.2f} us', flush=True)
        if not args.save_models:
            path.unlink()
    tree = profile['models']['hist_gb']['structure']
    nf = len(features)
    # Illustrative, optimistic storage only: each internal node holds one 64-bit
    # threshold, a 16-bit feature index, two 32-bit child indices; each leaf one
    # 64-bit output. Ignores node tags, alignment, missing-value routing, binning,
    # feature preprocessing, and hardware buffering. Not an FPGA utilization report.
    internal_bits = 64 + 16 + 32 + 32
    leaf_bits = 64
    node_bits = tree['n_internal_nodes']*internal_bits + tree['n_leaves']*leaf_bits
    ridge_bits = (3*nf + 1)*64  # coefficients, intercept, scaler means/scales
    profile['fpga_analytical'] = dict(
        precision_bits=64,
        ridge_parameter_storage_bits=ridge_bits,
        ridge_parameter_storage_kib=ridge_bits/8/1024,
        ridge_fused_linear_mac_operations_per_prediction=nf,
        hist_internal_node_record_bits=internal_bits,
        hist_leaf_record_bits=leaf_bits,
        hist_tree_storage_lower_bound_bits=node_bits,
        hist_tree_storage_lower_bound_kib=node_bits/8/1024,
        hist_feature_input_storage_bits=nf*64,
        hist_comparisons_per_prediction_upper_bound=tree['sum_tree_max_depths'],
        hist_tree_output_additions_per_prediction=max(0,tree['n_trees']-1),
        caveat='Storage and operation counts only. No FPGA synthesis, LUT/FF/BRAM/DSP estimates, clock frequency, or timing closure. '
               'HistGB CPU implementation uses binning and metadata not included in the simplified tree records.'
    )
    out = args.output_dir/'profile.json'
    out.write_text(json.dumps(profile, indent=2)+'\n')
    print('\n=== MODEL PROFILE ===')
    print(f'Features: {nf}; training rows: {len(y):,}')
    for name, item in profile['models'].items():
        print(f'{name}: serialized {item["serialized_bytes"]:,} bytes')
        print('  structure:', json.dumps(item['structure']))
        print('  CPU timing:', json.dumps(item['cpu_latency']))
    print('\n=== ANALYTICAL HARDWARE ESTIMATES (NOT SYNTHESIS) ===')
    print(json.dumps(profile['fpga_analytical'], indent=2))
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
