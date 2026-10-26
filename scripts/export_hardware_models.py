#!/usr/bin/env python3
"""Export trained H=10 HistGB and Ridge into explicit, float64 hardware-oriented records.

This is an inference reference export, NOT HDL, HLS, synthesis or timing closure.
HistGB node format (32 bytes, little-endian):
  offset  type   field
   0      f64    threshold (internal) or leaf contribution (leaf)
   8      u32    left child index within tree
  12      u32    right child index within tree
  16      u16    feature index
  18      u8     is_leaf
  19      u8     missing_go_to_left
  20..31  u8[12] reserved (zero)
Missing values follow sklearn's recorded routing. Node contributions include
learning-rate scaling as stored in sklearn's fitted predictor nodes.
"""
import argparse
import json
import struct
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq

NODE = struct.Struct('<dIIHBB12x')
assert NODE.size == 32


def dump_hist(model, dest, features):
    trees = [p for stage in model._predictors for p in stage]
    manifest = []
    records = 0
    with (dest / 'hist_nodes_f64.bin').open('wb') as f:
        for tree_id, tree in enumerate(trees):
            nodes = tree.nodes
            required = {'is_leaf', 'value', 'num_threshold', 'left', 'right',
                        'feature_idx', 'missing_go_to_left'}
            missing = required.difference(nodes.dtype.names)
            if missing:
                raise RuntimeError(f'Unsupported sklearn node layout; missing {missing}')
            offset = records * NODE.size
            for node in nodes:
                leaf = bool(node['is_leaf'])
                value = float(node['value'] if leaf else node['num_threshold'])
                f.write(NODE.pack(value,
                                  0 if leaf else int(node['left']),
                                  0 if leaf else int(node['right']),
                                  0 if leaf else int(node['feature_idx']),
                                  int(leaf),
                                  0 if leaf else int(node['missing_go_to_left'])))
            manifest.append({'tree_id': tree_id, 'offset_bytes': offset,
                             'node_count': len(nodes)})
            records += len(nodes)
    baseline = np.asarray(model._baseline_prediction, dtype=np.float64).ravel()
    if baseline.size != 1:
        raise ValueError('Only scalar-output HistGB supported')
    return {'trees': manifest, 'node_count': records,
            'baseline_prediction': float(baseline[0]),
            'feature_names': features,
            'input_dtype': 'float64', 'node_record_bytes': NODE.size,
            'node_layout': '<dIIHBB12x',
            'note': 'Raw numeric thresholds; fitted leaf contributions; float64 arithmetic.'}


def predict_export_hist(data, meta, X):
    # Independent interpreter of the exported binary, deliberately not using sklearn predictors.
    raw = np.frombuffer(data, dtype=np.dtype([
        ('v', '<f8'), ('left', '<u4'), ('right', '<u4'),
        ('feature', '<u2'), ('leaf', 'u1'), ('missing_left', 'u1'),
        ('padding', 'u1', (12,))]))
    out = np.full(len(X), meta['baseline_prediction'], dtype=np.float64)
    for tree in meta['trees']:
        base = tree['offset_bytes'] // NODE.size
        n = tree['node_count']
        for row_idx, x in enumerate(X):
            k = 0
            for _ in range(n + 1):
                node = raw[base + k]
                if node['leaf']:
                    out[row_idx] += float(node['v'])
                    break
                value = x[int(node['feature'])]
                left = bool(node['missing_left']) if np.isnan(value) else value <= node['v']
                k = int(node['left'] if left else node['right'])
                if k >= n:
                    raise RuntimeError('Corrupt child index')
            else:
                raise RuntimeError('Tree traversal did not terminate')
    return np.clip(out, 0, 1) if meta['clip_output_to_unit_interval'] else out


def dump_ridge(pipe, dest, features):
    scaler = pipe.named_steps['standardscaler']
    ridge = pipe.named_steps['ridge']
    # Fuse scaling into weights and bias, preserving float64 coefficients.
    w = np.asarray(ridge.coef_, dtype=np.float64).ravel() / np.asarray(scaler.scale_, dtype=np.float64)
    b = float(ridge.intercept_) - float(np.dot(w, scaler.mean_))
    np.asarray(np.r_[b, w], dtype='<f8').tofile(dest / 'ridge_fused_f64.bin')
    return {'feature_names': features, 'parameter_dtype': 'float64',
            'parameter_order': 'bias_then_70_weights', 'bias': b,
            'parameter_count': len(w) + 1,
            'note': 'Output clipping, if used by benchmark, is not part of sklearn Ridge.predict.'}


def load_validation_rows(path, features, count):
    pf = pq.ParquetFile(path)
    if not set(features).issubset(set(pf.schema_arrow.names)):
        raise ValueError('Validation Parquet does not contain all model features')
    batch = next(pf.iter_batches(batch_size=count, columns=features))
    return np.asarray(batch.to_pandas()[features].to_numpy(dtype=np.float64))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, default=Path('/workspace/results/model_profile_h10'))
    p.add_argument('--output-dir', type=Path, default=Path('/workspace/results/hardware_export_h10'))
    p.add_argument('--validation-parquet', type=Path, required=True,
                   help='A temporal_h10 parquet file not used in reference model training')
    p.add_argument('--verify-rows', type=int, default=128)
    args = p.parse_args()
    if args.verify_rows < 1:
        p.error('--verify-rows must be positive')
    profile = json.loads((args.model_dir / 'profile.json').read_text())
    features = profile['features']
    if len(features) != 70:
        raise ValueError(f'Expected H10 70 features, got {len(features)}')
    hist = joblib.load(args.model_dir / 'hist_gb_h10.joblib')
    ridge = joblib.load(args.model_dir / 'ridge_h10.joblib')
    if hist.n_features_in_ != len(features) or ridge.n_features_in_ != len(features):
        raise ValueError('Model and profile feature counts disagree')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hist_meta = dump_hist(hist, args.output_dir, features)
    hist_meta['clip_output_to_unit_interval'] = False
    ridge_meta = dump_ridge(ridge, args.output_dir, features)
    X = load_validation_rows(args.validation_parquet, features, args.verify_rows)
    hist_pred = hist.predict(X)
    exported_pred = predict_export_hist((args.output_dir / 'hist_nodes_f64.bin').read_bytes(), hist_meta, X)
    ridge_pred = ridge.predict(X)
    w = np.fromfile(args.output_dir / 'ridge_fused_f64.bin', dtype='<f8')
    ridge_exported = X @ w[1:] + w[0]
    # Differences can arise from summation order and scaling algebra; enforce small tolerance.
    hist_error = float(np.max(np.abs(hist_pred - exported_pred)))
    ridge_error = float(np.max(np.abs(ridge_pred - ridge_exported)))
    verified = bool(np.allclose(hist_pred, exported_pred, atol=1e-9, rtol=1e-9)
                    and np.allclose(ridge_pred, ridge_exported, atol=1e-9, rtol=1e-9))
    report = {'source_training': profile['training_source'],
              'validation_parquet': str(args.validation_parquet),
              'verification_rows': len(X),
              'hist_max_abs_error': hist_error,
              'ridge_max_abs_error': ridge_error,
              'verification_passed': verified,
              'hist_tree_bytes': (args.output_dir / 'hist_nodes_f64.bin').stat().st_size,
              'ridge_fused_bytes': (args.output_dir / 'ridge_fused_f64.bin').stat().st_size,
              'hist_node_count': hist_meta['node_count'],
              'hist_tree_count': len(hist_meta['trees']),
              'caveat': 'Storage counts exclude feature acquisition, buffering, clocking, and full hardware control. No synthesis performed.'}
    (args.output_dir / 'hist_manifest.json').write_text(json.dumps(hist_meta, indent=2) + '\n')
    (args.output_dir / 'ridge_manifest.json').write_text(json.dumps(ridge_meta, indent=2) + '\n')
    (args.output_dir / 'export_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not verified:
        raise SystemExit('FAILED: exported inference differs from sklearn. Do not use hardware estimates until resolved.')


if __name__ == '__main__':
    main()
