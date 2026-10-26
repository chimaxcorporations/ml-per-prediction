#!/usr/bin/env python3
"""Scenario-held-out PER benchmark: four original and two residual models."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["per_t", "dist_m", "v_src_mps", "v_dst_mps", "cbr_tx", "cbr_rx", "cbr_global"]
EVAL_COLS = FEATURES + ["target_per", "target_tx", "target_n_loss"]
MODELS = ("zero", "persistence", "ridge", "hist_gb", "residual_ridge", "residual_hist_gb")


def load_training(input_dir, names, per_scenario, seed):
    chunks = []
    for idx, name in enumerate(names):
        path = input_dir / name
        pf = pq.ParquetFile(path)
        n = pf.metadata.num_rows
        if n == 0:
            raise ValueError(f"Empty training file: {name}")
        # Reservoir sampling from parquet batches: bounded memory, uniform rows.
        rng = np.random.default_rng(seed + idx)
        k = min(per_scenario, n)
        reservoir = np.empty((k, len(FEATURES) + 1), dtype=np.float64)
        seen = 0
        for batch in pf.iter_batches(batch_size=100_000, columns=FEATURES + ["target_per"]):
            arr = batch.to_pandas()[FEATURES + ["target_per"]].to_numpy(dtype=np.float64)
            for row in arr:
                if seen < k:
                    reservoir[seen] = row
                else:
                    j = rng.integers(seen + 1)
                    if j < k:
                        reservoir[j] = row
                seen += 1
        chunks.append(reservoir)
        print(f"Training sample {idx+1}/{len(names)}: {name}: {k:,} of {n:,}", flush=True)
    data = np.vstack(chunks)
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite training features/targets")
    return data[:, :-1], data[:, -1]


def accumulator():
    return {name: {"n": 0, "abs": 0., "sq": 0., "wabs": 0., "wsq": 0.,
                   "sum_y": 0., "sum_y2": 0., "tx": 0, "loss": 0} for name in MODELS}


def merge(dst, src):
    for model in MODELS:
        for key in dst[model]:
            dst[model][key] += src[model][key]


def finalize(a, split, scenario):
    rows = []
    for model, s in a.items():
        n = s["n"]
        if not n or not s["tx"]:
            raise ValueError(f"No samples for {split}/{scenario}/{model}")
        variance_ss = s["sum_y2"] - s["sum_y"] ** 2 / n
        rows.append({
            "split": split, "scenario": scenario, "model": model,
            "samples": n, "total_tx": s["tx"], "total_loss": s["loss"],
            "packet_weighted_per": s["loss"] / s["tx"],
            "mae": s["abs"] / n, "rmse": np.sqrt(s["sq"] / n),
            "r2": 1 - s["sq"] / variance_ss if variance_ss > 1e-12 else np.nan,
            "weighted_mae": s["wabs"] / s["tx"],
            "weighted_rmse": np.sqrt(s["wsq"] / s["tx"]),
        })
    return rows


def evaluate_file(path, split, fitted, batch_size):
    pf = pq.ParquetFile(path)
    stats = accumulator()
    for batch in pf.iter_batches(batch_size=batch_size, columns=EVAL_COLS):
        df = batch.to_pandas()
        x = df[FEATURES].to_numpy(dtype=np.float64)
        y = df["target_per"].to_numpy(dtype=np.float64)
        tx = df["target_tx"].to_numpy(dtype=np.int64)
        loss = df["target_n_loss"].to_numpy(dtype=np.int64)
        if (not np.isfinite(x).all() or not np.isfinite(y).all()
                or np.any(tx <= 0) or np.any(loss < 0) or np.any(loss > tx)):
            raise ValueError(f"Invalid evaluation data: {path}")
        if not np.allclose(y, loss / tx, rtol=1e-6, atol=1e-6):
            raise ValueError(f"Target/count mismatch: {path}")
        predictions = {
            "zero": np.zeros_like(y),
            "persistence": df["per_t"].to_numpy(dtype=np.float64),
            "ridge": fitted["ridge"].predict(x),
            "hist_gb": fitted["hist_gb"].predict(x),
            "residual_ridge": x[:, 0] + fitted["residual_ridge"].predict(x),
            "residual_hist_gb": x[:, 0] + fitted["residual_hist_gb"].predict(x),
        }
        for model, pred in predictions.items():
            pred = np.clip(pred, 0., 1.)
            error = y - pred
            s = stats[model]
            s["n"] += len(y)
            s["abs"] += float(np.abs(error).sum())
            s["sq"] += float(np.dot(error, error))
            s["wabs"] += float(np.dot(tx, np.abs(error)))
            s["wsq"] += float(np.dot(tx, error * error))
            s["sum_y"] += float(y.sum())
            s["sum_y2"] += float(np.dot(y, y))
            s["tx"] += int(tx.sum())
            s["loss"] += int(loss.sum())
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("/workspace/data/processed"))
    parser.add_argument("--split-file", type=Path, default=Path("/workspace/configs/scenario_splits.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/results/ml_benchmark_v2"))
    parser.add_argument("--train-per-scenario", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train_per_scenario < 1 or args.batch_size < 1:
        parser.error("Sample and batch sizes must be positive")
    splits = json.loads(args.split_file.read_text())
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation, test in split manifest")
    all_names = sum((splits[s] for s in ("train", "validation", "test")), [])
    if len(all_names) != 27 or len(set(all_names)) != 27:
        raise ValueError("Expected 27 distinct scenarios across splits")
    for name in all_names:
        if Path(name).name != name or not (args.input_dir / name).is_file():
            raise ValueError(f"Missing or invalid scenario: {name}")
    print("Loading bounded training sample...", flush=True)
    x_train, y_train = load_training(args.input_dir, splits["train"], args.train_per_scenario, args.seed)
    print(f"Training on {len(y_train):,} samples", flush=True)
    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "hist_gb": HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1,
                  max_leaf_nodes=31, random_state=args.seed),
    }
    models["residual_ridge"] = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    models["residual_hist_gb"] = HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.1, max_leaf_nodes=31, random_state=args.seed
    )
    residual_target = y_train - x_train[:, 0]
    for name, model in models.items():
        print(f"Fitting {name}...", flush=True)
        target = residual_target if name.startswith("residual_") else y_train
        model.fit(x_train, target)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, args.output_dir / f"{name}.joblib")
    rows = []
    for split in ("validation", "test"):
        pooled = accumulator()
        for name in splits[split]:
            print(f"Evaluating {split}: {name}", flush=True)
            stats = evaluate_file(args.input_dir / name, split, models, args.batch_size)
            rows.extend(finalize(stats, split, name))
            merge(pooled, stats)
        rows.extend(finalize(pooled, split, "__pooled__"))
    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "metrics.csv", index=False)
    pooled = results[results["scenario"] == "__pooled__"]
    print("\n=== POOLED RESULTS ===")
    print(pooled[["split", "model", "samples", "mae", "rmse", "r2", "weighted_mae", "weighted_rmse"]].to_string(index=False))
    print(f"\nSaved: {args.output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
