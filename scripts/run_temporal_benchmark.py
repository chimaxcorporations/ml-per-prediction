#!/usr/bin/env python3
"""Fair 1/2/5/10-second PER-history benchmark on common H10-eligible rows."""
import argparse
import json
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
sys.path.insert(0, str(ROOT / "src"))
from per_prediction.data.temporal import feature_columns

HISTORIES = (1, 2, 5, 10)
META = ["target_per", "target_tx", "per_t_lag0"]


class MetricAccumulator:
    def __init__(self):
        self.n = 0
        self.sum_y = self.sum_y2 = self.sum_abs = self.sum_sq = 0.0
        self.sum_w = self.sum_wabs = self.sum_wsq = 0.0

    def update(self, y, prediction, weight):
        diff = y - prediction
        self.n += len(y)
        self.sum_y += float(np.sum(y, dtype=np.float64))
        self.sum_y2 += float(np.dot(y, y))
        self.sum_abs += float(np.sum(np.abs(diff), dtype=np.float64))
        self.sum_sq += float(np.dot(diff, diff))
        self.sum_w += float(np.sum(weight, dtype=np.float64))
        self.sum_wabs += float(np.dot(weight, np.abs(diff)))
        self.sum_wsq += float(np.dot(weight, diff * diff))

    def result(self, split, scenario, history, model):
        if self.n == 0 or self.sum_w <= 0:
            raise ValueError("Empty evaluation accumulator")
        sst = self.sum_y2 - self.sum_y ** 2 / self.n
        return dict(split=split, scenario=scenario, history=history, model=model,
                    samples=self.n, total_tx=self.sum_w,
                    mae=self.sum_abs / self.n,
                    rmse=np.sqrt(self.sum_sq / self.n),
                    r2=1 - self.sum_sq / sst if sst > 1e-12 else np.nan,
                    weighted_mae=self.sum_wabs / self.sum_w,
                    weighted_rmse=np.sqrt(self.sum_wsq / self.sum_w))


def sample_scenario(path, count, seed):
    """Uniform without-replacement sampling by row index, without reading full file."""
    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows
    if n == 0:
        raise ValueError(f"Empty file: {path}")
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(n, size=min(n, count), replace=False))
    selected_chunks = []
    offset = 0
    columns = list(dict.fromkeys(feature_columns(10) + ["target_per"]))
    for batch in pf.iter_batches(batch_size=100_000, columns=columns):
        stop = offset + batch.num_rows
        a = np.searchsorted(selected, offset, side="left")
        b = np.searchsorted(selected, stop, side="left")
        if b > a:
            frame = batch.to_pandas()
            selected_chunks.append(frame.iloc[selected[a:b] - offset].copy())
        offset = stop
    return pd.concat(selected_chunks, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("/workspace/data/temporal_h10"))
    ap.add_argument("--splits", type=Path, default=Path("/workspace/configs/scenario_splits.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("/workspace/results/temporal_benchmark_v1"))
    ap.add_argument("--samples-per-scenario", type=int, default=30_000)
    ap.add_argument("--batch-size", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.samples_per_scenario <= 0 or args.batch_size <= 0:
        ap.error("Sampling count and batch size must be positive")
    splits = json.loads(args.splits.read_text())
    if any(len(splits.get(s, [])) != n for s, n in (("train", 18), ("validation", 3), ("test", 6))):
        raise ValueError("Expected 18/3/6 split manifest")
    names = sum((splits[s] for s in ("train", "validation", "test")), [])
    if len(set(names)) != 27:
        raise ValueError("Duplicate or missing scenario in split manifest")
    for name in names:
        if not (args.input_dir / name).is_file():
            raise FileNotFoundError(args.input_dir / name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Sampling the same training observations for all history lengths...", flush=True)
    frames = []
    for i, name in enumerate(splits["train"]):
        frame = sample_scenario(args.input_dir / name, args.samples_per_scenario, args.seed + i)
        frames.append(frame)
        print(f"  {name}: {len(frame):,}", flush=True)
    training = pd.concat(frames, ignore_index=True)
    del frames
    y = training["target_per"].to_numpy(dtype=np.float64)
    models = {}
    for history in HISTORIES:
        columns = feature_columns(history)
        X = training[columns].to_numpy(dtype=np.float64)
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Training contains non-finite values")
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        ridge.fit(X, y)
        gb = HistGradientBoostingRegressor(max_iter=150, random_state=args.seed)
        gb.fit(X, y)
        models[history] = {"ridge": ridge, "hist_gb": gb}
        joblib.dump(ridge, args.output_dir / f"ridge_h{history}.joblib")
        joblib.dump(gb, args.output_dir / f"hist_gb_h{history}.joblib")
        print(f"Trained H={history} on {len(training):,} samples", flush=True)
    del training

    rows = []
    for split in ("validation", "test"):
        pooled = {(h, m): MetricAccumulator() for h in HISTORIES for m in ("zero", "persistence", "ridge", "hist_gb")}
        for name in splits[split]:
            scenario_acc = {(h, m): MetricAccumulator() for h in HISTORIES for m in ("zero", "persistence", "ridge", "hist_gb")}
            pf = pq.ParquetFile(args.input_dir / name)
            needed = list(dict.fromkeys(feature_columns(10) + META))
            for batch in pf.iter_batches(batch_size=args.batch_size, columns=needed):
                frame = batch.to_pandas()
                target = frame["target_per"].to_numpy(dtype=np.float64)
                weight = frame["target_tx"].to_numpy(dtype=np.float64)
                persistence = frame["per_t_lag0"].to_numpy(dtype=np.float64)
                if not (np.isfinite(target).all() and np.isfinite(weight).all() and (weight > 0).all()):
                    raise ValueError(f"Invalid evaluation target/weight: {name}")
                for history in HISTORIES:
                    X = frame[feature_columns(history)].to_numpy(dtype=np.float64)
                    preds = {"zero": np.zeros_like(target),
                             "persistence": persistence,
                             "ridge": np.clip(models[history]["ridge"].predict(X), 0, 1),
                             "hist_gb": np.clip(models[history]["hist_gb"].predict(X), 0, 1)}
                    for model, prediction in preds.items():
                        scenario_acc[(history, model)].update(target, prediction, weight)
                        pooled[(history, model)].update(target, prediction, weight)
            for (history, model), acc in scenario_acc.items():
                rows.append(acc.result(split, name, history, model))
            print(f"Evaluated {split}: {name} ({pf.metadata.num_rows:,} rows)", flush=True)
        for (history, model), acc in pooled.items():
            rows.append(acc.result(split, "__pooled__", history, model))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    print("\n=== POOLED RESULTS ===")
    print(metrics[metrics.scenario == "__pooled__"].drop(columns="scenario").to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("Saved:", args.output_dir / "metrics.csv")


if __name__ == "__main__":
    main()
