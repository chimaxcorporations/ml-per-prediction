#!/usr/bin/env python3
"""Zero-shot Juelich/Berlin common-40 transfer: GRU, LSTM, TCN, tiny Transformer.

Mirrors the sampling, source-only StandardScaler, training epochs, and evaluation
protocol of run_cross_dataset_common40.py. Original benchmark is untouched.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

BASE = ('per_t', 'dist_m', 'v_src_mps', 'v_dst_mps')
FEATURES = [f'{c}_lag{lag}' for lag in range(10) for c in BASE]
COLS = FEATURES + ['target_per', 'target_tx']
PATTERN = re.compile(r'^juelich_core_d(10|30|60)_r(2|5|10)_(log_distance|log_distance_nakagami|urban)_s\d+_w\d+\.parquet$')
MODEL_NAMES = ('gru', 'lstm', 'tcn', 'transformer')


def chronological(x):
    # Source columns are lag0 (newest), lag1, ..., lag9 (oldest).
    # Reverse time to oldest -> newest for recurrent/causal temporal models.
    return x.reshape(-1, 10, 4).flip(dims=(1,))


class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(4, 32, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        _, h = self.rnn(chronological(x))
        return self.head(h[-1]).squeeze(-1)


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(4, 32, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        _, (h, _) = self.rnn(chronological(x))
        return self.head(h[-1]).squeeze(-1)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, dilation=dilation)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(4, 16, 1), nn.ReLU(),
            CausalConv1d(16, 16, 2), nn.ReLU(),
            CausalConv1d(16, 16, 4), nn.ReLU(),
        )
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        z = chronological(x).transpose(1, 2)
        return self.head(self.net(z)[:, :, -1]).squeeze(-1)


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(4, 32)
        self.position = nn.Parameter(torch.zeros(1, 10, 32))
        layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64, dropout=0.0,
            activation='relu', batch_first=True, norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        z = self.project(chronological(x)) + self.position
        return self.head(self.encoder(z)[:, -1, :]).squeeze(-1)


CLASSES = {'gru': GRU, 'lstm': LSTM, 'tcn': TCN, 'transformer': TinyTransformer}


class Metrics:
    def __init__(self):
        self.n = 0
        self.sy = self.sy2 = self.ae = self.se = self.w = self.wae = self.wse = 0.0

    def update(self, y, p, w):
        e = y - p
        self.n += len(y)
        self.sy += float(y.sum())
        self.sy2 += float(np.dot(y, y))
        self.ae += float(np.abs(e).sum())
        self.se += float(np.dot(e, e))
        self.w += float(w.sum())
        self.wae += float(np.dot(w, np.abs(e)))
        self.wse += float(np.dot(w, e * e))

    def result(self):
        ss = self.sy2 - self.sy * self.sy / self.n
        return dict(samples=self.n, total_tx=self.w, mae=self.ae / self.n,
                    rmse=(self.se / self.n) ** 0.5,
                    r2=1 - self.se / ss if ss > 0 else np.nan,
                    weighted_mae=self.wae / self.w,
                    weighted_rmse=(self.wse / self.w) ** 0.5)


def check(frame, origin):
    missing = set(COLS) - set(frame.columns)
    if missing:
        raise ValueError(f'{origin}: missing {sorted(missing)}')
    arr = frame[COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f'{origin}: nonfinite values')
    if (frame.target_tx.to_numpy() <= 0).any():
        raise ValueError(f'{origin}: nonpositive target_tx')
    if ((frame.target_per.to_numpy() < 0) | (frame.target_per.to_numpy() > 1)).any():
        raise ValueError(f'{origin}: target_per outside [0,1]')


def sample_juelich(files, n, seed, batch):
    frames = []
    for idx, path in enumerate(files):
        pf = pq.ParquetFile(path)
        sel = np.sort(np.random.default_rng(seed + idx).choice(
            pf.metadata.num_rows, min(n, pf.metadata.num_rows), replace=False))
        offset, pieces = 0, []
        for chunk in pf.iter_batches(batch_size=batch, columns=FEATURES + ['target_per']):
            end = offset + chunk.num_rows
            lo, hi = np.searchsorted(sel, [offset, end])
            if hi > lo:
                pieces.append(chunk.to_pandas().iloc[sel[lo:hi] - offset])
            offset = end
        frames.append(pd.concat(pieces, ignore_index=True))
        print(f'  sampled {len(frames[-1]):,}: {path.name}', flush=True)
    return pd.concat(frames, ignore_index=True)


def train_models(X, y, args, direction):
    scaler = StandardScaler().fit(X)
    x32 = np.ascontiguousarray(scaler.transform(X).astype(np.float32))
    y32 = np.asarray(y, dtype=np.float32)
    models, counts = {}, {}
    for idx, name in enumerate(args.models):
        seed = args.seed + 20 + idx
        torch.manual_seed(seed)
        model = CLASSES[name]().to(args.device)
        counts[name] = sum(p.numel() for p in model.parameters())
        loader = DataLoader(TensorDataset(torch.from_numpy(x32), torch.from_numpy(y32)),
                            batch_size=args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(args.epochs):
            model.train()
            loss_sum = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(args.device), yb.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                loss = ((model(xb) - yb) ** 2).mean()
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(yb)
            print(f'  {direction} {name} epoch {epoch+1}/{args.epochs}: '
                  f'MSE={loss_sum/len(y):.7f}', flush=True)
        models[name] = model.eval()
        if args.save_models:
            out = args.output_dir / 'models'
            out.mkdir(exist_ok=True)
            torch.save({'state_dict': model.cpu().state_dict(), 'model': name,
                        'features': FEATURES, 'seed': seed, 'epochs': args.epochs,
                        'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_},
                       out / f'{direction}_{name}.pt')
            models[name] = model.to(args.device).eval()
    return scaler, models, counts


def predict(model, scaler, X, batch, device):
    output = np.empty(len(X), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(X), batch):
            end = min(start + batch, len(X))
            inp = np.ascontiguousarray(scaler.transform(X[start:end]).astype(np.float32))
            output[start:end] = model(torch.from_numpy(inp).to(device)).cpu().numpy()
    return np.clip(output, 0, 1)


def evaluate(files, scaler, models, args, direction):
    rows = []
    overall = {name: Metrics() for name in args.models}
    for path in files:
        acc = {name: Metrics() for name in args.models}
        pf = pq.ParquetFile(path)
        for chunk in pf.iter_batches(batch_size=args.eval_batch_size, columns=COLS):
            df = chunk.to_pandas()
            check(df, path.name)
            X = df[FEATURES].to_numpy(dtype=np.float64)
            y = df.target_per.to_numpy(dtype=np.float64)
            w = df.target_tx.to_numpy(dtype=np.float64)
            for name, model in models.items():
                p = predict(model, scaler, X, args.batch_size * 4, args.device)
                acc[name].update(y, p, w)
                overall[name].update(y, p, w)
        for name in args.models:
            rows.append(dict(direction=direction, level='scenario', scenario=path.stem,
                             model=name, **acc[name].result()))
        print(f'  evaluated {path.name}: {pf.metadata.num_rows:,} samples', flush=True)
        pd.DataFrame(rows).to_csv(args.output_dir / f'{direction}_partial.csv', index=False)
    for name in args.models:
        rows.append(dict(direction=direction, level='overall', scenario='ALL',
                         model=name, **overall[name].result()))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--juelich-dir', type=Path, default=Path('/workspace/data/temporal_h10'))
    ap.add_argument('--berlin-dir', type=Path, default=Path('/workspace/data/berlin/temporal_h10'))
    ap.add_argument('--output-dir', type=Path, default=Path('/workspace/results/temporal_transfer_v1'))
    ap.add_argument('--direction', choices=['both', 'juelich_to_berlin', 'berlin_to_juelich'], default='both')
    ap.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=list(MODEL_NAMES))
    ap.add_argument('--samples-per-scenario', type=int, default=30000)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--eval-batch-size', type=int, default=50000)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--save-models', action='store_true')
    args = ap.parse_args()
    if min(args.samples_per_scenario, args.epochs, args.batch_size,
           args.eval_batch_size, args.threads) < 1:
        ap.error('All numerical settings must be positive')
    if len(set(args.models)) != len(args.models):
        ap.error('Duplicate model names are not allowed')
    if args.device == 'cuda' and not torch.cuda.is_available():
        ap.error('CUDA requested but unavailable')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        ap.error('Output directory is not empty; choose a new one')
    files = sorted(p for p in args.juelich_dir.glob('juelich_core_*.parquet') if PATTERN.fullmatch(p.name))
    if len(files) != 27:
        raise ValueError(f'Expected 27 Juelich files, found {len(files)}')
    berlin = [args.berlin_dir / f'berlin_{s}_h10.parquet' for s in ('S1', 'S2')]
    if any(not p.is_file() for p in berlin):
        raise FileNotFoundError('Berlin S1/S2 Parquet files missing')
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(features=FEATURES, history=10, seed=args.seed, models=args.models,
                    samples_per_juelich_scenario=args.samples_per_scenario,
                    epochs=args.epochs, batch_size=args.batch_size,
                    eval_batch_size=args.eval_batch_size, direction=args.direction,
                    protocol='Train on source only, no target-domain training or tuning',
                    caution='Cross-technology IEEE 802.11p / LTE-V2X; PER denominators differ')
    (args.output_dir / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    rows, parameter_rows = [], []
    if args.direction in ('both', 'juelich_to_berlin'):
        direction = 'juelich_to_berlin'
        print('\n=== TRAIN JUELICH -> TEST BERLIN ===', flush=True)
        tr = sample_juelich(files, args.samples_per_scenario, args.seed, args.eval_batch_size)
        X = tr[FEATURES].to_numpy(dtype=np.float64)
        y = tr.target_per.to_numpy(dtype=np.float64)
        if not (np.isfinite(X).all() and np.isfinite(y).all()):
            raise ValueError('Invalid Juelich training data')
        scaler, models, counts = train_models(X, y, args, direction)
        parameter_rows += [dict(direction=direction, model=k, parameters=v) for k, v in counts.items()]
        del X, y, tr
        rows += evaluate(berlin, scaler, models, args, direction)
        del models, scaler
    if args.direction in ('both', 'berlin_to_juelich'):
        direction = 'berlin_to_juelich'
        print('\n=== TRAIN BERLIN -> TEST JUELICH ===', flush=True)
        tr = pd.concat([pd.read_parquet(p, columns=COLS) for p in berlin], ignore_index=True)
        check(tr, 'Berlin training')
        X = tr[FEATURES].to_numpy(dtype=np.float64)
        y = tr.target_per.to_numpy(dtype=np.float64)
        scaler, models, counts = train_models(X, y, args, direction)
        parameter_rows += [dict(direction=direction, model=k, parameters=v) for k, v in counts.items()]
        del X, y, tr
        rows += evaluate(files, scaler, models, args, direction)
        del models, scaler
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / 'metrics.csv', index=False)
    pd.DataFrame(parameter_rows).to_csv(args.output_dir / 'parameter_counts.csv', index=False)
    print('\n=== OVERALL TRANSFER METRICS ===', flush=True)
    print(df[df.level == 'overall'].to_string(index=False), flush=True)
    print('Saved:', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
