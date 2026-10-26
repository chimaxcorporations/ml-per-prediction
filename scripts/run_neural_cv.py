#!/usr/bin/env python3
"""Scenario-held-out nine-fold CV: compact MLP and 1D CNN, H=10.

Run in the same repository as scripts/run_scenario_cv.py. Torch is required.
Fixed architectures and epochs; no outer-test-driven hyperparameter selection.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise SystemExit('PyTorch missing. Install torch in the per-ml Docker image and rebuild it.') from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_scenario_cv import FEATURES, Metrics, build_folds, sample_rows

HISTORY, CHANNELS = 10, 7
assert len(FEATURES) == HISTORY * CHANNELS


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(70, 64), nn.ReLU(),
                                    nn.Linear(64, 32), nn.ReLU(),
                                    nn.Linear(32, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(7, 16, kernel_size=3, padding=1), nn.ReLU(),
                                  nn.Conv1d(16, 16, kernel_size=3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16 * 10, 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    def forward(self, x):
        # Features ordered [lag0's 7 values, lag1's 7 values, ...].
        x = x.reshape(-1, HISTORY, CHANNELS).transpose(1, 2)
        return self.head(self.conv(x)).squeeze(-1)


def fit_model(model, X, y, epochs, batch_size, lr, seed, device):
    torch.manual_seed(seed)
    model.to(device)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=generator, num_workers=0, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        loss_sum = count = 0
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(features)
            loss = loss_fn(pred, targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(targets)
            count += len(targets)
        print(f'    epoch {epoch + 1}/{epochs}: train MSE={loss_sum / count:.7f}', flush=True)
    return time.perf_counter() - start


def predict(model, X, batch_size, device):
    model.eval()
    output = np.empty(len(X), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            inp = torch.from_numpy(X[start:end]).to(device)
            output[start:end] = model(inp).cpu().numpy().astype(np.float64)
    return np.clip(output, 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', type=Path, default=Path('/workspace/data/temporal_h10'))
    ap.add_argument('--output-dir', type=Path, default=Path('/workspace/results/neural_cv_h10'))
    ap.add_argument('--samples-per-scenario', type=int, default=30000)
    ap.add_argument('--batch-size', type=int, default=2048, help='Training batch size')
    ap.add_argument('--eval-batch-size', type=int, default=20000)
    ap.add_argument('--read-batch-size', type=int, default=50000)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--fold', type=int, choices=range(1, 10), default=None,
                    help='Run just one fold (1-9) for smoke testing; does not produce overall metrics')
    ap.add_argument('--save-models', action='store_true')
    args = ap.parse_args()
    if min(args.samples_per_scenario, args.batch_size, args.eval_batch_size,
           args.read_batch_size, args.epochs, args.threads) < 1 or args.lr <= 0:
        ap.error('All counts must be positive, and learning rate > 0')
    if args.device == 'cuda' and not torch.cuda.is_available():
        ap.error('CUDA requested but not available')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    names = sorted(p.name for p in args.input_dir.glob('juelich_core_*.parquet'))
    if len(names) != 27:
        raise ValueError(f'Expected 27 scenarios, found {len(names)}')
    folds = build_folds(names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'folds.json').write_text(json.dumps({str(k+1):v for k,v in folds.items()}, indent=2)+'\n')
    (args.output_dir / 'settings.json').write_text(json.dumps({
        'samples_per_scenario':args.samples_per_scenario, 'epochs':args.epochs,
        'lr':args.lr, 'batch_size':args.batch_size, 'seed':args.seed,
        'features':FEATURES, 'models':['mlp','cnn'],
        'target':'target_per', 'scaling':'StandardScaler fitted on outer-train only',
        'fold_protocol':'same nine balanced scenario-held-out folds as run_scenario_cv.py',
        'no_early_stopping':'fixed epochs to avoid touching held-out scenarios'
    },indent=2)+'\n')
    if args.save_models:
        (args.output_dir / 'models').mkdir(exist_ok=True)
    rows = []
    overall = {name: Metrics() for name in ('persistence','mlp','cnn')}
    folds_to_run = [args.fold - 1] if args.fold else list(range(9))
    for fold in folds_to_run:
        test_names = folds[fold]
        train_names = [n for n in names if n not in set(test_names)]
        print(f'\nFOLD {fold+1}/9; train=24 scenarios, holdout=3', flush=True)
        for n in test_names: print('  HOLDOUT',n,flush=True)
        frames = [sample_rows(args.input_dir/n, args.samples_per_scenario,
                              args.seed + names.index(n), args.read_batch_size)
                  for n in train_names]
        train = pd.concat(frames, ignore_index=True)
        del frames
        Xraw = train[FEATURES].to_numpy(dtype=np.float64)
        y = train.target_per.to_numpy(dtype=np.float32)
        if not np.isfinite(Xraw).all() or not np.isfinite(y).all():
            raise ValueError('Non-finite training data')
        scaler = StandardScaler().fit(Xraw)
        X = np.ascontiguousarray(scaler.transform(Xraw).astype(np.float32))
        del Xraw, train
        models = {}
        for j, (name, cls) in enumerate([('mlp', MLP), ('cnn', CNN)]):
            torch.manual_seed(args.seed + fold * 10 + j)
            model = cls()
            print(f'  training {name}: {sum(p.numel() for p in model.parameters()):,} parameters',flush=True)
            seconds = fit_model(model, X, y, args.epochs, args.batch_size, args.lr,
                                args.seed + fold * 10 + j, args.device)
            print(f'  {name} training seconds={seconds:.1f}',flush=True)
            models[name] = model
            if args.save_models:
                torch.save({'state_dict':model.cpu().state_dict(), 'features':FEATURES,
                            'scaler_mean':scaler.mean_, 'scaler_scale':scaler.scale_,
                            'fold':fold+1, 'model':name},
                           args.output_dir/'models'/f'{name}_fold{fold+1}.pt')
                model.to(args.device)
        del X, y
        fold_acc = {name:Metrics() for name in overall}
        for scenario in test_names:
            scenario_acc = {name:Metrics() for name in overall}
            pf = pq.ParquetFile(args.input_dir/scenario)
            for batch in pf.iter_batches(batch_size=args.read_batch_size,
                                         columns=FEATURES+['target_per','target_tx']):
                frame = batch.to_pandas()
                Xraw = frame[FEATURES].to_numpy(dtype=np.float64)
                target = frame.target_per.to_numpy(dtype=np.float64)
                weight = frame.target_tx.to_numpy(dtype=np.float64)
                if not (np.isfinite(Xraw).all() and np.isfinite(target).all()
                        and np.isfinite(weight).all() and (weight > 0).all()):
                    raise ValueError(f'Invalid evaluation data: {scenario}')
                Xeval = np.ascontiguousarray(scaler.transform(Xraw).astype(np.float32))
                preds = {'persistence':frame.per_t_lag0.to_numpy(dtype=np.float64)}
                for name, model in models.items():
                    preds[name] = predict(model, Xeval, args.eval_batch_size, args.device)
                for name, p in preds.items():
                    scenario_acc[name].update(target,p,weight)
                    fold_acc[name].update(target,p,weight)
                    overall[name].update(target,p,weight)
            for name, acc in scenario_acc.items():
                rows.append(dict(level='scenario',fold=fold+1,scenario=scenario,
                                 model=name,**acc.result()))
            print(f'  evaluated {scenario}: {pf.metadata.num_rows:,} rows',flush=True)
        for name,acc in fold_acc.items():
            rows.append(dict(level='fold',fold=fold+1,scenario='__fold__',
                             model=name,**acc.result()))
        pd.DataFrame(rows).to_csv(args.output_dir/'metrics_partial.csv',index=False)
        del models,scaler
    if args.fold is None:
        for name,acc in overall.items():
            rows.append(dict(level='overall',fold=0,scenario='__pooled__',
                             model=name,**acc.result()))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir/'metrics.csv',index=False)
    print('\n=== RESULTS ===')
    print(metrics[metrics.level==('overall' if args.fold is None else 'fold')]
          .drop(columns=['level','fold','scenario']).to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    if args.fold is None:
        pivot = metrics[metrics.level=='scenario'].pivot(index='scenario',columns='model',values='rmse')
        gains = pd.DataFrame({n+'_gain_pct':100*(1-pivot[n]/pivot.persistence) for n in ('mlp','cnn')})
        gains.to_csv(args.output_dir/'scenario_gains.csv')
        print('\n=== SCENARIO GAINS ===')
        print(gains.agg(['mean','median','std','min','max']).to_string())
        for n in ('mlp','cnn'):
            print(f'{n} wins: {(gains[n+"_gain_pct"]>0).sum()}/27')
    print('Saved:',args.output_dir/'metrics.csv')


if __name__=='__main__':
    main()
