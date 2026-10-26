#!/usr/bin/env python3
"""Paired stratified link-cluster bootstrap from Berlin grouped SSE exports."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('grouped_csv', type=Path)
p.add_argument('--out-dir', type=Path, default=Path('results/berlin_grouped_uncertainty'))
p.add_argument('--bootstrap', type=int, default=10000)
p.add_argument('--seed', type=int, default=20261018)
p.add_argument('--expected-rmse-csv', type=Path)
p.add_argument('--tolerance', type=float, default=1e-5)
a = p.parse_args()
if a.bootstrap < 100: p.error('Use at least 100 replicates')
df = pd.read_csv(a.grouped_csv)
need = {'experiment','scenario','source','destination','model','samples','sse'}
if need - set(df): raise ValueError(f'Missing columns: {need-set(df)}')
if df.duplicated(['experiment','scenario','source','destination','model']).any():
    raise ValueError('Duplicate grouped rows')
if (df.samples <= 0).any() or (df.sse < 0).any(): raise ValueError('Invalid sufficient statistics')
a.out_dir.mkdir(parents=True,exist_ok=True)
rng = np.random.default_rng(a.seed)
ci_rows, pair_rows, checks = [], [], []
for experiment, sub in df.groupby('experiment'):
    sub = sub.copy()
    sub['link'] = list(zip(sub.scenario.astype(str), sub.source.astype(str), sub.destination.astype(str)))
    models = sorted(sub.model.unique())
    counts = sub.pivot(index='link', columns='model', values='samples').reindex(columns=models)
    sse = sub.pivot(index='link', columns='model', values='sse').reindex(columns=models)
    if counts.isna().any().any() or sse.isna().any().any():
        raise ValueError(f'{experiment}: not all models share all links')
    if not (counts.eq(counts.iloc[:,0],axis=0)).all().all():
        raise ValueError(f'{experiment}: model sample counts differ on same link')
    n = counts.iloc[:,0].to_numpy(dtype=float)
    ss = sse.to_numpy(dtype=float)
    original = np.sqrt(ss.sum(axis=0)/n.sum())
    scenarios = np.array([link[0] for link in counts.index])
    unique_s = sorted(set(scenarios))
    boot = np.empty((a.bootstrap,len(models)))
    for b in range(a.bootstrap):
        ids = []
        for scenario in unique_s:
            pool = np.flatnonzero(scenarios == scenario)
            ids.append(rng.choice(pool,size=len(pool),replace=True))
        idx = np.concatenate(ids)
        boot[b] = np.sqrt(ss[idx].sum(axis=0)/n[idx].sum())
    for j,model in enumerate(models):
        lo,hi = np.quantile(boot[:,j],[.025,.975])
        ci_rows.append(dict(experiment=experiment,model=model,rmse=original[j],ci_low=lo,ci_high=hi,
                            clusters=len(n),scenarios=len(unique_s),samples=int(n.sum())))
        checks.append(dict(experiment=experiment,model=model,rmse=original[j],samples=int(n.sum())))
    for i in range(len(models)):
        for j in range(i+1,len(models)):
            delta = boot[:,i]-boot[:,j]
            lo,hi = np.quantile(delta,[.025,.975])
            pair_rows.append(dict(experiment=experiment,model_a=models[i],model_b=models[j],
                                  delta_rmse=original[i]-original[j],ci_low=lo,ci_high=hi))
if a.expected_rmse_csv:
    expected = pd.read_csv(a.expected_rmse_csv)
    expected = expected[expected.level == 'overall']
    if 'direction' in expected.columns:
        expected = expected[expected.direction.isin(df.experiment.unique())]
    for r in checks:
        match = expected[expected.model.eq(r['model'])]
        if 'direction' in expected.columns: match = match[match.direction.eq(r['experiment'])]
        if len(match) != 1: raise ValueError(f'No unique reference metric: {r}')
        if int(match.iloc[0].samples) != r['samples'] or abs(float(match.iloc[0].rmse)-r['rmse']) > a.tolerance:
            raise ValueError(f'Published benchmark mismatch: {r}, reference={match.iloc[0].to_dict()}')
pd.DataFrame(ci_rows).to_csv(a.out_dir/'rmse_ci.csv',index=False)
pd.DataFrame(pair_rows).to_csv(a.out_dir/'paired_rmse_ci.csv',index=False)
print(pd.DataFrame(ci_rows).to_string(index=False))
print('Saved:',a.out_dir)
