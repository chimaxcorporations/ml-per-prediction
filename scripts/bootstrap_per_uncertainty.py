#!/usr/bin/env python3
"""Paired cluster bootstrap for PER RMSE from scenario-level sufficient statistics.
Run from ml-per-prediction repository root. No model retraining required.
"""
from pathlib import Path
import argparse, re
import numpy as np
import pandas as pd

P = {
 'juelich_classical':'results/juelich_cv_h10_common40/classical/metrics.csv',
 'juelich_neural':'results/juelich_cv_h10_common40/neural/metrics.csv',
 'berlin_to_juelich':'results/cross_dataset_h10_common40/berlin_to_juelich/metrics.csv',
 'temporal_transfer':'results/temporal_transfer_v1/metrics.csv',
}

def load(path, direction=None):
    d=pd.read_csv(path)
    d=d.loc[d.level.eq('scenario')].copy()
    if direction is not None: d=d.loc[d.direction.eq(direction)].copy()
    d['scenario']=d.scenario.astype(str)
    return d[['scenario','model','samples','rmse']].copy()

def mobility(s):
    m=re.search(r'_d(\d+)_r(\d+)_',s)
    if not m: raise ValueError(f'Cannot parse density/rate: {s}')
    return 'd'+m.group(1)+'_r'+m.group(2)

def combine(parts, expected):
    d=pd.concat(parts,ignore_index=True)
    duplicates=d[d.duplicated(['scenario','model'],keep=False)]
    if not duplicates.empty:
        for (_, _), g in duplicates.groupby(['scenario','model']):
            if g.samples.nunique()!=1 or not np.allclose(g.rmse,g.rmse.iloc[0],atol=1e-9,rtol=1e-9):
                raise ValueError('Conflicting duplicate metrics: '+str(g[['scenario','model','samples','rmse']].to_dict('records')))
        d=d.drop_duplicates(['scenario','model'])
    models=sorted(d.model.unique())
    if set(models)!=set(expected): raise ValueError(f'Model mismatch: {models}')
    counts=d.pivot(index='scenario',columns='model',values='samples').sort_index()
    errors=d.pivot(index='scenario',columns='model',values='rmse').reindex(counts.index)
    if counts.isna().any().any() or errors.isna().any().any(): raise ValueError('Missing scenario-model cells')
    if not counts.eq(counts.iloc[:,0],axis=0).all().all(): raise ValueError('Inconsistent test sample counts across models')
    if (counts<=0).any().any() or not np.isfinite(errors.to_numpy()).all(): raise ValueError('Invalid values')
    return counts.index.to_list(),models,counts.iloc[:,0].to_numpy(dtype=float),errors.to_numpy(dtype=float)

def bootstrap(name, parts, expected, nboot, seed, out):
    scenarios,models,n,r=combine(parts,expected)
    sq=r*r
    total=(n[:,None]*sq).sum(axis=0)
    point=np.sqrt(total/n.sum())
    groupings={'scenario':scenarios,'mobility_configuration':[mobility(s) for s in scenarios]}
    for method,labels in groupings.items():
        keys=sorted(set(labels))
        gid=np.array([keys.index(x) for x in labels])
        G=len(keys)
        n_g=np.bincount(gid,weights=n,minlength=G)
        sse_g=np.stack([np.bincount(gid,weights=n*sq[:,j],minlength=G) for j in range(len(models))],axis=1)
        rng=np.random.default_rng(seed)
        boot=np.empty((nboot,len(models)),dtype=float)
        chunk=1000
        for start in range(0,nboot,chunk):
            k=min(chunk,nboot-start)
            draws=rng.integers(0,G,size=(k,G))
            counts=np.zeros((k,G),dtype=np.int32)
            for row in range(k): counts[row]=np.bincount(draws[row],minlength=G)
            boot[start:start+k]=np.sqrt((counts@sse_g)/(counts@n_g)[:,None])
        ci=np.quantile(boot,[.025,.975],axis=0)
        rows=[dict(experiment=name,resampling=method,clusters=G,bootstrap_replicates=nboot,model=m,samples=int(n.sum()),rmse=float(point[j]),ci_low=float(ci[0,j]),ci_high=float(ci[1,j])) for j,m in enumerate(models)]
        pd.DataFrame(rows).to_csv(out/f'{name}_{method}_rmse_ci.csv',index=False)
        comparisons=[]
        for a in range(len(models)):
            for b in range(a+1,len(models)):
                delta=boot[:,a]-boot[:,b]
                lo,hi=np.quantile(delta,[.025,.975])
                comparisons.append(dict(experiment=name,resampling=method,clusters=G,model_a=models[a],model_b=models[b],delta_rmse=float(point[a]-point[b]),ci_low=float(lo),ci_high=float(hi)))
        pd.DataFrame(comparisons).to_csv(out/f'{name}_{method}_paired_differences.csv',index=False)
        print(f'\n{name}: {method}, {G} clusters, {len(scenarios)} scenarios')
        print(pd.DataFrame(rows)[['model','rmse','ci_low','ci_high']].to_string(index=False,float_format=lambda x:f'{x:.6f}'))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bootstrap',type=int,default=10000)
    p.add_argument('--seed',type=int,default=20261018)
    p.add_argument('--output-dir',type=Path,default=Path('results/bootstrap_uncertainty_h10_common40'))
    a=p.parse_args()
    if a.bootstrap<100: p.error('--bootstrap must be at least 100')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    jc=load(P['juelich_classical']); jn=load(P['juelich_neural'])
    bj=load(P['berlin_to_juelich'])
    bt=load(P['temporal_transfer'],'berlin_to_juelich')
    bootstrap('juelich_cv',[jc,jn],['persistence','ridge','hist_gb','mlp','cnn'],a.bootstrap,a.seed,a.output_dir)
    bootstrap('berlin_to_juelich',[bj,bt],['persistence','ridge','hist_gb','mlp','cnn','gru','lstm','tcn','transformer'],a.bootstrap,a.seed,a.output_dir)
    print('\nSaved outputs to',a.output_dir)

if __name__=='__main__':main()
