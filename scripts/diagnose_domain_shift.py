#!/usr/bin/env python3
"""Non-destructive domain shift and scenario transfer diagnostics for H10 common40."""
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = ('per_t','dist_m','v_src_mps','v_dst_mps')
COLUMNS = [f'{c}_lag0' for c in BASE] + ['target_per']
PATTERN = re.compile(r'^juelich_core_d(10|30|60)_r(2|5|10)_(log_distance|log_distance_nakagami|urban)_s\d+_w\d+\.parquet$')

def summarize(arr, dataset, scenario, variable):
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a): return None
    q = np.quantile(a, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    out = dict(dataset=dataset, scenario=scenario, variable=variable, sampled_rows=len(a),
               mean=float(a.mean()), std=float(a.std()), zero_fraction=float(np.mean(a==0)),
               negative_fraction=float(np.mean(a<0)))
    out.update({f'q{int(p*100):02d}':float(v) for p,v in zip([0,.01,.05,.25,.5,.75,.95,.99,1],q)})
    return out

def sampled_file(path, max_rows, seed):
    pf = pq.ParquetFile(path)
    absent = set(COLUMNS)-set(pf.schema_arrow.names)
    if absent: raise ValueError(f'{path}: missing {sorted(absent)}')
    n = pf.metadata.num_rows
    # Deterministic uniform random sample, reading only the five relevant columns.
    chosen = np.sort(np.random.default_rng(seed).choice(n, min(n,max_rows), replace=False))
    offset=0; parts=[]
    for batch in pf.iter_batches(batch_size=100000, columns=COLUMNS):
        end=offset+batch.num_rows
        lo,hi=np.searchsorted(chosen,[offset,end])
        if hi>lo: parts.append(batch.to_pandas().iloc[chosen[lo:hi]-offset])
        offset=end
    return pd.concat(parts, ignore_index=True), n

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--juelich-dir',type=Path,default=Path('/workspace/data/temporal_h10'))
    p.add_argument('--berlin-dir',type=Path,default=Path('/workspace/data/berlin/temporal_h10'))
    p.add_argument('--transfer-dir',type=Path,default=Path('/workspace/results/cross_dataset_h10_common40'))
    p.add_argument('--output-dir',type=Path,default=Path('/workspace/results/diagnostics_h10_common40'))
    p.add_argument('--sample-per-scenario',type=int,default=30000)
    p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args()
    if a.sample_per_scenario<1:p.error('sample-per-scenario must be positive')
    if a.output_dir.exists() and any(a.output_dir.iterdir()):p.error('Output directory is not empty; choose a new one')
    jf=sorted(x for x in a.juelich_dir.glob('juelich_core_*.parquet') if PATTERN.fullmatch(x.name))
    bf=[a.berlin_dir/f'berlin_{s}_h10.parquet' for s in ('S1','S2')]
    if len(jf)!=27 or not all(x.is_file() for x in bf):raise ValueError('Expected 27 Jülich files and 2 Berlin files')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    summary=[]; samples=[]; counts=[]
    for i,(dataset,path) in enumerate([('juelich',x) for x in jf]+[('berlin',x) for x in bf]):
        df,n=sampled_file(path,a.sample_per_scenario,a.seed+i)
        scenario=path.stem
        counts.append(dict(dataset=dataset,scenario=scenario,full_rows=n,sampled_rows=len(df)))
        df.insert(0,'dataset',dataset);df.insert(1,'scenario',scenario)
        samples.append(df)
        for c in COLUMNS:
            summary.append(summarize(df[c],dataset,scenario,c))
        print(f'{dataset:7} {scenario}: {len(df):,} sampled / {n:,} total',flush=True)
    samples=pd.concat(samples,ignore_index=True)
    pd.DataFrame(counts).to_csv(a.output_dir/'sampling_counts.csv',index=False)
    pd.DataFrame(summary).to_csv(a.output_dir/'scenario_distributions.csv',index=False)
    # Equal-scenario-weight distributions, to avoid dominance of dense Jülich scenarios.
    samples.to_parquet(a.output_dir/'distribution_sample.parquet',index=False)
    pooled=[]
    for dataset,group in samples.groupby('dataset'):
        for c in COLUMNS:pooled.append(summarize(group[c],dataset,'EQUAL_SCENARIO_SAMPLE',c))
    pd.DataFrame(pooled).to_csv(a.output_dir/'domain_distributions.csv',index=False)
    print('\n=== DOMAIN DISTRIBUTIONS (equal-size scenario samples) ===')
    print(pd.DataFrame(pooled)[['dataset','variable','mean','std','zero_fraction','q50','q95','q99']].to_string(index=False))
    # Existing cross-domain metrics: no retraining and no modification.
    metric_files=[a.transfer_dir/d/'metrics.csv' for d in ('juelich_to_berlin','berlin_to_juelich')]
    if all(x.exists() for x in metric_files):
        metrics=pd.concat([pd.read_csv(x) for x in metric_files],ignore_index=True)
        scenario=metrics[metrics.level=='scenario'].copy()
        if len(scenario):
            baseline=scenario[scenario.model=='persistence'][['direction','scenario','rmse']].rename(columns={'rmse':'persistence_rmse'})
            scenario=scenario.merge(baseline,on=['direction','scenario'],validate='many_to_one')
            scenario['rmse_gain_pct']=100*(1-scenario.rmse/scenario.persistence_rmse)
            scenario.to_csv(a.output_dir/'transfer_scenario_metrics.csv',index=False)
            print('\n=== TRANSFER SCENARIO RMSE GAIN (%) ===')
            print(scenario.pivot_table(index=['direction','scenario'],columns='model',values='rmse_gain_pct').round(2).to_string())
        metrics[metrics.level=='overall'].to_csv(a.output_dir/'transfer_overall_metrics.csv',index=False)
    else:
        print('Transfer metrics missing: skipped transfer scenario summary.')
    # Visualizations use matplotlib, no fixed colors.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for c in COLUMNS:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for dataset,group in samples.groupby('dataset'):
            values=group[c].to_numpy(dtype=float)
            values=values[np.isfinite(values)]
            lo,hi=np.quantile(values,[.005,.995])
            ax.hist(values,bins=70,range=(lo,hi),density=True,histtype='step',linewidth=1.6,label=dataset)
        ax.set_title(f'Common-feature distribution: {c}');ax.set_xlabel(c);ax.set_ylabel('Density')
        ax.legend();fig.tight_layout();fig.savefig(a.output_dir/f'distribution_{c}.png',dpi=160);plt.close(fig)
    print('\nSaved diagnostic outputs:',a.output_dir)
    print('NOTE: prediction bias/error quantiles require saved per-row predictions or a fresh inference pass.')

if __name__=='__main__':main()
