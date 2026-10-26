#!/usr/bin/env python3
"""Reproduce paired Berlin Ridge predictions; verify metrics before bootstrap."""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

BASE=('per_t','dist_m','v_src_mps','v_dst_mps')
FEATURES=[f'{c}_lag{lag}' for lag in range(10) for c in BASE]
PATTERN=re.compile(r'^juelich_core_d(10|30|60)_r(2|5|10)_(log_distance|log_distance_nakagami|urban)_s\d+_w\d+\.parquet$')

def sample_juelich(files,n,seed,batch):
    frames=[]
    for idx,path in enumerate(files):
        pf=pq.ParquetFile(path)
        sel=np.sort(np.random.default_rng(seed+idx).choice(pf.metadata.num_rows,min(n,pf.metadata.num_rows),replace=False))
        offset=0;pieces=[]
        for chunk in pf.iter_batches(batch_size=batch,columns=FEATURES+['target_per']):
            end=offset+chunk.num_rows;lo=np.searchsorted(sel,offset);hi=np.searchsorted(sel,end)
            if hi>lo:pieces.append(chunk.to_pandas().iloc[sel[lo:hi]-offset])
            offset=end
        frames.append(pd.concat(pieces,ignore_index=True))
        print('Sampled',path.name,len(frames[-1]),flush=True)
    return pd.concat(frames,ignore_index=True)

def fit(frame):
    X=frame[FEATURES].to_numpy(dtype=np.float64)
    y=frame['target_per'].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all() or not np.isfinite(y).all():raise ValueError('Nonfinite training values')
    return make_pipeline(StandardScaler(),Ridge(alpha=1.0)).fit(X,y)

def metrics(y,p):
    se=np.square(y-p)
    return dict(samples=len(y),rmse=float(np.sqrt(se.mean())),mae=float(np.abs(y-p).mean()),r2=float(1-se.sum()/np.square(y-y.mean()).sum()))

def reference(path,scenario):
    d=pd.read_csv(path)
    d=d[(d['level']=='scenario')&(d['model']=='ridge')].copy()
    d['scenario']=d['scenario'].astype(str).str.replace(r'^berlin_','',regex=True).str.replace(r'_h10$','',regex=True)
    r=d[d['scenario']==scenario]
    if len(r)!=1:raise ValueError(f'Expected one Ridge scenario row in {path}: {scenario}, found {len(r)}')
    return r.iloc[0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--juelich-dir',type=Path,default=Path('/workspace/data/temporal_h10'))
    ap.add_argument('--berlin-dir',type=Path,default=Path('/workspace/data/berlin/temporal_h10'))
    ap.add_argument('--within-metrics',type=Path,default=Path('/workspace/results/berlin_cv_h10/metrics.csv'))
    ap.add_argument('--transfer-metrics',type=Path,default=Path('/workspace/results/cross_dataset_h10_common40/juelich_to_berlin/metrics.csv'))
    ap.add_argument('--output-dir',type=Path,default=Path('/workspace/results/pass_b_ridge_paired'))
    ap.add_argument('--samples-per-scenario',type=int,default=30000)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--batch-size',type=int,default=50000)
    ap.add_argument('--bootstrap-reps',type=int,default=10000)
    ap.add_argument('--tolerance',type=float,default=1e-8)
    args=ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):ap.error('Output directory must be empty/new; refusing to overwrite')
    if args.bootstrap_reps<1:ap.error('bootstrap-reps must be positive')
    files=sorted(p for p in args.juelich_dir.glob('juelich_core_*.parquet') if PATTERN.fullmatch(p.name))
    if len(files)!=27:raise ValueError(f'Expected 27 Juelich files, found {len(files)}')
    berlin={s:args.berlin_dir/f'berlin_{s}_h10.parquet' for s in ('S1','S2')}
    if not all(p.is_file() for p in berlin.values()):raise FileNotFoundError('Berlin input parquet missing')
    print('Reproducing Juelich training sample and fitting transfer Ridge',flush=True)
    transfer=fit(sample_juelich(files,args.samples_per_scenario,args.seed,args.batch_size))
    print('Fitting Berlin S1/S2 Ridge models',flush=True)
    frames={s:pd.read_parquet(p) for s,p in berlin.items()}
    within={s:fit(frames['S1' if s=='S2' else 'S2']) for s in ('S1','S2')}
    records=[];checks=[];groups=[]
    for s in ('S2','S1'):
        df=frames[s];X=df[FEATURES].to_numpy(dtype=np.float64)
        y=df['target_per'].to_numpy(dtype=np.float64)
        p_within=np.clip(within[s].predict(X),0,1)
        p_transfer=np.clip(transfer.predict(X),0,1)
        for name,p,ref_path in (('within',p_within,args.within_metrics),('transfer',p_transfer,args.transfer_metrics)):
            actual=metrics(y,p);ref=reference(ref_path,s)
            delta=abs(actual['rmse']-float(ref['rmse']))
            ok=(int(actual['samples'])==int(ref['samples']) and delta<=args.tolerance)
            checks.append(dict(scenario=s,model=name,**actual,reference_rmse=float(ref['rmse']),absolute_rmse_difference=delta,passed=bool(ok)))
            print(f'{s} {name}: RMSE={actual["rmse"]:.12f} reference={float(ref["rmse"]):.12f} passed={ok}',flush=True)
        row=pd.DataFrame({'scenario':s,'row_index':np.arange(len(df),dtype=np.int64),'true_per':y,'within_ridge':p_within,'transfer_ridge':p_transfer})
        for key in ('Source','Destination','timestamp'):
            if key in df.columns:row[key]=df[key].to_numpy()
        records.append(row)
        if 'Source' in df and 'Destination' in df:
            # Stable link clusters within scenario; avoid mixing scenarios or directed links.
            keys=list(zip(df['Source'].astype(str),df['Destination'].astype(str)))
            ids=pd.factorize(pd.Series(keys,dtype=object),sort=False)[0]
            for k in np.unique(ids):
                mask=ids==k
                groups.append((s,int(k),int(mask.sum()),float(np.square(y[mask]-p_within[mask]).sum()),float(np.square(y[mask]-p_transfer[mask]).sum())))
        else:
            raise ValueError(f'No Source/Destination identifiers in {s}; cannot perform link bootstrap')
    args.output_dir.mkdir(parents=True)
    pd.DataFrame(checks).to_csv(args.output_dir/'reproduction_checks.csv',index=False)
    passed=all(c['passed'] for c in checks)
    status={'reproduction_passed':passed,'tolerance':args.tolerance,'seed':args.seed,'bootstrap_reps':args.bootstrap_reps}
    if not passed:
        status['message']='Reproduction failed. No paired confidence interval computed or predictions accepted.'
        (args.output_dir/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(status,indent=2));return
    predictions=pd.concat(records,ignore_index=True)
    predictions.to_parquet(args.output_dir/'paired_predictions.parquet',index=False)
    g=pd.DataFrame(groups,columns=['scenario','link_id','n','sse_within','sse_transfer'])
    g.to_csv(args.output_dir/'link_sse.csv',index=False)
    # Stratified link-cluster bootstrap: sample directed links with replacement within S1 and S2.
    rng=np.random.default_rng(args.seed+20261018)
    draws=np.empty(args.bootstrap_reps)
    parts=[g[g.scenario==s].reset_index(drop=True) for s in ('S1','S2')]
    for rep in range(args.bootstrap_reps):
        n=0;sw=0.;st=0.
        for part in parts:
            idx=rng.integers(0,len(part),size=len(part))
            sampled=part.iloc[idx]
            n+=int(sampled.n.sum());sw+=float(sampled.sse_within.sum());st+=float(sampled.sse_transfer.sum())
        draws[rep]=np.sqrt(st/n)-np.sqrt(sw/n)
    point=np.sqrt(g.sse_transfer.sum()/g.n.sum())-np.sqrt(g.sse_within.sum()/g.n.sum())
    status.update({'n_predictions':len(predictions),'n_links':len(g),'delta_rmse_transfer_minus_within':float(point),'bootstrap_ci_95_percentile':np.quantile(draws,[.025,.975]).tolist(),'message':'Conditional stratified directed-link bootstrap; no retraining or multiplicity correction.'})
    (args.output_dir/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    pd.DataFrame({'delta_rmse':draws}).to_csv(args.output_dir/'bootstrap_draws.csv',index=False)
    print(json.dumps(status,indent=2),flush=True)

if __name__=='__main__':main()
