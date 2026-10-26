#!/usr/bin/env python3
"""Pass C3: matched Berlin test rows, persistence, size-matched Juelich Ridge.
Requires the original Pass B paired_predictions.parquet and 27 temporal Juelich files.
Does not replace/overwrite any previous experiment results.
"""
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

def sample_file(path, n, seed, batch_size):
    pf=pq.ParquetFile(path)
    if pf.metadata.num_rows<n: raise ValueError(f'{path}: only {pf.metadata.num_rows} rows, need {n}')
    idx=np.sort(np.random.default_rng(seed).choice(pf.metadata.num_rows,n,replace=False))
    pieces=[]; offset=0
    for batch in pf.iter_batches(batch_size=batch_size,columns=FEATURES+['target_per']):
        end=offset+batch.num_rows;lo=np.searchsorted(idx,offset);hi=np.searchsorted(idx,end)
        if hi>lo:pieces.append(batch.to_pandas().iloc[idx[lo:hi]-offset])
        offset=end
    return pd.concat(pieces,ignore_index=True)

def metric(y,p):
    return {'rmse':float(np.sqrt(np.mean((y-p)**2))),'mae':float(np.mean(np.abs(y-p)))}

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--juelich-dir',type=Path,default=Path('data/temporal_h10'))
    a.add_argument('--berlin-dir',type=Path,default=Path('data/berlin/temporal_h10'))
    a.add_argument('--pass-b-predictions',type=Path,default=Path('results/pass_b_ridge_paired/paired_predictions.parquet'))
    a.add_argument('--out',type=Path,default=Path('results/pass_c3_size_matched_ridge'))
    a.add_argument('--per-scenario',type=int,default=3704,help='3704 x 27 = 100008 training observations')
    a.add_argument('--seed',type=int,default=42)
    a.add_argument('--bootstrap-seed',type=int,default=20261018)
    a.add_argument('--reps',type=int,default=10000)
    a.add_argument('--batch-size',type=int,default=50000)
    a.add_argument('--tolerance',type=float,default=1e-8)
    args=a.parse_args()
    if args.out.exists() and any(args.out.iterdir()):a.error('Output directory is nonempty; refusing overwrite')
    if args.per_scenario<1 or args.reps<100:a.error('per-scenario >=1 and reps >=100 required')
    files=sorted(p for p in args.juelich_dir.glob('juelich_core_*.parquet') if PATTERN.fullmatch(p.name))
    if len(files)!=27:raise ValueError(f'Expected 27 Juelich files, found {len(files)}')
    reference=pd.read_parquet(args.pass_b_predictions)
    required={'scenario','row_index','true_per','within_ridge','transfer_ridge'}
    if required-set(reference):raise ValueError(f'Missing Pass B columns: {required-set(reference)}')
    if len(reference)!=201229:raise ValueError(f'Expected 201229 matched observations, found {len(reference)}')
    train=pd.concat([sample_file(p,args.per_scenario,args.seed+i,args.batch_size) for i,p in enumerate(files)],ignore_index=True)
    X=train[FEATURES].to_numpy(dtype=np.float64);y=train.target_per.to_numpy(dtype=np.float64)
    if not np.isfinite(X).all() or not np.isfinite(y).all():raise ValueError('Nonfinite training features/targets')
    model=make_pipeline(StandardScaler(),Ridge(alpha=1.0)).fit(X,y)
    records=[]
    for s in ('S1','S2'):
        frame=pd.read_parquet(args.berlin_dir/f'berlin_{s}_h10.parquet')
        original=reference[reference.scenario.eq(s)].sort_values('row_index').reset_index(drop=True)
        if len(frame)!=len(original) or not np.array_equal(original.row_index.to_numpy(),np.arange(len(frame))):
            raise ValueError(f'{s}: reference and target row order mismatch')
        truth=frame.target_per.to_numpy(dtype=float)
        if not np.allclose(truth,original.true_per.to_numpy(dtype=float),rtol=0,atol=1e-12):
            raise ValueError(f'{s}: target PER differs from original matched evaluation')
        for c in ('Source','Destination'):
            if c not in frame:raise ValueError(f'{s}: missing directed link column {c}')
            if c in original and not np.array_equal(frame[c].astype(str).to_numpy(),original[c].astype(str).to_numpy()):
                raise ValueError(f'{s}: {c} differs from original matched evaluation')
        if 'per_t_lag0' not in frame:raise ValueError('Missing per_t_lag0 for persistence')
        persist=np.clip(frame.per_t_lag0.to_numpy(dtype=float),0,1)
        fitted=np.clip(model.predict(frame[FEATURES].to_numpy(dtype=float)),0,1)
        if not np.isfinite(persist).all() or not np.isfinite(fitted).all():raise ValueError('Nonfinite predictions')
        rec=original.copy();rec['size_matched_ridge']=fitted;rec['persistence']=persist
        records.append(rec)
    predictions=pd.concat(records,ignore_index=True)
    # Assert previously reported paired point estimates are recovered exactly on these same observations.
    expected={'S1':(.136911822753,.118063121432),'S2':(.233035701866,.238426709541)}
    for s,(w,t) in expected.items():
        d=predictions[predictions.scenario.eq(s)]; yy=d.true_per.to_numpy()
        if abs(metric(yy,d.within_ridge.to_numpy())['rmse']-w)>args.tolerance:raise ValueError(f'{s}: Berlin Ridge reference mismatch')
        if abs(metric(yy,d.transfer_ridge.to_numpy())['rmse']-t)>args.tolerance:raise ValueError(f'{s}: full Juelich Ridge reference mismatch')
    cols=['within_ridge','transfer_ridge','size_matched_ridge','persistence']
    rows=[]
    for s in ('S1','S2','pooled'):
        d=predictions if s=='pooled' else predictions[predictions.scenario.eq(s)]
        for name in cols:rows.append({'scenario':s,'model':name,'n_predictions':len(d),**metric(d.true_per.to_numpy(),d[name].to_numpy())})
    metrics=pd.DataFrame(rows)
    # Paired directed-link bootstrap, stratified by Berlin scenario, with identical draws for every model.
    groups=[]
    for (s,src,dst),d in predictions.groupby(['scenario','Source','Destination'],sort=False,dropna=False):
        entry={'scenario':s,'source':str(src),'destination':str(dst),'n':len(d)}
        yy=d.true_per.to_numpy()
        for name in cols:entry['sse_'+name]=float(np.square(yy-d[name].to_numpy()).sum())
        groups.append(entry)
    groups=pd.DataFrame(groups)
    if len(groups)!=25 or groups.scenario.value_counts().to_dict()!={'S1':12,'S2':13}:
        raise ValueError(f'Unexpected link grouping: {groups.scenario.value_counts().to_dict()}')
    comparisons=[('size_matched_ridge','within_ridge'),('size_matched_ridge','transfer_ridge'),('transfer_ridge','within_ridge'),('size_matched_ridge','persistence')]
    rng=np.random.default_rng(args.bootstrap_seed)
    draws={(s,x,y):np.empty(args.reps) for s in ('S1','S2','pooled') for x,y in comparisons}
    parts={s:groups[groups.scenario.eq(s)].reset_index(drop=True) for s in ('S1','S2')}
    for rep in range(args.reps):
        picked={s:parts[s].iloc[rng.integers(len(parts[s]),size=len(parts[s]))] for s in ('S1','S2')}
        for s in ('S1','S2','pooled'):
            d=pd.concat([picked['S1'],picked['S2']]) if s=='pooled' else picked[s]
            n=d.n.sum()
            for x,y in comparisons:
                draws[(s,x,y)][rep]=np.sqrt(d['sse_'+x].sum()/n)-np.sqrt(d['sse_'+y].sum()/n)
    ci=[]
    for s in ('S1','S2','pooled'):
        d=groups if s=='pooled' else parts[s];n=d.n.sum()
        for x,y in comparisons:
            delta=np.sqrt(d['sse_'+x].sum()/n)-np.sqrt(d['sse_'+y].sum()/n)
            lo,hi=np.quantile(draws[(s,x,y)],[.025,.975])
            ci.append({'scenario':s,'model_a':x,'model_b':y,'delta_a_minus_b':delta,'ci_low':lo,'ci_high':hi,'n_links':len(d)})
    args.out.mkdir(parents=True)
    predictions.to_parquet(args.out/'matched_predictions.parquet',index=False)
    metrics.to_csv(args.out/'model_metrics.csv',index=False)
    groups.to_csv(args.out/'link_sse.csv',index=False)
    pd.DataFrame(ci).to_csv(args.out/'paired_bootstrap_intervals.csv',index=False)
    report={'training_samples':len(train),'samples_per_scenario':args.per_scenario,'n_scenarios':27,'seed':args.seed,'bootstrap_seed':args.bootstrap_seed,'bootstrap_replicates':args.reps,'notes':'Same 201229 Berlin target rows. Fixed-model conditional paired link bootstrap. Size-matched training count is approximate; domain diversity and sampling remain confounded. Berlin-trained models use opposite Berlin scenario for fitting.'}
    (args.out/'run_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS C3: verified original matched predictions and completed sensitivity analysis')
    print(metrics.to_string(index=False,float_format=lambda x:f'{x:.9f}'))
    print(pd.DataFrame(ci).to_string(index=False,float_format=lambda x:f'{x:.9f}'))
    print('Saved',args.out)
if __name__=='__main__':main()
