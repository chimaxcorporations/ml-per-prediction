#!/usr/bin/env python3
"""Pass C2: independently audit saved grouped statistics without retraining."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
p=argparse.ArgumentParser()
p.add_argument('--ridge-links',type=Path,default=Path('results/pass_b_ridge_paired/link_sse.csv'))
p.add_argument('--juelich-classical',type=Path,default=Path('results/juelich_cv_h10_common40/classical/metrics.csv'))
p.add_argument('--juelich-neural',type=Path,default=Path('results/juelich_cv_h10_common40/neural/metrics.csv'))
p.add_argument('--berlin-to-juelich',type=Path,default=Path('results/cross_dataset_h10_common40/berlin_to_juelich/metrics.csv'))
p.add_argument('--temporal-transfer',type=Path,default=Path('results/temporal_transfer_v1/metrics.csv'))
p.add_argument('--out',type=Path,default=Path('results/pass_c2_bootstrap_audit'))
p.add_argument('--reps',type=int,default=10000)
p.add_argument('--seed',type=int,default=20261018)
a=p.parse_args()
if a.reps<100: p.error('At least 100 replicates required')
if a.out.exists() and any(a.out.iterdir()):p.error('Output directory is not empty; refusing to overwrite')
a.out.mkdir(parents=True,exist_ok=True)
report={'bootstrap_reps':a.reps,'base_seed':a.seed,'ridge':{},'nine_group':{}}
# Preserve original matched Ridge audit seed and scenario ordering for direct pooled comparison.
g=pd.read_csv(a.ridge_links)
need={'scenario','n','sse_within','sse_transfer'}
if need-set(g):raise ValueError('Missing Ridge columns '+str(need-set(g)))
if (g.n<=0).any() or (g[['sse_within','sse_transfer']]<0).any().any():raise ValueError('Invalid Ridge SSE')
scenarios=['S1','S2']
if set(g.scenario)!=set(scenarios):raise ValueError('Expected exactly S1 and S2')
parts={s:g[g.scenario==s].reset_index(drop=True) for s in scenarios}
# The original paired script draws S2 then S1, so use that order to reproduce its pooled interval.
rng=np.random.default_rng(42+20261018)
boot={s:np.empty(a.reps) for s in scenarios};pool=np.empty(a.reps)
for b in range(a.reps):
    totals={s:[] for s in scenarios}
    for s in ['S2','S1']:
        part=parts[s]
        idx=rng.integers(0,len(part),size=len(part))
        sampled=part.iloc[idx]
        totals[s]=[sampled.n.sum(),sampled.sse_within.sum(),sampled.sse_transfer.sum()]
        boot[s][b]=np.sqrt(totals[s][2]/totals[s][0])-np.sqrt(totals[s][1]/totals[s][0])
    n=sum(totals[s][0] for s in scenarios)
    pool[b]=np.sqrt(sum(totals[s][2] for s in scenarios)/n)-np.sqrt(sum(totals[s][1] for s in scenarios)/n)
rows=[]
for s in scenarios+['pooled']:
    part=g if s=='pooled' else parts[s]
    n=part.n.sum(); w=np.sqrt(part.sse_within.sum()/n);t=np.sqrt(part.sse_transfer.sum()/n)
    delta=boot[s] if s!='pooled' else pool
    lo,hi=np.quantile(delta,[.025,.975])
    rows.append({'scenario':s,'n_predictions':int(n),'n_links':len(part),'within_rmse':w,'transfer_rmse':t,'delta_transfer_minus_within':t-w,'ci_low':lo,'ci_high':hi})
pd.DataFrame(rows).to_csv(a.out/'ridge_scenario_paired_ci.csv',index=False)
pd.DataFrame({'S1_delta':boot['S1'],'S2_delta':boot['S2'],'pooled_delta':pool}).to_csv(a.out/'ridge_scenario_bootstrap_draws.csv',index=False)
report['ridge']={'rows':rows,'method':'scenario-stratified paired directed-link bootstrap; conditional on fitted models','seed':42+20261018}
# Nine mobility groups: verify group memberships, scenario counts, point estimates, and compare 27-vs-9 resampling.
import re
def load(path,direction=None):
    d=pd.read_csv(path)
    d=d[d.level.eq('scenario')].copy()
    if direction is not None:d=d[d.direction.eq(direction)].copy()
    return d[['scenario','model','samples','rmse']].copy()
def mobility(s):
    m=re.search(r'_d(\d+)_r(\d+)_',str(s))
    if not m:raise ValueError('Cannot parse mobility group '+str(s))
    return f'd{m.group(1)}_r{m.group(2)}'
for exp,paths in {'juelich_cv':[(a.juelich_classical,None),(a.juelich_neural,None)],'berlin_to_juelich':[(a.berlin_to_juelich,None),(a.temporal_transfer,'berlin_to_juelich')]}.items():
    if not all(path.exists() for path,_ in paths):
        report['nine_group'][exp]={'status':'missing_input','paths':[str(path) for path,_ in paths if not path.exists()]};continue
    d=pd.concat([load(path,direction) for path,direction in paths],ignore_index=True)
    if d.duplicated(['scenario','model']).any():
        duplicates=d[d.duplicated(['scenario','model'],keep=False)]
        for _,sub in duplicates.groupby(['scenario','model']):
            if sub.samples.nunique()!=1 or not np.allclose(sub.rmse,sub.rmse.iloc[0],rtol=1e-9,atol=1e-9):
                raise ValueError(f'{exp}: conflicting duplicate scenario metrics')
        d=d.drop_duplicates(['scenario','model'])
    counts=d.pivot(index='scenario',columns='model',values='samples').sort_index()
    rmses=d.pivot(index='scenario',columns='model',values='rmse').reindex(index=counts.index,columns=counts.columns)
    if counts.isna().any().any() or rmses.isna().any().any() or not counts.eq(counts.iloc[:,0],axis=0).all().all():
        raise ValueError(f'{exp}: incomplete or inconsistent scenario-model grid')
    names=counts.index.to_list();groups=[mobility(s) for s in names];unique=sorted(set(groups))
    if len(names)!=27 or len(unique)!=9 or sorted(groups.count(k) for k in unique)!=[3]*9:
        raise ValueError(f'{exp}: expected 27 scenarios in 9 groups of 3, got {len(names)}, {len(unique)}')
    n=counts.iloc[:,0].to_numpy(dtype=float);r=rmses.to_numpy(dtype=float)
    sse=n[:,None]*r*r
    labels=np.array([unique.index(k) for k in groups]);ng=np.bincount(labels,weights=n,minlength=9)
    sg=np.stack([np.bincount(labels,weights=sse[:,j],minlength=9) for j in range(sse.shape[1])],axis=1)
    if not np.allclose(ng.sum(),n.sum()) or not np.allclose(sg.sum(axis=0),sse.sum(axis=0)):
        raise ValueError(f'{exp}: grouping fails SSE conservation')
    group_rows=[]
    for i,k in enumerate(unique):
        group_rows.append({'experiment':exp,'mobility_group':k,'n_scenarios':groups.count(k),'n_predictions':int(ng[i]),'scenario_names':';'.join(names[j] for j in range(len(names)) if labels[j]==i)})
    pd.DataFrame(group_rows).to_csv(a.out/f'{exp}_nine_group_membership.csv',index=False)
    out=[]
    for mode,gn,gs in [('scenario',n,sse),('mobility_configuration',ng,sg)]:
        rng=np.random.default_rng(a.seed);G=len(gn)
        draws=rng.integers(0,G,size=(a.reps,G))
        frequencies=np.zeros((a.reps,G),dtype=np.int32)
        for i,row in enumerate(draws):frequencies[i]=np.bincount(row,minlength=G)
        boot=np.sqrt((frequencies@gs)/(frequencies@gn)[:,None])
        modelnames=list(counts.columns)
        for x,y in [('hist_gb','cnn'),('mlp','persistence')]:
            if x in modelnames and y in modelnames:
                ix=modelnames.index(x);iy=modelnames.index(y)
                point=np.sqrt(gs[:,ix].sum()/gn.sum())-np.sqrt(gs[:,iy].sum()/gn.sum())
                lo,hi=np.quantile(boot[:,ix]-boot[:,iy],[.025,.975])
                out.append({'experiment':exp,'resampling':mode,'clusters':G,'comparison':f'{x}-{y}','delta':point,'ci_low':lo,'ci_high':hi})
    pd.DataFrame(out).to_csv(a.out/f'{exp}_resampling_comparison.csv',index=False)
    report['nine_group'][exp]={'status':'verified','n_scenarios':27,'n_mobility_groups':9,'scenarios_per_group':3,'sse_conserved':True,'comparisons':out}
(a.out/'audit_status.json').write_text(json.dumps(report,indent=2)+'\n')
print('Ridge paired per-scenario intervals:')
print(pd.DataFrame(rows).to_string(index=False,float_format=lambda x:f'{x:.9f}'))
for exp,v in report['nine_group'].items():
    print('\n',exp,v['status'])
    if v['status']=='verified':print(pd.DataFrame(v['comparisons']).to_string(index=False,float_format=lambda x:f'{x:.9f}'))
    else:print(v['paths'])
print('Saved',a.out)
