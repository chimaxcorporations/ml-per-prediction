#!/usr/bin/env python3
"""Read-only Pass B checks; run from ml-per-prediction repository root."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument('--models',default='results/juelich_cv_h10_common40_refit/models')
ap.add_argument('--berlin-within',default='results/berlin_cv_h10_common40/metrics.csv')
ap.add_argument('--transfer',default='results/cross_dataset_h10_common40/juelich_to_berlin/metrics.csv')
ap.add_argument('--target-variance',type=float,default=None,help='Exact variance of matched Berlin target PER, if known')
a=ap.parse_args()
report={'status':'partial; raw predictions required for matched-population proof and figure reconstruction'}
try:
 import joblib
 models=sorted(Path(a.models).glob('hist_gb_fold*.joblib'))
 fold=[]
 for path in models:
  obj=joblib.load(path)
  est=obj.steps[-1][1] if hasattr(obj,'steps') else obj
  trees=getattr(est,'_predictors',[])
  nodecounts=[len(t.nodes) for stage in trees for t in stage]
  leaves=[int(np.sum(t.nodes['is_leaf'])) for stage in trees for t in stage]
  fold.append({'file':str(path),'n_iter_':int(est.n_iter_), 'max_iter':int(est.max_iter), 'max_leaf_nodes':est.max_leaf_nodes,'early_stopping':str(est.early_stopping),'do_early_stopping_':str(getattr(est,'do_early_stopping_',None)), 'n_trees':len(nodecounts),'total_nodes':sum(nodecounts),'unique_node_counts':sorted(set(nodecounts)),'unique_leaf_counts':sorted(set(leaves)), 'all_trees_31_leaves':all(x==31 for x in leaves)})
 report['histgb']=fold if fold else 'No fold joblib files found'
except Exception as e: report['histgb_error']=repr(e)

def overall(path):
 p=Path(path)
 if not p.exists():return {'status':'file not found','path':str(p)}
 d=pd.read_csv(p)
 if 'level' in d:d=d[d.level.eq('overall')]
 cols=[x for x in ['direction','model','samples','rmse','mae','r2'] if x in d]
 return d[cols].to_dict('records')
report['berlin_within']=overall(a.berlin_within)
report['juelich_to_berlin']=overall(a.transfer)
if a.target_variance is not None and a.target_variance>0:
 report['nrmse_note']='NRMSE by target standard deviation, on an identical target population'
 for key in ['berlin_within','juelich_to_berlin']:
  if isinstance(report[key],list):
   for r in report[key]:
    if 'rmse' in r:r['nrmse_sd']=float(r['rmse'])/np.sqrt(a.target_variance)
print(json.dumps(report,indent=2,default=str))
