#!/usr/bin/env python3
"""Two-direction Berlin scenario-transfer benchmark: persistence, Ridge, HistGB, MLP, CNN."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

BASE = ['per_t', 'dist_m', 'v_src_mps', 'v_dst_mps']
FEATURES = [f'{c}_lag{lag}' for lag in range(10) for c in BASE]

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(40,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(-1)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv1d(4,16,3,padding=1),nn.ReLU(),nn.Conv1d(16,16,3,padding=1),nn.ReLU())
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(160,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x): return self.head(self.conv(x.reshape(-1,10,4).transpose(1,2))).squeeze(-1)

def fit_nn(model, X, y, epochs, batch_size, seed):
    torch.manual_seed(seed)
    ds=TensorDataset(torch.from_numpy(X),torch.from_numpy(y))
    loader=DataLoader(ds,batch_size=batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed))
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    for epoch in range(epochs):
        model.train(); total=0.; count=0
        for xx,yy in loader:
            opt.zero_grad(set_to_none=True)
            loss=((model(xx)-yy)**2).mean()
            loss.backward();opt.step()
            total+=float(loss.detach())*len(yy);count+=len(yy)
        print(f'    epoch {epoch+1}/{epochs}: MSE={total/count:.7f}',flush=True)
    return model

def predict_nn(model,X,batch_size=8192):
    model.eval(); output=[]
    with torch.inference_mode():
        for i in range(0,len(X),batch_size):
            output.append(model(torch.from_numpy(X[i:i+batch_size])).numpy())
    return np.clip(np.concatenate(output).astype(np.float64),0,1)

def metrics(y,p,w):
    e=y-p; avg=y.mean(); sst=np.sum((y-avg)**2)
    return dict(samples=len(y),total_tx=float(w.sum()),mae=float(np.mean(abs(e))),
                rmse=float(np.sqrt(np.mean(e**2))),r2=float(1-np.sum(e**2)/sst) if sst>0 else None,
                weighted_mae=float(np.average(abs(e),weights=w)),
                weighted_rmse=float(np.sqrt(np.average(e**2,weights=w))))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir',type=Path,default=Path('/workspace/data/berlin/temporal_h10'))
    ap.add_argument('--output-dir',type=Path,default=Path('/workspace/results/berlin_cv_h10'))
    ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--batch-size',type=int,default=2048)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--fold',choices=['S1','S2'],help='Only evaluate named scenario (smoke test)')
    args=ap.parse_args()
    if min(args.epochs,args.batch_size,args.threads)<1: ap.error('epochs, batch-size, threads must be positive')
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    frames={s:pd.read_parquet(args.input_dir/f'berlin_{s}_h10.parquet') for s in ['S1','S2']}
    for s,df in frames.items():
        needed=FEATURES+['target_per','target_tx']
        missing=set(needed)-set(df.columns)
        if missing: raise ValueError(f'{s} missing columns: {sorted(missing)}')
        if not np.isfinite(df[needed].to_numpy(dtype=np.float64)).all(): raise ValueError(f'{s} nonfinite values')
    results=[]; pooled={m:[] for m in ['persistence','ridge','hist_gb','mlp','cnn']}
    for fold, test_s in enumerate(['S2','S1'],start=1):
        if args.fold and test_s!=args.fold: continue
        train_s='S1' if test_s=='S2' else 'S2'
        print(f'\nFold {fold}: train {train_s} ({len(frames[train_s]):,}), test {test_s} ({len(frames[test_s]):,})',flush=True)
        tr=frames[train_s];te=frames[test_s]
        Xtr=tr[FEATURES].to_numpy(dtype=np.float64);ytr=tr.target_per.to_numpy(dtype=np.float64)
        Xte=te[FEATURES].to_numpy(dtype=np.float64);yte=te.target_per.to_numpy(dtype=np.float64)
        w=te.target_tx.to_numpy(dtype=np.float64)
        scaler=StandardScaler().fit(Xtr)
        Xtr_nn=np.ascontiguousarray(scaler.transform(Xtr).astype(np.float32))
        Xte_nn=np.ascontiguousarray(scaler.transform(Xte).astype(np.float32))
        ridge=make_pipeline(StandardScaler(),Ridge(alpha=1.0)).fit(Xtr,ytr)
        hist=HistGradientBoostingRegressor(max_iter=150,random_state=args.seed).fit(Xtr,ytr)
        predictions={'persistence':te['per_t_lag0'].to_numpy(dtype=np.float64),
                     'ridge':np.clip(ridge.predict(Xte),0,1),
                     'hist_gb':np.clip(hist.predict(Xte),0,1)}
        for j,(name,cls) in enumerate([('mlp',MLP),('cnn',CNN)]):
            seed=args.seed+fold*10+j
            torch.manual_seed(seed)
            model=cls()
            print(f'  {name}: {sum(p.numel() for p in model.parameters()):,} parameters',flush=True)
            fit_nn(model,Xtr_nn,ytr.astype(np.float32),args.epochs,args.batch_size,seed)
            predictions[name]=predict_nn(model,Xte_nn)
        for name,p in predictions.items():
            record=dict(level='scenario',fold=fold,scenario=test_s,train_scenario=train_s,model=name,**metrics(yte,p,w))
            results.append(record)
            pooled[name].append((yte,p,w))
            print(f'  {name:12s} RMSE={record["rmse"]:.6f} weighted_RMSE={record["weighted_rmse"]:.6f}',flush=True)
        pd.DataFrame(results).to_csv(args.output_dir/'metrics_partial.csv',index=False)
    if not args.fold:
        for name,parts in pooled.items():
            y=np.concatenate([p[0] for p in parts]);p=np.concatenate([p[1] for p in parts]);w=np.concatenate([p[2] for p in parts])
            results.append(dict(level='overall',fold=None,scenario='ALL',train_scenario='both',model=name,**metrics(y,p,w)))
    result=pd.DataFrame(results)
    result.to_csv(args.output_dir/'metrics.csv',index=False)
    (args.output_dir/'settings.json').write_text(json.dumps({'features':FEATURES,'history':10,'epochs':args.epochs,'batch_size':args.batch_size,'seed':args.seed,'protocol':'Train S1 test S2, train S2 test S1; no test-based tuning','weights':'nominal future transmission rate','fold':args.fold},indent=2)+'\n')
    print('\n=== RESULTS ===')
    print(result.to_string(index=False))
    print('Saved:',args.output_dir/'metrics.csv')
if __name__=='__main__': main()
