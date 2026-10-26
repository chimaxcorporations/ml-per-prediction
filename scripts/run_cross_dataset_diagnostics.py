#!/usr/bin/env python3
"""Frozen train-on-one-dataset/test-on-the-other H10 common-40 PER transfer benchmark."""
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

BASE=('per_t','dist_m','v_src_mps','v_dst_mps')
FEATURES=[f'{c}_lag{lag}' for lag in range(10) for c in BASE]
COLS=FEATURES+['target_per','target_tx']
PATTERN=re.compile(r'^juelich_core_d(10|30|60)_r(2|5|10)_(log_distance|log_distance_nakagami|urban)_s\d+_w\d+\.parquet$')
MODELS=('persistence','ridge','hist_gb','mlp','cnn')

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(40,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(-1)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv1d(4,16,3,padding=1),nn.ReLU(),nn.Conv1d(16,16,3,padding=1),nn.ReLU())
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(160,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x): return self.head(self.conv(x.reshape(-1,10,4).transpose(1,2))).squeeze(-1)

class Metrics:
    def __init__(self): self.n=0;self.sy=0.;self.sy2=0.;self.ae=0.;self.se=0.;self.w=0.;self.wae=0.;self.wse=0.
    def update(self,y,p,w):
        e=y-p; self.n+=len(y);self.sy+=float(y.sum());self.sy2+=float(np.dot(y,y))
        self.ae+=float(np.abs(e).sum());self.se+=float(np.dot(e,e));self.w+=float(w.sum())
        self.wae+=float(np.dot(w,np.abs(e)));self.wse+=float(np.dot(w,e*e))
    def result(self):
        ss=self.sy2-self.sy*self.sy/self.n
        return dict(samples=self.n,total_tx=self.w,mae=self.ae/self.n,rmse=(self.se/self.n)**.5,
                    r2=1-self.se/ss if ss>0 else np.nan,weighted_mae=self.wae/self.w,
                    weighted_rmse=(self.wse/self.w)**.5)

def check(frame,origin):
    missing=set(COLS)-set(frame.columns)
    if missing: raise ValueError(f'{origin} missing {sorted(missing)}')
    arr=frame[COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all():raise ValueError(f'{origin}: nonfinite values')
    if (frame.target_tx.to_numpy()<=0).any():raise ValueError(f'{origin}: invalid target_tx')
    if ((frame.target_per.to_numpy()<0)|(frame.target_per.to_numpy()>1)).any():raise ValueError(f'{origin}: invalid target PER')

def sample_juelich(files,n,seed,batch):
    frames=[]
    for idx,path in enumerate(files):
        pf=pq.ParquetFile(path)
        sel=np.sort(np.random.default_rng(seed+idx).choice(pf.metadata.num_rows,min(n,pf.metadata.num_rows),replace=False))
        offset=0;pieces=[]
        for chunk in pf.iter_batches(batch_size=batch,columns=FEATURES+['target_per']):
            end=offset+chunk.num_rows;lo=np.searchsorted(sel,offset);hi=np.searchsorted(sel,end)
            if hi>lo: pieces.append(chunk.to_pandas().iloc[sel[lo:hi]-offset])
            offset=end
        frames.append(pd.concat(pieces,ignore_index=True))
        print(f'  sampled {len(frames[-1]):,}: {path.name}',flush=True)
    return pd.concat(frames,ignore_index=True)

def train(X,y,args):
    scaler=StandardScaler().fit(X)
    x32=np.ascontiguousarray(scaler.transform(X).astype(np.float32)); y32=np.asarray(y,dtype=np.float32)
    models={'ridge':make_pipeline(StandardScaler(),Ridge(alpha=1.)).fit(X,y),
            'hist_gb':HistGradientBoostingRegressor(max_iter=150,random_state=args.seed).fit(X,y)}
    for i,(name,cls) in enumerate((('mlp',MLP),('cnn',CNN))):
        seed=args.seed+10+i;torch.manual_seed(seed)
        model=cls();loader=DataLoader(TensorDataset(torch.from_numpy(x32),torch.from_numpy(y32)),
            batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed))
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        for epoch in range(args.epochs):
            model.train();loss_sum=0.
            for xb,yb in loader:
                opt.zero_grad(set_to_none=True);loss=((model(xb)-yb)**2).mean();loss.backward();opt.step()
                loss_sum+=float(loss.detach())*len(yb)
            print(f'  {name} epoch {epoch+1}/{args.epochs}: MSE={loss_sum/len(y):.7f}',flush=True)
        models[name]=model.eval()
    return scaler,models

def predict_nn(model,scaler,X,batch):
    result=[]
    with torch.inference_mode():
        for i in range(0,len(X),batch):
            x=np.ascontiguousarray(scaler.transform(X[i:i+batch]).astype(np.float32))
            result.append(model(torch.from_numpy(x)).numpy())
    return np.clip(np.concatenate(result).astype(np.float64),0,1)

# Fixed diagnostic intervals; upper edges are inclusive only in the last bin.
PER_EDGES=np.array([0., .01, .1, .25, .5, .75, 1.0000001])
DIST_EDGES=np.array([0., 25., 50., 100., 200., 300., 400., np.inf])

class ErrorSummary:
    def __init__(self):
        self.n=0; self.sum_y=0.; self.sum_p=0.; self.sum_e=0.; self.sum_ae=0.; self.sum_se=0.
    def update(self,y,p):
        e=p-y
        self.n+=len(y); self.sum_y+=float(np.sum(y)); self.sum_p+=float(np.sum(p))
        self.sum_e+=float(np.sum(e)); self.sum_ae+=float(np.sum(np.abs(e)))
        self.sum_se+=float(np.dot(e,e))
    def result(self):
        return dict(samples=self.n, mean_actual_per=self.sum_y/self.n,
                    mean_predicted_per=self.sum_p/self.n, bias=self.sum_e/self.n,
                    mae=self.sum_ae/self.n, rmse=(self.sum_se/self.n)**.5)

def interval_label(edges,idx):
    right=edges[idx+1]
    return f'[{edges[idx]:g}, {right:g}{"]" if np.isinf(right) else ")"}'

def record_bins(groups,direction,scenario,model,dimension,values,edges,y,p):
    ids=np.searchsorted(edges,values,side='right')-1
    ids=np.clip(ids,0,len(edges)-2)
    for j in np.unique(ids):
        mask=ids==j
        groups[(direction,scenario,model,dimension,interval_label(edges,int(j)))].update(y[mask],p[mask])

def evaluate(files,scaler,models,args,direction):
    rows=[]; overall={m:Metrics() for m in MODELS}
    diagnostic_overall={m:ErrorSummary() for m in MODELS}
    group_stats=defaultdict(ErrorSummary)
    diag_rows=[]; sampled=[]
    rng=np.random.default_rng(args.diagnostic_seed)
    for path in files:
        acc={m:Metrics() for m in MODELS}
        diag_acc={m:ErrorSummary() for m in MODELS}
        pf=pq.ParquetFile(path)
        for chunk in pf.iter_batches(batch_size=args.eval_batch_size,columns=COLS):
            df=chunk.to_pandas();check(df,path.name)
            X=df[FEATURES].to_numpy(dtype=np.float64);y=df.target_per.to_numpy(dtype=np.float64)
            w=df.target_tx.to_numpy(dtype=np.float64)
            predictions={'persistence':df.per_t_lag0.to_numpy(dtype=np.float64),
                'ridge':np.clip(models['ridge'].predict(X),0,1),
                'hist_gb':np.clip(models['hist_gb'].predict(X),0,1)}
            for name in ('mlp','cnn'):
                predictions[name]=predict_nn(models[name],scaler,X,args.batch_size*4)
            for name,p in predictions.items():
                acc[name].update(y,p,w);overall[name].update(y,p,w)
                diag_acc[name].update(y,p);diagnostic_overall[name].update(y,p)
                record_bins(group_stats,direction,path.stem,name,'true_per',y,PER_EDGES,y,p)
                record_bins(group_stats,direction,path.stem,name,'distance_m',
                            df.dist_m_lag0.to_numpy(dtype=np.float64),DIST_EDGES,y,p)
                # Identical random indices across models within each batch.
                if name=='persistence':
                    selected=rng.random(len(y)) < args.prediction_sample_fraction
                if selected.any():
                    sampled.append(pd.DataFrame(dict(direction=direction,scenario=path.stem,
                        model=name,true_per=y[selected],predicted_per=p[selected],
                        distance_m=df.dist_m_lag0.to_numpy(dtype=np.float64)[selected],
                        abs_error=np.abs(p[selected]-y[selected]),
                        signed_error=p[selected]-y[selected])))
        for name in MODELS:
            rows.append(dict(direction=direction,level='scenario',scenario=path.stem,model=name,**acc[name].result()))
            diag_rows.append(dict(direction=direction,level='scenario',scenario=path.stem,
                                  model=name,**diag_acc[name].result()))
        print(f'  evaluated {path.name}: {pf.metadata.num_rows:,} samples',flush=True)
        pd.DataFrame(rows).to_csv(args.output_dir/f'{direction}_partial.csv',index=False)
    for name in MODELS:
        rows.append(dict(direction=direction,level='overall',scenario='ALL',model=name,**overall[name].result()))
        diag_rows.append(dict(direction=direction,level='overall',scenario='ALL',
                              model=name,**diagnostic_overall[name].result()))
    pd.DataFrame(diag_rows).to_csv(args.output_dir/f'{direction}_error_summary.csv',index=False)
    pd.DataFrame([dict(direction=d,scenario=s,model=m,dimension=dim,interval=interval,
                       **stat.result()) for (d,s,m,dim,interval),stat in group_stats.items()]
                ).to_csv(args.output_dir/f'{direction}_error_by_bin.csv',index=False)
    if sampled:
        sample=pd.concat(sampled,ignore_index=True)
        sample.to_parquet(args.output_dir/f'{direction}_prediction_sample.parquet',index=False)
        quantiles=(sample.groupby('model').abs_error.quantile([.5,.9,.95,.99])
                   .unstack().rename(columns={.5:'ae_q50',.9:'ae_q90',.95:'ae_q95',.99:'ae_q99'}))
        quantiles.to_csv(args.output_dir/f'{direction}_sample_error_quantiles.csv')
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--juelich-dir',type=Path,default=Path('/workspace/data/temporal_h10'))
    ap.add_argument('--berlin-dir',type=Path,default=Path('/workspace/data/berlin/temporal_h10'))
    ap.add_argument('--output-dir',type=Path,default=Path('/workspace/results/cross_dataset_h10_common40'))
    ap.add_argument('--direction',choices=['both','juelich_to_berlin','berlin_to_juelich'],default='both')
    ap.add_argument('--samples-per-scenario',type=int,default=30000)
    ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--batch-size',type=int,default=2048)
    ap.add_argument('--eval-batch-size',type=int,default=50000)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--diagnostic-seed',type=int,default=2026)
    ap.add_argument('--prediction-sample-fraction',type=float,default=.01)
    args=ap.parse_args()
    if min(args.samples_per_scenario,args.epochs,args.batch_size,args.eval_batch_size,args.threads)<1:ap.error('All numerical settings must be positive')
    if not (0 < args.prediction_sample_fraction <= 1):ap.error('prediction-sample-fraction must be in (0,1]')
    if args.output_dir.resolve() in [Path('/workspace/results/berlin_cv_h10').resolve(),Path('/workspace/results/juelich_cv_h10_common40').resolve()]:
        ap.error('Output directory must not be an existing benchmark directory')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):ap.error('Output directory is not empty; use a new directory to preserve previous results')
    files=sorted(p for p in args.juelich_dir.glob('juelich_core_*.parquet') if PATTERN.fullmatch(p.name))
    if len(files)!=27:raise ValueError(f'Expected 27 Jülich files, found {len(files)}')
    berlin=[args.berlin_dir/f'berlin_{s}_h10.parquet' for s in ('S1','S2')]
    if any(not p.is_file() for p in berlin):raise FileNotFoundError('Berlin S1/S2 temporal Parquet files missing')
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'settings.json').write_text(json.dumps(dict(features=FEATURES,history=10,seed=args.seed,
        samples_per_juelich_scenario=args.samples_per_scenario,epochs=args.epochs,batch_size=args.batch_size,
        direction=args.direction,diagnostic_seed=args.diagnostic_seed,
        prediction_sample_fraction=args.prediction_sample_fraction,protocol='No target-domain training, scaler fitting, fine-tuning or model selection',
        caution='Cross-technology transfer: IEEE 802.11p to/from LTE-V2X; different PER denominators'),indent=2)+'\n')
    rows=[]
    if args.direction in ('both','juelich_to_berlin'):
        print('\n=== TRAIN JÜLICH -> TEST BERLIN ===',flush=True)
        tr=sample_juelich(files,args.samples_per_scenario,args.seed,args.eval_batch_size)
        X=tr[FEATURES].to_numpy(dtype=np.float64);y=tr.target_per.to_numpy(dtype=np.float64)
        if not (np.isfinite(X).all() and np.isfinite(y).all()):raise ValueError('Invalid Jülich training data')
        scaler,models=train(X,y,args);del X,y,tr
        rows+=evaluate(berlin,scaler,models,args,'juelich_to_berlin');del models,scaler
    if args.direction in ('both','berlin_to_juelich'):
        print('\n=== TRAIN BERLIN -> TEST JÜLICH ===',flush=True)
        tr=pd.concat([pd.read_parquet(p,columns=COLS) for p in berlin],ignore_index=True)
        check(tr,'Berlin training')
        X=tr[FEATURES].to_numpy(dtype=np.float64);y=tr.target_per.to_numpy(dtype=np.float64)
        scaler,models=train(X,y,args);del X,y,tr
        rows+=evaluate(files,scaler,models,args,'berlin_to_juelich');del models,scaler
    df=pd.DataFrame(rows);df.to_csv(args.output_dir/'metrics.csv',index=False)
    print('\n=== CROSS-DATASET RESULTS ===',flush=True)
    print(df[df.level=='overall'].to_string(index=False),flush=True)
    print('Saved:',args.output_dir/'metrics.csv',flush=True)
if __name__=='__main__':main()
