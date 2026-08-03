#!/usr/bin/env python3
"""Disk-backed strict-holdout R3 base-capacity ladder."""
from __future__ import annotations
import argparse,json,gc
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
PARAMS=dict(n_estimators=300,learning_rate=.04,num_leaves=63,min_child_samples=400,subsample=.8,colsample_bytree=.8,reg_lambda=12.,random_state=20260802,n_jobs=1,verbosity=-1)
TRAIN_END=pd.Timestamp('2024-03-01',tz='UTC');CAL_END=pd.Timestamp('2024-05-01',tz='UTC');EVAL_END=pd.Timestamp('2024-12-01',tz='UTC')
def read(p,cols=None): return pd.read_parquet(p,columns=cols)
def main():
 p=argparse.ArgumentParser();p.add_argument('--matrix',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--n-features',type=int,default=40);p.add_argument('--fit-only',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 man=json.loads((a.matrix/'manifest.json').read_text());pool=man['feature_pool'];files=[a.matrix/x['path'] for x in man['parts']]
 # Streaming train-only covariance ranking; zero-filled values were materialised
 # before this stage and coverage was already enforced by the source panel.
 stats={c:np.zeros(5) for c in pool};total=0
 for f in files:
  x=read(f,['available','y',*pool]);x=x.loc[pd.to_datetime(x.available,utc=True)<TRAIN_END];y=x.y.to_numpy(float);total+=len(x)
  for c in pool:
   v=x[c].to_numpy(float);stats[c]+=np.array([len(v),v.sum(),np.dot(v,v),y.sum(),np.dot(v,y)])
  del x;gc.collect()
 rank=[]
 for c,(n,sx,sxx,sy,sxy) in stats.items(): rank.append((abs((n*sxy-sx*sy)/np.sqrt(max(n*sxx-sx*sx,1))),c))
 selected=[c for _,c in sorted(rank,reverse=True)[:a.n_features]]
 def load_window(start,end,identity=False):
  frames=[]; cols=['available','y','t4_tp6_sl4_net_bps',*selected]
  if identity:cols=['candidate_id','__ts__','t4_tp6_sl4_gross_bps',*cols]
  for f in files:
   x=read(f,cols);available=pd.to_datetime(x.available,utc=True);x=x.loc[(available>=start)&(available<end)];frames.append(x)
  return pd.concat(frames,ignore_index=True)
 train=load_window(pd.Timestamp('2020-01-01',tz='UTC'),TRAIN_END);X=train[selected].to_numpy('float32');m=lgb.LGBMClassifier(objective='multiclass',num_class=3,**PARAMS).fit(X,train.y);del X,train;gc.collect()
 cal=load_window(TRAIN_END,CAL_END);pc=m.predict_proba(cal[selected].to_numpy('float32'));raw_cal=pc[:,2]-pc[:,0];edges=np.unique(np.quantile(raw_cal,np.linspace(0,1,11)));bins=np.clip(np.digitize(raw_cal,edges[1:-1],right=True),0,9);net=cal.t4_tp6_sl4_net_bps.to_numpy();means=np.array([net[bins==i].mean() if (bins==i).any() else net.mean() for i in range(10)]);m.booster_.save_model(str(a.out/'model.txt'));(a.out/'fit_state.json').write_text(json.dumps({'selected_features':selected,'edges':edges.tolist(),'means':means.tolist(),'source_matrix':str(a.matrix)},indent=2));
 if a.fit_only: return
 del cal,pc;gc.collect()
 pred_root=a.out/'prediction_parts';pred_root.mkdir();pred_paths=[]
 for f in files:
  test=read(f,['candidate_id','__ts__','available','t4_tp6_sl4_gross_bps','t4_tp6_sl4_net_bps',*selected]);available=pd.to_datetime(test.available,utc=True);test=test.loc[(available>=CAL_END)&(available<EVAL_END)]
  if test.empty: continue
  pt=m.predict_proba(test[selected].to_numpy('float32'));raw=pt[:,2]-pt[:,0];test['score_bps']=means[np.clip(np.digitize(raw,edges[1:-1],right=True),0,9)];test['raw_prediction']=raw;out_part=test.rename(columns={'t4_tp6_sl4_gross_bps':'gross_bps','t4_tp6_sl4_net_bps':'net_bps'})[['candidate_id','__ts__','gross_bps','net_bps','score_bps','raw_prediction']];dst=pred_root/f.name;out_part.to_parquet(dst,index=False);pred_paths.append(dst);del test,out_part,pt;gc.collect()
 out=pd.concat([read(x) for x in pred_paths],ignore_index=True);out.to_parquet(a.out/'predictions.parquet',index=False)
 results=[]
 for f in (.01,.05,.10):
  z=out.nlargest(int(np.ceil(len(out)*f)),'score_bps');results.append({'top_fraction':f,'rows':len(z),'gross_bps':z.gross_bps.mean(),'net_bps':z.net_bps.mean()})
 pd.DataFrame(results).to_parquet(a.out/'results.parquet',index=False);(a.out/'manifest.json').write_text(json.dumps({'source_matrix':str(a.matrix),'selected_features':selected,'selection':'strict pre-March only covariance ranking; configured base pool only','rows':{'train':int(total),'test':len(out)},'metrics':results},indent=2));print(pd.DataFrame(results).to_string(index=False))
if __name__=='__main__':main()
