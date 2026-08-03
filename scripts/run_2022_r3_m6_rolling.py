#!/usr/bin/env python3
"""Expanding 2022 R3 base-OOF followed by M6 conversion OOF experiment."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

P=dict(n_estimators=180,learning_rate=.05,num_leaves=31,min_child_samples=120,subsample=.8,colsample_bytree=.8,reg_lambda=8.,random_state=20260809,n_jobs=1,verbosity=-1)
BASE_PREFIX=('ret_','rv_','downside_','atr_','range_','drawdown_','recovery_','trend_','path_','volume_','jump_')
META_PREFIX=('market_','btc_minus_','transition_raw__')
def matrix(x,c):return x[c].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy('float32')
def main():
 p=argparse.ArgumentParser();p.add_argument('--labels',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 x=pd.concat([pd.read_parquet(z) for z in sorted((a.labels/'parts').glob('*.parquet'))],ignore_index=True);x=x[x.label_valid].copy();x['__ts__']=pd.to_datetime(x.__ts__,utc=True);x['month']=x.__ts__.dt.to_period('M').astype(str)
 cols=x.columns.tolist();base=[c for c in cols if c.startswith(BASE_PREFIX) and not c.startswith('trend_')][:35];meta=[c for c in cols if c.startswith(META_PREFIX)]+['base_p_upper','base_p_lower','base_p_timeout','base_margin','base_entropy'];meta=list(dict.fromkeys(meta))
 outputs=[];history=[];months=sorted(x.month.unique())
 for month in months[4:]:
  test=x[x.month.eq(month)].copy(); prior=x[x.__ts__<test.__ts__.min()].copy();base_parts=[]
  for side in ('long','short'):
   tr=prior[prior.side_name.eq(side)];te=test[test.side_name.eq(side)]
   b=lgb.LGBMClassifier(objective='multiclass',num_class=3,**P).fit(matrix(tr,base),tr.r3_class)
   prob=b.predict_proba(matrix(te,base));z=te[['candidate_id','__ts__','side_name','net_bps',*meta[:-5]]].copy();z[['base_p_upper','base_p_lower','base_p_timeout']]=prob;z['base_margin']=prob[:,2]-prob[:,0];z['base_entropy']=-(prob*np.log(np.maximum(prob,1e-12))).sum(1);base_parts.append(z)
  base_test=pd.concat(base_parts,ignore_index=True)
  # Meta may only use previous months' base-OOF rows, never base in-sample predictions.
  if history:
   hist=pd.concat(history,ignore_index=True);meta_parts=[]
   for side in ('long','short'):
    tr=hist[hist.side_name.eq(side)];te=base_test[base_test.side_name.eq(side)]
    m=lgb.LGBMClassifier(objective='binary',**P).fit(matrix(tr,meta),tr.net_bps.gt(50).astype(int));z=te.copy();z['m6_probability']=m.predict_proba(matrix(te,meta))[:,1];meta_parts.append(z)
   scored=pd.concat(meta_parts,ignore_index=True);outputs.append(scored)
  history.append(base_test)
 if not outputs:raise ValueError('no M6 folds')
 out=pd.concat(outputs,ignore_index=True);out['month']=pd.to_datetime(out.__ts__,utc=True).dt.to_period('M').astype(str);rows=[]
 for month,g in out.groupby('month'):
  for f in (.01,.05):
   z=g.nlargest(max(1,int(np.ceil(len(g)*f))),'m6_probability');rows.append({'month':month,'tail':f,'n':len(z),'net_bps':z.net_bps.mean(),'long_n':int((z.side_name=='long').sum()),'short_n':int((z.side_name=='short').sum())})
 for f in (.01,.05):
  z=out.nlargest(max(1,int(np.ceil(len(out)*f))),'m6_probability');rows.append({'month':'pooled','tail':f,'n':len(z),'net_bps':z.net_bps.mean(),'long_n':int((z.side_name=='long').sum()),'short_n':int((z.side_name=='short').sum())})
 out.to_parquet(a.out/'rolling_predictions.parquet',index=False);pd.DataFrame(rows).to_parquet(a.out/'rolling_metrics.parquet',index=False);(a.out/'manifest.json').write_text(json.dumps({'base_target':'R3 robust-clear b25/adverse/weak','conversion_target':'M6 net>50bps','base_features':base,'meta_features':meta,'folds':months[4:]},indent=2));print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
