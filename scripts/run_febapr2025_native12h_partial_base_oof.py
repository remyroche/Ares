#!/usr/bin/env python3
"""March--April-only side-local base OOF on native decision+12h labels.

This is deliberately partial: February cannot be scored until the January
native 12h backfill resolves.  It reuses the frozen 31/8 features and HPO, and
does not read execution-EV labels.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_febapr2025_canonical_base_oof import _load_contracts,_materialize_features,_deterministic_cap,_lgbm_regressor,_identity_hash
NEW=ROOT/'data_perp/artifacts/febapr2025_native_first_touch_full_12h_labels_20260729_v1'
OLD=ROOT/'data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels'
PROMOTION=ROOT/'docs/pipeline_roadmap/20260724/r3/packb_side_fs_hpo_promotion_v1.json';AE=ROOT/'data_perp/artifacts/packb_side_local_ae_20260724_v1';STORE=ROOT/'data_perp/features/20260711_070000'
OUT=ROOT/'data_perp/artifacts/febapr2025_native12h_partial_marapr_base_oof_20260729_v1'
def main():
 if (OUT/'manifest.json').exists():raise FileExistsError(OUT)
 new=pd.concat([pd.read_parquet(p,columns=['candidate_id','side_name','__symbol__','__ts__','__decision_ts__','__native_12h_resolution_ts__','__native_12h_first_touch_target_soft__','__native_12h_first_touch_capture_net__']) for p in sorted((NEW/'shards').glob('*_labels.parquet'))],ignore_index=True)
 old=pd.concat([pd.read_parquet(OLD/f'train_global_{s}_5_2025_{m:02d}.parquet',columns=['candidate_id','__w__']) for m in (2,3,4) for s in ('long','short')],ignore_index=True)
 x=new.merge(old,on='candidate_id',how='left',validate='one_to_one');x['__ts__']=pd.to_datetime(x.__ts__,utc=True);x['__decision_ts__']=pd.to_datetime(x.__decision_ts__,utc=True);x['base_label_resolution_utc']=pd.to_datetime(x.__native_12h_resolution_ts__,utc=True);x['__feature_symbol__']=x.candidate_id.str.split('|',n=1).str[0]
 x=x.rename(columns={'__native_12h_first_touch_target_soft__':'target_12h','__native_12h_first_touch_capture_net__':'capture_net_12h','__w__':'weight'});contracts=_load_contracts(PROMOTION,AE);OUT.mkdir(parents=True,exist_ok=True);preds=[];folds=[]
 for side_no,side in enumerate(('long','short')):
  sx=x[x.side_name.eq(side)].copy()
  for fold_no,(start,end) in enumerate(((pd.Timestamp('2025-03-01',tz='UTC'),pd.Timestamp('2025-04-01',tz='UTC')),(pd.Timestamp('2025-04-01',tz='UTC'),pd.Timestamp('2025-05-01',tz='UTC'))),1):
   train=_deterministic_cap(sx[sx.base_label_resolution_utc.lt(start)].copy(),100000).reset_index(drop=True);valid=sx[sx.__ts__.ge(start)&sx.__ts__.lt(end)].reset_index(drop=True)
   if train.empty or valid.empty:raise RuntimeError(f'{side} {start}: empty legal train/valid')
   folder=OUT/side/f'month_{start:%Y_%m}'
   existing=folder/'oof_predictions.parquet'
   if existing.exists():
    out=pd.read_parquet(existing);preds.append(out);folds.append({'side':side,'fold':folder.name,'resumed_existing':True,'train_rows':None,'valid_rows':len(out)});continue
   folder.mkdir(parents=True,exist_ok=True)
   tx,tc=_materialize_features(train,contracts[side],STORE,folder/'train_features.parquet');vx,vc=_materialize_features(valid,contracts[side],STORE,folder/'validation_features.parquet')
   if tc['exact_key_fraction']!=1 or vc['exact_key_fraction']!=1:raise RuntimeError('PIT feature coverage failure')
   model=_lgbm_regressor(contracts[side]['params'],seed=12900+side_no*100+fold_no);model.fit(tx,train.target_12h,sample_weight=train.weight)
   out=valid[['candidate_id','side_name','__symbol__','__ts__','__decision_ts__','base_label_resolution_utc','target_12h','weight','capture_net_12h']].copy();out['fold_id']=folder.name;out['base_oof_score']=model.predict(vx).astype('float64');out.to_parquet(folder/'oof_predictions.parquet',index=False,compression='zstd');preds.append(out);folds.append({'side':side,'fold':folder.name,'train_rows':len(train),'valid_rows':len(valid),'train_resolution_max':str(train.base_label_resolution_utc.max()),'features':list(contracts[side]['features']),'hpo_trial':contracts[side]['trial_id'],'feature_coverage':{'train':tc,'validation':vc}})
 result=pd.concat(preds,ignore_index=True).sort_values(['__ts__','candidate_id'],kind='stable');result.to_parquet(OUT/'oof_predictions.parquet',index=False,compression='zstd');pd.DataFrame(folds).to_parquet(OUT/'fold_provenance.parquet',index=False,compression='zstd')
 (OUT/'manifest.json').write_text(json.dumps({'schema':'native12h_partial_marapr_base_oof_v1','status':'COMPLETE_PARTIAL_MARCH_APRIL_ONLY','rows':len(result),'strict_native_only':True,'not_execution_ev_evaluated':True,'target':'decision+12h native first-touch soft','purge':'base_label_resolution_utc < fold_start','missing_february_reason':'January native 12h backfill is in progress; February fold is intentionally withheld','contracts':{s:{'features':list(contracts[s]['features']),'hpo_trial':contracts[s]['trial_id']} for s in contracts}},indent=2)+'\n')
if __name__=='__main__':main()
