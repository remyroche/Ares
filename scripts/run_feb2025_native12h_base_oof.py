#!/usr/bin/env python3
"""Legal February side-local base OOF using resolved January native 12h labels."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_febapr2025_canonical_base_oof import _load_contracts,_materialize_features,_deterministic_cap,_lgbm_regressor
JAN=ROOT/'data_perp/artifacts/january2025_native_first_touch_full_12h_labels_20260729_v1';FEB=ROOT/'data_perp/artifacts/febapr2025_native_first_touch_full_12h_labels_20260729_v1';OLD=ROOT/'data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels';PROMOTION=ROOT/'docs/pipeline_roadmap/20260724/r3/packb_side_fs_hpo_promotion_v1.json';AE=ROOT/'data_perp/artifacts/packb_side_local_ae_20260724_v1';STORE=ROOT/'data_perp/features/20260711_070000';OUT=ROOT/'data_perp/artifacts/feb2025_native12h_base_oof_20260729_v1'
def load(root):return pd.concat([pd.read_parquet(p,columns=['candidate_id','side_name','__symbol__','__ts__','__decision_ts__','__native_12h_resolution_ts__','__native_12h_first_touch_target_soft__','__native_12h_first_touch_capture_net__']) for p in sorted((root/'shards').glob('*_labels.parquet'))],ignore_index=True)
def main():
 if (OUT/'manifest.json').exists():raise FileExistsError(OUT)
 jan=load(JAN);feb=load(FEB);feb=feb[pd.to_datetime(feb.__ts__,utc=True).dt.month.eq(2)].copy();old=pd.concat([pd.read_parquet(OLD/f'train_global_{s}_5_2025_{m:02d}.parquet',columns=['candidate_id','__w__']) for m in (1,2) for s in ('long','short')],ignore_index=True);contracts=_load_contracts(PROMOTION,AE);OUT.mkdir(parents=True,exist_ok=True);parts=[]
 for side_no,side in enumerate(('long','short')):
  def prep(x):
   x=x[x.side_name.eq(side)].merge(old,on='candidate_id',how='left',validate='one_to_one');x['__ts__']=pd.to_datetime(x.__ts__,utc=True);x['base_label_resolution_utc']=pd.to_datetime(x.__native_12h_resolution_ts__,utc=True);x['__feature_symbol__']=x.candidate_id.str.split('|',n=1).str[0];return x.rename(columns={'__native_12h_first_touch_target_soft__':'target_12h','__native_12h_first_touch_capture_net__':'capture_net_12h','__w__':'weight'})
  train_source=prep(jan);train=_deterministic_cap(train_source.loc[train_source.base_label_resolution_utc.lt(pd.Timestamp('2025-02-01',tz='UTC'))],100000).reset_index(drop=True);valid=prep(feb).reset_index(drop=True);folder=OUT/side/'month_2025_02';existing=folder/'oof_predictions.parquet'
  if existing.exists():parts.append(pd.read_parquet(existing));continue
  folder.mkdir(parents=True,exist_ok=True);tx,tc=_materialize_features(train,contracts[side],STORE,folder/'train_features.parquet');vx,vc=_materialize_features(valid,contracts[side],STORE,folder/'validation_features.parquet');
  if tc['exact_key_fraction']!=1 or vc['exact_key_fraction']!=1:raise RuntimeError('PIT coverage failure')
  model=_lgbm_regressor(contracts[side]['params'],seed=12900+side_no*100);model.fit(tx,train.target_12h,sample_weight=train.weight);out=valid[['candidate_id','side_name','__symbol__','__ts__','__decision_ts__','base_label_resolution_utc','target_12h','weight','capture_net_12h']].copy();out['fold_id']='month_2025_02';out['base_oof_score']=model.predict(vx);out.to_parquet(folder/'oof_predictions.parquet',index=False,compression='zstd');parts.append(out)
 result=pd.concat(parts,ignore_index=True);result.to_parquet(OUT/'oof_predictions.parquet',index=False,compression='zstd');(OUT/'manifest.json').write_text(json.dumps({'status':'COMPLETE_NATIVE_12H_FEBRUARY_OOF','rows':len(result),'native_only':True,'purge':'January decision+12h resolution < 2025-02-01','execution_ev_joined':False},indent=2)+'\n')
if __name__=='__main__':main()
