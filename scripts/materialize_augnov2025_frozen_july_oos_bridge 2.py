#!/usr/bin/env python3
"""Resumable Aug--Nov 2025 common-30 OOS score bridge.

Exact execution-label ledgers provide the OOS candidate identity/economics.
They are never substituted for native first-touch training targets: base and
residual models fit only on native labels resolved strictly before 2025-08-01.
All model rows are 1h; 1m remains nested in the execution-label replay.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pickle, tempfile
from pathlib import Path
from typing import Any, Mapping
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT=Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.run_febapr2025_canonical_base_oof import IDENTITY,SIDES,TARGET,WEIGHT,ECONOMIC,_deterministic_cap,_load_contracts,_materialize_features,_sha256
from scripts.run_packb_pre_march_side_fs_hpo import _lgbm_regressor
from scripts.materialize_july2025_common30_final_base_residual_oof_bridge import _native,_base_training_features_with_frozen_cache
from scripts.run_mayjun2025_canonical_residual_continuation import _feature_matrix,_sha as rsha

LABEL_DIR=ROOT/'data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels'
PROMOTION=ROOT/'docs/pipeline_roadmap/20260724/r3/packb_side_fs_hpo_promotion_v1.json';AE=ROOT/'data_perp/artifacts/packb_side_local_ae_20260724_v1';STORE=ROOT/'data_perp/features/20260711_070000'
MAYJUN=ROOT/'data_perp/artifacts/mayjun2025_canonical_base_continuation_20260730_v1';JULY=ROOT/'data_perp/artifacts/july2025_common30_final_base_residual_oof_bridge_20260730_v1';HIST=ROOT/'data_perp/artifacts/febapr2025_canonical_residual_top40_20260727_v1/population.parquet';RCON=ROOT/'data_perp/artifacts/packb_side_local_residual_oof_20260724_v1_31_8'
EXEC={8:ROOT/'data_perp/artifacts/augoct2025_execution_ev_common30_labels_20260727_v1/labels.parquet',9:ROOT/'data_perp/artifacts/augoct2025_execution_ev_common30_labels_20260727_v1/labels.parquet',10:ROOT/'data_perp/artifacts/augoct2025_execution_ev_common30_labels_20260727_v1/labels.parquet',11:ROOT/'data_perp/artifacts/nov2025_execution_ev_common30_labels_20260727_v1/labels.parquet'}
OUT=ROOT/'data_perp/artifacts/augnov2025_common30_frozen_july_base_residual_oos_bridge_20260730_v1'; CUT=pd.Timestamp('2025-08-01',tz='UTC'); RTARGET='__first_touch_capture_net__'
class BridgeError(RuntimeError):pass
def sha(p):return _sha256(Path(p))
def dump(p,x):
 q=Path(p).with_name('.'+Path(p).name+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)
def load_pickle(p):
 with Path(p).open('rb') as f:return pickle.load(f)
def save_pickle(p,x):
 q=Path(p).with_name('.'+Path(p).name+'.partial')
 with q.open('wb') as f:pickle.dump(x,f)
 os.replace(q,p)
def candidates(month:int)->pd.DataFrame:
 x=pd.read_parquet(EXEC[month]);x['__ts__']=pd.to_datetime(x.__ts__,utc=True);x['execution_label_end_utc']=pd.to_datetime(x.execution_label_end_utc,utc=True);x['execution_label_available_at']=pd.to_datetime(x.execution_label_available_at,utc=True)
 x=x.loc[x.__ts__.dt.month.eq(month)].copy()
 expected_rows={8:44_640,9:43_200,10:44_640,11:43_200}[month]
 if len(x)!=expected_rows or x.candidate_id.duplicated().any() or not x.execution_label_end_utc.gt(x.__ts__).all():raise BridgeError(f'month {month} invalid exact execution candidate ledger')
 if not (x.__ts__.astype('int64')%pd.Timedelta(hours=1).value==0).all():raise BridgeError('non-hourly candidate')
 return x.sort_values(['__ts__','candidate_id'],kind='stable').reset_index(drop=True)
def required_file(out,side,name):return Path(out)/'models'/side/name
def fit_base(out:Path,side:str):
 native=_native(LABEL_DIR);contracts=_load_contracts(PROMOTION,AE);train=_deterministic_cap(native.loc[native.side_name.eq(side)&native.base_label_resolution_utc.lt(CUT)].copy(),100_000).reset_index(drop=True)
 if not train.base_label_resolution_utc.lt(CUT).all():raise BridgeError('future native base label')
 d=required_file(out,side,'base.pkl');audit=required_file(out,side,'base_fit.json');d.parent.mkdir(parents=True,exist_ok=True)
 if d.exists():return
 x,cov=_base_training_features_with_frozen_cache(train,side=side,route=contracts[side],feature_store=STORE,output=required_file(out,side,'base_train_features.parquet'),mayjun_base_dir=MAYJUN)
 model=_lgbm_regressor(contracts[side]['params'],seed=9700+(0 if side=='long' else 1));model.fit(x.loc[:,list(contracts[side]['features'])],train[TARGET],sample_weight=train[WEIGHT]);save_pickle(d,model);dump(audit,{'side':side,'train_rows':len(train),'train_label_resolution_max_utc':train.base_label_resolution_utc.max(),'label_cutoff':'native decision+24h < 2025-08-01T00:00:00Z','features':list(contracts[side]['features']),'feature_coverage':cov,'hpo_trial_id':contracts[side]['trial_id'],'no_aug_nov_native_labels_read':True,'model_sha256':sha(d)})
def score_base(out:Path,side:str,month:int):
 path=Path(out)/side/f'month_2025_{month:02d}'/'base_oos_predictions.parquet'
 if path.exists():return
 model=load_pickle(required_file(out,side,'base.pkl'));contracts=_load_contracts(PROMOTION,AE);v=candidates(month);v=v.loc[v.side_name.eq(side)].reset_index(drop=True);v['__feature_symbol__']=v.candidate_id.str.split('|',n=1).str[0];path.parent.mkdir(parents=True,exist_ok=True);x,cov=_materialize_features(v,contracts[side],STORE,path.parent/'base_oos_features.parquet')
 if cov['exact_key_fraction']!=1:raise BridgeError('base OOS PIT coverage fail')
 o=v.copy();o['base_oos_score']=model.predict(x.loc[:,list(contracts[side]['features'])]);o['score_base_alpha']=o.base_oos_score;o['base_rank_timestamp_side']=o.groupby('__ts__').base_oos_score.rank(method='first',ascending=False).astype(int);o['base_group_rows']=o.groupby('__ts__').candidate_id.transform('size').astype(int);o['base_rank_pct_timestamp_side']=o.base_rank_timestamp_side/o.base_group_rows;o['base_score_fit_cutoff_utc']=CUT;o['base_is_oos']=True;o.to_parquet(path,index=False,compression='zstd');dump(path.with_suffix('.json'),{'side':side,'month':month,'rows':len(o),'feature_coverage':cov,'candidate_identity_sha256':hashlib.sha256(pd.util.hash_pandas_object(o[list(IDENTITY)].astype(str),index=False).values.tobytes()).hexdigest(),'model_sha256':sha(required_file(out,side,'base.pkl'))})
def fit_residual(out:Path,side:str):
 p=required_file(out,side,'residual.pkl');a=required_file(out,side,'residual_fit.json');p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():return
 hist=pd.read_parquet(HIST);hist['__ts__']=pd.to_datetime(hist.__ts__,utc=True);hist['native_label_resolution_utc']=pd.to_datetime(hist.native_label_resolution_utc,utc=True)
 mj=pd.read_parquet(MAYJUN/'oof_predictions.parquet');mj['__ts__']=pd.to_datetime(mj.__ts__,utc=True);mj['native_label_resolution_utc']=pd.to_datetime(mj.base_label_resolution_utc,utc=True)
 jy=pd.read_parquet(JULY/'oof_predictions.parquet');jy['__ts__']=pd.to_datetime(jy.__ts__,utc=True);jy['native_label_resolution_utc']=pd.to_datetime(jy.base_label_resolution_utc,utc=True)
 frame=pd.concat([hist,mj,jy],ignore_index=True,sort=False);frame=frame.loc[frame.side_name.eq(side)&frame.native_label_resolution_utc.lt(CUT)].reset_index(drop=True)
 if frame.candidate_id.duplicated().any() or not frame.native_label_resolution_utc.lt(CUT).all():raise BridgeError('residual train future/duplicate')
 x,cov=_feature_matrix(frame,side,STORE);hp=json.loads((RCON/side/'hpo_contract.json').read_text());iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(frame.base_oof_score,frame[RTARGET],sample_weight=frame[WEIGHT]);base=iso.predict(frame.base_oof_score);m=lgb.LGBMRegressor(**hp['params'],n_estimators=int(hp['rounds']),random_state=9800+(0 if side=='long' else 1)).fit(x,frame[RTARGET].to_numpy()-base,sample_weight=frame[WEIGHT]);save_pickle(p,{'iso':iso,'model':m,'alpha':float(hp['alpha'])});dump(a,{'side':side,'train_rows':len(frame),'train_label_resolution_max_utc':frame.native_label_resolution_utc.max(),'label_cutoff':'native label resolution < 2025-08-01T00:00:00Z','feature_coverage':cov,'hpo_sha256':rsha(RCON/side/'hpo_contract.json'),'model_sha256':sha(p),'no_aug_nov_native_labels_read':True})
def score_residual(out:Path,side:str,month:int):
 d=Path(out)/side/f'month_2025_{month:02d}';p=d/'oos_predictions.parquet'
 if p.exists():return
 base=pd.read_parquet(d/'base_oos_predictions.parquet');base['base_oof_score']=base.base_oos_score;pack=load_pickle(required_file(out,side,'residual.pkl'));x,cov=_feature_matrix(base,side,STORE);o=base.copy();o['base_expected_ev']=pack['iso'].predict(o.base_oos_score);o['residual_delta_ev']=pack['model'].predict(x);o['residual_expected_ev']=o.base_expected_ev+pack['alpha']*o.residual_delta_ev;o['score_residual_expected_ev']=o.residual_expected_ev;o['residual_score_fit_cutoff_utc']=CUT;o['residual_is_oos']=True;o.to_parquet(p,index=False,compression='zstd');dump(d/'residual_oos.json',{'side':side,'month':month,'rows':len(o),'feature_coverage':cov,'model_sha256':sha(required_file(out,side,'residual.pkl'))})
def seal(out:Path):
 files=sorted(Path(out).glob('*/month_2025_*/oos_predictions.parquet'))
 if len(files)!=8:raise BridgeError('all four month x both side residual score files required')
 o=pd.concat([pd.read_parquet(p) for p in files],ignore_index=True).sort_values(['__ts__','candidate_id'],kind='stable').reset_index(drop=True)
 if len(o)!=175_680 or o.candidate_id.duplicated().any() or not o.residual_is_oos.all():raise BridgeError('final candidate coverage failure')
 o.to_parquet(Path(out)/'oos_predictions.parquet',index=False,compression='zstd');manifest={'schema':'augnov2025_common30_frozen_july_base_residual_oos_bridge_v1','status':'SEALED_COMMON30_FROZEN_JULY_OOS_SCORE_BRIDGE_NON_PROMOTION','promotion_eligible':False,'scope':'Aug-Nov exact common30 candidate population; not identical to wider final v3 population','decision_cadence':'1h','exact_replay_bar_cadence':'1m_labels_only','score_fit':'frozen side-local base/residual contracts fit only on native labels resolved strictly before 2025-08-01; no Aug-Nov native target labels read','no_2026_outcomes':True,'rows':len(o),'by_month':o.__ts__.dt.strftime('%Y-%m').value_counts().sort_index().to_dict(),'by_side':o.side_name.value_counts().to_dict(),'inputs_sha256':{str(x):sha(x) for x in [*EXEC.values(),JULY/'manifest.json',HIST,MAYJUN/'oof_predictions.parquet']},'outputs_sha256':{'oos_predictions.parquet':sha(Path(out)/'oos_predictions.parquet')}};dump(Path(out)/'bridge_contract.json',manifest);dump(Path(out)/'manifest.json',manifest);(Path(out)/'manifest.sha256').write_text(f"{sha(Path(out)/'manifest.json')}  manifest.json\n")
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUT);ap.add_argument('--stage',choices=('fit_base','score_base','fit_residual','score_residual','seal'),required=True);ap.add_argument('--side',choices=SIDES);ap.add_argument('--month',type=int,choices=(8,9,10,11));z=ap.parse_args();o=z.output;o.mkdir(parents=True,exist_ok=True)
 if z.stage in ('fit_base','fit_residual'):
  if not z.side: ap.error('--side required')
  (fit_base if z.stage=='fit_base' else fit_residual)(o,z.side)
 elif z.stage in ('score_base','score_residual'):
  if not z.side or not z.month: ap.error('--side and --month required')
  (score_base if z.stage=='score_base' else score_residual)(o,z.side,z.month)
 else:seal(o)
if __name__=='__main__':main()
