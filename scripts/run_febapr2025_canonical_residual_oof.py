#!/usr/bin/env python3
"""Strict historical residual OOF with February base-passthrough warm-up."""
from __future__ import annotations
import json, hashlib, sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from extreme_price_movements.packb_static_point_feature_loader import iter_point_in_time_feature_batches

TOP=ROOT/'data_perp/artifacts/febapr2025_canonical_residual_top40_20260727_v1'
CONTRACT=ROOT/'data_perp/artifacts/packb_side_local_residual_oof_20260724_v1_31_8'
AE=ROOT/'data_perp/artifacts/packb_side_local_ae_20260724_v1'
STORE=ROOT/'data_perp/features/20260711_070000'
OUT=ROOT/'data_perp/artifacts/febapr2025_canonical_residual_oof_20260727_v1'
SIDE_FILTER=None; MONTH_FILTER=None
TARGET='__first_touch_capture_net__'; WEIGHT='__w__'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def econ(x,score):
 x=x.assign(_s=score).sort_values('_s',ascending=False,kind='stable'); k=max(1,int(np.ceil(len(x)*.1))); t=x.head(k)
 return {'rows':len(x),'top10_rows':k,'top10_execution_net_ev':float(t.execution_net_ev_12h.mean()),'top10_positive_fraction':float((t.execution_net_ev_12h>0).mean()),'score_native_target_spearman':float(x[['_s',TARGET]].corr(method='spearman').iloc[0,1])}
def point_features(point,contract):
 parts=[]; matched=0
 for batch in iter_point_in_time_feature_batches(point,feature_store_dir=STORE,feature_contract=contract,verify_frozen_schema=False,max_rows_per_batch=2048,max_columns_per_read=64):
  x=batch.features; x['__row__']=batch.ledger_row_positions; parts.append(x); matched+=int(batch.matched_exact_keys.sum())
 out=pd.concat(parts,ignore_index=True).sort_values('__row__',kind='stable'); rows=out.pop('__row__').to_numpy()
 if not np.array_equal(rows,np.arange(len(point))): raise RuntimeError('PIT feature row order failed')
 out=out.reset_index(drop=True)
 return out,{'exact_key_rows':matched,'exact_key_fraction':matched/max(len(point),1),'raw_per_feature_finite_fraction':{c:float(out[c].notna().mean()) for c in out.columns}}
def main():
 if OUT.exists(): raise FileExistsError(OUT)
 pop=pd.read_parquet(TOP/'population.parquet'); pop['__ts__']=pd.to_datetime(pop.__ts__,utc=True); pop['native_label_resolution_utc']=pd.to_datetime(pop.native_label_resolution_utc,utc=True)
 OUT.mkdir(parents=True); allout=[]; folds=[]; coverage={}
 for side in ((SIDE_FILTER,) if SIDE_FILTER else ('long','short')):
  frame=pop[pop.side_name.eq(side)].reset_index(drop=True); features=json.load(open(CONTRACT/side/'feature_contract.json'))['features']; hp=json.load(open(CONTRACT/side/'hpo_contract.json')); raw_contract=json.load(open(AE/side/'loader_evidence/frozen_feature_contract.json'))
  point=frame[['candidate_id','side_name','__ts__','__symbol__']].copy(); point['__top40_symbol__']=point['__symbol__']; point['__symbol__']=point.candidate_id.str.split('|',n=1).str[0]
  raw,raw_coverage=point_features(point.drop(columns='__top40_symbol__'),raw_contract)
  ts=frame.__ts__
  # Some raw contracts already contain calendar fields.  Canonical residual
  # anchors must overwrite those columns rather than create duplicate labels.
  raw=raw.reset_index(drop=True)
  raw['base_prediction']=frame.base_oof_score.to_numpy()
  raw['base_rank_pct_timestamp_side']=frame.base_rank_pct_timestamp_side.to_numpy()
  raw['base_rank_timestamp_side']=frame.base_rank_timestamp_side.to_numpy()
  raw['hour_sin']=np.sin(2*np.pi*ts.dt.hour.to_numpy()/24)
  raw['hour_cos']=np.cos(2*np.pi*ts.dt.hour.to_numpy()/24)
  matrix=raw.reindex(columns=features); finite={c:float(matrix[c].notna().mean()) for c in features}
  bad={c:v for c,v in finite.items() if v<.95}
  if raw_coverage['exact_key_fraction'] != 1.0 or bad: raise RuntimeError(f'{side} residual PIT coverage gate failed: {bad}')
  coverage[side]={'exact_key':raw_coverage['exact_key_fraction'],'residual_feature_finite_fraction':finite,'symbol_binding':'top40 display symbol audited; candidate_id slash spelling used because it matches immutable PIT payload'}
  sideout=[]
  for month in ((MONTH_FILTER,) if MONTH_FILTER else (2,3,4)):
   valid=frame.__ts__.dt.month.eq(month); v=frame.loc[valid].copy()
   if month==2:
    v['base_expected_ev']=np.nan; v['residual_expected_ev']=np.nan; v['residual_delta_ev']=np.nan; v['residual_fold']='february_base_passthrough_warmup'; v['residual_is_oof']=False; sideout.append(v); folds.append({'side':side,'fold':v.residual_fold.iloc[0],'train_rows':0,'test_rows':len(v),'purge':'warm-up metadata only; no residual EV claim'}); continue
   start=pd.Timestamp(f'2025-{month:02d}-01',tz='UTC'); train=frame.native_label_resolution_utc.lt(start); tr=frame.loc[train];
   iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(tr.base_oof_score,tr[TARGET],sample_weight=tr[WEIGHT]); base_tr=iso.predict(tr.base_oof_score); y=tr[TARGET].to_numpy()-base_tr
   model=lgb.LGBMRegressor(**hp['params'],n_estimators=int(hp['rounds']),random_state=month).fit(matrix.loc[train],y,sample_weight=tr[WEIGHT]); base_v=iso.predict(v.base_oof_score); delta=model.predict(matrix.loc[valid]); v['base_expected_ev']=base_v; v['residual_expected_ev']=base_v+float(hp['alpha'])*delta; v['residual_delta_ev']=delta; v['residual_fold']=f'month_2025_{month:02d}'; v['residual_is_oof']=True; sideout.append(v); folds.append({'side':side,'fold':v.residual_fold.iloc[0],'train_rows':len(tr),'test_rows':len(v),'train_resolution_max':str(tr.native_label_resolution_utc.max()),'purge':'native label resolution < validation start','features':features,'hpo_sha256':sha(CONTRACT/side/'hpo_contract.json')})
  allout.extend(sideout)
 out=pd.concat(allout,ignore_index=True).sort_values(['__ts__','candidate_id'],kind='stable'); out.to_parquet(OUT/'oof_predictions.parquet',index=False,compression='zstd')
 strict=out.loc[out.residual_is_oof].copy(); base=econ(strict,strict.base_expected_ev); resid=econ(strict,strict.residual_expected_ev); gate={'schema':'febapr2025_canonical_residual_oof_v1','status':'FEBRUARY_WARMUP_MARCH_APRIL_STRICT_RESIDUAL_OOF','rows':len(out),'strict_residual_oof_rows':len(strict),'identity_sha256':sha(TOP/'population.parquet'),'base_metrics_identical_rows':base,'residual_metrics_identical_rows':resid,'folds':folds,'feature_coverage':coverage,'february_boundary':'no December 2024 native label shards; explicit warm-up metadata only, no residual EV claim','native_target':TARGET,'native_weight':WEIGHT,'execution_economic_diagnostic':'execution_net_ev_12h'}
 (OUT/'coverage_economics_gate.json').write_text(json.dumps(gate,indent=2,default=str)); (OUT/'manifest.json').write_text(json.dumps({'schema':'febapr2025_canonical_residual_oof_v1','status':gate['status'],'oof_sha256':sha(OUT/'oof_predictions.parquet'),'gate_sha256':sha(OUT/'coverage_economics_gate.json')},indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser(); parser.add_argument('--output-dir',type=Path,default=OUT); parser.add_argument('--side',choices=('long','short')); parser.add_argument('--month',type=int,choices=(3,4)); args=parser.parse_args(); OUT=args.output_dir; SIDE_FILTER=args.side; MONTH_FILTER=args.month; main()
