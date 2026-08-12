#!/usr/bin/env python3
"""Build the compatible 2022--2026 R3/TP6-SL4 correctness-regime input.

This uses the Stage-I selector contract rather than concatenating it with the
shorter shared-regime ledger.  The same-side base value map is recomputed
prequentially from strict OOF R3 probabilities and prior-resolved TP6/SL4 net
labels.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from extreme_price_movements.prequential_r3_value_map import PrequentialR3ValueMapConfig,prequential_same_side_r3_value_map

SAMPLE=ROOT/'data_perp/artifacts/stage_i_selector_sample_20260803_v5'
BASE=ROOT/'data_perp/artifacts/stage_i_base_selection_20260803_v5'
OUT=ROOT/'data_perp/artifacts/correctness_leaf_regime_two_year_input_20260803_v1'

def run(out:Path=OUT):
 out.mkdir(parents=True,exist_ok=True)
 ledger=pd.read_parquet(SAMPLE/'selector_ledger.parquet')
 base=pd.concat([pd.read_parquet(BASE/side/'selector_base_oof.parquet') for side in ('long','short')],ignore_index=True)
 keep=['candidate_id','r3_p_adverse','r3_p_weak','r3_p_clear','r3_opportunity_score']
 d=ledger.merge(base[keep],on='candidate_id',how='inner',validate='one_to_one')
 d['__ts__']=pd.to_datetime(d['__ts__'],utc=True);d['label_available_ts']=pd.to_datetime(d.label_available_ts,utc=True)
 p=d[['r3_p_adverse','r3_p_weak','r3_p_clear']].to_numpy(float);good=np.isfinite(p).all(1)&(p>=0).all(1)&np.isclose(p.sum(1),1.,atol=1e-5)
 d=d[good].copy();d[['r3_p_adverse','r3_p_weak','r3_p_clear']]=p[good]
 mapped=[]
 for side,x in d.groupby('side_name',observed=True,sort=False):
  value,audit,_=prequential_same_side_r3_value_map(exact_net_bps=x.exact_net_bps,decision_timestamps=x.decision_ts,label_available_timestamps=x.label_available_ts,side=side,p_adverse=x.r3_p_adverse,p_weak=x.r3_p_weak,p_clear=x.r3_p_clear,config=PrequentialR3ValueMapConfig(side=side,mapping_mode='monotone_pava'))
  z=x[['candidate_id']].copy();z['prequential_base_expected_net_bps']=value
  z['base_value_map_prior_global_support_log1p']=np.log1p(audit.prior_resolved_global_support.to_numpy(float))
  z['base_value_map_prior_bin_support_log1p']=np.log1p(audit.prior_resolved_bin_support.to_numpy(float))
  z['base_value_map_neutral_fallback']=audit.value_map_fallback.astype(str).str.contains('neutral').astype('float32')
  mapped.append(z)
 d=d.merge(pd.concat(mapped),on='candidate_id',how='inner',validate='one_to_one')
 p=d[['r3_p_adverse','r3_p_weak','r3_p_clear']].to_numpy(float)
 d['base_entropy']=-(p*np.log(np.maximum(p,1e-12))).sum(1);q=np.sort(p,axis=1);d['base_top2_margin']=q[:,-1]-q[:,-2];d['base_max_probability']=q[:,-1];d['side_is_long']=d.side_name.eq('long').astype('float32')
 d['gross_bps']=d.exact_gross_bps.astype('float32');d['net_bps']=d.exact_net_bps.astype('float32');d['era']=d.__ts__.dt.strftime('%Y-%m')
 features=pd.read_parquet(SAMPLE/'selector_features.parquet');d=d.merge(features,on=['candidate_id','__ts__','__symbol__'],how='inner',validate='one_to_one')
 core={'candidate_id','__ts__','__symbol__','decision_ts','label_available_ts','side_name','era','gross_bps','net_bps','prequential_base_expected_net_bps'}
 # These are realised path outcomes or downstream policy labels.  They may
 # exist in the selector ledger for auditing, but are never decision-time
 # inputs to a correctness discovery or residual meta model.
 label_or_policy_fields={
  'exact_net_bps','exact_gross_bps','label_valid','t2_tp6_sl4_event',
  'robust_clear_event_b25','robust_clear_soft_b25_t50','r3_class',
  'r3_metric_target','source_month','population_segment','selector_month',
  'selector_economic_bin',
 }
 # Retain base probability/economics/uncertainty and prequential support as
 # first-class meta inputs alongside the full selector feature universe.
 numeric=[c for c in d.columns if c not in core|label_or_policy_fields and pd.api.types.is_numeric_dtype(d[c])]
 coverage=1-d[numeric].replace([np.inf,-np.inf],np.nan).isna().mean();variation=d[numeric].replace([np.inf,-np.inf],np.nan).std()>1e-12
 usable=coverage.ge(.90)&variation
 audit=pd.DataFrame({'feature':numeric,'coverage':coverage,'nonconstant':variation,'usable_90pct_nonconstant':usable}).reset_index(drop=True);audit.to_parquet(out/'feature_availability.parquet',index=False)
 final=[c for c in d.columns if c in core or c in audit.loc[audit.usable_90pct_nonconstant,'feature'].tolist()]
 d[final].sort_values(['__ts__','candidate_id']).to_parquet(out/'input.parquet',index=False,compression='zstd')
 manifest={'status':'COMPLETED','source_contract':'stage_i_selector_sample_v5 + stage_i_base_selection_v5','geometry':'R3 TP6/SL4/H12, 100 bps fixed cost','rows':len(d),'time_range':[str(d.__ts__.min()),str(d.__ts__.max())],'base_oof_match_after_validity':len(d)/len(ledger),'usable_meta_features':int(usable.sum()),'value_map':'same-side strict-OOF, prior label_available_ts < decision_ts, fixed bins with monotone PAVA'}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args();run(args.out)
