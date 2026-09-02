#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from pathlib import Path
import pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]
SHARDS=ROOT/'data_perp/artifacts/febapr2025_residual_shards_20260727_v1'
TOP=ROOT/'data_perp/artifacts/febapr2025_canonical_residual_top40_20260727_v1/population.parquet'
OUT=ROOT/'data_perp/artifacts/febapr2025_canonical_residual_oof_20260727_v1'
NAMES=('long_2025_03_v3','long_2025_04_v2','short_2025_03_v2','short_2025_04_v2')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(x,col):
 x=x.sort_values(col,ascending=False,kind='stable');k=max(1,int(np.ceil(len(x)*.1)));t=x.head(k)
 return {'rows':len(x),'top10_rows':k,'top10_execution_net_ev':float(t.execution_net_ev_12h.mean()),'top10_positive_fraction':float((t.execution_net_ev_12h>0).mean()),'score_native_target_spearman':float(x[[col,'__first_touch_capture_net__']].corr(method='spearman').iloc[0,1])}
def main():
 if OUT.exists(): raise FileExistsError(OUT)
 parts=[pd.read_parquet(SHARDS/n/'oof_predictions.parquet') for n in NAMES]; strict=pd.concat(parts,ignore_index=True); top=pd.read_parquet(TOP); top['__ts__']=pd.to_datetime(top.__ts__,utc=True); warm=top.loc[top.__ts__.dt.month.eq(2)].copy();warm['base_expected_ev']=np.nan;warm['residual_expected_ev']=np.nan;warm['residual_delta_ev']=np.nan;warm['residual_fold']='february_base_passthrough_warmup';warm['residual_is_oof']=False
 out=pd.concat([warm,strict],ignore_index=True).sort_values(['__ts__','candidate_id'],kind='stable').reset_index(drop=True)
 if len(out)!=len(top) or out.candidate_id.duplicated().any() or len(strict)!=140682 or not strict.residual_is_oof.all(): raise RuntimeError('identity/strict OOF gate failed')
 manifests=[json.load(open(SHARDS/n/'manifest.json')) for n in NAMES];gates=[json.load(open(SHARDS/n/'coverage_economics_gate.json')) for n in NAMES]
 for g in gates:
  for f in g.get('folds',[]):
   if 'train_resolution_max' in f and not pd.Timestamp(f['train_resolution_max'])<pd.Timestamp('2025-'+f['fold'][-2:]+'-01',tz='UTC'): raise RuntimeError('purge gate failed')
 OUT.mkdir(parents=True);out.to_parquet(OUT/'oof_predictions.parquet',index=False,compression='zstd')
 gate={'schema':'febapr2025_canonical_residual_oof_aggregate_v1','status':'FEBRUARY_WARMUP_MARCH_APRIL_STRICT_RESIDUAL_OOF','rows':len(out),'warmup_rows':len(warm),'strict_oof_rows':len(strict),'base_metrics_identical_rows':metric(strict,'base_expected_ev'),'residual_metrics_identical_rows':metric(strict,'residual_expected_ev'),'per_side':{s:{'base':metric(strict[strict.side_name.eq(s)],'base_expected_ev'),'residual':metric(strict[strict.side_name.eq(s)],'residual_expected_ev')} for s in ('long','short')},'per_month':{m:{'base':metric(strict[strict.__ts__.dt.month.eq(m)],'base_expected_ev'),'residual':metric(strict[strict.__ts__.dt.month.eq(m)],'residual_expected_ev')} for m in (3,4)},'shards':[{'name':n,'manifest_sha256':sha(SHARDS/n/'manifest.json'),'gate_sha256':sha(SHARDS/n/'coverage_economics_gate.json')} for n in NAMES]}
 (OUT/'coverage_economics_gate.json').write_text(json.dumps(gate,indent=2));(OUT/'manifest.json').write_text(json.dumps({'schema':'febapr2025_canonical_residual_oof_v1','status':gate['status'],'oof_sha256':sha(OUT/'oof_predictions.parquet'),'gate_sha256':sha(OUT/'coverage_economics_gate.json')},indent=2))
if __name__=='__main__':main()
