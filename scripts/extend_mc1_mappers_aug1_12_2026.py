#!/usr/bin/env python3
"""Causally extend the frozen MC mapper finalists through 2026-08-12."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import scripts.run_strict_r3_six_mapper_families as m

LEDGER=ROOT/'data_perp/artifacts/strict_r3_lockstep_history_long_2024apr_jul2026_strictfull_prior28_optimizedpolicy_20260812_v1/walkforward_scored_label_ledger.parquet'
MAPS=ROOT/'data_perp/artifacts/strict_r3_recovery_detection_funnel_long_2024may_2026_20260813_v4/multiwindow_ev_maps.parquet'
AUG=ROOT/'data_perp/artifacts/strict_r3_homogeneous28_a5_forward_long_aug1_12_append_only_complete_20260813_v1/scored_label_ledger.parquet'
AUGMAP=ROOT/'data_perp/artifacts/strict_r3_multiwindow_ev_map_aug1_12_2026_20260813_v3/mapped_blocks/producer_20260801T000000Z.parquet'
OUT=ROOT/'data_perp/artifacts/strict_r3_mc1_aug1_12_extension_20260813_v2'
ARMS=('B0_R21','B1_R28','MC1_d2','MC1_d3','MC3_d2')

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def prep_aug():
 d=pd.read_parquet(AUG);d['__decision_ts__']=pd.to_datetime(d.__decision_ts__,utc=True);d['policy_label_available_ts']=pd.to_datetime(d.policy_label_available_ts,utc=True)
 d=d.sort_values(['__decision_ts__','final_score','candidate_id'],ascending=[True,False,True],kind='stable');d['rank_n']=d.groupby('__decision_ts__').cumcount()+1;d['group_n']=d.groupby('__decision_ts__').candidate_id.transform('size')
 d['rank_pct']=(d.rank_n-.5)/d.group_n;d['score_band']=np.minimum(9,(d.rank_pct*10).astype(int));d['day']=d.__decision_ts__.dt.normalize()
 mp=pd.read_parquet(AUGMAP,columns=['candidate_id','robust_21d__expected_net_bps','robust_28d__expected_net_bps']).rename(columns={'robust_21d__expected_net_bps':'m21','robust_28d__expected_net_bps':'m28'})
 return d.merge(mp,on='candidate_id',how='left',validate='one_to_one')

def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir(parents=True)
 hist=m.prepare(LEDGER,MAPS);aug=prep_aug();d=pd.concat([hist,aug],ignore_index=True,sort=False);opp=m.opportunity(d)
 pieces=[]
 for _,g in d.groupby('day',sort=True):
  top=g[g.rank_n.le(50)];rest=g.drop(top.index);pieces.append(pd.concat([top,rest.sample(min(250,len(rest)),random_state=1729)]))
 m._HISTORY_SOURCE=pd.concat(pieces,ignore_index=True).sort_values('policy_label_available_ts',kind='stable')
 selected=[s for s in m.specs() if s['id'] in ARMS]
 predictions=[]
 for day in sorted(aug.day.unique()):
  day=pd.Timestamp(day)
  for spec in selected:predictions.append(m.map_day(d,day,spec,opp).assign(arm=spec['id']))
 out=pd.concat(predictions,ignore_index=True);out.to_parquet(OUT/'causal_predictions.parquet',index=False)
 manifest={'status':'complete','period':'2026-08-01 through 2026-08-12','arms':ARMS,'history_ledger':str(LEDGER.relative_to(ROOT)),'history_sha256':sha(LEDGER),'aug_ledger':str(AUG.relative_to(ROOT)),'aug_sha256':sha(AUGMAP),'causality':'monthly model fitted at Aug 1 from labels available before Aug 1; daily recent-global shift updates from all labels available before the UTC day, including already-resolved August labels','ranking':'frozen final_score','admission_floor_bps':50}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'event':'complete','rows':len(out),'out':str(OUT)}))
if __name__=='__main__':main()
