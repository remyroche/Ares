#!/usr/bin/env python3
"""Training-only query-weight screen for the two advancing short targets."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.run_short_policy_conversion_funnel import PolicySpec,run
def u(x):
 y=pd.Timestamp(x); return y.tz_localize('UTC') if y.tzinfo is None else y.tz_convert('UTC')
WEIGHTS=('uniform','month_query','recency6m','recency9m','opportunity_spread','opportunity_tercile')
SPECS=tuple(PolicySpec(f'W_{target}_{weight}',f'{target} with {weight} training authority',kind,truncation=40,gain_family='linear',query_hours=1,weight_kind=weight) for target,kind in (('policy','policy_bps'),('activation','activation_grade')) for weight in WEIGHTS)
def main():
 p=argparse.ArgumentParser()
 for n in ('out','selection','policies','features','candidates','supportive-path'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();root=a.out.resolve();root.mkdir(parents=True);fields=json.loads(a.selection.read_text())['feature_sets']['90']
 for fold,start,end in [('mayjun','2024-05-01','2024-07-01'),('julaug','2024-07-01','2024-09-01'),('sep','2024-09-01','2024-10-01')]:run(out=root/fold,policies=a.policies.resolve(),features_path=a.features.resolve(),candidates_path=a.candidates.resolve(),supportive_path=a.supportive_path.resolve(),fields=fields,train_start=u('2023-10-01'),oos_start=u(start),oos_end=u(end),specs=SPECS)
 (root/'run_manifest.json').write_text(json.dumps({'schema':'strict_r3_short_policy_weight_screen_v1','status':'complete','selection_window':'pre-2024-10 only','targets':['policy_bps','activation_grade'],'weights':list(WEIGHTS),'fixed_ranker':{'query_hours':1,'truncation':40,'gain':'linear','objective':'lambdarank','norm':True}},indent=2)+'\n')
if __name__=='__main__':main()
