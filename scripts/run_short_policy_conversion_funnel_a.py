#!/usr/bin/env python3
"""Sequential Funnel-A formulation screen on pre-October short development data."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.run_short_policy_conversion_funnel import PolicySpec, run

def u(x:str)->pd.Timestamp:
    y=pd.Timestamp(x); return y.tz_localize('UTC') if y.tzinfo is None else y.tz_convert('UTC')

SPECS=(
 PolicySpec('A0_control','P1 control','policy_bps',truncation=32,gain_family='linear'),
 PolicySpec('A1_coarse','100bps relevance','policy_coarse',truncation=32,gain_family='linear'),
 PolicySpec('A2_deadzone','wide dead-zone relevance','policy_deadzone',truncation=32,gain_family='linear'),
 PolicySpec('A3_fine_tail','fine positive-tail relevance','policy_fine_tail',truncation=32,gain_family='linear'),
 PolicySpec('A4_train_quantile','training-derived absolute policy quantiles','train_quantile',truncation=32,gain_family='linear'),
 PolicySpec('A5_hybrid25','25% absolute / 75% relative','hybrid_rank',truncation=32,gain_family='linear',absolute_weight=.25),
 PolicySpec('A6_hybrid50','50% absolute / 50% relative','hybrid_rank',truncation=32,gain_family='linear',absolute_weight=.50),
 PolicySpec('A7_hybrid75','75% absolute / 25% relative','hybrid_rank',truncation=32,gain_family='linear',absolute_weight=.75),
)
def main()->None:
 p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--policies',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--candidates',type=Path,required=True);p.add_argument('--only-t4',action='store_true');a=p.parse_args(); root=a.out.resolve(); root.mkdir()
 fields=json.loads(a.selection.read_text())['feature_sets']['90']
 specs=(SPECS[4],) if a.only_t4 else SPECS
 # Three chronological development validations; no Oct-Dec rows are opened.
 for fold,train_end,end in [('mayjun','2024-05-01','2024-07-01'),('julaug','2024-07-01','2024-09-01'),('sep','2024-09-01','2024-10-01')]:
  run(out=root/fold,policies=a.policies.resolve(),features_path=a.features.resolve(),candidates_path=a.candidates.resolve(),fields=fields,train_start=u('2023-10-01'),oos_start=u(train_end),oos_end=u(end),specs=specs)
 (root/'run_manifest.json').write_text(json.dumps({'schema':'strict_r3_short_policy_funnel_a_v1','status':'complete','selection_window':'pre-2024-10 only','specs':[s.name for s in specs],'f90_selection':str(a.selection.resolve())},indent=2)+'\n')
 print(root)
if __name__=='__main__':main()
