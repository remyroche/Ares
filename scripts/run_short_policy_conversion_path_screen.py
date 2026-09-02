#!/usr/bin/env python3
"""Path-shape target screen for the frozen short policy-conversion ranker."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.run_short_policy_conversion_funnel import PolicySpec, run
def _utc(x: str) -> pd.Timestamp:
    y=pd.Timestamp(x); return y.tz_localize('UTC') if y.tzinfo is None else y.tz_convert('UTC')
SPECS=(
 PolicySpec('P0_policy_bps','Frozen policy-bps control','policy_bps',truncation=40,gain_family='linear',query_hours=1),
 PolicySpec('P5_early_mfe1','One-hour favorable MFE rank','early_mfe1_rank',truncation=40,gain_family='linear',query_hours=1),
 PolicySpec('P5_early_mfe2','Two-hour favorable MFE rank','early_mfe2_rank',truncation=40,gain_family='linear',query_hours=1),
 PolicySpec('P5_early_mfe3','Three-hour favorable MFE rank','early_mfe3_rank',truncation=40,gain_family='linear',query_hours=1),
 PolicySpec('P7_squeeze_l025','Three-hour MFE minus 0.25x pre-peak adverse rank','squeeze_l025_rank',truncation=40,gain_family='linear',query_hours=1),
 PolicySpec('P7_squeeze_l050','Three-hour MFE minus 0.5x pre-peak adverse rank','squeeze_l050_rank',truncation=40,gain_family='linear',query_hours=1),
 PolicySpec('P7_squeeze_l100','Three-hour MFE minus 1.0x pre-peak adverse rank','squeeze_l100_rank',truncation=40,gain_family='linear',query_hours=1),
 PolicySpec('P9_activation','Activation-before-adverse ordinal grade','activation_grade',truncation=40,gain_family='linear',query_hours=1),
 PolicySpec('P10_quality_a','Fast/clean/persistent conversion composite A','conversion_quality_a',truncation=40,gain_family='linear',query_hours=1),
 PolicySpec('P10_quality_b','Fast/clean/persistent conversion composite B','conversion_quality_b',truncation=40,gain_family='linear',query_hours=1),
)
def main()->None:
 p=argparse.ArgumentParser();
 for n in ('out','selection','policies','features','candidates','supportive-path'): p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args(); root=a.out.resolve(); root.mkdir(parents=True)
 fields=json.loads(a.selection.read_text())['feature_sets']['90']
 for fold,start,end in [('mayjun','2024-05-01','2024-07-01'),('julaug','2024-07-01','2024-09-01'),('sep','2024-09-01','2024-10-01')]:
  run(out=root/fold,policies=a.policies.resolve(),features_path=a.features.resolve(),candidates_path=a.candidates.resolve(),supportive_path=getattr(a,'supportive_path').resolve(),fields=fields,train_start=_utc('2023-10-01'),oos_start=_utc(start),oos_end=_utc(end),specs=SPECS)
 (root/'run_manifest.json').write_text(json.dumps({'schema':'strict_r3_short_policy_path_target_screen_v1','status':'complete','selection_window':'pre-2024-10 only','frozen_ranker':{'query_hours':1,'truncation':40,'gain':'linear','objective':'lambdarank','norm':True},'specs':[x.name for x in SPECS]},indent=2)+'\n')
if __name__=='__main__': main()
