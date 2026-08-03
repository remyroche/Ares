#!/usr/bin/env python3
"""Materialise compact, windowed R3 matrices one symbol at a time.

This separates memory handling from target research: each parquet contains one
side/window/symbol only and can be consumed without retaining the population.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_tp6_r3_base_feature_ladder import POOL

def main():
 p=argparse.ArgumentParser();p.add_argument('--side',required=True,choices=('long','short'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 panel=ROOT/'data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3';winner=ROOT/'data_perp/artifacts/full_universe_tp6_sl4_h12_sidecar_20260802_v1';labels=ROOT/'data_perp/artifacts/tp6_sl4_robust_clear_labels_20260802_v1'
 available=set(pd.read_parquet(next((panel/'parts').glob('*.parquet'))).columns);cols=[c for c in POOL if c in available]
 parts=[]
 for part in sorted((panel/'parts').glob('*.parquet')):
  x=pd.read_parquet(part,columns=['candidate_id','side_name','__ts__',*cols]);x=x.loc[x.side_name.eq(a.side)]
  w=pd.read_parquet(winner/'parts'/part.name,columns=['candidate_id','t4_tp6_sl4_gross_bps','t4_tp6_sl4_net_bps'])
  y=pd.read_parquet(labels/'parts'/part.name,columns=['candidate_id','label_valid','lower_touch_minute','robust_clear_event_b25','robust_clear_soft_b25_t50','__label_available_at__'])
  x=x.merge(w,on='candidate_id',validate='one_to_one').merge(y,on='candidate_id',validate='one_to_one');x=x.loc[x.label_valid]
  x['y']=np.select([x.robust_clear_event_b25.eq(1),x.lower_touch_minute.ge(0)],[2,0],default=1).astype('int8');x['available']=pd.to_datetime(x.__label_available_at__,utc=True)
  x[cols]=x[cols].replace([np.inf,-np.inf],np.nan).fillna(0.).astype('float32');dst=a.out/part.name;x[['candidate_id','__ts__','available','y','robust_clear_soft_b25_t50','t4_tp6_sl4_gross_bps','t4_tp6_sl4_net_bps',*cols]].to_parquet(dst,index=False,compression='zstd');parts.append({'path':dst.name,'rows':len(x)});print(parts[-1],flush=True)
 (a.out/'manifest.json').write_text(json.dumps({'side':a.side,'feature_pool':cols,'parts':parts,'schema':'tp6_r3_ladder_matrix_v1'},indent=2))
if __name__=='__main__':main()
