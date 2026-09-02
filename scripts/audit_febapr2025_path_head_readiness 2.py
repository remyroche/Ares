#!/usr/bin/env python3
"""Fail-closed readiness gate for historical CatBoost and five path heads."""
from __future__ import annotations
import json,hashlib
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
TOP=ROOT/'data_perp/artifacts/febapr2025_canonical_residual_top40_20260727_v1/population.parquet'
ARCH=ROOT/'data_perp/artifacts/20260724_path_archetype_labels_v9_packb31_8_top40/path_archetype_labels.parquet'
OUT=ROOT/'data_perp/artifacts/febapr2025_path_head_readiness_20260727_v1'
def main():
 if OUT.exists():raise FileExistsError(OUT)
 top=pd.read_parquet(TOP,columns=['candidate_id','__ts__']);arch=pd.read_parquet(ARCH,columns=['candidate_id','__ts__','side_name','path_archetype'])
 top['__ts__']=pd.to_datetime(top.__ts__,utc=True);arch['__ts__']=pd.to_datetime(arch.__ts__,utc=True)
 overlap=len(set(top.candidate_id)&set(arch.candidate_id));OUT.mkdir(parents=True)
 gate={'schema':'febapr2025_path_head_readiness_v1','status':'BLOCKED_NO_MATCHING_HISTORICAL_PATH_TARGETS','historical_top40_rows':len(top),'historical_period':[str(top.__ts__.min()),str(top.__ts__.max())],'frozen_archetype_rows':len(arch),'frozen_archetype_period':[str(arch.__ts__.min()),str(arch.__ts__.max())],'candidate_id_overlap':overlap,'required_heads':['CatBoost path archetype','peak_mfe_12h_atr','time_to_first_meaningful_mfe','mae_before_meaningful_mfe_atr','bars_before_price_stops_decreasing','future_slope_atr_per_hour'],'reason':'Current frozen labels are April-July 2026 only. No February-April 2025 identity overlap; proxy targets are forbidden.','next_requirement':'Materialise exact historical 12h path/archetype and five auxiliary targets from accepted 1m paths and frozen geometry, then rerun per-side March/April OOF.'}
 (OUT/'evidence_gate.json').write_text(json.dumps(gate,indent=2));
if __name__=='__main__':main()
