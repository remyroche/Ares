#!/usr/bin/env python3
"""Materialize deterministic selection-audit tables for the funnel."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extreme_price_movements.funnel_selection import select_winner
ROOT=Path(__file__).resolve().parents[1]
def main():
 specialist=ROOT/'data_perp/artifacts/frozen_specialist_query_hpo_20260810_v1'; a=pd.read_parquet(specialist/'specialist_target_query_winners.parquet').rename(columns={'target':'arm'}); w=select_winner(a,tie_tolerance_bps=0.0); pd.DataFrame([w]).to_parquet(specialist/'selection_winner.parquet',index=False)
 residual=ROOT/'data_perp/artifacts/frozen_residual_grade_ablation_20260810_v1'; b=pd.read_parquet(residual/'metrics.parquet').rename(columns={'grade_definition':'arm'}); wr=select_winner(b,tie_tolerance_bps=0.0); pd.DataFrame([wr]).to_parquet(residual/'selection_winner.parquet',index=False)
 (ROOT/'data_perp/artifacts/funnel_selection_audit_20260810_v1').mkdir(parents=True,exist_ok=True)
 payload={'specialist_winner':str(specialist/'selection_winner.parquet'),'residual_grade_winner':str(residual/'selection_winner.parquet'),'tie_tolerance_bps':0.0,'primary_rule':'global top5 net; exact ties by monthly stability; then top1 net'}
 (ROOT/'data_perp/artifacts/funnel_selection_audit_20260810_v1/manifest.json').write_text(json.dumps(payload,indent=2)+'\n')
if __name__=='__main__': main()
