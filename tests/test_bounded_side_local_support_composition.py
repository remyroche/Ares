"""Focused artifact checks for the side-local support-composition screen."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data_perp/artifacts/bounded_side_local_support_composition_20260730_v1'

def test_predeclared_side_grid_global_selection_and_adverse_contract() -> None:
 m=json.loads((OUT/'manifest.json').read_text());grid=pd.read_csv(OUT/'march_oof_side_local_weight_grid.csv')
 assert len(grid)==64
 assert m['contract']['selection'].startswith('64 predeclared')
 assert 'no side quota' in m['contract']['selection']
 assert m['adverse_support']['status']=='AVAILABLE_STRICT_OOF_ADVERSE_SEVERITY'
 assert 'no MAE action' in m['adverse_support']['semantic']
 assert m['contract']['portfolio_replay']=='NOT_RUN'

def test_control_cutoff_and_mapped_tie_gate() -> None:
 parity=json.loads((OUT/'control_parity.json').read_text())
 assert parity['bit_identical'] is True and parity['max_abs_delta']==0.0
 ties=pd.read_csv(OUT/'tie_bounds.csv')
 top1=ties[(ties.score_kind=='mapped')&(ties.top_fraction==.01)].iloc[0]
 assert top1.cutoff_tie_rows>top1.rows
 assert top1.random_tie_expected_net_bps<0.0
 predictions=pd.read_parquet(OUT/'april_confirmation_predictions.parquet')
 assert len(predictions)==69258 and predictions['execution_label_end_utc'].isna().sum()==0
