"""Focused checks for the frozen-control short payoff/loss screen."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from scripts.correct_bounded_side_local_support_composition_ties import expected_precision

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data_perp/artifacts/bounded_short_conditional_payoff_ablation_20260730_v2'
DAY=ROOT/'data_perp/artifacts/bounded_short_conditional_payoff_ablation_20260730_v2_day_blocks'

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()

def test_frozen_control_march_only_selection_and_cutoff_contract():
 m=json.loads((OUT/'manifest.json').read_text());parity=json.loads((OUT/'control_parity.json').read_text());runner=ROOT/'scripts/run_bounded_short_conditional_payoff_ablation.py'
 assert m['runner_sha256']==sha(runner)
 assert parity['long_robust_fixed'] and parity['short_A_control_tail_2_bit_identical'] and parity['max_abs_delta']==0.0
 assert 'March is development OOF only' in m['contract']['selection'] and m['contract']['map']=='March OOF fit applied April only'
 grid=pd.read_csv(OUT/'march_oof_arm_selection.csv');assert grid.iloc[0]['arm']=='B_peak_slope' and grid.iloc[0]['short_tail_weight']==2.0
 assert len(pd.read_parquet(OUT/'april_confirmation_predictions.parquet'))==69258

def test_corrected_ties_and_required_gates_fail():
 assert np.isclose(expected_precision(np.array([1.]),np.array([2.,-2.]),2),2/3)
 ties=pd.read_csv(OUT/'tie_bounds.csv');mapped=ties[(ties.score_kind=='mapped')&(ties.top_fraction==.1)].iloc[0]
 assert mapped.random_tie_expected_precision!=float(mapped.random_tie_expected_net_bps>0)
 gates=pd.read_csv(OUT/'promotion_gates.csv');required=['expected top10 economics','latest week','side allocation','asset max','calibration']
 for phrase in required: assert any(gates.gate.str.contains(phrase))
 assert not gates['pass'].all()

def test_day_block_sources_coverage_and_intervals():
 dm=json.loads((DAY/'manifest.json').read_text());assert dm['source_manifest_sha256']==sha(OUT/'manifest.json');assert dm['source_predictions_sha256']==sha(OUT/'april_confirmation_predictions.parquet')
 x=pd.read_csv(DAY/'utc_day_block_intervals.csv');top1=x[x.top_fraction.eq(.01)].iloc[0];top10=x[x.top_fraction.eq(.1)].iloc[0]
 assert top1.utc_days==29 and top10.utc_days==30
 assert top1.ci95_low_bps<0<top1.ci95_high_bps and top10.ci95_low_bps<0<top10.ci95_high_bps
