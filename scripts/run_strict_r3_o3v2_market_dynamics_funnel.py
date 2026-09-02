#!/usr/bin/env python3
"""Legacy candidate-ranker screen for market-wide labels.

Market-dynamics labels have one value for every decision timestamp, rather
than one value for every asset candidate.  Ranking asset candidates directly
on such a label lets candidate-specific stack fields break otherwise-global
ties, which is not a valid test of a market-state target.  The production
research path is therefore :mod:`run_strict_r3_o3v2_market_context_funnel`:
it learns one causal context value per timestamp and evaluates its incremental
temporal calibration effect separately from cross-sectional alpha.

This legacy entry point is retained only to reproduce the already-written
diagnostic receipts.  New runs must explicitly acknowledge that it is not a
promotion-eligible experiment.
"""
from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import run_strict_r3_o3v2_path_auxiliary_funnel as f
from audit_strict_r3_o3v2_market_dynamics_inputs import FAMILY_CANDIDATES

GROUPS={
 'trend':(('market_trend_continuation_12h',1.),('market_signed_directional_efficiency_12h',1.),('market_time_to_trend_break_12h',-1.)),
 'volatility':(('market_vol_change_12h',1.),('market_vol_acceleration_12h',1.)),
 'breadth':(('market_breadth_change_12h',1.),),
 'dispersion':(('cross_sectional_dispersion_change_12h',1.),),
 'flow':(('market_turnover_change_12h',1.),('market_volume_concentration_change_12h',-1.)),
 'stress':(('market_future_max_drawdown_12h',-1.),('market_jump_asymmetry_12h',-1.)),
}
BLOCK={'trend':'trend_persistence','volatility':'volatility_regime','breadth':'breadth_participation','dispersion':'cross_sectional_dispersion','flow':'volume_flow','stress':'tail_stress'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--group',choices=GROUPS,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--ledger',type=Path,default=f.DEFAULT_LEDGER);p.add_argument('--auxiliary',type=Path,default=ROOT/'data_perp/artifacts/strict_r3_o3v2_market_label_adapter_20260825_v1');p.add_argument('--legacy-unsafe-candidate-ranker',action='store_true');a=p.parse_args()
 if not a.legacy_unsafe_candidate_ranker:
  raise SystemExit(
   'Market-wide targets must be screened with run_strict_r3_o3v2_market_context_funnel.py; '
   'this legacy candidate-ranker does not measure a valid market-context effect. '
   'Pass --legacy-unsafe-candidate-ranker only to reproduce an already-declared diagnostic.'
  )
 f.TARGETS=tuple(f.TargetSpec('market_'+c,'market',c,d) for c,d in GROUPS[a.group]); f._fields=lambda _ledger:tuple(FAMILY_CANDIDATES[BLOCK[a.group]])
 print(f.run(ledger=a.ledger.resolve(),auxiliary=a.auxiliary.resolve(),out=a.out.resolve(),families=('market',),folds=f.FOLDS,resume=False,max_jobs=None))
if __name__=='__main__':main()
