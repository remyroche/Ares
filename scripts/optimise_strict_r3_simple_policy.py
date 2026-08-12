#!/usr/bin/env python3
"""Optimise strict-R3 trailing exits with ``simple_policy_optimiser``.

The score is held fixed.  Only 2025 candidates are used to choose the policy;
all later replay months must use the saved winner unchanged.  A flat 100-bps
round-trip cost is applied once in the objective, matching the declared
execution contract.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.replay_strict_r3_simple_policy_15m import _load_15m,_paths_for_group,_load_labels,HORIZON_BARS,COST_BPS
from extreme_price_movements.simple_policy_optimiser import simulate_and_score

SCORES=ROOT/'data_perp/artifacts/strict_r3_full_inference_2025_2026_v2/predictions.parquet'

def main():
    out=ROOT/'data_perp/artifacts/strict_r3_simple_policy_optimised_20260809_v1'; out.mkdir(parents=True,exist_ok=False)
    scores=pd.read_parquet(SCORES,columns=['candidate_id','__ts__','__symbol__','month','final_score'])
    scores.__ts__=pd.to_datetime(scores.__ts__,utc=True)
    # The 95th monthly-percentile score threshold is frozen before policy HPO.
    scores=scores[(scores.month.str.startswith('2025-'))&(scores.final_score>=.95)].copy()
    labels=_load_labels(sorted(scores.month.unique()))
    x=scores.merge(labels,on=['candidate_id','__ts__','__symbol__'],how='inner',validate='one_to_one')
    x=x[x.policy_atr_valid].copy().reset_index(drop=True)
    # Deterministic, equal-month cap keeps the optimisation practical without
    # allowing the densest month to dominate its geometry.
    x['_h']=pd.util.hash_pandas_object(x.candidate_id,index=False).astype('uint64')
    x=x.sort_values(['month','_h']).groupby('month',group_keys=False).head(3500).reset_index(drop=True)
    rows=[]; op=[]; hi=[]; lo=[]; cl=[]
    for sym,g in x.groupby('__symbol__',sort=True):
        ts,o,h,l,c=_load_15m(str(sym)); valid,fo,fh,fl,fc=_paths_for_group(g,ts,o,h,l,c)
        if not valid.any(): continue
        take=np.flatnonzero(valid); z=g.iloc[take].copy()
        rows.append(z); op.append(fo);hi.append(fh);lo.append(fl);cl.append(fc)
    x=pd.concat(rows,ignore_index=True); op=np.concatenate(op);hi=np.concatenate(hi);lo=np.concatenate(lo);cl=np.concatenate(cl)
    run=pd.DataFrame({'timestamp':x.__ts__,'symbol':x.__symbol__,'side':1.,'rank_pct':x.final_score,'barrier_pct':x.atr_1h.to_numpy(float)/op[:,0], 'expected_half_spread_bps':0.,'exit_quote_half_spread_bps':0.,'entry_slippage_proxy_bps':0.,'market_mode':'perps'})
    def objective(t):
        m=simulate_and_score(run,op,hi,lo,cl,cost_pct=0.,size_power=1.,replay_timeframe='15m',market_mode='perps',sl_mult=t.suggest_float('sl_mult',1.,5.),sl_abs_cap_pct=0.,trailing_activation_mult=t.suggest_float('trailing_activation_mult',.25,4.),trailing_activation_cap_pct=0.,trailing_activation_max_bars=HORIZON_BARS,fixed_trailing_gap_mult=t.suggest_float('fixed_trailing_gap_mult',.10,2.),capital_protect_mfe_mult=0.,adverse_exit_enabled=False,hard_tp_abs_pct=0.,max_concurrent_trades=10**9,max_concurrent_per_asset=10**9,max_new_entries_per_bar=10**9)
        net=np.asarray(m['gross_returns'],float)*1e4-COST_BPS
        monthly=pd.DataFrame({'m':x.month.to_numpy(),'net':net}).groupby('m').net.mean()
        return float(monthly.median()-.5*(monthly-monthly.median()).abs().median())
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=17))
    study.optimize(objective,n_trials=40,show_progress_bar=False)
    pd.DataFrame([{'trial':t.number,'value':t.value,**t.params} for t in study.trials if t.value is not None]).to_parquet(out/'trials.parquet',index=False)
    payload={'score_population':str(SCORES),'policy_development':'2025 only; score >= monthly 95th percentile; equal-month deterministic cap 3,500','execution':'15m entry at signal+1h; H12; flat 100 bps cost once','objective':'median monthly net bps/trade - 0.5 monthly MAD','winner':study.best_params,'winner_objective':study.best_value,'rows':len(x)}
    (out/'winner.json').write_text(json.dumps(payload,indent=2)); print(json.dumps(payload))
if __name__=='__main__': main()
