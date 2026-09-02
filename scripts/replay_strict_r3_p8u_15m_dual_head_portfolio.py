#!/usr/bin/env python3
"""Compare no/entry/continuation/both heads in one offline OOS auction replay."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from extreme_price_movements.portfolio_policy_replay import compute_replay_metrics, normalise_candidate_table, replay_candidates
from scripts import replay_strict_r3_p8u_15m_continuation_portfolio as port
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import CAUSAL_AUCTION_CURVE, _params as portfolio_params


def _selected_entry(path:Path,floor:float,spec:str,interaction:str)->set[str]:
    d=pd.read_parquet(path)
    col=f"selected__{interaction}"
    d=d.loc[d.floor_bps.eq(floor)&d.model_spec.eq(spec)&d[col].fillna(False)]
    return set(d.candidate_id.astype(str))

def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument('--entry-predictions',type=Path,required=True);p.add_argument('--continuation-outcomes',type=Path,required=True)
    p.add_argument('--state-root',type=Path,default=port.DEFAULT_STATE_ROOT);p.add_argument('--floor',type=float,default=30.)
    p.add_argument('--entry-model',default='lgb_huber_bps');p.add_argument('--entry-interaction',default='veto_pred_ge_0')
    p.add_argument('--continuation-arm',default='C1_activation_only');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    selected=_selected_entry(a.entry_predictions,a.floor,a.entry_model,a.entry_interaction)
    raw=pd.read_parquet(a.continuation_outcomes)
    raw=raw.loc[pd.to_numeric(raw.mc1_threshold_bps,errors='coerce').eq(a.floor)].copy();raw.candidate_id=raw.candidate_id.astype(str)
    parent=raw.loc[raw.arm.eq('C0_parent')].copy();cont=raw.loc[raw.arm.eq(a.continuation_arm)].copy()
    if parent.candidate_id.duplicated().any() or cont.candidate_id.duplicated().any():raise AssertionError('one outcome per candidate/arm required')
    frames=[]
    for name,frame in [('no_head',parent),('ordinal_policy_only',parent.loc[parent.candidate_id.isin(selected)]),('stateful_continuation_only',cont),('both_heads',cont.loc[cont.candidate_id.isin(selected)])]:
        x=frame.copy();x['arm']=name;frames.append(x)
    outcomes=pd.concat(frames,ignore_index=True)
    prices=port._entry_prices(a.state_root.resolve());priority=port._bcf_priority();params=portfolio_params();summary=[]
    for arm,frame in outcomes.groupby('arm',sort=False):
        candidates=port._candidate_table(frame,prices,priority);decisions,equity,_=replay_candidates(candidates,params,mode='global_auction',ev_curve=CAUSAL_AUCTION_CURVE,market_mode='perp');decisions=port._attach_ids(decisions,candidates);accepted=decisions.loc[decisions.accepted.fillna(False)].copy();tag=arm
        candidates.to_parquet(out/f'{tag}_candidates.parquet',index=False);decisions.to_parquet(out/f'{tag}_decisions.parquet',index=False);accepted.to_parquet(out/f'{tag}_accepted.parquet',index=False);equity.to_parquet(out/f'{tag}_equity.parquet',index=False)
        met=compute_replay_metrics(candidates,decisions,equity,params=params);summary.append({'arm':arm,'routed_candidates':len(candidates),'portfolio_accepted':len(accepted),**met})
    table=pd.DataFrame(summary);table.to_parquet(out/'portfolio_summary.parquet',index=False)
    (out/'run_manifest.json').write_text(json.dumps({'schema':'p8u-dual-head-four-arm-portfolio-v1','scope':'offline strict-OOS only','floor_bps':a.floor,'entry':{'model':a.entry_model,'interaction':a.entry_interaction,'authority':'veto/blend only'},'continuation':{'arm':a.continuation_arm,'authority':'tightening-only C1'},'priority':'BCF MC1 only','portfolio':asdict(params),'arms':['no_head','ordinal_policy_only','stateful_continuation_only','both_heads']},indent=2,default=str)+'\n')

if __name__=='__main__':main()
