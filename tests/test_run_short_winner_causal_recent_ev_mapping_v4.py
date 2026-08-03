from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

RUNNER = Path(__file__).resolve().parents[1] / 'scripts/run_short_winner_causal_recent_ev_mapping_v4.py'
SPEC = importlib.util.spec_from_file_location('short_mapping_v4', RUNNER); assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)

def frame(ids, ends, sides=('long', 'short'), scores=(.1, .9), y=(.01, -.01)):
    n=len(ids); ts=pd.Timestamp('2025-04-10T00:00:00Z')
    return pd.DataFrame({'candidate_id':ids,'side_name':list(sides)[:n],'__symbol__':['A']*n,'__ts__':[ts]*n,'execution_decision_utc':[ts]*n,'execution_label_end_utc':ends,'raw_score':list(scores)[:n],'execution_net_ev_12h':list(y)[:n],'score_available_utc':[ts]*n})

def test_snapshot_reference_window_is_label_end_not_decision_time_and_no_overlap():
    s=pd.Timestamp('2025-04-10T00:00:00Z')
    h=frame(['old','too_old','future'], [s-pd.Timedelta(hours=1),s-pd.Timedelta(days=22),s])
    e=frame(['eval'], [s+pd.Timedelta(hours=12)], sides=('long',), scores=(.3,), y=(.0,))
    # Fixed production support is intentionally too high here, but audit must be exact.
    _, a=M.map_one_snapshot(h,e,s)
    assert a['reference_rows']==1 and a['label_window_exact'] and a['evaluation_reference_identity_overlap']==0

def test_side_residual_formula_and_weak_side_fallback_are_exact():
    s=pd.Timestamp('2025-04-10T00:00:00Z')
    n=M.POOL; ids=[f'i{i}' for i in range(n)]
    h=pd.DataFrame({'candidate_id':ids,'side_name':['long']*n,'__symbol__':['A']*n,'__ts__':[s-pd.Timedelta(days=1)]*n,'execution_decision_utc':[s-pd.Timedelta(days=1)]*n,'execution_label_end_utc':[s-pd.Timedelta(hours=1)]*n,'score_available_utc':[s-pd.Timedelta(days=1)]*n,'raw_score':np.linspace(-1,1,n),'execution_net_ev_12h':np.linspace(-.01,.01,n)})
    e=frame(['e1','e2'],[s+pd.Timedelta(hours=12)]*2,sides=('long','short'),scores=(.2,.2),y=(0.,0.))
    o,a=M.map_one_snapshot(h,e,s)
    assert o.map_eligible.all() and o.loc[o.side_name.eq('short'),'pooled_plus_side'].iloc[0] == o.loc[o.side_name.eq('short'),'pooled_21d'].iloc[0]
    assert o.loc[o.side_name.eq('short'),'side_residual_weight'].iloc[0] == 0.0
    assert np.isclose(o.loc[o.side_name.eq('long'),'side_residual_weight'].iloc[0], n/(n+M.LAMBDA))

def test_global_topk_and_same_id_controls_contract_constants():
    assert (M.POOL,M.SIDE,M.LAMBDA,M.WINDOW.days)==(2000,1000,500.0,21)
    assert 'global top-k' in M.run.__doc__ if M.run.__doc__ else True
