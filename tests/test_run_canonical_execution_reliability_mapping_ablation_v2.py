from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1]/'scripts/run_canonical_execution_reliability_mapping_ablation_v2.py';S=importlib.util.spec_from_file_location('m2',P);M=importlib.util.module_from_spec(S);assert S and S.loader;S.loader.exec_module(M)
def test_fixed_contract_constants():assert (M.POOL,M.SIDE,M.LAMBDA,M.WINDOW.days)==(2000,1000,1000.,21)
def test_strict_snapshot_has_no_plateaus_or_raw_order_inversions():
 raw=np.array([.3,.1,.2]);z=M.strict_snapshot(raw,np.array([1.,1.,0.]));a=np.argsort(raw);assert np.all(np.diff(z[a])>0)
def test_positive_huber_forces_nonnegative_slope():
 n=M.POOL;h=pd.DataFrame({'raw_score':np.linspace(-1,1,n),'execution_net_ev_12h':np.linspace(.01,-.01,n)});z=M.positive_huber(h,np.array([-.5,0,.5]));assert np.all(np.diff(z)>=0)
def test_fractional_tie_book_uses_exact_equality_and_reconciles():
 x=pd.DataFrame({'candidate_id':['a','b','c'],'side_name':['long']*3,'__symbol__':['x']*3,'__ts__':pd.date_range('2025-01-01',periods=3,tz='UTC'),'score':[1.,1.,0.],'execution_net_ev_12h':[.01,-.01,0.]});w,m=M.fractional_book(x,'score',1/3);assert m['boundary_tie_population']==2 and np.isclose(w.sum(),1.) and np.isclose((w*x.execution_net_ev_12h).sum(),0.)
def test_m3_is_pooled_only_and_timestamp_order_preserving():
 n=M.POOL;ts=pd.Timestamp('2025-04-01',tz='UTC');h=pd.DataFrame({'raw_score':np.linspace(0,1,n),'execution_net_ev_12h':np.linspace(-.01,.01,n),'execution_decision_utc':[ts]*n});e=pd.DataFrame({'raw_score':[.1,.5,.9],'execution_decision_utc':[ts]*3});z=M.strict_snapshot(e.raw_score.to_numpy(),M.m3_bins(h,e));assert np.all(np.diff(z)>0)
