from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1]/'scripts/run_canonical_execution_reliability_mapping_ablation.py';S=importlib.util.spec_from_file_location('m',P);M=importlib.util.module_from_spec(S);assert S and S.loader;S.loader.exec_module(M)
def test_constants_are_fixed_and_no_hpo(): assert (M.POOL,M.SIDE,M.LAMBDA,M.WINDOW.days)==(2000,1000,1000.,21)
def test_strict_pava_is_strictly_ordered():
 x=np.array([.1,.2,.3]);y=np.array([1.,0.,1.]);z=M.pava_strict(x,y);assert np.all(np.diff(z)>0)
def test_huber_map_is_monotone_for_positive_slope():
 n=M.POOL;x=np.linspace(-2,2,n);h=pd.DataFrame({'raw_score':x,'execution_net_ev_12h':.01*np.tanh(x)});z=M.huber_tanh(h,np.array([-.5,0,.5]));assert np.all(np.diff(z)>=0)
def test_timestamp_percentile_preserves_order_within_timestamp():
 n=M.POOL;ts=pd.Timestamp('2025-04-01',tz='UTC');h=pd.DataFrame({'raw_score':np.linspace(0,1,n),'execution_net_ev_12h':np.linspace(-.01,.01,n),'execution_decision_utc':[ts]*n,'side_name':['long']*n});e=pd.DataFrame({'raw_score':[.1,.5,.9],'execution_decision_utc':[ts]*3,'side_name':['long']*3});z=M.timestamp_percentile(h,e);assert np.all(np.diff(z)>0)
