from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
RUNNER=Path(__file__).resolve().parents[1]/"scripts"/"run_canonical_execution_reliability_ablation_v2.py"
SPEC=importlib.util.spec_from_file_location("reliability_v2",RUNNER); assert SPEC and SPEC.loader
M=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=M;SPEC.loader.exec_module(M)
def _x():
 ts=pd.date_range("2025-03-01",periods=36,freq="h",tz="UTC")
 return pd.DataFrame({"candidate_id":[f"c{i}" for i in range(36)],"side_name":["long"]*36,"__symbol__":["A"]*36,"__ts__":ts,M.TIME:ts,M.END:ts+pd.Timedelta(hours=12),M.NET:np.arange(36)/1000,"base_oof_score":np.arange(36,dtype=float),**{s:np.arange(36,dtype=float) for s in ["preentry_transition__range_24h_pct__delta_12h","preentry_transition__meta_raw__volatility_zscore__delta_12h","preentry_transition__trend_r2_24__delta_12h","preentry_transition__jump_intensity__delta_12h","preentry_transition__meta_raw__chop_score__delta_12h"]}})
def test_outer_train_is_strictly_resolved_before_validation():
 x=_x();f={"validation_start_utc":"2025-03-01T18:00:00Z","validation_end_utc":"2025-03-02T00:00:00Z"};tr,va=M.outer_masks(x,f);assert x.loc[tr,M.END].lt(pd.Timestamp(f["validation_start_utc"])).all();assert va.sum()==6
def test_interactions_use_train_stats_are_bounded_and_exactly_five():
 x=_x();out=M.interaction_features(x.iloc[:8],x.base_oof_score,["preentry_transition__range_24h_pct__delta_12h","preentry_transition__meta_raw__volatility_zscore__delta_12h","preentry_transition__trend_r2_24__delta_12h","preentry_transition__jump_intensity__delta_12h","preentry_transition__meta_raw__chop_score__delta_12h"],x.iloc[8:]);assert out.shape==(len(x)-8,5);assert (out.abs()<=1).all().all()
def test_random_tie_expected_is_pooled_not_side_quota():
 x=_x().iloc[:4].copy();x["raw_score"]=[1,1,0,0];x[M.NET]=[.02,-.02,0,0];r=M.random_tie_expected(x,"raw_score",.25);assert r["selected_rows"]==1 and r["random_tie_expected_net_bps"]==0
def test_contract_requires_exactly_five_a5_interactions():
 c=M.load_contract();assert len(c["feature_arms"]["transition_interaction_sources"])==5

def test_capture_condition_is_applied_to_validation_head_metrics():
 x=_x().iloc[:4].copy()
 x["target_pre_exit_capture_valid"]=[1,1,0,1]
 x["target_pre_exit_meaningful_mfe"]=[1,0,1,1]
 assert M.condition_mask(x,"capture_valid_and_meaningful").tolist()==[True,False,False,True]

def test_multiclass_head_metrics_have_required_per_class_fields():
 y=np.array(["favorable_first","adverse_first_or_conflict","timeout"])
 p=np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]])
 out=M.head_metrics(y,p,"multi",classes=["favorable_first","adverse_first_or_conflict","timeout"])
 assert np.isfinite(out["macro_one_vs_rest_AUC"]) and "timeout" in out["per_class_Brier"]

def test_multiclass_head_metrics_accept_numpy_class_order():
 y=np.array(["favorable_first","adverse_first_or_conflict","timeout"])
 p=np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]])
 classes=np.array(["favorable_first","adverse_first_or_conflict","timeout"])
 out=M.head_metrics(y,p,"multi",classes=classes)
 assert np.isfinite(out["log_loss"])

def test_verified_v4_regime_attribution_is_exhaustive():
 c=M.load_contract()
 x,_=M.verify_input(M.ROOT/c["input_artifact"],c)
 assert set(x["regime_execution_risk_quintile"].unique())=={"Q1","Q2","Q3","Q4","Q5"}
 assert len(x)==110730

def test_binary_prediction_selects_only_the_positive_class_column():
 class Fake:
  classes_=np.array([0,1])
  def predict_proba(self,x): return np.tile(np.array([[.8,.2]]),(len(x),1))
 assert M._prediction(Fake(),pd.DataFrame({"x":[1,2]}),"binary").tolist()==[.2,.2]

def test_ece_is_ten_bin_weighted_calibration_error_with_diagnostics():
 ece,bins=M.ece_10bin(np.array([0,1]),np.array([.1,.9]))
 assert len(bins)==10 and np.isclose(ece,.1)

def test_cutoff_tie_selected_share_uses_needed_rows_not_tie_population():
 x=_x().iloc[:12].copy();x["raw_score"]=[2,2]+[1]*10;x[M.NET]=0
 r=M.random_tie_expected(x,"raw_score",.5)
 assert r["selected_rows"]==6 and r["boundary_tie_population"]==10 and np.isclose(r["cutoff_tie_selected_share"],4/6)

def test_context_variants_use_the_frozen_support_arm_and_allow_none():
 c=M.load_contract(); variants=M.context_variants(c,["raw_score","support__soft__S0__compact_d4"])
 assert any(name=="context_none" and fields==["raw_score","support__soft__S0__compact_d4"] for name,fields in variants)
