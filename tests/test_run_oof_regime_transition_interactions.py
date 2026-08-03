from __future__ import annotations

import pandas as pd
import pytest

from extreme_price_movements.regime_oof_stack import RegimeOOFStackError
from scripts.run_oof_regime_transition_interactions import _safe_predictors, build_panel


def test_predictor_denylist_keeps_realized_economics_out() -> None:
    selected = _safe_predictors(["mvreg__vol", "regime_state_entropy", "execution_net_ev_12h", "target_soft", "score_residual_expected_ev"])
    assert selected == ["mvreg__vol", "regime_state_entropy", "score_residual_expected_ev"]


def test_build_panel_fails_closed_on_missing_asof_multiview_coverage(tmp_path) -> None:
    keys = pd.DataFrame({"candidate_id": ["a"], "__ts__": pd.to_datetime(["2024-01-01"], utc=True), "__symbol__": ["BTC"], "side_name": ["long"]})
    soft = keys.assign(regime_state_p__0=1., regime_state_p__1=0., regime_state_ood_score=0., regime_state_id="0", regime_state_entropy=0., regime_state_margin=1., regime_state_uncertainty=0., regime_fold_id="f", regime_train_end_utc=pd.Timestamp("2023-12-01",tz="UTC"), regime_available_utc=pd.Timestamp("2024-01-01",tz="UTC"), transition_state_p__stable=1., transition_state_p__approach=0., transition_state_p__immediate_lead=0., transition_state_p__transition=0., transition_state_p__acceleration=0., transition_state_p__early_destination=0., transition_state_p__settled_destination=0., transition_state_ood_score=0., transition_state_id="stable", transition_state_entropy=0., transition_state_margin=1., transition_state_uncertainty=0., transition_fold_id="t", transition_train_end_utc=pd.Timestamp("2023-12-01",tz="UTC"), transition_available_utc=pd.Timestamp("2024-01-01",tz="UTC"))
    scores = keys.assign(execution_net_ev_12h=.01, score_residual_expected_ev=.02)
    root=tmp_path/'mv'; root.mkdir()
    # Deliberately stale beyond the two-hour causal join tolerance.
    mv=pd.DataFrame({"source_utc":pd.to_datetime(["2023-12-01"],utc=True),"mv": [1.]})
    soft.to_parquet(tmp_path/'soft.parquet',index=False); scores.to_parquet(tmp_path/'scores.parquet',index=False); mv.to_parquet(root/'regime_oof_features.parquet',index=False); mv.to_parquet(root/'transition_oof_features.parquet',index=False)
    with pytest.raises(RegimeOOFStackError, match="missing coverage"):
        build_panel(soft_path=tmp_path/'soft.parquet',multiview_root=root,scores_path=tmp_path/'scores.parquet',max_multiview_features=1)
