from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.regime_oof_stack import RegimeOOFStackError
from scripts.run_oof_gam_regime_calibrator import _fit_gam, _load_source_sidecars, feature_lists


def test_gam_feature_contract_is_additive_and_keeps_layers_distinct() -> None:
    fields = feature_lists()
    assert fields["baseline_spline"] == ["raw_trust_score"]
    assert "regime_state_entropy" in fields["regime_gam"]
    assert "transition_state_entropy" not in fields["regime_gam"]
    assert "transition_state_entropy" in fields["transition_gam"]
    assert "adverse_competing_risk_p__regime_plus_transition" not in fields["combined_gam"]
    assert "adverse_competing_risk_p__regime_plus_transition" in fields["combined_plus_adverse_gam"]
    assert all("timing" not in field and "mae" not in field and "wait" not in field for values in fields.values() for field in values)


def test_additive_spline_gam_outputs_finite_causal_predictions() -> None:
    rng = np.random.default_rng(41)
    fields = feature_lists()["combined_plus_adverse_gam"]
    train = pd.DataFrame({field: rng.uniform(.01, .99, 180) for field in fields})
    train["execution_net_ev_12h"] = .02 * train["raw_trust_score"] - .01 * train["adverse_competing_risk_p__regime_plus_transition"] + rng.normal(0, .002, len(train))
    evaluation = pd.DataFrame({field: rng.uniform(.01, .99, 30) for field in fields})
    raw, mapped = _fit_gam(train, evaluation, fields)
    assert raw.shape == mapped.shape == (30,)
    assert np.isfinite(raw).all() and np.isfinite(mapped).all()


def test_source_sidecars_fail_closed_when_candidate_support_differs(tmp_path) -> None:
    root = tmp_path / "trust"
    sidecars = root / "prediction_sidecars"
    sidecars.mkdir(parents=True)
    keys = pd.DataFrame({"candidate_id": ["a"], "__ts__": pd.to_datetime(["2024-01-01"], utc=True), "__symbol__": ["BTC"], "side_name": ["long"], "trust_fold_id": ["q1"], "trust_train_end_utc": pd.to_datetime(["2024-01-01"], utc=True), "raw_trust_score": [.1], "mapped_score": [.1]})
    for index, source_arm in enumerate(["baseline", "regime_only", "transition_only", "regime_plus_transition", "regime_plus_transition_plus_adverse_risk"]):
        frame = keys.copy()
        if index == 1:
            frame.loc[0, "candidate_id"] = "other"
        frame.to_parquet(sidecars / f"{source_arm}.parquet", index=False)
    with pytest.raises(RegimeOOFStackError, match="exact candidate support"):
        _load_source_sidecars(root)
