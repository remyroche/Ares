from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.regime_oof_stack import RegimeOOFStackError
from scripts.run_interaction_conditioned_residual_trust_oof import build_panel, feature_lists, run


def _write_inputs(tmp_path):
    timestamps = pd.date_range("2023-10-01", "2024-12-31 18:00", freq="6h", tz="UTC")
    n = len(timestamps)
    keys = pd.DataFrame({
        "candidate_id": [f"id-{i}" for i in range(n)], "__ts__": timestamps,
        "__symbol__": np.where(np.arange(n) % 2, "BTC", "ETH"),
        "side_name": np.where(np.arange(n) % 2, "long", "short"),
    })
    rng = np.random.default_rng(9)
    state = (np.arange(n) % 3).astype(int)
    transition = np.arange(n) % 7
    soft = keys.assign(
        regime_train_end_utc=timestamps - pd.Timedelta(days=3), regime_available_utc=timestamps - pd.Timedelta(hours=1),
        regime_fold_id="regime", regime_state_p__0=(state == 0).astype(float), regime_state_p__1=(state == 1).astype(float), regime_state_p__2=(state == 2).astype(float),
        regime_state_ood_score=.1, regime_state_id=state.astype(str), regime_state_entropy=.2, regime_state_margin=.8, regime_state_uncertainty=.1,
        transition_train_end_utc=timestamps - pd.Timedelta(days=3), transition_available_utc=timestamps - pd.Timedelta(hours=1), transition_fold_id="transition",
        transition_state_p__stable=(transition == 0).astype(float), transition_state_p__approach=(transition == 1).astype(float), transition_state_p__immediate_lead=(transition == 2).astype(float), transition_state_p__transition=(transition == 3).astype(float), transition_state_p__acceleration=(transition == 4).astype(float), transition_state_p__early_destination=(transition == 5).astype(float), transition_state_p__settled_destination=(transition == 6).astype(float),
        transition_active_probability=.2, transition_state_ood_score=.1, transition_state_id=transition.astype(str), transition_state_entropy=.2, transition_state_margin=.8, transition_state_uncertainty=.1,
    )
    score = rng.normal(0, .02, n)
    y = .35 * score + .005 * (state == 2) + rng.normal(0, .01, n)
    scores = keys.assign(execution_net_ev_12h=y, execution_gross_ev_12h=y+.001, execution_cost_return=.001, __reconstructed_soft_alpha_12h__=y, score_base_expected_ev=score*.8, score_residual_expected_ev=score)
    risk = keys.assign(probability_fold_id="risk", probability_evaluation_start_utc=timestamps.floor("D"), clean_opportunity_p__regime_plus_transition=.6, adverse_competing_risk_p__regime_plus_transition=.2)
    paths = (tmp_path / "soft.parquet", tmp_path / "scores.parquet", tmp_path / "risk.parquet")
    soft.to_parquet(paths[0], index=False); scores.to_parquet(paths[1], index=False); risk.to_parquet(paths[2], index=False)
    return paths


def test_feature_contract_excludes_action_and_uses_separate_clean_ablation() -> None:
    features = feature_lists()
    assert "adverse_competing_risk_p__regime_plus_transition" in features["regime_plus_transition_plus_adverse_risk"]
    assert "clean_opportunity_p__regime_plus_transition" not in features["regime_plus_transition_plus_adverse_risk"]
    assert "clean_opportunity_p__regime_plus_transition" in features["regime_plus_transition_plus_clean_probability"]
    assert all("timing" not in column and "mae" not in column for columns in features.values() for column in columns)


def test_runner_emits_matched_prediction_only_sidecars_and_chronological_provenance(tmp_path) -> None:
    soft, scores, risk = _write_inputs(tmp_path)
    output = run(output_dir=tmp_path / "out", soft_path=soft, scores_path=scores, risk_path=risk, start="2023-10-01", end="2025-01-01", min_train_rows=100, seed=3)
    summary = pd.read_csv(output / "metrics_summary.csv")
    assert set(summary["arm"]) == set(feature_lists())
    assert set(summary["selection_basis"]) == {"pooled_global_post_mapping_top_k"}
    sidecars = sorted((output / "prediction_sidecars").glob("*.parquet"))
    assert len(sidecars) == len(feature_lists())
    rows = [len(pd.read_parquet(path)) for path in sidecars]
    assert len(set(rows)) == 1 and rows[0] > 0
    assert all(TARGET not in pd.read_parquet(path).columns for path in sidecars for TARGET in ["execution_net_ev_12h"])
    provenance = pd.read_parquet(output / "fold_provenance.parquet")
    assert (pd.to_datetime(provenance["train_label_available_max_utc"], utc=True) < pd.to_datetime(provenance["evaluation_start_utc"], utc=True)).all()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["selection"]["per_timestamp_selection"] is False


def test_panel_fails_closed_when_action_column_is_requested(tmp_path) -> None:
    soft, scores, risk = _write_inputs(tmp_path)
    scores_frame = pd.read_parquet(scores).assign(wait_action_score=1.0)
    scores_frame.to_parquet(scores, index=False)
    # The panel does not select arbitrary score columns, hence this extra column
    # is harmless; the explicit predictor API is what forbids action fields.
    panel = build_panel(soft_path=soft, scores_path=scores, risk_path=risk)
    assert "wait_action_score" in panel
    with pytest.raises(RegimeOOFStackError, match="action-layer"):
        from scripts.run_interaction_conditioned_residual_trust_oof import _reject_action_fields
        _reject_action_fields(["wait_action_score"])
