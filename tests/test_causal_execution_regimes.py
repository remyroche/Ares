import numpy as np
import pandas as pd

from extreme_price_movements.causal_execution_regimes import (
    CausalRegimeStateModel,
    add_regime_transition_labels,
)


def test_transform_uses_training_ood_reference_not_evaluation_batch():
    rng = np.random.default_rng(4)
    train = pd.DataFrame({"a": rng.normal(size=150), "b": rng.normal(size=150)})
    model = CausalRegimeStateModel.fit(train, ["a", "b"], k_values=(3,))
    point = pd.DataFrame({"a": [8.0], "b": [8.0]})
    single = model.transform(point)
    batched = model.transform(pd.concat([point, pd.DataFrame({"a": [-8.0], "b": [-8.0]})], ignore_index=True))
    assert single.loc[0, "causal_regime_ood_z"] == batched.loc[0, "causal_regime_ood_z"]
    assert np.isclose(single.filter(like="posterior_").sum(axis=1).iloc[0], 1.0)
    assert "causal_regime_state" not in model.predictor_feature_columns
    assert 0.0 <= single.loc[0, "causal_regime_distance_percentile"] <= 1.0
    assert "causal_regime_distance_percentile" in model.stable_geometry_feature_columns


def test_transition_labels_are_unavailable_until_horizon_resolves():
    ts = pd.to_datetime(["2026-07-01T00:00:00Z", "2026-07-01T01:00:00Z", "2026-07-01T07:00:00Z"])
    frame = pd.DataFrame({"__ts__": ts, "__symbol__": ["BTC"] * 3, "side_name": ["long"] * 3, "causal_regime_state": [0, 1, 1]})
    out = add_regime_transition_labels(frame, observed_through=pd.Timestamp("2026-07-01T06:00:00Z"))
    assert out.loc[0, "causal_regime_change_within_6h"] == 1.0
    assert pd.isna(out.loc[1, "causal_regime_change_within_6h"])
    assert out.loc[0, "causal_regime_change_6h_label_resolution_utc"] == pd.Timestamp("2026-07-01T06:00:00Z")


def test_transition_label_uses_post_period_buffer_and_decision_time_column():
    # First row is in the evaluation period; second is only the 6h label
    # buffer.  The state change must be visible, not incorrectly marked false
    # merely because the weekly feature output stops after the first row.
    frame = pd.DataFrame({
        "execution_decision_utc": pd.to_datetime(["2026-07-07T23:00:00Z", "2026-07-08T02:00:00Z"]),
        "__symbol__": ["ETH", "ETH"], "side_name": ["short", "short"],
        "causal_regime_state": [1, 0],
    })
    out = add_regime_transition_labels(
        frame, observed_through=pd.Timestamp("2026-07-08T05:00:00Z"), time_column="execution_decision_utc"
    )
    assert out.loc[0, "causal_regime_change_within_6h"] == 1.0
