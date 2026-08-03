from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_execution_ev_mixed_period_remedies import (
    ForwardWindow,
    add_trust_composites,
    build_forward_split,
    training_weights,
)


def _frame() -> pd.DataFrame:
    decision = pd.date_range("2026-05-01", periods=12, freq="12h", tz="UTC")
    return pd.DataFrame(
        {
            "execution_decision_utc": decision,
            "execution_label_end_utc": decision + pd.Timedelta(hours=12),
            "execution_net_ev_12h": np.linspace(-0.02, 0.03, len(decision)),
            "existing_alpha_ev": np.linspace(-1.0, 1.0, len(decision)),
            "oof_clean_favorable_probability": np.linspace(0.2, 0.8, len(decision)),
            "base_margin_to_cutoff_z": np.linspace(-2.0, 2.0, len(decision)),
            "alpha_prediction_uncertainty": np.linspace(0.1, 1.0, len(decision)),
            "catboost_archetype": ["a", "b"] * 6,
            "mkt_state__atr_compression_ratio__h0": np.linspace(0.1, 1.0, len(decision)),
            "mkt_state__atr_pct_change__h0": np.linspace(-1.0, 1.0, len(decision)),
            "mkt_state__atr_slope__h0": np.sin(np.arange(len(decision))),
            "mkt_state__mkt_atr_expansion_4h__h0": np.cos(np.arange(len(decision))),
            "mkt_state__volatility_of_volatility_48__h0": np.linspace(2.0, 0.0, len(decision)),
        }
    )


def test_forward_split_requires_resolved_labels_and_purge() -> None:
    frame = _frame()
    window = ForwardWindow(
        name="test",
        train_start="2026-05-01T00:00:00Z",
        cutoff="2026-05-05T00:00:00Z",
        evaluation_end="2026-05-07T00:00:00Z",
        retention_role="forward",
    )
    train, evaluation, audit = build_forward_split(frame, window, purge_hours=12.0)
    assert len(train) > 0 and len(evaluation) > 0
    assert pd.to_datetime(frame.iloc[train].execution_label_end_utc, utc=True).max() < pd.Timestamp(window.cutoff)
    assert pd.to_datetime(frame.iloc[evaluation].execution_decision_utc, utc=True).min() >= pd.Timestamp(window.cutoff)
    assert audit["promotion_eligible"] is False


def test_regime_balanced_weights_are_train_only_finite_and_normalized() -> None:
    frame = _frame()
    frame.loc[:8, "catboost_archetype"] = "common"
    frame.loc[9:, "catboost_archetype"] = "rare"
    weights, report = training_weights(frame, "regime_balanced")
    assert np.isfinite(weights).all()
    assert (weights > 0.0).all()
    assert np.isclose(weights.mean(), 1.0)
    assert weights[frame.catboost_archetype.eq("rare")].mean() > weights[
        frame.catboost_archetype.eq("common")
    ].mean()
    assert 0.0 < report["effective_sample_fraction"] <= 1.0


def test_trust_composites_depend_only_on_reference_transform_and_are_finite() -> None:
    frame = _frame()
    reference = frame.iloc[:8]
    target = frame.iloc[8:]
    first = add_trust_composites(reference, target)
    changed_target = target.copy()
    changed_target["execution_net_ev_12h"] = 99.0
    second = add_trust_composites(reference, changed_target)
    columns = [column for column in first if column.startswith("trust_")]
    assert len(columns) == 6
    np.testing.assert_allclose(first[columns], second[columns])
    assert np.isfinite(first[columns].to_numpy(dtype=float)).all()


def test_observable_volatility_weights_are_train_only_and_balanced() -> None:
    frame = pd.concat([_frame()] * 100, ignore_index=True)
    weights, report = training_weights(frame, "observable_volatility_balanced")
    assert np.isfinite(weights).all()
    assert (weights > 0.0).all()
    assert np.isclose(weights.mean(), 1.0)
    assert report["observable_state_count"] == 2
    assert len(report["observable_features"]) == 5
    assert sum(report["observable_state_rows"].values()) == len(frame)
