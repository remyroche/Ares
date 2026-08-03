from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_execution_ev_decomposition_calibration_ablation import (
    DIRECT,
    MULTITASK_FEATURES,
    _join_clean_labels,
    _compact_feature_columns,
    pooled_global_metrics,
    temporal_hierarchical_oof_calibration,
    temporal_multitask_oof_blend,
    temporal_shared_multitask_oof_meta,
    temporal_side_oof_isotonic,
)


def _frame() -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "execution_decision_utc": ts,
            "execution_label_end_utc": ts + pd.Timedelta(hours=1),
            "side_name": ["long", "short"] * 6,
        }
    )


def test_side_temporal_oof_isotonic_never_uses_current_or_future_fold() -> None:
    frame = _frame()
    raw = np.linspace(-1.0, 1.0, len(frame))
    target = raw * 0.01
    fold = np.repeat(np.arange(3, dtype=float), 4)
    mapped, audit = temporal_side_oof_isotonic(
        frame,
        raw,
        target,
        fold,
        decision_col="execution_decision_utc",
        resolution_col="execution_label_end_utc",
        side_col="side_name",
        min_rows=1,
    )
    assert np.isfinite(mapped).all()
    first_fold = [row for row in audit if row["fold"] == 0]
    assert all(row["reference_oof_rows"] == 0 for row in first_fold)
    later = [row for row in audit if row["fold"] == 2]
    # The row resolving exactly at the validation boundary remains excluded.
    assert all(3 <= row["reference_oof_rows"] <= 4 for row in later)
    assert all(
        pd.Timestamp(row["reference_max_resolution_utc"]) < pd.Timestamp(row["validation_start_utc"])
        for row in later
    )


def test_pooled_global_metrics_ranks_one_cross_side_tail() -> None:
    score = np.array([0.9, 0.8, 0.2, 0.1])
    net = np.array([0.01, -0.01, 0.03, 0.04])
    gross = net + 0.002
    result = pooled_global_metrics(
        score,
        net,
        gross,
        np.array(["long", "short", "long", "short"]),
        np.ones(4, dtype=bool),
        top_fraction=0.25,
    )
    assert result["top_k_rows"] == 1
    assert result["top_k_mean_net_ev"] == 0.01
    assert result["top_k_long_rows"] == 1
    assert result["top_k_short_rows"] == 0
    assert result["ranking_scope"].startswith("one_pooled_global")


def test_exact_clean_join_requires_risk_class_favorable_first(tmp_path) -> None:
    frame = _frame().assign(
        __ts__=pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC"),
        __symbol__="BTC/USD:USD",
        candidate_id=[f"candidate-{index}" for index in range(12)],
    )
    labels = frame.loc[:, ["__ts__", "__symbol__", "side_name", "candidate_id"]].copy()
    labels["tb_hard_label"] = (np.arange(len(labels)) % 3 == 2).astype(float)
    labels["risk_class"] = np.where(labels["tb_hard_label"].eq(1.0), 2, 0)
    labels["meaningful_mfe_reached"] = labels["tb_hard_label"]
    path = tmp_path / "labels.parquet"
    labels.to_parquet(path, index=False)
    joined = _join_clean_labels(frame, path)
    assert joined.sum() == 4


def test_compact_contract_accepts_h0_only_and_requires_point_in_time_source() -> None:
    frame = _frame().assign(
        existing_alpha_ev=0.1,
        catboost_p_0=0.2,
        raw_state_source_utc_h0=lambda item: item["execution_decision_utc"],
        mkt_state__volatility_of_volatility_48__h0=0.3,
        mkt_state__breakout_efficiency_4h__h0=np.nan,
    )
    core, state = _compact_feature_columns(frame)
    assert set(core) == {"existing_alpha_ev", "catboost_p_0"}
    assert set(state) == {
        "mkt_state__volatility_of_volatility_48__h0",
        "mkt_state__breakout_efficiency_4h__h0",
    }


def test_hierarchical_calibration_uses_disjoint_prior_oof_side_and_anchor_rows() -> None:
    rows = 60
    ts = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "execution_decision_utc": ts,
            "execution_label_end_utc": ts + pd.Timedelta(hours=1),
            "side_name": ["long", "short"] * (rows // 2),
        }
    )
    raw = np.linspace(-1.0, 1.0, rows)
    target = raw * 0.01 + np.where(frame.side_name.eq("long"), 0.001, -0.001)
    fold = np.repeat(np.arange(3, dtype=float), 20)
    mapped, audit = temporal_hierarchical_oof_calibration(
        frame,
        raw,
        target,
        fold,
        decision_col="execution_decision_utc",
        resolution_col="execution_label_end_utc",
        side_col="side_name",
        min_rows=2,
    )
    assert np.isfinite(mapped).all()
    assert audit[0]["status"].startswith("identity_")
    final = audit[-1]
    assert pd.Timestamp(final["side_fit_max_resolution_utc"]) < pd.Timestamp(
        final["anchor_start_utc"]
    )
    assert pd.Timestamp(final["pooled_anchor_max_resolution_utc"]) < pd.Timestamp(
        final["validation_start_utc"]
    )


def test_multitask_blend_falls_back_to_direct_then_uses_prior_outer_oof() -> None:
    rows = 60
    frame = pd.DataFrame(
        {
            "execution_decision_utc": pd.date_range(
                "2026-01-01", periods=rows, freq="h", tz="UTC"
            ),
            "side_name": ["long", "short"] * (rows // 2),
        }
    )
    frame["execution_label_end_utc"] = (
        frame["execution_decision_utc"] + pd.Timedelta(hours=1)
    )
    base = np.linspace(-0.02, 0.02, rows)
    predictions = {
        name: base + index * 1e-4
        for index, name in enumerate(MULTITASK_FEATURES)
    }
    predictions[DIRECT] = base.copy()
    target = 0.8 * base + 0.002
    fold = np.repeat(np.arange(3, dtype=float), 20)
    blend, audit = temporal_multitask_oof_blend(
        frame,
        predictions,
        target,
        fold,
        decision_col="execution_decision_utc",
        resolution_col="execution_label_end_utc",
        min_rows=5,
    )
    np.testing.assert_allclose(blend[:20], base[:20])
    assert audit[0]["status"].startswith("direct_primary_fallback")
    assert audit[-1]["status"] == "pooled_ridge_on_prior_outer_oof_heads"
    assert audit[-1]["reference_oof_rows"] > 0


def test_shared_multitask_meta_keeps_direct_fallback_then_uses_prior_oof() -> None:
    rows = 90
    frame = pd.DataFrame(
        {
            "execution_decision_utc": pd.date_range(
                "2026-01-01", periods=rows, freq="h", tz="UTC"
            ),
            "side_name": ["long", "short"] * (rows // 2),
        }
    )
    frame["execution_label_end_utc"] = (
        frame["execution_decision_utc"] + pd.Timedelta(hours=1)
    )
    base = np.linspace(-0.02, 0.02, rows)
    predictions = {
        name: base + index * 1e-4
        for index, name in enumerate(MULTITASK_FEATURES)
    }
    predictions[DIRECT] = base.copy()
    target = 0.8 * base + 0.002
    fold = np.repeat(np.arange(3, dtype=float), 30)
    shared, audit = temporal_shared_multitask_oof_meta(
        frame,
        predictions,
        target,
        clean_event=target > 0.0,
        severe_floor=np.full(rows, 0.01),
        fold_id=fold,
        decision_col="execution_decision_utc",
        resolution_col="execution_label_end_utc",
        side_col="side_name",
        min_rows=5,
        random_state=1,
    )
    np.testing.assert_allclose(shared[:30], base[:30])
    assert audit[0]["status"].startswith("direct_primary_fallback")
    assert audit[-1]["status"] == (
        "shared_trunk_prior_outer_oof_direct_primary"
    )
    assert audit[-1]["reference_oof_rows"] > 0
