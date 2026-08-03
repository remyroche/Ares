from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_execution_ev_calibration_drift_uncertainty_ablation import (
    CalibrationArm,
    load_panel,
    score_scale_coverage_audit,
    global_top_k_metrics,
    strict_prior_oof_drift_heads,
    temporal_hierarchical_calibration,
)


def _frame(rows: int = 120) -> pd.DataFrame:
    timestamp = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "execution_decision_utc": timestamp,
            "execution_label_end_utc": timestamp + pd.Timedelta(hours=1),
            "side_name": ["long", "short"] * (rows // 2),
            "feature": np.linspace(-1.0, 1.0, rows),
        }
    )


def test_hierarchical_calibration_has_identity_first_fold_and_disjoint_anchor() -> None:
    frame = _frame()
    direct = np.linspace(-0.02, 0.02, len(frame))
    target = 0.7 * direct + np.where(frame.side_name.eq("long"), 0.001, -0.001)
    fold = np.repeat(np.arange(3, dtype=float), 40)
    arm = CalibrationArm("test", 21, 7.0, 0.75, 0.25)
    mapped, audit = temporal_hierarchical_calibration(frame, direct, target, fold, arm, min_rows=5)
    np.testing.assert_allclose(mapped[:40], direct[:40])
    final = audit[-1]
    assert pd.Timestamp(final["side_fit_max_resolution_utc"]) < pd.Timestamp(final["anchor_start_utc"])
    assert pd.Timestamp(final["anchor_max_resolution_utc"]) < pd.Timestamp(final["validation_start_utc"])


def test_drift_heads_never_use_current_or_future_fold_outcomes() -> None:
    frame = _frame()
    score = np.linspace(-0.01, 0.02, len(frame))
    target = 0.6 * score
    fold = np.repeat(np.arange(3, dtype=float), 40)
    heads, audit = strict_prior_oof_drift_heads(frame, ["feature"], score, target, fold, min_rows=8, seed=7)
    assert all(np.allclose(values[:40], 0.0) for values in heads.values())
    final = audit[-1]
    for side in ("long", "short"):
        assert pd.Timestamp(final["per_side"][side]["max_resolution_utc"]) < pd.Timestamp(final["validation_start_utc"])


def test_global_top_k_is_one_cross_side_tail_after_eligibility() -> None:
    score = np.array([0.9, 0.8, 0.2, 0.1])
    target = np.array([0.01, -0.01, 0.03, 0.04])
    side = np.array(["long", "short", "long", "short"])
    result = global_top_k_metrics(score, target, side, np.ones(4, dtype=bool), top_fraction=0.25, eligibility=np.array([False, True, True, True]))
    assert result["top_k_requested_rows"] == 1
    assert result["top_k_rows"] == 1
    assert result["top_k_mean_net_ev"] == -0.01
    assert result["ranking_scope"].startswith("one_pooled_global")


def test_score_scale_audit_keeps_fold_and_side_separate() -> None:
    frame = _frame()
    direct = np.linspace(-0.01, 0.02, len(frame))
    target = 0.5 * direct
    fold = np.repeat(np.arange(3, dtype=float), 40)
    report = score_scale_coverage_audit(frame, direct, target, fold)
    assert len(report) == 9
    assert {(row["fold"], row["side"]) for row in report} >= {(0, "long"), (2, "short"), (1, "all")}
    assert all(row["prediction_coverage"] == 1.0 for row in report)


def test_context_loader_admits_h0_state_but_rejects_later_state_horizons(tmp_path) -> None:
    frame = _frame(4).assign(
        __ts__=pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
        __symbol__="BTC/USD:USD",
        candidate_id=[f"c{index}" for index in range(4)],
        execution_net_ev_12h=0.01,
        direct_net_ev=0.01,
        oof_fold=0.0,
    )
    predictions = frame.loc[:, [
        "__ts__", "__symbol__", "side_name", "candidate_id", "execution_decision_utc",
        "execution_label_end_utc", "execution_net_ev_12h", "direct_net_ev", "oof_fold",
    ]]
    context = frame.loc[:, ["__ts__", "__symbol__", "side_name", "candidate_id"]].copy()
    context["mkt_state__causal__h0"] = 1.0
    context["mkt_state__future_transition__h3"] = 99.0
    predictions_path, context_path = tmp_path / "predictions.parquet", tmp_path / "context.parquet"
    predictions.to_parquet(predictions_path, index=False)
    context.to_parquet(context_path, index=False)
    joined, features = load_panel(predictions_path, context_path)
    assert len(joined) == 4
    assert features == ["mkt_state__causal__h0"]
