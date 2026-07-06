import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_candidate_source_tags import (
    ARCHETYPE_COLS,
    COMPONENT_COLS,
    SOURCE_KEY_COL,
    SOURCE_ROW_IDX_COL,
    TAG_COLS,
    add_source_identity,
    build_row_alignment_audit,
    build_source_archetype_promotion_scorecard,
    build_source_archetype_walkforward_readiness,
    build_outcome_frame,
    evaluate_capture_utility_gap,
    evaluate_failure_modes,
    evaluate_opportunity_capture,
    evaluate_proxy_learnability,
    evaluate_source_score_target_diagnostics,
    build_feature_registry,
    build_quality_label_candidates,
    join_predictions,
    materialize_source_tags,
    prior_recent_source_strength,
)


def _config():
    return {
        "timestamp_col": "__ts__",
        "symbol_col": "__symbol__",
        "regime_head_columns": ["regime"],
        "proxy_score_columns": ["proxy_score"],
        "explicit_causal_whitelist": [],
        "allowed_causal_feature_groups": {
            "trend_path_features": ["trend_strength", "path_strength", "future_trend_leak"],
            "shock_impulse_features": ["speed", "shock_12h", "breakout_24h", "range_24h_pct"],
            "liquidity_execution_features": ["spread_bps", "depth_score", "rejection_proxy"],
            "oi_agreement_features": ["oi_delta", "oi_trend_agreement"],
            "location_features": ["location_quality", "overextension_atr"],
            "pullback_retest_features": ["pullback_depth_atr", "retest_score"],
            "compression_features": ["atr_compression"],
            "volume_confirmation_features": ["volume_z", "turnover_z"],
            "barrier_pressure_features": ["barrier_pressure"],
        },
        "lower_is_better_features": ["spread_bps", "overextension_atr", "pullback_depth_atr"],
        "high_risk_features": ["rejection_proxy", "overextension_atr"],
        "outcome_columns": {
            "realized_net_utility": ["__u_policy_net__"],
            "adverse_mae": ["__mae_ret__"],
            "favorable_mfe": ["__mfe_ret__"],
            "barrier_width": ["__barrier_pct__"],
            "timeout_flag": ["__is_timeout__"],
        },
        "tag_thresholds": {
            "min_timestamp_rows": 2,
            "quiet_continuation": 0.5,
            "loud_breakout_impulse": 0.5,
            "dirty_shock_avoid": 0.5,
            "retest_reversal": 0.5,
            "compression_release": 0.5,
            "run_entry": 0.5,
            "late_run_continuation": 0.5,
            "run_prior_hours": 3.0,
            "run_prior_rows": 3,
            "run_prior_low_threshold": 0.35,
            "run_prior_high_threshold": 0.35,
        },
    }


def _sample_frame():
    ts = pd.date_range("2026-01-01", periods=8, freq="h")
    rows = []
    for i, stamp in enumerate(ts):
        for symbol_idx, symbol in enumerate(["AAA/USD:USD", "BBB/USD:USD"]):
            base = float(i + 1 + symbol_idx)
            rows.append(
                {
                    "__ts__": stamp,
                    "__symbol__": symbol,
                    "regime": "r0" if i < 4 else "r1",
                    "trend_strength": base,
                    "path_strength": base * 0.7,
                    "future_trend_leak": 9999.0 - base,
                    "speed": base if symbol_idx else 0.2 * base,
                    "shock_12h": base * 0.3,
                    "breakout_24h": base * 0.4,
                    "range_24h_pct": base * 0.2,
                    "spread_bps": 20.0 - base,
                    "depth_score": base,
                    "rejection_proxy": 0.1 * base,
                    "oi_delta": base * 0.2,
                    "oi_trend_agreement": base * 0.3,
                    "location_quality": base * 0.4,
                    "overextension_atr": 10.0 - base,
                    "pullback_depth_atr": 5.0 - 0.2 * base,
                    "retest_score": base * 0.2,
                    "atr_compression": 1.0 / base,
                    "volume_z": base * 0.5,
                    "turnover_z": base * 0.6,
                    "barrier_pressure": 0.1 * base,
                    "proxy_score": base,
                    "__u_policy_net__": (-1.0) ** i * 0.01 * base,
                    "__mae_ret__": -0.001 * base,
                    "__mfe_ret__": 0.006 * base,
                    "__barrier_pct__": 0.01,
                    "__is_timeout__": float(i % 3 == 0),
                }
            )
    return pd.DataFrame(rows)


def _source_signature(frame):
    cols = [
        col
        for col in frame.columns
        if col in COMPONENT_COLS + ARCHETYPE_COLS or col in TAG_COLS or col in {"primary_source_tag"}
    ]
    return frame[cols].reset_index(drop=True)


def test_source_tags_do_not_change_when_outcomes_change_or_drop():
    cfg = _config()
    frame = _sample_frame()
    base, _ = materialize_source_tags(frame, cfg)
    shuffled = frame.copy()
    shuffled["__u_policy_net__"] = np.random.default_rng(42).permutation(shuffled["__u_policy_net__"])
    shuffled["__mae_ret__"] = np.random.default_rng(43).normal(size=len(shuffled))
    shuffled["__is_timeout__"] = np.random.default_rng(44).integers(0, 2, size=len(shuffled))
    changed, _ = materialize_source_tags(shuffled, cfg)
    dropped, _ = materialize_source_tags(
        frame.drop(columns=["__u_policy_net__", "__mae_ret__", "__barrier_pct__", "__is_timeout__"]),
        cfg,
    )

    pdt.assert_frame_equal(_source_signature(base), _source_signature(changed), check_dtype=False)
    pdt.assert_frame_equal(_source_signature(base), _source_signature(dropped), check_dtype=False)


def test_v7_capture_and_location_risk_archetypes_are_materialized():
    cfg = _config()
    cfg["tag_thresholds"]["compression_capture_candidate"] = 0.5
    cfg["tag_thresholds"]["risk_adjusted_capture_candidate"] = 0.5
    cfg["tag_thresholds"]["clean_economic_capture_candidate"] = 0.5
    cfg["tag_thresholds"]["misleading_location_risk"] = 0.5
    frame = _sample_frame()
    source, _ = materialize_source_tags(frame, cfg)

    assert "compression_capture_candidate_score" in source.columns
    assert "risk_adjusted_capture_candidate_score" in source.columns
    assert "clean_economic_capture_candidate_score" in source.columns
    assert "misleading_location_risk_score" in source.columns
    assert "tag_compression_capture_candidate" in source.columns
    assert "tag_risk_adjusted_capture_candidate" in source.columns
    assert "tag_clean_economic_capture_candidate" in source.columns
    assert "tag_misleading_location_risk" in source.columns
    assert source["compression_capture_candidate_score"].between(0.0, 1.0).all()
    assert source["risk_adjusted_capture_candidate_score"].between(0.0, 1.0).all()
    assert source["clean_economic_capture_candidate_score"].between(0.0, 1.0).all()
    assert source["misleading_location_risk_score"].between(0.0, 1.0).all()


def test_source_tags_for_past_rows_do_not_change_after_future_append():
    cfg = _config()
    frame = _sample_frame()
    base, _ = materialize_source_tags(frame, cfg)
    future = frame.copy()
    future["__ts__"] = pd.to_datetime(future["__ts__"]) + pd.Timedelta(days=30)
    appended, _ = materialize_source_tags(pd.concat([frame, future], ignore_index=True), cfg)

    pdt.assert_frame_equal(
        _source_signature(base),
        _source_signature(appended.iloc[: len(frame)]),
        check_dtype=False,
    )


def test_run_entry_history_uses_only_prior_same_symbol_rows():
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-01-01", periods=4, freq="h"),
            "__symbol__": ["AAA"] * 4,
        }
    )
    score = pd.Series([0.9, 0.8, 0.1, 0.7], dtype=float)
    prior = prior_recent_source_strength(
        frame,
        score,
        symbol_col="__symbol__",
        timestamp_col="__ts__",
        hours=2.0,
        rows=2,
    )

    assert prior.iloc[0] == 0.0
    assert prior.iloc[1] == 0.9
    assert prior.iloc[2] == 0.9
    assert prior.iloc[3] == 0.8


def test_registry_excludes_outcome_like_columns_from_source_scores():
    cfg = _config()
    frame = _sample_frame()
    registry = build_feature_registry(frame, cfg)

    assert "future_trend_leak" not in registry["available"]["trend_path_features"]
    assert "future_trend_leak" in registry["excluded_outcome_like"]


def test_explicit_causal_whitelist_allows_trailing_realized_vol_feature():
    cfg = _config()
    cfg["explicit_causal_whitelist"] = ["realized_volatility_24h"]
    cfg["allowed_causal_feature_groups"]["compression_features"].append("realized_volatility_24h")
    frame = _sample_frame()
    frame["realized_volatility_24h"] = np.linspace(0.1, 0.9, len(frame))
    registry = build_feature_registry(frame, cfg)

    assert "realized_volatility_24h" in registry["available"]["compression_features"]
    assert "realized_volatility_24h" not in registry["excluded_outcome_like"]


def test_quality_label_candidates_include_source_conditioned_variants():
    cfg = _config()
    cfg["label_thresholds"] = {
        "good_top_frac": 0.30,
        "bad_bottom_frac": 0.40,
        "neutral_weight": 0.0,
        "source_conditioned_min_timestamp_rows": 1,
        "source_conditioned_min_prior_rows": 1,
        "source_conditioned_global_min_prior_rows": 1,
    }
    frame = _sample_frame()
    source, _ = materialize_source_tags(frame, cfg)
    outcomes, _ = build_outcome_frame(frame, source, cfg)
    labels, report = build_quality_label_candidates(source, outcomes, cfg)

    assert "quality_label_source_rank_v1" in labels.columns
    assert "quality_label_source_wf_v1" in labels.columns
    assert "quality_label_clean_path_v2" in labels.columns
    assert "quality_label_recoverable_opportunity_v2" in labels.columns
    assert "quality_label_opportunity_capture_v3" in labels.columns
    assert "quality_label_economic_capture_v4" in labels.columns
    assert "sample_weight_source_wf_v1" in labels.columns
    assert "sample_weight_opportunity_v2" in labels.columns
    assert "sample_weight_capture_v3" in labels.columns
    assert "sample_weight_economic_capture_v4" in labels.columns
    assert "train_include_calm_positive_score_top10_v1" in labels.columns
    assert "train_include_compression_capture_candidate_v3" in labels.columns
    assert "train_include_compression_capture_score_top10_v3" in labels.columns
    assert "train_include_risk_adjusted_capture_candidate_v4" in labels.columns
    assert "train_include_risk_adjusted_capture_score_top10_v4" in labels.columns
    assert "train_include_economic_capture_non_neutral_v4" in labels.columns
    assert "train_include_compression_economic_capture_score_top10_v4" in labels.columns
    assert "train_include_risk_adjusted_economic_capture_score_top10_v4" in labels.columns
    assert "train_include_clean_economic_capture_candidate_v5" in labels.columns
    assert "train_include_clean_economic_capture_score_top10_v5" in labels.columns
    assert "train_include_misleading_location_risk_excluded_v3" in labels.columns
    assert "train_include_misleading_location_risk_bottom70_v3" in labels.columns
    assert "source_rank_label_counts" in report
    assert "source_wf_label_counts" in report
    assert "recoverable_opportunity_label_counts" in report
    assert "opportunity_capture_label_counts" in report
    assert "economic_capture_label_counts" in report
    assert "failure_mode_counts" in report
    assert labels["quality_label_source_rank_v1"].isin([-1, 0, 1]).all()
    assert labels["quality_label_source_wf_v1"].isin([-1, 0, 1]).all()
    assert labels["quality_label_recoverable_opportunity_v2"].isin([-1, 0, 1]).all()
    assert labels["quality_label_opportunity_capture_v3"].isin([-1, 0, 1]).all()
    assert labels["quality_label_economic_capture_v4"].isin([-1, 0, 1]).all()


def test_quality_label_candidates_preserve_source_id_regime_and_proxy_columns():
    cfg = _config()
    frame = add_source_identity(_sample_frame(), cfg)
    source, _ = materialize_source_tags(frame, cfg)
    outcomes, _ = build_outcome_frame(frame, source, cfg)
    labels, report = build_quality_label_candidates(source, outcomes, cfg)

    for col in [SOURCE_ROW_IDX_COL, SOURCE_KEY_COL, "__ts__", "__symbol__", "regime", "proxy_score"]:
        assert col in source.columns
        assert col in labels.columns
    assert len(labels) == len(source)
    assert report["quality_label_rows"] == len(source)
    assert set([SOURCE_ROW_IDX_COL, SOURCE_KEY_COL, "regime", "proxy_score"]).issubset(
        set(report["metadata_columns_preserved"])
    )


def test_realized_outcome_columns_cannot_be_used_as_proxy_scores():
    cfg = _config()
    frame = add_source_identity(_sample_frame(), cfg)

    with pytest.raises(ValueError, match="Proxy/model score columns"):
        join_predictions(frame, None, cfg, proxy_score_cols=["__u_policy_net__"])


def test_expected_utility_proxy_score_is_allowed_when_explicit():
    cfg = _config()
    frame = add_source_identity(_sample_frame(), cfg)
    frame["expected_utility"] = np.linspace(0.0, 1.0, len(frame))
    joined, report = join_predictions(frame, None, cfg, proxy_score_cols=["expected_utility"])

    assert "expected_utility" in joined.columns
    assert "expected_utility" in report["proxy_score_columns"]
    assert report["prediction_match_rate"] == 1.0


def test_prediction_join_reports_duplicate_keys_and_low_match_rate(tmp_path):
    cfg = _config()
    frame = add_source_identity(_sample_frame(), cfg)
    prediction_rows = frame.loc[:0, ["__ts__", "__symbol__"]].copy()
    prediction_rows["pred_score"] = [0.9]
    prediction_rows = pd.concat([prediction_rows, prediction_rows], ignore_index=True)
    pred_path = tmp_path / "predictions.csv"
    prediction_rows.to_csv(pred_path, index=False)

    joined, report = join_predictions(
        frame,
        pred_path,
        cfg,
        prediction_key_cols=["__ts__", "__symbol__"],
        proxy_score_cols=["pred_score"],
    )

    assert "pred_score" in joined.columns
    assert report["prediction_duplicate_keys"] == 2
    assert report["rows_with_multiple_predictions_joined"] == 1
    assert report["prediction_match_rate"] < 0.80
    assert report["alignment_status"] == "fail"


def test_source_tags_unchanged_when_proxy_or_regime_columns_are_shuffled():
    cfg = _config()
    frame = add_source_identity(_sample_frame(), cfg)
    base, _ = materialize_source_tags(frame, cfg)
    shuffled = frame.copy()
    rng = np.random.default_rng(123)
    shuffled["proxy_score"] = rng.permutation(shuffled["proxy_score"].to_numpy(copy=True))
    shuffled["regime"] = rng.permutation(shuffled["regime"].to_numpy(copy=True))
    changed, _ = materialize_source_tags(shuffled, cfg)

    pdt.assert_frame_equal(_source_signature(base), _source_signature(changed), check_dtype=False)


def test_row_alignment_audit_reports_pass_and_preserved_metadata():
    cfg = _config()
    frame = add_source_identity(_sample_frame(), cfg)
    source, _ = materialize_source_tags(frame, cfg)
    outcomes, _ = build_outcome_frame(frame, source, cfg)
    labels, _ = build_quality_label_candidates(source, outcomes, cfg)

    audit, report = build_row_alignment_audit(
        frame=frame,
        source=source,
        label_candidates=labels,
        outcomes=outcomes,
        config=cfg,
        label_join_report={
            "label_duplicate_keys": 0,
            "rows_with_multiple_outcomes_joined": 0,
        },
        prediction_report={
            "prediction_rows": 0,
            "prediction_match_rate": 1.0,
            "prediction_duplicate_keys": 0,
            "rows_with_multiple_predictions_joined": 0,
            "alignment_status": "pass",
        },
    )

    assert audit.iloc[0]["alignment_quality"] == "pass"
    assert report["metadata_columns_preserved"] == 1
    assert report["quality_label_candidates_rows"] == len(frame)


def test_proxy_learnability_reports_source_and_regime_metrics():
    cfg = _config()
    cfg["diagnostics"] = {"proxy_top_fracs": [0.5]}
    frame = add_source_identity(_sample_frame(), cfg)
    source, _ = materialize_source_tags(frame, cfg)
    outcomes, _ = build_outcome_frame(frame, source, cfg)
    labels, _ = build_quality_label_candidates(source, outcomes, cfg)
    full = pd.concat([source, outcomes, labels[["quality_label_v0", "quality_label_economic_capture_v4"]]], axis=1)
    overall, by_month, by_week, source_x_regime, report = evaluate_proxy_learnability(full, cfg)

    assert report["proxy_available"] is True
    assert not overall.empty
    assert not by_month.empty
    assert not by_week.empty
    assert not source_x_regime.empty
    assert "proxy_ic_spearman" in overall.columns
    assert "proxy_topk_economic_capture_good_rate" in overall.columns
    assert "source_lift_vs_overall" in source_x_regime.columns
    lift_row = source_x_regime[source_x_regime["source_tag"].eq("run_entry")].iloc[0]
    regime_value = lift_row["regime_value"]
    regime_group = full[full["regime"].astype(str).eq(str(regime_value))]
    expected_concentration = float(regime_group["tag_run_entry"].astype(bool).mean())
    expected_lift = expected_concentration / float(full["tag_run_entry"].astype(bool).mean())
    assert np.isclose(lift_row["source_lift_vs_overall"], expected_lift)


def test_quality_score_is_missing_when_realized_utility_is_missing():
    cfg = _config()
    cfg["label_thresholds"] = {
        "source_conditioned_min_prior_rows": 1,
        "source_conditioned_global_min_prior_rows": 1,
    }
    frame = _sample_frame()
    frame.loc[0, "__u_policy_net__"] = np.nan
    source, _ = materialize_source_tags(frame, cfg)
    outcomes, _ = build_outcome_frame(frame, source, cfg)
    labels, _ = build_quality_label_candidates(source, outcomes, cfg)

    assert pd.isna(labels.loc[0, "realized_quality_score_v0"])
    assert labels.loc[0, "quality_label_v0"] == -1
    assert labels.loc[0, "quality_label_source_wf_v1"] == -1


def test_failure_mode_summary_reports_source_and_month_rows():
    cfg = _config()
    frame = _sample_frame()
    source, _ = materialize_source_tags(frame, cfg)
    outcomes, _ = build_outcome_frame(frame, source, cfg)
    full = pd.concat([source, outcomes], axis=1)
    by_source, by_month, by_source_month = evaluate_failure_modes(full, cfg)

    assert outcomes["mfe_mae_recovery_ratio"].dropna().max() <= 20.0
    assert not by_source.empty
    assert not by_month.empty
    assert not by_source_month.empty
    assert "recoverable_opportunity_rate" in by_source.columns
    assert "path_failure_rate" in by_month.columns


def test_opportunity_capture_summary_reports_capture_rates():
    cfg = _config()
    frame = _sample_frame()
    source, _ = materialize_source_tags(frame, cfg)
    outcomes, _ = build_outcome_frame(frame, source, cfg)
    full = pd.concat([source, outcomes], axis=1)
    by_source, by_month, by_source_month = evaluate_opportunity_capture(full, cfg)

    assert not by_source.empty
    assert not by_month.empty
    assert not by_source_month.empty
    assert "capture_rate" in by_source.columns
    assert "economic_capture_rate" in by_source.columns
    assert "expensive_capture_rate" in by_source.columns
    assert "capture_loss_rate" in by_month.columns


def test_capture_utility_gap_reports_economic_capture_split():
    cfg = _config()
    cfg["diagnostics"] = {"source_score_top_fracs": [0.5]}
    frame = _sample_frame()
    source, _ = materialize_source_tags(frame, cfg)
    outcomes, _ = build_outcome_frame(frame, source, cfg)
    full = pd.concat([source, outcomes], axis=1)
    overall, by_month = evaluate_capture_utility_gap(full, cfg)

    assert not overall.empty
    assert not by_month.empty
    assert "selected_economic_capture_rate" in overall.columns
    assert "selected_expensive_capture_rate" in overall.columns
    assert "economic_capture_among_opportunity_rate" in overall.columns


def test_source_score_target_diagnostics_report_ic_and_top_slice_delta():
    cfg = _config()
    cfg["diagnostics"] = {"source_score_top_fracs": [0.5]}
    frame = _sample_frame()
    source, _ = materialize_source_tags(frame, cfg)
    outcomes, _ = build_outcome_frame(frame, source, cfg)
    full = pd.concat([source, outcomes], axis=1)
    overall, by_month = evaluate_source_score_target_diagnostics(full, cfg)

    assert not overall.empty
    assert not by_month.empty
    assert "score_ic_target" in overall.columns
    assert "selected_target_delta" in overall.columns
    assert "outcome_recoverable_opportunity_flag" in set(overall["target_col"])


def test_source_archetype_promotion_scorecard_classifies_promotion_and_risk():
    quality = pd.DataFrame(
        [
            {
                "scope": "overall",
                "bucket": "all",
                "score_col": "capture_score",
                "top_frac": 0.10,
                "mean_net_utility": 0.001,
                "score_ic_utility": 0.01,
                "bad_mae_rate": 0.20,
                "p90_mae": 0.70,
                "timeout_rate": 0.10,
            },
            {
                "scope": "overall",
                "bucket": "all",
                "score_col": "risk_score",
                "top_frac": 0.10,
                "mean_net_utility": -0.002,
                "score_ic_utility": -0.01,
                "bad_mae_rate": 0.55,
                "p90_mae": 1.30,
                "timeout_rate": 0.20,
            },
            {
                "scope": "overall",
                "bucket": "all",
                "score_col": "unstable_score",
                "top_frac": 0.10,
                "mean_net_utility": -0.001,
                "score_ic_utility": -0.01,
                "bad_mae_rate": 0.30,
                "p90_mae": 0.90,
                "timeout_rate": 0.15,
            },
        ]
    )
    target_rows = [
        ("capture_score", "outcome_opportunity_captured_flag", 0.05, 0.04, 0.60),
        ("capture_score", "outcome_recoverable_opportunity_flag", 0.06, 0.05, np.nan),
        ("capture_score", "opportunity_capture_efficiency", 0.04, 0.03, np.nan),
        ("capture_score", "outcome_path_failure_flag", -0.03, -0.02, np.nan),
        ("capture_score", "outcome_no_edge_flag", 0.00, 0.00, np.nan),
        ("capture_score", "outcome_opportunity_capture_loss_flag", -0.02, -0.01, np.nan),
        ("risk_score", "outcome_opportunity_captured_flag", 0.00, 0.00, 0.30),
        ("risk_score", "outcome_recoverable_opportunity_flag", 0.00, 0.00, np.nan),
        ("risk_score", "opportunity_capture_efficiency", -0.03, -0.02, np.nan),
        ("risk_score", "outcome_path_failure_flag", 0.08, 0.06, np.nan),
        ("risk_score", "outcome_no_edge_flag", 0.05, 0.04, np.nan),
        ("risk_score", "outcome_opportunity_capture_loss_flag", 0.04, 0.03, np.nan),
        ("unstable_score", "outcome_opportunity_captured_flag", 0.05, 0.04, 0.50),
        ("unstable_score", "outcome_recoverable_opportunity_flag", 0.02, 0.02, np.nan),
        ("unstable_score", "opportunity_capture_efficiency", 0.01, 0.01, np.nan),
        ("unstable_score", "outcome_path_failure_flag", 0.00, 0.00, np.nan),
        ("unstable_score", "outcome_no_edge_flag", 0.00, 0.00, np.nan),
        ("unstable_score", "outcome_opportunity_capture_loss_flag", 0.00, 0.00, np.nan),
    ]
    target = pd.DataFrame(
        [
            {
                "scope": "overall",
                "bucket": "all",
                "score_col": score_col,
                "target_col": target_col,
                "top_frac": 0.10,
                "selected_target_delta": delta,
                "score_ic_target": ic,
                "selected_capture_among_opportunity_rate": capture_among_opp,
            }
            for score_col, target_col, delta, ic, capture_among_opp in target_rows
        ]
    )
    quality_by_month = pd.DataFrame(
        [
            {
                "scope": "month",
                "bucket": month,
                "score_col": score_col,
                "top_frac": 0.10,
                "mean_net_utility": utility,
            }
            for score_col, utilities in {
                "capture_score": [0.01, 0.02, -0.01, -0.02],
                "risk_score": [-0.01, -0.02, -0.03, -0.04],
                "unstable_score": [0.01, -0.02, -0.03, -0.04],
            }.items()
            for month, utility in zip(["2026-03", "2026-04", "2026-05", "2026-06"], utilities)
        ]
    )
    monthly_target_rows = []
    monthly_deltas = {
        "capture_score": {
            "outcome_opportunity_captured_flag": [0.04, 0.03, 0.05, 0.02],
            "outcome_recoverable_opportunity_flag": [0.04, 0.05, 0.04, 0.03],
            "opportunity_capture_efficiency": [0.02, 0.03, 0.02, 0.01],
            "outcome_path_failure_flag": [-0.03, -0.02, -0.01, 0.00],
            "outcome_no_edge_flag": [-0.01, -0.02, 0.00, -0.03],
            "outcome_opportunity_capture_loss_flag": [-0.01, -0.02, -0.01, 0.00],
        },
        "risk_score": {
            "outcome_opportunity_captured_flag": [0.00, 0.00, 0.00, 0.00],
            "outcome_recoverable_opportunity_flag": [0.00, 0.00, 0.00, 0.00],
            "opportunity_capture_efficiency": [-0.02, -0.01, -0.03, -0.01],
            "outcome_path_failure_flag": [0.05, 0.06, 0.07, 0.08],
            "outcome_no_edge_flag": [0.01, 0.02, 0.01, 0.03],
            "outcome_opportunity_capture_loss_flag": [0.02, 0.03, 0.01, 0.02],
        },
        "unstable_score": {
            "outcome_opportunity_captured_flag": [0.04, -0.01, -0.02, -0.03],
            "outcome_recoverable_opportunity_flag": [0.01, -0.01, 0.00, -0.02],
            "opportunity_capture_efficiency": [0.01, -0.01, -0.02, -0.03],
            "outcome_path_failure_flag": [0.00, 0.01, 0.02, 0.03],
            "outcome_no_edge_flag": [0.00, 0.01, 0.01, 0.02],
            "outcome_opportunity_capture_loss_flag": [0.00, 0.01, 0.01, 0.02],
        },
    }
    for score_col, target_map in monthly_deltas.items():
        for target_col, deltas in target_map.items():
            for month, delta in zip(["2026-03", "2026-04", "2026-05", "2026-06"], deltas):
                monthly_target_rows.append(
                    {
                        "scope": "month",
                        "bucket": month,
                        "score_col": score_col,
                        "target_col": target_col,
                        "top_frac": 0.10,
                        "selected_target_delta": delta,
                        "score_ic_target": delta,
                        "selected_capture_among_opportunity_rate": np.nan,
                    }
                )
    target_by_month = pd.DataFrame(monthly_target_rows)
    scorecard = build_source_archetype_promotion_scorecard(
        source_score_target_overall=target,
        source_score_quality_overall=quality,
        source_score_target_by_month=target_by_month,
        source_score_quality_by_month=quality_by_month,
        config={"diagnostics": {"promotion_scorecard_top_frac": 0.10}},
    )

    buckets = dict(zip(scorecard["score_col"], scorecard["promotion_bucket"]))
    actions = dict(zip(scorecard["score_col"], scorecard["training_action"]))
    stability = dict(zip(scorecard["score_col"], scorecard["stability_bucket"]))
    confidence = dict(zip(scorecard["score_col"], scorecard["promotion_confidence"]))
    assert buckets["capture_score"] == "safer_opportunity"
    assert actions["capture_score"] == "promote_as_clean_opportunity_filter_ablation"
    assert stability["capture_score"] == "stable_promoted_signal"
    assert confidence["capture_score"] == "high"
    assert buckets["risk_score"] == "risk_avoidance"
    assert actions["risk_score"] == "use_as_exclusion_or_downweight_flag"
    assert stability["risk_score"] == "stable_risk_signal"
    assert confidence["risk_score"] == "high"
    assert buckets["unstable_score"] == "capture_maximizer"
    assert stability["unstable_score"] == "unstable_promoted_signal"
    assert confidence["unstable_score"] == "low"


def test_source_archetype_walkforward_readiness_uses_prior_months_only():
    months = ["2026-01", "2026-02", "2026-03"]
    quality_by_month = pd.DataFrame(
        [
            {
                "scope": "month",
                "bucket": month,
                "score_col": score_col,
                "top_frac": 0.10,
                "mean_net_utility": utility,
                "score_ic_utility": 0.01,
            }
            for score_col, utilities in {
                "capture_score": [0.01, 0.02, -0.01],
                "risk_score": [-0.01, -0.02, -0.03],
                "fail_score": [0.01, 0.01, -0.02],
            }.items()
            for month, utility in zip(months, utilities)
        ]
    )
    monthly_deltas = {
        "capture_score": {
            "outcome_opportunity_captured_flag": [0.05, 0.04, 0.03],
            "outcome_recoverable_opportunity_flag": [0.06, 0.05, 0.02],
            "opportunity_capture_efficiency": [0.04, 0.03, 0.02],
            "outcome_path_failure_flag": [-0.03, -0.02, -0.01],
            "outcome_no_edge_flag": [-0.01, -0.02, 0.00],
            "outcome_opportunity_capture_loss_flag": [-0.01, -0.02, 0.00],
        },
        "risk_score": {
            "outcome_opportunity_captured_flag": [0.00, 0.00, 0.00],
            "outcome_recoverable_opportunity_flag": [0.00, 0.00, 0.00],
            "opportunity_capture_efficiency": [-0.01, -0.02, -0.01],
            "outcome_path_failure_flag": [0.06, 0.07, 0.08],
            "outcome_no_edge_flag": [0.01, 0.02, 0.03],
            "outcome_opportunity_capture_loss_flag": [0.02, 0.03, 0.04],
        },
        "fail_score": {
            "outcome_opportunity_captured_flag": [0.05, 0.04, -0.02],
            "outcome_recoverable_opportunity_flag": [0.06, 0.05, -0.01],
            "opportunity_capture_efficiency": [0.04, 0.03, -0.01],
            "outcome_path_failure_flag": [-0.03, -0.02, 0.04],
            "outcome_no_edge_flag": [-0.01, -0.02, 0.02],
            "outcome_opportunity_capture_loss_flag": [-0.01, -0.02, 0.03],
        },
    }
    rows = []
    for score_col, target_map in monthly_deltas.items():
        for target_col, values in target_map.items():
            for month, value in zip(months, values):
                rows.append(
                    {
                        "scope": "month",
                        "bucket": month,
                        "score_col": score_col,
                        "target_col": target_col,
                        "top_frac": 0.10,
                        "selected_target_delta": value,
                        "score_ic_target": value,
                    }
                )
    target_by_month = pd.DataFrame(rows)
    readiness = build_source_archetype_walkforward_readiness(
        source_score_target_by_month=target_by_month,
        source_score_quality_by_month=quality_by_month,
        config={
            "diagnostics": {
                "promotion_scorecard_top_frac": 0.10,
                "promotion_walkforward_min_history_months": 2,
            }
        },
    )

    assert set(readiness["eval_month"]) == {"2026-03"}
    by_score = readiness.set_index("score_col")
    assert by_score.loc["capture_score", "history_promotion_bucket"] == "safer_opportunity"
    assert bool(by_score.loc["capture_score", "eval_signal_success"]) is True
    assert by_score.loc["risk_score", "history_promotion_bucket"] == "risk_avoidance"
    assert bool(by_score.loc["risk_score", "eval_signal_success"]) is True
    assert by_score.loc["fail_score", "history_promotion_bucket"] == "safer_opportunity"
    assert bool(by_score.loc["fail_score", "eval_signal_success"]) is False
