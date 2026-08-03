from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_mayjul_failure_attribution import (
    SCORES,
    _within_decision_percentile,
    build_attribution,
    june_july_slice_decomposition,
    materialize_attribution_rows,
)


def _frame(rows: int = 40) -> pd.DataFrame:
    times = pd.date_range("2026-06-01", periods=rows // 2, freq="h", tz="UTC")
    decision = np.repeat(times, 2)
    data = pd.DataFrame(
        {
            "candidate_id": [f"c{i:03d}" for i in range(rows)],
            "side_name": ["long", "short"] * (rows // 2),
            "__symbol__": [f"a{i % 5}" for i in range(rows)],
            "__ts__": decision - pd.Timedelta(hours=1),
            "execution_decision_utc": decision,
            "candidate_month": ["2026-06"] * rows,
            "candidate_day": decision.floor("D"),
            "catboost_archetype": ["fast", "slow"] * (rows // 2),
            "oof_entry_atr_fraction": np.linspace(0.003, 0.03, rows),
            "transition_context_available": [True] * rows,
            "prediction_p_adverse_0_5r_before_mfe": np.linspace(0.0, 1.0, rows),
            "prediction_p_stop_1r_before_mfe": np.linspace(0.1, 0.9, rows),
            "direct_minus_base_decision_percentile": np.linspace(-0.5, 0.5, rows),
            "execution_mfe_return_12h": np.linspace(0.0, 0.04, rows),
            "execution_gross_ev_12h": np.linspace(-0.02, 0.03, rows),
            "execution_cost_return": [0.01] * rows,
            "execution_net_ev_12h": np.linspace(-0.03, 0.02, rows),
            "meaningful_mfe_any_touch": [0] * 20 + [1] * 20,
            "meaningful_mfe_clean_first": [0] * 24 + [1] * 16,
            "adverse_first": [1] * 24 + [0] * 16,
            "path_opportunity_above_exact_cost": [0] * 10 + [1] * 30,
            "exact_net_positive": [0] * 24 + [1] * 16,
        }
    )
    for index, score in enumerate(SCORES):
        data[score] = np.arange(rows, dtype=float) + index * 0.01
        data[f"{score}__decision_percentile"] = _within_decision_percentile(
            data, score
        )
    return data


def test_within_decision_rank_is_unit_invariant_and_has_no_side_quota() -> None:
    frame = _frame()
    rank = _within_decision_percentile(frame, "score_base_alpha")
    scaled = _within_decision_percentile(
        frame.assign(other=frame["score_base_alpha"] * 1000 + 7), "other"
    )
    np.testing.assert_allclose(rank, scaled)
    # Both candidates per decision are ranked together, not one per side.
    assert set(rank.iloc[:2]) == {0.5, 1.0}


def test_attribution_axes_reconcile_to_frozen_global_book() -> None:
    frame = _frame()
    historical = pd.DataFrame()
    for feature in (
        "context__state_mean__median__atr_compression_ratio",
        "context__state_mean__median__breakout_24h",
        "context__state_mean__median__dir_path_risk_skew_2h",
        "context__state_mean__median__jump_intensity",
        "context__state_mean__median__leverage_build_score",
        "context__state_mean__median__memory_asymmetry_1ATR",
        "context__state_mean__median__spread_proxy_abs_return_bps_robust_z",
        "context__past_geometry_shift_3h",
        "context__mapping_current__above_p90_share",
    ):
        frame[feature] = np.linspace(-2, 2, len(frame))
        historical[feature] = np.linspace(-3, 3, 200)
    summary, slices, replacements, _ = build_attribution(frame, historical)
    assert len(summary) == len(SCORES)
    audit = slices.groupby(["score", "axis"]).agg(
        rows=("selected_rows", "sum"),
        book=("book_rows", "first"),
        contribution=("net_shortfall_contribution_bps", "sum"),
    )
    assert (audit["rows"] == audit["book"]).all()
    assert np.max(np.abs(audit["contribution"])) < 1e-8
    assert set(replacements["replacement_role"]) == {
        "common",
        "base_only",
        "direct_only",
    }


def test_june_july_decomposition_reconciles_composition_and_within_effects() -> None:
    summaries = pd.DataFrame(
        [
            {"candidate_month": "2026-06", "score": score, "mean_net_bps": 0.0}
            for score in SCORES
        ]
        + [
            {"candidate_month": "2026-07", "score": score, "mean_net_bps": -20.0}
            for score in SCORES
        ]
    )
    slices = []
    for score in SCORES:
        slices.extend(
            [
                {
                    "candidate_month": "2026-06",
                    "score": score,
                    "axis": "side",
                    "slice": "long",
                    "slice_share": 0.5,
                    "mean_net_bps": 10.0,
                },
                {
                    "candidate_month": "2026-06",
                    "score": score,
                    "axis": "side",
                    "slice": "short",
                    "slice_share": 0.5,
                    "mean_net_bps": -10.0,
                },
                {
                    "candidate_month": "2026-07",
                    "score": score,
                    "axis": "side",
                    "slice": "long",
                    "slice_share": 0.25,
                    "mean_net_bps": 0.0,
                },
                {
                    "candidate_month": "2026-07",
                    "score": score,
                    "axis": "side",
                    "slice": "short",
                    "slice_share": 0.75,
                    "mean_net_bps": -26.666666666666668,
                },
            ]
        )
    out = june_july_slice_decomposition(summaries, pd.DataFrame(slices))
    audit = out.groupby(["score", "axis"]).first()
    np.testing.assert_allclose(
        audit["book_delta_bps"],
        audit["axis_composition_effect_bps"]
        + audit["axis_within_slice_effect_bps"],
    )


def test_row_handoff_reproduces_one_global_top_decile_without_side_quota() -> None:
    frame = _frame()
    frame["execution_label_end_utc"] = (
        frame["execution_decision_utc"] + pd.Timedelta(hours=12)
    )
    frame["label_resolution_utc"] = frame["execution_label_end_utc"]
    frame["execution_mae_return_12h"] = 0.01
    frame["soft_label"] = 0.5
    frame["timeout"] = 0
    frame["available_at"] = frame["__ts__"]
    frame["train_decision_cutoff"] = frame["__ts__"] - pd.Timedelta(hours=1)
    frame["label_resolution_available_at"] = frame["execution_label_end_utc"]
    frame["oof_fold"] = 1
    frame["cohort_anchor_utc"] = frame["execution_decision_utc"]
    for feature in (
        "context__state_mean__median__atr_compression_ratio",
        "context__state_mean__median__breakout_24h",
        "context__state_mean__median__dir_path_risk_skew_2h",
        "context__state_mean__median__jump_intensity",
        "context__state_mean__median__leverage_build_score",
        "context__state_mean__median__memory_asymmetry_1ATR",
        "context__state_mean__median__spread_proxy_abs_return_bps_robust_z",
        "context__past_geometry_shift_3h",
        "context__mapping_current__above_p90_share",
    ):
        frame[feature] = 0.0
    handoff = materialize_attribution_rows(frame)
    flag = "selected_global_top10__score_base_alpha"
    assert int(handoff[flag].sum()) == 4
    # Scores increase with row index, whose sides alternate; selection is
    # score-global and is not forced to contain a fixed side share.
    assert handoff.loc[handoff[flag], "candidate_id"].tolist() == [
        "c036",
        "c037",
        "c038",
        "c039",
    ]


def test_attribution_rejects_degenerate_preperiod_transition_reference() -> None:
    frame = _frame()
    historical = pd.DataFrame(index=np.arange(200))
    for feature in (
        "context__state_mean__median__atr_compression_ratio",
        "context__state_mean__median__breakout_24h",
        "context__state_mean__median__dir_path_risk_skew_2h",
        "context__state_mean__median__jump_intensity",
        "context__state_mean__median__leverage_build_score",
        "context__state_mean__median__memory_asymmetry_1ATR",
        "context__state_mean__median__spread_proxy_abs_return_bps_robust_z",
        "context__past_geometry_shift_3h",
        "context__mapping_current__above_p90_share",
    ):
        frame[feature] = 0.0
        historical[feature] = 0.0
    with pytest.raises(ValueError, match="degenerate transition reference"):
        build_attribution(frame, historical)
