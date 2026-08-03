from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.materialize_cross_era_global_book_transition_research_panel import (
    SOURCE_CONTRACTS,
    _add_persistent_adverse_labels,
    build_transition_panel,
)


def _geometry() -> pd.DataFrame:
    rows = []
    for stamp in pd.date_range("2025-01-01", periods=60, freq="h", tz="UTC"):
        for side, offset in (("long", 0.0), ("short", 1.0)):
            rows.append(
                {
                    "__ts__": stamp,
                    "side_name": side,
                    "median__range_12h_pct": stamp.hour + offset,
                    "iqr__range_12h_pct": 2.0 + offset,
                }
            )
    return pd.DataFrame(rows)


def _labels() -> pd.DataFrame:
    rows = []
    for horizon in (3, 12):
        for anchor in pd.date_range(
            "2025-01-02", periods=12, freq="h", tz="UTC"
        ):
            before = 0.002
            after = -0.001
            rows.append(
                {
                    "cohort_anchor_utc": anchor,
                    "horizon_hours": horizon,
                    "horizon_role": "primary" if horizon == 12 else "auxiliary",
                    "book_fraction": 0.10,
                    "before_global_hour_complete_flag": True,
                    "after_global_hour_complete_flag": True,
                    "before_selected_candidate_support": 10,
                    "after_selected_candidate_support": 10,
                    "outcome_only_not_model_feature": True,
                    "before_target_available_utc": anchor,
                    "after_target_available_utc": anchor
                    + pd.Timedelta(hours=horizon + 12),
                    "before_direct_mean_net": before,
                    "after_direct_mean_net": after,
                    "after_mean_conversion_residual": -0.01,
                    "delta_direct_mean_net": after - before,
                    "delta_mean_gross": -0.002,
                    "delta_mean_cost": 0.0,
                    "delta_mean_conversion_residual": -0.003,
                    "delta_opportunity_probability_0bps": -0.2,
                    "delta_opportunity_probability_25bps": -0.1,
                    "delta_positive_net_contribution": -0.002,
                    "delta_positive_net_contribution_robust_mean": -0.002,
                    "delta_loss_net_contribution": 0.003,
                    "delta_loss_net_contribution_robust_mean": 0.003,
                    "delta_p_exit_trailing": -0.1,
                    "delta_p_exit_timeout": 0.1,
                    "delta_p_exit_full_stop": 0.0,
                    "delta_p_exit_adverse_exit": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_panel_has_causal_features_and_separate_soft_targets() -> None:
    families = {
        name: _labels() for name in SOURCE_CONTRACTS
    }
    panel, features, targets = build_transition_panel(
        families, _geometry()
    )
    assert panel["source_family"].nunique() == 3
    assert panel["context_available"].all()
    assert "target__soft_net_deterioration_25bps" in targets
    assert panel["target__soft_net_deterioration_25bps"].eq(1.0).all()
    assert panel["target__adverse_transition_any"].eq(1.0).all()
    assert features
    assert not any(
        token in column.lower()
        for column in features
        for token in ("future", "target", "outcome", "execution")
    )


def test_nonoverlap_stride_uses_full_two_horizon_window() -> None:
    panel, _, _ = build_transition_panel(
        {"canonical_spread_febapr2025": _labels()}, _geometry()
    )
    for horizon, group in panel.groupby("horizon_hours"):
        selected = group.loc[group["nonoverlap_anchor_flag"]].sort_values(
            "cohort_anchor_utc"
        )
        if len(selected) > 1:
            gaps = selected["cohort_anchor_utc"].diff().dropna()
            assert gaps.ge(pd.Timedelta(hours=2 * int(horizon))).all()


def test_adverse_sensitivities_and_conditional_mechanisms_keep_exact_availability() -> None:
    anchors = pd.date_range("2025-01-01", periods=14, freq="h", tz="UTC")
    rows = []
    for index, anchor in enumerate(anchors):
        adverse = 8 <= index <= 12
        rows.append(
            {
                "source_family": "canonical_spread_febapr2025",
                "horizon_hours": 12,
                "book_fraction": 0.10,
                "cohort_anchor_utc": anchor,
                "after_mean_conversion_residual": -0.010,
                "delta_mean_conversion_residual": -0.008 if adverse else 0.0,
                "delta_direct_mean_net": -0.008 if adverse else 0.0,
                "delta_positive_net_contribution": -0.006,
                "delta_loss_net_contribution": 0.006,
                "before_target_available_utc": anchor + pd.Timedelta(hours=2),
                "after_target_available_utc": anchor + pd.Timedelta(hours=12),
            }
        )
    result = _add_persistent_adverse_labels(pd.DataFrame(rows)).set_index(
        "cohort_anchor_utc"
    )
    anchor = anchors[7]
    assert result.loc[anchor, "target__active_adverse_sensitivity_50bps"] == 1.0
    assert result.loc[anchor, "target__active_adverse_sensitivity_75bps"] == 1.0
    assert result.loc[anchor, "target__active_adverse_sensitivity_100bps"] == 0.0
    expected_available = anchors[9] + pd.Timedelta(hours=12)
    assert (
        result.loc[
            anchor, "target__active_adverse_sensitivity_75bps_available_utc"
        ]
        == expected_available
    )
    assert result.loc[anchor, "target__adverse_onset_sensitivity_75bps"] == 1.0
    assert (
        result.loc[
            anchor, "target__adverse_onset_sensitivity_75bps_available_utc"
        ]
        == expected_available
    )
    assert result.loc[anchor, "target__mechanism_upside_collapse"] == 1.0
    assert result.loc[anchor, "target__mechanism_loss_expansion"] == 1.0
    assert (
        result.loc[
            anchor, "target__mechanism_upside_collapse_available_utc"
        ]
        == expected_available
    )
    assert pd.isna(result.loc[anchors[0], "target__mechanism_upside_collapse"])
