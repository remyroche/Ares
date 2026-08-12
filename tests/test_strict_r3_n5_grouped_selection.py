from __future__ import annotations

import json

import numpy as np
import pandas as pd

import scripts.run_strict_r3_n5_canonical_selection as selection


def test_every_challenger_field_maps_to_one_versioned_group() -> None:
    contract = json.loads(
        open("config/strict_r3_n5_forest_support_v2_challenger.json").read()
    )
    config = selection._feature_group_config()
    groups = [selection._mda_group(field, config) for field in contract["features"]]
    assert len(groups) == len(contract["features"])
    assert len(set(groups)) == 8


def test_legacy_canonical_covariance_fields_have_a_stable_group() -> None:
    contract = json.loads(open("config/strict_r3_ldf_support_v3.json").read())
    config = selection._feature_group_config()
    groups = {
        field: selection._mda_group(field, config)
        for field in contract["features"]
    }
    assert groups["reliability_cov_break_upstream_7v28"] == (
        "I_legacy_score_outcome_covariance"
    )
    assert groups["reliability_corr_break_disagreement_7v28"] == (
        "I_legacy_score_outcome_covariance"
    )


def test_joint_conditional_permutation_preserves_rows_and_context_multisets() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": [f"c{index}" for index in range(12)],
            "a": np.arange(12),
            "b": np.arange(12) * 10,
        },
        index=np.arange(100, 112),
    )
    strata = np.repeat(["x", "y", "z"], 4)
    result = selection._permute_feature_frame(
        frame, ["a", "b"], strata, np.random.default_rng(17), joint=True,
    )
    assert result.index.equals(frame.index)
    assert result["candidate_id"].equals(frame["candidate_id"])
    for token in np.unique(strata):
        mask = strata == token
        before = sorted(map(tuple, frame.loc[mask, ["a", "b"]].to_numpy()))
        after = sorted(map(tuple, result.loc[mask, ["a", "b"]].to_numpy()))
        assert after == before


def test_portability_score_uses_predeclared_mad_and_negative_worst_penalties() -> None:
    detail = pd.DataFrame(
        {
            "fold": [0, 1, 2],
            "field": ["x"] * 3,
            "mda_loss": [10.0, 2.0, -2.0],
        }
    )
    result = selection._portable_summary(detail, ("field",), 3)
    assert result.iloc[0]["portable_mda_score"] == -0.5
    assert result.iloc[0]["positive_fold_recurrence"] == 2.0 / 3.0


def test_backward_elimination_is_deterministic(monkeypatch) -> None:
    fields = [
        "base_score", "base_rank", "base_anchor_bps", "consensus_rank",
        "final_score", "correctness_raw", "correctness_rank",
        "reliability_base_consensus_gap", "reliability_base_consensus_abs_gap",
        "reliability_base_consensus_mean", "reliability_upstream_rank",
        "k9_entropy", "k9_top2_margin", "leaf_support_effective",
    ]
    metrics = {
        "selection_score": 100.0, "weighted_tail_score": 100.0,
        "mean_portability_top1_2_5": 50.0, "worst_month_top1_2_5": 20.0,
        "top1_net_bps": 100.0, "top2_net_bps": 80.0, "top5_net_bps": 40.0,
    }

    def fake_evaluate(_folds, candidate_fields, *, params, arm):
        return pd.DataFrame(), dict(metrics)

    monkeypatch.setattr(selection, "_evaluate_contract", fake_evaluate)
    group_rows = []
    feature_rows = []
    for index, field in enumerate(fields):
        group = selection._mda_group(field)
        group_rows.append(
            {"group": group, "portable_mda_score": float(index), "positive_fold_recurrence": 1.0}
        )
        feature_rows.append(
            {"field": field, "portable_mda_score": float(index), "positive_fold_recurrence": 1.0}
        )
    first, _ = selection._backward_grouped_elimination(
        [], fields, params=selection.BASELINE_N5_PARAMS,
        group_mda=pd.DataFrame(group_rows), feature_mda=pd.DataFrame(feature_rows),
    )
    second, _ = selection._backward_grouped_elimination(
        [], fields, params=selection.BASELINE_N5_PARAMS,
        group_mda=pd.DataFrame(group_rows), feature_mda=pd.DataFrame(feature_rows),
    )
    assert first == second
    assert len(first) >= 12
