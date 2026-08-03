from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.inverse_fixed_geometry_evaluation import (
    evaluate_inverse_fixed_geometry_arms,
)


def _panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for arm, offset in (("direct", 0.0), ("aux", 0.2)):
        for month in ("2022-01-01", "2022-02-01"):
            decision = pd.Timestamp(month, tz="UTC")
            for position, candidate in enumerate(("a", "b", "c", "d")):
                # The aux arm makes c the top candidate, direct makes a top.
                score = [0.9, 0.8, 0.7, 0.6][position]
                if arm == "aux":
                    score = [0.4, 0.5, 1.1, 0.3][position]
                rows.append({
                    "arm": arm, "candidate_id": f"{month}-{candidate}",
                    "architecture": "direct_ev" if arm == "direct" else "direct_ev_plus_aux",
                    "feature_arm": "alpha_context" if arm == "direct" else "alpha_context_plus_aux",
                    "execution_decision_utc": decision + pd.Timedelta(hours=position),
                    "mapped_score": score + offset * 0.0,
                    "execution_net_ev_12h": [0.01, -0.02, 0.03, -0.04][position],
                    "execution_gross_ev_12h": [0.02, -0.01, 0.04, -0.03][position],
                    "side_name": "long" if position % 2 == 0 else "short",
                    "eligible": True,
                    "mapping_status": "causal_train_only",
                    "mapping_max_label_resolution_utc": decision - pd.Timedelta(hours=1),
                })
    return pd.DataFrame(rows)


def test_monthly_global_topk_is_cross_side_and_stably_tie_broken() -> None:
    result = evaluate_inverse_fixed_geometry_arms(
        _panel(), top_fractions=(0.25, 0.50), expected_months=("2022-01", "2022-02"), baseline_arm="direct",
    )
    direct_top = result.selections.loc[
        (result.selections.arm.eq("direct")) & result.selections.top_fraction.eq(0.25)
    ]
    assert direct_top.groupby("evaluation_month").size().tolist() == [1, 1]
    assert direct_top.candidate_id.str.endswith("-a").all()
    monthly = result.monthly.loc[result.monthly.top_fraction.eq(0.25)]
    assert monthly.selection_scope.eq(
        "one_pooled_global_top_k_per_evaluation_month_after_declared_train_only_mapping"
    ).all()
    assert monthly.book_depth.eq(1).all()
    assert set(result.summary.top_fraction) == {0.25, 0.5}
    assert result.contract["promotion_eligible"] is False


def test_aux_arm_comparison_has_economics_and_worst_month() -> None:
    result = evaluate_inverse_fixed_geometry_arms(
        _panel(), top_fractions=(0.25,), baseline_arm="direct", arm_metadata_cols=("architecture", "feature_arm"),
    )
    aux = result.summary.loc[result.summary.arm.eq("aux")].iloc[0]
    assert aux["mean_net_bps"] == pytest.approx(300.0)
    assert aux["mean_gross_bps"] == pytest.approx(400.0)
    assert aux["worst_month_mean_net_bps"] == pytest.approx(300.0)
    comparison = result.comparisons.loc[result.comparisons.arm.eq("aux")].iloc[0]
    assert comparison["delta_mean_net_bps_vs_baseline"] == pytest.approx(200.0)
    assert comparison["matched_candidate_population"]
    assert comparison["architecture"] == "direct_ev_plus_aux"
    assert result.monotonicity.groupby(["arm", "evaluation_month"]).size().eq(4).all()


def test_rejects_future_or_unattested_mapping_and_unmatched_arms() -> None:
    future = _panel()
    future.loc[0, "mapping_max_label_resolution_utc"] = future.loc[0, "execution_decision_utc"]
    with pytest.raises(ValueError, match="strictly before"):
        evaluate_inverse_fixed_geometry_arms(future, top_fractions=(0.1,))

    unmatched = _panel().iloc[:-1].copy()
    with pytest.raises(ValueError, match="same candidate identities"):
        evaluate_inverse_fixed_geometry_arms(unmatched, top_fractions=(0.1,))

    altered_target = _panel()
    altered_target.loc[altered_target.arm.eq("aux").idxmax(), "execution_net_ev_12h"] = 9.0
    with pytest.raises(ValueError, match="same candidate identities, outcomes"):
        evaluate_inverse_fixed_geometry_arms(altered_target, top_fractions=(0.1,))


def test_identity_mapping_is_explicit_and_never_claims_a_fit() -> None:
    panel = _panel()
    direct = panel.arm.eq("direct")
    panel.loc[direct, "mapping_status"] = "identity_no_fit"
    panel.loc[direct, "mapping_max_label_resolution_utc"] = pd.NaT
    result = evaluate_inverse_fixed_geometry_arms(panel, top_fractions=(0.2,))
    assert result.summary.month_coverage_complete.all()


def test_noncausal_leave_block_out_mapping_is_explicitly_accepted() -> None:
    panel = _panel()
    panel["mapping_status"] = "out_of_block_train_only_noncausal"
    panel["mapping_max_label_resolution_utc"] = (
        panel["execution_decision_utc"].max() + pd.Timedelta(days=1)
    )
    result = evaluate_inverse_fixed_geometry_arms(panel, top_fractions=(0.2,))
    assert result.contract["promotion_eligible"] is False
