from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_iii_residual_target_challengers import (
    ONE_SHARED_MODEL,
    ORDINAL_EDGES_BPS,
    PAIR_SEPARATIONS_BPS,
    QUANTILE_LEVELS,
    PairConstructionConfig,
    StageIIIResidualTargetError,
    candidate_residual_bps,
    construct_context_matched_residual_pairs,
    fit_quantile_residual_targets,
    fit_regime_centered_ordinal_residual,
    reconstruct_expected_net_bps,
    reconstruct_ordinal_candidate_residual_bps,
    reconstruct_quantile_residual_outputs,
)


def _frame() -> pd.DataFrame:
    decision = pd.to_datetime(
        [
            "2024-01-01 10:00", "2024-01-01 10:05", "2024-01-01 10:10",
            "2024-01-01 10:15", "2024-01-01 10:20", "2024-01-01 10:25",
            "2024-01-01 10:30", "2024-01-01 10:35",
        ],
        utc=True,
    )
    # Candidate residuals: -200, -150, -50, 0, 50, 100, 150, 250 bps.
    residual = np.asarray([-200, -150, -50, 0, 50, 100, 150, 250], dtype=float)
    base = np.asarray([20, 22, 24, 26, 28, 30, 32, 34], dtype=float)
    prior = np.asarray([-10, -10, -5, -5, 5, 5, 10, 10], dtype=float)
    return pd.DataFrame(
        {
            "candidate_id": [f"c{i}" for i in range(8)],
            "decision_ts": decision,
            "label_available_ts": decision + pd.Timedelta(hours=12),
            "side_name": ["long"] * 6 + ["short"] * 2,
            "exact_net_bps": base + prior + residual,
            "prequential_base_expected_net_bps": base,
            "prequential_soft_regime_prior_residual_bps": prior,
            "cost_to_atr": [1.00, 1.05, 1.10, 1.15, 1.12, 1.08, 2.0, 2.1],
            "p_regime_calm": [0.9, 0.85, 0.88, 0.84, 0.86, 0.90, 0.1, 0.15],
            "p_regime_stress": [0.1, 0.15, 0.12, 0.16, 0.14, 0.10, 0.9, 0.85],
            "base_map_is_prequential": True,
            "base_map_source_side": ["long"] * 6 + ["short"] * 2,
            "base_map_max_label_available_ts": decision - pd.Timedelta(hours=1),
            "soft_regime_is_causal_prequential": True,
            "soft_regime_fit_end_ts": decision - pd.Timedelta(hours=1),
            "prior_resolved_max_label_available_ts": decision - pd.Timedelta(minutes=30),
            "cost_atr_is_causal": True,
        }
    )


def test_ordinal_uses_fixed_regime_centered_bins_and_training_only_means() -> None:
    frame = _frame()
    fit, labels = fit_regime_centered_ordinal_residual(
        frame, fit_before_utc="2024-01-03"
    )
    assert fit.edges_bps == ORDINAL_EDGES_BPS
    assert fit.routing == ONE_SHARED_MODEL
    assert fit.class_support == (2, 2, 2, 2)
    np.testing.assert_array_equal(labels, [0, 0, 1, 1, 2, 2, 3, 3])
    np.testing.assert_allclose(fit.class_mean_bps, [-175, -25, 75, 200])

    probability = np.eye(4, dtype=float)
    residual_prediction = reconstruct_ordinal_candidate_residual_bps(probability, fit)
    np.testing.assert_allclose(residual_prediction, fit.class_mean_bps)
    expected_net = reconstruct_expected_net_bps(
        frame.iloc[:4], residual_prediction
    )
    np.testing.assert_allclose(
        expected_net,
        frame["prequential_base_expected_net_bps"].iloc[:4]
        + frame["prequential_soft_regime_prior_residual_bps"].iloc[:4]
        + residual_prediction,
    )


def test_ordinal_reconstruction_rejects_non_simplex_probabilities() -> None:
    fit, _ = fit_regime_centered_ordinal_residual(_frame(), fit_before_utc="2024-01-03")
    with pytest.raises(StageIIIResidualTargetError, match="sum to one"):
        reconstruct_ordinal_candidate_residual_bps([[0.2, 0.2, 0.2, 0.2]], fit)


def test_quantile_contract_has_five_shared_heads_and_risk_outputs() -> None:
    frame = _frame()
    fit, targets = fit_quantile_residual_targets(frame, fit_before_utc="2024-01-03")
    assert fit.quantiles == QUANTILE_LEVELS
    assert fit.routing == ONE_SHARED_MODEL
    assert tuple(targets) == ("q10", "q25", "q50", "q75", "q90")
    for target in targets.values():
        np.testing.assert_allclose(target, candidate_residual_bps(frame))

    outputs = reconstruct_quantile_residual_outputs(
        {
            "q10": [-100, -50], "q25": [-50, -20], "q50": [20, 10],
            "q75": [60, 40], "q90": [100, 80],
        },
        fit,
    )
    np.testing.assert_allclose(outputs["candidate_residual_median_bps"], [20, 10])
    np.testing.assert_allclose(outputs["candidate_residual_downside_bps"], [120, 60])
    np.testing.assert_allclose(outputs["candidate_residual_width_bps"], [200, 130])
    np.testing.assert_allclose(outputs["candidate_residual_iqr_bps"], [110, 60])
    assert not outputs["quantile_crossing_repaired"].any()


def test_quantile_crossing_is_causally_repaired_without_outcomes() -> None:
    fit, _ = fit_quantile_residual_targets(_frame(), fit_before_utc="2024-01-03")
    crossed = {"q10": [0], "q25": [-1], "q50": [5], "q75": [4], "q90": [10]}
    output = reconstruct_quantile_residual_outputs(crossed, fit)
    np.testing.assert_allclose(
        output.loc[0, ["q10", "q25", "q50", "q75", "q90"]].to_numpy(dtype=float),
        [0, 0, 5, 5, 10],
    )
    assert bool(output.loc[0, "quantile_crossing_repaired"])
    with pytest.raises(StageIIIResidualTargetError, match="cross"):
        reconstruct_quantile_residual_outputs(crossed, fit, repair_crossing=False)


def test_pair_builder_matches_side_date_soft_regime_base_ev_and_cost_atr() -> None:
    frame = _frame()
    target = candidate_residual_bps(frame)
    pairs = construct_context_matched_residual_pairs(
        frame,
        target,
        soft_regime_columns=("p_regime_calm", "p_regime_stress"),
        fit_before_utc="2024-01-03",
        config=PairConstructionConfig(
            min_soft_regime_similarity=0.50,
            max_base_ev_difference_bps=20.0,
            max_cost_atr_difference=0.25,
            max_pairs_per_better_row=2,
        ),
    )
    assert not pairs.empty
    assert set(PAIR_SEPARATIONS_BPS) == {50.0, 100.0}
    assert pairs["routing"].eq(ONE_SHARED_MODEL).all()
    assert pairs["residual_gap_bps"].ge(50.0).all()
    assert pairs["soft_regime_similarity"].ge(0.50).all()
    assert pairs["base_ev_difference_bps"].abs().le(20.0).all()
    assert pairs["cost_atr_difference"].abs().le(0.25).all()
    assert pairs["eligible_50bps"].all()
    assert (
        pairs["eligible_100bps"].to_numpy()
        == pairs["residual_gap_bps"].ge(100.0).to_numpy()
    ).all()
    # Pair context is side-local; no long/short comparison can be emitted.
    id_side = frame.set_index("candidate_id")["side_name"]
    assert all(
        id_side[better] == id_side[worse]
        for better, worse in zip(pairs["better_candidate_id"], pairs["worse_candidate_id"])
    )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda x: x.assign(label_available_ts=pd.Timestamp("2024-01-03", tz="UTC")), "unresolved"),
        (lambda x: x.assign(base_map_is_prequential=False), "base_map_is_prequential"),
        (lambda x: x.assign(soft_regime_is_causal_prequential=False), "soft_regime_is_causal"),
        (lambda x: x.assign(cost_atr_is_causal=False), "cost_atr_is_causal"),
    ],
)
def test_pair_builder_rejects_cutoff_or_noncausal_inputs(mutation, message: str) -> None:
    frame = mutation(_frame())
    with pytest.raises(StageIIIResidualTargetError, match=message):
        construct_context_matched_residual_pairs(
            frame,
            candidate_residual_bps(frame),
            soft_regime_columns=("p_regime_calm", "p_regime_stress"),
            fit_before_utc="2024-01-03",
        )


def test_all_target_fits_reject_rows_not_resolved_before_cutoff() -> None:
    frame = _frame()
    frame.loc[0, "label_available_ts"] = pd.Timestamp("2024-01-03", tz="UTC")
    with pytest.raises(StageIIIResidualTargetError, match="unresolved"):
        fit_regime_centered_ordinal_residual(frame, fit_before_utc="2024-01-03")
    with pytest.raises(StageIIIResidualTargetError, match="unresolved"):
        fit_quantile_residual_targets(frame, fit_before_utc="2024-01-03")


@pytest.mark.parametrize("bad_value", ["False", np.nan, 2, -1])
def test_target_and_pair_lineage_rejects_truthy_non_booleans(bad_value: object) -> None:
    frame = _frame()
    frame["base_map_is_prequential"] = bad_value
    with pytest.raises(StageIIIResidualTargetError, match="explicit true booleans"):
        fit_regime_centered_ordinal_residual(frame, fit_before_utc="2024-01-03")
    with pytest.raises(StageIIIResidualTargetError, match="explicit true booleans"):
        construct_context_matched_residual_pairs(
            frame,
            candidate_residual_bps(frame),
            soft_regime_columns=("p_regime_calm", "p_regime_stress"),
            fit_before_utc="2024-01-03",
        )


def test_target_lineage_rejects_cross_side_or_current_priors() -> None:
    cross_side = _frame()
    cross_side.loc[0, "base_map_source_side"] = "short"
    with pytest.raises(StageIIIResidualTargetError, match="same-side"):
        fit_quantile_residual_targets(cross_side, fit_before_utc="2024-01-03")

    current = _frame()
    current.loc[0, "prior_resolved_max_label_available_ts"] = current.loc[0, "decision_ts"]
    with pytest.raises(StageIIIResidualTargetError, match="current/future lineage"):
        fit_regime_centered_ordinal_residual(current, fit_before_utc="2024-01-03")
