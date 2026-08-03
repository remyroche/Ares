import numpy as np
import pandas as pd

from scripts.run_cross_era_tail_payoff_challenger import (
    ADVERSE,
    OTHER,
    POSITIVE,
    TIMEOUT,
    add_regime_composites,
    apply_class_calibrators,
    compose_tail_scores,
    economic_event_code,
    feature_arms,
    fit_class_calibrators,
    normalise_probabilities,
)


def test_economic_events_are_exhaustive_and_negative_only():
    frame = pd.DataFrame(
        {
            "positive_net": [1, 0, 0, 0, 1],
            "adverse_first": [1, 1, 0, 0, 0],
            "timeout_event": [0, 0, 1, 0, 1],
        }
    )
    result = economic_event_code(frame)
    assert result.tolist() == [POSITIVE, ADVERSE, TIMEOUT, OTHER, POSITIVE]
    assert set(result) == {POSITIVE, ADVERSE, TIMEOUT, OTHER}


def test_domain_calibration_is_separate_and_returns_simplex():
    raw = normalise_probabilities(
        np.array(
            [
                [.8, .1, .05, .05], [.7, .1, .1, .1], [.1, .7, .1, .1], [.1, .1, .7, .1],
                [.3, .3, .2, .2], [.2, .2, .3, .3], [.4, .2, .2, .2], [.2, .4, .2, .2],
            ]
        )
    )
    target = np.array([POSITIVE, POSITIVE, ADVERSE, TIMEOUT, OTHER, TIMEOUT, POSITIVE, ADVERSE])
    eras = np.array(["old"] * 4 + ["recent"] * 4)
    calibrators = fit_class_calibrators(raw, target, eras)
    assert set(calibrators) == {"old", "recent"}
    calibrated = apply_class_calibrators(raw, calibrators, eras)
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-12)
    assert np.isfinite(calibrated).all()


def test_tail_score_uses_conservative_gain_and_adverse_quantile():
    frame = pd.DataFrame(
        {
            "p_positive": [.5], "p_adverse_negative": [.25], "p_timeout_negative": [.125], "p_other_negative": [.125],
            "q25_positive_bps": [100.0], "q50_positive_bps": [150.0],
            "q50_adverse_bps": [200.0], "q85_adverse_bps": [400.0],
            "q75_timeout_bps": [80.0], "q75_other_bps": [40.0],
        }
    )
    result = compose_tail_scores(frame)
    assert result.loc[0, "ev50_bps"] == 10.0
    assert result.loc[0, "tail_ev_bps"] == -65.0


def test_feature_arms_exclude_unstable_raw_rank_margin_and_group_context():
    contract = {
        "feature_columns": ["raw_a", "raw_b"],
        "candidate_context_columns": [
            "base_oof_score", "candidate_group_size", "base_rank_timestamp_side",
            "base_rank_pct_timestamp_side", "base_score_z_timestamp_side", "base_margin_to_candidate_cutoff",
        ],
    }
    arms = feature_arms(contract, ["regime_x"])
    assert "base_rank_pct_timestamp_side" in arms["raw_context"]
    assert "base_oof_score" in arms["raw_context"]
    assert "base_score_z_timestamp_side" in arms["raw_context"]
    assert "candidate_group_size" not in arms["raw_context"]
    assert "base_rank_timestamp_side" not in arms["raw_context"]
    assert "base_margin_to_candidate_cutoff" not in arms["raw_context"]


def test_regime_composites_add_explicit_domain_interactions():
    frame = pd.DataFrame(
        {
            "era": ["2025_feb_apr", "2026_may_jul19"],
            "regime_transition_entropy_48h": [.2, .4], "regime_stability_24h": [.5, .25],
            "market_breadth_24h": [.1, .2], "negative_breadth_pct": [.3, .1],
            "eth_btc_ret_24h": [.01, -.02], "xs_dispersion__amihud_illiq": [.5, .75],
            "volatility_of_volatility_48": [.25, .5],
            "base_rank_pct_timestamp_side": [.4, .6], "base_oof_score": [.1, .2],
            "base_score_z_timestamp_side": [.2, .3],
        }
    )
    result, names = add_regime_composites(frame)
    assert len(names) == 4
    assert result["__era_is_2026__"].tolist() == [0.0, 1.0]
    assert result.loc[0, "__base_rank_pct_x_era__"] == 0.0
    assert result.loc[1, "__base_rank_pct_x_era__"] == .6
