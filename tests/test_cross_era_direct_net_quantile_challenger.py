import numpy as np
import pandas as pd
import pytest

from scripts.run_cross_era_direct_net_quantile_challenger import (
    _assert_complete_current_labels,
    _probability_calibration_metrics,
    _tail_economics_by_period,
    compose_scores,
    monotone_severe_probabilities,
)


def test_severe_probabilities_are_monotone_after_projection():
    frame = pd.DataFrame(
        {
            "p_loss_le_100": [.1, .8],
            "p_loss_le_200": [.4, .2],
            "p_loss_le_400": [.3, .5],
        }
    )
    result = monotone_severe_probabilities(frame)
    assert (result["p_loss_le_100"] >= result["p_loss_le_200"]).all()
    assert (result["p_loss_le_200"] >= result["p_loss_le_400"]).all()
    assert ((result[["p_loss_le_100", "p_loss_le_200", "p_loss_le_400"]] >= 0) & (result[["p_loss_le_100", "p_loss_le_200", "p_loss_le_400"]] <= 1)).all().all()


def test_severe_expected_loss_uses_non_overlapping_threshold_buckets():
    frame = pd.DataFrame(
        {
            "p_loss_le_100": [.50], "p_loss_le_200": [.30], "p_loss_le_400": [.10],
            "q75_loss_100_200_bps": [150.], "q75_loss_200_400_bps": [300.], "q75_loss_400_plus_bps": [600.],
            "q10_net_bps": [-50.], "q25_net_bps": [-20.], "q50_net_bps": [40.], "q75_net_bps": [120.],
        }
    )
    result = compose_scores(frame)
    # (.50-.30)*150 + (.30-.10)*300 + .10*600 = 150 bps.
    assert result.loc[0, "severe_expected_loss_bps"] == 150.
    assert result.loc[0, "score_median_bps"] == 40.
    assert result.loc[0, "score_lower_quantile_bps"] == -20.
    assert result.loc[0, "score_median_minus_severe_bps"] == -110.


def test_direct_score_does_not_need_competing_risk_or_action_inputs():
    frame = pd.DataFrame(
        {
            "p_loss_le_100": [.1], "p_loss_le_200": [.05], "p_loss_le_400": [.01],
            "q75_loss_100_200_bps": [150.], "q75_loss_200_400_bps": [300.], "q75_loss_400_plus_bps": [600.],
            "q10_net_bps": [-100.], "q25_net_bps": [-50.], "q50_net_bps": [10.], "q75_net_bps": [80.],
        }
    )
    result = compose_scores(frame)
    assert np.isfinite(result[["score_median_bps", "score_lower_quantile_bps", "score_median_minus_severe_bps"]].to_numpy(float)).all()


def test_direct_quantiles_are_monotone_before_scoring():
    frame = pd.DataFrame(
        {
            "p_loss_le_100": [.1], "p_loss_le_200": [.05], "p_loss_le_400": [.01],
            "q75_loss_100_200_bps": [150.], "q75_loss_200_400_bps": [300.], "q75_loss_400_plus_bps": [600.],
            "q10_net_bps": [25.], "q25_net_bps": [-50.], "q50_net_bps": [-100.], "q75_net_bps": [80.],
        }
    )
    result = compose_scores(frame)
    values = result.loc[0, ["q10_net_bps", "q25_net_bps", "q50_net_bps", "q75_net_bps"]].to_numpy(float)
    assert np.all(values[:-1] <= values[1:])


def _evaluation_frame() -> pd.DataFrame:
    rows = []
    for index in range(20):
        side = "long" if index < 10 else "short"
        rows.append(
            {
                "candidate_id": f"c{index:02d}",
                "__ts__": pd.Timestamp("2026-07-20T00:00:00Z") + pd.Timedelta(hours=index),
                "__symbol__": f"S{index % 3}",
                "side_name": side,
                "mapped_q25_bps": float(index),
                "execution_net_ev_12h": (index - 10) / 10_000,
                "raw_p_loss_le_100": .7 if index < 10 else .2,
                "p_loss_le_100": .6 if index < 10 else .3,
                "raw_p_loss_le_200": .4 if index < 10 else .1,
                "p_loss_le_200": .3 if index < 10 else .1,
                "raw_p_loss_le_400": .2 if index < 10 else .05,
                "p_loss_le_400": .1 if index < 10 else .05,
            }
        )
    return pd.DataFrame(rows)


def test_current_label_coverage_fails_closed():
    predictions = _evaluation_frame()
    labels = predictions.loc[:, ["candidate_id", "__ts__", "__symbol__", "side_name"]].copy()
    evidence = _assert_complete_current_labels(predictions, labels)
    assert evidence["identity_complete_one_to_one"] is True
    with pytest.raises(ValueError, match="coverage mismatch"):
        _assert_complete_current_labels(predictions, labels.iloc[:-1])
    duplicate = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate identities"):
        _assert_complete_current_labels(predictions, duplicate)


def test_period_economics_preserves_one_global_book_and_side_diagnostics():
    frame = _evaluation_frame()
    result = _tail_economics_by_period(frame, "mapped_q25_bps", "current")
    aggregate = result.loc[result["level"].eq("aggregate")]
    global_row = aggregate.loc[aggregate["scope"].eq("global")].iloc[0]
    assert global_row["rows"] == 2
    assert global_row["long_rows"] == 0
    assert global_row["short_rows"] == 2
    assert set(aggregate["scope"]) == {"global", "side_local_long", "side_local_short"}


def test_probability_calibration_is_reported_raw_and_calibrated_by_side():
    result = _probability_calibration_metrics(_evaluation_frame(), "current")
    aggregate = result.loc[result["month"].eq("all")]
    assert set(aggregate["side_name"]) == {"long", "short"}
    assert set(aggregate["calibration"]) == {"raw", "calibrated"}
    assert set(aggregate["head"]) == {"p_loss_le_100", "p_loss_le_200", "p_loss_le_400"}
    assert np.isfinite(aggregate[["brier", "ece10"]].to_numpy(float)).all()
