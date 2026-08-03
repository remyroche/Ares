import numpy as np
import pandas as pd
import pytest
from time import perf_counter

from extreme_price_movements.prequential_r3_value_map import (
    PrequentialR3ValueMapConfig,
    prequential_same_side_r3_value_map,
    r3_opportunity_score,
)


def test_r3_value_map_uses_only_prior_resolved_same_side_outcomes() -> None:
    decision = pd.to_datetime(
        ["2024-01-01T00:00:00Z", "2024-01-01T14:00:00Z", "2024-01-01T14:00:00Z"]
    )
    available = decision + pd.Timedelta(hours=13)
    # The first row is resolved before 14:00; the two later rows are mapped
    # together and cannot use one another's future exact-net outcomes.
    expected, audit, provenance = prequential_same_side_r3_value_map(
        exact_net_bps=[100.0, -500.0, 500.0],
        decision_timestamps=decision,
        label_available_timestamps=available,
        side="long",
        score=[0.9, 0.9, 0.9],
        config=PrequentialR3ValueMapConfig(side="long", min_global_rows=1, bin_shrink_rows=1),
    )
    assert expected[0] == pytest.approx(0.0)  # no prior-resolved label
    assert expected[1] == pytest.approx(100.0)
    assert expected[2] == pytest.approx(100.0)
    assert audit.loc[1, "prior_resolved_global_support"] == 1
    assert audit.loc[2, "prior_resolved_global_support"] == 1
    assert provenance["is_21_day_admission_map"] is False
    assert provenance["prior_resolution_rule"] == "label_available_ts < decision_ts"


def test_r3_probability_simplex_and_side_contract_are_validated() -> None:
    score = r3_opportunity_score(
        p_clear=[0.8, 0.1], p_adverse=[0.1, 0.3], p_weak=[0.1, 0.6]
    )
    np.testing.assert_allclose(score, [0.7, -0.2])
    ts = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="config side"):
        prequential_same_side_r3_value_map(
            exact_net_bps=[1.0, 2.0], decision_timestamps=ts,
            label_available_timestamps=ts + pd.Timedelta(hours=13), side="short",
            score=[0.1, 0.2], config=PrequentialR3ValueMapConfig(side="long"),
        )
    with pytest.raises(ValueError, match="sum to one"):
        r3_opportunity_score(p_clear=[0.8], p_adverse=[0.3], p_weak=[0.1])
    with pytest.raises(ValueError, match="strictly after decision"):
        prequential_same_side_r3_value_map(
            exact_net_bps=[1.0, 2.0], decision_timestamps=ts,
            label_available_timestamps=ts, side="long", score=[0.1, 0.2],
            config=PrequentialR3ValueMapConfig(side="long"),
        )


def _quadratic_reference(
    *, net: np.ndarray, decision: pd.DatetimeIndex, available: pd.DatetimeIndex,
    score: np.ndarray, config: PrequentialR3ValueMapConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Small-input specification matching the original production routine."""
    n = len(score)
    bucket = np.minimum(
        int(config.bins) - 1,
        np.floor((np.clip(score, -1.0, 1.0) + 1.0) * int(config.bins) / 2.0).astype(int),
    )
    output = np.zeros(n, dtype=np.float32)
    support = np.zeros(n, dtype=np.int32)
    global_support = np.zeros(n, dtype=np.int32)
    fallback = np.empty(n, dtype=object)
    max_resolution = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    decision_series = pd.Series(decision)
    available_series = pd.Series(available)
    order = np.argsort(decision_series.to_numpy(dtype="datetime64[ns]"), kind="stable")
    ordered_decision = decision_series.to_numpy(dtype="datetime64[ns]")[order]
    start = 0
    while start < n:
        stop = start + 1
        while stop < n and ordered_decision[stop] == ordered_decision[start]:
            stop += 1
        positions = order[start:stop]
        cutoff = decision_series.iloc[positions[0]]
        prior = available_series < cutoff
        prior_count = int(prior.sum())
        global_support[positions] = prior_count
        if prior_count < int(config.min_global_rows):
            fallback[positions] = "neutral_no_prior_resolved_support"
        else:
            prior_net = net[prior]
            global_mean = float(prior_net.mean())
            max_resolution[positions] = available_series[prior].max().to_datetime64()
            for value in np.unique(bucket[positions]):
                target_positions = positions[bucket[positions] == value]
                in_bin = prior.to_numpy() & (bucket == value)
                count = int(in_bin.sum())
                support[target_positions] = count
                if count == 0:
                    output[target_positions] = global_mean
                    fallback[target_positions] = "global_prior_fallback_empty_bin"
                else:
                    bin_mean = float(net[in_bin].mean())
                    shrink = float(config.bin_shrink_rows)
                    output[target_positions] = (
                        count * bin_mean + shrink * global_mean
                    ) / (count + shrink)
                    fallback[target_positions] = "shrunk_bin_prior_resolved"
        start = stop
    return output, pd.DataFrame({
        "r3_score_bin": bucket.astype(np.int16),
        "prior_resolved_global_support": global_support,
        "prior_resolved_bin_support": support,
        "value_map_fallback": fallback,
        "value_map_max_label_available_ts": pd.to_datetime(max_resolution, utc=True),
    })


@pytest.mark.parametrize("seed", [1, 7, 41])
def test_linear_event_sweep_matches_quadratic_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 240
    # Repeated, shuffled decisions exercise group stability independently of
    # input order.  Availability can coincide exactly with a later decision,
    # which must remain excluded at that cutoff.
    decision = pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(
        rng.integers(0, 48, size=n), unit="h"
    )
    available = decision + pd.to_timedelta(rng.integers(1, 15, size=n), unit="h")
    score = rng.uniform(-1.0, 1.0, size=n)
    net = rng.normal(15.0, 280.0, size=n)
    permutation = rng.permutation(n)
    decision = pd.DatetimeIndex(decision[permutation])
    available = pd.DatetimeIndex(available[permutation])
    score = score[permutation]
    net = net[permutation]
    config = PrequentialR3ValueMapConfig(
        side="long", bins=9, min_global_rows=11, bin_shrink_rows=17.0
    )
    expected, expected_audit = _quadratic_reference(
        net=net, decision=decision, available=available, score=score, config=config
    )
    actual, actual_audit, _ = prequential_same_side_r3_value_map(
        exact_net_bps=net, decision_timestamps=decision,
        label_available_timestamps=available, side="long", score=score,
        config=config,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-5)
    for column in (
        "r3_score_bin", "prior_resolved_global_support",
        "prior_resolved_bin_support", "value_map_fallback",
        "value_map_max_label_available_ts",
    ):
        pd.testing.assert_series_equal(
            actual_audit[column].reset_index(drop=True),
            expected_audit[column].reset_index(drop=True),
            check_names=False,
        )


def test_label_resolving_exactly_at_decision_is_not_prior_support() -> None:
    decision = pd.to_datetime([
        "2024-01-01T00:00:00Z", "2024-01-01T13:00:00Z",
        "2024-01-01T14:00:00Z",
    ])
    available = pd.to_datetime([
        "2024-01-01T13:00:00Z", "2024-01-02T02:00:00Z",
        "2024-01-02T03:00:00Z",
    ])
    _, audit, _ = prequential_same_side_r3_value_map(
        exact_net_bps=[100.0, -20.0, 40.0], decision_timestamps=decision,
        label_available_timestamps=available, side="short", score=[0.8, 0.1, -0.2],
        config=PrequentialR3ValueMapConfig(side="short", min_global_rows=1),
    )
    assert audit.prior_resolved_global_support.tolist() == [0, 0, 1]


def test_large_value_map_has_production_scale_linear_sanity() -> None:
    # Forty candidates per hourly decision approximates the production
    # candidate density.  The old all-rows-per-timestamp implementation scans
    # roughly 400 million row positions here; the event sweep is linear after
    # its two stable sorts.
    n = 100_000
    decision = pd.Timestamp("2022-01-01", tz="UTC") + pd.to_timedelta(
        np.arange(n) // 40, unit="h"
    )
    available = decision + pd.Timedelta(hours=13)
    score = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    net = np.sin(np.arange(n, dtype=np.float64) / 37.0) * 300.0
    started = perf_counter()
    mapped, audit, _ = prequential_same_side_r3_value_map(
        exact_net_bps=net, decision_timestamps=decision,
        label_available_timestamps=available, side="long", score=score,
        config=PrequentialR3ValueMapConfig(side="long"),
    )
    elapsed = perf_counter() - started
    assert len(mapped) == len(audit) == n
    assert audit.prior_resolved_global_support.iloc[-1] > 99_000
    # Deliberately generous for shared CI machines while still separating the
    # event sweep from the former quadratic implementation by orders of magnitude.
    assert elapsed < 10.0
