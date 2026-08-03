from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_kraken_inverse_hourly_feature_candidates import (
    EVIDENCE_SCOPE,
    POPULATION_LINEAGE,
    PRODUCT_LINEAGE,
    PRODUCTS,
    _hourly_from_exact_minutes,
    build_feature_candidates,
)


def _minutes(start: pd.Timestamp, end: pd.Timestamp, offset: float) -> pd.DataFrame:
    index = pd.date_range(start, end, freq="1min", inclusive="left", tz="UTC")
    step = np.arange(len(index), dtype=float)
    close = 100.0 + offset + 0.002 * step + 0.2 * np.sin(step / 37.0)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": 10.0 + (step % 17),
        },
        index=index,
    )


def test_hourly_bars_are_right_labelled_and_require_every_minute() -> None:
    start = pd.Timestamp("2022-01-01T00:00:00Z")
    end = start + pd.Timedelta(hours=3)
    minute = _minutes(start - pd.Timedelta(hours=1), end, 0.0)
    hourly = _hourly_from_exact_minutes(
        minute, required_start=start, required_end=end
    )
    assert hourly.index.tolist() == [
        start,
        start + pd.Timedelta(hours=1),
        start + pd.Timedelta(hours=2),
        start + pd.Timedelta(hours=3),
    ]
    first_window = minute.loc[
        (minute.index >= start - pd.Timedelta(hours=1))
        & (minute.index < start)
    ]
    assert hourly.loc[start, "close"] == pytest.approx(first_window["close"].iloc[-1])

    broken = minute.drop(minute.index[15])
    with pytest.raises(ValueError, match="uninterrupted"):
        _hourly_from_exact_minutes(
            broken, required_start=start, required_end=end
        )


def test_feature_candidates_are_paired_causal_and_population_complete() -> None:
    start = pd.Timestamp("2022-04-01T00:00:00Z")
    end = start + pd.Timedelta(hours=48)
    warmup_days = 10
    minute_start = start - pd.Timedelta(days=warmup_days) - pd.Timedelta(hours=1)
    minute_end = end + pd.Timedelta(hours=12)
    sources = {
        symbol: _minutes(minute_start, minute_end, float(position))
        for position, symbol in enumerate(PRODUCTS)
    }
    candidates, hourly = build_feature_candidates(
        sources,
        start=start,
        end_exclusive=end,
        warmup_days=warmup_days,
        cadence_hours=1,
    )
    assert len(candidates) == 48 * len(PRODUCTS) * 2
    assert len(hourly) == len(PRODUCTS) * (
        warmup_days * 24 + 48 + 12 + 1
    )
    assert candidates.groupby(["__ts__", "__symbol__"])["side_name"].nunique().eq(2).all()
    assert candidates.groupby("side_name").size().to_dict() == {
        "long": 48 * len(PRODUCTS),
        "short": 48 * len(PRODUCTS),
    }
    assert set(candidates["archetype_policy_key"]) == {"parent"}
    assert set(candidates["policy_archetype_assignment_source"]) == {
        "explicit_deployed_side_parent_inverse_grid"
    }
    assert set(candidates["evidence_scope"]) == {EVIDENCE_SCOPE}
    assert set(candidates["candidate_population_lineage"]) == {
        POPULATION_LINEAGE
    }
    assert set(candidates["source_product_lineage"]) == {PRODUCT_LINEAGE}
    assert not candidates["bootstrap_barrier_data_acquisition_only"].any()
    assert candidates["__barrier_pct__"].between(0.005, 0.05).all()
    assert candidates["selected_for_monitor"].all()
    assert set(candidates["product_id"]) == set(PRODUCTS.values())
    assert candidates["source_product_id"].equals(candidates["product_id"])
    assert set(candidates["source_contract_family"]) == {"PI"}
    assert candidates.filter(like="transition_raw__").shape[1] >= 20
    assert np.isfinite(candidates["base_score"]).all()


def test_future_mutation_does_not_change_prior_signal_features() -> None:
    start = pd.Timestamp("2022-04-01T00:00:00Z")
    end = start + pd.Timedelta(hours=24)
    warmup_days = 10
    minute_start = start - pd.Timedelta(days=warmup_days) - pd.Timedelta(hours=1)
    minute_end = end + pd.Timedelta(hours=12)
    sources = {
        symbol: _minutes(minute_start, minute_end, float(position))
        for position, symbol in enumerate(PRODUCTS)
    }
    original, _ = build_feature_candidates(
        sources,
        start=start,
        end_exclusive=end,
        warmup_days=warmup_days,
        cadence_hours=1,
    )
    changed = {symbol: frame.copy() for symbol, frame in sources.items()}
    future = changed["BTC/USD:BTC"].index >= start + pd.Timedelta(hours=12)
    changed["BTC/USD:BTC"].loc[future, ["open", "high", "low", "close"]] *= 2.0
    mutated, _ = build_feature_candidates(
        changed,
        start=start,
        end_exclusive=end,
        warmup_days=warmup_days,
        cadence_hours=1,
    )
    columns = [
        "base_score",
        "__barrier_pct__",
        "ret_24h",
        "market_median_ret_24h",
        "transition_raw__market_median_rv_24h__z_72h",
    ]
    before = original.loc[original["__ts__"] <= start + pd.Timedelta(hours=12)]
    after = mutated.loc[mutated["__ts__"] <= start + pd.Timedelta(hours=12)]
    pd.testing.assert_frame_equal(
        before.loc[:, columns].reset_index(drop=True),
        after.loc[:, columns].reset_index(drop=True),
    )
