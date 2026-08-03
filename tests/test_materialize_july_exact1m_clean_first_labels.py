from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_july_exact1m_clean_first_labels import (
    HORIZON_MINUTES,
    _load_historical_feature_source,
    build_exact_clean_first_labels,
)


def _paths() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    open_ = np.full((1, HORIZON_MINUTES), 100.0)
    return open_, open_.copy(), open_.copy()


def _labels(
    high: np.ndarray,
    low: np.ndarray,
    *,
    side: float = 1.0,
    entry_spread: float = 0.0,
    exit_spread: float = 0.0,
) -> pd.DataFrame:
    open_, _, _ = _paths()
    return build_exact_clean_first_labels(
        open_, high, low,
        atr_fraction=np.array([0.01]),
        side_sign=np.array([side]),
        entry_half_spread_bps=np.array([entry_spread]),
        exit_half_spread_bps=np.array([exit_spread]),
        decision_utc=pd.DatetimeIndex([pd.Timestamp("2026-07-20T00:00:00Z")]),
    )


def test_exact_first_favorable_timestamp_and_12h_resolution() -> None:
    _, high, low = _paths()
    high[0, 7] = 101.6
    labels = _labels(high, low).iloc[0]
    assert labels["__soft_tb_first_event__"] == "favorable_first"
    assert labels["__soft_tb_first_favorable_minute__"] == 7.0
    assert labels["__soft_tb_first_favorable_utc__"] == pd.Timestamp("2026-07-20T00:07:00Z")
    assert labels["__soft_tb_label_end_utc__"] == pd.Timestamp("2026-07-20T12:00:00Z")
    assert labels["__soft_tb_upper_return__"] == pytest.approx(0.015)
    assert labels["__soft_tb_lower_return__"] == pytest.approx(0.01)


def test_same_minute_conflict_is_adverse_deterministically() -> None:
    _, high, low = _paths()
    high[0, 3] = 101.6
    low[0, 3] = 99.0
    labels = _labels(high, low).iloc[0]
    assert labels["__soft_tb_first_event__"] == "adverse_first_or_conflict"
    assert labels["__soft_tb_order_ambiguous__"] == 1
    assert labels["__soft_tb_first_favorable_minute__"] == 3.0
    assert labels["__soft_tb_first_adverse_minute__"] == 3.0


def test_short_side_mirrors_barriers_and_timeout_is_exhaustive() -> None:
    _, high, low = _paths()
    low[0, 9] = 98.4
    labels = _labels(high, low, side=-1.0).iloc[0]
    assert labels["__soft_tb_first_event__"] == "favorable_first"
    assert labels["__soft_tb_first_favorable_minute__"] == 9.0
    _, high, low = _paths()
    timeout = _labels(high, low, side=-1.0).iloc[0]
    assert timeout["__soft_tb_first_event__"] == "timeout"
    assert pd.isna(timeout["__soft_tb_first_favorable_utc__"])
    assert pd.isna(timeout["__soft_tb_first_adverse_utc__"])


def test_rejects_non_12h_path() -> None:
    with pytest.raises(ValueError, match="720"):
        build_exact_clean_first_labels(
            np.ones((1, 3)), np.ones((1, 3)), np.ones((1, 3)),
            atr_fraction=np.array([0.01]), side_sign=np.array([1.0]),
            entry_half_spread_bps=np.array([0.0]), exit_half_spread_bps=np.array([0.0]),
            decision_utc=pd.DatetimeIndex([pd.Timestamp("2026-07-20T00:00:00Z")]),
        )


def test_executable_spread_adjustment_matches_cross_era_barrier_semantics() -> None:
    """A raw 2% high is no longer a clean 1.5% move after executable spreads."""
    _, high, low = _paths()
    high[0, 5] = 102.0
    raw = _labels(high, low).iloc[0]
    executable = _labels(high, low, entry_spread=50.0, exit_spread=50.0).iloc[0]
    assert raw["__soft_tb_first_event__"] == "favorable_first"
    assert executable["__soft_tb_first_event__"] == "timeout"
    assert executable["__soft_tb_executable_entry__"] == pytest.approx(100.5)


def test_historical_source_mode_binds_feature_atr_to_exact_policy_spreads(tmp_path) -> None:
    identity = {
        "__ts__": [pd.Timestamp("2026-05-01T00:00:00Z")],
        "__symbol__": ["X/USD:USD"],
        "side_name": ["short"],
        "candidate_id": ["x"],
    }
    decision = pd.Timestamp("2026-05-01T01:00:00Z")
    feature = pd.DataFrame({
        **identity,
        "oof_entry_atr_fraction": [0.01],
        "execution_decision_utc": [decision],
        "execution_label_end_utc": [decision + pd.Timedelta(hours=12)],
        "execution_cost_return": [0.001],
    })
    policy = pd.DataFrame({
        **identity,
        "execution_decision_utc": [decision],
        "execution_label_end_utc": [decision + pd.Timedelta(hours=12)],
        "execution_entry_half_spread_bps": [3.0],
        "execution_exit_half_spread_bps": [4.0],
        "execution_cost_return": [0.001],
    })
    feature_path, policy_path = tmp_path / "feature.parquet", tmp_path / "policy.parquet"
    feature.to_parquet(feature_path, index=False)
    policy.to_parquet(policy_path, index=False)
    actual = _load_historical_feature_source(feature_path, policy_path)
    assert len(actual) == 1
    assert actual.loc[0, "__path_auxiliary_atr_fraction__"] == pytest.approx(0.01)
    assert actual.loc[0, "execution_entry_half_spread_bps"] == pytest.approx(3.0)
