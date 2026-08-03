from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import materialize_canonical_pre_exit_capture_labels as capture


def test_path_symbol_normalization_is_explicit_and_bounded() -> None:
    values = capture.canonicalize_path_symbol(
        pd.Series(["BTC/USD:USD", "ETH_USD:USD"])
    )
    assert values.tolist() == ["BTC_USD:USD", "ETH_USD:USD"]


def test_capture_uses_only_prices_through_deployed_exit() -> None:
    open_ = np.array([[100.0, 100.0, 100.0]])
    high = np.array([[101.0, 102.0, 120.0]])
    low = np.array([[99.0, 99.0, 99.0]])
    close = np.array([[100.0, 100.0, 100.0]])
    result = capture.capture_columns(
        open_=open_,
        high=high,
        low=low,
        close=close,
        side_name=np.array(["long"]),
        entry_spread_bps=np.array([0.0]),
        exit_spread_bps=np.array([0.0]),
        exit_minute=np.array([1.0]),
        atr_fraction=np.array([0.01]),
        gross=np.array([0.015]),
        cost=np.array([0.01]),
        net=np.array([0.005]),
    )
    assert np.allclose(result["pre_exit_mfe_return"], 0.02)
    assert result["target_pre_exit_economic_opportunity"].tolist() == [1]
    assert np.allclose(result["target_pre_exit_economic_capture_ratio"], 0.5)


def test_short_capture_is_side_relative_and_cost_aware() -> None:
    open_ = np.array([[100.0, 100.0]])
    high = np.array([[101.0, 101.0]])
    low = np.array([[98.0, 90.0]])
    close = np.array([[99.0, 95.0]])
    result = capture.capture_columns(
        open_=open_,
        high=high,
        low=low,
        close=close,
        side_name=np.array(["short"]),
        entry_spread_bps=np.array([0.0]),
        exit_spread_bps=np.array([0.0]),
        exit_minute=np.array([0.0]),
        atr_fraction=np.array([0.01]),
        gross=np.array([0.015]),
        cost=np.array([0.01]),
        net=np.array([0.005]),
    )
    assert np.allclose(result["pre_exit_mfe_return"], 0.02)
    assert result["target_pre_exit_meaningful_mfe"].tolist() == [1]
    assert result["target_pre_exit_capture_net_positive"].tolist() == [1]


def test_gross_above_pre_exit_mfe_is_excluded_from_capture_training() -> None:
    result = capture.capture_columns(
            open_=np.array([[100.0]]),
            high=np.array([[101.0]]),
            low=np.array([[99.0]]),
            close=np.array([[100.0]]),
            side_name=np.array(["long"]),
            entry_spread_bps=np.array([0.0]),
            exit_spread_bps=np.array([0.0]),
            exit_minute=np.array([0.0]),
            atr_fraction=np.array([0.01]),
            gross=np.array([0.02]),
            cost=np.array([0.01]),
            net=np.array([0.01]),
    )
    assert result["pre_exit_path_policy_parity"].tolist() == [0]
    assert result["target_pre_exit_capture_valid"].tolist() == [0]
    assert np.isnan(result["target_pre_exit_capture_ratio"]).all()
