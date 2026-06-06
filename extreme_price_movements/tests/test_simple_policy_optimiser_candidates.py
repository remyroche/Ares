import numpy as np
import pandas as pd
import pytest

from extreme_price_movements import simple_policy_optimiser as spo


def test_policy_candidate_export_includes_entry_slippage_in_friction(monkeypatch):
    def fake_simulate_and_score(*args, **kwargs):
        return {
            "selected_mask": np.array([True]),
            "raw_gains": np.array([0.009], dtype=np.float64),
            "gross_gains": np.array([0.010], dtype=np.float64),
            "sizes": np.array([1.0], dtype=np.float64),
            "exit_bars": np.array([2], dtype=np.int32),
            "exit_reason": np.array(["trailing"], dtype=object),
        }

    monkeypatch.setattr(spo, "simulate_and_score", fake_simulate_and_score)
    rows = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01 00:00:00", tz="UTC")],
            "symbol": ["BTC/USD:USD"],
            "side": [1.0],
            "rank_pct": [0.91],
            "calibrated_score": [0.8],
            "barrier_pct": [0.02],
            "theoretical_entry_price": [100.0],
            "entry_gap_bps": [15.0],
            "entry_slippage_proxy_bps": [4.5],
        }
    )
    paths = (
        np.array([[100.15, 101.0]], dtype=np.float32),
        np.array([[101.0, 102.0]], dtype=np.float32),
        np.array([[99.0, 100.0]], dtype=np.float32),
        np.array([[100.5, 101.0]], dtype=np.float32),
    )

    out = spo._build_simple_policy_candidate_rows(
        strategy_id="long_test",
        df_top=rows,
        paths=paths,
        cost_pct=0.001,
        best_params={"sl_mult": 1.0, "trailing_activation_mult": 1.0},
        best_size_power=1.0,
        base_strategy_threshold=0.75,
        market_mode="perps",
    )

    assert out["slippage_bps"].iloc[0] == 4.5
    assert out["orderbook_slippage_bps"].iloc[0] == 4.5
    assert out["expected_spread_bps"].iloc[0] == pytest.approx(
        spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS
    )
    assert out["spread_cost_bps"].iloc[0] == pytest.approx(
        spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS / 2.0
    )
    assert out["net_return_before_spread"].iloc[0] == pytest.approx(0.009)
    assert out["net_return"].iloc[0] == pytest.approx(
        0.009 - (spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS / 2.0) / 10_000.0
    )
    assert out["expected_friction_bps"].iloc[0] == pytest.approx(
        14.5 + spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS / 2.0
    )
    assert out["entry_gap_bps"].iloc[0] == 15.0


def test_policy_band_metrics_include_execution_cost_assumptions():
    rows = pd.DataFrame(
        {
            "rank_pct": [0.82, 0.87],
            "net_return": [0.003, 0.001],
            "gross_return": [0.008, 0.006],
            "expected_spread_bps": [spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS] * 2,
            "expected_half_spread_bps": [
                spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS / 2.0
            ]
            * 2,
            "spread_cost_bps": [spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS / 2.0]
            * 2,
            "entry_delay_target_minutes": [float(spo.POLICY_DELAYED_ENTRY_MINUTES)] * 2,
            "entry_delay_actual_minutes": [5.0, 6.0],
            "entry_slippage_proxy_bps": [4.0, 8.0],
            "fees_bps": [20.0, 20.0],
            "expected_friction_bps": [
                20.0 + spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS / 2.0 + 4.0,
                20.0 + spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS / 2.0 + 8.0,
            ],
            "entry_execution_source": [
                "delayed_1m_intraminute_proxy",
                "delayed_1m_intraminute_proxy",
            ],
        }
    )

    metrics = spo._candidate_band_metrics(
        rows,
        rank_col="rank_pct",
        group_name="strategy",
        strategy_id="long_test",
        band_lo=0.8,
        band_hi=0.85,
        selection_type="local_band",
    )

    assert spo.POLICY_DELAYED_ENTRY_MINUTES == 5
    assert metrics["mean_expected_spread_bps"] == pytest.approx(
        spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS
    )
    assert metrics["mean_spread_cost_bps"] == pytest.approx(
        spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS / 2.0
    )
    assert metrics["mean_entry_delay_target_minutes"] == pytest.approx(5.0)
    assert metrics["mean_entry_delay_actual_minutes"] == pytest.approx(5.5)
    assert metrics["mean_entry_slippage_proxy_bps"] == pytest.approx(6.0)
    assert metrics["mean_expected_friction_bps"] == pytest.approx(
        20.0 + spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS / 2.0 + 6.0
    )
    assert metrics["entry_execution_source_counts"] == {
        "delayed_1m_intraminute_proxy": 2
    }
