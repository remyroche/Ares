import json

import pandas as pd

from extreme_price_movements.data_store import PartitionedOHLCVStore
from scripts.replay_live_signal_predictions import (
    _live_feature_cache_symbols_for_end,
    _load_panel,
    _load_recent_decisions,
    _parity_failures,
)


def test_load_recent_decisions_filters_rank_source_and_start(tmp_path):
    ledger_path = tmp_path / "prediction_ledger.parquet"
    trades_path = tmp_path / "missing_trades.csv"
    ledger = pd.DataFrame(
        {
            "decision_ts": pd.to_datetime(
                [
                    "2026-05-15T10:00:00Z",
                    "2026-05-15T11:00:00Z",
                    "2026-05-15T12:00:00Z",
                ],
                utc=True,
            ),
            "signal_bar_ts": pd.to_datetime(
                [
                    "2026-05-15T09:00:00Z",
                    "2026-05-15T10:00:00Z",
                    "2026-05-15T11:00:00Z",
                ],
                utc=True,
            ),
            "symbol": ["AAAUSDC", "BBB/USDC", "CCC/USDC"],
            "side": ["long", "long", "short"],
            "strategy_id": ["long_demo", "long_demo", "short_demo"],
            "rank_score_source": [
                "historical_meta_oof_percentile",
                "policy_rank_reference_percentile",
                "policy_rank_reference_percentile",
            ],
        }
    )
    ledger.to_parquet(ledger_path, index=False)

    decisions = _load_recent_decisions(
        ledger_path=ledger_path,
        trades_path=trades_path,
        max_rows=10,
        decision_start="2026-05-15T10:30:00Z",
        require_rank_source="policy_rank_reference_percentile",
    )

    assert decisions["symbol"].tolist() == ["BBB/USDC", "CCC/USDC"]
    assert set(decisions["rank_score_source"]) == {"policy_rank_reference_percentile"}


def test_parity_failures_require_live_values_and_tolerance():
    frame = pd.DataFrame(
        {
            "live_base_pred": [0.1],
            "live_meta_pred": [0.2],
            "live_calibrated_score": [0.2],
            "live_policy_rank_pct": [0.6],
            "replay_policy_rank_reference_n": [10],
            "base_pred_delta": [0.0],
            "meta_pred_delta": [0.02],
            "calibrated_score_delta": [0.0],
            "rank_percentile_delta": [0.0],
        }
    )

    failures = _parity_failures(
        frame,
        tolerance=0.01,
        require_policy_rank_reference=True,
        require_live_values=True,
    )

    assert failures == ["meta_pred_delta_max_abs=0.02"]


def test_parity_failures_detect_missing_policy_reference_and_live_fields():
    frame = pd.DataFrame(
        {
            "live_meta_pred": [0.2],
            "replay_policy_rank_reference_n": [0],
            "meta_pred_delta": [0.0],
        }
    )

    failures = _parity_failures(
        frame,
        tolerance=0.01,
        require_policy_rank_reference=True,
        require_live_values=True,
    )

    assert "missing_policy_rank_reference_rows=1" in failures
    assert "missing_live_base_pred_rows=1" in failures
    assert "missing_live_calibrated_score_rows=1" in failures
    assert "missing_live_policy_rank_pct_rows=1" in failures


def test_live_feature_cache_symbols_prefers_smallest_matching_universe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "cache" / "inference_live_features" / "run_a"
    small = root / "small"
    large = root / "large"
    small.mkdir(parents=True)
    large.mkdir(parents=True)
    ts = "2026-05-15T23:45:00+00:00"
    small_symbols = [f"S{i}/USDC" for i in range(30)]
    large_symbols = [f"S{i}/USDC" for i in range(45)]
    (small / "meta.json").write_text(json.dumps({"end_ts": ts, "symbols": small_symbols}))
    (large / "meta.json").write_text(json.dumps({"end_ts": ts, "symbols": large_symbols}))

    symbols = _live_feature_cache_symbols_for_end(
        tmp_path,
        run_id="run_a",
        end_ts=pd.Timestamp(ts),
    )

    assert symbols == sorted(small_symbols)


def test_load_panel_preserves_perp_ohlcv_extras_and_overlays_microdata(tmp_path):
    symbol = "AAA/USD:USD"
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2026-05-15 10:00", tz="UTC")],
        name="ts",
    )
    store = PartitionedOHLCVStore(str(tmp_path), timeframe="1h")
    store.save_partitioned(
        symbol,
        pd.DataFrame(
            {
                "open": [1.0],
                "high": [1.1],
                "low": [0.9],
                "close": [1.0],
                "volume": [100.0],
                "mark_open": [1.01],
                "mark_price": [1.02],
                "index_price": [1.00],
            },
            index=idx,
        ),
    )
    funding_dir = tmp_path / "funding_hourly"
    funding_dir.mkdir()
    pd.DataFrame(
        {
            "mark_price": [1.03],
            "index_price": [1.00],
            "funding_rate": [0.0001],
        },
        index=idx,
    ).to_parquet(funding_dir / "AAA_USD_USD.parquet")

    panel = _load_panel(
        data_root=tmp_path,
        symbols=[symbol],
        start_ts=idx[0],
        end_ts=idx[0],
    )

    import math
    assert math.isclose(float(panel["mark_open"].loc[idx[0], symbol]), 1.01, rel_tol=1e-5)
    assert math.isclose(float(panel["mark_price"].loc[idx[0], symbol]), 1.03, rel_tol=1e-5)
    assert math.isclose(float(panel["funding_rate"].loc[idx[0], symbol]), 0.0001, rel_tol=1e-5)
