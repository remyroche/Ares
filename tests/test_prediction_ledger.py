import pandas as pd
import numpy as np
import pytest
from extreme_price_movements.inference.prediction_ledger import PredictionLedger

def test_prediction_ledger_mark_resolved(tmp_path):
    ledger = PredictionLedger(path=tmp_path / "ledger.parquet")

    # Initial data
    initial_data = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=3, tz="UTC"),
        "signal_bar_ts": pd.date_range("2023-01-01", periods=3, tz="UTC"),
        "symbol": ["BTC", "ETH", "SOL"],
        "side": [1, -1, 1],
        "strategy_id": ["A", "B", "A"],
        "meta_head_hash": ["h1", "h2", "h3"],
        "value": [10.0, 20.0, 30.0],
        "status": ["pending", "open", "closed"]
    })
    ledger._write_atomic(initial_data)

    # Updates: update ETH, add new BNB
    updates = pd.DataFrame({
        "timestamp": [pd.Timestamp("2023-01-02 00:00:00", tz="UTC"), pd.Timestamp("2023-01-04 00:00:00", tz="UTC")],
        "signal_bar_ts": [pd.Timestamp("2023-01-02 00:00:00", tz="UTC"), pd.Timestamp("2023-01-04 00:00:00", tz="UTC")],
        "symbol": ["ETH", "BNB"],
        "side": [-1, 1],
        "strategy_id": ["B", "C"],
        "meta_head_hash": ["h2", "h4"],
        "value": [25.0, 40.0],
        "status": ["closed", "open"],
        "new_col": ["a", "b"]
    })

    ledger.mark_resolved(updates)

    result = ledger._read()

    # The ledger should have 4 rows
    assert len(result) == 4

    # ETH should be updated
    eth_row = result[result["symbol"] == "ETH"].iloc[0]
    assert eth_row["value"] == 25.0
    assert eth_row["status"] == "closed"
    assert eth_row["new_col"] == "a"

    # BTC should remain the same
    btc_row = result[result["symbol"] == "BTC"].iloc[0]
    assert btc_row["value"] == 10.0
    assert btc_row["status"] == "pending"
    assert pd.isna(btc_row["new_col"])

    # BNB should be added
    bnb_row = result[result["symbol"] == "BNB"].iloc[0]
    assert bnb_row["value"] == 40.0
    assert bnb_row["status"] == "open"
    assert bnb_row["new_col"] == "b"

def test_mark_resolved_with_empty_updates(tmp_path):
    ledger = PredictionLedger(path=tmp_path / "ledger.parquet")
    initial_data = pd.DataFrame({"timestamp": [1], "symbol": ["BTC"]})
    ledger._write_atomic(initial_data)

    ledger.mark_resolved(pd.DataFrame())

    result = ledger._read()
    assert len(result) == 1
    assert result["symbol"].iloc[0] == "BTC"
