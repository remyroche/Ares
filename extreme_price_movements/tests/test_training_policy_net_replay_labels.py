from __future__ import annotations

import os

import numpy as np
import pandas as pd

from extreme_price_movements import simple_policy_optimiser as spo
from extreme_price_movements import training


def test_policy_net_replay_labels_use_positional_rows_and_contract_exchange(monkeypatch):
    captured = {"make_exchange": None, "fetch_exchange": None, "apply_exchange": None}

    def fake_make_policy_replay_store(data_root, market_mode):
        captured["make_exchange"] = os.environ.get("EPM_EXCHANGE")
        return {"data_root": data_root, "market_mode": market_mode}

    def fake_fetch_policy_paths(rows, store):
        captured["fetch_exchange"] = os.environ.get("EPM_EXCHANGE")
        assert list(rows.index) == [0, 1, 2]
        assert list(rows["symbol"]) == ["BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD"]
        return (np.arange(len(rows), dtype=float),)

    def fake_apply_delayed_entry_execution_model(rows, paths, *, data_root, market_mode):
        captured["apply_exchange"] = os.environ.get("EPM_EXCHANGE")
        assert list(rows.index) == [0, 1, 2]
        assert data_root == "data_perp"
        assert market_mode == "perps"
        return rows.copy(), paths

    def fake_simulate_and_score(rows_exec, *paths, **kwargs):
        assert list(rows_exec.index) == [0, 1, 2]
        return {
            "selected_mask": np.array([True, True, True]),
            "raw_gains": np.array([1.0, -2.0, 3.0]),
            "sizes": np.array([100.0, 100.0, 100.0]),
        }

    monkeypatch.setattr(spo, "_make_policy_replay_store", fake_make_policy_replay_store)
    monkeypatch.setattr(spo, "_fetch_policy_paths", fake_fetch_policy_paths)
    monkeypatch.setattr(
        spo,
        "_apply_delayed_entry_execution_model",
        fake_apply_delayed_entry_execution_model,
    )
    monkeypatch.setattr(spo, "simulate_and_score", fake_simulate_and_score)
    monkeypatch.setattr(training._policy_replay_store_for_labels, "_cache", {}, raising=False)
    monkeypatch.setenv("EPM_EXCHANGE", "binance")

    timestamps = pd.Series(
        pd.date_range("2026-03-01", periods=3, freq="h", tz="UTC"),
        index=[5000, 5001, 5002],
    )
    symbols = pd.Series(
        ["BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD"],
        index=[5000, 5001, 5002],
    )
    vals, stats = training._materialize_policy_net_replay_labels(
        timestamps=timestamps,
        symbols=symbols,
        side="long",
        barrier_pct=np.array([0.01, 0.02, 0.03], dtype=np.float32),
        cfg={
            "data_root": "data_perp",
            "market_mode": "perps",
            "training_exchange_contract": {
                "exchange_id": "krakenfutures",
                "market_mode": "perps",
            },
            "label_policy_net_replay_min_coverage": 1.0,
        },
        label="unit_test",
    )

    assert captured == {
        "make_exchange": "krakenfutures",
        "fetch_exchange": "krakenfutures",
        "apply_exchange": "krakenfutures",
    }
    assert os.environ.get("EPM_EXCHANGE") == "binance"
    assert stats["exchange_id"] == "krakenfutures"
    np.testing.assert_allclose(vals, np.array([0.01, -0.02, 0.03], dtype=np.float32))


def test_training_soft_label_dispatches_s14_policy_path_blend(monkeypatch):
    monkeypatch.delenv("EPM_LABEL_ABLATION_MODE", raising=False)
    monkeypatch.delenv("EPM_LABEL_WEIGHT_RECIPE", raising=False)
    monkeypatch.delenv("EPM_LABEL_WEIGHT_USE_BEST_DEFAULT", raising=False)

    df = pd.DataFrame(
        {
            "__u_policy_net__": [-0.006, 0.0, 0.006, 0.018],
            "__mfe_ret__": [0.010, 0.018, 0.022, 0.028],
            "__mae_ret__": [-0.004, -0.026, -0.005, -0.002],
            "__barrier_pct__": [0.010, 0.010, 0.010, 0.010],
            "__bars_to_mfe__": [2.0, 4.0, 3.0, 2.0],
            "__y_ret__": [0.004, -0.012, 0.011, 0.022],
            "__y_outcome__": [1.0, 0.0, 1.0, 1.0],
        }
    )

    soft, stats = training._build_mfe_mae_soft_label(
        df,
        np.array([0, 0, 1, 1], dtype=np.float32),
        cfg={
            "label_ablation_mode": "s14_policy_net_path_blend",
            "policy_net_label_center": 0.0,
            "policy_net_label_temperature": 0.012,
        },
        label="training_s14_dispatch",
    )

    plain_policy = 1.0 / (1.0 + np.exp(-df["__u_policy_net__"].to_numpy(dtype=float) / 0.012))
    assert stats["target_mode"] == "policy_net_path_blend"
    assert stats["bad_path_rate"] > 0.0
    assert soft[1] < plain_policy[1]
    assert soft[3] > soft[2] > soft[0]
