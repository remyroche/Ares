from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.execution_ev_labels import (
    REASON_FULL_STOP,
    REASON_TIMEOUT,
    REASON_TRAILING,
    ExecutionLabelGeometry,
    simulate_execution_ev_12h,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_execution_ev_12h_labels",
    ROOT / "scripts" / "materialize_execution_ev_12h_labels.py",
)
assert SPEC and SPEC.loader
materializer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(materializer)


def _geometry(**overrides: object) -> ExecutionLabelGeometry:
    values: dict[str, object] = {
        "sl_mult": 1.0,
        "trailing_activation_mult": 1.0,
        "trailing_activation_cap_pct": 0.0,
        "trailing_activation_decay_half_life_minutes": 0.0,
        "trailing_activation_decay_start_minutes": 0.0,
        "trailing_activation_min_mult": 1.0,
        "trailing_power": 1.5,
        "trailing_squash_divisor": 2.0,
        "giveback_beta": 0.5,
        "adverse_exit_enabled": False,
        "adverse_exit_min_mae_atr": 1.0,
        "adverse_exit_min_speed_per_15m": 0.3,
        "adverse_exit_theta": 1e9,
        "adverse_exit_fast_minutes": 0.0,
        "adverse_exit_max_mfe_atr": 0.25,
    }
    values.update(overrides)
    return ExecutionLabelGeometry.from_mapping(values)


def test_timeout_exits_at_last_close_and_deducts_cost_once() -> None:
    opens = np.full((1, 12), 100.0)
    highs = np.full((1, 12), 100.4)
    lows = np.full((1, 12), 99.6)
    closes = np.full((1, 12), 101.0)
    params = _geometry(sl_mult=4.0, trailing_activation_mult=5.0).vector()
    gross, net, reason, exit_bar, _, _ = simulate_execution_ev_12h(
        opens, highs, lows, closes, np.array([1.0]), np.array([0.01]),
        np.array([0.003]), params, params,
    )
    assert reason[0] == REASON_TIMEOUT
    assert exit_bar[0] == 11
    assert gross[0] == pytest.approx(0.01)
    assert net[0] == pytest.approx(0.007)


def test_pessimistic_full_stop_precedes_same_bar_favorable_move() -> None:
    opens = np.full((1, 12), 100.0)
    highs = np.full((1, 12), 105.0)
    lows = np.full((1, 12), 98.0)
    closes = np.full((1, 12), 100.0)
    params = _geometry().vector()
    _, net, reason, exit_bar, _, _ = simulate_execution_ev_12h(
        opens, highs, lows, closes, np.array([1.0]), np.array([0.01]),
        np.array([0.003]), params, params,
    )
    assert reason[0] == REASON_FULL_STOP
    assert exit_bar[0] == 0
    assert net[0] == pytest.approx(-0.013)


def test_trailing_uses_prior_completed_candle_mfe() -> None:
    opens = np.full((1, 12), 100.0)
    highs = np.full((1, 12), 100.2)
    lows = np.full((1, 12), 99.9)
    closes = np.full((1, 12), 100.0)
    highs[0, 0] = 102.0
    lows[0, 1] = 100.5
    params = _geometry(sl_mult=4.0, trailing_activation_mult=1.0).vector()
    gross, _, reason, exit_bar, _, _ = simulate_execution_ev_12h(
        opens, highs, lows, closes, np.array([1.0]), np.array([0.01]),
        np.array([0.0]), params, params,
    )
    assert reason[0] == REASON_TRAILING
    assert exit_bar[0] == 1
    assert gross[0] > 0.0


def test_materializer_starts_path_at_decision_and_persists_cost_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    signal = pd.Timestamp("2026-01-01T00:00:00Z")
    candidates = pd.DataFrame(
        {
            "__ts__": [signal],
            "__symbol__": ["BTC/USD:USD"],
            "candidate_id": ["candidate-0"],
            "side_name": ["long"],
            "__path_auxiliary_atr_fraction__": [0.01],
            "path_cost_return": [0.003],
        }
    )
    candidate_path = tmp_path / "candidates.parquet"
    candidates.to_parquet(candidate_path, index=False)
    geometry = _geometry(sl_mult=4.0, trailing_activation_mult=5.0).__dict__
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps({"geometry": {"long": geometry, "short": geometry}}),
        encoding="utf-8",
    )
    bar_index = pd.date_range(signal + pd.Timedelta(hours=1), periods=12, freq="h")
    bars = pd.DataFrame(
        {"open": 100.0, "high": 100.4, "low": 99.6, "close": 101.0},
        index=bar_index,
    )

    class FakeStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def load(self, *_args: object, **_kwargs: object) -> pd.DataFrame:
            return bars

    monkeypatch.setattr(materializer, "PartitionedOHLCVStore", FakeStore)
    output = tmp_path / "execution.parquet"
    manifest = tmp_path / "execution.manifest.json"
    materializer.materialize(
        candidate_path,
        tmp_path / "store",
        policy_path,
        output,
        manifest,
    )
    result = pd.read_parquet(output)
    assert result.loc[0, "__decision_ts__"] == signal + pd.Timedelta(hours=1)
    assert result.loc[0, "execution_label_end_utc"] == signal + pd.Timedelta(hours=13)
    assert result.loc[0, "candidate_id"] == "candidate-0"
    assert result.loc[0, "execution_label_available_at"] == signal + pd.Timedelta(hours=13)
    assert result.loc[0, "execution_exit_reason"] == "timeout"
    assert result.loc[0, "execution_gross_ev_12h"] == pytest.approx(0.01)
    assert result.loc[0, "execution_net_ev_12h"] == pytest.approx(0.007)
    payload = json.loads(manifest.read_text())
    assert payload["accounting"]["timeout"].startswith("exit at final")
    assert payload["rows"]["path_coverage_on_valid_atr"] == pytest.approx(1.0)
    assert payload["rows"]["invalid_atr_excluded"] == 0
    assert payload["prediction_role_manifest_sha256"]


def test_candidate_filter_resets_sparse_indices(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
            "__symbol__": ["BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD"],
            "candidate_id": ["candidate-0", "candidate-1", "candidate-2"],
            "side_name": ["long", "short", "long"],
            "__path_auxiliary_atr_fraction__": [0.01, np.nan, 0.02],
            "path_cost_return": [0.003, 0.003, 0.003],
        }
    )
    path = tmp_path / "candidate-filter.parquet"
    frame.to_parquet(path, index=False)
    result = materializer._canonical_candidates(path)
    assert result.index.tolist() == [0, 1]
    assert result["__symbol__"].tolist() == ["BTC/USD:USD", "SOL/USD:USD"]
    assert result.attrs["invalid_atr_rows_excluded"] == 1


def test_explicit_fee_plus_p90_spread_is_deducted_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    signal = pd.Timestamp("2026-01-01T00:00:00Z")
    candidates = pd.DataFrame(
        {
            "__ts__": [signal],
            "__symbol__": ["BTC/USD:USD"],
            "candidate_id": ["candidate-0"],
            "side_name": ["long"],
            "__path_auxiliary_atr_fraction__": [0.01],
            "path_cost_return": [0.99],
        }
    )
    candidate_path = tmp_path / "candidates.parquet"
    candidates.to_parquet(candidate_path, index=False)
    spread_map = tmp_path / "spread.csv"
    pd.DataFrame(
        {"symbol": ["BTC/USD:USD"], "p90_spread_bps": [4.0]}
    ).to_csv(spread_map, index=False)
    geometry = _geometry(sl_mult=4.0, trailing_activation_mult=5.0).__dict__
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps({"geometry": {"long": geometry, "short": geometry}}),
        encoding="utf-8",
    )
    bars = pd.DataFrame(
        {"open": 100.0, "high": 100.4, "low": 99.6, "close": 101.0},
        index=pd.date_range(signal + pd.Timedelta(hours=1), periods=12, freq="h"),
    )

    class FakeStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def load(self, *_args: object, **_kwargs: object) -> pd.DataFrame:
            return bars

    monkeypatch.setattr(materializer, "PartitionedOHLCVStore", FakeStore)
    output = tmp_path / "execution.parquet"
    manifest = tmp_path / "execution.manifest.json"
    materializer.materialize(
        candidate_path,
        tmp_path / "store",
        policy_path,
        output,
        manifest,
        fee_round_trip_return=0.003,
        spread_map_csv=spread_map,
    )
    result = pd.read_parquet(output)
    assert result.loc[0, "execution_fee_return"] == pytest.approx(0.003)
    assert result.loc[0, "execution_spread_return"] == pytest.approx(0.0004)
    assert result.loc[0, "execution_cost_return"] == pytest.approx(0.0034)
    assert result.loc[0, "execution_net_ev_12h"] == pytest.approx(0.0066)
    payload = json.loads(manifest.read_text())
    assert payload["accounting"]["cost_contract"] == "explicit_fee_plus_full_p90_spread"
    assert payload["accounting"]["fee_round_trip_return"] == pytest.approx(0.003)
    assert payload["accounting"]["spread_map_sha256"]
    assert payload["accounting"]["fee"] == "explicit round-trip fee deducted once"
    assert "__p90_full_spread_return__" in payload["accounting"]["spread"]
