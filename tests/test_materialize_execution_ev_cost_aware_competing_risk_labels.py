from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_execution_ev_cost_aware_competing_risk_labels import (
    HORIZON_MINUTES,
    _parser,
    _buffer_change_summary,
    _complete_path_mask,
    _load_source,
    _materialize_symbol,
    _sha256,
    build_row_cost_aware_competing_risk_labels,
)


def _paths(rows: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    open_ = np.full((rows, HORIZON_MINUTES), 100.0)
    return open_, open_.copy(), open_.copy(), open_.copy()


def _labels(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    side: float = 1.0,
    atr: float = 0.01,
    cost: float = 0.005,
    buffer_bps: int = 0,
) -> pd.DataFrame:
    open_, _, _, _ = _paths(len(high))
    return build_row_cost_aware_competing_risk_labels(
        open_, high, low, close,
        oof_entry_atr_fraction=np.full(len(high), atr),
        execution_cost_return=np.full(len(high), cost),
        execution_entry_price=np.full(len(high), 100.0),
        side_sign=np.full(len(high), side),
        decision_utc=pd.date_range("2026-07-01", periods=len(high), freq="h", tz="UTC"),
        buffer_bps=buffer_bps,
    )


def test_long_favorable_and_short_favorable_are_side_signed() -> None:
    _, high, low, close = _paths(2)
    high[0, 4] = 101.6
    low[1, 7] = 98.4
    long = _labels(high[:1], low[:1], close[:1], side=1.0).iloc[0]
    short = _labels(high[1:], low[1:], close[1:], side=-1.0).iloc[0]
    assert long["competing_risk_event"] == "clean_economic_favorable_first"
    assert long["first_event_minute"] == 4.0
    assert short["competing_risk_event"] == "clean_economic_favorable_first"
    assert short["first_event_minute"] == 7.0


def test_same_minute_favorable_adverse_tie_is_adverse() -> None:
    _, high, low, close = _paths()
    high[0, 3] = 101.6
    low[0, 3] = 99.0
    label = _labels(high, low, close).iloc[0]
    assert label["competing_risk_event"] == "adverse_first"
    assert label["adverse_first"] == 1
    assert label["same_minute_favorable_adverse_conflict"] == 1
    assert label["first_favorable_minute"] == label["first_adverse_minute"] == 3.0


def test_timeout_has_a_timeout_only_soft_viability_simplex() -> None:
    _, high, low, close = _paths()
    close[0, -1] = 100.5
    label = _labels(high, low, close).iloc[0]
    assert label["competing_risk_event"] == "timeout"
    assert label["label_resolution_utc"] == pd.Timestamp("2026-07-01T12:00:00Z")
    simplex = [
        label["timeout_soft_clean_economic_favorable_viability"],
        label["timeout_soft_adverse_viability"],
        label["timeout_soft_timeout_viability"],
    ]
    assert np.isclose(sum(simplex), 1.0)
    assert all(0.0 <= value <= 1.0 for value in simplex)


def test_row_cost_plus_buffer_can_dominate_favorable_barrier() -> None:
    _, high, low, close = _paths()
    high[0, 5] = 102.0
    # 50 bps cost plus 50 bps buffer -> 1.0% is not dominant.  Raise cost
    # until the 2.0% path is economically insufficient after the buffer.
    label = _labels(high, low, close, atr=0.005, cost=0.018, buffer_bps=50).iloc[0]
    assert label["upper_barrier_driver"] == "cost_plus_buffer"
    assert label["economic_upper_return"] == pytest.approx(0.023)
    assert label["competing_risk_event"] == "timeout"


def test_upper_barrier_driver_uses_declared_tie_precedence() -> None:
    _, high, low, close = _paths()
    # 1.5 ATR and cost are both exactly the 1.5% floor.  The output must not
    # conceal the equal geometry: cost_plus_buffer has declared precedence.
    label = _labels(high, low, close, atr=0.01, cost=0.015).iloc[0]
    assert label["economic_upper_return"] == pytest.approx(0.015)
    assert label["upper_barrier_driver"] == "cost_plus_buffer"


def test_optional_no_floor_sensitivity_preserves_cost_aware_formula() -> None:
    open_, high, low, close = _paths()
    high[0, 4] = 101.1
    labels = build_row_cost_aware_competing_risk_labels(
        open_, high, low, close,
        oof_entry_atr_fraction=[0.005],
        execution_cost_return=[0.01],
        execution_entry_price=[100.0],
        side_sign=[1.0],
        decision_utc=[pd.Timestamp("2026-07-01T00:00:00Z")],
        buffer_bps=0,
        use_upper_return_floor=False,
    )
    assert labels.loc[0, "economic_upper_return"] == pytest.approx(0.01)
    assert labels.loc[0, "upper_barrier_driver"] == "cost_plus_buffer"
    assert labels.loc[0, "competing_risk_event"] == "clean_economic_favorable_first"


def test_cli_requires_output_and_keeps_primary_floor_by_default() -> None:
    args = _parser().parse_args(["--output-dir", "unused"])
    assert args.omit_upper_return_floor is False
    assert args.buffer_bps == []


def test_buffer_change_summary_reports_inert_and_changed_geometry() -> None:
    _, high, low, close = _paths()
    # 2.0% favourable excursion: 0 bps reaches (floor 1.5%), whereas 100 bps
    # above a 1.8% cost has a 2.8% upper barrier and times out.
    zero = _labels(high, low, close, cost=0.018, buffer_bps=0)
    high[0, 4] = 102.0
    zero = _labels(high, low, close, cost=0.010, buffer_bps=0)
    hundred = _labels(high, low, close, cost=0.018, buffer_bps=100)
    identity = {
        "__ts__": pd.Timestamp("2026-07-01T00:00:00Z"),
        "__symbol__": "X/USD:USD",
        "side_name": "long",
        "candidate_id": "x",
    }
    frame = pd.concat([zero, hundred], ignore_index=True)
    for key, value in identity.items():
        frame[key] = value
    summary = _buffer_change_summary(frame).set_index("cost_buffer_bps")
    assert summary.loc[0, "competing_risk_label_changed_rows"] == 0
    assert summary.loc[100, "upper_return_changed_rows"] == 1
    assert summary.loc[100, "competing_risk_label_changed_rows"] == 1


def test_complete_path_mask_fails_closed_on_one_missing_minute() -> None:
    values = np.full((HORIZON_MINUTES + 1, 4), 100.0)
    values[:, 1] = 101.0
    values[:, 2] = 99.0
    assert _complete_path_mask(values, np.array([0, 1])).tolist() == [True, True]
    values[30, 3] = np.nan
    assert _complete_path_mask(values, np.array([0, 1])).tolist() == [False, False]


def test_materialized_symbol_retains_one_authoritative_atr_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = pd.Timestamp("2026-07-01T01:00:00Z")
    grid = pd.date_range(decision, periods=HORIZON_MINUTES, freq="min", tz="UTC")
    bars = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        index=grid,
    )
    monkeypatch.setattr(
        "scripts.materialize_execution_ev_cost_aware_competing_risk_labels._load_symbol_bars",
        lambda *_args, **_kwargs: bars,
    )
    source = pd.DataFrame(
        {
            "__ts__": [decision - pd.Timedelta(hours=1)],
            "__symbol__": ["X/USD:USD"],
            "side_name": ["long"],
            "candidate_id": ["x"],
            "execution_decision_utc": [decision],
            "execution_label_end_utc": [decision + pd.Timedelta(hours=12)],
            "execution_entry_price": [100.0],
            "execution_gross_ev_12h": [0.01],
            "execution_cost_return": [0.01],
            "execution_net_ev_12h": [0.0],
            "execution_exit_hour": [12.0],
            "oof_entry_atr_fraction": [0.01],
        }
    )
    labels, coverage = _materialize_symbol(
        source,
        data_root=Path("unused"),
        buffers_bps=[0, 25],
        batch_rows=1,
        use_upper_return_floor=True,
    )
    assert labels.columns.is_unique
    assert labels.columns.tolist().count("oof_entry_atr_fraction") == 1
    assert len(labels) == 2
    assert coverage["coverage"] == 1.0


def test_source_loader_binds_input_hash_pit_atr_and_gross_cost_net(tmp_path: Path) -> None:
    identity = {
        "__ts__": [pd.Timestamp("2026-07-01T00:00:00Z")],
        "__symbol__": ["X/USD:USD"],
        "side_name": ["long"],
        "candidate_id": ["x"],
    }
    labels_path = tmp_path / "labels.parquet"
    targets_path = tmp_path / "targets.parquet"
    manifest_path = tmp_path / "manifest.json"
    pd.DataFrame({
        **identity,
        "execution_decision_utc": [pd.Timestamp("2026-07-01T01:00:00Z")],
        "execution_label_end_utc": [pd.Timestamp("2026-07-01T13:00:00Z")],
        "execution_entry_price": [100.0],
        "execution_gross_ev_12h": [0.02],
        "execution_cost_return": [0.01],
        "execution_net_ev_12h": [0.02],  # deliberately inconsistent
        "execution_exit_hour": [0.1],
    }).to_parquet(labels_path, index=False)
    pd.DataFrame({**identity, "__path_auxiliary_atr_fraction__": [0.01]}).to_parquet(targets_path, index=False)
    manifest = {
        "schema": "execution_ev_deployed_policy_1m_labels_v1",
        "output": {"path": str(labels_path), "sha256": _sha256(labels_path)},
        "source": {"path_targets": str(targets_path), "path_targets_sha256": _sha256(targets_path)},
        "exit_policy_contract": {"horizon_minutes": 720},
    }
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="gross - cost"):
        _load_source(labels_path, manifest_path)

    fixed = pd.read_parquet(labels_path)
    fixed["execution_net_ev_12h"] = 0.01
    fixed.to_parquet(labels_path, index=False)
    # A stale source hash is rejected before target construction can read a
    # path or use the joined decision-time ATR fraction.
    with pytest.raises(ValueError, match="does not bind"):
        _load_source(labels_path, manifest_path)
