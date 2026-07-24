from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import report_meta_p90_spread_cost as report


def _row(ts: str, symbol: str, side: str, base: float, meta: float, gross: float, archetype: str) -> dict[str, object]:
    return {
        "__ts__": pd.Timestamp(ts, tz="UTC"),
        "__symbol__": symbol,
        "side_name": side,
        "score_base": base,
        "score_meta_base_soft_label": meta,
        "first_touch_gross": gross,
        "archetype_label_family": archetype,
        "clean_exec": 1.0,
        "dirty_positive": 0.0,
        "first_touch_bad_mae_1r": 0.0,
        "full_path_bad_mae_1r": 0.0,
        "timeout": 0.0,
    }


def _spread(symbols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, symbol in enumerate(symbols, start=1):
        rows.extend(
            [
                {"observed_ts": pd.Timestamp("2026-07-01", tz="UTC"), "symbol": symbol, "spread_bps": index * 10.0},
                {"observed_ts": pd.Timestamp("2026-07-02", tz="UTC"), "symbol": symbol, "spread_bps": index * 20.0},
            ]
        )
    return pd.DataFrame(rows)


def test_universe_is_fixed_from_base_candidates_before_meta_score_validity() -> None:
    predictions = pd.DataFrame(
        [
            _row("2026-04-01T00:00:00", "AAA", "long", 0.9, 0.8, 0.02, "a"),
            _row("2026-04-01T00:00:00", "BBB", "long", 0.8, np.nan, 0.03, "a"),
            _row("2026-04-01T00:00:00", "CCC", "long", 0.7, 0.9, 0.04, "a"),
        ]
    )
    candidates = predictions.loc[:, list(report.KEY_COLUMNS)].copy()
    candidates["selected_top30"] = True
    result = report.evaluate_frames(
        predictions,
        candidates,
        _spread(["AAA", "BBB", "CCC"]),
        eligible_symbols=2,
    )

    assert set(result.eligible.index) == {"AAA", "BBB"}
    assert result.integrity["prediction_rows_after_base_candidate_join_before_score_validity"] == 2
    assert result.integrity["identical_score_valid_oos_rows"] == 1
    assert result.integrity["dropped_for_nonfinite_base_or_meta_score_or_outcome"] == 1


def test_ranking_is_within_timestamp_and_side_and_emits_path_metrics() -> None:
    rows = [
        _row("2026-04-01T00:00:00", "AAA", "long", 0.9, 0.1, 0.020, "a"),
        _row("2026-04-01T00:00:00", "BBB", "long", 0.8, 0.9, 0.010, "a"),
        _row("2026-04-01T00:00:00", "CCC", "short", 0.02, 0.01, 0.030, "b"),
        _row("2026-04-01T00:00:00", "DDD", "short", 0.01, 0.02, 0.040, "b"),
    ]
    predictions = pd.DataFrame(rows)
    candidates = predictions.loc[:, list(report.KEY_COLUMNS)].copy()
    candidates["selected_top30"] = True
    result = report.evaluate_frames(predictions, candidates, _spread(["AAA", "BBB", "CCC", "DDD"]), eligible_symbols=4)

    base = result.metrics.loc[(result.metrics.selector == "base_score") & (result.metrics.scope == "overall") & (result.metrics.top_frac == 0.10)].iloc[0]
    meta = result.metrics.loc[(result.metrics.selector == "meta_base_soft_label") & (result.metrics.scope == "overall") & (result.metrics.top_frac == 0.10)].iloc[0]
    assert base.selected_rows == 2
    assert meta.selected_rows == 2
    assert base.clean_positive_rate == 1.0
    assert meta.timeout_rate == 0.0
    assert set(result.metrics.scope) >= {"overall", "month", "week", "side", "archetype", "side_archetype"}
    delta = result.deltas.loc[(result.deltas.scope == "overall") & (result.deltas.top_frac == 0.10)].iloc[0]
    assert delta.delta_mean_net_ev_vs_base != 0.0


def test_cost_is_gross_minus_fee_and_pooled_full_spread_once() -> None:
    predictions = pd.DataFrame([
        _row("2026-04-01T00:00:00", "AAA", "long", 0.9, 0.9, 0.020, "a"),
    ])
    candidates = predictions.loc[:, list(report.KEY_COLUMNS)].copy()
    candidates["selected_top30"] = True
    spreads = pd.DataFrame(
        [
            {"observed_ts": pd.Timestamp("2026-07-01", tz="UTC"), "symbol": "AAA", "spread_bps": 10.0},
            {"observed_ts": pd.Timestamp("2026-07-02", tz="UTC"), "symbol": "AAA", "spread_bps": 30.0},
        ]
    )
    result = report.evaluate_frames(predictions, candidates, spreads, eligible_symbols=1)
    row = result.metrics.loc[(result.metrics.selector == "base_score") & (result.metrics.scope == "overall") & (result.metrics.top_frac == 0.10)].iloc[0]
    # pandas' default linear 90th percentile of 10 and 30 bps is 28 bps.
    assert row.mean_net_ev == pytest.approx(0.020 - 0.0015 - 0.0028)
    assert row.positive_net_ev_rate == pytest.approx(1.0)
    assert result.provenance["cost_formula"] == "first_touch_gross - fee_round_trip_pct - pooled_p90_full_spread_bps/10000"


def test_path_evaluation_writes_integrity_and_provenance(tmp_path: Path) -> None:
    predictions = pd.DataFrame([
        _row("2026-04-01T00:00:00", "AAA", "long", 0.9, 0.9, 0.020, "a"),
    ])
    candidates = predictions.loc[:, list(report.KEY_COLUMNS)].copy()
    candidates["selected_top30"] = True
    spread = _spread(["AAA"])
    predictions_path = tmp_path / "predictions.parquet"
    candidates_path = tmp_path / "candidates.parquet"
    spread_path = tmp_path / "spread.parquet"
    predictions.to_parquet(predictions_path, index=False)
    candidates.to_parquet(candidates_path, index=False)
    spread.to_parquet(spread_path, index=False)
    result = report.evaluate_paths(
        predictions_path=predictions_path,
        base_candidate_ledger_path=candidates_path,
        spread_history_path=spread_path,
        eligible_symbols=1,
        spread_quantile=0.90,
        fee_round_trip_pct=0.0015,
        base_candidate_selector_column="selected_top30",
    )
    out_dir = tmp_path / "report"
    report._write_result(result, out_dir)
    assert (out_dir / "metrics.csv").is_file()
    assert (out_dir / "delta_vs_base.csv").is_file()
    integrity = json.loads((out_dir / "integrity.json").read_text())
    provenance = json.loads((out_dir / "provenance.json").read_text())
    assert integrity["rank_scope"] == "timestamp_side"
    assert provenance["schema"] == report.SCHEMA


def test_fails_when_base_candidate_universe_cannot_supply_requested_symbols() -> None:
    predictions = pd.DataFrame([
        _row("2026-04-01T00:00:00", "AAA", "long", 0.9, 0.9, 0.020, "a"),
    ])
    candidates = predictions.loc[:, list(report.KEY_COLUMNS)].copy()
    candidates["selected_top30"] = True
    with pytest.raises(ValueError, match="requires 2"):
        report.evaluate_frames(predictions, candidates, _spread(["AAA"]), eligible_symbols=2)


def test_fails_closed_when_base_archetype_identity_is_missing() -> None:
    predictions = pd.DataFrame([
        _row("2026-04-01T00:00:00", "AAA", "long", 0.9, 0.9, 0.020, "a"),
    ]).drop(columns="archetype_label_family")
    candidates = predictions.loc[:, list(report.KEY_COLUMNS)].copy()
    candidates["selected_top30"] = True
    with pytest.raises(ValueError, match="base archetype identity"):
        report.evaluate_frames(predictions, candidates, _spread(["AAA"]), eligible_symbols=1)
