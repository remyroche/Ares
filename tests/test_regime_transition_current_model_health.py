from __future__ import annotations

from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.regime_transition_current_model_health import (
    CURRENT_MODEL_HEALTH_COLUMNS,
    build_hourly_current_model_health,
)
from scripts.run_regime_transition_current_model_health_ablation import run as run_ablation


def _sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.date_range("2026-05-01", periods=8, freq="h", tz="UTC")
    rows: list[dict[str, object]] = []
    rich: list[dict[str, object]] = []
    for hour, stamp in enumerate(source):
        for index, side in enumerate(("long", "short", "long", "short")):
            candidate = f"asset{index}|{stamp.isoformat()}|{side}"
            decision = stamp + pd.Timedelta(hours=1)
            rows.append({
                "candidate_id": candidate,
                "__ts__": stamp,
                "__symbol__": f"asset{index}",
                "side_name": side,
                "execution_decision_utc": decision,
                "execution_label_end_utc": decision + pd.Timedelta(hours=1),
                "execution_gross_ev_12h": 0.02 if index == 0 else 0.005,
                "execution_net_ev_12h": 0.01 if index == 0 else -0.005,
                "causal_recent_isotonic_ev": 0.002 * (index - 1),
                "causal_recent_side_isotonic_ev": 0.003 * (index - 1),
                "catboost__residual__without_hpo__all_features": 0.01 * (index - 1.5),
                "causal_recent_side_isotonic_ev__is_oof": True,
                "causal_recent_side_isotonic_ev__is_forward_oos": False,
            })
            rich.append({
                "candidate_id": candidate,
                "__ts__": stamp,
                "execution_decision_utc": decision,
                "base_oof_score": 0.1 * (index + 1),
                "base_margin_to_cutoff_z": float(index - 1),
                "catboost_entropy": 0.4 + 0.1 * index,
                "alpha_prediction_uncertainty": 0.01 * index,
            })
    return pd.DataFrame(rows), pd.DataFrame(rich)


def test_current_health_is_compact_and_resolution_strict() -> None:
    ledger, rich = _sources()
    health, report = build_hourly_current_model_health(ledger, rich)
    assert len(CURRENT_MODEL_HEALTH_COLUMNS) == 29
    assert set(CURRENT_MODEL_HEALTH_COLUMNS).issubset(health.columns)
    assert report["lineage"].startswith("current execution-EV")
    # At 02:00 the first cohort resolves at the exact decision timestamp, so
    # strict-prior history must still be empty.  It appears only at 03:00.
    exact = health.loc[health["execution_decision_utc"].eq(pd.Timestamp("2026-05-01T02:00:00Z"))].iloc[0]
    later = health.loc[health["execution_decision_utc"].eq(pd.Timestamp("2026-05-01T03:00:00Z"))].iloc[0]
    assert np.isnan(exact["health__recent_resolved_net_ev_hl3d"])
    assert np.isfinite(later["health__recent_resolved_net_ev_hl3d"])
    assert health["health__mapped_ev_global_side_abs_gap_mean"].notna().all()


def test_current_health_rejects_candidate_timestamp_mismatch() -> None:
    ledger, rich = _sources()
    rich.loc[0, "__ts__"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="timestamp mismatch"):
        build_hourly_current_model_health(ledger, rich)


def test_ablation_refuses_invalid_grouped_oof_when_only_one_event(tmp_path) -> None:
    ledger, rich = _sources()
    health, _ = build_hourly_current_model_health(ledger, rich)
    market = health.loc[:, ["source_utc", "execution_decision_utc"]].copy()
    market["segment_id"] = 1
    market["observable_market_feature"] = np.arange(len(market), dtype=np.float32)
    market["target__pooled_state"] = 0
    market["target__event_id"] = None
    market["target__onset_within_3h"] = 0
    market.loc[2:4, "target__event_id"] = "only_event"
    market.loc[2:4, "target__onset_within_3h"] = 1
    market_path = tmp_path / "market.parquet"
    health_path = tmp_path / "health.parquet"
    market.to_parquet(market_path, index=False)
    health.to_parquet(health_path, index=False)
    report = run_ablation(Namespace(
        dataset=market_path,
        health=health_path,
        output_dir=tmp_path / "out",
        folds=5,
        seed=7,
    ))
    assert report["status"] == "INSUFFICIENT_INDEPENDENT_EVENTS_FOR_GROUPED_OOF"
    assert report["metrics"] is None
