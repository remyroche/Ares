from __future__ import annotations

import pandas as pd
import pytest

from scripts.materialize_residual_only_policy_oos_candidates import (
    materialize_candidates,
    repair_prediction_path_end_from_ledger,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.Timestamp("2026-07-01T00:00:00Z")
    predictions = pd.DataFrame(
        {
            "__ts__": [ts],
            "__symbol__": ["BTC/USD:USD"],
            "side_name": ["long"],
            "__label_path_end_ts__": [ts + pd.Timedelta(hours=25)],
            "ev_after_1pct": [0.01],
            "clean_exec": [1.0],
            "dirty_positive": [0.0],
            "full_path_bad_mae_1r": [0.0],
            "timeout": [0.0],
            "score_base": [0.5],
            "score_base_ev_mapped": [0.006],
            "score_base_ev_residual_expert": [0.008],
            "score_base_ev_residual_expert_hier_mapped": [0.009],
            "meta_residual_expert_delta_ev": [0.002],
            "score_base_ev_rank_train_reference": [0.80],
            "score_base_residual_ev_rank_train_reference": [0.95],
            "archetype_policy_key": ["long_mixed"],
            "calendar_month": ["2026-07"],
            "week_start": ["2026-06-29"],
        }
    )
    ledger = pd.DataFrame(
        {
            "__ts__": [ts],
            "__symbol__": ["BTC/USD:USD"],
            "side_name": ["long"],
            "__signal_ts__": [ts],
            "__decision_ts__": [ts + pd.Timedelta(hours=1)],
            "__first_path_ts__": [ts + pd.Timedelta(hours=1)],
            "__entry_ts__": [ts + pd.Timedelta(hours=1)],
            "__label_path_end_ts__": [ts + pd.Timedelta(hours=25)],
            "__barrier_pct__": [0.01],
            "__archetype_policy_key__": ["long_mixed"],
            "__archetype_policy_tp_r__": [0.4],
            "__archetype_policy_sl_r__": [1.5],
            "__archetype_policy_trail_r__": [0.2],
            "__archetype_policy_confidence__": [0.8],
        }
    )
    return predictions, ledger


def test_materializer_preserves_frozen_rank_ev_and_causal_timing() -> None:
    predictions, ledger = _frames()
    rows, audit = materialize_candidates(predictions, ledger)
    assert len(rows) == 1
    assert rows.loc[0, "rank_pct"] == pytest.approx(0.95)
    assert rows.loc[0, "calibrated_score"] == pytest.approx(0.009)
    assert rows.loc[0, "decision_timestamp"] == rows.loc[0, "signal_timestamp"] + pd.Timedelta(
        hours=1
    )
    assert audit["score_contract"]["no_materializer_rerank"] is True


def test_materializer_rejects_same_bar_decision() -> None:
    predictions, ledger = _frames()
    ledger.loc[0, "__decision_ts__"] = ledger.loc[0, "__signal_ts__"]
    with pytest.raises(ValueError, match="causal path timing invalid"):
        materialize_candidates(predictions, ledger)


def test_materializer_rejects_archetype_mismatch() -> None:
    predictions, ledger = _frames()
    ledger.loc[0, "__archetype_policy_key__"] = "other"
    with pytest.raises(ValueError, match="archetype mismatch"):
        materialize_candidates(predictions, ledger)


def test_explicit_path_end_repair_uses_authoritative_ledger() -> None:
    predictions, ledger = _frames()
    predictions["__label_path_end_ts__"] += pd.Timedelta(hours=2)

    repaired, audit = repair_prediction_path_end_from_ledger(predictions, ledger)
    rows, _ = materialize_candidates(repaired, ledger)

    assert len(rows) == 1
    assert audit["source_mismatch_rows"] == 1
    assert audit["delta_seconds_min"] == 7200.0
    assert audit["delta_seconds_max"] == 7200.0


def test_materializer_can_bound_download_universe_with_frozen_rank() -> None:
    predictions, ledger = _frames()
    rows, audit = materialize_candidates(predictions, ledger, min_rank=0.90)
    assert len(rows) == 1
    assert audit["min_rank"] == 0.90
    with pytest.raises(ValueError, match="no candidates remain"):
        materialize_candidates(predictions, ledger, min_rank=0.99)


def test_materializer_preserves_short_side_and_archetype_contract() -> None:
    predictions, ledger = _frames()
    predictions["side_name"] = "short"
    predictions["archetype_policy_key"] = "short_default"
    ledger["side_name"] = "short"
    ledger["__archetype_policy_key__"] = "short_default"

    rows, audit = materialize_candidates(predictions, ledger, side_name="short")

    assert rows.loc[0, "side"] == pytest.approx(-1.0)
    assert rows.loc[0, "side_name"] == "short"
    assert rows.loc[0, "strategy_id"] == "short_s59_residual_only_oos"
    assert rows.loc[0, "local_side_archetype"] == "short__short_default"
    assert audit["side"] == "short"
