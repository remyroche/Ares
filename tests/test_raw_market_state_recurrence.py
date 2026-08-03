from __future__ import annotations

import pandas as pd
import pytest

from scripts.diagnose_raw_market_state_recurrence import (
    BASE_SCORE,
    DECISION,
    RESOLUTION,
    SOURCE_TIME,
    SIDE,
    TARGET,
    attach_frozen_scores,
    gate_context,
    resolved_prior_blocks,
)


def _hand_off_rows(*, source_age_minutes: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    decision = pd.Timestamp("2026-06-08T01:00:00Z")
    shared = {
        "__ts__": decision - pd.Timedelta(hours=1),
        "__symbol__": "BTCUSDT",
        SIDE: "long",
        "candidate_id": "one",
        DECISION: decision,
        RESOLUTION: decision + pd.Timedelta(hours=12),
        TARGET: 0.002,
    }
    raw = pd.DataFrame([{**shared, SOURCE_TIME: decision - pd.Timedelta(minutes=source_age_minutes)}])
    scores = pd.DataFrame([{**shared, BASE_SCORE: 0.001}])
    return raw, scores


def test_attach_frozen_scores_enforces_completed_bar_source_contract() -> None:
    raw, scores = _hand_off_rows(source_age_minutes=60)
    joined = attach_frozen_scores(raw, scores)
    assert len(joined) == 1
    stale_raw, _ = _hand_off_rows(source_age_minutes=91)
    with pytest.raises(ValueError, match="source timing"):
        attach_frozen_scores(stale_raw, scores)


def test_resolved_prior_blocks_purges_open_execution_labels() -> None:
    cutoff = pd.Timestamp("2026-06-15T00:00:00Z")
    frame = pd.DataFrame(
        {
            DECISION: [
                cutoff - pd.Timedelta(days=8),
                cutoff - pd.Timedelta(hours=1),
            ],
            RESOLUTION: [
                cutoff - pd.Timedelta(days=7, hours=12),
                cutoff + pd.Timedelta(hours=11),
            ],
            "candidate_id": ["resolved", "open"],
        }
    )
    prior = resolved_prior_blocks(frame, cutoff=cutoff, block_start=cutoff)
    assert prior["candidate_id"].tolist() == ["resolved"]


class _FixedStateModel:
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        # It deliberately ignores outcomes and only supplies the gate's
        # categorical diagnostic field, as a frozen state transformer would.
        return pd.DataFrame({"causal_regime_state": [0] * len(frame)})


def test_gate_context_has_generic_chronological_blocks_not_calendar_features() -> None:
    index = pd.date_range("2026-06-01", periods=2, freq="7D", tz="UTC")
    prior = pd.DataFrame(
        {
            DECISION: index,
            RESOLUTION: index + pd.Timedelta(hours=12),
            SIDE: ["long", "long"],
            "__symbol__": ["BTCUSDT", "BTCUSDT"],
            "candidate_id": ["a", "b"],
            TARGET: [0.0, 0.0],
            BASE_SCORE: [0.0, 0.0],
            "raw": [1.0, 2.0],
        }
    )
    context = gate_context(prior, _FixedStateModel(), ["raw"])
    assert context["july_block"].tolist() == ["block_20260601", "block_20260608"]
    assert not any(column.startswith("mkt_state__calendar") for column in context.columns)
