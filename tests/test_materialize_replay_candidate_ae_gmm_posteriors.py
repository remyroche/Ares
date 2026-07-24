from __future__ import annotations

import pandas as pd
import pytest

from scripts.materialize_replay_candidate_ae_gmm_posteriors import (
    _append_live_source_regime_inputs,
    _build_group_features,
    _normalize_candidate_schema,
)


def test_source_regime_requires_full_cross_sectional_panel() -> None:
    """Candidate-only rows must not emit degenerate cross-sectional scores."""

    raw = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-07-01T00:00:00Z"] * 3, utc=True),
            "__symbol__": ["A/USD:USD", "B/USD:USD", "C/USD:USD"],
        }
    )

    with pytest.raises(ValueError, match="full cross-sectional panel"):
        _append_live_source_regime_inputs(
            raw,
            required_columns=["__regime_source_dirty_shock_avoid_score__"],
            min_timestamp_symbols=4,
        )


def test_source_regime_is_not_requested_without_source_columns() -> None:
    raw = pd.DataFrame({"__ts__": pd.to_datetime(["2026-07-01T00:00:00Z"], utc=True)})
    out, report = _append_live_source_regime_inputs(raw, required_columns=[])

    assert out.equals(raw)
    assert report == {"source_regime_requested": False}


def test_frozen_transform_consumes_full_universe_source_overlay() -> None:
    index = pd.Index([10, 11])
    group = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-07-01T00:00:00Z", "2026-07-01T01:00:00Z"], utc=True),
            "side_name": ["long", "long"],
            "__regime_source_trend_path_score__": [0.12, 0.88],
        },
        index=index,
    )
    features = pd.DataFrame({"ordinary_feature": [1.0, 2.0]}, index=group["timestamp"])
    out = _build_group_features(
        group,
        features,
        ["ordinary_feature", "__regime_source_trend_path_score__", "side"],
    )

    assert out.loc[10, "__regime_source_trend_path_score__"] == pytest.approx(0.12)
    assert out.loc[11, "__regime_source_trend_path_score__"] == pytest.approx(0.88)


def test_candidate_schema_accepts_canonical_static_ledger_columns() -> None:
    raw = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-07-01T00:00:00Z"], utc=True),
            "__symbol__": ["A/USD:USD"],
            "side_name": ["short"],
        }
    )
    normalized = _normalize_candidate_schema(raw)

    assert normalized.loc[0, "timestamp"] == raw.loc[0, "__ts__"]
    assert normalized.loc[0, "symbol"] == "A/USD:USD"
