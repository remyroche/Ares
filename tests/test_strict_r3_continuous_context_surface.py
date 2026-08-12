"""Causality boundary tests for the continuous-context reliability sidecar."""

from __future__ import annotations

import pandas as pd
import pytest

from extreme_price_movements.causal_market_regime_systems import (
    CONTINUOUS_CONTEXT_FEATURE_KEYS,
    RELATIONSHIP_BREAK_FEATURE_KEYS,
)
from scripts.materialize_strict_r3_k9weighted_mda_surface import (
    _join_continuous_market_context,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__ts__": pd.to_datetime(["2025-01-01T01:00:00Z", "2025-01-01T02:00:00Z"]),
        "__symbol__": ["X/USD:USD", "Y/USD:USD"],
        "side_name": ["long", "long"],
    })


def _sidecar(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["source_utc"] = out["__ts__"]
    out["continuous_context_available_utc"] = out["__ts__"]
    for index, field in enumerate([
        *CONTINUOUS_CONTEXT_FEATURE_KEYS,
        *RELATIONSHIP_BREAK_FEATURE_KEYS,
    ]):
        out[field] = float(index + 1)
    return out


def test_continuous_context_join_is_exact_and_target_free(tmp_path) -> None:
    frame = _frame()
    path = tmp_path / "continuous.parquet"
    _sidecar(frame).to_parquet(path, index=False)

    got = _join_continuous_market_context(frame, path)

    assert len(got) == len(frame)
    assert got["candidate_id"].tolist() == ["a", "b"]
    assert set(CONTINUOUS_CONTEXT_FEATURE_KEYS).issubset(got)
    assert set(RELATIONSHIP_BREAK_FEATURE_KEYS).issubset(got)
    assert not any(column.startswith("market_regime__state_p_") for column in got)


def test_continuous_context_join_rejects_future_available_state(tmp_path) -> None:
    frame = _frame()
    sidecar = _sidecar(frame)
    sidecar.loc[0, "continuous_context_available_utc"] = frame.loc[0, "__ts__"] + pd.Timedelta(hours=1)
    path = tmp_path / "future.parquet"
    sidecar.to_parquet(path, index=False)

    with pytest.raises(AssertionError, match="looks ahead"):
        _join_continuous_market_context(frame, path)


def test_continuous_context_join_rejects_latent_posterior_schema(tmp_path) -> None:
    frame = _frame()
    sidecar = _sidecar(frame)
    sidecar["market_regime__state_p_0"] = 1.0
    path = tmp_path / "latent.parquet"
    sidecar.to_parquet(path, index=False)

    with pytest.raises(AssertionError, match="fold-local latent-state"):
        _join_continuous_market_context(frame, path)
