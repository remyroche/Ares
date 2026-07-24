from __future__ import annotations

import pandas as pd
import pytest

from scripts.backfill_residual_event_taxonomy_state import _read_projection


def test_state_projection_requires_observable_identity_columns(tmp_path) -> None:
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"__ts__": ["2026-01-01"], "side_name": ["long"]}).to_parquet(path)
    with pytest.raises(KeyError, match="missing keys"):
        _read_projection(path)


def test_state_projection_keeps_only_observable_keys(tmp_path) -> None:
    path = tmp_path / "state.parquet"
    pd.DataFrame(
        {
            "__ts__": ["2026-01-01"],
            "__symbol__": ["BTC/USD:USD"],
            "side_name": ["long"],
            "archetype_policy_key": ["long_mixed"],
            "ev_after_1pct": [-0.5],
        }
    ).to_parquet(path)
    result = _read_projection(path)
    assert "ev_after_1pct" not in result.columns
    assert result["selected_top30"].tolist() == [True]
