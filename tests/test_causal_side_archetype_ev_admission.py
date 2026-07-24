from __future__ import annotations

import pandas as pd

from scripts.run_causal_side_archetype_ev_admission import _load, _reference


def test_reference_defers_test_outcome_by_horizon() -> None:
    rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-07-01T00:00:00Z"], utc=True),
            "symbol": ["BTC/USD:USD"],
            "side_name": ["long"],
            "policy_archetype": ["long_breakout"],
            "rank_mlp_direct": [0.95],
            "expected_net_ev_after_1pct_mlp_direct": [0.01],
            "ev_after_1pct": [0.02],
        }
    )

    reference = _reference(rows, 12)

    assert reference.loc[0, "outcome_resolved_at"] == pd.Timestamp(
        "2026-07-01T12:00:00Z"
    )
    assert reference.loc[0, "mapped_expected_ev"] == 0.01


def test_load_accepts_categorical_archetype_with_missing(tmp_path) -> None:
    path = tmp_path / "rows.parquet"
    pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-07-01T00:00:00Z"], utc=True),
            "__symbol__": ["BTC/USD:USD"],
            "side_name": ["long"],
            "archetype_policy_key": pd.Categorical([None], categories=["known"]),
            "rank_mlp_direct": [0.95],
            "expected_net_ev_after_1pct_mlp_direct": [0.01],
            "ev_after_1pct": [0.02],
        }
    ).to_parquet(path, index=False)

    rows = _load(path)

    assert rows.loc[0, "policy_archetype"] == "missing"
