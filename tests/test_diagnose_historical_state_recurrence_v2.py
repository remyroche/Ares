from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_historical_state_recurrence_v2 import (
    DECISION,
    IDENTITY,
    LIVE_PREFIX,
    STATE_FEATURES,
    _strict_join,
    diagnose_expanding_states,
    verify_live_alignment,
)


def _state_rows(
    *,
    start: str,
    periods: int,
    side: str,
    origin: str,
) -> pd.DataFrame:
    stamps = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    values = np.arange(periods, dtype=float)
    frame = pd.DataFrame(
        {
            "__ts__": stamps - pd.Timedelta(hours=1),
            "__symbol__": [f"S{index % 5}" for index in range(periods)],
            "side_name": side,
            "candidate_id": [f"{side}-{origin}-{index}" for index in range(periods)],
            DECISION: stamps,
            "panel_origin": origin,
        }
    )
    for index, feature in enumerate(STATE_FEATURES):
        frame[feature] = values * (index + 1) / 100.0 + (0.01 if side == "short" else 0.0)
    return frame


def test_strict_join_rejects_source_decision_disagreement() -> None:
    population = _state_rows(
        start="2026-05-05", periods=2, side="long", origin="current_canonical_top30"
    ).drop(columns=["panel_origin", *STATE_FEATURES])
    source = _state_rows(
        start="2026-05-05", periods=2, side="long", origin="current_canonical_top30"
    ).drop(columns=["panel_origin"])
    source.loc[source.index[0], DECISION] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="decision timestamps disagree"):
        _strict_join(population, source, name="test")


def test_expanding_state_fit_uses_only_prior_rows_and_emits_no_economics() -> None:
    historical = pd.concat(
        [
            _state_rows(
                start="2026-04-01", periods=160, side="long", origin="historical_strict_oof_top30"
            ),
            _state_rows(
                start="2026-04-01", periods=160, side="short", origin="historical_strict_oof_top30"
            ),
        ],
        ignore_index=True,
    )
    current = pd.concat(
        [
            _state_rows(
                start="2026-05-05", periods=48, side="long", origin="current_canonical_top30"
            ),
            _state_rows(
                start="2026-05-05", periods=48, side="short", origin="current_canonical_top30"
            ),
        ],
        ignore_index=True,
    )
    summary, rows = diagnose_expanding_states(
        pd.concat([historical, current], ignore_index=True),
        first_evaluation=pd.Timestamp("2026-05-05", tz="UTC"),
        end=pd.Timestamp("2026-05-12", tz="UTC"),
        min_state_fit_rows=100,
    )
    assert summary["status"].eq("evaluated").all()
    assert (pd.to_datetime(summary["state_fit_max_decision_utc"], utc=True) < pd.Timestamp("2026-05-05", tz="UTC")).all()
    assert len(rows) == len(current)
    forbidden = ("score", "target", "outcome", "economic", "label", "calendar")
    assert not any(token in column.lower() for column in rows.columns for token in forbidden)
    assert {"causal_regime_state", "causal_regime_entropy", "causal_regime_ood_z"}.issubset(rows.columns)


def test_live_alignment_is_validation_only_and_requires_all_fixed_fields(tmp_path: Path) -> None:
    current = _state_rows(
        start="2026-05-05", periods=12, side="long", origin="current_canonical_top30"
    )
    capture = current.loc[:, list(IDENTITY)].copy()
    for feature in STATE_FEATURES:
        capture[f"{LIVE_PREFIX}{feature}"] = current[feature].to_numpy()
    universe = tmp_path / "capture.parquet"
    capture.to_parquet(universe, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "eligible_full_period_feature_columns": [
                    f"{LIVE_PREFIX}{feature}" for feature in STATE_FEATURES
                ]
            }
        ),
        encoding="utf-8",
    )
    result = verify_live_alignment(
        current,
        capture_universe=universe,
        capture_manifest=manifest,
        minimum_coverage=1.0,
        minimum_spearman=0.99,
    )
    assert len(result) == 23
    assert result["paired_coverage"].eq(1.0).all()
    assert result["spearman"].eq(1.0).all()
