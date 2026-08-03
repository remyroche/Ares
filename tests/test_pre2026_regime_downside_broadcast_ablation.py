from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_pre2026_regime_downside_broadcast_ablation import (
    expected_downside_hourly,
    global_top10,
)


def test_expected_downside_is_opportunity_failure_severity_product() -> None:
    raw = pd.DataFrame(
        {
            "arm": ["regime"] * 6,
            "kind": ["score_only"] * 3 + ["context"] * 3,
            "target": ["book_opportunity", "book_failure_rate_if_selected", "book_downside_severity_if_selected"] * 2,
            "prediction": [.5, .4, .01, .8, .25, .02],
            "__ts__": ["2025-01-01T00:00:00Z"] * 6,
            "era": ["2025_x"] * 6,
        }
    )
    out = expected_downside_hourly(raw)
    assert len(out) == 1
    assert out.loc[0, "score_only_expected_downside"] == pytest.approx(.002)
    assert out.loc[0, "context_expected_downside"] == pytest.approx(.004)


def test_global_top10_is_pooled_and_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["b", "a", "c", "d", "e", "f", "g", "h", "i", "j"],
            "score": [1.0, 1.0, .5, .4, .3, .2, .1, 0.0, -.1, -.2],
        }
    )
    selected = global_top10(frame, "score")
    assert selected.sum() == 1
    assert selected.iloc[1]


def test_hour_broadcast_preserves_candidate_order() -> None:
    score = pd.Series([.10, .20, .30])
    broadcast_penalty = .0025 * .4
    adjusted = score - broadcast_penalty
    assert adjusted.sort_values().index.tolist() == score.sort_values().index.tolist()
