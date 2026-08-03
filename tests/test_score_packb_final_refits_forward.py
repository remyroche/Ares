from __future__ import annotations

import pandas as pd

from scripts.score_packb_final_refits_forward import build_base_context


def test_base_context_selects_top40_with_deterministic_margin() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-08-01"] * 10, utc=True),
            "__symbol__": [f"S{i:02d}" for i in range(10)],
            "side_name": ["long"] * 10,
            "candidate_id": [f"id{i}" for i in range(10)],
            "prediction": [float(i) for i in range(10)],
        }
    )
    selected, full = build_base_context(frame)
    selected = selected.sort_values("base_candidate_rank_timestamp_side")
    assert len(full) == 10
    assert len(selected) == 4
    assert selected["prediction"].tolist() == [9.0, 8.0, 7.0, 6.0]
    assert selected["base_cutoff_score"].eq(6.0).all()
    assert selected["base_margin_to_cutoff"].tolist() == [3.0, 2.0, 1.0, 0.0]
    assert selected["base_candidate_rank_timestamp_side"].tolist() == [1, 2, 3, 4]


def test_base_context_is_side_local() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-08-01"] * 6, utc=True),
            "__symbol__": ["A", "B", "C", "A", "B", "C"],
            "side_name": ["long"] * 3 + ["short"] * 3,
            "candidate_id": [f"id{i}" for i in range(6)],
            "prediction": [3.0, 2.0, 1.0, 30.0, 20.0, 10.0],
        }
    )
    selected, _ = build_base_context(frame)
    # ceil(40% * 3) is two for each side.
    assert selected.groupby("side_name").size().to_dict() == {"long": 2, "short": 2}
