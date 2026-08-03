from __future__ import annotations

import pandas as pd

from scripts.map_exact_h12_residual_recent_ev import _side_attribution


def test_side_attribution_uses_frozen_global_book_without_reranking() -> None:
    books = pd.DataFrame(
        {
            "candidate_month": ["2026-05"] * 3,
            "score_name": ["residual_exact_h12"] * 3,
            "mode": ["recent_ev_mapped"] * 3,
            "side_name": ["long", "short", "short"],
            "execution_net_ev_12h": [0.03, -0.01, 0.02],
        }
    )
    result = _side_attribution(books)
    long = result.loc[result["side_name"].eq("long")].iloc[0]
    short = result.loc[result["side_name"].eq("short")].iloc[0]
    assert long["rows"] == 1
    assert short["rows"] == 2
    assert long["contribution_net_bps"] == 100.0
    assert short["contribution_net_bps"] == 100.0 / 3.0
