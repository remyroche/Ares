import numpy as np
import pandas as pd
import pytest

from scripts.ablate_execution_ev_inference_spread_universe import (
    normalized_spread_eligibility,
    pooled_top_fraction,
    summarize,
)


def test_eligibility_uses_exact_inference_normalization() -> None:
    symbols = pd.Series(["BTC_USD:USD", "AAVE/USD:USD", "SOL/USD:USD"])
    eligible = normalized_spread_eligibility(symbols, {"BTC/USD:USD", "AAVE_USD:USD"})
    assert eligible.tolist() == [False, False, True]


def test_eligible_slice_and_rerank_are_distinct_books() -> None:
    frame = pd.DataFrame(
        {
            "canonical_recent_ev_score": np.arange(10, dtype=float),
            "execution_net_ev_12h": np.linspace(-0.01, 0.01, 10),
            "execution_gross_ev_12h": np.linspace(0.0, 0.02, 10),
            "execution_cost_return": np.full(10, 0.01),
            "side_name": ["long"] * 5 + ["short"] * 5,
            "eligible": [True] * 8 + [False, False],
        }
    )
    unrestricted = pooled_top_fraction(frame, fraction=0.2)
    assert unrestricted["canonical_recent_ev_score"].tolist() == [9.0, 8.0]
    assert unrestricted.loc[unrestricted["eligible"]].empty
    reranked = pooled_top_fraction(frame.loc[frame["eligible"]], fraction=0.2)
    assert reranked["canonical_recent_ev_score"].tolist() == [7.0, 6.0]
    metrics = summarize(
        reranked,
        candidate_rows=8,
        unrestricted_candidate_rows=10,
        original_top10_rows=2,
        book="eligible_universe_reranked_global_top10",
    )
    assert metrics["selected_rows"] == 2
    assert metrics["mean_gross_ev_bps"] == pytest.approx(144.44444444444443)
