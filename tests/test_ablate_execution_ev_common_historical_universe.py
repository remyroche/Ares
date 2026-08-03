import numpy as np
import pandas as pd

from scripts.ablate_execution_ev_common_historical_universe import (
    pooled_top_fraction,
    summarize,
)


def test_common_universe_reranks_one_global_book() -> None:
    frame = pd.DataFrame(
        {
            "canonical_recent_ev_score": np.arange(20, dtype=float),
            "execution_net_ev_12h": np.arange(20, dtype=float) / 100.0,
            "execution_gross_ev_12h": np.arange(20, dtype=float) / 100.0 + 0.01,
            "execution_cost_return": np.full(20, 0.01),
            "side_name": ["long"] * 10 + ["short"] * 10,
        }
    )
    selected = pooled_top_fraction(frame)
    assert len(selected) == 2
    assert selected["canonical_recent_ev_score"].min() == 18
    metrics = summarize(selected, candidate_rows=20, book="test")
    assert metrics["selected_fraction"] == 0.1
    assert metrics["long_rows"] == 0
    assert metrics["short_rows"] == 2
