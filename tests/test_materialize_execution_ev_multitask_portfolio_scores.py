from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.materialize_execution_ev_multitask_portfolio_scores import (
    IDENTITY,
    materialize,
)


def test_materializer_keeps_only_strict_oof_rows_and_marks_each_score() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
            "__symbol__": ["A", "B", "C"],
            "side_name": ["long", "short", "long"],
            "candidate_id": ["a", "b", "c"],
            "oof_fold": [0.0, np.nan, 1.0],
            "direct": [0.1, 0.2, 0.3],
            "challenger": [0.4, 0.5, 0.6],
        }
    )
    output = materialize(frame, ["direct", "challenger"])
    assert output.loc[:, list(IDENTITY)].candidate_id.tolist() == ["a", "c"]
    assert output["execution_ev_model_ablation_oof_fold"].tolist() == [0, 1]
    assert output["direct__is_oof"].all()
    assert output["challenger__is_oof"].all()
