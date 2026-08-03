from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.audit_deployed_policy_label_parity import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    compare_label_frames,
    select_stratified_sample,
)


def _labels() -> pd.DataFrame:
    rows = []
    for month in ("2026-05", "2026-06"):
        for side in ("long", "short"):
            for offset in range(3):
                row = {
                        "__ts__": pd.Timestamp(f"{month}-01T00:00:00Z") + pd.Timedelta(hours=offset),
                        "__symbol__": "BTC/USD:USD",
                        "side_name": side,
                        "candidate_id": f"{month}-{side}-{offset}",
                        **{column: float(offset) for column in NUMERIC_COLUMNS},
                        **{column: "same" for column in CATEGORICAL_COLUMNS},
                    }
                row["execution_gross_ev_12h"] = float(offset + 1)
                row["execution_cost_return"] = 1.0
                row["execution_net_ev_12h"] = float(offset)
                rows.append(row)
    return pd.DataFrame(rows)


def test_sample_is_side_month_stratified_and_deterministic() -> None:
    frame = _labels()
    sampled = select_stratified_sample(frame, per_side_month=2)
    assert len(sampled) == 8
    counts = sampled.assign(month=sampled["__ts__"].dt.strftime("%Y-%m")).groupby(["month", "side_name"]).size()
    assert counts.to_dict() == {
        ("2026-05", "long"): 2,
        ("2026-05", "short"): 2,
        ("2026-06", "long"): 2,
        ("2026-06", "short"): 2,
    }
    assert sampled["candidate_id"].tolist() == select_stratified_sample(frame, per_side_month=2)["candidate_id"].tolist()


def test_comparison_fails_closed_for_economic_difference() -> None:
    reference = _labels().iloc[:2].copy()
    replayed = reference.copy()
    comparison, summary = compare_label_frames(replayed, reference, atol=1e-10)
    assert summary["parity_pass"]
    replayed.loc[0, "execution_net_ev_12h"] += 1e-3
    comparison, summary = compare_label_frames(replayed, reference, atol=1e-10)
    assert not summary["parity_pass"]
    assert comparison.loc[comparison["field"].eq("execution_net_ev_12h"), "mismatch_rows"].item() == 1
