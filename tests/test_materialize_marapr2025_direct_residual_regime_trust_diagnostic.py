from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import materialize_marapr2025_direct_residual_regime_trust_diagnostic as m


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "side_name": ["long", "long", "short", "short"],
            "__symbol__": ["A", "B", "C", "D"],
            "__ts__": pd.to_datetime(["2025-03-18"] * 4, utc=True),
            m.SCORES["direct_q25"]: [4.0, 3.0, 2.0, 1.0],
            m.SCORES["residual"]: [1.0, 3.0, 4.0, 2.0],
            m.NET: [0.04, 0.03, -0.02, -0.01],
            m.GROSS: [0.05, 0.04, -0.01, 0.00],
            m.COST: [0.01] * 4,
            m.MFE: [0.06, 0.05, 0.01, 0.02],
            m.MAE: [-0.01, -0.02, -0.04, -0.03],
        }
    )


def test_period_boundaries_are_exact_and_exclusive() -> None:
    values = pd.Series(
        pd.to_datetime(
            [
                "2025-03-03T00:00:00Z",
                "2025-03-19T23:00:00Z",
                "2025-03-20T00:00:00Z",
                "2025-03-31T23:00:00Z",
                "2025-04-01T00:00:00Z",
                "2025-04-30T23:00:00Z",
            ],
            utc=True,
        )
    )
    assert m.assign_period(values).tolist() == [
        "march03_19",
        "march03_19",
        "march20_31",
        "march20_31",
        "april",
        "april",
    ]


def test_candidate_coordinates_are_complete_group_relative() -> None:
    result = m.add_candidate_coordinates(_rows())
    assert set(result["candidate_group_rows_timestamp_side"]) == {2}
    assert set(result["candidate_group_rows_timestamp_global"]) == {4}
    assert result.loc[result.candidate_id.eq("a"), "direct_q25_rank_pct_timestamp_global"].iat[0] == 0
    assert result.loc[result.candidate_id.eq("d"), "direct_q25_rank_pct_timestamp_global"].iat[0] == 1
    assert np.isfinite(result.filter(regex="rank|score_z|group_rows").to_numpy(float)).all()


def test_global_top_is_pooled_and_deterministic() -> None:
    rows = _rows()
    rows[m.SCORES["direct_q25"]] = 1.0
    selected = m.global_top(rows, m.SCORES["direct_q25"], fraction=0.50)
    assert selected.candidate_id.tolist() == ["a", "b"]
    assert selected.side_name.tolist() == ["long", "long"]


def test_overlap_attribution_reconciles_direct_minus_residual() -> None:
    rows = _rows()
    direct = rows.loc[rows.candidate_id.isin(["a", "b"])]
    residual = rows.loc[rows.candidate_id.isin(["b", "c"])]
    records = m.overlap_records(rows, direct, residual, period="test")
    delta = direct[m.NET].mean() - residual[m.NET].mean()
    assert np.isclose(
        sum(row["direct_minus_residual_contribution_bps"] for row in records),
        delta * 1e4,
    )
    shared = next(row for row in records if row["membership"] == "shared")
    assert shared["rows"] == 1


def test_quantile_edges_use_reference_only() -> None:
    reference = pd.Series(np.arange(1000, dtype=float))
    edges = m.quantile_edges(reference)
    assert np.allclose(edges, np.quantile(reference, [0.2, 0.4, 0.6, 0.8]))
    assert np.all(np.diff(edges) > 0)
