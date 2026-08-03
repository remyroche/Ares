from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.audit_execution_ev_promotion_candidates import audit_candidate


def test_promotion_audit_uses_pooled_global_topk_and_latest_gates(tmp_path) -> None:
    rows = 400
    decision = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "execution_decision_utc": decision,
        "execution_net_ev_12h": np.where(np.arange(rows) % 2, 0.02, -0.01),
        "score": np.arange(rows, dtype=float),
        "side_name": np.where(np.arange(rows) % 2, "long", "short"),
        "oof_fold": np.repeat([0, 1], rows // 2),
    })
    path = tmp_path / "candidate.parquet"
    frame.to_parquet(path, index=False)
    promotion, slices, composition, deciles = audit_candidate(
        "candidate", path, "score", top_fraction=0.10,
        min_latest_month_rows=1, min_latest_7d_rows=1,
        min_side_rows=1, min_fold_coverage=0.99,
    )
    all_oof = next(row for row in slices if row["scope"] == "all_oof")
    assert all_oof["selected_rows"] == 40
    assert all_oof["selected_long_rows"] == 20
    assert all_oof["selected_short_rows"] == 20
    assert any(row["dimension"] == "fold" for row in composition)
    assert len(deciles) == 40
    assert promotion["gate__latest_month_coverage"]
    assert promotion["latest_7d_selected_rows"] == 40


def test_missing_latest_scores_fail_fold_coverage(tmp_path) -> None:
    rows = 400
    frame = pd.DataFrame({
        "execution_decision_utc": pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC"),
        "execution_net_ev_12h": 0.01,
        "score": [1.0] * 200 + [np.nan] * 200,
        "side_name": ["long", "short"] * (rows // 2),
        "oof_fold": [0] * 200 + [1] * 200,
    })
    path = tmp_path / "candidate.parquet"
    frame.to_parquet(path, index=False)
    promotion, *_ = audit_candidate(
        "candidate", path, "score", top_fraction=0.10,
        min_latest_month_rows=1, min_latest_7d_rows=1,
        min_side_rows=1, min_fold_coverage=0.99,
    )
    assert not promotion["gate__fold_score_coverage"]
    assert promotion["latest_7d_selected_rows"] == 0
    assert not promotion["promotion_eligible"]
