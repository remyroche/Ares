import numpy as np
import pandas as pd

from scripts.materialize_prehead_symbol_guard_ablation_candidates import (
    GuardConfig,
    _apply_variant,
    _rank_failure_decisions,
)


def _rank_failure_frame() -> pd.DataFrame:
    rows = []
    row_pos = 0
    # Prior day: top decile loses while 70-90 band wins for head_a.
    for rank, ret in [(0.95, -0.02), (0.94, -0.01), (0.80, 0.02), (0.78, 0.01)]:
        rows.append(
            {
                "__row_pos": row_pos,
                "timestamp": pd.Timestamp("2026-06-01", tz="UTC"),
                "head": "head_a",
                "symbol": f"S{row_pos}",
                "side": "long",
                "__rank_for_guard": rank,
                "__guard_net_return": ret,
                "normalized_rank_score": rank,
                "policy_rank_pct": rank,
                "rank_pct": rank,
                "strategy_rank_pct": rank,
            }
        )
        row_pos += 1
    # Scoring day rows should inherit the rank-failure state.
    for rank in (0.96, 0.75):
        rows.append(
            {
                "__row_pos": row_pos,
                "timestamp": pd.Timestamp("2026-06-02", tz="UTC"),
                "head": "head_a",
                "symbol": f"S{row_pos}",
                "side": "long",
                "__rank_for_guard": rank,
                "__guard_net_return": 0.0,
                "normalized_rank_score": rank,
                "policy_rank_pct": rank,
                "rank_pct": rank,
                "strategy_rank_pct": rank,
            }
        )
        row_pos += 1
    return pd.DataFrame(rows)


def test_rank_failure_soft_penalizes_following_day_scores():
    frame = _rank_failure_frame()
    cfg = GuardConfig(
        "rank_soft_test",
        scope="head",
        mode="rank_failure_soft",
        rank_min_top_count=2,
        rank_min_lower_count=2,
        rank_hr_margin=0.10,
        rank_soft_penalty=0.07,
        rank_severe_penalty=0.12,
        rank_require_both_edges=True,
    )

    decisions = _rank_failure_decisions(frame, cfg)
    applied = _apply_variant(frame, cfg, decisions)
    scored = applied.loc[applied["timestamp"].eq(pd.Timestamp("2026-06-02", tz="UTC"))]

    assert scored["prehead_symbol_guard_penalty"].tolist() == [0.12, 0.12]
    assert scored["prehead_symbol_guard_reason"].tolist() == [
        "rank_failure_severe",
        "rank_failure_severe",
    ]
    assert np.allclose(scored["policy_rank_pct"], [0.84, 0.63])
    assert np.allclose(scored["prehead_symbol_guard_original_policy_rank_pct"], [0.96, 0.75])


def test_rank_failure_hard_removes_following_day_rows():
    frame = _rank_failure_frame()
    cfg = GuardConfig(
        "rank_hard_test",
        scope="head",
        mode="rank_failure_hard",
        rank_min_top_count=2,
        rank_min_lower_count=2,
        rank_hr_margin=0.10,
        rank_require_both_edges=True,
    )

    decisions = _rank_failure_decisions(frame, cfg)
    applied = _apply_variant(frame, cfg, decisions)

    assert len(applied.loc[applied["timestamp"].eq(pd.Timestamp("2026-06-02", tz="UTC"))]) == 0
