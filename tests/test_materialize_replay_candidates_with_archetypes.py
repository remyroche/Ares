import pandas as pd

from scripts.materialize_replay_candidates_with_archetypes import (
    _deduplicate_decisions,
    _join_archetype_ledger,
    _materialize_policy_archetype,
)


def test_join_archetype_ledger_and_dedupe_decision_rows() -> None:
    candidates = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-05-01 00:00:00", "2026-05-01 00:00:00"], utc=True
            ),
            "symbol": ["BTC/USD:USD", "BTC/USD:USD"],
            "side": ["short", "short"],
            "strategy_id": ["short_meta", "short_meta"],
            "meta_variant": ["weak", "strong"],
            "meta_score_rank_pct": [0.2, 0.8],
            "meta_score_rank_pct_selected": [0.1, 0.9],
            "calibrated_score": [0.0, 1.0],
            "rank_pct": [0.2, 0.8],
        }
    )
    ledger = pd.DataFrame(
        {
            "timestamp": candidates["timestamp"],
            "symbol": candidates["symbol"],
            "side": candidates["side"],
            "meta_variant": candidates["meta_variant"],
            "meta_score_rank_pct": candidates["meta_score_rank_pct"],
            "local_side_archetype": ["short_0", "short_1"],
        }
    )

    joined, join_report = _join_archetype_ledger(
        candidates,
        ledger,
        join_keys=["timestamp", "symbol", "side", "meta_variant", "meta_score_rank_pct"],
    )
    with_archetype = _materialize_policy_archetype(joined)
    deduped, dedupe_report = _deduplicate_decisions(
        with_archetype,
        keys=["timestamp", "symbol", "strategy_id"],
        sort_columns=["meta_score_rank_pct_selected", "meta_score_rank_pct"],
    )

    assert join_report["matched_rows"] == 2
    assert dedupe_report["before_rows"] == 2
    assert dedupe_report["after_rows"] == 1
    assert deduped["policy_archetype"].tolist() == ["short_1"]
