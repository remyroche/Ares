from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_chronological_regime_observability import (
    DECISION,
    RESOLUTION,
    TARGET,
    WEEK,
    add_week_economic_labels,
    chronological_observability,
    feature_families,
)


def _frame() -> pd.DataFrame:
    records = []
    for week_index in range(8):
        week = pd.Timestamp("2026-05-04", tz="UTC") + pd.Timedelta(days=7 * week_index)
        for row in range(20):
            decision = week + pd.Timedelta(hours=row)
            profitable = week_index % 2
            records.append(
                {
                    "__ts__": decision,
                    "__symbol__": f"S{row % 3}",
                    "side_name": "long" if row % 2 else "short",
                    "candidate_id": f"{week_index}-{row}",
                    DECISION: decision,
                    RESOLUTION: decision + pd.Timedelta(hours=12),
                    TARGET: 0.01 if profitable else -0.01,
                    "score": float(row),
                    "existing_alpha_ev": float(profitable) + row / 1000.0,
                    "mkt_state__atr_slope__h0": float(profitable),
                }
            )
    return pd.DataFrame(records)


def test_week_label_availability_and_global_topk() -> None:
    frame, weeks = add_week_economic_labels(
        _frame(), score_column="score", top_k_fraction=0.10
    )
    assert len(weeks) == 8
    assert set(weeks["topk_rows"]) == {2}
    assert set(weeks["population_rows"]) == {20}
    assert (weeks["week_label_available_at"] >= weeks["week_end_exclusive"]).all()
    assert frame[WEEK].nunique() == 8


def test_chronological_classifier_excludes_unresolved_previous_week() -> None:
    frame, weeks = add_week_economic_labels(
        _frame(), score_column="score", top_k_fraction=0.10
    )
    families = {"tiny": ["existing_alpha_ev", "mkt_state__atr_slope__h0"]}
    predictions, folds, importance = chronological_observability(
        frame,
        weeks,
        families,
        first_evaluation=pd.Timestamp("2026-06-15", tz="UTC"),
        min_train_weeks=4,
        min_feature_coverage=1.0,
        seed=42,
    )
    assert not predictions.empty
    assert predictions["observability_oos"].all()
    first = folds.sort_values("evaluation_week").iloc[0]
    evaluation = pd.Timestamp(first["evaluation_week"])
    eligible = weeks.loc[weeks["week_label_available_at"] < evaluation]
    assert first["train_weeks"] == len(eligible)
    assert not importance.empty


def test_feature_families_use_h0_only() -> None:
    families = feature_families(
        [
            "existing_alpha_ev",
            "mkt_state__atr_slope__h0",
            "mkt_state__atr_slope__h1",
            "mkt_state__market_breadth_4h__h0",
        ]
    )
    assert "existing_alpha_ev" in families["head_context"]
    assert "mkt_state__atr_slope__h0" in families["volatility"]
    assert "mkt_state__atr_slope__h1" not in families["market_h0"]
    assert "mkt_state__market_breadth_4h__h0" in families["breadth"]
