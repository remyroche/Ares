from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.meta_residual_archetypes import (
    OUTCOME_COLUMNS,
    ResidualArchetypeConfig,
    strip_outcomes_for_oos,
)
from extreme_price_movements.meta_residual_surprise_heads import (
    SURPRISE_HEAD_OUTPUTS,
    ResidualSurpriseHeadState,
)


def _frame(rows: int = 12_000) -> pd.DataFrame:
    rng = np.random.default_rng(52)
    ts = pd.date_range("2025-01-01", periods=rows // 8, freq="h", tz="UTC").repeat(8)
    side = np.where(np.arange(rows) % 2 == 0, "long", "short")
    archetype = np.where(np.arange(rows) % 4 < 2, "continuation", "compression")
    shock = rng.normal(0.0, 1.0, rows).astype(np.float32)
    breadth = rng.normal(0.0, 1.0, rows).astype(np.float32)
    score = np.clip(
        0.55 + 0.16 * shock - 0.08 * breadth + rng.normal(0.0, 0.05, rows),
        0.01,
        0.99,
    ).astype(np.float32)
    clean_prob = np.clip(
        score - 0.25 * (shock > 1.0) + 0.20 * (breadth > 1.0), 0.02, 0.98
    )
    clean = (rng.random(rows) < clean_prob).astype(np.float32)
    bad_mae = ((shock > 0.8) & (clean < 0.5)).astype(np.float32)
    timeout = ((breadth < -1.3) & (clean < 0.5)).astype(np.float32)
    ev = (0.012 * clean - 0.016 * (1.0 - clean) + 0.003 * breadth).astype(np.float32)
    return pd.DataFrame(
        {
            "__ts__": ts,
            "__symbol__": [f"S{i % 8}" for i in range(rows)],
            "side_name": side,
            "archetype_policy_key": archetype,
            "oos_fold": ts.to_period("M").astype(str),
            "score_meta_base_soft_label": score,
            "clean_exec": clean,
            "dirty_positive": ((ev > 0.0) & (bad_mae > 0.5)).astype(np.float32),
            "full_path_bad_mae_1r": bad_mae,
            "timeout": timeout,
            "ev_after_1pct": ev,
            "mkt_shock": shock,
            "market_breadth": breadth,
            "oi_flush": (shock - breadth).astype(np.float32),
            "base_score": score,
        }
    )


def test_surprise_head_is_side_specific_and_oos_transform_is_outcome_free() -> None:
    frame = _frame(12_000)
    state = ResidualSurpriseHeadState(
        candidate_features=["mkt_shock", "market_breadth", "oi_flush", "base_score"],
        config=ResidualArchetypeConfig(min_side_rows=300, random_state=17),
        max_fit_rows_per_side=1_000,
    ).fit(frame.iloc[:10_000].copy())
    assert set(state.side_models) == {"long", "short"}
    safe = strip_outcomes_for_oos(frame.iloc[10_000:].copy())
    assert not (set(safe.columns) & OUTCOME_COLUMNS)
    output = state.transform(safe)
    assert list(output.columns) == list(SURPRISE_HEAD_OUTPUTS)
    assert np.isfinite(output.to_numpy(dtype=np.float32)).all()
    assert output["meta_resid_negative_tail_probability"].between(0.0, 1.0).all()
    assert output["meta_resid_positive_tail_probability"].between(0.0, 1.0).all()
    assert output["meta_resid_surprise_head_support_log1p"].gt(0.0).all()


def test_surprise_head_unknown_side_uses_neutral_zero_outputs() -> None:
    frame = _frame(12_000)
    state = ResidualSurpriseHeadState(
        candidate_features=["mkt_shock", "market_breadth", "oi_flush", "base_score"],
        config=ResidualArchetypeConfig(min_side_rows=300, random_state=23),
        max_fit_rows_per_side=1_000,
    ).fit(frame.iloc[:10_000].copy())
    safe = strip_outcomes_for_oos(frame.iloc[10_000:10_020].copy())
    safe["side_name"] = "unknown"
    output = state.transform(safe)
    assert output.eq(0.0).all().all()
