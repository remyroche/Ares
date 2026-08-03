from __future__ import annotations

import pandas as pd
import pytest

from scripts.train_freeze_contextual_score_arms import (
    FrozenContextError,
    REGIME_FEATURES,
    TRANSITION_FEATURES,
    build_training_panel,
    feature_sets,
    fit_frozen_arms,
)


def _frame(stamp: str, ident: str, *, available: str | None = None) -> pd.DataFrame:
    when = pd.Timestamp(stamp, tz="UTC")
    row = {"candidate_id": ident, "__ts__": when, "__symbol__": "BTC/USD:USD", "side_name": "long",
           "baseline_context_free_raw_score": .2, "execution_net_ev_12h": .1,
           "execution_label_available_at": pd.Timestamp(available, tz="UTC") if available else when + pd.Timedelta(hours=13)}
    row.update({field: .1 for field in REGIME_FEATURES})
    row.update({field: .2 for field in TRANSITION_FEATURES})
    return pd.DataFrame([row])


def test_panel_rejects_label_resolved_at_or_after_freeze() -> None:
    historical = _frame("2024-12-01", "hist")
    after = _frame("2025-06-30 12:00", "after", available="2025-07-01 01:00")
    panel = build_training_panel(historical, after)
    assert panel.candidate_id.tolist() == ["hist"]
    assert panel.side_is_long.tolist() == [1.0]


def test_panel_rejects_duplicate_exact_identity() -> None:
    historical = _frame("2024-12-01", "dup")
    gap = _frame("2024-12-01", "dup")
    with pytest.raises(FrozenContextError, match="duplicate"):
        build_training_panel(historical, gap)


def test_fixed_models_cover_all_declared_arms() -> None:
    rows = []
    for number in range(12):
        frame = _frame(f"2024-12-{number + 1:02d}", str(number))
        frame.loc[:, "side_name"] = "long" if number % 2 else "short"
        frame.loc[:, "execution_net_ev_12h"] = number / 100
        rows.append(frame)
    panel = build_training_panel(pd.concat(rows[:6]), pd.concat(rows[6:]))
    models, diagnostics = fit_frozen_arms(panel)
    assert set(models) == set(feature_sets())
    assert diagnostics.set_index("arm").loc["combined", "features"] == len(feature_sets()["combined"])
