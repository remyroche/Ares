from __future__ import annotations

import pandas as pd
import pytest

from extreme_price_movements.structural_family_health import (
    StructuralFamilyHealthError,
    build_structural_family_historical_health,
)


def _frame() -> pd.DataFrame:
    decision = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    return pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d"],
        "decision_ts": decision,
        # b/c resolve exactly when d is scored and must therefore be excluded.
        "label_available_ts": decision + pd.Timedelta(hours=1),
        "net_bps": [100.0, -100.0, 100.0, 0.0],
        "base_expected_bps": [0.0, 0.0, 0.0, 0.0],
        "base_structural_family__abc": [1.0, 1.0, 1.0, 1.0],
        "frozen_aegmm_context": [0.0, 0.0, 0.0, 0.0],
    })


def test_health_uses_only_strictly_prior_resolved_labels() -> None:
    out = build_structural_family_historical_health(
        _frame(), context_columns=["frozen_aegmm_context"],
    ).set_index("candidate_id")
    assert out.loc["a", "structural_health__historical_log_support"] == pytest.approx(0.0)
    # At c (02:00), only a resolves strictly before; b itself is still
    # unresolved.  At d (03:00), a/b have resolved while c resolves exactly
    # at 03:00 and is still excluded.
    assert out.loc["c", "structural_health__historical_residual_bps"] == pytest.approx(100.0)
    assert out.loc["d", "structural_health__historical_residual_bps"] == pytest.approx(0.0)
    assert out.loc["c", "structural_health__context_compatibility"] == pytest.approx(1.0)


def test_health_rejects_raw_leaf_like_posterior_fields() -> None:
    frame = _frame().rename(columns={"base_structural_family__abc": "base_structural_family__leaf_abc"})
    with pytest.raises(StructuralFamilyHealthError, match="forbidden"):
        build_structural_family_historical_health(frame, context_columns=["frozen_aegmm_context"])
