from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.ablate_strict_r3_ldf_contract_blend import blend_outputs


def _frame(multiplier: tuple[float, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "__decision_ts__": pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True),
            "final_score": [0.8, 0.9],
            "policy_net_bps": [150.0, 200.0],
            "trust_size_multiplier": multiplier,
        }
    )


def test_blend_keeps_upstream_identity_and_bounds_multiplier() -> None:
    compact = _frame((0.25, 1.75))
    full = _frame((1.75, 0.25))
    output = blend_outputs(compact, full, compact_weight=0.5)
    assert output["candidate_id"].tolist() == ["a", "b"]
    assert np.allclose(output["trust_size_multiplier"], [1.0, 1.0])
    assert output["final_score"].equals(compact["final_score"])


def test_blend_rejects_nonidentical_frozen_score() -> None:
    compact = _frame((1.0, 1.0))
    full = _frame((1.0, 1.0))
    full.loc[1, "final_score"] = 0.95
    with pytest.raises(ValueError, match="final_score"):
        blend_outputs(compact, full, compact_weight=0.5)
