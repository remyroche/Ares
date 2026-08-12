from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.leaf_failure_explainability import (
    LeafFailureExplainabilityError,
    analyze_leaf_failure_explainability,
    write_leaf_failure_explainability,
)


def _frame(rows: int = 10) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq="MS", tz="UTC")
    x = np.linspace(-1.0, 1.0, rows)
    return pd.DataFrame({
        "side_name": "long", "layer": "base", "head_name": "p_clear", "rule_signature": "r:abc",
        "period_start": ts, "label_available_ts": ts - pd.Timedelta(hours=1), "feature_generation_ts": ts,
        "economic_effect": 0.5 * x, "effect_standard_error": 0.1,
        "support": np.arange(rows) + 10, "context": x, "covariance_break": x * 0.2,
    })


def test_chronological_ladder_and_immutable_writer(tmp_path):
    result = analyze_leaf_failure_explainability(
        _frame(),
        groups={"D2": ["support"], "D4": ["context"], "D6": ["covariance_break"]},
    )
    assert set(result.diagnostics["step"]) == {"D0", "D1", "D2", "D3", "D4", "D5", "D6"}
    assert result.predictions["period_start"].min() == pd.Timestamp("2024-05-01", tz="UTC")
    out = write_leaf_failure_explainability(result, tmp_path / "diagnostic")
    assert (out / "leaf_failure_classification.yaml").is_file()
    with pytest.raises(FileExistsError):
        write_leaf_failure_explainability(result, out)


def test_rejects_unresolved_or_raw_leaf_contract():
    bad = _frame()
    bad.loc[0, "label_available_ts"] = bad.loc[0, "feature_generation_ts"]
    with pytest.raises(LeafFailureExplainabilityError, match="unresolved"):
        analyze_leaf_failure_explainability(bad, groups={})
    bad = _frame()
    bad["leaf_token"] = 1
    with pytest.raises(LeafFailureExplainabilityError, match="raw leaf"):
        analyze_leaf_failure_explainability(bad, groups={})
