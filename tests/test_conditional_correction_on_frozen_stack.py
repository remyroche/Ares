"""Pure helper tests for the anchored conditional-correction audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_conditional_correction_on_frozen_stack.py"
spec = importlib.util.spec_from_file_location("conditional_correction", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_empirical_rank_is_monotone_and_bounded() -> None:
    ranked = module._empirical_rank(np.array([-1.0, 0.0, 1.0, np.nan]), np.array([-1.0, 0.0, 1.0]))
    assert np.all(np.isfinite(ranked))
    assert np.all((ranked >= 0.0) & (ranked <= 1.0))
    assert np.all(np.diff(ranked[:3]) >= 0.0)
    assert ranked[-1] == 0.5


def test_sigmoid_is_bounded_and_saturates() -> None:
    values = module._sigmoid(np.array([-100.0, 0.0, 100.0]))
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert values[0] < 1e-10
    assert np.isclose(values[1], 0.5)
    assert values[2] > 1.0 - 1e-10


def test_tail_rows_are_global_and_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "score": [0.2, 0.9, 0.1, 0.8],
            "gross_bps": [10.0, 30.0, -5.0, 20.0],
            "net_bps": [-90.0, -70.0, -105.0, -80.0],
        }
    )
    rows = module._tail_rows(frame, "score")
    assert rows[0]["trades"] == 1
    assert rows[0]["net_bps_per_trade"] == -70.0
    assert rows[1]["trades"] == 1
    assert rows[1]["net_bps_per_trade"] == -70.0


def test_forbidden_outcome_fields_are_not_in_feature_groups() -> None:
    groups = {
        "head": ["incumbent_score"],
        "condition": ["condition__family__membership"],
        "context": ["regime_entropy"],
    }
    assert not any(field in module.FORBIDDEN for fields in groups.values() for field in fields)
