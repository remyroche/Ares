from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_strict_r3_p8u_precision_preservation_weight_funnel_v1.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("p8u_weight_funnel", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_weight_schemes_are_fixed_and_unique() -> None:
    assert MODULE._schemes("uniform,positive_125") == ("uniform", "positive_125")
    with pytest.raises(ValueError):
        MODULE._schemes("uniform,uniform")


def test_query_safe_weights_preserve_exact_query_mass_and_bounds() -> None:
    frame = pd.DataFrame({"__decision_ts__": pd.to_datetime(["2026-01-01T00:00Z"] * 3 + ["2026-01-01T01:00Z"] * 2)})
    weights = MODULE._query_safe_weights(frame, np.asarray([0, 3, 5, 1, 4]), "tail_linear_250")
    assert weights.min() >= .5 and weights.max() <= 2.0
    means = pd.Series(weights, index=frame["__decision_ts__"]).groupby(level=0).mean()
    assert np.allclose(means.to_numpy(), 1.0)


def test_strong_tail_weights_still_respect_the_exact_bounds() -> None:
    frame = pd.DataFrame({"__decision_ts__": pd.to_datetime(["2026-01-01T00:00Z"] * 10)})
    weights = MODULE._query_safe_weights(frame, np.asarray([0] * 9 + [5]), "tail_linear_250")
    assert weights.min() >= .5 and weights.max() <= 2.0
    assert np.isclose(weights.mean(), 1.0)


def test_weighting_development_months_remain_cross_year() -> None:
    assert len(MODULE._months("2025-11,2026-03,2026-07")) == 3
    with pytest.raises(ValueError):
        MODULE._months("2026-01,2026-03,2026-07")


def test_fitted_boosters_are_released_between_schemes() -> None:
    source = SCRIPT.read_text()
    assert "del model, x_train, x_held" in source
    assert 'stage="scheme_complete"' in source


def test_history_mode_is_explicitly_target_free_and_single_scheme() -> None:
    source = SCRIPT.read_text()
    assert "--history-only" in source
    assert "history-only mode requires exactly one frozen scheme" in source
    assert "held_scores_target_free_without_policy_join" in source


def test_declared_feature_contract_is_not_silently_restricted_to_72_fields() -> None:
    source = SCRIPT.read_text()
    assert "def _load_fields" in source
    assert "MIN_FEATURES = 16" in source
    assert "MAX_FEATURES = 160" in source
    assert "frozen_declared_feature_contract" in source


def test_shared_loader_ceiling_allows_the_declared_130_field_beam_finalist() -> None:
    shared = (SCRIPT.parent / "run_strict_r3_router_single_base_prescreen_v1.py").read_text()
    assert "1 <= len(fields) <= 160" in shared
