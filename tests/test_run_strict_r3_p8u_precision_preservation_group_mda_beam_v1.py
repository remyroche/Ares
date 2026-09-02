from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_strict_r3_p8u_precision_preservation_group_mda_beam_v1.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("p8u_group_mda_beam", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_development_months_remain_cross_year() -> None:
    assert [f"{item:%Y-%m}" for item in MODULE._months("2025-11,2026-03,2026-07")] == [
        "2025-11", "2026-03", "2026-07",
    ]
    with pytest.raises(ValueError):
        MODULE._months("2026-01,2026-03,2026-07")


def test_mda_permutation_never_crosses_timestamp_and_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["b", "a", "d", "c", "e"],
            "__decision_ts__": pd.to_datetime(
                ["2026-01-01T00:00Z", "2026-01-01T00:00Z", "2026-01-01T01:00Z", "2026-01-01T01:00Z", "2026-01-01T02:00Z"]
            ),
        }
    )
    first = MODULE._permutation(frame, 1729)
    second = MODULE._permutation(frame, 1729)
    assert np.array_equal(first, second)
    assert np.all(frame.loc[first, "__decision_ts__"].to_numpy() == frame["__decision_ts__"].to_numpy())
    assert first[4] == 4  # singleton timestamp is unchanged


def test_beam_variants_preserve_width_and_offer_bounded_swap() -> None:
    ranked = pd.DataFrame(
        {
            "feature": ["a", "b", "c", "d"],
            "gain_median": [1.0, 4.0, 3.0, 2.0],
            "random_subspace_inclusion_uplift": [4.0, 3.0, 1.0, 2.0],
        }
    )
    variants = MODULE._beam_variants(ranked, 3)
    assert set(variants) == {"mda_blend", "gain_swap", "inclusion_swap"}
    assert all(len(value) == 3 and len(set(value)) == 3 for value in variants.values())


def test_beam_uses_common_seed_for_all_feature_contracts() -> None:
    source = SCRIPT.read_text()
    assert "_score_fields(folds, candidate, params, SEED)" in source
    assert "SEED + 100_000 + size" not in source
