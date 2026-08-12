from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "run_ten_head_k9_consensus_ablation.py"
SPEC = importlib.util.spec_from_file_location("ten_head_k9_consensus", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _sample() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = 2_000
    target = rng.integers(0, 5, rows)
    frame = pd.DataFrame(
        {
            "__target__": target,
            "prequential_base_rank42": rng.random(rows),
        }
    )
    for position, field in enumerate(MODULE.K9_MEMBERSHIPS):
        frame[field] = (
            target / 4.0 + rng.normal(0.0, 0.03, rows)
            if position == 2
            else rng.random(rows)
        )
    return frame


def test_cmi_selector_is_train_only_and_selects_three_memberships() -> None:
    result = MODULE.conditional_membership_mi(_sample())
    assert result["selected"].sum() == 3
    assert result.iloc[0]["field"] == MODULE.K9_MEMBERSHIPS[2]
    assert set(result["field"]) == set(MODULE.K9_MEMBERSHIPS)


def test_k9_modes_have_distinct_stable_contracts() -> None:
    cmi = MODULE.conditional_membership_mi(_sample())
    assert MODULE._mode_fields("conditional_none", cmi) == ()
    assert MODULE._mode_fields("conditional_summary", cmi) == MODULE.K9_SUMMARIES
    assert MODULE._mode_fields("conditional_raw9", cmi) == MODULE.K9_MEMBERSHIPS
    selected = MODULE._mode_fields("conditional_cmi3", cmi)
    assert selected[:3] == MODULE.K9_SUMMARIES
    assert len(selected) == 6


def test_frozen_k9_definition_precedes_every_scored_fold() -> None:
    assert MODULE.K9_FIT_END == pd.Timestamp("2025-01-01", tz="UTC")
    assert MODULE.K9_FIT_START == pd.Timestamp("2024-10-01", tz="UTC")
    assert MODULE.K9_FIT_END <= pd.Timestamp("2025-01-01", tz="UTC")
