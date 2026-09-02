from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "orthogonal_meta_funnel",
    ROOT / "scripts" / "run_strict_r3_orthogonal_meta_label_funnel.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _train() -> pd.DataFrame:
    decision = pd.to_datetime([
        "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z",
        "2026-01-01T01:00:00Z", "2026-01-01T01:00:00Z",
    ])
    return pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d"],
        "__decision_ts__": decision,
        "policy_net_bps": [250.0, -150.0, 80.0, -10.0],
        "base_rank_ts": [0.9, 0.8, 0.2, 0.3],
        "semantic_composite": ["clean_fast_persistent_winner", "early_adverse_failure", "clean_fast_persistent_winner", "no_opportunity_timeout"],
        "semantic_tbm_event": ["upper_first", "lower_first", "upper_first", "vertical"],
    })


def test_semantic_weights_are_bounded_and_training_only() -> None:
    train = _train()
    weights = MODULE._semantic_weights(train)
    assert weights.shape == (len(train),)
    assert np.isfinite(weights).all()
    assert (weights >= 0.25).all()
    assert (weights <= 4.0).all()
    # The weighting helper consumes resolved labels only; it emits a numeric
    # loss weight, not a candidate feature or score column.
    assert not any("semantic" in column for column in train.columns if column.startswith("om_"))


def test_rank_error_target_is_query_relative() -> None:
    grade, audit = MODULE._targets(_train(), "O5_base_rank_error_semantic")
    assert np.unique(grade).size >= 2
    assert audit["target_std"] > 0.0
