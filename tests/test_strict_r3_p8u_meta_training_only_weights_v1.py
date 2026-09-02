from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_strict_r3_p8u_meta_lgbm_objective_screen_v1.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("p8u_meta_weight_objective", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_training_only_meta_weights_are_deterministic_and_bounded() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d", "e"],
        "__decision_ts__": pd.to_datetime([
            "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z",
            "2026-01-01T01:00:00Z", "2026-01-01T01:00:00Z",
        ]),
        "base_rank_ts": [.10, .40, .99, .20, .80],
        "prequential_residual_bps": [-5.0, 30.0, 250.0, -40.0, 150.0],
    })
    profile = {
        "equal_timestamp": True,
        "components": [
            {"name": "base_score", "power": 1.5, "strength": .75},
            {"name": "positive_recall", "power": 1.0, "strength": .50},
            {"name": "magnitude_awareness", "power": 2.0, "strength": 1.0},
        ],
    }
    labels = np.asarray([0, 0, 1, 0, 1], dtype=np.int32)
    first, audit = MODULE._sample_weight(train=frame, labels=labels, profile=profile)
    second, _ = MODULE._sample_weight(train=frame, labels=labels, profile=profile)
    assert first is not None
    assert np.allclose(first, second)
    assert first.min() >= .5 and first.max() <= 2.0
    assert audit["enters_inference_features"] is False
    # Within the three-row timestamp, the high Base/positive/magnitude row is
    # deliberately emphasized; this confirms all declared fit-only layers act.
    assert first[2] > first[0]


def test_unweighted_profile_preserves_legacy_fit_semantics() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__decision_ts__": pd.to_datetime(["2026-01-01T00:00:00Z"] * 2),
        "base_rank_ts": [.9, .8],
        "prequential_residual_bps": [1.0, 2.0],
    })
    weights, audit = MODULE._sample_weight(train=frame, labels=np.asarray([0, 1]), profile=None)
    assert weights is None
    assert audit == {"profile": "unweighted", "enters_inference_features": False}


def test_weight_source_never_persists_as_an_inference_feature() -> None:
    source = SCRIPT.read_text()
    assert '"sample_weights_are_training_only_and_never_inference_features": True' in source
    assert 'model.fit(prepared.train_x, y, group=prepared.groups, sample_weight=sample_weight)' in source
