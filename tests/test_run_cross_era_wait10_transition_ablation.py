from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_cross_era_wait10_transition_ablation import (
    FEATURE_SETS,
    SCORE_COMMON,
    TRANSITION_COMMON,
    feature_shift_records,
)


def test_cross_era_feature_contract_is_explicitly_common() -> None:
    assert set(FEATURE_SETS["score_common"]) == set(SCORE_COMMON)
    assert set(FEATURE_SETS["transition_common"]) == set(TRANSITION_COMMON)
    assert set(FEATURE_SETS["score_plus_transition_common"]) == set(
        (*SCORE_COMMON, *TRANSITION_COMMON)
    )
    assert not set(SCORE_COMMON).intersection(TRANSITION_COMMON)


def test_feature_shift_uses_train_distribution_only() -> None:
    train = pd.DataFrame({"x": np.arange(100, dtype=float)})
    valid = pd.DataFrame({"x": [50.0, 200.0]})
    records = feature_shift_records(
        train,
        valid,
        ["x"],
        source="history",
        feature_set="one",
        evaluation="future",
        side="long",
    )
    assert len(records) == 1
    assert records[0]["evaluation_outside_train_1_99_rate"] == 0.5
    assert records[0]["train_median"] == 49.5
