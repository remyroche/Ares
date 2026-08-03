from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_trust_base_signal_mapping_ablation import (
    apply_pooled_global_topk,
    compose_arms,
    trust_mapping_targets,
)


def test_trust_target_rewards_positive_accurate_base_signal() -> None:
    targets = trust_mapping_targets(
        [0.01, -0.01, 0.05],
        [0.009, -0.009, 0.0],
    )
    trust = targets["trust_base_signal"]
    assert trust[0] > trust[1]
    assert trust[0] > trust[2]
    np.testing.assert_allclose(
        targets["residual_utility"],
        np.asarray([0.001, -0.001, 0.05], dtype=np.float32),
    )


def test_trust_is_never_exposed_as_raw_ranking_score() -> None:
    frozen = np.asarray([0.1, 0.2])
    residual = np.asarray([0.01, -0.01])
    error = np.asarray([0.02, 0.03])
    trust = np.asarray([0.9, 0.1])
    arms = compose_arms(frozen, residual, error, trust)
    for score, _ in arms.values():
        assert not np.array_equal(score, trust)


def test_pooled_topk_is_global_not_timestamp_local() -> None:
    frame = pd.DataFrame(
        {
            "arm": ["baseline"] * 20,
            "ranking_score": np.arange(20, dtype=float),
            "eligible": True,
        }
    )
    result = apply_pooled_global_topk(frame, top_fraction=0.10)
    assert result["pooled_global_selected"].sum() == 2
    assert result.loc[result["pooled_global_selected"]].index.tolist() == [
        18,
        19,
    ]
