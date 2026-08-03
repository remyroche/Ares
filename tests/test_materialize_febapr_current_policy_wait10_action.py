from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.materialize_febapr_current_policy_wait10_action import (
    MODEL_FEATURES,
    normalize_symbol,
)


def test_normalize_symbol_preserves_canonical_path_identity() -> None:
    values = pd.Series(["BTC/USD:USD", "ETH_USD:USD"])
    assert normalize_symbol(values).tolist() == ["BTC_USD:USD", "ETH_USD:USD"]


def test_action_model_features_are_strictly_preentry() -> None:
    forbidden_fragments = (
        "future",
        "target",
        "execution_",
        "label",
        "mfe",
        "mae",
        "wait_delta",
        "month",
    )
    assert len(MODEL_FEATURES) == len(set(MODEL_FEATURES))
    assert not [
        name
        for name in MODEL_FEATURES
        if any(fragment in name.lower() for fragment in forbidden_fragments)
    ]


def test_transition_context_is_explicit_in_action_contract() -> None:
    required = {
        "regime_stability_24h",
        "regime_transition_entropy_12h",
        "regime_transition_entropy_48h",
        "correlation_breakdown_dispersion",
        "leverage_build_score",
        "liquidation_onset_score",
    }
    assert required.issubset(MODEL_FEATURES)
    assert all(np.isfinite([10.0, 12.0]))
