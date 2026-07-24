from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.live_meta_feature_overlays import (
    BASE_ANCHOR_FEATURES,
    apply_live_meta_reliability_priors,
)
from extreme_price_movements.inference.feature_generator import (
    raw_required_feature_keys,
)


def test_live_reliability_overlay_materializes_all_frozen_base_anchors() -> None:
    features = pd.DataFrame(
        {"source_tag": ["long__model_frontier_top10"]}, index=["BTC/USD:USD"]
    )
    payload = {
        "feature_names": [],
        "score_reference_quantiles": [0.10, 0.20, 0.30, 0.40],
        "score_quantile_edges": [0.20, 0.30],
        "margin_quantile_edges": [-0.05, 0.0, 0.05],
        "global_prior": {"cutoff": 0.20, "score_mean": 0.25, "score_std": 0.10},
        "side_arch_priors": {
            "long|long__model_frontier_top10": {
                "cutoff": 0.30,
                "mean": 0.35,
                "std": 0.05,
                "rows": 1_000,
            }
        },
    }

    out = apply_live_meta_reliability_priors(
        features,
        side="long",
        base_predictions={"BTC/USD:USD": {"base_pred": 0.40}},
        prior_payload=payload,
    )

    assert set(BASE_ANCHOR_FEATURES).issubset(out.columns)
    np.testing.assert_allclose(out.loc["BTC/USD:USD", "score"], 0.40)
    np.testing.assert_allclose(
        out.loc["BTC/USD:USD", "base_score_rank_pct_train_prior"], 1.0
    )
    np.testing.assert_allclose(
        out.loc["BTC/USD:USD", "base_margin_to_cutoff"], 0.10
    )
    np.testing.assert_allclose(
        out.loc["BTC/USD:USD", "base_margin_to_cutoff_z"], 2.0
    )
    np.testing.assert_allclose(
        out.loc["BTC/USD:USD", "base_signal_zscore_within_archetype"], 1.0
    )


def test_post_base_anchors_are_not_requested_from_raw_feature_store() -> None:
    required = raw_required_feature_keys(
        {"ret1h", "volatility_zscore", *BASE_ANCHOR_FEATURES}
    )

    assert required == {"ret1h", "volatility_zscore"}
