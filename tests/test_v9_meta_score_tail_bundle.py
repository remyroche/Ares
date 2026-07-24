from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.meta_historical_rank import HistoricalScoreRankReference
from extreme_price_movements.v9_meta_score_tail_bundle import MetaScoreV9TailBundle


def _bundle() -> MetaScoreV9TailBundle:
    reference = HistoricalScoreRankReference(
        score_col="score_meta_base_soft_label", side_col="side_name"
    ).fit(
        pd.DataFrame(
            {
                "side_name": ["long"] * 100 + ["short"] * 100,
                "score_meta_base_soft_label": np.concatenate(
                    [np.linspace(0, 1, 100), np.linspace(0, 2, 100)]
                ),
            }
        )
    )
    return MetaScoreV9TailBundle(
        historical_rank_reference=reference,
        local_references={
            ("long", "breakout", "adverse"): [
                ("resid_event_aegmm_expected_adverse_path_event", 1.0, np.linspace(0, 1, 1000))
            ]
        },
        threshold=0.95,
        alpha_down=0.01,
    )


def test_parent_rank_uses_side_specific_meta_score_reference() -> None:
    bundle = _bundle()
    frame = pd.DataFrame(
        {
            "side_name": ["long", "short"],
            "archetype_policy_key": ["breakout", "breakout"],
            "score_meta_base_soft_label": [0.90, 0.90],
        }
    )
    predicted = bundle.predict(frame)
    assert predicted.loc[0, "historical_rank"] > predicted.loc[1, "historical_rank"]


def test_tail_overlay_is_down_only_and_normalizes_live_archetype_prefix() -> None:
    bundle = _bundle()
    frame = pd.DataFrame(
        {
            "side_name": ["long", "long"],
            "archetype_policy_key": ["long__breakout", "long__breakout"],
            "score_meta_base_soft_label": [0.99, 0.50],
            "resid_event_aegmm_expected_adverse_path_event": [1.0, 1.0],
        }
    )
    predecessor = bundle.predict(frame)
    adjusted = bundle.apply_residual_overlay(frame, predecessor)
    assert adjusted.loc[0, "historical_rank"] < predecessor.loc[0, "historical_rank"]
    assert adjusted.loc[1, "historical_rank"] == predecessor.loc[1, "historical_rank"]
    assert np.all(
        adjusted["historical_rank"].to_numpy()
        <= predecessor["historical_rank"].to_numpy()
    )
