from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_meta_market_state_threshold_calibration import (
    KEYS,
    _merge_observable_context,
    _probability_map_quality,
)


def test_context_override_preserves_fixed_parent_rows(tmp_path) -> None:
    ts = pd.to_datetime(["2026-04-01", "2026-04-02"], utc=True)
    parent = pd.DataFrame(
        {
            "__ts__": ts,
            "__symbol__": ["A", "B"],
            "side_name": ["long", "short"],
            "archetype_policy_key": ["x", "y"],
            "resid_event_aegmm_gmm_entropy": [0.1, 0.2],
        }
    )
    context = pd.concat(
        [
            parent.loc[:, KEYS].assign(
                resid_event_aegmm_gmm_entropy=[0.8, 0.9],
                resid_event_aegmm_gmm_ood_score=[1.0, 2.0],
            ),
            pd.DataFrame(
                {
                    "__ts__": [pd.Timestamp("2026-04-03", tz="UTC")],
                    "__symbol__": ["C"],
                    "side_name": ["long"],
                    "archetype_policy_key": ["z"],
                    "resid_event_aegmm_gmm_entropy": [0.7],
                    "resid_event_aegmm_gmm_ood_score": [3.0],
                }
            ),
        ],
        ignore_index=True,
    )
    path = tmp_path / "context.parquet"
    context.to_parquet(path, index=False)

    result = _merge_observable_context(
        parent,
        state_artifact=path,
        expanded_source=None,
        override_existing=True,
    )

    assert result.loc[:, KEYS].reset_index(drop=True).equals(
        parent.loc[:, KEYS].reset_index(drop=True)
    )
    assert result["resid_event_aegmm_gmm_entropy"].tolist() == [0.8, 0.9]
    assert result["resid_event_aegmm_gmm_ood_score"].tolist() == [1.0, 2.0]


def test_probability_quality_rejects_collapsed_mapping() -> None:
    y = np.tile(np.array([0.0, 1.0]), 100)
    baseline = np.linspace(0.2, 0.8, len(y))
    collapsed = np.full(len(y), 0.99)

    quality = _probability_map_quality(collapsed, y, baseline)

    assert quality["valid"] is False


def test_probability_quality_accepts_informative_mapping() -> None:
    y = np.tile(np.array([0.0, 1.0]), 100)
    baseline = np.where(y > 0.5, 0.65, 0.35)
    mapped = np.clip(baseline + np.linspace(-0.03, 0.03, len(y)), 0.01, 0.99)

    quality = _probability_map_quality(mapped, y, baseline)

    assert quality["valid"] is True
