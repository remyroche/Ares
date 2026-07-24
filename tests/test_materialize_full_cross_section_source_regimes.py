from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_frozen_aegmm_transition_context import (
    ALL_TRANSITION_FEATURES,
    TRANSITION_FEATURES,
    add_causal_aegmm_transition_features,
)
from scripts.materialize_full_cross_section_source_regimes import (
    materialize_full_universe_passthrough,
    materialize_source_regimes,
    retain_candidate_fraction,
)


def _panel() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        ["2026-06-01T00:00:00Z"] * 4 + ["2026-06-01T01:00:00Z"] * 4,
        utc=True,
    )
    return pd.DataFrame(
        {
            "__ts__": timestamps,
            "__symbol__": ["A", "B", "C", "D"] * 2,
            "trend_strength_percentile": np.array([0.1, 0.3, 0.7, 0.9] * 2, dtype=np.float32),
            "speed": np.array([0.2, 0.4, 0.6, 0.8] * 2, dtype=np.float32),
        }
    )


def test_source_scores_use_full_cross_section_not_candidate_subset() -> None:
    required = ["__regime_source_trend_path_score__"]
    scored, report = materialize_source_regimes(
        _panel(), required_columns=required, min_timestamp_symbols=4
    )

    assert report["min_symbols_per_timestamp"] == 4
    # Symbol A is not assigned the candidate-only neutral 0.5: the full
    # timestamp cross-section produces a genuinely low trend-source rank.
    first = scored.loc[scored["__symbol__"].eq("A")].iloc[0]
    assert float(first[required[0]]) < 0.5


def test_source_scores_require_full_universe_support() -> None:
    with np.testing.assert_raises_regex(ValueError, "lacks adequate cross-sectional coverage"):
        materialize_source_regimes(
            _panel().iloc[:3],
            required_columns=["__regime_source_trend_path_score__"],
            min_timestamp_symbols=4,
        )


def test_candidate_fraction_preserves_full_stream_score_percentile() -> None:
    candidates = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-06-01T00:00:00Z"] * 5, utc=True),
            "score": [0.10, 0.20, 0.30, 0.40, 0.50],
        }
    )

    retained = retain_candidate_fraction(candidates, fraction=0.20)

    assert len(retained) == 1
    assert float(retained.iloc[0]["base_rank_pct_by_timestamp"]) == 1.0
    # The next surviving rank would be 0.8 in the original stream, rather than
    # an artificial 1.0 after a top-20-only recomputation.
    retained_two = retain_candidate_fraction(candidates, fraction=0.40)
    np.testing.assert_allclose(
        retained_two["base_rank_pct_by_timestamp"].to_numpy(),
        np.array([0.8, 1.0]),
    )


def test_full_universe_passthrough_uses_market_median_and_asset_values() -> None:
    panel = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-06-01T00:00:00Z"] * 3, utc=True),
            "__symbol__": ["A", "B", "C"],
            "mkt_transition": [0.1, 0.3, 0.9],
            "asset_transition": [1.0, 2.0, 3.0],
        }
    )

    out = materialize_full_universe_passthrough(
        panel,
        feature_columns=["mkt_transition", "asset_transition"],
    )

    assert out["mkt_transition"].tolist() == [0.3, 0.3, 0.3]
    assert out["asset_transition"].tolist() == [1.0, 2.0, 3.0]


def test_aegmm_transition_features_are_causal_and_gap_safe() -> None:
    timestamps = pd.to_datetime(
        ["2026-06-01T00:00:00Z", "2026-06-01T01:00:00Z", "2026-06-01T03:00:00Z"],
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "__ts__": timestamps,
            "__symbol__": ["A"] * 3,
            "side_name": ["long"] * 3,
            "gmm_cluster_id": [0.0, 1.0, 1.0],
            "gmm_entropy": [0.2, 0.3, 0.5],
            "gmm_posterior_max": [0.9, 0.6, 0.7],
            "AE_reconstruction_error": [0.1, 0.2, 0.4],
            "mahalanobis_distance": [1.0, 2.0, 3.0],
            "gmm_ood_score": [0.0, 0.1, 0.2],
            **{f"gmm_prob_{idx}": ([0.9, 0.1, 0.2] if idx == 0 else [0.1, 0.9, 0.8] if idx == 1 else [0.0, 0.0, 0.0]) for idx in range(12)},
            **{f"dae_b16_{idx:02d}": ([0.0, 1.0, 2.0] if idx == 0 else [0.0, 0.0, 0.0]) for idx in range(16)},
        }
    )

    out = add_causal_aegmm_transition_features(frame)

    assert list(out.columns) == ["__ts__", "__symbol__", "side_name", *ALL_TRANSITION_FEATURES]
    assert np.isnan(out.loc[0, "meta_aegmm_transition_posterior_tv_1h"])
    assert out.loc[1, "meta_aegmm_transition_cluster_switch_1h"] == 1.0
    assert out.loc[1, "meta_aegmm_transition_posterior_tv_1h"] == pytest.approx(0.8)
    assert out.loc[1, "meta_aegmm_transition_prob_0_delta_1h"] == pytest.approx(-0.8)
    assert out.loc[1, "meta_aegmm_transition_prob_1_enter_breadth_1h"] == 1.0
    # The component change at 01:00 begins a new state run.  The first row of
    # a valid sequence is an age of zero and cannot fabricate a switch count.
    assert out.loc[1, "meta_aegmm_transition_dominant_state_age_24h_norm"] == 0.0
    assert out.loc[1, "meta_aegmm_transition_dominant_switch_count_4h"] == 1.0
    # There is no 02:00 row: the 03:00 value must not bridge a two-hour gap.
    assert np.isnan(out.loc[2, "meta_aegmm_transition_posterior_tv_1h"])
    assert out.loc[2, "meta_aegmm_transition_posterior_tv_4h"] != out.loc[2, "meta_aegmm_transition_posterior_tv_4h"]
    assert np.isnan(out.loc[2, "meta_aegmm_transition_dominant_switch_count_4h"])
