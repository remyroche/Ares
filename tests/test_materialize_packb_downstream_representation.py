from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_packb_downstream_representation import (
    DownstreamRepresentationError,
    append_side_representation,
)


def _context() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-04-01T00:00:00Z",
                    "2026-04-01T01:00:00Z",
                    "2026-04-01T00:00:00Z",
                    "2026-04-01T01:00:00Z",
                ],
                utc=True,
            ),
            "__symbol__": ["A", "B", "A", "B"],
            "side_name": ["long", "long", "short", "short"],
            "candidate_id": ["l1", "l2", "s1", "s2"],
        }
    )


def test_append_side_representation_preserves_order_and_side_scope() -> None:
    context = _context()
    side_frames = {
        "long": pd.DataFrame(
            {"dae_b16_00": [1.0, 2.0], "gmm_cluster_posterior_0": [0.8, 0.7]}
        ),
        "short": pd.DataFrame(
            {"dae_b16_00": [-1.0, -2.0], "gmm_cluster_posterior_0": [0.2, 0.3]}
        ),
    }

    result, report = append_side_representation(
        context,
        side_frames=side_frames,
        generated_features_by_side={
            side: list(frame.columns) for side, frame in side_frames.items()
        },
        minimum_joint_finite_fraction=1.0,
    )

    assert result["candidate_id"].tolist() == context["candidate_id"].tolist()
    assert result["dae_b16_00"].tolist() == [1.0, 2.0, -1.0, -2.0]
    assert report["generated_feature_count"] == 2
    assert report["coverage_by_side"]["long"]["joint_finite_fraction"] == 1.0
    assert all(
        result[column].dtype == np.float32
        for column in ("dae_b16_00", "gmm_cluster_posterior_0")
    )


def test_append_side_representation_fails_closed_on_sparse_generated_rows() -> None:
    context = _context()
    side_frames = {
        "long": pd.DataFrame({"dae_b16_00": [1.0, np.nan]}),
        "short": pd.DataFrame({"dae_b16_00": [1.0, 2.0]}),
    }

    with pytest.raises(DownstreamRepresentationError, match="joint finite"):
        append_side_representation(
            context,
            side_frames=side_frames,
            generated_features_by_side={
                side: list(frame.columns) for side, frame in side_frames.items()
            },
            minimum_joint_finite_fraction=0.75,
        )


def test_append_side_representation_rejects_different_side_contracts() -> None:
    context = _context()
    with pytest.raises(DownstreamRepresentationError, match="contracts differ"):
        append_side_representation(
            context,
            side_frames={
                "long": pd.DataFrame({"dae_b16_00": [1.0, 2.0]}),
                "short": pd.DataFrame({"gmm_entropy": [1.0, 2.0]}),
            },
            generated_features_by_side={
                "long": ["dae_b16_00"],
                "short": ["gmm_entropy"],
            },
            minimum_joint_finite_fraction=1.0,
        )
