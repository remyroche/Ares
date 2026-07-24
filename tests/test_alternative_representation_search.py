from __future__ import annotations

import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from scripts.run_alternative_representation_search import (
    _dae_gmm_input,
    _downstream_candidate_rows,
    _validate_downstream_label_contract,
)
from scripts.report_dae150k_gmm_density_ablation import _next_stage_recommendation
from extreme_price_movements.alternative_latent_encoders import NativeLatentOutput
from extreme_price_movements.alternative_representation_search import (
    EncoderCandidate,
    cap_candidates,
    idec_final_candidates,
    materialize_representation_features,
    select_family_finalists,
    ssl_candidates,
)
from extreme_price_movements.representation_proxy_metrics import (
    GmmPanelFit,
    GmmPanelSpec,
)


def _panel() -> GmmPanelFit:
    return GmmPanelFit(
        GmmPanelSpec(2, "diag", 0.003),
        1,
        {
            "weights": np.asarray([0.5, 0.5]),
            "means": np.asarray([[-1.0, 0.0], [1.0, 0.0]]),
            "covariances": np.ones((2, 2)),
        },
    )


def test_ssl_policy_fields_are_active_encoder_fields() -> None:
    config = {
        "ssl": {
            "objectives": ["vicreg"],
            "policies": {
                "P3": {
                    "element_replace": 0.10,
                    "noise": 0.01,
                    "group_masks": 0,
                    "donor": "joint_group_empirical",
                }
            },
            "view_pairs": ["weak_strong"],
        }
    }
    candidate = ssl_candidates(config)[0]
    assert candidate.config["group_donor_replacement_rate"] == 0.10
    assert candidate.config["additive_noise_std"] == 0.01
    assert candidate.config["corruption_rate"] == 0.0


def test_family_selection_is_independent() -> None:
    rows = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "family": ["x", "x", "y", "y"],
            "best_robust_panel_score": [0.2, 0.8, 0.7, 0.1],
        }
    )
    ranked = select_family_finalists(rows, top_per_family=1)
    assert set(ranked.loc[ranked["promoted"], "candidate_id"]) == {"b", "c"}


def test_candidate_cap_preserves_search_axis_coverage() -> None:
    candidates = [
        EncoderCandidate(
            f"c{index}",
            "ssl",
            {
                "kind": "vicreg",
                "objective": objective,
                "policy": policy,
                "view_pair": view,
            },
        )
        for index, (objective, policy, view) in enumerate(
            [
                ("denoise", "P0", "weak_strong"),
                ("denoise", "P1", "weak_strong"),
                ("masked", "P2", "weak_strong"),
                ("scarf", "P3", "weak_weak"),
                ("scarf", "P4", "weak_strong"),
                ("vicreg", "P5", "weak_weak"),
                ("vicreg", "P0", "weak_strong"),
            ]
        )
    ]

    selected = cap_candidates(candidates, max_per_family=6, seed=7)

    assert {row.config["objective"] for row in selected} == {
        "denoise",
        "masked",
        "scarf",
        "vicreg",
    }
    assert {row.config["policy"] for row in selected} == {
        "P0",
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
    }
    assert {row.config["view_pair"] for row in selected} == {
        "weak_weak",
        "weak_strong",
    }


def test_idec_final_grid_inherits_proxy_component_count() -> None:
    config = {
        "idec": {
            "final": {
                "latent_dim": [8],
                "target_update_frequency": [1],
                "initialization": ["incumbent_gmm_means"],
                "student_t_df": [5.0],
                "output_mode": ["direct", "embedding_gmm", "embedding_only"],
            }
        }
    }
    parent = EncoderCandidate(
        "proxy_k8",
        "idec",
        {"kind": "idec", "latent_dim": 16, "n_clusters": 8, "pretraining_fraction": 0.66},
    )
    candidates = idec_final_candidates(config, [parent])
    assert len(candidates) == 3
    assert {candidate.config["n_clusters"] for candidate in candidates} == {8}
    assert {candidate.output_mode for candidate in candidates} == {
        "direct",
        "embedding_gmm",
        "embedding_only",
    }


def test_materializer_honors_embedding_only_and_elbo_outputs() -> None:
    keys = pd.DataFrame(
        {
            "__ts__": pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC"),
            "__symbol__": ["A"] * 4,
            "side": [1, 1, -1, -1],
        }
    )
    native = NativeLatentOutput(
        latent=np.asarray([[-1, 0], [-0.5, 0], [0.5, 0], [1, 0]], dtype=np.float32),
        reconstruction_error=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        cluster_probabilities=np.asarray([[0.9, 0.1]] * 4, dtype=np.float32),
        mean=np.zeros((4, 2), dtype=np.float32),
        logvar=np.zeros((4, 2), dtype=np.float32),
    )
    result = materialize_representation_features(
        keys=keys,
        native=native,
        panel=_panel(),
        reference_indices=np.arange(4),
        output_mode="embedding_only",
    )
    assert "repr_latent_00" in result
    assert "repr_elbo_novelty_reference_pct" in result
    assert not any("component_posterior" in column for column in result)
    assert not any("native_posterior" in column for column in result)


def test_explicit_downstream_density_selection_bypasses_proxy_rank(tmp_path) -> None:
    root = tmp_path / "search"
    root.mkdir()
    pd.DataFrame(
        {
            "candidate_id": ["dae", "dae"],
            "family": ["incumbent", "incumbent"],
            "density_id": ["dae__s1__k6", "dae__s1__k8"],
            "stage": [1, 1],
            "density_proxy_score": [0.9, 0.1],
        }
    ).to_csv(root / "density_stage1_summary.csv", index=False)
    pd.DataFrame(
        {
            "candidate_id": ["dae"],
            "output_mode": ["embedding_gmm"],
        }
    ).to_csv(root / "proxy_candidate_summary.csv", index=False)
    args = Namespace(
        output_root=root,
        downstream_density_ids=["dae__s1__k8", "dae__s1__k6"],
    )

    selected = _downstream_candidate_rows(args, top_per_family=1)

    assert selected["density_id"].tolist() == ["dae__s1__k8", "dae__s1__k6"]


def test_dae_gmm_input_includes_log_reconstruction_novelty() -> None:
    native = NativeLatentOutput(
        latent=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        reconstruction_error=np.asarray([0.0, 3.0], dtype=np.float32),
    )

    values, schema = _dae_gmm_input(native)

    assert values.shape == (2, 3)
    assert np.allclose(values[:, :2], native.latent)
    assert np.allclose(values[:, -1], np.log1p(native.reconstruction_error))
    assert schema["derivative_columns"] == ["reconstruction_error_log1p"]


def test_economic_winner_at_density_grid_edge_requires_local_continuation() -> None:
    config = {
        "gmm_search": {
            "stage1": {
                "components": [6, 8, 10, 12],
                "reg_covar": [0.001, 0.003, 0.01],
            }
        }
    }
    recommendation = _next_stage_recommendation(
        winner={
            "density_id": "dae__s1__whitened__diag__k6__r0.01",
            "economic_rank": 1,
            "components": 6,
            "reg_covar": 0.01,
            "latent_preprocessing": "whitened",
        },
        config=config,
    )

    assert recommendation["boundary_expansion_required"]
    assert recommendation["components"] == [4, 6, 8]
    assert recommendation["reg_covar"] == [0.003, 0.01, 0.03333333333333333]


def test_downstream_labels_require_causal_signal_close_offset(tmp_path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    contract = {
        "materialized_side_archetype_trailing_labels": {
            "timeframe": "1h",
            "entry_delay_hours": 1,
            "path_start_contract": (
                "signal_timestamp_plus_timeframe_then_optional_delayed_execution"
            ),
            "round_trip_cost": 0.01,
        }
    }
    (labels_dir / "labels_manifest.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )

    result = _validate_downstream_label_contract(labels_dir)

    assert result["entry_delay_hours"] == 1
    assert result["round_trip_cost"] == 0.01


def test_downstream_labels_reject_same_candle_path_contract(tmp_path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    contract = {
        "materialized_side_archetype_trailing_labels": {
            "timeframe": "1h",
            "entry_delay_hours": 0,
            "path_start_contract": "signal_timestamp",
            "round_trip_cost": 0.01,
        }
    }
    (labels_dir / "labels_manifest.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="canonical one-hour close offset"):
        _validate_downstream_label_contract(labels_dir)
