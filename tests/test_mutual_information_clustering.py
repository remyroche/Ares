from __future__ import annotations

import numpy as np
import pytest

from extreme_price_movements.mutual_information_clustering import (
    FrozenEmbeddingMutualInformationClustering,
    MutualInformationClusteringConfig,
    torch_available,
)

pytestmark = pytest.mark.skipif(not torch_available(), reason="PyTorch is optional")


def _paired_embeddings() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    centers = np.asarray(
        [[-2.0, -1.0, 0.5], [0.0, 2.0, -0.5], [2.0, -1.0, 1.0]], dtype=np.float32
    )
    labels = np.repeat(np.arange(len(centers)), 12)
    weak = centers[labels] + rng.normal(0.0, 0.08, (len(labels), 3))
    strong = centers[labels] + rng.normal(0.0, 0.20, (len(labels), 3))
    return weak.astype(np.float32), strong.astype(np.float32)


def _config() -> MutualInformationClusteringConfig:
    return MutualInformationClusteringConfig(
        cluster_counts=(4, 6),
        overcluster_counts=(12,),
        shared_bottleneck_dim=8,
        reconstruction_weight=0.05,
        epochs=6,
        batch_size=12,
        learning_rate=0.01,
        random_state=17,
        device="cpu",
    )


def test_frozen_embedding_heads_emit_assignments_and_pair_diagnostics() -> None:
    weak, strong = _paired_embeddings()
    model = FrozenEmbeddingMutualInformationClustering(_config()).fit(weak, strong)

    diagnostics = model.diagnostics(weak, strong)
    assert set(diagnostics) == {"cluster_4", "cluster_6", "overcluster_12"}
    for name, values in diagnostics.items():
        probabilities = values["probabilities"]
        assert probabilities.dtype == np.float32
        assert probabilities.shape == (len(weak), int(name.rsplit("_", 1)[1]))
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        assert values["assignments"].shape == (len(weak),)
        assert np.all(
            (values["normalized_entropy"] >= 0.0)
            & (values["normalized_entropy"] <= 1.0)
        )
        assert np.all(values["margin"] >= 0.0)
        assert np.all(np.isfinite(values["augmentation_consistency"]))
        assert np.isclose(values["occupancy"].sum(), 1.0, atol=1e-6)
        assert np.isfinite(values["conditional_entropy"])


def test_fit_is_deterministic_and_state_round_trip_is_exact(tmp_path) -> None:
    weak, strong = _paired_embeddings()
    first = FrozenEmbeddingMutualInformationClustering(_config()).fit_pairs(weak, strong)
    second = FrozenEmbeddingMutualInformationClustering(_config()).fit(weak, strong)

    first_probabilities = first.predict_proba(weak)
    second_probabilities = second.predict_proba(weak)
    for name in first_probabilities:
        assert np.array_equal(first_probabilities[name], second_probabilities[name])

    path = tmp_path / "frozen_iic.pkl"
    first.save(path)
    restored = FrozenEmbeddingMutualInformationClustering.load(path)
    for name, probabilities in first_probabilities.items():
        assert np.array_equal(probabilities, restored.predict_proba(weak)[name])


def test_configuration_and_pair_shape_contracts_are_enforced() -> None:
    with pytest.raises(ValueError, match="reconstruction_weight requires"):
        MutualInformationClusteringConfig(reconstruction_weight=0.1)
    with pytest.raises(ValueError, match="unsupported"):
        MutualInformationClusteringConfig(cluster_counts=(5,))

    weak, strong = _paired_embeddings()
    model = FrozenEmbeddingMutualInformationClustering(_config())
    with pytest.raises(ValueError, match="identical shapes"):
        model.fit(weak, strong[:-1])
