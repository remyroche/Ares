import numpy as np

from src.training.utils.embedding_postprocessing import filter_embedding_features


def test_filter_embedding_features_filters_correlated_and_low_ic_dimensions():
    rng = np.random.default_rng(42)
    parent = rng.normal(size=(256, 4))
    target = rng.normal(size=256)

    embeddings = rng.normal(size=(256, 12))
    embeddings[:, 0] = parent[:, 0]  # Perfect correlation with parent feature
    embeddings[:, 1] = parent[:, 1] * 0.9 + rng.normal(scale=0.01, size=256)  # High correlation
    embeddings[:, 2] = rng.normal(scale=1e-3, size=256)  # Near-zero information coefficient

    for idx in range(3, 12):
        embeddings[:, idx] = target * (0.25 + 0.02 * idx) + rng.normal(scale=0.05, size=256)

    filtered_embeddings, metadata = filter_embedding_features(
        parent_features=parent,
        embedding_features=embeddings,
        target=target,
        min_embeddings=6,
        max_embeddings=10,
    )

    assert 'embedding_0' in metadata['dropped_due_to_corr']
    assert 'embedding_1' in metadata['dropped_due_to_corr']
    assert 'embedding_2' in metadata['dropped_due_to_ic']
    assert metadata['within_budget']
    assert 6 <= filtered_embeddings.shape[1] <= 10


def test_filter_embedding_features_limits_maximum_embeddings():
    rng = np.random.default_rng(7)
    parent = rng.normal(size=(300, 3))
    target = rng.normal(size=300)

    embedding_columns = [
        target * (0.3 + 0.01 * i) + rng.normal(scale=0.05, size=300)
        for i in range(20)
    ]
    embeddings = np.column_stack(embedding_columns)

    filtered_embeddings, metadata = filter_embedding_features(
        parent_features=parent,
        embedding_features=embeddings,
        target=target,
        min_embeddings=6,
        max_embeddings=10,
    )

    assert filtered_embeddings.shape[1] == 10
    assert metadata['within_budget']
