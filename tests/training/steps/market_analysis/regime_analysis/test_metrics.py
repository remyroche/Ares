import numpy as np

from src.training.steps.market_analysis.regime_analysis import metrics


def test_calculate_regime_distribution():
    labels = np.array([0, 0, 1, 2, 2, 2])
    distribution = metrics.calculate_regime_distribution(labels, "NAS")

    assert distribution["regime_counts"] == {"regime_0": 2, "regime_1": 1, "regime_2": 3}
    assert distribution["regime_type"] == "NAS"
    assert distribution["total_samples"] == 6


def test_calculate_clustering_metrics_returns_interpretation():
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=0.0, scale=0.05, size=(10, 3))
    cluster_b = rng.normal(loc=3.0, scale=0.05, size=(10, 3))
    features = np.vstack([cluster_a, cluster_b])
    labels = np.array([0] * len(cluster_a) + [1] * len(cluster_b))

    result = metrics.calculate_clustering_metrics(features, labels, "NAS")

    assert set(result.keys()) == {
        "regime_type",
        "silhouette_score",
        "davies_bouldin_score",
        "calinski_harabasz_score",
        "cv_score",
        "interpretation",
    }
    assert result["interpretation"]["silhouette"] in {
        "Excellent clustering",
        "Good clustering",
        "Fair clustering",
        "Poor clustering",
    }


def test_cv_interpretations_cover_ranges():
    assert metrics.interpret_cv_score(0.9) == "Excellent regime distinction"
    assert metrics.interpret_cv_score(0.7) == "Good regime distinction"
    assert metrics.interpret_cv_score(0.5) == "Fair regime distinction"
    assert metrics.interpret_cv_score(0.1) == "Poor regime distinction"
