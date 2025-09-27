"""Aggregated helpers for market state clustering and validation."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.utils.logger import system_logger

from .optimal_cluster_selection import (
    DataDrivenClusteringConfig,
    DataDrivenClusteringFramework,
    DataDrivenClusteringResult,
)
from .similarity_clustering import (
    SimilarityClusteringConfig,
    SimilarityClusteringResult,
    SimilarityMatrixClusterer,
)
from .validation_metrics import (
    RegimeValidationMetrics,
    ValidationConfig,
    ValidationMetric,
    ValidationResult,
)


class MarketStateClusterer:
    """Main orchestrator for market state clustering."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.discovered_states: Dict[str, Dict[str, object]] = {}
        self.logger = system_logger.getChild("MarketStateClusterer")

    def discover_market_states(self, market_dimensions, n_clusters: Optional[int] = None):
        """Discover market states from market dimensions."""

        if isinstance(market_dimensions, dict):
            all_features = pd.concat(market_dimensions.values(), axis=1)
        else:
            all_features = market_dimensions

        all_features_clean = all_features.fillna(method="ffill").fillna(0)

        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(all_features_clean)

        features_scaled = self.scaler.fit_transform(all_features_clean)

        self.logger.info("Running KMeans clustering for market state discovery")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)

        state_labels = pd.Series(cluster_labels, index=all_features.index)

        distances = kmeans.transform(features_scaled)
        probabilities_array = np.exp(-distances) / np.exp(-distances).sum(axis=1, keepdims=True)

        state_probabilities = pd.DataFrame(
            probabilities_array,
            index=all_features.index,
            columns=[f"state_{i}" for i in range(n_clusters)],
        )

        cluster_profiles: Dict[str, Dict[str, object]] = {}
        for i in range(n_clusters):
            state_mask = state_labels == i
            state_features = all_features_clean[state_mask]

            cluster_profiles[f"state_{i}"] = {
                "size": int(state_mask.sum()),
                "frequency": float(state_mask.mean()),
                "feature_means": state_features.mean().to_dict(),
                "description": f"Market State {i} ({state_mask.sum()} periods, {state_mask.mean():.1%} frequency)",
            }

        results = {
            "labels": state_labels,
            "probabilities": state_probabilities,
            "profiles": cluster_profiles,
            "validation": {"n_clusters": n_clusters, "inertia": kmeans.inertia_},
        }

        self.discovered_states = results
        return results

    def find_optimal_clusters(self, market_dimensions, k_range: Tuple[int, int] = (2, 8)):
        """Find optimal number of clusters using a simple elbow method."""

        if isinstance(market_dimensions, dict):
            all_features = pd.concat(market_dimensions.values(), axis=1)
        else:
            all_features = market_dimensions

        all_features_clean = all_features.fillna(method="ffill").fillna(0)
        features_scaled = self.scaler.fit_transform(all_features_clean)

        inertias = []
        k_values = list(range(k_range[0], k_range[1] + 1))

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features_scaled)
            inertias.append(kmeans.inertia_)

        decreases = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
        optimal_k = k_values[np.argmax(decreases)]

        min_samples_per_cluster = 50
        max_k = len(all_features) // min_samples_per_cluster
        optimal_k = min(optimal_k, max_k)

        return max(2, optimal_k)

    def validate_clusters(self, market_dimensions, cluster_labels):
        """Validate cluster quality."""

        if isinstance(market_dimensions, dict):
            all_features = pd.concat(market_dimensions.values(), axis=1)
        else:
            all_features = market_dimensions

        all_features_clean = all_features.fillna(method="ffill").fillna(0)

        try:
            from sklearn.metrics import silhouette_score

            features_scaled = self.scaler.fit_transform(all_features_clean)
            silhouette = silhouette_score(features_scaled, cluster_labels)
        except Exception:  # pragma: no cover - optional dependency failures
            silhouette = 0.0

        unique_labels = np.unique(cluster_labels)
        cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels]

        validation_results = {
            "silhouette_score": float(silhouette),
            "n_clusters": int(len(unique_labels)),
            "cluster_sizes": cluster_sizes,
            "min_cluster_size": int(min(cluster_sizes)),
            "max_cluster_size": int(max(cluster_sizes)),
            "avg_cluster_size": float(np.mean(cluster_sizes)),
        }

        return validation_results


class RegimeDiscoverer:
    """Combine similarity clustering with optional data-driven optimisation."""

    def __init__(
        self,
        similarity_config: Optional[SimilarityClusteringConfig] = None,
        framework_config: Optional[DataDrivenClusteringConfig] = None,
    ) -> None:
        self.logger = system_logger.getChild("RegimeDiscoverer")
        self.clusterer = SimilarityMatrixClusterer(similarity_config)
        self.framework_config = framework_config
        self.framework: Optional[DataDrivenClusteringFramework] = (
            DataDrivenClusteringFramework(framework_config) if framework_config else None
        )

    def discover(
        self,
        features: pd.DataFrame,
        price_data: Optional[pd.DataFrame] = None,
        use_framework: bool = False,
    ) -> Union[SimilarityClusteringResult, DataDrivenClusteringResult]:
        """Run regime clustering with the requested strategy."""

        if features is None or features.empty:
            raise ValueError("features must contain observations for clustering")

        if use_framework:
            if self.framework is None:
                self.framework = DataDrivenClusteringFramework(self.framework_config)
            if price_data is None or price_data.empty:
                raise ValueError("price_data is required for data-driven clustering")

            self.logger.info("🚀 Running data-driven regime discovery framework")
            return self.framework.discover_optimal_regimes(features, price_data)

        self.logger.info("🔍 Running similarity matrix regime discovery")
        return self.clusterer.fit_predict(features, price_data)

    @staticmethod
    def summarise_similarity(result: SimilarityClusteringResult) -> Dict[str, Dict[str, float]]:
        """Create a serialisable summary of similarity clustering outputs."""

        summary: Dict[str, Dict[str, float]] = {}
        for validation in result.cluster_validations:
            summary[str(validation.cluster_id)] = {
                "cv_score": validation.cv_score,
                "similarity_score": validation.similarity_score,
                "economic_significance": validation.economic_significance,
                "n_samples": float(validation.n_samples),
            }
        return summary


class OptimalClusterSelector:
    """Wrap :class:`DataDrivenClusteringFramework` for optimal cluster discovery."""

    def __init__(self, framework: Optional[DataDrivenClusteringFramework] = None) -> None:
        self.logger = system_logger.getChild("OptimalClusterSelector")
        self.framework = framework or DataDrivenClusteringFramework()

    def select_optimal_clusters(
        self,
        features: pd.DataFrame,
        price_data: pd.DataFrame,
    ) -> Tuple[int, DataDrivenClusteringResult]:
        """Return the optimal cluster count and the full framework result."""

        if features is None or features.empty:
            raise ValueError("features must contain observations for clustering")
        if price_data is None or price_data.empty:
            raise ValueError("price_data is required for validation")

        self.logger.info("⚙️ Selecting optimal cluster configuration")
        result = self.framework.discover_optimal_regimes(features, price_data)
        return result.n_clusters, result


class SimilarityClusterer(SimilarityMatrixClusterer):
    """Backwards-compatible alias with convenience helpers."""

    def fit_predict_with_summary(
        self,
        features: pd.DataFrame,
        price_data: Optional[pd.DataFrame] = None,
    ) -> Tuple[SimilarityClusteringResult, Dict[str, Dict[str, float]]]:
        """Return both the raw result and a condensed summary."""

        result = self.fit_predict(features, price_data)
        summary = RegimeDiscoverer.summarise_similarity(result)
        return result, summary


class ClusterValidator:
    """Execute comprehensive cluster validation pipelines."""

    def __init__(self, config: Optional[ValidationConfig] = None) -> None:
        self.logger = system_logger.getChild("ClusterValidator")
        self.validator = RegimeValidationMetrics(config)

    def validate(
        self,
        market_data: pd.DataFrame,
        regime_labels: Union[np.ndarray, Iterable[int]],
        **kwargs,
    ) -> Dict[ValidationMetric, ValidationResult]:
        """Run statistical validation across all metrics."""

        labels = np.asarray(regime_labels)
        results = self.validator.validate_all_metrics(market_data, labels, **kwargs)
        return results

    def economic_summary(
        self,
        market_data: pd.DataFrame,
        regime_labels: Union[np.ndarray, Iterable[int]],
    ) -> Dict[str, object]:
        """Return the economic validation summary for the regimes."""

        labels = np.asarray(regime_labels)
        return self.validator.validate_economic_significance(market_data, labels)

    def composite_score(self, weights: Optional[Dict[ValidationMetric, float]] = None) -> float:
        """Expose the composite validation scoring helper."""

        score = self.validator.calculate_composite_score(weights)
        self.logger.info("Composite validation score: %.3f", score)
        return score


__all__ = [
    "ClusterValidator",
    "DataDrivenClusteringConfig",
    "DataDrivenClusteringFramework",
    "DataDrivenClusteringResult",
    "MarketStateClusterer",
    "OptimalClusterSelector",
    "RegimeDiscoverer",
    "SimilarityClusteringConfig",
    "SimilarityClusteringResult",
    "SimilarityClusterer",
    "ValidationConfig",
    "ValidationMetric",
]
