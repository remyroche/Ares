"""Utilities for market factor and dimension analysis.

This package exposes high-level helpers that orchestrate the lower level
dimension discovery, clustering and validation utilities implemented in the
specialised modules.  The original audit flagged this ``__init__`` module for
exporting placeholder classes that contained ``pass`` statements which made the
public API unusable.  The helpers below provide concrete behaviour on top of the
fully fledged implementations in :mod:`dimension_discovery`,
:mod:`factor_extraction` and :mod:`feature_clustering`.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import system_logger

from .dimension_discovery import (
    DimensionAnalysisConfig,
    DimensionMetrics,
    MarketDimension,
    MarketDimensionAnalyzer,
)
from .factor_extraction import (
    AdvancedFeatureConfig,
    AdvancedMarkovFeatureEngine,
    FeatureTheme,
    LeakageSafeRollingStats,
)
from .feature_clustering import (
    ClusteringMethod,
    FeatureClusterResult,
    FeatureClusterer as _FeatureClusterer,
)
from .statistical_analysis import (
    DimensionalityMethod,
    DimensionalityResult,
    StatisticalDimensionAnalyzer,
)


class MarketFactorAnalyzer:
    """Main orchestrator for market factor analysis."""

    def __init__(self) -> None:
        self.feature_clusterer = _FeatureClusterer()
        self.discovered_dimensions: Dict[str, pd.DataFrame] = {}

    def discover_market_dimensions(self, feature_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Discover implicit market dimensions from the supplied features."""

        clustering_result = self.feature_clusterer.cluster_features(
            feature_data,
            method=ClusteringMethod.ENSEMBLE,
            similarity_threshold=0.6,
        )

        dimensions: Dict[str, pd.DataFrame] = {}
        for group_name, features in clustering_result.feature_groups.items():
            if features:
                dimensions[group_name] = feature_data[features]

        self.discovered_dimensions = dimensions
        return dimensions

    def extract_factors(self, feature_data: pd.DataFrame, n_factors: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Extract rotated factors from the feature set using PCA."""

        from sklearn.decomposition import PCA

        n_factors = n_factors or min(6, feature_data.shape[1])

        pca = PCA(n_components=n_factors)
        factor_scores = pca.fit_transform(feature_data.fillna(0))

        factor_df = pd.DataFrame(
            factor_scores,
            index=feature_data.index,
            columns=[f"factor_{i}" for i in range(n_factors)],
        )

        return {
            "factors": factor_df,
            "loadings": pd.DataFrame(
                pca.components_.T,
                index=feature_data.columns,
                columns=[f"factor_{i}" for i in range(n_factors)],
            ),
            "explained_variance": pd.Series(
                pca.explained_variance_ratio_,
                index=[f"factor_{i}" for i in range(n_factors)],
            ),
        }

    def cluster_features(self, feature_data: pd.DataFrame, similarity_threshold: float = 0.7) -> FeatureClusterResult:
        """Cluster features by similarity."""

        return self.feature_clusterer.cluster_features(
            feature_data,
            method=ClusteringMethod.CORRELATION,
            similarity_threshold=similarity_threshold,
        )


class DimensionDiscoverer:
    """High-level wrapper around :class:`MarketDimensionAnalyzer`."""

    def __init__(self, config: Optional[DimensionAnalysisConfig] = None) -> None:
        self.logger = system_logger.getChild("DimensionDiscoverer")
        self.analyzer = MarketDimensionAnalyzer(config)

    def discover(
        self,
        feature_data: pd.DataFrame,
        regime_labels: Optional[np.ndarray] = None,
        use_existing_features: bool = True,
    ) -> Dict[MarketDimension, DimensionMetrics]:
        """Run the comprehensive dimension analysis pipeline.

        Raises
        ------
        ValueError
            If ``feature_data`` is empty and no analysis can be performed.
        """

        if feature_data is None or feature_data.empty:
            raise ValueError("feature_data must contain at least one row")

        self.logger.info("🔍 Discovering market dimensions from feature matrix")
        return self.analyzer.analyze_all_dimensions(feature_data, regime_labels, use_existing_features)

    @staticmethod
    def summarise(results: Dict[MarketDimension, DimensionMetrics]) -> pd.DataFrame:
        """Convert analyzer results into a tabular summary."""

        records = [
            {
                "dimension": dimension.value,
                "importance_score": metrics.importance_score,
                "stability_score": metrics.stability_score,
                "predictive_power": metrics.predictive_power,
                "regime_discriminability": metrics.regime_discriminability,
                "feature_names": ", ".join(metrics.feature_names),
            }
            for dimension, metrics in results.items()
        ]

        return pd.DataFrame.from_records(records)


class FactorExtractor:
    """Coordinate statistical factor extraction utilities."""

    def __init__(self) -> None:
        self.logger = system_logger.getChild("FactorExtractor")
        self.statistical_analyzer = StatisticalDimensionAnalyzer()
        self.feature_engine = AdvancedMarkovFeatureEngine(AdvancedFeatureConfig())

    def extract(
        self,
        market_data: pd.DataFrame,
        methods: Optional[Iterable[DimensionalityMethod]] = None,
        n_components: Optional[int] = None,
    ) -> Dict[DimensionalityMethod, DimensionalityResult]:
        """Generate engineered features and run statistical factor extraction."""

        if market_data is None or market_data.empty:
            raise ValueError("market_data must contain at least one row")

        self.logger.info("🧮 Generating leakage-safe features for factor extraction")
        features = self.feature_engine.generate_features(market_data)
        feature_frame = features if isinstance(features, pd.DataFrame) else pd.DataFrame(features)

        self.logger.info("📊 Running statistical factor extraction")
        return self.statistical_analyzer.analyze_dimensions(
            feature_frame,
            list(methods) if methods else None,
            n_components,
        )

    @staticmethod
    def stack_transformed_data(results: Dict[DimensionalityMethod, DimensionalityResult]) -> pd.DataFrame:
        """Combine transformed datasets from multiple methods into a single DataFrame."""

        transformed: List[pd.DataFrame] = []
        for method, result in results.items():
            frame = pd.DataFrame(
                result.transformed_data,
                columns=[f"{method.name.lower()}_{i}" for i in range(result.n_components)],
            )
            transformed.append(frame)

        return pd.concat(transformed, axis=1) if transformed else pd.DataFrame()

    def generate_features_by_theme(
        self,
        market_data: pd.DataFrame,
        themes: Optional[Iterable[FeatureTheme]] = None,
    ) -> pd.DataFrame:
        """Expose themed feature generation for downstream analysis."""

        if market_data is None or market_data.empty:
            raise ValueError("market_data must contain at least one row")

        theme_list: Optional[List[FeatureTheme]]
        if themes is None:
            theme_list = None
        else:
            theme_list = [theme if isinstance(theme, FeatureTheme) else FeatureTheme(theme) for theme in themes]

        return self.feature_engine.generate_features(market_data, theme_filter=theme_list)

    def rolling_statistics(self) -> LeakageSafeRollingStats:
        """Return the leakage-safe rolling statistics helper used by the engine."""

        return self.feature_engine.rolling_stats


class DimensionValidator:
    """Validate discovered dimensions against configurable thresholds."""

    def __init__(
        self,
        importance_threshold: float = 0.05,
        stability_threshold: float = 0.2,
        predictive_power_threshold: float = 0.05,
    ) -> None:
        self.logger = system_logger.getChild("DimensionValidator")
        self.importance_threshold = importance_threshold
        self.stability_threshold = stability_threshold
        self.predictive_power_threshold = predictive_power_threshold

    def validate(self, metrics: Dict[MarketDimension, DimensionMetrics]) -> Dict[str, Dict[str, Tuple[bool, float]]]:
        """Evaluate whether each dimension meets the configured thresholds."""

        validation_report: Dict[str, Dict[str, Tuple[bool, float]]] = {}

        for dimension, dimension_metrics in metrics.items():
            dimension_key = dimension.value
            checks: Dict[str, Tuple[bool, float]] = {}

            checks["importance"] = (
                dimension_metrics.importance_score >= self.importance_threshold,
                dimension_metrics.importance_score,
            )
            checks["stability"] = (
                dimension_metrics.stability_score >= self.stability_threshold,
                dimension_metrics.stability_score,
            )
            checks["predictive_power"] = (
                dimension_metrics.predictive_power >= self.predictive_power_threshold,
                dimension_metrics.predictive_power,
            )

            validation_report[dimension_key] = checks

            if not all(flag for flag, _ in checks.values()):
                self.logger.warning("Dimension %s failed validation checks: %s", dimension_key, checks)
            else:
                self.logger.info("Dimension %s passed all validation checks", dimension_key)

        return validation_report


FeatureClusterer = _FeatureClusterer

__all__ = [
    "AdvancedFeatureConfig",
    "AdvancedMarkovFeatureEngine",
    "ClusteringMethod",
    "DimensionAnalysisConfig",
    "DimensionDiscoverer",
    "DimensionMetrics",
    "DimensionValidator",
    "DimensionalityMethod",
    "FactorExtractor",
    "FeatureClusterResult",
    "FeatureClusterer",
    "FeatureTheme",
    "LeakageSafeRollingStats",
    "MarketDimension",
    "MarketDimensionAnalyzer",
    "MarketFactorAnalyzer",
    "StatisticalDimensionAnalyzer",
]
