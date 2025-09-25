"""TAS analysis helpers built on the unified NAS/TAS shared utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from src.utils.nas_tas.shared_utils.analysis_components import (
    AnalysisComponentConfig,
    AnalysisResult,
    ClusterAnalyzer,
    RegimeAnalyzer,
    SharedClusteringUtilities,
    create_cluster_analyzer,
    create_regime_analyzer,
)

logger = logging.getLogger(__name__)


@dataclass
class TASAnalysisConfig:
    """Lightweight wrapper for configuring TAS analysis components."""

    analysis_config: AnalysisComponentConfig = AnalysisComponentConfig()
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if not isinstance(self.analysis_config, AnalysisComponentConfig):
            self.analysis_config = AnalysisComponentConfig(**(self.analysis_config or {}))
        if self.metadata is None:
            self.metadata = {}


class AnalysisComponents:
    """Thin orchestration layer that reuses the unified NAS/TAS analysis utilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.logger = logger.getChild("AnalysisComponents")
        self._config = TASAnalysisConfig(**config) if config else TASAnalysisConfig()

        self.analysis_config: AnalysisComponentConfig = self._config.analysis_config
        self.regime_analyzer: RegimeAnalyzer = create_regime_analyzer(self.analysis_config)
        self.cluster_analyzer: ClusterAnalyzer = create_cluster_analyzer(self.analysis_config)
        self.shared_clustering = SharedClusteringUtilities()
        self.metadata: Dict[str, Any] = self._config.metadata

        self.logger.info("✅ TAS AnalysisComponents initialized using shared NAS/TAS analyzers")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze_regimes(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        features: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> AnalysisResult:
        """Run regime analysis using the shared RegimeAnalyzer."""

        array = self._to_numpy(data)
        feature_array = self._to_numpy(features) if features is not None else None
        self.logger.debug("Running regime analysis with %s samples", len(array))
        return self.regime_analyzer.analyze(array, feature_array)

    def analyze_clusters(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        features: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> AnalysisResult:
        """Run cluster analysis via the shared ClusterAnalyzer."""

        array = self._to_numpy(data)
        feature_array = self._to_numpy(features) if features is not None else None
        self.logger.debug("Running cluster analysis with %s samples", len(array))
        return self.cluster_analyzer.analyze(array, feature_array)

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        analysis_type: str = "regime",
        features: Optional[pd.DataFrame] = None,
    ) -> AnalysisResult:
        """Convenience wrapper that accepts pandas objects directly."""

        if analysis_type not in {"regime", "cluster"}:
            raise ValueError(f"Unsupported analysis_type: {analysis_type}")

        if analysis_type == "regime":
            return self.analyze_regimes(df.values, features.values if features is not None else None)
        return self.analyze_clusters(df.values, features.values if features is not None else None)

    def perform_shared_clustering(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        n_clusters: int = 8,
        algorithm: str = "auto",
    ) -> Dict[str, Any]:
        """Expose the shared clustering helper for TAS workflows."""

        array = self._to_numpy(data)
        labels, centers, metrics = self.shared_clustering.perform_shared_clustering(array, n_clusters, algorithm)
        return {
            "labels": labels,
            "centers": centers,
            "metrics": metrics,
            "metadata": self.metadata,
        }

    def analyze_components(
        self,
        components: Any,
        data: Union[np.ndarray, pd.DataFrame],
        features: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """Compatibility helper that delegates to regime analysis."""

        result = self.analyze_regimes(data, features)
        return {
            "analysis_result": result,
            "components": components,
            "metadata": self.metadata,
        }

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        """No-op cleanup retained for legacy callers."""
        self.logger.debug("Cleanup invoked - no resources to release")

    def __enter__(self) -> "AnalysisComponents":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.cleanup()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _to_numpy(self, data: Union[np.ndarray, pd.DataFrame, None]) -> np.ndarray:
        if data is None:
            return np.empty((0, 0))
        if isinstance(data, pd.DataFrame):
            return data.values
        if isinstance(data, np.ndarray):
            return data
        raise TypeError(f"Unsupported data type: {type(data)!r}")


def create_analysis_components(config: Optional[Dict[str, Any]] = None) -> AnalysisComponents:
    """Factory helper retained for backwards compatibility."""

    return AnalysisComponents(config)


def analyze_components(
    components: Any,
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Legacy helper that now leverages the shared regime analyzer."""

    _ = kwargs  # kwargs retained for API stability
    with AnalysisComponents(config) as analyzer:
        return analyzer.analyze_components(components, X_test, y_test)


def analyze_dataframe(
    df: pd.DataFrame,
    analysis_type: str = "regime",
    config: Optional[Dict[str, Any]] = None,
) -> AnalysisResult:
    """Convenience wrapper mirroring the previous API."""

    with AnalysisComponents(config) as analyzer:
        return analyzer.analyze_dataframe(df, analysis_type)


# Flags kept for legacy checks – the shared utilities are now always available
UNIFIED_FRAMEWORK_AVAILABLE = True
ML_UTILS_AVAILABLE = True
MATRIX_OPS_AVAILABLE = True
DATA_UTILS_AVAILABLE = True


__all__ = [
    "AnalysisComponents",
    "create_analysis_components",
    "analyze_components",
    "analyze_dataframe",
    "UNIFIED_FRAMEWORK_AVAILABLE",
    "ML_UTILS_AVAILABLE",
    "MATRIX_OPS_AVAILABLE",
    "DATA_UTILS_AVAILABLE",
]
