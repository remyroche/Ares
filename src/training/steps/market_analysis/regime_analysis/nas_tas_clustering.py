"""Pipeline implementation for the NAS-TAS clustering component."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.utils.tprint import tprint

from ..shared_utils import (
    calculate_economic_scores,
    calculate_stability_scores,
    calculate_trading_scores,
    generate_cluster_characteristics,
    prepare_market_features,
)


class NASTASClusteringPipeline:
    """Small pipeline that runs the NAS-TAS clustering stages."""

    def __init__(self, component: "NASTASClusteringComponent") -> None:
        self._component = component

    @property
    def _config(self):
        return self._component.config

    @property
    def _metrics_calculator(self):
        return self._component.metrics_calculator

    def extract_regime_counts(self, pipeline_state: Dict[str, Any]) -> int:
        """Extract the number of regimes to use for clustering."""
        tprint("📈 Step 1: Extracting regime count from previous step artifacts...", "INFO")

        regime_discovery_result = pipeline_state.get("nas_tas_regime_discovery_result", {})
        tas_regime_count = regime_discovery_result.get("tas_regime_count", 8)
        nas_regime_count = regime_discovery_result.get("nas_regime_count", 8)

        n_regimes = (
            max(tas_regime_count, nas_regime_count)
            if tas_regime_count and nas_regime_count
            else 8
        )
        n_regimes = max(5, min(15, n_regimes))

        tprint(
            f"Extracted regime counts - TAS: {tas_regime_count}, NAS: {nas_regime_count}, Using: {n_regimes}",
            "INFO",
        )
        return n_regimes

    def validate_configuration(self) -> None:
        """Validate component configuration using shared utilities."""
        tprint("Step 2: Validating inputs and configuration using shared utilities", "INFO")
        validation_errors = self._component.config_validator.validate_config(self._config)
        if validation_errors:
            tprint(f"Configuration validation failed: {validation_errors}", "ERROR")
            raise ValueError(f"Configuration validation failed: {validation_errors}")

        tprint("Configuration validation passed using shared utilities", "SUCCESS")

    def prepare_features(self, market_data: pd.DataFrame) -> Any:
        """Prepare market features for clustering."""
        tprint("Step 4: Preparing features using shared utilities", "INFO")
        features = prepare_market_features(market_data, self._component.feature_config, verbose=True)
        if features is None:
            tprint("Failed to prepare features for clustering", "ERROR")
            raise ValueError("Failed to prepare features for clustering")

        self._component.features = features
        tprint(f"Features prepared: {features.shape}", "SUCCESS")
        return features

    async def perform_clustering(self, features: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering using advanced optimization methods."""
        try:
            tprint("Performing advanced clustering optimization...", "INFO")

            clustering_result = await self._component._perform_advanced_clustering(features, market_data)
            tprint("Advanced clustering optimization completed", "SUCCESS")

            tprint(f"Clustering completed: {clustering_result['n_clusters']} clusters", "SUCCESS")
            return clustering_result

        except Exception as exc:  # pragma: no cover - error path
            tprint(f"Clustering failed: {exc}", "ERROR")
            raise ValueError(f"Clustering failed: {exc}")

    def generate_cluster_characteristics(
        self,
        market_data: pd.DataFrame,
        clustering_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate characteristics for each cluster."""
        tprint("Step 8: Generating cluster characteristics using shared utilities", "INFO")
        cluster_characteristics = generate_cluster_characteristics(
            market_data,
            clustering_result["cluster_assignments"],
            clustering_result.get("cluster_centers"),
            verbose=True,
        )
        tprint("Cluster characteristics generated", "SUCCESS")
        return cluster_characteristics

    def calculate_clustering_metrics(
        self,
        clustering_result: Dict[str, Any],
        cluster_characteristics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate clustering metrics using shared utilities."""
        try:
            tprint("Calculating clustering metrics using shared utilities...", "INFO")
            tprint("Calculating clustering metrics using shared utilities")

            cluster_assignments = clustering_result["cluster_assignments"]
            n_clusters = clustering_result["n_clusters"]
            tprint(
                f"Processing {n_clusters} clusters with {len(cluster_assignments)} samples",
                "INFO",
            )

            tprint("Calculating regime distribution...", "INFO")
            regime_distribution = self._metrics_calculator.calculate_regime_distribution(cluster_assignments)
            tprint(f"Regime distribution calculated: {len(regime_distribution)} regimes", "SUCCESS")

            clustering_quality = clustering_result.get("clustering_quality", {})
            tprint("Clustering quality metrics retrieved", "SUCCESS")

            tprint("Calculating economic scores...", "INFO")
            economic_scores = calculate_economic_scores(cluster_assignments, verbose=True)
            tprint("Economic scores calculated", "SUCCESS")

            tprint("Calculating trading scores...", "INFO")
            trading_scores = calculate_trading_scores(cluster_assignments, verbose=True)
            tprint("Trading scores calculated", "SUCCESS")

            tprint("Calculating stability scores...", "INFO")
            stability_scores = calculate_stability_scores(cluster_assignments, verbose=True)
            tprint("Stability scores calculated", "SUCCESS")

            tprint("Compiling final metrics...", "INFO")
            metrics = {
                "n_clusters": n_clusters,
                "total_samples": len(cluster_assignments),
                "regime_distribution": regime_distribution,
                "clustering_quality": clustering_quality,
                "economic_scores": economic_scores,
                "trading_scores": trading_scores,
                "stability_scores": stability_scores,
                "regime_balance": (
                    1.0
                    - (
                        np.std(list(regime_distribution.values()))
                        / np.mean(list(regime_distribution.values()))
                    )
                    if regime_distribution
                    else 0.0
                ),
            }
            tprint("Final metrics compiled", "SUCCESS")

            self._component._log("Clustering metrics calculated using shared utilities", "SUCCESS")
            return metrics

        except Exception as exc:  # pragma: no cover - error path
            tprint(f"Clustering metrics calculation failed: {exc}")
            return {
                "n_clusters": clustering_result.get("n_clusters", 0),
                "total_samples": len(clustering_result.get("cluster_assignments", [])),
                "regime_distribution": {},
                "clustering_quality": {},
                "economic_scores": [],
                "trading_scores": [],
                "stability_scores": [],
            }

    def build_artifacts(
        self,
        clustering_result: Dict[str, Any],
        cluster_characteristics: Dict[str, Any],
        clustering_metrics: Dict[str, Any],
        market_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Create consolidated artifacts from clustering outputs."""
        tprint("Step 10: Creating consolidated artifacts", "INFO")
        artifacts = self._component._create_consolidated_artifacts(
            clustering_result,
            cluster_characteristics,
            clustering_metrics,
            market_data,
        )
        tprint("Consolidated artifacts created", "SUCCESS")
        return artifacts
