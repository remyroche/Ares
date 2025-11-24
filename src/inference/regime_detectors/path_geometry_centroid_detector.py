"""
Path Geometry Centroid-Based Regime Detector

This module provides a simpler alternative to GMM-direct inference using
distance-based classification with regime centroids.

Key Features:
- Extremely fast inference (distance calculation only)
- Simple, interpretable decision logic
- No complex model dependencies
- Suitable for embedded/edge deployments

Architecture:
- Compute regime centroids from GMM/SA labeled data
- At inference: find nearest centroid (Mahalanobis or Euclidean distance)
- Optional confidence based on distance ratios

Usage:
    detector = PathGeometryCentroidDetector.load("path_centroids_ETHUSDT_15m.pkl")
    regime, confidence = detector.predict(live_features_dict)
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.spatial.distance import mahalanobis

logger = logging.getLogger(__name__)


@dataclass
class CentroidRegimeDetection:
    """Result of a single centroid-based regime detection."""
    regime_id: int
    confidence: float  # Based on distance ratio to 2nd nearest
    distance_to_centroid: float
    all_distances: Dict[int, float]  # regime_id → distance
    regime_name: Optional[str] = None


class PathGeometryCentroidDetector:
    """
    Distance-based regime detector using precomputed regime centroids.

    This is a simpler alternative to GMM-direct that trades some accuracy
    for extreme simplicity and speed. Useful for:
    - Embedded systems with limited dependencies
    - Ultra-low-latency requirements
    - Interpretable decision boundaries
    """

    def __init__(
        self,
        regime_centroids: pd.DataFrame,
        regime_covariances: Optional[Dict[int, np.ndarray]] = None,
        feature_names: List[str] = None,
        scaler: Optional[Any] = None,
        regime_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
        distance_metric: str = "mahalanobis",
    ):
        """
        Initialize the centroid detector.

        Args:
            regime_centroids: DataFrame with regime_id as index, features as columns
            regime_covariances: Optional dict mapping regime_id → covariance matrix
                               (required for Mahalanobis distance)
            feature_names: Ordered list of feature names
            scaler: Optional fitted scaler
            regime_metadata: Optional regime descriptions
            distance_metric: "mahalanobis" or "euclidean"
        """
        self.regime_centroids = regime_centroids
        self.regime_covariances = regime_covariances or {}
        self.feature_names = feature_names or list(regime_centroids.columns)
        self.scaler = scaler
        self.regime_metadata = regime_metadata or {}
        self.distance_metric = distance_metric

        self.regime_ids = sorted(regime_centroids.index.tolist())

        # Validate Mahalanobis setup
        if distance_metric == "mahalanobis" and not regime_covariances:
            logger.warning(
                "Mahalanobis distance requested but no covariances provided. "
                "Falling back to Euclidean distance."
            )
            self.distance_metric = "euclidean"

        logger.info(
            f"Initialized PathGeometryCentroidDetector: "
            f"{len(self.regime_ids)} regimes, "
            f"{len(self.feature_names)} features, "
            f"distance={self.distance_metric}"
        )

    def predict(
        self,
        features: Dict[str, float],
        return_distances: bool = True,
    ) -> CentroidRegimeDetection:
        """
        Predict regime by finding nearest centroid.

        Args:
            features: Dict mapping feature_name → value
            return_distances: If True, include all distances in result

        Returns:
            CentroidRegimeDetection with regime_id, confidence, distances
        """
        # Prepare feature vector
        feature_vector = self._prepare_features(features)

        # Compute distances to all centroids
        distances = {}
        for regime_id in self.regime_ids:
            centroid = self.regime_centroids.loc[regime_id].values
            distances[regime_id] = self._compute_distance(
                feature_vector, centroid, regime_id
            )

        # Find nearest regime
        nearest_regime = min(distances, key=distances.get)
        nearest_distance = distances[nearest_regime]

        # Compute confidence from distance ratios
        # Confidence is high when nearest is much closer than 2nd nearest
        sorted_distances = sorted(distances.values())
        if len(sorted_distances) > 1:
            d1, d2 = sorted_distances[0], sorted_distances[1]
            if d2 > 0:
                # Confidence ∈ [0, 1]: ratio of distance to 2nd vs nearest
                # If d1 << d2, confidence → 1
                confidence = 1.0 - (d1 / (d1 + d2))
            else:
                confidence = 1.0
        else:
            confidence = 1.0

        # Get regime name
        regime_name = None
        if nearest_regime in self.regime_metadata:
            regime_name = self.regime_metadata[nearest_regime].get("name")

        return CentroidRegimeDetection(
            regime_id=nearest_regime,
            confidence=confidence,
            distance_to_centroid=nearest_distance,
            all_distances=distances if return_distances else None,
            regime_name=regime_name,
        )

    def predict_batch(
        self,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict regimes for a batch of observations.

        Args:
            features_df: DataFrame with feature columns

        Returns:
            DataFrame with columns: regime_id, confidence, distance
        """
        results = []
        for idx, row in features_df.iterrows():
            features_dict = row.to_dict()
            detection = self.predict(features_dict, return_distances=False)
            results.append({
                "regime_id": detection.regime_id,
                "confidence": detection.confidence,
                "distance": detection.distance_to_centroid,
            })

        return pd.DataFrame(results, index=features_df.index)

    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """Extract and scale features in correct order."""
        feature_vector = np.array([
            features.get(fname, 0.0) for fname in self.feature_names
        ])

        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))[0]

        return feature_vector

    def _compute_distance(
        self,
        point: np.ndarray,
        centroid: np.ndarray,
        regime_id: int,
    ) -> float:
        """Compute distance from point to centroid."""
        if self.distance_metric == "mahalanobis" and regime_id in self.regime_covariances:
            try:
                cov = self.regime_covariances[regime_id]
                # Add regularization to avoid singular matrices
                cov_reg = cov + np.eye(len(cov)) * 1e-6
                distance = mahalanobis(point, centroid, np.linalg.inv(cov_reg))
            except Exception as e:
                logger.warning(
                    f"Mahalanobis distance failed for regime {regime_id}: {e}. "
                    f"Falling back to Euclidean."
                )
                distance = np.linalg.norm(point - centroid)
        else:
            # Euclidean distance
            distance = np.linalg.norm(point - centroid)

        return float(distance)

    @classmethod
    def from_labeled_data(
        cls,
        data: pd.DataFrame,
        label_column: str,
        feature_columns: List[str],
        distance_metric: str = "mahalanobis",
        scaler: Optional[Any] = None,
    ) -> "PathGeometryCentroidDetector":
        """
        Create detector by computing centroids from labeled data.

        Args:
            data: DataFrame with features and regime labels
            label_column: Name of column containing regime labels
            feature_columns: List of feature column names
            distance_metric: "mahalanobis" or "euclidean"
            scaler: Optional fitted scaler

        Returns:
            Initialized PathGeometryCentroidDetector
        """
        # Compute centroids
        regime_centroids = data.groupby(label_column)[feature_columns].mean()

        # Compute covariances if Mahalanobis
        regime_covariances = None
        if distance_metric == "mahalanobis":
            regime_covariances = {}
            for regime_id in regime_centroids.index:
                regime_data = data[data[label_column] == regime_id][feature_columns]
                regime_covariances[regime_id] = regime_data.cov().values

        return cls(
            regime_centroids=regime_centroids,
            regime_covariances=regime_covariances,
            feature_names=feature_columns,
            scaler=scaler,
            distance_metric=distance_metric,
        )

    @classmethod
    def load(cls, model_path: str) -> "PathGeometryCentroidDetector":
        """Load a persisted centroid detector from disk."""
        model_data = joblib.load(model_path)

        return cls(
            regime_centroids=model_data["regime_centroids"],
            regime_covariances=model_data.get("regime_covariances"),
            feature_names=model_data["feature_names"],
            scaler=model_data.get("scaler"),
            regime_metadata=model_data.get("regime_metadata"),
            distance_metric=model_data.get("distance_metric", "mahalanobis"),
        )

    def save(self, model_path: str) -> None:
        """Persist the centroid detector to disk."""
        model_data = {
            "regime_centroids": self.regime_centroids,
            "regime_covariances": self.regime_covariances,
            "feature_names": self.feature_names,
            "scaler": self.scaler,
            "regime_metadata": self.regime_metadata,
            "distance_metric": self.distance_metric,
        }

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, model_path)

        logger.info(f"Saved PathGeometryCentroidDetector to {model_path}")

    def __repr__(self) -> str:
        return (
            f"PathGeometryCentroidDetector("
            f"n_regimes={len(self.regime_ids)}, "
            f"n_features={len(self.feature_names)}, "
            f"distance={self.distance_metric})"
        )
