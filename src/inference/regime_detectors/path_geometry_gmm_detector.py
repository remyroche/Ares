"""
Path Geometry GMM-Direct Regime Detector

This module provides live-trading inference for Price Path Geometry regimes using
a persisted GMM model trained via Simulated Annealing optimization.

Key Features:
- Direct GMM inference (no student classifier compression)
- Probabilistic regime assignments with confidence scores
- Fast inference suitable for live trading (<1ms)
- Deterministic predictions from stable GMM boundaries

Architecture:
- Teacher: GMM + Simulated Annealing (trained offline)
- Inference: Direct GMM.predict_proba() on live features
- No information loss, no class collapse risk

Usage:
    detector = PathGeometryGMMDetector.load("path_gmm_ETHUSDT_15m.pkl")
    regime, confidence = detector.predict(live_features_dict)
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)


@dataclass
class RegimeDetection:
    """Result of a single regime detection."""
    regime_id: int
    confidence: float  # Max probability across regimes
    regime_probs: np.ndarray  # Full probability distribution
    regime_name: Optional[str] = None
    geometry_signature: Optional[Dict[str, float]] = None


class PathGeometryGMMDetector:
    """
    Direct GMM-based regime detector for Price Path Geometry.

    This detector uses a persisted GMM model (trained with Simulated Annealing)
    to classify live market data into structural path regimes without any
    student classifier compression.
    """

    # Core structural features for path geometry detection
    # These align with the user's conceptual framework:
    # - Roughness: hurst_exponent_path
    # - Linearity: path_trend_r2
    # - Directness: path_efficiency_return_3h
    # - Shape/Bend: quadratic_fit_curvature (if available)
    # - Steepness: linear_reg_slope (if available)
    # - Timing: path_center_of_gravity (if available)
    # - Morphology: body_range_ratio

    GEOMETRY_FEATURES = [
        "hurst_exponent_path",      # Roughness
        "path_trend_r2",             # Linearity
        "path_efficiency_return_3h", # Directness
        "body_range_ratio",          # Morphology
        "path_fractal_dimension",    # Complexity
        "traffic_overlap_3h",        # Overlap characteristic
        "path_efficiency_dropping",  # Efficiency drop pattern
        "path_alpha_state",          # Alpha state indicator
        "path_directional_eff_3h",   # Directional efficiency
    ]

    def __init__(
        self,
        gmm_model: Any,
        feature_names: List[str],
        scaler: Optional[Any] = None,
        regime_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
        model_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the GMM detector.

        Args:
            gmm_model: Trained sklearn.mixture.GaussianMixture model
            feature_names: Ordered list of feature names expected by GMM
            scaler: Optional fitted scaler (e.g., StandardScaler, MinMaxScaler)
            regime_metadata: Optional dict mapping regime_id → {name, description, centroids, etc.}
            model_metadata: Optional metadata about training (timestamp, symbol, timeframe, metrics)
        """
        self.gmm_model = gmm_model
        self.feature_names = feature_names
        self.scaler = scaler
        self.regime_metadata = regime_metadata or {}
        self.model_metadata = model_metadata or {}

        self.n_regimes = gmm_model.n_components

        logger.info(
            f"Initialized PathGeometryGMMDetector: {self.n_regimes} regimes, "
            f"{len(feature_names)} features"
        )

    def predict(
        self,
        features: Dict[str, float],
        return_probs: bool = True,
        min_confidence: float = 0.0,
    ) -> RegimeDetection:
        """
        Predict regime for a single observation.

        Args:
            features: Dict mapping feature_name → value
            return_probs: If True, include full probability distribution
            min_confidence: Minimum confidence threshold (if below, regime=-1)

        Returns:
            RegimeDetection object with regime_id, confidence, and optional probs
        """
        # Extract features in correct order
        feature_vector = self._prepare_features(features)

        # Get probabilities from GMM
        probs = self.gmm_model.predict_proba(feature_vector)[0]  # Shape: (n_regimes,)

        # Determine regime and confidence
        regime_id = int(np.argmax(probs))
        confidence = float(probs[regime_id])

        # Apply confidence threshold
        if confidence < min_confidence:
            regime_id = -1  # Unknown/uncertain regime

        # Get regime name if available
        regime_name = None
        if regime_id in self.regime_metadata:
            regime_name = self.regime_metadata[regime_id].get("name")

        # Compute geometry signature (feature contributions)
        geometry_signature = self._compute_geometry_signature(features)

        return RegimeDetection(
            regime_id=regime_id,
            confidence=confidence,
            regime_probs=probs if return_probs else None,
            regime_name=regime_name,
            geometry_signature=geometry_signature,
        )

    def predict_batch(
        self,
        features_df: pd.DataFrame,
        min_confidence: float = 0.0,
    ) -> pd.DataFrame:
        """
        Predict regimes for a batch of observations.

        Args:
            features_df: DataFrame with feature columns
            min_confidence: Minimum confidence threshold

        Returns:
            DataFrame with columns: regime_id, confidence, regime_probs
        """
        # Prepare feature matrix
        X = self._prepare_features_batch(features_df)

        # Get probabilities
        probs = self.gmm_model.predict_proba(X)  # Shape: (n_samples, n_regimes)

        # Determine regimes and confidences
        regime_ids = np.argmax(probs, axis=1)
        confidences = probs[np.arange(len(probs)), regime_ids]

        # Apply confidence threshold
        regime_ids[confidences < min_confidence] = -1

        # Build result DataFrame
        result = pd.DataFrame({
            "regime_id": regime_ids,
            "confidence": confidences,
        }, index=features_df.index)

        # Add probability columns
        for i in range(self.n_regimes):
            result[f"prob_regime_{i}"] = probs[:, i]

        return result

    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """Extract and scale features in correct order."""
        # Extract features in order
        feature_vector = np.array([
            features.get(fname, 0.0) for fname in self.feature_names
        ]).reshape(1, -1)

        # Apply scaling if scaler exists
        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector)

        return feature_vector

    def _prepare_features_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """Extract and scale features for batch prediction."""
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(features_df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Extract features in order
        X = features_df[self.feature_names].values

        # Apply scaling if scaler exists
        if self.scaler is not None:
            X = self.scaler.transform(X)

        return X

    def _compute_geometry_signature(
        self, features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute a human-readable geometry signature.

        Maps raw feature values to interpretable geometry characteristics:
        - roughness (from hurst_exponent_path)
        - linearity (from path_trend_r2)
        - directness (from path_efficiency_return_3h)
        - morphology (from body_range_ratio)
        - etc.
        """
        signature = {}

        # Roughness (Hurst exponent: <0.5 = mean-reverting, >0.5 = trending)
        if "hurst_exponent_path" in features:
            signature["roughness"] = features["hurst_exponent_path"]

        # Linearity (R² of trend fit: 0-1)
        if "path_trend_r2" in features:
            signature["linearity"] = features["path_trend_r2"]

        # Directness (efficiency: 0-1)
        if "path_efficiency_return_3h" in features:
            signature["directness"] = features["path_efficiency_return_3h"]

        # Morphology (body-to-range ratio: 0-1)
        if "body_range_ratio" in features:
            signature["morphology"] = features["body_range_ratio"]

        # Fractal complexity (typically 1-2)
        if "path_fractal_dimension" in features:
            signature["fractal_complexity"] = features["path_fractal_dimension"]

        return signature

    def get_regime_description(self, regime_id: int) -> str:
        """Get human-readable description of a regime."""
        if regime_id in self.regime_metadata:
            meta = self.regime_metadata[regime_id]
            name = meta.get("name", f"Regime {regime_id}")
            desc = meta.get("description", "No description available")
            return f"{name}: {desc}"
        return f"Regime {regime_id} (no metadata available)"

    @classmethod
    def load(cls, model_path: str) -> "PathGeometryGMMDetector":
        """
        Load a persisted GMM detector from disk.

        Args:
            model_path: Path to pickled model file (created by save())

        Returns:
            Loaded PathGeometryGMMDetector instance
        """
        model_data = joblib.load(model_path)

        return cls(
            gmm_model=model_data["gmm_model"],
            feature_names=model_data["feature_names"],
            scaler=model_data.get("scaler"),
            regime_metadata=model_data.get("regime_metadata"),
            model_metadata=model_data.get("model_metadata"),
        )

    def save(self, model_path: str) -> None:
        """
        Persist the GMM detector to disk.

        Args:
            model_path: Output path for pickled model
        """
        model_data = {
            "gmm_model": self.gmm_model,
            "feature_names": self.feature_names,
            "scaler": self.scaler,
            "regime_metadata": self.regime_metadata,
            "model_metadata": self.model_metadata,
        }

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, model_path)

        logger.info(f"Saved PathGeometryGMMDetector to {model_path}")

    def __repr__(self) -> str:
        meta_str = ""
        if self.model_metadata:
            meta_str = f", trained on {self.model_metadata.get('symbol', 'unknown')} " \
                       f"{self.model_metadata.get('timeframe', 'unknown')}"

        return (
            f"PathGeometryGMMDetector("
            f"n_regimes={self.n_regimes}, "
            f"n_features={len(self.feature_names)}"
            f"{meta_str})"
        )
