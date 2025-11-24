"""
Regime Detectors for Live Trading Inference

This module provides production-ready regime detection systems that eliminate
the teacher-student compression bottleneck.

Available Detectors:
-------------------
1. PathGeometryGMMDetector (Recommended)
   - Direct GMM inference for path geometry regimes
   - Probabilistic regime assignments with confidence scores
   - Fast inference (<1ms for 4 components)
   - Zero risk of class collapse

2. PathGeometryCentroidDetector (Simpler Backup)
   - Distance-based classification using regime centroids
   - Extremely fast (distance calculation only)
   - Suitable for embedded systems or edge deployments

Usage:
------
# GMM-Direct (Recommended)
from src.inference.regime_detectors import PathGeometryGMMDetector

detector = PathGeometryGMMDetector.load("path_gmm_ETHUSDT_15m.pkl")
detection = detector.predict(live_features)

# Centroid-Based (Backup)
from src.inference.regime_detectors import PathGeometryCentroidDetector

detector = PathGeometryCentroidDetector.load("path_centroid_ETHUSDT_15m.pkl")
detection = detector.predict(live_features)
"""

from .path_geometry_gmm_detector import (
    PathGeometryGMMDetector,
    RegimeDetection,
)

from .path_geometry_centroid_detector import (
    PathGeometryCentroidDetector,
    CentroidRegimeDetection,
)

__all__ = [
    "PathGeometryGMMDetector",
    "RegimeDetection",
    "PathGeometryCentroidDetector",
    "CentroidRegimeDetection",
]
