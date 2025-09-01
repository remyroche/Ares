"""
Probability Calculation Framework

This module provides base probability calculation functions for different model types
and market scenarios. It serves as the foundation for generating the 4 required
probability outputs for the Enhanced Prediction Service.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score
import logging

logger = logging.getLogger(__name__)


class BaseProbabilityCalculator:
    """Base class for probability calculations across different model types."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def validate_probability(self, prob: float, name: str) -> float:
        """Validate that probability is between 0.0 and 1.0."""
        if not 0.0 <= prob <= 1.0:
            self.logger.warning(f"{name} probability {prob} out of range [0,1], clamping")
            return np.clip(prob, 0.0, 1.0)
        return prob

    def calculate_confidence_from_proba(self, y_pred_proba: np.ndarray) -> float:
        """Calculate confidence from prediction probabilities."""
        if y_pred_proba.ndim == 1:
            # Binary classification
            return np.mean(np.maximum(y_pred_proba, 1 - y_pred_proba))
        else:
            # Multi-class classification
            return np.mean(np.max(y_pred_proba, axis=1))


class ClassificationProbabilityCalculator(BaseProbabilityCalculator):
    """Probability calculator for classification models."""


class RegressionProbabilityCalculator(BaseProbabilityCalculator):
    """Probability calculator for regression models."""

