"""
Tactician Validator

This module provides validation utilities for tactician models.
"""

import logging
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Import ML libraries with fallback support
try:
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import roc_auc_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from src.utils.tprint import tprint

logger = logging.getLogger(__name__)

class TacticianValidator:
    """
    Validator for tactician models with comprehensive evaluation metrics.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the validator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.validation_results = {}

    async def validate_models(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        signal_type: str = 'long'
    ) -> Dict[str, Any]:
        """
        Validate trained models for the given signal type.

        Args:
            models: Dictionary of trained models
            X: Feature matrix
            y: Target labels
            signal_type: Type of signal ('long' or 'short')

        Returns:
            Dictionary with validation results
        """
        try:
            tprint(f"🔍 [TACTICIAN_VALIDATOR] Validating models for {signal_type} signals...", color="blue")

            # Handle missing values
            X_clean = X.fillna(X.median())
            y_clean = y.fillna(0)

            validation_results = {}

            for model_key, model in models.items():
                try:
                    # Cross-validation
                    cv_scores = self._cross_validate_model(model, X_clean, y_clean)

                    # Single model evaluation
                    y_pred = model.predict(X_clean)
                    y_pred_proba = getattr(model, 'predict_proba', lambda x: np.zeros((len(x), 2)))(X_clean)

                    # Calculate metrics
                    metrics = self._calculate_metrics(y_clean, y_pred, y_pred_proba)

                    validation_results[model_key] = {
                        'cv_scores': cv_scores,
                        'metrics': metrics,
                        'model_type': model_key.split('_')[0],
                        'signal_type': signal_type
                    }

                    tprint(f"✅ [TACTICIAN_VALIDATOR] Validated {model_key}: CV Score: {cv_scores['mean']:.3f} ± {cv_scores['std']:.3f}", color="green")

                except Exception as e:
                    tprint(f"❌ [TACTICIAN_VALIDATOR] Error validating {model_key}: {e}", color="red")
                    validation_results[model_key] = {
                        'error': str(e),
                        'model_type': model_key.split('_')[0],
                        'signal_type': signal_type
                    }

            # Store results
            self.validation_results[signal_type] = validation_results

            tprint(f"✅ [TACTICIAN_VALIDATOR] Completed validation for {signal_type} signals", color="green")

            return {
                'success': True,
                'validation_results': validation_results,
                'signal_type': signal_type,
                'n_models': len(models)
            }

        except Exception as e:
            tprint(f"❌ [TACTICIAN_VALIDATOR] Error during validation: {e}", color="red")
            return {
                'success': False,
                'error': str(e),
                'signal_type': signal_type
            }

    def _cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Perform cross-validation on a model."""
        if not SKLEARN_AVAILABLE:
            return {'mean': 0.0, 'std': 0.0}

        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

            return {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {'mean': 0.0, 'std': 0.0}

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        if not SKLEARN_AVAILABLE:
            return {}

        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }

            # Add ROC AUC if probabilities are available
            if y_pred_proba.shape[1] > 1:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                except:
                    metrics['roc_auc'] = 0.0
            else:
                metrics['roc_auc'] = 0.0

            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}

    def get_best_model(self, signal_type: str) -> Optional[str]:
        """
        Get the best performing model for a signal type.

        Args:
            signal_type: Type of signal ('long' or 'short')

        Returns:
            Name of the best model or None
        """
        if signal_type not in self.validation_results:
            return None

        best_model = None
        best_score = -1

        for model_key, results in self.validation_results[signal_type].items():
            if 'error' in results:
                continue

            cv_score = results.get('cv_scores', {}).get('mean', 0)
            if cv_score > best_score:
                best_score = cv_score
                best_model = model_key

        return best_model
