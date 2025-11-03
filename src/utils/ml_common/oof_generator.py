"""
Out-of-Fold (OOF) Prediction Generator

This module provides a simplified interface for generating out-of-fold predictions,
wrapping the more comprehensive OOF stacking functionality from the ensembles module.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import clone, is_classifier

# Import existing OOF functionality
from src.utils.ml_common.ensembles.oof_stacking_ensemble_manager import (
    OOFStackingEnsembleManager,
    OOFStackingEnsembleConfig
)

logger = logging.getLogger(__name__)


@dataclass
class OOFConfig:
    """Configuration for OOF generation."""
    n_splits: int = 5
    shuffle: bool = False
    random_state: Optional[int] = 42
    verbose: bool = True


class OOFGenerator:
    """
    Simplified OOF (Out-of-Fold) prediction generator.
    
    This class provides a simple interface for generating out-of-fold predictions
    from a single model or multiple models, useful for cross-validation and
    ensemble stacking.
    """
    
    def __init__(self, config: Optional[OOFConfig] = None):
        """
        Initialize OOF Generator.
        
        Args:
            config: Configuration for OOF generation
        """
        self.config = config or OOFConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.oof_predictions_ = None
        self.oof_scores_ = None
    
    def generate_oof_predictions(
        self,
        model,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        fit_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Generate out-of-fold predictions for a single model.
        
        Args:
            model: Sklearn-compatible model
            X: Feature matrix
            y: Target vector
            fit_params: Additional parameters for model.fit()
            
        Returns:
            Tuple of (oof_predictions, metrics_dict)
        """
        self.logger.info(f"Generating OOF predictions with {self.config.n_splits} folds")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Initialize OOF predictions array
        oof_preds = np.zeros(len(y))
        
        # Determine if classification or regression
        is_classification = is_classifier(model)
        
        # Create cross-validation splitter
        if is_classification:
            cv = StratifiedKFold(
                n_splits=self.config.n_splits,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state
            )
        else:
            cv = KFold(
                n_splits=self.config.n_splits,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state
            )
        
        # Generate OOF predictions
        fold_scores = []
        fit_params = fit_params or {}
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y if is_classification else None)):
            if self.config.verbose:
                self.logger.debug(f"Processing fold {fold_idx + 1}/{self.config.n_splits}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Clone and fit model
            model_clone = clone(model)
            model_clone.fit(X_train, y_train, **fit_params)
            
            # Generate predictions
            if is_classification and hasattr(model_clone, 'predict_proba'):
                # For classification, use probabilities if available
                val_preds = model_clone.predict_proba(X_val)
                if val_preds.shape[1] == 2:
                    # Binary classification - use positive class probability
                    oof_preds[val_idx] = val_preds[:, 1]
                else:
                    # Multi-class - store full probability matrix (simplified to argmax)
                    oof_preds[val_idx] = np.argmax(val_preds, axis=1)
            else:
                # For regression or classification without predict_proba
                oof_preds[val_idx] = model_clone.predict(X_val)
            
            # Calculate fold score
            if is_classification:
                from sklearn.metrics import accuracy_score
                fold_score = accuracy_score(y_val, (oof_preds[val_idx] > 0.5).astype(int))
            else:
                from sklearn.metrics import r2_score
                fold_score = r2_score(y_val, oof_preds[val_idx])
            
            fold_scores.append(fold_score)
        
        # Calculate overall metrics
        if is_classification:
            from sklearn.metrics import accuracy_score, f1_score
            oof_accuracy = accuracy_score(y, (oof_preds > 0.5).astype(int))
            oof_f1 = f1_score(y, (oof_preds > 0.5).astype(int), average='weighted', zero_division=0)
            metrics = {
                'oof_accuracy': oof_accuracy,
                'oof_f1': oof_f1,
                'mean_fold_score': np.mean(fold_scores),
                'std_fold_score': np.std(fold_scores)
            }
        else:
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            oof_r2 = r2_score(y, oof_preds)
            oof_mse = mean_squared_error(y, oof_preds)
            oof_mae = mean_absolute_error(y, oof_preds)
            metrics = {
                'oof_r2': oof_r2,
                'oof_mse': oof_mse,
                'oof_mae': oof_mae,
                'mean_fold_score': np.mean(fold_scores),
                'std_fold_score': np.std(fold_scores)
            }
        
        # Store results
        self.oof_predictions_ = oof_preds
        self.oof_scores_ = metrics
        
        if self.config.verbose:
            self.logger.info(f"OOF generation complete. Metrics: {metrics}")
        
        return oof_preds, metrics
    
    def generate_multi_model_oof(
        self,
        models: Dict[str, Any],
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        fit_params: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
        """
        Generate OOF predictions for multiple models.
        
        Args:
            models: Dictionary of model_name -> model
            X: Feature matrix
            y: Target vector
            fit_params: Dictionary of model_name -> fit_params
            
        Returns:
            Tuple of (oof_predictions_dict, metrics_dict)
        """
        self.logger.info(f"Generating OOF predictions for {len(models)} models")
        
        all_oof_preds = {}
        all_metrics = {}
        fit_params = fit_params or {}
        
        for model_name, model in models.items():
            self.logger.info(f"Processing model: {model_name}")
            
            model_fit_params = fit_params.get(model_name, {})
            oof_preds, metrics = self.generate_oof_predictions(
                model, X, y, model_fit_params
            )
            
            all_oof_preds[model_name] = oof_preds
            all_metrics[model_name] = metrics
        
        return all_oof_preds, all_metrics
    
    def get_oof_predictions(self) -> Optional[np.ndarray]:
        """Get the last generated OOF predictions."""
        return self.oof_predictions_
    
    def get_oof_scores(self) -> Optional[Dict[str, float]]:
        """Get the last generated OOF scores."""
        return self.oof_scores_


def generate_oof_predictions(
    model,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = False
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Convenience function to generate OOF predictions.
    
    Args:
        model: Sklearn-compatible model
        X: Feature matrix
        y: Target vector
        n_splits: Number of cross-validation folds
        random_state: Random seed for reproducibility
        verbose: Whether to print progress
        
    Returns:
        Tuple of (oof_predictions, metrics_dict)
    """
    config = OOFConfig(
        n_splits=n_splits,
        random_state=random_state,
        verbose=verbose
    )
    
    generator = OOFGenerator(config)
    return generator.generate_oof_predictions(model, X, y)


__all__ = [
    'OOFGenerator',
    'OOFConfig',
    'generate_oof_predictions'
]

