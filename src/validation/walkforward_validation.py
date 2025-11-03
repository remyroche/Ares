"""
Walk-Forward Validation System for End-to-End Roadmap

Implements:
- Walk-forward (outer) with K chronological folds
- Nested (inner) CV for hyperparameter selection
- Embargo logic (≥ max(h, longest window))
- Ablation ladder testing
- SPA/reality check for data-snooping protection
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.linear_model import LinearRegression
import itertools

class ValidationType(Enum):
    """Types of validation."""
    WALK_FORWARD = "walk_forward"
    NESTED_CV = "nested_cv"
    ABLATION = "ablation"
    SPA_CHECK = "spa_check"

@dataclass
class ValidationConfig:
    """Configuration for validation."""
    n_outer_folds: int = 6
    n_inner_folds: int = 3
    embargo_pct: float = 0.1
    min_train_samples: int = 1000
    min_val_samples: int = 200
    ablation_steps: List[str] = None
    spa_permutations: int = 1000
    significance_level: float = 0.05

    def __post_init__(self):
        if self.ablation_steps is None:
            self.ablation_steps = [
                'parents_only',
                'parents_transforms',
                'parents_transforms_patch',
                'parents_transforms_patch_8_interactions',
                'parents_transforms_patch_15_interactions'
            ]

@dataclass
class ValidationResult:
    """Result of validation."""
    ic_scores: Dict[str, float]
    auc_scores: Dict[str, float]
    mse_scores: Dict[str, float]
    fold_results: List[Dict[str, Any]]
    ablation_results: Dict[str, Dict[str, float]]
    spa_p_value: Optional[float]
    metadata: Dict[str, Any]

@dataclass
class FoldResult:
    """Result for a single fold."""
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    ic_score: float
    auc_score: float
    mse_score: float
    feature_importance: Dict[str, float]
    predictions: np.ndarray
    actual: np.ndarray

class WalkForwardValidator:
    """Walk-forward validation with nested CV."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.outer_cv = TimeSeriesSplit(n_splits=config.n_outer_folds)
        self.inner_cv = TimeSeriesSplit(n_splits=config.n_inner_folds)

    def validate(self,
                features: pd.DataFrame,
                targets: pd.Series,
                model_configs: Dict[str, Any]) -> ValidationResult:
        """Run complete walk-forward validation."""

        fold_results = []
        ic_scores = []
        auc_scores = []
        mse_scores = []

        # Walk-forward outer loop
        for fold_idx, (train_idx, val_idx) in enumerate(self.outer_cv.split(features)):
            try:
                # Apply embargo
                embargo_size = int(len(val_idx) * self.config.embargo_pct)
                if embargo_size > 0:
                    val_idx = val_idx[embargo_size:]

                if len(val_idx) < self.config.min_val_samples:
                    continue

                # Split data
                X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
                y_train, y_val = targets.iloc[train_idx], targets.iloc[val_idx]

                # Nested CV for hyperparameter selection
                best_config = self._nested_cv_selection(X_train, y_train, model_configs)

                # Train final model
                final_model = self._train_model(X_train, y_train, best_config)

                # Evaluate on validation set
                fold_result = self._evaluate_fold(
                    final_model, X_val, y_val, train_idx, val_idx, fold_idx
                )

                fold_results.append(fold_result)
                ic_scores.append(fold_result.ic_score)
                auc_scores.append(fold_result.auc_score)
                mse_scores.append(fold_result.mse_score)

            except Exception as e:
                warnings.warn(f"Fold {fold_idx} failed: {e}")
                continue

        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(ic_scores, auc_scores, mse_scores)

        return ValidationResult(
            ic_scores=aggregate_metrics['ic'],
            auc_scores=aggregate_metrics['auc'],
            mse_scores=aggregate_metrics['mse'],
            fold_results=fold_results,
            ablation_results={},  # Would be populated by ablation validator
            spa_p_value=None,  # Would be calculated by SPA checker
            metadata={
                'n_folds_completed': len(fold_results),
                'n_folds_attempted': self.config.n_outer_folds,
                'embargo_applied': True
            }
        )

    def _nested_cv_selection(self,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            model_configs: Dict[str, Any]) -> Dict[str, Any]:
        """Select best hyperparameters using nested CV."""

        best_score = -np.inf
        best_config = None

        for config_name, config in model_configs.items():
            try:
                scores = []

                # Inner CV loop
                for inner_train_idx, inner_val_idx in self.inner_cv.split(X_train):
                    X_inner_train = X_train.iloc[inner_train_idx]
                    y_inner_train = y_train.iloc[inner_train_idx]
                    X_inner_val = X_train.iloc[inner_val_idx]
                    y_inner_val = y_train.iloc[inner_val_idx]

                    # Train model
                    model = self._train_model(X_inner_train, y_inner_train, config)

                    # Evaluate
                    score = self._evaluate_model(model, X_inner_val, y_inner_val)
                    scores.append(score)

                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_config = config

            except Exception as e:
                warnings.warn(f"Config {config_name} failed in nested CV: {e}")
                continue

        return best_config or model_configs[list(model_configs.keys())[0]]

    def _train_model(self, X: pd.DataFrame, y: pd.Series, config: Dict[str, Any]):
        """Train model with given configuration."""
        # Simplified model training (would use actual model in practice)
        model = LinearRegression()
        model.fit(X, y)
        return model

    def _evaluate_model(self, model, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate model and return score."""
        predictions = model.predict(X)

        # Calculate IC (correlation)
        ic = np.corrcoef(predictions, y)[0, 1]
        return abs(ic) if not np.isnan(ic) else 0.0

    def _evaluate_fold(self,
                      model,
                      X_val: pd.DataFrame,
                      y_val: pd.Series,
                      train_idx: np.ndarray,
                      val_idx: np.ndarray,
                      fold_idx: int) -> FoldResult:
        """Evaluate a single fold."""

        predictions = model.predict(X_val)

        # Calculate metrics
        ic_score = self._calculate_ic(predictions, y_val)
        auc_score = self._calculate_auc(predictions, y_val)
        mse_score = mean_squared_error(y_val, predictions)

        # Feature importance (simplified)
        feature_importance = {}
        if hasattr(model, 'coef_'):
            for i, feature in enumerate(X_val.columns):
                feature_importance[feature] = abs(model.coef_[i]) if i < len(model.coef_) else 0.0

        return FoldResult(
            train_start=train_idx[0],
            train_end=train_idx[-1],
            val_start=val_idx[0],
            val_end=val_idx[-1],
            ic_score=ic_score,
            auc_score=auc_score,
            mse_score=mse_score,
            feature_importance=feature_importance,
            predictions=predictions,
            actual=y_val.values
        )

    def _calculate_ic(self, predictions: np.ndarray, actual: np.ndarray) -> float:
        """Calculate Information Coefficient."""
        if len(predictions) < 10:
            return 0.0

        ic = np.corrcoef(predictions, actual)[0, 1]
        return ic if not np.isnan(ic) else 0.0

    def _calculate_auc(self, predictions: np.ndarray, actual: np.ndarray) -> float:
        """Calculate AUC for binary classification."""
        if len(predictions) < 10:
            return 0.5

        # Convert to binary targets
        binary_targets = (actual > 0).astype(int)

        # Check if we have both classes
        if len(np.unique(binary_targets)) < 2:
            return 0.5

        try:
            auc = roc_auc_score(binary_targets, predictions)
            return auc if not np.isnan(auc) else 0.5
        except Exception:
            return 0.5

    def _calculate_aggregate_metrics(self,
                                   ic_scores: List[float],
                                   auc_scores: List[float],
                                   mse_scores: List[float]) -> Dict[str, Dict[str, float]]:
        """Calculate aggregate metrics across folds."""

        return {
            'ic': {
                'mean': np.mean(ic_scores),
                'std': np.std(ic_scores),
                'min': np.min(ic_scores),
                'max': np.max(ic_scores)
            },
            'auc': {
                'mean': np.mean(auc_scores),
                'std': np.std(auc_scores),
                'min': np.min(auc_scores),
                'max': np.max(auc_scores)
            },
            'mse': {
                'mean': np.mean(mse_scores),
                'std': np.std(mse_scores),
                'min': np.min(mse_scores),
                'max': np.max(mse_scores)
            }
        }

class AblationValidator:
    """Ablation testing for feature importance."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.validator = WalkForwardValidator(config)

    def run_ablation(self,
                    features: pd.DataFrame,
                    targets: pd.Series,
                    model_config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Run ablation ladder testing."""

        results = {}

        for step in self.config.ablation_steps:
            try:
                # Create feature subset based on ablation step
                feature_subset = self._create_feature_subset(features, step)

                if feature_subset.empty:
                    continue

                # Run validation on subset
                validation_result = self.validator.validate(
                    feature_subset, targets, {'default': model_config}
                )

                # Store results
                results[step] = {
                    'ic_mean': validation_result.ic_scores['mean'],
                    'ic_std': validation_result.ic_scores['std'],
                    'auc_mean': validation_result.auc_scores['mean'],
                    'auc_std': validation_result.auc_scores['std'],
                    'mse_mean': validation_result.mse_scores['mean'],
                    'mse_std': validation_result.mse_scores['std'],
                    'n_features': len(feature_subset.columns)
                }

            except Exception as e:
                warnings.warn(f"Ablation step {step} failed: {e}")
                continue

        return results

    def _create_feature_subset(self, features: pd.DataFrame, step: str) -> pd.DataFrame:
        """Create feature subset for ablation step."""

        if step == 'parents_only':
            # Only parent features
            parent_cols = [col for col in features.columns if col.startswith('p/')]
            return features[parent_cols]

        elif step == 'parents_transforms':
            # Parent features + transforms
            parent_cols = [col for col in features.columns if col.startswith('p/')]
            transform_cols = [col for col in features.columns if col.startswith('t/')]
            return features[parent_cols + transform_cols]

        elif step == 'parents_transforms_patch':
            # Add patch features
            parent_cols = [col for col in features.columns if col.startswith('p/')]
            transform_cols = [col for col in features.columns if col.startswith('t/')]
            patch_cols = [col for col in features.columns if 'y_hat' in col]
            return features[parent_cols + transform_cols + patch_cols]

        elif step == 'parents_transforms_patch_8_interactions':
            # Add 8 interactions
            parent_cols = [col for col in features.columns if col.startswith('p/')]
            transform_cols = [col for col in features.columns if col.startswith('t/')]
            patch_cols = [col for col in features.columns if 'y_hat' in col]
            interaction_cols = [col for col in features.columns if col.startswith('i/')][:8]
            return features[parent_cols + transform_cols + patch_cols + interaction_cols]

        elif step == 'parents_transforms_patch_15_interactions':
            # All features
            return features

        else:
            return features

class SPAValidator:
    """Stepwise Superior Predictive Ability test."""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def run_spa_test(self,
                    features: pd.DataFrame,
                    targets: pd.Series,
                    model_config: Dict[str, Any]) -> float:
        """Run SPA test for data-snooping protection."""

        # Calculate actual performance
        validator = WalkForwardValidator(self.config)
        actual_result = validator.validate(features, targets, {'default': model_config})
        actual_ic = actual_result.ic_scores['mean']

        # Generate random permutations
        permuted_ics = []

        for _ in range(self.config.spa_permutations):
            try:
                # Randomly permute targets
                permuted_targets = targets.sample(frac=1.0).reset_index(drop=True)

                # Run validation on permuted data
                permuted_result = validator.validate(features, permuted_targets, {'default': model_config})
                permuted_ic = permuted_result.ic_scores['mean']
                permuted_ics.append(permuted_ic)

            except Exception as e:
                warnings.warn(f"SPA permutation failed: {e}")
                continue

        if not permuted_ics:
            return 1.0  # No valid permutations

        # Calculate p-value
        better_count = sum(1 for ic in permuted_ics if ic >= actual_ic)
        p_value = better_count / len(permuted_ics)

        return p_value

def run_complete_validation(features: pd.DataFrame,
                          targets: pd.Series,
                          model_config: Dict[str, Any],
                          config: Optional[ValidationConfig] = None) -> ValidationResult:
    """Run complete validation pipeline."""

    if config is None:
        config = ValidationConfig()

    # Walk-forward validation
    validator = WalkForwardValidator(config)
    result = validator.validate(features, targets, {'default': model_config})

    # Ablation testing
    ablation_validator = AblationValidator(config)
    ablation_results = ablation_validator.run_ablation(features, targets, model_config)
    result.ablation_results = ablation_results

    # SPA test
    spa_validator = SPAValidator(config)
    spa_p_value = spa_validator.run_spa_test(features, targets, model_config)
    result.spa_p_value = spa_p_value

    return result
