"""
Enhanced OOF Stacking Ensemble Manager with Confidence Intervals

This module extends the OOF stacking ensemble manager to include:
- Prediction confidence intervals for OOF predictions
- Ensemble diversity metrics for base model complementarity
- Enhanced uncertainty quantification
- Bootstrap-based confidence estimation
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet
import warnings
from scipy import stats
from scipy.stats import bootstrap

# Import base OOF stacking manager
from .oof_stacking_ensemble_manager import (
    OOFStackingEnsembleManager,
    OOFStackingEnsembleConfig,
    OOFStackingEnsembleResult
)

# Import purged K-fold for temporal validation
from src.utils.purged_kfold import PurgedKFoldTime

logger = logging.getLogger(__name__)

@dataclass
class ConfidenceIntervalConfig:
    """Configuration for confidence interval estimation."""

    # Bootstrap configuration
    n_bootstrap_samples: int = 100
    confidence_level: float = 0.95
    bootstrap_method: str = "percentile"  # percentile, bias_corrected, studentized

    # Prediction uncertainty
    enable_prediction_uncertainty: bool = True
    uncertainty_method: str = "ensemble_variance"  # ensemble_variance, model_uncertainty, both

    # Diversity metrics
    enable_diversity_metrics: bool = True
    diversity_threshold: float = 0.3  # Minimum diversity required

    # Performance
    parallel_bootstrap: bool = True
    max_workers: Optional[int] = None

@dataclass
class EnhancedOOFResult:
    """Enhanced OOF result with confidence intervals and diversity metrics."""

    # Basic OOF predictions
    oof_predictions: Dict[str, np.ndarray]
    oof_scores: Dict[str, float]

    # Confidence intervals
    confidence_intervals: Dict[str, Tuple[np.ndarray, np.ndarray]]  # (lower, upper)
    prediction_uncertainty: Dict[str, np.ndarray]

    # Diversity metrics
    ensemble_diversity: Dict[str, float]
    model_correlations: Optional[np.ndarray] = None
    diversity_score: float = 0.0

    # Bootstrap results
    bootstrap_predictions: Optional[Dict[str, np.ndarray]] = None
    bootstrap_scores: Optional[Dict[str, List[float]]] = None

class EnhancedOOFStackingEnsembleManager(OOFStackingEnsembleManager):
    """Enhanced OOF stacking ensemble manager with confidence intervals and diversity metrics."""

    def __init__(self, config: OOFStackingEnsembleConfig, confidence_config: Optional[ConfidenceIntervalConfig] = None):
        """Initialize enhanced OOF stacking ensemble manager."""
        super().__init__(config)
        self.confidence_config = confidence_config or ConfidenceIntervalConfig()
        self.logger = logger.getChild('EnhancedOOFStackingEnsembleManager')

        self.logger.info("🚀 Enhanced OOF StackingEnsembleManager initialized with confidence intervals")

    def _generate_oof_predictions_with_confidence(self,
                                                 models: Dict[str, Any],
                                                 X: np.ndarray,
                                                 y: np.ndarray,
                                                 cv: Any,
                                                 is_classification: bool) -> EnhancedOOFResult:
        """Generate OOF predictions with confidence intervals and diversity metrics."""
        self.logger.info("🔄 Generating enhanced OOF predictions with confidence intervals...")

        # Generate base OOF predictions
        oof_predictions = {}
        oof_scores = {}
        model_predictions = {}  # Store individual model predictions for diversity analysis

        for output_name, model_dict in models.items():
            self.logger.debug(f"🔄 Processing output: {output_name}")

            # Get target for this output
            if isinstance(y, pd.DataFrame):
                y_output = y[output_name].values if output_name in y.columns else y.iloc[:, 0].values
            else:
                output_idx = self.config.output_names.index(output_name)
                y_output = y[:, output_idx]

            # Initialize arrays
            n_samples = len(X)
            oof_preds = np.zeros(n_samples)
            model_preds_dict = {}

            # Generate OOF predictions for each base model
            for model_name, model in model_dict.items():
                self.logger.debug(f"🔄 Generating OOF predictions for {model_name}")

                model_oof_preds = np.zeros(n_samples)

                for train_idx, val_idx in cv.split(X, y_output):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y_output[train_idx], y_output[val_idx]

                    # Clone model to avoid state issues
                    model_clone = clone(model)

                    # Setup early stopping if enabled
                    if self.config.enable_early_stopping:
                        model_clone = self._setup_early_stopping(
                            model_clone, X_train, X_val, y_train, y_val, model_name
                        )

                    # Train model
                    model_clone.fit(X_train, y_train)

                    # Make predictions
                    if hasattr(model_clone, 'predict_proba') and is_classification:
                        val_preds = model_clone.predict_proba(X_val)
                        if val_preds.ndim > 1 and val_preds.shape[1] > 1:
                            val_preds = val_preds[:, 1]  # Use positive class probability
                    else:
                        val_preds = model_clone.predict(X_val)

                    model_oof_preds[val_idx] = val_preds

                # Store model predictions
                model_preds_dict[model_name] = model_oof_preds

                # Calculate OOF score
                if hasattr(model, 'predict_proba') and is_classification:
                    pred_probs = model_oof_preds
                    if pred_probs.ndim == 1:
                        pred_probs = np.column_stack([1 - pred_probs, pred_probs])
                    score = self._calculate_score(y_output, pred_probs, is_classification, self.config.oof_validation_metric)
                else:
                    score = self._calculate_score(y_output, model_oof_preds, is_classification, self.config.oof_validation_metric)

                if output_name not in oof_scores:
                    oof_scores[output_name] = {}
                oof_scores[output_name][model_name] = score

                self.logger.debug(f"✅ OOF predictions generated for {model_name}, score: {score:.4f}")

            # Calculate ensemble prediction (average of base models)
            if model_preds_dict:
                ensemble_preds = np.mean(list(model_preds_dict.values()), axis=0)
                oof_predictions[output_name] = ensemble_preds
                model_predictions[output_name] = model_preds_dict

            self.logger.info(f"✅ Enhanced OOF predictions generated for {output_name}")

        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(model_predictions, y)

        # Calculate prediction uncertainty
        prediction_uncertainty = self._calculate_prediction_uncertainty(model_predictions)

        # Calculate diversity metrics
        ensemble_diversity = self._calculate_ensemble_diversity(model_predictions)
        model_correlations = self._calculate_model_correlations(model_predictions)
        diversity_score = self._calculate_overall_diversity_score(ensemble_diversity, model_correlations)

        # Generate bootstrap predictions if enabled
        bootstrap_predictions = None
        bootstrap_scores = None
        if self.confidence_config.n_bootstrap_samples > 0:
            bootstrap_predictions, bootstrap_scores = self._generate_bootstrap_predictions(
                models, X, y, cv, is_classification
            )

        return EnhancedOOFResult(
            oof_predictions=oof_predictions,
            oof_scores=oof_scores,
            confidence_intervals=confidence_intervals,
            prediction_uncertainty=prediction_uncertainty,
            ensemble_diversity=ensemble_diversity,
            model_correlations=model_correlations,
            diversity_score=diversity_score,
            bootstrap_predictions=bootstrap_predictions,
            bootstrap_scores=bootstrap_scores
        )

    def _calculate_confidence_intervals(self,
                                      model_predictions: Dict[str, Dict[str, np.ndarray]],
                                      y: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Calculate confidence intervals for OOF predictions."""
        self.logger.info("📊 Calculating confidence intervals...")

        confidence_intervals = {}

        for output_name, model_preds in model_predictions.items():
            if not model_preds:
                continue

            # Get all model predictions for this output
            pred_arrays = list(model_preds.values())
            n_samples = len(pred_arrays[0])

            # Calculate confidence intervals using bootstrap
            if self.confidence_config.bootstrap_method == "percentile":
                # Use percentile method
                alpha = 1 - self.confidence_config.confidence_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100

                # Calculate ensemble predictions for each bootstrap sample
                bootstrap_ensemble_preds = []
                for _ in range(self.confidence_config.n_bootstrap_samples):
                    # Bootstrap sample of models
                    bootstrap_models = np.random.choice(len(pred_arrays), size=len(pred_arrays), replace=True)
                    bootstrap_preds = [pred_arrays[i] for i in bootstrap_models]
                    ensemble_pred = np.mean(bootstrap_preds, axis=0)
                    bootstrap_ensemble_preds.append(ensemble_pred)

                bootstrap_ensemble_preds = np.array(bootstrap_ensemble_preds)

                # Calculate percentiles
                lower_bound = np.percentile(bootstrap_ensemble_preds, lower_percentile, axis=0)
                upper_bound = np.percentile(bootstrap_ensemble_preds, upper_percentile, axis=0)

            elif self.confidence_config.bootstrap_method == "bias_corrected":
                # Use bias-corrected bootstrap
                bootstrap_ensemble_preds = []
                for _ in range(self.confidence_config.n_bootstrap_samples):
                    bootstrap_models = np.random.choice(len(pred_arrays), size=len(pred_arrays), replace=True)
                    bootstrap_preds = [pred_arrays[i] for i in bootstrap_models]
                    ensemble_pred = np.mean(bootstrap_preds, axis=0)
                    bootstrap_ensemble_preds.append(ensemble_pred)

                bootstrap_ensemble_preds = np.array(bootstrap_ensemble_preds)

                # Calculate bias-corrected confidence intervals
                mean_pred = np.mean(bootstrap_ensemble_preds, axis=0)
                std_pred = np.std(bootstrap_ensemble_preds, axis=0)

                # Bias correction
                z_alpha = stats.norm.ppf(1 - self.confidence_config.confidence_level / 2)
                lower_bound = mean_pred - z_alpha * std_pred
                upper_bound = mean_pred + z_alpha * std_pred

            else:  # studentized
                # Use studentized bootstrap
                bootstrap_ensemble_preds = []
                for _ in range(self.confidence_config.n_bootstrap_samples):
                    bootstrap_models = np.random.choice(len(pred_arrays), size=len(pred_arrays), replace=True)
                    bootstrap_preds = [pred_arrays[i] for i in bootstrap_models]
                    ensemble_pred = np.mean(bootstrap_preds, axis=0)
                    bootstrap_ensemble_preds.append(ensemble_pred)

                bootstrap_ensemble_preds = np.array(bootstrap_ensemble_preds)

                # Studentized confidence intervals
                mean_pred = np.mean(bootstrap_ensemble_preds, axis=0)
                std_pred = np.std(bootstrap_ensemble_preds, axis=0)

                # Use t-distribution for small samples
                df = self.confidence_config.n_bootstrap_samples - 1
                t_alpha = stats.t.ppf(1 - self.confidence_config.confidence_level / 2, df)
                lower_bound = mean_pred - t_alpha * std_pred
                upper_bound = mean_pred + t_alpha * std_pred

            confidence_intervals[output_name] = (lower_bound, upper_bound)
            self.logger.debug(f"✅ Confidence intervals calculated for {output_name}")

        return confidence_intervals

    def _calculate_prediction_uncertainty(self,
                                        model_predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Calculate prediction uncertainty for each output."""
        self.logger.info("📊 Calculating prediction uncertainty...")

        prediction_uncertainty = {}

        for output_name, model_preds in model_predictions.items():
            if not model_preds:
                continue

            pred_arrays = list(model_preds.values())
            n_samples = len(pred_arrays[0])

            if self.confidence_config.uncertainty_method == "ensemble_variance":
                # Calculate variance across ensemble predictions
                pred_matrix = np.array(pred_arrays)
                uncertainty = np.var(pred_matrix, axis=0)

            elif self.confidence_config.uncertainty_method == "model_uncertainty":
                # Calculate uncertainty based on model disagreement
                pred_matrix = np.array(pred_arrays)
                mean_pred = np.mean(pred_matrix, axis=0)
                uncertainty = np.mean((pred_matrix - mean_pred) ** 2, axis=0)

            else:  # both
                # Combine ensemble variance and model uncertainty
                pred_matrix = np.array(pred_arrays)
                ensemble_var = np.var(pred_matrix, axis=0)
                mean_pred = np.mean(pred_matrix, axis=0)
                model_uncertainty = np.mean((pred_matrix - mean_pred) ** 2, axis=0)
                uncertainty = (ensemble_var + model_uncertainty) / 2

            prediction_uncertainty[output_name] = uncertainty
            self.logger.debug(f"✅ Prediction uncertainty calculated for {output_name}")

        return prediction_uncertainty

    def _calculate_ensemble_diversity(self,
                                    model_predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, float]:
        """Calculate ensemble diversity metrics."""
        self.logger.info("📊 Calculating ensemble diversity metrics...")

        ensemble_diversity = {}

        for output_name, model_preds in model_predictions.items():
            if len(model_preds) < 2:
                ensemble_diversity[output_name] = 0.0
                continue

            pred_arrays = list(model_preds.values())
            n_models = len(pred_arrays)
            n_samples = len(pred_arrays[0])

            # Calculate pairwise diversity
            diversity_scores = []
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    # Calculate correlation-based diversity
                    corr = np.corrcoef(pred_arrays[i], pred_arrays[j])[0, 1]
                    diversity = 1 - abs(corr)  # Higher diversity = lower correlation
                    diversity_scores.append(diversity)

            # Calculate average diversity
            avg_diversity = np.mean(diversity_scores) if diversity_scores else 0.0
            ensemble_diversity[output_name] = avg_diversity

            self.logger.debug(f"✅ Ensemble diversity calculated for {output_name}: {avg_diversity:.4f}")

        return ensemble_diversity

    def _calculate_model_correlations(self,
                                    model_predictions: Dict[str, Dict[str, np.ndarray]]) -> Optional[np.ndarray]:
        """Calculate correlation matrix between models."""
        if not model_predictions:
            return None

        # Get all model predictions
        all_predictions = []
        model_names = []

        for output_name, model_preds in model_predictions.items():
            for model_name, preds in model_preds.items():
                all_predictions.append(preds)
                model_names.append(f"{output_name}_{model_name}")

        if len(all_predictions) < 2:
            return None

        # Calculate correlation matrix
        pred_matrix = np.array(all_predictions)
        correlation_matrix = np.corrcoef(pred_matrix)

        self.logger.debug(f"✅ Model correlation matrix calculated: {correlation_matrix.shape}")
        return correlation_matrix

    def _calculate_overall_diversity_score(self,
                                         ensemble_diversity: Dict[str, float],
                                         model_correlations: Optional[np.ndarray]) -> float:
        """Calculate overall diversity score."""
        if not ensemble_diversity:
            return 0.0

        # Average diversity across outputs
        avg_diversity = np.mean(list(ensemble_diversity.values()))

        # Penalty for high correlations
        correlation_penalty = 0.0
        if model_correlations is not None:
            # Calculate average absolute correlation (excluding diagonal)
            mask = ~np.eye(model_correlations.shape[0], dtype=bool)
            avg_correlation = np.mean(np.abs(model_correlations[mask]))
            correlation_penalty = avg_correlation * 0.5  # Penalty for high correlations

        # Overall diversity score
        diversity_score = avg_diversity - correlation_penalty
        diversity_score = max(0.0, min(1.0, diversity_score))  # Clamp to [0, 1]

        self.logger.info(f"📊 Overall diversity score: {diversity_score:.4f}")
        return diversity_score

    def _generate_bootstrap_predictions(self,
                                      models: Dict[str, Any],
                                      X: np.ndarray,
                                      y: np.ndarray,
                                      cv: Any,
                                      is_classification: bool) -> Tuple[Dict[str, np.ndarray], Dict[str, List[float]]]:
        """Generate bootstrap predictions for additional uncertainty quantification."""
        self.logger.info("🔄 Generating bootstrap predictions...")

        bootstrap_predictions = {}
        bootstrap_scores = {}

        for output_name, model_dict in models.items():
            if not model_dict:
                continue

            # Get target for this output
            if isinstance(y, pd.DataFrame):
                y_output = y[output_name].values if output_name in y.columns else y.iloc[:, 0].values
            else:
                output_idx = self.config.output_names.index(output_name)
                y_output = y[:, output_idx]

            n_samples = len(X)
            bootstrap_preds_list = []
            bootstrap_scores_list = []

            for bootstrap_idx in range(self.confidence_config.n_bootstrap_samples):
                # Bootstrap sample of data
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_bootstrap = X[bootstrap_indices]
                y_bootstrap = y_output[bootstrap_indices]

                # Train ensemble on bootstrap sample
                ensemble_preds = []
                for model_name, model in model_dict.items():
                    model_clone = clone(model)
                    model_clone.fit(X_bootstrap, y_bootstrap)

                    if hasattr(model_clone, 'predict_proba') and is_classification:
                        preds = model_clone.predict_proba(X)
                        if preds.ndim > 1 and preds.shape[1] > 1:
                            preds = preds[:, 1]
                    else:
                        preds = model_clone.predict(X)

                    ensemble_preds.append(preds)

                # Average ensemble predictions
                ensemble_pred = np.mean(ensemble_preds, axis=0)
                bootstrap_preds_list.append(ensemble_pred)

                # Calculate score
                if is_classification:
                    score = accuracy_score(y_output, np.round(ensemble_pred))
                else:
                    score = 1 - np.mean((y_output - ensemble_pred) ** 2) / np.var(y_output)

                bootstrap_scores_list.append(score)

            # Store bootstrap results
            bootstrap_predictions[output_name] = np.array(bootstrap_preds_list)
            bootstrap_scores[output_name] = bootstrap_scores_list

            self.logger.debug(f"✅ Bootstrap predictions generated for {output_name}")

        return bootstrap_predictions, bootstrap_scores

    def get_enhanced_oof_results(self) -> EnhancedOOFResult:
        """Get enhanced OOF results with confidence intervals and diversity metrics."""
        if not hasattr(self, 'enhanced_oof_results'):
            raise ValueError("Enhanced OOF results not available. Call fit() first.")
        return self.enhanced_oof_results

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnhancedOOFStackingEnsembleManager':
        """Fit the enhanced OOF stacking ensemble with confidence intervals."""
        self.logger.info(f"🚀 Fitting Enhanced OOF StackingEnsemble with confidence intervals...")
        start_time = time.time()

        try:
            # Setup cross-validation strategy
            is_classification = len(np.unique(y.ravel())) <= 10
            cv = self._setup_cross_validation(X, y, is_classification)

            # Generate enhanced OOF predictions
            if self.base_models:
                self.enhanced_oof_results = self._generate_oof_predictions_with_confidence(
                    self.base_models, X, y, cv, is_classification
                )
            else:
                self.logger.warning("No base models provided, creating defaults")
                self._create_default_base_models()
                self.enhanced_oof_results = self._generate_oof_predictions_with_confidence(
                    self.base_models, X, y, cv, is_classification
                )

            # Train meta-models on OOF predictions
            if self.meta_models:
                meta_models = self._train_meta_models(X, y, self.enhanced_oof_results.oof_predictions, cv)
            else:
                self.logger.warning("No meta-models provided, creating defaults")
                meta_models = self._train_meta_models(X, y, self.enhanced_oof_results.oof_predictions, cv)

            # Update state
            self.is_fitted = True

            # Log results
            training_time = time.time() - start_time
            self.logger.info(f"✅ Enhanced OOF StackingEnsemble fitted in {training_time:.3f}s")
            self.logger.info(f"📊 Diversity score: {self.enhanced_oof_results.diversity_score:.4f}")
            self.logger.info(f"📊 Confidence level: {self.confidence_config.confidence_level}")

            return self

        except Exception as e:
            self.logger.error(f"Failed to fit enhanced OOF stacking ensemble: {e}")
            raise

# Convenience function
def create_enhanced_oof_stacking_ensemble(
    ensemble_name: str,
    output_dir: str,
    confidence_config: Optional[ConfidenceIntervalConfig] = None,
    **kwargs
) -> EnhancedOOFStackingEnsembleManager:
    """Create enhanced OOF stacking ensemble with confidence intervals."""

    config = OOFStackingEnsembleConfig(
        ensemble_name=ensemble_name,
        output_dir=output_dir,
        **kwargs
    )

    return EnhancedOOFStackingEnsembleManager(config, confidence_config)
