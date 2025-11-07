"""
Regime-specific Walk-Forward Validation

Extends the general walk-forward validation system with regime-specific features:
- Regime-aware splits (ensures minimum samples per regime in each fold)
- Temporal metrics (MEL, TR, SFPR, etc.)
- Confidence intervals across folds
- Model selection based on OOS performance
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.utils.logger import system_logger
from src.utils.tprint import tprint

logger = system_logger.getChild('RegimeWalkForwardValidator')


@dataclass
class RegimeValidationConfig:
    """Configuration for regime walk-forward validation."""
    n_outer_folds: int = 5
    n_inner_folds: int = 3
    embargo_pct: float = 0.05
    min_train_samples: int = 100
    min_val_samples: int = 30
    min_regime_samples: int = 10
    test_size: float = 0.3
    gap_size: int = 1


@dataclass
class RegimeValidationResult:
    """Result of regime validation."""
    # Per-fold metrics
    fold_metrics: List[Dict[str, float]]

    # Aggregate metrics with confidence intervals
    accuracy: Dict[str, float]  # mean, std, min, max, ci_lower, ci_upper
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]

    # Temporal metrics
    temporal_metrics: Dict[str, Dict[str, float]]  # MEL, TR, SFPR, etc.

    # Model rankings
    model_rankings: List[Dict[str, Any]]

    # Metadata
    metadata: Dict[str, Any]


class RegimeWalkForwardValidator:
    """Walk-forward validation with regime-specific features."""

    def __init__(self, config: Optional[RegimeValidationConfig] = None):
        """Initialize validator."""
        self.config = config or RegimeValidationConfig()
        self.outer_cv = TimeSeriesSplit(
            n_splits=self.config.n_outer_folds,
            gap=self.config.gap_size
        )
        self.inner_cv = TimeSeriesSplit(
            n_splits=self.config.n_inner_folds,
            gap=self.config.gap_size
        )

        logger.info(f"Initialized RegimeWalkForwardValidator with {self.config.n_outer_folds} outer folds")

    def validate_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: Dict[str, Any],
        model_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> RegimeValidationResult:
        """
        Validate multiple models using walk-forward validation.

        Args:
            X: Feature matrix
            y: Target labels (regime labels)
            models: Dictionary of trained models {model_name: model}
            model_configs: Optional model configurations for retraining

        Returns:
            RegimeValidationResult with comprehensive metrics
        """
        tprint(f"🔍 [REGIME_WF_VAL] Starting walk-forward validation for {len(models)} models", color="cyan")

        # Track metrics across all folds for all models
        model_fold_results = {model_name: [] for model_name in models.keys()}

        # Walk-forward outer loop
        fold_idx = 0
        for train_idx, val_idx in self.outer_cv.split(X):
            try:
                fold_idx += 1
                tprint(f"📊 [REGIME_WF_VAL] Processing fold {fold_idx}/{self.config.n_outer_folds}", color="blue")

                # Apply embargo
                embargo_size = int(len(val_idx) * self.config.embargo_pct)
                if embargo_size > 0:
                    val_idx = val_idx[embargo_size:]

                # Check minimum samples
                if len(val_idx) < self.config.min_val_samples:
                    tprint(f"⚠️ [REGIME_WF_VAL] Fold {fold_idx} has insufficient validation samples, skipping", color="yellow")
                    continue

                # Split data
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Check regime distribution
                if not self._check_regime_distribution(y_train, y_val):
                    tprint(f"⚠️ [REGIME_WF_VAL] Fold {fold_idx} has insufficient regime samples, skipping", color="yellow")
                    continue

                # Evaluate each model on this fold
                for model_name, model in models.items():
                    try:
                        # Retrain model if configs provided (for true OOS)
                        if model_configs and model_name in model_configs:
                            # TODO: Implement retraining with HPO
                            # For now, use the provided model
                            pass

                        # Get predictions
                        y_pred = model.predict(X_val)
                        y_pred_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None

                        # Calculate metrics
                        fold_metrics = self._calculate_fold_metrics(
                            y_val, y_pred, y_pred_proba, fold_idx
                        )
                        fold_metrics['fold'] = fold_idx
                        fold_metrics['model_name'] = model_name

                        model_fold_results[model_name].append(fold_metrics)

                    except Exception as e:
                        logger.warning(f"Model {model_name} failed on fold {fold_idx}: {e}")
                        continue

            except Exception as e:
                logger.warning(f"Fold {fold_idx} failed: {e}")
                continue

        # Calculate aggregate metrics across folds for each model
        tprint("📊 [REGIME_WF_VAL] Calculating aggregate metrics across folds", color="cyan")
        model_results = {}

        for model_name, fold_results in model_fold_results.items():
            if not fold_results:
                tprint(f"⚠️ [REGIME_WF_VAL] No valid folds for model {model_name}", color="yellow")
                continue

            model_results[model_name] = self._aggregate_fold_metrics(fold_results)

        # Rank models by OOS performance
        model_rankings = self._rank_models(model_results)

        # Create result object (using first model's results as template for now)
        # TODO: Make this more flexible to support multiple models
        first_model_name = list(model_results.keys())[0] if model_results else None
        first_model_results = model_results.get(first_model_name, {}) if first_model_name else {}

        result = RegimeValidationResult(
            fold_metrics=model_fold_results.get(first_model_name, []) if first_model_name else [],
            accuracy=first_model_results.get('accuracy', {}),
            precision=first_model_results.get('precision', {}),
            recall=first_model_results.get('recall', {}),
            f1_score=first_model_results.get('f1_score', {}),
            temporal_metrics=first_model_results.get('temporal_metrics', {}),
            model_rankings=model_rankings,
            metadata={
                'n_folds_attempted': self.config.n_outer_folds,
                'n_folds_completed': len(first_model_results.get('fold_scores', {}).get('accuracy', [])) if first_model_results else 0,
                'n_models_evaluated': len(models),
                'embargo_applied': True
            }
        )

        tprint(f"✅ [REGIME_WF_VAL] Walk-forward validation completed for {len(model_results)} models", color="green")
        return result

    def _check_regime_distribution(self, y_train: np.ndarray, y_val: np.ndarray) -> bool:
        """Check if regime distribution meets minimum requirements."""
        train_regimes = np.unique(y_train)
        val_regimes = np.unique(y_val)

        # Check if all regimes present in training
        for regime in val_regimes:
            train_count = np.sum(y_train == regime)
            if train_count < self.config.min_regime_samples:
                return False

        return True

    def _calculate_fold_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray],
        fold_idx: int
    ) -> Dict[str, float]:
        """Calculate metrics for a single fold."""
        metrics = {}

        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1

        # Temporal metrics (simplified - can be expanded)
        if len(y_pred) > 1:
            # Mean Episode Length (MEL)
            episode_lengths = []
            current_regime = y_pred[0]
            current_length = 1

            for i in range(1, len(y_pred)):
                if y_pred[i] == current_regime:
                    current_length += 1
                else:
                    episode_lengths.append(current_length)
                    current_regime = y_pred[i]
                    current_length = 1
            episode_lengths.append(current_length)

            metrics['mel'] = np.mean(episode_lengths)
            metrics['transition_rate'] = len(episode_lengths) / len(y_pred)

            # Switch False Positive Rate (SFPR)
            sfpr_count = 0
            for i in range(1, len(y_pred) - 1):
                if y_pred[i] != y_pred[i-1] and y_pred[i] != y_pred[i+1] and y_pred[i-1] == y_pred[i+1]:
                    sfpr_count += 1
            metrics['sfpr'] = sfpr_count / max(1, len(episode_lengths))

        # Confidence metrics
        if y_pred_proba is not None:
            metrics['mean_confidence'] = np.mean(np.max(y_pred_proba, axis=1))
            metrics['std_confidence'] = np.std(np.max(y_pred_proba, axis=1))

        return metrics

    def _aggregate_fold_metrics(self, fold_results: List[Dict[str, float]]) -> Dict[str, Any]:
        """Aggregate metrics across folds with confidence intervals."""
        aggregated = {}

        # Get all metric names
        metric_names = set()
        for fold_result in fold_results:
            metric_names.update(fold_result.keys())
        metric_names.discard('fold')
        metric_names.discard('model_name')

        # Store fold-level scores
        fold_scores = {}

        # Calculate statistics for each metric
        for metric_name in metric_names:
            values = [fr[metric_name] for fr in fold_results if metric_name in fr]

            if not values:
                continue

            # Store fold-level scores
            fold_scores[metric_name] = values

            # Calculate statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)

            # Calculate 95% confidence interval
            ci_lower, ci_upper = self._calculate_confidence_interval(values)

            aggregated[metric_name] = {
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_folds': len(values)
            }

        # Group temporal metrics
        temporal_metric_names = ['mel', 'transition_rate', 'sfpr', 'mean_confidence', 'std_confidence']
        temporal_metrics = {}
        for tm_name in temporal_metric_names:
            if tm_name in aggregated:
                temporal_metrics[tm_name] = aggregated[tm_name]

        aggregated['temporal_metrics'] = temporal_metrics
        aggregated['fold_scores'] = fold_scores

        return aggregated

    def _calculate_confidence_interval(self, values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval using t-distribution."""
        from scipy import stats

        n = len(values)
        if n < 2:
            return (np.mean(values), np.mean(values))

        mean = np.mean(values)
        std_err = stats.sem(values)
        margin = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)

        return (mean - margin, mean + margin)

    def _rank_models(self, model_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank models by OOS performance."""
        rankings = []

        for model_name, results in model_results.items():
            # Calculate composite score (weighted average of metrics)
            accuracy = results.get('accuracy', {}).get('mean', 0)
            f1 = results.get('f1_score', {}).get('mean', 0)
            precision = results.get('precision', {}).get('mean', 0)
            recall = results.get('recall', {}).get('mean', 0)

            # Temporal stability bonus
            mel = results.get('temporal_metrics', {}).get('mel', {}).get('mean', 0)
            sfpr = results.get('temporal_metrics', {}).get('sfpr', {}).get('mean', 1)

            # Composite score: accuracy (40%) + f1 (30%) + stability (30%)
            stability_score = (mel / 10.0) * (1 - sfpr)  # Normalize MEL and penalize SFPR
            composite_score = (accuracy * 0.4) + (f1 * 0.3) + (stability_score * 0.3)

            rankings.append({
                'model_name': model_name,
                'composite_score': composite_score,
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'mel': mel,
                'sfpr': sfpr,
                'accuracy_ci': (
                    results.get('accuracy', {}).get('ci_lower', 0),
                    results.get('accuracy', {}).get('ci_upper', 0)
                ),
                'f1_ci': (
                    results.get('f1_score', {}).get('ci_lower', 0),
                    results.get('f1_score', {}).get('ci_upper', 0)
                )
            })

        # Sort by composite score
        rankings.sort(key=lambda x: x['composite_score'], reverse=True)

        return rankings


def select_top_models(
    validation_result: RegimeValidationResult,
    top_n: int = 3
) -> List[str]:
    """
    Select top N models based on OOS performance.

    Args:
        validation_result: Result from walk-forward validation
        top_n: Number of top models to select

    Returns:
        List of top model names
    """
    rankings = validation_result.model_rankings
    top_models = [r['model_name'] for r in rankings[:top_n]]

    tprint(f"🏆 [REGIME_WF_VAL] Top {len(top_models)} models selected:", color="green")
    for i, model_name in enumerate(top_models, 1):
        rank_info = rankings[i-1]
        tprint(
            f"   {i}. {model_name}: "
            f"score={rank_info['composite_score']:.4f}, "
            f"acc={rank_info['accuracy']:.4f} "
            f"[{rank_info['accuracy_ci'][0]:.4f}, {rank_info['accuracy_ci'][1]:.4f}]",
            color="blue"
        )

    return top_models
