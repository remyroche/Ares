"""
Data-Driven Regime-to-Model Mapping System for Hybrid NAS-TAS

This module implements a completely data-driven approach to map market regimes to optimal models
without any heuristics or hardcoded choices. It automatically discovers the best model for each
regime through performance analysis and continuous learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from collections import defaultdict, deque
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a model in a specific regime."""
    regime_id: int
    model_name: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    execution_time: float = 0.0
    stability_score: float = 0.0
    confidence_score: float = 0.0
    sample_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    performance_history: List[float] = field(default_factory=list)

@dataclass
class RegimeModelMapping:
    """Mapping of regime to optimal model with confidence scores."""
    regime_id: int
    primary_model: str
    secondary_models: List[str] = field(default_factory=list)
    model_weights: Dict[str, float] = field(default_factory=dict)
    confidence_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    performance_trend: str = "stable"  # "improving", "stable", "declining"
    regime_characteristics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelSelectorConfig:
    """Configuration for data-driven model selector."""

    # Performance tracking
    min_samples_for_evaluation: int = 100
    performance_window: int = 1000  # Number of recent samples to consider
    confidence_threshold: float = 0.7
    stability_threshold: float = 0.8

    # Model selection criteria
    primary_metric: str = "f1_score"  # Primary metric for model selection
    secondary_metrics: List[str] = field(default_factory=lambda: ["accuracy", "sharpe_ratio", "stability_score"])
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "f1_score": 0.4,
        "accuracy": 0.2,
        "sharpe_ratio": 0.2,
        "stability_score": 0.2
    })

    # Learning parameters
    learning_rate: float = 0.1
    adaptation_threshold: float = 0.05  # Minimum improvement to switch models
    exploration_rate: float = 0.1  # Rate of exploring new models
    decay_factor: float = 0.95  # Decay factor for old performance data

    # Ensemble parameters
    enable_ensemble: bool = True
    max_ensemble_models: int = 3
    ensemble_weight_threshold: float = 0.1

    # Continuous learning
    enable_continuous_learning: bool = True
    retraining_frequency: int = 1000  # Retrain every N samples
    performance_drift_threshold: float = 0.1

    # Data storage
    save_mappings: bool = True
    mapping_file_path: str = "regime_model_mappings.pkl"
    performance_file_path: str = "model_performance_history.pkl"

class DataDrivenModelSelector:
    """
    Data-driven model selector that automatically maps regimes to optimal models.

    This system learns the best model for each regime through continuous performance
    evaluation and adapts to changing market conditions without any hardcoded rules.
    """

    def __init__(self, config: ModelSelectorConfig):
        """Initialize data-driven model selector."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Performance tracking
        self.model_performance: Dict[Tuple[int, str], ModelPerformanceMetrics] = {}
        self.regime_mappings: Dict[int, RegimeModelMapping] = {}
        self.performance_history: deque = deque(maxlen=config.performance_window)

        # Learning state
        self.regime_characteristics: Dict[int, Dict[str, Any]] = {}
        self.model_exploration_count: Dict[Tuple[int, str], int] = defaultdict(int)
        self.last_retraining: Dict[int, datetime] = {}

        # Load existing mappings if available
        self._load_existing_mappings()

        self.logger.info("✅ Data-driven model selector initialized")
        self.logger.info(f"   Primary metric: {config.primary_metric}")
        self.logger.info(f"   Ensemble enabled: {config.enable_ensemble}")
        self.logger.info(f"   Continuous learning: {config.enable_continuous_learning}")

    def register_model_performance(self,
                                 regime_id: int,
                                 model_name: str,
                                 predictions: np.ndarray,
                                 actual_values: np.ndarray,
                                 execution_time: float,
                                 regime_characteristics: Optional[Dict[str, Any]] = None) -> ModelPerformanceMetrics:
        """
        Register performance metrics for a model in a specific regime.

        Args:
            regime_id: ID of the market regime
            model_name: Name of the model
            predictions: Model predictions
            actual_values: Actual values
            execution_time: Time taken for inference
            regime_characteristics: Characteristics of the regime

        Returns:
            Updated performance metrics
        """
        try:
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(
                regime_id, model_name, predictions, actual_values, execution_time
            )

            # Update performance tracking
            key = (regime_id, model_name)
            if key in self.model_performance:
                # Update existing metrics
                old_metrics = self.model_performance[key]
                metrics = self._update_performance_metrics(old_metrics, metrics)
            else:
                # Create new metrics
                metrics.regime_id = regime_id
                metrics.model_name = model_name
                metrics.last_updated = datetime.now()

            self.model_performance[key] = metrics

            # Update regime characteristics
            if regime_characteristics:
                self.regime_characteristics[regime_id] = regime_characteristics

            # Add to performance history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'regime_id': regime_id,
                'model_name': model_name,
                'metrics': metrics
            })

            # Check if we need to update regime mappings
            self._update_regime_mappings(regime_id)

            # Check for continuous learning
            if self.config.enable_continuous_learning:
                self._check_continuous_learning(regime_id)

            self.logger.debug(f"Updated performance for regime {regime_id}, model {model_name}: "
                            f"F1={metrics.f1_score:.3f}, Accuracy={metrics.accuracy:.3f}")

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to register model performance: {e}")
            raise

    def select_model_for_regime(self, regime_id: int, available_models: List[str]) -> Tuple[str, Dict[str, float]]:
        """
        Select the best model for a specific regime.

        Args:
            regime_id: ID of the market regime
            available_models: List of available model names

        Returns:
            Tuple of (selected_model, ensemble_weights)
        """
        try:
            # Check if we have a mapping for this regime
            if regime_id in self.regime_mappings:
                mapping = self.regime_mappings[regime_id]

                # Check if primary model is still available and performing well
                if (mapping.primary_model in available_models and
                    mapping.confidence_score >= self.config.confidence_threshold):

                    if self.config.enable_ensemble:
                        return mapping.primary_model, mapping.model_weights
                    else:
                        return mapping.primary_model, {mapping.primary_model: 1.0}

            # No mapping or low confidence - select best available model
            return self._select_best_available_model(regime_id, available_models)

        except Exception as e:
            self.logger.error(f"Failed to select model for regime {regime_id}: {e}")
            # Fallback to first available model
            return available_models[0] if available_models else "default", {"default": 1.0}

    def get_ensemble_weights(self, regime_id: int, available_models: List[str]) -> Dict[str, float]:
        """
        Get ensemble weights for models in a specific regime.

        Args:
            regime_id: ID of the market regime
            available_models: List of available model names

        Returns:
            Dictionary of model weights
        """
        try:
            if not self.config.enable_ensemble:
                # Return single model weight
                model, _ = self.select_model_for_regime(regime_id, available_models)
                return {model: 1.0}

            # Get performance scores for available models
            model_scores = {}
            for model_name in available_models:
                key = (regime_id, model_name)
                if key in self.model_performance:
                    metrics = self.model_performance[key]
                    score = self._calculate_composite_score(metrics)
                    model_scores[model_name] = score
                else:
                    # Unknown model - give it a small weight for exploration
                    model_scores[model_name] = self.config.exploration_rate

            # Normalize weights
            total_score = sum(model_scores.values())
            if total_score > 0:
                weights = {model: score / total_score for model, score in model_scores.items()}
            else:
                # Equal weights if no scores available
                weights = {model: 1.0 / len(available_models) for model in available_models}

            # Filter out models with very low weights
            filtered_weights = {
                model: weight for model, weight in weights.items()
                if weight >= self.config.ensemble_weight_threshold
            }

            # Renormalize if we filtered out models
            if filtered_weights:
                total_weight = sum(filtered_weights.values())
                weights = {model: weight / total_weight for model, weight in filtered_weights.items()}

            return weights

        except Exception as e:
            self.logger.error(f"Failed to get ensemble weights for regime {regime_id}: {e}")
            # Fallback to equal weights
            return {model: 1.0 / len(available_models) for model in available_models}

    def get_regime_insights(self, regime_id: int) -> Dict[str, Any]:
        """
        Get insights about model performance in a specific regime.

        Args:
            regime_id: ID of the market regime

        Returns:
            Dictionary with regime insights
        """
        try:
            insights = {
                'regime_id': regime_id,
                'mapping': self.regime_mappings.get(regime_id),
                'model_performance': {},
                'regime_characteristics': self.regime_characteristics.get(regime_id, {}),
                'performance_trend': 'unknown',
                'recommendations': []
            }

            # Get performance for all models in this regime
            for (r_id, model_name), metrics in self.model_performance.items():
                if r_id == regime_id:
                    insights['model_performance'][model_name] = {
                        'f1_score': metrics.f1_score,
                        'accuracy': metrics.accuracy,
                        'sharpe_ratio': metrics.sharpe_ratio,
                        'stability_score': metrics.stability_score,
                        'sample_count': metrics.sample_count,
                        'last_updated': metrics.last_updated.isoformat()
                    }

            # Analyze performance trend
            if regime_id in self.regime_mappings:
                mapping = self.regime_mappings[regime_id]
                insights['performance_trend'] = mapping.performance_trend

            # Generate recommendations
            insights['recommendations'] = self._generate_recommendations(regime_id)

            return insights

        except Exception as e:
            self.logger.error(f"Failed to get regime insights for regime {regime_id}: {e}")
            return {'regime_id': regime_id, 'error': str(e)}

    def _calculate_performance_metrics(self,
                                     regime_id: int,
                                     model_name: str,
                                     predictions: np.ndarray,
                                     actual_values: np.ndarray,
                                     execution_time: float) -> ModelPerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        try:
            # Classification metrics
            accuracy = accuracy_score(actual_values, predictions)
            precision = precision_score(actual_values, predictions, average='weighted', zero_division=0)
            recall = recall_score(actual_values, predictions, average='weighted', zero_division=0)
            f1 = f1_score(actual_values, predictions, average='weighted', zero_division=0)

            # Regression metrics (if applicable)
            mse = mean_squared_error(actual_values, predictions)
            mae = mean_absolute_error(actual_values, predictions)
            r2 = r2_score(actual_values, predictions)

            # Trading metrics (simplified)
            returns = np.diff(predictions) / (predictions[:-1] + 1e-8)
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) if len(returns) > 1 else 0.0

            # Calculate drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

            # Win rate
            win_rate = np.mean(returns > 0) if len(returns) > 0 else 0.0

            # Profit factor
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            profit_factor = (np.sum(positive_returns) / abs(np.sum(negative_returns))) if len(negative_returns) > 0 else 1.0

            # Stability score (based on consistency of performance)
            stability_score = 1.0 - np.std([accuracy, precision, recall, f1]) / (np.mean([accuracy, precision, recall, f1]) + 1e-8)

            # Confidence score (based on sample size and consistency)
            confidence_score = min(1.0, len(predictions) / self.config.min_samples_for_evaluation)

            return ModelPerformanceMetrics(
                regime_id=regime_id,
                model_name=model_name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                mse=mse,
                mae=mae,
                r2_score=r2,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                execution_time=execution_time,
                stability_score=stability_score,
                confidence_score=confidence_score,
                sample_count=len(predictions)
            )

        except Exception as e:
            self.logger.error(f"Failed to calculate performance metrics: {e}")
            # Return default metrics
            return ModelPerformanceMetrics(
                regime_id=regime_id,
                model_name=model_name,
                sample_count=len(predictions)
            )

    def _update_performance_metrics(self, old_metrics: ModelPerformanceMetrics, new_metrics: ModelPerformanceMetrics) -> ModelPerformanceMetrics:
        """Update existing performance metrics with new data."""
        try:
            # Weighted average with decay factor
            decay = self.config.decay_factor
            weight_old = decay
            weight_new = 1.0 - decay

            # Update metrics with weighted average
            old_metrics.accuracy = weight_old * old_metrics.accuracy + weight_new * new_metrics.accuracy
            old_metrics.precision = weight_old * old_metrics.precision + weight_new * new_metrics.precision
            old_metrics.recall = weight_old * old_metrics.recall + weight_new * new_metrics.recall
            old_metrics.f1_score = weight_old * old_metrics.f1_score + weight_new * new_metrics.f1_score
            old_metrics.sharpe_ratio = weight_old * old_metrics.sharpe_ratio + weight_new * new_metrics.sharpe_ratio
            old_metrics.stability_score = weight_old * old_metrics.stability_score + weight_new * new_metrics.stability_score

            # Update sample count
            old_metrics.sample_count += new_metrics.sample_count

            # Update execution time (exponential moving average)
            old_metrics.execution_time = 0.9 * old_metrics.execution_time + 0.1 * new_metrics.execution_time

            # Add to performance history
            old_metrics.performance_history.append(new_metrics.f1_score)
            if len(old_metrics.performance_history) > 100:  # Keep last 100 samples
                old_metrics.performance_history.pop(0)

            old_metrics.last_updated = datetime.now()

            return old_metrics

        except Exception as e:
            self.logger.error(f"Failed to update performance metrics: {e}")
            return old_metrics

    def _update_regime_mappings(self, regime_id: int):
        """Update regime-to-model mappings based on current performance."""
        try:
            # Get all models for this regime
            regime_models = [(r_id, model_name) for (r_id, model_name) in self.model_performance.keys() if r_id == regime_id]

            if not regime_models:
                return

            # Calculate composite scores for all models
            model_scores = {}
            for (r_id, model_name) in regime_models:
                metrics = self.model_performance[(r_id, model_name)]
                if metrics.sample_count >= self.config.min_samples_for_evaluation:
                    score = self._calculate_composite_score(metrics)
                    model_scores[model_name] = score

            if not model_scores:
                return

            # Sort models by score
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

            # Get best model
            primary_model = sorted_models[0][0]
            primary_score = sorted_models[0][1]

            # Get secondary models for ensemble
            secondary_models = []
            model_weights = {}

            if self.config.enable_ensemble:
                for model_name, score in sorted_models[:self.config.max_ensemble_models]:
                    if model_name != primary_model:
                        secondary_models.append(model_name)

                    # Calculate weight based on score
                    weight = score / sum(model_scores.values())
                    if weight >= self.config.ensemble_weight_threshold:
                        model_weights[model_name] = weight

                # Normalize weights
                total_weight = sum(model_weights.values())
                if total_weight > 0:
                    model_weights = {model: weight / total_weight for model, weight in model_weights.items()}

            # Calculate confidence score
            confidence_score = min(1.0, primary_score)

            # Determine performance trend
            performance_trend = "stable"
            if regime_id in self.regime_mappings:
                old_mapping = self.regime_mappings[regime_id]
                if old_mapping.primary_model == primary_model:
                    # Check if performance is improving
                    old_score = model_scores.get(old_mapping.primary_model, 0.0)
                    if primary_score > old_score + self.config.adaptation_threshold:
                        performance_trend = "improving"
                    elif primary_score < old_score - self.config.adaptation_threshold:
                        performance_trend = "declining"

            # Create or update mapping
            mapping = RegimeModelMapping(
                regime_id=regime_id,
                primary_model=primary_model,
                secondary_models=secondary_models,
                model_weights=model_weights,
                confidence_score=confidence_score,
                last_updated=datetime.now(),
                performance_trend=performance_trend,
                regime_characteristics=self.regime_characteristics.get(regime_id, {})
            )

            self.regime_mappings[regime_id] = mapping

            self.logger.info(f"Updated mapping for regime {regime_id}: {primary_model} "
                           f"(confidence: {confidence_score:.3f}, trend: {performance_trend})")

        except Exception as e:
            self.logger.error(f"Failed to update regime mappings for regime {regime_id}: {e}")

    def _select_best_available_model(self, regime_id: int, available_models: List[str]) -> Tuple[str, Dict[str, float]]:
        """Select the best available model for a regime."""
        try:
            best_model = available_models[0]  # Default fallback
            best_score = 0.0

            # Check performance of available models
            for model_name in available_models:
                key = (regime_id, model_name)
                if key in self.model_performance:
                    metrics = self.model_performance[key]
                    score = self._calculate_composite_score(metrics)
                    if score > best_score:
                        best_score = score
                        best_model = model_name

            # Return single model weight
            return best_model, {best_model: 1.0}

        except Exception as e:
            self.logger.error(f"Failed to select best available model: {e}")
            return available_models[0], {available_models[0]: 1.0}

    def _calculate_composite_score(self, metrics: ModelPerformanceMetrics) -> float:
        """Calculate composite score based on weighted metrics."""
        try:
            score = 0.0

            # Primary metric
            primary_metric_value = getattr(metrics, self.config.primary_metric, 0.0)
            score += self.config.metric_weights.get(self.config.primary_metric, 0.4) * primary_metric_value

            # Secondary metrics
            for metric_name in self.config.secondary_metrics:
                metric_value = getattr(metrics, metric_name, 0.0)
                weight = self.config.metric_weights.get(metric_name, 0.2)
                score += weight * metric_value

            # Apply confidence penalty
            confidence_penalty = 1.0 - metrics.confidence_score
            score *= (1.0 - confidence_penalty * 0.2)  # 20% penalty for low confidence

            return max(0.0, score)

        except Exception as e:
            self.logger.error(f"Failed to calculate composite score: {e}")
            return 0.0

    def _check_continuous_learning(self, regime_id: int):
        """Check if continuous learning is needed for a regime."""
        try:
            current_time = datetime.now()

            # Check if enough time has passed since last retraining
            if regime_id in self.last_retraining:
                time_since_retraining = (current_time - self.last_retraining[regime_id]).total_seconds()
                if time_since_retraining < 3600:  # 1 hour minimum between retraining
                    return

            # Check if performance has drifted
            if regime_id in self.regime_mappings:
                mapping = self.regime_mappings[regime_id]
                if mapping.performance_trend == "declining":
                    self.logger.info(f"Triggering continuous learning for regime {regime_id} due to declining performance")
                    self._trigger_continuous_learning(regime_id)
                    self.last_retraining[regime_id] = current_time

        except Exception as e:
            self.logger.error(f"Failed to check continuous learning for regime {regime_id}: {e}")

    def _trigger_continuous_learning(self, regime_id: int):
        """Trigger continuous learning for a regime."""
        try:
            # This would integrate with the training pipeline
            # For now, we'll just log the trigger
            self.logger.info(f"Continuous learning triggered for regime {regime_id}")

            # In a real implementation, this would:
            # 1. Collect recent data for the regime
            # 2. Retrain models with new data
            # 3. Update performance metrics
            # 4. Re-evaluate regime mappings

        except Exception as e:
            self.logger.error(f"Failed to trigger continuous learning for regime {regime_id}: {e}")

    def _generate_recommendations(self, regime_id: int) -> List[str]:
        """Generate recommendations for a regime."""
        try:
            recommendations = []

            if regime_id not in self.regime_mappings:
                recommendations.append("No model mapping available - consider training models for this regime")
                return recommendations

            mapping = self.regime_mappings[regime_id]

            # Performance-based recommendations
            if mapping.confidence_score < self.config.confidence_threshold:
                recommendations.append("Low confidence in current model - consider collecting more data or trying different models")

            if mapping.performance_trend == "declining":
                recommendations.append("Performance declining - consider retraining or switching models")

            if mapping.performance_trend == "improving":
                recommendations.append("Performance improving - current model selection is working well")

            # Ensemble recommendations
            if self.config.enable_ensemble and len(mapping.secondary_models) > 0:
                recommendations.append(f"Consider using ensemble with {len(mapping.secondary_models)} secondary models")

            # Sample size recommendations
            regime_models = [(r_id, model_name) for (r_id, model_name) in self.model_performance.keys() if r_id == regime_id]
            total_samples = sum(self.model_performance[key].sample_count for key in regime_models)

            if total_samples < self.config.min_samples_for_evaluation:
                recommendations.append(f"Low sample count ({total_samples}) - consider collecting more data for better evaluation")

            return recommendations

        except Exception as e:
            self.logger.error(f"Failed to generate recommendations for regime {regime_id}: {e}")
            return ["Error generating recommendations"]

    def _load_existing_mappings(self):
        """Load existing mappings from file."""
        try:
            if self.config.save_mappings and Path(self.config.mapping_file_path).exists():
                with open(self.config.mapping_file_path, 'rb') as f:
                    data = pickle.load(f)
                    self.regime_mappings = data.get('regime_mappings', {})
                    self.model_performance = data.get('model_performance', {})
                    self.regime_characteristics = data.get('regime_characteristics', {})

                self.logger.info(f"Loaded existing mappings: {len(self.regime_mappings)} regimes, "
                               f"{len(self.model_performance)} model-regime pairs")

        except Exception as e:
            self.logger.warning(f"Failed to load existing mappings: {e}")

    def save_mappings(self):
        """Save current mappings to file."""
        try:
            if self.config.save_mappings:
                data = {
                    'regime_mappings': self.regime_mappings,
                    'model_performance': self.model_performance,
                    'regime_characteristics': self.regime_characteristics,
                    'timestamp': datetime.now().isoformat()
                }

                with open(self.config.mapping_file_path, 'wb') as f:
                    pickle.dump(data, f)

                self.logger.info(f"Saved mappings: {len(self.regime_mappings)} regimes, "
                               f"{len(self.model_performance)} model-regime pairs")

        except Exception as e:
            self.logger.error(f"Failed to save mappings: {e}")

    def get_system_summary(self) -> Dict[str, Any]:
        """Get summary of the entire system."""
        try:
            summary = {
                'total_regimes': len(self.regime_mappings),
                'total_model_regime_pairs': len(self.model_performance),
                'regime_mappings': {},
                'performance_summary': {},
                'system_health': 'healthy'
            }

            # Regime mappings summary
            for regime_id, mapping in self.regime_mappings.items():
                summary['regime_mappings'][regime_id] = {
                    'primary_model': mapping.primary_model,
                    'confidence_score': mapping.confidence_score,
                    'performance_trend': mapping.performance_trend,
                    'secondary_models': mapping.secondary_models,
                    'last_updated': mapping.last_updated.isoformat()
                }

            # Performance summary
            if self.model_performance:
                all_f1_scores = [metrics.f1_score for metrics in self.model_performance.values()]
                all_accuracies = [metrics.accuracy for metrics in self.model_performance.values()]

                summary['performance_summary'] = {
                    'average_f1_score': np.mean(all_f1_scores),
                    'average_accuracy': np.mean(all_accuracies),
                    'best_f1_score': np.max(all_f1_scores),
                    'best_accuracy': np.max(all_accuracies),
                    'total_samples': sum(metrics.sample_count for metrics in self.model_performance.values())
                }

            return summary

        except Exception as e:
            self.logger.error(f"Failed to get system summary: {e}")
            return {'error': str(e)}
