"""
Regime-Specific ML Model Training Framework.

This module provides a comprehensive framework for training machine learning
models on different market regimes. It implements regime-aware training
strategies, model selection, and performance evaluation specifically designed
for trading applications.

Key Features:
- Regime-specific model training (separate models per regime)
- Multi-regime ensemble models (single model aware of all regimes)
- Regime transition models (predict regime changes)
- Adaptive models (online learning for regime drift)
- Performance evaluation across regimes
- Model selection and hyperparameter optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json
import pickle
from abc import ABC, abstractmethod

from src.utils.logger import system_logger


class ModelType(Enum):
    """Enumeration of model types."""
    # Traditional ML models
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    SVM = "svm"
    LOGISTIC_REGRESSION = "logistic_regression"
    
    # Neural networks
    MLP = "mlp"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    
    # Ensemble methods
    VOTING_CLASSIFIER = "voting_classifier"
    STACKING_CLASSIFIER = "stacking_classifier"
    BAGGING_CLASSIFIER = "bagging_classifier"
    
    # Specialized models
    REGIME_AWARE_NN = "regime_aware_nn"
    ADAPTIVE_MODEL = "adaptive_model"


class TrainingStrategy(Enum):
    """Enumeration of training strategies."""
    REGIME_SPECIFIC = "regime_specific"  # Separate model per regime
    MULTI_REGIME = "multi_regime"      # Single model with regime features
    ENSEMBLE = "ensemble"              # Ensemble of regime-specific models
    HIERARCHICAL = "hierarchical"      # Hierarchical regime-aware training
    ADAPTIVE = "adaptive"              # Online/adaptive learning
    TRANSFER_LEARNING = "transfer_learning"  # Transfer between regimes


@dataclass
class TrainingConfig:
    """Configuration for ML model training."""
    # General parameters
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    cross_validation_folds: int = 5
    
    # Training strategy
    strategy: TrainingStrategy = TrainingStrategy.REGIME_SPECIFIC
    
    # Model parameters
    model_params: Dict[ModelType, Dict[str, Any]] = None
    
    # Feature engineering
    use_regime_features: bool = True
    use_temporal_features: bool = True
    feature_selection: bool = True
    feature_selection_k: int = 50
    
    # Hyperparameter optimization
    use_hyperopt: bool = True
    hyperopt_trials: int = 100
    hyperopt_scoring: str = "f1_weighted"
    
    # Training parameters
    early_stopping: bool = True
    early_stopping_patience: int = 10
    batch_size: int = 32
    max_epochs: int = 100
    
    # Ensemble parameters
    ensemble_methods: List[ModelType] = None
    ensemble_weights: Optional[Dict[ModelType, float]] = None
    
    # Adaptive learning parameters
    adaptation_window: int = 252  # Trading days
    adaptation_threshold: float = 0.05
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.model_params is None:
            self.model_params = {
                ModelType.RANDOM_FOREST: {'n_estimators': 100, 'max_depth': 10},
                ModelType.XGBOOST: {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
                ModelType.LIGHTGBM: {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
                ModelType.MLP: {'hidden_layer_sizes': (100, 50), 'max_iter': 1000}
            }
        if self.ensemble_methods is None:
            self.ensemble_methods = [ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.LIGHTGBM]


@dataclass
class TrainingResult:
    """Result container for model training."""
    model_type: ModelType
    strategy: TrainingStrategy
    models: Dict[str, Any]  # Can contain single model or regime-specific models
    performance_metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]]
    regime_performance: Optional[Dict[int, Dict[str, float]]]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (excluding models)."""
        return {
            'model_type': self.model_type.value,
            'strategy': self.strategy.value,
            'performance_metrics': self.performance_metrics,
            'feature_importance': self.feature_importance,
            'regime_performance': self.regime_performance,
            'metadata': self.metadata
        }


class BaseModelTrainer(ABC):
    """Abstract base class for model trainers."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = system_logger.getChild(f'Trainer.{self.__class__.__name__}')
    
    @abstractmethod
    def train(self, 
             features: pd.DataFrame,
             target: np.ndarray,
             regime_labels: np.ndarray) -> TrainingResult:
        """Train the model."""
        pass
    
    def _prepare_features(self, 
                         features: pd.DataFrame,
                         regime_labels: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Prepare features for training."""
        prepared_features = features.copy()
        
        # Add regime features if requested
        if self.config.use_regime_features and regime_labels is not None:
            # One-hot encode regime labels
            unique_regimes = np.unique(regime_labels)
            for regime in unique_regimes:
                prepared_features[f'regime_{regime}'] = (regime_labels == regime).astype(int)
        
        # Add temporal features if requested
        if self.config.use_temporal_features:
            prepared_features = self._add_temporal_features(prepared_features)
        
        # Feature selection
        if self.config.feature_selection and len(prepared_features.columns) > self.config.feature_selection_k:
            prepared_features = self._select_features(prepared_features, target)
        
        return prepared_features
    
    def _add_temporal_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features."""
        # Add lagged features
        for lag in [1, 5, 10]:
            for col in features.select_dtypes(include=[np.number]).columns[:10]:  # Limit to prevent explosion
                features[f'{col}_lag_{lag}'] = features[col].shift(lag)
        
        # Add rolling statistics
        for window in [5, 10, 20]:
            for col in features.select_dtypes(include=[np.number]).columns[:5]:
                features[f'{col}_ma_{window}'] = features[col].rolling(window).mean()
                features[f'{col}_std_{window}'] = features[col].rolling(window).std()
        
        return features.fillna(method='ffill').fillna(0)
    
    def _select_features(self, features: pd.DataFrame, target: np.ndarray) -> pd.DataFrame:
        """Select top K features."""
        from sklearn.feature_selection import SelectKBest, f_classif, f_regression
        
        # Determine if classification or regression
        unique_targets = len(np.unique(target))
        is_classification = unique_targets < min(len(target) * 0.1, 50)
        
        # Select features
        if is_classification:
            selector = SelectKBest(f_classif, k=min(self.config.feature_selection_k, len(features.columns)))
        else:
            selector = SelectKBest(f_regression, k=min(self.config.feature_selection_k, len(features.columns)))
        
        X_selected = selector.fit_transform(features.fillna(0), target)
        selected_features = features.columns[selector.get_support()]
        
        return pd.DataFrame(X_selected, columns=selected_features, index=features.index)
    
    def _calculate_performance_metrics(self, 
                                     y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate performance metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
        )
        
        metrics = {}
        
        # Determine if classification or regression
        unique_targets = len(np.unique(y_true))
        is_classification = unique_targets < min(len(y_true) * 0.1, 50)
        
        if is_classification:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # ROC AUC for binary classification
            if unique_targets == 2 and y_prob is not None:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
                except:
                    pass
        else:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
        
        return metrics


class RegimeSpecificTrainer(BaseModelTrainer):
    """Trainer for regime-specific models."""
    
    def train(self, 
             features: pd.DataFrame,
             target: np.ndarray,
             regime_labels: np.ndarray) -> TrainingResult:
        """Train separate models for each regime."""
        
        # Prepare features (without regime features for regime-specific training)
        config_copy = self.config
        config_copy.use_regime_features = False
        prepared_features = self._prepare_features(features, regime_labels)
        
        unique_regimes = np.unique(regime_labels)
        regime_models = {}
        regime_performance = {}
        all_feature_importance = {}
        
        for regime in unique_regimes:
            self.logger.info(f"🎯 Training model for regime {regime}")
            
            # Get regime-specific data
            regime_mask = regime_labels == regime
            regime_features = prepared_features[regime_mask]
            regime_target = target[regime_mask]
            
            if len(regime_target) < 10:  # Skip regimes with too few samples
                self.logger.warning(f"Skipping regime {regime}: insufficient samples ({len(regime_target)})")
                continue
            
            # Train regime-specific model
            model = self._train_single_model(regime_features, regime_target)
            regime_models[f'regime_{regime}'] = model
            
            # Evaluate performance
            y_pred = model.predict(regime_features)
            y_prob = model.predict_proba(regime_features) if hasattr(model, 'predict_proba') else None
            
            performance = self._calculate_performance_metrics(regime_target, y_pred, y_prob)
            regime_performance[int(regime)] = performance
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(regime_features.columns, model.feature_importances_))
                all_feature_importance[f'regime_{regime}'] = importance
        
        # Calculate overall performance (weighted by regime size)
        overall_performance = self._calculate_weighted_performance(regime_performance, regime_labels)
        
        return TrainingResult(
            model_type=ModelType.RANDOM_FOREST,  # Default, should be parameterized
            strategy=TrainingStrategy.REGIME_SPECIFIC,
            models=regime_models,
            performance_metrics=overall_performance,
            feature_importance=all_feature_importance,
            regime_performance=regime_performance,
            metadata={
                'n_regimes': len(regime_models),
                'regime_sizes': {int(r): int(np.sum(regime_labels == r)) for r in unique_regimes}
            }
        )
    
    def _train_single_model(self, features: pd.DataFrame, target: np.ndarray):
        """Train a single model."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        # Determine if classification or regression
        unique_targets = len(np.unique(target))
        is_classification = unique_targets < min(len(target) * 0.1, 50)
        
        # Get model parameters
        model_params = self.config.model_params.get(ModelType.RANDOM_FOREST, {})
        model_params['random_state'] = self.config.random_state
        
        if is_classification:
            model = RandomForestClassifier(**model_params)
        else:
            model = RandomForestRegressor(**model_params)
        
        # Train model
        model.fit(features.fillna(0), target)
        
        return model
    
    def _calculate_weighted_performance(self, 
                                      regime_performance: Dict[int, Dict[str, float]],
                                      regime_labels: np.ndarray) -> Dict[str, float]:
        """Calculate weighted average performance across regimes."""
        if not regime_performance:
            return {}
        
        # Calculate weights based on regime sizes
        regime_sizes = {}
        for regime in regime_performance.keys():
            regime_sizes[regime] = np.sum(regime_labels == regime)
        
        total_size = sum(regime_sizes.values())
        weights = {regime: size / total_size for regime, size in regime_sizes.items()}
        
        # Calculate weighted averages
        weighted_performance = {}
        
        # Get all metric names
        all_metrics = set()
        for performance in regime_performance.values():
            all_metrics.update(performance.keys())
        
        for metric in all_metrics:
            weighted_sum = 0
            total_weight = 0
            
            for regime, performance in regime_performance.items():
                if metric in performance:
                    weight = weights[regime]
                    weighted_sum += weight * performance[metric]
                    total_weight += weight
            
            if total_weight > 0:
                weighted_performance[f'weighted_{metric}'] = weighted_sum / total_weight
        
        return weighted_performance


class MultiRegimeTrainer(BaseModelTrainer):
    """Trainer for multi-regime aware models."""
    
    def train(self, 
             features: pd.DataFrame,
             target: np.ndarray,
             regime_labels: np.ndarray) -> TrainingResult:
        """Train a single model that uses regime information as features."""
        
        # Prepare features (including regime features)
        prepared_features = self._prepare_features(features, regime_labels)
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            prepared_features, target,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=regime_labels if len(np.unique(regime_labels)) > 1 else None
        )
        
        # Train model
        model = self._train_single_model(X_train, y_train)
        
        # Evaluate performance
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        overall_performance = self._calculate_performance_metrics(y_test, y_pred, y_prob)
        
        # Calculate regime-specific performance
        regime_performance = {}
        test_regime_labels = regime_labels[X_test.index] if hasattr(X_test, 'index') else regime_labels[-len(X_test):]
        
        for regime in np.unique(test_regime_labels):
            regime_mask = test_regime_labels == regime
            if np.sum(regime_mask) > 0:
                regime_y_true = y_test[regime_mask]
                regime_y_pred = y_pred[regime_mask]
                regime_y_prob = y_prob[regime_mask] if y_prob is not None else None
                
                regime_perf = self._calculate_performance_metrics(regime_y_true, regime_y_pred, regime_y_prob)
                regime_performance[int(regime)] = regime_perf
        
        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(prepared_features.columns, model.feature_importances_))
        
        return TrainingResult(
            model_type=ModelType.RANDOM_FOREST,  # Default
            strategy=TrainingStrategy.MULTI_REGIME,
            models={'main_model': model},
            performance_metrics=overall_performance,
            feature_importance=feature_importance,
            regime_performance=regime_performance,
            metadata={
                'n_features': len(prepared_features.columns),
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
        )
    
    def _train_single_model(self, features: pd.DataFrame, target: np.ndarray):
        """Train a single model."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        # Determine if classification or regression
        unique_targets = len(np.unique(target))
        is_classification = unique_targets < min(len(target) * 0.1, 50)
        
        # Get model parameters
        model_params = self.config.model_params.get(ModelType.RANDOM_FOREST, {})
        model_params['random_state'] = self.config.random_state
        
        if is_classification:
            model = RandomForestClassifier(**model_params)
        else:
            model = RandomForestRegressor(**model_params)
        
        # Train model
        model.fit(features.fillna(0), target)
        
        return model


class EnsembleTrainer(BaseModelTrainer):
    """Trainer for ensemble models."""
    
    def train(self, 
             features: pd.DataFrame,
             target: np.ndarray,
             regime_labels: np.ndarray) -> TrainingResult:
        """Train ensemble of different model types."""
        
        from sklearn.ensemble import VotingClassifier, VotingRegressor
        from sklearn.model_selection import train_test_split
        
        # Prepare features
        prepared_features = self._prepare_features(features, regime_labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            prepared_features, target,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        # Determine if classification or regression
        unique_targets = len(np.unique(target))
        is_classification = unique_targets < min(len(target) * 0.1, 50)
        
        # Create base models
        base_models = self._create_base_models(is_classification)
        
        # Create ensemble
        if is_classification:
            ensemble = VotingClassifier(
                estimators=list(base_models.items()),
                voting='soft'
            )
        else:
            ensemble = VotingRegressor(
                estimators=list(base_models.items())
            )
        
        # Train ensemble
        ensemble.fit(X_train.fillna(0), y_train)
        
        # Evaluate performance
        y_pred = ensemble.predict(X_test.fillna(0))
        y_prob = ensemble.predict_proba(X_test.fillna(0)) if hasattr(ensemble, 'predict_proba') else None
        
        overall_performance = self._calculate_performance_metrics(y_test, y_pred, y_prob)
        
        # Calculate regime-specific performance
        regime_performance = {}
        test_regime_labels = regime_labels[X_test.index] if hasattr(X_test, 'index') else regime_labels[-len(X_test):]
        
        for regime in np.unique(test_regime_labels):
            regime_mask = test_regime_labels == regime
            if np.sum(regime_mask) > 0:
                regime_y_true = y_test[regime_mask]
                regime_y_pred = y_pred[regime_mask]
                regime_y_prob = y_prob[regime_mask] if y_prob is not None else None
                
                regime_perf = self._calculate_performance_metrics(regime_y_true, regime_y_pred, regime_y_prob)
                regime_performance[int(regime)] = regime_perf
        
        return TrainingResult(
            model_type=ModelType.VOTING_CLASSIFIER if is_classification else ModelType.VOTING_CLASSIFIER,
            strategy=TrainingStrategy.ENSEMBLE,
            models={'ensemble_model': ensemble},
            performance_metrics=overall_performance,
            feature_importance=None,  # Ensemble feature importance is complex
            regime_performance=regime_performance,
            metadata={
                'base_models': list(base_models.keys()),
                'n_features': len(prepared_features.columns),
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
        )
    
    def _create_base_models(self, is_classification: bool) -> Dict[str, Any]:
        """Create base models for ensemble."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        
        base_models = {}
        
        for model_type in self.config.ensemble_methods:
            model_params = self.config.model_params.get(model_type, {})
            model_params['random_state'] = self.config.random_state
            
            if model_type == ModelType.RANDOM_FOREST:
                if is_classification:
                    model = RandomForestClassifier(**model_params)
                else:
                    model = RandomForestRegressor(**model_params)
                base_models['rf'] = model
            
            elif model_type == ModelType.LOGISTIC_REGRESSION and is_classification:
                model = LogisticRegression(**model_params)
                base_models['lr'] = model
            
            elif model_type == ModelType.LOGISTIC_REGRESSION and not is_classification:
                model = Ridge(**model_params)
                base_models['ridge'] = model
            
            # Add more model types as needed
        
        return base_models


class RegimeMLTrainer:
    """
    Main ML training framework for market regimes.
    
    This class provides comprehensive machine learning model training
    specifically designed for market regime analysis and trading applications.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize the regime ML trainer.
        
        Args:
            config: Configuration for ML training
        """
        self.config = config or TrainingConfig()
        self.logger = system_logger.getChild('RegimeMLTrainer')
        self.results: Dict[str, TrainingResult] = {}
        
        # Initialize trainers
        self.trainers = {
            TrainingStrategy.REGIME_SPECIFIC: RegimeSpecificTrainer(self.config),
            TrainingStrategy.MULTI_REGIME: MultiRegimeTrainer(self.config),
            TrainingStrategy.ENSEMBLE: EnsembleTrainer(self.config)
        }
    
    def train_single_strategy(self,
                            features: pd.DataFrame,
                            target: np.ndarray,
                            regime_labels: np.ndarray,
                            strategy: TrainingStrategy,
                            experiment_name: Optional[str] = None) -> TrainingResult:
        """
        Train using a single strategy.
        
        Args:
            features: Feature matrix
            target: Target variable
            regime_labels: Regime assignments
            strategy: Training strategy to use
            experiment_name: Name for this experiment
            
        Returns:
            Training result
        """
        experiment_name = experiment_name or f"{strategy.value}_{len(self.results)}"
        
        self.logger.info(f"🚀 Training {strategy.value} model: {experiment_name}")
        
        if strategy not in self.trainers:
            raise ValueError(f"Training strategy {strategy.value} not supported")
        
        result = self.trainers[strategy].train(features, target, regime_labels)
        self.results[experiment_name] = result
        
        # Log results
        main_metric = 'weighted_f1_score' if 'weighted_f1_score' in result.performance_metrics else list(result.performance_metrics.keys())[0]
        score = result.performance_metrics.get(main_metric, 0)
        
        self.logger.info(f"✅ {experiment_name} completed: {main_metric}={score:.3f}")
        
        return result
    
    def train_all_strategies(self,
                           features: pd.DataFrame,
                           target: np.ndarray,
                           regime_labels: np.ndarray) -> Dict[str, TrainingResult]:
        """
        Train using all available strategies.
        
        Args:
            features: Feature matrix
            target: Target variable
            regime_labels: Regime assignments
            
        Returns:
            Dictionary mapping experiment names to results
        """
        self.logger.info("🚀 Training all strategies")
        
        results = {}
        
        for strategy in TrainingStrategy:
            if strategy in self.trainers:
                try:
                    experiment_name = f"{strategy.value}_experiment"
                    result = self.train_single_strategy(
                        features, target, regime_labels, strategy, experiment_name
                    )
                    results[experiment_name] = result
                except Exception as e:
                    self.logger.error(f"❌ {strategy.value} failed: {e}")
                    continue
        
        self.logger.info(f"✅ Completed {len(results)} training strategies")
        return results
    
    def compare_strategies(self) -> pd.DataFrame:
        """
        Compare training strategies.
        
        Returns:
            DataFrame with strategy comparison
        """
        if not self.results:
            self.logger.warning("No training results available for comparison")
            return pd.DataFrame()
        
        comparison_data = []
        
        for experiment_name, result in self.results.items():
            row = {
                'experiment': experiment_name,
                'strategy': result.strategy.value,
                'model_type': result.model_type.value
            }
            
            # Add performance metrics
            for metric, value in result.performance_metrics.items():
                row[metric] = value
            
            # Add metadata
            if 'n_regimes' in result.metadata:
                row['n_regimes'] = result.metadata['n_regimes']
            if 'n_features' in result.metadata:
                row['n_features'] = result.metadata['n_features']
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Add rankings based on main metric
        if not df.empty:
            # Find best metric for ranking
            metric_cols = [col for col in df.columns if col not in ['experiment', 'strategy', 'model_type', 'n_regimes', 'n_features']]
            if metric_cols:
                main_metric = metric_cols[0]  # Use first metric
                df['rank'] = df[main_metric].rank(ascending=False)
                df = df.sort_values('rank')
        
        return df
    
    def get_best_strategy(self, 
                         metric: str = 'weighted_f1_score') -> Optional[Tuple[str, TrainingResult]]:
        """
        Get the best training strategy based on a metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (experiment_name, result) for best strategy
        """
        if not self.results:
            return None
        
        best_score = -np.inf
        best_experiment = None
        best_result = None
        
        for experiment_name, result in self.results.items():
            score = result.performance_metrics.get(metric, -np.inf)
            if score > best_score:
                best_score = score
                best_experiment = experiment_name
                best_result = result
        
        return (best_experiment, best_result) if best_experiment else None
    
    def analyze_regime_performance(self, experiment_name: str) -> pd.DataFrame:
        """
        Analyze performance across regimes for a specific experiment.
        
        Args:
            experiment_name: Name of the experiment to analyze
            
        Returns:
            DataFrame with regime-wise performance analysis
        """
        if experiment_name not in self.results:
            return pd.DataFrame()
        
        result = self.results[experiment_name]
        
        if not result.regime_performance:
            return pd.DataFrame()
        
        # Create DataFrame from regime performance
        regime_data = []
        for regime, performance in result.regime_performance.items():
            row = {'regime': regime}
            row.update(performance)
            regime_data.append(row)
        
        df = pd.DataFrame(regime_data)
        
        # Add regime size information if available
        if 'regime_sizes' in result.metadata:
            regime_sizes = result.metadata['regime_sizes']
            df['regime_size'] = df['regime'].map(regime_sizes)
        
        return df.sort_values('regime')
    
    def save_models(self, experiment_name: str, filepath: str):
        """Save trained models to file."""
        if experiment_name not in self.results:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        result = self.results[experiment_name]
        
        # Save models
        with open(filepath, 'wb') as f:
            pickle.dump(result.models, f)
        
        # Save metadata
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        self.logger.info(f"💾 Saved models for {experiment_name} to {filepath}")
    
    def load_models(self, experiment_name: str, filepath: str):
        """Load trained models from file."""
        # Load models
        with open(filepath, 'rb') as f:
            models = pickle.load(f)
        
        # Load metadata
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'r') as f:
            result_dict = json.load(f)
        
        # Reconstruct TrainingResult
        result = TrainingResult(
            model_type=ModelType(result_dict['model_type']),
            strategy=TrainingStrategy(result_dict['strategy']),
            models=models,
            performance_metrics=result_dict['performance_metrics'],
            feature_importance=result_dict['feature_importance'],
            regime_performance=result_dict['regime_performance'],
            metadata=result_dict['metadata']
        )
        
        self.results[experiment_name] = result
        
        self.logger.info(f"📂 Loaded models for {experiment_name} from {filepath}")
    
    def predict(self, 
               experiment_name: str,
               features: pd.DataFrame,
               regime_labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            experiment_name: Name of the experiment to use
            features: Features for prediction
            regime_labels: Regime labels (for regime-specific models)
            
        Returns:
            Predictions
        """
        if experiment_name not in self.results:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        result = self.results[experiment_name]
        
        if result.strategy == TrainingStrategy.REGIME_SPECIFIC:
            return self._predict_regime_specific(result, features, regime_labels)
        else:
            # Multi-regime or ensemble
            main_model = result.models.get('main_model') or result.models.get('ensemble_model')
            if main_model is None:
                raise ValueError("No main model found")
            
            # Prepare features (including regime features if needed)
            prepared_features = self.trainers[result.strategy]._prepare_features(features, regime_labels)
            return main_model.predict(prepared_features.fillna(0))
    
    def _predict_regime_specific(self, 
                               result: TrainingResult,
                               features: pd.DataFrame,
                               regime_labels: np.ndarray) -> np.ndarray:
        """Make predictions using regime-specific models."""
        if regime_labels is None:
            raise ValueError("Regime labels required for regime-specific predictions")
        
        predictions = np.zeros(len(features))
        
        for regime in np.unique(regime_labels):
            regime_mask = regime_labels == regime
            regime_model_key = f'regime_{regime}'
            
            if regime_model_key in result.models:
                regime_features = features[regime_mask]
                prepared_features = self.trainers[TrainingStrategy.REGIME_SPECIFIC]._prepare_features(regime_features)
                regime_predictions = result.models[regime_model_key].predict(prepared_features.fillna(0))
                predictions[regime_mask] = regime_predictions
            else:
                self.logger.warning(f"No model found for regime {regime}")
        
        return predictions
    
    def generate_training_report(self) -> str:
        """Generate a comprehensive training report."""
        if not self.results:
            return "No training results available. Run training first."
        
        report = []
        report.append("# ML Model Training Report")
        report.append("=" * 50)
        report.append("")
        
        # Strategy comparison
        comparison_df = self.compare_strategies()
        if not comparison_df.empty:
            report.append("## Strategy Comparison")
            report.append("")
            
            for _, row in comparison_df.iterrows():
                report.append(f"**{row['experiment'].upper()}** ({row['strategy']})")
                
                # Performance metrics
                metric_cols = [col for col in row.index if col not in ['experiment', 'strategy', 'model_type', 'n_regimes', 'n_features', 'rank']]
                for metric in metric_cols:
                    if not pd.isna(row[metric]):
                        report.append(f"- {metric.replace('_', ' ').title()}: {row[metric]:.3f}")
                
                if 'rank' in row:
                    report.append(f"- Overall Rank: {int(row['rank'])}")
                
                report.append("")
        
        # Best strategy
        best_strategy = self.get_best_strategy()
        if best_strategy:
            best_name, best_result = best_strategy
            report.append("## Recommended Strategy")
            report.append("")
            report.append(f"**{best_name.upper()}** is recommended based on performance metrics.")
            report.append(f"- Strategy: {best_result.strategy.value}")
            report.append(f"- Model Type: {best_result.model_type.value}")
            
            # Top performance metrics
            for metric, value in list(best_result.performance_metrics.items())[:5]:
                report.append(f"- {metric.replace('_', ' ').title()}: {value:.3f}")
            
            report.append("")
        
        # Detailed results
        report.append("## Detailed Results")
        report.append("")
        
        for experiment_name, result in self.results.items():
            report.append(f"### {experiment_name.upper()}")
            report.append(f"**Strategy**: {result.strategy.value}")
            report.append(f"**Model Type**: {result.model_type.value}")
            report.append("")
            
            # Performance metrics
            report.append("**Overall Performance:**")
            for metric, value in result.performance_metrics.items():
                report.append(f"- {metric.replace('_', ' ').title()}: {value:.3f}")
            
            # Regime-specific performance
            if result.regime_performance:
                report.append("")
                report.append("**Regime-Specific Performance:**")
                for regime, performance in result.regime_performance.items():
                    report.append(f"- Regime {regime}:")
                    for metric, value in list(performance.items())[:3]:  # Top 3 metrics
                        report.append(f"  - {metric.replace('_', ' ').title()}: {value:.3f}")
            
            # Feature importance (top 10)
            if result.feature_importance:
                if isinstance(result.feature_importance, dict) and not any(isinstance(v, dict) for v in result.feature_importance.values()):
                    # Single model feature importance
                    sorted_features = sorted(result.feature_importance.items(), key=lambda x: x[1], reverse=True)
                    report.append("")
                    report.append("**Top 10 Important Features:**")
                    for feature, importance in sorted_features[:10]:
                        report.append(f"- {feature}: {importance:.3f}")
            
            report.append("")
        
        return "\n".join(report)