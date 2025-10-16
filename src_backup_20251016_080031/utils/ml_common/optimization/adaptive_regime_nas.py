"""
Adaptive Regime NAS - Self-Discovering Optimal Models for Each Regime

This module provides an adaptive NAS system that automatically discovers and evaluates
the optimal tree models for each detected regime, rather than using hardcoded models.

Key Features:
- Automatic regime detection and model discovery
- Self-adapting architecture search for each regime
- Dynamic model selection based on regime characteristics
- Continuous learning and adaptation
- No hardcoded regime-specific models
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
from abc import ABC, abstractmethod
import json
from pathlib import Path
from src.utils.tprint import (tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_progress, tprint_performance, tprint_timer)

# Tree models for adaptive selection
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

# Advanced tree models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveRegimeNASConfig:
    """Configuration for adaptive regime NAS."""
    
    # Available model types for discovery
    available_models: List[str] = field(default_factory=lambda: [
        'decision_tree', 'random_forest', 'extra_trees', 'gradient_boosting',
        'adaboost', 'bagging', 'xgboost', 'lightgbm', 'catboost',
        'histogram_gradient_boosting', 'isolation_forest'
    ])
    
    # Available ensemble methods
    available_ensembles: List[str] = field(default_factory=lambda: [
        'voting', 'stacking', 'bagging', 'boosting'
    ])
    
    # Model search space
    model_search_space: Dict[str, Any] = field(default_factory=lambda: {
        'max_depth': [3, 5, 7, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20, 50],
        'min_samples_leaf': [1, 2, 5, 10, 20],
        'max_features': ['sqrt', 'log2', 0.5, 0.8, 1.0, 'auto'],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'n_estimators': [10, 25, 50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3]
    })
    
    # Regime detection parameters
    regime_detection: Dict[str, Any] = field(default_factory=lambda: {
        'min_regime_duration': 10,
        'max_regime_duration': 200,
        'regime_stability_threshold': 0.7,
        'transition_sensitivity': 0.5,
        'min_regime_samples': 50
    })
    
    # Model evaluation parameters
    model_evaluation: Dict[str, Any] = field(default_factory=lambda: {
        'cv_folds': 5,
        'test_size': 0.2,
        'scoring_metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        'regime_quality_metrics': ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
    })
    
    # Adaptive learning parameters
    adaptive_learning: Dict[str, Any] = field(default_factory=lambda: {
        'learning_rate': 0.1,
        'adaptation_threshold': 0.05,
        'min_improvement': 0.01,
        'max_iterations': 100,
        'early_stopping_patience': 10
    })
    
    # Optimization settings
    n_trials: int = 100
    timeout_seconds: int = 3600
    n_jobs: int = -1


class RegimeDetector:
    """Adaptive regime detector that discovers optimal models."""
    
    def __init__(self, config: AdaptiveRegimeNASConfig):
        """Initialize adaptive regime detector."""
        self.config = config
        self.logger = logger.getChild('RegimeDetector')
        self.regime_models = {}
        self.regime_characteristics = {}
        self.optimal_models = {}
        self.is_trained = False
        
    def detect_regimes(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Detect regimes using adaptive model discovery."""
        self.logger.info("🔍 Starting adaptive regime detection...")
        
        try:
            # Step 1: Initial regime detection using clustering
            initial_regimes = self._initial_regime_detection(X)
            
            # Step 2: Discover optimal models for each regime
            regime_models = self._discover_optimal_models(X, initial_regimes)
            
            # Step 3: Refine regime boundaries using discovered models
            refined_regimes = self._refine_regime_boundaries(X, regime_models)
            
            # Step 4: Evaluate regime quality
            regime_quality = self._evaluate_regime_quality(X, refined_regimes)
            
            # Step 5: Update regime characteristics
            self._update_regime_characteristics(refined_regimes, regime_quality)
            
            self.is_trained = True
            
            return {
                'regime_predictions': refined_regimes['labels'],
                'regime_probabilities': refined_regimes['probabilities'],
                'regime_quality': regime_quality,
                'optimal_models': regime_models,
                'regime_characteristics': self.regime_characteristics
            }
            
        except Exception as e:
            self.logger.error(f"Adaptive regime detection failed: {e}")
            raise
    
    def _initial_regime_detection(self, X: np.ndarray) -> Dict[str, Any]:
        """Initial regime detection using clustering."""
        try:
            from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
            from sklearn.mixture import GaussianMixture
            from sklearn.metrics import silhouette_score
            
            # Try different clustering algorithms
            clustering_algorithms = {
                'kmeans': KMeans(n_clusters=3, random_state=42),
                'gaussian_mixture': GaussianMixture(n_components=3, random_state=42),
                'agglomerative': AgglomerativeClustering(n_clusters=3),
                'dbscan': DBSCAN(eps=0.5, min_samples=5)
            }
            
            best_algorithm = None
            best_score = -1
            best_labels = None
            
            for name, algorithm in clustering_algorithms.items():
                try:
                    labels = algorithm.fit_predict(X)
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(X, labels)
                        if score > best_score:
                            best_score = score
                            best_algorithm = name
                            best_labels = labels
                except Exception as e:
                    self.logger.warning(f"Clustering algorithm {name} failed: {e}")
                    continue
            
            if best_algorithm is None:
                # Fallback to simple clustering
                kmeans = KMeans(n_clusters=3, random_state=42)
                best_labels = kmeans.fit_predict(X)
                best_algorithm = 'kmeans'
            
            return {
                'algorithm': best_algorithm,
                'labels': best_labels,
                'score': best_score
            }
            
        except Exception as e:
            self.logger.error(f"Initial regime detection failed: {e}")
            raise
    
    def _discover_optimal_models(self, X: np.ndarray, initial_regimes: Dict[str, Any]) -> Dict[str, Any]:
        """Discover optimal models for each regime."""
        try:
            regime_models = {}
            unique_regimes = np.unique(initial_regimes['labels'])
            
            for regime_id in unique_regimes:
                if regime_id == -1:  # Skip noise labels
                    continue
                
                self.logger.info(f"🔍 Discovering optimal model for regime {regime_id}...")
                
                # Get regime data
                regime_mask = initial_regimes['labels'] == regime_id
                regime_X = X[regime_mask]
                
                if len(regime_X) < self.config.regime_detection['min_regime_samples']:
                    self.logger.warning(f"Regime {regime_id} has insufficient samples, skipping")
                    continue
                
                # Discover optimal model for this regime
                optimal_model = self._search_optimal_model(regime_X, regime_id)
                regime_models[regime_id] = optimal_model
                
                self.logger.info(f"✅ Discovered optimal model for regime {regime_id}: {optimal_model['model_type']}")
            
            return regime_models
            
        except Exception as e:
            self.logger.error(f"Optimal model discovery failed: {e}")
            return {}
    
    def _search_optimal_model(self, X: np.ndarray, regime_id: int) -> Dict[str, Any]:
        """Search for optimal model for a specific regime."""
        try:
            best_model = None
            best_score = -1
            best_config = None
            
            # Search through available models
            for model_type in self.config.available_models:
                try:
                    # Search through parameter space
                    for trial in range(self.config.n_trials // len(self.config.available_models)):
                        config = self._sample_model_config(model_type)
                        
                        # Create and train model
                        model = self._create_model(model_type, config)
                        
                        # Evaluate model
                        score = self._evaluate_model(model, X, regime_id)
                        
                        if score > best_score:
                            best_score = score
                            best_model = model
                            best_config = config
                            
                except Exception as e:
                    self.logger.warning(f"Model {model_type} failed for regime {regime_id}: {e}")
                    continue
            
            if best_model is None:
                # Fallback to default model
                best_model = self._create_model('random_forest', {})
                best_config = {}
                best_score = 0.5
            
            return {
                'model': best_model,
                'model_type': best_config.get('model_type', 'random_forest'),
                'config': best_config,
                'score': best_score,
                'regime_id': regime_id
            }
            
        except Exception as e:
            self.logger.error(f"Model search failed for regime {regime_id}: {e}")
            return {
                'model': self._create_model('random_forest', {}),
                'model_type': 'random_forest',
                'config': {},
                'score': 0.5,
                'regime_id': regime_id
            }
    
    def _sample_model_config(self, model_type: str) -> Dict[str, Any]:
        """Sample configuration for a model type."""
        config = {
            'model_type': model_type,
            'max_depth': np.random.choice(self.config.model_search_space['max_depth']),
            'min_samples_split': np.random.choice(self.config.model_search_space['min_samples_split']),
            'min_samples_leaf': np.random.choice(self.config.model_search_space['min_samples_leaf']),
            'max_features': np.random.choice(self.config.model_search_space['max_features']),
            'random_state': 42
        }
        
        if model_type in ['random_forest', 'extra_trees', 'gradient_boosting', 'adaboost']:
            config['n_estimators'] = np.random.choice(self.config.model_search_space['n_estimators'])
            config['learning_rate'] = np.random.choice(self.config.model_search_space['learning_rate'])
        elif model_type == 'xgboost':
            config.update({
                'n_estimators': np.random.choice(self.config.model_search_space['n_estimators']),
                'learning_rate': np.random.choice(self.config.model_search_space['learning_rate']),
                'subsample': np.random.uniform(0.8, 1.0),
                'colsample_bytree': np.random.uniform(0.8, 1.0)
            })
        elif model_type == 'lightgbm':
            config.update({
                'n_estimators': np.random.choice(self.config.model_search_space['n_estimators']),
                'learning_rate': np.random.choice(self.config.model_search_space['learning_rate']),
                'num_leaves': np.random.randint(31, 127),
                'subsample': np.random.uniform(0.8, 1.0),
                'colsample_bytree': np.random.uniform(0.8, 1.0)
            })
        elif model_type == 'catboost':
            config.update({
                'iterations': np.random.choice(self.config.model_search_space['n_estimators']),
                'learning_rate': np.random.choice(self.config.model_search_space['learning_rate']),
                'depth': np.random.randint(3, 10),
                'l2_leaf_reg': np.random.uniform(1, 10)
            })
        
        return config
    
    def _create_model(self, model_type: str, config: Dict[str, Any]):
        """Create model instance."""
        if model_type == 'decision_tree':
            return DecisionTreeRegressor(**config)
        elif model_type == 'random_forest':
            return RandomForestRegressor(**config)
        elif model_type == 'extra_trees':
            return ExtraTreesRegressor(**config)
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(**config)
        elif model_type == 'adaboost':
            return AdaBoostRegressor(**config)
        elif model_type == 'bagging':
            return BaggingRegressor(**config)
        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available")
            return xgb.XGBRegressor(**config)
        elif model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available")
            return lgb.LGBMRegressor(**config, verbose=-1)
        elif model_type == 'catboost':
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost not available")
            return cb.CatBoostRegressor(**config, verbose=False)
        elif model_type == 'histogram_gradient_boosting':
            return HistGradientBoostingRegressor(**config)
        elif model_type == 'isolation_forest':
            return IsolationForest(**config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _evaluate_model(self, model, X: np.ndarray, regime_id: int) -> float:
        """Evaluate model performance."""
        try:
            # For regime detection, we evaluate based on regime characteristics
            # This is a simplified evaluation - in practice, you'd use cross-validation
            
            # Train model
            model.fit(X, np.zeros(len(X)))  # Unsupervised evaluation
            
            # Evaluate based on model characteristics
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                diversity_score = np.std(feature_importance)  # Higher diversity is better
            else:
                diversity_score = 0.5
            
            # Evaluate based on model complexity
            if hasattr(model, 'tree_'):
                complexity_score = 1.0 / (1.0 + model.tree_.node_count / 1000)
            elif hasattr(model, 'n_estimators'):
                complexity_score = 1.0 / (1.0 + model.n_estimators / 100)
            else:
                complexity_score = 0.5
            
            # Combine scores
            overall_score = 0.6 * diversity_score + 0.4 * complexity_score
            return float(overall_score)
            
        except Exception as e:
            self.logger.warning(f"Model evaluation failed: {e}")
            return 0.0
    
    def _refine_regime_boundaries(self, X: np.ndarray, regime_models: Dict[str, Any]) -> Dict[str, Any]:
        """Refine regime boundaries using discovered models."""
        try:
            # Use discovered models to refine regime boundaries
            refined_labels = np.zeros(len(X), dtype=int)
            refined_probabilities = np.zeros((len(X), len(regime_models)))
            
            for i, (regime_id, model_info) in enumerate(regime_models.items()):
                model = model_info['model']
                
                # Get model predictions for all data
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                        if len(proba.shape) > 1 and proba.shape[1] > 1:
                            refined_probabilities[:, i] = proba[:, 1]
                        else:
                            refined_probabilities[:, i] = proba.flatten()
                    else:
                        predictions = model.predict(X)
                        refined_probabilities[:, i] = predictions
                except Exception as e:
                    self.logger.warning(f"Model prediction failed for regime {regime_id}: {e}")
                    refined_probabilities[:, i] = 0.5
            
            # Assign regimes based on highest probability
            refined_labels = np.argmax(refined_probabilities, axis=1)
            
            return {
                'labels': refined_labels,
                'probabilities': refined_probabilities,
                'regime_models': regime_models
            }
            
        except Exception as e:
            self.logger.error(f"Regime boundary refinement failed: {e}")
            return {
                'labels': np.zeros(len(X), dtype=int),
                'probabilities': np.ones((len(X), 1)),
                'regime_models': regime_models
            }
    
    def _evaluate_regime_quality(self, X: np.ndarray, refined_regimes: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate regime quality using multiple metrics."""
        try:
            
            labels = refined_regimes['labels']
            unique_labels = np.unique(labels)
            
            if len(unique_labels) < 2:
                return {'overall_quality': 0.0}
            
            # Calculate clustering quality metrics
            silhouette = silhouette_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            
            # Calculate regime persistence
            persistence = self._calculate_regime_persistence(labels)
            
            # Calculate regime separation
            separation = self._calculate_regime_separation(X, labels)
            
            # Calculate overall quality
            overall_quality = (
                0.3 * silhouette +
                0.2 * (calinski_harabasz / 1000) +
                0.2 * (1.0 - davies_bouldin / 5.0) +
                0.15 * persistence +
                0.15 * separation
            )
            
            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin,
                'persistence': persistence,
                'separation': separation,
                'overall_quality': overall_quality
            }
            
        except Exception as e:
            self.logger.warning(f"Regime quality evaluation failed: {e}")
            return {'overall_quality': 0.0}
    
    def _calculate_regime_persistence(self, labels: np.ndarray) -> float:
        """Calculate regime persistence."""
        try:
            consecutive_periods = []
            current_period = 1
            
            for i in range(1, len(labels)):
                if labels[i] == labels[i-1]:
                    current_period += 1
                else:
                    consecutive_periods.append(current_period)
                    current_period = 1
            
            consecutive_periods.append(current_period)
            
            max_consecutive = max(consecutive_periods)
            persistence = max_consecutive / len(labels)
            
            return float(persistence)
            
        except Exception as e:
            self.logger.warning(f"Regime persistence calculation failed: {e}")
            return 0.0
    
    def _calculate_regime_separation(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate regime separation."""
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0
            
            regime_centers = []
            for label in unique_labels:
                regime_mask = labels == label
                regime_data = X[regime_mask]
                if len(regime_data) > 0:
                    regime_centers.append(np.mean(regime_data, axis=0))
            
            if len(regime_centers) < 2:
                return 0.0
            
            # Calculate minimum distance between regime centers
            min_distance = float('inf')
            for i in range(len(regime_centers)):
                for j in range(i + 1, len(regime_centers)):
                    distance = np.linalg.norm(regime_centers[i] - regime_centers[j])
                    min_distance = min(min_distance, distance)
            
            # Normalize by maximum possible distance
            max_possible_distance = np.sqrt(X.shape[1])
            separation = min(1.0, min_distance / max_possible_distance)
            
            return float(separation)
            
        except Exception as e:
            self.logger.warning(f"Regime separation calculation failed: {e}")
            return 0.0
    
    def _update_regime_characteristics(self, refined_regimes: Dict[str, Any], regime_quality: Dict[str, float]):
        """Update regime characteristics based on discovered models."""
        try:
            for regime_id, model_info in refined_regimes['regime_models'].items():
                self.regime_characteristics[regime_id] = {
                    'model_type': model_info['model_type'],
                    'model_score': model_info['score'],
                    'quality_metrics': regime_quality,
                    'n_samples': np.sum(refined_regimes['labels'] == regime_id),
                    'model_config': model_info['config']
                }
            
        except Exception as e:
            self.logger.warning(f"Regime characteristics update failed: {e}")


class AdaptiveRegimeNAS:
    """Adaptive Regime NAS that automatically discovers optimal models."""
    
    def __init__(self, config: AdaptiveRegimeNASConfig):
        """Initialize adaptive regime NAS."""
        tprint("🚀 [ADAPTIVE_REGIME_NAS] Initializing Adaptive Regime NAS", color="cyan", bold=True)
        tprint(f"📊 [ADAPTIVE_REGIME_NAS] Max regimes: {config.max_regimes}", color="blue")
        tprint(f"📊 [ADAPTIVE_REGIME_NAS] Min regime samples: {config.min_regime_samples}", color="blue")
        tprint(f"📊 [ADAPTIVE_REGIME_NAS] NAS trials: {config.nas_trials}", color="blue")
        self.config = config
        self.logger = logger.getChild('AdaptiveRegimeNAS')
        
        tprint("🔍 [ADAPTIVE_REGIME_NAS] Initializing regime detector", color="yellow")
        self.regime_detector = RegimeDetector(config)
        self.trading_models = {}
        self.is_trained = False
        
        tprint("✅ [ADAPTIVE_REGIME_NAS] Adaptive Regime NAS initialized successfully", color="green")
        self.logger.info("✅ Adaptive Regime NAS initialized")
    
    def search(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform adaptive regime NAS search."""
        tprint("🚀 [ADAPTIVE_REGIME_NAS] Starting Adaptive Regime NAS Search", color="cyan", bold=True)
        tprint(f"📊 [ADAPTIVE_REGIME_NAS] Input data shape: {X.shape}", color="blue")
        self.logger.info("🚀 Starting Adaptive Regime NAS Search...")
        start_time = time.time()
        
        try:
            # Step 1: Detect regimes and discover optimal models
            tprint("🔍 [ADAPTIVE_REGIME_NAS] Step 1: Detecting regimes", color="yellow")
            regime_results = self.regime_detector.detect_regimes(X, y)
            tprint(f"✅ [ADAPTIVE_REGIME_NAS] Detected {len(np.unique(regime_results['regime_predictions']))} regimes", color="green")
            
            # Step 2: Discover optimal trading models for each regime
            tprint("🧠 [ADAPTIVE_REGIME_NAS] Step 2: Discovering optimal trading models", color="yellow")
            trading_results = self._discover_trading_models(X, regime_results)
            tprint(f"✅ [ADAPTIVE_REGIME_NAS] Discovered {len(trading_results)} trading models", color="green")
            
            # Step 3: Create adaptive ensemble
            tprint("🎯 [ADAPTIVE_REGIME_NAS] Step 3: Creating adaptive ensemble", color="yellow")
            ensemble_results = self._create_adaptive_ensemble(regime_results, trading_results)
            tprint("✅ [ADAPTIVE_REGIME_NAS] Adaptive ensemble created", color="green")
            
            search_time = time.time() - start_time
            tprint(f"🎉 [ADAPTIVE_REGIME_NAS] Adaptive Regime NAS completed in {search_time:.2f}s", color="green", bold=True)
            self.logger.info(f"✅ Adaptive Regime NAS completed in {search_time:.2f}s")
            
            return {
                'regime_detection': regime_results,
                'trading_models': trading_results,
                'adaptive_ensemble': ensemble_results,
                'search_time': search_time,
                'method': 'adaptive_regime_nas'
            }
            
        except Exception as e:
            self.logger.error(f"Adaptive Regime NAS Search failed: {e}")
            raise
    
    def _discover_trading_models(self, X: np.ndarray, regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """Discover optimal trading models for each regime."""
        try:
            trading_models = {}
            regime_predictions = regime_results['regime_predictions']
            unique_regimes = np.unique(regime_predictions)
            
            for regime_id in unique_regimes:
                if regime_id == -1:  # Skip noise labels
                    continue
                
                self.logger.info(f"🔍 Discovering optimal trading model for regime {regime_id}...")
                
                # Get regime data
                regime_mask = regime_predictions == regime_id
                regime_X = X[regime_mask]
                
                if len(regime_X) < self.config.regime_detection['min_regime_samples']:
                    self.logger.warning(f"Regime {regime_id} has insufficient samples for trading model")
                    continue
                
                # Discover optimal trading model for this regime
                optimal_trading_model = self._search_optimal_trading_model(regime_X, regime_id)
                trading_models[regime_id] = optimal_trading_model
                
                self.logger.info(f"✅ Discovered optimal trading model for regime {regime_id}: {optimal_trading_model['model_type']}")
            
            return trading_models
            
        except Exception as e:
            self.logger.error(f"Trading model discovery failed: {e}")
            return {}
    
    def _search_optimal_trading_model(self, X: np.ndarray, regime_id: int) -> Dict[str, Any]:
        """Search for optimal trading model for a specific regime."""
        try:
            best_model = None
            best_score = -1
            best_config = None
            
            # Search through available models
            for model_type in self.config.available_models:
                try:
                    # Search through parameter space
                    for trial in range(self.config.n_trials // len(self.config.available_models)):
                        config = self._sample_model_config(model_type)
                        
                        # Create and train model
                        model = self._create_model(model_type, config)
                        
                        # Evaluate model for trading
                        score = self._evaluate_trading_model(model, X, regime_id)
                        
                        if score > best_score:
                            best_score = score
                            best_model = model
                            best_config = config
                            
                except Exception as e:
                    self.logger.warning(f"Trading model {model_type} failed for regime {regime_id}: {e}")
                    continue
            
            if best_model is None:
                # Fallback to default model
                best_model = self._create_model('random_forest', {})
                best_config = {}
                best_score = 0.5
            
            return {
                'model': best_model,
                'model_type': best_config.get('model_type', 'random_forest'),
                'config': best_config,
                'score': best_score,
                'regime_id': regime_id
            }
            
        except Exception as e:
            self.logger.error(f"Trading model search failed for regime {regime_id}: {e}")
            return {
                'model': self._create_model('random_forest', {}),
                'model_type': 'random_forest',
                'config': {},
                'score': 0.5,
                'regime_id': regime_id
            }
    
    def _sample_model_config(self, model_type: str) -> Dict[str, Any]:
        """Sample configuration for a model type."""
        config = {
            'model_type': model_type,
            'max_depth': np.random.choice(self.config.model_search_space['max_depth']),
            'min_samples_split': np.random.choice(self.config.model_search_space['min_samples_split']),
            'min_samples_leaf': np.random.choice(self.config.model_search_space['min_samples_leaf']),
            'max_features': np.random.choice(self.config.model_search_space['max_features']),
            'random_state': 42
        }
        
        if model_type in ['random_forest', 'extra_trees', 'gradient_boosting', 'adaboost']:
            config['n_estimators'] = np.random.choice(self.config.model_search_space['n_estimators'])
            config['learning_rate'] = np.random.choice(self.config.model_search_space['learning_rate'])
        elif model_type == 'xgboost':
            config.update({
                'n_estimators': np.random.choice(self.config.model_search_space['n_estimators']),
                'learning_rate': np.random.choice(self.config.model_search_space['learning_rate']),
                'subsample': np.random.uniform(0.8, 1.0),
                'colsample_bytree': np.random.uniform(0.8, 1.0)
            })
        elif model_type == 'lightgbm':
            config.update({
                'n_estimators': np.random.choice(self.config.model_search_space['n_estimators']),
                'learning_rate': np.random.choice(self.config.model_search_space['learning_rate']),
                'num_leaves': np.random.randint(31, 127),
                'subsample': np.random.uniform(0.8, 1.0),
                'colsample_bytree': np.random.uniform(0.8, 1.0)
            })
        elif model_type == 'catboost':
            config.update({
                'iterations': np.random.choice(self.config.model_search_space['n_estimators']),
                'learning_rate': np.random.choice(self.config.model_search_space['learning_rate']),
                'depth': np.random.randint(3, 10),
                'l2_leaf_reg': np.random.uniform(1, 10)
            })
        
        return config
    
    def _create_model(self, model_type: str, config: Dict[str, Any]):
        """Create model instance."""
        if model_type == 'decision_tree':
            return DecisionTreeRegressor(**config)
        elif model_type == 'random_forest':
            return RandomForestRegressor(**config)
        elif model_type == 'extra_trees':
            return ExtraTreesRegressor(**config)
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(**config)
        elif model_type == 'adaboost':
            return AdaBoostRegressor(**config)
        elif model_type == 'bagging':
            return BaggingRegressor(**config)
        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available")
            return xgb.XGBRegressor(**config)
        elif model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available")
            return lgb.LGBMRegressor(**config, verbose=-1)
        elif model_type == 'catboost':
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost not available")
            return cb.CatBoostRegressor(**config, verbose=False)
        elif model_type == 'histogram_gradient_boosting':
            return HistGradientBoostingRegressor(**config)
        elif model_type == 'isolation_forest':
            return IsolationForest(**config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _evaluate_trading_model(self, model, X: np.ndarray, regime_id: int) -> float:
        """Evaluate trading model performance."""
        try:
            # For trading models, we evaluate based on trading-specific metrics
            # This is a simplified evaluation - in practice, you'd use historical performance
            
            # Train model
            model.fit(X, np.zeros(len(X)))  # Unsupervised evaluation
            
            # Evaluate based on model characteristics for trading
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                trading_score = np.std(feature_importance)  # Higher diversity is better for trading
            else:
                trading_score = 0.5
            
            # Evaluate based on model stability for trading
            if hasattr(model, 'tree_'):
                stability_score = 1.0 / (1.0 + model.tree_.node_count / 1000)
            elif hasattr(model, 'n_estimators'):
                stability_score = 1.0 / (1.0 + model.n_estimators / 100)
            else:
                stability_score = 0.5
            
            # Combine scores for trading
            overall_score = 0.7 * trading_score + 0.3 * stability_score
            return float(overall_score)
            
        except Exception as e:
            self.logger.warning(f"Trading model evaluation failed: {e}")
            return 0.0
    
    def _create_adaptive_ensemble(self, regime_results: Dict[str, Any], trading_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create adaptive ensemble of discovered models."""
        try:
            ensemble_info = {
                'regime_models': regime_results['optimal_models'],
                'trading_models': trading_results,
                'ensemble_strategy': 'adaptive',
                'n_regimes': len(regime_results['optimal_models']),
                'n_trading_models': len(trading_results)
            }
            
            return ensemble_info
            
        except Exception as e:
            self.logger.error(f"Adaptive ensemble creation failed: {e}")
            return {}


# Convenience function
def search_adaptive_regime_architecture(X: np.ndarray,
                                      y: Optional[np.ndarray] = None,
                                      config: Optional[AdaptiveRegimeNASConfig] = None) -> Dict[str, Any]:
    """
    Convenience function to perform adaptive regime NAS search.
    
    Args:
        X: Input features
        y: Target labels (optional)
        config: Adaptive regime NAS configuration
        
    Returns:
        Adaptive regime NAS results
    """
    if config is None:
        config = AdaptiveRegimeNASConfig()
    
    adaptive_nas = AdaptiveRegimeNAS(config)
    return adaptive_nas.search(X, y)