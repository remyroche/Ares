"""
Enhanced ML Integration for Hybrid NAS-TAS Regime System

This module integrates ML utilities from src/utils/ml_common/ for enhanced
machine learning operations, cross-validation, feature selection, and model optimization.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pathlib import Path
import time
from dataclasses import dataclass, field
from enum import Enum

# Import enhanced utility integration
from .enhanced_utility_integration import EnhancedUtilityIntegration, UtilityIntegrationConfig

# Import ML common utilities (conditional imports)
try:
    from src.utils.ml_common.common_operations import MLCommonOperations
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False
    MLCommonOperations = None

try:
    from src.utils.ml_common.confidence_metrics import ConfidenceMetrics
    CONFIDENCE_METRICS_AVAILABLE = True
except ImportError:
    CONFIDENCE_METRICS_AVAILABLE = False
    ConfidenceMetrics = None

try:
    from src.utils.ml_common.feature_selection import FeatureSelector
    FEATURE_SELECTION_AVAILABLE = True
except ImportError:
    FEATURE_SELECTION_AVAILABLE = False
    FeatureSelector = None

try:
    from src.utils.ml_common.matrix_cross_validation import MatrixCrossValidation
    MATRIX_CV_AVAILABLE = True
except ImportError:
    MATRIX_CV_AVAILABLE = False
    MatrixCrossValidation = None

try:
    from src.utils.ml_common.hmm_regime_detection import HMMRegimeDetection
    HMM_REGIME_DETECTION_AVAILABLE = True
except ImportError:
    HMM_REGIME_DETECTION_AVAILABLE = False
    HMMRegimeDetection = None

try:
    from src.utils.ml_common.parallel_processing_optimizer import ParallelProcessingOptimizer
    PARALLEL_PROCESSING_AVAILABLE = True
except ImportError:
    PARALLEL_PROCESSING_AVAILABLE = False
    ParallelProcessingOptimizer = None

try:
    from src.utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager
    VECTORIZATION_MANAGER_AVAILABLE = True
except ImportError:
    VECTORIZATION_MANAGER_AVAILABLE = False
    UnifiedVectorizationManager = None

# Import optimization utilities
try:
    from src.utils.ml_common.optimization.grid_search import GridSearchOptimizer
    GRID_SEARCH_AVAILABLE = True
except ImportError:
    GRID_SEARCH_AVAILABLE = False
    GridSearchOptimizer = None

try:
    from src.utils.ml_common.optimization.bayesian_optimization import BayesianOptimizer
    BAYESIAN_OPTIMIZATION_AVAILABLE = True
except ImportError:
    BAYESIAN_OPTIMIZATION_AVAILABLE = False
    BayesianOptimizer = None

try:
    from src.utils.ml_common.optimization.tpe_optimization import TPEOptimizer
    TPE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    TPE_OPTIMIZATION_AVAILABLE = False
    TPEOptimizer = None

# Import ensemble utilities
try:
    from src.utils.ml_common.ensembles.ensemble_manager import EnsembleManager
    ENSEMBLE_MANAGER_AVAILABLE = True
except ImportError:
    ENSEMBLE_MANAGER_AVAILABLE = False
    EnsembleManager = None

try:
    from src.utils.ml_common.ensembles.model_ensemble import ModelEnsemble
    MODEL_ENSEMBLE_AVAILABLE = True
except ImportError:
    MODEL_ENSEMBLE_AVAILABLE = False
    ModelEnsemble = None

# Import evaluation utilities
try:
    from src.utils.ml_common.evaluation.model_evaluator import ModelEvaluator
    MODEL_EVALUATOR_AVAILABLE = True
except ImportError:
    MODEL_EVALUATOR_AVAILABLE = False
    ModelEvaluator = None

try:
    from src.utils.ml_common.evaluation.performance_metrics import PerformanceMetrics
    PERFORMANCE_METRICS_AVAILABLE = True
except ImportError:
    PERFORMANCE_METRICS_AVAILABLE = False
    PerformanceMetrics = None

# Setup logging
logger = logging.getLogger(__name__)


class MLIntegrationStatus(Enum):
    """Status of ML integration."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass
class MLIntegrationConfig:
    """Configuration for ML integration."""
    # Core ML operations
    enable_ml_common: bool = True
    enable_feature_selection: bool = True
    enable_cross_validation: bool = True
    enable_confidence_metrics: bool = True
    
    # Regime detection
    enable_hmm_regime_detection: bool = True
    enable_regime_analysis: bool = True
    
    # Optimization
    enable_grid_search: bool = True
    enable_bayesian_optimization: bool = True
    enable_tpe_optimization: bool = True
    
    # Ensembles
    enable_ensemble_management: bool = True
    enable_model_ensembles: bool = True
    
    # Evaluation
    enable_model_evaluation: bool = True
    enable_performance_metrics: bool = True
    
    # Performance
    enable_parallel_processing: bool = True
    enable_vectorization: bool = True
    
    # Advanced features
    enable_lookahead_bias_detection: bool = True
    enable_overfitting_detection: bool = True
    enable_data_leakage_detection: bool = True


class EnhancedMLIntegration:
    """
    Enhanced ML integration manager for hybrid NAS-TAS regime system.
    
    This class integrates all available ML utilities from src/utils/ml_common/
    to provide enhanced machine learning operations, optimization, and evaluation.
    """
    
    def __init__(self, config: Optional[MLIntegrationConfig] = None, utility_config: Optional[UtilityIntegrationConfig] = None):
        """Initialize the enhanced ML integration."""
        self.config = config or MLIntegrationConfig()
        self.utility_integration = EnhancedUtilityIntegration(utility_config)
        self.logger = logger.getChild('EnhancedMLIntegration')
        
        # Initialize integration status
        self.integration_status = self._check_integration_status()
        
        # Initialize ML managers
        self._initialize_ml_managers()
        
        self.logger.info("🤖 Enhanced ML Integration initialized")
        self.logger.info(f"📊 Integration Status: {self.integration_status}")
    
    def _check_integration_status(self) -> Dict[str, MLIntegrationStatus]:
        """Check the status of all ML integrations."""
        status = {}
        
        # Check core ML operations
        status['ml_common'] = MLIntegrationStatus.AVAILABLE if ML_COMMON_AVAILABLE else MLIntegrationStatus.UNAVAILABLE
        status['feature_selection'] = MLIntegrationStatus.AVAILABLE if FEATURE_SELECTION_AVAILABLE else MLIntegrationStatus.UNAVAILABLE
        status['cross_validation'] = MLIntegrationStatus.AVAILABLE if MATRIX_CV_AVAILABLE else MLIntegrationStatus.UNAVAILABLE
        status['confidence_metrics'] = MLIntegrationStatus.AVAILABLE if CONFIDENCE_METRICS_AVAILABLE else MLIntegrationStatus.UNAVAILABLE
        
        # Check regime detection
        status['hmm_regime_detection'] = MLIntegrationStatus.AVAILABLE if HMM_REGIME_DETECTION_AVAILABLE else MLIntegrationStatus.UNAVAILABLE
        
        # Check optimization
        status['grid_search'] = MLIntegrationStatus.AVAILABLE if GRID_SEARCH_AVAILABLE else MLIntegrationStatus.UNAVAILABLE
        status['bayesian_optimization'] = MLIntegrationStatus.AVAILABLE if BAYESIAN_OPTIMIZATION_AVAILABLE else MLIntegrationStatus.UNAVAILABLE
        status['tpe_optimization'] = MLIntegrationStatus.AVAILABLE if TPE_OPTIMIZATION_AVAILABLE else MLIntegrationStatus.UNAVAILABLE
        
        # Check ensembles
        status['ensemble_manager'] = MLIntegrationStatus.AVAILABLE if ENSEMBLE_MANAGER_AVAILABLE else MLIntegrationStatus.UNAVAILABLE
        status['model_ensemble'] = MLIntegrationStatus.AVAILABLE if MODEL_ENSEMBLE_AVAILABLE else MLIntegrationStatus.UNAVAILABLE
        
        # Check evaluation
        status['model_evaluator'] = MLIntegrationStatus.AVAILABLE if MODEL_EVALUATOR_AVAILABLE else MLIntegrationStatus.UNAVAILABLE
        status['performance_metrics'] = MLIntegrationStatus.AVAILABLE if PERFORMANCE_METRICS_AVAILABLE else MLIntegrationStatus.UNAVAILABLE
        
        # Check performance
        status['parallel_processing'] = MLIntegrationStatus.AVAILABLE if PARALLEL_PROCESSING_AVAILABLE else MLIntegrationStatus.UNAVAILABLE
        status['vectorization'] = MLIntegrationStatus.AVAILABLE if VECTORIZATION_MANAGER_AVAILABLE else MLIntegrationStatus.UNAVAILABLE
        
        return status
    
    def _initialize_ml_managers(self):
        """Initialize ML managers."""
        # Initialize core ML operations
        if self.config.enable_ml_common and ML_COMMON_AVAILABLE:
            self.ml_common = MLCommonOperations()
        else:
            self.ml_common = None
            
        if self.config.enable_feature_selection and FEATURE_SELECTION_AVAILABLE:
            self.feature_selector = FeatureSelector()
        else:
            self.feature_selector = None
            
        if self.config.enable_cross_validation and MATRIX_CV_AVAILABLE:
            self.matrix_cv = MatrixCrossValidation()
        else:
            self.matrix_cv = None
            
        if self.config.enable_confidence_metrics and CONFIDENCE_METRICS_AVAILABLE:
            self.confidence_metrics = ConfidenceMetrics()
        else:
            self.confidence_metrics = None
        
        # Initialize regime detection
        if self.config.enable_hmm_regime_detection and HMM_REGIME_DETECTION_AVAILABLE:
            self.hmm_regime_detection = HMMRegimeDetection()
        else:
            self.hmm_regime_detection = None
        
        # Initialize optimization
        if self.config.enable_grid_search and GRID_SEARCH_AVAILABLE:
            self.grid_search = GridSearchOptimizer()
        else:
            self.grid_search = None
            
        if self.config.enable_bayesian_optimization and BAYESIAN_OPTIMIZATION_AVAILABLE:
            self.bayesian_optimizer = BayesianOptimizer()
        else:
            self.bayesian_optimizer = None
            
        if self.config.enable_tpe_optimization and TPE_OPTIMIZATION_AVAILABLE:
            self.tpe_optimizer = TPEOptimizer()
        else:
            self.tpe_optimizer = None
        
        # Initialize ensembles
        if self.config.enable_ensemble_management and ENSEMBLE_MANAGER_AVAILABLE:
            self.ensemble_manager = EnsembleManager()
        else:
            self.ensemble_manager = None
            
        if self.config.enable_model_ensembles and MODEL_ENSEMBLE_AVAILABLE:
            self.model_ensemble = ModelEnsemble()
        else:
            self.model_ensemble = None
        
        # Initialize evaluation
        if self.config.enable_model_evaluation and MODEL_EVALUATOR_AVAILABLE:
            self.model_evaluator = ModelEvaluator()
        else:
            self.model_evaluator = None
            
        if self.config.enable_performance_metrics and PERFORMANCE_METRICS_AVAILABLE:
            self.performance_metrics = PerformanceMetrics()
        else:
            self.performance_metrics = None
        
        # Initialize performance
        if self.config.enable_parallel_processing and PARALLEL_PROCESSING_AVAILABLE:
            self.parallel_processor = ParallelProcessingOptimizer()
        else:
            self.parallel_processor = None
            
        if self.config.enable_vectorization and VECTORIZATION_MANAGER_AVAILABLE:
            self.vectorization_manager = UnifiedVectorizationManager()
        else:
            self.vectorization_manager = None
    
    # =============================================================================
    # FEATURE SELECTION AND ENGINEERING
    # =============================================================================
    
    def select_features(self, X: np.ndarray, y: np.ndarray, method: str = "mutual_info", 
                       n_features: int = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Select features using enhanced feature selection utilities."""
        if self.feature_selector:
            try:
                selected_features = self.feature_selector.select_features(
                    X, y, method=method, n_features=n_features, **kwargs
                )
                X_selected = X[:, selected_features]
                self.logger.info(f"✅ Selected {len(selected_features)} features using {method}")
                return X_selected, selected_features
            except Exception as e:
                self.logger.error(f"❌ Error in feature selection: {e}")
                return X, np.arange(X.shape[1])
        else:
            self.logger.warning("⚠️ Feature selector not available")
            return X, np.arange(X.shape[1])
    
    def engineer_features_ml(self, data: pd.DataFrame, target_column: str = None, 
                            feature_types: List[str] = None) -> pd.DataFrame:
        """Engineer features using ML common utilities."""
        if self.ml_common:
            try:
                features = self.ml_common.engineer_features(data, target_column, feature_types)
                self.logger.info(f"✅ Engineered {len(features.columns)} ML features")
                return features
            except Exception as e:
                self.logger.error(f"❌ Error engineering ML features: {e}")
                return data
        else:
            self.logger.warning("⚠️ ML common operations not available")
            return data
    
    # =============================================================================
    # CROSS-VALIDATION AND MODEL EVALUATION
    # =============================================================================
    
    def cross_validate_model(self, estimator, X: np.ndarray, y: np.ndarray, 
                           cv: int = 5, scoring: str = "accuracy", **kwargs) -> Dict[str, Any]:
        """Perform cross-validation using enhanced CV utilities."""
        if self.matrix_cv:
            try:
                cv_results = self.matrix_cv.cross_validate(
                    estimator, X, y, cv=cv, scoring=scoring, **kwargs
                )
                self.logger.info(f"✅ Cross-validation completed with {cv} folds")
                return cv_results
            except Exception as e:
                self.logger.error(f"❌ Error in cross-validation: {e}")
                return {}
        else:
            # Fallback to sklearn
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
            return {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std(),
                'cv': cv,
                'scoring': scoring
            }
    
    def evaluate_model(self, estimator, X_test: np.ndarray, y_test: np.ndarray, 
                      y_pred: np.ndarray = None, y_proba: np.ndarray = None) -> Dict[str, Any]:
        """Evaluate model using enhanced evaluation utilities."""
        if self.model_evaluator:
            try:
                evaluation_results = self.model_evaluator.evaluate(
                    estimator, X_test, y_test, y_pred, y_proba
                )
                self.logger.info("✅ Model evaluation completed")
                return evaluation_results
            except Exception as e:
                self.logger.error(f"❌ Error in model evaluation: {e}")
                return {}
        else:
            # Fallback to basic evaluation
            from sklearn.metrics import accuracy_score, classification_report
            if y_pred is None:
                y_pred = estimator.predict(X_test)
            
            return {
                'accuracy': accuracy_score(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred),
                'predictions': y_pred
            }
    
    def calculate_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    y_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate performance metrics using enhanced metrics utilities."""
        if self.performance_metrics:
            try:
                metrics = self.performance_metrics.calculate_metrics(y_true, y_pred, y_proba)
                self.logger.info("✅ Performance metrics calculated")
                return metrics
            except Exception as e:
                self.logger.error(f"❌ Error calculating performance metrics: {e}")
                return {}
        else:
            # Fallback to basic metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted')
            }
    
    # =============================================================================
    # HYPERPARAMETER OPTIMIZATION
    # =============================================================================
    
    def optimize_hyperparameters(self, estimator, X: np.ndarray, y: np.ndarray, 
                              param_grid: Dict[str, List], method: str = "grid_search",
                              cv: int = 5, scoring: str = "accuracy", **kwargs) -> Dict[str, Any]:
        """Optimize hyperparameters using enhanced optimization utilities."""
        if method == "grid_search" and self.grid_search:
            try:
                optimization_results = self.grid_search.optimize(
                    estimator, X, y, param_grid, cv=cv, scoring=scoring, **kwargs
                )
                self.logger.info("✅ Grid search optimization completed")
                return optimization_results
            except Exception as e:
                self.logger.error(f"❌ Error in grid search optimization: {e}")
                return {}
        
        elif method == "bayesian" and self.bayesian_optimizer:
            try:
                optimization_results = self.bayesian_optimizer.optimize(
                    estimator, X, y, param_grid, cv=cv, scoring=scoring, **kwargs
                )
                self.logger.info("✅ Bayesian optimization completed")
                return optimization_results
            except Exception as e:
                self.logger.error(f"❌ Error in Bayesian optimization: {e}")
                return {}
        
        elif method == "tpe" and self.tpe_optimizer:
            try:
                optimization_results = self.tpe_optimizer.optimize(
                    estimator, X, y, param_grid, cv=cv, scoring=scoring, **kwargs
                )
                self.logger.info("✅ TPE optimization completed")
                return optimization_results
            except Exception as e:
                self.logger.error(f"❌ Error in TPE optimization: {e}")
                return {}
        
        else:
            # Fallback to sklearn GridSearchCV
            from sklearn.model_selection import GridSearchCV
            grid_search = GridSearchCV(estimator, param_grid, cv=cv, scoring=scoring, **kwargs)
            grid_search.fit(X, y)
            return {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'best_estimator': grid_search.best_estimator_,
                'cv_results': grid_search.cv_results_
            }
    
    # =============================================================================
    # REGIME DETECTION
    # =============================================================================
    
    def detect_regimes_hmm(self, data: pd.DataFrame, n_regimes: int = 3, 
                          features: List[str] = None) -> Dict[str, Any]:
        """Detect regimes using HMM-based regime detection."""
        if self.hmm_regime_detection:
            try:
                regime_results = self.hmm_regime_detection.detect_regimes(
                    data, n_regimes=n_regimes, features=features
                )
                self.logger.info(f"✅ HMM regime detection completed with {n_regimes} regimes")
                return regime_results
            except Exception as e:
                self.logger.error(f"❌ Error in HMM regime detection: {e}")
                return {}
        else:
            self.logger.warning("⚠️ HMM regime detection not available")
            return {}
    
    def analyze_regime_transitions(self, regime_sequence: np.ndarray) -> Dict[str, Any]:
        """Analyze regime transitions and patterns."""
        if self.hmm_regime_detection:
            try:
                transition_analysis = self.hmm_regime_detection.analyze_transitions(regime_sequence)
                self.logger.info("✅ Regime transition analysis completed")
                return transition_analysis
            except Exception as e:
                self.logger.error(f"❌ Error in regime transition analysis: {e}")
                return {}
        else:
            # Basic transition analysis
            unique_regimes = np.unique(regime_sequence)
            transition_matrix = np.zeros((len(unique_regimes), len(unique_regimes)))
            
            for i in range(len(regime_sequence) - 1):
                current_regime = regime_sequence[i]
                next_regime = regime_sequence[i + 1]
                current_idx = np.where(unique_regimes == current_regime)[0][0]
                next_idx = np.where(unique_regimes == next_regime)[0][0]
                transition_matrix[current_idx, next_idx] += 1
            
            # Normalize transition matrix
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = transition_matrix / row_sums[:, np.newaxis]
            
            return {
                'transition_matrix': transition_matrix,
                'regime_counts': {regime: np.sum(regime_sequence == regime) for regime in unique_regimes},
                'unique_regimes': unique_regimes
            }
    
    # =============================================================================
    # ENSEMBLE METHODS
    # =============================================================================
    
    def create_ensemble(self, models: List[Any], method: str = "voting", 
                       weights: List[float] = None) -> Any:
        """Create ensemble model using enhanced ensemble utilities."""
        if self.model_ensemble:
            try:
                ensemble = self.model_ensemble.create_ensemble(models, method, weights)
                self.logger.info(f"✅ Ensemble created with {len(models)} models using {method}")
                return ensemble
            except Exception as e:
                self.logger.error(f"❌ Error creating ensemble: {e}")
                return None
        else:
            # Fallback to sklearn VotingClassifier
            from sklearn.ensemble import VotingClassifier
            if method == "voting":
                return VotingClassifier(models, weights=weights)
            else:
                self.logger.warning("⚠️ Ensemble method not supported")
                return None
    
    def manage_ensemble(self, ensemble: Any, X: np.ndarray, y: np.ndarray, 
                       method: str = "bagging") -> Dict[str, Any]:
        """Manage ensemble using enhanced ensemble management utilities."""
        if self.ensemble_manager:
            try:
                management_results = self.ensemble_manager.manage_ensemble(
                    ensemble, X, y, method=method
                )
                self.logger.info(f"✅ Ensemble management completed using {method}")
                return management_results
            except Exception as e:
                self.logger.error(f"❌ Error in ensemble management: {e}")
                return {}
        else:
            self.logger.warning("⚠️ Ensemble manager not available")
            return {}
    
    # =============================================================================
    # CONFIDENCE AND BIAS DETECTION
    # =============================================================================
    
    def calculate_confidence_metrics(self, predictions: np.ndarray, 
                                    probabilities: np.ndarray) -> Dict[str, float]:
        """Calculate confidence metrics using enhanced confidence utilities."""
        if self.confidence_metrics:
            try:
                confidence_metrics = self.confidence_metrics.calculate_metrics(
                    predictions, probabilities
                )
                self.logger.info("✅ Confidence metrics calculated")
                return confidence_metrics
            except Exception as e:
                self.logger.error(f"❌ Error calculating confidence metrics: {e}")
                return {}
        else:
            # Basic confidence calculation
            return {
                'mean_confidence': np.mean(probabilities),
                'std_confidence': np.std(probabilities),
                'min_confidence': np.min(probabilities),
                'max_confidence': np.max(probabilities)
            }
    
    def detect_lookahead_bias(self, X: np.ndarray, y: np.ndarray, 
                            model: Any = None) -> Dict[str, Any]:
        """Detect lookahead bias in the model."""
        if self.ml_common and hasattr(self.ml_common, 'detect_lookahead_bias'):
            try:
                bias_results = self.ml_common.detect_lookahead_bias(X, y, model)
                self.logger.info("✅ Lookahead bias detection completed")
                return bias_results
            except Exception as e:
                self.logger.error(f"❌ Error detecting lookahead bias: {e}")
                return {}
        else:
            self.logger.warning("⚠️ Lookahead bias detection not available")
            return {}
    
    def detect_overfitting(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Detect overfitting in the model."""
        if self.ml_common and hasattr(self.ml_common, 'detect_overfitting'):
            try:
                overfitting_results = self.ml_common.detect_overfitting(
                    model, X_train, y_train, X_val, y_val
                )
                self.logger.info("✅ Overfitting detection completed")
                return overfitting_results
            except Exception as e:
                self.logger.error(f"❌ Error detecting overfitting: {e}")
                return {}
        else:
            # Basic overfitting detection
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            overfitting_score = train_score - val_score
            
            return {
                'train_score': train_score,
                'val_score': val_score,
                'overfitting_score': overfitting_score,
                'is_overfitting': overfitting_score > 0.1
            }
    
    def detect_data_leakage(self, X: np.ndarray, y: np.ndarray, 
                           feature_names: List[str] = None) -> Dict[str, Any]:
        """Detect data leakage in the dataset."""
        if self.ml_common and hasattr(self.ml_common, 'detect_data_leakage'):
            try:
                leakage_results = self.ml_common.detect_data_leakage(X, y, feature_names)
                self.logger.info("✅ Data leakage detection completed")
                return leakage_results
            except Exception as e:
                self.logger.error(f"❌ Error detecting data leakage: {e}")
                return {}
        else:
            self.logger.warning("⚠️ Data leakage detection not available")
            return {}
    
    # =============================================================================
    # PERFORMANCE OPTIMIZATION
    # =============================================================================
    
    def optimize_parallel_processing(self, n_jobs: int = -1, backend: str = "threading") -> Dict[str, Any]:
        """Optimize parallel processing settings."""
        if self.parallel_processor:
            try:
                optimization_results = self.parallel_processor.optimize_processing(
                    n_jobs=n_jobs, backend=backend
                )
                self.logger.info("✅ Parallel processing optimization completed")
                return optimization_results
            except Exception as e:
                self.logger.error(f"❌ Error optimizing parallel processing: {e}")
                return {}
        else:
            self.logger.warning("⚠️ Parallel processor not available")
            return {}
    
    def optimize_vectorization(self, operations: List[str]) -> Dict[str, Any]:
        """Optimize vectorization for operations."""
        if self.vectorization_manager:
            try:
                vectorization_results = self.vectorization_manager.optimize_operations(operations)
                self.logger.info("✅ Vectorization optimization completed")
                return vectorization_results
            except Exception as e:
                self.logger.error(f"❌ Error optimizing vectorization: {e}")
                return {}
        else:
            self.logger.warning("⚠️ Vectorization manager not available")
            return {}
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def get_integration_status(self) -> Dict[str, MLIntegrationStatus]:
        """Get the status of all ML integrations."""
        return self.integration_status
    
    def get_available_ml_utilities(self) -> List[str]:
        """Get list of available ML utilities."""
        available = []
        for utility, status in self.integration_status.items():
            if status == MLIntegrationStatus.AVAILABLE:
                available.append(utility)
        return available
    
    def get_unavailable_ml_utilities(self) -> List[str]:
        """Get list of unavailable ML utilities."""
        unavailable = []
        for utility, status in self.integration_status.items():
            if status == MLIntegrationStatus.UNAVAILABLE:
                unavailable.append(utility)
        return unavailable
    
    def cleanup_ml_resources(self) -> bool:
        """Clean up ML resources."""
        try:
            if self.parallel_processor and hasattr(self.parallel_processor, 'cleanup'):
                self.parallel_processor.cleanup()
            
            if self.vectorization_manager and hasattr(self.vectorization_manager, 'cleanup'):
                self.vectorization_manager.cleanup()
            
            self.logger.info("🧹 ML resources cleaned up successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error during ML cleanup: {e}")
            return False


# Factory function for easy initialization
def create_enhanced_ml_integration(
    config: Optional[MLIntegrationConfig] = None,
    utility_config: Optional[UtilityIntegrationConfig] = None
) -> EnhancedMLIntegration:
    """Create an enhanced ML integration instance."""
    return EnhancedMLIntegration(config, utility_config)


# Convenience functions for common ML operations
def select_features_enhanced(X: np.ndarray, y: np.ndarray, method: str = "mutual_info", 
                           n_features: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Enhanced feature selection."""
    integration = create_enhanced_ml_integration()
    return integration.select_features(X, y, method, n_features)


def cross_validate_enhanced(estimator, X: np.ndarray, y: np.ndarray, 
                          cv: int = 5, scoring: str = "accuracy") -> Dict[str, Any]:
    """Enhanced cross-validation."""
    integration = create_enhanced_ml_integration()
    return integration.cross_validate_model(estimator, X, y, cv, scoring)


def optimize_hyperparameters_enhanced(estimator, X: np.ndarray, y: np.ndarray, 
                                     param_grid: Dict[str, List], method: str = "grid_search") -> Dict[str, Any]:
    """Enhanced hyperparameter optimization."""
    integration = create_enhanced_ml_integration()
    return integration.optimize_hyperparameters(estimator, X, y, param_grid, method)


def detect_regimes_enhanced(data: pd.DataFrame, n_regimes: int = 3, 
                           features: List[str] = None) -> Dict[str, Any]:
    """Enhanced regime detection."""
    integration = create_enhanced_ml_integration()
    return integration.detect_regimes_hmm(data, n_regimes, features)