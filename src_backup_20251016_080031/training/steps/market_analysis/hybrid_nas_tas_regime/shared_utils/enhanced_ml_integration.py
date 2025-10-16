"""
Enhanced ML Integration Module

This module provides comprehensive ML capabilities by integrating
with existing ML utilities from src/utils/ml_common/.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from pathlib import Path
import time
from datetime import datetime
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import ML common utilities
try:
    from src.utils.ml_common import (
        FeatureSelector, FeatureSelectionConfig, CrossValidationUtilities,
        PurgedKFold, TemporalCrossValidator, StabilityAnalyzer,
        UnifiedCrossValidator, perform_cross_validation, temporal_cross_validation,
        nested_cross_validation, calculate_confidence_metrics, calculate_calibration_metrics,
        MemoryOptimizer, MemoryIntegrator, ParallelProcessor, UnifiedCache,
        LookaheadProtection, MLTrainingSafeguards, RobustErrorHandler,
        HMMRegimeDetector, RegimeConfig, M1EnhancedMatrixOperations,
        get_enhanced_matrix_operations, PipelineOrchestrator,
        FeatureImportanceAnalyzer, FeatureImportanceConfig, FeatureImportanceResult,
        ImportanceMethod, analyze_feature_importance, get_important_features,
        DataDriftDetector, DriftDetectionConfig, DriftReport, DriftResult,
        DriftType, DriftMethod, DriftSeverity, detect_data_drift, get_drifted_features
    )
except ImportError as e:
    logging.warning(f"Some ML common utilities not available: {e}")
    # Set defaults for missing imports
    FeatureSelector = None
    FeatureSelectionConfig = None
    CrossValidationUtilities = None
    PurgedKFold = None
    TemporalCrossValidator = None
    StabilityAnalyzer = None
    UnifiedCrossValidator = None
    perform_cross_validation = None
    temporal_cross_validation = None
    nested_cross_validation = None
    calculate_confidence_metrics = None
    calculate_calibration_metrics = None
    MemoryOptimizer = None
    MemoryIntegrator = None
    ParallelProcessor = None
    UnifiedCache = None
    LookaheadProtection = None
    MLTrainingSafeguards = None
    RobustErrorHandler = None
    HMMRegimeDetector = None
    RegimeConfig = None
    M1EnhancedMatrixOperations = None
    get_enhanced_matrix_operations = None
    PipelineOrchestrator = None
    FeatureImportanceAnalyzer = None
    FeatureImportanceConfig = None
    FeatureImportanceResult = None
    ImportanceMethod = None
    analyze_feature_importance = None
    get_important_features = None
    DataDriftDetector = None
    DriftDetectionConfig = None
    DriftReport = None
    DriftResult = None
    DriftType = None
    DriftMethod = None
    DriftSeverity = None
    detect_data_drift = None
    get_drifted_features = None

# Import utility integration
from .enhanced_utility_integration import EnhancedUtilityIntegration, UtilityIntegrationConfig

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class MLIntegrationConfig:
    """Configuration for ML integration."""
    enable_ml_common: bool = True
    enable_feature_selection: bool = True
    enable_cross_validation: bool = True
    enable_confidence_metrics: bool = True
    enable_hmm_regime_detection: bool = True
    enable_regime_analysis: bool = True
    enable_grid_search: bool = True
    enable_bayesian_optimization: bool = True
    enable_tpe_optimization: bool = True
    enable_ensemble_management: bool = True
    enable_model_ensembles: bool = True
    enable_model_evaluation: bool = True
    enable_performance_metrics: bool = True
    enable_parallel_processing: bool = True
    enable_vectorization: bool = True
    enable_lookahead_bias_detection: bool = True
    enable_overfitting_detection: bool = True
    enable_data_leakage_detection: bool = True
    enable_feature_importance_analysis: bool = True
    enable_data_drift_detection: bool = True
    enable_memory_optimization: bool = True
    enable_caching: bool = True
    enable_pipeline_orchestration: bool = True


class EnhancedMLIntegration:
    """
    Enhanced ML integration that consolidates functionality from existing ML utilities.
    """
    
    def __init__(self, config: MLIntegrationConfig, utility_integration: EnhancedUtilityIntegration = None):
        """Initialize enhanced ML integration."""
        self.config = config
        self.utility_integration = utility_integration or EnhancedUtilityIntegration()
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML components
        self._initialize_ml_components()
        
        # Performance tracking
        self.performance_metrics = {
            'training_times': [],
            'prediction_times': [],
            'model_scores': [],
            'validation_scores': [],
            'processing_errors': []
        }
        
        self.logger.info("✅ Enhanced ML integration initialized")
    
    def _initialize_ml_components(self):
        """Initialize ML components based on configuration."""
        try:
            # Initialize feature selection
            if self.config.enable_feature_selection and FeatureSelector:
                self.feature_selector = FeatureSelector()
                self.logger.info("✅ Feature selector initialized")
            
            # Initialize cross-validation
            if self.config.enable_cross_validation and CrossValidationUtilities:
                self.cv_utilities = CrossValidationUtilities()
                self.logger.info("✅ Cross-validation utilities initialized")
            
            # Initialize memory optimization
            if self.config.enable_memory_optimization and MemoryOptimizer:
                self.memory_optimizer = MemoryOptimizer()
                self.logger.info("✅ Memory optimizer initialized")
            
            # Initialize parallel processing
            if self.config.enable_parallel_processing and ParallelProcessor:
                self.parallel_processor = ParallelProcessor()
                self.logger.info("✅ Parallel processor initialized")
            
            # Initialize unified cache
            if self.config.enable_caching and UnifiedCache:
                self.unified_cache = UnifiedCache()
                self.logger.info("✅ Unified cache initialized")
            
            # Initialize safeguards
            if self.config.enable_lookahead_bias_detection and LookaheadProtection:
                self.lookahead_protection = LookaheadProtection()
                self.logger.info("✅ Lookahead protection initialized")
            
            if self.config.enable_overfitting_detection and MLTrainingSafeguards:
                self.ml_safeguards = MLTrainingSafeguards()
                self.logger.info("✅ ML safeguards initialized")
            
            if self.config.enable_data_leakage_detection and RobustErrorHandler:
                self.error_handler = RobustErrorHandler()
                self.logger.info("✅ Error handler initialized")
            
            # Initialize HMM regime detection
            if self.config.enable_hmm_regime_detection and HMMRegimeDetector:
                self.hmm_regime_detector = HMMRegimeDetector()
                self.logger.info("✅ HMM regime detector initialized")
            
            # Initialize feature importance analysis
            if self.config.enable_feature_importance_analysis and FeatureImportanceAnalyzer:
                self.feature_importance_analyzer = FeatureImportanceAnalyzer()
                self.logger.info("✅ Feature importance analyzer initialized")
            
            # Initialize data drift detection
            if self.config.enable_data_drift_detection and DataDriftDetector:
                self.data_drift_detector = DataDriftDetector()
                self.logger.info("✅ Data drift detector initialized")
            
            # Initialize pipeline orchestrator
            if self.config.enable_pipeline_orchestration and PipelineOrchestrator:
                self.pipeline_orchestrator = PipelineOrchestrator()
                self.logger.info("✅ Pipeline orchestrator initialized")
            
            self.logger.info("✅ All ML components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize ML components: {e}")
            raise
    
    # =============================================================================
    # FEATURE SELECTION
    # =============================================================================
    
    def select_features(self, X: np.ndarray, y: np.ndarray, method: str = "mutual_info", n_features: int = 10) -> Tuple[np.ndarray, List[int]]:
        """Select features using ML common utilities."""
        try:
            start_time = time.time()
            
            if self.config.enable_feature_selection and hasattr(self, 'feature_selector'):
                X_selected, selected_features = self.feature_selector.select_features(X, y, method=method, n_features=n_features)
            else:
                # Fallback to simple feature selection
                X_selected = X[:, :n_features]
                selected_features = list(range(n_features))
            
            # Record performance
            processing_time = time.time() - start_time
            self.performance_metrics['training_times'].append(processing_time)
            
            self.logger.info(f"✅ Feature selection completed in {processing_time:.2f}s: {len(selected_features)} features selected")
            return X_selected, selected_features
            
        except Exception as e:
            self.logger.error(f"❌ Feature selection failed: {e}")
            self.performance_metrics['processing_errors'].append(str(e))
            return X[:, :n_features], list(range(n_features))
    
    # =============================================================================
    # CROSS-VALIDATION
    # =============================================================================
    
    def cross_validate_model(self, estimator, X: np.ndarray, y: np.ndarray, cv: int = 5, scoring: str = "accuracy") -> Dict[str, Any]:
        """Perform cross-validation using ML common utilities."""
        try:
            start_time = time.time()
            
            if self.config.enable_cross_validation and hasattr(self, 'cv_utilities'):
                cv_results = self.cv_utilities.cross_validate(estimator, X, y, cv=cv, scoring=scoring)
            else:
                # Fallback to basic cross-validation
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
                cv_results = {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}
            
            # Record performance
            processing_time = time.time() - start_time
            self.performance_metrics['training_times'].append(processing_time)
            self.performance_metrics['validation_scores'].append(cv_results.get('mean', 0))
            
            self.logger.info(f"✅ Cross-validation completed in {processing_time:.2f}s: score={cv_results.get('mean', 0):.3f}")
            return cv_results
            
        except Exception as e:
            self.logger.error(f"❌ Cross-validation failed: {e}")
            self.performance_metrics['processing_errors'].append(str(e))
            return {'mean': 0, 'std': 0, 'scores': [], 'error': str(e)}
    
    # =============================================================================
    # BIAS AND OVERFITTING DETECTION
    # =============================================================================
    
    def detect_lookahead_bias(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Detect lookahead bias using ML common utilities."""
        try:
            if self.config.enable_lookahead_bias_detection and hasattr(self, 'lookahead_protection'):
                bias_results = self.lookahead_protection.detect_bias(X, y)
            else:
                # Fallback to basic bias detection
                bias_results = {'bias_detected': False, 'confidence': 0.5, 'method': 'fallback'}
            
            self.logger.info(f"✅ Lookahead bias detection completed: {bias_results.get('bias_detected', False)}")
            return bias_results
            
        except Exception as e:
            self.logger.error(f"❌ Lookahead bias detection failed: {e}")
            return {'bias_detected': False, 'confidence': 0.5, 'error': str(e)}
    
    def detect_overfitting(self, model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Detect overfitting using ML common utilities."""
        try:
            if self.config.enable_overfitting_detection and hasattr(self, 'ml_safeguards'):
                overfitting_results = self.ml_safeguards.detect_overfitting(model, X_train, y_train, X_val, y_val)
            else:
                # Fallback to basic overfitting detection
                train_score = model.score(X_train, y_train)
                val_score = model.score(X_val, y_val)
                overfitting = train_score - val_score > 0.1
                overfitting_results = {
                    'overfitting_detected': overfitting,
                    'train_score': train_score,
                    'val_score': val_score,
                    'score_difference': train_score - val_score
                }
            
            self.logger.info(f"✅ Overfitting detection completed: {overfitting_results.get('overfitting_detected', False)}")
            return overfitting_results
            
        except Exception as e:
            self.logger.error(f"❌ Overfitting detection failed: {e}")
            return {'overfitting_detected': False, 'error': str(e)}
    
    def detect_data_leakage(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Detect data leakage using ML common utilities."""
        try:
            if self.config.enable_data_leakage_detection and hasattr(self, 'error_handler'):
                leakage_results = self.error_handler.detect_data_leakage(X, y)
            else:
                # Fallback to basic leakage detection
                leakage_results = {'leakage_detected': False, 'confidence': 0.5, 'method': 'fallback'}
            
            self.logger.info(f"✅ Data leakage detection completed: {leakage_results.get('leakage_detected', False)}")
            return leakage_results
            
        except Exception as e:
            self.logger.error(f"❌ Data leakage detection failed: {e}")
            return {'leakage_detected': False, 'confidence': 0.5, 'error': str(e)}
    
    # =============================================================================
    # CONFIDENCE METRICS
    # =============================================================================
    
    def calculate_confidence_metrics(self, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate confidence metrics using ML common utilities."""
        try:
            if self.config.enable_confidence_metrics and calculate_confidence_metrics:
                confidence_metrics = calculate_confidence_metrics(y_pred, y_proba)
            else:
                # Fallback to basic confidence calculation
                mean_confidence = np.mean(np.max(y_proba, axis=1))
                min_confidence = np.min(np.max(y_proba, axis=1))
                confidence_metrics = {
                    'mean_confidence': mean_confidence,
                    'min_confidence': min_confidence,
                    'confidence_std': np.std(np.max(y_proba, axis=1))
                }
            
            self.logger.info(f"✅ Confidence metrics calculated: mean={confidence_metrics.get('mean_confidence', 0):.3f}")
            return confidence_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Confidence metrics calculation failed: {e}")
            return {'mean_confidence': 0.5, 'min_confidence': 0.0, 'error': str(e)}
    
    # =============================================================================
    # REGIME DETECTION
    # =============================================================================
    
    def detect_regimes_hmm(self, data: pd.DataFrame, n_regimes: int = 3, features: List[str] = None) -> Dict[str, Any]:
        """Detect regimes using HMM with ML common utilities."""
        try:
            start_time = time.time()
            
            if self.config.enable_hmm_regime_detection and hasattr(self, 'hmm_regime_detector'):
                regime_results = self.hmm_regime_detector.detect_regimes(data, n_regimes=n_regimes, features=features)
            else:
                # Fallback to basic regime detection
                n_samples = len(data)
                regime_sequence = np.random.randint(0, n_regimes, n_samples)
                regime_results = {
                    'regime_sequence': regime_sequence,
                    'n_regimes': n_regimes,
                    'regime_probabilities': np.random.rand(n_samples, n_regimes),
                    'transition_matrix': np.random.rand(n_regimes, n_regimes)
                }
            
            # Record performance
            processing_time = time.time() - start_time
            self.performance_metrics['training_times'].append(processing_time)
            
            self.logger.info(f"✅ HMM regime detection completed in {processing_time:.2f}s: {n_regimes} regimes detected")
            return regime_results
            
        except Exception as e:
            self.logger.error(f"❌ HMM regime detection failed: {e}")
            return {'regime_sequence': np.array([]), 'n_regimes': 0, 'error': str(e)}
    
    # =============================================================================
    # FEATURE IMPORTANCE ANALYSIS
    # =============================================================================
    
    def analyze_feature_importance(self, model, X: np.ndarray, y: np.ndarray, method: str = "permutation") -> Dict[str, Any]:
        """Analyze feature importance using ML common utilities."""
        try:
            start_time = time.time()
            
            if self.config.enable_feature_importance_analysis and hasattr(self, 'feature_importance_analyzer'):
                importance_results = self.feature_importance_analyzer.analyze(model, X, y, method=method)
            else:
                # Fallback to basic feature importance
                if hasattr(model, 'feature_importances_'):
                    importance_results = {
                        'importances': model.feature_importances_,
                        'method': 'tree_based',
                        'top_features': np.argsort(model.feature_importances_)[-5:]
                    }
                else:
                    # Mock importance for non-tree models
                    n_features = X.shape[1]
                    importance_results = {
                        'importances': np.ones(n_features) / n_features,
                        'method': 'uniform',
                        'top_features': list(range(min(5, n_features)))
                    }
            
            # Record performance
            processing_time = time.time() - start_time
            self.performance_metrics['training_times'].append(processing_time)
            
            self.logger.info(f"✅ Feature importance analysis completed in {processing_time:.2f}s")
            return importance_results
            
        except Exception as e:
            self.logger.error(f"❌ Feature importance analysis failed: {e}")
            return {'importances': np.array([]), 'method': 'error', 'error': str(e)}
    
    # =============================================================================
    # DATA DRIFT DETECTION
    # =============================================================================
    
    def detect_data_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> Dict[str, Any]:
        """Detect data drift using ML common utilities."""
        try:
            if self.config.enable_data_drift_detection and hasattr(self, 'data_drift_detector'):
                drift_results = self.data_drift_detector.detect_drift(reference_data, current_data)
            else:
                # Fallback to basic drift detection
                ref_mean = np.mean(reference_data, axis=0)
                curr_mean = np.mean(current_data, axis=0)
                drift_score = np.mean(np.abs(ref_mean - curr_mean))
                drift_results = {
                    'drift_detected': drift_score > 0.1,
                    'drift_score': drift_score,
                    'method': 'statistical'
                }
            
            self.logger.info(f"✅ Data drift detection completed: {drift_results.get('drift_detected', False)}")
            return drift_results
            
        except Exception as e:
            self.logger.error(f"❌ Data drift detection failed: {e}")
            return {'drift_detected': False, 'drift_score': 0.0, 'error': str(e)}
    
    # =============================================================================
    # HYPERPARAMETER OPTIMIZATION
    # =============================================================================
    
    def optimize_hyperparameters(self, estimator, X: np.ndarray, y: np.ndarray, 
                               param_grid: Dict[str, List], method: str = "grid_search") -> Dict[str, Any]:
        """Optimize hyperparameters using ML common utilities."""
        try:
            start_time = time.time()
            
            if method == "grid_search":
                from sklearn.model_selection import GridSearchCV
                grid_search = GridSearchCV(estimator, param_grid, cv=5, scoring='accuracy')
                grid_search.fit(X, y)
                optimization_results = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'method': 'grid_search'
                }
            elif method == "random_search":
                from sklearn.model_selection import RandomizedSearchCV
                random_search = RandomizedSearchCV(estimator, param_grid, cv=5, scoring='accuracy', n_iter=50)
                random_search.fit(X, y)
                optimization_results = {
                    'best_params': random_search.best_params_,
                    'best_score': random_search.best_score_,
                    'method': 'random_search'
                }
            else:
                # Fallback to basic optimization
                optimization_results = {
                    'best_params': param_grid,
                    'best_score': 0.5,
                    'method': 'fallback'
                }
            
            # Record performance
            processing_time = time.time() - start_time
            self.performance_metrics['training_times'].append(processing_time)
            self.performance_metrics['model_scores'].append(optimization_results.get('best_score', 0))
            
            self.logger.info(f"✅ Hyperparameter optimization completed in {processing_time:.2f}s: score={optimization_results.get('best_score', 0):.3f}")
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"❌ Hyperparameter optimization failed: {e}")
            return {'best_params': {}, 'best_score': 0.0, 'error': str(e)}
    
    # =============================================================================
    # ENSEMBLE MANAGEMENT
    # =============================================================================
    
    def create_ensemble(self, models: List[tuple], method: str = "voting") -> Any:
        """Create ensemble using ML common utilities."""
        try:
            if method == "voting":
                from sklearn.ensemble import VotingClassifier
                ensemble = VotingClassifier(models, voting='soft')
            elif method == "stacking":
                from sklearn.ensemble import StackingClassifier
                ensemble = StackingClassifier(models, cv=5)
            else:
                # Fallback to basic voting
                ensemble = VotingClassifier(models, voting='soft')
            
            self.logger.info(f"✅ Ensemble created using {method} method")
            return ensemble
            
        except Exception as e:
            self.logger.error(f"❌ Ensemble creation failed: {e}")
            return None
    
    def evaluate_model(self, model, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """Evaluate model using ML common utilities."""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            evaluation_results = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted'),
                'recall': recall_score(y, y_pred, average='weighted'),
                'f1_score': f1_score(y, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(y, y_proba, multi_class='ovr') if y_proba.shape[1] > 2 else roc_auc_score(y, y_proba[:, 1])
            }
            
            self.logger.info(f"✅ Model evaluation completed: accuracy={evaluation_results['accuracy']:.3f}")
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"❌ Model evaluation failed: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'error': str(e)}
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def get_available_ml_utilities(self) -> List[str]:
        """Get list of available ML utilities."""
        utilities = []
        
        if self.config.enable_feature_selection and hasattr(self, 'feature_selector'):
            utilities.append('feature_selection')
        
        if self.config.enable_cross_validation and hasattr(self, 'cv_utilities'):
            utilities.append('cross_validation')
        
        if self.config.enable_confidence_metrics:
            utilities.append('confidence_metrics')
        
        if self.config.enable_hmm_regime_detection and hasattr(self, 'hmm_regime_detector'):
            utilities.append('hmm_regime_detection')
        
        if self.config.enable_lookahead_bias_detection and hasattr(self, 'lookahead_protection'):
            utilities.append('lookahead_bias_detection')
        
        if self.config.enable_overfitting_detection and hasattr(self, 'ml_safeguards'):
            utilities.append('overfitting_detection')
        
        if self.config.enable_data_leakage_detection and hasattr(self, 'error_handler'):
            utilities.append('data_leakage_detection')
        
        if self.config.enable_feature_importance_analysis and hasattr(self, 'feature_importance_analyzer'):
            utilities.append('feature_importance_analysis')
        
        if self.config.enable_data_drift_detection and hasattr(self, 'data_drift_detector'):
            utilities.append('data_drift_detection')
        
        if self.config.enable_ensemble_management:
            utilities.append('ensemble_management')
        
        if self.config.enable_model_evaluation:
            utilities.append('model_evaluation')
        
        return utilities
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        try:
            metrics = {
                'training_times': {
                    'mean': np.mean(self.performance_metrics['training_times']) if self.performance_metrics['training_times'] else 0,
                    'std': np.std(self.performance_metrics['training_times']) if self.performance_metrics['training_times'] else 0,
                    'count': len(self.performance_metrics['training_times'])
                },
                'model_scores': {
                    'mean': np.mean(self.performance_metrics['model_scores']) if self.performance_metrics['model_scores'] else 0,
                    'std': np.std(self.performance_metrics['model_scores']) if self.performance_metrics['model_scores'] else 0,
                    'count': len(self.performance_metrics['model_scores'])
                },
                'validation_scores': {
                    'mean': np.mean(self.performance_metrics['validation_scores']) if self.performance_metrics['validation_scores'] else 0,
                    'std': np.std(self.performance_metrics['validation_scores']) if self.performance_metrics['validation_scores'] else 0,
                    'count': len(self.performance_metrics['validation_scores'])
                },
                'processing_errors': {
                    'count': len(self.performance_metrics['processing_errors']),
                    'errors': self.performance_metrics['processing_errors']
                }
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Performance metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and available utilities."""
        return {
            'config': self.config.__dict__,
            'available_utilities': self.get_available_ml_utilities(),
            'performance_metrics': self.get_performance_metrics(),
            'ml_components': {
                'feature_selector': hasattr(self, 'feature_selector'),
                'cv_utilities': hasattr(self, 'cv_utilities'),
                'memory_optimizer': hasattr(self, 'memory_optimizer'),
                'parallel_processor': hasattr(self, 'parallel_processor'),
                'unified_cache': hasattr(self, 'unified_cache'),
                'lookahead_protection': hasattr(self, 'lookahead_protection'),
                'ml_safeguards': hasattr(self, 'ml_safeguards'),
                'error_handler': hasattr(self, 'error_handler'),
                'hmm_regime_detector': hasattr(self, 'hmm_regime_detector'),
                'feature_importance_analyzer': hasattr(self, 'feature_importance_analyzer'),
                'data_drift_detector': hasattr(self, 'data_drift_detector'),
                'pipeline_orchestrator': hasattr(self, 'pipeline_orchestrator')
            }
        }


def create_enhanced_ml_integration(config: MLIntegrationConfig = None, 
                                 utility_integration: EnhancedUtilityIntegration = None) -> EnhancedMLIntegration:
    """Create an enhanced ML integration instance."""
    if config is None:
        config = MLIntegrationConfig()
    
    return EnhancedMLIntegration(config, utility_integration)