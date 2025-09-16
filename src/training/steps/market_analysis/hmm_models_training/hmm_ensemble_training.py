"""
HMM Ensemble Training Component

This component handles per-regime ensemble training of HMM models using common dependencies.
The HMM Ensemble operates on 1h timeframe and combines individual HMM models
to create robust ensemble predictions for market regime detection.

Enhanced with vectorized training capabilities for improved performance.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
import traceback
from pathlib import Path

from src.utils.tprint import tprint
from src.utils.ml_common.config.base_training_config import EnsembleTrainingConfig
from src.utils.ml_common.training.ensemble_training_step import EnsembleTrainingStep

# Shared utilities
from .shared_utilities import (
    TrainingErrorHandler,
    UnifiedModelFactory,
    CircuitBreaker,
    ValidationUtils,
    ProgressReporter,
    MemoryTracker
)
from .shared_utilities.training_error_handler import TrainingMetrics, ModelResult

# Import vectorized training manager
try:
    from src.utils.ml_common.training.vectorized_training_manager import VectorizedTrainingManager
    VECTORIZED_TRAINING_AVAILABLE = True
except ImportError:
    VECTORIZED_TRAINING_AVAILABLE = False

# Using tprint for all logging - no logger needed


class HMMEnsembleTrainingComponent(EnsembleTrainingStep):
    """
    HMM Ensemble Training Component with per-regime ensemble training, HPO, saving, and metrics.
    
    The HMM Ensemble operates on 1h timeframe and combines individual HMM models
    to create robust ensemble predictions for market regime detection.
    """
    
    def __init__(self, config: Optional[EnsembleTrainingConfig] = None, enable_vectorization: bool = True):
        """
        Initialize HMM ensemble training component with vectorization support.

        Args:
            config: Per-regime training configuration
            enable_vectorization: Whether to enable vectorized training
        """
        self.start_time = time.time()
        
        try:
            # Set default configuration for HMM ensemble models
            if config is None:
                config = EnsembleTrainingConfig(
                    model_name="hmm_ensemble_models",
                    timeframe="1h",
                    model_types=["lightgbm", "elastic_net", "xgboost"],
                    hpo_n_trials=100,
                    hpo_timeout_seconds=3600,
                    min_samples_per_regime=1000,
                    enable_data_augmentation=True,
                    augmentation_method="smote",
                    model_save_path="./models/hmm_ensemble_models",
                    evaluation_metrics=["accuracy", "f1_score", "precision", "recall", "auc"]
                )
                tprint("📋 Using default configuration for HMM ensemble training")

            # Validate configuration with fast-fail
            self._validate_config(config)
            
            # Initialize parent class
            super().__init__(config, enable_vectorization=enable_vectorization and VECTORIZED_TRAINING_AVAILABLE)
            
            # Initialize tracking variables
            self.training_stats = {
                'initialization_time': time.time() - self.start_time,
                'vectorization_enabled': self.enable_vectorization,
                'config_used': config.model_name,
                'model_types': config.model_types,
                'timeframe': config.timeframe
            }
            
            # Log initialization success
            if self.enable_vectorization:
                tprint("🚀 HMM Ensemble Training Component initialized with vectorization")
            else:
                tprint("✅ HMM Ensemble Training Component initialized (standard mode)")
                
            tprint(f"📊 Configuration: {len(config.model_types)} ensemble types, {config.timeframe} timeframe")
            
        except Exception as e:
            tprint(f"❌ Failed to initialize HMM Ensemble Training Component: {e}")
            tprint(f"🔍 Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"HMM Ensemble Training Component initialization failed: {e}") from e
    
    def _validate_config(self, config: EnsembleTrainingConfig) -> None:
        """
        Validate configuration parameters with fast-fail for critical issues.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            # Validate model types - FAST FAIL
            if not config.model_types or len(config.model_types) == 0:
                tprint("❌ CRITICAL: No model types specified - FAILING FAST")
                raise ValueError("At least one model type must be specified")
            
            # Validate timeframe - FAST FAIL
            valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            if not config.timeframe or config.timeframe not in valid_timeframes:
                tprint(f"❌ CRITICAL: Invalid timeframe '{config.timeframe}' - FAILING FAST")
                raise ValueError(f"Invalid timeframe '{config.timeframe}' - must be one of: {valid_timeframes}")
            
            # Validate HPO parameters - FAST FAIL
            if config.enable_hpo:
                if config.hpo_n_trials <= 0:
                    tprint("❌ CRITICAL: HPO trials must be positive - FAILING FAST")
                    raise ValueError("HPO trials must be positive")
                if config.hpo_timeout_seconds <= 0:
                    tprint("❌ CRITICAL: HPO timeout must be positive - FAILING FAST")
                    raise ValueError("HPO timeout must be positive")
            
            # Validate minimum samples - FAST FAIL
            if config.min_samples_per_regime <= 0:
                tprint("❌ CRITICAL: Minimum samples per regime must be positive - FAILING FAST")
                raise ValueError("Minimum samples per regime must be positive")
            
            # Validate save path - WARNING ONLY
            if config.save_models and config.model_save_path:
                save_path = Path(config.model_save_path)
                if not save_path.parent.exists():
                    tprint(f"⚠️ WARNING: Save path parent directory does not exist: {save_path.parent}")
            
            tprint("✅ Configuration validation passed")
            
        except Exception as e:
            tprint(f"❌ Configuration validation failed: {e}")
            raise ValueError(f"Invalid configuration: {e}") from e
    
    def _validate_input_data(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> None:
        """
        Validate input data with fast-fail for critical issues.
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels
            
        Raises:
            ValueError: If input data is invalid
        """
        try:
            # Check data shapes - FAST FAIL
            if X.shape[0] != y.shape[0] or X.shape[0] != regime_labels.shape[0]:
                tprint(f"❌ CRITICAL: Data shape mismatch - FAILING FAST")
                tprint(f"   X={X.shape}, y={y.shape}, regimes={regime_labels.shape}")
                raise ValueError(f"Data shape mismatch: X={X.shape}, y={y.shape}, regimes={regime_labels.shape}")
            
            # Check for empty data - FAST FAIL
            if X.shape[0] == 0:
                tprint("❌ CRITICAL: Input data is empty - FAILING FAST")
                raise ValueError("Input data is empty")
            
            # Check for NaN values - FAST FAIL
            if np.isnan(X).any():
                nan_count = np.isnan(X).sum()
                tprint(f"❌ CRITICAL: Found {nan_count} NaN values in input features - FAILING FAST")
                raise ValueError(f"Input data contains {nan_count} NaN values - training cannot proceed")
            
            if np.isnan(y).any():
                nan_count = np.isnan(y).sum()
                tprint(f"❌ CRITICAL: Found {nan_count} NaN values in target values - FAILING FAST")
                raise ValueError(f"Target data contains {nan_count} NaN values - training cannot proceed")
            
            # Check for infinite values - FAST FAIL
            if np.isinf(X).any():
                inf_count = np.isinf(X).sum()
                tprint(f"❌ CRITICAL: Found {inf_count} infinite values in input features - FAILING FAST")
                raise ValueError(f"Input data contains {inf_count} infinite values - training cannot proceed")
            
            if np.isinf(y).any():
                inf_count = np.isinf(y).sum()
                tprint(f"❌ CRITICAL: Found {inf_count} infinite values in target values - FAILING FAST")
                raise ValueError(f"Target data contains {inf_count} infinite values - training cannot proceed")
            
            # Check regime distribution - WARNING ONLY
            unique_regimes, regime_counts = np.unique(regime_labels, return_counts=True)
            min_regime_samples = regime_counts.min()
            
            if min_regime_samples < self.config.min_samples_per_regime:
                insufficient_regimes = unique_regimes[regime_counts < self.config.min_samples_per_regime]
                tprint(f"⚠️ WARNING: {len(insufficient_regimes)} regimes have insufficient samples (< {self.config.min_samples_per_regime})")
            
            tprint(f"✅ Data validation passed: {X.shape[0]} samples, {X.shape[1]} features, {len(unique_regimes)} regimes")
            
        except Exception as e:
            tprint(f"❌ Data validation failed: {e}")
            raise ValueError(f"Invalid input data: {e}") from e
    
    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        base_hmm_models: Optional[Dict[str, Any]] = None,
        hmm_training_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute HMM ensemble training component with comprehensive error handling and progress tracking.
        
        Args:
            X: Input features (1h timeframe with cross-timeframe features)
            y: Target values (HMM regime predictions)
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            base_hmm_models: Individual HMM models to ensemble
            hmm_training_metrics: Performance metrics of base models
            
        Returns:
            Dictionary containing training results and metadata
        """
        execution_start_time = time.time()
        tprint("🚀 Starting HMM ensemble training component")
        
        try:
            # Step 1: Validate inputs
            tprint("🔄 Step 1: Validating inputs...")
            self._validate_input_data(X, y, regime_labels)
            
            # Step 2: Validate and prepare base models
            tprint("🔄 Step 2: Validating base models...")
            if base_hmm_models is None or not base_hmm_models:
                tprint("⚠️ No base HMM models provided, creating proper ensemble models")
                base_hmm_models = self._create_ensemble_models()
            else:
                tprint(f"✅ Using {len(base_hmm_models)} provided base models")
            
            # Step 3: Execute training with enhanced error handling
            tprint("🔄 Step 3: Executing ensemble training...")
            results = self._execute_training_with_error_handling(
                X, y, regime_labels, feature_names, hmm_states, base_hmm_models
            )
            
            # Step 4: Add ensemble-specific metadata
            tprint("🔄 Step 4: Adding ensemble-specific metadata...")
            if 'error' not in results:
                results = self._add_ensemble_specific_metadata(results, base_hmm_models, hmm_training_metrics)
            
            # Step 5: Generate comprehensive report
            execution_time = time.time() - execution_start_time
            results = self._generate_comprehensive_report(results, execution_time, base_hmm_models, hmm_training_metrics)
            
            tprint(f"✅ HMM ensemble training completed successfully in {execution_time:.2f}s")
            return results
            
        except Exception as e:
            execution_time = time.time() - execution_start_time
            error_msg = f"HMM ensemble training failed after {execution_time:.2f}s: {e}"
            tprint(f"❌ {error_msg}")
            tprint(f"🔍 Traceback: {traceback.format_exc()}")
            
            return {
                'error': error_msg,
                'execution_time': execution_time,
                'traceback': traceback.format_exc(),
                'training_stats': self.training_stats
            }
    
    def _execute_training_with_error_handling(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]],
        hmm_states: Optional[np.ndarray],
        base_hmm_models: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute training with comprehensive error handling and recovery.
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels
            feature_names: Feature names
            hmm_states: HMM states
            base_hmm_models: Base models
            
        Returns:
            Training results
        """
        try:
            # Use the parent class execute method with additional ensemble-specific logic
            results = super().execute(
                X=X,
                y=y,
                regime_labels=regime_labels,
                feature_names=feature_names,
                hmm_states=hmm_states,
                is_classification=True,  # HMM ensemble models are classification
                base_models=base_hmm_models,
                symbol=None,  # Can be passed as kwargs
                exchange=None,
                timeframe=self.config.timeframe
            )
            
            # Update training stats
            self.training_stats.update({
                'training_completed': True,
                'base_models_used': len(base_hmm_models),
                'feature_count': X.shape[1],
                'sample_count': X.shape[0]
            })
            
            return results
            
        except Exception as e:
            tprint(f"❌ Training execution failed: {e}")
            self.training_stats.update({
                'training_completed': False,
                'training_error': str(e)
            })
            raise
    
    def _create_ensemble_models(self) -> Dict[str, Any]:
        """
        Create proper ensemble models for HMM training with enhanced error handling.
        
        Returns:
            Dictionary of ensemble models
        """
        try:
            from lightgbm import LGBMRegressor
            from sklearn.linear_model import ElasticNet
            from xgboost import XGBRegressor
            
            ensemble_models = {
                'lightgbm': LGBMRegressor(
                    n_estimators=100,
                    random_state=42,
                    max_depth=6,
                    learning_rate=0.1,
                    verbose=-1
                ),
                'elastic_net': ElasticNet(
                    random_state=43,
                    max_iter=1000,
                    alpha=0.1,
                    l1_ratio=0.5
                ),
                'xgboost': XGBRegressor(
                    n_estimators=100,
                    random_state=44,
                    max_depth=6,
                    learning_rate=0.1,
                    verbosity=0
                )
            }
            
            tprint(f"📊 Created {len(ensemble_models)} ensemble models for HMM training")
            tprint(f"   Models: {list(ensemble_models.keys())}")
            self.training_stats['ensemble_models_created'] = len(ensemble_models)
            return ensemble_models
            
        except ImportError as e:
            tprint(f"❌ CRITICAL: Failed to import required model libraries - FAILING FAST")
            tprint(f"   Error: {e}")
            raise RuntimeError(f"Required model libraries not available: {e}") from e
        except Exception as e:
            tprint(f"❌ Failed to create ensemble models: {e}")
            raise RuntimeError(f"Ensemble model creation failed: {e}") from e
    
    def _generate_comprehensive_report(
        self,
        results: Dict[str, Any],
        execution_time: float,
        base_hmm_models: Dict[str, Any],
        hmm_training_metrics: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive training report with detailed statistics and analysis.
        
        Args:
            results: Training results
            execution_time: Total execution time
            base_hmm_models: Base models used
            hmm_training_metrics: Base model metrics
            
        Returns:
            Enhanced results with comprehensive reporting
        """
        try:
            # Create comprehensive report
            comprehensive_report = {
                'execution_summary': {
                    'total_execution_time': execution_time,
                    'initialization_time': self.training_stats.get('initialization_time', 0),
                    'training_time': execution_time - self.training_stats.get('initialization_time', 0),
                    'vectorization_enabled': self.training_stats.get('vectorization_enabled', False),
                    'success': 'error' not in results
                },
                'data_summary': {
                    'sample_count': self.training_stats.get('sample_count', 0),
                    'feature_count': self.training_stats.get('feature_count', 0),
                    'base_models_used': self.training_stats.get('base_models_used', 0),
                    'mock_models_created': self.training_stats.get('mock_models_created', 0)
                },
                'configuration_summary': {
                    'model_name': self.training_stats.get('config_used', 'unknown'),
                    'timeframe': self.training_stats.get('timeframe', 'unknown'),
                    'model_types': self.training_stats.get('model_types', []),
                    'hpo_enabled': self.config.enable_hpo,
                    'hpo_trials': self.config.hpo_n_trials if self.config.enable_hpo else 0
                },
                'performance_analysis': self._analyze_performance(results),
                'regime_analysis': self._analyze_regime_performance(results),
                'base_model_integration': self._analyze_base_model_integration(base_hmm_models, hmm_training_metrics),
                'recommendations': self._generate_recommendations(results, execution_time)
            }
            
            # Add comprehensive report to results
            results['comprehensive_report'] = comprehensive_report
            
            # Log summary
            self._log_comprehensive_summary(comprehensive_report)
            
            return results
            
        except Exception as e:
            tprint(f"❌ Failed to generate comprehensive report: {e}")
            results['comprehensive_report'] = {'error': f"Report generation failed: {e}"}
            return results
    
    def _analyze_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze overall training performance.
        
        Args:
            results: Training results
            
        Returns:
            Performance analysis
        """
        try:
            performance_analysis = {
                'training_success': 'error' not in results,
                'models_trained': 0,
                'best_performance': {},
                'performance_distribution': {}
            }
            
            if 'evaluation_results' in results:
                evaluation_results = results['evaluation_results']
                performance_analysis['models_trained'] = len(evaluation_results)
                
                # Find best performing model
                best_accuracy = -np.inf
                best_model = None
                
                for regime, regime_metrics in evaluation_results.items():
                    if isinstance(regime_metrics, dict) and 'accuracy' in regime_metrics:
                        if regime_metrics['accuracy'] > best_accuracy:
                            best_accuracy = regime_metrics['accuracy']
                            best_model = regime
                
                if best_model is not None:
                    performance_analysis['best_performance'] = {
                        'regime': best_model,
                        'accuracy': best_accuracy
                    }
            
            return performance_analysis
            
        except Exception as e:
            tprint(f"⚠️ Performance analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_regime_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze regime-specific performance.
        
        Args:
            results: Training results
            
        Returns:
            Regime performance analysis
        """
        try:
            regime_analysis = {
                'total_regimes': 0,
                'successful_regimes': 0,
                'failed_regimes': 0,
                'regime_details': {}
            }
            
            if 'regime_analysis' in results:
                regime_data = results['regime_analysis']
                regime_analysis['total_regimes'] = len(regime_data.get('unique_regimes', []))
                regime_analysis['successful_regimes'] = len(regime_data.get('sufficient_regimes', []))
                regime_analysis['failed_regimes'] = len(regime_data.get('insufficient_regimes', []))
            
            return regime_analysis
            
        except Exception as e:
            tprint(f"⚠️ Regime analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_base_model_integration(
        self,
        base_hmm_models: Dict[str, Any],
        hmm_training_metrics: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze base model integration.
        
        Args:
            base_hmm_models: Base models used
            hmm_training_metrics: Base model metrics
            
        Returns:
            Base model integration analysis
        """
        try:
            integration_analysis = {
                'base_models_count': len(base_hmm_models) if base_hmm_models else 0,
                'base_model_types': list(base_hmm_models.keys()) if base_hmm_models else [],
                'metrics_available': hmm_training_metrics is not None,
                'integration_quality': 'good' if base_hmm_models and len(base_hmm_models) >= 3 else 'limited'
            }
            
            if hmm_training_metrics:
                integration_analysis['base_model_performance'] = hmm_training_metrics
            
            return integration_analysis
            
        except Exception as e:
            tprint(f"⚠️ Base model integration analysis failed: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, results: Dict[str, Any], execution_time: float) -> List[str]:
        """
        Generate recommendations based on training results.
        
        Args:
            results: Training results
            execution_time: Execution time
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        try:
            # Performance-based recommendations
            if 'error' in results:
                recommendations.append("❌ Training failed - review error logs and data quality")
            else:
                recommendations.append("✅ Training completed successfully")
            
            # Time-based recommendations
            if execution_time > 3600:  # More than 1 hour
                recommendations.append("⏰ Consider enabling vectorization for faster training")
            elif execution_time < 60:  # Less than 1 minute
                recommendations.append("⚡ Training completed quickly - consider increasing HPO trials for better performance")
            
            # Data-based recommendations
            if self.training_stats.get('sample_count', 0) < 10000:
                recommendations.append("📊 Consider collecting more training data for better model performance")
            
            # Model-based recommendations
            if self.training_stats.get('base_models_used', 0) < 3:
                recommendations.append("🤖 Consider using more diverse base models for better ensemble performance")
            
            # Vectorization recommendations
            if not self.training_stats.get('vectorization_enabled', False):
                recommendations.append("🚀 Enable vectorization for improved performance on multi-regime training")
            
            return recommendations
            
        except Exception as e:
            tprint(f"⚠️ Recommendation generation failed: {e}")
            return [f"⚠️ Could not generate recommendations: {e}"]
    
    def _log_comprehensive_summary(self, comprehensive_report: Dict[str, Any]) -> None:
        """
        Log comprehensive training summary using tprint.
        
        Args:
            comprehensive_report: Comprehensive report data
        """
        try:
            tprint("📊 COMPREHENSIVE TRAINING SUMMARY")
            tprint("=" * 50)
            
            # Execution summary
            exec_summary = comprehensive_report.get('execution_summary', {})
            tprint(f"⏱️ Total execution time: {exec_summary.get('total_execution_time', 0):.2f}s")
            tprint(f"🚀 Vectorization enabled: {exec_summary.get('vectorization_enabled', False)}")
            tprint(f"✅ Training success: {exec_summary.get('success', False)}")
            
            # Data summary
            data_summary = comprehensive_report.get('data_summary', {})
            tprint(f"📊 Samples processed: {data_summary.get('sample_count', 0):,}")
            tprint(f"🔢 Features used: {data_summary.get('feature_count', 0)}")
            tprint(f"🤖 Base models: {data_summary.get('base_models_used', 0)}")
            
            # Performance analysis
            perf_analysis = comprehensive_report.get('performance_analysis', {})
            if perf_analysis.get('best_performance'):
                best_perf = perf_analysis['best_performance']
                tprint(f"🏆 Best performance: Accuracy = {best_perf.get('accuracy', 0):.4f} (Regime {best_perf.get('regime', 'N/A')})")
            
            # Recommendations
            recommendations = comprehensive_report.get('recommendations', [])
            if recommendations:
                tprint("💡 RECOMMENDATIONS:")
                for rec in recommendations:
                    tprint(f"   {rec}")
            
            tprint("=" * 50)
            
        except Exception as e:
            tprint(f"❌ Failed to log comprehensive summary: {e}")
    
    def _add_ensemble_specific_metadata(self, results: Dict[str, Any], base_models: Dict[str, Any], base_metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add ensemble-specific metadata to results with enhanced error handling.
        
        Args:
            results: Training results
            base_models: Base HMM models used in ensemble
            base_metrics: Performance metrics of base models
            
        Returns:
            Enhanced results with ensemble-specific metadata
        """
        try:
            # Add ensemble-specific analysis
            if 'regime_analysis' in results:
                regime_analysis = results['regime_analysis']
                
                # Calculate ensemble-specific metrics
                ensemble_metrics = {
                    'total_regimes': len(regime_analysis.get('unique_regimes', [])),
                    'sufficient_regimes': len(regime_analysis.get('sufficient_regimes', [])),
                    'insufficient_regimes': len(regime_analysis.get('insufficient_regimes', [])),
                    'regime_balance': regime_analysis.get('regime_balance_train', 0.0),
                    'timeframe': self.config.timeframe,
                    'ensemble_model_types': self.config.model_types,
                    'base_models_count': len(base_models) if base_models else 0,
                    'training_timestamp': time.time(),
                    'vectorization_used': self.training_stats.get('vectorization_enabled', False)
                }
                
                # Add base model performance analysis if available
                if base_metrics:
                    ensemble_metrics['base_model_performance'] = base_metrics
                    tprint("📊 Integrated base model performance metrics")
                
                results['ensemble_metrics'] = ensemble_metrics
            
            # Add ensemble performance summary with enhanced analysis
            if 'evaluation_results' in results:
                evaluation_results = results['evaluation_results']
                
                # Calculate best performing ensemble per regime
                best_ensembles = {}
                performance_summary = {
                    'total_regimes_evaluated': 0,
                    'successful_evaluations': 0,
                    'failed_evaluations': 0,
                    'average_accuracy': 0.0,
                    'best_overall_accuracy': -np.inf
                }
                
                accuracies = []
                
                for regime, regime_metrics in evaluation_results.items():
                    performance_summary['total_regimes_evaluated'] += 1
                    
                    if isinstance(regime_metrics, dict) and 'error' not in regime_metrics:
                        performance_summary['successful_evaluations'] += 1
                        
                        best_ensemble = None
                        best_accuracy = -np.inf
                        
                        for ensemble_name, metrics in regime_metrics.items():
                            if isinstance(metrics, dict) and 'accuracy' in metrics:
                                accuracies.append(metrics['accuracy'])
                                if metrics['accuracy'] > best_accuracy:
                                    best_accuracy = metrics['accuracy']
                                    best_ensemble = ensemble_name
                        
                        if best_ensemble:
                            best_ensembles[regime] = {
                                'ensemble': best_ensemble,
                                'accuracy': best_accuracy,
                                'regime_samples': regime_metrics.get('samples', 0)
                            }
                            
                            if best_accuracy > performance_summary['best_overall_accuracy']:
                                performance_summary['best_overall_accuracy'] = best_accuracy
                    else:
                        performance_summary['failed_evaluations'] += 1
                
                # Calculate average performance
                if accuracies:
                    performance_summary['average_accuracy'] = np.mean(accuracies)
                    performance_summary['accuracy_std'] = np.std(accuracies)
                    performance_summary['accuracy_min'] = np.min(accuracies)
                    performance_summary['accuracy_max'] = np.max(accuracies)
                
                results['best_ensembles_per_regime'] = best_ensembles
                results['performance_summary'] = performance_summary
                
                tprint(f"📊 Performance summary: {performance_summary['successful_evaluations']}/{performance_summary['total_regimes_evaluated']} regimes successful")
                if performance_summary['average_accuracy'] > 0:
                    tprint(f"🏆 Average Accuracy: {performance_summary['average_accuracy']:.4f}, Best Accuracy: {performance_summary['best_overall_accuracy']:.4f}")
            
            # Add enhanced ensemble-specific analysis
            ensemble_analysis = {
                'base_timeframe': self.config.timeframe,
                'cross_timeframe_features': True,
                'ensemble_method': 'per_regime',
                'base_models_integrated': len(base_models) if base_models else 0,
                'ensemble_role': 'market_regime_detection',
                'training_configuration': {
                    'hpo_enabled': self.config.enable_hpo,
                    'hpo_trials': self.config.hpo_n_trials if self.config.enable_hpo else 0,
                    'min_samples_per_regime': self.config.min_samples_per_regime,
                    'evaluation_metrics': self.config.evaluation_metrics
                },
                'data_characteristics': {
                    'total_samples': self.training_stats.get('sample_count', 0),
                    'feature_count': self.training_stats.get('feature_count', 0),
                    'mock_models_used': self.training_stats.get('mock_models_created', 0) > 0
                }
            }
            results['ensemble_analysis'] = ensemble_analysis
            
            return results
            
        except Exception as e:
            tprint(f"❌ Failed to add ensemble-specific metadata: {e}")
            results['ensemble_metadata_error'] = str(e)
            return results


# Convenience functions for backward compatibility
def create_hmm_ensemble_training_component(
    config: Optional[EnsembleTrainingConfig] = None
) -> HMMEnsembleTrainingComponent:
    """Create HMM ensemble training component."""
    return HMMEnsembleTrainingComponent(config)


def execute_hmm_ensemble_training(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[EnsembleTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None,
    base_hmm_models: Optional[Dict[str, Any]] = None,
    hmm_training_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute HMM ensemble training component."""
    component = create_hmm_ensemble_training_component(config)
    return component.execute(X, y, regime_labels, feature_names, hmm_states, base_hmm_models, hmm_training_metrics)


# Example usage and comparison
if __name__ == "__main__":
    # Example of how to use the HMM ensemble training component
    print("HMM Ensemble Training Component")
    print("=" * 50)
    
    # Create configuration
    config = EnsembleTrainingConfig(
        model_name="hmm_ensemble_models",
        timeframe="1h",
        model_types=["lightgbm", "elastic_net", "xgboost"],
        hpo_n_trials=50,  # Reduced for demo
        enable_hpo=True,
        save_models=True,
        model_save_path="./models/hmm_ensemble_models_refactored"
    )
    
    # Create training component
    training_component = create_hmm_ensemble_training_component(config)
    
    print(f"✅ Created HMM ensemble training component with {len(config.model_types)} ensemble types")
    print(f"📊 HPO enabled: {config.enable_hpo}")
    print(f"💾 Save models: {config.save_models}")
    print(f"📁 Save path: {config.model_save_path}")
    print(f"⏰ Base timeframe: {config.timeframe}")
    
    # The actual training would be called with:
    # results = training_component.execute(X, y, regime_labels, feature_names, hmm_states, base_hmm_models, hmm_training_metrics)
    
    print("\n🎯 HMM Ensemble Component Features:")
    print("- Operates on 1h timeframe with cross-timeframe features")
    print("- Combines individual HMM models into robust ensembles")
    print("- Per-regime ensemble training for regime-specific optimization")
    print("- Enhanced market regime detection accuracy through model combination")
    print("- Models: LightGBM, Elastic Net, XGBoost")
    print("- Comprehensive context from multi-timeframe dynamics")
    
    print("\n🔄 Integration with Individual HMM Models:")
    print("- Receives individual HMM model predictions")
    print("- Uses base model performance metrics for weighting")
    print("- Creates regime-specific ensemble combinations")
    print("- Provides enhanced market regime detection signals")