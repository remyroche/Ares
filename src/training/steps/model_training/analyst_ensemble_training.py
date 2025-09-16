"""
Analyst Ensemble Training Step

This step handles per-regime ensemble training of Analyst models using common dependencies.
The Analyst Ensemble operates on 5m timeframe and combines individual analyst models
to create robust ensemble predictions for trade decisions.

Enhanced with vectorized training capabilities for improved performance.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
import traceback
from pathlib import Path

from src.utils.logger import system_logger
from src.utils.ml_common.config.base_training_config import EnsembleTrainingConfig
from src.utils.ml_common.training.ensemble_training_step import EnsembleTrainingStep

# Import vectorized training manager
try:
    from src.utils.ml_common.training.vectorized_training_manager import VectorizedTrainingManager
    VECTORIZED_TRAINING_AVAILABLE = True
except ImportError:
    VECTORIZED_TRAINING_AVAILABLE = False

logger = system_logger.getChild('AnalystEnsembleTraining')


class AnalystEnsembleTrainingStep(EnsembleTrainingStep):
    """
    Analyst Ensemble Training Step with per-regime ensemble training, HPO, saving, and metrics.
    
    The Analyst Ensemble operates on 5m timeframe and combines individual analyst models
    to create robust ensemble predictions for trade decisions.
    """
    
    def __init__(self, config: Optional[EnsembleTrainingConfig] = None, enable_vectorization: bool = True):
        """
        Initialize Analyst ensemble training step with vectorization support.

        Args:
            config: Per-regime training configuration
            enable_vectorization: Whether to enable vectorized training
        """
        self.logger = logger.getChild('AnalystEnsembleTrainingStep')
        self.start_time = time.time()
        
        try:
            # Set default configuration for analyst ensemble models
            if config is None:
                config = EnsembleTrainingConfig(
                    model_name="analyst_ensemble_models",
                    timeframe="5m",
                    model_types=["tcn", "catboost", "lightgbm", "ensemble_rf"],
                    hpo_n_trials=100,
                    hpo_timeout_seconds=3600,
                    min_samples_per_regime=1000,
                    enable_data_augmentation=True,
                    augmentation_method="smote",
                    model_save_path="./models/analyst_ensemble_models",
                    evaluation_metrics=["mse", "mae", "r2", "mape", "smape"]
                )
                self.logger.info("📋 Using default configuration for analyst ensemble training")

            # Validate configuration
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
                self.logger.info("🚀 Analyst Ensemble Training Step initialized with vectorization")
            else:
                self.logger.info("✅ Analyst Ensemble Training Step initialized (standard mode)")
                
            self.logger.info(f"📊 Configuration: {len(config.model_types)} ensemble types, {config.timeframe} timeframe")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Analyst Ensemble Training Step: {e}")
            self.logger.error(f"🔍 Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Analyst Ensemble Training Step initialization failed: {e}") from e
    
    def _validate_config(self, config: EnsembleTrainingConfig) -> None:
        """
        Validate configuration parameters to prevent runtime failures.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            # Validate model types
            if not config.model_types or len(config.model_types) == 0:
                raise ValueError("At least one model type must be specified")
            
            # Validate timeframe
            if not config.timeframe or config.timeframe not in ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]:
                self.logger.warning(f"⚠️ Unusual timeframe specified: {config.timeframe}")
            
            # Validate HPO parameters
            if config.enable_hpo:
                if config.hpo_n_trials <= 0:
                    raise ValueError("HPO trials must be positive")
                if config.hpo_timeout_seconds <= 0:
                    raise ValueError("HPO timeout must be positive")
            
            # Validate minimum samples
            if config.min_samples_per_regime <= 0:
                raise ValueError("Minimum samples per regime must be positive")
            
            # Validate save path
            if config.save_models and config.model_save_path:
                save_path = Path(config.model_save_path)
                if not save_path.parent.exists():
                    self.logger.warning(f"⚠️ Save path parent directory does not exist: {save_path.parent}")
            
            self.logger.info("✅ Configuration validation passed")
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            raise ValueError(f"Invalid configuration: {e}") from e
    
    def _validate_input_data(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> None:
        """
        Validate input data to prevent runtime failures.
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels
            
        Raises:
            ValueError: If input data is invalid
        """
        try:
            # Check data shapes
            if X.shape[0] != y.shape[0] or X.shape[0] != regime_labels.shape[0]:
                raise ValueError(f"Data shape mismatch: X={X.shape}, y={y.shape}, regimes={regime_labels.shape}")
            
            # Check for empty data
            if X.shape[0] == 0:
                raise ValueError("Input data is empty")
            
            # Check for NaN values
            if np.isnan(X).any():
                nan_count = np.isnan(X).sum()
                self.logger.warning(f"⚠️ Found {nan_count} NaN values in input features")
            
            if np.isnan(y).any():
                nan_count = np.isnan(y).sum()
                self.logger.warning(f"⚠️ Found {nan_count} NaN values in target values")
            
            # Check for infinite values
            if np.isinf(X).any():
                inf_count = np.isinf(X).sum()
                self.logger.warning(f"⚠️ Found {inf_count} infinite values in input features")
            
            if np.isinf(y).any():
                inf_count = np.isinf(y).sum()
                self.logger.warning(f"⚠️ Found {inf_count} infinite values in target values")
            
            # Check regime distribution
            unique_regimes, regime_counts = np.unique(regime_labels, return_counts=True)
            min_regime_samples = regime_counts.min()
            
            if min_regime_samples < self.config.min_samples_per_regime:
                insufficient_regimes = unique_regimes[regime_counts < self.config.min_samples_per_regime]
                self.logger.warning(f"⚠️ {len(insufficient_regimes)} regimes have insufficient samples (< {self.config.min_samples_per_regime})")
            
            self.logger.info(f"✅ Data validation passed: {X.shape[0]} samples, {X.shape[1]} features, {len(unique_regimes)} regimes")
            
        except Exception as e:
            self.logger.error(f"❌ Data validation failed: {e}")
            raise ValueError(f"Invalid input data: {e}") from e
    
    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        base_analyst_models: Optional[Dict[str, Any]] = None,
        analyst_training_metrics: Optional[Dict[str, Any]] = None,
        hmm_base_models: Optional[Dict[str, Any]] = None,
        hmm_training_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute Analyst ensemble training step with comprehensive error handling and progress tracking.
        
        Args:
            X: Input features (5m timeframe with cross-timeframe features)
            y: Target values (analyst outputs)
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            base_analyst_models: Individual analyst models to ensemble
            analyst_training_metrics: Performance metrics of base models
            hmm_base_models: HMM base models for integration
            hmm_training_metrics: Performance metrics of HMM base models
            
        Returns:
            Dictionary containing training results and metadata
        """
        execution_start_time = time.time()
        self.logger.info("🚀 Starting Analyst ensemble training step")
        
        try:
            # Step 1: Validate inputs
            self.logger.info("🔄 Step 1: Validating inputs...")
            self._validate_input_data(X, y, regime_labels)
            
            # Step 2: Validate and prepare base models
            self.logger.info("🔄 Step 2: Validating base models...")
            if base_analyst_models is None or not base_analyst_models:
                self.logger.warning("⚠️ No base analyst models provided, using mock models")
                base_analyst_models = self._create_mock_base_models()
            else:
                self.logger.info(f"✅ Using {len(base_analyst_models)} provided base models")
            
            # Step 2.5: Integrate HMM base models if available
            if hmm_base_models is not None:
                self.logger.info("🔄 Step 2.5: Integrating HMM base models...")
                X = self._integrate_hmm_base_models(X, hmm_base_models, hmm_training_metrics)
                self.logger.info(f"✅ HMM base models integrated, enhanced features: {X.shape[1]}")
            else:
                self.logger.info("ℹ️ No HMM base models provided, using original features")
            
            # Step 3: Execute training with enhanced error handling
            self.logger.info("🔄 Step 3: Executing ensemble training...")
            results = self._execute_training_with_error_handling(
                X, y, regime_labels, feature_names, hmm_states, base_analyst_models
            )
            
            # Step 4: Add ensemble-specific metadata
            self.logger.info("🔄 Step 4: Adding ensemble-specific metadata...")
            if 'error' not in results:
                results = self._add_ensemble_specific_metadata(results, base_analyst_models, analyst_training_metrics)
            
            # Step 5: Generate comprehensive report
            execution_time = time.time() - execution_start_time
            results = self._generate_comprehensive_report(results, execution_time, base_analyst_models, analyst_training_metrics)
            
            self.logger.info(f"✅ Analyst ensemble training completed successfully in {execution_time:.2f}s")
            return results
            
        except Exception as e:
            execution_time = time.time() - execution_start_time
            error_msg = f"Analyst ensemble training failed after {execution_time:.2f}s: {e}"
            self.logger.error(f"❌ {error_msg}")
            self.logger.error(f"🔍 Traceback: {traceback.format_exc()}")
            
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
        base_analyst_models: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute training with comprehensive error handling and recovery.
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels
            feature_names: Feature names
            hmm_states: HMM states
            base_analyst_models: Base models
            
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
                is_classification=False,  # Analyst ensemble models are typically regression
                base_models=base_analyst_models,
                symbol=None,  # Can be passed as kwargs
                exchange=None,
                timeframe=self.config.timeframe
            )
            
            # Update training stats
            self.training_stats.update({
                'training_completed': True,
                'base_models_used': len(base_analyst_models),
                'feature_count': X.shape[1],
                'sample_count': X.shape[0]
            })
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Training execution failed: {e}")
            self.training_stats.update({
                'training_completed': False,
                'training_error': str(e)
            })
            raise
    
    def _create_mock_base_models(self) -> Dict[str, Any]:
        """
        Create mock base models for testing purposes with enhanced error handling.
        
        Returns:
            Dictionary of mock base models
        """
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import GradientBoostingRegressor
            
            mock_models = {
                'tcn_model': RandomForestRegressor(n_estimators=10, random_state=42, max_depth=5),
                'catboost_model': RandomForestRegressor(n_estimators=10, random_state=43, max_depth=5),
                'lightgbm_model': GradientBoostingRegressor(n_estimators=10, random_state=44, max_depth=3),
                'ensemble_rf_model': RandomForestRegressor(n_estimators=10, random_state=45, max_depth=5)
            }
            
            self.logger.info(f"📊 Created {len(mock_models)} mock base models for ensemble training")
            self.training_stats['mock_models_created'] = len(mock_models)
            return mock_models
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create mock base models: {e}")
            raise RuntimeError(f"Mock model creation failed: {e}") from e
    
    def _integrate_hmm_base_models(
        self, 
        X: np.ndarray, 
        hmm_base_models: Dict[str, Any], 
        hmm_training_metrics: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Integrate HMM base models as additional features for analyst ensemble training.
        
        Args:
            X: Original input features
            hmm_base_models: HMM base models to integrate
            hmm_training_metrics: Performance metrics of HMM base models
            
        Returns:
            Enhanced feature matrix with HMM model predictions
        """
        try:
            enhanced_features = [X]
            hmm_features_count = 0
            
            self.logger.info(f"🔄 Integrating {len(hmm_base_models)} HMM base models...")
            
            for model_name, model_data in hmm_base_models.items():
                try:
                    # Extract model object and generate predictions
                    if isinstance(model_data, dict) and 'model_object' in model_data:
                        model = model_data['model_object']
                    else:
                        model = model_data
                    
                    if model is not None and hasattr(model, 'predict'):
                        # Generate predictions
                        predictions = model.predict(X)
                        
                        # Ensure predictions are 2D
                        if predictions.ndim == 1:
                            predictions = predictions.reshape(-1, 1)
                        
                        # Validate predictions
                        if predictions.shape[0] == X.shape[0] and not np.any(np.isnan(predictions)):
                            enhanced_features.append(predictions)
                            hmm_features_count += predictions.shape[1]
                            self.logger.info(f"✅ Added {predictions.shape[1]} features from HMM model: {model_name}")
                        else:
                            self.logger.warning(f"⚠️ Invalid predictions from HMM model: {model_name}")
                    else:
                        self.logger.warning(f"⚠️ HMM model {model_name} is None or has no predict method")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to integrate HMM model {model_name}: {e}")
                    continue
            
            # Combine all features
            if len(enhanced_features) > 1:
                X_enhanced = np.column_stack(enhanced_features)
                self.logger.info(f"📊 HMM integration complete: {X.shape[1]} original + {hmm_features_count} HMM = {X_enhanced.shape[1]} total features")
                return X_enhanced
            else:
                self.logger.warning("⚠️ No valid HMM models integrated, using original features")
                return X
                
        except Exception as e:
            self.logger.error(f"❌ Failed to integrate HMM base models: {e}")
            self.logger.warning("⚠️ Returning original features due to HMM integration failure")
            return X
    
    def _generate_comprehensive_report(
        self,
        results: Dict[str, Any],
        execution_time: float,
        base_analyst_models: Dict[str, Any],
        analyst_training_metrics: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive training report with detailed statistics and analysis.
        
        Args:
            results: Training results
            execution_time: Total execution time
            base_analyst_models: Base models used
            analyst_training_metrics: Base model metrics
            
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
                'base_model_integration': self._analyze_base_model_integration(base_analyst_models, analyst_training_metrics),
                'recommendations': self._generate_recommendations(results, execution_time)
            }
            
            # Add comprehensive report to results
            results['comprehensive_report'] = comprehensive_report
            
            # Log summary
            self._log_comprehensive_summary(comprehensive_report)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate comprehensive report: {e}")
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
                best_r2 = -np.inf
                best_model = None
                
                for regime, regime_metrics in evaluation_results.items():
                    if isinstance(regime_metrics, dict) and 'r2' in regime_metrics:
                        if regime_metrics['r2'] > best_r2:
                            best_r2 = regime_metrics['r2']
                            best_model = regime
                
                if best_model is not None:
                    performance_analysis['best_performance'] = {
                        'regime': best_model,
                        'r2_score': best_r2
                    }
            
            return performance_analysis
            
        except Exception as e:
            self.logger.warning(f"⚠️ Performance analysis failed: {e}")
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
            self.logger.warning(f"⚠️ Regime analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_base_model_integration(
        self,
        base_analyst_models: Dict[str, Any],
        analyst_training_metrics: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze base model integration.
        
        Args:
            base_analyst_models: Base models used
            analyst_training_metrics: Base model metrics
            
        Returns:
            Base model integration analysis
        """
        try:
            integration_analysis = {
                'base_models_count': len(base_analyst_models) if base_analyst_models else 0,
                'base_model_types': list(base_analyst_models.keys()) if base_analyst_models else [],
                'metrics_available': analyst_training_metrics is not None,
                'integration_quality': 'good' if base_analyst_models and len(base_analyst_models) >= 3 else 'limited'
            }
            
            if analyst_training_metrics:
                integration_analysis['base_model_performance'] = analyst_training_metrics
            
            return integration_analysis
            
        except Exception as e:
            self.logger.warning(f"⚠️ Base model integration analysis failed: {e}")
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
            self.logger.warning(f"⚠️ Recommendation generation failed: {e}")
            return [f"⚠️ Could not generate recommendations: {e}"]
    
    def _log_comprehensive_summary(self, comprehensive_report: Dict[str, Any]) -> None:
        """
        Log comprehensive training summary.
        
        Args:
            comprehensive_report: Comprehensive report data
        """
        try:
            self.logger.info("📊 COMPREHENSIVE TRAINING SUMMARY")
            self.logger.info("=" * 50)
            
            # Execution summary
            exec_summary = comprehensive_report.get('execution_summary', {})
            self.logger.info(f"⏱️ Total execution time: {exec_summary.get('total_execution_time', 0):.2f}s")
            self.logger.info(f"🚀 Vectorization enabled: {exec_summary.get('vectorization_enabled', False)}")
            self.logger.info(f"✅ Training success: {exec_summary.get('success', False)}")
            
            # Data summary
            data_summary = comprehensive_report.get('data_summary', {})
            self.logger.info(f"📊 Samples processed: {data_summary.get('sample_count', 0):,}")
            self.logger.info(f"🔢 Features used: {data_summary.get('feature_count', 0)}")
            self.logger.info(f"🤖 Base models: {data_summary.get('base_models_used', 0)}")
            
            # Performance analysis
            perf_analysis = comprehensive_report.get('performance_analysis', {})
            if perf_analysis.get('best_performance'):
                best_perf = perf_analysis['best_performance']
                self.logger.info(f"🏆 Best performance: R² = {best_perf.get('r2_score', 0):.4f} (Regime {best_perf.get('regime', 'N/A')})")
            
            # Recommendations
            recommendations = comprehensive_report.get('recommendations', [])
            if recommendations:
                self.logger.info("💡 RECOMMENDATIONS:")
                for rec in recommendations:
                    self.logger.info(f"   {rec}")
            
            self.logger.info("=" * 50)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log comprehensive summary: {e}")
    
    def _add_ensemble_specific_metadata(self, results: Dict[str, Any], base_models: Dict[str, Any], base_metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add ensemble-specific metadata to results with enhanced error handling.
        
        Args:
            results: Training results
            base_models: Base analyst models used in ensemble
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
                    self.logger.info("📊 Integrated base model performance metrics")
                
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
                    'average_r2': 0.0,
                    'best_overall_r2': -np.inf
                }
                
                r2_scores = []
                
                for regime, regime_metrics in evaluation_results.items():
                    performance_summary['total_regimes_evaluated'] += 1
                    
                    if isinstance(regime_metrics, dict) and 'error' not in regime_metrics:
                        performance_summary['successful_evaluations'] += 1
                        
                        best_ensemble = None
                        best_r2 = -np.inf
                        
                        for ensemble_name, metrics in regime_metrics.items():
                            if isinstance(metrics, dict) and 'r2' in metrics:
                                r2_scores.append(metrics['r2'])
                                if metrics['r2'] > best_r2:
                                    best_r2 = metrics['r2']
                                    best_ensemble = ensemble_name
                        
                        if best_ensemble:
                            best_ensembles[regime] = {
                                'ensemble': best_ensemble,
                                'r2_score': best_r2,
                                'regime_samples': regime_metrics.get('samples', 0)
                            }
                            
                            if best_r2 > performance_summary['best_overall_r2']:
                                performance_summary['best_overall_r2'] = best_r2
                    else:
                        performance_summary['failed_evaluations'] += 1
                
                # Calculate average performance
                if r2_scores:
                    performance_summary['average_r2'] = np.mean(r2_scores)
                    performance_summary['r2_std'] = np.std(r2_scores)
                    performance_summary['r2_min'] = np.min(r2_scores)
                    performance_summary['r2_max'] = np.max(r2_scores)
                
                results['best_ensembles_per_regime'] = best_ensembles
                results['performance_summary'] = performance_summary
                
                self.logger.info(f"📊 Performance summary: {performance_summary['successful_evaluations']}/{performance_summary['total_regimes_evaluated']} regimes successful")
                if performance_summary['average_r2'] > 0:
                    self.logger.info(f"🏆 Average R²: {performance_summary['average_r2']:.4f}, Best R²: {performance_summary['best_overall_r2']:.4f}")
            
            # Add enhanced ensemble-specific analysis
            ensemble_analysis = {
                'base_timeframe': self.config.timeframe,
                'cross_timeframe_features': True,
                'ensemble_method': 'per_regime',
                'base_models_integrated': len(base_models) if base_models else 0,
                'ensemble_role': 'trade_decision_enhancement',
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
            self.logger.error(f"❌ Failed to add ensemble-specific metadata: {e}")
            results['ensemble_metadata_error'] = str(e)
            return results
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive training statistics.
        
        Returns:
            Dictionary containing training statistics
        """
        return {
            'training_stats': self.training_stats.copy(),
            'configuration': {
                'model_name': self.config.model_name,
                'timeframe': self.config.timeframe,
                'model_types': self.config.model_types,
                'hpo_enabled': self.config.enable_hpo,
                'vectorization_enabled': self.enable_vectorization
            },
            'performance_metrics': getattr(self, 'training_results', {}).get('performance_summary', {}),
            'timestamp': time.time()
        }
    
    def validate_training_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate training results and provide quality assessment.
        
        Args:
            results: Training results to validate
            
        Returns:
            Validation report
        """
        validation_report = {
            'validation_passed': True,
            'issues_found': [],
            'warnings': [],
            'quality_score': 0.0
        }
        
        try:
            # Check for errors
            if 'error' in results:
                validation_report['validation_passed'] = False
                validation_report['issues_found'].append(f"Training failed: {results['error']}")
                return validation_report
            
            # Check for required components
            required_components = ['ensemble_metrics', 'ensemble_analysis']
            for component in required_components:
                if component not in results:
                    validation_report['warnings'].append(f"Missing component: {component}")
            
            # Check performance metrics
            if 'performance_summary' in results:
                perf_summary = results['performance_summary']
                success_rate = perf_summary.get('successful_evaluations', 0) / max(perf_summary.get('total_regimes_evaluated', 1), 1)
                
                if success_rate < 0.5:
                    validation_report['warnings'].append(f"Low success rate: {success_rate:.2%}")
                
                avg_r2 = perf_summary.get('average_r2', 0)
                if avg_r2 < 0.1:
                    validation_report['warnings'].append(f"Low average R²: {avg_r2:.4f}")
                
                # Calculate quality score
                validation_report['quality_score'] = min(1.0, success_rate * (1 + avg_r2) / 2)
            
            # Check data quality
            if 'ensemble_metrics' in results:
                ensemble_metrics = results['ensemble_metrics']
                if ensemble_metrics.get('base_models_count', 0) < 2:
                    validation_report['warnings'].append("Limited base models for ensemble")
            
            self.logger.info(f"✅ Training validation completed - Quality score: {validation_report['quality_score']:.2f}")
            
        except Exception as e:
            validation_report['validation_passed'] = False
            validation_report['issues_found'].append(f"Validation failed: {e}")
            self.logger.error(f"❌ Training validation failed: {e}")
        
        return validation_report


# Convenience functions for backward compatibility
def create_analyst_ensemble_training_step(
    config: Optional[EnsembleTrainingConfig] = None
) -> AnalystEnsembleTrainingStep:
    """Create Analyst ensemble training step."""
    return AnalystEnsembleTrainingStep(config)


def execute_analyst_ensemble_training(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[EnsembleTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None,
    base_analyst_models: Optional[Dict[str, Any]] = None,
    analyst_training_metrics: Optional[Dict[str, Any]] = None,
    hmm_base_models: Optional[Dict[str, Any]] = None,
    hmm_training_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute Analyst ensemble training step."""
    step = create_analyst_ensemble_training_step(config)
    return step.execute(X, y, regime_labels, feature_names, hmm_states, base_analyst_models, analyst_training_metrics, hmm_base_models, hmm_training_metrics)


# Example usage and comparison
if __name__ == "__main__":
    # Example of how to use the ensemble training version
    print("Analyst Ensemble Training Step")
    print("=" * 50)
    
    # Create configuration
    config = EnsembleTrainingConfig(
        model_name="analyst_ensemble_models",
        timeframe="5m",
        model_types=["tcn", "catboost", "lightgbm", "ensemble_rf"],
        hpo_n_trials=50,  # Reduced for demo
        enable_hpo=True,
        save_models=True,
        model_save_path="./models/analyst_ensemble_models_refactored"
    )
    
    # Create training step
    training_step = create_analyst_ensemble_training_step(config)
    
    print(f"✅ Created analyst ensemble training step with {len(config.model_types)} ensemble types")
    print(f"📊 HPO enabled: {config.enable_hpo}")
    print(f"💾 Save models: {config.save_models}")
    print(f"📁 Save path: {config.model_save_path}")
    print(f"⏰ Base timeframe: {config.timeframe}")
    
    # The actual training would be called with:
    # results = training_step.execute(X, y, regime_labels, feature_names, hmm_states, base_analyst_models, analyst_training_metrics)
    
    print("\n🎯 Analyst Ensemble Module Features:")
    print("- Operates on 5m timeframe with cross-timeframe features")
    print("- Combines individual analyst models into robust ensembles")
    print("- Per-regime ensemble training for regime-specific optimization")
    print("- Enhanced trade decision accuracy through model combination")
    print("- Models: TCN (Temporal Convolutional Network), CatBoost, LightGBM, RandomForest")
    print("- Comprehensive context from multi-timeframe dynamics")
    
    print("\n🔄 Integration with Individual Analyst Models:")
    print("- Receives individual analyst model predictions")
    print("- Uses base model performance metrics for weighting")
    print("- Creates regime-specific ensemble combinations")
    print("- Provides enhanced trade decision signals")