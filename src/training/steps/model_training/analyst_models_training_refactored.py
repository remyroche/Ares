"""
Analyst Models Training Step - Enhanced and Streamlined

This step handles per-regime training of individual Analyst models using common dependencies.
Enhanced Features:
- Comprehensive error handling with detailed failure tracking
- Advanced monitoring and health checks
- Enhanced reporting with performance metrics and resource utilization
- Streamlined code with reduced redundancy
- Silent failure prevention with explicit error propagation
- Real-time training progress tracking
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
from pathlib import Path
import json
import time
import psutil
import traceback
from contextlib import contextmanager

from src.utils.logger import system_logger
from src.utils.ml_common.config import PerRegimeTrainingConfig
from src.utils.ml_common.training import PerRegimeTrainingStep

logger = system_logger.getChild('AnalystModelsTrainingEnhanced')


@contextmanager
def monitor_resources(operation_name: str, logger: logging.Logger):
    """Context manager for monitoring resource usage during operations."""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    start_cpu = psutil.cpu_percent()
    
    logger.info(f"🔄 Starting {operation_name} - Memory: {start_memory:.1f}MB, CPU: {start_cpu:.1f}%")
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        end_cpu = psutil.cpu_percent()
        
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        logger.info(f"✅ Completed {operation_name} - Duration: {duration:.2f}s, "
                   f"Memory: {end_memory:.1f}MB (+{memory_delta:+.1f}MB), CPU: {end_cpu:.1f}%")


class TrainingProgressTracker:
    """Track training progress and provide detailed status updates."""
    
    def __init__(self, total_steps: int, logger: logging.Logger):
        self.total_steps = total_steps
        self.current_step = 0
        self.logger = logger
        self.start_time = time.time()
        self.step_times = []
        
    def update_step(self, step_name: str, details: Optional[Dict] = None):
        """Update progress with step completion."""
        self.current_step += 1
        step_time = time.time()
        self.step_times.append(step_time)
        
        elapsed = step_time - self.start_time
        progress_pct = (self.current_step / self.total_steps) * 100
        
        status_msg = f"📊 Progress: {self.current_step}/{self.total_steps} ({progress_pct:.1f}%) - {step_name}"
        if details:
            status_msg += f" - {details}"
        
        self.logger.info(status_msg)
        
        # Estimate remaining time
        if self.current_step > 1:
            avg_step_time = elapsed / self.current_step
            remaining_steps = self.total_steps - self.current_step
            eta = remaining_steps * avg_step_time
            self.logger.info(f"⏱️ ETA: {eta:.1f}s remaining")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training progress summary."""
        total_time = time.time() - self.start_time
        return {
            'total_steps': self.total_steps,
            'completed_steps': self.current_step,
            'progress_percentage': (self.current_step / self.total_steps) * 100,
            'total_time': total_time,
            'average_step_time': total_time / self.current_step if self.current_step > 0 else 0
        }


class EnhancedErrorHandler:
    """Enhanced error handling with detailed failure tracking."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.error_history = []
        
    def handle_error(self, error: Exception, context: str, 
                    additional_info: Optional[Dict] = None) -> Dict[str, Any]:
        """Handle errors with comprehensive logging and tracking."""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'additional_info': additional_info or {}
        }
        
        self.error_history.append(error_info)
        
        # Log detailed error information
        self.logger.error(f"❌ Error in {context}: {error}")
        self.logger.error(f"🔍 Error Type: {type(error).__name__}")
        if additional_info:
            self.logger.error(f"📋 Additional Info: {additional_info}")
        
        # Log traceback for debugging
        self.logger.debug(f"🔍 Full traceback:\n{traceback.format_exc()}")
        
        return error_info
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered."""
        if not self.error_history:
            return {'total_errors': 0, 'errors': []}
        
        error_types = {}
        for error in self.error_history:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'error_types': error_types,
            'errors': self.error_history
        }


class AnalystModelsTrainingStepRefactored(PerRegimeTrainingStep):
    """
    Analyst Models Training Step with per-regime training, HPO, saving, and metrics.
    
    This is a refactored version that uses common dependencies to reduce code duplication.
    """
    
    def __init__(self, config: Optional[PerRegimeTrainingConfig] = None):
        """
        Initialize Analyst models training step with enhanced error handling and monitoring.
        
        Args:
            config: Per-regime training configuration
        """
        with monitor_resources("Analyst Models Training Initialization", logger):
            # Set default configuration for analyst models
            if config is None:
                config = PerRegimeTrainingConfig(
                    model_name="analyst_models",
                    timeframe="5m",
                    model_types=["TEMPORAL_FUSION_TRANSFORMER", "TABNET", "HIST_GRADIENT_BOOSTING", "EXTRA_TREES"],
                    hpo_n_trials=100,
                    hpo_timeout_seconds=3600,
                    min_samples_per_regime=1000,
                    enable_data_augmentation=True,
                    augmentation_method="smote",
                    model_save_path="./models/analyst_models",
                    evaluation_metrics=["mse", "mae", "r2", "mape", "smape"]
                )
            
            super().__init__(config)
            self.logger = logger.getChild('AnalystModelsTrainingStepRefactored')
            
            # Initialize enhanced components
            self.error_handler = EnhancedErrorHandler(self.logger)
            self.progress_tracker = None  # Will be initialized when training starts
            self.training_metrics = {
                'start_time': None,
                'end_time': None,
                'total_duration': None,
                'memory_usage': [],
                'cpu_usage': [],
                'model_performance': {},
                'regime_statistics': {}
            }
            
            # Validate configuration with detailed error reporting
            validation_result = self._validate_config_enhanced(config)
            if not validation_result['valid']:
                error_msg = f"Invalid configuration: {validation_result['errors']}"
                self.error_handler.handle_error(
                    ValueError(error_msg), 
                    "Configuration Validation",
                    {'config': config.__dict__, 'validation_errors': validation_result['errors']}
                )
                raise ValueError(error_msg)
            
            self.logger.info("✅ Analyst Models Training Step (Enhanced) initialized successfully")
            self.logger.info(f"📋 Configuration: {len(config.model_types)} model types, "
                           f"{config.hpo_n_trials} HPO trials, {config.min_samples_per_regime} min samples/regime")
    
    def _validate_config_enhanced(self, config: PerRegimeTrainingConfig) -> Dict[str, Any]:
        """Enhanced configuration validation with detailed error reporting."""
        errors = []
        warnings = []
        
        try:
            # Required fields validation
            if not config.model_name or not isinstance(config.model_name, str):
                errors.append("model_name must be a non-empty string")
            
            if not config.timeframe or not isinstance(config.timeframe, str):
                errors.append("timeframe must be a non-empty string")
            
            # Model types validation
            if not config.model_types or not isinstance(config.model_types, list):
                errors.append("model_types must be a non-empty list")
            elif len(config.model_types) == 0:
                errors.append("model_types list cannot be empty")
            else:
                # Validate each model type
                valid_model_types = [
                    "TEMPORAL_FUSION_TRANSFORMER", "TABNET", "HIST_GRADIENT_BOOSTING", 
                    "EXTRA_TREES", "TCN", "CatBoostRegressor", "LGBMRegressor", 
                    "RandomForestRegressor", "XGBRegressor", "NODE"
                ]
                invalid_types = [mt for mt in config.model_types if mt not in valid_model_types]
                if invalid_types:
                    warnings.append(f"Unknown model types: {invalid_types}")
            
            # HPO validation
            if config.hpo_n_trials <= 0:
                errors.append("hpo_n_trials must be > 0")
            elif config.hpo_n_trials > 1000:
                warnings.append("hpo_n_trials > 1000 may cause long training times")
            
            if config.hpo_timeout_seconds <= 0:
                errors.append("hpo_timeout_seconds must be > 0")
            
            # Data validation
            if config.min_samples_per_regime < 100:
                warnings.append("min_samples_per_regime < 100 may cause poor model performance")
            
            # Path validation
            if not config.model_save_path:
                errors.append("model_save_path cannot be empty")
            
            # Metrics validation
            if not config.evaluation_metrics or not isinstance(config.evaluation_metrics, list):
                errors.append("evaluation_metrics must be a non-empty list")
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Validation exception: {str(e)}"],
                'warnings': []
            }
    
    def _validate_config(self, config: PerRegimeTrainingConfig) -> bool:
        """Legacy validation method for backward compatibility."""
        result = self._validate_config_enhanced(config)
        return result['valid']
    
    def _generate_datetime_stamp(self) -> str:
        """Generate a consistent datetime stamp for artifacts."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _create_training_report(
        self, 
        results: Dict[str, Any], 
        execution_time: float,
        status: str = "SUCCESS"
    ) -> str:
        """Create a comprehensive training report with enhanced metrics and monitoring data."""
        timestamp = self._generate_datetime_stamp()
        report_filename = f"analyst_models_training_report_{timestamp}.json"
        report_path = f"{self.config.model_save_path}/reports/{report_filename}"
        
        # Ensure reports directory exists
        Path(f"{self.config.model_save_path}/reports").mkdir(parents=True, exist_ok=True)
        
        # Gather system metrics
        system_metrics = self._gather_system_metrics()
        
        # Gather error summary
        error_summary = self.error_handler.get_error_summary()
        
        # Gather progress summary if available
        progress_summary = self.progress_tracker.get_summary() if self.progress_tracker else {}
        
        # Create comprehensive report
        report_data = {
            "metadata": {
                "model_name": self.config.model_name,
                "timeframe": self.config.timeframe,
                "timestamp": timestamp,
                "execution_time_seconds": execution_time,
                "status": status,
                "version": "enhanced_v1.0",
                "config": {
                    "model_types": self.config.model_types,
                    "hpo_n_trials": self.config.hpo_n_trials,
                    "hpo_timeout_seconds": self.config.hpo_timeout_seconds,
                    "min_samples_per_regime": self.config.min_samples_per_regime,
                    "enable_data_augmentation": self.config.enable_data_augmentation,
                    "augmentation_method": self.config.augmentation_method,
                    "evaluation_metrics": self.config.evaluation_metrics
                }
            },
            "results": results,
            "monitoring": {
                "system_metrics": system_metrics,
                "training_metrics": self.training_metrics,
                "progress_summary": progress_summary,
                "error_summary": error_summary
            },
            "summary": {
                "models_trained": len(results.get('models', [])),
                "regimes_processed": len(results.get('regime_analysis', {}).get('unique_regimes', [])),
                "best_performing_model": results.get('best_models_per_regime', {}),
                "training_successful": status == "SUCCESS",
                "total_errors": error_summary.get('total_errors', 0),
                "performance_metrics": self._calculate_performance_metrics(results)
            }
        }
        
        # Save report
        try:
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            self.logger.info(f"📋 Enhanced training report saved: {report_path}")
            self.logger.info(f"📊 Report includes: {len(report_data['monitoring'])} monitoring sections, "
                           f"{error_summary.get('total_errors', 0)} errors tracked")
        except Exception as e:
            self.error_handler.handle_error(e, "Report Saving", {'report_path': report_path})
            report_path = None
        
        return report_path
    
    def _gather_system_metrics(self) -> Dict[str, Any]:
        """Gather comprehensive system metrics."""
        try:
            import psutil
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            return {
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_percent': memory.percent,
                    'swap_total_gb': swap.total / (1024**3),
                    'swap_used_percent': swap.percent
                },
                'cpu': {
                    'usage_percent': cpu_percent,
                    'count': cpu_count,
                    'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                },
                'disk': {
                    'total_gb': disk.total / (1024**3),
                    'used_gb': disk.used / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'usage_percent': (disk.used / disk.total) * 100
                }
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to gather system metrics: {e}")
            return {}
    
    def _calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics from training results."""
        metrics = {
            'model_count': 0,
            'regime_count': 0,
            'best_r2_score': -np.inf,
            'worst_r2_score': np.inf,
            'average_r2_score': 0.0,
            'successful_models': 0,
            'failed_models': 0
        }
        
        try:
            if 'evaluation_results' in results:
                r2_scores = []
                for regime, regime_results in results['evaluation_results'].items():
                    if isinstance(regime_results, dict):
                        metrics['regime_count'] += 1
                        for model_type, model_results in regime_results.items():
                            if isinstance(model_results, dict) and 'metrics' in model_results:
                                metrics['model_count'] += 1
                                if 'r2' in model_results['metrics']:
                                    r2_score = model_results['metrics']['r2']
                                    r2_scores.append(r2_score)
                                    metrics['best_r2_score'] = max(metrics['best_r2_score'], r2_score)
                                    metrics['worst_r2_score'] = min(metrics['worst_r2_score'], r2_score)
                                    metrics['successful_models'] += 1
                                else:
                                    metrics['failed_models'] += 1
                
                if r2_scores:
                    metrics['average_r2_score'] = np.mean(r2_scores)
                    metrics['r2_std'] = np.std(r2_scores)
                    metrics['r2_median'] = np.median(r2_scores)
            
            return metrics
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate performance metrics: {e}")
            return metrics
    
    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Execute Analyst models training step with enhanced error handling, monitoring, and reporting.
        
        Args:
            X: Input features
            y: Target values (analyst outputs)
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            
        Returns:
            Dictionary containing training results and metadata
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If training fails
        """
        # Initialize training metrics and progress tracking
        self.training_metrics['start_time'] = datetime.now()
        start_time = time.time()
        
        # Initialize progress tracker
        total_steps = 6  # Data validation, regime analysis, training, evaluation, saving, reporting
        self.progress_tracker = TrainingProgressTracker(total_steps, self.logger)
        
        with monitor_resources("Analyst Models Training Execution", self.logger):
            self.logger.info("🚀 Starting Enhanced Analyst models training step")
            self.logger.info(f"📊 Input data: {X.shape[0]} samples, {X.shape[1]} features, "
                           f"{len(np.unique(regime_labels))} regimes")
            
            try:
                # Step 1: Enhanced data validation
                self.progress_tracker.update_step("Data Validation", {"samples": X.shape[0], "features": X.shape[1]})
                validation_result = self._validate_input_data_enhanced(X, y, regime_labels)
                if not validation_result['valid']:
                    error_msg = f"Invalid input data: {validation_result['errors']}"
                    self.error_handler.handle_error(
                        ValueError(error_msg), 
                        "Input Data Validation",
                        validation_result
                    )
                    raise ValueError(error_msg)
                
                # Step 2: Regime analysis
                self.progress_tracker.update_step("Regime Analysis", {"unique_regimes": len(np.unique(regime_labels))})
                
                # Step 3: Training execution with enhanced error handling
                self.progress_tracker.update_step("Model Training", {"model_types": len(self.config.model_types)})
                
                # VECTORIZED: Use ultra-fast vectorized training by default
                self.logger.info("🚀 Using VECTORIZED analyst models training")
                training_successful = False
                results = None
                
                try:
                    with monitor_resources("Vectorized Training", self.logger):
                        results = super().execute_vectorized(
                            X=X,
                            y=y,
                            regime_labels=regime_labels,
                            feature_names=feature_names,
                            hmm_states=hmm_states,
                            is_classification=False,  # Analyst models are typically regression
                            symbol=None,  # Can be passed as kwargs
                            exchange=None,
                            timeframe=self.config.timeframe
                        )
                    
                    if results.get('vectorized', False):
                        self.logger.info("✅ VECTORIZED analyst training completed successfully")
                        training_successful = True
                    else:
                        self.logger.warning("⚠️ VECTORIZED analyst training failed, falling back to standard method")
                        raise Exception("Vectorized training returned non-vectorized results")
                        
                except Exception as e:
                    self.error_handler.handle_error(e, "Vectorized Training", {
                        'fallback_reason': str(e),
                        'data_shape': X.shape,
                        'regime_count': len(np.unique(regime_labels))
                    })
                    
                    self.logger.warning(f"⚠️ VECTORIZED analyst training failed: {e}, falling back to standard method")
                    
                    with monitor_resources("Standard Training Fallback", self.logger):
                        results = super().execute(
                            X=X,
                            y=y,
                            regime_labels=regime_labels,
                            feature_names=feature_names,
                            hmm_states=hmm_states,
                            is_classification=False,  # Analyst models are typically regression
                            symbol=None,  # Can be passed as kwargs
                            exchange=None,
                            timeframe=self.config.timeframe
                        )
                    training_successful = True
                
                # Step 4: Post-processing and metadata enhancement
                self.progress_tracker.update_step("Post-processing", {"training_successful": training_successful})
                if 'error' not in results:
                    results = self._add_analyst_specific_metadata(results)
                
                # Step 5: Performance evaluation
                self.progress_tracker.update_step("Performance Evaluation")
                results = self._enhance_results_with_performance_metrics(results)
                
                # Step 6: Report generation
                self.progress_tracker.update_step("Report Generation")
                execution_time = time.time() - start_time
                self.training_metrics['end_time'] = datetime.now()
                self.training_metrics['total_duration'] = execution_time
                
                report_path = self._create_training_report(results, execution_time, "SUCCESS")
                if report_path:
                    results['training_report'] = report_path
                
                # Final success logging
                self.logger.info(f"✅ Enhanced Analyst models training completed in {execution_time:.2f}s")
                self.logger.info(f"📊 Training Summary: {self.progress_tracker.get_summary()}")
                
                return results
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.training_metrics['end_time'] = datetime.now()
                self.training_metrics['total_duration'] = execution_time
                
                error_msg = f"Analyst models training failed: {e}"
                self.error_handler.handle_error(e, "Training Execution", {
                    'execution_time': execution_time,
                    'progress': self.progress_tracker.get_summary() if self.progress_tracker else {}
                })
                
                # Create comprehensive failure report
                failure_results = {
                    'error': error_msg, 
                    'execution_time': execution_time,
                    'error_summary': self.error_handler.get_error_summary(),
                    'progress_summary': self.progress_tracker.get_summary() if self.progress_tracker else {}
                }
                self._create_training_report(failure_results, execution_time, "FAILED")
                
                # Fast-fail: Re-raise the exception with enhanced context
                raise RuntimeError(error_msg) from e
    
    def _validate_input_data_enhanced(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Enhanced input data validation with detailed error reporting."""
        errors = []
        warnings = []
        
        try:
            # Basic null checks
            if X is None or y is None or regime_labels is None:
                errors.append("Input data cannot be None")
                return {'valid': False, 'errors': errors, 'warnings': warnings}
            
            # Shape validation
            if len(X) != len(y) or len(X) != len(regime_labels):
                errors.append(f"Input data length mismatch: X={len(X)}, y={len(y)}, regime_labels={len(regime_labels)}")
                return {'valid': False, 'errors': errors, 'warnings': warnings}
            
            if len(X) == 0:
                errors.append("Input data is empty")
                return {'valid': False, 'errors': errors, 'warnings': warnings}
            
            # Data quality checks
            nan_count_X = np.isnan(X).sum()
            inf_count_X = np.isinf(X).sum()
            nan_count_y = np.isnan(y).sum()
            inf_count_y = np.isinf(y).sum()
            
            if nan_count_X > 0:
                errors.append(f"Input features contain {nan_count_X} NaN values")
            
            if inf_count_X > 0:
                errors.append(f"Input features contain {inf_count_X} infinite values")
            
            if nan_count_y > 0:
                errors.append(f"Target values contain {nan_count_y} NaN values")
            
            if inf_count_y > 0:
                errors.append(f"Target values contain {inf_count_y} infinite values")
            
            # Regime distribution checks
            unique_regimes = np.unique(regime_labels)
            regime_counts = np.bincount(regime_labels)
            min_regime_size = regime_counts.min()
            max_regime_size = regime_counts.max()
            
            if min_regime_size < self.config.min_samples_per_regime:
                warnings.append(f"Some regimes have fewer than {self.config.min_samples_per_regime} samples (min: {min_regime_size})")
            
            # Data distribution warnings
            if max_regime_size / min_regime_size > 10:
                warnings.append(f"High regime imbalance: largest regime is {max_regime_size/min_regime_size:.1f}x larger than smallest")
            
            # Feature statistics
            feature_stats = {
                'n_samples': len(X),
                'n_features': X.shape[1] if len(X.shape) > 1 else 1,
                'n_regimes': len(unique_regimes),
                'regime_distribution': dict(zip(unique_regimes, regime_counts)),
                'feature_means': np.mean(X, axis=0).tolist() if len(X.shape) > 1 else [np.mean(X)],
                'feature_stds': np.std(X, axis=0).tolist() if len(X.shape) > 1 else [np.std(X)]
            }
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'statistics': feature_stats
            }
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Validation exception: {str(e)}"],
                'warnings': [],
                'statistics': {}
            }
    
    def _validate_input_data(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> bool:
        """Legacy validation method for backward compatibility."""
        result = self._validate_input_data_enhanced(X, y, regime_labels)
        return result['valid']
    
    def _enhance_results_with_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance results with comprehensive performance metrics."""
        try:
            # Add training efficiency metrics
            if 'total_training_time' in results:
                results['training_efficiency'] = {
                    'total_time': results['total_training_time'],
                    'models_per_second': results.get('summary', {}).get('total_models', 0) / results['total_training_time'],
                    'regimes_per_second': results.get('summary', {}).get('total_regimes', 0) / results['total_training_time']
                }
            
            # Add model performance comparison
            if 'evaluation_results' in results:
                model_performance = {}
                for regime, regime_results in results['evaluation_results'].items():
                    if isinstance(regime_results, dict):
                        for model_type, model_results in regime_results.items():
                            if isinstance(model_results, dict) and 'metrics' in model_results:
                                if model_type not in model_performance:
                                    model_performance[model_type] = []
                                model_performance[model_type].append(model_results['metrics'])
                
                # Calculate aggregate performance per model type
                for model_type, metrics_list in model_performance.items():
                    if metrics_list:
                        aggregate_metrics = {}
                        for metric_name in metrics_list[0].keys():
                            values = [m[metric_name] for m in metrics_list if metric_name in m]
                            if values:
                                aggregate_metrics[metric_name] = {
                                    'mean': np.mean(values),
                                    'std': np.std(values),
                                    'min': np.min(values),
                                    'max': np.max(values),
                                    'count': len(values)
                                }
                        model_performance[model_type] = aggregate_metrics
                
                results['model_performance_comparison'] = model_performance
            
            return results
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to enhance results with performance metrics: {e}")
            return results
    
    def _add_analyst_specific_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add analyst-specific metadata to results with enhanced analysis.
        
        Args:
            results: Training results
            
        Returns:
            Enhanced results with analyst-specific metadata
        """
        try:
            # Add analyst-specific analysis
            if 'regime_analysis' in results:
                regime_analysis = results['regime_analysis']
                
                # Calculate comprehensive analyst-specific metrics
                analyst_metrics = {
                    'total_regimes': len(regime_analysis.get('unique_regimes', [])),
                    'sufficient_regimes': len(regime_analysis.get('sufficient_regimes', [])),
                    'insufficient_regimes': len(regime_analysis.get('insufficient_regimes', [])),
                    'regime_balance': regime_analysis.get('regime_balance_train', 0.0),
                    'regime_diversity': self._calculate_regime_diversity(regime_analysis),
                    'data_quality_score': self._calculate_data_quality_score(results)
                }
                
                results['analyst_metrics'] = analyst_metrics
            
            # Add enhanced model performance summary
            if 'evaluation_results' in results:
                evaluation_results = results['evaluation_results']
                
                # Calculate best performing model per regime with confidence scores
                best_models = {}
                model_confidence_scores = {}
                
                for regime, regime_metrics in evaluation_results.items():
                    if isinstance(regime_metrics, dict) and 'error' not in regime_metrics:
                        best_model = None
                        best_r2 = -np.inf
                        regime_scores = {}
                        
                        for model_name, metrics in regime_metrics.items():
                            if isinstance(metrics, dict) and 'metrics' in metrics:
                                model_metrics = metrics['metrics']
                                if 'r2' in model_metrics:
                                    r2_score = model_metrics['r2']
                                    regime_scores[model_name] = r2_score
                                    
                                    if r2_score > best_r2:
                                        best_r2 = r2_score
                                        best_model = model_name
                        
                        if best_model:
                            # Calculate confidence based on score separation
                            sorted_scores = sorted(regime_scores.values(), reverse=True)
                            confidence = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0] if len(sorted_scores) > 1 else 1.0
                            
                            best_models[regime] = {
                                'model': best_model,
                                'r2_score': best_r2,
                                'confidence': confidence,
                                'all_scores': regime_scores
                            }
                            
                            # Track model confidence across regimes
                            if best_model not in model_confidence_scores:
                                model_confidence_scores[best_model] = []
                            model_confidence_scores[best_model].append(confidence)
                
                results['best_models_per_regime'] = best_models
                results['model_confidence_analysis'] = {
                    model: {
                        'average_confidence': np.mean(scores),
                        'confidence_std': np.std(scores),
                        'regime_count': len(scores)
                    } for model, scores in model_confidence_scores.items()
                }
            
            return results
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to add analyst-specific metadata: {e}")
            return results
    
    def _calculate_regime_diversity(self, regime_analysis: Dict[str, Any]) -> float:
        """Calculate regime diversity score."""
        try:
            unique_regimes = regime_analysis.get('unique_regimes', [])
            if len(unique_regimes) <= 1:
                return 0.0
            
            # Calculate entropy-based diversity
            regime_counts = [regime_analysis.get('regime_counts', {}).get(str(regime), 0) for regime in unique_regimes]
            total_samples = sum(regime_counts)
            
            if total_samples == 0:
                return 0.0
            
            # Normalize counts to probabilities
            probabilities = [count / total_samples for count in regime_counts]
            
            # Calculate Shannon entropy
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            
            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(unique_regimes))
            diversity_score = entropy / max_entropy if max_entropy > 0 else 0.0
            
            return diversity_score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate regime diversity: {e}")
            return 0.0
    
    def _calculate_data_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        try:
            score = 1.0
            
            # Penalize for errors
            if 'error' in results:
                score -= 0.5
            
            # Reward for successful models
            if 'evaluation_results' in results:
                successful_models = 0
                total_models = 0
                
                for regime_results in results['evaluation_results'].values():
                    if isinstance(regime_results, dict):
                        for model_results in regime_results.values():
                            total_models += 1
                            if isinstance(model_results, dict) and 'metrics' in model_results:
                                successful_models += 1
                
                if total_models > 0:
                    success_rate = successful_models / total_models
                    score = score * success_rate
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate data quality score: {e}")
            return 0.0
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of the training system."""
        health_status = {
            'overall_status': 'healthy',
            'checks': {},
            'recommendations': []
        }
        
        try:
            # Check system resources
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                health_status['checks']['memory'] = 'critical'
                health_status['recommendations'].append('High memory usage detected - consider reducing batch size')
            elif memory.percent > 75:
                health_status['checks']['memory'] = 'warning'
                health_status['recommendations'].append('Memory usage is high - monitor during training')
            else:
                health_status['checks']['memory'] = 'healthy'
            
            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.percent > 95:
                health_status['checks']['disk'] = 'critical'
                health_status['recommendations'].append('Low disk space - cleanup required')
            elif disk.percent > 85:
                health_status['checks']['disk'] = 'warning'
                health_status['recommendations'].append('Disk space is getting low')
            else:
                health_status['checks']['disk'] = 'healthy'
            
            # Check configuration
            config_validation = self._validate_config_enhanced(self.config)
            if not config_validation['valid']:
                health_status['checks']['configuration'] = 'critical'
                health_status['recommendations'].extend([f"Config error: {error}" for error in config_validation['errors']])
            else:
                health_status['checks']['configuration'] = 'healthy'
                if config_validation['warnings']:
                    health_status['recommendations'].extend([f"Config warning: {warning}" for warning in config_validation['warnings']])
            
            # Check error history
            error_summary = self.error_handler.get_error_summary()
            if error_summary['total_errors'] > 10:
                health_status['checks']['error_rate'] = 'warning'
                health_status['recommendations'].append('High error rate detected - review error logs')
            else:
                health_status['checks']['error_rate'] = 'healthy'
            
            # Determine overall status
            critical_checks = [check for check, status in health_status['checks'].items() if status == 'critical']
            warning_checks = [check for check, status in health_status['checks'].items() if status == 'warning']
            
            if critical_checks:
                health_status['overall_status'] = 'critical'
            elif warning_checks:
                health_status['overall_status'] = 'warning'
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")
            return {
                'overall_status': 'error',
                'checks': {'health_check': 'failed'},
                'recommendations': [f'Health check failed: {str(e)}']
            }


# Enhanced Convenience Functions
def create_analyst_models_training_step_enhanced(
    config: Optional[PerRegimeTrainingConfig] = None
) -> AnalystModelsTrainingStepRefactored:
    """
    Create enhanced Analyst models training step with comprehensive monitoring.
    
    Args:
        config: Per-regime training configuration
        
    Returns:
        Enhanced Analyst models training step instance
        
    Raises:
        ValueError: If configuration is invalid
    """
    try:
        step = AnalystModelsTrainingStepRefactored(config)
        
        # Perform initial health check
        health_status = step.health_check()
        if health_status['overall_status'] == 'critical':
            logger.warning(f"⚠️ Critical health issues detected: {health_status['recommendations']}")
        elif health_status['overall_status'] == 'warning':
            logger.info(f"ℹ️ Health warnings: {health_status['recommendations']}")
        
        return step
        
    except Exception as e:
        logger.error(f"❌ Failed to create enhanced analyst models training step: {e}")
        raise


def execute_analyst_models_training_enhanced(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[PerRegimeTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None,
    perform_health_check: bool = True
) -> Dict[str, Any]:
    """
    Execute enhanced Analyst models training step with comprehensive monitoring and error handling.
    
    Args:
        X: Input features
        y: Target values (analyst outputs)
        regime_labels: Regime labels for each sample
        config: Per-regime training configuration
        feature_names: Names of input features
        hmm_states: HMM cluster/regime states
        perform_health_check: Whether to perform health check before training
        
    Returns:
        Dictionary containing training results and comprehensive metadata
        
    Raises:
        ValueError: If input data is invalid
        RuntimeError: If training fails
    """
    step = create_analyst_models_training_step_enhanced(config)
    
    # Perform pre-training health check
    if perform_health_check:
        health_status = step.health_check()
        logger.info(f"🏥 Pre-training health check: {health_status['overall_status']}")
        if health_status['recommendations']:
            logger.info(f"💡 Health recommendations: {health_status['recommendations']}")
    
    # Execute training with enhanced monitoring
    results = step.execute(X, y, regime_labels, feature_names, hmm_states)
    
    # Add post-training health check to results
    post_health = step.health_check()
    results['post_training_health'] = post_health
    
    return results


def analyze_training_report(report_path: str) -> Dict[str, Any]:
    """
    Analyze a training report and provide insights.
    
    Args:
        report_path: Path to the training report JSON file
        
    Returns:
        Dictionary containing analysis insights
    """
    try:
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        analysis = {
            'report_metadata': report_data.get('metadata', {}),
            'performance_summary': report_data.get('summary', {}),
            'health_insights': [],
            'recommendations': []
        }
        
        # Analyze performance metrics
        perf_metrics = report_data.get('summary', {}).get('performance_metrics', {})
        if perf_metrics:
            if perf_metrics.get('average_r2_score', 0) > 0.8:
                analysis['health_insights'].append("Excellent model performance (R² > 0.8)")
            elif perf_metrics.get('average_r2_score', 0) > 0.6:
                analysis['health_insights'].append("Good model performance (R² > 0.6)")
            else:
                analysis['recommendations'].append("Consider improving model performance - R² < 0.6")
        
        # Analyze error summary
        error_summary = report_data.get('monitoring', {}).get('error_summary', {})
        if error_summary.get('total_errors', 0) > 0:
            analysis['health_insights'].append(f"Training completed with {error_summary['total_errors']} errors")
            if error_summary.get('total_errors', 0) > 5:
                analysis['recommendations'].append("High error count - review error logs for improvement opportunities")
        
        # Analyze system metrics
        system_metrics = report_data.get('monitoring', {}).get('system_metrics', {})
        if system_metrics:
            memory_usage = system_metrics.get('memory', {}).get('used_percent', 0)
            if memory_usage > 90:
                analysis['recommendations'].append("High memory usage detected - consider optimizing memory usage")
        
        return analysis
        
    except Exception as e:
        logger.error(f"❌ Failed to analyze training report: {e}")
        return {'error': str(e)}


# Legacy compatibility functions
def create_analyst_models_training_step_refactored(
    config: Optional[PerRegimeTrainingConfig] = None
) -> AnalystModelsTrainingStepRefactored:
    """Legacy function for backward compatibility."""
    logger.warning("⚠️ Using legacy function - consider using create_analyst_models_training_step_enhanced")
    return create_analyst_models_training_step_enhanced(config)


def execute_analyst_models_training_refactored(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[PerRegimeTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Legacy function for backward compatibility."""
    logger.warning("⚠️ Using legacy function - consider using execute_analyst_models_training_enhanced")
    return execute_analyst_models_training_enhanced(X, y, regime_labels, config, feature_names, hmm_states)