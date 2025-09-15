"""
Improved HMM Training Manager

Demonstrates implementation of key suggestions for streamlining, enhanced reporting,
and silent failure prevention.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import time
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
from enum import Enum

# Core imports
from src.utils.logger import system_logger
from src.utils.ml_common.config.base_training_config import HMMTrainingConfig
from src.utils.ml_common.training.base_training_step import BaseTrainingStep

logger = system_logger.getChild('ImprovedTrainingManager')


class ValidationLevel(Enum):
    """Validation levels for different strictness."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"


class ValidationResult(Enum):
    """Validation result types."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class ValidationCheck:
    """Individual validation check result."""
    name: str
    result: ValidationResult
    message: str
    details: Optional[Dict[str, Any]] = None
    severity: str = "medium"


@dataclass
class TrainingMetrics:
    """Enhanced training metrics with more detailed information."""
    accuracy: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    training_time: float = 0.0
    convergence_epochs: int = 0
    memory_usage_mb: float = 0.0
    validation_loss: Optional[float] = None
    test_accuracy: Optional[float] = None
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class ModelResult:
    """Enhanced model result container."""
    model: Any
    metrics: TrainingMetrics
    feature_importance: Optional[Dict[str, float]] = None
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    training_history: Optional[Dict[str, List[float]]] = None


class TrainingErrorHandler:
    """Centralized error handling for training operations."""
    
    @staticmethod
    def handle_model_creation_error(model_type: str, error: Exception) -> ModelResult:
        """Standardized model creation error handling."""
        return ModelResult(
            model=None,
            metrics=TrainingMetrics(
                error_message=f"Failed to create {model_type}: {str(error)}",
                training_time=0.0
            )
        )
    
    @staticmethod
    def handle_training_error(model_type: str, error: Exception, training_time: float) -> ModelResult:
        """Standardized training error handling."""
        return ModelResult(
            model=None,
            metrics=TrainingMetrics(
                error_message=f"Failed to train {model_type}: {str(error)}",
                training_time=training_time
            )
        )


class ModelFactory:
    """Factory for creating model instances with standardized configuration."""
    
    _model_configs = {
        'logistic_regression': {
            'class': 'sklearn.linear_model.LogisticRegression',
            'default_params': {
                'C': 1.0, 'max_iter': 1000, 'random_state': 42,
                'class_weight': 'balanced'
            }
        },
        'lightgbm': {
            'class': 'lightgbm.LGBMClassifier',
            'default_params': {
                'n_estimators': 100, 'learning_rate': 0.1,
                'max_depth': 6, 'random_state': 42, 'verbose': -1
            }
        },
        'random_forest': {
            'class': 'sklearn.ensemble.RandomForestClassifier',
            'default_params': {
                'n_estimators': 100, 'max_depth': 10,
                'random_state': 42, 'n_jobs': -1
            }
        }
    }
    
    @classmethod
    def create_model(cls, model_type: str, **custom_params) -> Any:
        """Create model instance with standardized configuration."""
        if model_type not in cls._model_configs:
            raise ValueError(f"Unknown model type: {model_type}")
        
        config = cls._model_configs[model_type]
        
        # Import the class dynamically
        class_path = config['class']
        module_name, class_name = class_path.rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        model_class = getattr(module, class_name)
        
        # Merge default and custom parameters
        params = {**config['default_params'], **custom_params}
        return model_class(**params)


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures."""
    
    def __init__(self, failure_threshold: int = 3, timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN - too many failures")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e


class RealTimeProgressReporter:
    """Real-time progress reporting during training."""
    
    def __init__(self, total_models: int):
        self.total_models = total_models
        self.completed_models = 0
        self.start_time = time.time()
        self.model_times = []
        self.successful_models = 0
        self.failed_models = 0
    
    def update_progress(self, model_name: str, success: bool, training_time: float):
        """Update progress after each model training."""
        self.completed_models += 1
        self.model_times.append(training_time)
        
        if success:
            self.successful_models += 1
        else:
            self.failed_models += 1
        
        progress_percent = (self.completed_models / self.total_models) * 100
        avg_time = np.mean(self.model_times)
        eta = avg_time * (self.total_models - self.completed_models)
        
        status = "✅" if success else "❌"
        
        print(f"\r{status} {model_name} | Progress: {progress_percent:.1f}% | "
              f"Success: {self.successful_models}/{self.completed_models} | "
              f"ETA: {eta:.1f}s", end="", flush=True)
    
    def finish_report(self):
        """Generate final progress report."""
        total_time = time.time() - self.start_time
        print(f"\n\n🎯 Training Summary:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average time per model: {np.mean(self.model_times):.2f}s")
        print(f"   Successful models: {self.successful_models}/{self.total_models}")
        print(f"   Success rate: {(self.successful_models/self.total_models)*100:.1f}%")


class PerformanceMetricsCollector:
    """Collect detailed performance metrics during training."""
    
    def __init__(self):
        self.metrics = {
            'training_times': [],
            'memory_usage': [],
            'model_accuracies': [],
            'feature_counts': []
        }
    
    def start_training_timer(self) -> str:
        """Start timer for training session."""
        timer_id = f"training_{int(time.time())}"
        self.metrics[timer_id] = {'start_time': time.time()}
        return timer_id
    
    def end_training_timer(self, timer_id: str):
        """End timer and record training time."""
        if timer_id in self.metrics:
            self.metrics[timer_id]['end_time'] = time.time()
            training_time = self.metrics[timer_id]['end_time'] - self.metrics[timer_id]['start_time']
            self.metrics['training_times'].append(training_time)
    
    def record_model_performance(self, accuracy: float, training_time: float, memory_mb: float):
        """Record individual model performance."""
        self.metrics['model_accuracies'].append(accuracy)
        self.metrics['training_times'].append(training_time)
        self.metrics['memory_usage'].append(memory_mb)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of collected metrics."""
        return {
            'avg_training_time': np.mean(self.metrics['training_times']) if self.metrics['training_times'] else 0,
            'max_memory_usage': np.max(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
            'avg_accuracy': np.mean(self.metrics['model_accuracies']) if self.metrics['model_accuracies'] else 0,
            'total_training_sessions': len(self.metrics['training_times'])
        }


class ImprovedHMMTrainingManager(BaseTrainingStep):
    """
    Improved HMM Training Manager implementing key suggestions for:
    - Streamlined architecture
    - Enhanced error handling
    - Silent failure prevention
    - Real-time reporting
    """
    
    def __init__(self, config: Optional[Union[HMMTrainingConfig, Dict[str, Any]]] = None):
        """Initialize improved training manager."""
        if config is None:
            config = HMMTrainingConfig(
                model_name="improved_hmm_models",
                timeframe="1h",
                n_features=50,
                sequence_length=20,
                n_regimes=3,
                model_types=["logistic_regression", "lightgbm", "random_forest"],
                hpo_trials=25,
                enable_multi_objective=True
            )
        elif isinstance(config, dict):
            default_config = HMMTrainingConfig()
            config_dict = {**default_config.__dict__, **config}
            config = HMMTrainingConfig(**config_dict)

        super().__init__(config)
        self.logger = logger.getChild('ImprovedTrainingManager')
        
        # Initialize improved components
        self.circuit_breaker = CircuitBreaker(failure_threshold=2, timeout=60)
        self.performance_collector = PerformanceMetricsCollector()
        self.progress_reporter = None
        
        # Training state
        self.training_start_time = None
        self.training_results = {}
        
        self.logger.info("✅ Improved HMM Training Manager initialized")
    
    def _validate_inputs_enhanced(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> bool:
        """
        Enhanced input validation with early exit on critical failures.
        
        Returns:
            True if validation passes, False otherwise
        """
        critical_failures = []
        warnings = []
        
        try:
            # Critical checks (cause early exit)
            if len(X) == 0:
                critical_failures.append("Input data is empty")
            
            if len(X) != len(y):
                critical_failures.append(f"X length ({len(X)}) != y length ({len(y)})")
            
            if len(X) != len(regime_labels):
                critical_failures.append(f"X length ({len(X)}) != regime_labels length ({len(regime_labels)})")
            
            # Check for NaN values
            if isinstance(X, np.ndarray):
                if np.any(np.isnan(X)):
                    critical_failures.append("X contains NaN values")
                if np.any(np.isinf(X)):
                    critical_failures.append("X contains infinite values")
            
            if np.any(np.isnan(y)):
                critical_failures.append("y contains NaN values")
            
            if np.any(np.isnan(regime_labels)):
                critical_failures.append("regime_labels contains NaN values")
            
            # Warning checks (don't cause early exit)
            if len(X) < 1000:
                warnings.append(f"Small dataset: {len(X)} samples (recommended: >1000)")
            
            unique_regimes = np.unique(regime_labels)
            if len(unique_regimes) < 2:
                critical_failures.append("Need at least 2 regimes")
            
            # Check regime balance
            regime_counts = [np.sum(regime_labels == regime) for regime in unique_regimes]
            min_regime_count = min(regime_counts)
            if min_regime_count < 10:
                warnings.append(f"Some regimes have very few samples (minimum: {min_regime_count})")
            
            # Early exit on critical failures
            if critical_failures:
                self.logger.error(f"❌ Critical validation failures: {critical_failures}")
                return False
            
            # Log warnings
            if warnings:
                for warning in warnings:
                    self.logger.warning(f"⚠️ {warning}")
            
            self.logger.info(f"✅ Enhanced validation passed: {len(X)} samples, {len(unique_regimes)} regimes")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Validation error: {e}")
            return False
    
    def _train_single_model_improved(self, model_type: str, X: np.ndarray, y: np.ndarray) -> ModelResult:
        """
        Train a single model with improved error handling and metrics collection.
        
        Args:
            model_type: Type of model to train
            X: Training features
            y: Training targets
            
        Returns:
            ModelResult with trained model and metrics
        """
        start_time = time.time()
        metrics = TrainingMetrics()
        
        try:
            self.logger.info(f"🔄 Training {model_type}...")
            
            # Create model using factory
            model = ModelFactory.create_model(model_type)
            
            # Train model with circuit breaker protection
            def train_model():
                return model.fit(X, y)
            
            self.circuit_breaker.call(train_model)
            training_time = time.time() - start_time
            
            # Evaluate model
            try:
                predictions = model.predict(X)
                accuracy = np.mean(predictions == y)
                metrics.accuracy = accuracy
                
                # Calculate additional metrics
                from sklearn.metrics import f1_score, precision_score, recall_score
                metrics.f1_score = f1_score(y, predictions, average='weighted')
                metrics.precision = precision_score(y, predictions, average='weighted')
                metrics.recall = recall_score(y, predictions, average='weighted')
                
            except Exception as e:
                metrics.warnings.append(f"Evaluation failed: {e}")
                # Fallback to simple accuracy
                predictions = model.predict(X)
                metrics.accuracy = np.mean(predictions == y)
            
            metrics.training_time = training_time
            
            # Get predictions and probabilities
            predictions = None
            probabilities = None
            try:
                predictions = model.predict(X)
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X)
            except Exception as e:
                metrics.warnings.append(f"Failed to get predictions: {e}")
            
            # Get feature importance
            feature_importance = None
            try:
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(range(len(model.feature_importances_)), model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    feature_importance = dict(zip(range(len(model.coef_[0])), np.abs(model.coef_[0])))
            except Exception as e:
                self.logger.debug(f"Feature importance not available for {model_type}: {e}")
            
            # Get hyperparameters
            hyperparameters = None
            try:
                if hasattr(model, 'get_params'):
                    hyperparameters = model.get_params()
            except Exception as e:
                self.logger.debug(f"Could not get hyperparameters: {e}")
            
            # Record performance metrics
            self.performance_collector.record_model_performance(
                metrics.accuracy, training_time, metrics.memory_usage_mb
            )
            
            self.logger.info(f"✅ {model_type} trained successfully (accuracy: {metrics.accuracy:.4f}, time: {training_time:.2f}s)")
            
            return ModelResult(
                model=model,
                metrics=metrics,
                feature_importance=feature_importance,
                predictions=predictions,
                probabilities=probabilities,
                hyperparameters=hyperparameters
            )
            
        except Exception as e:
            training_time = time.time() - start_time
            metrics.training_time = training_time
            metrics.error_message = str(e)
            
            self.logger.error(f"❌ Failed to train {model_type}: {e}")
            
            return TrainingErrorHandler.handle_training_error(model_type, e, training_time)
    
    def _generate_enhanced_report(self, results: Dict[str, Any], execution_time: float) -> Dict[str, Any]:
        """
        Generate enhanced report with real metrics and actionable insights.
        
        Args:
            results: Training results
            execution_time: Total execution time
            
        Returns:
            Enhanced report dictionary
        """
        try:
            # Get performance summary
            perf_summary = self.performance_collector.get_summary_stats()
            
            report = {
                "report_metadata": {
                    "report_type": "Improved HMM Training Report",
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "version": "2.1",
                    "generator": "Improved Training Manager"
                },
                "execution_summary": {
                    "total_execution_time": execution_time,
                    "models_trained": len(results.get('model_results', {})),
                    "successful_models": sum(1 for r in results.get('model_results', {}).values() 
                                           if r.metrics.error_message is None),
                    "failed_models": sum(1 for r in results.get('model_results', {}).values() 
                                        if r.metrics.error_message is not None),
                    "circuit_breaker_state": self.circuit_breaker.state
                },
                "performance_metrics": perf_summary,
                "model_performance": {},
                "feature_analysis": {
                    "total_features": results.get('total_features', 0),
                    "selected_features": results.get('selected_features', 0),
                    "feature_selection_ratio": results.get('selected_features', 0) / max(results.get('total_features', 1), 1)
                },
                "regime_analysis": {
                    "total_regimes": results.get('n_regimes', 0),
                    "regime_distribution": results.get('regime_distribution', {})
                },
                "recommendations": [],
                "warnings": []
            }
            
            # Analyze model performance
            model_results = results.get('model_results', {})
            if model_results:
                best_accuracy = -1
                best_model = None
                accuracies = []
                warnings = []
                
                for model_name, model_result in model_results.items():
                    metrics = model_result.metrics
                    
                    report["model_performance"][model_name] = {
                        "accuracy": metrics.accuracy,
                        "f1_score": metrics.f1_score,
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "training_time": metrics.training_time,
                        "status": "success" if metrics.error_message is None else "failed",
                        "error": metrics.error_message,
                        "warnings": metrics.warnings
                    }
                    
                    if metrics.error_message is None:
                        accuracies.append(metrics.accuracy)
                        if metrics.accuracy > best_accuracy:
                            best_accuracy = metrics.accuracy
                            best_model = model_name
                    
                    # Collect warnings
                    warnings.extend(metrics.warnings)
                
                # Add performance summary
                if accuracies:
                    report["performance_summary"] = {
                        "best_model": best_model,
                        "best_accuracy": best_accuracy,
                        "average_accuracy": np.mean(accuracies),
                        "accuracy_std": np.std(accuracies),
                        "performance_variance": np.var(accuracies)
                    }
                
                report["warnings"] = list(set(warnings))  # Remove duplicates
            
            # Generate recommendations
            recommendations = []
            
            if report["execution_summary"]["failed_models"] > 0:
                recommendations.append(f"Address {report['execution_summary']['failed_models']} failed model(s)")
            
            if report["performance_summary"]["average_accuracy"] < 0.7:
                recommendations.append("Consider feature engineering or data preprocessing improvements")
            
            if report["execution_summary"]["circuit_breaker_state"] == "OPEN":
                recommendations.append("Circuit breaker is OPEN - investigate systematic failures")
            
            if len(report["warnings"]) > 0:
                recommendations.append(f"Address {len(report['warnings'])} warnings for better performance")
            
            if report["feature_analysis"]["feature_selection_ratio"] > 0.5:
                recommendations.append("High feature selection ratio - consider more aggressive feature selection")
            
            report["recommendations"] = recommendations
            
            self.logger.info("✅ Enhanced report generated successfully")
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate enhanced report: {e}")
            return {
                "report_type": "Improved HMM Training Report (Error)",
                "error": str(e),
                "timestamp": pd.Timestamp.now().isoformat(),
                "status": "Report generation failed"
            }
    
    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute improved HMM models training with enhanced error handling and reporting.
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing training results and enhanced report
        """
        self.logger.info("🚀 Starting Improved HMM Models Training")
        self.training_start_time = time.time()
        
        try:
            # Step 1: Enhanced input validation
            self.logger.info("🔄 Step 1: Enhanced input validation...")
            if not self._validate_inputs_enhanced(X, y, regime_labels):
                raise ValueError("Enhanced input validation failed")
            
            # Step 2: Initialize progress reporter
            self.progress_reporter = RealTimeProgressReporter(len(self.config.model_types))
            
            # Step 3: Train models with improved error handling
            self.logger.info("🔄 Step 3: Training models with improved error handling...")
            model_results = {}
            
            for model_type in self.config.model_types:
                try:
                    # Train model with improved method
                    model_result = self._train_single_model_improved(model_type, X, y)
                    model_results[model_type] = model_result
                    
                    # Update progress
                    success = model_result.metrics.error_message is None
                    self.progress_reporter.update_progress(model_type, success, model_result.metrics.training_time)
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to train {model_type}: {e}")
                    # Create failed result using error handler
                    model_results[model_type] = TrainingErrorHandler.handle_training_error(model_type, e, 0.0)
                    self.progress_reporter.update_progress(model_type, False, 0.0)
            
            # Step 4: Finish progress reporting
            self.progress_reporter.finish_report()
            
            # Step 5: Analyze regimes
            unique_regimes, regime_counts = np.unique(regime_labels, return_counts=True)
            regime_distribution = {
                f"regime_{regime}": {
                    "count": int(count),
                    "percentage": float(count / len(regime_labels) * 100)
                }
                for regime, count in zip(unique_regimes, regime_counts)
            }
            
            # Step 6: Create final results
            execution_time = time.time() - self.training_start_time
            
            results = {
                'model_results': model_results,
                'metadata': {
                    'total_features': X.shape[1] if hasattr(X, 'shape') else len(X[0]) if len(X) > 0 else 0,
                    'selected_features': X.shape[1] if hasattr(X, 'shape') else len(X[0]) if len(X) > 0 else 0,
                    'n_regimes': len(unique_regimes),
                    'regime_distribution': regime_distribution,
                    'execution_time': execution_time,
                    'config': asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else str(self.config),
                    'circuit_breaker_state': self.circuit_breaker.state,
                    'performance_summary': self.performance_collector.get_summary_stats()
                },
                'training_time': execution_time
            }
            
            # Step 7: Generate enhanced report
            self.logger.info("🔄 Step 7: Generating enhanced report...")
            enhanced_report = self._generate_enhanced_report(results, execution_time)
            results['enhanced_report'] = enhanced_report
            
            # Log final summary
            successful_count = sum(1 for r in model_results.values() if r.metrics.error_message is None)
            self.logger.info(f"✅ Improved HMM Models Training completed: {successful_count}/{len(model_results)} models successful")
            self.logger.info(f"📊 Total execution time: {execution_time:.2f}s")
            self.logger.info(f"🔧 Circuit breaker state: {self.circuit_breaker.state}")
            
            return results
            
        except Exception as e:
            execution_time = time.time() - self.training_start_time if self.training_start_time else 0
            self.logger.error(f"❌ Improved HMM Models Training failed: {e}")
            
            return {
                'model_results': {},
                'metadata': {
                    'error': str(e),
                    'execution_time': execution_time,
                    'config': asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else str(self.config),
                    'circuit_breaker_state': self.circuit_breaker.state
                },
                'training_time': execution_time,
                'enhanced_report': {
                    "report_type": "Improved HMM Training Report (Error)",
                    "error": str(e),
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "status": "Training failed"
                }
            }


# Convenience functions
def create_improved_hmm_training_manager(
    config: Optional[HMMTrainingConfig] = None
) -> ImprovedHMMTrainingManager:
    """Create improved HMM training manager."""
    return ImprovedHMMTrainingManager(config)


def execute_improved_hmm_training(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[HMMTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Execute improved HMM training step."""
    manager = create_improved_hmm_training_manager(config)
    return manager.execute(X, y, regime_labels, feature_names, hmm_states)


# Example usage
if __name__ == "__main__":
    print("Improved HMM Training Manager")
    print("=" * 50)
    
    # Create configuration
    config = HMMTrainingConfig(
        model_name="improved_hmm_models",
        timeframe="1h",
        n_features=50,
        sequence_length=20,
        n_regimes=3,
        model_types=["logistic_regression", "lightgbm", "random_forest"],
        hpo_trials=25,
        enable_multi_objective=True
    )
    
    # Create training manager
    training_manager = create_improved_hmm_training_manager(config)
    
    print(f"✅ Created improved training manager with {len(config.model_types)} model types")
    print(f"📊 Features: {config.n_features}")
    print(f"📊 Sequence length: {config.sequence_length}")
    print(f"📊 HPO trials: {config.hpo_trials}")
    
    print("\n🎯 Key improvements implemented:")
    print("- ✅ Centralized error handling")
    print("- ✅ Model factory pattern")
    print("- ✅ Circuit breaker for failure prevention")
    print("- ✅ Real-time progress reporting")
    print("- ✅ Enhanced input validation with early exit")
    print("- ✅ Performance metrics collection")
    print("- ✅ Comprehensive reporting with actionable insights")
    print("- ✅ Silent failure prevention")