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


# All duplicated classes now imported from shared utilities


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
                timeframe="15m",
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
        self.memory_tracker = MemoryTracker()
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
            
            # Create model using shared unified factory
            model = UnifiedModelFactory.create_model(model_type)
            
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
            
            # Record performance metrics if available
            if hasattr(self, 'performance_collector') and self.performance_collector:
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
            # Get performance summary if available
            perf_summary = {}
            if hasattr(self, 'performance_collector') and self.performance_collector:
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
            self.progress_reporter = ProgressReporter(len(self.config.model_types))
            
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
                    accuracy = model_result.metrics.accuracy if success else None
                    error_message = model_result.metrics.error_message if not success else None
                    self.progress_reporter.update_progress(
                        model_type, success, model_result.metrics.training_time, 
                        accuracy, error_message
                    )
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to train {model_type}: {e}")
                    # Create failed result using error handler
                    model_results[model_type] = TrainingErrorHandler.handle_training_error(model_type, e, 0.0)
                    self.progress_reporter.update_progress(model_type, False, 0.0, None, str(e))
            
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
                    'performance_summary': perf_summary
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