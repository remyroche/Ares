"""
Model Evaluation Component

This module provides comprehensive model evaluation capabilities including
pre and post HPO metrics comparison, performance analysis, and evaluation reporting.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import gc
import psutil
from pathlib import Path

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size, check_disk_space
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, MathValidationError
)
from src.utils.parquet_utils import get_parquet_utils, ParquetUtils
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time, 
    timeout, error_boundary, compose, validate_data_quality, 
    monitor_step_execution, ensure_data_integrity, validate_pipeline_step
)
from src.utils.intensity_scaler import (
    get_intensity_from_environment, get_scaled_hpo_trials,
    get_scaled_hpo_timeout, log_intensity_info
)
from src.core.errors import (
    ValidationError, DataIntegrityError, FileOperationError,
    ConfigurationError, ModelTrainingError
)
from src.utils.logger import system_logger
from src.utils.ml_common.evaluation.unified_evaluator import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_sharpe_ratio,
)

# ML metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    
    # Regression metrics
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    
    # Additional metrics
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    
    # Metadata
    evaluation_time: Optional[float] = None
    sample_count: Optional[int] = None
    feature_count: Optional[int] = None

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Evaluation settings
    enable_pre_hpo_evaluation: bool = True
    enable_post_hpo_evaluation: bool = True
    enable_cross_validation: bool = True
    cv_folds: int = 5
    
    # Metrics to calculate
    calculate_classification_metrics: bool = True
    calculate_regression_metrics: bool = True
    calculate_trading_metrics: bool = True
    
    # Trading-specific metrics
    enable_sharpe_ratio: bool = True
    enable_max_drawdown: bool = True
    enable_win_rate: bool = True
    enable_profit_factor: bool = True
    
    # Performance thresholds
    min_accuracy_threshold: float = 0.5
    min_f1_threshold: float = 0.5
    min_r2_threshold: float = 0.0
    min_sharpe_threshold: float = 0.0
    
    # Output settings
    save_evaluation_results: bool = True
    generate_evaluation_report: bool = True
    evaluation_report_path: Optional[str] = None

@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    
    # Pre-HPO metrics
    pre_hpo_metrics: Optional[EvaluationMetrics] = None
    
    # Post-HPO metrics
    post_hpo_metrics: Optional[EvaluationMetrics] = None
    
    # Improvement metrics
    accuracy_improvement: Optional[float] = None
    f1_improvement: Optional[float] = None
    r2_improvement: Optional[float] = None
    sharpe_improvement: Optional[float] = None
    
    # Overall assessment
    evaluation_passed: bool = False
    performance_grade: str = "F"  # A, B, C, D, F
    
    # Metadata
    evaluation_time: float = 0.0
    model_name: str = ""
    evaluation_timestamp: str = ""
    
    def __post_init__(self):
        """Calculate improvement metrics."""
        if self.pre_hpo_metrics and self.post_hpo_metrics:
            self._calculate_improvements()
            self._determine_grade()
    
    def _calculate_improvements(self):
        """Calculate improvement metrics."""
        if self.pre_hpo_metrics.accuracy and self.post_hpo_metrics.accuracy:
            self.accuracy_improvement = self.post_hpo_metrics.accuracy - self.pre_hpo_metrics.accuracy
        
        if self.pre_hpo_metrics.f1_score and self.post_hpo_metrics.f1_score:
            self.f1_improvement = self.post_hpo_metrics.f1_score - self.pre_hpo_metrics.f1_score
        
        if self.pre_hpo_metrics.r2_score and self.post_hpo_metrics.r2_score:
            self.r2_improvement = self.post_hpo_metrics.r2_score - self.pre_hpo_metrics.r2_score
        
        if self.pre_hpo_metrics.sharpe_ratio and self.post_hpo_metrics.sharpe_ratio:
            self.sharpe_improvement = self.post_hpo_metrics.sharpe_ratio - self.pre_hpo_metrics.sharpe_ratio
    
    def _determine_grade(self):
        """Determine performance grade based on metrics."""
        if not self.post_hpo_metrics:
            return
        
        score = 0
        total_metrics = 0
        
        # Classification metrics
        if self.post_hpo_metrics.accuracy:
            score += min(self.post_hpo_metrics.accuracy * 100, 100)
            total_metrics += 1
        
        if self.post_hpo_metrics.f1_score:
            score += min(self.post_hpo_metrics.f1_score * 100, 100)
            total_metrics += 1
        
        # Regression metrics
        if self.post_hpo_metrics.r2_score:
            score += max(0, min(self.post_hpo_metrics.r2_score * 100, 100))
            total_metrics += 1
        
        # Trading metrics
        if self.post_hpo_metrics.sharpe_ratio:
            score += max(0, min(self.post_hpo_metrics.sharpe_ratio * 20, 100))
            total_metrics += 1
        
        if total_metrics > 0:
            avg_score = score / total_metrics
            
            if avg_score >= 90:
                self.performance_grade = "A"
            elif avg_score >= 80:
                self.performance_grade = "B"
            elif avg_score >= 70:
                self.performance_grade = "C"
            elif avg_score >= 60:
                self.performance_grade = "D"
            else:
                self.performance_grade = "F"

class ModelEvaluator:
    """Comprehensive model evaluator with pre/post HPO metrics comparison."""
    
    def __init__(self, config: EvaluationConfig):
        """Initialize the model evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.logger = system_logger.getChild('ModelEvaluator')
        
        # Apply intensity scaling
        intensity_pct = get_intensity_from_environment()
        if intensity_pct < 1.0:
            self.config = self._apply_intensity_scaling(intensity_pct)
            self.logger.info(f"🔧 Applied intensity scaling ({intensity_pct*100:.0f}%) to evaluation config")
    
    def _apply_intensity_scaling(self, intensity_pct: float) -> EvaluationConfig:
        """Apply intensity scaling to the configuration."""
        return EvaluationConfig(
            enable_pre_hpo_evaluation=self.config.enable_pre_hpo_evaluation,
            enable_post_hpo_evaluation=self.config.enable_post_hpo_evaluation,
            enable_cross_validation=self.config.enable_cross_validation and intensity_pct > 0.5,
            cv_folds=max(3, int(self.config.cv_folds * intensity_pct)),
            calculate_classification_metrics=self.config.calculate_classification_metrics,
            calculate_regression_metrics=self.config.calculate_regression_metrics,
            calculate_trading_metrics=self.config.calculate_trading_metrics,
            enable_sharpe_ratio=self.config.enable_sharpe_ratio,
            enable_max_drawdown=self.config.enable_max_drawdown,
            enable_win_rate=self.config.enable_win_rate,
            enable_profit_factor=self.config.enable_profit_factor,
            min_accuracy_threshold=self.config.min_accuracy_threshold,
            min_f1_threshold=self.config.min_f1_threshold,
            min_r2_threshold=self.config.min_r2_threshold,
            min_sharpe_threshold=self.config.min_sharpe_threshold,
            save_evaluation_results=self.config.save_evaluation_results,
            generate_evaluation_report=self.config.generate_evaluation_report,
            evaluation_report_path=self.config.evaluation_report_path
        )
    
    @handles_errors(default_return=None, context='Model evaluation')
    # @log_execution_time  # Temporarily disabled due to import conflicts
    async def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                           model_name: str = "", pre_hpo_metrics: Optional[EvaluationMetrics] = None) -> EvaluationResult:
        """Evaluate a trained model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            pre_hpo_metrics: Pre-HPO evaluation metrics for comparison
            
        Returns:
            EvaluationResult with comprehensive metrics
        """
        try:
            self.logger.info(f"🔍 Evaluating model: {model_name}")
            start_time = time.time()
            
            # Get predictions
            y_pred = self._get_predictions(model, X_test)
            y_pred_proba = self._get_prediction_probabilities(model, X_test)
            
            # Calculate metrics
            post_hpo_metrics = await self._calculate_metrics(
                y_test, y_pred, y_pred_proba, X_test, model_name
            )
            
            # Create evaluation result
            result = EvaluationResult(
                pre_hpo_metrics=pre_hpo_metrics,
                post_hpo_metrics=post_hpo_metrics,
                evaluation_time=time.time() - start_time,
                model_name=model_name,
                evaluation_timestamp=get_current_datetime()
            )
            
            # Check if evaluation passes thresholds
            result.evaluation_passed = self._check_evaluation_thresholds(post_hpo_metrics)
            
            # Save results if configured
            if self.config.save_evaluation_results:
                await self._save_evaluation_results(result)
            
            # Generate report if configured
            if self.config.generate_evaluation_report:
                await self._generate_evaluation_report(result)
            
            self.logger.info(f"✅ Model evaluation completed: {result.performance_grade} grade")
            return result
            
        except Exception as e:
            self.logger.exception(f"💥 Error evaluating model: {e}")
            return EvaluationResult(
                evaluation_passed=False,
                model_name=model_name,
                evaluation_timestamp=get_current_datetime()
            )
    
    def _get_predictions(self, model: Any, X_test: np.ndarray) -> np.ndarray:
        """Get predictions from the model."""
        try:
            if hasattr(model, 'predict'):
                return model.predict(X_test)
            else:
                self.logger.warning("⚠️ Model does not have predict method")
                return np.zeros(len(X_test))
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting predictions: {e}")
            return np.zeros(len(X_test))
    
    def _get_prediction_probabilities(self, model: Any, X_test: np.ndarray) -> Optional[np.ndarray]:
        """Get prediction probabilities from the model."""
        try:
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X_test)
            else:
                return None
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting prediction probabilities: {e}")
            return None
    
    @handles_errors(default_return=EvaluationMetrics(), context='Metrics calculation')
    async def _calculate_metrics(self, y_test: np.ndarray, y_pred: np.ndarray, 
                               y_pred_proba: Optional[np.ndarray], X_test: np.ndarray,
                               model_name: str) -> EvaluationMetrics:
        """Calculate comprehensive evaluation metrics."""
        try:
            metrics = EvaluationMetrics()
            metrics.sample_count = len(y_test)
            metrics.feature_count = X_test.shape[1] if len(X_test.shape) > 1 else 1
            
            # Determine if this is classification or regression
            is_classification = self._is_classification_task(y_test, y_pred)
            
            if is_classification and self.config.calculate_classification_metrics:
                await self._calculate_classification_metrics(y_test, y_pred, y_pred_proba, metrics)
            
            if not is_classification and self.config.calculate_regression_metrics:
                await self._calculate_regression_metrics(y_test, y_pred, metrics)
            
            if self.config.calculate_trading_metrics:
                await self._calculate_trading_metrics(y_test, y_pred, metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.exception(f"💥 Error calculating metrics: {e}")
            return EvaluationMetrics()
    
    def _is_classification_task(self, y_test: np.ndarray, y_pred: np.ndarray) -> bool:
        """Determine if this is a classification or regression task."""
        try:
            # Check if targets are discrete
            unique_test = len(np.unique(y_test))
            unique_pred = len(np.unique(y_pred))
            
            # If we have few unique values, it's likely classification
            return unique_test <= 10 and unique_pred <= 10
        except:
            return False
    
    @handles_errors(default_return=None, context='Classification metrics')
    async def _calculate_classification_metrics(self, y_test: np.ndarray, y_pred: np.ndarray,
                                              y_pred_proba: Optional[np.ndarray], metrics: EvaluationMetrics):
        """Calculate classification metrics."""
        try:
            all_metrics = compute_classification_metrics(y_test, y_pred, y_pred_proba)
            metrics.accuracy = all_metrics.get('accuracy')
            metrics.precision = all_metrics.get('precision')
            metrics.recall = all_metrics.get('recall')
            metrics.f1_score = all_metrics.get('f1_score')
            metrics.roc_auc = all_metrics.get('roc_auc')
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating classification metrics: {e}")
    
    @handles_errors(default_return=None, context='Regression metrics')
    async def _calculate_regression_metrics(self, y_test: np.ndarray, y_pred: np.ndarray, metrics: EvaluationMetrics):
        """Calculate regression metrics."""
        try:
            all_metrics = compute_regression_metrics(y_test, y_pred)
            metrics.mse = all_metrics.get('mse')
            metrics.rmse = all_metrics.get('rmse')
            metrics.mae = all_metrics.get('mae')
            metrics.r2_score = all_metrics.get('r2')
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating regression metrics: {e}")
    
    @handles_errors(default_return=None, context='Trading metrics')
    async def _calculate_trading_metrics(self, y_test: np.ndarray, y_pred: np.ndarray, metrics: EvaluationMetrics):
        """Calculate trading-specific metrics."""
        try:
            # Calculate returns based on predictions
            returns = y_pred - y_test  # Assuming y_test contains actual returns
            
            if len(returns) > 0:
                # Sharpe ratio
                if self.config.enable_sharpe_ratio:
                    metrics.sharpe_ratio = compute_sharpe_ratio(returns)
                
                # Win rate
                if self.config.enable_win_rate:
                    metrics.win_rate = np.mean(returns > 0)
                
                # Max drawdown
                if self.config.enable_max_drawdown:
                    cumulative_returns = np.cumsum(returns)
                    running_max = np.maximum.accumulate(cumulative_returns)
                    drawdown = cumulative_returns - running_max
                    metrics.max_drawdown = np.min(drawdown)
                
                # Profit factor
                if self.config.enable_profit_factor:
                    positive_returns = returns[returns > 0]
                    negative_returns = returns[returns < 0]
                    
                    if len(positive_returns) > 0 and len(negative_returns) > 0:
                        total_profit = np.sum(positive_returns)
                        total_loss = abs(np.sum(negative_returns))
                        
                        if total_loss > 0:
                            metrics.profit_factor = total_profit / total_loss
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating trading metrics: {e}")
    
    def _check_evaluation_thresholds(self, metrics: EvaluationMetrics) -> bool:
        """Check if evaluation passes configured thresholds."""
        try:
            # Check accuracy threshold
            if metrics.accuracy is not None and metrics.accuracy < self.config.min_accuracy_threshold:
                return False
            
            # Check F1 threshold
            if metrics.f1_score is not None and metrics.f1_score < self.config.min_f1_threshold:
                return False
            
            # Check R2 threshold
            if metrics.r2_score is not None and metrics.r2_score < self.config.min_r2_threshold:
                return False
            
            # Check Sharpe threshold
            if metrics.sharpe_ratio is not None and metrics.sharpe_ratio < self.config.min_sharpe_threshold:
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error checking evaluation thresholds: {e}")
            return False
    
    @handles_errors(default_return=None, context='Evaluation results saving')
    async def _save_evaluation_results(self, result: EvaluationResult):
        """Save evaluation results to file."""
        try:
            results_data = {
                'model_name': result.model_name,
                'evaluation_timestamp': result.evaluation_timestamp,
                'evaluation_time': result.evaluation_time,
                'evaluation_passed': result.evaluation_passed,
                'performance_grade': result.performance_grade,
                'pre_hpo_metrics': result.pre_hpo_metrics.__dict__ if result.pre_hpo_metrics else None,
                'post_hpo_metrics': result.post_hpo_metrics.__dict__ if result.post_hpo_metrics else None,
                'accuracy_improvement': result.accuracy_improvement,
                'f1_improvement': result.f1_improvement,
                'r2_improvement': result.r2_improvement,
                'sharpe_improvement': result.sharpe_improvement
            }
            
            # Save to file
            results_path = f"data_cache/evaluation_results_{result.model_name}_{get_current_datetime()}.json"
            ensure_directory(Path(results_path).parent)
            safe_json_dump(results_data, results_path)
            
            self.logger.info(f"💾 Evaluation results saved to {results_path}")
            
        except Exception as e:
            self.logger.exception(f"💥 Error saving evaluation results: {e}")
    
    @handles_errors(default_return=None, context='Evaluation report generation')
    async def _generate_evaluation_report(self, result: EvaluationResult):
        """Generate comprehensive evaluation report."""
        try:
            report_path = self.config.evaluation_report_path or f"data_cache/evaluation_report_{result.model_name}_{get_current_datetime()}.txt"
            ensure_directory(Path(report_path).parent)
            
            with open(report_path, 'w') as f:
                f.write(f"Model Evaluation Report\n")
                f.write(f"======================\n\n")
                f.write(f"Model Name: {result.model_name}\n")
                f.write(f"Evaluation Timestamp: {result.evaluation_timestamp}\n")
                f.write(f"Evaluation Time: {result.evaluation_time:.2f}s\n")
                f.write(f"Performance Grade: {result.performance_grade}\n")
                f.write(f"Evaluation Passed: {result.evaluation_passed}\n\n")
                
                if result.pre_hpo_metrics:
                    f.write(f"Pre-HPO Metrics:\n")
                    f.write(f"----------------\n")
                    self._write_metrics_to_file(f, result.pre_hpo_metrics)
                    f.write(f"\n")
                
                if result.post_hpo_metrics:
                    f.write(f"Post-HPO Metrics:\n")
                    f.write(f"-----------------\n")
                    self._write_metrics_to_file(f, result.post_hpo_metrics)
                    f.write(f"\n")
                
                if result.accuracy_improvement is not None:
                    f.write(f"Improvement Metrics:\n")
                    f.write(f"-------------------\n")
                    f.write(f"Accuracy Improvement: {result.accuracy_improvement:.4f}\n")
                    if result.f1_improvement is not None:
                        f.write(f"F1 Improvement: {result.f1_improvement:.4f}\n")
                    if result.r2_improvement is not None:
                        f.write(f"R2 Improvement: {result.r2_improvement:.4f}\n")
                    if result.sharpe_improvement is not None:
                        f.write(f"Sharpe Improvement: {result.sharpe_improvement:.4f}\n")
            
            self.logger.info(f"📊 Evaluation report generated: {report_path}")
            
        except Exception as e:
            self.logger.exception(f"💥 Error generating evaluation report: {e}")
    
    def _write_metrics_to_file(self, file, metrics: EvaluationMetrics):
        """Write metrics to file."""
        try:
            if metrics.accuracy is not None:
                file.write(f"Accuracy: {metrics.accuracy:.4f}\n")
            if metrics.precision is not None:
                file.write(f"Precision: {metrics.precision:.4f}\n")
            if metrics.recall is not None:
                file.write(f"Recall: {metrics.recall:.4f}\n")
            if metrics.f1_score is not None:
                file.write(f"F1 Score: {metrics.f1_score:.4f}\n")
            if metrics.roc_auc is not None:
                file.write(f"ROC AUC: {metrics.roc_auc:.4f}\n")
            if metrics.mse is not None:
                file.write(f"MSE: {metrics.mse:.4f}\n")
            if metrics.rmse is not None:
                file.write(f"RMSE: {metrics.rmse:.4f}\n")
            if metrics.mae is not None:
                file.write(f"MAE: {metrics.mae:.4f}\n")
            if metrics.r2_score is not None:
                file.write(f"R2 Score: {metrics.r2_score:.4f}\n")
            if metrics.sharpe_ratio is not None:
                file.write(f"Sharpe Ratio: {metrics.sharpe_ratio:.4f}\n")
            if metrics.max_drawdown is not None:
                file.write(f"Max Drawdown: {metrics.max_drawdown:.4f}\n")
            if metrics.win_rate is not None:
                file.write(f"Win Rate: {metrics.win_rate:.4f}\n")
            if metrics.profit_factor is not None:
                file.write(f"Profit Factor: {metrics.profit_factor:.4f}\n")
            if metrics.sample_count is not None:
                file.write(f"Sample Count: {metrics.sample_count}\n")
            if metrics.feature_count is not None:
                file.write(f"Feature Count: {metrics.feature_count}\n")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error writing metrics to file: {e}")