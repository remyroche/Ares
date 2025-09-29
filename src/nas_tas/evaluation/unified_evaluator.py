"""
Unified Evaluator for NAS/TAS Systems

This module provides a comprehensive evaluation framework that consolidates
evaluation logic previously scattered across NAS and TAS implementations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer,
    tprint_structured, tprint_with_level, tprint_logged, LogLevel
)

from .financial_metrics import (
    FinancialMetricsCalculator,
    TradingPerformanceMetrics,
    RiskMetrics,
    FinancialValidationResult
)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    # Performance metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    precision_recall_auc: float = 0.0
    
    # Regression metrics
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0
    
    # Financial metrics
    financial_metrics: TradingPerformanceMetrics = field(default_factory=TradingPerformanceMetrics)
    risk_metrics: RiskMetrics = field(default_factory=RiskMetrics)
    
    # Regime-specific metrics
    regime_accuracy: float = 0.0
    regime_stability: float = 0.0
    adaptation_speed: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    evaluation_timestamp: datetime = field(default_factory=datetime.now)
    evaluation_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        
        # Basic metrics
        for field_name, field_value in self.__dict__.items():
            if field_name in ['financial_metrics', 'risk_metrics', 'evaluation_timestamp']:
                continue
            elif isinstance(field_value, (int, float)):
                result[field_name] = float(field_value)
            else:
                result[field_name] = field_value
        
        # Nested objects
        result['financial_metrics'] = self.financial_metrics.to_dict()
        result['risk_metrics'] = self.risk_metrics.to_dict()
        result['evaluation_timestamp'] = self.evaluation_timestamp.isoformat()
        
        return result


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    
    # Evaluation type
    evaluation_type: str = "comprehensive"  # comprehensive, quick, financial, performance
    
    # Data splitting
    validation_split: float = 0.2
    test_split: float = 0.2
    cv_folds: int = 5
    
    # Performance metrics
    calculate_performance_metrics: bool = True
    calculate_financial_metrics: bool = True
    calculate_regime_metrics: bool = True
    calculate_risk_metrics: bool = True
    
    # Financial validation
    financial_validation: bool = True
    financial_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'min_sharpe_ratio': 1.0,
        'max_drawdown': 0.15,
        'min_win_rate': 0.4,
        'min_profit_factor': 1.2
    })
    
    # Statistical validation
    statistical_tests: bool = True
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    
    # Parallel processing
    enable_parallel_evaluation: bool = True
    max_workers: int = 4
    
    # Custom evaluation functions
    custom_evaluators: List[Callable] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        config_dict = {}
        for field_name, field_value in self.__dict__.items():
            if field_name == 'custom_evaluators':
                config_dict[field_name] = [func.__name__ for func in field_value]
            else:
                config_dict[field_name] = field_value
        return config_dict


@dataclass
class EvaluationResult:
    """Result of comprehensive evaluation."""
    
    # Evaluation status
    evaluation_successful: bool = False
    evaluation_score: float = 0.0
    
    # Metrics
    metrics: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    
    # Validation results
    financial_validation: Optional[FinancialValidationResult] = None
    performance_validation: bool = False
    regime_validation: bool = False
    
    # Model comparison
    model_comparison: Dict[str, float] = field(default_factory=dict)
    
    # Error analysis
    error_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    evaluation_timestamp: datetime = field(default_factory=datetime.now)
    evaluation_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'evaluation_successful': self.evaluation_successful,
            'evaluation_score': self.evaluation_score,
            'metrics': self.metrics.to_dict(),
            'performance_validation': self.performance_validation,
            'regime_validation': self.regime_validation,
            'model_comparison': self.model_comparison,
            'error_analysis': self.error_analysis,
            'recommendations': self.recommendations,
            'evaluation_timestamp': self.evaluation_timestamp.isoformat(),
            'evaluation_duration': self.evaluation_duration
        }
        
        if self.financial_validation:
            result['financial_validation'] = self.financial_validation.to_dict()
        
        return result


class EvaluationStrategy(ABC):
    """Abstract base class for evaluation strategies."""
    
    @abstractmethod
    def evaluate(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray,
        config: EvaluationConfig
    ) -> EvaluationMetrics:
        """Evaluate model and return metrics."""
        pass


class PerformanceEvaluationStrategy(EvaluationStrategy):
    """Strategy for performance evaluation."""
    
    def evaluate(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray,
        config: EvaluationConfig
    ) -> EvaluationMetrics:
        """Evaluate model performance."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, precision_recall_curve, auc,
            mean_squared_error, mean_absolute_error, r2_score
        )
        
        tprint_info("Evaluating model performance")
        
        try:
            # Make predictions
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred = model.predict(X)
                y_pred_proba = None
            
            metrics = EvaluationMetrics()
            
            # Classification metrics
            if len(np.unique(y)) == 2:  # Binary classification
                metrics.accuracy = accuracy_score(y, y_pred)
                metrics.precision = precision_score(y, y_pred, zero_division=0)
                metrics.recall = recall_score(y, y_pred, zero_division=0)
                metrics.f1_score = f1_score(y, y_pred, zero_division=0)
                
                if y_pred_proba is not None:
                    try:
                        metrics.roc_auc = roc_auc_score(y, y_pred_proba)
                        precision_vals, recall_vals, _ = precision_recall_curve(y, y_pred_proba)
                        metrics.precision_recall_auc = auc(recall_vals, precision_vals)
                    except ValueError:
                        pass  # Handle edge cases
            
            else:  # Regression
                metrics.mse = mean_squared_error(y, y_pred)
                metrics.rmse = np.sqrt(metrics.mse)
                metrics.mae = mean_absolute_error(y, y_pred)
                metrics.r2_score = r2_score(y, y_pred)
            
            tprint_success(f"Performance evaluation completed: Accuracy={metrics.accuracy:.3f}, "
                          f"F1={metrics.f1_score:.3f}")
            
            return metrics
            
        except Exception as e:
            tprint_error(f"Error in performance evaluation: {e}")
            return EvaluationMetrics()


class FinancialEvaluationStrategy(EvaluationStrategy):
    """Strategy for financial evaluation."""
    
    def __init__(self, financial_calculator: Optional[FinancialMetricsCalculator] = None):
        self.financial_calculator = financial_calculator or FinancialMetricsCalculator()
    
    def evaluate(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray,
        config: EvaluationConfig
    ) -> EvaluationMetrics:
        """Evaluate financial performance."""
        tprint_info("Evaluating financial performance")
        
        try:
            # Make predictions
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)[:, 1]
            else:
                y_pred = model.predict(X)
                y_pred_proba = y_pred  # Use predictions as probabilities
            
            # Calculate returns (assuming y contains returns or prices)
            if len(y) > 1:
                # If y is already returns, use it directly
                if np.all(np.abs(y) <= 1):  # Likely returns
                    returns = y
                else:  # Likely prices, calculate returns
                    returns = np.diff(y) / y[:-1]
                
                # Calculate financial metrics
                financial_metrics = self.financial_calculator.calculate_performance_metrics(returns)
                risk_metrics = self.financial_calculator.calculate_risk_metrics(returns)
                
                metrics = EvaluationMetrics()
                metrics.financial_metrics = financial_metrics
                metrics.risk_metrics = risk_metrics
                
                tprint_success(f"Financial evaluation completed: Sharpe={financial_metrics.sharpe_ratio:.3f}, "
                              f"Max DD={financial_metrics.max_drawdown:.3f}")
                
                return metrics
            else:
                tprint_warning("Insufficient data for financial evaluation")
                return EvaluationMetrics()
                
        except Exception as e:
            tprint_error(f"Error in financial evaluation: {e}")
            return EvaluationMetrics()


class RegimeEvaluationStrategy(EvaluationStrategy):
    """Strategy for regime-specific evaluation."""
    
    def evaluate(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray,
        config: EvaluationConfig
    ) -> EvaluationMetrics:
        """Evaluate regime-specific performance."""
        tprint_info("Evaluating regime performance")
        
        try:
            metrics = EvaluationMetrics()
            
            # Regime detection using volatility clustering
            if len(y) > 50:  # Need sufficient data for regime analysis
                # Calculate rolling volatility as regime indicator
                window_size = min(20, len(y) // 4)
                rolling_vol = pd.Series(y).rolling(window=window_size).std()
                
                # Detect regime changes using volatility thresholds
                vol_threshold = rolling_vol.quantile(0.7)  # 70th percentile as threshold
                regimes = (rolling_vol > vol_threshold).astype(int)
                
                # Calculate regime accuracy (how well model adapts to regime changes)
                if hasattr(model, 'predict'):
                    predictions = model.predict(X)
                    if len(predictions) == len(regimes):
                        # Calculate accuracy within each regime
                        regime_0_mask = regimes == 0
                        regime_1_mask = regimes == 1
                        
                        if np.sum(regime_0_mask) > 0 and np.sum(regime_1_mask) > 0:
                            # Calculate regime-specific accuracy
                            regime_0_acc = self._calculate_regime_accuracy(
                                y[regime_0_mask], predictions[regime_0_mask]
                            )
                            regime_1_acc = self._calculate_regime_accuracy(
                                y[regime_1_mask], predictions[regime_1_mask]
                            )
                            
                            # Overall regime accuracy as weighted average
                            metrics.regime_accuracy = (
                                regime_0_acc * np.sum(regime_0_mask) + 
                                regime_1_acc * np.sum(regime_1_mask)
                            ) / len(regimes)
                        else:
                            metrics.regime_accuracy = 0.5  # Default if no regime separation
                    else:
                        metrics.regime_accuracy = 0.5
                else:
                    metrics.regime_accuracy = 0.5
                
                # Calculate regime stability (consistency of regime predictions)
                regime_changes = np.sum(np.diff(regimes) != 0)
                total_periods = len(regimes) - 1
                metrics.regime_stability = 1.0 - (regime_changes / max(total_periods, 1))
                
                # Calculate adaptation speed (how quickly model responds to regime changes)
                if regime_changes > 0:
                    # Measure prediction consistency around regime changes
                    change_points = np.where(np.diff(regimes) != 0)[0]
                    adaptation_scores = []
                    
                    for change_point in change_points:
                        # Look at prediction consistency before and after change
                        before_window = max(0, change_point - 5)
                        after_window = min(len(predictions), change_point + 6)
                        
                        if after_window - before_window > 5:
                            before_preds = predictions[before_window:change_point]
                            after_preds = predictions[change_point:after_window]
                            
                            # Calculate consistency (lower variance = better adaptation)
                            before_consistency = 1.0 - np.std(before_preds) if len(before_preds) > 1 else 1.0
                            after_consistency = 1.0 - np.std(after_preds) if len(after_preds) > 1 else 1.0
                            
                            adaptation_scores.append((before_consistency + after_consistency) / 2)
                    
                    metrics.adaptation_speed = np.mean(adaptation_scores) if adaptation_scores else 0.5
                else:
                    metrics.adaptation_speed = 1.0  # No regime changes = perfect adaptation
            else:
                # Insufficient data for regime analysis
                metrics.regime_accuracy = 0.5
                metrics.regime_stability = 0.5
                metrics.adaptation_speed = 0.5
            
            tprint_success(f"Regime evaluation completed: Accuracy={metrics.regime_accuracy:.3f}, "
                          f"Stability={metrics.regime_stability:.3f}, "
                          f"Adaptation={metrics.adaptation_speed:.3f}")
            
            return metrics
            
        except Exception as e:
            tprint_error(f"Error in regime evaluation: {e}")
            return EvaluationMetrics()
    
    def _calculate_regime_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy for a specific regime."""
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.5
        
        # For regression, use R² score
        if len(np.unique(y_true)) > 2:
            from sklearn.metrics import r2_score
            try:
                return max(0, r2_score(y_true, y_pred))
            except:
                return 0.5
        else:
            # For classification, use accuracy
            from sklearn.metrics import accuracy_score
            try:
                return accuracy_score(y_true, y_pred)
            except:
                return 0.5


class UnifiedEvaluator:
    """
    Unified evaluator for NAS/TAS systems.
    
    This class consolidates evaluation logic that was previously scattered
    across NAS and TAS implementations, providing a comprehensive evaluation
    framework with multiple strategies.
    """
    
    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        financial_calculator: Optional[FinancialMetricsCalculator] = None
    ):
        """
        Initialize unified evaluator.
        
        Args:
            config: Evaluation configuration
            financial_calculator: Financial metrics calculator
        """
        tprint_info("Initializing Unified Evaluator")
        
        self.config = config or EvaluationConfig()
        self.financial_calculator = financial_calculator or FinancialMetricsCalculator()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Log configuration
        tprint_structured({
            "evaluation_config": self.config.to_dict()
        }, LogLevel.INFO)
        
        # Initialize evaluation strategies
        tprint_debug("Initializing evaluation strategies")
        self.strategies = {
            'performance': PerformanceEvaluationStrategy(),
            'financial': FinancialEvaluationStrategy(self.financial_calculator),
            'regime': RegimeEvaluationStrategy()
        }
        
        tprint_success(f"Unified evaluator initialized with {len(self.strategies)} strategies")
    
    async def evaluate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        benchmark_model: Optional[Any] = None,
        benchmark_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> EvaluationResult:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Targets
            benchmark_model: Optional benchmark model for comparison
            benchmark_data: Optional benchmark data (X, y)
            
        Returns:
            EvaluationResult with comprehensive evaluation
        """
        tprint_info("Starting comprehensive model evaluation")
        start_time = datetime.now()
        
        # Log evaluation parameters
        tprint_structured({
            "evaluation_parameters": {
                "model_type": type(model).__name__,
                "data_shape": X.shape,
                "target_shape": y.shape,
                "benchmark_model": type(benchmark_model).__name__ if benchmark_model else None,
                "benchmark_data_shape": benchmark_data[0].shape if benchmark_data else None,
                "evaluation_type": self.config.evaluation_type,
                "enable_parallel_evaluation": self.config.enable_parallel_evaluation
            }
        }, LogLevel.INFO)
        
        try:
            result = EvaluationResult()
            
            # Validate inputs
            tprint_debug("Validating evaluation inputs")
            if len(X) != len(y):
                error_msg = f"X and y must have same length: {len(X)} vs {len(y)}"
                tprint_error(error_msg)
                raise ValueError(error_msg)
            
            if len(X) == 0:
                error_msg = "Empty dataset provided"
                tprint_error(error_msg)
                raise ValueError(error_msg)
            
            tprint_success("Input validation passed")
            
            # Perform evaluations based on configuration
            tprint_debug("Setting up evaluation tasks")
            evaluation_tasks = []
            
            if self.config.calculate_performance_metrics:
                tprint_debug("Adding performance evaluation task")
                evaluation_tasks.append(('performance', self._evaluate_performance(model, X, y)))
            
            if self.config.calculate_financial_metrics:
                tprint_debug("Adding financial evaluation task")
                evaluation_tasks.append(('financial', self._evaluate_financial(model, X, y)))
            
            if self.config.calculate_regime_metrics:
                tprint_debug("Adding regime evaluation task")
                evaluation_tasks.append(('regime', self._evaluate_regime(model, X, y)))
            
            tprint_success(f"Configured {len(evaluation_tasks)} evaluation tasks")
            
            # Execute evaluations
            tprint_info("Executing evaluation tasks")
            if self.config.enable_parallel_evaluation and len(evaluation_tasks) > 1:
                tprint_debug("Using parallel evaluation")
                with tprint_timer("parallel_evaluation", LogLevel.INFO):
                    metrics_results = await self._execute_parallel_evaluations(model, X, y, evaluation_tasks)
                tprint_success("Parallel evaluation completed")
            else:
                tprint_debug("Using sequential evaluation")
                with tprint_timer("sequential_evaluation", LogLevel.INFO):
                    metrics_results = await self._execute_sequential_evaluations(model, X, y, evaluation_tasks)
                tprint_success("Sequential evaluation completed")
            
            # Combine metrics
            tprint_debug("Combining evaluation metrics")
            combined_metrics = self._combine_metrics(metrics_results)
            result.metrics = combined_metrics
            tprint_success("Metrics combined successfully")
            
            # Financial validation
            if self.config.financial_validation and self.config.calculate_financial_metrics:
                tprint_debug("Performing financial validation")
                with tprint_timer("financial_validation", LogLevel.DEBUG):
                    result.financial_validation = await self._validate_financial_performance(combined_metrics)
                tprint_success("Financial validation completed")
            
            # Performance validation
            if self.config.calculate_performance_metrics:
                tprint_debug("Performing performance validation")
                with tprint_timer("performance_validation", LogLevel.DEBUG):
                    result.performance_validation = self._validate_performance(combined_metrics)
                tprint_success("Performance validation completed")
            
            # Regime validation
            if self.config.calculate_regime_metrics:
                tprint_debug("Performing regime validation")
                with tprint_timer("regime_validation", LogLevel.DEBUG):
                    result.regime_validation = self._validate_regime_performance(combined_metrics)
                tprint_success("Regime validation completed")
            
            # Model comparison
            if benchmark_model is not None and benchmark_data is not None:
                tprint_debug("Performing model comparison")
                with tprint_timer("model_comparison", LogLevel.DEBUG):
                    result.model_comparison = await self._compare_models(model, benchmark_model, X, y, benchmark_data)
                tprint_success("Model comparison completed")
            
            # Error analysis
            tprint_debug("Performing error analysis")
            with tprint_timer("error_analysis", LogLevel.DEBUG):
                result.error_analysis = self._analyze_errors(model, X, y)
            tprint_success("Error analysis completed")
            
            # Generate recommendations
            tprint_debug("Generating recommendations")
            with tprint_timer("recommendations_generation", LogLevel.DEBUG):
                result.recommendations = self._generate_recommendations(result)
            tprint_success("Recommendations generated")
            
            # Calculate overall evaluation score
            tprint_debug("Calculating overall evaluation score")
            result.evaluation_score = self._calculate_evaluation_score(result)
            result.evaluation_successful = result.evaluation_score >= 0.7
            
            tprint_structured({
                "evaluation_results": {
                    "evaluation_score": result.evaluation_score,
                    "evaluation_successful": result.evaluation_successful,
                    "metrics_count": len(combined_metrics.to_dict()) if hasattr(combined_metrics, 'to_dict') else 0,
                    "has_financial_validation": hasattr(result, 'financial_validation') and result.financial_validation is not None,
                    "has_performance_validation": hasattr(result, 'performance_validation') and result.performance_validation is not None,
                    "has_regime_validation": hasattr(result, 'regime_validation') and result.regime_validation is not None,
                    "has_model_comparison": hasattr(result, 'model_comparison') and result.model_comparison is not None
                }
            }, LogLevel.INFO)
            
            # Calculate duration
            result.evaluation_duration = (datetime.now() - start_time).total_seconds()
            
            tprint_success(f"Model evaluation completed: {'SUCCESS' if result.evaluation_successful else 'FAILED'} "
                          f"(Score: {result.evaluation_score:.3f}, Duration: {result.evaluation_duration:.2f}s)")
            
            return result
            
        except Exception as e:
            tprint_error(f"Error during model evaluation: {e}")
            tprint_structured({
                "evaluation_error": {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "evaluation_duration_seconds": (datetime.now() - start_time).total_seconds(),
                    "timestamp": datetime.now().isoformat()
                }
            }, LogLevel.ERROR)
            self.logger.error(f"Error during model evaluation: {e}", exc_info=True)
            
            result = EvaluationResult()
            result.evaluation_successful = False
            result.evaluation_duration = (datetime.now() - start_time).total_seconds()
            return result
    
    async def _execute_parallel_evaluations(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        evaluation_tasks: List[Tuple[str, str]]
    ) -> Dict[str, EvaluationMetrics]:
        """Execute evaluations in parallel."""
        tprint_info(f"Executing {len(evaluation_tasks)} evaluations in parallel")
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit tasks
            future_to_task = {
                executor.submit(self.strategies[task_name].evaluate, model, X, y, self.config): task_name
                for task_name, _ in evaluation_tasks
            }
            
            # Collect results
            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    result = future.result()
                    results[task_name] = result
                    tprint_debug(f"Completed {task_name} evaluation")
                except Exception as e:
                    tprint_error(f"Error in {task_name} evaluation: {e}")
                    results[task_name] = EvaluationMetrics()
        
        return results
    
    async def _execute_sequential_evaluations(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        evaluation_tasks: List[Tuple[str, str]]
    ) -> Dict[str, EvaluationMetrics]:
        """Execute evaluations sequentially."""
        tprint_info(f"Executing {len(evaluation_tasks)} evaluations sequentially")
        
        results = {}
        
        for task_name, _ in evaluation_tasks:
            try:
                result = self.strategies[task_name].evaluate(model, X, y, self.config)
                results[task_name] = result
                tprint_debug(f"Completed {task_name} evaluation")
            except Exception as e:
                tprint_error(f"Error in {task_name} evaluation: {e}")
                results[task_name] = EvaluationMetrics()
        
        return results
    
    def _combine_metrics(self, metrics_results: Dict[str, EvaluationMetrics]) -> EvaluationMetrics:
        """Combine metrics from different evaluation strategies."""
        combined = EvaluationMetrics()
        
        for strategy_name, metrics in metrics_results.items():
            if strategy_name == 'performance':
                combined.accuracy = metrics.accuracy
                combined.precision = metrics.precision
                combined.recall = metrics.recall
                combined.f1_score = metrics.f1_score
                combined.roc_auc = metrics.roc_auc
                combined.precision_recall_auc = metrics.precision_recall_auc
                combined.mse = metrics.mse
                combined.rmse = metrics.rmse
                combined.mae = metrics.mae
                combined.r2_score = metrics.r2_score
            
            elif strategy_name == 'financial':
                combined.financial_metrics = metrics.financial_metrics
                combined.risk_metrics = metrics.risk_metrics
            
            elif strategy_name == 'regime':
                combined.regime_accuracy = metrics.regime_accuracy
                combined.regime_stability = metrics.regime_stability
                combined.adaptation_speed = metrics.adaptation_speed
        
        return combined
    
    async def _validate_financial_performance(self, metrics: EvaluationMetrics) -> FinancialValidationResult:
        """Validate financial performance."""
        if metrics.financial_metrics.total_return == 0:
            return FinancialValidationResult()
        
        # Extract returns from financial metrics (simplified)
        returns = np.array([metrics.financial_metrics.total_return])  # Simplified for demo
        
        return self.financial_calculator.validate_financial_performance(
            returns,
            self.config.financial_thresholds
        )
    
    def _validate_performance(self, metrics: EvaluationMetrics) -> bool:
        """Validate performance metrics."""
        thresholds = {
            'min_accuracy': 0.6,
            'min_f1_score': 0.5,
            'min_roc_auc': 0.7
        }
        
        checks = [
            metrics.accuracy >= thresholds['min_accuracy'],
            metrics.f1_score >= thresholds['min_f1_score'],
            metrics.roc_auc >= thresholds['min_roc_auc'] if metrics.roc_auc > 0 else True
        ]
        
        return sum(checks) / len(checks) >= 0.67  # Pass if 2/3 checks pass
    
    def _validate_regime_performance(self, metrics: EvaluationMetrics) -> bool:
        """Validate regime performance metrics."""
        return (
            metrics.regime_accuracy >= 0.6 and
            metrics.regime_stability >= 0.7
        )
    
    async def _compare_models(
        self,
        model: Any,
        benchmark_model: Any,
        X: np.ndarray,
        y: np.ndarray,
        benchmark_data: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        """Compare model with benchmark."""
        tprint_info("Comparing model with benchmark")
        
        try:
            # Evaluate both models
            model_result = await self.evaluate_model(model, X, y)
            benchmark_result = await self.evaluate_model(benchmark_model, benchmark_data[0], benchmark_data[1])
            
            # Calculate relative performance
            comparison = {
                'accuracy_improvement': model_result.metrics.accuracy - benchmark_result.metrics.accuracy,
                'f1_improvement': model_result.metrics.f1_score - benchmark_result.metrics.f1_score,
                'sharpe_improvement': (
                    model_result.metrics.financial_metrics.sharpe_ratio - 
                    benchmark_result.metrics.financial_metrics.sharpe_ratio
                ),
                'drawdown_reduction': (
                    benchmark_result.metrics.financial_metrics.max_drawdown - 
                    model_result.metrics.financial_metrics.max_drawdown
                )
            }
            
            return comparison
            
        except Exception as e:
            tprint_error(f"Error in model comparison: {e}")
            return {}
    
    def _analyze_errors(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction errors."""
        try:
            y_pred = model.predict(X)
            
            # Calculate residuals
            residuals = y - y_pred
            
            error_analysis = {
                'mean_absolute_error': np.mean(np.abs(residuals)),
                'mean_squared_error': np.mean(residuals ** 2),
                'error_std': np.std(residuals),
                'error_skewness': float(stats.skew(residuals)) if len(residuals) > 2 else 0.0,
                'error_kurtosis': float(stats.kurtosis(residuals)) if len(residuals) > 3 else 0.0,
                'outlier_percentage': np.mean(np.abs(residuals) > 2 * np.std(residuals)) * 100
            }
            
            return error_analysis
            
        except Exception as e:
            tprint_error(f"Error in error analysis: {e}")
            return {}
    
    def _generate_recommendations(self, result: EvaluationResult) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        # Performance recommendations
        if result.metrics.accuracy < 0.7:
            recommendations.append("Improve model accuracy through feature engineering or hyperparameter tuning")
        
        if result.metrics.f1_score < 0.6:
            recommendations.append("Address class imbalance or improve precision/recall balance")
        
        # Financial recommendations
        if result.financial_validation and not result.financial_validation.passed_validation:
            recommendations.extend(result.financial_validation.recommendations)
        
        # Regime recommendations
        if result.metrics.regime_accuracy < 0.7:
            recommendations.append("Improve regime detection accuracy")
        
        if result.metrics.regime_stability < 0.8:
            recommendations.append("Enhance regime stability through better feature selection")
        
        return recommendations
    
    def _calculate_evaluation_score(self, result: EvaluationResult) -> float:
        """Calculate overall evaluation score."""
        scores = []
        
        # Performance score
        if result.metrics.accuracy > 0:
            scores.append(min(result.metrics.accuracy, 1.0))
        
        if result.metrics.f1_score > 0:
            scores.append(min(result.metrics.f1_score, 1.0))
        
        # Financial score
        if result.financial_validation:
            scores.append(result.financial_validation.validation_score)
        
        # Regime score
        if result.metrics.regime_accuracy > 0:
            scores.append(result.metrics.regime_accuracy)
        
        return np.mean(scores) if scores else 0.0