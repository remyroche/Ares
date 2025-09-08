"""
Financial metrics logging for Step09 HMM-Based Training Per Regime.
Independent logging module that can be used without the reporting system.

Enhanced with per-HMM regime logging and fail-fast validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from src.utils.financial_metrics_logger import (
    get_financial_metrics_logger, 
    financial_metrics_context,
    get_smart_financial_metrics_logger,
    log_financial_metric_with_regime_awareness
)
from src.utils.logger import system_logger

# Import enhanced functionality if available
try:
    from src.utils.enhanced_financial_metrics_logger import (
        get_enhanced_financial_metrics_logger,
        validate_and_log_regime_data
    )
    ENHANCED_LOGGING_AVAILABLE = True
except ImportError:
    ENHANCED_LOGGING_AVAILABLE = False
    get_enhanced_financial_metrics_logger = None
    validate_and_log_regime_data = None

logger = system_logger.getChild('Step09FinancialLogging')


class Step09FinancialLogger:
    """Independent financial metrics logger for Step09 HMM-Based Training Per Regime with enhanced regime logging."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str, enable_enhanced_logging: bool = True):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.enable_enhanced_logging = enable_enhanced_logging
        
        # Use smart logger that automatically chooses enhanced or base logger
        self.financial_logger = get_smart_financial_metrics_logger(use_enhanced=enable_enhanced_logging)
        
        # Store enhanced logger separately if available
        if ENHANCED_LOGGING_AVAILABLE and enable_enhanced_logging:
            self.enhanced_logger = get_enhanced_financial_metrics_logger()
        else:
            self.enhanced_logger = None
    
    def log_step_execution(self, training_results: Dict[str, Any], model_performance: Dict[str, Any], 
                          execution_data: Dict[str, Any], regime_models: Dict[str, Any], 
                          data: Optional[pd.DataFrame] = None) -> bool:
        """
        Log comprehensive financial metrics for Step09 execution with enhanced regime validation.
        
        Args:
            training_results: Training execution results
            model_performance: Model performance metrics
            execution_data: Execution metadata
            regime_models: Regime-specific model data
            data: DataFrame for regime validation (optional)
            
        Returns:
            True if logging succeeded, False if fail-fast conditions triggered
        """
        try:
            # Use enhanced logging if available and data is provided
            if self.enhanced_logger and data is not None:
                return self._log_with_enhanced_regime_validation(
                    training_results, model_performance, execution_data, regime_models, data
                )
            else:
                # Fallback to standard logging
                return self._log_with_standard_method(
                    training_results, model_performance, execution_data, regime_models
                )
        except Exception as e:
            logger.error(f"Failed to log financial metrics: {e}")
            return False
    
    def _log_with_enhanced_regime_validation(self, training_results: Dict[str, Any], 
                                           model_performance: Dict[str, Any], 
                                           execution_data: Dict[str, Any], 
                                           regime_models: Dict[str, Any], 
                                           data: pd.DataFrame) -> bool:
        """Log with enhanced regime validation and fail-fast checks."""
        try:
            # Validate regime data first
            if validate_and_log_regime_data:
                validation_success = validate_and_log_regime_data(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step09_HMM_Based_Training_Per_Regime",
                    data=data,
                    regime_column='composite_cluster_id'
                )
                
                if not validation_success:
                    logger.error("🚨 Regime validation failed for Step09")
                    return False
            
            # Log step start
            self.financial_logger.log_step_start("Step09_HMM_Based_Training_Per_Regime", self.symbol, self.exchange, self.timeframe)
            
            # Log all financial metrics with regime awareness
            success = self._log_financial_metrics_with_regime_awareness(
                training_results, model_performance, execution_data, regime_models, data
            )
            
            # Log file paths
            self._log_created_file_paths()
            
            # Log step end
            self.financial_logger.log_step_end(
                "Step09_HMM_Based_Training_Per_Regime", 
                self.symbol, 
                self.exchange, 
                self.timeframe, 
                success=success
            )
            
            return success
            
        except Exception as e:
            self.financial_logger.log_step_end(
                "Step09_HMM_Based_Training_Per_Regime", 
                self.symbol, 
                self.exchange, 
                self.timeframe, 
                success=False, 
                error_message=str(e)
            )
            logger.error(f"Enhanced regime validation logging failed: {e}")
            return False
    
    def _log_with_standard_method(self, training_results: Dict[str, Any], 
                                model_performance: Dict[str, Any], 
                                execution_data: Dict[str, Any], 
                                regime_models: Dict[str, Any]) -> bool:
        """Log using standard method (fallback)."""
        with financial_metrics_context(
            step_name="Step09_HMM_Based_Training_Per_Regime",
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe
        ):
            try:
                self.financial_logger.log_step_start("Step09_HMM_Based_Training_Per_Regime", self.symbol, self.exchange, self.timeframe)
                
                # Log all financial metrics
                self._log_financial_metrics_from_results(training_results, model_performance, execution_data, regime_models)
                
                # Log file paths
                self._log_created_file_paths()
                
                self.financial_logger.log_step_end("Step09_HMM_Based_Training_Per_Regime", self.symbol, self.exchange, self.timeframe, success=True)
                
                return True
                
            except Exception as e:
                self.financial_logger.log_step_end("Step09_HMM_Based_Training_Per_Regime", self.symbol, self.exchange, self.timeframe, success=False, error_message=str(e))
                logger.error(f"Failed to log financial metrics: {e}")
                return False
    
    def _log_financial_metrics_from_results(self, training_results: Dict[str, Any], model_performance: Dict[str, Any], 
                                          execution_data: Dict[str, Any], regime_models: Dict[str, Any]) -> None:
        """Log key financial metrics directly from step results."""
        try:
            # Note: Data quality and performance metrics are logged in regular system logs
            # Financial metrics logger focuses only on financial/trading metrics
            
            # Log training results metrics (financial relevance)
            if training_results:
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="total_models_trained",
                    metric_value=float(training_results.get('total_models_trained', 0)),
                    metric_type="performance",
                    step_name="Step09_HMM_Based_Training_Per_Regime"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="successful_trainings",
                    metric_value=float(training_results.get('successful_trainings', 0)),
                    metric_type="performance",
                    step_name="Step09_HMM_Based_Training_Per_Regime"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="failed_trainings",
                    metric_value=float(training_results.get('failed_trainings', 0)),
                    metric_type="performance",
                    step_name="Step09_HMM_Based_Training_Per_Regime"
                )
                
                # Log training success rate (financial relevance)
                total_trainings = training_results.get('total_models_trained', 1)
                successful_trainings = training_results.get('successful_trainings', 0)
                training_success_rate = successful_trainings / total_trainings
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="training_success_rate",
                    metric_value=training_success_rate,
                    metric_type="trading",
                    step_name="Step09_HMM_Based_Training_Per_Regime"
                )
            
            # Log model performance metrics
            if model_performance:
                # Log overall model performance
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="overall_model_accuracy",
                    metric_value=model_performance.get('overall_accuracy', 0.0),
                    metric_type="performance",
                    step_name="Step09_HMM_Based_Training_Per_Regime"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="overall_model_precision",
                    metric_value=model_performance.get('overall_precision', 0.0),
                    metric_type="performance",
                    step_name="Step09_HMM_Based_Training_Per_Regime"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="overall_model_recall",
                    metric_value=model_performance.get('overall_recall', 0.0),
                    metric_type="performance",
                    step_name="Step09_HMM_Based_Training_Per_Regime"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="overall_model_f1_score",
                    metric_value=model_performance.get('overall_f1_score', 0.0),
                    metric_type="performance",
                    step_name="Step09_HMM_Based_Training_Per_Regime"
                )
                
                # Log model stability metrics (financial relevance)
                if 'model_stability_score' in model_performance:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="model_stability_score",
                        metric_value=model_performance['model_stability_score'],
                        metric_type="trading",
                        step_name="Step09_HMM_Based_Training_Per_Regime"
                    )
                
                # Log ensemble performance if available
                if 'ensemble_performance' in model_performance:
                    ensemble_perf = model_performance['ensemble_performance']
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="ensemble_accuracy",
                        metric_value=ensemble_perf.get('accuracy', 0.0),
                        metric_type="performance",
                        step_name="Step09_HMM_Based_Training_Per_Regime"
                    )
                    
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="ensemble_diversity_score",
                        metric_value=ensemble_perf.get('diversity_score', 0.0),
                        metric_type="trading",
                        step_name="Step09_HMM_Based_Training_Per_Regime"
                    )
            
            # Log regime-specific model metrics
            if regime_models:
                for regime_id, regime_model_data in regime_models.items():
                    # Log regime model performance
                    if 'accuracy' in regime_model_data:
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"regime_{regime_id}_model_accuracy",
                            metric_value=regime_model_data['accuracy'],
                            metric_type="performance",
                            step_name="Step09_HMM_Based_Training_Per_Regime",
                            regime_id=str(regime_id)
                        )
                    
                    if 'precision' in regime_model_data:
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"regime_{regime_id}_model_precision",
                            metric_value=regime_model_data['precision'],
                            metric_type="performance",
                            step_name="Step09_HMM_Based_Training_Per_Regime",
                            regime_id=str(regime_id)
                        )
                    
                    if 'recall' in regime_model_data:
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"regime_{regime_id}_model_recall",
                            metric_value=regime_model_data['recall'],
                            metric_type="performance",
                            step_name="Step09_HMM_Based_Training_Per_Regime",
                            regime_id=str(regime_id)
                        )
                    
                    if 'f1_score' in regime_model_data:
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"regime_{regime_id}_model_f1_score",
                            metric_value=regime_model_data['f1_score'],
                            metric_type="performance",
                            step_name="Step09_HMM_Based_Training_Per_Regime",
                            regime_id=str(regime_id)
                        )
                    
                    # Log regime model training metrics
                    if 'training_time' in regime_model_data:
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"regime_{regime_id}_training_time",
                            metric_value=regime_model_data['training_time'],
                            metric_type="performance",
                            step_name="Step09_HMM_Based_Training_Per_Regime",
                            regime_id=str(regime_id)
                        )
                    
                    if 'convergence_score' in regime_model_data:
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"regime_{regime_id}_convergence_score",
                            metric_value=regime_model_data['convergence_score'],
                            metric_type="trading",
                            step_name="Step09_HMM_Based_Training_Per_Regime",
                            regime_id=str(regime_id)
                        )
                    
                    if 'training_samples' in regime_model_data:
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"regime_{regime_id}_training_samples",
                            metric_value=float(regime_model_data['training_samples']),
                            metric_type="performance",
                            step_name="Step09_HMM_Based_Training_Per_Regime",
                            regime_id=str(regime_id)
                        )
                    
                    # Log regime model feature importance
                    if 'feature_importance' in regime_model_data:
                        feature_importance = regime_model_data['feature_importance']
                        for feature_name, importance in feature_importance.items():
                            self.financial_logger.log_financial_metric(
                                symbol=self.symbol,
                                exchange=self.exchange,
                                timeframe=self.timeframe,
                                metric_name=f"regime_{regime_id}_feature_importance_{feature_name}",
                                metric_value=importance,
                                metric_type="feature",
                                step_name="Step09_HMM_Based_Training_Per_Regime",
                                regime_id=str(regime_id),
                                additional_data={'feature_name': feature_name}
                            )
            
            # Note: Execution performance metrics are logged in regular system logs
            # Financial metrics logger focuses only on financial/trading metrics
            
            # Log comprehensive trading performance estimation
            if training_results and model_performance:
                # Estimate trading performance based on model training results
                overall_accuracy = model_performance.get('overall_accuracy', 0.5)
                training_success_rate = training_results.get('successful_trainings', 0) / max(training_results.get('total_models_trained', 1), 1)
                
                # Estimate returns based on model performance
                estimated_return = (overall_accuracy * 0.02) - ((1 - overall_accuracy) * 0.01)  # Rough estimate
                estimated_volatility = 0.02  # Default estimate
                
                estimated_performance = {
                    'total_return': estimated_return,
                    'annualized_return': estimated_return * 252,  # Assuming daily signals
                    'volatility': estimated_volatility,
                    'sharpe_ratio': estimated_return / estimated_volatility if estimated_volatility > 0 else 0.0,
                    'sortino_ratio': estimated_return / (estimated_volatility * 0.5) if estimated_volatility > 0 else 0.0,
                    'calmar_ratio': 0.0,  # Would need max drawdown
                    'max_drawdown': estimated_volatility * 2,  # Estimate
                    'max_drawdown_duration': 25,  # Default estimate
                    'var_95': estimated_volatility * 1.5,  # Estimate
                    'cvar_95': estimated_volatility * 2,  # Estimate
                    'win_rate': overall_accuracy,
                    'profit_factor': 1.0 + (overall_accuracy - 0.5) * 2,  # Estimate based on accuracy
                    'avg_win': 0.02,  # Default estimate
                    'avg_loss': 0.01,  # Default estimate
                    'largest_win': 0.05,  # Default estimate
                    'largest_loss': estimated_volatility * 2,  # Estimate
                    'total_trades': 30,  # Default estimate
                    'winning_trades': int(30 * overall_accuracy),
                    'losing_trades': int(30 * (1 - overall_accuracy)),
                    'additional_metrics': {
                        'total_models_trained': training_results.get('total_models_trained', 0),
                        'training_success_rate': training_success_rate,
                        'overall_model_accuracy': overall_accuracy,
                        'overall_model_f1_score': model_performance.get('overall_f1_score', 0.0),
                        'model_stability_score': model_performance.get('model_stability_score', 0.0)
                    }
                }
                
                self.financial_logger.log_trading_performance(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step09_HMM_Based_Training_Per_Regime",
                    **estimated_performance
                )
            
        except Exception as e:
            logger.error(f"Failed to log financial metrics from results: {e}")
    
    def _log_financial_metrics_with_regime_awareness(self, training_results: Dict[str, Any], 
                                                   model_performance: Dict[str, Any], 
                                                   execution_data: Dict[str, Any], 
                                                   regime_models: Dict[str, Any], 
                                                   data: pd.DataFrame) -> bool:
        """Log financial metrics with enhanced regime awareness and fail-fast validation."""
        try:
            success = True
            
            # Log training results metrics with regime awareness
            if training_results:
                success &= log_financial_metric_with_regime_awareness(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="total_models_trained",
                    metric_value=float(training_results.get('total_models_trained', 0)),
                    metric_type="performance",
                    step_name="Step09_HMM_Based_Training_Per_Regime",
                    data=data
                )
                
                success &= log_financial_metric_with_regime_awareness(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="training_success_rate",
                    metric_value=float(training_results.get('successful_trainings', 0)) / max(training_results.get('total_models_trained', 1), 1),
                    metric_type="trading",
                    step_name="Step09_HMM_Based_Training_Per_Regime",
                    data=data
                )
            
            # Log model performance metrics with regime awareness
            if model_performance:
                success &= log_financial_metric_with_regime_awareness(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="overall_model_accuracy",
                    metric_value=model_performance.get('overall_accuracy', 0.0),
                    metric_type="performance",
                    step_name="Step09_HMM_Based_Training_Per_Regime",
                    data=data
                )
            
            # Log regime-specific model metrics with enhanced validation
            if regime_models and self.enhanced_logger:
                regime_metrics = {}
                for regime_id, regime_model_data in regime_models.items():
                    regime_metrics[str(regime_id)] = {
                        'accuracy': regime_model_data.get('accuracy', 0.0),
                        'precision': regime_model_data.get('precision', 0.0),
                        'recall': regime_model_data.get('recall', 0.0),
                        'f1_score': regime_model_data.get('f1_score', 0.0),
                        'training_samples': regime_model_data.get('training_samples', 0)
                    }
                
                # Use enhanced logger for per-regime metrics
                success &= self.enhanced_logger.log_per_regime_metrics(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step09_HMM_Based_Training_Per_Regime",
                    regime_metrics=regime_metrics,
                    data=data
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to log financial metrics with regime awareness: {e}")
            return False
    
    def _log_created_file_paths(self) -> None:
        """Log file paths that were created during this step."""
        try:
            if hasattr(self.financial_logger, 'current_file_path') and self.financial_logger.current_file_path:
                logger.info(f"📁 Financial metrics file created: {self.financial_logger.current_file_path}")
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="metrics_file_path",
                    metric_value=0.0,
                    metric_type="file_path",
                    step_name="Step09_HMM_Based_Training_Per_Regime",
                    additional_data={'file_path': str(self.financial_logger.current_file_path)}
                )
            logger.info("📁 File paths logged for Step09")
        except Exception as e:
            logger.warning(f"Could not log file paths: {e}")


# Enhanced Step09 Financial Logger with Regime-Aware Decorator Support
class EnhancedStep09FinancialLogger(Step09FinancialLogger):
    """Enhanced Step09 Financial Logger with automatic regime-aware logging decorator support."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str, enable_enhanced_logging: bool = True):
        super().__init__(symbol, exchange, timeframe, enable_enhanced_logging)
        
        # Import regime-aware decorator if available
        try:
            from src.utils.regime_aware_financial_logging_decorator import (
                regime_aware_financial_logging,
                auto_regime_aware_logging
            )
            self.regime_aware_decorator = regime_aware_financial_logging
            self.auto_regime_aware_decorator = auto_regime_aware_logging
            self.decorator_available = True
        except ImportError:
            self.decorator_available = False
    
    def get_decorated_execute_method(self, original_execute_method):
        """Get the execute method decorated with regime-aware logging."""
        if self.decorator_available:
            return self.auto_regime_aware_decorator(
                enable_regime_validation=True,
                enable_fail_fast=True,
                min_regime_samples=100,
                max_regime_imbalance=0.8,
                regime_column='composite_cluster_id',
                min_data_quality=0.7
            )(original_execute_method)
        else:
            return original_execute_method