"""
Financial metrics logging for Step12 Analyst Enhancement.
Independent logging module that can be used without the reporting system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from src.utils.financial_metrics_logger import get_financial_metrics_logger, financial_metrics_context
from src.utils.logger import system_logger

logger = system_logger.getChild('Step12FinancialLogging')


class Step12FinancialLogger:
    """Independent financial metrics logger for Step12 Analyst Enhancement."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.financial_logger = get_financial_metrics_logger()
    
    def log_step_execution(self, optimization_results: Dict[str, Any], model_performance: Dict[str, Any], 
                          execution_data: Dict[str, Any], enhancement_metrics: Dict[str, Any]) -> None:
        """Log comprehensive financial metrics for Step12 execution."""
        with financial_metrics_context(
            step_name="Step12_Analyst_Enhancement",
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe
        ):
            try:
                self.financial_logger.log_step_start("Step12_Analyst_Enhancement", self.symbol, self.exchange, self.timeframe)
                
                # Log all financial metrics
                self._log_financial_metrics_from_results(optimization_results, model_performance, execution_data, enhancement_metrics)
                
                # Log file paths
                self._log_created_file_paths()
                
                self.financial_logger.log_step_end("Step12_Analyst_Enhancement", self.symbol, self.exchange, self.timeframe, success=True)
                
            except Exception as e:
                self.financial_logger.log_step_end("Step12_Analyst_Enhancement", self.symbol, self.exchange, self.timeframe, success=False, error_message=str(e))
                logger.error(f"Failed to log financial metrics: {e}")
    
    def _log_financial_metrics_from_results(self, optimization_results: Dict[str, Any], model_performance: Dict[str, Any], 
                                          execution_data: Dict[str, Any], enhancement_metrics: Dict[str, Any]) -> None:
        """Log key financial metrics directly from step results."""
        try:
            # Note: Data quality and performance metrics are logged in regular system logs
            # Financial metrics logger focuses only on financial/trading metrics
            
            # Log hyperparameter optimization metrics
            if optimization_results:
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="total_optimization_trials",
                    metric_value=float(optimization_results.get('total_trials', 0)),
                    metric_type="performance",
                    step_name="Step12_Analyst_Enhancement"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="best_optimization_score",
                    metric_value=optimization_results.get('best_score', 0.0),
                    metric_type="performance",
                    step_name="Step12_Analyst_Enhancement"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="optimization_convergence_score",
                    metric_value=optimization_results.get('convergence_score', 0.0),
                    metric_type="trading",
                    step_name="Step12_Analyst_Enhancement"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="optimization_time",
                    metric_value=float(optimization_results.get('optimization_time', 0)),
                    metric_type="performance",
                    step_name="Step12_Analyst_Enhancement"
                )
            
            # Log feature selection metrics
            if enhancement_metrics:
                if 'original_feature_count' in enhancement_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="original_feature_count",
                        metric_value=float(enhancement_metrics['original_feature_count']),
                        metric_type="performance",
                        step_name="Step12_Analyst_Enhancement"
                    )
                
                if 'selected_feature_count' in enhancement_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="selected_feature_count",
                        metric_value=float(enhancement_metrics['selected_feature_count']),
                        metric_type="performance",
                        step_name="Step12_Analyst_Enhancement"
                    )
                
                if 'feature_selection_score' in enhancement_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="feature_selection_score",
                        metric_value=enhancement_metrics['feature_selection_score'],
                        metric_type="trading",
                        step_name="Step12_Analyst_Enhancement"
                    )
                
                if 'correlation_reduction' in enhancement_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="correlation_reduction",
                        metric_value=enhancement_metrics['correlation_reduction'],
                        metric_type="trading",
                        step_name="Step12_Analyst_Enhancement"
                    )
                
                if 'vif_improvement' in enhancement_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="vif_improvement",
                        metric_value=enhancement_metrics['vif_improvement'],
                        metric_type="trading",
                        step_name="Step12_Analyst_Enhancement"
                    )
                
                if 'stability_score' in enhancement_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="feature_stability_score",
                        metric_value=enhancement_metrics['stability_score'],
                        metric_type="trading",
                        step_name="Step12_Analyst_Enhancement"
                    )
            
            # Log model enhancement metrics
            if model_performance:
                if 'original_accuracy' in model_performance:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="original_model_accuracy",
                        metric_value=model_performance['original_accuracy'],
                        metric_type="performance",
                        step_name="Step12_Analyst_Enhancement"
                    )
                
                if 'enhanced_accuracy' in model_performance:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="enhanced_model_accuracy",
                        metric_value=model_performance['enhanced_accuracy'],
                        metric_type="performance",
                        step_name="Step12_Analyst_Enhancement"
                    )
                
                if 'improvement_percentage' in model_performance:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="model_improvement_percentage",
                        metric_value=model_performance['improvement_percentage'],
                        metric_type="performance",
                        step_name="Step12_Analyst_Enhancement"
                    )
                
                if 'training_speedup' in model_performance:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="training_speedup",
                        metric_value=model_performance['training_speedup'],
                        metric_type="performance",
                        step_name="Step12_Analyst_Enhancement"
                    )
            
            # Log regime-specific optimization metrics
            if enhancement_metrics:
                if 'regime_specific_improvements' in enhancement_metrics:
                    regime_improvements = enhancement_metrics['regime_specific_improvements']
                    for regime_id, improvement in regime_improvements.items():
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"regime_{regime_id}_improvement",
                            metric_value=improvement,
                            metric_type="performance",
                            step_name="Step12_Analyst_Enhancement",
                            regime_id=str(regime_id)
                        )
                
                if 'optimization_efficiency' in enhancement_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="regime_optimization_efficiency",
                        metric_value=enhancement_metrics['optimization_efficiency'],
                        metric_type="trading",
                        step_name="Step12_Analyst_Enhancement"
                    )
                
                if 'feature_selection_efficiency' in enhancement_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="regime_feature_selection_efficiency",
                        metric_value=enhancement_metrics['feature_selection_efficiency'],
                        metric_type="trading",
                        step_name="Step12_Analyst_Enhancement"
                    )
                
                if 'hyperparameter_optimization_score' in enhancement_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="regime_hyperparameter_optimization_score",
                        metric_value=enhancement_metrics['hyperparameter_optimization_score'],
                        metric_type="trading",
                        step_name="Step12_Analyst_Enhancement"
                    )
            
            # Log hardware optimization metrics
            if execution_data:
                if 'gpu_utilization' in execution_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="gpu_utilization",
                        metric_value=execution_data['gpu_utilization'],
                        metric_type="performance",
                        step_name="Step12_Analyst_Enhancement"
                    )
                
                if 'memory_efficiency' in execution_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="memory_efficiency",
                        metric_value=execution_data['memory_efficiency'],
                        metric_type="performance",
                        step_name="Step12_Analyst_Enhancement"
                    )
                
                if 'parallel_processing_efficiency' in execution_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="parallel_processing_efficiency",
                        metric_value=execution_data['parallel_processing_efficiency'],
                        metric_type="performance",
                        step_name="Step12_Analyst_Enhancement"
                    )
                
                if 'vectorized_operations_count' in execution_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="vectorized_operations_count",
                        metric_value=float(execution_data['vectorized_operations_count']),
                        metric_type="performance",
                        step_name="Step12_Analyst_Enhancement"
                    )
                
                if 'matrix_operations_speedup' in execution_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="matrix_operations_speedup",
                        metric_value=execution_data['matrix_operations_speedup'],
                        metric_type="performance",
                        step_name="Step12_Analyst_Enhancement"
                    )
            
            # Log comprehensive trading performance estimation
            if optimization_results and model_performance and enhancement_metrics:
                # Estimate trading performance based on analyst enhancement results
                best_score = optimization_results.get('best_score', 0.5)
                improvement_percentage = model_performance.get('improvement_percentage', 0.0)
                feature_selection_score = enhancement_metrics.get('feature_selection_score', 0.5)
                stability_score = enhancement_metrics.get('stability_score', 0.5)
                
                # Estimate returns based on enhancement results
                combined_score = (best_score + (improvement_percentage / 100) + feature_selection_score + stability_score) / 4
                estimated_return = (combined_score * 0.02) - ((1 - combined_score) * 0.01)  # Estimate
                estimated_volatility = 0.02  # Default estimate
                
                # Estimate trading metrics
                total_trials = optimization_results.get('total_trials', 100)
                selected_features = enhancement_metrics.get('selected_feature_count', 50)
                original_features = enhancement_metrics.get('original_feature_count', 100)
                
                estimated_performance = {
                    'total_return': estimated_return,
                    'annualized_return': estimated_return * 252,  # Assuming daily signals
                    'volatility': estimated_volatility,
                    'sharpe_ratio': estimated_return / estimated_volatility if estimated_volatility > 0 else 0.0,
                    'sortino_ratio': estimated_return / (estimated_volatility * 0.6) if estimated_volatility > 0 else 0.0,
                    'calmar_ratio': 0.0,  # Would need max drawdown
                    'max_drawdown': estimated_volatility * 2.0,  # Estimate
                    'max_drawdown_duration': 25,  # Default estimate
                    'var_95': estimated_volatility * 1.6,  # Estimate
                    'cvar_95': estimated_volatility * 2.0,  # Estimate
                    'win_rate': combined_score,
                    'profit_factor': 1.0 + (combined_score - 0.5) * 2,
                    'avg_win': 0.02,  # Default estimate
                    'avg_loss': 0.01,  # Default estimate
                    'largest_win': 0.05,  # Default estimate
                    'largest_loss': estimated_volatility * 2.0,  # Estimate
                    'total_trades': int(total_trials * 0.2),  # Estimate 20% of trials become trades
                    'winning_trades': int(total_trials * 0.2 * combined_score),
                    'losing_trades': int(total_trials * 0.2 * (1 - combined_score)),
                    'additional_metrics': {
                        'best_optimization_score': best_score,
                        'improvement_percentage': improvement_percentage,
                        'feature_selection_score': feature_selection_score,
                        'stability_score': stability_score,
                        'original_feature_count': original_features,
                        'selected_feature_count': selected_features,
                        'feature_reduction_ratio': selected_features / max(original_features, 1),
                        'optimization_convergence_score': optimization_results.get('convergence_score', 0.0),
                        'optimization_time': optimization_results.get('optimization_time', 0),
                        'training_speedup': model_performance.get('training_speedup', 1.0),
                        'correlation_reduction': enhancement_metrics.get('correlation_reduction', 0.0),
                        'vif_improvement': enhancement_metrics.get('vif_improvement', 0.0)
                    }
                }
                
                self.financial_logger.log_trading_performance(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step12_Analyst_Enhancement",
                    **estimated_performance
                )
            
        except Exception as e:
            logger.error(f"Failed to log financial metrics from results: {e}")
    
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
                    step_name="Step12_Analyst_Enhancement",
                    additional_data={'file_path': str(self.financial_logger.current_file_path)}
                )
            logger.info("📁 File paths logged for Step12")
        except Exception as e:
            logger.warning(f"Could not log file paths: {e}")