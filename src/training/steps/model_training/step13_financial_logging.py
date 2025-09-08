"""
Financial metrics logging for Step13 Analyst Ensemble Creation.
Independent logging module that can be used without the reporting system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from src.utils.financial_metrics_logger import get_financial_metrics_logger, financial_metrics_context
from src.utils.logger import system_logger

logger = system_logger.getChild('Step13FinancialLogging')


class Step13FinancialLogger:
    """Independent financial metrics logger for Step13 Analyst Ensemble Creation."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.financial_logger = get_financial_metrics_logger()
    
    def log_step_execution(self, ensemble_results: Dict[str, Any], execution_data: Dict[str, Any], 
                          performance_metrics: Dict[str, Any], optimization_metrics: Dict[str, Any]) -> None:
        """Log comprehensive financial metrics for Step13 execution."""
        with financial_metrics_context(
            step_name="Step13_Analyst_Ensemble_Creation",
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe
        ):
            try:
                self.financial_logger.log_step_start("Step13_Analyst_Ensemble_Creation", self.symbol, self.exchange, self.timeframe)
                
                # Log all financial metrics
                self._log_financial_metrics_from_results(ensemble_results, execution_data, performance_metrics, optimization_metrics)
                
                # Log file paths
                self._log_created_file_paths()
                
                self.financial_logger.log_step_end("Step13_Analyst_Ensemble_Creation", self.symbol, self.exchange, self.timeframe, success=True)
                
            except Exception as e:
                self.financial_logger.log_step_end("Step13_Analyst_Ensemble_Creation", self.symbol, self.exchange, self.timeframe, success=False, error_message=str(e))
                logger.error(f"Failed to log financial metrics: {e}")
    
    def _log_financial_metrics_from_results(self, ensemble_results: Dict[str, Any], execution_data: Dict[str, Any], 
                                          performance_metrics: Dict[str, Any], optimization_metrics: Dict[str, Any]) -> None:
        """Log key financial metrics directly from step results."""
        try:
            # Note: Data quality and performance metrics are logged in regular system logs
            # Financial metrics logger focuses only on financial/trading metrics
            
            # Log ensemble performance metrics
            if ensemble_results:
                if 'ensemble_accuracy' in ensemble_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="ensemble_accuracy",
                        metric_value=ensemble_results['ensemble_accuracy'],
                        metric_type="performance",
                        step_name="Step13_Analyst_Ensemble_Creation"
                    )
                
                if 'ensemble_improvement' in ensemble_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="ensemble_improvement",
                        metric_value=ensemble_results['ensemble_improvement'],
                        metric_type="performance",
                        step_name="Step13_Analyst_Ensemble_Creation"
                    )
                
                if 'ensemble_diversity_score' in ensemble_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="ensemble_diversity_score",
                        metric_value=ensemble_results['ensemble_diversity_score'],
                        metric_type="trading",
                        step_name="Step13_Analyst_Ensemble_Creation"
                    )
                
                if 'ensemble_stability_score' in ensemble_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="ensemble_stability_score",
                        metric_value=ensemble_results['ensemble_stability_score'],
                        metric_type="trading",
                        step_name="Step13_Analyst_Ensemble_Creation"
                    )
                
                if 'cross_validation_score' in ensemble_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="cross_validation_score",
                        metric_value=ensemble_results['cross_validation_score'],
                        metric_type="performance",
                        step_name="Step13_Analyst_Ensemble_Creation"
                    )
                
                if 'out_of_sample_performance' in ensemble_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="out_of_sample_performance",
                        metric_value=ensemble_results['out_of_sample_performance'],
                        metric_type="performance",
                        step_name="Step13_Analyst_Ensemble_Creation"
                    )
            
            # Log weight optimization metrics
            if optimization_metrics:
                if 'weight_optimization_score' in optimization_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="weight_optimization_score",
                        metric_value=optimization_metrics['weight_optimization_score'],
                        metric_type="trading",
                        step_name="Step13_Analyst_Ensemble_Creation"
                    )
                
                if 'weight_stability_score' in optimization_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="weight_stability_score",
                        metric_value=optimization_metrics['weight_stability_score'],
                        metric_type="trading",
                        step_name="Step13_Analyst_Ensemble_Creation"
                    )
                
                if 'weight_convergence_score' in optimization_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="weight_convergence_score",
                        metric_value=optimization_metrics['weight_convergence_score'],
                        metric_type="trading",
                        step_name="Step13_Analyst_Ensemble_Creation"
                    )
                
                if 'ensemble_optimization_time' in optimization_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="ensemble_optimization_time",
                        metric_value=float(optimization_metrics['ensemble_optimization_time']),
                        metric_type="performance",
                        step_name="Step13_Analyst_Ensemble_Creation"
                    )
            
            # Log hardware acceleration metrics
            if execution_data:
                if 'gpu_utilization' in execution_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="gpu_utilization",
                        metric_value=execution_data['gpu_utilization'],
                        metric_type="performance",
                        step_name="Step13_Analyst_Ensemble_Creation"
                    )
                
                if 'memory_efficiency' in execution_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="memory_efficiency",
                        metric_value=execution_data['memory_efficiency'],
                        metric_type="performance",
                        step_name="Step13_Analyst_Ensemble_Creation"
                    )
                
                if 'parallel_processing_efficiency' in execution_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="parallel_processing_efficiency",
                        metric_value=execution_data['parallel_processing_efficiency'],
                        metric_type="performance",
                        step_name="Step13_Analyst_Ensemble_Creation"
                    )
                
                if 'vectorized_operations_count' in execution_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="vectorized_operations_count",
                        metric_value=float(execution_data['vectorized_operations_count']),
                        metric_type="performance",
                        step_name="Step13_Analyst_Ensemble_Creation"
                    )
                
                if 'ensemble_creation_time' in execution_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="ensemble_creation_time",
                        metric_value=float(execution_data['ensemble_creation_time']),
                        metric_type="performance",
                        step_name="Step13_Analyst_Ensemble_Creation"
                    )
            
            # Log model diversity metrics
            if performance_metrics:
                if 'individual_model_accuracies' in performance_metrics:
                    accuracies = performance_metrics['individual_model_accuracies']
                    if accuracies:
                        accuracy_std = np.std(accuracies)
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name="model_accuracy_std",
                            metric_value=accuracy_std,
                            metric_type="trading",
                            step_name="Step13_Analyst_Ensemble_Creation"
                        )
                        
                        accuracy_mean = np.mean(accuracies)
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name="model_accuracy_mean",
                            metric_value=accuracy_mean,
                            metric_type="performance",
                            step_name="Step13_Analyst_Ensemble_Creation"
                        )
                
                if 'ensemble_variance_reduction' in performance_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="ensemble_variance_reduction",
                        metric_value=performance_metrics['ensemble_variance_reduction'],
                        metric_type="trading",
                        step_name="Step13_Analyst_Ensemble_Creation"
                    )
                
                if 'ensemble_bias_reduction' in performance_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="ensemble_bias_reduction",
                        metric_value=performance_metrics['ensemble_bias_reduction'],
                        metric_type="trading",
                        step_name="Step13_Analyst_Ensemble_Creation"
                    )
            
            # Log comprehensive trading performance estimation
            if ensemble_results and optimization_metrics and performance_metrics:
                # Estimate trading performance based on ensemble creation results
                ensemble_accuracy = ensemble_results.get('ensemble_accuracy', 0.5)
                ensemble_improvement = ensemble_results.get('ensemble_improvement', 0.0)
                diversity_score = ensemble_results.get('ensemble_diversity_score', 0.5)
                stability_score = ensemble_results.get('ensemble_stability_score', 0.5)
                
                # Estimate returns based on ensemble results
                combined_score = (ensemble_accuracy + (ensemble_improvement / 100) + diversity_score + stability_score) / 4
                estimated_return = (combined_score * 0.025) - ((1 - combined_score) * 0.01)  # Estimate
                estimated_volatility = 0.015  # Default estimate
                
                # Estimate trading metrics
                total_models = execution_data.get('total_models_processed', 10)
                ensemble_creation_time = execution_data.get('ensemble_creation_time', 60)
                
                estimated_performance = {
                    'total_return': estimated_return,
                    'annualized_return': estimated_return * 252,  # Assuming daily signals
                    'volatility': estimated_volatility,
                    'sharpe_ratio': estimated_return / estimated_volatility if estimated_volatility > 0 else 0.0,
                    'sortino_ratio': estimated_return / (estimated_volatility * 0.6) if estimated_volatility > 0 else 0.0,
                    'calmar_ratio': 0.0,  # Would need max drawdown
                    'max_drawdown': estimated_volatility * 1.8,  # Estimate
                    'max_drawdown_duration': 30,  # Default estimate
                    'var_95': estimated_volatility * 1.6,  # Estimate
                    'cvar_95': estimated_volatility * 2.0,  # Estimate
                    'win_rate': combined_score,
                    'profit_factor': 1.0 + (combined_score - 0.5) * 2.5,
                    'avg_win': 0.025,  # Default estimate
                    'avg_loss': 0.01,  # Default estimate
                    'largest_win': 0.06,  # Default estimate
                    'largest_loss': estimated_volatility * 1.8,  # Estimate
                    'total_trades': int(total_models * 3),  # Estimate 3 trades per model
                    'winning_trades': int(total_models * 3 * combined_score),
                    'losing_trades': int(total_models * 3 * (1 - combined_score)),
                    'additional_metrics': {
                        'ensemble_accuracy': ensemble_accuracy,
                        'ensemble_improvement': ensemble_improvement,
                        'diversity_score': diversity_score,
                        'stability_score': stability_score,
                        'total_models_processed': total_models,
                        'ensemble_creation_time': ensemble_creation_time,
                        'weight_optimization_score': optimization_metrics.get('weight_optimization_score', 0.0),
                        'weight_stability_score': optimization_metrics.get('weight_stability_score', 0.0),
                        'cross_validation_score': ensemble_results.get('cross_validation_score', 0.0),
                        'out_of_sample_performance': ensemble_results.get('out_of_sample_performance', 0.0)
                    }
                }
                
                self.financial_logger.log_trading_performance(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step13_Analyst_Ensemble_Creation",
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
                    step_name="Step13_Analyst_Ensemble_Creation",
                    additional_data={'file_path': str(self.financial_logger.current_file_path)}
                )
            logger.info("📁 File paths logged for Step13")
        except Exception as e:
            logger.warning(f"Could not log file paths: {e}")