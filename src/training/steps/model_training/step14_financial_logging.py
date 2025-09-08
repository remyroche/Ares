"""
Financial metrics logging for Step14 Tactician Labeling.
Independent logging module that can be used without the reporting system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from src.utils.financial_metrics_logger import get_financial_metrics_logger, financial_metrics_context
from src.utils.logger import system_logger

logger = system_logger.getChild('Step14FinancialLogging')


class Step14FinancialLogger:
    """Independent financial metrics logger for Step14 Tactician Labeling."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.financial_logger = get_financial_metrics_logger()
    
    def log_step_execution(self, labeling_results: Dict[str, Any], execution_data: Dict[str, Any], 
                          performance_metrics: Dict[str, Any], barrier_metrics: Dict[str, Any]) -> None:
        """Log comprehensive financial metrics for Step14 execution."""
        with financial_metrics_context(
            step_name="Step14_Tactician_Labeling",
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe
        ):
            try:
                self.financial_logger.log_step_start("Step14_Tactician_Labeling", self.symbol, self.exchange, self.timeframe)
                
                # Log all financial metrics
                self._log_financial_metrics_from_results(labeling_results, execution_data, performance_metrics, barrier_metrics)
                
                # Log file paths
                self._log_created_file_paths()
                
                self.financial_logger.log_step_end("Step14_Tactician_Labeling", self.symbol, self.exchange, self.timeframe, success=True)
                
            except Exception as e:
                self.financial_logger.log_step_end("Step14_Tactician_Labeling", self.symbol, self.exchange, self.timeframe, success=False, error_message=str(e))
                logger.error(f"Failed to log financial metrics: {e}")
    
    def _log_financial_metrics_from_results(self, labeling_results: Dict[str, Any], execution_data: Dict[str, Any], 
                                          performance_metrics: Dict[str, Any], barrier_metrics: Dict[str, Any]) -> None:
        """Log key financial metrics directly from step results."""
        try:
            # Note: Data quality and performance metrics are logged in regular system logs
            # Financial metrics logger focuses only on financial/trading metrics
            
            # Log barrier performance metrics
            if barrier_metrics:
                if 'total_barriers_calculated' in barrier_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="total_barriers_calculated",
                        metric_value=float(barrier_metrics['total_barriers_calculated']),
                        metric_type="performance",
                        step_name="Step14_Tactician_Labeling"
                    )
                
                if 'barrier_effectiveness_score' in barrier_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="barrier_effectiveness_score",
                        metric_value=barrier_metrics['barrier_effectiveness_score'],
                        metric_type="trading",
                        step_name="Step14_Tactician_Labeling"
                    )
                
                if 'average_profit_barrier' in barrier_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="average_profit_barrier",
                        metric_value=barrier_metrics['average_profit_barrier'],
                        metric_type="trading",
                        step_name="Step14_Tactician_Labeling"
                    )
                
                if 'average_loss_barrier' in barrier_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="average_loss_barrier",
                        metric_value=barrier_metrics['average_loss_barrier'],
                        metric_type="trading",
                        step_name="Step14_Tactician_Labeling"
                    )
                
                if 'barrier_adaptation_rate' in barrier_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="barrier_adaptation_rate",
                        metric_value=barrier_metrics['barrier_adaptation_rate'],
                        metric_type="trading",
                        step_name="Step14_Tactician_Labeling"
                    )
                
                if 'barrier_success_rate' in barrier_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="barrier_success_rate",
                        metric_value=barrier_metrics['barrier_success_rate'],
                        metric_type="trading",
                        step_name="Step14_Tactician_Labeling"
                    )
            
            # Log labeling quality metrics
            if performance_metrics:
                if 'labeling_accuracy' in performance_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="labeling_accuracy",
                        metric_value=performance_metrics['labeling_accuracy'],
                        metric_type="performance",
                        step_name="Step14_Tactician_Labeling"
                    )
                
                if 'labeling_precision' in performance_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="labeling_precision",
                        metric_value=performance_metrics['labeling_precision'],
                        metric_type="performance",
                        step_name="Step14_Tactician_Labeling"
                    )
                
                if 'labeling_recall' in performance_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="labeling_recall",
                        metric_value=performance_metrics['labeling_recall'],
                        metric_type="performance",
                        step_name="Step14_Tactician_Labeling"
                    )
                
                if 'labeling_f1_score' in performance_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="labeling_f1_score",
                        metric_value=performance_metrics['labeling_f1_score'],
                        metric_type="performance",
                        step_name="Step14_Tactician_Labeling"
                    )
                
                if 'labeling_consistency_score' in performance_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="labeling_consistency_score",
                        metric_value=performance_metrics['labeling_consistency_score'],
                        metric_type="trading",
                        step_name="Step14_Tactician_Labeling"
                    )
                
                if 'labeling_stability_score' in performance_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="labeling_stability_score",
                        metric_value=performance_metrics['labeling_stability_score'],
                        metric_type="trading",
                        step_name="Step14_Tactician_Labeling"
                    )
            
            # Log regime-specific metrics
            if labeling_results:
                if 'regime_specific_results' in labeling_results:
                    regime_results = labeling_results['regime_specific_results']
                    for regime_id, regime_data in regime_results.items():
                        if isinstance(regime_data, dict):
                            if 'regime_accuracy' in regime_data:
                                self.financial_logger.log_financial_metric(
                                    symbol=self.symbol,
                                    exchange=self.exchange,
                                    timeframe=self.timeframe,
                                    metric_name=f"regime_{regime_id}_accuracy",
                                    metric_value=regime_data['regime_accuracy'],
                                    metric_type="performance",
                                    step_name="Step14_Tactician_Labeling",
                                    regime_id=str(regime_id)
                                )
                            
                            if 'regime_barrier_effectiveness' in regime_data:
                                self.financial_logger.log_financial_metric(
                                    symbol=self.symbol,
                                    exchange=self.exchange,
                                    timeframe=self.timeframe,
                                    metric_name=f"regime_{regime_id}_barrier_effectiveness",
                                    metric_value=regime_data['regime_barrier_effectiveness'],
                                    metric_type="trading",
                                    step_name="Step14_Tactician_Labeling",
                                    regime_id=str(regime_id)
                                )
                            
                            if 'regime_labeling_quality' in regime_data:
                                self.financial_logger.log_financial_metric(
                                    symbol=self.symbol,
                                    exchange=self.exchange,
                                    timeframe=self.timeframe,
                                    metric_name=f"regime_{regime_id}_labeling_quality",
                                    metric_value=regime_data['regime_labeling_quality'],
                                    metric_type="trading",
                                    step_name="Step14_Tactician_Labeling",
                                    regime_id=str(regime_id)
                                )
                
                if 'total_labels_generated' in labeling_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="total_labels_generated",
                        metric_value=float(labeling_results['total_labels_generated']),
                        metric_type="performance",
                        step_name="Step14_Tactician_Labeling"
                    )
                
                if 'labeling_efficiency' in labeling_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="labeling_efficiency",
                        metric_value=labeling_results['labeling_efficiency'],
                        metric_type="performance",
                        step_name="Step14_Tactician_Labeling"
                    )
            
            # Log execution metrics
            if execution_data:
                if 'total_execution_time' in execution_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="total_execution_time",
                        metric_value=float(execution_data['total_execution_time']),
                        metric_type="performance",
                        step_name="Step14_Tactician_Labeling"
                    )
                
                if 'regimes_processed' in execution_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="regimes_processed",
                        metric_value=float(execution_data['regimes_processed']),
                        metric_type="performance",
                        step_name="Step14_Tactician_Labeling"
                    )
                
                if 'data_points_processed' in execution_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="data_points_processed",
                        metric_value=float(execution_data['data_points_processed']),
                        metric_type="performance",
                        step_name="Step14_Tactician_Labeling"
                    )
            
            # Log comprehensive trading performance estimation
            if labeling_results and barrier_metrics and performance_metrics:
                # Estimate trading performance based on tactician labeling results
                labeling_accuracy = performance_metrics.get('labeling_accuracy', 0.5)
                barrier_effectiveness = barrier_metrics.get('barrier_effectiveness_score', 0.5)
                barrier_success_rate = barrier_metrics.get('barrier_success_rate', 0.5)
                labeling_consistency = performance_metrics.get('labeling_consistency_score', 0.5)
                
                # Estimate returns based on labeling results
                combined_score = (labeling_accuracy + barrier_effectiveness + barrier_success_rate + labeling_consistency) / 4
                estimated_return = (combined_score * 0.03) - ((1 - combined_score) * 0.015)  # Estimate
                estimated_volatility = 0.02  # Default estimate
                
                # Estimate trading metrics
                total_labels = labeling_results.get('total_labels_generated', 1000)
                total_barriers = barrier_metrics.get('total_barriers_calculated', 100)
                
                estimated_performance = {
                    'total_return': estimated_return,
                    'annualized_return': estimated_return * 252,  # Assuming daily signals
                    'volatility': estimated_volatility,
                    'sharpe_ratio': estimated_return / estimated_volatility if estimated_volatility > 0 else 0.0,
                    'sortino_ratio': estimated_return / (estimated_volatility * 0.6) if estimated_volatility > 0 else 0.0,
                    'calmar_ratio': 0.0,  # Would need max drawdown
                    'max_drawdown': estimated_volatility * 2.2,  # Estimate
                    'max_drawdown_duration': 35,  # Default estimate
                    'var_95': estimated_volatility * 1.6,  # Estimate
                    'cvar_95': estimated_volatility * 2.0,  # Estimate
                    'win_rate': combined_score,
                    'profit_factor': 1.0 + (combined_score - 0.5) * 3.0,
                    'avg_win': 0.03,  # Default estimate
                    'avg_loss': 0.015,  # Default estimate
                    'largest_win': 0.08,  # Default estimate
                    'largest_loss': estimated_volatility * 2.2,  # Estimate
                    'total_trades': int(total_labels * 0.1),  # Estimate 10% of labels become trades
                    'winning_trades': int(total_labels * 0.1 * combined_score),
                    'losing_trades': int(total_labels * 0.1 * (1 - combined_score)),
                    'additional_metrics': {
                        'labeling_accuracy': labeling_accuracy,
                        'barrier_effectiveness': barrier_effectiveness,
                        'barrier_success_rate': barrier_success_rate,
                        'labeling_consistency': labeling_consistency,
                        'total_labels_generated': total_labels,
                        'total_barriers_calculated': total_barriers,
                        'average_profit_barrier': barrier_metrics.get('average_profit_barrier', 0.0),
                        'average_loss_barrier': barrier_metrics.get('average_loss_barrier', 0.0),
                        'barrier_adaptation_rate': barrier_metrics.get('barrier_adaptation_rate', 0.0),
                        'labeling_precision': performance_metrics.get('labeling_precision', 0.0),
                        'labeling_recall': performance_metrics.get('labeling_recall', 0.0),
                        'labeling_f1_score': performance_metrics.get('labeling_f1_score', 0.0),
                        'labeling_stability_score': performance_metrics.get('labeling_stability_score', 0.0)
                    }
                }
                
                self.financial_logger.log_trading_performance(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step14_Tactician_Labeling",
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
                    step_name="Step14_Tactician_Labeling",
                    additional_data={'file_path': str(self.financial_logger.current_file_path)}
                )
            logger.info("📁 File paths logged for Step14")
        except Exception as e:
            logger.warning(f"Could not log file paths: {e}")