"""
Financial metrics logging for Step04_5 Triple Barrier Method.
Independent logging module that can be used without the reporting system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from src.utils.financial_metrics_logger import get_financial_metrics_logger, financial_metrics_context
from src.utils.logger import system_logger

logger = system_logger.getChild('Step04_5FinancialLogging')


class Step04_5FinancialLogger:
    """Independent financial metrics logger for Step04_5 Triple Barrier Method."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.financial_logger = get_financial_metrics_logger()
    
    def log_step_execution(self, labeled_data: pd.DataFrame, label_stats: Dict[str, Any], 
                          execution_data: Dict[str, Any], triple_barrier_results: Dict[str, Any]) -> None:
        """Log comprehensive financial metrics for Step04_5 execution."""
        with financial_metrics_context(
            step_name="Step04_5_Triple_Barrier_Method",
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe
        ):
            try:
                self.financial_logger.log_step_start("Step04_5_Triple_Barrier_Method", self.symbol, self.exchange, self.timeframe)
                
                # Log all financial metrics
                self._log_financial_metrics_from_results(labeled_data, label_stats, execution_data, triple_barrier_results)
                
                # Log file paths
                self._log_created_file_paths()
                
                self.financial_logger.log_step_end("Step04_5_Triple_Barrier_Method", self.symbol, self.exchange, self.timeframe, success=True)
                
            except Exception as e:
                self.financial_logger.log_step_end("Step04_5_Triple_Barrier_Method", self.symbol, self.exchange, self.timeframe, success=False, error_message=str(e))
                logger.error(f"Failed to log financial metrics: {e}")
    
    def _log_financial_metrics_from_results(self, labeled_data: pd.DataFrame, label_stats: Dict[str, Any], 
                                          execution_data: Dict[str, Any], triple_barrier_results: Dict[str, Any]) -> None:
        """Log key financial metrics directly from step results."""
        try:
            # Note: Data quality metrics are logged in regular system logs
            # Financial metrics logger focuses only on financial/trading metrics
            
            # Log label statistics
            if label_stats:
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="total_signals_generated",
                    metric_value=float(label_stats.get('total_signals', 0)),
                    metric_type="performance",
                    step_name="Step04_5_Triple_Barrier_Method"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="buy_signals_count",
                    metric_value=float(label_stats.get('buy_signals', 0)),
                    metric_type="performance",
                    step_name="Step04_5_Triple_Barrier_Method"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="sell_signals_count",
                    metric_value=float(label_stats.get('sell_signals', 0)),
                    metric_type="performance",
                    step_name="Step04_5_Triple_Barrier_Method"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="hold_signals_count",
                    metric_value=float(label_stats.get('hold_signals', 0)),
                    metric_type="performance",
                    step_name="Step04_5_Triple_Barrier_Method"
                )
                
                # Log signal distribution balance (financial relevance)
                total_signals = label_stats.get('total_signals', 1)
                buy_ratio = label_stats.get('buy_signals', 0) / total_signals
                sell_ratio = label_stats.get('sell_signals', 0) / total_signals
                hold_ratio = label_stats.get('hold_signals', 0) / total_signals
                
                # Calculate signal distribution balance (closer to 0.33 each is better)
                ideal_ratio = 1.0 / 3.0
                distribution_balance = 1.0 - (abs(buy_ratio - ideal_ratio) + abs(sell_ratio - ideal_ratio) + abs(hold_ratio - ideal_ratio))
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="signal_distribution_balance",
                    metric_value=distribution_balance,
                    metric_type="trading",
                    step_name="Step04_5_Triple_Barrier_Method"
                )
                
                # Log profit target and stop loss statistics
                if 'profit_target_achieved' in label_stats:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="profit_target_achieved",
                        metric_value=float(label_stats['profit_target_achieved']),
                        metric_type="performance",
                        step_name="Step04_5_Triple_Barrier_Method"
                    )
                
                if 'stop_loss_hit' in label_stats:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="stop_loss_hit",
                        metric_value=float(label_stats['stop_loss_hit']),
                        metric_type="performance",
                        step_name="Step04_5_Triple_Barrier_Method"
                    )
                
                if 'timeout_reached' in label_stats:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="timeout_reached",
                        metric_value=float(label_stats['timeout_reached']),
                        metric_type="performance",
                        step_name="Step04_5_Triple_Barrier_Method"
                    )
                
                # Log win rate if available
                if 'win_rate' in label_stats:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="label_win_rate",
                        metric_value=label_stats['win_rate'],
                        metric_type="performance",
                        step_name="Step04_5_Triple_Barrier_Method"
                    )
                
                # Log profit factor if available
                if 'profit_factor' in label_stats:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="label_profit_factor",
                        metric_value=label_stats['profit_factor'],
                        metric_type="performance",
                        step_name="Step04_5_Triple_Barrier_Method"
                    )
            
            # Log triple barrier method specific metrics
            if triple_barrier_results:
                # Log barrier configuration
                if 'profit_take_multiplier' in triple_barrier_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="profit_take_multiplier",
                        metric_value=triple_barrier_results['profit_take_multiplier'],
                        metric_type="hyperparameter",
                        step_name="Step04_5_Triple_Barrier_Method"
                    )
                
                if 'stop_loss_multiplier' in triple_barrier_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="stop_loss_multiplier",
                        metric_value=triple_barrier_results['stop_loss_multiplier'],
                        metric_type="hyperparameter",
                        step_name="Step04_5_Triple_Barrier_Method"
                    )
                
                if 'time_barrier_minutes' in triple_barrier_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="time_barrier_minutes",
                        metric_value=float(triple_barrier_results['time_barrier_minutes']),
                        metric_type="hyperparameter",
                        step_name="Step04_5_Triple_Barrier_Method"
                    )
                
                # Log signal generation rate
                if 'signal_generation_rate' in triple_barrier_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="signal_generation_rate",
                        metric_value=triple_barrier_results['signal_generation_rate'],
                        metric_type="performance",
                        step_name="Step04_5_Triple_Barrier_Method"
                    )
                
                # Log label success rate (financial relevance)
                if 'label_success_rate' in triple_barrier_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="label_success_rate",
                        metric_value=triple_barrier_results['label_success_rate'],
                        metric_type="trading",
                        step_name="Step04_5_Triple_Barrier_Method"
                    )
            
            # Note: Execution performance metrics are logged in regular system logs
            # Financial metrics logger focuses only on financial/trading metrics
            
            # Log comprehensive trading performance estimation
            if labeled_data is not None and not labeled_data.empty and label_stats:
                # Estimate trading performance based on triple barrier results
                total_signals = label_stats.get('total_signals', 0)
                win_rate = label_stats.get('win_rate', 0.5)
                profit_factor = label_stats.get('profit_factor', 1.0)
                
                # Estimate returns based on signal quality
                estimated_return = (win_rate * 0.02) - ((1 - win_rate) * 0.01)  # Rough estimate
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
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'avg_win': 0.02,  # Default estimate
                    'avg_loss': 0.01,  # Default estimate
                    'largest_win': 0.05,  # Default estimate
                    'largest_loss': estimated_volatility * 2,  # Estimate
                    'total_trades': total_signals,
                    'winning_trades': int(total_signals * win_rate),
                    'losing_trades': int(total_signals * (1 - win_rate)),
                    'additional_metrics': {
                        'signal_distribution_balance': distribution_balance if 'distribution_balance' in locals() else 0.0,
                        'label_success_rate': triple_barrier_results.get('label_success_rate', 0.0) if triple_barrier_results else 0.0,
                        'profit_target_achieved': label_stats.get('profit_target_achieved', 0),
                        'stop_loss_hit': label_stats.get('stop_loss_hit', 0),
                        'timeout_reached': label_stats.get('timeout_reached', 0)
                    }
                }
                
                self.financial_logger.log_trading_performance(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step04_5_Triple_Barrier_Method",
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
                    step_name="Step04_5_Triple_Barrier_Method",
                    additional_data={'file_path': str(self.financial_logger.current_file_path)}
                )
            logger.info("📁 File paths logged for Step04_5")
        except Exception as e:
            logger.warning(f"Could not log file paths: {e}")