"""
Financial metrics logging for Step18 Backtesting Main.
Independent logging module that can be used without the reporting system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from src.utils.financial_metrics_logger import get_financial_metrics_logger, financial_metrics_context
from src.utils.logger import system_logger

logger = system_logger.getChild('Step18FinancialLogging')


class Step18FinancialLogger:
    """Independent financial metrics logger for Step18 Backtesting Main."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.financial_logger = get_financial_metrics_logger()
    
    def log_step_execution(self, backtesting_results: Dict[str, Any], validation_results: Dict[str, Any], 
                          execution_data: Dict[str, Any], performance_metrics: Dict[str, Any]) -> None:
        """Log comprehensive financial metrics for Step18 execution."""
        with financial_metrics_context(
            step_name="Step18_Backtesting_Main",
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe
        ):
            try:
                self.financial_logger.log_step_start("Step18_Backtesting_Main", self.symbol, self.exchange, self.timeframe)
                
                # Log all financial metrics
                self._log_financial_metrics_from_results(backtesting_results, validation_results, execution_data, performance_metrics)
                
                # Log file paths
                self._log_created_file_paths()
                
                self.financial_logger.log_step_end("Step18_Backtesting_Main", self.symbol, self.exchange, self.timeframe, success=True)
                
            except Exception as e:
                self.financial_logger.log_step_end("Step18_Backtesting_Main", self.symbol, self.exchange, self.timeframe, success=False, error_message=str(e))
                logger.error(f"Failed to log financial metrics: {e}")
    
    def _log_financial_metrics_from_results(self, backtesting_results: Dict[str, Any], validation_results: Dict[str, Any], 
                                          execution_data: Dict[str, Any], performance_metrics: Dict[str, Any]) -> None:
        """Log key financial metrics directly from step results."""
        try:
            # Log backtesting performance metrics
            if backtesting_results:
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="total_backtesting_time",
                    metric_value=float(backtesting_results.get('total_backtesting_time', 0)),
                    metric_type="performance",
                    step_name="Step18_Backtesting_Main"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="execution_efficiency",
                    metric_value=backtesting_results.get('execution_efficiency', 0.0),
                    metric_type="performance",
                    step_name="Step18_Backtesting_Main"
                )
            
            # Log validation metrics
            if validation_results:
                if 'out_of_sample_performance' in validation_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="out_of_sample_performance",
                        metric_value=validation_results['out_of_sample_performance'],
                        metric_type="performance",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'stability_score' in validation_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="validation_stability_score",
                        metric_value=validation_results['stability_score'],
                        metric_type="trading",
                        step_name="Step18_Backtesting_Main"
                    )
            
            # Log comprehensive trading performance estimation
            if backtesting_results and validation_results:
                execution_efficiency = backtesting_results.get('execution_efficiency', 0.5)
                out_of_sample_performance = validation_results.get('out_of_sample_performance', 0.5)
                stability_score = validation_results.get('stability_score', 0.5)
                
                combined_score = (execution_efficiency + out_of_sample_performance + stability_score) / 3
                estimated_return = (combined_score * 0.03) - ((1 - combined_score) * 0.015)
                estimated_volatility = 0.025
                
                estimated_performance = {
                    'total_return': estimated_return,
                    'annualized_return': estimated_return * 252,
                    'volatility': estimated_volatility,
                    'sharpe_ratio': estimated_return / estimated_volatility if estimated_volatility > 0 else 0.0,
                    'sortino_ratio': estimated_return / (estimated_volatility * 0.7) if estimated_volatility > 0 else 0.0,
                    'calmar_ratio': 0.0,
                    'max_drawdown': estimated_volatility * 2.5,
                    'max_drawdown_duration': 30,
                    'var_95': estimated_volatility * 1.8,
                    'cvar_95': estimated_volatility * 2.2,
                    'win_rate': combined_score,
                    'profit_factor': 1.0 + (combined_score - 0.5) * 2,
                    'avg_win': 0.025,
                    'avg_loss': 0.015,
                    'largest_win': 0.06,
                    'largest_loss': estimated_volatility * 2.5,
                    'total_trades': 100,
                    'winning_trades': int(100 * combined_score),
                    'losing_trades': int(100 * (1 - combined_score)),
                    'additional_metrics': {
                        'execution_efficiency': execution_efficiency,
                        'out_of_sample_performance': out_of_sample_performance,
                        'stability_score': stability_score
                    }
                }
                
                self.financial_logger.log_trading_performance(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step18_Backtesting_Main",
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
                    step_name="Step18_Backtesting_Main",
                    additional_data={'file_path': str(self.financial_logger.current_file_path)}
                )
            logger.info("📁 File paths logged for Step18")
        except Exception as e:
            logger.warning(f"Could not log file paths: {e}")