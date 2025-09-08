"""
Financial metrics logging for Step18 Backtesting.
Independent logging module that can be used without the reporting system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from src.utils.financial_metrics_logger import get_financial_metrics_logger, financial_metrics_context
from src.utils.logger import system_logger

logger = system_logger.getChild('Step18FinancialLogging')


class Step18FinancialLogger:
    """Independent financial metrics logger for Step18 Backtesting."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.financial_logger = get_financial_metrics_logger()
    
    def log_step_execution(self, backtest_results: Dict[str, Any], performance_metrics: Dict[str, Any], 
                          execution_data: Dict[str, Any], trading_results: Dict[str, Any]) -> None:
        """Log comprehensive financial metrics for Step18 execution."""
        with financial_metrics_context(
            step_name="Step18_Backtesting",
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe
        ):
            try:
                self.financial_logger.log_step_start("Step18_Backtesting", self.symbol, self.exchange, self.timeframe)
                
                # Log all financial metrics
                self._log_backtesting_metrics(backtest_results, performance_metrics, execution_data, trading_results)
                
                # Log file paths
                self._log_created_file_paths()
                
                self.financial_logger.log_step_end("Step18_Backtesting", self.symbol, self.exchange, self.timeframe, success=True)
                
            except Exception as e:
                self.financial_logger.log_step_end("Step18_Backtesting", self.symbol, self.exchange, self.timeframe, success=False, error_message=str(e))
                logger.error(f"Failed to log financial metrics: {e}")
    
    def _log_backtesting_metrics(self, backtest_results: Dict[str, Any], performance_metrics: Dict[str, Any],
                               execution_data: Dict[str, Any], trading_results: Dict[str, Any]) -> None:
        """Log key financial metrics directly from step results."""
        try:
            # Log comprehensive trading performance metrics
            if performance_metrics:
                # Log all performance metrics using the comprehensive trading performance method
                self.financial_logger.log_trading_performance(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step18_Backtesting",
                    total_return=performance_metrics.get('total_return', 0.0),
                    annualized_return=performance_metrics.get('annualized_return', 0.0),
                    volatility=performance_metrics.get('volatility', 0.0),
                    sharpe_ratio=performance_metrics.get('sharpe_ratio', 0.0),
                    sortino_ratio=performance_metrics.get('sortino_ratio', 0.0),
                    calmar_ratio=performance_metrics.get('calmar_ratio', 0.0),
                    max_drawdown=performance_metrics.get('max_drawdown', 0.0),
                    max_drawdown_duration=performance_metrics.get('max_drawdown_duration', 0),
                    var_95=performance_metrics.get('var_95', 0.0),
                    cvar_95=performance_metrics.get('cvar_95', 0.0),
                    win_rate=performance_metrics.get('win_rate', 0.0),
                    profit_factor=performance_metrics.get('profit_factor', 0.0),
                    avg_win=performance_metrics.get('avg_win', 0.0),
                    avg_loss=performance_metrics.get('avg_loss', 0.0),
                    largest_win=performance_metrics.get('largest_win', 0.0),
                    largest_loss=performance_metrics.get('largest_loss', 0.0),
                    total_trades=performance_metrics.get('total_trades', 0),
                    winning_trades=performance_metrics.get('winning_trades', 0),
                    losing_trades=performance_metrics.get('losing_trades', 0),
                    additional_metrics=performance_metrics.get('additional_metrics', {})
                )
            
            # Log individual key metrics for detailed tracking
            if backtest_results:
                # Log backtest execution metrics
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="backtest_duration_days",
                    metric_value=float(backtest_results.get('duration_days', 0)),
                    metric_type="performance",
                    step_name="Step18_Backtesting"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="backtest_trades_executed",
                    metric_value=float(backtest_results.get('trades_executed', 0)),
                    metric_type="performance",
                    step_name="Step18_Backtesting"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="backtest_initial_capital",
                    metric_value=float(backtest_results.get('initial_capital', 0)),
                    metric_type="performance",
                    step_name="Step18_Backtesting"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="backtest_final_capital",
                    metric_value=float(backtest_results.get('final_capital', 0)),
                    metric_type="performance",
                    step_name="Step18_Backtesting"
                )
            
            # Log trading execution metrics
            if trading_results:
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="execution_slippage_avg",
                    metric_value=trading_results.get('avg_slippage', 0.0),
                    metric_type="trading",
                    step_name="Step18_Backtesting"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="execution_commission_total",
                    metric_value=float(trading_results.get('total_commission', 0)),
                    metric_type="trading",
                    step_name="Step18_Backtesting"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="execution_fill_rate",
                    metric_value=trading_results.get('fill_rate', 0.0),
                    metric_type="trading",
                    step_name="Step18_Backtesting"
                )
            
            # Log risk metrics
            if performance_metrics:
                risk_metrics = self._extract_risk_metrics(performance_metrics)
                for metric_name, metric_value in risk_metrics.items():
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name=f"risk_{metric_name}",
                        metric_value=metric_value,
                        metric_type="risk",
                        step_name="Step18_Backtesting"
                    )
            
        except Exception as e:
            logger.error(f"Failed to log backtesting metrics: {e}")
    
    def _extract_risk_metrics(self, performance_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract risk-related metrics from performance data."""
        try:
            return {
                'max_drawdown': performance_metrics.get('max_drawdown', 0.0),
                'max_drawdown_duration': float(performance_metrics.get('max_drawdown_duration', 0)),
                'var_95': performance_metrics.get('var_95', 0.0),
                'cvar_95': performance_metrics.get('cvar_95', 0.0),
                'volatility': performance_metrics.get('volatility', 0.0),
                'downside_deviation': performance_metrics.get('downside_deviation', 0.0),
                'tail_ratio': performance_metrics.get('tail_ratio', 0.0),
                'skewness': performance_metrics.get('skewness', 0.0),
                'kurtosis': performance_metrics.get('kurtosis', 0.0)
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract risk metrics: {e}")
            return {'max_drawdown': 0.0}
    
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
                    step_name="Step18_Backtesting",
                    additional_data={'file_path': str(self.financial_logger.current_file_path)}
                )
            logger.info("📁 File paths logged for Step18")
        except Exception as e:
            logger.warning(f"Could not log file paths: {e}")