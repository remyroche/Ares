"""
Financial metrics logging for Step19 Advanced Backtesting.
Independent logging module that can be used without the reporting system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from src.utils.financial_metrics_logger import get_financial_metrics_logger, financial_metrics_context
from src.utils.logger import system_logger

logger = system_logger.getChild('Step19FinancialLogging')


class Step19FinancialLogger:
    """Independent financial metrics logger for Step19 Advanced Backtesting."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.financial_logger = get_financial_metrics_logger()
    
    def log_step_execution(self, advanced_backtest_results: Dict[str, Any], performance_metrics: Dict[str, Any], 
                          execution_data: Dict[str, Any], optimization_results: Dict[str, Any]) -> None:
        """Log comprehensive financial metrics for Step19 execution."""
        with financial_metrics_context(
            step_name="Step19_Advanced_Backtesting",
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe
        ):
            try:
                self.financial_logger.log_step_start("Step19_Advanced_Backtesting", self.symbol, self.exchange, self.timeframe)
                
                # Log all financial metrics
                self._log_advanced_backtesting_metrics(advanced_backtest_results, performance_metrics, execution_data, optimization_results)
                
                # Log file paths
                self._log_created_file_paths()
                
                self.financial_logger.log_step_end("Step19_Advanced_Backtesting", self.symbol, self.exchange, self.timeframe, success=True)
                
            except Exception as e:
                self.financial_logger.log_step_end("Step19_Advanced_Backtesting", self.symbol, self.exchange, self.timeframe, success=False, error_message=str(e))
                logger.error(f"Failed to log financial metrics: {e}")
    
    def _log_advanced_backtesting_metrics(self, advanced_backtest_results: Dict[str, Any], performance_metrics: Dict[str, Any],
                                        execution_data: Dict[str, Any], optimization_results: Dict[str, Any]) -> None:
        """Log key financial metrics directly from step results."""
        try:
            # Log comprehensive trading performance metrics
            if performance_metrics:
                # Log all performance metrics using the comprehensive trading performance method
                self.financial_logger.log_trading_performance(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step19_Advanced_Backtesting",
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
            
            # Log advanced backtesting specific metrics
            if advanced_backtest_results:
                # Log Monte Carlo simulation results
                if 'monte_carlo_results' in advanced_backtest_results:
                    mc_results = advanced_backtest_results['monte_carlo_results']
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="monte_carlo_simulations_count",
                        metric_value=float(mc_results.get('simulations_count', 0)),
                        metric_type="performance",
                        step_name="Step19_Advanced_Backtesting"
                    )
                    
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="monte_carlo_confidence_95",
                        metric_value=mc_results.get('confidence_95', 0.0),
                        metric_type="risk",
                        step_name="Step19_Advanced_Backtesting"
                    )
                
                # Log walk-forward analysis results
                if 'walk_forward_results' in advanced_backtest_results:
                    wf_results = advanced_backtest_results['walk_forward_results']
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="walk_forward_periods",
                        metric_value=float(wf_results.get('periods_count', 0)),
                        metric_type="performance",
                        step_name="Step19_Advanced_Backtesting"
                    )
                    
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="walk_forward_consistency",
                        metric_value=wf_results.get('consistency_score', 0.0),
                        metric_type="performance",
                        step_name="Step19_Advanced_Backtesting"
                    )
                
                # Log regime-specific performance
                if 'regime_performance' in advanced_backtest_results:
                    regime_perf = advanced_backtest_results['regime_performance']
                    for regime_id, regime_metrics in regime_perf.items():
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"regime_{regime_id}_return",
                            metric_value=regime_metrics.get('return', 0.0),
                            metric_type="performance",
                            step_name="Step19_Advanced_Backtesting",
                            regime_id=regime_id
                        )
            
            # Log optimization results
            if optimization_results:
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="optimization_iterations",
                    metric_value=float(optimization_results.get('iterations', 0)),
                    metric_type="performance",
                    step_name="Step19_Advanced_Backtesting"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="optimization_best_score",
                    metric_value=optimization_results.get('best_score', 0.0),
                    metric_type="performance",
                    step_name="Step19_Advanced_Backtesting"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="optimization_convergence_rate",
                    metric_value=optimization_results.get('convergence_rate', 0.0),
                    metric_type="performance",
                    step_name="Step19_Advanced_Backtesting"
                )
            
            # Log advanced risk metrics
            if performance_metrics:
                advanced_risk_metrics = self._extract_advanced_risk_metrics(performance_metrics)
                for metric_name, metric_value in advanced_risk_metrics.items():
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name=f"advanced_risk_{metric_name}",
                        metric_value=metric_value,
                        metric_type="risk",
                        step_name="Step19_Advanced_Backtesting"
                    )
            
        except Exception as e:
            logger.error(f"Failed to log advanced backtesting metrics: {e}")
    
    def _extract_advanced_risk_metrics(self, performance_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract advanced risk-related metrics from performance data."""
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
                'kurtosis': performance_metrics.get('kurtosis', 0.0),
                'ulcer_index': performance_metrics.get('ulcer_index', 0.0),
                'sterling_ratio': performance_metrics.get('sterling_ratio', 0.0),
                'burke_ratio': performance_metrics.get('burke_ratio', 0.0),
                'kappa_3': performance_metrics.get('kappa_3', 0.0),
                'omega_ratio': performance_metrics.get('omega_ratio', 0.0),
                'gain_loss_ratio': performance_metrics.get('gain_loss_ratio', 0.0)
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract advanced risk metrics: {e}")
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
                    step_name="Step19_Advanced_Backtesting",
                    additional_data={'file_path': str(self.financial_logger.current_file_path)}
                )
            logger.info("📁 File paths logged for Step19")
        except Exception as e:
            logger.warning(f"Could not log file paths: {e}")