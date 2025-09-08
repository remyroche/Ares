"""
Financial metrics logging for Step20 Final Backtesting.
Independent logging module that can be used without the reporting system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from src.utils.financial_metrics_logger import get_financial_metrics_logger, financial_metrics_context
from src.utils.logger import system_logger

logger = system_logger.getChild('Step20FinancialLogging')


class Step20FinancialLogger:
    """Independent financial metrics logger for Step20 Final Backtesting."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.financial_logger = get_financial_metrics_logger()
    
    def log_step_execution(self, final_backtest_results: Dict[str, Any], performance_metrics: Dict[str, Any], 
                          execution_data: Dict[str, Any], final_analysis: Dict[str, Any]) -> None:
        """Log comprehensive financial metrics for Step20 execution."""
        with financial_metrics_context(
            step_name="Step20_Final_Backtesting",
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe
        ):
            try:
                self.financial_logger.log_step_start("Step20_Final_Backtesting", self.symbol, self.exchange, self.timeframe)
                
                # Log all financial metrics
                self._log_final_backtesting_metrics(final_backtest_results, performance_metrics, execution_data, final_analysis)
                
                # Log file paths
                self._log_created_file_paths()
                
                self.financial_logger.log_step_end("Step20_Final_Backtesting", self.symbol, self.exchange, self.timeframe, success=True)
                
            except Exception as e:
                self.financial_logger.log_step_end("Step20_Final_Backtesting", self.symbol, self.exchange, self.timeframe, success=False, error_message=str(e))
                logger.error(f"Failed to log financial metrics: {e}")
    
    def _log_final_backtesting_metrics(self, final_backtest_results: Dict[str, Any], performance_metrics: Dict[str, Any],
                                     execution_data: Dict[str, Any], final_analysis: Dict[str, Any]) -> None:
        """Log key financial metrics directly from step results."""
        try:
            # Log comprehensive trading performance metrics
            if performance_metrics:
                # Log all performance metrics using the comprehensive trading performance method
                self.financial_logger.log_trading_performance(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step20_Final_Backtesting",
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
            
            # Log final backtesting specific metrics
            if final_backtest_results:
                # Log final strategy performance
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="final_strategy_score",
                    metric_value=final_backtest_results.get('strategy_score', 0.0),
                    metric_type="performance",
                    step_name="Step20_Final_Backtesting"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="final_validation_score",
                    metric_value=final_backtest_results.get('validation_score', 0.0),
                    metric_type="performance",
                    step_name="Step20_Final_Backtesting"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="final_robustness_score",
                    metric_value=final_backtest_results.get('robustness_score', 0.0),
                    metric_type="performance",
                    step_name="Step20_Final_Backtesting"
                )
                
                # Log production readiness metrics
                if 'production_readiness' in final_backtest_results:
                    prod_readiness = final_backtest_results['production_readiness']
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="production_readiness_score",
                        metric_value=prod_readiness.get('readiness_score', 0.0),
                        metric_type="performance",
                        step_name="Step20_Final_Backtesting"
                    )
                    
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="production_risk_score",
                        metric_value=prod_readiness.get('risk_score', 0.0),
                        metric_type="risk",
                        step_name="Step20_Final_Backtesting"
                    )
            
            # Log final analysis results
            if final_analysis:
                # Log strategy comparison metrics
                if 'strategy_comparison' in final_analysis:
                    comparison = final_analysis['strategy_comparison']
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="strategy_ranking",
                        metric_value=float(comparison.get('ranking', 0)),
                        metric_type="performance",
                        step_name="Step20_Final_Backtesting"
                    )
                    
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="strategy_percentile",
                        metric_value=comparison.get('percentile', 0.0),
                        metric_type="performance",
                        step_name="Step20_Final_Backtesting"
                    )
                
                # Log market condition analysis
                if 'market_conditions' in final_analysis:
                    market_conditions = final_analysis['market_conditions']
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="market_condition_score",
                        metric_value=market_conditions.get('condition_score', 0.0),
                        metric_type="performance",
                        step_name="Step20_Final_Backtesting"
                    )
                    
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="market_adaptability_score",
                        metric_value=market_conditions.get('adaptability_score', 0.0),
                        metric_type="performance",
                        step_name="Step20_Final_Backtesting"
                    )
            
            # Log comprehensive risk assessment
            if performance_metrics:
                comprehensive_risk_metrics = self._extract_comprehensive_risk_metrics(performance_metrics)
                for metric_name, metric_value in comprehensive_risk_metrics.items():
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name=f"comprehensive_risk_{metric_name}",
                        metric_value=metric_value,
                        metric_type="risk",
                        step_name="Step20_Final_Backtesting"
                    )
            
        except Exception as e:
            logger.error(f"Failed to log final backtesting metrics: {e}")
    
    def _extract_comprehensive_risk_metrics(self, performance_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract comprehensive risk-related metrics from performance data."""
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
                'gain_loss_ratio': performance_metrics.get('gain_loss_ratio', 0.0),
                'information_ratio': performance_metrics.get('information_ratio', 0.0),
                'treynor_ratio': performance_metrics.get('treynor_ratio', 0.0),
                'jensen_alpha': performance_metrics.get('jensen_alpha', 0.0),
                'tracking_error': performance_metrics.get('tracking_error', 0.0),
                'beta': performance_metrics.get('beta', 0.0)
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract comprehensive risk metrics: {e}")
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
                    step_name="Step20_Final_Backtesting",
                    additional_data={'file_path': str(self.financial_logger.current_file_path)}
                )
            logger.info("📁 File paths logged for Step20")
        except Exception as e:
            logger.warning(f"Could not log file paths: {e}")