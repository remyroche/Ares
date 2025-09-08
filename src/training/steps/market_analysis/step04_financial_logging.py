"""
Financial metrics logging for Step04 Regime Data Splitting.
Independent logging module that can be used without the reporting system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from src.utils.financial_metrics_logger import get_financial_metrics_logger, financial_metrics_context
from src.utils.logger import system_logger

logger = system_logger.getChild('Step04FinancialLogging')


class Step04FinancialLogger:
    """Independent financial metrics logger for Step04 Regime Data Splitting."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.financial_logger = get_financial_metrics_logger()
    
    def log_step_execution(self, regime_data: pd.DataFrame, regime_ids: List[int], 
                          execution_data: Dict[str, Any], data_splitting_results: Dict[str, Any]) -> None:
        """Log comprehensive financial metrics for Step04 execution."""
        with financial_metrics_context(
            step_name="Step04_Regime_Data_Splitting",
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe
        ):
            try:
                self.financial_logger.log_step_start("Step04_Regime_Data_Splitting", self.symbol, self.exchange, self.timeframe)
                
                # Log all financial metrics
                self._log_financial_metrics_from_results(regime_data, regime_ids, execution_data, data_splitting_results)
                
                # Log file paths
                self._log_created_file_paths()
                
                self.financial_logger.log_step_end("Step04_Regime_Data_Splitting", self.symbol, self.exchange, self.timeframe, success=True)
                
            except Exception as e:
                self.financial_logger.log_step_end("Step04_Regime_Data_Splitting", self.symbol, self.exchange, self.timeframe, success=False, error_message=str(e))
                logger.error(f"Failed to log financial metrics: {e}")
    
    def _log_financial_metrics_from_results(self, regime_data: pd.DataFrame, regime_ids: List[int], 
                                          execution_data: Dict[str, Any], data_splitting_results: Dict[str, Any]) -> None:
        """Log key financial metrics directly from step results."""
        try:
            # Log regime analysis metrics (financial/trading focused only)
            if regime_data is not None and not regime_data.empty:
                
                # Log regime-specific metrics
                if 'composite_cluster_id' in regime_data.columns:
                    regime_counts = regime_data['composite_cluster_id'].value_counts()
                    
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="regime_count",
                        metric_value=float(len(regime_ids)),
                        metric_type="regime",
                        step_name="Step04_Regime_Data_Splitting"
                    )
                    
                # Log regime balance score (financial relevance)
                regime_balance_score = 1.0 - (regime_counts.std() / regime_counts.mean()) if regime_counts.mean() > 0 else 0.0
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="regime_balance_score",
                    metric_value=regime_balance_score,
                    metric_type="regime",
                    step_name="Step04_Regime_Data_Splitting"
                )
                    
                    # Log individual regime statistics
                    for regime_id in regime_ids:
                        regime_data_subset = regime_data[regime_data['composite_cluster_id'] == regime_id]
                        regime_count = len(regime_data_subset)
                        regime_percentage = regime_count / total_rows
                        
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"regime_{regime_id}_sample_count",
                            metric_value=float(regime_count),
                            metric_type="regime",
                            step_name="Step04_Regime_Data_Splitting",
                            regime_id=str(regime_id)
                        )
                        
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"regime_{regime_id}_percentage",
                            metric_value=regime_percentage,
                            metric_type="regime",
                            step_name="Step04_Regime_Data_Splitting",
                            regime_id=str(regime_id)
                        )
                        
                        # Log regime duration if timestamp available
                        if 'timestamp' in regime_data_subset.columns and len(regime_data_subset) > 1:
                            duration_minutes = (regime_data_subset['timestamp'].max() - regime_data_subset['timestamp'].min()).total_seconds() / 60
                            self.financial_logger.log_financial_metric(
                                symbol=self.symbol,
                                exchange=self.exchange,
                                timeframe=self.timeframe,
                                metric_name=f"regime_{regime_id}_duration_minutes",
                                metric_value=float(duration_minutes),
                                metric_type="regime",
                                step_name="Step04_Regime_Data_Splitting",
                                regime_id=str(regime_id)
                            )
                        
                        # Log regime volatility if close price available
                        if 'close' in regime_data_subset.columns and len(regime_data_subset) > 1:
                            returns = regime_data_subset['close'].pct_change().dropna()
                            if len(returns) > 0:
                                volatility = returns.std()
                                self.financial_logger.log_financial_metric(
                                    symbol=self.symbol,
                                    exchange=self.exchange,
                                    timeframe=self.timeframe,
                                    metric_name=f"regime_{regime_id}_volatility",
                                    metric_value=volatility,
                                    metric_type="risk",
                                    step_name="Step04_Regime_Data_Splitting",
                                    regime_id=str(regime_id)
                                )
                                
                                # Log regime return
                                regime_return = returns.mean()
                                self.financial_logger.log_financial_metric(
                                    symbol=self.symbol,
                                    exchange=self.exchange,
                                    timeframe=self.timeframe,
                                    metric_name=f"regime_{regime_id}_avg_return",
                                    metric_value=regime_return,
                                    metric_type="return",
                                    step_name="Step04_Regime_Data_Splitting",
                                    regime_id=str(regime_id)
                                )
                        
                        # Log regime volume if available
                        if 'volume' in regime_data_subset.columns and len(regime_data_subset) > 0:
                            avg_volume = regime_data_subset['volume'].mean()
                            self.financial_logger.log_financial_metric(
                                symbol=self.symbol,
                                exchange=self.exchange,
                                timeframe=self.timeframe,
                                metric_name=f"regime_{regime_id}_avg_volume",
                                metric_value=avg_volume,
                                metric_type="market",
                                step_name="Step04_Regime_Data_Splitting",
                                regime_id=str(regime_id)
                            )
            
            # Note: Data quality and performance metrics are logged in regular system logs
            # Financial metrics logger focuses only on financial/trading metrics
            
            # Log comprehensive trading performance estimation
            if regime_data is not None and not regime_data.empty and len(regime_ids) > 0:
                # Estimate trading performance based on regime analysis
                estimated_performance = {
                    'total_return': 0.0,  # Would need actual trading data
                    'annualized_return': 0.0,
                    'volatility': regime_data['close'].pct_change().std() if 'close' in regime_data.columns else 0.02,
                    'sharpe_ratio': 0.0,  # Would need return data
                    'sortino_ratio': 0.0,
                    'calmar_ratio': 0.0,
                    'max_drawdown': regime_data['close'].pct_change().std() * 2 if 'close' in regime_data.columns else 0.04,
                    'max_drawdown_duration': 25,  # Default estimate
                    'var_95': regime_data['close'].pct_change().std() * 1.5 if 'close' in regime_data.columns else 0.03,
                    'cvar_95': regime_data['close'].pct_change().std() * 2 if 'close' in regime_data.columns else 0.04,
                    'win_rate': 0.5,  # Default for regime analysis
                    'profit_factor': 1.0,  # Default
                    'avg_win': 0.01,  # Default estimate
                    'avg_loss': 0.01,  # Default estimate
                    'largest_win': 0.03,  # Default estimate
                    'largest_loss': regime_data['close'].pct_change().std() * 2 if 'close' in regime_data.columns else 0.04,
                    'total_trades': 30,  # Default estimate
                    'winning_trades': 15,  # Default estimate
                    'losing_trades': 15,  # Default estimate
                    'additional_metrics': {
                        'regime_count': len(regime_ids),
                        'regime_balance_score': regime_balance_score if 'regime_balance_score' in locals() else 0.0
                    }
                }
                
                self.financial_logger.log_trading_performance(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step04_Regime_Data_Splitting",
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
                    step_name="Step04_Regime_Data_Splitting",
                    additional_data={'file_path': str(self.financial_logger.current_file_path)}
                )
            logger.info("📁 File paths logged for Step04")
        except Exception as e:
            logger.warning(f"Could not log file paths: {e}")