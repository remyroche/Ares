"""
Financial metrics logging for Step05 Labeling.
Independent logging module that can be used without the reporting system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from src.utils.financial_metrics_logger import get_financial_metrics_logger, financial_metrics_context
from src.utils.logger import system_logger

logger = system_logger.getChild('Step05FinancialLogging')


class Step05FinancialLogger:
    """Independent financial metrics logger for Step05 Labeling."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.financial_logger = get_financial_metrics_logger()
    
    def log_step_execution(self, labeled_data: pd.DataFrame, label_stats: Dict[str, Any], 
                          execution_data: Dict[str, Any], labeling_results: Dict[str, Any]) -> None:
        """Log comprehensive financial metrics for Step05 execution."""
        with financial_metrics_context(
            step_name="Step05_Labeling",
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe
        ):
            try:
                self.financial_logger.log_step_start("Step05_Labeling", self.symbol, self.exchange, self.timeframe)
                
                # Log all financial metrics
                self._log_financial_metrics_from_results(labeled_data, label_stats, execution_data, labeling_results)
                
                # Log file paths
                self._log_created_file_paths()
                
                self.financial_logger.log_step_end("Step05_Labeling", self.symbol, self.exchange, self.timeframe, success=True)
                
            except Exception as e:
                self.financial_logger.log_step_end("Step05_Labeling", self.symbol, self.exchange, self.timeframe, success=False, error_message=str(e))
                logger.error(f"Failed to log financial metrics: {e}")
    
    def _log_financial_metrics_from_results(self, labeled_data: pd.DataFrame, label_stats: Dict[str, Any], 
                                          execution_data: Dict[str, Any], labeling_results: Dict[str, Any]) -> None:
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
                    metric_name="total_labels_created",
                    metric_value=float(label_stats.get('total_labels', 0)),
                    metric_type="performance",
                    step_name="Step05_Labeling"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="buy_labels_count",
                    metric_value=float(label_stats.get('buy_labels', 0)),
                    metric_type="performance",
                    step_name="Step05_Labeling"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="sell_labels_count",
                    metric_value=float(label_stats.get('sell_labels', 0)),
                    metric_type="performance",
                    step_name="Step05_Labeling"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="hold_labels_count",
                    metric_value=float(label_stats.get('hold_labels', 0)),
                    metric_type="performance",
                    step_name="Step05_Labeling"
                )
                
                # Log label distribution balance (financial relevance)
                total_labels = label_stats.get('total_labels', 1)
                buy_ratio = label_stats.get('buy_labels', 0) / total_labels
                sell_ratio = label_stats.get('sell_labels', 0) / total_labels
                hold_ratio = label_stats.get('hold_labels', 0) / total_labels
                
                # Calculate label distribution balance (closer to 0.33 each is better)
                ideal_ratio = 1.0 / 3.0
                distribution_balance = 1.0 - (abs(buy_ratio - ideal_ratio) + abs(sell_ratio - ideal_ratio) + abs(hold_ratio - ideal_ratio))
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="label_distribution_balance",
                    metric_value=distribution_balance,
                    metric_type="trading",
                    step_name="Step05_Labeling"
                )
                
                # Log label quality metrics (financial relevance)
                if 'label_confidence_score' in label_stats:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="label_confidence_score",
                        metric_value=label_stats['label_confidence_score'],
                        metric_type="trading",
                        step_name="Step05_Labeling"
                    )
                
                if 'label_consistency_score' in label_stats:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="label_consistency_score",
                        metric_value=label_stats['label_consistency_score'],
                        metric_type="trading",
                        step_name="Step05_Labeling"
                    )
                
                if 'label_purity_score' in label_stats:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="label_purity_score",
                        metric_value=label_stats['label_purity_score'],
                        metric_type="trading",
                        step_name="Step05_Labeling"
                    )
                
                # Log false positive and negative rates (financial relevance)
                if 'false_positive_rate' in label_stats:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="false_positive_rate",
                        metric_value=label_stats['false_positive_rate'],
                        metric_type="trading",
                        step_name="Step05_Labeling"
                    )
                
                if 'false_negative_rate' in label_stats:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="false_negative_rate",
                        metric_value=label_stats['false_negative_rate'],
                        metric_type="trading",
                        step_name="Step05_Labeling"
                    )
                
                # Log label accuracy estimate (financial relevance)
                if 'label_accuracy_estimate' in label_stats:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="label_accuracy_estimate",
                        metric_value=label_stats['label_accuracy_estimate'],
                        metric_type="trading",
                        step_name="Step05_Labeling"
                    )
            
            # Log meta-labeling analysis if available
            if labeling_results and 'meta_labeling_analysis' in labeling_results:
                meta_analysis = labeling_results['meta_labeling_analysis']
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="meta_labels_created",
                    metric_value=float(meta_analysis.get('meta_labels_created', 0)),
                    metric_type="performance",
                    step_name="Step05_Labeling"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="meta_labeling_success_rate",
                    metric_value=meta_analysis.get('meta_labeling_success_rate', 0.0),
                    metric_type="trading",
                    step_name="Step05_Labeling"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="meta_label_confidence_avg",
                    metric_value=meta_analysis.get('meta_label_confidence_avg', 0.0),
                    metric_type="trading",
                    step_name="Step05_Labeling"
                )
            
            # Note: Execution performance metrics are logged in regular system logs
            # Financial metrics logger focuses only on financial/trading metrics
            
            # Log comprehensive trading performance estimation
            if labeled_data is not None and not labeled_data.empty and label_stats:
                # Estimate trading performance based on labeling results
                total_labels = label_stats.get('total_labels', 0)
                label_accuracy = label_stats.get('label_accuracy_estimate', 0.5)
                label_confidence = label_stats.get('label_confidence_score', 0.5)
                
                # Estimate returns based on label quality
                estimated_return = (label_accuracy * 0.02) - ((1 - label_accuracy) * 0.01)  # Rough estimate
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
                    'win_rate': label_accuracy,
                    'profit_factor': 1.0 + (label_accuracy - 0.5) * 2,  # Estimate based on accuracy
                    'avg_win': 0.02,  # Default estimate
                    'avg_loss': 0.01,  # Default estimate
                    'largest_win': 0.05,  # Default estimate
                    'largest_loss': estimated_volatility * 2,  # Estimate
                    'total_trades': total_labels,
                    'winning_trades': int(total_labels * label_accuracy),
                    'losing_trades': int(total_labels * (1 - label_accuracy)),
                    'additional_metrics': {
                        'label_distribution_balance': distribution_balance if 'distribution_balance' in locals() else 0.0,
                        'label_confidence_score': label_confidence,
                        'label_consistency_score': label_stats.get('label_consistency_score', 0.0),
                        'label_purity_score': label_stats.get('label_purity_score', 0.0),
                        'false_positive_rate': label_stats.get('false_positive_rate', 0.0),
                        'false_negative_rate': label_stats.get('false_negative_rate', 0.0)
                    }
                }
                
                self.financial_logger.log_trading_performance(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step05_Labeling",
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
                    step_name="Step05_Labeling",
                    additional_data={'file_path': str(self.financial_logger.current_file_path)}
                )
            logger.info("📁 File paths logged for Step05")
        except Exception as e:
            logger.warning(f"Could not log file paths: {e}")