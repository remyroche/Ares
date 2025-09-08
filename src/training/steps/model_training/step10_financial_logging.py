"""
Financial metrics logging for Step10 Unified Regime Intelligence.
Independent logging module that can be used without the reporting system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from src.utils.financial_metrics_logger import get_financial_metrics_logger, financial_metrics_context
from src.utils.logger import system_logger

logger = system_logger.getChild('Step10FinancialLogging')


class Step10FinancialLogger:
    """Independent financial metrics logger for Step10 Unified Regime Intelligence."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.financial_logger = get_financial_metrics_logger()
    
    def log_step_execution(self, analysis_results: Dict[str, Any], prediction_results: Dict[str, Any], 
                          integration_metrics: Dict[str, Any], performance_data: Dict[str, Any]) -> None:
        """Log comprehensive financial metrics for Step10 execution."""
        with financial_metrics_context(
            step_name="Step10_Unified_Regime_Intelligence",
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe
        ):
            try:
                self.financial_logger.log_step_start("Step10_Unified_Regime_Intelligence", self.symbol, self.exchange, self.timeframe)
                
                # Log all financial metrics
                self._log_financial_metrics_from_results(analysis_results, prediction_results, integration_metrics, performance_data)
                
                # Log file paths
                self._log_created_file_paths()
                
                self.financial_logger.log_step_end("Step10_Unified_Regime_Intelligence", self.symbol, self.exchange, self.timeframe, success=True)
                
            except Exception as e:
                self.financial_logger.log_step_end("Step10_Unified_Regime_Intelligence", self.symbol, self.exchange, self.timeframe, success=False, error_message=str(e))
                logger.error(f"Failed to log financial metrics: {e}")
    
    def _log_financial_metrics_from_results(self, analysis_results: Dict[str, Any], prediction_results: Dict[str, Any], 
                                          integration_metrics: Dict[str, Any], performance_data: Dict[str, Any]) -> None:
        """Log key financial metrics directly from step results."""
        try:
            # Note: Data quality and performance metrics are logged in regular system logs
            # Financial metrics logger focuses only on financial/trading metrics
            
            # Log multi-timeframe HMM analysis metrics (financial relevance)
            if analysis_results:
                # Log regime detection confidence
                if 'regime_detection_confidence' in analysis_results:
                    for timeframe, confidence in analysis_results['regime_detection_confidence'].items():
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"regime_detection_confidence_{timeframe}",
                            metric_value=confidence,
                            metric_type="trading",
                            step_name="Step10_Unified_Regime_Intelligence",
                            additional_data={'timeframe': timeframe}
                        )
                
                # Log temporal consistency score
                if 'temporal_consistency_score' in analysis_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="temporal_consistency_score",
                        metric_value=analysis_results['temporal_consistency_score'],
                        metric_type="trading",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
                
                # Log cross-timeframe regime alignment
                if 'cross_timeframe_regime_alignment' in analysis_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="cross_timeframe_regime_alignment",
                        metric_value=analysis_results['cross_timeframe_regime_alignment'],
                        metric_type="trading",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
            
            # Log intensity-based prediction metrics
            if prediction_results:
                # Log intensity-based confidence
                if 'intensity_based_confidence' in prediction_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="intensity_based_confidence",
                        metric_value=prediction_results['intensity_based_confidence'],
                        metric_type="trading",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
                
                # Log prediction accuracy by intensity
                if 'prediction_accuracy_by_intensity' in prediction_results:
                    for intensity_level, accuracy in prediction_results['prediction_accuracy_by_intensity'].items():
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"prediction_accuracy_intensity_{intensity_level}",
                            metric_value=accuracy,
                            metric_type="performance",
                            step_name="Step10_Unified_Regime_Intelligence",
                            additional_data={'intensity_level': intensity_level}
                        )
                
                # Log false positive/negative rates
                if 'false_positive_rate' in prediction_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="false_positive_rate",
                        metric_value=prediction_results['false_positive_rate'],
                        metric_type="risk",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
                
                if 'false_negative_rate' in prediction_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="false_negative_rate",
                        metric_value=prediction_results['false_negative_rate'],
                        metric_type="risk",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
            
            # Log TPSL integration metrics
            if integration_metrics:
                # Log TPSL signal generation
                if 'take_profit_signals_generated' in integration_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="take_profit_signals_generated",
                        metric_value=float(integration_metrics['take_profit_signals_generated']),
                        metric_type="trading",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
                
                if 'stop_loss_signals_generated' in integration_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="stop_loss_signals_generated",
                        metric_value=float(integration_metrics['stop_loss_signals_generated']),
                        metric_type="trading",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
                
                # Log combined TPSL accuracy
                if 'combined_tpsl_accuracy' in integration_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="combined_tpsl_accuracy",
                        metric_value=integration_metrics['combined_tpsl_accuracy'],
                        metric_type="performance",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
                
                # Log direction prediction confidence
                if 'direction_prediction_confidence' in integration_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="direction_prediction_confidence",
                        metric_value=integration_metrics['direction_prediction_confidence'],
                        metric_type="trading",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
                
                # Log risk management effectiveness
                if 'risk_management_effectiveness' in integration_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="risk_management_effectiveness",
                        metric_value=integration_metrics['risk_management_effectiveness'],
                        metric_type="risk",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
                
                # Log profit factor
                if 'profit_factor' in integration_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="profit_factor",
                        metric_value=integration_metrics['profit_factor'],
                        metric_type="performance",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
                
                # Log S/R integration metrics
                if 'sr_levels_identified' in integration_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="sr_levels_identified",
                        metric_value=float(integration_metrics['sr_levels_identified']),
                        metric_type="trading",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
                
                if 'sr_based_signals' in integration_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="sr_based_signals",
                        metric_value=float(integration_metrics['sr_based_signals']),
                        metric_type="trading",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
                
                if 'sr_confidence_boost' in integration_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="sr_confidence_boost",
                        metric_value=integration_metrics['sr_confidence_boost'],
                        metric_type="trading",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
                
                if 'combined_sr_regime_accuracy' in integration_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="combined_sr_regime_accuracy",
                        metric_value=integration_metrics['combined_sr_regime_accuracy'],
                        metric_type="performance",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
            
            # Log position logic metrics
            if prediction_results:
                # Log trading signals
                if 'total_trading_signals' in prediction_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="total_trading_signals",
                        metric_value=float(prediction_results['total_trading_signals']),
                        metric_type="trading",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
                
                if 'buy_signals_generated' in prediction_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="buy_signals_generated",
                        metric_value=float(prediction_results['buy_signals_generated']),
                        metric_type="trading",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
                
                if 'sell_signals_generated' in prediction_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="sell_signals_generated",
                        metric_value=float(prediction_results['sell_signals_generated']),
                        metric_type="trading",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
                
                if 'hold_signals_generated' in prediction_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="hold_signals_generated",
                        metric_value=float(prediction_results['hold_signals_generated']),
                        metric_type="trading",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
                
                # Log position transition accuracy
                if 'position_transition_accuracy' in prediction_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="position_transition_accuracy",
                        metric_value=prediction_results['position_transition_accuracy'],
                        metric_type="performance",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
                
                # Log risk-adjusted returns
                if 'risk_adjusted_returns' in prediction_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="risk_adjusted_returns",
                        metric_value=prediction_results['risk_adjusted_returns'],
                        metric_type="performance",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
            
            # Log unified model performance metrics
            if performance_data:
                # Log overall model performance
                if 'overall_accuracy' in performance_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="unified_model_accuracy",
                        metric_value=performance_data['overall_accuracy'],
                        metric_type="performance",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
                
                if 'precision_score' in performance_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="unified_model_precision",
                        metric_value=performance_data['precision_score'],
                        metric_type="performance",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
                
                if 'recall_score' in performance_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="unified_model_recall",
                        metric_value=performance_data['recall_score'],
                        metric_type="performance",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
                
                if 'f1_score' in performance_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="unified_model_f1_score",
                        metric_value=performance_data['f1_score'],
                        metric_type="performance",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
                
                # Log multi-timeframe consistency
                if 'multi_timeframe_consistency' in performance_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="multi_timeframe_consistency",
                        metric_value=performance_data['multi_timeframe_consistency'],
                        metric_type="trading",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
                
                # Log prediction stability
                if 'prediction_stability' in performance_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="prediction_stability",
                        metric_value=performance_data['prediction_stability'],
                        metric_type="trading",
                        step_name="Step10_Unified_Regime_Intelligence"
                    )
                
                # Log regime classification accuracy
                if 'regime_classification_accuracy' in performance_data:
                    for regime_id, accuracy in performance_data['regime_classification_accuracy'].items():
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"regime_classification_accuracy_{regime_id}",
                            metric_value=accuracy,
                            metric_type="performance",
                            step_name="Step10_Unified_Regime_Intelligence",
                            regime_id=str(regime_id)
                        )
            
            # Log comprehensive trading performance estimation
            if analysis_results and prediction_results and integration_metrics and performance_data:
                # Estimate trading performance based on unified regime intelligence results
                overall_accuracy = performance_data.get('overall_accuracy', 0.5)
                intensity_confidence = prediction_results.get('intensity_based_confidence', 0.5)
                tpsl_accuracy = integration_metrics.get('combined_tpsl_accuracy', 0.5)
                sr_accuracy = integration_metrics.get('combined_sr_regime_accuracy', 0.5)
                
                # Estimate returns based on combined model performance
                combined_accuracy = (overall_accuracy + intensity_confidence + tpsl_accuracy + sr_accuracy) / 4
                estimated_return = (combined_accuracy * 0.03) - ((1 - combined_accuracy) * 0.015)  # Improved estimate
                estimated_volatility = 0.025  # Slightly higher for regime-based trading
                
                # Estimate trading metrics
                total_signals = prediction_results.get('total_trading_signals', 50)
                buy_signals = prediction_results.get('buy_signals_generated', 25)
                sell_signals = prediction_results.get('sell_signals_generated', 20)
                hold_signals = prediction_results.get('hold_signals_generated', 5)
                
                estimated_performance = {
                    'total_return': estimated_return,
                    'annualized_return': estimated_return * 252,  # Assuming daily signals
                    'volatility': estimated_volatility,
                    'sharpe_ratio': estimated_return / estimated_volatility if estimated_volatility > 0 else 0.0,
                    'sortino_ratio': estimated_return / (estimated_volatility * 0.6) if estimated_volatility > 0 else 0.0,
                    'calmar_ratio': 0.0,  # Would need max drawdown
                    'max_drawdown': estimated_volatility * 2.5,  # Estimate
                    'max_drawdown_duration': 30,  # Default estimate
                    'var_95': estimated_volatility * 1.8,  # Estimate
                    'cvar_95': estimated_volatility * 2.2,  # Estimate
                    'win_rate': combined_accuracy,
                    'profit_factor': integration_metrics.get('profit_factor', 1.0 + (combined_accuracy - 0.5) * 2),
                    'avg_win': 0.025,  # Default estimate
                    'avg_loss': 0.015,  # Default estimate
                    'largest_win': 0.06,  # Default estimate
                    'largest_loss': estimated_volatility * 2.5,  # Estimate
                    'total_trades': total_signals,
                    'winning_trades': int(total_signals * combined_accuracy),
                    'losing_trades': int(total_signals * (1 - combined_accuracy)),
                    'additional_metrics': {
                        'overall_accuracy': overall_accuracy,
                        'intensity_based_confidence': intensity_confidence,
                        'combined_tpsl_accuracy': tpsl_accuracy,
                        'combined_sr_regime_accuracy': sr_accuracy,
                        'temporal_consistency_score': analysis_results.get('temporal_consistency_score', 0.0),
                        'cross_timeframe_regime_alignment': analysis_results.get('cross_timeframe_regime_alignment', 0.0),
                        'multi_timeframe_consistency': performance_data.get('multi_timeframe_consistency', 0.0),
                        'prediction_stability': performance_data.get('prediction_stability', 0.0),
                        'total_trading_signals': total_signals,
                        'buy_signals': buy_signals,
                        'sell_signals': sell_signals,
                        'hold_signals': hold_signals
                    }
                }
                
                self.financial_logger.log_trading_performance(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step10_Unified_Regime_Intelligence",
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
                    step_name="Step10_Unified_Regime_Intelligence",
                    additional_data={'file_path': str(self.financial_logger.current_file_path)}
                )
            logger.info("📁 File paths logged for Step10")
        except Exception as e:
            logger.warning(f"Could not log file paths: {e}")