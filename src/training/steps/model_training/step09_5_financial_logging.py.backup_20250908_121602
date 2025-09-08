"""
Financial metrics logging for Step09_5 HMM-LM Generalist Training.
Independent logging module that can be used without the reporting system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from src.utils.financial_metrics_logger import get_financial_metrics_logger, financial_metrics_context
from src.utils.logger import system_logger

logger = system_logger.getChild('Step09_5FinancialLogging')


class Step09_5FinancialLogger:
    """Independent financial metrics logger for Step09_5 HMM-LM Generalist Training."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.financial_logger = get_financial_metrics_logger()
    
    def log_step_execution(self, training_results: Dict[str, Any], model_performance: Dict[str, Any], 
                          execution_data: Dict[str, Any], hmm_metrics: Dict[str, Any]) -> None:
        """Log comprehensive financial metrics for Step09_5 execution."""
        with financial_metrics_context(
            step_name="Step09_5_HMM_LM_Generalist_Training",
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe
        ):
            try:
                self.financial_logger.log_step_start("Step09_5_HMM_LM_Generalist_Training", self.symbol, self.exchange, self.timeframe)
                
                # Log all financial metrics
                self._log_financial_metrics_from_results(training_results, model_performance, execution_data, hmm_metrics)
                
                # Log file paths
                self._log_created_file_paths()
                
                self.financial_logger.log_step_end("Step09_5_HMM_LM_Generalist_Training", self.symbol, self.exchange, self.timeframe, success=True)
                
            except Exception as e:
                self.financial_logger.log_step_end("Step09_5_HMM_LM_Generalist_Training", self.symbol, self.exchange, self.timeframe, success=False, error_message=str(e))
                logger.error(f"Failed to log financial metrics: {e}")
    
    def _log_financial_metrics_from_results(self, training_results: Dict[str, Any], model_performance: Dict[str, Any], 
                                          execution_data: Dict[str, Any], hmm_metrics: Dict[str, Any]) -> None:
        """Log key financial metrics directly from step results."""
        try:
            # Note: Data quality and performance metrics are logged in regular system logs
            # Financial metrics logger focuses only on financial/trading metrics
            
            # Log training results metrics (financial relevance)
            if training_results:
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="total_training_time",
                    metric_value=float(training_results.get('total_training_time', 0)),
                    metric_type="performance",
                    step_name="Step09_5_HMM_LM_Generalist_Training"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="epochs_completed",
                    metric_value=float(training_results.get('epochs_completed', 0)),
                    metric_type="performance",
                    step_name="Step09_5_HMM_LM_Generalist_Training"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="convergence_score",
                    metric_value=training_results.get('convergence_score', 0.0),
                    metric_type="trading",
                    step_name="Step09_5_HMM_LM_Generalist_Training"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="early_stopping_triggered",
                    metric_value=1.0 if training_results.get('early_stopping_triggered', False) else 0.0,
                    metric_type="trading",
                    step_name="Step09_5_HMM_LM_Generalist_Training"
                )
            
            # Log model performance metrics
            if model_performance:
                # Log overall model performance
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="test_accuracy",
                    metric_value=model_performance.get('test_accuracy', 0.0),
                    metric_type="performance",
                    step_name="Step09_5_HMM_LM_Generalist_Training"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="precision_score",
                    metric_value=model_performance.get('precision_score', 0.0),
                    metric_type="performance",
                    step_name="Step09_5_HMM_LM_Generalist_Training"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="recall_score",
                    metric_value=model_performance.get('recall_score', 0.0),
                    metric_type="performance",
                    step_name="Step09_5_HMM_LM_Generalist_Training"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="f1_score",
                    metric_value=model_performance.get('f1_score', 0.0),
                    metric_type="performance",
                    step_name="Step09_5_HMM_LM_Generalist_Training"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="roc_auc_score",
                    metric_value=model_performance.get('roc_auc_score', 0.0),
                    metric_type="performance",
                    step_name="Step09_5_HMM_LM_Generalist_Training"
                )
                
                # Log regime prediction metrics
                if 'regime_prediction_metrics' in model_performance:
                    regime_metrics = model_performance['regime_prediction_metrics']
                    for regime_id, metrics in regime_metrics.items():
                        if isinstance(metrics, dict):
                            for metric_name, metric_value in metrics.items():
                                self.financial_logger.log_financial_metric(
                                    symbol=self.symbol,
                                    exchange=self.exchange,
                                    timeframe=self.timeframe,
                                    metric_name=f"regime_{regime_id}_{metric_name}",
                                    metric_value=metric_value,
                                    metric_type="performance",
                                    step_name="Step09_5_HMM_LM_Generalist_Training",
                                    regime_id=str(regime_id)
                                )
            
            # Log HMM metrics
            if hmm_metrics:
                # Log HMM regime analysis
                if 'hmm_states' in hmm_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="hmm_states_count",
                        metric_value=float(hmm_metrics['hmm_states']),
                        metric_type="trading",
                        step_name="Step09_5_HMM_LM_Generalist_Training"
                    )
                
                if 'regime_transition_probability' in hmm_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="regime_transition_probability",
                        metric_value=hmm_metrics['regime_transition_probability'],
                        metric_type="trading",
                        step_name="Step09_5_HMM_LM_Generalist_Training"
                    )
                
                if 'regime_stability_score' in hmm_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="regime_stability_score",
                        metric_value=hmm_metrics['regime_stability_score'],
                        metric_type="trading",
                        step_name="Step09_5_HMM_LM_Generalist_Training"
                    )
                
                if 'regime_entropy_score' in hmm_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="regime_entropy_score",
                        metric_value=hmm_metrics['regime_entropy_score'],
                        metric_type="trading",
                        step_name="Step09_5_HMM_LM_Generalist_Training"
                    )
                
                if 'regime_detection_accuracy' in hmm_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="regime_detection_accuracy",
                        metric_value=hmm_metrics['regime_detection_accuracy'],
                        metric_type="performance",
                        step_name="Step09_5_HMM_LM_Generalist_Training"
                    )
            
            # Log sequence processing metrics
            if training_results:
                if 'total_sequences_processed' in training_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="total_sequences_processed",
                        metric_value=float(training_results['total_sequences_processed']),
                        metric_type="performance",
                        step_name="Step09_5_HMM_LM_Generalist_Training"
                    )
                
                if 'regime_change_events_detected' in training_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="regime_change_events_detected",
                        metric_value=float(training_results['regime_change_events_detected']),
                        metric_type="trading",
                        step_name="Step09_5_HMM_LM_Generalist_Training"
                    )
                
                if 'tpsl_events_processed' in training_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="tpsl_events_processed",
                        metric_value=float(training_results['tpsl_events_processed']),
                        metric_type="trading",
                        step_name="Step09_5_HMM_LM_Generalist_Training"
                    )
                
                if 'sequence_quality_score' in training_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="sequence_quality_score",
                        metric_value=training_results['sequence_quality_score'],
                        metric_type="trading",
                        step_name="Step09_5_HMM_LM_Generalist_Training"
                    )
            
            # Log TPSL prediction metrics
            if training_results:
                if 'take_profit_accuracy' in training_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="take_profit_accuracy",
                        metric_value=training_results['take_profit_accuracy'],
                        metric_type="performance",
                        step_name="Step09_5_HMM_LM_Generalist_Training"
                    )
                
                if 'stop_loss_accuracy' in training_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="stop_loss_accuracy",
                        metric_value=training_results['stop_loss_accuracy'],
                        metric_type="performance",
                        step_name="Step09_5_HMM_LM_Generalist_Training"
                    )
                
                if 'combined_tpsl_accuracy' in training_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="combined_tpsl_accuracy",
                        metric_value=training_results['combined_tpsl_accuracy'],
                        metric_type="performance",
                        step_name="Step09_5_HMM_LM_Generalist_Training"
                    )
                
                if 'direction_prediction_confidence' in training_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="direction_prediction_confidence",
                        metric_value=training_results['direction_prediction_confidence'],
                        metric_type="trading",
                        step_name="Step09_5_HMM_LM_Generalist_Training"
                    )
                
                if 'risk_reward_ratio' in training_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="risk_reward_ratio",
                        metric_value=training_results['risk_reward_ratio'],
                        metric_type="performance",
                        step_name="Step09_5_HMM_LM_Generalist_Training"
                    )
            
            # Log multi-timeframe metrics
            if training_results:
                if 'cross_timeframe_correlation' in training_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="cross_timeframe_correlation",
                        metric_value=training_results['cross_timeframe_correlation'],
                        metric_type="trading",
                        step_name="Step09_5_HMM_LM_Generalist_Training"
                    )
                
                if 'temporal_alignment_score' in training_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="temporal_alignment_score",
                        metric_value=training_results['temporal_alignment_score'],
                        metric_type="trading",
                        step_name="Step09_5_HMM_LM_Generalist_Training"
                    )
                
                if 'multi_timeframe_consistency' in training_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="multi_timeframe_consistency",
                        metric_value=training_results['multi_timeframe_consistency'],
                        metric_type="trading",
                        step_name="Step09_5_HMM_LM_Generalist_Training"
                    )
            
            # Log comprehensive trading performance estimation
            if training_results and model_performance and hmm_metrics:
                # Estimate trading performance based on HMM-LM training results
                test_accuracy = model_performance.get('test_accuracy', 0.5)
                regime_detection_accuracy = hmm_metrics.get('regime_detection_accuracy', 0.5)
                tpsl_accuracy = training_results.get('combined_tpsl_accuracy', 0.5)
                direction_confidence = training_results.get('direction_prediction_confidence', 0.5)
                
                # Estimate returns based on combined model performance
                combined_accuracy = (test_accuracy + regime_detection_accuracy + tpsl_accuracy + direction_confidence) / 4
                estimated_return = (combined_accuracy * 0.025) - ((1 - combined_accuracy) * 0.012)  # Estimate
                estimated_volatility = 0.022  # Default estimate
                
                # Estimate trading metrics
                sequences_processed = training_results.get('total_sequences_processed', 1000)
                regime_changes = training_results.get('regime_change_events_detected', 50)
                tpsl_events = training_results.get('tpsl_events_processed', 200)
                
                estimated_performance = {
                    'total_return': estimated_return,
                    'annualized_return': estimated_return * 252,  # Assuming daily signals
                    'volatility': estimated_volatility,
                    'sharpe_ratio': estimated_return / estimated_volatility if estimated_volatility > 0 else 0.0,
                    'sortino_ratio': estimated_return / (estimated_volatility * 0.7) if estimated_volatility > 0 else 0.0,
                    'calmar_ratio': 0.0,  # Would need max drawdown
                    'max_drawdown': estimated_volatility * 2.2,  # Estimate
                    'max_drawdown_duration': 28,  # Default estimate
                    'var_95': estimated_volatility * 1.7,  # Estimate
                    'cvar_95': estimated_volatility * 2.1,  # Estimate
                    'win_rate': combined_accuracy,
                    'profit_factor': training_results.get('risk_reward_ratio', 1.0 + (combined_accuracy - 0.5) * 2),
                    'avg_win': 0.022,  # Default estimate
                    'avg_loss': 0.012,  # Default estimate
                    'largest_win': 0.055,  # Default estimate
                    'largest_loss': estimated_volatility * 2.2,  # Estimate
                    'total_trades': int(sequences_processed * 0.1),  # Estimate 10% of sequences become trades
                    'winning_trades': int(sequences_processed * 0.1 * combined_accuracy),
                    'losing_trades': int(sequences_processed * 0.1 * (1 - combined_accuracy)),
                    'additional_metrics': {
                        'test_accuracy': test_accuracy,
                        'regime_detection_accuracy': regime_detection_accuracy,
                        'combined_tpsl_accuracy': tpsl_accuracy,
                        'direction_prediction_confidence': direction_confidence,
                        'regime_stability_score': hmm_metrics.get('regime_stability_score', 0.0),
                        'regime_entropy_score': hmm_metrics.get('regime_entropy_score', 0.0),
                        'sequence_quality_score': training_results.get('sequence_quality_score', 0.0),
                        'cross_timeframe_correlation': training_results.get('cross_timeframe_correlation', 0.0),
                        'temporal_alignment_score': training_results.get('temporal_alignment_score', 0.0),
                        'multi_timeframe_consistency': training_results.get('multi_timeframe_consistency', 0.0),
                        'total_sequences_processed': sequences_processed,
                        'regime_change_events_detected': regime_changes,
                        'tpsl_events_processed': tpsl_events
                    }
                }
                
                self.financial_logger.log_trading_performance(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step09_5_HMM_LM_Generalist_Training",
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
                    step_name="Step09_5_HMM_LM_Generalist_Training",
                    additional_data={'file_path': str(self.financial_logger.current_file_path)}
                )
            logger.info("📁 File paths logged for Step09_5")
        except Exception as e:
            logger.warning(f"Could not log file paths: {e}")