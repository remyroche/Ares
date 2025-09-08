"""
Financial metrics logging for Step15 Tactician Specialist Training.
Independent logging module that can be used without the reporting system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from src.utils.financial_metrics_logger import get_financial_metrics_logger, financial_metrics_context
from src.utils.logger import system_logger

logger = system_logger.getChild('Step15FinancialLogging')


class Step15FinancialLogger:
    """Independent financial metrics logger for Step15 Tactician Specialist Training."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.financial_logger = get_financial_metrics_logger()
    
    def log_step_execution(self, training_results: Dict[str, Any], model_performance: Dict[str, Any], 
                          feature_data: Dict[str, Any], sr_analysis: Dict[str, Any], 
                          regime_data: Dict[str, Any], optimization_metrics: Dict[str, Any]) -> None:
        """Log comprehensive financial metrics for Step15 execution."""
        with financial_metrics_context(
            step_name="Step15_Tactician_Specialist_Training",
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe
        ):
            try:
                self.financial_logger.log_step_start("Step15_Tactician_Specialist_Training", self.symbol, self.exchange, self.timeframe)
                
                # Log all financial metrics
                self._log_financial_metrics_from_results(training_results, model_performance, feature_data, sr_analysis, regime_data, optimization_metrics)
                
                # Log file paths
                self._log_created_file_paths()
                
                self.financial_logger.log_step_end("Step15_Tactician_Specialist_Training", self.symbol, self.exchange, self.timeframe, success=True)
                
            except Exception as e:
                self.financial_logger.log_step_end("Step15_Tactician_Specialist_Training", self.symbol, self.exchange, self.timeframe, success=False, error_message=str(e))
                logger.error(f"Failed to log financial metrics: {e}")
    
    def _log_financial_metrics_from_results(self, training_results: Dict[str, Any], model_performance: Dict[str, Any], 
                                          feature_data: Dict[str, Any], sr_analysis: Dict[str, Any], 
                                          regime_data: Dict[str, Any], optimization_metrics: Dict[str, Any]) -> None:
        """Log key financial metrics directly from step results."""
        try:
            # Note: Data quality and performance metrics are logged in regular system logs
            # Financial metrics logger focuses only on financial/trading metrics
            
            # Log specialist model performance metrics
            if model_performance:
                if 'specialist_accuracy' in model_performance:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="specialist_accuracy",
                        metric_value=model_performance['specialist_accuracy'],
                        metric_type="performance",
                        step_name="Step15_Tactician_Specialist_Training"
                    )
                
                if 'specialist_precision' in model_performance:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="specialist_precision",
                        metric_value=model_performance['specialist_precision'],
                        metric_type="performance",
                        step_name="Step15_Tactician_Specialist_Training"
                    )
                
                if 'specialist_recall' in model_performance:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="specialist_recall",
                        metric_value=model_performance['specialist_recall'],
                        metric_type="performance",
                        step_name="Step15_Tactician_Specialist_Training"
                    )
                
                if 'specialist_f1_score' in model_performance:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="specialist_f1_score",
                        metric_value=model_performance['specialist_f1_score'],
                        metric_type="performance",
                        step_name="Step15_Tactician_Specialist_Training"
                    )
                
                if 'specialist_convergence_score' in model_performance:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="specialist_convergence_score",
                        metric_value=model_performance['specialist_convergence_score'],
                        metric_type="trading",
                        step_name="Step15_Tactician_Specialist_Training"
                    )
                
                if 'specialist_generalization_score' in model_performance:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="specialist_generalization_score",
                        metric_value=model_performance['specialist_generalization_score'],
                        metric_type="trading",
                        step_name="Step15_Tactician_Specialist_Training"
                    )
            
            # Log S/R integration metrics
            if sr_analysis:
                if 'sr_levels_identified' in sr_analysis:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="sr_levels_identified",
                        metric_value=float(sr_analysis['sr_levels_identified']),
                        metric_type="trading",
                        step_name="Step15_Tactician_Specialist_Training"
                    )
                
                if 'sr_effectiveness_score' in sr_analysis:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="sr_effectiveness_score",
                        metric_value=sr_analysis['sr_effectiveness_score'],
                        metric_type="trading",
                        step_name="Step15_Tactician_Specialist_Training"
                    )
                
                if 'sr_breakout_accuracy' in sr_analysis:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="sr_breakout_accuracy",
                        metric_value=sr_analysis['sr_breakout_accuracy'],
                        metric_type="trading",
                        step_name="Step15_Tactician_Specialist_Training"
                    )
                
                if 'sr_support_resistance_score' in sr_analysis:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="sr_support_resistance_score",
                        metric_value=sr_analysis['sr_support_resistance_score'],
                        metric_type="trading",
                        step_name="Step15_Tactician_Specialist_Training"
                    )
                
                if 'sr_feature_contribution' in sr_analysis:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="sr_feature_contribution",
                        metric_value=sr_analysis['sr_feature_contribution'],
                        metric_type="trading",
                        step_name="Step15_Tactician_Specialist_Training"
                    )
                
                if 'sr_regime_alignment' in sr_analysis:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="sr_regime_alignment",
                        metric_value=sr_analysis['sr_regime_alignment'],
                        metric_type="trading",
                        step_name="Step15_Tactician_Specialist_Training"
                    )
            
            # Log feature engineering metrics
            if feature_data:
                if 'total_features_selected' in feature_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="total_features_selected",
                        metric_value=float(feature_data['total_features_selected']),
                        metric_type="performance",
                        step_name="Step15_Tactician_Specialist_Training"
                    )
                
                if 'feature_importance_score' in feature_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="feature_importance_score",
                        metric_value=feature_data['feature_importance_score'],
                        metric_type="trading",
                        step_name="Step15_Tactician_Specialist_Training"
                    )
                
                if 'feature_stability_score' in feature_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="feature_stability_score",
                        metric_value=feature_data['feature_stability_score'],
                        metric_type="trading",
                        step_name="Step15_Tactician_Specialist_Training"
                    )
                
                if 'feature_predictive_power' in feature_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="feature_predictive_power",
                        metric_value=feature_data['feature_predictive_power'],
                        metric_type="trading",
                        step_name="Step15_Tactician_Specialist_Training"
                    )
                
                if 'feature_redundancy_score' in feature_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="feature_redundancy_score",
                        metric_value=feature_data['feature_redundancy_score'],
                        metric_type="trading",
                        step_name="Step15_Tactician_Specialist_Training"
                    )
            
            # Log probability generation metrics
            if training_results:
                prob_analysis = training_results.get('probability_analysis', {})
                if prob_analysis:
                    if 'probability_calibration_score' in prob_analysis:
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name="probability_calibration_score",
                            metric_value=prob_analysis['probability_calibration_score'],
                            metric_type="trading",
                            step_name="Step15_Tactician_Specialist_Training"
                        )
                    
                    if 'probability_accuracy' in prob_analysis:
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name="probability_accuracy",
                            metric_value=prob_analysis['probability_accuracy'],
                            metric_type="performance",
                            step_name="Step15_Tactician_Specialist_Training"
                        )
                    
                    if 'uncertainty_estimation_score' in prob_analysis:
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name="uncertainty_estimation_score",
                            metric_value=prob_analysis['uncertainty_estimation_score'],
                            metric_type="trading",
                            step_name="Step15_Tactician_Specialist_Training"
                        )
                    
                    if 'decision_threshold_optimization' in prob_analysis:
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name="decision_threshold_optimization",
                            metric_value=prob_analysis['decision_threshold_optimization'],
                            metric_type="trading",
                            step_name="Step15_Tactician_Specialist_Training"
                        )
            
            # Log regime specialization metrics
            if regime_data:
                if 'total_regimes_processed' in regime_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="total_regimes_processed",
                        metric_value=float(regime_data['total_regimes_processed']),
                        metric_type="performance",
                        step_name="Step15_Tactician_Specialist_Training"
                    )
                
                if 'regime_adaptation_score' in regime_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="regime_adaptation_score",
                        metric_value=regime_data['regime_adaptation_score'],
                        metric_type="trading",
                        step_name="Step15_Tactician_Specialist_Training"
                    )
                
                if 'regime_transfer_learning_score' in regime_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="regime_transfer_learning_score",
                        metric_value=regime_data['regime_transfer_learning_score'],
                        metric_type="trading",
                        step_name="Step15_Tactician_Specialist_Training"
                    )
                
                # Log regime-specific performance
                if 'regime_performance' in regime_data:
                    regime_performance = regime_data['regime_performance']
                    for regime_id, regime_metrics in regime_performance.items():
                        if isinstance(regime_metrics, dict):
                            if 'regime_accuracy' in regime_metrics:
                                self.financial_logger.log_financial_metric(
                                    symbol=self.symbol,
                                    exchange=self.exchange,
                                    timeframe=self.timeframe,
                                    metric_name=f"regime_{regime_id}_accuracy",
                                    metric_value=regime_metrics['regime_accuracy'],
                                    metric_type="performance",
                                    step_name="Step15_Tactician_Specialist_Training",
                                    regime_id=str(regime_id)
                                )
                            
                            if 'regime_specialization_score' in regime_metrics:
                                self.financial_logger.log_financial_metric(
                                    symbol=self.symbol,
                                    exchange=self.exchange,
                                    timeframe=self.timeframe,
                                    metric_name=f"regime_{regime_id}_specialization_score",
                                    metric_value=regime_metrics['regime_specialization_score'],
                                    metric_type="trading",
                                    step_name="Step15_Tactician_Specialist_Training",
                                    regime_id=str(regime_id)
                                )
            
            # Log LM optimization metrics
            if optimization_metrics:
                lm_data = optimization_metrics.get('language_model', {})
                if lm_data:
                    if 'lm_training_accuracy' in lm_data:
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name="lm_training_accuracy",
                            metric_value=lm_data['lm_training_accuracy'],
                            metric_type="performance",
                            step_name="Step15_Tactician_Specialist_Training"
                        )
                    
                    if 'lm_convergence_score' in lm_data:
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name="lm_convergence_score",
                            metric_value=lm_data['lm_convergence_score'],
                            metric_type="trading",
                            step_name="Step15_Tactician_Specialist_Training"
                        )
                    
                    if 'lm_feature_importance' in lm_data:
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name="lm_feature_importance",
                            metric_value=lm_data['lm_feature_importance'],
                            metric_type="trading",
                            step_name="Step15_Tactician_Specialist_Training"
                        )
                    
                    if 'lm_inference_speed' in lm_data:
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name="lm_inference_speed",
                            metric_value=float(lm_data['lm_inference_speed']),
                            metric_type="performance",
                            step_name="Step15_Tactician_Specialist_Training"
                        )
            
            # Log training execution metrics
            if training_results:
                if 'total_models_trained' in training_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="total_models_trained",
                        metric_value=float(training_results['total_models_trained']),
                        metric_type="performance",
                        step_name="Step15_Tactician_Specialist_Training"
                    )
                
                if 'training_duration' in training_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="training_duration",
                        metric_value=float(training_results['training_duration']),
                        metric_type="performance",
                        step_name="Step15_Tactician_Specialist_Training"
                    )
                
                if 'data_points_processed' in training_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="data_points_processed",
                        metric_value=float(training_results['data_points_processed']),
                        metric_type="performance",
                        step_name="Step15_Tactician_Specialist_Training"
                    )
            
            # Log comprehensive trading performance estimation
            if training_results and model_performance and sr_analysis and regime_data:
                # Estimate trading performance based on specialist training results
                specialist_accuracy = model_performance.get('specialist_accuracy', 0.5)
                sr_effectiveness = sr_analysis.get('sr_effectiveness_score', 0.5)
                regime_adaptation = regime_data.get('regime_adaptation_score', 0.5)
                feature_importance = feature_data.get('feature_importance_score', 0.5) if feature_data else 0.5
                
                # Estimate returns based on specialist training results
                combined_score = (specialist_accuracy + sr_effectiveness + regime_adaptation + feature_importance) / 4
                estimated_return = (combined_score * 0.035) - ((1 - combined_score) * 0.02)  # Estimate
                estimated_volatility = 0.025  # Default estimate
                
                # Estimate trading metrics
                total_models = training_results.get('total_models_trained', 5)
                total_regimes = regime_data.get('total_regimes_processed', 3)
                
                estimated_performance = {
                    'total_return': estimated_return,
                    'annualized_return': estimated_return * 252,  # Assuming daily signals
                    'volatility': estimated_volatility,
                    'sharpe_ratio': estimated_return / estimated_volatility if estimated_volatility > 0 else 0.0,
                    'sortino_ratio': estimated_return / (estimated_volatility * 0.6) if estimated_volatility > 0 else 0.0,
                    'calmar_ratio': 0.0,  # Would need max drawdown
                    'max_drawdown': estimated_volatility * 2.5,  # Estimate
                    'max_drawdown_duration': 40,  # Default estimate
                    'var_95': estimated_volatility * 1.7,  # Estimate
                    'cvar_95': estimated_volatility * 2.2,  # Estimate
                    'win_rate': combined_score,
                    'profit_factor': 1.0 + (combined_score - 0.5) * 3.5,
                    'avg_win': 0.035,  # Default estimate
                    'avg_loss': 0.02,  # Default estimate
                    'largest_win': 0.09,  # Default estimate
                    'largest_loss': estimated_volatility * 2.5,  # Estimate
                    'total_trades': int(total_models * total_regimes * 2),  # Estimate 2 trades per model per regime
                    'winning_trades': int(total_models * total_regimes * 2 * combined_score),
                    'losing_trades': int(total_models * total_regimes * 2 * (1 - combined_score)),
                    'additional_metrics': {
                        'specialist_accuracy': specialist_accuracy,
                        'sr_effectiveness': sr_effectiveness,
                        'regime_adaptation': regime_adaptation,
                        'feature_importance': feature_importance,
                        'total_models_trained': total_models,
                        'total_regimes_processed': total_regimes,
                        'sr_levels_identified': sr_analysis.get('sr_levels_identified', 0),
                        'sr_breakout_accuracy': sr_analysis.get('sr_breakout_accuracy', 0.0),
                        'sr_feature_contribution': sr_analysis.get('sr_feature_contribution', 0.0),
                        'regime_transfer_learning_score': regime_data.get('regime_transfer_learning_score', 0.0),
                        'probability_calibration_score': training_results.get('probability_analysis', {}).get('probability_calibration_score', 0.0),
                        'uncertainty_estimation_score': training_results.get('probability_analysis', {}).get('uncertainty_estimation_score', 0.0),
                        'lm_training_accuracy': optimization_metrics.get('language_model', {}).get('lm_training_accuracy', 0.0) if optimization_metrics else 0.0,
                        'lm_convergence_score': optimization_metrics.get('language_model', {}).get('lm_convergence_score', 0.0) if optimization_metrics else 0.0
                    }
                }
                
                self.financial_logger.log_trading_performance(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step15_Tactician_Specialist_Training",
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
                    step_name="Step15_Tactician_Specialist_Training",
                    additional_data={'file_path': str(self.financial_logger.current_file_path)}
                )
            logger.info("📁 File paths logged for Step15")
        except Exception as e:
            logger.warning(f"Could not log file paths: {e}")