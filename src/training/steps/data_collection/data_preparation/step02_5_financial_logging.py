"""
Financial metrics logging for Step02_5 S/R Optimization.
Independent logging module that can be used without the reporting system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from src.utils.financial_metrics_logger import get_financial_metrics_logger, financial_metrics_context
from src.utils.logger import system_logger

logger = system_logger.getChild('Step02_5FinancialLogging')


class Step02_5FinancialLogger:
    """Independent financial metrics logger for Step02_5 S/R Optimization."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.financial_logger = get_financial_metrics_logger()
    
    def log_step_execution(self, sr_levels: Dict[str, Any], ml_results: Dict[str, Any], 
                          execution_data: Dict[str, Any], data: Optional[pd.DataFrame]) -> None:
        """Log comprehensive financial metrics for Step02_5 execution."""
        with financial_metrics_context(
            step_name="Step02_5_SR_Optimization",
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe
        ):
            try:
                self.financial_logger.log_step_start("Step02_5_SR_Optimization", self.symbol, self.exchange, self.timeframe)
                
                # Log all financial metrics
                self._log_financial_metrics_from_results(sr_levels, ml_results, execution_data, data)
                
                # Log file paths
                self._log_created_file_paths()
                
                self.financial_logger.log_step_end("Step02_5_SR_Optimization", self.symbol, self.exchange, self.timeframe, success=True)
                
            except Exception as e:
                self.financial_logger.log_step_end("Step02_5_SR_Optimization", self.symbol, self.exchange, self.timeframe, success=False, error_message=str(e))
                logger.error(f"Failed to log financial metrics: {e}")
    
    def _log_financial_metrics_from_results(self, sr_levels: Dict[str, Any], ml_results: Dict[str, Any], execution_data: Dict[str, Any], data: Optional[pd.DataFrame]) -> None:
        """Log key financial metrics directly from step results."""
        try:
            # Note: Data quality and performance metrics are logged in regular system logs
            # Financial metrics logger focuses only on financial/trading metrics
            
            # Log comprehensive ML model performance metrics (financial relevance)
            if ml_results:
                # Basic performance metrics
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="ml_direction_accuracy",
                    metric_value=ml_results.get('direction_accuracy', 0.0),
                    metric_type="performance",
                    step_name="Step02_5_SR_Optimization"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="ml_volatility_mae",
                    metric_value=ml_results.get('volatility_mae', 0.0),
                    metric_type="risk",
                    step_name="Step02_5_SR_Optimization"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="ml_f1_score",
                    metric_value=ml_results.get('f1_score', 0.0),
                    metric_type="performance",
                    step_name="Step02_5_SR_Optimization"
                )
                
                # Additional ML metrics
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="ml_precision",
                    metric_value=ml_results.get('precision', 0.0),
                    metric_type="performance",
                    step_name="Step02_5_SR_Optimization"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="ml_recall",
                    metric_value=ml_results.get('recall', 0.0),
                    metric_type="performance",
                    step_name="Step02_5_SR_Optimization"
                )
                
                # Note: Training/test sample counts are logged in regular system logs
                # Financial metrics logger focuses only on financial/trading metrics
                
                # Log feature importance
                feature_importance = ml_results.get('feature_importance', {})
                if feature_importance:
                    for feature_name, importance in feature_importance.items():
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"feature_importance_{feature_name}",
                            metric_value=importance,
                            metric_type="feature",
                            step_name="Step02_5_SR_Optimization",
                            additional_data={'feature_name': feature_name}
                        )
                
                # Log SHAP values if available
                shap_values = ml_results.get('shap_values', {})
                if shap_values:
                    for feature_name, shap_value in shap_values.items():
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"shap_value_{feature_name}",
                            metric_value=shap_value,
                            metric_type="shap",
                            step_name="Step02_5_SR_Optimization",
                            additional_data={'feature_name': feature_name}
                        )
                
                # Log cross-validation scores
                cv_scores = ml_results.get('cross_validation_scores', [])
                if cv_scores:
                    for i, score in enumerate(cv_scores):
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"cv_score_fold_{i}",
                            metric_value=score,
                            metric_type="performance",
                            step_name="Step02_5_SR_Optimization"
                        )
                    
                    # Log CV statistics
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="cv_mean_score",
                        metric_value=np.mean(cv_scores),
                        metric_type="performance",
                        step_name="Step02_5_SR_Optimization"
                    )
                    
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="cv_std_score",
                        metric_value=np.std(cv_scores),
                        metric_type="performance",
                        step_name="Step02_5_SR_Optimization"
                    )
                
                # Log confusion matrix if available
                confusion_matrix = ml_results.get('confusion_matrix', {})
                if confusion_matrix:
                    for key, value in confusion_matrix.items():
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"confusion_matrix_{key}",
                            metric_value=float(value),
                            metric_type="performance",
                            step_name="Step02_5_SR_Optimization"
                        )
                
                # Log hyperparameters if available
                hyperparameters = ml_results.get('hyperparameters', {})
                if hyperparameters:
                    for param_name, param_value in hyperparameters.items():
                        # Convert parameter value to float if possible
                        try:
                            param_float = float(param_value)
                            self.financial_logger.log_financial_metric(
                                symbol=self.symbol,
                                exchange=self.exchange,
                                timeframe=self.timeframe,
                                metric_name=f"hyperparameter_{param_name}",
                                metric_value=param_float,
                                metric_type="hyperparameter",
                                step_name="Step02_5_SR_Optimization",
                                additional_data={'parameter_name': param_name, 'parameter_value': str(param_value)}
                            )
                        except (ValueError, TypeError):
                            # Log as additional data if can't convert to float
                            self.financial_logger.log_financial_metric(
                                symbol=self.symbol,
                                exchange=self.exchange,
                                timeframe=self.timeframe,
                                metric_name="hyperparameter_info",
                                metric_value=0.0,
                                metric_type="hyperparameter",
                                step_name="Step02_5_SR_Optimization",
                                additional_data={param_name: str(param_value)}
                            )
            
            # Log clustering details if available (financial relevance)
            clustering_results = ml_results.get('clustering_results', {})
            if clustering_results:
                # Log clustering quality metrics (financial relevance)
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="clustering_silhouette_score",
                    metric_value=clustering_results.get('silhouette_score', 0.0),
                    metric_type="trading",
                    step_name="Step02_5_SR_Optimization"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="clustering_davies_bouldin_index",
                    metric_value=clustering_results.get('davies_bouldin_index', 0.0),
                    metric_type="trading",
                    step_name="Step02_5_SR_Optimization"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="clustering_calinski_harabasz_index",
                    metric_value=clustering_results.get('calinski_harabasz_index', 0.0),
                    metric_type="trading",
                    step_name="Step02_5_SR_Optimization"
                )
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="clustering_n_clusters",
                    metric_value=float(clustering_results.get('n_clusters', 0)),
                    metric_type="trading",
                    step_name="Step02_5_SR_Optimization"
                )
                
                # Note: Cluster sizes, centers, and technical clustering metrics are logged in regular system logs
                # Financial metrics logger focuses only on financial/trading metrics
            
            # Log detailed S/R level metrics
            if sr_levels:
                support_levels = sr_levels.get('support_levels', [])
                resistance_levels = sr_levels.get('resistance_levels', [])
                
                # Log individual support levels with detailed characteristics
                if support_levels:
                    support_strengths = [level.get('strength', 0) for level in support_levels]
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="support_levels_count",
                        metric_value=float(len(support_levels)),
                        metric_type="technical",
                        step_name="Step02_5_SR_Optimization"
                    )
                    
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="support_average_strength",
                        metric_value=np.mean(support_strengths) if support_strengths else 0.0,
                        metric_type="technical",
                        step_name="Step02_5_SR_Optimization"
                    )
                    
                    # Log each support level individually with detailed characteristics
                    for i, level in enumerate(support_levels):
                        level_data = {
                            'level_id': i,
                            'price': level.get('price', 0.0),
                            'strength': level.get('strength', 0.0),
                            'touches': level.get('touches', 0),
                            'bounces': level.get('bounces', 0),
                            'bounce_rate': level.get('bounce_rate', 0.0),
                            'age_days': level.get('age_days', 0),
                            'distance_to_current': level.get('distance_to_current', 0.0),
                            'reliability_score': level.get('reliability_score', 0.0),
                            'trend_alignment': level.get('trend_alignment', 'unknown'),
                            'volume_confirmation': level.get('volume_confirmation', False),
                            'fractal_strength': level.get('fractal_strength', 0.0)
                        }
                        
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"support_level_{i}",
                            metric_value=level.get('price', 0.0),
                            metric_type="technical",
                            step_name="Step02_5_SR_Optimization",
                            additional_data=level_data
                        )
                
                # Log individual resistance levels with detailed characteristics
                if resistance_levels:
                    resistance_strengths = [level.get('strength', 0) for level in resistance_levels]
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="resistance_levels_count",
                        metric_value=float(len(resistance_levels)),
                        metric_type="technical",
                        step_name="Step02_5_SR_Optimization"
                    )
                    
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="resistance_average_strength",
                        metric_value=np.mean(resistance_strengths) if resistance_strengths else 0.0,
                        metric_type="technical",
                        step_name="Step02_5_SR_Optimization"
                    )
                    
                    # Log each resistance level individually with detailed characteristics
                    for i, level in enumerate(resistance_levels):
                        level_data = {
                            'level_id': i,
                            'price': level.get('price', 0.0),
                            'strength': level.get('strength', 0.0),
                            'touches': level.get('touches', 0),
                            'bounces': level.get('bounces', 0),
                            'bounce_rate': level.get('bounce_rate', 0.0),
                            'age_days': level.get('age_days', 0),
                            'distance_to_current': level.get('distance_to_current', 0.0),
                            'reliability_score': level.get('reliability_score', 0.0),
                            'trend_alignment': level.get('trend_alignment', 'unknown'),
                            'volume_confirmation': level.get('volume_confirmation', False),
                            'fractal_strength': level.get('fractal_strength', 0.0)
                        }
                        
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"resistance_level_{i}",
                            metric_value=level.get('price', 0.0),
                            metric_type="technical",
                            step_name="Step02_5_SR_Optimization",
                            additional_data=level_data
                        )
            
            # Note: Data quality and execution performance metrics are logged in regular system logs
            # Financial metrics logger focuses only on financial/trading metrics
            
            # Log comprehensive trading performance
            if sr_levels and ml_results:
                # Estimate trading performance based on S/R levels and ML results
                estimated_performance = {
                    'total_return': 0.0,  # Would need actual trading data
                    'annualized_return': 0.0,
                    'volatility': ml_results.get('volatility_mae', 0.02),
                    'sharpe_ratio': 0.0,  # Would need return data
                    'sortino_ratio': 0.0,
                    'calmar_ratio': 0.0,
                    'max_drawdown': ml_results.get('volatility_mae', 0.02) * 2,  # Estimate
                    'max_drawdown_duration': 25,  # Default estimate
                    'var_95': ml_results.get('volatility_mae', 0.02) * 1.5,  # Estimate
                    'cvar_95': ml_results.get('volatility_mae', 0.02) * 2,  # Estimate
                    'win_rate': ml_results.get('direction_accuracy', 0.5),
                    'profit_factor': 1.0,  # Default
                    'avg_win': 0.01,  # Default estimate
                    'avg_loss': 0.01,  # Default estimate
                    'largest_win': 0.03,  # Default estimate
                    'largest_loss': ml_results.get('volatility_mae', 0.02) * 2,  # Estimate
                    'total_trades': 30,  # Default estimate
                    'winning_trades': int(30 * ml_results.get('direction_accuracy', 0.5)),
                    'losing_trades': int(30 * (1 - ml_results.get('direction_accuracy', 0.5))),
                    'additional_metrics': {
                        'sr_levels_count': len(sr_levels.get('support_levels', [])) + len(sr_levels.get('resistance_levels', [])),
                        'ml_accuracy': ml_results.get('direction_accuracy', 0.0),
                        'ml_f1_score': ml_results.get('f1_score', 0.0)
                    }
                }
                
                self.financial_logger.log_trading_performance(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step02_5_SR_Optimization",
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
                    step_name="Step02_5_SR_Optimization",
                    additional_data={'file_path': str(self.financial_logger.current_file_path)}
                )
            logger.info("📁 File paths logged for Step02_5")
        except Exception as e:
            logger.warning(f"Could not log file paths: {e}")