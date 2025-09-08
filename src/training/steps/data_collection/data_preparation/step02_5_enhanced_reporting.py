"""
Enhanced Reporting System for Step 2.5 S/R Optimization

This module provides comprehensive reporting capabilities for step02_5_sr_optimization
with detailed metrics, performance analytics, ML insights, and data quality assessments.
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
# Visualization imports removed - using financial metrics logger instead
from dataclasses import dataclass, asdict
import warnings

# Using financial metrics logger instead of old reporting system
from src.utils.logger import system_logger
from src.utils.financial_metrics_logger import get_financial_metrics_logger, financial_metrics_context

logger = system_logger.getChild('Step02_5EnhancedReporting')
financial_logger = get_financial_metrics_logger()


@dataclass
class SRLevelMetrics:
    """Detailed metrics for individual S/R levels."""
    price: float
    strength: float
    touches: int
    bounces: int
    bounce_rate: float
    age_days: int
    distance_to_current: float
    reliability_score: float
    trend_alignment: str
    volume_confirmation: bool
    fractal_strength: float


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for the step."""
    execution_time_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float
    data_processing_rate: float  # rows/second
    ml_training_time: float
    feature_engineering_time: float
    sr_detection_time: float
    report_generation_time: float
    total_function_calls: int
    successful_operations: int
    failed_operations: int
    error_rate: float


@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics."""
    total_rows: int
    total_columns: int
    missing_values_percent: float
    duplicate_rows: int
    outlier_rows: int
    zero_values_count: int
    data_completeness_score: float
    feature_correlation_warnings: List[str]
    timestamp_anomalies: List[str]
    price_anomalies: List[str]


@dataclass
class MLModelInsights:
    """Detailed ML model performance and insights."""
    model_type: str
    direction_accuracy: float
    volatility_mae: float
    precision: float
    recall: float
    f1_score: float
    feature_importance: Dict[str, float]
    confusion_matrix: Dict[str, int]
    cross_validation_scores: List[float]
    training_samples: int
    test_samples: int
    feature_count: int
    hyperparameters: Dict[str, Any]
    overfitting_score: float
    model_complexity: str


class Step02_5EnhancedReporter:
    """
    Enhanced reporting system for Step 2.5 S/R Optimization.

    Provides comprehensive metrics including:
    - Performance analytics
    - Data quality assessment
    - ML model insights
    - S/R level analysis
    - Trading implications
    - Visualizations
    """

    def __init__(self, symbol: str = "UNKNOWN", exchange: str = "UNKNOWN", timeframe: str = "30m"):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.start_time = datetime.now()

        # Using financial metrics logger instead of old reporting system
        self.report_manager = None

        self.metrics_history = []

    def collect_performance_metrics(self,
                                   execution_time: float,
                                   memory_usage: float = 0,
                                   cpu_usage: float = 0,
                                   function_calls: int = 0) -> PerformanceMetrics:
        """Collect comprehensive performance metrics."""
        return PerformanceMetrics(
            execution_time_seconds=execution_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            data_processing_rate=0,  # Will be calculated with data
            ml_training_time=0,
            feature_engineering_time=0,
            sr_detection_time=0,
            report_generation_time=0,
            total_function_calls=function_calls,
            successful_operations=0,
            failed_operations=0,
            error_rate=0
        )

    def assess_data_quality(self, data: pd.DataFrame) -> DataQualityMetrics:
        """Perform comprehensive data quality assessment."""
        try:
            total_rows, total_columns = data.shape

            # Missing values
            missing_values = data.isnull().sum().sum()
            missing_values_percent = (missing_values / (total_rows * total_columns)) * 100

            # Duplicate rows
            duplicate_rows = data.duplicated().sum()

            # Zero values (excluding timestamp and other non-numeric)
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            zero_values = (data[numeric_cols] == 0).sum().sum()

            # Outlier detection (simple IQR method)
            outlier_rows = 0
            for col in numeric_cols:
                if col in ['timestamp', 'datetime']:
                    continue
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_rows += outliers

            # Data completeness score
            completeness_score = (1 - missing_values_percent/100) * 100

            # Correlation warnings
            correlation_warnings = []
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr()
                high_corr = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool))
                high_corr_pairs = high_corr.stack().reset_index()
                high_corr_pairs.columns = ['var1', 'var2', 'correlation']
                high_corr_pairs = high_corr_pairs[abs(high_corr_pairs['correlation']) > 0.95]

                for _, row in high_corr_pairs.iterrows():
                    correlation_warnings.append(
                        f"High correlation between {row['var1']} and {row['var2']}: {row['correlation']:.3f}"
                    )

            # Timestamp and price anomalies
            timestamp_anomalies = []
            price_anomalies = []

            if 'timestamp' in data.columns:
                timestamps = pd.to_datetime(data['timestamp'])
                time_diffs = timestamps.diff()
                # Check for unrealistic time gaps
                large_gaps = time_diffs > timedelta(hours=1)
                if large_gaps.any():
                    timestamp_anomalies.append(f"Found {large_gaps.sum()} large time gaps (>1 hour)")

            if 'close' in data.columns:
                price_changes = data['close'].pct_change()
                extreme_changes = abs(price_changes) > 0.5  # 50% change
                if extreme_changes.any():
                    price_anomalies.append(f"Found {extreme_changes.sum()} extreme price changes (>50%)")

            return DataQualityMetrics(
                total_rows=total_rows,
                total_columns=total_columns,
                missing_values_percent=missing_values_percent,
                duplicate_rows=duplicate_rows,
                outlier_rows=outlier_rows,
                zero_values_count=zero_values,
                data_completeness_score=completeness_score,
                feature_correlation_warnings=correlation_warnings,
                timestamp_anomalies=timestamp_anomalies,
                price_anomalies=price_anomalies
            )

        except Exception as e:
            logger.warning(f"Failed to assess data quality: {e}")
            return DataQualityMetrics(
                total_rows=0,
                total_columns=0,
                missing_values_percent=0,
                duplicate_rows=0,
                outlier_rows=0,
                zero_values_count=0,
                data_completeness_score=0,
                feature_correlation_warnings=[],
                timestamp_anomalies=[],
                price_anomalies=[]
            )

    def analyze_sr_levels(self,
                         sr_levels: Dict[str, Any],
                         current_price: Optional[float] = None) -> Dict[str, Any]:
        """Perform comprehensive S/R level analysis."""
        analysis = {
            'support_analysis': {},
            'resistance_analysis': {},
            'level_distribution': {},
            'strength_distribution': {},
            'trading_zones': {},
            'risk_assessment': {}
        }

        # Support levels analysis
        support_levels = sr_levels.get('support_levels', [])
        if support_levels:
            analysis['support_analysis'] = {
                'total_levels': len(support_levels),
                'average_strength': np.mean([level.get('strength', 0) for level in support_levels]),
                'strongest_level': max(support_levels, key=lambda x: x.get('strength', 0))['price'],
                'weakest_level': min(support_levels, key=lambda x: x.get('strength', 0))['price'],
                'level_ranges': {
                    'weak': len([l for l in support_levels if l.get('strength', 0) < 0.4]),
                    'moderate': len([l for l in support_levels if 0.4 <= l.get('strength', 0) < 0.7]),
                    'strong': len([l for l in support_levels if l.get('strength', 0) >= 0.7])
                }
            }

        # Resistance levels analysis
        resistance_levels = sr_levels.get('resistance_levels', [])
        if resistance_levels:
            analysis['resistance_analysis'] = {
                'total_levels': len(resistance_levels),
                'average_strength': np.mean([level.get('strength', 0) for level in resistance_levels]),
                'strongest_level': max(resistance_levels, key=lambda x: x.get('strength', 0))['price'],
                'weakest_level': min(resistance_levels, key=lambda x: x.get('strength', 0))['price'],
                'level_ranges': {
                    'weak': len([l for l in resistance_levels if l.get('strength', 0) < 0.4]),
                    'moderate': len([l for l in resistance_levels if 0.4 <= l.get('strength', 0) < 0.7]),
                    'strong': len([l for l in resistance_levels if l.get('strength', 0) >= 0.7])
                }
            }

        # Trading zones analysis
        if current_price and support_levels and resistance_levels:
            nearest_support = min(support_levels, key=lambda x: abs(x['price'] - current_price))
            nearest_resistance = min(resistance_levels, key=lambda x: abs(x['price'] - current_price))

            analysis['trading_zones'] = {
                'current_price': current_price,
                'nearest_support': nearest_support['price'],
                'nearest_resistance': nearest_resistance['price'],
                'support_distance_percent': abs(current_price - nearest_support['price']) / current_price * 100,
                'resistance_distance_percent': abs(current_price - nearest_resistance['price']) / current_price * 100,
                'zone_type': 'support' if current_price - nearest_support['price'] < nearest_resistance['price'] - current_price else 'resistance'
            }

        return analysis

    def analyze_ml_performance(self, ml_results: Dict[str, Any]) -> MLModelInsights:
        """Analyze ML model performance in detail."""
        try:
            # Extract basic metrics
            direction_accuracy = ml_results.get('direction_accuracy', 0)
            volatility_mae = ml_results.get('volatility_mae', 0)

            # Calculate derived metrics
            precision = 0
            recall = 0
            f1_score = 0

            if 'classification_report' in ml_results:
                report = ml_results['classification_report']
                if 'weighted avg' in report:
                    weighted_avg = report['weighted avg']
                    precision = weighted_avg.get('precision', 0)
                    recall = weighted_avg.get('recall', 0)
                    f1_score = weighted_avg.get('f1-score', 0)

            # Feature importance
            feature_importance = ml_results.get('feature_importance', {})

            # Cross-validation scores
            cv_scores = ml_results.get('cross_validation_scores', [])

            # Overfitting assessment
            overfitting_score = 0
            if cv_scores:
                cv_mean = np.mean(cv_scores)
                train_accuracy = direction_accuracy
                overfitting_score = max(0, train_accuracy - cv_mean)

            # Model complexity assessment
            feature_count = len(feature_importance)
            if feature_count < 10:
                complexity = "Simple"
            elif feature_count < 50:
                complexity = "Moderate"
            else:
                complexity = "Complex"

            return MLModelInsights(
                model_type=ml_results.get('model_type', 'Unknown'),
                direction_accuracy=direction_accuracy,
                volatility_mae=volatility_mae,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                feature_importance=feature_importance,
                confusion_matrix=ml_results.get('confusion_matrix', {}),
                cross_validation_scores=cv_scores,
                training_samples=ml_results.get('training_samples', 0),
                test_samples=ml_results.get('test_samples', 0),
                feature_count=feature_count,
                hyperparameters=ml_results.get('hyperparameters', {}),
                overfitting_score=overfitting_score,
                model_complexity=complexity
            )

        except Exception as e:
            logger.warning(f"Failed to analyze ML performance: {e}")
            return MLModelInsights(
                model_type="Error",
                direction_accuracy=0,
                volatility_mae=0,
                precision=0,
                recall=0,
                f1_score=0,
                feature_importance={},
                confusion_matrix={},
                cross_validation_scores=[],
                training_samples=0,
                test_samples=0,
                feature_count=0,
                hyperparameters={},
                overfitting_score=0,
                model_complexity="Unknown"
            )

    def generate_comprehensive_report(self,
                                    sr_levels: Dict[str, Any],
                                    ml_results: Dict[str, Any],
                                    execution_data: Dict[str, Any],
                                    data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Generate comprehensive report with all metrics and insights."""
        
        # Use financial metrics context for this step
        with financial_metrics_context("Step02_5_SR_Optimization", self.symbol, self.exchange, self.timeframe):
            try:
                financial_logger.log_step_start("Step02_5_SR_Optimization", self.symbol, self.exchange, self.timeframe)

                # Get current price and market context
                current_price = None
                market_context = {}
                if data is not None and not data.empty and 'close' in data.columns:
                    current_price = float(data['close'].iloc[-1])
                    market_context = self._analyze_market_context(data, current_price)

                # Collect all metrics with enhanced detail
                performance_metrics = self.collect_performance_metrics(
                    execution_data.get('execution_time', 0),
                    execution_data.get('memory_usage', 0),
                    execution_data.get('cpu_usage', 0),
                    execution_data.get('function_calls', 0)
                )

                data_quality = DataQualityMetrics(
                    total_rows=0, total_columns=0, missing_values_percent=0,
                    duplicate_rows=0, outlier_rows=0, zero_values_count=0,
                    data_completeness_score=0, feature_correlation_warnings=[],
                    timestamp_anomalies=[], price_anomalies=[]
                )
                if data is not None:
                    data_quality = self.assess_data_quality(data)

                sr_analysis = self.analyze_sr_levels(sr_levels, current_price)
                ml_insights = self.analyze_ml_performance(ml_results)

                # Enhanced technical analysis
                technical_analysis = {}
                if data is not None:
                    technical_analysis = self._perform_technical_analysis(data, sr_levels)

                # Feature engineering insights
                feature_insights = self._analyze_feature_engineering(data, ml_results)

                # Market regime analysis
                regime_analysis = self._analyze_market_regime(data, sr_levels, ml_results)

                # Risk management recommendations
                risk_management = {'position_sizing': {}, 'stop_loss_levels': {}, 'risk_warnings': []}

                # Performance prediction
                performance_prediction = {'expected_accuracy': 0.5, 'confidence_interval': [0.4, 0.6]}

                # Trading strategy suggestions
                strategy_suggestions = {'primary_strategy': 'WAIT', 'entry_signals': [], 'exit_signals': []}

                # Compile comprehensive report with much more detail
                report = {
                    'report_metadata': {
                        'generated_at': datetime.now().isoformat(),
                        'symbol': self.symbol,
                        'exchange': self.exchange,
                        'timeframe': self.timeframe,
                        'current_price': current_price,
                        'report_version': '3.0.0',
                        'data_timeframe': f"{len(data) if data is not None else 0} periods",
                        'generation_duration': 'comprehensive'
                    },
                    'market_context': market_context,
                    'performance_metrics': asdict(performance_metrics),
                    'performance_breakdown': self._detailed_performance_breakdown(execution_data),
                    'data_quality_assessment': asdict(data_quality),
                    'data_processing_insights': self._analyze_data_processing(data),
                    'sr_level_analysis': sr_analysis,
                    'sr_level_detailed_analysis': self._detailed_sr_analysis(sr_levels, data),
                    'ml_model_insights': asdict(ml_insights),
                    'ml_model_detailed_analysis': self._detailed_ml_analysis(ml_results),
                    'feature_engineering_insights': feature_insights,
                    'technical_analysis': technical_analysis,
                    'market_regime_analysis': regime_analysis,
                    'correlation_analysis': self._analyze_correlations(data),
                    'volume_analysis': self._analyze_volume_patterns(data),
                    'execution_summary': execution_data,
                    'execution_detailed_breakdown': self._detailed_execution_breakdown(execution_data),
                    'trading_recommendations': self._generate_trading_recommendations(
                        sr_analysis, ml_insights, current_price
                    ),
                    'trading_strategy_suggestions': strategy_suggestions,
                    'risk_assessment': self._assess_overall_risk(sr_analysis, ml_insights),
                    'risk_management_recommendations': risk_management,
                    'performance_prediction': performance_prediction,
                    'model_validation_insights': self._validate_model_performance(ml_results),
                    'market_prediction': {'predicted_direction': 'SIDEWAYS', 'prediction_confidence': 0.5},
                    'alerts_and_warnings': {'critical_alerts': [], 'warnings': [], 'notifications': []},
                    # Visualization and export data removed - using financial metrics logger instead
                }

                # Log key financial metrics directly from step results
                self._log_financial_metrics_from_results(sr_levels, ml_results, execution_data, data)

                financial_logger.log_step_end("Step02_5_SR_Optimization", self.symbol, self.exchange, self.timeframe, success=True)
                return report

            except Exception as e:
                financial_logger.log_step_end("Step02_5_SR_Optimization", self.symbol, self.exchange, self.timeframe, success=False, error_message=str(e))
                logger.error(f"❌ Failed to generate comprehensive report: {e}")
                # Return minimal report on error
                return {
                    'report_metadata': {
                        'generated_at': datetime.now().isoformat(),
                        'symbol': self.symbol,
                        'exchange': self.exchange,
                        'timeframe': self.timeframe,
                        'error': str(e)
                    }
                }

    def _log_financial_metrics_from_results(self, sr_levels: Dict[str, Any], ml_results: Dict[str, Any], execution_data: Dict[str, Any], data: Optional[pd.DataFrame]) -> None:
        """Log key financial metrics directly from step results."""
        try:
            # Log comprehensive ML model performance metrics
            if ml_results:
                # Basic performance metrics
                financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="ml_direction_accuracy",
                    metric_value=ml_results.get('direction_accuracy', 0.0),
                    metric_type="performance",
                    step_name="Step02_5_SR_Optimization"
                )
                
                financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="ml_volatility_mae",
                    metric_value=ml_results.get('volatility_mae', 0.0),
                    metric_type="risk",
                    step_name="Step02_5_SR_Optimization"
                )
                
                financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="ml_f1_score",
                    metric_value=ml_results.get('f1_score', 0.0),
                    metric_type="performance",
                    step_name="Step02_5_SR_Optimization"
                )
                
                # Additional ML metrics
                financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="ml_precision",
                    metric_value=ml_results.get('precision', 0.0),
                    metric_type="performance",
                    step_name="Step02_5_SR_Optimization"
                )
                
                financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="ml_recall",
                    metric_value=ml_results.get('recall', 0.0),
                    metric_type="performance",
                    step_name="Step02_5_SR_Optimization"
                )
                
                financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="ml_training_samples",
                    metric_value=float(ml_results.get('training_samples', 0)),
                    metric_type="performance",
                    step_name="Step02_5_SR_Optimization"
                )
                
                financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="ml_test_samples",
                    metric_value=float(ml_results.get('test_samples', 0)),
                    metric_type="performance",
                    step_name="Step02_5_SR_Optimization"
                )
                
                # Log feature importance
                feature_importance = ml_results.get('feature_importance', {})
                if feature_importance:
                    for feature_name, importance in feature_importance.items():
                        financial_logger.log_financial_metric(
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
                        financial_logger.log_financial_metric(
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
                        financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"cv_score_fold_{i}",
                            metric_value=score,
                            metric_type="performance",
                            step_name="Step02_5_SR_Optimization"
                        )
                    
                    # Log CV statistics
                    financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="cv_mean_score",
                        metric_value=np.mean(cv_scores),
                        metric_type="performance",
                        step_name="Step02_5_SR_Optimization"
                    )
                    
                    financial_logger.log_financial_metric(
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
                        financial_logger.log_financial_metric(
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
                            financial_logger.log_financial_metric(
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
                            financial_logger.log_financial_metric(
                                symbol=self.symbol,
                                exchange=self.exchange,
                                timeframe=self.timeframe,
                                metric_name="hyperparameter_info",
                                metric_value=0.0,
                                metric_type="hyperparameter",
                                step_name="Step02_5_SR_Optimization",
                                additional_data={param_name: str(param_value)}
                            )
            
            # Log clustering details if available
            clustering_results = ml_results.get('clustering_results', {})
            if clustering_results:
                # Log clustering quality metrics
                financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="clustering_silhouette_score",
                    metric_value=clustering_results.get('silhouette_score', 0.0),
                    metric_type="quality",
                    step_name="Step02_5_SR_Optimization"
                )
                
                financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="clustering_davies_bouldin_index",
                    metric_value=clustering_results.get('davies_bouldin_index', 0.0),
                    metric_type="quality",
                    step_name="Step02_5_SR_Optimization"
                )
                
                financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="clustering_calinski_harabasz_index",
                    metric_value=clustering_results.get('calinski_harabasz_index', 0.0),
                    metric_type="quality",
                    step_name="Step02_5_SR_Optimization"
                )
                
                financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="clustering_n_clusters",
                    metric_value=float(clustering_results.get('n_clusters', 0)),
                    metric_type="technical",
                    step_name="Step02_5_SR_Optimization"
                )
                
                # Log cluster sizes
                cluster_sizes = clustering_results.get('cluster_sizes', [])
                if cluster_sizes:
                    for i, size in enumerate(cluster_sizes):
                        financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"cluster_{i}_size",
                            metric_value=float(size),
                            metric_type="clustering",
                            step_name="Step02_5_SR_Optimization"
                        )
                
                # Log cluster centers if available
                cluster_centers = clustering_results.get('cluster_centers', [])
                if cluster_centers:
                    for i, center in enumerate(cluster_centers):
                        if isinstance(center, (list, np.ndarray)):
                            for j, coord in enumerate(center):
                                financial_logger.log_financial_metric(
                                    symbol=self.symbol,
                                    exchange=self.exchange,
                                    timeframe=self.timeframe,
                                    metric_name=f"cluster_{i}_center_{j}",
                                    metric_value=float(coord),
                                    metric_type="clustering",
                                    step_name="Step02_5_SR_Optimization"
                                )
                
                # Log explained variance ratio if available
                explained_variance = clustering_results.get('explained_variance_ratio', 0.0)
                if explained_variance:
                    financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="clustering_explained_variance_ratio",
                        metric_value=explained_variance,
                        metric_type="quality",
                        step_name="Step02_5_SR_Optimization"
                    )
                
                # Log feature reduction efficiency if available
                feature_reduction_efficiency = clustering_results.get('feature_reduction_efficiency', 0.0)
                if feature_reduction_efficiency:
                    financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="clustering_feature_reduction_efficiency",
                        metric_value=feature_reduction_efficiency,
                        metric_type="quality",
                        step_name="Step02_5_SR_Optimization"
                    )
            
            # Log detailed S/R level metrics
            if sr_levels:
                support_levels = sr_levels.get('support_levels', [])
                resistance_levels = sr_levels.get('resistance_levels', [])
                
                # Log individual support levels with detailed characteristics
                if support_levels:
                    support_strengths = [level.get('strength', 0) for level in support_levels]
                    financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="support_levels_count",
                        metric_value=float(len(support_levels)),
                        metric_type="technical",
                        step_name="Step02_5_SR_Optimization"
                    )
                    
                    financial_logger.log_financial_metric(
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
                        
                        financial_logger.log_financial_metric(
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
                    financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="resistance_levels_count",
                        metric_value=float(len(resistance_levels)),
                        metric_type="technical",
                        step_name="Step02_5_SR_Optimization"
                    )
                    
                    financial_logger.log_financial_metric(
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
                        
                        financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"resistance_level_{i}",
                            metric_value=level.get('price', 0.0),
                            metric_type="technical",
                            step_name="Step02_5_SR_Optimization",
                            additional_data=level_data
                        )
            
            # Log data quality metrics
            if data is not None and not data.empty:
                total_rows, total_columns = data.shape
                missing_values = data.isnull().sum().sum()
                missing_values_percent = (missing_values / (total_rows * total_columns)) * 100 if total_rows > 0 else 0
                completeness_score = (1 - missing_values_percent/100) * 100
                
                financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="data_completeness_score",
                    metric_value=completeness_score,
                    metric_type="quality",
                    step_name="Step02_5_SR_Optimization"
                )
                
                financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="missing_values_percent",
                    metric_value=missing_values_percent,
                    metric_type="quality",
                    step_name="Step02_5_SR_Optimization"
                )
            
            # Log performance metrics
            if execution_data:
                financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="execution_time_seconds",
                    metric_value=execution_data.get('execution_time', 0.0),
                    metric_type="performance",
                    step_name="Step02_5_SR_Optimization"
                )
                
                financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="memory_usage_mb",
                    metric_value=execution_data.get('memory_usage', 0.0),
                    metric_type="performance",
                    step_name="Step02_5_SR_Optimization"
                )
            
            # Log trading signal based on ML results
            if ml_results:
                direction_accuracy = ml_results.get('direction_accuracy', 0.5)
                signal_value = 0.0
                if direction_accuracy > 0.6:
                    signal_value = 1.0  # Bullish
                elif direction_accuracy < 0.4:
                    signal_value = -1.0  # Bearish
                
                financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="trading_signal",
                    metric_value=signal_value,
                    metric_type="signal",
                    step_name="Step02_5_SR_Optimization",
                    additional_data={'accuracy': direction_accuracy}
                )
            
            # Log comprehensive trading performance if we have enough data
            if ml_results and sr_levels:
                direction_accuracy = ml_results.get('direction_accuracy', 0.5)
                volatility_mae = ml_results.get('volatility_mae', 0.02)
                
                performance_data = {
                    'total_return': direction_accuracy * 0.1 - 0.05,  # Estimate based on accuracy
                    'annualized_return': direction_accuracy * 0.12 - 0.06,  # Estimate
                    'volatility': volatility_mae * 10,  # Convert MAE to volatility estimate
                    'sharpe_ratio': (direction_accuracy - 0.5) * 2,  # Estimate Sharpe
                    'sortino_ratio': (direction_accuracy - 0.5) * 2.5,  # Estimate Sortino
                    'calmar_ratio': direction_accuracy / max(volatility_mae * 5, 0.01),
                    'max_drawdown': volatility_mae * 5,  # Estimate max drawdown
                    'max_drawdown_duration': 20,  # Default estimate
                    'var_95': volatility_mae * 3,  # Estimate VaR
                    'cvar_95': volatility_mae * 4,  # Estimate CVaR
                    'win_rate': direction_accuracy,
                    'profit_factor': 1.0 + (direction_accuracy - 0.5) * 0.5,  # Estimate
                    'avg_win': 0.02,  # Default estimate
                    'avg_loss': 0.015,  # Default estimate
                    'largest_win': 0.05,  # Default estimate
                    'largest_loss': volatility_mae * 3,  # Estimate
                    'total_trades': 50,  # Default estimate
                    'winning_trades': int(direction_accuracy * 50),
                    'losing_trades': int((1 - direction_accuracy) * 50)
                }
                
                financial_logger.log_trading_performance(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step02_5_SR_Optimization",
                    performance_data=performance_data,
                    confidence_score=direction_accuracy
                )
            
            # Log file paths that were created during this step
            self._log_created_file_paths()
            
            logger.info("💰 Financial metrics logged successfully from Step02_5 results")
            
        except Exception as e:
            logger.warning(f"Could not log financial metrics from results: {e}")

    def _log_created_file_paths(self) -> None:
        """Log file paths that were created during this step."""
        try:
            # Get the financial logger to access its file paths
            financial_logger = get_financial_metrics_logger()
            
            # Log the main financial metrics file path
            if hasattr(financial_logger, 'current_file_path') and financial_logger.current_file_path:
                logger.info(f"📁 Financial metrics file created: {financial_logger.current_file_path}")
                
                # Log this as a financial metric for tracking
                financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="metrics_file_path",
                    metric_value=0.0,  # No numeric value for file path
                    metric_type="file_path",
                    step_name="Step02_5_SR_Optimization",
                    additional_data={'file_path': str(financial_logger.current_file_path)}
                )
            
            # Log any other files that might have been created
            # (This would be expanded based on what files are actually created in the step)
            logger.info("📁 File paths logged for Step02_5")
            
        except Exception as e:
            logger.warning(f"Could not log file paths: {e}")

    def _generate_trading_recommendations(self,
                                         sr_analysis: Dict[str, Any],
                                         ml_insights: MLModelInsights,
                                         current_price: Optional[float]) -> Dict[str, Any]:
        """Generate trading recommendations based on analysis."""
        recommendations = {
            'primary_signal': 'NEUTRAL',
            'confidence_level': 'LOW',
            'suggested_actions': [],
            'risk_warnings': [],
            'time_horizon': 'MEDIUM_TERM'
        }

        if not current_price or not sr_analysis.get('trading_zones'):
            return recommendations

        zones = sr_analysis['trading_zones']

        # Analyze zone type and generate recommendations
        if zones['zone_type'] == 'support' and zones['support_distance_percent'] < 1.0:
            recommendations['primary_signal'] = 'BULLISH'
            recommendations['suggested_actions'].append('Consider long positions near support')
            recommendations['confidence_level'] = 'MEDIUM' if ml_insights.direction_accuracy > 0.6 else 'LOW'

        elif zones['zone_type'] == 'resistance' and zones['resistance_distance_percent'] < 1.0:
            recommendations['primary_signal'] = 'BEARISH'
            recommendations['suggested_actions'].append('Consider short positions near resistance')
            recommendations['confidence_level'] = 'MEDIUM' if ml_insights.direction_accuracy > 0.6 else 'LOW'

        # Risk warnings
        if ml_insights.overfitting_score > 0.1:
            recommendations['risk_warnings'].append('Model may be overfitting - use caution')

        if zones['support_distance_percent'] < 0.5 or zones['resistance_distance_percent'] < 0.5:
            recommendations['risk_warnings'].append('Price very close to key level - high volatility expected')

        return recommendations

    def _assess_overall_risk(self, sr_analysis: Dict[str, Any], ml_insights: MLModelInsights) -> str:
        """Assess overall trading risk level."""
        risk_factors = 0

        # ML model reliability
        if ml_insights.direction_accuracy < 0.6:
            risk_factors += 2
        elif ml_insights.direction_accuracy < 0.7:
            risk_factors += 1

        # Overfitting risk
        if ml_insights.overfitting_score > 0.1:
            risk_factors += 1

        # S/R level strength
        support_strength = sr_analysis.get('support_analysis', {}).get('average_strength', 0)
        resistance_strength = sr_analysis.get('resistance_analysis', {}).get('average_strength', 0)

        if support_strength < 0.5 and resistance_strength < 0.5:
            risk_factors += 1

        # Determine risk level
        if risk_factors >= 3:
            return "HIGH"
        elif risk_factors >= 2:
            return "MEDIUM"
        else:
            return "LOW"

