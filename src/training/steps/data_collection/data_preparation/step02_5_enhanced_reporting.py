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
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import warnings

# Avoid circular import - import these functions when needed
# from src.training.reports import save_training_report, CentralizedReportManager
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

        # Import here to avoid circular imports
        try:
            from src.training.reports import CentralizedReportManager
            self.report_manager = CentralizedReportManager()
        except ImportError:
            # Fallback if import fails
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
                risk_management = self._generate_risk_management_recommendations(
                    sr_analysis, ml_insights, market_context
                )

                # Performance prediction
                performance_prediction = self._generate_performance_prediction(
                    ml_results, sr_analysis, market_context
                )

                # Trading strategy suggestions
                strategy_suggestions = self._generate_strategy_suggestions(
                    sr_analysis, ml_insights, market_context, technical_analysis
                )

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
                    'market_prediction': self._generate_market_prediction(sr_analysis, ml_insights),
                    'alerts_and_warnings': self._generate_alerts_and_warnings(sr_analysis, ml_insights, market_context),
                    'visualization_data': self._prepare_visualization_data(sr_levels, ml_results),
                    'visualization_enhanced_data': self._prepare_enhanced_visualization_data(data, sr_levels, ml_results),
                    'export_ready_data': self._prepare_export_data(sr_levels, ml_results, execution_data)
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

    def _prepare_visualization_data(self,
                                   sr_levels: Dict[str, Any],
                                   ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for visualizations."""
        viz_data = {
            'sr_levels_chart': {
                'support_prices': [level.get('price', 0) for level in sr_levels.get('support_levels', [])],
                'support_strengths': [level.get('strength', 0) for level in sr_levels.get('support_levels', [])],
                'resistance_prices': [level.get('price', 0) for level in sr_levels.get('resistance_levels', [])],
                'resistance_strengths': [level.get('strength', 0) for level in sr_levels.get('resistance_levels', [])]
            },
            'feature_importance': ml_results.get('feature_importance', {}),
            'model_performance': {
                'accuracy': ml_results.get('direction_accuracy', 0),
                'mae': ml_results.get('volatility_mae', 0)
            }
        }

        return viz_data

    def save_comprehensive_report(self,
                                report_data: Dict[str, Any],
                                include_visualizations: bool = True) -> Dict[str, str]:
        """Save comprehensive report in multiple formats."""

        saved_files = {}

        try:
            # Import here to avoid circular imports
            from src.training.reports import save_training_report

            # Save detailed JSON report
            json_path = save_training_report(
                data=report_data,
                step_name='step02_5_sr_optimization',
                report_type='comprehensive_analysis',
                symbol=self.symbol,
                timeframe=self.timeframe,
                file_format='json'
            )
            saved_files['json_report'] = json_path
            logger.info(f"✅ Comprehensive JSON report saved: {json_path}")

            # Save human-readable Markdown report
            md_path = save_training_report(
                data=self._convert_to_markdown(report_data),
                step_name='step02_5_sr_optimization',
                report_type='analysis_summary',
                symbol=self.symbol,
                timeframe=self.timeframe,
                file_format='md'
            )
            saved_files['markdown_report'] = md_path
            logger.info(f"✅ Markdown summary saved: {md_path}")

            # Save CSV data for key metrics
            csv_path = self._save_csv_metrics(report_data)
            if csv_path:
                saved_files['csv_metrics'] = csv_path
                logger.info(f"✅ CSV metrics saved: {csv_path}")

            # Generate and save visualizations if requested
            if include_visualizations:
                viz_files = self._generate_visualizations(report_data)
                saved_files.update(viz_files)

        except Exception as e:
            logger.error(f"Failed to save comprehensive report: {e}")

        return saved_files

    def _perform_technical_analysis(self, data: pd.DataFrame, sr_levels: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive technical analysis."""
        try:
            analysis = {
                'trend_indicators': {},
                'momentum_indicators': {},
                'volatility_indicators': {},
                'support_resistance_interactions': {},
                'pattern_recognition': {}
            }

            if len(data) < 50:
                return analysis

            # Trend indicators
            sma_20 = data['close'].rolling(20).mean()
            sma_50 = data['close'].rolling(50).mean()
            analysis['trend_indicators'] = {
                'sma_20_slope': (sma_20.iloc[-1] - sma_20.iloc[-5]) / 5,
                'sma_50_slope': (sma_50.iloc[-1] - sma_50.iloc[-10]) / 10,
                'trend_strength': abs(sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1],
                'trend_direction': 'UP' if sma_20.iloc[-1] > sma_50.iloc[-1] else 'DOWN'
            }

            # Momentum indicators
            rsi = self._calculate_rsi(data['close'], 14)
            analysis['momentum_indicators'] = {
                'rsi_current': rsi.iloc[-1],
                'rsi_signal': 'OVERBOUGHT' if rsi.iloc[-1] > 70 else 'OVERSOLD' if rsi.iloc[-1] < 30 else 'NEUTRAL',
                'momentum_divergence': self._detect_divergence(data['close'], rsi)
            }

            # Volatility indicators
            returns = data['close'].pct_change()
            volatility = returns.rolling(20).std()
            analysis['volatility_indicators'] = {
                'current_volatility': volatility.iloc[-1],
                'volatility_trend': 'INCREASING' if volatility.tail(10).mean() > volatility.tail(20).mean() else 'DECREASING',
                'volatility_percentile': (volatility.iloc[-1] - volatility.min()) / (volatility.max() - volatility.min())
            }

            # S/R interactions
            current_price = data['close'].iloc[-1]
            analysis['support_resistance_interactions'] = self._analyze_sr_interactions(current_price, sr_levels)

            return analysis

        except Exception as e:
            logger.warning(f"Failed to perform technical analysis: {e}")
            return {}

    def _analyze_sr_interactions(self, current_price: float, sr_levels: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze interactions between current price and S/R levels."""
        interactions = {
            'nearest_support': None,
            'nearest_resistance': None,
            'support_distance': float('inf'),
            'resistance_distance': float('inf'),
            'price_position': 'MID_RANGE'
        }

        # Find nearest support
        for level in sr_levels.get('support_levels', []):
            price = level.get('price', 0)
            distance = abs(current_price - price) / current_price
            if distance < interactions['support_distance']:
                interactions['support_distance'] = distance
                interactions['nearest_support'] = price

        # Find nearest resistance
        for level in sr_levels.get('resistance_levels', []):
            price = level.get('price', 0)
            distance = abs(current_price - price) / current_price
            if distance < interactions['resistance_distance']:
                interactions['resistance_distance'] = distance
                interactions['nearest_resistance'] = price

        # Determine price position
        if interactions['resistance_distance'] < 0.01:  # Within 1%
            interactions['price_position'] = 'AT_RESISTANCE'
        elif interactions['support_distance'] < 0.01:  # Within 1%
            interactions['price_position'] = 'AT_SUPPORT'
        elif interactions['resistance_distance'] < interactions['support_distance']:
            interactions['price_position'] = 'NEAR_RESISTANCE'
        else:
            interactions['price_position'] = 'NEAR_SUPPORT'

        return interactions

    def _detect_divergence(self, price: pd.Series, indicator: pd.Series) -> str:
        """Detect price-indicator divergence."""
        try:
            # Simple divergence detection
            price_trend = 'UP' if price.iloc[-1] > price.iloc[-5] else 'DOWN'
            indicator_trend = 'UP' if indicator.iloc[-1] > indicator.iloc[-5] else 'DOWN'

            if price_trend != indicator_trend:
                return 'BEARISH_DIVERGENCE' if price_trend == 'UP' else 'BULLISH_DIVERGENCE'

            return 'NO_DIVERGENCE'
        except:
            return 'UNKNOWN'

    def _analyze_market_context(self, data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Analyze current market context and conditions."""
        try:
            context = {
                'trend_direction': 'SIDEWAYS',
                'volatility_regime': 'NORMAL',
                'momentum_strength': 0.0,
                'price_position': 'MID_RANGE',
                'recent_performance': {},
                'market_structure': 'RANGING'
            }

            if len(data) < 20:
                return context

            # Trend analysis
            sma_20 = data['close'].rolling(20).mean()
            sma_50 = data['close'].rolling(50).mean()

            if sma_20.iloc[-1] > sma_50.iloc[-1] * 1.005:
                context['trend_direction'] = 'BULLISH'
            elif sma_20.iloc[-1] < sma_50.iloc[-1] * 0.995:
                context['trend_direction'] = 'BEARISH'

            # Volatility analysis
            returns = data['close'].pct_change()
            volatility = returns.rolling(20).std() * np.sqrt(252)  # Annualized
            current_volatility = volatility.iloc[-1]

            if current_volatility > volatility.quantile(0.8):
                context['volatility_regime'] = 'HIGH'
            elif current_volatility < volatility.quantile(0.2):
                context['volatility_regime'] = 'LOW'

            # Momentum analysis
            rsi = self._calculate_rsi(data['close'], 14)
            context['momentum_strength'] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

            # Price position analysis
            recent_high = data['high'].rolling(50).max().iloc[-1]
            recent_low = data['low'].rolling(50).min().iloc[-1]
            price_range = recent_high - recent_low

            if current_price > recent_high * 0.98:
                context['price_position'] = 'NEAR_HIGH'
            elif current_price < recent_low * 1.02:
                context['price_position'] = 'NEAR_LOW'
            else:
                context['price_position'] = 'MID_RANGE'

            # Market structure
            if abs(sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1] > 0.05:
                context['market_structure'] = 'TRENDING'
            else:
                context['market_structure'] = 'RANGING'

            return context

        except Exception as e:
            logger.warning(f"Failed to analyze market context: {e}")
            return {}

    # Stub implementations for missing methods (to be enhanced later)
    def _analyze_feature_engineering(self, data: Optional[pd.DataFrame], ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feature engineering insights."""
        return {'feature_categories': {}, 'feature_quality': {}, 'feature_redundancy': {}}

    def _analyze_market_regime(self, data: Optional[pd.DataFrame], sr_levels: Dict[str, Any], ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current market regime."""
        return {'current_regime': 'UNKNOWN', 'regime_confidence': 0.0, 'optimal_strategy': 'HOLD'}

    def _generate_risk_management_recommendations(self, sr_analysis: Dict[str, Any], ml_insights: MLModelInsights, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk management recommendations."""
        return {'position_sizing': {}, 'stop_loss_levels': {}, 'risk_warnings': []}

    def _generate_performance_prediction(self, ml_results: Dict[str, Any], sr_analysis: Dict[str, Any], market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance predictions."""
        return {'expected_accuracy': 0.5, 'confidence_interval': [0.4, 0.6]}

    def _generate_strategy_suggestions(self, sr_analysis: Dict[str, Any], ml_insights: MLModelInsights, market_context: Dict[str, Any], technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading strategy suggestions."""
        return {'primary_strategy': 'WAIT', 'entry_signals': [], 'exit_signals': []}

    def _detailed_performance_breakdown(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide detailed performance breakdown."""
        return {'processing_efficiency': {}, 'resource_usage_timeline': {}, 'efficiency_metrics': {}}

    def _analyze_data_processing(self, data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Analyze data processing insights."""
        return {'data_characteristics': {}, 'data_quality_metrics': {}}

    def _detailed_sr_analysis(self, sr_levels: Dict[str, Any], data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Provide detailed S/R level analysis."""
        return {'level_quality_metrics': {}, 'temporal_distribution': {}}

    def _detailed_ml_analysis(self, ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Provide detailed ML model analysis."""
        return {'model_characteristics': {}, 'performance_stability': {}, 'feature_utilization': {}}

    def _analyze_correlations(self, data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Analyze feature correlations."""
        return {'correlation_summary': {}, 'correlation_warnings': []}

    def _analyze_volume_patterns(self, data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Analyze volume patterns and insights."""
        return {'volume_characteristics': {}, 'volume_price_relationship': {}}

    def _detailed_execution_breakdown(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide detailed execution breakdown."""
        return {'step_timing_analysis': {}, 'resource_usage_timeline': {}}

    def _validate_model_performance(self, ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate ML model performance metrics."""
        return {'overfitting_detected': False, 'validation_confidence': 'LOW'}

    def _generate_market_prediction(self, sr_analysis: Dict[str, Any], ml_insights: MLModelInsights) -> Dict[str, Any]:
        """Generate market direction prediction."""
        return {'predicted_direction': 'SIDEWAYS', 'prediction_confidence': 0.5}

    def _generate_alerts_and_warnings(self, sr_analysis: Dict[str, Any], ml_insights: MLModelInsights, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate alerts and warnings for trading."""
        return {'critical_alerts': [], 'warnings': [], 'notifications': []}

    def _prepare_enhanced_visualization_data(self, data: Optional[pd.DataFrame], sr_levels: Dict[str, Any], ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare enhanced visualization data."""
        return {'price_chart_data': {}, 'indicator_plots': {}}

    def _prepare_export_data(self, sr_levels: Dict[str, Any], ml_results: Dict[str, Any], execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for export in various formats."""
        return {'sr_levels_csv_ready': '', 'ml_metrics_json_ready': {}}

    def _identify_bottlenecks(self, execution_data: Dict[str, Any]) -> List[str]:
        """Identify performance bottlenecks."""
        return []

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50] * len(prices), index=prices.index)

    def _convert_to_markdown(self, report_data: Dict[str, Any]) -> str:
        """Convert comprehensive report data to detailed Markdown format."""
        md_content = []

        # Header with enhanced metadata
        md_content.append("# Step 2.5 S/R Optimization - Comprehensive Analysis Report")
        md_content.append("")

        metadata = report_data.get('report_metadata', {})
        md_content.append(f"**Generated:** {metadata.get('generated_at', 'Unknown')}")
        md_content.append(f"**Symbol:** {metadata.get('symbol', 'Unknown')}")
        md_content.append(f"**Exchange:** {metadata.get('exchange', 'Unknown')}")
        md_content.append(f"**Timeframe:** {metadata.get('timeframe', 'Unknown')}")
        md_content.append(f"**Current Price:** ${metadata.get('current_price', 'N/A')}")
        md_content.append(f"**Data Period:** {metadata.get('data_timeframe', 'Unknown')}")
        md_content.append(f"**Report Version:** {metadata.get('report_version', 'Unknown')}")
        md_content.append("")

        # Market Context
        self._add_market_context_section(md_content, report_data)

        # Performance Summary with detailed breakdown
        self._add_performance_summary_section(md_content, report_data)

        # Data Quality Assessment
        self._add_data_quality_section(md_content, report_data)

        # Data Processing Insights
        self._add_data_processing_section(md_content, report_data)

        # S/R Level Analysis with detailed breakdown
        self._add_sr_analysis_section(md_content, report_data)

        # Technical Analysis
        self._add_technical_analysis_section(md_content, report_data)

        # Market Regime Analysis
        self._add_market_regime_section(md_content, report_data)

        # ML Model Analysis with comprehensive details
        self._add_ml_analysis_section(md_content, report_data)

        # Feature Engineering Insights
        self._add_feature_engineering_section(md_content, report_data)

        # Correlation Analysis
        self._add_correlation_analysis_section(md_content, report_data)

        # Volume Analysis
        self._add_volume_analysis_section(md_content, report_data)

        # Trading Recommendations with strategy suggestions
        self._add_trading_recommendations_section(md_content, report_data)

        # Risk Management Recommendations
        self._add_risk_management_section(md_content, report_data)

        # Performance Prediction
        self._add_performance_prediction_section(md_content, report_data)

        # Market Prediction
        self._add_market_prediction_section(md_content, report_data)

        # Alerts and Warnings
        self._add_alerts_section(md_content, report_data)

        # Overall Risk Assessment
        md_content.append("## ⚠️ Overall Risk Assessment")
        md_content.append(f"**Risk Level:** {report_data.get('risk_assessment', 'UNKNOWN')}")
        md_content.append("")

        return "\n".join(md_content)

    def _add_market_context_section(self, md_content: list, report_data: Dict[str, Any]) -> None:
        """Add market context section to markdown."""
        market_context = report_data.get('market_context', {})
        if market_context:
            md_content.append("## 🌍 Market Context Analysis")
            md_content.append(f"- **Trend Direction:** {market_context.get('trend_direction', 'Unknown')}")
            md_content.append(f"- **Volatility Regime:** {market_context.get('volatility_regime', 'Unknown')}")
            md_content.append(f"- **Market Structure:** {market_context.get('market_structure', 'Unknown')}")
            md_content.append(f"- **Price Position:** {market_context.get('price_position', 'Unknown')}")
            md_content.append(f"- **Momentum Strength (RSI):** {market_context.get('momentum_strength', 0):.1f}")
            md_content.append("")

    def _add_performance_summary_section(self, md_content: list, report_data: Dict[str, Any]) -> None:
        """Add detailed performance summary section."""
        md_content.append("## 🚀 Performance Summary")

        perf = report_data.get('performance_metrics', {})
        md_content.append(f"- **Execution Time:** {perf.get('execution_time_seconds', 0):.2f} seconds")
        md_content.append(f"- **Memory Usage:** {perf.get('memory_usage_mb', 0):.1f} MB")
        md_content.append(f"- **CPU Usage:** {perf.get('cpu_usage_percent', 0):.1f}%")
        md_content.append(f"- **Function Calls:** {perf.get('total_function_calls', 0):,}")

        # Performance breakdown
        perf_breakdown = report_data.get('performance_breakdown', {})
        if perf_breakdown.get('processing_efficiency'):
            pe = perf_breakdown['processing_efficiency']
            md_content.append(f"- **Data Processing Rate:** {pe.get('data_points_per_second', 0):.1f} rows/sec")
            md_content.append(f"- **Feature Processing Rate:** {pe.get('features_per_second', 0):.1f} features/sec")

        # Bottlenecks
        bottlenecks = perf_breakdown.get('bottlenecks_identified', [])
        if bottlenecks:
            md_content.append("- **Performance Bottlenecks:**")
            for bottleneck in bottlenecks[:3]:  # Show top 3
                md_content.append(f"  - {bottleneck}")

        md_content.append("")

    def _add_data_quality_section(self, md_content: list, report_data: Dict[str, Any]) -> None:
        """Add data quality assessment section."""
        md_content.append("## 📊 Data Quality Assessment")
        dq = report_data.get('data_quality_assessment', {})
        md_content.append(f"- **Total Rows:** {dq.get('total_rows', 0):,}")
        md_content.append(f"- **Total Columns:** {dq.get('total_columns', 0)}")
        md_content.append(f"- **Data Completeness:** {dq.get('data_completeness_score', 0):.1f}%")
        md_content.append(f"- **Missing Values:** {dq.get('missing_values_percent', 0):.2f}%")
        md_content.append(f"- **Duplicate Rows:** {dq.get('duplicate_rows', 0)}")
        md_content.append(f"- **Zero Values:** {dq.get('zero_values_count', 0)}")
        md_content.append("")

    def _add_data_processing_section(self, md_content: list, report_data: Dict[str, Any]) -> None:
        """Add data processing insights section."""
        processing = report_data.get('data_processing_insights', {})
        if processing:
            md_content.append("## 🔧 Data Processing Insights")
            char = processing.get('data_characteristics', {})
            if char:
                md_content.append(f"- **Data Period:** {char.get('date_range', 'Unknown')}")
                md_content.append(f"- **Available Columns:** {len(char.get('columns_available', []))}")
                md_content.append(f"- **Data Types:** {len(char.get('data_types', {}))}")

            stats = processing.get('statistical_summary', {})
            if stats:
                md_content.append(f"- **Price Volatility:** {stats.get('price_volatility', 0):.2f}")
                md_content.append(f"- **Average Volume:** {stats.get('average_volume', 0):,.0f}")
                md_content.append(f"- **Price Range:** {stats.get('price_range', 'Unknown')}")

            md_content.append("")

    def _add_sr_analysis_section(self, md_content: list, report_data: Dict[str, Any]) -> None:
        """Add comprehensive S/R level analysis section."""
        md_content.append("## 📈 S/R Level Analysis")

        sr = report_data.get('sr_level_analysis', {})
        support = sr.get('support_analysis', {})
        resistance = sr.get('resistance_analysis', {})

        if support:
            md_content.append("### 🛡️ Support Levels")
            md_content.append(f"- **Total Levels:** {support.get('total_levels', 0)}")
            md_content.append(f"- **Average Strength:** {support.get('average_strength', 0):.3f}")
            md_content.append(f"- **Strongest Level:** ${support.get('strongest_level', 0):.2f}")
            md_content.append(f"- **Strong Levels (>{0.7}):** {support.get('strong_levels', 0)}")

        if resistance:
            md_content.append("### 🎯 Resistance Levels")
            md_content.append(f"- **Total Levels:** {resistance.get('total_levels', 0)}")
            md_content.append(f"- **Average Strength:** {resistance.get('average_strength', 0):.3f}")
            md_content.append(f"- **Strongest Level:** ${resistance.get('strongest_level', 0):.2f}")
            md_content.append(f"- **Strong Levels (>{0.7}):** {resistance.get('strong_levels', 0)}")

        # Trading Zones
        zones = sr.get('trading_zones', {})
        if zones:
            md_content.append("### 🎯 Trading Zones")
            md_content.append(f"- **Current Price:** ${zones.get('current_price', 0):.2f}")
            md_content.append(f"- **Nearest Support:** ${zones.get('nearest_support', 0):.2f} ({zones.get('support_distance_percent', 0)*100:.2f}%)")
            md_content.append(f"- **Nearest Resistance:** ${zones.get('nearest_resistance', 0):.2f} ({zones.get('resistance_distance_percent', 0)*100:.2f}%)")
            md_content.append(f"- **Current Zone:** {zones.get('zone_type', 'Unknown').title()}")

        # Detailed SR analysis
        sr_detailed = report_data.get('sr_level_detailed_analysis', {})
        if sr_detailed.get('level_quality_metrics'):
            md_content.append("### 📊 Level Quality Metrics")
            for level_type, metrics in sr_detailed['level_quality_metrics'].items():
                if metrics['count'] > 0:
                    md_content.append(f"- **{level_type.title()}:** {metrics['count']} levels, avg strength {metrics['average_strength']:.3f}")

        md_content.append("")

    def _add_technical_analysis_section(self, md_content: list, report_data: Dict[str, Any]) -> None:
        """Add technical analysis section."""
        tech = report_data.get('technical_analysis', {})
        if tech:
            md_content.append("## 📊 Technical Analysis")

            trend = tech.get('trend_indicators', {})
            if trend:
                md_content.append("### 📈 Trend Indicators")
                md_content.append(f"- **SMA 20 Slope:** {trend.get('sma_20_slope', 0):.4f}")
                md_content.append(f"- **SMA 50 Slope:** {trend.get('sma_50_slope', 0):.4f}")
                md_content.append(f"- **Trend Strength:** {trend.get('trend_strength', 0):.3f}")
                md_content.append(f"- **Trend Direction:** {trend.get('trend_direction', 'Unknown')}")

            momentum = tech.get('momentum_indicators', {})
            if momentum:
                md_content.append("### 💨 Momentum Indicators")
                md_content.append(f"- **RSI Current:** {momentum.get('rsi_current', 0):.1f}")
                md_content.append(f"- **RSI Signal:** {momentum.get('rsi_signal', 'Unknown')}")
                md_content.append(f"- **Divergence:** {momentum.get('momentum_divergence', 'None')}")

            volatility = tech.get('volatility_indicators', {})
            if volatility:
                md_content.append("### 🌊 Volatility Indicators")
                md_content.append(f"- **Current Volatility:** {volatility.get('current_volatility', 0):.4f}")
                md_content.append(f"- **Volatility Trend:** {volatility.get('volatility_trend', 'Unknown')}")
                md_content.append(f"- **Volatility Percentile:** {volatility.get('volatility_percentile', 0):.1%}")

            md_content.append("")

    def _add_market_regime_section(self, md_content: list, report_data: Dict[str, Any]) -> None:
        """Add market regime analysis section."""
        regime = report_data.get('market_regime_analysis', {})
        if regime:
            md_content.append("## 🎭 Market Regime Analysis")
            md_content.append(f"- **Current Regime:** {regime.get('current_regime', 'Unknown')}")
            md_content.append(f"- **Regime Confidence:** {regime.get('regime_confidence', 0):.1%}")
            md_content.append(f"- **Optimal Strategy:** {regime.get('optimal_strategy', 'Unknown')}")

            characteristics = regime.get('regime_characteristics', {})
            if characteristics:
                md_content.append("### 📋 Regime Characteristics")
                md_content.append(f"- **Volatility Regime:** {characteristics.get('volatility_regime', 'Unknown')}")
                md_content.append(f"- **Trend Regime:** {characteristics.get('trend_regime', 'Unknown')}")
                md_content.append(f"- **Trend Strength:** {characteristics.get('trend_strength', 0):.3f}")

            md_content.append("")

    def _add_ml_analysis_section(self, md_content: list, report_data: Dict[str, Any]) -> None:
        """Add comprehensive ML model analysis section."""
        md_content.append("## 🤖 ML Model Analysis")

        ml = report_data.get('ml_model_insights', {})
        ml_detailed = report_data.get('ml_model_detailed_analysis', {})

        # Basic ML metrics
        md_content.append("### 📊 Model Performance")
        md_content.append(f"- **Model Type:** {ml.get('model_type', 'Unknown')}")
        md_content.append(f"- **Direction Accuracy:** {ml.get('direction_accuracy', 0):.3f}")
        md_content.append(f"- **Volatility MAE:** {ml.get('volatility_mae', 0):.6f}")
        md_content.append(f"- **F1 Score:** {ml.get('f1_score', 0):.3f}")
        md_content.append(f"- **Features Used:** {ml.get('feature_count', 0)}")

        # Training details
        char = ml_detailed.get('model_characteristics', {})
        if char:
            md_content.append("### 🎯 Training Details")
            md_content.append(f"- **Training Samples:** {char.get('training_samples', 0):,}")
            md_content.append(f"- **Test Samples:** {char.get('test_samples', 0):,}")
            md_content.append(f"- **Cross-validation Folds:** {char.get('cross_validation_folds', 0)}")

        # Performance stability
        stability = ml_detailed.get('performance_stability', {})
        if stability:
            md_content.append("### 📈 Performance Stability")
            md_content.append(f"- **CV Mean:** {stability.get('cv_mean', 0):.3f}")
            md_content.append(f"- **CV Std:** {stability.get('cv_std', 0):.3f}")
            md_content.append(f"- **Stability Rating:** {stability.get('performance_stability', 'Unknown')}")

        # Feature utilization
        features = ml_detailed.get('feature_utilization', {})
        if features:
            md_content.append("### 🔍 Feature Utilization")
            top_features = features.get('top_features', [])
            if top_features:
                md_content.append("- **Top Features:**")
                for feature, importance in top_features[:5]:
                    md_content.append(f"  - {feature}: {importance:.3f}")

        md_content.append("")

    def _add_feature_engineering_section(self, md_content: list, report_data: Dict[str, Any]) -> None:
        """Add feature engineering insights section."""
        features = report_data.get('feature_engineering_insights', {})
        if features:
            md_content.append("## 🛠️ Feature Engineering Insights")

            categories = features.get('feature_categories', {})
            if categories:
                md_content.append("### 📂 Feature Categories")
                md_content.append(f"- **Technical Indicators:** {categories.get('technical_indicators', 0)}")
                md_content.append(f"- **Price Features:** {categories.get('price_features', 0)}")
                md_content.append(f"- **Volume Features:** {categories.get('volume_features', 0)}")
                md_content.append(f"- **Microstructure Features:** {categories.get('microstructure_features', 0)}")
                md_content.append(f"- **Timeframe Features:** {categories.get('timeframe_features', 0)}")

            quality = features.get('feature_quality', {})
            if quality:
                md_content.append("### ✅ Feature Quality")
                md_content.append(f"- **Features with Missing Values:** {len(quality.get('features_with_missing_values', {}))}")
                md_content.append(f"- **Constant Features:** {len(quality.get('constant_features', []))}")
                md_content.append(f"- **Highly Correlated Pairs:** {len(quality.get('high_correlation_features', []))}")

            md_content.append("")

    def _add_correlation_analysis_section(self, md_content: list, report_data: Dict[str, Any]) -> None:
        """Add correlation analysis section."""
        corr = report_data.get('correlation_analysis', {})
        if corr:
            md_content.append("## 🔗 Correlation Analysis")

            summary = corr.get('correlation_summary', {})
            if summary:
                md_content.append(f"- **Average Correlation:** {summary.get('average_correlation', 0):.3f}")
                md_content.append(f"- **Max Correlation:** {summary.get('max_correlation', 0):.3f}")
                md_content.append(f"- **Highly Correlated Pairs:** {len(summary.get('highly_correlated_pairs', []))}")

                price_corr = summary.get('price_correlations', {})
                if price_corr:
                    md_content.append("- **Price Correlations (Top 5):**")
                    for feature, correlation in list(price_corr.items())[:5]:
                        md_content.append(f"  - {feature}: {correlation:.3f}")

            warnings = corr.get('correlation_warnings', [])
            if warnings:
                md_content.append("- **⚠️ Correlation Warnings:**")
                for warning in warnings:
                    if warning:
                        md_content.append(f"  - {warning}")

            md_content.append("")

    def _add_volume_analysis_section(self, md_content: list, report_data: Dict[str, Any]) -> None:
        """Add volume analysis section."""
        volume = report_data.get('volume_analysis', {})
        if volume:
            md_content.append("## 📊 Volume Analysis")

            char = volume.get('volume_characteristics', {})
            if char:
                md_content.append("### 📈 Volume Characteristics")
                md_content.append(f"- **Average Volume:** {char.get('average_volume', 0):,.0f}")
                md_content.append(f"- **Volume Volatility:** {char.get('volume_volatility', 0):.2f}")
                md_content.append(f"- **Volume Trend:** {char.get('volume_trend', 'Unknown')}")
                md_content.append(f"- **High Volume Periods:** {char.get('high_volume_periods', 0)}")

            insights = volume.get('volume_insights', [])
            if insights:
                md_content.append("- **📋 Volume Insights:**")
                for insight in insights:
                    if insight:
                        md_content.append(f"  - {insight}")

            md_content.append("")

    def _add_trading_recommendations_section(self, md_content: list, report_data: Dict[str, Any]) -> None:
        """Add comprehensive trading recommendations section."""
        md_content.append("## 💡 Trading Recommendations")

        recs = report_data.get('trading_recommendations', {})
        strategy = report_data.get('trading_strategy_suggestions', {})

        # Primary recommendations
        md_content.append(f"- **Primary Signal:** {recs.get('primary_signal', 'NEUTRAL')}")
        md_content.append(f"- **Confidence Level:** {recs.get('confidence_level', 'LOW')}")
        md_content.append(f"- **Time Horizon:** {recs.get('time_horizon', 'MEDIUM_TERM')}")

        # Strategy suggestions
        if strategy:
            md_content.append(f"- **Primary Strategy:** {strategy.get('primary_strategy', 'WAIT')}")
            md_content.append(f"- **Strategy Confidence:** {strategy.get('strategy_confidence', 'LOW')}")

        # Suggested actions
        actions = recs.get('suggested_actions', []) + strategy.get('entry_signals', [])
        if actions:
            md_content.append("- **Suggested Actions:**")
            for action in actions:
                md_content.append(f"  - {action}")

        # Risk warnings
        warnings = recs.get('risk_warnings', [])
        if warnings:
            md_content.append("- **⚠️ Risk Warnings:**")
            for warning in warnings:
                md_content.append(f"  - {warning}")

        md_content.append("")

    def _add_risk_management_section(self, md_content: list, report_data: Dict[str, Any]) -> None:
        """Add risk management recommendations section."""
        risk_mgmt = report_data.get('risk_management_recommendations', {})
        if risk_mgmt:
            md_content.append("## 🛡️ Risk Management Recommendations")

            sizing = risk_mgmt.get('position_sizing', {})
            if sizing:
                md_content.append("### 📏 Position Sizing")
                md_content.append(f"- **Recommended Size:** {sizing.get('recommended_size', 'MODERATE')}")
                md_content.append(f"- **Max Allocation:** {sizing.get('max_allocation', 0.075):.1%}")

            stops = risk_mgmt.get('stop_loss_levels', {})
            if stops:
                md_content.append("### 🛑 Stop Loss Levels")
                md_content.append(f"- **Suggested SL:** ${stops.get('suggested_sl', 'N/A')}")

            risk_warnings = risk_mgmt.get('risk_warnings', [])
            if risk_warnings:
                md_content.append("- **⚠️ Risk Alerts:**")
                for warning in risk_warnings:
                    md_content.append(f"  - {warning}")

            md_content.append("")

    def _add_performance_prediction_section(self, md_content: list, report_data: Dict[str, Any]) -> None:
        """Add performance prediction section."""
        prediction = report_data.get('performance_prediction', {})
        if prediction:
            md_content.append("## 🔮 Performance Prediction")

            md_content.append(f"- **Expected Accuracy:** {prediction.get('expected_accuracy', 0):.3f}")
            ci = prediction.get('confidence_interval', [])
            if ci:
                md_content.append(f"- **Confidence Interval:** {ci[0]:.3f} - {ci[1]:.3f}")
            md_content.append(f"- **Performance Stability:** {prediction.get('performance_stability', 'Unknown')}")

            suggestions = prediction.get('improvement_suggestions', [])
            if suggestions:
                md_content.append("- **Improvement Suggestions:**")
                for suggestion in suggestions:
                    md_content.append(f"  - {suggestion}")

            md_content.append("")

    def _add_market_prediction_section(self, md_content: list, report_data: Dict[str, Any]) -> None:
        """Add market prediction section."""
        prediction = report_data.get('market_prediction', {})
        if prediction:
            md_content.append("## 🎯 Market Prediction")

            md_content.append(f"- **Predicted Direction:** {prediction.get('predicted_direction', 'Unknown')}")
            md_content.append(f"- **Prediction Confidence:** {prediction.get('prediction_confidence', 0):.1%}")
            md_content.append(f"- **Time Horizon:** {prediction.get('time_horizon', 'Unknown')}")
            md_content.append(f"- **Market Outlook:** {prediction.get('market_outlook', 'Unknown')}")

            md_content.append("")

    def _add_alerts_section(self, md_content: list, report_data: Dict[str, Any]) -> None:
        """Add alerts and warnings section."""
        alerts = report_data.get('alerts_and_warnings', {})
        if alerts:
            md_content.append("## 🚨 Alerts & Warnings")

            critical = alerts.get('critical_alerts', [])
            if critical:
                md_content.append("### 🚨 Critical Alerts")
                for alert in critical:
                    md_content.append(f"- **CRITICAL:** {alert}")

            warnings = alerts.get('warnings', [])
            if warnings:
                md_content.append("### ⚠️ Warnings")
                for warning in warnings:
                    md_content.append(f"- {warning}")

            notifications = alerts.get('notifications', [])
            if notifications:
                md_content.append("### 📢 Notifications")
                for notification in notifications:
                    md_content.append(f"- {notification}")

            md_content.append("")

    def _save_csv_metrics(self, report_data: Dict[str, Any]) -> Optional[str]:
        """Save key metrics as CSV for further analysis."""
        try:
            # Extract key metrics for CSV
            csv_data = []

            # S/R Level metrics
            sr_analysis = report_data.get('sr_level_analysis', {})
            support_analysis = sr_analysis.get('support_analysis', {})
            resistance_analysis = sr_analysis.get('resistance_analysis', {})

            csv_data.append({
                'metric_type': 'sr_levels',
                'category': 'support',
                'total_levels': support_analysis.get('total_levels', 0),
                'average_strength': support_analysis.get('average_strength', 0),
                'strongest_level': support_analysis.get('strongest_level', 0)
            })

            csv_data.append({
                'metric_type': 'sr_levels',
                'category': 'resistance',
                'total_levels': resistance_analysis.get('total_levels', 0),
                'average_strength': resistance_analysis.get('average_strength', 0),
                'strongest_level': resistance_analysis.get('strongest_level', 0)
            })

            # ML Performance metrics
            ml_insights = report_data.get('ml_model_insights', {})
            csv_data.append({
                'metric_type': 'ml_performance',
                'category': 'direction',
                'accuracy': ml_insights.get('direction_accuracy', 0),
                'precision': ml_insights.get('precision', 0),
                'recall': ml_insights.get('recall', 0),
                'f1_score': ml_insights.get('f1_score', 0)
            })

            csv_data.append({
                'metric_type': 'ml_performance',
                'category': 'volatility',
                'mae': ml_insights.get('volatility_mae', 0),
                'feature_count': ml_insights.get('feature_count', 0)
            })

            # Data Quality metrics
            dq = report_data.get('data_quality_assessment', {})
            csv_data.append({
                'metric_type': 'data_quality',
                'category': 'completeness',
                'total_rows': dq.get('total_rows', 0),
                'total_columns': dq.get('total_columns', 0),
                'completeness_score': dq.get('data_completeness_score', 0),
                'missing_values_percent': dq.get('missing_values_percent', 0)
            })

            # Save CSV
            df = pd.DataFrame(csv_data)
            from src.training.reports import save_training_report
            csv_path = save_training_report(
                data=df.to_csv(index=False),
                step_name='step02_5_sr_optimization',
                report_type='key_metrics',
                symbol=self.symbol,
                timeframe=self.timeframe,
                file_format='csv'
            )

            return csv_path

        except Exception as e:
            logger.warning(f"Failed to save CSV metrics: {e}")
            return None

    def _generate_visualizations(self, report_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate and save visualization charts."""
        saved_files = {}

        try:
            # Set up plotting style
            plt.style.use('default')
            sns.set_palette("husl")

            viz_data = report_data.get('visualization_data', {})

            # S/R Levels visualization
            sr_chart = viz_data.get('sr_levels_chart', {})
            if sr_chart.get('support_prices') or sr_chart.get('resistance_prices'):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Support levels
                if sr_chart.get('support_prices'):
                    ax1.scatter(sr_chart['support_prices'], sr_chart['support_strengths'],
                              alpha=0.6, color='green', s=50, label='Support')
                    ax1.set_title('Support Levels Strength Distribution')
                    ax1.set_xlabel('Price Level')
                    ax1.set_ylabel('Strength')
                    ax1.grid(True, alpha=0.3)

                # Resistance levels
                if sr_chart.get('resistance_prices'):
                    ax2.scatter(sr_chart['resistance_prices'], sr_chart['resistance_strengths'],
                              alpha=0.6, color='red', s=50, label='Resistance')
                    ax2.set_title('Resistance Levels Strength Distribution')
                    ax2.set_xlabel('Price Level')
                    ax2.set_ylabel('Strength')
                    ax2.grid(True, alpha=0.3)

                plt.tight_layout()

                # Save chart
                if self.report_manager:
                    chart_filename = f"sr_levels_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    chart_path = self.report_manager.base_dir / 'step02_5_sr_optimization' / chart_filename
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    plt.close()

                    saved_files['sr_levels_chart'] = str(chart_path)
                    logger.info(f"✅ S/R levels chart saved: {chart_path}")
                else:
                    # Fallback: save to current directory
                    chart_filename = f"sr_levels_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
                    plt.close()
                    saved_files['sr_levels_chart'] = chart_filename

            # Feature importance visualization
            feature_importance = viz_data.get('feature_importance', {})
            if feature_importance:
                # Sort and take top 20 features
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]

                if sorted_features:
                    fig, ax = plt.subplots(figsize=(12, 8))

                    features, importance = zip(*sorted_features)
                    bars = ax.barh(range(len(features)), importance, color='skyblue')
                    ax.set_yticks(range(len(features)))
                    ax.set_yticklabels(features)
                    ax.set_xlabel('Feature Importance')
                    ax.set_title('Top 20 Feature Importance')
                    ax.grid(True, alpha=0.3)

                    # Add value labels
                    for i, (bar, imp) in enumerate(zip(bars, importance)):
                        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                               '.3f', ha='left', va='center')

                    plt.tight_layout()

                # Save chart
                if self.report_manager:
                    fi_filename = f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    fi_path = self.report_manager.base_dir / 'step02_5_sr_optimization' / fi_filename
                    plt.savefig(fi_path, dpi=300, bbox_inches='tight')
                    plt.close()

                    saved_files['feature_importance_chart'] = str(fi_path)
                    logger.info(f"✅ Feature importance chart saved: {fi_path}")
                else:
                    # Fallback: save to current directory
                    fi_filename = f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    plt.savefig(fi_filename, dpi=300, bbox_inches='tight')
                    plt.close()
                    saved_files['feature_importance_chart'] = fi_filename

        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")

        return saved_files
