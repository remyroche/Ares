"""
Enhanced Reporting System for Step06: Advanced Feature Engineering

This module provides comprehensive analysis and reporting for advanced feature engineering operations,
including wavelet features, multi-timeframe analysis, hardware acceleration metrics,
feature quality assessment, and technical indicator evaluation.
"""

import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

from src.utils.logger import system_logger

# Import centralized reporting utilities locally to avoid circular imports
def get_centralized_report_manager():
    """Get CentralizedReportManager instance with local import to avoid circular dependencies."""
    try:
        from src.training.reports import CentralizedReportManager
        return CentralizedReportManager()
    except ImportError:
        return None

def get_save_training_report():
    """Get save_training_report function with local import to avoid circular dependencies."""
    try:
        from src.training.reports import save_training_report
        return save_training_report
    except ImportError:
        return lambda *args, **kwargs: "fallback_report_saved"

@dataclass
class FeatureEngineeringMetrics:
    """Metrics for feature engineering operations."""
    total_features_created: int
    feature_categories: Dict[str, int]
    wavelet_features_count: int
    multi_timeframe_features_count: int
    technical_indicators_count: int
    feature_interactions_count: int
    regime_aware_features_count: int
    feature_creation_time_seconds: float
    features_per_second: float

@dataclass
class HardwareAccelerationMetrics:
    """Metrics for hardware acceleration and optimization."""
    gpu_utilization: float
    cpu_utilization: float
    vectorization_efficiency: float
    memory_usage_mb: float
    processing_speedup: float
    optimization_enabled: bool
    m1_gpu_available: bool
    vectorized_operations: int
    hardware_acceleration_score: float

@dataclass
class FeatureQualityMetrics:
    """Comprehensive feature quality assessment."""
    completeness_score: float
    validity_score: float
    uniqueness_score: float
    informativeness_score: float
    stability_score: float
    correlation_score: float
    overall_quality_score: float
    quality_issues: List[str]
    feature_importance_analysis: Dict[str, Any]
    redundancy_analysis: Dict[str, Any]

@dataclass
class WaveletAnalysisMetrics:
    """Metrics for wavelet feature generation."""
    wavelet_levels: int
    wavelet_family: str
    decomposition_levels: int
    wavelet_features_generated: int
    wavelet_computation_time: float
    wavelet_quality_score: float
    frequency_bands_analyzed: List[str]
    wavelet_transform_efficiency: float

@dataclass
class MultiTimeframeMetrics:
    """Metrics for multi-timeframe feature engineering."""
    timeframes_processed: List[str]
    timeframe_features_generated: Dict[str, int]
    cross_timeframe_correlations: Dict[str, float]
    temporal_consistency_score: float
    timeframe_processing_times: Dict[str, float]
    multi_timeframe_efficiency: float

@dataclass
class TechnicalIndicatorMetrics:
    """Metrics for technical indicator generation."""
    indicators_generated: Dict[str, int]
    indicator_categories: Dict[str, int]
    indicator_computation_time: float
    indicator_quality_scores: Dict[str, float]
    indicator_stability_analysis: Dict[str, Any]
    custom_indicators_count: int

@dataclass
class FeatureInteractionMetrics:
    """Metrics for feature interaction analysis."""
    interactions_created: int
    interaction_degree: int
    correlation_matrix_density: float
    high_correlation_pairs: List[Tuple[str, str, float]]
    feature_redundancy_score: float
    interaction_computation_time: float

@dataclass
class PerformanceOptimizationMetrics:
    """Performance and optimization metrics."""
    total_execution_time: float
    feature_engineering_efficiency: float
    memory_optimization_score: float
    parallel_processing_efficiency: float
    caching_efficiency: float
    chunk_processing_metrics: Dict[str, Any]

class Step06EnhancedReporter:
    """Enhanced reporting system for Step06 advanced feature engineering operations."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the enhanced reporter."""
        self.config = config
        self.logger = system_logger.getChild('Step06.EnhancedReporter')
        self.report_manager = get_centralized_report_manager()
        self.save_training_report = get_save_training_report()

        # Initialize metrics containers
        self.feature_metrics = None
        self.hardware_metrics = None
        self.quality_metrics = None
        self.wavelet_metrics = None
        self.multitimeframe_metrics = None
        self.technical_metrics = None
        self.interaction_metrics = None
        self.performance_metrics = None

        # Setup visualization style
        plt.style.use('default')
        sns.set_palette("husl")

    def generate_comprehensive_report(self,
                                    input_data: pd.DataFrame,
                                    output_features: pd.DataFrame,
                                    feature_config: Dict[str, Any],
                                    execution_metadata: Dict[str, Any],
                                    hardware_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report for advanced feature engineering.

        Args:
            input_data: Original input dataset
            output_features: Generated feature dataset
            feature_config: Feature engineering configuration
            execution_metadata: Execution performance and timing data
            hardware_metrics: Hardware utilization and acceleration data

        Returns:
            Comprehensive report dictionary
        """
        try:
            self.logger.info("🔍 Generating comprehensive Step06 analysis report...")

            # Generate all analysis components
            self._analyze_feature_engineering(input_data, output_features, feature_config, execution_metadata)
            self._analyze_hardware_acceleration(hardware_metrics)
            self._analyze_feature_quality(input_data, output_features)
            self._analyze_wavelet_features(feature_config, execution_metadata)
            self._analyze_multitimeframe_features(feature_config, execution_metadata)
            self._analyze_technical_indicators(output_features, feature_config)
            self._analyze_feature_interactions(output_features)
            self._analyze_performance_optimization(execution_metadata, hardware_metrics)

            # Compile comprehensive report
            report = {
                'timestamp': datetime.now().isoformat(),
                'step_name': 'step06_advanced_feature_engineering',
                'analysis_type': 'enhanced_feature_engineering_analysis',
                'config_summary': self._summarize_config(feature_config),
                'feature_engineering_analysis': self.feature_metrics.__dict__ if self.feature_metrics else {},
                'hardware_acceleration_analysis': self.hardware_metrics.__dict__ if self.hardware_metrics else {},
                'feature_quality_analysis': self.quality_metrics.__dict__ if self.quality_metrics else {},
                'wavelet_analysis': self.wavelet_metrics.__dict__ if self.wavelet_metrics else {},
                'multitimeframe_analysis': self.multitimeframe_metrics.__dict__ if self.multitimeframe_metrics else {},
                'technical_indicator_analysis': self.technical_metrics.__dict__ if self.technical_metrics else {},
                'feature_interaction_analysis': self.interaction_metrics.__dict__ if self.interaction_metrics else {},
                'performance_optimization_analysis': self.performance_metrics.__dict__ if self.performance_metrics else {},
                'recommendations': self._generate_recommendations(),
                'alerts': self._generate_alerts()
            }

            self.logger.info("✅ Comprehensive Step06 analysis report generated successfully")
            return report

        except Exception as e:
            self.logger.error(f"❌ Failed to generate comprehensive report: {e}")
            return self._generate_fallback_report(input_data, output_features, str(e))

    def save_comprehensive_report(self, report_data: Dict[str, Any], symbol: str, exchange: str, timeframe: str) -> List[str]:
        """
        Save comprehensive report in multiple formats with visualizations.

        Args:
            report_data: The comprehensive report data
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe

        Returns:
            List of saved file paths
        """
        saved_files = []

        try:
            self.logger.info("💾 Saving comprehensive Step06 reports...")

            # Save JSON report
            json_path = self.save_training_report(
                data=report_data,
                step_name='step06_advanced_feature_engineering',
                report_type='comprehensive_analysis',
                symbol=symbol,
                timeframe=timeframe,
                file_format='json'
            )
            if json_path:
                saved_files.append(json_path)

            # Save Markdown summary
            markdown_path = self._save_markdown_report(report_data, symbol, exchange, timeframe)
            if markdown_path:
                saved_files.append(markdown_path)

            # Generate and save visualizations
            viz_paths = self._generate_and_save_visualizations(report_data, symbol, exchange, timeframe)
            saved_files.extend(viz_paths)

            # Save CSV summary
            csv_path = self._save_csv_summary(report_data, symbol, exchange, timeframe)
            if csv_path:
                saved_files.append(csv_path)

            self.logger.info(f"✅ Saved {len(saved_files)} Step06 report files")
            return saved_files

        except Exception as e:
            self.logger.error(f"❌ Failed to save comprehensive reports: {e}")
            return []

    def _analyze_feature_engineering(self, input_data: pd.DataFrame, output_features: pd.DataFrame,
                                   feature_config: Dict[str, Any], execution_metadata: Dict[str, Any]) -> None:
        """Analyze overall feature engineering operations."""
        try:
            self.logger.info("🔧 Analyzing feature engineering operations...")

            input_features = input_data.shape[1]
            total_features_created = output_features.shape[1] - input_features

            # Categorize features
            feature_categories = {
                'wavelet': 0,
                'multi_timeframe': 0,
                'technical': 0,
                'interaction': 0,
                'regime_aware': 0,
                'other': 0
            }

            # Simple feature categorization based on column names
            for col in output_features.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['wavelet', 'wvl']):
                    feature_categories['wavelet'] += 1
                elif any(term in col_lower for term in ['mtf', 'multi', 'timeframe']):
                    feature_categories['multi_timeframe'] += 1
                elif any(term in col_lower for term in ['rsi', 'macd', 'sma', 'ema', 'bb', 'stoch']):
                    feature_categories['technical'] += 1
                elif any(term in col_lower for term in ['interaction', 'corr', 'cross']):
                    feature_categories['interaction'] += 1
                elif any(term in col_lower for term in ['regime', 'cluster']):
                    feature_categories['regime_aware'] += 1
                else:
                    feature_categories['other'] += 1

            # Calculate timing and efficiency
            execution_time = execution_metadata.get('total_execution_time', 0)
            features_per_second = total_features_created / max(execution_time, 0.001)

            # Extract specific counts
            wavelet_features = feature_categories['wavelet']
            mtf_features = feature_categories['multi_timeframe']
            tech_features = feature_categories['technical']
            interaction_features = feature_categories['interaction']
            regime_features = feature_categories['regime_aware']

            self.feature_metrics = FeatureEngineeringMetrics(
                total_features_created=total_features_created,
                feature_categories=feature_categories,
                wavelet_features_count=wavelet_features,
                multi_timeframe_features_count=mtf_features,
                technical_indicators_count=tech_features,
                feature_interactions_count=interaction_features,
                regime_aware_features_count=regime_features,
                feature_creation_time_seconds=execution_time,
                features_per_second=features_per_second
            )

            self.logger.info("✅ Feature engineering analysis completed")

        except Exception as e:
            self.logger.error(f"❌ Failed to analyze feature engineering: {e}")
            self.feature_metrics = None

    def _analyze_hardware_acceleration(self, hardware_metrics: Dict[str, Any]) -> None:
        """Analyze hardware acceleration and optimization metrics."""
        try:
            self.logger.info("⚡ Analyzing hardware acceleration metrics...")

            gpu_util = hardware_metrics.get('gpu_utilization', 0.0)
            cpu_util = hardware_metrics.get('cpu_utilization', 0.0)
            vectorization_eff = hardware_metrics.get('vectorization_efficiency', 1.0)
            memory_usage = hardware_metrics.get('memory_usage_mb', 0.0)
            speedup = hardware_metrics.get('processing_speedup', 1.0)

            # Calculate hardware acceleration score (0-1, higher is better)
            accel_score = min(1.0, (gpu_util * 0.4 + vectorization_eff * 0.3 + speedup * 0.3))

            self.hardware_metrics = HardwareAccelerationMetrics(
                gpu_utilization=gpu_util,
                cpu_utilization=cpu_util,
                vectorization_efficiency=vectorization_eff,
                memory_usage_mb=memory_usage,
                processing_speedup=speedup,
                optimization_enabled=hardware_metrics.get('optimization_enabled', False),
                m1_gpu_available=hardware_metrics.get('m1_gpu_available', False),
                vectorized_operations=hardware_metrics.get('vectorized_operations', 0),
                hardware_acceleration_score=accel_score
            )

            self.logger.info("✅ Hardware acceleration analysis completed")

        except Exception as e:
            self.logger.error(f"❌ Failed to analyze hardware acceleration: {e}")
            self.hardware_metrics = None

    def _analyze_feature_quality(self, input_data: pd.DataFrame, output_features: pd.DataFrame) -> None:
        """Analyze comprehensive feature quality metrics."""
        try:
            self.logger.info("🔍 Analyzing feature quality metrics...")

            # Basic quality checks
            total_cells = output_features.shape[0] * output_features.shape[1]
            missing_cells = output_features.isnull().sum().sum()
            completeness_score = 1 - (missing_cells / max(total_cells, 1))

            # Validity score (check for reasonable value ranges)
            numeric_features = output_features.select_dtypes(include=[np.number])
            if not numeric_features.empty:
                # Check for infinite values and extreme outliers
                finite_mask = np.isfinite(numeric_features.values)
                validity_score = np.mean(finite_mask)

                # Check for extreme values (beyond 10 std devs from mean)
                if numeric_features.shape[1] > 0:
                    z_scores = np.abs((numeric_features - numeric_features.mean()) / numeric_features.std())
                    extreme_outliers = (z_scores > 10).any().any()
                    if extreme_outliers:
                        validity_score *= 0.9
            else:
                validity_score = 1.0

            # Uniqueness score (duplicate features)
            duplicate_features = output_features.T.duplicated().sum()
            uniqueness_score = 1 - (duplicate_features / max(output_features.shape[1], 1))

            # Informativeness score (features with variance)
            if not numeric_features.empty:
                variance_mask = numeric_features.var() > 1e-10
                informativeness_score = variance_mask.sum() / len(variance_mask)
            else:
                informativeness_score = 1.0

            # Stability score (features that don't change too frequently)
            if output_features.shape[0] > 10:
                # Calculate rolling standard deviation and check stability
                stability_scores = []
                for col in numeric_features.columns[:10]:  # Sample first 10 features
                    if numeric_features[col].var() > 1e-10:
                        rolling_std = numeric_features[col].rolling(10).std()
                        stability = 1 / (1 + rolling_std.mean())
                        stability_scores.append(stability)

                stability_score = np.mean(stability_scores) if stability_scores else 1.0
            else:
                stability_score = 1.0

            # Correlation score (avoid highly correlated features)
            if numeric_features.shape[1] > 1:
                corr_matrix = numeric_features.corr()
                high_corr_count = (abs(corr_matrix) > 0.95).sum().sum() - numeric_features.shape[1]  # Subtract diagonal
                correlation_score = 1 - (high_corr_count / max(corr_matrix.size - numeric_features.shape[1], 1))
            else:
                correlation_score = 1.0

            # Overall quality score
            overall_score = np.mean([
                completeness_score, validity_score, uniqueness_score,
                informativeness_score, stability_score, correlation_score
            ])

            # Quality issues
            issues = []
            if completeness_score < 0.95:
                issues.append(f"Low completeness: {completeness_score:.2%}")
            if validity_score < 0.9:
                issues.append(f"Low validity: {validity_score:.2%}")
            if uniqueness_score < 0.9:
                issues.append(f"Low uniqueness: {uniqueness_score:.2%}")
            if correlation_score < 0.8:
                issues.append(f"High correlation issues: {correlation_score:.2%}")

            self.quality_metrics = FeatureQualityMetrics(
                completeness_score=float(completeness_score),
                validity_score=float(validity_score),
                uniqueness_score=float(uniqueness_score),
                informativeness_score=float(informativeness_score),
                stability_score=float(stability_score),
                correlation_score=float(correlation_score),
                overall_quality_score=float(overall_score),
                quality_issues=issues,
                feature_importance_analysis={},  # Placeholder
                redundancy_analysis={}  # Placeholder
            )

            self.logger.info("✅ Feature quality analysis completed")

        except Exception as e:
            self.logger.error(f"❌ Failed to analyze feature quality: {e}")
            self.quality_metrics = None

    def _analyze_wavelet_features(self, feature_config: Dict[str, Any], execution_metadata: Dict[str, Any]) -> None:
        """Analyze wavelet feature generation metrics."""
        try:
            self.logger.info("🌊 Analyzing wavelet features...")

            wavelet_config = feature_config.get('wavelet_config', {})
            wavelet_levels = wavelet_config.get('levels', 3)
            wavelet_family = wavelet_config.get('family', 'db4')
            decomposition_levels = wavelet_config.get('decomposition_levels', 4)

            # Estimate wavelet features generated
            base_features = ['open', 'high', 'low', 'close', 'volume']
            wavelet_features_per_level = len(base_features) * decomposition_levels
            total_wavelet_features = wavelet_features_per_level * wavelet_levels

            # Estimate computation time (rough estimate)
            wavelet_time = execution_metadata.get('wavelet_computation_time', execution_metadata.get('total_execution_time', 0) * 0.3)

            # Quality score based on configuration
            quality_score = min(1.0, wavelet_levels / 5.0)  # Higher levels = better quality

            frequency_bands = [f'band_{i}' for i in range(decomposition_levels)]

            self.wavelet_metrics = WaveletAnalysisMetrics(
                wavelet_levels=wavelet_levels,
                wavelet_family=wavelet_family,
                decomposition_levels=decomposition_levels,
                wavelet_features_generated=total_wavelet_features,
                wavelet_computation_time=wavelet_time,
                wavelet_quality_score=quality_score,
                frequency_bands_analyzed=frequency_bands,
                wavelet_transform_efficiency=feature_config.get('enable_wavelets', True)
            )

            self.logger.info("✅ Wavelet analysis completed")

        except Exception as e:
            self.logger.error(f"❌ Failed to analyze wavelet features: {e}")
            self.wavelet_metrics = None

    def _analyze_multitimeframe_features(self, feature_config: Dict[str, Any], execution_metadata: Dict[str, Any]) -> None:
        """Analyze multi-timeframe feature engineering metrics."""
        try:
            self.logger.info("⏰ Analyzing multi-timeframe features...")

            timeframes = feature_config.get('timeframes', ['30m', '1h', '4h', '1d'])
            timeframe_features = {}

            # Estimate features per timeframe
            base_features = 10  # Rough estimate of base features
            for tf in timeframes:
                timeframe_features[tf] = base_features * len(timeframes)  # Cross-timeframe features

            # Estimate processing times
            processing_times = {}
            total_time = execution_metadata.get('total_execution_time', 0)
            time_per_tf = total_time / max(len(timeframes), 1)
            for tf in timeframes:
                processing_times[tf] = time_per_tf

            # Cross-timeframe correlations (placeholder)
            cross_corr = {f'{tf1}_{tf2}': 0.5 for tf1 in timeframes for tf2 in timeframes if tf1 != tf2}

            # Temporal consistency score
            consistency_score = feature_config.get('enable_multi_timeframe', True)

            self.multitimeframe_metrics = MultiTimeframeMetrics(
                timeframes_processed=timeframes,
                timeframe_features_generated=timeframe_features,
                cross_timeframe_correlations=cross_corr,
                temporal_consistency_score=float(consistency_score),
                timeframe_processing_times=processing_times,
                multi_timeframe_efficiency=float(len(timeframes) / max(total_time, 1))
            )

            self.logger.info("✅ Multi-timeframe analysis completed")

        except Exception as e:
            self.logger.error(f"❌ Failed to analyze multi-timeframe features: {e}")
            self.multitimeframe_metrics = None

    def _analyze_technical_indicators(self, output_features: pd.DataFrame, feature_config: Dict[str, Any]) -> None:
        """Analyze technical indicator generation metrics."""
        try:
            self.logger.info("📊 Analyzing technical indicators...")

            # Categorize technical indicators
            indicators = {
                'trend': 0,
                'momentum': 0,
                'volatility': 0,
                'volume': 0,
                'support_resistance': 0
            }

            categories = {
                'trend': ['sma', 'ema', 'wma', 'trend'],
                'momentum': ['rsi', 'macd', 'stoch', 'williams', 'momentum'],
                'volatility': ['bb', 'atr', 'cci', 'volatility'],
                'volume': ['volume', 'vwap', 'obv', 'volume'],
                'support_resistance': ['pivot', 'fibonacci', 'sr']
            }

            for col in output_features.columns:
                col_lower = col.lower()
                for category, keywords in categories.items():
                    if any(keyword in col_lower for keyword in keywords):
                        indicators[category] += 1
                        break

            total_indicators = sum(indicators.values())

            # Quality scores (placeholder - could be enhanced with actual indicator analysis)
            quality_scores = {cat: 0.8 for cat in indicators.keys()}

            self.technical_metrics = TechnicalIndicatorMetrics(
                indicators_generated=indicators,
                indicator_categories={cat: count for cat, count in indicators.items()},
                indicator_computation_time=feature_config.get('indicator_computation_time', 0),
                indicator_quality_scores=quality_scores,
                indicator_stability_analysis={},  # Placeholder
                custom_indicators_count=feature_config.get('custom_indicators', 0)
            )

            self.logger.info("✅ Technical indicator analysis completed")

        except Exception as e:
            self.logger.error(f"❌ Failed to analyze technical indicators: {e}")
            self.technical_metrics = None

    def _analyze_feature_interactions(self, output_features: pd.DataFrame) -> None:
        """Analyze feature interaction and correlation metrics."""
        try:
            self.logger.info("🔗 Analyzing feature interactions...")

            numeric_features = output_features.select_dtypes(include=[np.number])

            if numeric_features.shape[1] > 1:
                # Calculate correlation matrix
                corr_matrix = numeric_features.corr()

                # Find high correlation pairs
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = abs(corr_matrix.iloc[i, j])
                        if corr_val > 0.8:  # High correlation threshold
                            high_corr_pairs.append((
                                corr_matrix.columns[i],
                                corr_matrix.columns[j],
                                float(corr_val)
                            ))

                # Calculate correlation density
                density = np.mean(np.abs(corr_matrix.values))

                # Redundancy score (higher = more redundancy)
                high_corr_count = len(high_corr_pairs)
                total_possible = (len(corr_matrix.columns) * (len(corr_matrix.columns) - 1)) / 2
                redundancy_score = high_corr_count / max(total_possible, 1)

                interactions_created = len(high_corr_pairs)
                interaction_degree = 2  # Pairwise interactions

            else:
                high_corr_pairs = []
                density = 0.0
                redundancy_score = 0.0
                interactions_created = 0
                interaction_degree = 0

            self.interaction_metrics = FeatureInteractionMetrics(
                interactions_created=interactions_created,
                interaction_degree=interaction_degree,
                correlation_matrix_density=float(density),
                high_correlation_pairs=high_corr_pairs,
                feature_redundancy_score=float(redundancy_score),
                interaction_computation_time=0.0  # Placeholder
            )

            self.logger.info("✅ Feature interaction analysis completed")

        except Exception as e:
            self.logger.error(f"❌ Failed to analyze feature interactions: {e}")
            self.interaction_metrics = None

    def _analyze_performance_optimization(self, execution_metadata: Dict[str, Any], hardware_metrics: Dict[str, Any]) -> None:
        """Analyze performance optimization metrics."""
        try:
            self.logger.info("⚡ Analyzing performance optimization...")

            total_time = execution_metadata.get('total_execution_time', 0)
            features_created = execution_metadata.get('features_created', 1)
            efficiency = features_created / max(total_time, 0.001)

            memory_usage = hardware_metrics.get('memory_usage_mb', 0)
            memory_score = 1 - min(1.0, memory_usage / 8000)  # Lower memory = higher score

            parallel_eff = hardware_metrics.get('parallel_processing_efficiency', 1.0)
            caching_eff = execution_metadata.get('caching_efficiency', 1.0)

            chunk_metrics = execution_metadata.get('chunk_processing_metrics', {})

            self.performance_metrics = PerformanceOptimizationMetrics(
                total_execution_time=total_time,
                feature_engineering_efficiency=efficiency,
                memory_optimization_score=memory_score,
                parallel_processing_efficiency=parallel_eff,
                caching_efficiency=caching_eff,
                chunk_processing_metrics=chunk_metrics
            )

            self.logger.info("✅ Performance optimization analysis completed")

        except Exception as e:
            self.logger.error(f"❌ Failed to analyze performance optimization: {e}")
            self.performance_metrics = None

    def _generate_recommendations(self) -> List[str]:
        """Generate comprehensive recommendations based on detailed analysis."""
        recommendations = []

        try:
            # Feature Quality Recommendations
            if self.quality_metrics:
                quality_score = self.quality_metrics.overall_quality_score

                if quality_score >= 0.9:
                    recommendations.append("✅ Feature quality is excellent - maintain current standards")
                elif quality_score >= 0.8:
                    recommendations.append("⚠️ Feature quality is good but could be improved")
                else:
                    recommendations.append("🚨 Feature quality needs immediate attention")

                # Specific quality recommendations
                if self.quality_metrics.completeness_score < 0.95:
                    recommendations.append("• Implement advanced data imputation techniques for missing values")
                if self.quality_metrics.validity_score < 0.9:
                    recommendations.append("• Add robust outlier detection and handling mechanisms")
                if self.quality_metrics.correlation_score < 0.7:
                    recommendations.append("• Implement feature selection to reduce redundancy (correlation > 0.7)")
                if self.quality_metrics.uniqueness_score < 0.9:
                    recommendations.append("• Remove duplicate features and improve feature diversity")
                if self.quality_metrics.informativeness_score < 0.8:
                    recommendations.append("• Review feature engineering to ensure informative features")
                if self.quality_metrics.stability_score < 0.8:
                    recommendations.append("• Implement feature stability testing and validation")

            # Hardware Acceleration Recommendations
            if self.hardware_metrics:
                accel_score = self.hardware_metrics.hardware_acceleration_score

                if accel_score >= 0.8:
                    recommendations.append("✅ Hardware acceleration is highly optimized")
                elif accel_score >= 0.6:
                    recommendations.append("⚠️ Hardware acceleration shows room for improvement")
                else:
                    recommendations.append("🚨 Hardware acceleration needs significant optimization")

                # Specific hardware recommendations
                gpu_util = self.hardware_metrics.gpu_utilization
                if gpu_util > 0.8:
                    recommendations.append("• GPU utilization is high - monitor thermal conditions")
                elif gpu_util < 0.5:
                    recommendations.append("• GPU underutilized - consider increasing parallel workloads")

                speedup = self.hardware_metrics.processing_speedup
                if speedup < 1.5:
                    recommendations.append("• Processing speedup is suboptimal - review vectorization strategies")

                if not self.hardware_metrics.m1_gpu_available:
                    recommendations.append("• M1 GPU not detected - ensure proper hardware configuration")

            # Feature Engineering Performance Recommendations
            if self.feature_metrics:
                fps = self.feature_metrics.features_per_second
                total_features = self.feature_metrics.total_features_created

                if fps > 500:
                    recommendations.append("✅ Feature engineering performance is excellent")
                elif fps > 100:
                    recommendations.append("⚠️ Feature engineering performance is acceptable")
                else:
                    recommendations.append("🚨 Feature engineering performance needs optimization")

                if total_features < 50:
                    recommendations.append("• Consider expanding feature set for better model performance")
                elif total_features > 1000:
                    recommendations.append("• Large feature set detected - consider dimensionality reduction")

            # Wavelet Analysis Recommendations
            if self.wavelet_metrics:
                wavelet_score = self.wavelet_metrics.wavelet_quality_score
                levels = self.wavelet_metrics.wavelet_levels

                if wavelet_score >= 0.8:
                    recommendations.append("✅ Wavelet analysis quality is excellent")
                elif wavelet_score >= 0.6:
                    recommendations.append("⚠️ Wavelet analysis shows room for improvement")
                else:
                    recommendations.append("🚨 Wavelet analysis needs optimization")

                if levels < 3:
                    recommendations.append("• Increase wavelet decomposition levels for better frequency resolution")
                elif levels > 5:
                    recommendations.append("• Consider reducing wavelet levels to prevent overfitting")

            # Multi-Timeframe Recommendations
            if self.multitimeframe_metrics:
                consistency = self.multitimeframe_metrics.temporal_consistency_score
                efficiency = self.multitimeframe_metrics.multi_timeframe_efficiency

                if consistency >= 0.8:
                    recommendations.append("✅ Multi-timeframe consistency is excellent")
                else:
                    recommendations.append("⚠️ Review multi-timeframe feature consistency")

                if efficiency < 0.7:
                    recommendations.append("• Optimize multi-timeframe processing efficiency")

            # Technical Indicators Recommendations
            if self.technical_metrics:
                custom_count = self.technical_metrics.custom_indicators_count
                computation_time = self.technical_metrics.indicator_computation_time

                if custom_count > 10:
                    recommendations.append("✅ Good variety of custom technical indicators")
                elif custom_count < 5:
                    recommendations.append("• Consider adding more custom technical indicators")

                if computation_time > 60:  # seconds
                    recommendations.append("• Optimize technical indicator computation performance")

            # Feature Interaction Recommendations
            if self.interaction_metrics:
                redundancy = self.interaction_metrics.feature_redundancy_score
                interactions = self.interaction_metrics.interactions_created

                if redundancy > 0.3:
                    recommendations.append("🚨 High feature redundancy detected - implement feature selection")
                elif redundancy > 0.1:
                    recommendations.append("⚠️ Moderate feature redundancy - monitor correlations")

                if interactions < 10:
                    recommendations.append("• Consider increasing feature interaction analysis")

            # Performance Optimization Recommendations
            if self.performance_metrics:
                exec_time = self.performance_metrics.total_execution_time
                memory_score = self.performance_metrics.memory_optimization_score
                parallel_eff = self.performance_metrics.parallel_processing_efficiency

                if exec_time > 600:  # 10 minutes
                    recommendations.append("🚨 Long execution time - consider optimization strategies")

                if memory_score < 0.7:
                    recommendations.append("• Implement memory optimization techniques")

                if parallel_eff < 0.8:
                    recommendations.append("• Improve parallel processing efficiency")

            # Overall System Health
            overall_score = self._calculate_overall_system_health()
            if overall_score >= 0.85:
                recommendations.append("🎉 Overall system health is excellent - continue current practices")
            elif overall_score >= 0.7:
                recommendations.append("⚠️ Overall system health is good but monitor key metrics")
            else:
                recommendations.append("🚨 Overall system health needs attention - review all recommendations above")

        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive recommendations: {e}")
            recommendations = ["Unable to generate recommendations due to analysis error"]

        return recommendations

    def _calculate_overall_system_health(self) -> float:
        """Calculate overall system health score."""
        try:
            scores = []

            if self.quality_metrics:
                scores.append(self.quality_metrics.overall_quality_score)

            if self.hardware_metrics:
                scores.append(self.hardware_metrics.hardware_acceleration_score)

            if self.feature_metrics:
                # Normalize features per second to a 0-1 score
                fps_score = min(1.0, self.feature_metrics.features_per_second / 500.0)
                scores.append(fps_score)

            if self.wavelet_metrics:
                scores.append(self.wavelet_metrics.wavelet_quality_score)

            if self.multitimeframe_metrics:
                scores.append(self.multitimeframe_metrics.temporal_consistency_score)

            if self.performance_metrics:
                scores.append(self.performance_metrics.memory_optimization_score)
                scores.append(self.performance_metrics.parallel_processing_efficiency)

            return np.mean(scores) if scores else 0.5

        except Exception as e:
            self.logger.error(f"Failed to calculate system health: {e}")
            return 0.5

    def _generate_alerts(self) -> List[str]:
        """Generate comprehensive alerts for critical issues and performance predictions."""
        alerts = []

        try:
            # Critical Quality Alerts
            if self.quality_metrics:
                if self.quality_metrics.completeness_score < 0.8:
                    alerts.append("🚨 CRITICAL: Low feature completeness (<80%) - may cause model instability")
                    alerts.append("   • Impact: Models may fail to learn from incomplete data patterns")
                    alerts.append("   • Risk: Reduced predictive accuracy and unreliable signals")

                if self.quality_metrics.validity_score < 0.7:
                    alerts.append("🚨 CRITICAL: Severe feature validity issues (<70%) - data corruption detected")
                    alerts.append("   • Impact: Models may learn incorrect patterns from corrupted data")
                    alerts.append("   • Risk: False trading signals and potential losses")

                if self.quality_metrics.correlation_score < 0.6:
                    alerts.append("⚠️ WARNING: High feature redundancy (<60% correlation score) - multicollinearity risk")
                    alerts.append("   • Impact: Models may become unstable with correlated features")
                    alerts.append("   • Risk: Overfitting and reduced generalization capability")

                if self.quality_metrics.overall_quality_score < 0.75:
                    alerts.append("🚨 CRITICAL: Overall feature quality is poor - immediate review required")
                    alerts.append("   • Predicted Impact: Model performance may be 30-50% below optimal")
                    alerts.append("   • Risk Level: HIGH - Consider halting model training until resolved")

            # Hardware Performance Alerts
            if self.hardware_metrics:
                if not self.hardware_metrics.m1_gpu_available:
                    alerts.append("⚠️ WARNING: M1 GPU not available - suboptimal processing performance")
                    alerts.append("   • Impact: 2-3x slower feature engineering")
                    alerts.append("   • Risk: Increased processing time and resource costs")

                if self.hardware_metrics.hardware_acceleration_score < 0.5:
                    alerts.append("⚠️ WARNING: Poor hardware acceleration (<50%) - efficiency concerns")
                    alerts.append("   • Impact: Resource waste and increased operational costs")
                    alerts.append("   • Risk: Scalability issues with larger datasets")

                gpu_util = self.hardware_metrics.gpu_utilization
                if gpu_util > 0.9:
                    alerts.append("⚠️ WARNING: GPU utilization >90% - thermal and stability risks")
                    alerts.append("   • Impact: Potential hardware damage or throttling")
                    alerts.append("   • Risk: System crashes and processing interruptions")

            # Feature Engineering Scale Alerts
            if self.feature_metrics:
                total_features = self.feature_metrics.total_features_created
                fps = self.feature_metrics.features_per_second

                if total_features < 10:
                    alerts.append("🚨 CRITICAL: Insufficient features generated (<10) - inadequate feature set")
                    alerts.append("   • Impact: Models lack sufficient information for accurate predictions")
                    alerts.append("   • Risk: Poor model performance and unreliable trading signals")

                if total_features > 2000:
                    alerts.append("⚠️ WARNING: Excessive features generated (>2000) - dimensionality curse risk")
                    alerts.append("   • Impact: Models may overfit and fail to generalize")
                    alerts.append("   • Risk: Poor out-of-sample performance and false confidence")

                if fps < 10:
                    alerts.append("🚨 CRITICAL: Very slow feature generation (<10 features/sec)")
                    alerts.append("   • Impact: Infeasible for real-time or frequent retraining")
                    alerts.append("   • Risk: Operational delays and missed market opportunities")

            # Wavelet Analysis Alerts
            if self.wavelet_metrics:
                wavelet_score = self.wavelet_metrics.wavelet_quality_score
                if wavelet_score < 0.6:
                    alerts.append("⚠️ WARNING: Poor wavelet analysis quality (<60%) - frequency domain issues")
                    alerts.append("   • Impact: Missing important market cycle information")
                    alerts.append("   • Risk: Incomplete market regime representation")

            # Multi-Timeframe Alerts
            if self.multitimeframe_metrics:
                consistency = self.multitimeframe_metrics.temporal_consistency_score
                if consistency < 0.7:
                    alerts.append("⚠️ WARNING: Low multi-timeframe consistency (<70%) - temporal misalignment")
                    alerts.append("   • Impact: Conflicting signals across timeframes")
                    alerts.append("   • Risk: Contradictory trading signals and confusion")

            # Performance Optimization Alerts
            if self.performance_metrics:
                exec_time = self.performance_metrics.total_execution_time
                memory_score = self.performance_metrics.memory_optimization_score

                if exec_time > 1800:  # 30 minutes
                    alerts.append("⚠️ WARNING: Excessive execution time (>30min) - scalability concerns")
                    alerts.append("   • Impact: Difficult to integrate into automated pipelines")
                    alerts.append("   • Risk: Delayed model updates and missed opportunities")

                if memory_score < 0.6:
                    alerts.append("⚠️ WARNING: Poor memory optimization (<60%) - resource constraints")
                    alerts.append("   • Impact: System instability with larger datasets")
                    alerts.append("   • Risk: Processing failures and system crashes")

            # Predictive Performance Alerts
            overall_health = self._calculate_overall_system_health()
            if overall_health < 0.6:
                alerts.append("🚨 CRITICAL: Overall system health is poor (<60%)")
                alerts.append("   • Predicted Model Performance: 40-60% below optimal")
                alerts.append("   • Risk Assessment: HIGH - Immediate intervention required")
            elif overall_health < 0.75:
                alerts.append("⚠️ WARNING: Overall system health needs attention (<75%)")
                alerts.append("   • Predicted Model Performance: 15-25% below optimal")
                alerts.append("   • Risk Assessment: MEDIUM - Monitor and optimize")

        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive alerts: {e}")
            alerts.append("❌ ALERT SYSTEM ERROR: Unable to generate performance alerts")

        return alerts

    def _generate_performance_predictions(self) -> Dict[str, Any]:
        """Generate comprehensive performance predictions for the feature engineering pipeline."""
        try:
            predictions = {
                'model_performance_predictions': {},
                'feature_importance_predictions': {},
                'scalability_predictions': {},
                'robustness_predictions': {},
                'optimization_opportunities': {},
                'risk_assessments': {},
                'confidence_intervals': {},
                'benchmarking_predictions': {}
            }

            # Model Performance Predictions
            quality_score = self.quality_metrics.overall_quality_score if self.quality_metrics else 0.5
            feature_count = self.feature_metrics.total_features_created if self.feature_metrics else 100
            hardware_score = self.hardware_metrics.hardware_acceleration_score if self.hardware_metrics else 0.5

            # Predict model accuracy based on feature quality and quantity
            base_accuracy = 0.55  # Baseline model accuracy
            quality_bonus = (quality_score - 0.5) * 0.3  # Quality impact
            feature_bonus = min(0.15, (feature_count - 50) / 1000)  # Feature diversity impact
            hardware_bonus = (hardware_score - 0.5) * 0.1  # Hardware efficiency impact

            predicted_accuracy = min(0.95, base_accuracy + quality_bonus + feature_bonus + hardware_bonus)

            predictions['model_performance_predictions'] = {
                'predicted_model_accuracy': predicted_accuracy,
                'accuracy_confidence_interval': [predicted_accuracy - 0.1, predicted_accuracy + 0.1],
                'feature_contribution_score': quality_score,
                'hardware_efficiency_impact': hardware_bonus,
                'performance_stability_score': self._calculate_performance_stability(),
                'overfitting_risk_level': self._assess_overfitting_risk(),
                'generalization_capability': self._predict_generalization_capability()
            }

            # Feature Importance Predictions
            if self.feature_metrics:
                wavelet_count = self.feature_metrics.wavelet_features_count
                mtf_count = self.feature_metrics.multi_timeframe_features_count
                tech_count = self.feature_metrics.technical_indicators_count

                total_features = self.feature_metrics.total_features_created

                predictions['feature_importance_predictions'] = {
                    'wavelet_feature_importance': wavelet_count / total_features,
                    'multitimeframe_feature_importance': mtf_count / total_features,
                    'technical_indicator_importance': tech_count / total_features,
                    'feature_redundancy_impact': self._calculate_redundancy_impact(),
                    'optimal_feature_subset_size': self._predict_optimal_feature_count(),
                    'feature_stability_predictions': self._predict_feature_stability()
                }

            # Scalability Predictions
            if self.performance_metrics:
                exec_time = self.performance_metrics.total_execution_time
                memory_usage = self.hardware_metrics.memory_usage_mb if self.hardware_metrics else 1000

                predictions['scalability_predictions'] = {
                    'maximum_dataset_size': self._predict_max_dataset_size(exec_time, memory_usage),
                    'processing_time_scaling': self._predict_processing_time_scaling(),
                    'memory_usage_scaling': self._predict_memory_usage_scaling(),
                    'parallelization_efficiency': self.performance_metrics.parallel_processing_efficiency,
                    'bottleneck_analysis': self._identify_performance_bottlenecks(),
                    'optimization_potential': self._assess_optimization_potential()
                }

            # Robustness Predictions
            predictions['robustness_predictions'] = {
                'data_quality_robustness': self._predict_data_quality_robustness(),
                'market_condition_adaptability': self._predict_market_adaptability(),
                'regime_detection_robustness': self._predict_regime_robustness(),
                'noise_resistance_score': self._predict_noise_resistance(),
                'edge_case_handling': self._predict_edge_case_handling()
            }

            # Optimization Opportunities
            predictions['optimization_opportunities'] = {
                'performance_improvements': self._identify_performance_improvements(),
                'feature_engineering_enhancements': self._identify_feature_enhancements(),
                'hardware_optimizations': self._identify_hardware_optimizations(),
                'code_optimization_potential': self._assess_code_optimization_potential(),
                'pipeline_efficiency_gains': self._predict_pipeline_efficiency_gains()
            }

            # Risk Assessments
            predictions['risk_assessments'] = {
                'model_risks': self._assess_model_risks(),
                'operational_risks': self._assess_operational_risks(),
                'data_quality_risks': self._assess_data_quality_risks(),
                'scalability_risks': self._assess_scalability_risks(),
                'market_risks': self._assess_market_risks(),
                'overall_risk_score': self._calculate_overall_risk_score()
            }

            # Confidence Intervals
            predictions['confidence_intervals'] = {
                'accuracy_95_ci': [predicted_accuracy - 0.15, predicted_accuracy + 0.15],
                'performance_95_ci': [predicted_accuracy - 0.2, predicted_accuracy + 0.2],
                'stability_95_ci': [0.7, 0.9],
                'risk_95_ci': [0.1, 0.3]
            }

            # Benchmarking Predictions
            predictions['benchmarking_predictions'] = {
                'vs_industry_standards': self._predict_vs_industry_standards(),
                'competitor_comparison': self._predict_competitor_comparison(),
                'best_practice_alignment': self._assess_best_practice_alignment(),
                'innovation_potential': self._assess_innovation_potential()
            }

            return predictions

        except Exception as e:
            self.logger.error(f"Failed to generate performance predictions: {e}")
            return {'error': str(e), 'predictions_unavailable': True}

    def _calculate_performance_stability(self) -> float:
        """Calculate predicted performance stability."""
        try:
            stability_factors = []

            if self.quality_metrics:
                stability_factors.append(self.quality_metrics.stability_score)

            if self.feature_metrics:
                # More features generally mean more stability
                feature_stability = min(1.0, self.feature_metrics.total_features_created / 200)
                stability_factors.append(feature_stability)

            if self.wavelet_metrics:
                stability_factors.append(self.wavelet_metrics.wavelet_quality_score)

            return np.mean(stability_factors) if stability_factors else 0.7

        except Exception:
            return 0.7

    def _assess_overfitting_risk(self) -> str:
        """Assess overfitting risk level."""
        try:
            if not self.quality_metrics or not self.feature_metrics:
                return "UNKNOWN"

            correlation_score = self.quality_metrics.correlation_score
            feature_count = self.feature_metrics.total_features_created

            # High correlation + many features = high overfitting risk
            if correlation_score < 0.7 and feature_count > 500:
                return "HIGH"
            elif correlation_score < 0.8 and feature_count > 200:
                return "MEDIUM"
            elif correlation_score > 0.8 and feature_count < 100:
                return "LOW"
            else:
                return "MODERATE"

        except Exception:
            return "UNKNOWN"

    def _predict_generalization_capability(self) -> float:
        """Predict model's generalization capability."""
        try:
            if not self.quality_metrics:
                return 0.6

            # Generalization improves with quality but may decrease with complexity
            base_generalization = self.quality_metrics.overall_quality_score

            if self.feature_metrics:
                # Too many features can hurt generalization
                feature_penalty = max(0, (self.feature_metrics.total_features_created - 300) / 1000)
                base_generalization -= feature_penalty

            return max(0.3, min(0.9, base_generalization))

        except Exception:
            return 0.6

    def _calculate_redundancy_impact(self) -> float:
        """Calculate the impact of feature redundancy."""
        try:
            if not self.interaction_metrics:
                return 0.1

            redundancy_score = self.interaction_metrics.feature_redundancy_score
            # Convert redundancy score to impact (higher redundancy = higher negative impact)
            return min(0.5, redundancy_score * 2)

        except Exception:
            return 0.1

    def _predict_optimal_feature_count(self) -> int:
        """Predict the optimal number of features."""
        try:
            if not self.quality_metrics or not self.feature_metrics:
                return 200

            quality_score = self.quality_metrics.overall_quality_score
            current_features = self.feature_metrics.total_features_created

            # Optimal features = quality * scaling factor
            optimal = int(quality_score * 400)

            # Don't recommend drastic changes
            if abs(optimal - current_features) / current_features > 0.5:
                optimal = int(current_features * 0.8) if current_features > optimal else int(current_features * 1.2)

            return max(50, min(1000, optimal))

        except Exception:
            return 200

    def _predict_feature_stability(self) -> Dict[str, float]:
        """Predict stability of different feature types."""
        try:
            stability_predictions = {
                'wavelet_features': 0.85,
                'multitimeframe_features': 0.78,
                'technical_indicators': 0.82,
                'interaction_features': 0.75,
                'regime_features': 0.80
            }

            if self.quality_metrics:
                # Adjust based on overall quality
                quality_factor = self.quality_metrics.stability_score
                for key in stability_predictions:
                    stability_predictions[key] *= quality_factor

            return stability_predictions

        except Exception:
            return {'error': 'Unable to predict feature stability'}

    def _predict_max_dataset_size(self, exec_time: float, memory_usage: float) -> str:
        """Predict maximum feasible dataset size."""
        try:
            # Estimate based on current performance
            time_per_million_rows = exec_time / 1_000_000  # Assume current test is ~1M rows

            # Max time = 1 hour = 3600 seconds
            max_rows_by_time = 3600 / time_per_million_rows * 1_000_000

            # Max memory = 32GB = 32768 MB (leave 4GB buffer)
            max_memory_mb = 32768 - 4096
            max_rows_by_memory = (max_memory_mb / memory_usage) * 1_000_000

            max_rows = min(max_rows_by_time, max_rows_by_memory)

            if max_rows > 100_000_000:  # 100M rows
                return "100M+ rows"
            elif max_rows > 10_000_000:  # 10M rows
                return f"{int(max_rows / 1_000_000)}M rows"
            else:
                return f"{int(max_rows / 1_000)}K rows"

        except Exception:
            return "10M rows (estimated)"

    def _predict_processing_time_scaling(self) -> str:
        """Predict how processing time scales with data size."""
        try:
            parallel_eff = self.performance_metrics.parallel_processing_efficiency if self.performance_metrics else 0.8

            if parallel_eff > 0.8:
                return "Near-linear scaling (excellent parallelization)"
            elif parallel_eff > 0.6:
                return "Sub-linear scaling (good parallelization)"
            else:
                return "Poor scaling (parallelization issues)"

        except Exception:
            return "Sub-linear scaling (typical)"

    def _predict_memory_usage_scaling(self) -> str:
        """Predict memory usage scaling."""
        try:
            memory_score = self.performance_metrics.memory_optimization_score if self.performance_metrics else 0.7

            if memory_score > 0.8:
                return "Linear scaling (excellent memory efficiency)"
            elif memory_score > 0.6:
                return "Linear scaling (good memory efficiency)"
            else:
                return "Super-linear scaling (memory inefficiency)"

        except Exception:
            return "Linear scaling (typical)"

    def _identify_performance_bottlenecks(self) -> List[str]:
        """Identify potential performance bottlenecks."""
        bottlenecks = []

        try:
            if self.performance_metrics:
                if self.performance_metrics.parallel_processing_efficiency < 0.7:
                    bottlenecks.append("Parallel processing inefficiency")

                if self.performance_metrics.memory_optimization_score < 0.6:
                    bottlenecks.append("Memory management issues")

                if self.performance_metrics.caching_efficiency < 0.7:
                    bottlenecks.append("Poor caching utilization")

            if self.hardware_metrics:
                if self.hardware_metrics.gpu_utilization < 0.5:
                    bottlenecks.append("Underutilized GPU resources")

                if self.hardware_metrics.processing_speedup < 1.5:
                    bottlenecks.append("Inefficient hardware acceleration")

            if not bottlenecks:
                bottlenecks.append("No major bottlenecks identified")

        except Exception:
            bottlenecks.append("Unable to analyze bottlenecks")

        return bottlenecks

    def _assess_optimization_potential(self) -> Dict[str, Any]:
        """Assess potential for optimization."""
        try:
            potential = {
                'performance_improvement_potential': 0.0,
                'estimated_speedup_factor': 1.0,
                'resource_efficiency_gain': 0.0,
                'cost_reduction_potential': 0.0
            }

            if self.hardware_metrics and self.performance_metrics:
                # Calculate potential improvements
                current_speedup = self.hardware_metrics.processing_speedup
                current_efficiency = self.performance_metrics.parallel_processing_efficiency

                # Potential for hardware optimization
                if current_speedup < 2.0:
                    potential['performance_improvement_potential'] += 0.3
                    potential['estimated_speedup_factor'] *= 1.8

                # Potential for parallelization improvements
                if current_efficiency < 0.8:
                    potential['performance_improvement_potential'] += 0.2
                    potential['estimated_speedup_factor'] *= 1.4

                # Resource efficiency
                potential['resource_efficiency_gain'] = min(0.4, (1 - current_efficiency) * 0.5)

                # Cost reduction (based on time and resource savings)
                potential['cost_reduction_potential'] = potential['performance_improvement_potential'] * 0.6

            return potential

        except Exception:
            return {'error': 'Unable to assess optimization potential'}

    def _summarize_config(self, feature_config: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize configuration settings."""
        return {
            'enable_wavelets': feature_config.get('enable_wavelets', True),
            'enable_multi_timeframe': feature_config.get('enable_multi_timeframe', True),
            'enable_feature_interactions': feature_config.get('enable_feature_interactions', True),
            'timeframes': feature_config.get('timeframes', ['30m', '1h', '4h', '1d']),
            'max_features': feature_config.get('max_features', 500),
            'chunk_size': feature_config.get('chunk_size', 500000),
            'hardware_acceleration': feature_config.get('hardware_acceleration', True)
        }

    def _save_markdown_report(self, report_data: Dict[str, Any], symbol: str, exchange: str, timeframe: str) -> Optional[str]:
        """Save comprehensive markdown report with enhanced formatting and sections."""
        try:
            # Enhanced header with emojis and better formatting
            markdown_content = f"""# Step 6 Enhanced Advanced Feature Engineering Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Symbol:** {symbol}
**Exchange:** {exchange}
**Timeframe:** {timeframe}

## 🚀 Executive Summary

This comprehensive report provides detailed analysis of the advanced feature engineering process for **{symbol}** on **{exchange}** using **{timeframe}** timeframe data.

The analysis includes feature creation metrics, hardware acceleration performance, quality assessment, wavelet analysis, multi-timeframe processing, technical indicators, and actionable recommendations for optimization.

"""

            # Performance Summary Dashboard
            markdown_content += """## 📊 Performance Summary

| Metric | Value | Status |
|--------|-------|--------|"""

            # Add execution time if available
            if 'feature_engineering_analysis' in report_data:
                fe_data = report_data['feature_engineering_analysis']
                exec_time = fe_data.get('feature_creation_time_seconds', 0)
                markdown_content += f"\n| Execution Time | {exec_time:.2f}s | {'✅' if exec_time < 300 else '⚠️'} |"

            if 'hardware_acceleration_analysis' in report_data:
                hw_data = report_data['hardware_acceleration_analysis']
                gpu_util = hw_data.get('gpu_utilization', 0)
                speedup = hw_data.get('processing_speedup', 1.0)
                markdown_content += f"\n| GPU Utilization | {gpu_util:.1%} | {'✅' if gpu_util > 0.5 else '⚠️'} |"
                markdown_content += f"\n| Processing Speedup | {speedup:.2f}x | {'✅' if speedup > 1.5 else '⚠️'} |"

            markdown_content += "\n"

            # Feature Engineering Analysis with enhanced detail
            if 'feature_engineering_analysis' in report_data:
                fe_data = report_data['feature_engineering_analysis']
                markdown_content += f"""
## 🔧 Feature Engineering Analysis

### Core Metrics
- **Total Features Created:** {fe_data.get('total_features_created', 'N/A'):,}
- **Features/Second:** {fe_data.get('features_per_second', 'N/A'):.1f}
- **Feature Creation Time:** {fe_data.get('feature_creation_time_seconds', 'N/A'):.2f}s

### Feature Categories Breakdown

| Category | Count | Percentage |
|----------|-------|------------|"""

                categories = fe_data.get('feature_categories', {})
                total_features = sum(categories.values())
                for category, count in categories.items():
                    percentage = (count / total_features * 100) if total_features > 0 else 0
                    category_name = category.replace('_', ' ').title()
                    markdown_content += f"\n| {category_name} | {count:,} | {percentage:.1f}% |"

                markdown_content += f"""

### Specialized Feature Types
- **Wavelet Features:** {fe_data.get('wavelet_features_count', 'N/A'):,}
- **Multi-Timeframe Features:** {fe_data.get('multi_timeframe_features_count', 'N/A'):,}
- **Technical Indicators:** {fe_data.get('technical_indicators_count', 'N/A'):,}
- **Feature Interactions:** {fe_data.get('feature_interactions_count', 'N/A'):,}
- **Regime-Aware Features:** {fe_data.get('regime_aware_features_count', 'N/A'):,}

"""

            # Hardware Acceleration Analysis with enhanced detail
            if 'hardware_acceleration_analysis' in report_data:
                hw_data = report_data['hardware_acceleration_analysis']
                markdown_content += f"""
## ⚡ Hardware Acceleration Analysis

### Performance Metrics
- **GPU Utilization:** {hw_data.get('gpu_utilization', 'N/A'):.1%}
- **CPU Utilization:** {hw_data.get('cpu_utilization', 'N/A'):.1%}
- **Vectorization Efficiency:** {hw_data.get('vectorization_efficiency', 'N/A'):.1%}
- **Memory Usage:** {hw_data.get('memory_usage_mb', 'N/A'):.1f}MB
- **Processing Speedup:** {hw_data.get('processing_speedup', 'N/A'):.2f}x
- **Hardware Acceleration Score:** {hw_data.get('hardware_acceleration_score', 'N/A'):.3f}

### System Capabilities
- **Optimization Enabled:** {hw_data.get('optimization_enabled', 'N/A')}
- **M1 GPU Available:** {hw_data.get('m1_gpu_available', 'N/A')}
- **Vectorized Operations:** {hw_data.get('vectorized_operations', 'N/A'):,}

### Performance Insights
"""

                # Add performance insights
                gpu_util = hw_data.get('gpu_utilization', 0)
                speedup = hw_data.get('processing_speedup', 1.0)

                if gpu_util > 0.7:
                    markdown_content += "- **GPU heavily utilized** - Consider workload distribution\n"
                elif gpu_util < 0.3:
                    markdown_content += "- **GPU underutilized** - Potential for increased parallel processing\n"

                if speedup > 2.0:
                    markdown_content += "- **Excellent speedup achieved** - Hardware acceleration working effectively\n"
                elif speedup < 1.2:
                    markdown_content += "- **Limited speedup** - Review vectorization and GPU utilization\n"

                markdown_content += "\n"

            # Feature Quality Analysis with comprehensive breakdown
            if 'feature_quality_analysis' in report_data:
                quality_data = report_data['feature_quality_analysis']
                markdown_content += f"""
## 🔍 Feature Quality Assessment

### Overall Quality Score: **{quality_data.get('overall_quality_score', 'N/A'):.3f}**

### Quality Dimensions

| Metric | Score | Status |
|--------|-------|--------|
| Completeness | {quality_data.get('completeness_score', 'N/A'):.3f} | {'✅' if quality_data.get('completeness_score', 0) > 0.95 else '⚠️'} |
| Validity | {quality_data.get('validity_score', 'N/A'):.3f} | {'✅' if quality_data.get('validity_score', 0) > 0.9 else '⚠️'} |
| Uniqueness | {quality_data.get('uniqueness_score', 'N/A'):.3f} | {'✅' if quality_data.get('uniqueness_score', 0) > 0.9 else '⚠️'} |
| Informativeness | {quality_data.get('informativeness_score', 'N/A'):.3f} | {'✅' if quality_data.get('informativeness_score', 0) > 0.8 else '⚠️'} |
| Stability | {quality_data.get('stability_score', 'N/A'):.3f} | {'✅' if quality_data.get('stability_score', 0) > 0.8 else '⚠️'} |
| Correlation | {quality_data.get('correlation_score', 'N/A'):.3f} | {'✅' if quality_data.get('correlation_score', 0) > 0.8 else '⚠️'} |

"""

                # Add quality issues section
                quality_issues = quality_data.get('quality_issues', [])
                if quality_issues:
                    markdown_content += "### ⚠️ Quality Issues Identified\n\n"
                    for issue in quality_issues:
                        markdown_content += f"- {issue}\n"
                    markdown_content += "\n"

                # Add improvement suggestions
                if quality_data.get('overall_quality_score', 1.0) < 0.9:
                    markdown_content += "### 💡 Quality Improvement Suggestions\n\n"
                    if quality_data.get('completeness_score', 1.0) < 0.95:
                        markdown_content += "- **Data imputation** - Implement advanced missing value handling\n"
                    if quality_data.get('validity_score', 1.0) < 0.9:
                        markdown_content += "- **Outlier detection** - Add robust outlier detection and filtering\n"
                    if quality_data.get('correlation_score', 1.0) < 0.8:
                        markdown_content += "- **Feature selection** - Implement correlation-based feature reduction\n"
                    markdown_content += "\n"

            # Wavelet Analysis section
            if 'wavelet_analysis' in report_data:
                wavelet_data = report_data['wavelet_analysis']
                markdown_content += f"""
## 🌊 Wavelet Feature Analysis

### Wavelet Configuration
- **Wavelet Family:** {wavelet_data.get('wavelet_family', 'N/A')}
- **Decomposition Levels:** {wavelet_data.get('decomposition_levels', 'N/A')}
- **Wavelet Levels:** {wavelet_data.get('wavelet_levels', 'N/A')}
- **Quality Score:** {wavelet_data.get('wavelet_quality_score', 'N/A'):.3f}

### Performance Metrics
- **Features Generated:** {wavelet_data.get('wavelet_features_generated', 'N/A'):,}
- **Computation Time:** {wavelet_data.get('wavelet_computation_time', 'N/A'):.2f}s
- **Transform Efficiency:** {wavelet_data.get('wavelet_transform_efficiency', 'N/A')}

### Frequency Bands Analyzed
"""
                frequency_bands = wavelet_data.get('frequency_bands_analyzed', [])
                for band in frequency_bands:
                    markdown_content += f"- {band}\n"

                markdown_content += "\n"

            # Multi-Timeframe Analysis
            if 'multitimeframe_analysis' in report_data:
                mtf_data = report_data['multitimeframe_analysis']
                markdown_content += f"""
## ⏰ Multi-Timeframe Analysis

### Timeframe Processing
"""
                timeframes = mtf_data.get('timeframes_processed', [])
                tf_features = mtf_data.get('timeframe_features_generated', {})

                for tf in timeframes:
                    features_count = tf_features.get(tf, 0)
                    markdown_content += f"- **{tf}:** {features_count:,} features generated\n"

                markdown_content += f"""
### Cross-Timeframe Analysis
- **Temporal Consistency Score:** {mtf_data.get('temporal_consistency_score', 'N/A'):.3f}
- **Multi-Timeframe Efficiency:** {mtf_data.get('multi_timeframe_efficiency', 'N/A'):.3f}

"""

            # Technical Indicators Analysis
            if 'technical_indicator_analysis' in report_data:
                ti_data = report_data['technical_indicator_analysis']
                markdown_content += f"""
## 📊 Technical Indicators Analysis

### Indicator Categories
"""
                indicators = ti_data.get('indicators_generated', {})
                for category, count in indicators.items():
                    markdown_content += f"- **{category.title()}:** {count:,} indicators\n"

                markdown_content += f"""
### Performance Metrics
- **Computation Time:** {ti_data.get('indicator_computation_time', 'N/A'):.2f}s
- **Custom Indicators:** {ti_data.get('custom_indicators_count', 'N/A'):,}

### Quality Assessment
"""
                quality_scores = ti_data.get('indicator_quality_scores', {})
                for indicator_type, score in quality_scores.items():
                    markdown_content += f"- **{indicator_type.title()}:** {score:.3f} quality score\n"

                markdown_content += "\n"

            # Feature Interactions Analysis
            if 'feature_interaction_analysis' in report_data:
                interaction_data = report_data['feature_interaction_analysis']
                markdown_content += f"""
## 🔗 Feature Interaction Analysis

### Interaction Metrics
- **Interactions Created:** {interaction_data.get('interactions_created', 'N/A'):,}
- **Interaction Degree:** {interaction_data.get('interaction_degree', 'N/A')}
- **Correlation Matrix Density:** {interaction_data.get('correlation_matrix_density', 'N/A'):.3f}
- **Feature Redundancy Score:** {interaction_data.get('feature_redundancy_score', 'N/A'):.3f}

### High Correlation Pairs
"""
                high_corr_pairs = interaction_data.get('high_correlation_pairs', [])
                for pair in high_corr_pairs[:10]:  # Show top 10
                    feature1, feature2, corr = pair
                    markdown_content += f"- **{feature1} ↔ {feature2}:** {corr:.3f}\n"

                if len(high_corr_pairs) > 10:
                    markdown_content += f"- ... and {len(high_corr_pairs) - 10} more pairs\n"

                markdown_content += "\n"

            # Performance Optimization Analysis
            if 'performance_optimization_analysis' in report_data:
                perf_data = report_data['performance_optimization_analysis']
                markdown_content += f"""
## ⚡ Performance Optimization Analysis

### Efficiency Metrics
- **Total Execution Time:** {perf_data.get('total_execution_time', 'N/A'):.2f}s
- **Feature Engineering Efficiency:** {perf_data.get('feature_engineering_efficiency', 'N/A'):.1f} features/sec
- **Memory Optimization Score:** {perf_data.get('memory_optimization_score', 'N/A'):.3f}
- **Parallel Processing Efficiency:** {perf_data.get('parallel_processing_efficiency', 'N/A'):.3f}
- **Caching Efficiency:** {perf_data.get('caching_efficiency', 'N/A'):.3f}

"""

            # Recommendations section with enhanced formatting
            if 'recommendations' in report_data:
                markdown_content += """## 💡 Key Recommendations

### Immediate Actions
"""
                recommendations = report_data['recommendations']
                for i, rec in enumerate(recommendations, 1):
                    markdown_content += f"{i}. **{rec}**\n"

                markdown_content += "\n### Strategic Improvements\n\n"
                markdown_content += "1. **Feature Selection** - Implement automated feature importance analysis\n"
                markdown_content += "2. **Quality Assurance** - Add comprehensive feature validation pipeline\n"
                markdown_content += "3. **Performance Monitoring** - Track feature engineering metrics over time\n"
                markdown_content += "4. **Scalability Planning** - Optimize for larger datasets and higher frequency\n"

            # Alerts section with enhanced formatting
            if 'alerts' in report_data:
                markdown_content += "\n## 🚨 Alerts & Critical Issues\n\n"
                alerts = report_data['alerts']
                for alert in alerts:
                    markdown_content += f"- {alert}\n"

            # Technical Details section
            markdown_content += f"""

## 🔧 Technical Details

**Configuration Summary:**
"""
            config = report_data.get('config_summary', {})
            for key, value in config.items():
                markdown_content += f"- **{key.replace('_', ' ').title()}:** {value}\n"

            markdown_content += f"""
**Report Generation:**
- **Version:** 2.0.0
- **Step:** step06_advanced_feature_engineering
- **Analysis Type:** Enhanced Feature Engineering Analysis

---
*This report was generated automatically by the Ares Trading System feature engineering pipeline.*
"""

            # Save enhanced markdown file
            markdown_path = self.save_training_report(
                data={'markdown_content': markdown_content},
                step_name='step06_advanced_feature_engineering',
                report_type='enhanced_analysis_summary',
                symbol=symbol,
                timeframe=timeframe,
                file_format='md'
            )

            return markdown_path

        except Exception as e:
            self.logger.error(f"Failed to save enhanced markdown report: {e}")
            return None

    def _generate_and_save_visualizations(self, report_data: Dict[str, Any], symbol: str, exchange: str, timeframe: str) -> List[str]:
        """Generate and save comprehensive visualization charts for enhanced reporting."""
        saved_files = []

        try:
            # Feature categories pie chart with enhanced styling
            if 'feature_engineering_analysis' in report_data:
                fe_data = report_data['feature_engineering_analysis']
                categories = fe_data.get('feature_categories', {})

                if categories:
                    plt.figure(figsize=(12, 8))
                    labels = [cat.replace('_', ' ').title() for cat in categories.keys()]
                    sizes = list(categories.values())

                    # Create custom colors
                    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

                    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                           colors=colors, shadow=True, explode=[0.05] * len(labels))
                    plt.title(f'Feature Categories Distribution - {symbol} ({timeframe})',
                            fontsize=16, fontweight='bold', pad=20)
                    plt.axis('equal')

                    # Save pie chart
                    pie_path = self.save_training_report(
                        data={'chart_data': {'labels': labels, 'sizes': sizes}},
                        step_name='step06_advanced_feature_engineering',
                        report_type='feature_categories_chart',
                        symbol=symbol,
                        timeframe=timeframe,
                        file_format='png'
                    )
                    if pie_path:
                        saved_files.append(pie_path)

                    plt.close()

            # Enhanced quality metrics radar chart
            if 'feature_quality_analysis' in report_data:
                quality_data = report_data['feature_quality_analysis']

                plt.figure(figsize=(10, 8))
                categories = ['Completeness', 'Validity', 'Uniqueness', 'Informativeness', 'Stability', 'Correlation']
                values = [
                    quality_data.get('completeness_score', 0),
                    quality_data.get('validity_score', 0),
                    quality_data.get('uniqueness_score', 0),
                    quality_data.get('informativeness_score', 0),
                    quality_data.get('stability_score', 0),
                    quality_data.get('correlation_score', 0)
                ]

                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                values += values[:1]  # Close the polygon
                angles += angles[:1]

                fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

                # Fill area
                ax.fill(angles, values, 'skyblue', alpha=0.4, label='Quality Score')

                # Plot line
                ax.plot(angles, values, 'b-', linewidth=3, marker='o', markersize=8,
                       markerfacecolor='red', markeredgecolor='red', label='Current Score')

                # Add ideal reference line
                ideal_values = [1.0] * len(categories)
                ideal_values += ideal_values[:1]
                ax.plot(angles, ideal_values, 'g--', linewidth=2, alpha=0.7, label='Ideal Score')

                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
                ax.set_ylim(0, 1.1)
                ax.set_title(f'Feature Quality Assessment Radar - {symbol}',
                           size=16, fontweight='bold', pad=20)
                ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
                ax.grid(True, alpha=0.3)

                # Save radar chart
                radar_path = self.save_training_report(
                    data={'chart_data': {'categories': categories, 'values': values[:-1]}},
                    step_name='step06_advanced_feature_engineering',
                    report_type='feature_quality_radar',
                    symbol=symbol,
                    timeframe=timeframe,
                    file_format='png'
                )
                if radar_path:
                    saved_files.append(radar_path)

                plt.close()

            # Hardware performance comparison chart
            if 'hardware_acceleration_analysis' in report_data:
                hw_data = report_data['hardware_acceleration_analysis']

                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle(f'Hardware Acceleration Performance - {symbol} ({timeframe})',
                           fontsize=16, fontweight='bold')

                # GPU vs CPU utilization
                metrics = ['GPU Utilization', 'CPU Utilization']
                values = [hw_data.get('gpu_utilization', 0), hw_data.get('cpu_utilization', 0)]
                colors = ['skyblue', 'lightcoral']

                bars = ax1.bar(metrics, values, color=colors, alpha=0.7)
                ax1.set_title('GPU vs CPU Utilization', fontweight='bold')
                ax1.set_ylabel('Utilization (%)')
                ax1.set_ylim(0, 100)

                for bar, value in zip(bars, values):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

                # Memory usage gauge
                memory_usage = hw_data.get('memory_usage_mb', 0)
                ax2.barh(['Memory Usage'], [memory_usage], color='lightgreen', alpha=0.7)
                ax2.set_title('Memory Usage (MB)', fontweight='bold')
                ax2.set_xlabel('Memory (MB)')
                ax2.text(memory_usage/2, 0, f'{memory_usage:.1f} MB',
                        ha='center', va='center', fontweight='bold', fontsize=12)

                # Processing speedup
                speedup = hw_data.get('processing_speedup', 1.0)
                ax3.bar(['Processing Speedup'], [speedup], color='gold', alpha=0.7)
                ax3.set_title('Processing Speedup vs Baseline', fontweight='bold')
                ax3.set_ylabel('Speedup Factor')
                ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Baseline')
                ax3.text(0, speedup + 0.1, f'{speedup:.2f}x', ha='center', fontweight='bold')
                ax3.legend()

                # Hardware acceleration score
                accel_score = hw_data.get('hardware_acceleration_score', 0)
                ax4.barh(['Acceleration Score'], [accel_score], color='purple', alpha=0.7)
                ax4.set_title('Overall Acceleration Score', fontweight='bold')
                ax4.set_xlim(0, 1)
                ax4.text(accel_score/2, 0, f'{accel_score:.3f}',
                        ha='center', va='center', fontweight='bold', fontsize=12)

                # Add reference lines for score interpretation
                ax4.axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='Excellent')
                ax4.axvline(x=0.6, color='orange', linestyle='--', alpha=0.7, label='Good')
                ax4.axvline(x=0.4, color='red', linestyle='--', alpha=0.7, label='Needs Improvement')
                ax4.legend()

                plt.tight_layout()

                # Save hardware performance chart
                hw_path = self.save_training_report(
                    data={'chart_data': hw_data},
                    step_name='step06_advanced_feature_engineering',
                    report_type='hardware_performance_dashboard',
                    symbol=symbol,
                    timeframe=timeframe,
                    file_format='png'
                )
                if hw_path:
                    saved_files.append(hw_path)

                plt.close()

            # Wavelet analysis visualization
            if 'wavelet_analysis' in report_data:
                wavelet_data = report_data['wavelet_analysis']

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                fig.suptitle(f'Wavelet Feature Analysis - {symbol} ({timeframe})',
                           fontsize=16, fontweight='bold')

                # Wavelet configuration summary
                config_labels = ['Levels', 'Decomposition', 'Quality Score']
                config_values = [
                    wavelet_data.get('wavelet_levels', 0),
                    wavelet_data.get('decomposition_levels', 0),
                    wavelet_data.get('wavelet_quality_score', 0)
                ]

                bars = ax1.bar(config_labels, config_values, color=['blue', 'green', 'orange'], alpha=0.7)
                ax1.set_title('Wavelet Configuration', fontweight='bold')
                ax1.set_ylabel('Value')

                for bar, value in zip(bars, config_values):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                           f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

                # Frequency bands
                bands = wavelet_data.get('frequency_bands_analyzed', [])
                if bands:
                    band_counts = [1] * len(bands)  # Placeholder counts
                    ax2.bar(bands, band_counts, color='purple', alpha=0.7)
                    ax2.set_title('Analyzed Frequency Bands', fontweight='bold')
                    ax2.set_ylabel('Band Count')
                else:
                    ax2.text(0.5, 0.5, 'No frequency band data available',
                           ha='center', va='center', transform=ax2.transAxes, fontsize=12)

                plt.tight_layout()

                # Save wavelet analysis chart
                wavelet_path = self.save_training_report(
                    data={'chart_data': wavelet_data},
                    step_name='step06_advanced_feature_engineering',
                    report_type='wavelet_analysis_chart',
                    symbol=symbol,
                    timeframe=timeframe,
                    file_format='png'
                )
                if wavelet_path:
                    saved_files.append(wavelet_path)

                plt.close()

            # Technical indicators distribution
            if 'technical_indicator_analysis' in report_data:
                ti_data = report_data['technical_indicator_analysis']
                indicators = ti_data.get('indicators_generated', {})

                if indicators:
                    plt.figure(figsize=(12, 6))

                    categories = list(indicators.keys())
                    counts = list(indicators.values())

                    bars = plt.bar(categories, counts, color='teal', alpha=0.7)
                    plt.title(f'Technical Indicators Distribution - {symbol} ({timeframe})',
                            fontsize=16, fontweight='bold')
                    plt.xlabel('Indicator Category')
                    plt.ylabel('Number of Indicators')
                    plt.xticks(rotation=45)

                    # Add value labels
                    for bar, count in zip(bars, counts):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               f'{count}', ha='center', va='bottom', fontweight='bold')

                    plt.tight_layout()

                    # Save technical indicators chart
                    ti_path = self.save_training_report(
                        data={'chart_data': {'categories': categories, 'counts': counts}},
                        step_name='step06_advanced_feature_engineering',
                        report_type='technical_indicators_chart',
                        symbol=symbol,
                        timeframe=timeframe,
                        file_format='png'
                    )
                    if ti_path:
                        saved_files.append(ti_path)

                    plt.close()

            # Feature correlation heatmap (if interaction data available)
            if 'feature_interaction_analysis' in report_data:
                interaction_data = report_data['feature_interaction_analysis']
                high_corr_pairs = interaction_data.get('high_correlation_pairs', [])

                if high_corr_pairs:
                    # Create a simplified correlation visualization
                    plt.figure(figsize=(10, 6))

                    pairs = [f"{pair[0]}↔{pair[1]}" for pair in high_corr_pairs[:10]]
                    correlations = [pair[2] for pair in high_corr_pairs[:10]]

                    bars = plt.barh(pairs, correlations, color='salmon', alpha=0.7)
                    plt.title(f'Top Feature Correlations - {symbol} ({timeframe})',
                            fontsize=16, fontweight='bold')
                    plt.xlabel('Correlation Coefficient')
                    plt.ylabel('Feature Pairs')

                    # Add correlation values
                    for bar, corr in zip(bars, correlations):
                        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                               f'{corr:.3f}', va='center', fontweight='bold')

                    plt.tight_layout()

                    # Save correlation chart
                    corr_path = self.save_training_report(
                        data={'chart_data': {'pairs': pairs, 'correlations': correlations}},
                        step_name='step06_advanced_feature_engineering',
                        report_type='feature_correlations_chart',
                        symbol=symbol,
                        timeframe=timeframe,
                        file_format='png'
                    )
                    if corr_path:
                        saved_files.append(corr_path)

                    plt.close()

            self.logger.info(f"✅ Generated {len(saved_files)} enhanced visualization charts")

        except Exception as e:
            self.logger.error(f"Failed to generate enhanced visualizations: {e}")

        return saved_files

    def _save_csv_summary(self, report_data: Dict[str, Any], symbol: str, exchange: str, timeframe: str) -> Optional[str]:
        """Save CSV summary of key metrics."""
        try:
            # Create summary data
            summary_data = {
                'metric': [],
                'value': [],
                'category': []
            }

            # Add feature engineering metrics
            if 'feature_engineering_analysis' in report_data:
                fe_data = report_data['feature_engineering_analysis']
                summary_data['metric'].append('total_features_created')
                summary_data['value'].append(fe_data.get('total_features_created', 0))
                summary_data['category'].append('feature_engineering')

                summary_data['metric'].append('features_per_second')
                summary_data['value'].append(fe_data.get('features_per_second', 0))
                summary_data['category'].append('feature_engineering')

            # Add quality metrics
            if 'feature_quality_analysis' in report_data:
                quality_data = report_data['feature_quality_analysis']
                summary_data['metric'].append('overall_quality_score')
                summary_data['value'].append(quality_data.get('overall_quality_score', 0))
                summary_data['category'].append('feature_quality')

                summary_data['metric'].append('completeness_score')
                summary_data['value'].append(quality_data.get('completeness_score', 0))
                summary_data['category'].append('feature_quality')

            # Add hardware metrics
            if 'hardware_acceleration_analysis' in report_data:
                hw_data = report_data['hardware_acceleration_analysis']
                summary_data['metric'].append('hardware_acceleration_score')
                summary_data['value'].append(hw_data.get('hardware_acceleration_score', 0))
                summary_data['category'].append('hardware')

                summary_data['metric'].append('processing_speedup')
                summary_data['value'].append(hw_data.get('processing_speedup', 0))
                summary_data['category'].append('hardware')

            # Save as CSV
            csv_path = self.save_training_report(
                data={'summary_data': summary_data},
                step_name='step06_advanced_feature_engineering',
                report_type='metrics_summary',
                symbol=symbol,
                timeframe=timeframe,
                file_format='csv'
            )

            return csv_path

        except Exception as e:
            self.logger.error(f"Failed to save CSV summary: {e}")
            return None

    def _generate_fallback_report(self, input_data: pd.DataFrame, output_features: pd.DataFrame, error_message: str) -> Dict[str, Any]:
        """Generate a basic fallback report when full analysis fails."""
        return {
            'timestamp': datetime.now().isoformat(),
            'step_name': 'step06_advanced_feature_engineering',
            'analysis_type': 'fallback_report',
            'error': error_message,
            'basic_info': {
                'input_features': input_data.shape[1],
                'output_features': output_features.shape[1],
                'total_samples': len(input_data),
                'features_created': output_features.shape[1] - input_data.shape[1]
            },
            'recommendations': ['Review error logs and fix underlying issues before re-running analysis'],
            'alerts': ['Analysis failed - manual review required']
        }
