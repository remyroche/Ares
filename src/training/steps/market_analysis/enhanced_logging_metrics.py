#!/usr/bin/env python3
"""
Enhanced Logging and Metrics System for Market Analysis Pipeline

This module provides comprehensive logging with emojis and detailed metrics
for troubleshooting and monitoring the market analysis pipeline.
"""

import logging
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict

# Optional imports
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS_NUMPY = True
except ImportError:
    HAS_PANDAS_NUMPY = False
    # Create dummy classes for type hints
    class pd:
        class DataFrame:
            pass
        class Series:
            pass
    class np:
        @staticmethod
        def isnan(x):
            return False
        @staticmethod
        def isinf(x):
            return False
        class linalg:
            @staticmethod
            def cond(x):
                return 1.0
            @staticmethod
            def matrix_rank(x):
                return len(x) if hasattr(x, '__len__') else 1

# Core utilities
try:
    from src.utils.common_operations import get_logger, get_current_datetime, format_datetime
except ImportError:
    # Fallback logging setup
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    def get_current_datetime():
        return datetime.now()
    def format_datetime(dt):
        return dt.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class FeatureQualityMetrics:
    """Metrics for feature quality assessment."""
    total_features: int = 0
    nan_features: int = 0
    constant_features: int = 0
    high_correlation_pairs: int = 0
    low_variance_features: int = 0
    infinite_values: int = 0
    duplicate_features: int = 0
    quality_score: float = 0.0
    issues: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []


@dataclass
class RegimeQualityMetrics:
    """Metrics for regime clustering quality."""
    total_regimes: int = 0
    regime_counts: Dict[int, int] = None
    regime_balance_score: float = 0.0
    transition_stability: float = 0.0
    regime_persistence: float = 0.0
    quality_threshold_met: bool = False
    issues: List[str] = None
    
    def __post_init__(self):
        if self.regime_counts is None:
            self.regime_counts = {}
        if self.issues is None:
            self.issues = []


@dataclass
class StepMetrics:
    """Metrics for individual pipeline steps."""
    step_name: str = ""
    start_time: datetime = None
    end_time: datetime = None
    duration_seconds: float = 0.0
    success: bool = False
    error_message: str = ""
    input_data_shape: Tuple[int, int] = (0, 0)
    output_data_shape: Tuple[int, int] = (0, 0)
    memory_usage_mb: float = 0.0
    feature_metrics: FeatureQualityMetrics = None
    regime_metrics: RegimeQualityMetrics = None
    custom_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.feature_metrics is None:
            self.feature_metrics = FeatureQualityMetrics()
        if self.regime_metrics is None:
            self.regime_metrics = RegimeQualityMetrics()
        if self.custom_metrics is None:
            self.custom_metrics = {}


class EnhancedPipelineLogger:
    """
    Enhanced logger with emojis and comprehensive metrics for market analysis pipeline.
    """
    
    def __init__(self, name: str = "market_analysis", log_dir: str = "log"):
        """Initialize the enhanced logger."""
        self.logger = get_logger(name)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Metrics tracking
        self.step_metrics: Dict[str, StepMetrics] = {}
        self.pipeline_start_time = None
        self.pipeline_end_time = None
        self.correlation_id = None
        
        # Quality thresholds
        self.quality_thresholds = {
            'feature_quality_min_score': 0.7,
            'regime_balance_min_score': 0.3,
            'regime_persistence_min': 0.5,
            'max_nan_ratio': 0.1,
            'max_correlation_threshold': 0.95,
            'min_regime_samples': 100
        }
        
        # Emoji mapping for different types of messages
        self.emoji_map = {
            'start': '🚀',
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️',
            'progress': '📊',
            'feature': '🔧',
            'regime': '🎯',
            'matrix': '🧮',
            'validation': '🔍',
            'quality': '📈',
            'performance': '⚡',
            'memory': '💾',
            'time': '⏱️',
            'data': '📋',
            'config': '⚙️',
            'threshold': '🎚️',
            'issue': '🚨',
            'fix': '🔧',
            'complete': '🎉',
            'step': '📝',
            'pipeline': '🔄'
        }
    
    def start_pipeline(self, symbol: str, exchange: str, correlation_id: str = None):
        """Start pipeline logging with comprehensive initialization."""
        self.pipeline_start_time = get_current_datetime()
        self.correlation_id = correlation_id or f"market_analysis_{symbol}_{exchange}_{int(time.time())}"
        
        self.logger.info("=" * 100)
        self.logger.info(f"{self.emoji_map['pipeline']} MARKET ANALYSIS PIPELINE STARTED")
        self.logger.info("=" * 100)
        self.logger.info(f"{self.emoji_map['time']} Start Time: {format_datetime(self.pipeline_start_time)}")
        self.logger.info(f"{self.emoji_map['config']} Symbol: {symbol}")
        self.logger.info(f"{self.emoji_map['config']} Exchange: {exchange}")
        self.logger.info(f"{self.emoji_map['config']} Correlation ID: {self.correlation_id}")
        self.logger.info("=" * 100)
    
    def end_pipeline(self, success: bool = True, error_message: str = ""):
        """End pipeline logging with comprehensive summary."""
        self.pipeline_end_time = get_current_datetime()
        duration = (self.pipeline_end_time - self.pipeline_start_time).total_seconds()
        
        self.logger.info("=" * 100)
        if success:
            self.logger.info(f"{self.emoji_map['complete']} MARKET ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
        else:
            self.logger.info(f"{self.emoji_map['error']} MARKET ANALYSIS PIPELINE FAILED!")
        
        self.logger.info("=" * 100)
        self.logger.info(f"{self.emoji_map['time']} End Time: {format_datetime(self.pipeline_end_time)}")
        self.logger.info(f"{self.emoji_map['time']} Total Duration: {duration:.2f} seconds")
        
        if error_message:
            self.logger.error(f"{self.emoji_map['error']} Error: {error_message}")
        
        # Log step summary
        self._log_step_summary()
        
        # Save metrics to file
        self._save_metrics_to_file()
        
        self.logger.info("=" * 100)
    
    def start_step(self, step_name: str, description: str = ""):
        """Start logging for a specific step."""
        start_time = get_current_datetime()
        
        self.step_metrics[step_name] = StepMetrics(
            step_name=step_name,
            start_time=start_time
        )
        
        self.logger.info(f"{self.emoji_map['step']} Starting Step: {step_name}")
        if description:
            self.logger.info(f"{self.emoji_map['info']} Description: {description}")
        self.logger.info(f"{self.emoji_map['time']} Start Time: {format_datetime(start_time)}")
        self.logger.info("-" * 80)
    
    def end_step(self, step_name: str, success: bool = True, error_message: str = "", 
                 input_shape: Tuple[int, int] = None, output_shape: Tuple[int, int] = None,
                 memory_usage_mb: float = 0.0):
        """End logging for a specific step."""
        if step_name not in self.step_metrics:
            self.logger.warning(f"{self.emoji_map['warning']} Step {step_name} not found in metrics")
            return
        
        end_time = get_current_datetime()
        duration = (end_time - self.step_metrics[step_name].start_time).total_seconds()
        
        # Update metrics
        self.step_metrics[step_name].end_time = end_time
        self.step_metrics[step_name].duration_seconds = duration
        self.step_metrics[step_name].success = success
        self.step_metrics[step_name].error_message = error_message
        self.step_metrics[step_name].input_data_shape = input_shape or (0, 0)
        self.step_metrics[step_name].output_data_shape = output_shape or (0, 0)
        self.step_metrics[step_name].memory_usage_mb = memory_usage_mb
        
        # Log step completion
        self.logger.info("-" * 80)
        if success:
            self.logger.info(f"{self.emoji_map['success']} Step {step_name} completed successfully")
        else:
            self.logger.error(f"{self.emoji_map['error']} Step {step_name} failed: {error_message}")
        
        self.logger.info(f"{self.emoji_map['time']} Duration: {duration:.2f} seconds")
        if input_shape:
            self.logger.info(f"{self.emoji_map['data']} Input Shape: {input_shape}")
        if output_shape:
            self.logger.info(f"{self.emoji_map['data']} Output Shape: {output_shape}")
        if memory_usage_mb > 0:
            self.logger.info(f"{self.emoji_map['memory']} Memory Usage: {memory_usage_mb:.2f} MB")
    
    def log_feature_quality(self, step_name: str, data: pd.DataFrame, feature_columns: List[str] = None):
        """Log comprehensive feature quality metrics."""
        if feature_columns is None:
            feature_columns = data.columns.tolist()
        
        metrics = self._calculate_feature_quality_metrics(data, feature_columns)
        
        if step_name in self.step_metrics:
            self.step_metrics[step_name].feature_metrics = metrics
        
        # Log feature quality
        self.logger.info(f"{self.emoji_map['feature']} Feature Quality Analysis for {step_name}:")
        self.logger.info(f"  {self.emoji_map['data']} Total Features: {metrics.total_features}")
        self.logger.info(f"  {self.emoji_map['quality']} Quality Score: {metrics.quality_score:.3f}")
        
        if metrics.nan_features > 0:
            self.logger.warning(f"  {self.emoji_map['issue']} NaN Features: {metrics.nan_features}")
        if metrics.constant_features > 0:
            self.logger.warning(f"  {self.emoji_map['issue']} Constant Features: {metrics.constant_features}")
        if metrics.high_correlation_pairs > 0:
            self.logger.warning(f"  {self.emoji_map['issue']} High Correlation Pairs: {metrics.high_correlation_pairs}")
        if metrics.low_variance_features > 0:
            self.logger.warning(f"  {self.emoji_map['issue']} Low Variance Features: {metrics.low_variance_features}")
        if metrics.infinite_values > 0:
            self.logger.warning(f"  {self.emoji_map['issue']} Infinite Values: {metrics.infinite_values}")
        if metrics.duplicate_features > 0:
            self.logger.warning(f"  {self.emoji_map['issue']} Duplicate Features: {metrics.duplicate_features}")
        
        # Log quality assessment
        if metrics.quality_score >= self.quality_thresholds['feature_quality_min_score']:
            self.logger.info(f"  {self.emoji_map['success']} Feature quality meets threshold ({self.quality_thresholds['feature_quality_min_score']})")
        else:
            self.logger.warning(f"  {self.emoji_map['warning']} Feature quality below threshold ({self.quality_thresholds['feature_quality_min_score']})")
        
        # Log specific issues
        for issue in metrics.issues:
            self.logger.warning(f"  {self.emoji_map['issue']} {issue}")
    
    def log_regime_quality(self, step_name: str, regimes: pd.Series, regime_labels: List[str] = None):
        """Log comprehensive regime clustering quality metrics."""
        metrics = self._calculate_regime_quality_metrics(regimes, regime_labels)
        
        if step_name in self.step_metrics:
            self.step_metrics[step_name].regime_metrics = metrics
        
        # Log regime quality
        self.logger.info(f"{self.emoji_map['regime']} Regime Quality Analysis for {step_name}:")
        self.logger.info(f"  {self.emoji_map['data']} Total Regimes: {metrics.total_regimes}")
        self.logger.info(f"  {self.emoji_map['quality']} Balance Score: {metrics.regime_balance_score:.3f}")
        self.logger.info(f"  {self.emoji_map['quality']} Persistence: {metrics.regime_persistence:.3f}")
        self.logger.info(f"  {self.emoji_map['quality']} Transition Stability: {metrics.transition_stability:.3f}")
        
        # Log regime distribution
        for regime_id, count in metrics.regime_counts.items():
            percentage = (count / len(regimes)) * 100
            self.logger.info(f"  {self.emoji_map['data']} Regime {regime_id}: {count} samples ({percentage:.1f}%)")
        
        # Log quality assessment
        if metrics.quality_threshold_met:
            self.logger.info(f"  {self.emoji_map['success']} Regime quality meets all thresholds")
        else:
            self.logger.warning(f"  {self.emoji_map['warning']} Regime quality below thresholds")
        
        # Log specific issues
        for issue in metrics.issues:
            self.logger.warning(f"  {self.emoji_map['issue']} {issue}")
    
    def log_step6_metrics(self, step_name: str, feature_engine_results: Dict[str, Any]):
        """Log specific metrics for Step 6 (Feature Engineering)."""
        self.logger.info(f"{self.emoji_map['feature']} Step 6 Feature Engineering Metrics:")
        
        # Log feature engineering results
        if 'total_features_created' in feature_engine_results:
            self.logger.info(f"  {self.emoji_map['data']} Total Features Created: {feature_engine_results['total_features_created']}")
        
        if 'interaction_features' in feature_engine_results:
            self.logger.info(f"  {self.emoji_map['feature']} Interaction Features: {feature_engine_results['interaction_features']}")
        
        if 'selected_features' in feature_engine_results:
            self.logger.info(f"  {self.emoji_map['feature']} Selected Features: {feature_engine_results['selected_features']}")
        
        if 'feature_importance_top_10' in feature_engine_results:
            self.logger.info(f"  {self.emoji_map['quality']} Top 10 Feature Importance:")
            for i, (feature, importance) in enumerate(feature_engine_results['feature_importance_top_10'][:10], 1):
                self.logger.info(f"    {i:2d}. {feature}: {importance:.4f}")
        
        # Log optimization results
        if 'lookback_optimization' in feature_engine_results:
            opt_results = feature_engine_results['lookback_optimization']
            self.logger.info(f"  {self.emoji_map['performance']} Lookback Optimization:")
            self.logger.info(f"    {self.emoji_map['success']} Optimized Indicators: {opt_results.get('optimized_count', 0)}")
            self.logger.info(f"    {self.emoji_map['time']} Optimization Time: {opt_results.get('optimization_time', 0):.2f}s")
        
        # Store in step metrics
        if step_name in self.step_metrics:
            self.step_metrics[step_name].custom_metrics.update(feature_engine_results)
    
    def log_step7_metrics(self, step_name: str, matrix_results: Dict[str, Any]):
        """Log specific metrics for Step 7 (Matrix Operations)."""
        self.logger.info(f"{self.emoji_map['matrix']} Step 7 Matrix Operations Metrics:")
        
        # Log matrix operation results
        if 'matrix_operations_performed' in matrix_results:
            self.logger.info(f"  {self.emoji_map['matrix']} Operations Performed: {matrix_results['matrix_operations_performed']}")
        
        if 'eigenvalue_analysis' in matrix_results:
            eigen_results = matrix_results['eigenvalue_analysis']
            self.logger.info(f"  {self.emoji_map['matrix']} Eigenvalue Analysis:")
            self.logger.info(f"    {self.emoji_map['data']} Condition Number: {eigen_results.get('condition_number', 0):.2e}")
            self.logger.info(f"    {self.emoji_map['data']} Rank: {eigen_results.get('rank', 0)}")
            self.logger.info(f"    {self.emoji_map['data']} Effective Rank: {eigen_results.get('effective_rank', 0)}")
        
        if 'correlation_analysis' in matrix_results:
            corr_results = matrix_results['correlation_analysis']
            self.logger.info(f"  {self.emoji_map['matrix']} Correlation Analysis:")
            self.logger.info(f"    {self.emoji_map['data']} High Correlation Pairs: {corr_results.get('high_correlation_pairs', 0)}")
            self.logger.info(f"    {self.emoji_map['data']} Max Correlation: {corr_results.get('max_correlation', 0):.3f}")
        
        if 'performance_metrics' in matrix_results:
            perf_results = matrix_results['performance_metrics']
            self.logger.info(f"  {self.emoji_map['performance']} Performance Metrics:")
            self.logger.info(f"    {self.emoji_map['time']} Computation Time: {perf_results.get('computation_time', 0):.2f}s")
            self.logger.info(f"    {self.emoji_map['memory']} Memory Usage: {perf_results.get('memory_usage_mb', 0):.2f} MB")
        
        # Store in step metrics
        if step_name in self.step_metrics:
            self.step_metrics[step_name].custom_metrics.update(matrix_results)
    
    def log_progress(self, step_name: str, progress: float, message: str = ""):
        """Log progress updates with visual indicators."""
        progress_bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
        self.logger.info(f"{self.emoji_map['progress']} {step_name}: [{progress_bar}] {progress*100:.1f}% {message}")
    
    def log_issue(self, step_name: str, issue_type: str, message: str, severity: str = "warning"):
        """Log specific issues with appropriate emojis and severity."""
        emoji = self.emoji_map.get(issue_type, self.emoji_map['issue'])
        
        if severity == "error":
            self.logger.error(f"{emoji} {step_name} - {issue_type.upper()}: {message}")
        elif severity == "warning":
            self.logger.warning(f"{emoji} {step_name} - {issue_type.upper()}: {message}")
        else:
            self.logger.info(f"{emoji} {step_name} - {issue_type.upper()}: {message}")
    
    def _calculate_feature_quality_metrics(self, data, feature_columns: List[str]) -> FeatureQualityMetrics:
        """Calculate comprehensive feature quality metrics."""
        metrics = FeatureQualityMetrics()
        metrics.total_features = len(feature_columns)
        
        if not HAS_PANDAS_NUMPY:
            # Fallback when pandas/numpy not available
            metrics.quality_score = 1.0
            metrics.issues.append("Pandas/numpy not available - using fallback metrics")
            return metrics
        
        try:
            # Check for NaN values
            nan_counts = data[feature_columns].isnull().sum()
            metrics.nan_features = (nan_counts > 0).sum()
            
            # Check for constant features
            constant_features = []
            for col in feature_columns:
                if data[col].nunique() <= 1:
                    constant_features.append(col)
            metrics.constant_features = len(constant_features)
            
            # Check for infinite values
            inf_counts = np.isinf(data[feature_columns].select_dtypes(include=[np.number])).sum()
            metrics.infinite_values = (inf_counts > 0).sum()
            
            # Check for low variance features
            numeric_cols = data[feature_columns].select_dtypes(include=[np.number]).columns
            low_variance_features = []
            for col in numeric_cols:
                if data[col].var() < 1e-10:
                    low_variance_features.append(col)
            metrics.low_variance_features = len(low_variance_features)
            
            # Check for high correlation pairs
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr()
                high_corr_pairs = 0
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > self.quality_thresholds['max_correlation_threshold']:
                            high_corr_pairs += 1
                metrics.high_correlation_pairs = high_corr_pairs
            
            # Check for duplicate features
            metrics.duplicate_features = len(feature_columns) - len(set(feature_columns))
            
            # Calculate quality score
            total_issues = (metrics.nan_features + metrics.constant_features + 
                           metrics.high_correlation_pairs + metrics.low_variance_features + 
                           metrics.infinite_values + metrics.duplicate_features)
            metrics.quality_score = max(0, 1 - (total_issues / max(1, metrics.total_features)))
            
            # Collect specific issues
            if metrics.nan_features > 0:
                nan_cols = nan_counts[nan_counts > 0].index.tolist()
                metrics.issues.append(f"Features with NaN values: {nan_cols[:5]}{'...' if len(nan_cols) > 5 else ''}")
            
            if metrics.constant_features > 0:
                metrics.issues.append(f"Constant features: {constant_features[:5]}{'...' if len(constant_features) > 5 else ''}")
            
            if metrics.high_correlation_pairs > 0:
                metrics.issues.append(f"High correlation pairs detected: {metrics.high_correlation_pairs}")
            
            if metrics.low_variance_features > 0:
                metrics.issues.append(f"Low variance features: {low_variance_features[:5]}{'...' if len(low_variance_features) > 5 else ''}")
            
            if metrics.infinite_values > 0:
                metrics.issues.append(f"Features with infinite values: {metrics.infinite_values}")
            
            if metrics.duplicate_features > 0:
                metrics.issues.append(f"Duplicate feature names: {metrics.duplicate_features}")
                
        except Exception as e:
            metrics.quality_score = 0.5
            metrics.issues.append(f"Error calculating metrics: {str(e)}")
        
        return metrics
    
    def _calculate_regime_quality_metrics(self, regimes, regime_labels: List[str] = None) -> RegimeQualityMetrics:
        """Calculate comprehensive regime clustering quality metrics."""
        metrics = RegimeQualityMetrics()
        
        if not HAS_PANDAS_NUMPY:
            # Fallback when pandas/numpy not available
            metrics.quality_score = 1.0
            metrics.issues.append("Pandas/numpy not available - using fallback regime metrics")
            return metrics
        
        try:
            # Basic regime statistics
            unique_regimes = regimes.unique()
            metrics.total_regimes = len(unique_regimes)
            
            # Regime counts
            regime_counts = regimes.value_counts().sort_index()
            metrics.regime_counts = regime_counts.to_dict()
            
            # Calculate balance score (how evenly distributed the regimes are)
            if len(regime_counts) > 1:
                min_count = regime_counts.min()
                max_count = regime_counts.max()
                metrics.regime_balance_score = min_count / max_count if max_count > 0 else 0
            else:
                metrics.regime_balance_score = 1.0
            
            # Calculate regime persistence (average length of regime sequences)
            regime_changes = (regimes != regimes.shift()).sum()
            if regime_changes > 0:
                metrics.regime_persistence = len(regimes) / regime_changes
            else:
                metrics.regime_persistence = len(regimes)
            
            # Calculate transition stability (how stable regime transitions are)
            if len(regimes) > 1:
                transitions = (regimes != regimes.shift()).sum()
                metrics.transition_stability = 1 - (transitions / len(regimes))
            else:
                metrics.transition_stability = 1.0
            
            # Check quality thresholds
            min_samples = self.quality_thresholds['min_regime_samples']
            balance_threshold = self.quality_thresholds['regime_balance_min_score']
            persistence_threshold = self.quality_thresholds['regime_persistence_min']
            
            quality_checks = []
            
            # Check minimum samples per regime
            for regime_id, count in metrics.regime_counts.items():
                if count < min_samples:
                    quality_checks.append(False)
                    metrics.issues.append(f"Regime {regime_id} has only {count} samples (minimum: {min_samples})")
                else:
                    quality_checks.append(True)
            
            # Check balance score
            if metrics.regime_balance_score < balance_threshold:
                quality_checks.append(False)
                metrics.issues.append(f"Regime balance score {metrics.regime_balance_score:.3f} below threshold {balance_threshold}")
            else:
                quality_checks.append(True)
            
            # Check persistence
            if metrics.regime_persistence < persistence_threshold:
                quality_checks.append(False)
                metrics.issues.append(f"Regime persistence {metrics.regime_persistence:.3f} below threshold {persistence_threshold}")
            else:
                quality_checks.append(True)
            
            metrics.quality_threshold_met = all(quality_checks)
            
        except Exception as e:
            metrics.quality_threshold_met = False
            metrics.issues.append(f"Error calculating regime metrics: {str(e)}")
        
        return metrics
    
    def _log_step_summary(self):
        """Log a summary of all steps."""
        self.logger.info(f"{self.emoji_map['pipeline']} PIPELINE STEP SUMMARY:")
        self.logger.info("-" * 80)
        
        total_duration = 0
        successful_steps = 0
        failed_steps = 0
        
        for step_name, metrics in self.step_metrics.items():
            status_emoji = self.emoji_map['success'] if metrics.success else self.emoji_map['error']
            self.logger.info(f"{status_emoji} {step_name}: {metrics.duration_seconds:.2f}s")
            
            total_duration += metrics.duration_seconds
            if metrics.success:
                successful_steps += 1
            else:
                failed_steps += 1
        
        self.logger.info("-" * 80)
        self.logger.info(f"{self.emoji_map['time']} Total Step Duration: {total_duration:.2f}s")
        self.logger.info(f"{self.emoji_map['success']} Successful Steps: {successful_steps}")
        self.logger.info(f"{self.emoji_map['error']} Failed Steps: {failed_steps}")
    
    def _save_metrics_to_file(self):
        """Save all metrics to a JSON file."""
        try:
            metrics_data = {
                'correlation_id': self.correlation_id,
                'pipeline_start_time': self.pipeline_start_time.isoformat() if self.pipeline_start_time else None,
                'pipeline_end_time': self.pipeline_end_time.isoformat() if self.pipeline_end_time else None,
                'step_metrics': {}
            }
            
            for step_name, metrics in self.step_metrics.items():
                metrics_data['step_metrics'][step_name] = asdict(metrics)
                # Convert datetime objects to strings
                if metrics_data['step_metrics'][step_name]['start_time']:
                    metrics_data['step_metrics'][step_name]['start_time'] = metrics.start_time.isoformat()
                if metrics_data['step_metrics'][step_name]['end_time']:
                    metrics_data['step_metrics'][step_name]['end_time'] = metrics.end_time.isoformat()
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.log_dir / f"market_analysis_metrics_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            self.logger.info(f"{self.emoji_map['memory']} Metrics saved to: {filename}")
            
        except Exception as e:
            self.logger.error(f"{self.emoji_map['error']} Failed to save metrics: {e}")


# Global logger instance
enhanced_logger = EnhancedPipelineLogger()