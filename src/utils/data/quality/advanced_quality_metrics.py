"""Advanced data quality metrics with tolerant parameters for flagging obvious issues."""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Set

from datetime import datetime, timedelta
from dataclasses import dataclass, field
from scipy import stats
from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards
import logging
import time

@dataclass
class QualityMetric:
    """Represents a data quality metric."""
    name: str
    value: float
    threshold: float
    severity: str  # 'info', 'warning', 'error', 'critical'
    message: str
    suggested_action: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory = dict)

@dataclass
class QualityAssessment:
    """Comprehensive data quality assessment."""
    overall_score: float
    metrics: List[QualityMetric]
    issues_found: int
    warnings_found: int
    critical_issues: int
    assessment_timestamp: datetime
    data_shape: Tuple[int, int]
    metadata: Dict[str, Any] = field(default_factory = dict)

class AdvancedQualityMetrics:
    """Advanced data quality assessment with tolerant parameters."""
    
    def __init__(self):
        start_time = time.time()
        self.logger = system_logger.getChild('AdvancedQualityMetrics')
        self.standards = PipelineStandards(self.logger)
        
        # Tolerant thresholds - only flag obvious issues
        self.tolerant_thresholds = {
            'temporal_consistency': {
                'max_gap_hours': 24,  # 24 hours gap is acceptable
                'max_gap_percentage': 0.1,  # 10% of data can have gaps
                'min_continuity_percentage': 0.8  # 80% continuity is acceptable
            },
            'price_anomaly_detection': {
                'z_score_threshold': 5.0,  # Very high z-score (5σ)
                'price_change_threshold': 0.5,  # 50% price change
                'max_anomaly_percentage': 0.05  # 5% anomalies acceptable
            },
            'volume_pattern_analysis': {
                'volume_spike_threshold': 10.0,  # 10x normal volume
                'zero_volume_threshold': 0.1,  # 10% zero volume acceptable
                'volume_correlation_threshold': 0.3  # Low correlation acceptable
            },
            'market_microstructure': {
                'bid_ask_spread_threshold': 0.1,  # 10% spread acceptable
                'price_impact_threshold': 0.05,  # 5% price impact acceptable
                'liquidity_threshold': 0.01  # 1% liquidity threshold
            },
            'data_completeness': {
                'missing_data_threshold': 0.2,  # 20% missing data acceptable
                'duplicate_threshold': 0.1,  # 10% duplicates acceptable
                'null_threshold': 0.15  # 15% nulls acceptable
            },
            'statistical_consistency': {
                'distribution_skew_threshold': 3.0,  # High skew acceptable
                'distribution_kurtosis_threshold': 10.0,  # High kurtosis acceptable
                'correlation_threshold': 0.99  # Very high correlation
            }
        }
        
        # Quality assessment history
        self.assessment_history: List[QualityAssessment] = []
        
        self.logger.info("📊 AdvancedQualityMetrics initialized with tolerant parameters")
        
        # Add timing information (Numba-safe implementation)
        duration = time.time() - start_time
        try:
            from src.utils.tprint import tprint_performance
            tprint_performance("AdvancedQualityMetrics initialization", duration)
        except ImportError:
            # Fallback to basic logging (Numba-safe)
            self.logger.info(f"⏱️ AdvancedQualityMetrics initialized in {duration:.3f}s")
    
    def comprehensive_quality_assessment(self, 
                                       data: pd.DataFrame,
                                       context: Optional[str] = None,
                                       step_name: Optional[str] = None) -> QualityAssessment:
        """
        Perform comprehensive data quality assessment with tolerant parameters.
        
        Args:
            data: DataFrame to assess
            context: Assessment context (e.g., 'pre_processing', 'post_processing')
            step_name: Name of the pipeline step
            
        Returns:
            Comprehensive quality assessment
        """
        self.logger.info(f"🔍 Performing comprehensive quality assessment...")
        
        start_time = datetime.now()
        metrics = []
        
        try:
            # Run all quality checks
            quality_checks = [
                self._check_temporal_consistency,
                self._check_price_anomaly_detection,
                self._check_volume_pattern_analysis,
                self._check_market_microstructure,
                self._check_data_completeness,
                self._check_statistical_consistency,
                self._check_basic_data_integrity
            ]
            
            for check_func in quality_checks:
                try:
                    check_metrics = check_func(data)
                    metrics.extend(check_metrics)
                except Exception as e:
                    self.logger.warning(f"⚠️ Quality check failed: {e}")
                    metrics.append(QualityMetric(
                        name='check_error',
                        value = 0.0,
                        threshold = 1.0,
                        severity='warning',
                        message = f'Quality check failed: {str(e)}'
                    ))
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(metrics)
            
            # Count issues by severity
            issues_found = len([m for m in metrics if m.severity in ['error', 'critical']])
            warnings_found = len([m for m in metrics if m.severity == 'warning'])
            critical_issues = len([m for m in metrics if m.severity == 'critical'])
            
            # Create assessment
            assessment = QualityAssessment(
                overall_score = overall_score,
                metrics = metrics,
                issues_found = issues_found,
                warnings_found = warnings_found,
                critical_issues = critical_issues,
                assessment_timestamp = start_time,
                data_shape = data.shape,
                metadata={
                    'context': context,
                    'step_name': step_name,
                    'assessment_duration': (datetime.now() - start_time).total_seconds()
                }
            )
            
            # Store assessment
            self.assessment_history.append(assessment)
            
            # Log summary
            self.logger.info(f"✅ Quality assessment completed: "
                           f"score={overall_score:.1f}, issues={issues_found}, "
                           f"warnings={warnings_found}, critical={critical_issues}")
            
            return assessment
            
        except Exception as e:
            self.logger.exception(f"❌ Error in quality assessment: {e}")
            return QualityAssessment(
                overall_score = 0.0,
                metrics=[QualityMetric(
                    name='assessment_error',
                    value = 0.0,
                    threshold = 1.0,
                    severity='critical',
                    message = f'Assessment failed: {str(e)}'
                )],
                issues_found = 1,
                warnings_found = 0,
                critical_issues = 1,
                assessment_timestamp = start_time,
                data_shape = data.shape
            )
    
    def _check_temporal_consistency(self, data: pd.DataFrame) -> List[QualityMetric]:
        """Check temporal consistency with tolerant parameters."""
        metrics = []
        
        if 'timestamp' not in data.columns:
            return metrics
        
        try:
            timestamps = pd.to_datetime(data['timestamp']).sort_values()
            
            # Check for gaps
            time_diffs = timestamps.diff().dropna()
            gap_threshold = timedelta(hours = self.tolerant_thresholds['temporal_consistency']['max_gap_hours'])
            
            large_gaps = time_diffs[time_diffs > gap_threshold]
            gap_percentage = len(large_gaps) / len(time_diffs) if len(time_diffs) > 0 else 0
            
            if gap_percentage > self.tolerant_thresholds['temporal_consistency']['max_gap_percentage']:
                metrics.append(QualityMetric(
                    name='temporal_gaps',
                    value = gap_percentage,
                    threshold = self.tolerant_thresholds['temporal_consistency']['max_gap_percentage'],
                    severity='warning',
                    message = f'Large temporal gaps found: {gap_percentage:.1%} of intervals',
                    suggested_action='Check data collection continuity'
                ))
            
            # Check for duplicates
            duplicate_timestamps = timestamps.duplicated().sum()
            duplicate_percentage = duplicate_timestamps / len(timestamps) if len(timestamps) > 0 else 0
            
            if duplicate_percentage > 0.05:  # 5% duplicates
                metrics.append(QualityMetric(
                    name='temporal_duplicates',
                    value = duplicate_percentage,
                    threshold = 0.05,
                    severity='warning',
                    message = f'Duplicate timestamps found: {duplicate_percentage:.1%}',
                    suggested_action='Remove duplicate timestamps'
                ))
            
            # Check monotonicity
            if not timestamps.is_monotonic_increasing:
                metrics.append(QualityMetric(
                    name='temporal_monotonicity',
                    value = 0.0,
                    threshold = 1.0,
                    severity='warning',
                    message='Timestamps are not monotonically increasing',
                    suggested_action='Sort data by timestamp'
                ))
            
        except Exception as e:
            metrics.append(QualityMetric(
                name='temporal_consistency_error',
                value = 0.0,
                threshold = 1.0,
                severity='warning',
                message = f'Temporal consistency check failed: {str(e)}'
            ))
        
        return metrics
    
    def _check_price_anomaly_detection(self, data: pd.DataFrame) -> List[QualityMetric]:
        """Check for price anomalies with tolerant parameters."""
        metrics = []
        
        price_columns = ['open', 'high', 'low', 'close']
        available_price_cols = [col for col in price_columns if col in data.columns]
        
        if not available_price_cols:
            return metrics
        
        try:
            for col in available_price_cols:
                prices = data[col].dropna()
                
                if len(prices) < 10:  # Need minimum data for analysis
                    continue
                
                # Check for extreme price changes
                price_changes = prices.pct_change().dropna()
                extreme_changes = price_changes[abs(price_changes) > self.tolerant_thresholds['price_anomaly_detection']['price_change_threshold']]
                anomaly_percentage = len(extreme_changes) / len(price_changes) if len(price_changes) > 0 else 0
                
                if anomaly_percentage > self.tolerant_thresholds['price_anomaly_detection']['max_anomaly_percentage']:
                    metrics.append(QualityMetric(
                        name = f'{col}_price_anomalies',
                        value = anomaly_percentage,
                        threshold = self.tolerant_thresholds['price_anomaly_detection']['max_anomaly_percentage'],
                        severity='warning',
                        message = f'Price anomalies in {col}: {anomaly_percentage:.1%} extreme changes',
                        suggested_action='Review price data for errors'
                    ))
                
                # Check for negative prices
                negative_prices = (prices < 0).sum()
                if negative_prices > 0:
                    metrics.append(QualityMetric(
                        name = f'{col}_negative_prices',
                        value = negative_prices,
                        threshold = 0,
                        severity='critical',
                        message = f'Negative prices in {col}: {negative_prices} occurrences',
                        suggested_action='Fix price data source'
                    ))
                
                # Check for zero prices
                zero_prices = (prices == 0).sum()
                if zero_prices > len(prices) * 0.01:  # More than 1% zero prices
                    metrics.append(QualityMetric(
                        name = f'{col}_zero_prices',
                        value = zero_prices / len(prices),
                        threshold = 0.01,
                        severity='warning',
                        message = f'Zero prices in {col}: {zero_prices} occurrences',
                        suggested_action='Check for data collection issues'
                    ))
        
        except Exception as e:
            metrics.append(QualityMetric(
                name='price_anomaly_detection_error',
                value = 0.0,
                threshold = 1.0,
                severity='warning',
                message = f'Price anomaly detection failed: {str(e)}'
            ))
        
        return metrics
    
    def _check_volume_pattern_analysis(self, data: pd.DataFrame) -> List[QualityMetric]:
        """Check volume patterns with tolerant parameters."""
        metrics = []
        
        if 'volume' not in data.columns:
            return metrics
        
        try:
            volume = data['volume'].dropna()
            
            if len(volume) < 10:
                return metrics
            
            # Check for volume spikes
            volume_mean = volume.mean()
            volume_std = volume.std()
            
            if volume_std > 0:
                volume_spikes = volume[volume > volume_mean + 3 * volume_std]  # 3σ threshold
                spike_percentage = len(volume_spikes) / len(volume)
                
                if spike_percentage > 0.05:  # More than 5% spikes
                    metrics.append(QualityMetric(
                        name='volume_spikes',
                        value = spike_percentage,
                        threshold = 0.05,
                        severity='info',
                        message = f'Volume spikes detected: {spike_percentage:.1%} of data',
                        suggested_action='Review for market events or data errors'
                    ))
            
            # Check for zero volume
            zero_volume = (volume == 0).sum()
            zero_percentage = zero_volume / len(volume)
            
            if zero_percentage > self.tolerant_thresholds['volume_pattern_analysis']['zero_volume_threshold']:
                metrics.append(QualityMetric(
                    name='zero_volume',
                    value = zero_percentage,
                    threshold = self.tolerant_thresholds['volume_pattern_analysis']['zero_volume_threshold'],
                    severity='warning',
                    message = f'High zero volume percentage: {zero_percentage:.1%}',
                    suggested_action='Check for market closure or data issues'
                ))
            
            # Check for negative volume
            negative_volume = (volume < 0).sum()
            if negative_volume > 0:
                metrics.append(QualityMetric(
                    name='negative_volume',
                    value = negative_volume,
                    threshold = 0,
                    severity='critical',
                    message = f'Negative volume detected: {negative_volume} occurrences',
                    suggested_action='Fix volume data source'
                ))
        
        except Exception as e:
            metrics.append(QualityMetric(
                name='volume_pattern_analysis_error',
                value = 0.0,
                threshold = 1.0,
                severity='warning',
                message = f'Volume pattern analysis failed: {str(e)}'
            ))
        
        return metrics
    
    def _check_market_microstructure(self, data: pd.DataFrame) -> List[QualityMetric]:
        """Check market microstructure with tolerant parameters."""
        metrics = []
        
        # Check OHLC relationships
        ohlc_cols = ['open', 'high', 'low', 'close']
        available_ohlc = [col for col in ohlc_cols if col in data.columns]
        
        if len(available_ohlc) >= 4:
            try:
                # Check high >= low
                invalid_hl = (data['high'] < data['low']).sum()
                if invalid_hl > 0:
                    metrics.append(QualityMetric(
                        name='invalid_high_low',
                        value = invalid_hl,
                        threshold = 0,
                        severity='critical',
                        message = f'Invalid high < low relationships: {invalid_hl} occurrences',
                        suggested_action='Fix OHLC data source'
                    ))
                
                # Check high >= open, close
                invalid_ho = (data['high'] < data['open']).sum()
                invalid_hc = (data['high'] < data['close']).sum()
                if invalid_ho > 0 or invalid_hc > 0:
                    metrics.append(QualityMetric(
                        name='invalid_high_oc',
                        value = invalid_ho + invalid_hc,
                        threshold = 0,
                        severity='critical',
                        message = f'Invalid high < open/close relationships: {invalid_ho + invalid_hc} occurrences',
                        suggested_action='Fix OHLC data source'
                    ))
                
                # Check low <= open, close
                invalid_lo = (data['low'] > data['open']).sum()
                invalid_lc = (data['low'] > data['close']).sum()
                if invalid_lo > 0 or invalid_lc > 0:
                    metrics.append(QualityMetric(
                        name='invalid_low_oc',
                        value = invalid_lo + invalid_lc,
                        threshold = 0,
                        severity='critical',
                        message = f'Invalid low > open/close relationships: {invalid_lo + invalid_lc} occurrences',
                        suggested_action='Fix OHLC data source'
                    ))
            except Exception as e:
                metrics.append(QualityMetric(
                    name='market_microstructure_error',
                    value = 0.0,
                    threshold = 1.0,
                    severity='warning',
                    message = f'Market microstructure check failed: {str(e)}'
                ))
        
        return metrics
    
    def _check_data_completeness(self, data: pd.DataFrame) -> List[QualityMetric]:
        """Check data completeness with tolerant parameters."""
        metrics = []
        
        try:
            total_cells = data.shape[0] * data.shape[1]
            null_cells = data.isnull().sum().sum()
            null_percentage = null_cells / total_cells if total_cells > 0 else 0
            
            if null_percentage > self.tolerant_thresholds['data_completeness']['null_threshold']:
                metrics.append(QualityMetric(
                    name='data_completeness',
                    value = null_percentage,
                    threshold = self.tolerant_thresholds['data_completeness']['null_threshold'],
                    severity='warning',
                    message = f'High null percentage: {null_percentage:.1%}',
                    suggested_action='Review data collection and processing'
                ))
            
            # Check for completely empty columns
            empty_columns = data.columns[data.isnull().all()].tolist()
            if empty_columns:
                metrics.append(QualityMetric(
                    name='empty_columns',
                    value = len(empty_columns),
                    threshold = 0,
                    severity='warning',
                    message = f'Empty columns found: {empty_columns}',
                    suggested_action='Remove or populate empty columns'
                ))
            
            # Check for completely empty rows
            empty_rows = data.isnull().all(axis = 1).sum()
            if empty_rows > 0:
                metrics.append(QualityMetric(
                    name='empty_rows',
                    value = empty_rows,
                    threshold = 0,
                    severity='warning',
                    message = f'Empty rows found: {empty_rows}',
                    suggested_action='Remove empty rows'
                ))
        
        except Exception as e:
            metrics.append(QualityMetric(
                name='data_completeness_error',
                value = 0.0,
                threshold = 1.0,
                severity='warning',
                message = f'Data completeness check failed: {str(e)}'
            ))
        
        return metrics
    
    def _check_statistical_consistency(self, data: pd.DataFrame) -> List[QualityMetric]:
        """Check statistical consistency with tolerant parameters."""
        metrics = []
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            try:
                values = data[col].dropna()
                
                if len(values) < 10:
                    continue
                
                # Check for extreme skewness
                skewness = stats.skew(values)
                if abs(skewness) > self.tolerant_thresholds['statistical_consistency']['distribution_skew_threshold']:
                    metrics.append(QualityMetric(
                        name = f'{col}_skewness',
                        value = abs(skewness),
                        threshold = self.tolerant_thresholds['statistical_consistency']['distribution_skew_threshold'],
                        severity='info',
                        message = f'High skewness in {col}: {skewness:.2f}',
                        suggested_action='Consider data transformation'
                    ))
                
                # Check for extreme kurtosis
                kurtosis = stats.kurtosis(values)
                if abs(kurtosis) > self.tolerant_thresholds['statistical_consistency']['distribution_kurtosis_threshold']:
                    metrics.append(QualityMetric(
                        name = f'{col}_kurtosis',
                        value = abs(kurtosis),
                        threshold = self.tolerant_thresholds['statistical_consistency']['distribution_kurtosis_threshold'],
                        severity='info',
                        message = f'High kurtosis in {col}: {kurtosis:.2f}',
                        suggested_action='Review data distribution'
                    ))
                
                # Check for constant values
                if values.nunique() == 1:
                    metrics.append(QualityMetric(
                        name = f'{col}_constant_values',
                        value = 1.0,
                        threshold = 0,
                        severity='warning',
                        message = f'Constant values in {col}',
                        suggested_action='Check for data collection issues'
                    ))
            except Exception as e:
                metrics.append(QualityMetric(
                    name='statistical_consistency_error',
                    value = 0.0,
                    threshold = 1.0,
                    severity='warning',
                    message = f'Statistical consistency check failed: {str(e)}'
                ))
        
        return metrics
    
    def _check_basic_data_integrity(self, data: pd.DataFrame) -> List[QualityMetric]:
        """Check basic data integrity."""
        metrics = []
        
        try:
            # Check data shape
            if data.shape[0] == 0:
                metrics.append(QualityMetric(
                    name='empty_dataset',
                    value = 0,
                    threshold = 1,
                    severity='critical',
                    message='Dataset is empty',
                    suggested_action='Check data source'
                ))
            
            if data.shape[1] == 0:
                metrics.append(QualityMetric(
                    name='no_columns',
                    value = 0,
                    threshold = 1,
                    severity='critical',
                    message='Dataset has no columns',
                    suggested_action='Check data source'
                ))
            
            # Check for duplicate rows
            duplicate_rows = data.duplicated().sum()
            if duplicate_rows > len(data) * 0.1:  # More than 10% duplicates
                metrics.append(QualityMetric(
                    name='duplicate_rows',
                    value = duplicate_rows / len(data),
                    threshold = 0.1,
                    severity='warning',
                    message = f'High duplicate row percentage: {duplicate_rows / len(data):.1%}',
                    suggested_action='Remove duplicate rows'
                ))
        
        except Exception as e:
            metrics.append(QualityMetric(
                name='basic_integrity_error',
                value = 0.0,
                threshold = 1.0,
                severity='warning',
                message = f'Basic integrity check failed: {str(e)}'
            ))
        
        return metrics
    
    def _calculate_overall_score(self, metrics: List[QualityMetric]) -> float:
        """Calculate overall quality score."""
        if not metrics:
            return 100.0
        
        # Weight metrics by severity
        severity_weights = {
            'info': 0.1,
            'warning': 0.3,
            'error': 0.7,
            'critical': 1.0
        }
        
        total_penalty = 0.0
        total_weight = 0.0
        
        for metric in metrics:
            weight = severity_weights.get(metric.severity, 0.5)
            penalty = min(metric.value / metric.threshold, 1.0) * weight * 20  # Max 20 points per metric
            total_penalty += penalty
            total_weight += weight
        
        if total_weight == 0:
            return 100.0
        
        # Normalize penalty
        normalized_penalty = total_penalty / total_weight
        score = max(0.0, 100.0 - normalized_penalty)
        
        return score
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get quality assessment summary."""
        if not self.assessment_history:
            return {'assessments': 0, 'average_score': 100.0}
        
        total_assessments = len(self.assessment_history)
        average_score = np.mean([assessment.overall_score for assessment in self.assessment_history])
        total_issues = sum([assessment.issues_found for assessment in self.assessment_history])
        total_warnings = sum([assessment.warnings_found for assessment in self.assessment_history])
        total_critical = sum([assessment.critical_issues for assessment in self.assessment_history])
        
        return {
            'total_assessments': total_assessments,
            'average_score': average_score,
            'total_issues': total_issues,
            'total_warnings': total_warnings,
            'total_critical_issues': total_critical,
            'latest_assessment': self.assessment_history[-1].assessment_timestamp if self.assessment_history else None
        }
    
    def get_assessment_history(self) -> List[QualityAssessment]:
        """Get assessment history."""
        return self.assessment_history.copy()
    
    def reset_assessment_history(self):
        """Reset assessment history."""
        self.assessment_history.clear()
        self.logger.info("🔄 Quality assessment history reset")

# Global instance
advanced_quality_metrics = AdvancedQualityMetrics()