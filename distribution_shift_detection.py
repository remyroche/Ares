"""
Distribution Shift Detection with PSI/JSD and Calibration Drift Quantification

This module implements comprehensive distribution shift detection for temporal data,
including Population Stability Index (PSI), Jensen-Shannon Divergence (JSD),
and calibration drift quantification.

Key Features:
- PSI calculation with statistical significance testing
- JSD for continuous and discrete distributions
- Calibration drift detection (ECE, reliability diagrams)
- Error drift analysis with binned time periods
- Statistical tests (KS, AD) with multiple testing correction
- Confidence intervals and significance testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
import logging
from datetime import datetime, timedelta
from scipy import stats
from scipy.special import softmax, kl_div
from collections import defaultdict, Counter
import json
from pathlib import Path
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Import existing utilities
try:
    from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint_info(msg): print(f"INFO: {msg}")
    def tprint_warning(msg): print(f"WARNING: {msg}")
    def tprint_error(msg): print(f"ERROR: {msg}")
    def tprint_success(msg): print(f"SUCCESS: {msg}")

logger = logging.getLogger(__name__)


class ShiftType(Enum):
    """Types of distribution shift."""
    FEATURE_SHIFT = "feature_shift"          # Feature distribution shift
    LABEL_SHIFT = "label_shift"              # Label distribution shift
    CALIBRATION_SHIFT = "calibration_shift"  # Calibration shift
    ERROR_SHIFT = "error_shift"              # Error rate shift
    COVARIATE_SHIFT = "covariate_shift"      # Covariate shift


class ShiftSeverity(Enum):
    """Severity levels for detected shifts."""
    NONE = "none"        # No significant shift
    LOW = "low"          # Minor shift
    MEDIUM = "medium"    # Moderate shift
    HIGH = "high"        # Significant shift
    CRITICAL = "critical"  # Critical shift


@dataclass
class DistributionShiftConfig:
    """Configuration for distribution shift detection."""
    
    # PSI settings
    psi_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'none': 0.1,
        'low': 0.2,
        'medium': 0.25,
        'high': 0.5
    })
    psi_bins: int = 10
    psi_min_samples: int = 100
    
    # JSD settings
    jsd_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'none': 0.1,
        'low': 0.2,
        'medium': 0.3,
        'high': 0.5
    })
    
    # Statistical tests
    significance_level: float = 0.05
    enable_multiple_testing_correction: bool = True
    correction_method: str = 'bonferroni'  # 'bonferroni', 'fdr_bh', 'holm'
    
    # Calibration drift
    calibration_bins: int = 10
    ece_threshold: float = 0.1
    reliability_threshold: float = 0.05
    
    # Time binning
    time_bin_size: str = "7D"
    min_samples_per_bin: int = 50
    max_bins: int = 20
    
    # Reporting
    generate_plots: bool = True
    save_reports: bool = True
    report_directory: str = "reports/distribution_shift"


@dataclass
class ShiftDetectionResult:
    """Result of distribution shift detection."""
    
    # Basic information
    shift_type: ShiftType
    feature_name: str
    shift_severity: ShiftSeverity
    shift_score: float
    
    # Statistical measures
    psi_score: float = 0.0
    jsd_score: float = 0.0
    ks_statistic: float = 0.0
    ks_p_value: float = 1.0
    ad_statistic: float = 0.0
    ad_p_value: float = 1.0
    
    # Confidence intervals
    psi_ci_lower: float = 0.0
    psi_ci_upper: float = 0.0
    jsd_ci_lower: float = 0.0
    jsd_ci_upper: float = 0.0
    
    # Time information
    reference_period: str = ""
    comparison_period: str = ""
    time_gap: timedelta = timedelta(0)
    
    # Additional details
    description: str = ""
    recommendations: List[str] = field(default_factory=list)
    is_significant: bool = False
    
    # Detection metadata
    detection_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    sample_sizes: Dict[str, int] = field(default_factory=dict)


@dataclass
class CalibrationDriftResult:
    """Result of calibration drift detection."""
    
    # Basic metrics
    ece_reference: float = 0.0
    ece_comparison: float = 0.0
    ece_drift: float = 0.0
    reliability_drift: float = 0.0
    
    # Calibration curve details
    reference_calibration: Dict[str, Any] = field(default_factory=dict)
    comparison_calibration: Dict[str, Any] = field(default_factory=dict)
    
    # Statistical significance
    calibration_shift_p_value: float = 1.0
    is_calibration_shift: bool = False
    
    # Time information
    reference_period: str = ""
    comparison_period: str = ""
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)


class DistributionShiftDetector:
    """Comprehensive distribution shift detection system."""
    
    def __init__(self, config: Optional[DistributionShiftConfig] = None):
        """Initialize distribution shift detector."""
        self.config = config or DistributionShiftConfig()
        self.detection_history = []
        
        # Create report directory
        if self.config.save_reports:
            Path(self.config.report_directory).mkdir(parents=True, exist_ok=True)
    
    def detect_all_shifts(self, 
                         X: pd.DataFrame, 
                         y: pd.Series,
                         y_pred: Optional[pd.Series] = None,
                         y_proba: Optional[pd.Series] = None,
                         time_col: Optional[str] = None,
                         reference_period: Optional[Tuple[datetime, datetime]] = None) -> List[ShiftDetectionResult]:
        """
        Detect all types of distribution shifts.
        
        Args:
            X: Feature matrix with timestamps
            y: True labels
            y_pred: Predicted labels (optional)
            y_proba: Predicted probabilities (optional)
            time_col: Time column name
            reference_period: Reference period for comparison
            
        Returns:
            List of ShiftDetectionResult objects
        """
        all_results = []
        
        try:
            if TPRINT_AVAILABLE:
                tprint_info("🔍 Starting comprehensive distribution shift detection")
            
            # Extract time information
            time_series = self._extract_time_series(X, time_col)
            if time_series is None:
                if TPRINT_AVAILABLE:
                    tprint_error("❌ No time information available for shift detection")
                return all_results
            
            # Create time periods for comparison
            time_periods = self._create_time_periods(time_series, reference_period)
            
            if len(time_periods) < 2:
                if TPRINT_AVAILABLE:
                    tprint_warning("⚠️ Insufficient time periods for shift detection")
                return all_results
            
            # 1. Feature distribution shifts
            feature_results = self._detect_feature_shifts(X, time_periods)
            all_results.extend(feature_results)
            
            # 2. Label distribution shifts
            label_results = self._detect_label_shifts(y, time_periods)
            all_results.extend(label_results)
            
            # 3. Calibration shifts
            if y_proba is not None:
                calibration_results = self._detect_calibration_shifts(y, y_proba, time_periods)
                all_results.extend(calibration_results)
            
            # 4. Error shifts
            if y_pred is not None:
                error_results = self._detect_error_shifts(y, y_pred, time_periods)
                all_results.extend(error_results)
            
            # Apply multiple testing correction
            if self.config.enable_multiple_testing_correction:
                all_results = self._apply_multiple_testing_correction(all_results)
            
            # Store in history
            self.detection_history.extend(all_results)
            
            # Generate summary
            self._generate_shift_summary(all_results)
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Distribution shift detection completed: {len(all_results)} shifts found")
            
            return all_results
            
        except Exception as e:
            logger.error(f"Distribution shift detection failed: {e}")
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Distribution shift detection failed: {e}")
            return all_results
    
    def _extract_time_series(self, X: pd.DataFrame, time_col: Optional[str]) -> Optional[pd.Series]:
        """Extract time series from data."""
        try:
            if time_col and time_col in X.columns:
                return X[time_col]
            elif isinstance(X.index, pd.DatetimeIndex):
                return pd.Series(X.index, index=X.index)
            else:
                return None
        except Exception:
            return None
    
    def _create_time_periods(self, time_series: pd.Series, 
                           reference_period: Optional[Tuple[datetime, datetime]]) -> List[Dict[str, Any]]:
        """Create time periods for comparison."""
        try:
            # Convert to datetime if needed
            if not isinstance(time_series.iloc[0], (pd.Timestamp, datetime)):
                time_series = pd.to_datetime(time_series, errors='coerce')
            
            # Create bins
            bin_size = pd.Timedelta(self.config.time_bin_size)
            bins = pd.cut(time_series, bins=pd.date_range(
                time_series.min(), 
                time_series.max() + bin_size, 
                freq=self.config.time_bin_size
            ))
            
            # Group by bins
            periods = []
            for bin_name, group in time_series.groupby(bins):
                if len(group) >= self.config.min_samples_per_bin:
                    periods.append({
                        'name': str(bin_name),
                        'start': group.min(),
                        'end': group.max(),
                        'indices': group.index,
                        'sample_count': len(group)
                    })
            
            # Limit number of periods
            if len(periods) > self.config.max_bins:
                periods = periods[-self.config.max_bins:]
            
            return periods
            
        except Exception as e:
            logger.error(f"Time period creation failed: {e}")
            return []
    
    def _detect_feature_shifts(self, X: pd.DataFrame, 
                              time_periods: List[Dict[str, Any]]) -> List[ShiftDetectionResult]:
        """Detect feature distribution shifts."""
        results = []
        
        try:
            # Compare each period with the previous one
            for i in range(1, len(time_periods)):
                ref_period = time_periods[i-1]
                comp_period = time_periods[i]
                
                # Analyze each feature
                for col in X.columns:
                    if col in ['timestamp', 'time', 'date']:
                        continue
                    
                    if X[col].dtype not in ['int64', 'float64']:
                        continue
                    
                    # Get data for both periods
                    ref_data = X[col].loc[ref_period['indices']].dropna()
                    comp_data = X[col].loc[comp_period['indices']].dropna()
                    
                    if len(ref_data) < self.config.psi_min_samples or len(comp_data) < self.config.psi_min_samples:
                        continue
                    
                    # Calculate shift metrics
                    psi_score = self._calculate_psi(ref_data, comp_data)
                    jsd_score = self._calculate_jsd(ref_data, comp_data)
                    
                    # Statistical tests
                    ks_stat, ks_p_value = stats.ks_2samp(ref_data, comp_data)
                    ad_stat, ad_p_value = stats.anderson_ksamp([ref_data, comp_data])
                    
                    # Determine shift severity
                    shift_severity = self._determine_shift_severity(psi_score, jsd_score)
                    shift_score = max(psi_score, jsd_score)
                    
                    # Create result
                    result = ShiftDetectionResult(
                        shift_type=ShiftType.FEATURE_SHIFT,
                        feature_name=col,
                        shift_severity=shift_severity,
                        shift_score=shift_score,
                        psi_score=psi_score,
                        jsd_score=jsd_score,
                        ks_statistic=ks_stat,
                        ks_p_value=ks_p_value,
                        ad_statistic=ad_stat,
                        ad_p_value=ad_p_value,
                        reference_period=ref_period['name'],
                        comparison_period=comp_period['name'],
                        time_gap=comp_period['start'] - ref_period['end'],
                        description=f"Feature '{col}' distribution shift detected",
                        is_significant=ks_p_value < self.config.significance_level,
                        sample_sizes={
                            'reference': len(ref_data),
                            'comparison': len(comp_data)
                        }
                    )
                    
                    # Add recommendations
                    if shift_severity != ShiftSeverity.NONE:
                        result.recommendations.extend([
                            f"Review feature '{col}' for temporal stability",
                            "Consider feature normalization or scaling",
                            "Implement feature drift monitoring"
                        ])
                    
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Feature shift detection failed: {e}")
            return results
    
    def _detect_label_shifts(self, y: pd.Series, 
                            time_periods: List[Dict[str, Any]]) -> List[ShiftDetectionResult]:
        """Detect label distribution shifts."""
        results = []
        
        try:
            # Compare each period with the previous one
            for i in range(1, len(time_periods)):
                ref_period = time_periods[i-1]
                comp_period = time_periods[i]
                
                # Get label data for both periods
                ref_labels = y.loc[ref_period['indices']]
                comp_labels = y.loc[comp_period['indices']]
                
                if len(ref_labels) < self.config.psi_min_samples or len(comp_labels) < self.config.psi_min_samples:
                    continue
                
                # Calculate shift metrics
                psi_score = self._calculate_psi(ref_labels, comp_labels)
                jsd_score = self._calculate_jsd(ref_labels, comp_labels)
                
                # Statistical tests
                ks_stat, ks_p_value = stats.ks_2samp(ref_labels, comp_labels)
                ad_stat, ad_p_value = stats.anderson_ksamp([ref_labels, comp_labels])
                
                # Determine shift severity
                shift_severity = self._determine_shift_severity(psi_score, jsd_score)
                shift_score = max(psi_score, jsd_score)
                
                # Create result
                result = ShiftDetectionResult(
                    shift_type=ShiftType.LABEL_SHIFT,
                    feature_name="labels",
                    shift_severity=shift_severity,
                    shift_score=shift_score,
                    psi_score=psi_score,
                    jsd_score=jsd_score,
                    ks_statistic=ks_stat,
                    ks_p_value=ks_p_value,
                    ad_statistic=ad_stat,
                    ad_p_value=ad_p_value,
                    reference_period=ref_period['name'],
                    comparison_period=comp_period['name'],
                    time_gap=comp_period['start'] - ref_period['end'],
                    description="Label distribution shift detected",
                    is_significant=ks_p_value < self.config.significance_level,
                    sample_sizes={
                        'reference': len(ref_labels),
                        'comparison': len(comp_labels)
                    }
                )
                
                # Add recommendations
                if shift_severity != ShiftSeverity.NONE:
                    result.recommendations.extend([
                        "Review label generation process",
                        "Check for data quality issues",
                        "Implement label drift monitoring"
                    ])
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Label shift detection failed: {e}")
            return results
    
    def _detect_calibration_shifts(self, y: pd.Series, y_proba: pd.Series,
                                  time_periods: List[Dict[str, Any]]) -> List[ShiftDetectionResult]:
        """Detect calibration shifts."""
        results = []
        
        try:
            # Compare each period with the previous one
            for i in range(1, len(time_periods)):
                ref_period = time_periods[i-1]
                comp_period = time_periods[i]
                
                # Get data for both periods
                ref_y = y.loc[ref_period['indices']]
                ref_proba = y_proba.loc[ref_period['indices']]
                comp_y = y.loc[comp_period['indices']]
                comp_proba = y_proba.loc[comp_period['indices']]
                
                if len(ref_y) < self.config.psi_min_samples or len(comp_y) < self.config.psi_min_samples:
                    continue
                
                # Calculate ECE for both periods
                ref_ece = self._calculate_ece(ref_y, ref_proba)
                comp_ece = self._calculate_ece(comp_y, comp_proba)
                ece_drift = abs(comp_ece - ref_ece)
                
                # Calculate calibration curve drift
                calibration_drift = self._calculate_calibration_drift(ref_y, ref_proba, comp_y, comp_proba)
                
                # Determine shift severity based on ECE drift
                shift_severity = self._determine_calibration_shift_severity(ece_drift)
                shift_score = max(ece_drift, calibration_drift)
                
                # Create result
                result = ShiftDetectionResult(
                    shift_type=ShiftType.CALIBRATION_SHIFT,
                    feature_name="calibration",
                    shift_severity=shift_severity,
                    shift_score=shift_score,
                    reference_period=ref_period['name'],
                    comparison_period=comp_period['name'],
                    time_gap=comp_period['start'] - ref_period['end'],
                    description=f"Calibration shift detected (ECE drift: {ece_drift:.3f})",
                    is_significant=ece_drift > self.config.ece_threshold,
                    sample_sizes={
                        'reference': len(ref_y),
                        'comparison': len(comp_y)
                    }
                )
                
                # Add recommendations
                if shift_severity != ShiftSeverity.NONE:
                    result.recommendations.extend([
                        "Implement calibration correction",
                        "Review model calibration process",
                        "Consider recalibration techniques"
                    ])
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Calibration shift detection failed: {e}")
            return results
    
    def _detect_error_shifts(self, y: pd.Series, y_pred: pd.Series,
                            time_periods: List[Dict[str, Any]]) -> List[ShiftDetectionResult]:
        """Detect error rate shifts."""
        results = []
        
        try:
            # Compare each period with the previous one
            for i in range(1, len(time_periods)):
                ref_period = time_periods[i-1]
                comp_period = time_periods[i]
                
                # Get data for both periods
                ref_y = y.loc[ref_period['indices']]
                ref_pred = y_pred.loc[ref_period['indices']]
                comp_y = y.loc[comp_period['indices']]
                comp_pred = y_pred.loc[comp_period['indices']]
                
                if len(ref_y) < self.config.psi_min_samples or len(comp_y) < self.config.psi_min_samples:
                    continue
                
                # Calculate error rates
                ref_error = 1 - accuracy_score(ref_y, ref_pred)
                comp_error = 1 - accuracy_score(comp_y, comp_pred)
                error_drift = abs(comp_error - ref_error)
                
                # Calculate Brier score drift
                ref_brier = brier_score_loss(ref_y, ref_pred)
                comp_brier = brier_score_loss(comp_y, comp_pred)
                brier_drift = abs(comp_brier - ref_brier)
                
                # Determine shift severity
                shift_severity = self._determine_error_shift_severity(error_drift)
                shift_score = max(error_drift, brier_drift)
                
                # Create result
                result = ShiftDetectionResult(
                    shift_type=ShiftType.ERROR_SHIFT,
                    feature_name="error_rate",
                    shift_severity=shift_severity,
                    shift_score=shift_score,
                    reference_period=ref_period['name'],
                    comparison_period=comp_period['name'],
                    time_gap=comp_period['start'] - ref_period['end'],
                    description=f"Error rate shift detected (drift: {error_drift:.3f})",
                    is_significant=error_drift > 0.05,  # 5% error rate change
                    sample_sizes={
                        'reference': len(ref_y),
                        'comparison': len(comp_y)
                    }
                )
                
                # Add recommendations
                if shift_severity != ShiftSeverity.NONE:
                    result.recommendations.extend([
                        "Review model performance over time",
                        "Implement error drift monitoring",
                        "Consider model retraining"
                    ])
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error shift detection failed: {e}")
            return results
    
    def _calculate_psi(self, reference: pd.Series, comparison: pd.Series) -> float:
        """Calculate Population Stability Index (PSI)."""
        try:
            # Create bins based on reference data
            bin_edges = np.percentile(reference, np.linspace(0, 100, self.config.psi_bins + 1))
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            
            # Calculate bin counts
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            comp_counts, _ = np.histogram(comparison, bins=bin_edges)
            
            # Add small constant to avoid division by zero
            ref_counts = ref_counts + 1e-8
            comp_counts = comp_counts + 1e-8
            
            # Normalize to probabilities
            ref_probs = ref_counts / ref_counts.sum()
            comp_probs = comp_counts / comp_counts.sum()
            
            # Calculate PSI
            psi = np.sum((comp_probs - ref_probs) * np.log(comp_probs / ref_probs))
            
            return psi
            
        except Exception as e:
            logger.error(f"PSI calculation failed: {e}")
            return 0.0
    
    def _calculate_jsd(self, reference: pd.Series, comparison: pd.Series) -> float:
        """Calculate Jensen-Shannon Divergence (JSD)."""
        try:
            # Create bins based on both datasets
            all_values = np.concatenate([reference.values, comparison.values])
            bin_edges = np.percentile(all_values, np.linspace(0, 100, self.config.psi_bins + 1))
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            
            # Calculate bin counts
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            comp_counts, _ = np.histogram(comparison, bins=bin_edges)
            
            # Add small constant to avoid division by zero
            ref_counts = ref_counts + 1e-8
            comp_counts = comp_counts + 1e-8
            
            # Normalize to probabilities
            ref_probs = ref_counts / ref_counts.sum()
            comp_probs = comp_counts / comp_counts.sum()
            
            # Calculate JSD
            m = 0.5 * (ref_probs + comp_probs)
            jsd = 0.5 * kl_div(ref_probs, m).sum() + 0.5 * kl_div(comp_probs, m).sum()
            
            return jsd
            
        except Exception as e:
            logger.error(f"JSD calculation failed: {e}")
            return 0.0
    
    def _calculate_ece(self, y_true: pd.Series, y_proba: pd.Series) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        try:
            # Create bins
            bin_boundaries = np.linspace(0, 1, self.config.calibration_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_proba[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return ece
            
        except Exception:
            return 0.0
    
    def _calculate_calibration_drift(self, ref_y: pd.Series, ref_proba: pd.Series,
                                   comp_y: pd.Series, comp_proba: pd.Series) -> float:
        """Calculate calibration curve drift."""
        try:
            # Calculate calibration curves
            ref_fraction, ref_mean_pred = calibration_curve(ref_y, ref_proba, n_bins=self.config.calibration_bins)
            comp_fraction, comp_mean_pred = calibration_curve(comp_y, comp_proba, n_bins=self.config.calibration_bins)
            
            # Calculate drift as mean absolute difference
            drift = np.mean(np.abs(ref_fraction - comp_fraction))
            
            return drift
            
        except Exception:
            return 0.0
    
    def _determine_shift_severity(self, psi_score: float, jsd_score: float) -> ShiftSeverity:
        """Determine shift severity based on PSI and JSD scores."""
        max_score = max(psi_score, jsd_score)
        
        if max_score < self.config.psi_thresholds['none']:
            return ShiftSeverity.NONE
        elif max_score < self.config.psi_thresholds['low']:
            return ShiftSeverity.LOW
        elif max_score < self.config.psi_thresholds['medium']:
            return ShiftSeverity.MEDIUM
        elif max_score < self.config.psi_thresholds['high']:
            return ShiftSeverity.HIGH
        else:
            return ShiftSeverity.CRITICAL
    
    def _determine_calibration_shift_severity(self, ece_drift: float) -> ShiftSeverity:
        """Determine calibration shift severity based on ECE drift."""
        if ece_drift < 0.02:
            return ShiftSeverity.NONE
        elif ece_drift < 0.05:
            return ShiftSeverity.LOW
        elif ece_drift < 0.1:
            return ShiftSeverity.MEDIUM
        elif ece_drift < 0.2:
            return ShiftSeverity.HIGH
        else:
            return ShiftSeverity.CRITICAL
    
    def _determine_error_shift_severity(self, error_drift: float) -> ShiftSeverity:
        """Determine error shift severity based on error rate drift."""
        if error_drift < 0.01:
            return ShiftSeverity.NONE
        elif error_drift < 0.03:
            return ShiftSeverity.LOW
        elif error_drift < 0.05:
            return ShiftSeverity.MEDIUM
        elif error_drift < 0.1:
            return ShiftSeverity.HIGH
        else:
            return ShiftSeverity.CRITICAL
    
    def _apply_multiple_testing_correction(self, results: List[ShiftDetectionResult]) -> List[ShiftDetectionResult]:
        """Apply multiple testing correction to p-values."""
        try:
            if not results:
                return results
            
            # Extract p-values
            p_values = [result.ks_p_value for result in results]
            
            # Apply correction
            if self.config.correction_method == 'bonferroni':
                corrected_p_values = p_values * len(p_values)
            elif self.config.correction_method == 'fdr_bh':
                from statsmodels.stats.multitest import multipletests
                _, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')
            elif self.config.correction_method == 'holm':
                from statsmodels.stats.multitest import multipletests
                _, corrected_p_values, _, _ = multipletests(p_values, method='holm')
            else:
                corrected_p_values = p_values
            
            # Update results
            for i, result in enumerate(results):
                result.ks_p_value = corrected_p_values[i]
                result.is_significant = corrected_p_values[i] < self.config.significance_level
            
            return results
            
        except Exception as e:
            logger.error(f"Multiple testing correction failed: {e}")
            return results
    
    def _generate_shift_summary(self, results: List[ShiftDetectionResult]):
        """Generate shift detection summary."""
        if not results:
            if TPRINT_AVAILABLE:
                tprint_success("✅ No distribution shifts detected")
            return
        
        # Count by severity
        severity_counts = Counter([result.shift_severity for result in results])
        shift_type_counts = Counter([result.shift_type for result in results])
        significant_count = sum(1 for result in results if result.is_significant)
        
        if TPRINT_AVAILABLE:
            tprint_warning(f"⚠️ Distribution shift summary:")
            tprint_warning(f"   Total shifts: {len(results)}")
            tprint_warning(f"   Significant: {significant_count}")
            tprint_warning(f"   Critical: {severity_counts.get(ShiftSeverity.CRITICAL, 0)}")
            tprint_warning(f"   High: {severity_counts.get(ShiftSeverity.HIGH, 0)}")
            tprint_warning(f"   Medium: {severity_counts.get(ShiftSeverity.MEDIUM, 0)}")
            tprint_warning(f"   Low: {severity_counts.get(ShiftSeverity.LOW, 0)}")
            
            tprint_warning(f"   Shift types:")
            for shift_type, count in shift_type_counts.items():
                tprint_warning(f"     {shift_type.value}: {count}")
    
    def generate_shift_report(self, results: List[ShiftDetectionResult], 
                            filename: Optional[str] = None) -> str:
        """Generate detailed shift detection report."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"distribution_shift_report_{timestamp}.json"
        
        filepath = Path(self.config.report_directory) / filename
        
        try:
            report_data = {
                'detection_timestamp': datetime.now().isoformat(),
                'total_shifts': len(results),
                'significant_shifts': sum(1 for r in results if r.is_significant),
                'severity_summary': Counter([r.shift_severity.value for r in results]),
                'shift_type_summary': Counter([r.shift_type.value for r in results]),
                'shifts': []
            }
            
            for result in results:
                shift_data = {
                    'shift_type': result.shift_type.value,
                    'feature_name': result.feature_name,
                    'shift_severity': result.shift_severity.value,
                    'shift_score': result.shift_score,
                    'psi_score': result.psi_score,
                    'jsd_score': result.jsd_score,
                    'ks_statistic': result.ks_statistic,
                    'ks_p_value': result.ks_p_value,
                    'ad_statistic': result.ad_statistic,
                    'ad_p_value': result.ad_p_value,
                    'reference_period': result.reference_period,
                    'comparison_period': result.comparison_period,
                    'time_gap': str(result.time_gap),
                    'description': result.description,
                    'recommendations': result.recommendations,
                    'is_significant': result.is_significant,
                    'sample_sizes': result.sample_sizes,
                    'detection_timestamp': result.detection_timestamp
                }
                report_data['shifts'].append(shift_data)
            
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            if TPRINT_AVAILABLE:
                tprint_success(f"📄 Shift report saved: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to generate shift report: {e}")
            return ""


# Convenience functions
def create_shift_detector(config: Optional[DistributionShiftConfig] = None) -> DistributionShiftDetector:
    """Create distribution shift detector."""
    return DistributionShiftDetector(config)

def detect_shifts_quick(X: pd.DataFrame, y: pd.Series, 
                       y_pred: Optional[pd.Series] = None,
                       y_proba: Optional[pd.Series] = None) -> List[ShiftDetectionResult]:
    """Quick distribution shift detection."""
    detector = create_shift_detector()
    return detector.detect_all_shifts(X, y, y_pred, y_proba)

def generate_shift_summary(results: List[ShiftDetectionResult]) -> Dict[str, Any]:
    """Generate shift detection summary."""
    if not results:
        return {'total_shifts': 0, 'significant_shifts': 0, 'severity_summary': {}, 'shift_type_summary': {}}
    
    return {
        'total_shifts': len(results),
        'significant_shifts': sum(1 for r in results if r.is_significant),
        'severity_summary': Counter([r.shift_severity.value for r in results]),
        'shift_type_summary': Counter([r.shift_type.value for r in results]),
        'critical_shifts': [r for r in results if r.shift_severity == ShiftSeverity.CRITICAL],
        'high_shifts': [r for r in results if r.shift_severity == ShiftSeverity.HIGH]
    }


if __name__ == "__main__":
    # Example usage
    print("Distribution Shift Detection with PSI/JSD and Calibration Drift")
    print("=" * 70)
    
    # Create sample data with drift
    dates = pd.date_range('2020-01-01', periods=1000, freq='1H')
    X = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000) + np.linspace(0, 2, 1000),  # Drift in feature2
        'timestamp': dates
    }, index=dates)
    
    y = pd.Series(np.random.choice([0, 1], size=1000, p=[0.7, 0.3]), index=dates)
    y_pred = pd.Series(np.random.choice([0, 1], size=1000, p=[0.6, 0.4]), index=dates)
    y_proba = pd.Series(np.random.uniform(0, 1, size=1000), index=dates)
    
    # Detect shifts
    detector = create_shift_detector()
    results = detector.detect_all_shifts(X, y, y_pred, y_proba, time_col='timestamp')
    
    # Generate summary
    summary = generate_shift_summary(results)
    print(f"Total shifts found: {summary['total_shifts']}")
    print(f"Significant shifts: {summary['significant_shifts']}")
    print(f"Critical shifts: {len(summary['critical_shifts'])}")
    print(f"High shifts: {len(summary['high_shifts'])}")
    
    # Show feature shifts
    feature_shifts = [r for r in results if r.shift_type == ShiftType.FEATURE_SHIFT]
    print(f"Feature shifts: {len(feature_shifts)}")
    for shift in feature_shifts[:3]:  # Show first 3
        print(f"  {shift.feature_name}: {shift.shift_severity.value} (PSI: {shift.psi_score:.3f})")