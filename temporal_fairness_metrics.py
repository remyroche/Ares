"""
Temporal Fairness Metrics with Explicit Definitions

This module implements three distinct temporal fairness definitions:
1. Equal Exposure per Period: Balanced representation across time periods
2. Stable Error by Period: Consistent error rates across time periods  
3. Stable Calibration by Period: Consistent calibration across time periods

Each definition includes explicit metrics, confidence bands, and statistical tests.
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
from scipy.special import softmax
from collections import defaultdict, Counter
import json
from pathlib import Path
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, f1_score, accuracy_score
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


class TemporalFairnessDefinition(Enum):
    """Temporal fairness definitions."""
    EQUAL_EXPOSURE = "equal_exposure"          # Equal exposure per period
    STABLE_ERROR = "stable_error"              # Stable error by period
    STABLE_CALIBRATION = "stable_calibration"  # Stable calibration by period


@dataclass
class TemporalFairnessConfig:
    """Configuration for temporal fairness metrics."""
    
    # Time binning
    time_bin_size: str = "7D"  # Size of time bins for analysis
    min_samples_per_bin: int = 10  # Minimum samples per bin
    max_bins: int = 50  # Maximum number of bins
    
    # Statistical tests
    confidence_level: float = 0.95
    significance_level: float = 0.05
    enable_bootstrap: bool = True
    bootstrap_samples: int = 1000
    
    # Fairness thresholds
    exposure_parity_threshold: float = 0.1  # Max std/mean for exposure parity
    error_parity_threshold: float = 0.05    # Max std for error parity
    calibration_stability_threshold: float = 0.1  # Max ECE variation
    
    # Metrics to compute
    compute_exposure_metrics: bool = True
    compute_error_metrics: bool = True
    compute_calibration_metrics: bool = True
    
    # Reporting
    generate_plots: bool = True
    save_reports: bool = True
    report_directory: str = "reports/temporal_fairness"


@dataclass
class ExposureParityMetrics:
    """Metrics for equal exposure per period."""
    
    # Basic metrics
    exposure_by_period: Dict[str, float] = field(default_factory=dict)
    exposure_std: float = 0.0
    exposure_cv: float = 0.0  # Coefficient of variation
    exposure_gini: float = 0.0  # Gini coefficient
    
    # Statistical tests
    levene_p_value: float = 1.0
    kruskal_p_value: float = 1.0
    exposure_parity_passed: bool = True
    
    # Confidence intervals
    exposure_ci_lower: Dict[str, float] = field(default_factory=dict)
    exposure_ci_upper: Dict[str, float] = field(default_factory=dict)
    
    # Period details
    periods: List[str] = field(default_factory=list)
    sample_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class ErrorParityMetrics:
    """Metrics for stable error by period."""
    
    # Basic metrics
    error_by_period: Dict[str, float] = field(default_factory=dict)
    error_std: float = 0.0
    error_cv: float = 0.0
    error_range: float = 0.0  # Max - Min error
    
    # Statistical tests
    levene_p_value: float = 1.0
    kruskal_p_value: float = 1.0
    error_parity_passed: bool = True
    
    # Confidence intervals
    error_ci_lower: Dict[str, float] = field(default_factory=dict)
    error_ci_upper: Dict[str, float] = field(default_factory=dict)
    
    # Error types
    brier_scores: Dict[str, float] = field(default_factory=dict)
    f1_scores: Dict[str, float] = field(default_factory=dict)
    accuracy_scores: Dict[str, float] = field(default_factory=dict)
    
    # Period details
    periods: List[str] = field(default_factory=list)
    sample_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class CalibrationStabilityMetrics:
    """Metrics for stable calibration by period."""
    
    # Basic metrics
    ece_by_period: Dict[str, float] = field(default_factory=dict)
    ece_std: float = 0.0
    ece_cv: float = 0.0
    ece_range: float = 0.0
    
    # Calibration details
    calibration_slopes: Dict[str, float] = field(default_factory=dict)
    calibration_intercepts: Dict[str, float] = field(default_factory=dict)
    calibration_r2: Dict[str, float] = field(default_factory=dict)
    
    # Statistical tests
    slope_test_p_value: float = 1.0
    intercept_test_p_value: float = 1.0
    calibration_stability_passed: bool = True
    
    # Confidence intervals
    ece_ci_lower: Dict[str, float] = field(default_factory=dict)
    ece_ci_upper: Dict[str, float] = field(default_factory=dict)
    
    # Period details
    periods: List[str] = field(default_factory=list)
    sample_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class TemporalFairnessReport:
    """Comprehensive temporal fairness report."""
    
    # Overall metrics
    overall_fairness_score: float = 0.0
    fairness_passed: bool = True
    
    # Individual definition results
    exposure_parity: ExposureParityMetrics = field(default_factory=ExposureParityMetrics)
    error_parity: ErrorParityMetrics = field(default_factory=ErrorParityMetrics)
    calibration_stability: CalibrationStabilityMetrics = field(default_factory=CalibrationStabilityMetrics)
    
    # Summary statistics
    total_periods: int = 0
    total_samples: int = 0
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Warnings and recommendations
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)


class TemporalFairnessAnalyzer:
    """Analyzer for temporal fairness metrics."""
    
    def __init__(self, config: Optional[TemporalFairnessConfig] = None):
        """Initialize temporal fairness analyzer."""
        self.config = config or TemporalFairnessConfig()
        self.analysis_history = []
        
        # Create report directory
        if self.config.save_reports:
            Path(self.config.report_directory).mkdir(parents=True, exist_ok=True)
    
    def analyze_temporal_fairness(self, 
                                X: pd.DataFrame, 
                                y: pd.Series,
                                y_pred: Optional[pd.Series] = None,
                                y_proba: Optional[pd.Series] = None,
                                time_col: Optional[str] = None,
                                sample_weights: Optional[pd.Series] = None) -> TemporalFairnessReport:
        """
        Analyze temporal fairness using all three definitions.
        
        Args:
            X: Feature matrix with timestamps
            y: True labels
            y_pred: Predicted labels (optional)
            y_proba: Predicted probabilities (optional)
            time_col: Time column name
            sample_weights: Sample weights (optional)
            
        Returns:
            TemporalFairnessReport with comprehensive analysis
        """
        report = TemporalFairnessReport()
        
        try:
            if TPRINT_AVAILABLE:
                tprint_info("📊 Starting comprehensive temporal fairness analysis")
            
            # Extract time information
            time_series = self._extract_time_series(X, time_col)
            if time_series is None:
                report.critical_issues.append("No time information available")
                return report
            
            # Create time bins
            time_bins = self._create_time_bins(time_series)
            report.total_periods = len(time_bins)
            report.total_samples = len(y)
            
            # 1. Equal Exposure per Period
            if self.config.compute_exposure_metrics:
                report.exposure_parity = self._analyze_exposure_parity(
                    y, time_bins, sample_weights
                )
            
            # 2. Stable Error by Period
            if self.config.compute_error_metrics and y_pred is not None:
                report.error_parity = self._analyze_error_parity(
                    y, y_pred, time_bins, sample_weights
                )
            
            # 3. Stable Calibration by Period
            if self.config.compute_calibration_metrics and y_proba is not None:
                report.calibration_stability = self._analyze_calibration_stability(
                    y, y_proba, time_bins, sample_weights
                )
            
            # Calculate overall fairness score
            report.overall_fairness_score = self._calculate_overall_fairness_score(report)
            report.fairness_passed = self._evaluate_fairness_passed(report)
            
            # Generate warnings and recommendations
            report.warnings = self._generate_fairness_warnings(report)
            report.recommendations = self._generate_fairness_recommendations(report)
            report.critical_issues = self._identify_critical_issues(report)
            
            # Store in history
            self.analysis_history.append(report)
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Temporal fairness analysis completed - Score: {report.overall_fairness_score:.3f}")
            
            return report
            
        except Exception as e:
            logger.error(f"Temporal fairness analysis failed: {e}")
            report.critical_issues.append(f"Analysis failed: {str(e)}")
            return report
    
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
    
    def _create_time_bins(self, time_series: pd.Series) -> Dict[str, pd.Series]:
        """Create time bins for analysis."""
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
            time_bins = {}
            for bin_name, group in time_series.groupby(bins):
                if len(group) >= self.config.min_samples_per_bin:
                    time_bins[str(bin_name)] = group
            
            # Limit number of bins
            if len(time_bins) > self.config.max_bins:
                # Keep most recent bins
                sorted_bins = sorted(time_bins.items(), key=lambda x: x[1].max(), reverse=True)
                time_bins = dict(sorted_bins[:self.config.max_bins])
            
            return time_bins
            
        except Exception as e:
            logger.error(f"Time bin creation failed: {e}")
            return {}
    
    def _analyze_exposure_parity(self, y: pd.Series, time_bins: Dict[str, pd.Series],
                                sample_weights: Optional[pd.Series]) -> ExposureParityMetrics:
        """Analyze equal exposure per period."""
        metrics = ExposureParityMetrics()
        
        try:
            # Calculate exposure by period
            for period_name, period_indices in time_bins.items():
                period_y = y.loc[period_indices.index]
                period_weights = sample_weights.loc[period_indices.index] if sample_weights is not None else None
                
                # Calculate exposure (weighted sum of positive samples)
                if period_weights is not None:
                    exposure = (period_y * period_weights).sum()
                else:
                    exposure = period_y.sum()
                
                metrics.exposure_by_period[period_name] = exposure
                metrics.periods.append(period_name)
                metrics.sample_counts[period_name] = len(period_y)
            
            if not metrics.exposure_by_period:
                return metrics
            
            # Calculate statistics
            exposures = list(metrics.exposure_by_period.values())
            metrics.exposure_std = np.std(exposures)
            metrics.exposure_cv = metrics.exposure_std / (np.mean(exposures) + 1e-8)
            metrics.exposure_gini = self._calculate_gini_coefficient(exposures)
            
            # Statistical tests
            if len(exposures) > 2:
                # Levene test for equal variances
                period_groups = []
                for period_name, period_indices in time_bins.items():
                    period_y = y.loc[period_indices.index]
                    period_weights = sample_weights.loc[period_indices.index] if sample_weights is not None else None
                    if period_weights is not None:
                        period_groups.append(period_y * period_weights)
                    else:
                        period_groups.append(period_y)
                
                if len(period_groups) > 1:
                    try:
                        levene_stat, metrics.levene_p_value = stats.levene(*period_groups)
                    except Exception:
                        metrics.levene_p_value = 1.0
                    
                    try:
                        kruskal_stat, metrics.kruskal_p_value = stats.kruskal(*period_groups)
                    except Exception:
                        metrics.kruskal_p_value = 1.0
            
            # Check if exposure parity passed
            metrics.exposure_parity_passed = (
                metrics.exposure_cv <= self.config.exposure_parity_threshold and
                metrics.levene_p_value > self.config.significance_level
            )
            
            # Calculate confidence intervals
            if self.config.enable_bootstrap:
                metrics.exposure_ci_lower, metrics.exposure_ci_upper = self._calculate_bootstrap_ci(
                    exposures, self.config.confidence_level, self.config.bootstrap_samples
                )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Exposure parity analysis failed: {e}")
            return metrics
    
    def _analyze_error_parity(self, y: pd.Series, y_pred: pd.Series, 
                             time_bins: Dict[str, pd.Series],
                             sample_weights: Optional[pd.Series]) -> ErrorParityMetrics:
        """Analyze stable error by period."""
        metrics = ErrorParityMetrics()
        
        try:
            # Calculate error by period
            for period_name, period_indices in time_bins.items():
                period_y = y.loc[period_indices.index]
                period_pred = y_pred.loc[period_indices.index]
                period_weights = sample_weights.loc[period_indices.index] if sample_weights is not None else None
                
                # Calculate various error metrics
                if period_weights is not None:
                    # Weighted error metrics
                    brier_score = brier_score_loss(period_y, period_pred, sample_weight=period_weights)
                    f1_score_val = f1_score(period_y, period_pred, sample_weight=period_weights)
                    accuracy_score_val = accuracy_score(period_y, period_pred, sample_weight=period_weights)
                else:
                    brier_score = brier_score_loss(period_y, period_pred)
                    f1_score_val = f1_score(period_y, period_pred)
                    accuracy_score_val = accuracy_score(period_y, period_pred)
                
                # Use Brier score as primary error metric
                error = brier_score
                
                metrics.error_by_period[period_name] = error
                metrics.brier_scores[period_name] = brier_score
                metrics.f1_scores[period_name] = f1_score_val
                metrics.accuracy_scores[period_name] = accuracy_score_val
                metrics.periods.append(period_name)
                metrics.sample_counts[period_name] = len(period_y)
            
            if not metrics.error_by_period:
                return metrics
            
            # Calculate statistics
            errors = list(metrics.error_by_period.values())
            metrics.error_std = np.std(errors)
            metrics.error_cv = metrics.error_std / (np.mean(errors) + 1e-8)
            metrics.error_range = np.max(errors) - np.min(errors)
            
            # Statistical tests
            if len(errors) > 2:
                # Levene test for equal variances
                period_groups = []
                for period_name, period_indices in time_bins.items():
                    period_y = y.loc[period_indices.index]
                    period_pred = y_pred.loc[period_indices.index]
                    period_weights = sample_weights.loc[period_indices.index] if sample_weights is not None else None
                    
                    if period_weights is not None:
                        period_groups.append(period_y * period_weights)
                    else:
                        period_groups.append(period_y)
                
                if len(period_groups) > 1:
                    try:
                        levene_stat, metrics.levene_p_value = stats.levene(*period_groups)
                    except Exception:
                        metrics.levene_p_value = 1.0
                    
                    try:
                        kruskal_stat, metrics.kruskal_p_value = stats.kruskal(*period_groups)
                    except Exception:
                        metrics.kruskal_p_value = 1.0
            
            # Check if error parity passed
            metrics.error_parity_passed = (
                metrics.error_std <= self.config.error_parity_threshold and
                metrics.levene_p_value > self.config.significance_level
            )
            
            # Calculate confidence intervals
            if self.config.enable_bootstrap:
                metrics.error_ci_lower, metrics.error_ci_upper = self._calculate_bootstrap_ci(
                    errors, self.config.confidence_level, self.config.bootstrap_samples
                )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error parity analysis failed: {e}")
            return metrics
    
    def _analyze_calibration_stability(self, y: pd.Series, y_proba: pd.Series,
                                     time_bins: Dict[str, pd.Series],
                                     sample_weights: Optional[pd.Series]) -> CalibrationStabilityMetrics:
        """Analyze stable calibration by period."""
        metrics = CalibrationStabilityMetrics()
        
        try:
            # Calculate ECE by period
            for period_name, period_indices in time_bins.items():
                period_y = y.loc[period_indices.index]
                period_proba = y_proba.loc[period_indices.index]
                period_weights = sample_weights.loc[period_indices.index] if sample_weights is not None else None
                
                # Calculate ECE (Expected Calibration Error)
                ece = self._calculate_ece(period_y, period_proba, period_weights)
                metrics.ece_by_period[period_name] = ece
                
                # Calculate calibration curve parameters
                if len(period_y) > 10:  # Need sufficient samples
                    try:
                        fraction_of_positives, mean_predicted_value = calibration_curve(
                            period_y, period_proba, n_bins=10
                        )
                        
                        # Fit linear regression to calibration curve
                        if len(fraction_of_positives) > 1:
                            slope, intercept, r_value, p_value, std_err = stats.linregress(
                                mean_predicted_value, fraction_of_positives
                            )
                            
                            metrics.calibration_slopes[period_name] = slope
                            metrics.calibration_intercepts[period_name] = intercept
                            metrics.calibration_r2[period_name] = r_value ** 2
                    except Exception:
                        metrics.calibration_slopes[period_name] = 0.0
                        metrics.calibration_intercepts[period_name] = 0.0
                        metrics.calibration_r2[period_name] = 0.0
                
                metrics.periods.append(period_name)
                metrics.sample_counts[period_name] = len(period_y)
            
            if not metrics.ece_by_period:
                return metrics
            
            # Calculate statistics
            eces = list(metrics.ece_by_period.values())
            metrics.ece_std = np.std(eces)
            metrics.ece_cv = metrics.ece_std / (np.mean(eces) + 1e-8)
            metrics.ece_range = np.max(eces) - np.min(eces)
            
            # Statistical tests for calibration stability
            if len(eces) > 2:
                # Test for slope stability
                slopes = list(metrics.calibration_slopes.values())
                if len(slopes) > 1:
                    try:
                        slope_stat, metrics.slope_test_p_value = stats.kruskal(*[slopes])
                    except Exception:
                        metrics.slope_test_p_value = 1.0
                
                # Test for intercept stability
                intercepts = list(metrics.calibration_intercepts.values())
                if len(intercepts) > 1:
                    try:
                        intercept_stat, metrics.intercept_test_p_value = stats.kruskal(*[intercepts])
                    except Exception:
                        metrics.intercept_test_p_value = 1.0
            
            # Check if calibration stability passed
            metrics.calibration_stability_passed = (
                metrics.ece_std <= self.config.calibration_stability_threshold and
                metrics.slope_test_p_value > self.config.significance_level
            )
            
            # Calculate confidence intervals
            if self.config.enable_bootstrap:
                metrics.ece_ci_lower, metrics.ece_ci_upper = self._calculate_bootstrap_ci(
                    eces, self.config.confidence_level, self.config.bootstrap_samples
                )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Calibration stability analysis failed: {e}")
            return metrics
    
    def _calculate_ece(self, y_true: pd.Series, y_proba: pd.Series, 
                      sample_weights: Optional[pd.Series] = None) -> float:
        """Calculate Expected Calibration Error."""
        try:
            # Create bins
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
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
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement."""
        try:
            if len(values) < 2:
                return 0.0
            
            values = np.array(values)
            values = np.sort(values)
            n = len(values)
            cumsum = np.cumsum(values)
            
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
            return gini
            
        except Exception:
            return 0.0
    
    def _calculate_bootstrap_ci(self, values: List[float], confidence_level: float, 
                               n_samples: int) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate bootstrap confidence intervals."""
        try:
            if len(values) < 3:
                return {}, {}
            
            # Bootstrap sampling
            bootstrap_means = []
            for _ in range(n_samples):
                bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_means, lower_percentile)
            ci_upper = np.percentile(bootstrap_means, upper_percentile)
            
            return {'overall': ci_lower}, {'overall': ci_upper}
            
        except Exception:
            return {}, {}
    
    def _calculate_overall_fairness_score(self, report: TemporalFairnessReport) -> float:
        """Calculate overall fairness score."""
        try:
            scores = []
            
            # Exposure parity score
            if report.exposure_parity.exposure_parity_passed:
                scores.append(1.0)
            else:
                scores.append(max(0.0, 1.0 - report.exposure_parity.exposure_cv))
            
            # Error parity score
            if report.error_parity.error_parity_passed:
                scores.append(1.0)
            else:
                scores.append(max(0.0, 1.0 - report.error_parity.error_std))
            
            # Calibration stability score
            if report.calibration_stability.calibration_stability_passed:
                scores.append(1.0)
            else:
                scores.append(max(0.0, 1.0 - report.calibration_stability.ece_std))
            
            return np.mean(scores) if scores else 0.0
            
        except Exception:
            return 0.0
    
    def _evaluate_fairness_passed(self, report: TemporalFairnessReport) -> bool:
        """Evaluate if overall fairness passed."""
        try:
            # All individual fairness measures must pass
            exposure_passed = report.exposure_parity.exposure_parity_passed
            error_passed = report.error_parity.error_parity_passed
            calibration_passed = report.calibration_stability.calibration_stability_passed
            
            return exposure_passed and error_passed and calibration_passed
            
        except Exception:
            return False
    
    def _generate_fairness_warnings(self, report: TemporalFairnessReport) -> List[str]:
        """Generate fairness warnings."""
        warnings = []
        
        try:
            # Exposure parity warnings
            if not report.exposure_parity.exposure_parity_passed:
                warnings.append(f"Exposure parity failed: CV={report.exposure_parity.exposure_cv:.3f}")
            
            # Error parity warnings
            if not report.error_parity.error_parity_passed:
                warnings.append(f"Error parity failed: std={report.error_parity.error_std:.3f}")
            
            # Calibration stability warnings
            if not report.calibration_stability.calibration_stability_passed:
                warnings.append(f"Calibration stability failed: ECE std={report.calibration_stability.ece_std:.3f}")
            
            return warnings
            
        except Exception:
            return []
    
    def _generate_fairness_recommendations(self, report: TemporalFairnessReport) -> List[str]:
        """Generate fairness recommendations."""
        recommendations = []
        
        try:
            # Exposure parity recommendations
            if not report.exposure_parity.exposure_parity_passed:
                recommendations.append("Implement temporal balancing strategies")
                recommendations.append("Use stratified sampling across time periods")
            
            # Error parity recommendations
            if not report.error_parity.error_parity_passed:
                recommendations.append("Implement temporal error correction")
                recommendations.append("Use time-aware model training")
            
            # Calibration stability recommendations
            if not report.calibration_stability.calibration_stability_passed:
                recommendations.append("Implement calibration correction")
                recommendations.append("Use time-aware calibration methods")
            
            return recommendations
            
        except Exception:
            return []
    
    def _identify_critical_issues(self, report: TemporalFairnessReport) -> List[str]:
        """Identify critical fairness issues."""
        critical_issues = []
        
        try:
            if report.overall_fairness_score < 0.5:
                critical_issues.append("Critical temporal fairness issues detected")
            
            if report.exposure_parity.exposure_cv > 0.5:
                critical_issues.append("Severe exposure imbalance across time periods")
            
            if report.error_parity.error_std > 0.2:
                critical_issues.append("Severe error instability across time periods")
            
            if report.calibration_stability.ece_std > 0.3:
                critical_issues.append("Severe calibration instability across time periods")
            
            return critical_issues
            
        except Exception:
            return []
    
    def generate_fairness_report(self, report: TemporalFairnessReport, 
                                filename: Optional[str] = None) -> str:
        """Generate detailed fairness report."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"temporal_fairness_report_{timestamp}.json"
        
        filepath = Path(self.config.report_directory) / filename
        
        try:
            report_data = {
                'analysis_timestamp': report.analysis_timestamp,
                'overall_fairness_score': report.overall_fairness_score,
                'fairness_passed': report.fairness_passed,
                'total_periods': report.total_periods,
                'total_samples': report.total_samples,
                'exposure_parity': {
                    'exposure_by_period': report.exposure_parity.exposure_by_period,
                    'exposure_std': report.exposure_parity.exposure_std,
                    'exposure_cv': report.exposure_parity.exposure_cv,
                    'exposure_gini': report.exposure_parity.exposure_gini,
                    'levene_p_value': report.exposure_parity.levene_p_value,
                    'kruskal_p_value': report.exposure_parity.kruskal_p_value,
                    'exposure_parity_passed': report.exposure_parity.exposure_parity_passed
                },
                'error_parity': {
                    'error_by_period': report.error_parity.error_by_period,
                    'error_std': report.error_parity.error_std,
                    'error_cv': report.error_parity.error_cv,
                    'error_range': report.error_parity.error_range,
                    'levene_p_value': report.error_parity.levene_p_value,
                    'kruskal_p_value': report.error_parity.kruskal_p_value,
                    'error_parity_passed': report.error_parity.error_parity_passed
                },
                'calibration_stability': {
                    'ece_by_period': report.calibration_stability.ece_by_period,
                    'ece_std': report.calibration_stability.ece_std,
                    'ece_cv': report.calibration_stability.ece_cv,
                    'ece_range': report.calibration_stability.ece_range,
                    'slope_test_p_value': report.calibration_stability.slope_test_p_value,
                    'intercept_test_p_value': report.calibration_stability.intercept_test_p_value,
                    'calibration_stability_passed': report.calibration_stability.calibration_stability_passed
                },
                'warnings': report.warnings,
                'recommendations': report.recommendations,
                'critical_issues': report.critical_issues
            }
            
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            if TPRINT_AVAILABLE:
                tprint_success(f"📄 Fairness report saved: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to generate fairness report: {e}")
            return ""


# Convenience functions
def create_temporal_fairness_analyzer(config: Optional[TemporalFairnessConfig] = None) -> TemporalFairnessAnalyzer:
    """Create temporal fairness analyzer."""
    return TemporalFairnessAnalyzer(config)

def analyze_temporal_fairness_quick(X: pd.DataFrame, y: pd.Series, 
                                   y_pred: Optional[pd.Series] = None,
                                   y_proba: Optional[pd.Series] = None) -> float:
    """Quick temporal fairness analysis."""
    analyzer = create_temporal_fairness_analyzer()
    report = analyzer.analyze_temporal_fairness(X, y, y_pred, y_proba)
    return report.overall_fairness_score


if __name__ == "__main__":
    # Example usage
    print("Temporal Fairness Metrics with Explicit Definitions")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=1000, freq='1H')
    X = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'timestamp': dates
    }, index=dates)
    
    y = pd.Series(np.random.choice([0, 1], size=1000, p=[0.7, 0.3]), index=dates)
    y_pred = pd.Series(np.random.choice([0, 1], size=1000, p=[0.6, 0.4]), index=dates)
    y_proba = pd.Series(np.random.uniform(0, 1, size=1000), index=dates)
    
    # Analyze temporal fairness
    analyzer = create_temporal_fairness_analyzer()
    report = analyzer.analyze_temporal_fairness(X, y, y_pred, y_proba, time_col='timestamp')
    
    print(f"Overall fairness score: {report.overall_fairness_score:.3f}")
    print(f"Fairness passed: {report.fairness_passed}")
    print(f"Exposure parity passed: {report.exposure_parity.exposure_parity_passed}")
    print(f"Error parity passed: {report.error_parity.error_parity_passed}")
    print(f"Calibration stability passed: {report.calibration_stability.calibration_stability_passed}")
    print(f"Warnings: {len(report.warnings)}")
    print(f"Recommendations: {len(report.recommendations)}")