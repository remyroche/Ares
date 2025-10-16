#!/usr/bin/env python3
"""
Automated Data Drift Detection System

This module provides comprehensive data drift detection capabilities for the trading system:
- Statistical drift detection (KS test, Chi-square test, etc.)
- Distribution-based drift detection
- Feature-level drift monitoring
- Temporal drift analysis
- Regime-aware drift detection
- Automated alerting and reporting
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import joblib

# Import system utilities
from ..logger import get_logger
from ..matrix_operations import get_unified_matrix_operations

class DriftType(Enum):
    """Types of data drift."""
    STATISTICAL = "statistical"
    DISTRIBUTIONAL = "distributional"
    COVARIATE = "covariate"
    LABEL = "label"
    CONCEPT = "concept"
    TEMPORAL = "temporal"

class DriftMethod(Enum):
    """Available drift detection methods."""
    KS_TEST = "kolmogorov_smirnov"
    CHI_SQUARE = "chi_square"
    WASSERSTEIN = "wasserstein"
    MUTUAL_INFO = "mutual_information"
    PSI = "population_stability_index"
    JENSEN_SHANNON = "jensen_shannon"
    BETA_DRIFT = "beta_drift"
    PERMUTATION = "permutation"
    CORRELATION = "correlation"

class DriftSeverity(Enum):
    """Drift severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DriftDetectionConfig:
    """Configuration for drift detection."""
    # Detection methods
    methods: List[DriftMethod] = field(default_factory=lambda: [
        DriftMethod.KS_TEST,
        DriftMethod.PSI,
        DriftMethod.WASSERSTEIN,
        DriftMethod.CHI_SQUARE
    ])
    
    # Thresholds
    drift_threshold: float = 0.05
    warning_threshold: float = 0.1
    critical_threshold: float = 0.2
    
    # Statistical test parameters
    alpha: float = 0.05
    min_samples: int = 100
    max_samples: int = 10000
    
    # Temporal analysis
    temporal_window: int = 1000
    lookback_periods: int = 5
    
    # Performance settings
    n_jobs: int = -1
    chunk_size: int = 10000
    enable_parallel: bool = True
    
    # Alerting
    enable_alerts: bool = True
    alert_cooldown: int = 3600  # seconds
    
    # Output settings
    save_results: bool = True
    generate_plots: bool = True
    output_directory: Optional[str] = None

@dataclass
class DriftResult:
    """Result of drift detection analysis."""
    feature_name: str
    drift_type: DriftType
    method: DriftMethod
    statistic: float
    p_value: float
    severity: DriftSeverity
    threshold_exceeded: bool
    confidence_interval: Optional[Tuple[float, float]] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DriftReport:
    """Comprehensive drift detection report."""
    timestamp: str
    total_features: int
    drifted_features: int
    drift_results: List[DriftResult]
    severity_summary: Dict[DriftSeverity, int]
    feature_rankings: List[str]
    recommendations: List[str]
    meta_info: Dict[str, Any]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection results."""
        return {
            'total_features': self.total_features,
            'drifted_features': self.drifted_features,
            'drift_rate': self.drifted_features / self.total_features if self.total_features > 0 else 0,
            'severity_distribution': {sev.value: count for sev, count in self.severity_summary.items()},
            'top_drifted_features': self.feature_rankings[:10],
            'recommendations_count': len(self.recommendations)
        }

class DataDriftDetector:
    """Automated data drift detector."""
    
    def __init__(self, config: Optional[DriftDetectionConfig] = None):
        self.config = config or DriftDetectionConfig()
        self.logger = get_logger("DataDriftDetector")
        
        # Initialize matrix operations for performance
        self.matrix_ops = get_unified_matrix_operations()
        
        # Historical data storage for temporal analysis
        self.historical_data = defaultdict(lambda: deque(maxlen=self.config.lookback_periods))
        self.alert_history = defaultdict(float)
        
        self.logger.info("🚀 DataDriftDetector initialized")
    
    def detect_drift(self, 
                    reference_data: pd.DataFrame,
                    current_data: pd.DataFrame,
                    feature_names: Optional[List[str]] = None,
                    regime_labels: Optional[pd.Series] = None) -> DriftReport:
        """Perform comprehensive drift detection analysis."""
        
        start_time = time.time()
        self.logger.info(f"🔍 Starting drift detection on {len(reference_data.columns)} features")
        
        if feature_names is None:
            feature_names = reference_data.columns.tolist()
        
        # Filter features that exist in both datasets
        common_features = [f for f in feature_names if f in reference_data.columns and f in current_data.columns]
        
        if not common_features:
            self.logger.warning("⚠️ No common features found between reference and current data")
            return self._create_empty_report()
        
        # Perform drift detection for each feature
        drift_results = []
        
        if self.config.enable_parallel and len(common_features) > 10:
            drift_results = self._detect_drift_parallel(common_features, reference_data, current_data, regime_labels)
        else:
            drift_results = self._detect_drift_sequential(common_features, reference_data, current_data, regime_labels)
        
        # Generate report
        report = self._generate_report(drift_results, start_time)
        
        # Save results if configured
        if self.config.save_results:
            self._save_report(report)
        
        # Generate plots if configured
        if self.config.generate_plots:
            self._generate_plots(report, reference_data, current_data)
        
        # Send alerts if configured
        if self.config.enable_alerts:
            self._send_alerts(report)
        
        # Update historical data
        self._update_historical_data(current_data)
        
        self.logger.info(f"✅ Drift detection completed in {time.time() - start_time:.3f}s")
        return report
    
    def _detect_drift_parallel(self, features: List[str], reference_data: pd.DataFrame, 
                              current_data: pd.DataFrame, regime_labels: Optional[pd.Series]) -> List[DriftResult]:
        """Detect drift using parallel processing."""
        drift_results = []
        
        with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
            futures = {
                executor.submit(self._detect_feature_drift, feature, reference_data, current_data, regime_labels): feature
                for feature in features
            }
            
            for future in as_completed(futures):
                feature = futures[future]
                try:
                    results = future.result()
                    drift_results.extend(results)
                except Exception as e:
                    self.logger.error(f"❌ Error detecting drift for feature {feature}: {e}")
        
        return drift_results
    
    def _detect_drift_sequential(self, features: List[str], reference_data: pd.DataFrame, 
                                current_data: pd.DataFrame, regime_labels: Optional[pd.Series]) -> List[DriftResult]:
        """Detect drift using sequential processing."""
        drift_results = []
        
        for feature in features:
            try:
                results = self._detect_feature_drift(feature, reference_data, current_data, regime_labels)
                drift_results.extend(results)
            except Exception as e:
                self.logger.error(f"❌ Error detecting drift for feature {feature}: {e}")
        
        return drift_results
    
    def _detect_feature_drift(self, feature: str, reference_data: pd.DataFrame, 
                             current_data: pd.DataFrame, regime_labels: Optional[pd.Series]) -> List[DriftResult]:
        """Detect drift for a single feature."""
        results = []
        
        ref_values = reference_data[feature].dropna()
        cur_values = current_data[feature].dropna()
        
        # Skip if insufficient data
        if len(ref_values) < self.config.min_samples or len(cur_values) < self.config.min_samples:
            self.logger.warning(f"⚠️ Insufficient data for feature {feature}")
            return results
        
        # Sample data if too large
        if len(ref_values) > self.config.max_samples:
            ref_values = ref_values.sample(self.config.max_samples, random_state=42)
        if len(cur_values) > self.config.max_samples:
            cur_values = cur_values.sample(self.config.max_samples, random_state=42)
        
        # Apply drift detection methods
        for method in self.config.methods:
            try:
                result = self._apply_drift_method(method, feature, ref_values, cur_values)
                if result:
                    results.append(result)
            except Exception as e:
                self.logger.error(f"❌ Error applying {method.value} to feature {feature}: {e}")
        
        # Regime-aware analysis if regime labels available
        if regime_labels is not None:
            regime_results = self._detect_regime_drift(feature, reference_data, current_data, regime_labels)
            results.extend(regime_results)
        
        return results
    
    def _apply_drift_method(self, method: DriftMethod, feature: str, 
                           ref_values: pd.Series, cur_values: pd.Series) -> Optional[DriftResult]:
        """Apply specific drift detection method."""
        
        if method == DriftMethod.KS_TEST:
            return self._ks_test_drift(feature, ref_values, cur_values)
        
        elif method == DriftMethod.CHI_SQUARE:
            return self._chi_square_drift(feature, ref_values, cur_values)
        
        elif method == DriftMethod.WASSERSTEIN:
            return self._wasserstein_drift(feature, ref_values, cur_values)
        
        elif method == DriftMethod.PSI:
            return self._psi_drift(feature, ref_values, cur_values)
        
        elif method == DriftMethod.JENSEN_SHANNON:
            return self._jensen_shannon_drift(feature, ref_values, cur_values)
        
        elif method == DriftMethod.MUTUAL_INFO:
            return self._mutual_info_drift(feature, ref_values, cur_values)
        
        elif method == DriftMethod.CORRELATION:
            return self._correlation_drift(feature, ref_values, cur_values)
        
        else:
            self.logger.warning(f"⚠️ Unknown drift method: {method}")
            return None
    
    def _ks_test_drift(self, feature: str, ref_values: pd.Series, cur_values: pd.Series) -> DriftResult:
        """Kolmogorov-Smirnov test for drift detection."""
        statistic, p_value = ks_2samp(ref_values, cur_values)
        
        severity = self._determine_severity(statistic, p_value)
        
        return DriftResult(
            feature_name=feature,
            drift_type=DriftType.STATISTICAL,
            method=DriftMethod.KS_TEST,
            statistic=statistic,
            p_value=p_value,
            severity=severity,
            threshold_exceeded=p_value < self.config.alpha,
            additional_metrics={'ks_statistic': statistic}
        )
    
    def _chi_square_drift(self, feature: str, ref_values: pd.Series, cur_values: pd.Series) -> DriftResult:
        """Chi-square test for drift detection."""
        # Create bins for categorical analysis
        bins = np.linspace(min(ref_values.min(), cur_values.min()), 
                          max(ref_values.max(), cur_values.max()), 10)
        
        ref_binned = pd.cut(ref_values, bins=bins, include_lowest=True)
        cur_binned = pd.cut(cur_values, bins=bins, include_lowest=True)
        
        ref_counts = ref_binned.value_counts().sort_index()
        cur_counts = cur_binned.value_counts().sort_index()
        
        # Align counts
        all_bins = ref_counts.index.union(cur_counts.index)
        ref_aligned = ref_counts.reindex(all_bins, fill_value=0)
        cur_aligned = cur_counts.reindex(all_bins, fill_value=0)
        
        # Perform chi-square test
        contingency_table = np.array([ref_aligned.values, cur_aligned.values])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        severity = self._determine_severity(chi2 / (dof + 1e-8), p_value)
        
        return DriftResult(
            feature_name=feature,
            drift_type=DriftType.STATISTICAL,
            method=DriftMethod.CHI_SQUARE,
            statistic=chi2,
            p_value=p_value,
            severity=severity,
            threshold_exceeded=p_value < self.config.alpha,
            additional_metrics={'chi2_statistic': chi2, 'degrees_of_freedom': dof}
        )
    
    def _wasserstein_drift(self, feature: str, ref_values: pd.Series, cur_values: pd.Series) -> DriftResult:
        """Wasserstein distance for drift detection."""
        distance = wasserstein_distance(ref_values, cur_values)
        
        # Normalize distance by standard deviation
        std_ref = ref_values.std()
        std_cur = cur_values.std()
        normalized_distance = distance / (std_ref + std_cur + 1e-8)
        
        severity = self._determine_severity(normalized_distance, 1.0)
        
        return DriftResult(
            feature_name=feature,
            drift_type=DriftType.DISTRIBUTIONAL,
            method=DriftMethod.WASSERSTEIN,
            statistic=distance,
            p_value=1.0,  # No p-value for Wasserstein distance
            severity=severity,
            threshold_exceeded=normalized_distance > self.config.drift_threshold,
            additional_metrics={'normalized_distance': normalized_distance}
        )
    
    def _psi_drift(self, feature: str, ref_values: pd.Series, cur_values: pd.Series) -> DriftResult:
        """Population Stability Index for drift detection."""
        # Create bins
        bins = np.linspace(min(ref_values.min(), cur_values.min()), 
                          max(ref_values.max(), cur_values.max()), 10)
        
        ref_binned = pd.cut(ref_values, bins=bins, include_lowest=True)
        cur_binned = pd.cut(cur_values, bins=bins, include_lowest=True)
        
        ref_counts = ref_binned.value_counts().sort_index()
        cur_counts = cur_binned.value_counts().sort_index()
        
        # Calculate PSI
        ref_props = ref_counts / ref_counts.sum()
        cur_props = cur_counts / cur_counts.sum()
        
        # Align proportions
        all_bins = ref_props.index.union(cur_props.index)
        ref_aligned = ref_props.reindex(all_bins, fill_value=1e-8)
        cur_aligned = cur_props.reindex(all_bins, fill_value=1e-8)
        
        psi = np.sum((cur_aligned - ref_aligned) * np.log(cur_aligned / ref_aligned))
        
        severity = self._determine_severity(psi, 1.0)
        
        return DriftResult(
            feature_name=feature,
            drift_type=DriftType.DISTRIBUTIONAL,
            method=DriftMethod.PSI,
            statistic=psi,
            p_value=1.0,  # No p-value for PSI
            severity=severity,
            threshold_exceeded=psi > self.config.drift_threshold,
            additional_metrics={'psi_value': psi}
        )
    
    def _jensen_shannon_drift(self, feature: str, ref_values: pd.Series, cur_values: pd.Series) -> DriftResult:
        """Jensen-Shannon divergence for drift detection."""
        # Create bins
        bins = np.linspace(min(ref_values.min(), cur_values.min()), 
                          max(ref_values.max(), cur_values.max()), 20)
        
        ref_binned = pd.cut(ref_values, bins=bins, include_lowest=True)
        cur_binned = pd.cut(cur_values, bins=bins, include_lowest=True)
        
        ref_counts = ref_binned.value_counts().sort_index()
        cur_counts = cur_binned.value_counts().sort_index()
        
        # Calculate probabilities
        ref_props = ref_counts / ref_counts.sum()
        cur_props = cur_counts / cur_counts.sum()
        
        # Align proportions
        all_bins = ref_props.index.union(cur_props.index)
        ref_aligned = ref_props.reindex(all_bins, fill_value=1e-8)
        cur_aligned = cur_props.reindex(all_bins, fill_value=1e-8)
        
        # Jensen-Shannon divergence
        m = 0.5 * (ref_aligned + cur_aligned)
        js_div = 0.5 * stats.entropy(ref_aligned, m) + 0.5 * stats.entropy(cur_aligned, m)
        
        severity = self._determine_severity(js_div, 1.0)
        
        return DriftResult(
            feature_name=feature,
            drift_type=DriftType.DISTRIBUTIONAL,
            method=DriftMethod.JENSEN_SHANNON,
            statistic=js_div,
            p_value=1.0,  # No p-value for JS divergence
            severity=severity,
            threshold_exceeded=js_div > self.config.drift_threshold,
            additional_metrics={'js_divergence': js_div}
        )
    
    def _mutual_info_drift(self, feature: str, ref_values: pd.Series, cur_values: pd.Series) -> DriftResult:
        """Mutual information for drift detection."""
        # Create bins for discretization
        bins = np.linspace(min(ref_values.min(), cur_values.min()), 
                          max(ref_values.max(), cur_values.max()), 20)
        
        ref_binned = pd.cut(ref_values, bins=bins, include_lowest=True)
        cur_binned = pd.cut(cur_values, bins=bins, include_lowest=True)
        
        # Calculate mutual information between reference and current distributions
        ref_counts = ref_binned.value_counts()
        cur_counts = cur_binned.value_counts()
        
        # Create joint distribution
        ref_props = ref_counts / ref_counts.sum()
        cur_props = cur_counts / cur_counts.sum()
        
        # Align proportions
        all_bins = ref_props.index.union(cur_props.index)
        ref_aligned = ref_props.reindex(all_bins, fill_value=1e-8)
        cur_aligned = cur_props.reindex(all_bins, fill_value=1e-8)
        
        # Calculate mutual information
        mi = mutual_info_score(ref_aligned, cur_aligned)
        
        severity = self._determine_severity(mi, 1.0)
        
        return DriftResult(
            feature_name=feature,
            drift_type=DriftType.COVARIATE,
            method=DriftMethod.MUTUAL_INFO,
            statistic=mi,
            p_value=1.0,  # No p-value for mutual information
            severity=severity,
            threshold_exceeded=mi > self.config.drift_threshold,
            additional_metrics={'mutual_information': mi}
        )
    
    def _correlation_drift(self, feature: str, ref_values: pd.Series, cur_values: pd.Series) -> DriftResult:
        """Correlation-based drift detection."""
        # Calculate correlation between reference and current data
        correlation = np.corrcoef(ref_values, cur_values)[0, 1]
        
        # Convert correlation to drift measure (1 - |correlation|)
        drift_measure = 1 - abs(correlation) if not np.isnan(correlation) else 1.0
        
        severity = self._determine_severity(drift_measure, 1.0)
        
        return DriftResult(
            feature_name=feature,
            drift_type=DriftType.CORRELATION,
            method=DriftMethod.CORRELATION,
            statistic=drift_measure,
            p_value=1.0,  # No p-value for correlation drift
            severity=severity,
            threshold_exceeded=drift_measure > self.config.drift_threshold,
            additional_metrics={'correlation': correlation, 'drift_measure': drift_measure}
        )
    
    def _detect_regime_drift(self, feature: str, reference_data: pd.DataFrame, 
                            current_data: pd.DataFrame, regime_labels: pd.Series) -> List[DriftResult]:
        """Detect drift within specific regimes."""
        results = []
        
        for regime in regime_labels.unique():
            regime_mask = regime_labels == regime
            if regime_mask.sum() < self.config.min_samples:
                continue
            
            ref_regime = reference_data[regime_mask]
            cur_regime = current_data[regime_mask]
            
            if len(ref_regime) == 0 or len(cur_regime) == 0:
                continue
            
            # Apply drift detection to regime-specific data
            regime_results = self._detect_feature_drift(f"{feature}_regime_{regime}", ref_regime, cur_regime, None)
            results.extend(regime_results)
        
        return results
    
    def _determine_severity(self, statistic: float, p_value: float) -> DriftSeverity:
        """Determine drift severity based on statistic and p-value."""
        if p_value < self.config.alpha and statistic > self.config.critical_threshold:
            return DriftSeverity.CRITICAL
        elif p_value < self.config.alpha and statistic > self.config.warning_threshold:
            return DriftSeverity.HIGH
        elif p_value < self.config.alpha or statistic > self.config.drift_threshold:
            return DriftSeverity.MEDIUM
        elif statistic > self.config.drift_threshold * 0.5:
            return DriftSeverity.LOW
        else:
            return DriftSeverity.NONE
    
    def _generate_report(self, drift_results: List[DriftResult], start_time: float) -> DriftReport:
        """Generate comprehensive drift detection report."""
        timestamp = pd.Timestamp.now().isoformat()
        
        # Count drifted features
        drifted_features = len([r for r in drift_results if r.threshold_exceeded])
        
        # Severity summary
        severity_summary = defaultdict(int)
        for result in drift_results:
            severity_summary[result.severity] += 1
        
        # Feature rankings (by severity and statistic)
        feature_scores = defaultdict(float)
        for result in drift_results:
            severity_weight = {
                DriftSeverity.CRITICAL: 5,
                DriftSeverity.HIGH: 4,
                DriftSeverity.MEDIUM: 3,
                DriftSeverity.LOW: 2,
                DriftSeverity.NONE: 1
            }[result.severity]
            feature_scores[result.feature_name] += result.statistic * severity_weight
        
        feature_rankings = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        feature_rankings = [feature for feature, _ in feature_rankings]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(drift_results, severity_summary)
        
        return DriftReport(
            timestamp=timestamp,
            total_features=len(set(r.feature_name for r in drift_results)),
            drifted_features=drifted_features,
            drift_results=drift_results,
            severity_summary=dict(severity_summary),
            feature_rankings=feature_rankings,
            recommendations=recommendations,
            meta_info={
                'analysis_time': time.time() - start_time,
                'methods_used': [m.value for m in self.config.methods],
                'config': self.config.__dict__
            }
        )
    
    def _generate_recommendations(self, drift_results: List[DriftResult], 
                                 severity_summary: Dict[DriftSeverity, int]) -> List[str]:
        """Generate recommendations based on drift results."""
        recommendations = []
        
        critical_count = severity_summary.get(DriftSeverity.CRITICAL, 0)
        high_count = severity_summary.get(DriftSeverity.HIGH, 0)
        
        if critical_count > 0:
            recommendations.append(f"CRITICAL: {critical_count} features show critical drift - immediate retraining recommended")
        
        if high_count > 0:
            recommendations.append(f"HIGH: {high_count} features show high drift - consider model updates")
        
        if critical_count + high_count > len(drift_results) * 0.3:
            recommendations.append("WARNING: High proportion of features showing drift - investigate data pipeline")
        
        # Feature-specific recommendations
        critical_features = [r.feature_name for r in drift_results if r.severity == DriftSeverity.CRITICAL]
        if critical_features:
            recommendations.append(f"Focus on features: {', '.join(critical_features[:5])}")
        
        return recommendations
    
    def _create_empty_report(self) -> DriftReport:
        """Create empty report for error cases."""
        return DriftReport(
            timestamp=pd.Timestamp.now().isoformat(),
            total_features=0,
            drifted_features=0,
            drift_results=[],
            severity_summary={},
            feature_rankings=[],
            recommendations=["No common features found between datasets"],
            meta_info={'error': 'No common features found'}
        )
    
    def _save_report(self, report: DriftReport):
        """Save drift detection report."""
        if self.config.output_directory:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed report
            report_file = output_dir / f"drift_report_{int(time.time())}.json"
            
            # Convert to serializable format
            serializable_report = {
                'timestamp': report.timestamp,
                'total_features': report.total_features,
                'drifted_features': report.drifted_features,
                'severity_summary': {sev.value: count for sev, count in report.severity_summary.items()},
                'feature_rankings': report.feature_rankings,
                'recommendations': report.recommendations,
                'meta_info': report.meta_info
            }
            
            import json
            with open(report_file, 'w') as f:
                json.dump(serializable_report, f, indent=2)
            
            self.logger.info(f"💾 Drift report saved to {report_file}")
    
    def _generate_plots(self, report: DriftReport, reference_data: pd.DataFrame, current_data: pd.DataFrame):
        """Generate visualization plots for drift detection."""
        if self.config.output_directory:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot 1: Severity distribution
            self._plot_severity_distribution(report, output_dir)
            
            # Plot 2: Feature drift heatmap
            self._plot_drift_heatmap(report, output_dir)
            
            # Plot 3: Top drifted features
            self._plot_top_drifted_features(report, reference_data, current_data, output_dir)
    
    def _plot_severity_distribution(self, report: DriftReport, output_dir: Path):
        """Plot severity distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        severities = list(report.severity_summary.keys())
        counts = list(report.severity_summary.values())
        
        colors = ['green', 'yellow', 'orange', 'red', 'darkred']
        bars = ax.bar([s.value for s in severities], counts, color=colors[:len(severities)])
        
        ax.set_xlabel('Drift Severity')
        ax.set_ylabel('Number of Features')
        ax.set_title('Distribution of Drift Severity')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'severity_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_drift_heatmap(self, report: DriftReport, output_dir: Path):
        """Plot drift detection heatmap."""
        if not report.drift_results:
            return
        
        # Create feature-method matrix
        features = list(set(r.feature_name for r in report.drift_results))
        methods = list(set(r.method.value for r in report.drift_results))
        
        data_matrix = np.zeros((len(features), len(methods)))
        
        for result in report.drift_results:
            feature_idx = features.index(result.feature_name)
            method_idx = methods.index(result.method.value)
            data_matrix[feature_idx, method_idx] = result.statistic
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(data_matrix, 
                   xticklabels=methods, 
                   yticklabels=features,
                   annot=True, 
                   fmt='.3f',
                   cmap='Reds',
                   ax=ax)
        
        ax.set_title('Drift Detection Results Heatmap')
        ax.set_xlabel('Detection Methods')
        ax.set_ylabel('Features')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'drift_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_top_drifted_features(self, report: DriftReport, reference_data: pd.DataFrame, 
                                  current_data: pd.DataFrame, output_dir: Path):
        """Plot distributions of top drifted features."""
        top_features = report.feature_rankings[:5]
        
        if not top_features:
            return
        
        fig, axes = plt.subplots(len(top_features), 1, figsize=(12, 3 * len(top_features)))
        if len(top_features) == 1:
            axes = [axes]
        
        for i, feature in enumerate(top_features):
            if feature in reference_data.columns and feature in current_data.columns:
                ax = axes[i]
                
                # Plot distributions
                ax.hist(reference_data[feature].dropna(), bins=50, alpha=0.7, label='Reference', density=True)
                ax.hist(current_data[feature].dropna(), bins=50, alpha=0.7, label='Current', density=True)
                
                ax.set_xlabel(feature)
                ax.set_ylabel('Density')
                ax.set_title(f'Distribution Comparison: {feature}')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'top_drifted_features.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _send_alerts(self, report: DriftReport):
        """Send alerts for significant drift."""
        current_time = time.time()
        
        # Check for critical drift
        critical_features = [r for r in report.drift_results if r.severity == DriftSeverity.CRITICAL]
        
        if critical_features and (current_time - self.alert_history['critical']) > self.config.alert_cooldown:
            self.logger.critical(f"🚨 CRITICAL DRIFT DETECTED: {len(critical_features)} features")
            self.alert_history['critical'] = current_time
        
        # Check for high drift
        high_features = [r for r in report.drift_results if r.severity == DriftSeverity.HIGH]
        
        if high_features and (current_time - self.alert_history['high']) > self.config.alert_cooldown:
            self.logger.warning(f"⚠️ HIGH DRIFT DETECTED: {len(high_features)} features")
            self.alert_history['high'] = current_time
    
    def _update_historical_data(self, current_data: pd.DataFrame):
        """Update historical data for temporal analysis."""
        timestamp = time.time()
        self.historical_data[timestamp].append(current_data.copy())

# Convenience functions
def detect_data_drift(reference_data: pd.DataFrame,
                      current_data: pd.DataFrame,
                      feature_names: Optional[List[str]] = None,
                      regime_labels: Optional[pd.Series] = None,
                      config: Optional[DriftDetectionConfig] = None) -> DriftReport:
    """Convenience function for drift detection."""
    detector = DataDriftDetector(config)
    return detector.detect_drift(reference_data, current_data, feature_names, regime_labels)

def get_drifted_features(reference_data: pd.DataFrame,
                         current_data: pd.DataFrame,
                         severity_threshold: DriftSeverity = DriftSeverity.MEDIUM,
                         config: Optional[DriftDetectionConfig] = None) -> List[str]:
    """Get list of drifted features above severity threshold."""
    detector = DataDriftDetector(config)
    report = detector.detect_drift(reference_data, current_data)
    
    drifted_features = []
    for result in report.drift_results:
        if result.severity.value in ['medium', 'high', 'critical']:
            drifted_features.append(result.feature_name)
    
    return list(set(drifted_features))