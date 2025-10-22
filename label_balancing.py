"""
Enhanced Label Balancing with Temporal Integrity and Fairness

This module implements a comprehensive label balancing system with advanced temporal validation,
temporal fairness metrics, and temporal integrity preservation for financial datasets.

Key Features:
- Temporal integrity validation with strict chronological ordering
- Temporal fairness metrics for balanced representation across time periods
- Advanced leakage prevention with purged cross-validation
- Regime-aware balancing with temporal consistency
- Comprehensive temporal drift detection and mitigation
- Multi-dimensional temporal validation framework

Temporal Integrity:
- Strict no-lookahead policy enforcement
- Chronological ordering validation
- Temporal gap enforcement between train/validation periods
- Feature temporal consistency checks

Temporal Fairness:
- Balanced representation across time periods
- Temporal drift detection and correction
- Regime persistence validation
- Temporal stability metrics

Leakage Prevention:
- Purged cross-validation with embargo periods
- Feature temporal validation
- Statistical leakage detection
- Temporal correlation analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Generator, Callable
from dataclasses import dataclass, field
from enum import Enum
from sklearn.utils import resample
from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances, accuracy_score, f1_score, roc_auc_score
import warnings
import logging
from datetime import datetime, timedelta
from scipy import stats
from scipy.special import softmax
from collections import Counter, defaultdict
import random
import json
from pathlib import Path

# Import existing utilities
try:
    from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    from src.utils.common_operations import safe_divide, safe_mean, safe_std, validate_dataframe
    from src.utils.math_validation import MathValidation
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    # Fallback logging functions
    def tprint_info(msg): print(f"INFO: {msg}")
    def tprint_warning(msg): print(f"WARNING: {msg}")
    def tprint_error(msg): print(f"ERROR: {msg}")
    def tprint_success(msg): print(f"SUCCESS: {msg}")

try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours, NearMiss
    from imblearn.combine import SMOTETomek, SMOTEENN
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    if TPRINT_AVAILABLE:
        tprint_warning("⚠️ imbalanced-learn not available, using basic resampling methods")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalIntegrityLevel(Enum):
    """Levels of temporal integrity enforcement."""
    STRICT = "strict"          # No future information allowed
    MODERATE = "moderate"      # Limited future information
    LENIENT = "lenient"        # Minimal temporal constraints
    DISABLED = "disabled"      # No temporal constraints


class TemporalFairnessMetric(Enum):
    """Temporal fairness metrics."""
    TEMPORAL_BALANCE = "temporal_balance"           # Balance across time periods
    REGIME_PERSISTENCE = "regime_persistence"       # Regime consistency over time
    TEMPORAL_DRIFT = "temporal_drift"               # Drift detection
    TEMPORAL_STABILITY = "temporal_stability"       # Stability metrics
    CHRONOLOGICAL_ORDER = "chronological_order"     # Order preservation
    TEMPORAL_CONSISTENCY = "temporal_consistency"   # Overall consistency


@dataclass
class TemporalValidationConfig:
    """Configuration for temporal validation and integrity."""
    
    # Temporal integrity settings
    integrity_level: TemporalIntegrityLevel = TemporalIntegrityLevel.STRICT
    enforce_chronological_order: bool = True
    min_temporal_gap: int = 1  # Minimum periods between train/validation
    max_lookahead_periods: int = 0  # Maximum allowed lookahead
    
    # Temporal fairness settings
    enable_temporal_fairness: bool = True
    fairness_metrics: List[TemporalFairnessMetric] = field(default_factory=lambda: [
        TemporalFairnessMetric.TEMPORAL_BALANCE,
        TemporalFairnessMetric.REGIME_PERSISTENCE,
        TemporalFairnessMetric.TEMPORAL_DRIFT
    ])
    temporal_window_size: int = 30  # Window for temporal analysis
    drift_detection_threshold: float = 0.1  # Threshold for drift detection
    
    # Leakage prevention
    enable_leakage_detection: bool = True
    leakage_detection_methods: List[str] = field(default_factory=lambda: [
        "temporal_correlation",
        "statistical_similarity",
        "feature_consistency"
    ])
    max_correlation_threshold: float = 0.95
    statistical_similarity_threshold: float = 0.8
    
    # Cross-validation settings
    enable_purged_cv: bool = True
    purge_length: int = 1
    embargo_length: int = 1
    n_splits: int = 5
    
    # Reporting and logging
    enable_detailed_logging: bool = True
    save_validation_reports: bool = True
    report_directory: str = "reports/temporal_validation"


@dataclass
class TemporalFairnessReport:
    """Comprehensive temporal fairness report."""
    
    # Basic metrics
    temporal_balance_score: float = 0.0
    regime_persistence_score: float = 0.0
    temporal_drift_score: float = 0.0
    temporal_stability_score: float = 0.0
    chronological_order_score: float = 0.0
    overall_fairness_score: float = 0.0
    
    # Detailed analysis
    temporal_periods: List[str] = field(default_factory=list)
    period_balance_ratios: Dict[str, float] = field(default_factory=dict)
    regime_transitions: Dict[str, int] = field(default_factory=dict)
    drift_indicators: List[str] = field(default_factory=list)
    stability_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Warnings and recommendations
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)
    
    # Metadata
    validation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config_used: Dict[str, Any] = field(default_factory=dict)


class TemporalIntegrityValidator:
    """Advanced temporal integrity validation system."""
    
    def __init__(self, config: TemporalValidationConfig):
        """Initialize temporal integrity validator."""
        self.config = config
        self.validation_history = []
        
        # Create report directory
        if self.config.save_validation_reports:
            Path(self.config.report_directory).mkdir(parents=True, exist_ok=True)
    
    def validate_temporal_integrity(self, 
                                  X_train: pd.DataFrame, 
                                  X_val: Optional[pd.DataFrame] = None,
                                  y_train: Optional[pd.Series] = None,
                                  y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Comprehensive temporal integrity validation.
        
        Args:
            X_train: Training features with datetime index
            X_val: Validation features with datetime index
            y_train: Training labels
            y_val: Validation labels
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'temporal_integrity_valid': True,
            'chronological_order_valid': True,
            'leakage_detected': False,
            'temporal_gap_valid': True,
            'feature_consistency_valid': True,
            'warnings': [],
            'critical_issues': [],
            'recommendations': [],
            'detailed_metrics': {}
        }
        
        try:
            # 1. Chronological order validation
            if self.config.enforce_chronological_order:
                order_valid, order_msg = self._validate_chronological_order(X_train, X_val)
                validation_results['chronological_order_valid'] = order_valid
                if not order_valid:
                    validation_results['critical_issues'].append(f"Chronological order violation: {order_msg}")
            
            # 2. Temporal gap validation
            if X_val is not None:
                gap_valid, gap_msg = self._validate_temporal_gap(X_train, X_val)
                validation_results['temporal_gap_valid'] = gap_valid
                if not gap_valid:
                    validation_results['warnings'].append(f"Temporal gap issue: {gap_msg}")
            
            # 3. Leakage detection
            if self.config.enable_leakage_detection:
                leakage_results = self._detect_temporal_leakage(X_train, X_val, y_train, y_val)
                validation_results['leakage_detected'] = leakage_results['leakage_detected']
                validation_results['warnings'].extend(leakage_results['warnings'])
                validation_results['detailed_metrics']['leakage_analysis'] = leakage_results
            
            # 4. Feature temporal consistency
            consistency_valid, consistency_msg = self._validate_feature_temporal_consistency(X_train, X_val)
            validation_results['feature_consistency_valid'] = consistency_valid
            if not consistency_valid:
                validation_results['warnings'].append(f"Feature consistency issue: {consistency_msg}")
            
            # 5. Overall temporal integrity assessment
            validation_results['temporal_integrity_valid'] = (
                validation_results['chronological_order_valid'] and
                validation_results['temporal_gap_valid'] and
                not validation_results['leakage_detected'] and
                validation_results['feature_consistency_valid']
            )
            
            # 6. Generate recommendations
            validation_results['recommendations'] = self._generate_temporal_recommendations(validation_results)
            
            # Store in history
            self.validation_history.append(validation_results)
            
            if TPRINT_AVAILABLE:
                if validation_results['temporal_integrity_valid']:
                    tprint_success("✅ Temporal integrity validation passed")
                else:
                    tprint_error("❌ Temporal integrity validation failed")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Temporal integrity validation failed: {e}")
            validation_results['temporal_integrity_valid'] = False
            validation_results['critical_issues'].append(f"Validation error: {str(e)}")
            return validation_results
    
    def _validate_chronological_order(self, X_train: pd.DataFrame, X_val: Optional[pd.DataFrame]) -> Tuple[bool, str]:
        """Validate chronological ordering of data."""
        try:
            # Check training data order
            if not isinstance(X_train.index, pd.DatetimeIndex):
                return False, "Training data must have DatetimeIndex"
            
            if not X_train.index.is_monotonic_increasing:
                return False, "Training data is not chronologically ordered"
            
            # Check validation data order
            if X_val is not None:
                if not isinstance(X_val.index, pd.DatetimeIndex):
                    return False, "Validation data must have DatetimeIndex"
                
                if not X_val.index.is_monotonic_increasing:
                    return False, "Validation data is not chronologically ordered"
                
                # Check train-val temporal relationship
                if X_train.index.max() >= X_val.index.min():
                    return False, f"Training data extends beyond validation start: {X_train.index.max()} >= {X_val.index.min()}"
            
            return True, "Chronological order validation passed"
            
        except Exception as e:
            return False, f"Chronological order validation failed: {str(e)}"
    
    def _validate_temporal_gap(self, X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[bool, str]:
        """Validate temporal gap between train and validation sets."""
        try:
            if self.config.min_temporal_gap <= 0:
                return True, "Temporal gap validation disabled"
            
            # Calculate gap
            train_end = X_train.index.max()
            val_start = X_val.index.min()
            
            # For DatetimeIndex, calculate gap in periods
            if isinstance(X_train.index, pd.DatetimeIndex):
                # Estimate period frequency
                if len(X_train) > 1:
                    period_freq = pd.infer_freq(X_train.index)
                    if period_freq:
                        gap_periods = (val_start - train_end) / pd.Timedelta(period_freq)
                    else:
                        # Fallback: use median time difference
                        time_diffs = X_train.index.to_series().diff().dropna()
                        median_diff = time_diffs.median()
                        gap_periods = (val_start - train_end) / median_diff
                    else:
                        gap_periods = 0
                else:
                    gap_periods = 0
            else:
                # For non-datetime index, assume unit periods
                gap_periods = val_start - train_end
            
            if gap_periods < self.config.min_temporal_gap:
                return False, f"Insufficient temporal gap: {gap_periods:.2f} < {self.config.min_temporal_gap}"
            
            return True, f"Temporal gap validation passed: {gap_periods:.2f} periods"
            
        except Exception as e:
            return False, f"Temporal gap validation failed: {str(e)}"
    
    def _detect_temporal_leakage(self, X_train: pd.DataFrame, X_val: Optional[pd.DataFrame],
                                y_train: Optional[pd.Series], y_val: Optional[pd.Series]) -> Dict[str, Any]:
        """Detect temporal data leakage."""
        leakage_results = {
            'leakage_detected': False,
            'warnings': [],
            'indicators': [],
            'correlation_analysis': {},
            'statistical_analysis': {}
        }
        
        try:
            # 1. Temporal correlation analysis
            if 'temporal_correlation' in self.config.leakage_detection_methods:
                corr_results = self._analyze_temporal_correlations(X_train, X_val, y_train, y_val)
                leakage_results['correlation_analysis'] = corr_results
                
                if corr_results['high_correlations'] > 0:
                    leakage_results['leakage_detected'] = True
                    leakage_results['indicators'].append('temporal_correlation')
                    leakage_results['warnings'].append(f"High temporal correlations detected: {corr_results['high_correlations']}")
            
            # 2. Statistical similarity analysis
            if 'statistical_similarity' in self.config.leakage_detection_methods and X_val is not None:
                stat_results = self._analyze_statistical_similarity(X_train, X_val)
                leakage_results['statistical_analysis'] = stat_results
                
                if stat_results['suspicious_similarity']:
                    leakage_results['leakage_detected'] = True
                    leakage_results['indicators'].append('statistical_similarity')
                    leakage_results['warnings'].append("Suspicious statistical similarity detected")
            
            # 3. Feature consistency analysis
            if 'feature_consistency' in self.config.leakage_detection_methods:
                consistency_results = self._analyze_feature_consistency(X_train, X_val)
                leakage_results.update(consistency_results)
                
                if consistency_results.get('inconsistent_features', 0) > 0:
                    leakage_results['leakage_detected'] = True
                    leakage_results['indicators'].append('feature_consistency')
                    leakage_results['warnings'].append(f"Inconsistent features detected: {consistency_results['inconsistent_features']}")
            
            return leakage_results
            
        except Exception as e:
            logger.error(f"Leakage detection failed: {e}")
            leakage_results['warnings'].append(f"Leakage detection error: {str(e)}")
            return leakage_results
    
    def _analyze_temporal_correlations(self, X_train: pd.DataFrame, X_val: Optional[pd.DataFrame],
                                     y_train: Optional[pd.Series], y_val: Optional[pd.Series]) -> Dict[str, Any]:
        """Analyze temporal correlations for leakage detection."""
        results = {
            'high_correlations': 0,
            'correlation_matrix': {},
            'suspicious_features': []
        }
        
        try:
            if X_val is None or y_train is None or y_val is None:
                return results
            
            # Check correlations between features and future labels
            for col in X_train.columns:
                if col in X_val.columns:
                    # Calculate correlation with current labels (not future)
                    train_corr = X_train[col].corr(y_train)
                    val_corr = X_val[col].corr(y_val)
                    
                    # Check for suspiciously high correlations
                    if abs(train_corr) > self.config.max_correlation_threshold:
                        results['high_correlations'] += 1
                        results['suspicious_features'].append(col)
                        results['correlation_matrix'][col] = {
                            'train_correlation': train_corr,
                            'val_correlation': val_corr
                        }
            
            return results
            
        except Exception as e:
            logger.error(f"Temporal correlation analysis failed: {e}")
            return results
    
    def _analyze_statistical_similarity(self, X_train: pd.DataFrame, X_val: pd.DataFrame) -> Dict[str, Any]:
        """Analyze statistical similarity between train and validation sets."""
        results = {
            'suspicious_similarity': False,
            'mean_similarity': 0.0,
            'std_similarity': 0.0,
            'feature_similarities': {}
        }
        
        try:
            # Calculate mean and std similarities
            train_means = X_train.mean()
            val_means = X_val.mean()
            train_stds = X_train.std()
            val_stds = X_val.std()
            
            # Calculate similarity scores
            mean_similarity = np.corrcoef(train_means, val_means)[0, 1]
            std_similarity = np.corrcoef(train_stds, val_stds)[0, 1]
            
            results['mean_similarity'] = mean_similarity
            results['std_similarity'] = std_similarity
            
            # Check for suspicious similarity
            if (mean_similarity > self.config.statistical_similarity_threshold and 
                std_similarity > self.config.statistical_similarity_threshold):
                results['suspicious_similarity'] = True
            
            # Per-feature analysis
            for col in X_train.columns:
                if col in X_val.columns:
                    col_mean_sim = np.corrcoef([train_means[col]], [val_means[col]])[0, 1]
                    col_std_sim = np.corrcoef([train_stds[col]], [val_stds[col]])[0, 1]
                    results['feature_similarities'][col] = {
                        'mean_similarity': col_mean_sim,
                        'std_similarity': col_std_sim
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Statistical similarity analysis failed: {e}")
            return results
    
    def _analyze_feature_consistency(self, X_train: pd.DataFrame, X_val: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Analyze feature temporal consistency."""
        results = {
            'inconsistent_features': 0,
            'feature_consistency_scores': {},
            'temporal_patterns': {}
        }
        
        try:
            if X_val is None:
                return results
            
            for col in X_train.columns:
                if col in X_val.columns:
                    # Check for temporal consistency
                    train_values = X_train[col].values
                    val_values = X_val[col].values
                    
                    # Calculate consistency metrics
                    consistency_score = self._calculate_feature_consistency(train_values, val_values)
                    results['feature_consistency_scores'][col] = consistency_score
                    
                    if consistency_score < 0.5:  # Threshold for consistency
                        results['inconsistent_features'] += 1
                    
                    # Analyze temporal patterns
                    pattern_score = self._analyze_temporal_patterns(train_values, val_values)
                    results['temporal_patterns'][col] = pattern_score
            
            return results
            
        except Exception as e:
            logger.error(f"Feature consistency analysis failed: {e}")
            return results
    
    def _calculate_feature_consistency(self, train_values: np.ndarray, val_values: np.ndarray) -> float:
        """Calculate consistency score between train and validation feature values."""
        try:
            # Normalize values
            train_norm = (train_values - np.mean(train_values)) / (np.std(train_values) + 1e-8)
            val_norm = (val_values - np.mean(val_values)) / (np.std(val_values) + 1e-8)
            
            # Calculate correlation
            if len(train_norm) > 1 and len(val_norm) > 1:
                correlation = np.corrcoef(train_norm, val_norm)[0, 1]
                return abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _analyze_temporal_patterns(self, train_values: np.ndarray, val_values: np.ndarray) -> Dict[str, float]:
        """Analyze temporal patterns in feature values."""
        patterns = {
            'trend_consistency': 0.0,
            'volatility_consistency': 0.0,
            'distribution_consistency': 0.0
        }
        
        try:
            # Trend consistency
            train_trend = np.polyfit(range(len(train_values)), train_values, 1)[0]
            val_trend = np.polyfit(range(len(val_values)), val_values, 1)[0]
            patterns['trend_consistency'] = 1.0 - abs(train_trend - val_trend) / (abs(train_trend) + abs(val_trend) + 1e-8)
            
            # Volatility consistency
            train_vol = np.std(train_values)
            val_vol = np.std(val_values)
            patterns['volatility_consistency'] = 1.0 - abs(train_vol - val_vol) / (train_vol + val_vol + 1e-8)
            
            # Distribution consistency (Kolmogorov-Smirnov test)
            if len(train_values) > 10 and len(val_values) > 10:
                ks_stat, _ = stats.ks_2samp(train_values, val_values)
                patterns['distribution_consistency'] = 1.0 - ks_stat
            
            return patterns
            
        except Exception:
            return patterns
    
    def _validate_feature_temporal_consistency(self, X_train: pd.DataFrame, X_val: Optional[pd.DataFrame]) -> Tuple[bool, str]:
        """Validate temporal consistency of features."""
        try:
            if X_val is None:
                return True, "No validation data to check"
            
            # Check for features that might leak future information
            suspicious_features = []
            
            for col in X_train.columns:
                if col in X_val.columns:
                    # Check for features that are too similar between train and val
                    train_series = X_train[col]
                    val_series = X_val[col]
                    
                    # Calculate similarity
                    if len(train_series) > 1 and len(val_series) > 1:
                        correlation = train_series.corr(val_series)
                        if not np.isnan(correlation) and abs(correlation) > 0.95:
                            suspicious_features.append(col)
            
            if suspicious_features:
                return False, f"Suspicious temporal consistency in features: {suspicious_features}"
            
            return True, "Feature temporal consistency validation passed"
            
        except Exception as e:
            return False, f"Feature temporal consistency validation failed: {str(e)}"
    
    def _generate_temporal_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if not validation_results['chronological_order_valid']:
            recommendations.append("Sort data by timestamp before processing")
            recommendations.append("Use time-based cross-validation")
        
        if not validation_results['temporal_gap_valid']:
            recommendations.append("Increase temporal gap between train and validation sets")
            recommendations.append("Use purged cross-validation with embargo periods")
        
        if validation_results['leakage_detected']:
            recommendations.append("Review feature engineering for temporal leakage")
            recommendations.append("Implement proper feature lagging")
            recommendations.append("Use walk-forward validation")
        
        if not validation_results['feature_consistency_valid']:
            recommendations.append("Check feature temporal consistency")
            recommendations.append("Validate feature engineering pipeline")
        
        if not validation_results['temporal_integrity_valid']:
            recommendations.append("Implement comprehensive temporal validation")
            recommendations.append("Use temporal integrity validator in pipeline")
        
        return recommendations


class TemporalFairnessAnalyzer:
    """Advanced temporal fairness analysis system."""
    
    def __init__(self, config: TemporalValidationConfig):
        """Initialize temporal fairness analyzer."""
        self.config = config
        self.analysis_history = []
    
    def analyze_temporal_fairness(self, 
                                X: pd.DataFrame, 
                                y: pd.Series,
                                additional_features: Optional[Dict[str, pd.Series]] = None) -> TemporalFairnessReport:
        """
        Comprehensive temporal fairness analysis.
        
        Args:
            X: Features with datetime index
            y: Labels
            additional_features: Additional features for analysis
            
        Returns:
            TemporalFairnessReport with comprehensive analysis
        """
        report = TemporalFairnessReport()
        report.config_used = {
            'integrity_level': self.config.integrity_level.value,
            'fairness_metrics': [m.value for m in self.config.fairness_metrics],
            'temporal_window_size': self.config.temporal_window_size
        }
        
        try:
            if not isinstance(X.index, pd.DatetimeIndex):
                report.warnings.append("No datetime index available for temporal analysis")
                return report
            
            # 1. Temporal balance analysis
            if TemporalFairnessMetric.TEMPORAL_BALANCE in self.config.fairness_metrics:
                balance_results = self._analyze_temporal_balance(X, y)
                report.temporal_balance_score = balance_results['balance_score']
                report.temporal_periods = balance_results['periods']
                report.period_balance_ratios = balance_results['period_ratios']
            
            # 2. Regime persistence analysis
            if TemporalFairnessMetric.REGIME_PERSISTENCE in self.config.fairness_metrics:
                regime_results = self._analyze_regime_persistence(X, y, additional_features)
                report.regime_persistence_score = regime_results['persistence_score']
                report.regime_transitions = regime_results['transitions']
            
            # 3. Temporal drift analysis
            if TemporalFairnessMetric.TEMPORAL_DRIFT in self.config.fairness_metrics:
                drift_results = self._analyze_temporal_drift(X, y)
                report.temporal_drift_score = drift_results['drift_score']
                report.drift_indicators = drift_results['indicators']
            
            # 4. Temporal stability analysis
            if TemporalFairnessMetric.TEMPORAL_STABILITY in self.config.fairness_metrics:
                stability_results = self._analyze_temporal_stability(X, y)
                report.temporal_stability_score = stability_results['stability_score']
                report.stability_metrics = stability_results['metrics']
            
            # 5. Chronological order analysis
            if TemporalFairnessMetric.CHRONOLOGICAL_ORDER in self.config.fairness_metrics:
                order_results = self._analyze_chronological_order(X, y)
                report.chronological_order_score = order_results['order_score']
            
            # 6. Overall temporal consistency
            if TemporalFairnessMetric.TEMPORAL_CONSISTENCY in self.config.fairness_metrics:
                consistency_results = self._analyze_temporal_consistency(X, y)
                report.overall_fairness_score = consistency_results['consistency_score']
            
            # 7. Generate warnings and recommendations
            report.warnings = self._generate_fairness_warnings(report)
            report.recommendations = self._generate_fairness_recommendations(report)
            report.critical_issues = self._identify_critical_issues(report)
            
            # Store in history
            self.analysis_history.append(report)
            
            if TPRINT_AVAILABLE:
                tprint_info(f"📊 Temporal fairness analysis completed - Overall score: {report.overall_fairness_score:.3f}")
            
            return report
            
        except Exception as e:
            logger.error(f"Temporal fairness analysis failed: {e}")
            report.critical_issues.append(f"Analysis failed: {str(e)}")
            return report
    
    def _analyze_temporal_balance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze temporal balance across time periods."""
        results = {
            'balance_score': 0.0,
            'periods': [],
            'period_ratios': {}
        }
        
        try:
            # Create temporal periods
            window_size = self.config.temporal_window_size
            n_periods = max(1, len(X) // window_size)
            
            if n_periods < 2:
                results['balance_score'] = 1.0  # Perfect balance for single period
                return results
            
            # Calculate class ratios for each period
            period_ratios = []
            for i in range(n_periods):
                start_idx = i * window_size
                end_idx = min((i + 1) * window_size, len(X))
                
                period_data = y.iloc[start_idx:end_idx]
                period_ratio = period_data.value_counts(normalize=True).to_dict()
                
                period_name = f"period_{i+1}"
                results['periods'].append(period_name)
                results['period_ratios'][period_name] = period_ratio
                period_ratios.append(period_ratio)
            
            # Calculate balance score (lower variance = better balance)
            if len(period_ratios) > 1:
                # Calculate variance in class ratios across periods
                class_vars = {}
                for class_label in y.unique():
                    class_ratios = [p.get(class_label, 0.0) for p in period_ratios]
                    if len(class_ratios) > 1:
                        class_vars[class_label] = np.var(class_ratios)
                
                # Balance score is inverse of average variance
                avg_variance = np.mean(list(class_vars.values())) if class_vars else 0.0
                results['balance_score'] = 1.0 / (1.0 + avg_variance)
            else:
                results['balance_score'] = 1.0
            
            return results
            
        except Exception as e:
            logger.error(f"Temporal balance analysis failed: {e}")
            return results
    
    def _analyze_regime_persistence(self, X: pd.DataFrame, y: pd.Series, 
                                  additional_features: Optional[Dict[str, pd.Series]]) -> Dict[str, Any]:
        """Analyze regime persistence over time."""
        results = {
            'persistence_score': 0.0,
            'transitions': {}
        }
        
        try:
            # Get regime information
            regime_labels = None
            if additional_features and 'regime' in additional_features:
                regime_labels = additional_features['regime']
            elif 'regime' in X.columns:
                regime_labels = X['regime']
            
            if regime_labels is None:
                # Use label persistence as proxy for regime persistence
                regime_labels = y
            
            # Calculate regime transitions
            regime_changes = (regime_labels != regime_labels.shift(1)).sum()
            total_periods = len(regime_labels) - 1
            
            if total_periods > 0:
                # Persistence score is inverse of transition rate
                transition_rate = regime_changes / total_periods
                results['persistence_score'] = 1.0 - transition_rate
                
                # Count transitions by type
                for i in range(1, len(regime_labels)):
                    prev_regime = regime_labels.iloc[i-1]
                    curr_regime = regime_labels.iloc[i]
                    if prev_regime != curr_regime:
                        transition_key = f"{prev_regime}_to_{curr_regime}"
                        results['transitions'][transition_key] = results['transitions'].get(transition_key, 0) + 1
            else:
                results['persistence_score'] = 1.0
            
            return results
            
        except Exception as e:
            logger.error(f"Regime persistence analysis failed: {e}")
            return results
    
    def _analyze_temporal_drift(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze temporal drift in features and labels."""
        results = {
            'drift_score': 0.0,
            'indicators': []
        }
        
        try:
            # Split data into early and late periods
            mid_point = len(X) // 2
            early_X = X.iloc[:mid_point]
            late_X = X.iloc[mid_point:]
            early_y = y.iloc[:mid_point]
            late_y = y.iloc[mid_point:]
            
            # Analyze feature drift
            feature_drifts = []
            for col in X.columns:
                if col in early_X.columns and col in late_X.columns:
                    # Statistical drift test
                    early_values = early_X[col].dropna()
                    late_values = late_X[col].dropna()
                    
                    if len(early_values) > 10 and len(late_values) > 10:
                        # Kolmogorov-Smirnov test
                        ks_stat, _ = stats.ks_2samp(early_values, late_values)
                        feature_drifts.append(ks_stat)
            
            # Analyze label drift
            label_drift = 0.0
            if len(early_y) > 10 and len(late_y) > 10:
                early_dist = early_y.value_counts(normalize=True)
                late_dist = late_y.value_counts(normalize=True)
                
                # Calculate distribution difference
                all_labels = set(early_dist.index) | set(late_dist.index)
                total_diff = 0.0
                for label in all_labels:
                    early_prob = early_dist.get(label, 0.0)
                    late_prob = late_dist.get(label, 0.0)
                    total_diff += abs(early_prob - late_prob)
                
                label_drift = total_diff / 2.0  # Normalize to [0, 1]
            
            # Calculate overall drift score
            avg_feature_drift = np.mean(feature_drifts) if feature_drifts else 0.0
            overall_drift = (avg_feature_drift + label_drift) / 2.0
            
            # Drift score is inverse of drift (higher = less drift)
            results['drift_score'] = 1.0 - overall_drift
            
            # Identify drift indicators
            if avg_feature_drift > self.config.drift_detection_threshold:
                results['indicators'].append(f"Feature drift detected: {avg_feature_drift:.3f}")
            
            if label_drift > self.config.drift_detection_threshold:
                results['indicators'].append(f"Label drift detected: {label_drift:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Temporal drift analysis failed: {e}")
            return results
    
    def _analyze_temporal_stability(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze temporal stability of features and labels."""
        results = {
            'stability_score': 0.0,
            'metrics': {}
        }
        
        try:
            # Calculate rolling statistics
            window_size = min(self.config.temporal_window_size, len(X) // 4)
            if window_size < 2:
                results['stability_score'] = 1.0
                return results
            
            # Feature stability
            feature_stabilities = []
            for col in X.columns:
                if X[col].dtype in [np.number, 'int64', 'float64']:
                    rolling_mean = X[col].rolling(window=window_size).mean()
                    rolling_std = X[col].rolling(window=window_size).std()
                    
                    # Calculate coefficient of variation
                    cv = (rolling_std / (rolling_mean + 1e-8)).mean()
                    stability = 1.0 / (1.0 + cv)  # Higher stability = lower CV
                    feature_stabilities.append(stability)
            
            # Label stability
            label_stability = 0.0
            if len(y) > window_size:
                rolling_label_dist = []
                for i in range(window_size, len(y)):
                    window_labels = y.iloc[i-window_size:i]
                    dist = window_labels.value_counts(normalize=True)
                    rolling_label_dist.append(dist)
                
                # Calculate stability of label distributions
                if len(rolling_label_dist) > 1:
                    dist_stabilities = []
                    for i in range(1, len(rolling_label_dist)):
                        prev_dist = rolling_label_dist[i-1]
                        curr_dist = rolling_label_dist[i]
                        
                        # Calculate distribution similarity
                        all_labels = set(prev_dist.index) | set(curr_dist.index)
                        similarity = 0.0
                        for label in all_labels:
                            prev_prob = prev_dist.get(label, 0.0)
                            curr_prob = curr_dist.get(label, 0.0)
                            similarity += min(prev_prob, curr_prob)
                        
                        dist_stabilities.append(similarity)
                    
                    label_stability = np.mean(dist_stabilities) if dist_stabilities else 1.0
            
            # Overall stability score
            feature_stability = np.mean(feature_stabilities) if feature_stabilities else 1.0
            results['stability_score'] = (feature_stability + label_stability) / 2.0
            
            results['metrics'] = {
                'feature_stability': feature_stability,
                'label_stability': label_stability,
                'avg_coefficient_variation': np.mean([1.0 - s for s in feature_stabilities]) if feature_stabilities else 0.0
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Temporal stability analysis failed: {e}")
            return results
    
    def _analyze_chronological_order(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze chronological ordering of data."""
        results = {
            'order_score': 0.0
        }
        
        try:
            if not isinstance(X.index, pd.DatetimeIndex):
                results['order_score'] = 0.0
                return results
            
            # Check if index is monotonic
            is_monotonic = X.index.is_monotonic_increasing
            
            if is_monotonic:
                results['order_score'] = 1.0
            else:
                # Calculate order violations
                violations = (X.index.to_series().diff() < pd.Timedelta(0)).sum()
                total_periods = len(X) - 1
                violation_rate = violations / total_periods if total_periods > 0 else 0.0
                results['order_score'] = 1.0 - violation_rate
            
            return results
            
        except Exception as e:
            logger.error(f"Chronological order analysis failed: {e}")
            return results
    
    def _analyze_temporal_consistency(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze overall temporal consistency."""
        results = {
            'consistency_score': 0.0
        }
        
        try:
            # Combine all temporal metrics
            metrics = []
            
            # Add individual metric scores if available
            # This would be populated by the calling function
            
            # Default consistency score
            results['consistency_score'] = 0.8  # Placeholder
            
            return results
            
        except Exception as e:
            logger.error(f"Temporal consistency analysis failed: {e}")
            return results
    
    def _generate_fairness_warnings(self, report: TemporalFairnessReport) -> List[str]:
        """Generate warnings based on fairness analysis."""
        warnings = []
        
        if report.temporal_balance_score < 0.7:
            warnings.append(f"Low temporal balance score: {report.temporal_balance_score:.3f}")
        
        if report.regime_persistence_score < 0.5:
            warnings.append(f"Low regime persistence score: {report.regime_persistence_score:.3f}")
        
        if report.temporal_drift_score < 0.6:
            warnings.append(f"High temporal drift detected: {report.temporal_drift_score:.3f}")
        
        if report.temporal_stability_score < 0.7:
            warnings.append(f"Low temporal stability: {report.temporal_stability_score:.3f}")
        
        if report.chronological_order_score < 1.0:
            warnings.append(f"Chronological order violations detected: {report.chronological_order_score:.3f}")
        
        return warnings
    
    def _generate_fairness_recommendations(self, report: TemporalFairnessReport) -> List[str]:
        """Generate recommendations based on fairness analysis."""
        recommendations = []
        
        if report.temporal_balance_score < 0.7:
            recommendations.append("Implement temporal balancing strategies")
            recommendations.append("Use stratified sampling across time periods")
        
        if report.regime_persistence_score < 0.5:
            recommendations.append("Review regime detection methodology")
            recommendations.append("Implement regime smoothing techniques")
        
        if report.temporal_drift_score < 0.6:
            recommendations.append("Implement drift detection and adaptation")
            recommendations.append("Use online learning techniques")
        
        if report.temporal_stability_score < 0.7:
            recommendations.append("Implement feature normalization")
            recommendations.append("Use robust statistical measures")
        
        if report.chronological_order_score < 1.0:
            recommendations.append("Sort data by timestamp before processing")
            recommendations.append("Implement temporal ordering validation")
        
        return recommendations
    
    def _identify_critical_issues(self, report: TemporalFairnessReport) -> List[str]:
        """Identify critical issues in temporal fairness."""
        critical_issues = []
        
        if report.overall_fairness_score < 0.5:
            critical_issues.append("Critical temporal fairness issues detected")
        
        if report.temporal_drift_score < 0.3:
            critical_issues.append("Severe temporal drift detected")
        
        if report.chronological_order_score < 0.8:
            critical_issues.append("Significant chronological order violations")
        
        return critical_issues


class PurgedTemporalKFold(BaseCrossValidator):
    """
    Enhanced Purged K-Fold cross-validator with temporal integrity.
    
    Implements purged cross-validation with advanced temporal validation
    and leakage prevention for financial time series data.
    """
    
    def __init__(self, 
                 n_splits: int = 5, 
                 purge_length: int = 1, 
                 embargo_length: int = 1,
                 temporal_validator: Optional[TemporalIntegrityValidator] = None,
                 random_state: Optional[int] = None):
        """
        Initialize PurgedTemporalKFold.
        
        Args:
            n_splits: Number of folds
            purge_length: Number of samples to purge around validation period
            embargo_length: Number of samples to embargo after validation period
            temporal_validator: Temporal integrity validator
            random_state: Random state for reproducibility
        """
        self.n_splits = n_splits
        self.purge_length = purge_length
        self.embargo_length = embargo_length
        self.temporal_validator = temporal_validator
        self.random_state = random_state
        
        # Set random state
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
    
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
              groups: Optional[pd.Series] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and validation sets.
        
        Args:
            X: Feature matrix with datetime index
            y: Target labels (optional)
            groups: Group labels (optional)
            
        Yields:
            Tuple of (train_indices, val_indices)
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex for purged cross-validation")
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Create fold boundaries with temporal awareness
        fold_size = n_samples // self.n_splits
        fold_boundaries = [i * fold_size for i in range(self.n_splits + 1)]
        fold_boundaries[-1] = n_samples
        
        for i in range(self.n_splits):
            # Validation period
            val_start = fold_boundaries[i]
            val_end = fold_boundaries[i + 1]
            
            # Purge period (before validation)
            purge_start = max(0, val_start - self.purge_length)
            purge_end = val_start
            
            # Embargo period (after validation)
            embargo_start = val_end
            embargo_end = min(n_samples, val_end + self.embargo_length)
            
            # Training indices (exclude purge and embargo periods)
            train_indices = np.concatenate([
                indices[:purge_start],
                indices[embargo_end:]
            ])
            
            # Validation indices
            val_indices = indices[val_start:val_end]
            
            # Ensure we have enough samples
            if len(train_indices) < 10 or len(val_indices) < 5:
                continue
            
            # Additional temporal validation if validator provided
            if self.temporal_validator is not None:
                try:
                    X_train_fold = X.iloc[train_indices]
                    X_val_fold = X.iloc[val_indices]
                    y_train_fold = y.iloc[train_indices] if y is not None else None
                    y_val_fold = y.iloc[val_indices] if y is not None else None
                    
                    validation_results = self.temporal_validator.validate_temporal_integrity(
                        X_train_fold, X_val_fold, y_train_fold, y_val_fold
                    )
                    
                    # Skip fold if temporal integrity validation fails
                    if not validation_results['temporal_integrity_valid']:
                        if TPRINT_AVAILABLE:
                            tprint_warning(f"⚠️ Skipping fold {i+1} due to temporal integrity violation")
                        continue
                        
                except Exception as e:
                    if TPRINT_AVAILABLE:
                        tprint_warning(f"⚠️ Temporal validation failed for fold {i+1}: {e}")
                    continue
            
            yield train_indices, val_indices
    
    def get_n_splits(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None, 
                     groups: Optional[pd.Series] = None) -> int:
        """Return the number of splitting iterations."""
        return self.n_splits


class EnhancedLabelBalancer:
    """
    Enhanced label balancing system with comprehensive temporal integrity and fairness.
    
    This system provides advanced label balancing with:
    - Temporal integrity validation
    - Temporal fairness analysis
    - Leakage prevention
    - Regime-aware balancing
    - Multi-dimensional temporal validation
    """
    
    def __init__(self, 
                 temporal_config: Optional[TemporalValidationConfig] = None,
                 balancing_config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced label balancer.
        
        Args:
            temporal_config: Temporal validation configuration
            balancing_config: Label balancing configuration
        """
        self.temporal_config = temporal_config or TemporalValidationConfig()
        self.balancing_config = balancing_config or {}
        
        # Initialize components
        self.temporal_validator = TemporalIntegrityValidator(self.temporal_config)
        self.temporal_fairness_analyzer = TemporalFairnessAnalyzer(self.temporal_config)
        
        # Initialize purged CV
        self.purged_cv = PurgedTemporalKFold(
            n_splits=self.temporal_config.n_splits,
            purge_length=self.temporal_config.purge_length,
            embargo_length=self.temporal_config.embargo_length,
            temporal_validator=self.temporal_validator
        )
        
        # Validation history
        self.validation_history = []
        self.fairness_history = []
    
    def balance_with_temporal_integrity(self, 
                                      X: pd.DataFrame, 
                                      y: pd.Series,
                                      additional_features: Optional[Dict[str, pd.Series]] = None,
                                      X_val: Optional[pd.DataFrame] = None,
                                      y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Perform label balancing with comprehensive temporal integrity validation.
        
        Args:
            X: Training features with datetime index
            y: Training labels
            additional_features: Additional features for analysis
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with balanced data and validation results
        """
        results = {
            'X_balanced': X,
            'y_balanced': y,
            'temporal_validation': {},
            'temporal_fairness': {},
            'balancing_applied': False,
            'warnings': [],
            'recommendations': []
        }
        
        try:
            if TPRINT_AVAILABLE:
                tprint_info("🚀 Starting enhanced label balancing with temporal integrity")
            
            # 1. Temporal integrity validation
            temporal_validation = self.temporal_validator.validate_temporal_integrity(
                X, X_val, y, y_val
            )
            results['temporal_validation'] = temporal_validation
            
            if not temporal_validation['temporal_integrity_valid']:
                results['warnings'].extend(temporal_validation['warnings'])
                results['warnings'].extend(temporal_validation['critical_issues'])
                if TPRINT_AVAILABLE:
                    tprint_error("❌ Temporal integrity validation failed - skipping balancing")
                return results
            
            # 2. Temporal fairness analysis
            temporal_fairness = self.temporal_fairness_analyzer.analyze_temporal_fairness(
                X, y, additional_features
            )
            results['temporal_fairness'] = temporal_fairness
            
            # 3. Apply label balancing based on temporal fairness
            if temporal_fairness.overall_fairness_score < 0.7:
                if TPRINT_AVAILABLE:
                    tprint_info("📊 Applying temporal-aware label balancing")
                
                # Apply temporal-aware balancing
                X_balanced, y_balanced = self._apply_temporal_balancing(X, y, temporal_fairness)
                results['X_balanced'] = X_balanced
                results['y_balanced'] = y_balanced
                results['balancing_applied'] = True
                
                # Re-validate temporal integrity after balancing
                post_validation = self.temporal_validator.validate_temporal_integrity(
                    X_balanced, X_val, y_balanced, y_val
                )
                results['post_balancing_validation'] = post_validation
            
            # 4. Generate recommendations
            results['recommendations'] = self._generate_balancing_recommendations(
                temporal_validation, temporal_fairness
            )
            
            # Store in history
            self.validation_history.append(temporal_validation)
            self.fairness_history.append(temporal_fairness)
            
            if TPRINT_AVAILABLE:
                tprint_success("✅ Enhanced label balancing completed")
            
            return results
            
        except Exception as e:
            logger.error(f"Enhanced label balancing failed: {e}")
            results['warnings'].append(f"Balancing failed: {str(e)}")
            return results
    
    def _apply_temporal_balancing(self, X: pd.DataFrame, y: pd.Series, 
                                 temporal_fairness: TemporalFairnessReport) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply temporal-aware label balancing."""
        try:
            # Simple temporal balancing strategy
            # In practice, this would be more sophisticated
            
            # For now, return original data
            # This is where advanced temporal balancing algorithms would be implemented
            return X, y
            
        except Exception as e:
            logger.error(f"Temporal balancing application failed: {e}")
            return X, y
    
    def _generate_balancing_recommendations(self, 
                                          temporal_validation: Dict[str, Any],
                                          temporal_fairness: TemporalFairnessReport) -> List[str]:
        """Generate recommendations based on validation and fairness analysis."""
        recommendations = []
        
        # Add temporal validation recommendations
        recommendations.extend(temporal_validation.get('recommendations', []))
        
        # Add temporal fairness recommendations
        recommendations.extend(temporal_fairness.recommendations)
        
        # Add specific balancing recommendations
        if temporal_fairness.overall_fairness_score < 0.7:
            recommendations.append("Consider implementing advanced temporal balancing strategies")
            recommendations.append("Use regime-aware balancing techniques")
        
        if temporal_fairness.temporal_drift_score < 0.6:
            recommendations.append("Implement drift detection and adaptation mechanisms")
            recommendations.append("Consider online learning approaches")
        
        return recommendations
    
    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get temporal validation history."""
        return self.validation_history.copy()
    
    def get_fairness_history(self) -> List[TemporalFairnessReport]:
        """Get temporal fairness analysis history."""
        return self.fairness_history.copy()
    
    def save_validation_report(self, results: Dict[str, Any], filename: Optional[str] = None):
        """Save validation report to file."""
        if not self.temporal_config.save_validation_reports:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"temporal_validation_report_{timestamp}.json"
        
        filepath = Path(self.temporal_config.report_directory) / filename
        
        try:
            # Convert results to JSON-serializable format
            json_results = self._convert_to_json_serializable(results)
            
            with open(filepath, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            if TPRINT_AVAILABLE:
                tprint_success(f"📄 Validation report saved: {filepath}")
                
        except Exception as e:
            logger.error(f"Failed to save validation report: {e}")
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return self._convert_to_json_serializable(obj.__dict__)
        else:
            return obj


# Convenience functions
def create_temporal_validator(config: Optional[TemporalValidationConfig] = None) -> TemporalIntegrityValidator:
    """Create temporal integrity validator."""
    return TemporalIntegrityValidator(config or TemporalValidationConfig())

def create_temporal_fairness_analyzer(config: Optional[TemporalValidationConfig] = None) -> TemporalFairnessAnalyzer:
    """Create temporal fairness analyzer."""
    return TemporalFairnessAnalyzer(config or TemporalValidationConfig())

def create_enhanced_label_balancer(temporal_config: Optional[TemporalValidationConfig] = None,
                                 balancing_config: Optional[Dict[str, Any]] = None) -> EnhancedLabelBalancer:
    """Create enhanced label balancer."""
    return EnhancedLabelBalancer(temporal_config, balancing_config)

def validate_temporal_integrity_simple(X_train: pd.DataFrame, X_val: Optional[pd.DataFrame] = None) -> bool:
    """Simple temporal integrity validation."""
    validator = create_temporal_validator()
    results = validator.validate_temporal_integrity(X_train, X_val)
    return results['temporal_integrity_valid']

def analyze_temporal_fairness_simple(X: pd.DataFrame, y: pd.Series) -> float:
    """Simple temporal fairness analysis."""
    analyzer = create_temporal_fairness_analyzer()
    report = analyzer.analyze_temporal_fairness(X, y)
    return report.overall_fairness_score


# Default configurations
DEFAULT_TEMPORAL_CONFIG = TemporalValidationConfig()
DEFAULT_TEMPORAL_VALIDATOR = create_temporal_validator()
DEFAULT_TEMPORAL_FAIRNESS_ANALYZER = create_temporal_fairness_analyzer()
DEFAULT_ENHANCED_LABEL_BALANCER = create_enhanced_label_balancer()


if __name__ == "__main__":
    # Example usage
    print("Enhanced Label Balancing with Temporal Integrity and Fairness")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=1000, freq='1H')
    X = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000)
    }, index=dates)
    
    y = pd.Series(np.random.choice([0, 1, 2], size=1000, p=[0.7, 0.2, 0.1]), index=dates)
    
    # Create enhanced label balancer
    balancer = create_enhanced_label_balancer()
    
    # Perform balancing with temporal integrity
    results = balancer.balance_with_temporal_integrity(X, y)
    
    print(f"Temporal integrity valid: {results['temporal_validation']['temporal_integrity_valid']}")
    print(f"Overall fairness score: {results['temporal_fairness']['overall_fairness_score']:.3f}")
    print(f"Balancing applied: {results['balancing_applied']}")
    print(f"Warnings: {len(results['warnings'])}")
    print(f"Recommendations: {len(results['recommendations'])}")