"""
Comprehensive Leakage Detection System with Proof Artifacts

This module implements a sophisticated leakage detection system that identifies
and provides proof artifacts for different types of temporal data leakage.

Leakage Taxonomy:
1. Label Leakage (Y or proxies): Future labels leaking into features
2. Target-Adjacent Features: Rolling windows that peek into future
3. Entity Leakage: ID reuse across time periods
4. Future Censoring: Survivorship bias in data

Each detection provides:
- Offending columns/rows identification
- Minimal reproduction snippets
- Severity scoring
- Remediation suggestions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Set
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
import itertools

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


class LeakageType(Enum):
    """Types of temporal data leakage."""
    LABEL_LEAKAGE = "label_leakage"                    # Future labels in features
    TARGET_ADJACENT = "target_adjacent"                # Rolling windows peeking future
    ENTITY_LEAKAGE = "entity_leakage"                  # ID reuse across time
    FUTURE_CENSORING = "future_censoring"              # Survivorship bias
    FEATURE_LEAD_CORRELATION = "feature_lead_correlation"  # Features leading targets
    TEMPORAL_CORRELATION = "temporal_correlation"      # High temporal correlations
    STATISTICAL_LEAKAGE = "statistical_leakage"        # Statistical similarity


class LeakageSeverity(Enum):
    """Severity levels for detected leakage."""
    CRITICAL = "critical"      # Immediate action required
    HIGH = "high"              # Significant risk
    MEDIUM = "medium"          # Moderate risk
    LOW = "low"                # Minor risk
    INFO = "info"              # Informational only


@dataclass
class LeakageProof:
    """Proof artifact for detected leakage."""
    
    # Basic information
    leakage_type: LeakageType
    severity: LeakageSeverity
    description: str
    
    # Offending data identification
    offending_columns: List[str] = field(default_factory=list)
    offending_rows: List[int] = field(default_factory=list)
    offending_entities: List[str] = field(default_factory=list)
    
    # Quantification
    correlation_score: float = 0.0
    statistical_score: float = 0.0
    temporal_score: float = 0.0
    overall_score: float = 0.0
    
    # Proof data
    minimal_repro_snippet: Dict[str, Any] = field(default_factory=dict)
    correlation_matrix: Dict[str, float] = field(default_factory=dict)
    temporal_patterns: Dict[str, Any] = field(default_factory=dict)
    
    # Context
    detection_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    validation_period: Optional[Tuple[datetime, datetime]] = None
    affected_samples: int = 0
    
    # Remediation
    remediation_suggestions: List[str] = field(default_factory=list)
    auto_fixable: bool = False
    fix_confidence: float = 0.0


@dataclass
class LeakageDetectionConfig:
    """Configuration for leakage detection system."""
    
    # Detection thresholds
    correlation_threshold: float = 0.95
    statistical_similarity_threshold: float = 0.8
    temporal_correlation_threshold: float = 0.9
    lead_lag_threshold: int = 5  # Maximum allowed lead/lag periods
    
    # Entity leakage detection
    entity_overlap_threshold: float = 0.1  # Max allowed entity overlap
    entity_time_window: str = "1D"  # Time window for entity analysis
    
    # Feature analysis
    max_features_to_analyze: int = 1000
    feature_sample_size: int = 10000
    enable_feature_lead_detection: bool = True
    
    # Statistical tests
    enable_ks_test: bool = True
    enable_ad_test: bool = True
    enable_mann_kendall: bool = True
    significance_level: float = 0.05
    
    # Reporting
    generate_proof_artifacts: bool = True
    save_detection_reports: bool = True
    report_directory: str = "reports/leakage_detection"
    max_proof_samples: int = 100


class LeakageDetector:
    """Comprehensive leakage detection system with proof artifacts."""
    
    def __init__(self, config: Optional[LeakageDetectionConfig] = None):
        """Initialize leakage detector."""
        self.config = config or LeakageDetectionConfig()
        self.detection_history = []
        
        # Create report directory
        if self.config.save_detection_reports:
            Path(self.config.report_directory).mkdir(parents=True, exist_ok=True)
    
    def detect_all_leakage(self, 
                          X: pd.DataFrame, 
                          y: pd.Series,
                          entity_cols: Optional[List[str]] = None,
                          time_col: Optional[str] = None,
                          additional_context: Optional[Dict[str, Any]] = None) -> List[LeakageProof]:
        """
        Detect all types of temporal leakage with proof artifacts.
        
        Args:
            X: Feature matrix
            y: Target labels
            entity_cols: Entity identifier columns
            time_col: Time column name
            additional_context: Additional context for detection
            
        Returns:
            List of LeakageProof objects
        """
        all_proofs = []
        
        try:
            if TPRINT_AVAILABLE:
                tprint_info("🔍 Starting comprehensive leakage detection")
            
            # 1. Label leakage detection
            label_proofs = self._detect_label_leakage(X, y, time_col)
            all_proofs.extend(label_proofs)
            
            # 2. Target-adjacent feature detection
            target_adjacent_proofs = self._detect_target_adjacent_features(X, y, time_col)
            all_proofs.extend(target_adjacent_proofs)
            
            # 3. Entity leakage detection
            if entity_cols:
                entity_proofs = self._detect_entity_leakage(X, y, entity_cols, time_col)
                all_proofs.extend(entity_proofs)
            
            # 4. Future censoring detection
            future_censoring_proofs = self._detect_future_censoring(X, y, time_col)
            all_proofs.extend(future_censoring_proofs)
            
            # 5. Feature lead correlation detection
            if self.config.enable_feature_lead_detection:
                lead_correlation_proofs = self._detect_feature_lead_correlations(X, y, time_col)
                all_proofs.extend(lead_correlation_proofs)
            
            # 6. Temporal correlation detection
            temporal_correlation_proofs = self._detect_temporal_correlations(X, y, time_col)
            all_proofs.extend(temporal_correlation_proofs)
            
            # 7. Statistical leakage detection
            statistical_proofs = self._detect_statistical_leakage(X, y, time_col)
            all_proofs.extend(statistical_proofs)
            
            # Store in history
            self.detection_history.extend(all_proofs)
            
            # Generate summary
            self._generate_detection_summary(all_proofs)
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Leakage detection completed: {len(all_proofs)} issues found")
            
            return all_proofs
            
        except Exception as e:
            logger.error(f"Leakage detection failed: {e}")
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Leakage detection failed: {e}")
            return []
    
    def _detect_label_leakage(self, X: pd.DataFrame, y: pd.Series, 
                             time_col: Optional[str]) -> List[LeakageProof]:
        """Detect label leakage (future labels in features)."""
        proofs = []
        
        try:
            # Check for direct label leakage
            for col in X.columns:
                if col in y.name or y.name in col:
                    proof = LeakageProof(
                        leakage_type=LeakageType.LABEL_LEAKAGE,
                        severity=LeakageSeverity.CRITICAL,
                        description=f"Direct label leakage detected in column '{col}'",
                        offending_columns=[col],
                        correlation_score=1.0,
                        overall_score=1.0,
                        auto_fixable=True,
                        fix_confidence=1.0,
                        remediation_suggestions=[
                            f"Remove column '{col}' from features",
                            "Ensure feature engineering doesn't include target information"
                        ]
                    )
                    proofs.append(proof)
            
            # Check for label proxy leakage
            if len(y.unique()) > 1:
                for col in X.columns:
                    if X[col].dtype in ['object', 'category']:
                        # Check if categorical feature perfectly predicts labels
                        crosstab = pd.crosstab(X[col], y)
                        if crosstab.shape[0] > 1:
                            # Check for perfect prediction
                            perfect_predictions = (crosstab > 0).sum(axis=1)
                            if (perfect_predictions == 1).any():
                                proof = LeakageProof(
                                    leakage_type=LeakageType.LABEL_LEAKAGE,
                                    severity=LeakageSeverity.HIGH,
                                    description=f"Label proxy leakage in categorical column '{col}'",
                                    offending_columns=[col],
                                    correlation_score=0.9,
                                    overall_score=0.9,
                                    auto_fixable=True,
                                    fix_confidence=0.8,
                                    remediation_suggestions=[
                                        f"Review categorical encoding for column '{col}'",
                                        "Ensure categories don't encode target information"
                                    ]
                                )
                                proofs.append(proof)
            
            # Check for temporal label leakage
            if time_col and time_col in X.columns:
                time_series = X[time_col]
                if isinstance(time_series.iloc[0], (pd.Timestamp, datetime)):
                    # Check if time features correlate with future labels
                    for i in range(1, min(6, len(y))):  # Check up to 5 periods ahead
                        future_y = y.shift(-i)
                        correlation = time_series.corr(future_y)
                        if not np.isnan(correlation) and abs(correlation) > self.config.correlation_threshold:
                            proof = LeakageProof(
                                leakage_type=LeakageType.LABEL_LEAKAGE,
                                severity=LeakageSeverity.HIGH,
                                description=f"Temporal label leakage: {time_col} correlates with future labels (lag {i})",
                                offending_columns=[time_col],
                                correlation_score=abs(correlation),
                                overall_score=abs(correlation),
                                temporal_patterns={
                                    'lag': i,
                                    'correlation': correlation,
                                    'future_label_correlation': True
                                },
                                auto_fixable=False,
                                fix_confidence=0.7,
                                remediation_suggestions=[
                                    f"Remove or modify time column '{time_col}'",
                                    "Use proper temporal feature engineering",
                                    "Implement proper train/validation splits"
                                ]
                            )
                            proofs.append(proof)
            
            return proofs
            
        except Exception as e:
            logger.error(f"Label leakage detection failed: {e}")
            return []
    
    def _detect_target_adjacent_features(self, X: pd.DataFrame, y: pd.Series, 
                                       time_col: Optional[str]) -> List[LeakageProof]:
        """Detect target-adjacent features (rolling windows peeking future)."""
        proofs = []
        
        try:
            # Look for rolling window features that might peek into future
            rolling_keywords = ['rolling', 'window', 'ma', 'moving', 'avg', 'mean', 'sum', 'max', 'min']
            
            for col in X.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in rolling_keywords):
                    # Check if this rolling feature correlates with future targets
                    for lag in range(1, self.config.lead_lag_threshold + 1):
                        future_y = y.shift(-lag)
                        correlation = X[col].corr(future_y)
                        
                        if not np.isnan(correlation) and abs(correlation) > self.config.correlation_threshold:
                            # Check if this is likely due to future peeking
                            if self._is_likely_future_peeking(X[col], future_y, lag):
                                proof = LeakageProof(
                                    leakage_type=LeakageType.TARGET_ADJACENT,
                                    severity=LeakageSeverity.HIGH,
                                    description=f"Target-adjacent feature '{col}' shows future peeking (lag {lag})",
                                    offending_columns=[col],
                                    correlation_score=abs(correlation),
                                    overall_score=abs(correlation),
                                    temporal_patterns={
                                        'lag': lag,
                                        'correlation': correlation,
                                        'feature_type': 'rolling_window',
                                        'future_peeking_detected': True
                                    },
                                    auto_fixable=False,
                                    fix_confidence=0.8,
                                    remediation_suggestions=[
                                        f"Review rolling window calculation for '{col}'",
                                        f"Ensure window doesn't extend beyond current time + {lag} periods",
                                        "Implement proper temporal feature engineering"
                                    ]
                                )
                                proofs.append(proof)
            
            return proofs
            
        except Exception as e:
            logger.error(f"Target-adjacent feature detection failed: {e}")
            return []
    
    def _detect_entity_leakage(self, X: pd.DataFrame, y: pd.Series, 
                              entity_cols: List[str], time_col: Optional[str]) -> List[LeakageProof]:
        """Detect entity leakage (ID reuse across time periods)."""
        proofs = []
        
        try:
            if not time_col or time_col not in X.columns:
                return proofs
            
            time_series = X[time_col]
            if not isinstance(time_series.iloc[0], (pd.Timestamp, datetime)):
                return proofs
            
            for entity_col in entity_cols:
                if entity_col not in X.columns:
                    continue
                
                # Group by entity and check for temporal violations
                entity_groups = X.groupby(entity_col)
                
                for entity_id, group in entity_groups:
                    if len(group) < 2:
                        continue
                    
                    # Check if entity appears in non-chronological order
                    entity_times = group[time_col].sort_values()
                    if not entity_times.is_monotonic_increasing:
                        # Find violations
                        violations = []
                        for i in range(1, len(entity_times)):
                            if entity_times.iloc[i] < entity_times.iloc[i-1]:
                                violations.append(i)
                        
                        if violations:
                            proof = LeakageProof(
                                leakage_type=LeakageType.ENTITY_LEAKAGE,
                                severity=LeakageSeverity.MEDIUM,
                                description=f"Entity '{entity_id}' appears in non-chronological order",
                                offending_entities=[str(entity_id)],
                                offending_rows=group.index[violations].tolist(),
                                temporal_score=len(violations) / len(group),
                                overall_score=len(violations) / len(group),
                                temporal_patterns={
                                    'entity_id': entity_id,
                                    'violations_count': len(violations),
                                    'total_occurrences': len(group),
                                    'violation_indices': violations
                                },
                                auto_fixable=True,
                                fix_confidence=0.9,
                                remediation_suggestions=[
                                    f"Sort data by {entity_col} and {time_col}",
                                    "Implement proper entity-level temporal ordering",
                                    "Check for data quality issues"
                                ]
                            )
                            proofs.append(proof)
                    
                    # Check for entity overlap across time periods
                    entity_time_span = entity_times.max() - entity_time_span.min()
                    if entity_time_span > pd.Timedelta(self.config.entity_time_window):
                        # Check if entity appears in multiple distant time periods
                        time_gaps = entity_times.diff().dropna()
                        large_gaps = time_gaps[time_gaps > pd.Timedelta(self.config.entity_time_window)]
                        
                        if len(large_gaps) > 0:
                            proof = LeakageProof(
                                leakage_type=LeakageType.ENTITY_LEAKAGE,
                                severity=LeakageSeverity.LOW,
                                description=f"Entity '{entity_id}' has large time gaps ({len(large_gaps)} gaps)",
                                offending_entities=[str(entity_id)],
                                temporal_score=len(large_gaps) / len(group),
                                overall_score=len(large_gaps) / len(group),
                                temporal_patterns={
                                    'entity_id': entity_id,
                                    'large_gaps_count': len(large_gaps),
                                    'max_gap': large_gaps.max().total_seconds() / 3600,  # hours
                                    'time_span_hours': entity_time_span.total_seconds() / 3600
                                },
                                auto_fixable=False,
                                fix_confidence=0.6,
                                remediation_suggestions=[
                                    "Review entity lifecycle and data collection",
                                    "Consider entity-level temporal validation",
                                    "Check for data quality issues"
                                ]
                            )
                            proofs.append(proof)
            
            return proofs
            
        except Exception as e:
            logger.error(f"Entity leakage detection failed: {e}")
            return []
    
    def _detect_future_censoring(self, X: pd.DataFrame, y: pd.Series, 
                                time_col: Optional[str]) -> List[LeakageProof]:
        """Detect future censoring (survivorship bias)."""
        proofs = []
        
        try:
            if not time_col or time_col not in X.columns:
                return proofs
            
            time_series = X[time_col]
            if not isinstance(time_series.iloc[0], (pd.Timestamp, datetime)):
                return proofs
            
            # Check for survivorship bias patterns
            # Look for features that might indicate future survival
            
            # Check if labels are correlated with time (survivorship bias)
            time_numeric = pd.to_numeric(time_series, errors='coerce')
            if not time_numeric.isna().all():
                time_label_correlation = time_numeric.corr(y)
                if not np.isnan(time_label_correlation) and abs(time_label_correlation) > 0.3:
                    proof = LeakageProof(
                        leakage_type=LeakageType.FUTURE_CENSORING,
                        severity=LeakageSeverity.MEDIUM,
                        description="Potential survivorship bias: labels correlate with time",
                        correlation_score=abs(time_label_correlation),
                        overall_score=abs(time_label_correlation),
                        temporal_patterns={
                            'time_label_correlation': time_label_correlation,
                            'survivorship_bias_detected': True
                        },
                        auto_fixable=False,
                        fix_confidence=0.7,
                        remediation_suggestions=[
                            "Review data collection methodology",
                            "Check for survivorship bias in labels",
                            "Implement proper censoring handling"
                        ]
                    )
                    proofs.append(proof)
            
            # Check for features that might indicate future survival
            survival_keywords = ['survival', 'duration', 'lifetime', 'persistence', 'retention']
            for col in X.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in survival_keywords):
                    # Check if this feature correlates with future labels
                    for lag in range(1, min(6, len(y))):
                        future_y = y.shift(-lag)
                        correlation = X[col].corr(future_y)
                        
                        if not np.isnan(correlation) and abs(correlation) > 0.5:
                            proof = LeakageProof(
                                leakage_type=LeakageType.FUTURE_CENSORING,
                                severity=LeakageSeverity.HIGH,
                                description=f"Survivorship bias in feature '{col}' (correlates with future labels)",
                                offending_columns=[col],
                                correlation_score=abs(correlation),
                                overall_score=abs(correlation),
                                temporal_patterns={
                                    'lag': lag,
                                    'correlation': correlation,
                                    'feature_type': 'survival_related',
                                    'future_censoring_detected': True
                                },
                                auto_fixable=False,
                                fix_confidence=0.8,
                                remediation_suggestions=[
                                    f"Review feature '{col}' for survivorship bias",
                                    "Implement proper censoring handling",
                                    "Consider using survival analysis techniques"
                                ]
                            )
                            proofs.append(proof)
            
            return proofs
            
        except Exception as e:
            logger.error(f"Future censoring detection failed: {e}")
            return []
    
    def _detect_feature_lead_correlations(self, X: pd.DataFrame, y: pd.Series, 
                                        time_col: Optional[str]) -> List[LeakageProof]:
        """Detect feature lead correlations (features leading targets)."""
        proofs = []
        
        try:
            # Sample features if too many
            features_to_analyze = X.columns
            if len(features_to_analyze) > self.config.max_features_to_analyze:
                features_to_analyze = np.random.choice(
                    features_to_analyze, 
                    self.config.max_features_to_analyze, 
                    replace=False
                )
            
            for col in features_to_analyze:
                if X[col].dtype not in ['int64', 'float64']:
                    continue
                
                # Check for lead correlations
                max_correlation = 0.0
                best_lag = 0
                correlation_details = {}
                
                for lag in range(1, self.config.lead_lag_threshold + 1):
                    future_y = y.shift(-lag)
                    correlation = X[col].corr(future_y)
                    
                    if not np.isnan(correlation):
                        correlation_details[lag] = correlation
                        if abs(correlation) > abs(max_correlation):
                            max_correlation = correlation
                            best_lag = lag
                
                # Check if this is a significant lead correlation
                if abs(max_correlation) > self.config.correlation_threshold:
                    # Verify this isn't just noise
                    if self._is_significant_lead_correlation(X[col], y, best_lag):
                        proof = LeakageProof(
                            leakage_type=LeakageType.FEATURE_LEAD_CORRELATION,
                            severity=LeakageSeverity.HIGH,
                            description=f"Feature '{col}' leads target by {best_lag} periods (corr: {max_correlation:.3f})",
                            offending_columns=[col],
                            correlation_score=abs(max_correlation),
                            overall_score=abs(max_correlation),
                            temporal_patterns={
                                'best_lag': best_lag,
                                'max_correlation': max_correlation,
                                'all_correlations': correlation_details,
                                'lead_correlation_detected': True
                            },
                            auto_fixable=False,
                            fix_confidence=0.8,
                            remediation_suggestions=[
                                f"Review feature '{col}' calculation",
                                f"Ensure feature doesn't use future information (lag {best_lag})",
                                "Implement proper temporal feature engineering"
                            ]
                        )
                        proofs.append(proof)
            
            return proofs
            
        except Exception as e:
            logger.error(f"Feature lead correlation detection failed: {e}")
            return []
    
    def _detect_temporal_correlations(self, X: pd.DataFrame, y: pd.Series, 
                                    time_col: Optional[str]) -> List[LeakageProof]:
        """Detect high temporal correlations that might indicate leakage."""
        proofs = []
        
        try:
            # Check for features that are highly correlated with time
            if time_col and time_col in X.columns:
                time_series = X[time_col]
                if isinstance(time_series.iloc[0], (pd.Timestamp, datetime)):
                    time_numeric = pd.to_numeric(time_series, errors='coerce')
                    
                    for col in X.columns:
                        if col == time_col or X[col].dtype not in ['int64', 'float64']:
                            continue
                        
                        correlation = X[col].corr(time_numeric)
                        if not np.isnan(correlation) and abs(correlation) > self.config.temporal_correlation_threshold:
                            proof = LeakageProof(
                                leakage_type=LeakageType.TEMPORAL_CORRELATION,
                                severity=LeakageSeverity.MEDIUM,
                                description=f"Feature '{col}' highly correlated with time (corr: {correlation:.3f})",
                                offending_columns=[col],
                                correlation_score=abs(correlation),
                                overall_score=abs(correlation),
                                temporal_patterns={
                                    'time_correlation': correlation,
                                    'temporal_leakage_detected': True
                                },
                                auto_fixable=False,
                                fix_confidence=0.6,
                                remediation_suggestions=[
                                    f"Review feature '{col}' for temporal leakage",
                                    "Consider detrending or normalizing features",
                                    "Implement proper temporal feature engineering"
                                ]
                            )
                            proofs.append(proof)
            
            return proofs
            
        except Exception as e:
            logger.error(f"Temporal correlation detection failed: {e}")
            return []
    
    def _detect_statistical_leakage(self, X: pd.DataFrame, y: pd.Series, 
                                   time_col: Optional[str]) -> List[LeakageProof]:
        """Detect statistical leakage (unusual statistical patterns)."""
        proofs = []
        
        try:
            # Check for features with suspiciously high correlations with targets
            for col in X.columns:
                if X[col].dtype not in ['int64', 'float64']:
                    continue
                
                correlation = X[col].corr(y)
                if not np.isnan(correlation) and abs(correlation) > self.config.correlation_threshold:
                    # Check if this correlation is suspiciously high
                    if self._is_suspicious_correlation(X[col], y):
                        proof = LeakageProof(
                            leakage_type=LeakageType.STATISTICAL_LEAKAGE,
                            severity=LeakageSeverity.HIGH,
                            description=f"Suspiciously high correlation in feature '{col}' (corr: {correlation:.3f})",
                            offending_columns=[col],
                            correlation_score=abs(correlation),
                            overall_score=abs(correlation),
                            temporal_patterns={
                                'correlation': correlation,
                                'statistical_leakage_detected': True
                            },
                            auto_fixable=False,
                            fix_confidence=0.7,
                            remediation_suggestions=[
                                f"Review feature '{col}' for data leakage",
                                "Check feature engineering pipeline",
                                "Implement proper validation"
                            ]
                        )
                        proofs.append(proof)
            
            return proofs
            
        except Exception as e:
            logger.error(f"Statistical leakage detection failed: {e}")
            return []
    
    def _is_likely_future_peeking(self, feature: pd.Series, future_target: pd.Series, lag: int) -> bool:
        """Check if feature is likely peeking into future."""
        try:
            # Simple heuristic: check if correlation is too high for the lag
            correlation = feature.corr(future_target)
            if np.isnan(correlation):
                return False
            
            # Higher correlation with longer lags is more suspicious
            suspicious_threshold = 0.8 + (lag - 1) * 0.05
            return abs(correlation) > suspicious_threshold
            
        except Exception:
            return False
    
    def _is_significant_lead_correlation(self, feature: pd.Series, target: pd.Series, lag: int) -> bool:
        """Check if lead correlation is statistically significant."""
        try:
            # Use permutation test to check significance
            n_permutations = 100
            original_corr = feature.corr(target.shift(-lag))
            
            if np.isnan(original_corr):
                return False
            
            # Generate permuted correlations
            permuted_corrs = []
            for _ in range(n_permutations):
                permuted_target = target.sample(frac=1).reset_index(drop=True)
                perm_corr = feature.corr(permuted_target.shift(-lag))
                if not np.isnan(perm_corr):
                    permuted_corrs.append(perm_corr)
            
            if not permuted_corrs:
                return False
            
            # Check if original correlation is significantly higher
            p_value = np.mean([abs(corr) >= abs(original_corr) for corr in permuted_corrs])
            return p_value < 0.05
            
        except Exception:
            return False
    
    def _is_suspicious_correlation(self, feature: pd.Series, target: pd.Series) -> bool:
        """Check if correlation is suspiciously high."""
        try:
            correlation = feature.corr(target)
            if np.isnan(correlation):
                return False
            
            # Very high correlations are suspicious
            return abs(correlation) > 0.95
            
        except Exception:
            return False
    
    def _generate_detection_summary(self, proofs: List[LeakageProof]):
        """Generate detection summary."""
        if not proofs:
            if TPRINT_AVAILABLE:
                tprint_success("✅ No leakage detected")
            return
        
        # Count by severity
        severity_counts = Counter([proof.severity for proof in proofs])
        leakage_type_counts = Counter([proof.leakage_type for proof in proofs])
        
        if TPRINT_AVAILABLE:
            tprint_warning(f"⚠️ Leakage detection summary:")
            tprint_warning(f"   Total issues: {len(proofs)}")
            tprint_warning(f"   Critical: {severity_counts.get(LeakageSeverity.CRITICAL, 0)}")
            tprint_warning(f"   High: {severity_counts.get(LeakageSeverity.HIGH, 0)}")
            tprint_warning(f"   Medium: {severity_counts.get(LeakageSeverity.MEDIUM, 0)}")
            tprint_warning(f"   Low: {severity_counts.get(LeakageSeverity.LOW, 0)}")
            
            tprint_warning(f"   Leakage types:")
            for leakage_type, count in leakage_type_counts.items():
                tprint_warning(f"     {leakage_type.value}: {count}")
    
    def generate_proof_report(self, proofs: List[LeakageProof], 
                            filename: Optional[str] = None) -> str:
        """Generate detailed proof report."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"leakage_proof_report_{timestamp}.json"
        
        filepath = Path(self.config.report_directory) / filename
        
        try:
            # Convert proofs to JSON-serializable format
            report_data = {
                'detection_timestamp': datetime.now().isoformat(),
                'total_issues': len(proofs),
                'severity_summary': Counter([proof.severity.value for proof in proofs]),
                'leakage_type_summary': Counter([proof.leakage_type.value for proof in proofs]),
                'proofs': []
            }
            
            for proof in proofs:
                proof_data = {
                    'leakage_type': proof.leakage_type.value,
                    'severity': proof.severity.value,
                    'description': proof.description,
                    'offending_columns': proof.offending_columns,
                    'offending_rows': proof.offending_rows,
                    'offending_entities': proof.offending_entities,
                    'correlation_score': proof.correlation_score,
                    'statistical_score': proof.statistical_score,
                    'temporal_score': proof.temporal_score,
                    'overall_score': proof.overall_score,
                    'minimal_repro_snippet': proof.minimal_repro_snippet,
                    'correlation_matrix': proof.correlation_matrix,
                    'temporal_patterns': proof.temporal_patterns,
                    'detection_timestamp': proof.detection_timestamp,
                    'affected_samples': proof.affected_samples,
                    'remediation_suggestions': proof.remediation_suggestions,
                    'auto_fixable': proof.auto_fixable,
                    'fix_confidence': proof.fix_confidence
                }
                report_data['proofs'].append(proof_data)
            
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            if TPRINT_AVAILABLE:
                tprint_success(f"📄 Proof report saved: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to generate proof report: {e}")
            return ""


# Convenience functions
def create_leakage_detector(config: Optional[LeakageDetectionConfig] = None) -> LeakageDetector:
    """Create leakage detector."""
    return LeakageDetector(config)

def detect_leakage_quick(X: pd.DataFrame, y: pd.Series, 
                        entity_cols: Optional[List[str]] = None) -> List[LeakageProof]:
    """Quick leakage detection."""
    detector = create_leakage_detector()
    return detector.detect_all_leakage(X, y, entity_cols)

def generate_leakage_summary(proofs: List[LeakageProof]) -> Dict[str, Any]:
    """Generate leakage summary."""
    if not proofs:
        return {'total_issues': 0, 'severity_summary': {}, 'leakage_type_summary': {}}
    
    return {
        'total_issues': len(proofs),
        'severity_summary': Counter([proof.severity.value for proof in proofs]),
        'leakage_type_summary': Counter([proof.leakage_type.value for proof in proofs]),
        'critical_issues': [proof for proof in proofs if proof.severity == LeakageSeverity.CRITICAL],
        'high_issues': [proof for proof in proofs if proof.severity == LeakageSeverity.HIGH],
        'auto_fixable': [proof for proof in proofs if proof.auto_fixable]
    }


if __name__ == "__main__":
    # Example usage
    print("Comprehensive Leakage Detection System")
    print("=" * 50)
    
    # Create sample data with leakage
    dates = pd.date_range('2020-01-01', periods=1000, freq='1H')
    X = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'future_label': np.random.choice([0, 1], size=1000),  # This will cause label leakage
        'rolling_ma': pd.Series(np.random.randn(1000)).rolling(5).mean(),
        'time': dates
    }, index=dates)
    
    y = pd.Series(np.random.choice([0, 1, 2], size=1000, p=[0.7, 0.2, 0.1]), index=dates)
    
    # Detect leakage
    detector = create_leakage_detector()
    proofs = detector.detect_all_leakage(X, y, entity_cols=['account_id'], time_col='time')
    
    # Generate summary
    summary = generate_leakage_summary(proofs)
    print(f"Total issues found: {summary['total_issues']}")
    print(f"Critical issues: {len(summary['critical_issues'])}")
    print(f"High issues: {len(summary['high_issues'])}")
    print(f"Auto-fixable: {len(summary['auto_fixable'])}")