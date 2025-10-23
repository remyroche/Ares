"""
Time Alignment Contracts and Feature Availability Lag Registry

This module implements strict time alignment contracts to ensure temporal integrity
in financial datasets. It enforces the constraint:
timestamp_event ≤ timestamp_label ≤ timestamp_available

Key Features:
- Feature availability lag registry
- As-of join validation
- Temporal constraint enforcement
- Feature temporal validity checking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import warnings
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path

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


class TemporalConstraint(Enum):
    """Temporal constraint types."""
    EVENT_BEFORE_LABEL = "event_before_label"          # timestamp_event ≤ timestamp_label
    LABEL_BEFORE_AVAILABLE = "label_before_available"   # timestamp_label ≤ timestamp_available
    FEATURE_AVAILABILITY = "feature_availability"       # Feature available after lag
    AS_OF_JOIN = "as_of_join"                          # As-of join validity


@dataclass
class FeatureAvailabilityConfig:
    """Configuration for feature availability lag registry."""
    
    # Feature lag specifications
    feature_lags: Dict[str, str] = field(default_factory=dict)  # feature_name: lag_duration
    
    # Default lags by feature type
    default_lags: Dict[str, str] = field(default_factory=lambda: {
        'price': '00:01:00',           # 1 minute
        'volume': '00:01:00',          # 1 minute
        'technical': '00:05:00',       # 5 minutes
        'fundamental': '01:00:00',     # 1 hour
        'sentiment': '00:15:00',       # 15 minutes
        'news': '00:30:00',            # 30 minutes
        'alternative': '02:00:00'      # 2 hours
    })
    
    # Validation settings
    strict_mode: bool = True
    allow_override: bool = False
    default_lag: str = '00:05:00'      # Default lag for unregistered features
    
    # Error handling
    fail_on_violation: bool = True
    log_violations: bool = True
    max_violations: int = 100


@dataclass
class TimeAlignmentViolation:
    """Time alignment violation record."""
    
    violation_type: TemporalConstraint
    feature_name: str
    violation_timestamp: datetime
    expected_timestamp: datetime
    actual_timestamp: datetime
    violation_duration: timedelta
    severity: str = "medium"
    description: str = ""
    auto_fixable: bool = False
    fix_suggestion: str = ""


@dataclass
class AsOfJoinResult:
    """Result of as-of join validation."""
    
    is_valid: bool
    violations: List[TimeAlignmentViolation] = field(default_factory=list)
    feature_coverage: Dict[str, float] = field(default_factory=dict)
    temporal_gaps: Dict[str, timedelta] = field(default_factory=dict)
    validation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class FeatureAvailabilityRegistry:
    """Registry for feature availability lags."""
    
    def __init__(self, config: Optional[FeatureAvailabilityConfig] = None):
        """Initialize feature availability registry."""
        self.config = config or FeatureAvailabilityConfig()
        self.registry = self.config.feature_lags.copy()
        self.violation_history = []
    
    def register_feature(self, feature_name: str, lag_duration: str, 
                        feature_type: Optional[str] = None) -> bool:
        """
        Register a feature with its availability lag.
        
        Args:
            feature_name: Name of the feature
            lag_duration: Lag duration (e.g., '00:05:00' for 5 minutes)
            feature_type: Type of feature for default lag lookup
            
        Returns:
            True if registration successful
        """
        try:
            # Validate lag duration format
            if not self._validate_lag_duration(lag_duration):
                if TPRINT_AVAILABLE:
                    tprint_error(f"❌ Invalid lag duration format: {lag_duration}")
                return False
            
            # Check if feature already registered
            if feature_name in self.registry and not self.config.allow_override:
                if TPRINT_AVAILABLE:
                    tprint_warning(f"⚠️ Feature '{feature_name}' already registered")
                return False
            
            # Register feature
            self.registry[feature_name] = lag_duration
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Registered feature '{feature_name}' with lag {lag_duration}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register feature '{feature_name}': {e}")
            return False
    
    def get_feature_lag(self, feature_name: str, feature_type: Optional[str] = None) -> str:
        """
        Get availability lag for a feature.
        
        Args:
            feature_name: Name of the feature
            feature_type: Type of feature for default lookup
            
        Returns:
            Lag duration string
        """
        # Check if feature is directly registered
        if feature_name in self.registry:
            return self.registry[feature_name]
        
        # Check if feature type has default lag
        if feature_type and feature_type in self.config.default_lags:
            return self.config.default_lags[feature_type]
        
        # Use default lag
        return self.config.default_lag
    
    def register_features_from_config(self, feature_config: Dict[str, Any]) -> int:
        """
        Register multiple features from configuration.
        
        Args:
            feature_config: Dictionary with feature configurations
            
        Returns:
            Number of successfully registered features
        """
        registered_count = 0
        
        for feature_name, config in feature_config.items():
            if isinstance(config, str):
                # Simple string lag
                if self.register_feature(feature_name, config):
                    registered_count += 1
            elif isinstance(config, dict):
                # Complex configuration
                lag_duration = config.get('lag', self.config.default_lag)
                feature_type = config.get('type')
                if self.register_feature(feature_name, lag_duration, feature_type):
                    registered_count += 1
        
        if TPRINT_AVAILABLE:
            tprint_info(f"📝 Registered {registered_count} features from config")
        
        return registered_count
    
    def _validate_lag_duration(self, lag_duration: str) -> bool:
        """Validate lag duration format."""
        try:
            # Try to parse as timedelta
            pd.Timedelta(lag_duration)
            return True
        except Exception:
            return False
    
    def get_all_registered_features(self) -> Dict[str, str]:
        """Get all registered features with their lags."""
        return self.registry.copy()
    
    def export_config(self) -> Dict[str, Any]:
        """Export current configuration."""
        return {
            'feature_lags': self.registry,
            'default_lags': self.config.default_lags,
            'strict_mode': self.config.strict_mode,
            'allow_override': self.config.allow_override,
            'default_lag': self.config.default_lag
        }


class TimeAlignmentValidator:
    """Validator for time alignment contracts."""
    
    def __init__(self, 
                 feature_registry: Optional[FeatureAvailabilityRegistry] = None,
                 config: Optional[FeatureAvailabilityConfig] = None):
        """Initialize time alignment validator."""
        self.feature_registry = feature_registry or FeatureAvailabilityRegistry(config)
        self.config = config or FeatureAvailabilityConfig()
        self.validation_history = []
    
    def validate_time_alignment(self, 
                              X: pd.DataFrame,
                              y: Optional[pd.Series] = None,
                              event_timestamps: Optional[pd.Series] = None,
                              label_timestamps: Optional[pd.Series] = None,
                              available_timestamps: Optional[pd.Series] = None,
                              time_col: str = 'timestamp') -> AsOfJoinResult:
        """
        Validate time alignment contracts.
        
        Args:
            X: Feature matrix with timestamps
            y: Target labels (optional)
            event_timestamps: Event timestamps (optional)
            label_timestamps: Label timestamps (optional)
            available_timestamps: Available timestamps (optional)
            time_col: Name of time column
            
        Returns:
            AsOfJoinResult with validation results
        """
        result = AsOfJoinResult(is_valid=True)
        
        try:
            if TPRINT_AVAILABLE:
                tprint_info("🔍 Validating time alignment contracts")
            
            # Get timestamps
            timestamps = self._extract_timestamps(X, y, event_timestamps, 
                                                label_timestamps, available_timestamps, time_col)
            
            # 1. Validate event ≤ label ≤ available constraint
            if timestamps['event'] is not None and timestamps['label'] is not None:
                event_label_violations = self._validate_event_label_constraint(
                    timestamps['event'], timestamps['label']
                )
                result.violations.extend(event_label_violations)
            
            if timestamps['label'] is not None and timestamps['available'] is not None:
                label_available_violations = self._validate_label_available_constraint(
                    timestamps['label'], timestamps['available']
                )
                result.violations.extend(label_available_violations)
            
            # 2. Validate feature availability
            feature_violations = self._validate_feature_availability(X, timestamps['available'])
            result.violations.extend(feature_violations)
            
            # 3. Validate as-of join integrity
            as_of_violations = self._validate_as_of_join(X, timestamps)
            result.violations.extend(as_of_violations)
            
            # 4. Calculate feature coverage and temporal gaps
            result.feature_coverage = self._calculate_feature_coverage(X, timestamps['available'])
            result.temporal_gaps = self._calculate_temporal_gaps(X, timestamps)
            
            # 5. Determine overall validity
            result.is_valid = len(result.violations) == 0
            
            # Store in history
            self.validation_history.append(result)
            
            if TPRINT_AVAILABLE:
                if result.is_valid:
                    tprint_success("✅ Time alignment validation passed")
                else:
                    tprint_error(f"❌ Time alignment validation failed: {len(result.violations)} violations")
            
            return result
            
        except Exception as e:
            logger.error(f"Time alignment validation failed: {e}")
            result.is_valid = False
            result.violations.append(TimeAlignmentViolation(
                violation_type=TemporalConstraint.AS_OF_JOIN,
                feature_name="validation_error",
                violation_timestamp=datetime.now(),
                expected_timestamp=datetime.now(),
                actual_timestamp=datetime.now(),
                violation_duration=timedelta(0),
                severity="critical",
                description=f"Validation error: {str(e)}",
                auto_fixable=False,
                fix_suggestion="Review data format and timestamps"
            ))
            return result
    
    def _extract_timestamps(self, X: pd.DataFrame, y: Optional[pd.Series],
                           event_timestamps: Optional[pd.Series],
                           label_timestamps: Optional[pd.Series],
                           available_timestamps: Optional[pd.Series],
                           time_col: str) -> Dict[str, Optional[pd.Series]]:
        """Extract timestamps from various sources."""
        timestamps = {
            'event': None,
            'label': None,
            'available': None
        }
        
        # Extract event timestamps
        if event_timestamps is not None:
            timestamps['event'] = event_timestamps
        elif time_col in X.columns:
            timestamps['event'] = X[time_col]
        
        # Extract label timestamps
        if label_timestamps is not None:
            timestamps['label'] = label_timestamps
        elif y is not None and hasattr(y, 'index') and isinstance(y.index, pd.DatetimeIndex):
            timestamps['label'] = pd.Series(y.index, index=y.index)
        
        # Extract available timestamps
        if available_timestamps is not None:
            timestamps['available'] = available_timestamps
        else:
            # Use current timestamp as available timestamp
            timestamps['available'] = pd.Series([datetime.now()] * len(X), index=X.index)
        
        return timestamps
    
    def _validate_event_label_constraint(self, event_timestamps: pd.Series, 
                                       label_timestamps: pd.Series) -> List[TimeAlignmentViolation]:
        """Validate event ≤ label constraint."""
        violations = []
        
        try:
            # Align timestamps
            common_index = event_timestamps.index.intersection(label_timestamps.index)
            if len(common_index) == 0:
                return violations
            
            event_aligned = event_timestamps.loc[common_index]
            label_aligned = label_timestamps.loc[common_index]
            
            # Find violations
            violation_mask = event_aligned > label_aligned
            violation_indices = common_index[violation_mask]
            
            for idx in violation_indices:
                violation = TimeAlignmentViolation(
                    violation_type=TemporalConstraint.EVENT_BEFORE_LABEL,
                    feature_name="event_timestamp",
                    violation_timestamp=event_aligned.loc[idx],
                    expected_timestamp=label_aligned.loc[idx],
                    actual_timestamp=event_aligned.loc[idx],
                    violation_duration=event_aligned.loc[idx] - label_aligned.loc[idx],
                    severity="high",
                    description="Event timestamp is after label timestamp",
                    auto_fixable=False,
                    fix_suggestion="Ensure event timestamps are before label timestamps"
                )
                violations.append(violation)
            
            return violations
            
        except Exception as e:
            logger.error(f"Event-label constraint validation failed: {e}")
            return violations
    
    def _validate_label_available_constraint(self, label_timestamps: pd.Series,
                                           available_timestamps: pd.Series) -> List[TimeAlignmentViolation]:
        """Validate label ≤ available constraint."""
        violations = []
        
        try:
            # Align timestamps
            common_index = label_timestamps.index.intersection(available_timestamps.index)
            if len(common_index) == 0:
                return violations
            
            label_aligned = label_timestamps.loc[common_index]
            available_aligned = available_timestamps.loc[common_index]
            
            # Find violations
            violation_mask = label_aligned > available_aligned
            violation_indices = common_index[violation_mask]
            
            for idx in violation_indices:
                violation = TimeAlignmentViolation(
                    violation_type=TemporalConstraint.LABEL_BEFORE_AVAILABLE,
                    feature_name="label_timestamp",
                    violation_timestamp=label_aligned.loc[idx],
                    expected_timestamp=available_aligned.loc[idx],
                    actual_timestamp=label_aligned.loc[idx],
                    violation_duration=label_aligned.loc[idx] - available_aligned.loc[idx],
                    severity="high",
                    description="Label timestamp is after available timestamp",
                    auto_fixable=False,
                    fix_suggestion="Ensure label timestamps are before available timestamps"
                )
                violations.append(violation)
            
            return violations
            
        except Exception as e:
            logger.error(f"Label-available constraint validation failed: {e}")
            return violations
    
    def _validate_feature_availability(self, X: pd.DataFrame, 
                                     available_timestamps: Optional[pd.Series]) -> List[TimeAlignmentViolation]:
        """Validate feature availability based on lag registry."""
        violations = []
        
        try:
            if available_timestamps is None:
                return violations
            
            for feature_name in X.columns:
                if feature_name in ['timestamp', 'time', 'date']:
                    continue
                
                # Get feature lag
                feature_lag = self.feature_registry.get_feature_lag(feature_name)
                lag_duration = pd.Timedelta(feature_lag)
                
                # Check if feature is available at the required time
                feature_timestamps = available_timestamps - lag_duration
                
                # Find violations where feature is not available
                violation_mask = feature_timestamps < available_timestamps.min()
                violation_indices = feature_timestamps.index[violation_mask]
                
                for idx in violation_indices:
                    violation = TimeAlignmentViolation(
                        violation_type=TemporalConstraint.FEATURE_AVAILABILITY,
                        feature_name=feature_name,
                        violation_timestamp=available_timestamps.loc[idx],
                        expected_timestamp=feature_timestamps.loc[idx],
                        actual_timestamp=available_timestamps.loc[idx],
                        violation_duration=lag_duration,
                        severity="medium",
                        description=f"Feature '{feature_name}' not available with required lag {feature_lag}",
                        auto_fixable=True,
                        fix_suggestion=f"Ensure feature '{feature_name}' is available with lag {feature_lag}"
                    )
                    violations.append(violation)
            
            return violations
            
        except Exception as e:
            logger.error(f"Feature availability validation failed: {e}")
            return violations
    
    def _validate_as_of_join(self, X: pd.DataFrame, timestamps: Dict[str, Optional[pd.Series]]) -> List[TimeAlignmentViolation]:
        """Validate as-of join integrity."""
        violations = []
        
        try:
            # Check if all timestamps are properly aligned
            if timestamps['event'] is not None and timestamps['available'] is not None:
                # Check for temporal consistency
                event_times = timestamps['event']
                available_times = timestamps['available']
                
                # Find cases where event time is after available time
                violation_mask = event_times > available_times
                violation_indices = event_times.index[violation_mask]
                
                for idx in violation_indices:
                    violation = TimeAlignmentViolation(
                        violation_type=TemporalConstraint.AS_OF_JOIN,
                        feature_name="as_of_join",
                        violation_timestamp=event_times.loc[idx],
                        expected_timestamp=available_times.loc[idx],
                        actual_timestamp=event_times.loc[idx],
                        violation_duration=event_times.loc[idx] - available_times.loc[idx],
                        severity="high",
                        description="As-of join violation: event time after available time",
                        auto_fixable=False,
                        fix_suggestion="Ensure event times are before available times"
                    )
                    violations.append(violation)
            
            return violations
            
        except Exception as e:
            logger.error(f"As-of join validation failed: {e}")
            return violations
    
    def _calculate_feature_coverage(self, X: pd.DataFrame, 
                                   available_timestamps: Optional[pd.Series]) -> Dict[str, float]:
        """Calculate feature coverage over time."""
        coverage = {}
        
        try:
            if available_timestamps is None:
                return coverage
            
            for feature_name in X.columns:
                if feature_name in ['timestamp', 'time', 'date']:
                    continue
                
                # Calculate coverage as percentage of non-null values
                non_null_count = X[feature_name].notna().sum()
                total_count = len(X[feature_name])
                coverage[feature_name] = non_null_count / total_count if total_count > 0 else 0.0
            
            return coverage
            
        except Exception as e:
            logger.error(f"Feature coverage calculation failed: {e}")
            return coverage
    
    def _calculate_temporal_gaps(self, X: pd.DataFrame, 
                               timestamps: Dict[str, Optional[pd.Series]]) -> Dict[str, timedelta]:
        """Calculate temporal gaps between different timestamp types."""
        gaps = {}
        
        try:
            if timestamps['event'] is not None and timestamps['available'] is not None:
                # Calculate gap between event and available
                event_times = timestamps['event']
                available_times = timestamps['available']
                
                common_index = event_times.index.intersection(available_times.index)
                if len(common_index) > 0:
                    event_aligned = event_times.loc[common_index]
                    available_aligned = available_times.loc[common_index]
                    
                    gaps['event_to_available'] = (available_aligned - event_aligned).mean()
            
            if timestamps['label'] is not None and timestamps['available'] is not None:
                # Calculate gap between label and available
                label_times = timestamps['label']
                available_times = timestamps['available']
                
                common_index = label_times.index.intersection(available_times.index)
                if len(common_index) > 0:
                    label_aligned = label_times.loc[common_index]
                    available_aligned = available_times.loc[common_index]
                    
                    gaps['label_to_available'] = (available_aligned - label_aligned).mean()
            
            return gaps
            
        except Exception as e:
            logger.error(f"Temporal gap calculation failed: {e}")
            return gaps
    
    def generate_validation_report(self, result: AsOfJoinResult, 
                                 filename: Optional[str] = None) -> str:
        """Generate detailed validation report."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"time_alignment_report_{timestamp}.json"
        
        filepath = Path(self.config.report_directory) / filename
        
        try:
            report_data = {
                'validation_timestamp': result.validation_timestamp,
                'is_valid': result.is_valid,
                'total_violations': len(result.violations),
                'violations_by_type': Counter([v.violation_type.value for v in result.violations]),
                'violations_by_severity': Counter([v.severity for v in result.violations]),
                'feature_coverage': result.feature_coverage,
                'temporal_gaps': {k: str(v) for k, v in result.temporal_gaps.items()},
                'violations': []
            }
            
            for violation in result.violations:
                violation_data = {
                    'violation_type': violation.violation_type.value,
                    'feature_name': violation.feature_name,
                    'violation_timestamp': violation.violation_timestamp.isoformat(),
                    'expected_timestamp': violation.expected_timestamp.isoformat(),
                    'actual_timestamp': violation.actual_timestamp.isoformat(),
                    'violation_duration': str(violation.violation_duration),
                    'severity': violation.severity,
                    'description': violation.description,
                    'auto_fixable': violation.auto_fixable,
                    'fix_suggestion': violation.fix_suggestion
                }
                report_data['violations'].append(violation_data)
            
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            if TPRINT_AVAILABLE:
                tprint_success(f"📄 Time alignment report saved: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to generate validation report: {e}")
            return ""


# Convenience functions
def create_feature_registry(config: Optional[FeatureAvailabilityConfig] = None) -> FeatureAvailabilityRegistry:
    """Create feature availability registry."""
    return FeatureAvailabilityRegistry(config)

def create_time_alignment_validator(registry: Optional[FeatureAvailabilityRegistry] = None) -> TimeAlignmentValidator:
    """Create time alignment validator."""
    return TimeAlignmentValidator(registry)

def validate_time_alignment_quick(X: pd.DataFrame, 
                                 y: Optional[pd.Series] = None,
                                 time_col: str = 'timestamp') -> bool:
    """Quick time alignment validation."""
    validator = create_time_alignment_validator()
    result = validator.validate_time_alignment(X, y, time_col=time_col)
    return result.is_valid


if __name__ == "__main__":
    # Example usage
    print("Time Alignment Contracts and Feature Availability Registry")
    print("=" * 60)
    
    # Create feature registry
    registry = create_feature_registry()
    
    # Register some features
    registry.register_feature('price_5min_ma', '00:05:00', 'technical')
    registry.register_feature('volume_1min', '00:01:00', 'volume')
    registry.register_feature('sentiment_score', '00:15:00', 'sentiment')
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=1000, freq='1H')
    X = pd.DataFrame({
        'price_5min_ma': np.random.randn(1000),
        'volume_1min': np.random.randn(1000),
        'sentiment_score': np.random.randn(1000),
        'timestamp': dates
    }, index=dates)
    
    y = pd.Series(np.random.choice([0, 1], size=1000), index=dates)
    
    # Validate time alignment
    validator = create_time_alignment_validator(registry)
    result = validator.validate_time_alignment(X, y, time_col='timestamp')
    
    print(f"Time alignment valid: {result.is_valid}")
    print(f"Total violations: {len(result.violations)}")
    print(f"Feature coverage: {result.feature_coverage}")
    print(f"Temporal gaps: {result.temporal_gaps}")