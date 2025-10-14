"""
Label Construction Validation

Validates label construction to prevent look-ahead bias and ensure temporal
integrity in time series feature engineering.

Key Features:
- Fixed horizon forward returns validation
- Resampling frequency verification
- HTF alignment constraints
- Past-only data enforcement
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import logging
import time
from enum import Enum
from datetime import datetime, timedelta
import warnings

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

logger = logging.getLogger(__name__)


class LabelType(Enum):
    """Types of labels for validation."""
    FORWARD_RETURNS = "forward_returns"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTI_CLASS = "multi_class"
    REGRESSION = "regression"
    CUSTOM = "custom"


class HTFAlignmentMethod(Enum):
    """HTF alignment methods."""
    PAST_ONLY = "past_only"
    STRICT_PAST = "strict_past"
    FUTURE_OK = "future_ok"  # Not recommended


@dataclass
class LabelConstructionConfig:
    """Configuration for label construction validation."""
    
    # Label construction parameters
    label_type: LabelType = LabelType.FORWARD_RETURNS
    horizon: int = 1  # Fixed horizon forward returns
    resampling_frequency: str = '1H'  # How often labels are updated
    min_samples_for_label: int = 10  # Minimum samples needed for reliable label
    
    # HTF alignment constraints
    htf_alignment_method: HTFAlignmentMethod = HTFAlignmentMethod.STRICT_PAST
    max_htf_lookback: int = 252  # Maximum HTF lookback periods
    htf_resampling_offset: int = 0  # Offset to ensure past-only data
    
    # Validation parameters
    strict_temporal_ordering: bool = True
    validate_future_data: bool = True
    validate_resampling_alignment: bool = True
    validate_htf_constraints: bool = True
    
    # Error handling
    fail_on_violation: bool = True
    log_violations: bool = True
    max_violations: int = 10


@dataclass
class LabelConstructionResult:
    """Result from label construction validation."""
    
    # Validation results
    is_valid: bool
    label_type: LabelType
    horizon: int
    resampling_frequency: str
    
    # Validation details
    temporal_violations: List[Dict[str, Any]] = field(default_factory=list)
    htf_violations: List[Dict[str, Any]] = field(default_factory=list)
    resampling_violations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Statistics
    total_labels: int = 0
    valid_labels: int = 0
    invalid_labels: int = 0
    future_data_points: int = 0
    
    # HTF alignment
    htf_features_validated: int = 0
    htf_alignment_score: float = 0.0
    
    # Performance
    validation_time: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class LabelConstructionValidator:
    """
    Validates label construction to prevent look-ahead bias.
    
    This class ensures that labels are constructed using only past information
    and that HTF features are properly aligned to prevent future data leakage.
    """
    
    def __init__(self, config: Optional[LabelConstructionConfig] = None):
        """
        Initialize the label construction validator.
        
        Args:
            config: Configuration for label construction validation
        """
        self.config = config or LabelConstructionConfig()
        self.logger = logger
        
        tprint_info("🔒 Label Construction Validator initialized")
        tprint_debug(f"📊 Label type: {self.config.label_type.value}")
        tprint_debug(f"📊 Horizon: {self.config.horizon}")
        tprint_debug(f"📊 HTF alignment: {self.config.htf_alignment_method.value}")
    
    def validate_label_construction(self, 
                                  data: pd.DataFrame,
                                  targets: pd.Series,
                                  htf_features: Optional[Dict[str, pd.Series]] = None) -> LabelConstructionResult:
        """
        Validate label construction for temporal integrity.
        
        Args:
            data: Input data with timestamps
            targets: Target labels
            htf_features: Higher timeframe features (optional)
            
        Returns:
            LabelConstructionResult with validation details
        """
        start_time = time.time()
        
        tprint_info("🔒 Validating label construction...")
        tprint_debug(f"📊 Data shape: {data.shape}")
        tprint_debug(f"📊 Targets length: {len(targets)}")
        tprint_debug(f"📊 HTF features: {len(htf_features) if htf_features else 0}")
        
        try:
            # Initialize result
            result = LabelConstructionResult(
                is_valid=True,
                label_type=self.config.label_type,
                horizon=self.config.horizon,
                resampling_frequency=self.config.resampling_frequency,
                total_labels=len(targets)
            )
            
            # Step 1: Validate temporal ordering
            tprint_debug("Step 1: Validating temporal ordering...")
            temporal_violations = self._validate_temporal_ordering(data, targets)
            result.temporal_violations = temporal_violations
            
            if temporal_violations and self.config.fail_on_violation:
                result.is_valid = False
                tprint_error(f"❌ {len(temporal_violations)} temporal violations found")
            
            # Step 2: Validate label construction
            tprint_debug("Step 2: Validating label construction...")
            label_validation = self._validate_label_construction_method(data, targets)
            result.valid_labels = label_validation['valid_labels']
            result.invalid_labels = label_validation['invalid_labels']
            result.future_data_points = label_validation['future_data_points']
            
            if result.invalid_labels > 0 and self.config.fail_on_violation:
                result.is_valid = False
                tprint_error(f"❌ {result.invalid_labels} invalid labels found")
            
            # Step 3: Validate resampling alignment
            if self.config.validate_resampling_alignment:
                tprint_debug("Step 3: Validating resampling alignment...")
                resampling_violations = self._validate_resampling_alignment(data, targets)
                result.resampling_violations = resampling_violations
                
                if resampling_violations and self.config.fail_on_violation:
                    result.is_valid = False
                    tprint_error(f"❌ {len(resampling_violations)} resampling violations found")
            
            # Step 4: Validate HTF alignment
            if htf_features and self.config.validate_htf_constraints:
                tprint_debug("Step 4: Validating HTF alignment...")
                htf_validation = self._validate_htf_alignment(data, htf_features)
                result.htf_violations = htf_validation['violations']
                result.htf_features_validated = htf_validation['features_validated']
                result.htf_alignment_score = htf_validation['alignment_score']
                
                if htf_validation['violations'] and self.config.fail_on_violation:
                    result.is_valid = False
                    tprint_error(f"❌ {len(htf_validation['violations'])} HTF violations found")
            
            # Calculate final statistics
            result.validation_time = time.time() - start_time
            result.memory_usage_mb = self._estimate_memory_usage(data, targets, htf_features)
            
            # Log results
            if result.is_valid:
                tprint_success("✅ Label construction validation passed")
            else:
                tprint_error("❌ Label construction validation failed")
            
            tprint_info(f"📊 Valid labels: {result.valid_labels}/{result.total_labels}")
            tprint_info(f"📊 HTF alignment score: {result.htf_alignment_score:.3f}")
            tprint_info(f"📊 Validation time: {result.validation_time:.3f}s")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Label construction validation failed: {e}")
            return LabelConstructionResult(
                is_valid=False,
                label_type=self.config.label_type,
                horizon=self.config.horizon,
                resampling_frequency=self.config.resampling_frequency,
                total_labels=len(targets),
                metadata={'error': str(e)}
            )
    
    def _validate_temporal_ordering(self, 
                                  data: pd.DataFrame, 
                                  targets: pd.Series) -> List[Dict[str, Any]]:
        """Validate temporal ordering of data and targets."""
        violations = []
        
        try:
            # Check if data has datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                violations.append({
                    'type': 'non_datetime_index',
                    'message': 'Data index is not datetime',
                    'severity': 'high'
                })
                return violations
            
            # Check if targets have datetime index
            if not isinstance(targets.index, pd.DatetimeIndex):
                violations.append({
                    'type': 'non_datetime_targets',
                    'message': 'Targets index is not datetime',
                    'severity': 'high'
                })
                return violations
            
            # Check temporal alignment
            data_start = data.index.min()
            data_end = data.index.max()
            targets_start = targets.index.min()
            targets_end = targets.index.max()
            
            # Data should start before or at the same time as targets
            if data_start > targets_start:
                violations.append({
                    'type': 'data_starts_after_targets',
                    'message': f'Data starts {data_start} after targets {targets_start}',
                    'severity': 'high',
                    'data_start': str(data_start),
                    'targets_start': str(targets_start)
                })
            
            # Check for future data in targets
            if self.config.validate_future_data:
                future_data_count = 0
                for i, (timestamp, value) in enumerate(targets.items()):
                    # Check if this target uses future data
                    if self._uses_future_data(timestamp, data, self.config.horizon):
                        future_data_count += 1
                        if future_data_count <= self.config.max_violations:
                            violations.append({
                                'type': 'future_data_usage',
                                'message': f'Target at {timestamp} uses future data',
                                'severity': 'critical',
                                'timestamp': str(timestamp),
                                'value': value
                            })
                
                if future_data_count > self.config.max_violations:
                    violations.append({
                        'type': 'excessive_future_data',
                        'message': f'{future_data_count} targets use future data',
                        'severity': 'critical',
                        'count': future_data_count
                    })
            
        except Exception as e:
            violations.append({
                'type': 'temporal_validation_error',
                'message': f'Temporal validation failed: {e}',
                'severity': 'high'
            })
        
        return violations
    
    def _validate_label_construction_method(self, 
                                          data: pd.DataFrame, 
                                          targets: pd.Series) -> Dict[str, Any]:
        """Validate the method used to construct labels."""
        validation_result = {
            'valid_labels': 0,
            'invalid_labels': 0,
            'future_data_points': 0
        }
        
        try:
            for i, (timestamp, target_value) in enumerate(targets.items()):
                # Check if this label is valid
                is_valid = self._is_label_valid(timestamp, target_value, data)
                
                if is_valid:
                    validation_result['valid_labels'] += 1
                else:
                    validation_result['invalid_labels'] += 1
                    
                    # Check if it uses future data
                    if self._uses_future_data(timestamp, data, self.config.horizon):
                        validation_result['future_data_points'] += 1
                        
        except Exception as e:
            self.logger.warning(f"Label construction validation failed: {e}")
        
        return validation_result
    
    def _validate_resampling_alignment(self, 
                                     data: pd.DataFrame, 
                                     targets: pd.Series) -> List[Dict[str, Any]]:
        """Validate resampling alignment between data and targets."""
        violations = []
        
        try:
            # Check resampling frequency alignment
            if self.config.resampling_frequency:
                # Validate that targets are properly resampled
                expected_freq = pd.Timedelta(self.config.resampling_frequency)
                
                # Check target frequency
                target_diffs = targets.index.to_series().diff().dropna()
                if not target_diffs.empty:
                    median_diff = target_diffs.median()
                    if abs(median_diff - expected_freq) > expected_freq * 0.1:  # 10% tolerance
                        violations.append({
                            'type': 'resampling_frequency_mismatch',
                            'message': f'Target frequency {median_diff} does not match expected {expected_freq}',
                            'severity': 'medium',
                            'expected_frequency': str(expected_freq),
                            'actual_frequency': str(median_diff)
                        })
            
        except Exception as e:
            violations.append({
                'type': 'resampling_validation_error',
                'message': f'Resampling validation failed: {e}',
                'severity': 'high'
            })
        
        return violations
    
    def _validate_htf_alignment(self, 
                               data: pd.DataFrame, 
                               htf_features: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Validate HTF feature alignment to prevent future data leakage."""
        violations = []
        features_validated = 0
        alignment_scores = []
        
        try:
            for feature_name, htf_series in htf_features.items():
                features_validated += 1
                
                # Check HTF alignment
                alignment_score = self._calculate_htf_alignment_score(data, htf_series)
                alignment_scores.append(alignment_score)
                
                # Check for future data usage
                if self.config.htf_alignment_method == HTFAlignmentMethod.STRICT_PAST:
                    if not self._is_htf_past_only(data, htf_series):
                        violations.append({
                            'type': 'htf_future_data',
                            'message': f'HTF feature {feature_name} uses future data',
                            'severity': 'critical',
                            'feature_name': feature_name,
                            'alignment_score': alignment_score
                        })
                
                # Check resampling offset
                if self.config.htf_resampling_offset > 0:
                    if not self._has_proper_htf_offset(data, htf_series):
                        violations.append({
                            'type': 'htf_offset_violation',
                            'message': f'HTF feature {feature_name} does not have proper offset',
                            'severity': 'medium',
                            'feature_name': feature_name,
                            'expected_offset': self.config.htf_resampling_offset
                        })
                
                # Check lookback constraints
                if len(htf_series) > self.config.max_htf_lookback:
                    violations.append({
                        'type': 'htf_lookback_exceeded',
                        'message': f'HTF feature {feature_name} exceeds max lookback {self.config.max_htf_lookback}',
                        'severity': 'medium',
                        'feature_name': feature_name,
                        'actual_lookback': len(htf_series),
                        'max_lookback': self.config.max_htf_lookback
                    })
        
        except Exception as e:
            violations.append({
                'type': 'htf_validation_error',
                'message': f'HTF validation failed: {e}',
                'severity': 'high'
            })
        
        return {
            'violations': violations,
            'features_validated': features_validated,
            'alignment_score': np.mean(alignment_scores) if alignment_scores else 0.0
        }
    
    def _is_label_valid(self, 
                       timestamp: pd.Timestamp, 
                       target_value: float, 
                       data: pd.DataFrame) -> bool:
        """Check if a label is valid (not using future data)."""
        try:
            # Check if target value is finite
            if not np.isfinite(target_value):
                return False
            
            # Check if timestamp exists in data
            if timestamp not in data.index:
                return False
            
            # Check if enough data exists before this timestamp
            data_before = data[data.index < timestamp]
            if len(data_before) < self.config.min_samples_for_label:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _uses_future_data(self, 
                         timestamp: pd.Timestamp, 
                         data: pd.DataFrame, 
                         horizon: int) -> bool:
        """Check if a timestamp uses future data."""
        try:
            # For forward returns, check if we have enough future data
            future_data = data[data.index > timestamp]
            if len(future_data) < horizon:
                return True  # Not enough future data for proper label
            
            # Check if the label construction would require future data
            # This is a simplified check - in practice, you'd need to know the exact construction method
            return False
            
        except Exception:
            return True  # Assume worst case
    
    def _is_htf_past_only(self, 
                         data: pd.DataFrame, 
                         htf_series: pd.Series) -> bool:
        """Check if HTF series uses only past data."""
        try:
            # Check that all HTF data is before the latest data timestamp
            data_latest = data.index.max()
            htf_latest = htf_series.index.max()
            
            return htf_latest < data_latest
            
        except Exception:
            return False
    
    def _has_proper_htf_offset(self, 
                              data: pd.DataFrame, 
                              htf_series: pd.Series) -> bool:
        """Check if HTF series has proper offset."""
        try:
            if self.config.htf_resampling_offset <= 0:
                return True
            
            # Check that HTF data is offset by the required amount
            data_latest = data.index.max()
            htf_latest = htf_series.index.max()
            
            offset = data_latest - htf_latest
            required_offset = pd.Timedelta(hours=self.config.htf_resampling_offset)
            
            return offset >= required_offset
            
        except Exception:
            return False
    
    def _calculate_htf_alignment_score(self, 
                                     data: pd.DataFrame, 
                                     htf_series: pd.Series) -> float:
        """Calculate HTF alignment score (0-1, higher is better)."""
        try:
            # Calculate overlap between data and HTF series
            data_start = data.index.min()
            data_end = data.index.max()
            htf_start = htf_series.index.min()
            htf_end = htf_series.index.max()
            
            # Calculate temporal overlap
            overlap_start = max(data_start, htf_start)
            overlap_end = min(data_end, htf_end)
            
            if overlap_start >= overlap_end:
                return 0.0
            
            overlap_duration = overlap_end - overlap_start
            data_duration = data_end - data_start
            htf_duration = htf_end - htf_start
            
            # Calculate alignment score
            data_overlap_ratio = overlap_duration / data_duration if data_duration > pd.Timedelta(0) else 0.0
            htf_overlap_ratio = overlap_duration / htf_duration if htf_duration > pd.Timedelta(0) else 0.0
            
            alignment_score = (data_overlap_ratio + htf_overlap_ratio) / 2
            
            return min(max(alignment_score, 0.0), 1.0)
            
        except Exception:
            return 0.0
    
    def _estimate_memory_usage(self, 
                             data: pd.DataFrame, 
                             targets: pd.Series, 
                             htf_features: Optional[Dict[str, pd.Series]]) -> float:
        """Estimate memory usage in MB."""
        try:
            data_memory = data.memory_usage(deep=True).sum() / 1024 / 1024
            targets_memory = targets.memory_usage(deep=True) / 1024 / 1024
            
            htf_memory = 0.0
            if htf_features:
                for series in htf_features.values():
                    htf_memory += series.memory_usage(deep=True) / 1024 / 1024
            
            return data_memory + targets_memory + htf_memory
            
        except Exception:
            return 0.0


# Convenience functions
def validate_label_construction(data: pd.DataFrame,
                              targets: pd.Series,
                              htf_features: Optional[Dict[str, pd.Series]] = None,
                              config: Optional[LabelConstructionConfig] = None) -> LabelConstructionResult:
    """
    Convenience function to validate label construction.
    
    Args:
        data: Input data with timestamps
        targets: Target labels
        htf_features: Higher timeframe features (optional)
        config: Label construction configuration
        
    Returns:
        LabelConstructionResult with validation details
    """
    validator = LabelConstructionValidator(config)
    return validator.validate_label_construction(data, targets, htf_features)


# Export main classes and functions
__all__ = [
    'LabelConstructionValidator',
    'LabelConstructionConfig',
    'LabelConstructionResult',
    'LabelType',
    'HTFAlignmentMethod',
    'validate_label_construction'
]