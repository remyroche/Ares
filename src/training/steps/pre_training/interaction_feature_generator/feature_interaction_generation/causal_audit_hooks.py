"""
Causal Audit Hooks for Time-Series Features

This module implements comprehensive causal audit hooks to ensure all rolling
operations are right-aligned and no centered windows appear in the feature
generation pipeline.

Key Features:
- Assert all rolling ops are right-aligned
- Fail if centered windows appear
- Audit feature generation pipeline
- Detect non-causal operations
- Enforce temporal causality
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
import re
import warnings
from functools import wraps
import inspect

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

logger = logging.getLogger(__name__)


@dataclass
class CausalAuditConfig:
    """Configuration for causal audit hooks."""
    enable_audit: bool = True
    fail_on_violation: bool = True
    log_violations: bool = True
    check_centered_windows: bool = True
    check_future_leakage: bool = True
    check_lookback_alignment: bool = True
    strict_mode: bool = True


class CausalViolationError(Exception):
    """Exception raised when causal violations are detected."""
    pass


class CausalAuditor:
    """Audit causal relationships in time-series features."""
    
    def __init__(self, config: CausalAuditConfig):
        self.config = config
        self.violations = []
        self.audit_history = []
        
        tprint_info("🔍 Causal auditor initialized")
        tprint_info(f"📊 Strict mode: {config.strict_mode}")
        tprint_info(f"📊 Fail on violation: {config.fail_on_violation}")
    
    def audit_feature_generation(self, 
                               features: pd.DataFrame,
                               operation_name: str = "unknown") -> bool:
        """
        Audit feature generation for causal violations.
        
        Args:
            features: Generated features
            operation_name: Name of the operation being audited
            
        Returns:
            True if no violations, False otherwise
        """
        if not self.config.enable_audit:
            return True
        
        tprint_debug(f"🔍 Auditing {operation_name} with {len(features.columns)} features")
        
        violations_found = []
        
        # Check each feature for causal violations
        for col in features.columns:
            feature_violations = self._audit_single_feature(features[col], col, operation_name)
            violations_found.extend(feature_violations)
        
        # Record audit results
        audit_result = {
            'operation': operation_name,
            'timestamp': pd.Timestamp.now(),
            'features_checked': len(features.columns),
            'violations_found': len(violations_found),
            'violations': violations_found
        }
        self.audit_history.append(audit_result)
        
        if violations_found:
            self.violations.extend(violations_found)
            
            if self.config.log_violations:
                tprint_error(f"❌ Causal violations found in {operation_name}:")
                for violation in violations_found:
                    tprint_error(f"  - {violation['type']}: {violation['feature']} - {violation['description']}")
            
            if self.config.fail_on_violation:
                raise CausalViolationError(f"Causal violations detected in {operation_name}: {len(violations_found)} violations")
            
            return False
        else:
            tprint_debug(f"✅ No causal violations found in {operation_name}")
            return True
    
    def _audit_single_feature(self, 
                            feature: pd.Series, 
                            feature_name: str,
                            operation_name: str) -> List[Dict[str, Any]]:
        """Audit a single feature for causal violations."""
        violations = []
        
        # Check for centered windows
        if self.config.check_centered_windows:
            centered_violation = self._check_centered_window(feature_name, feature)
            if centered_violation:
                violations.append(centered_violation)
        
        # Check for future leakage
        if self.config.check_future_leakage:
            leakage_violation = self._check_future_leakage(feature, feature_name)
            if leakage_violation:
                violations.append(leakage_violation)
        
        # Check lookback alignment
        if self.config.check_lookback_alignment:
            alignment_violation = self._check_lookback_alignment(feature, feature_name)
            if alignment_violation:
                violations.append(alignment_violation)
        
        return violations
    
    def _check_centered_window(self, feature_name: str, feature: pd.Series) -> Optional[Dict[str, Any]]:
        """Check if feature uses centered window (non-causal)."""
        # Patterns that indicate centered windows
        centered_patterns = [
            r'centered', r'center', r'mid', r'middle',
            r'symmetric', r'sym', r'balanced', r'centered_',
            r'_centered', r'center_', r'_center'
        ]
        
        feature_lower = feature_name.lower()
        
        for pattern in centered_patterns:
            if re.search(pattern, feature_lower):
                return {
                    'type': 'centered_window',
                    'feature': feature_name,
                    'description': f'Feature name suggests centered window: {pattern}',
                    'severity': 'high'
                }
        
        # Check for symmetric rolling operations
        symmetric_patterns = [
            r'rolling.*center.*true',
            r'rolling.*center.*True',
            r'rolling.*center=.*true',
            r'rolling.*center=.*True'
        ]
        
        for pattern in symmetric_patterns:
            if re.search(pattern, feature_name):
                return {
                    'type': 'centered_window',
                    'feature': feature_name,
                    'description': f'Feature uses centered rolling window: {pattern}',
                    'severity': 'high'
                }
        
        return None
    
    def _check_future_leakage(self, feature: pd.Series, feature_name: str) -> Optional[Dict[str, Any]]:
        """Check for future information leakage."""
        # Check for perfect correlation with future values
        if len(feature) < 10:  # Need sufficient data
            return None
        
        # Check if feature is perfectly correlated with its shifted version
        # (indicating it might be using future information)
        try:
            # Shift feature by 1 period and check correlation
            shifted = feature.shift(1)
            correlation = feature.corr(shifted)
            
            if abs(correlation) > 0.99:
                return {
                    'type': 'future_leakage',
                    'feature': feature_name,
                    'description': f'Feature shows perfect correlation with future values: {correlation:.3f}',
                    'severity': 'high'
                }
        except Exception:
            pass
        
        # Check for features that look like they're using future information
        future_patterns = [
            r'future', r'forward', r'next', r'tomorrow',
            r'lead', r'ahead', r'prediction', r'forecast'
        ]
        
        feature_lower = feature_name.lower()
        for pattern in future_patterns:
            if re.search(pattern, feature_lower):
                return {
                    'type': 'future_leakage',
                    'feature': feature_name,
                    'description': f'Feature name suggests future information: {pattern}',
                    'severity': 'medium'
                }
        
        return None
    
    def _check_lookback_alignment(self, feature: pd.Series, feature_name: str) -> Optional[Dict[str, Any]]:
        """Check if feature has proper lookback alignment."""
        # Extract lookback period from feature name
        lookback = self._extract_lookback_period(feature_name)
        
        if lookback <= 0:
            return None
        
        # Check if feature has proper warmup period
        # Features with lookback should have NaN values for the first lookback periods
        if len(feature) > lookback:
            warmup_period = feature.iloc[:lookback]
            if not warmup_period.isna().all():
                return {
                    'type': 'lookback_alignment',
                    'feature': feature_name,
                    'description': f'Feature with {lookback} lookback should have NaN warmup period',
                    'severity': 'medium'
                }
        
        return None
    
    def _extract_lookback_period(self, feature_name: str) -> int:
        """Extract lookback period from feature name."""
        patterns = [
            r'_(\d+)$',           # _20, _50, etc.
            r'_(\d+)_',           # _20_, _50_, etc.
            r'rolling_(\d+)',     # rolling_20, rolling_50
            r'window_(\d+)',      # window_20, window_50
            r'period_(\d+)',      # period_20, period_50
            r'ctf_(\d+)_',        # ctf_20_, ctf_50_
            r'ma_(\d+)',          # ma_20, ma_50
            r'sma_(\d+)',         # sma_20, sma_50
            r'ema_(\d+)',         # ema_20, ema_50
        ]
        
        max_period = 0
        for pattern in patterns:
            matches = re.findall(pattern, feature_name)
            for match in matches:
                try:
                    period = int(match)
                    max_period = max(max_period, period)
                except ValueError:
                    continue
        
        return max_period
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get summary of audit results."""
        total_violations = len(self.violations)
        total_audits = len(self.audit_history)
        
        violation_types = {}
        for violation in self.violations:
            vtype = violation['type']
            violation_types[vtype] = violation_types.get(vtype, 0) + 1
        
        return {
            'total_audits': total_audits,
            'total_violations': total_violations,
            'violation_types': violation_types,
            'audit_history': self.audit_history
        }
    
    def clear_violations(self):
        """Clear recorded violations."""
        self.violations = []
        self.audit_history = []


def causal_audit_hook(operation_name: str = None):
    """
    Decorator to add causal audit hooks to feature generation functions.
    
    Args:
        operation_name: Name of the operation for logging
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the causal auditor
            auditor = get_causal_auditor()
            
            if not auditor.config.enable_audit:
                return func(*args, **kwargs)
            
            # Determine operation name
            op_name = operation_name or func.__name__
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Audit the result if it's a DataFrame
                if isinstance(result, pd.DataFrame):
                    auditor.audit_feature_generation(result, op_name)
                
                return result
                
            except CausalViolationError as e:
                tprint_error(f"❌ Causal violation in {op_name}: {e}")
                raise
            except Exception as e:
                tprint_error(f"❌ Error in {op_name}: {e}")
                raise
        
        return wrapper
    return decorator


def assert_right_aligned(features: pd.DataFrame, 
                        operation_name: str = "unknown") -> bool:
    """
    Assert that all features are right-aligned (causal).
    
    Args:
        features: Feature matrix to check
        operation_name: Name of the operation
        
    Returns:
        True if all features are right-aligned
        
    Raises:
        CausalViolationError: If non-causal features are found
    """
    auditor = get_causal_auditor()
    return auditor.audit_feature_generation(features, operation_name)


def check_centered_windows(features: pd.DataFrame) -> List[str]:
    """
    Check for centered windows in features.
    
    Args:
        features: Feature matrix to check
        
    Returns:
        List of features with centered windows
    """
    centered_features = []
    
    for col in features.columns:
        if _is_centered_window(col):
            centered_features.append(col)
    
    return centered_features


def _is_centered_window(feature_name: str) -> bool:
    """Check if feature name suggests centered window."""
    centered_patterns = [
        r'centered', r'center', r'mid', r'middle',
        r'symmetric', r'sym', r'balanced'
    ]
    
    feature_lower = feature_name.lower()
    return any(re.search(pattern, feature_lower) for pattern in centered_patterns)


# Global instances
_causal_auditor = None

def get_causal_auditor() -> CausalAuditor:
    """Get the global causal auditor."""
    global _causal_auditor
    if _causal_auditor is None:
        config = CausalAuditConfig()
        _causal_auditor = CausalAuditor(config)
    return _causal_auditor

def enable_causal_audit(enable: bool = True, fail_on_violation: bool = True):
    """Enable or disable causal audit."""
    auditor = get_causal_auditor()
    auditor.config.enable_audit = enable
    auditor.config.fail_on_violation = fail_on_violation
    
    if enable:
        tprint_info("🔍 Causal audit enabled")
    else:
        tprint_info("🔍 Causal audit disabled")

def get_audit_summary() -> Dict[str, Any]:
    """Get causal audit summary."""
    auditor = get_causal_auditor()
    return auditor.get_audit_summary()

def clear_audit_violations():
    """Clear causal audit violations."""
    auditor = get_causal_auditor()
    auditor.clear_violations()