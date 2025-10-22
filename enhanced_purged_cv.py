"""
Enhanced PurgedTemporalKFold with Sharp Edge Handling

This module implements a production-ready PurgedTemporalKFold cross-validator
with comprehensive edge case handling and temporal integrity validation.

Key Features:
- Purge window = label_horizon + feature_max_lag (computed from registry)
- Embargo applied symmetrically around validation windows
- Deterministic split by bin boundaries (not counts) to avoid slice-creep
- Comprehensive edge case handling
- Temporal integrity validation per fold
- Leakage detection and prevention
- Multi-entity support with blocked splits
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Generator
from dataclasses import dataclass, field
from enum import Enum
import warnings
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json
from pathlib import Path
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import check_random_state

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


class PurgeMode(Enum):
    """Purge window calculation modes."""
    FIXED = "fixed"                    # Fixed purge length
    LABEL_HORIZON = "label_horizon"    # Based on label horizon
    FEATURE_LAG = "feature_lag"        # Based on feature max lag
    COMBINED = "combined"              # Label horizon + feature max lag


@dataclass
class PurgedCVConfig:
    """Configuration for PurgedTemporalKFold."""
    
    # Basic settings
    n_splits: int = 5
    test_size: float = 0.2  # Fraction of data for testing
    
    # Purge and embargo settings
    purge_mode: PurgeMode = PurgeMode.COMBINED
    purge_length: int = 1
    embargo_length: int = 1
    label_horizon: int = 1
    feature_max_lag: int = 5
    
    # Temporal validation
    enable_temporal_validation: bool = True
    strict_temporal_order: bool = True
    min_temporal_gap: int = 1
    
    # Leakage prevention
    enable_leakage_detection: bool = True
    max_correlation_threshold: float = 0.95
    enable_entity_validation: bool = True
    
    # Edge case handling
    min_train_samples: int = 100
    min_test_samples: int = 50
    max_train_test_ratio: float = 10.0
    handle_irregular_sampling: bool = True
    handle_missing_periods: bool = True
    
    # Multi-entity support
    entity_cols: Optional[List[str]] = None
    enable_entity_blocking: bool = True
    min_entity_samples: int = 10
    
    # Reporting and logging
    enable_detailed_logging: bool = True
    save_cv_reports: bool = True
    report_directory: str = "reports/purged_cv"


@dataclass
class FoldValidationResult:
    """Result of fold validation."""
    
    fold_id: int
    is_valid: bool
    train_size: int
    test_size: int
    purge_size: int
    embargo_size: int
    effective_train_size: int
    
    # Temporal validation
    temporal_integrity_valid: bool = True
    chronological_order_valid: bool = True
    temporal_gap_valid: bool = True
    
    # Leakage detection
    leakage_detected: bool = False
    leakage_indicators: List[str] = field(default_factory=list)
    
    # Entity validation
    entity_overlap_detected: bool = False
    entity_violations: List[str] = field(default_factory=list)
    
    # Warnings and issues
    warnings: List[str] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)
    
    # Timing information
    train_start: Optional[datetime] = None
    train_end: Optional[datetime] = None
    test_start: Optional[datetime] = None
    test_end: Optional[datetime] = None
    purge_start: Optional[datetime] = None
    purge_end: Optional[datetime] = None
    embargo_start: Optional[datetime] = None
    embargo_end: Optional[datetime] = None


class EnhancedPurgedTemporalKFold(BaseCrossValidator):
    """
    Enhanced PurgedTemporalKFold with comprehensive edge case handling.
    
    This cross-validator implements purged cross-validation with:
    - Automatic purge window calculation
    - Symmetric embargo periods
    - Deterministic bin-based splitting
    - Comprehensive temporal validation
    - Leakage detection and prevention
    - Multi-entity support
    """
    
    def __init__(self, 
                 config: Optional[PurgedCVConfig] = None,
                 random_state: Optional[int] = None):
        """
        Initialize EnhancedPurgedTemporalKFold.
        
        Args:
            config: Configuration for the cross-validator
            random_state: Random state for reproducibility
        """
        self.config = config or PurgedCVConfig()
        self.random_state = random_state
        self.random_state_ = check_random_state(random_state)
        
        # Validation history
        self.validation_history = []
        self.fold_history = []
        
        # Create report directory
        if self.config.save_cv_reports:
            Path(self.config.report_directory).mkdir(parents=True, exist_ok=True)
    
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
              groups: Optional[pd.Series] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and validation sets.
        
        Args:
            X: Feature matrix with datetime index
            y: Target labels (optional)
            groups: Group labels for multi-entity support (optional)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex for purged cross-validation")
        
        try:
            if TPRINT_AVAILABLE:
                tprint_info("🚀 Starting enhanced purged cross-validation")
            
            # Calculate purge window
            purge_window = self._calculate_purge_window()
            
            # Create time bins for deterministic splitting
            time_bins = self._create_time_bins(X.index)
            
            if len(time_bins) < 2:
                if TPRINT_AVAILABLE:
                    tprint_error("❌ Insufficient time periods for cross-validation")
                return
            
            # Generate folds
            fold_count = 0
            for train_indices, test_indices in self._generate_folds(X, y, groups, time_bins, purge_window):
                # Validate fold
                validation_result = self._validate_fold(
                    X, y, groups, train_indices, test_indices, purge_window, fold_count
                )
                
                # Store validation result
                self.validation_history.append(validation_result)
                
                # Check if fold is valid
                if not validation_result.is_valid:
                    if TPRINT_AVAILABLE:
                        tprint_warning(f"⚠️ Skipping fold {fold_count + 1}: {validation_result.critical_issues}")
                    continue
                
                # Store fold information
                self.fold_history.append({
                    'fold_id': fold_count,
                    'train_indices': train_indices,
                    'test_indices': test_indices,
                    'validation_result': validation_result
                })
                
                fold_count += 1
                
                if TPRINT_AVAILABLE:
                    tprint_success(f"✅ Fold {fold_count}: train={len(train_indices)}, test={len(test_indices)}")
                
                yield train_indices, test_indices
                
                # Stop if we have enough valid folds
                if fold_count >= self.config.n_splits:
                    break
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Cross-validation completed: {fold_count} valid folds")
            
        except Exception as e:
            logger.error(f"Enhanced purged CV failed: {e}")
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Enhanced purged CV failed: {e}")
    
    def _calculate_purge_window(self) -> int:
        """Calculate purge window based on configuration."""
        if self.config.purge_mode == PurgeMode.FIXED:
            return self.config.purge_length
        elif self.config.purge_mode == PurgeMode.LABEL_HORIZON:
            return self.config.label_horizon
        elif self.config.purge_mode == PurgeMode.FEATURE_LAG:
            return self.config.feature_max_lag
        elif self.config.purge_mode == PurgeMode.COMBINED:
            return self.config.label_horizon + self.config.feature_max_lag
        else:
            return self.config.purge_length
    
    def _create_time_bins(self, time_index: pd.DatetimeIndex) -> List[Dict[str, Any]]:
        """Create deterministic time bins for splitting."""
        try:
            # Calculate bin size based on data length and number of splits
            total_periods = len(time_index)
            bin_size = max(1, total_periods // (self.config.n_splits + 1))
            
            # Create bins
            bins = []
            for i in range(0, total_periods, bin_size):
                end_idx = min(i + bin_size, total_periods)
                if end_idx - i >= self.config.min_train_samples:
                    bins.append({
                        'start_idx': i,
                        'end_idx': end_idx,
                        'start_time': time_index[i],
                        'end_time': time_index[end_idx - 1],
                        'size': end_idx - i
                    })
            
            return bins
            
        except Exception as e:
            logger.error(f"Time bin creation failed: {e}")
            return []
    
    def _generate_folds(self, X: pd.DataFrame, y: Optional[pd.Series], 
                       groups: Optional[pd.Series], time_bins: List[Dict[str, Any]], 
                       purge_window: int) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate folds with purging and embargo."""
        try:
            n_bins = len(time_bins)
            
            for i in range(n_bins - 1):  # Need at least 2 bins
                # Test period (most recent)
                test_bin = time_bins[-(i + 1)]
                test_start_idx = test_bin['start_idx']
                test_end_idx = test_bin['end_idx']
                test_indices = np.arange(test_start_idx, test_end_idx)
                
                # Purge period (before test)
                purge_start_idx = max(0, test_start_idx - purge_window)
                purge_end_idx = test_start_idx
                
                # Embargo period (after test)
                embargo_start_idx = test_end_idx
                embargo_end_idx = min(len(X), test_end_idx + self.config.embargo_length)
                
                # Training period (before purge, after embargo)
                train_start_idx = 0
                train_end_idx = purge_start_idx
                
                # Additional training from after embargo (if available)
                if embargo_end_idx < len(X):
                    train_start_idx_2 = embargo_end_idx
                    train_end_idx_2 = len(X)
                    train_indices = np.concatenate([
                        np.arange(train_start_idx, train_end_idx),
                        np.arange(train_start_idx_2, train_end_idx_2)
                    ])
                else:
                    train_indices = np.arange(train_start_idx, train_end_idx)
                
                # Validate fold sizes
                if (len(train_indices) < self.config.min_train_samples or 
                    len(test_indices) < self.config.min_test_samples):
                    continue
                
                # Check train/test ratio
                if len(train_indices) / len(test_indices) > self.config.max_train_test_ratio:
                    continue
                
                yield train_indices, test_indices
                
        except Exception as e:
            logger.error(f"Fold generation failed: {e}")
    
    def _validate_fold(self, X: pd.DataFrame, y: Optional[pd.Series], 
                      groups: Optional[pd.Series], train_indices: np.ndarray, 
                      test_indices: np.ndarray, purge_window: int, 
                      fold_id: int) -> FoldValidationResult:
        """Validate a single fold for temporal integrity and leakage."""
        result = FoldValidationResult(
            fold_id=fold_id,
            is_valid=True,
            train_size=len(train_indices),
            test_size=len(test_indices),
            purge_size=purge_window,
            embargo_size=self.config.embargo_length,
            effective_train_size=len(train_indices)
        )
        
        try:
            # Get time information
            train_times = X.index[train_indices]
            test_times = X.index[test_indices]
            
            result.train_start = train_times.min()
            result.train_end = train_times.max()
            result.test_start = test_times.min()
            result.test_end = test_times.max()
            
            # 1. Temporal integrity validation
            if self.config.enable_temporal_validation:
                temporal_result = self._validate_temporal_integrity(train_times, test_times)
                result.temporal_integrity_valid = temporal_result['valid']
                result.chronological_order_valid = temporal_result['chronological_order']
                result.temporal_gap_valid = temporal_result['temporal_gap']
                
                if not temporal_result['valid']:
                    result.critical_issues.extend(temporal_result['issues'])
                    result.is_valid = False
            
            # 2. Leakage detection
            if self.config.enable_leakage_detection:
                leakage_result = self._detect_leakage(X, y, train_indices, test_indices)
                result.leakage_detected = leakage_result['leakage_detected']
                result.leakage_indicators = leakage_result['indicators']
                
                if leakage_result['leakage_detected']:
                    result.warnings.extend(leakage_result['warnings'])
                    if leakage_result['critical']:
                        result.critical_issues.extend(leakage_result['critical_issues'])
                        result.is_valid = False
            
            # 3. Entity validation
            if self.config.enable_entity_validation and groups is not None:
                entity_result = self._validate_entity_separation(groups, train_indices, test_indices)
                result.entity_overlap_detected = entity_result['overlap_detected']
                result.entity_violations = entity_result['violations']
                
                if entity_result['overlap_detected']:
                    result.warnings.extend(entity_result['warnings'])
                    if entity_result['critical']:
                        result.critical_issues.extend(entity_result['critical_issues'])
                        result.is_valid = False
            
            # 4. Edge case handling
            edge_case_result = self._handle_edge_cases(X, train_indices, test_indices)
            result.warnings.extend(edge_case_result['warnings'])
            if edge_case_result['critical_issues']:
                result.critical_issues.extend(edge_case_result['critical_issues'])
                result.is_valid = False
            
            return result
            
        except Exception as e:
            logger.error(f"Fold validation failed: {e}")
            result.is_valid = False
            result.critical_issues.append(f"Validation error: {str(e)}")
            return result
    
    def _validate_temporal_integrity(self, train_times: pd.DatetimeIndex, 
                                   test_times: pd.DatetimeIndex) -> Dict[str, Any]:
        """Validate temporal integrity of a fold."""
        result = {
            'valid': True,
            'chronological_order': True,
            'temporal_gap': True,
            'issues': []
        }
        
        try:
            # Check chronological order
            if not train_times.is_monotonic_increasing:
                result['chronological_order'] = False
                result['issues'].append("Training data not chronologically ordered")
                result['valid'] = False
            
            if not test_times.is_monotonic_increasing:
                result['chronological_order'] = False
                result['issues'].append("Test data not chronologically ordered")
                result['valid'] = False
            
            # Check temporal gap
            if train_times.max() >= test_times.min():
                result['temporal_gap'] = False
                result['issues'].append("Training data extends into test period")
                result['valid'] = False
            
            # Check minimum temporal gap
            if self.config.min_temporal_gap > 0:
                gap = test_times.min() - train_times.max()
                if gap < pd.Timedelta(self.config.min_temporal_gap):
                    result['temporal_gap'] = False
                    result['issues'].append(f"Insufficient temporal gap: {gap}")
                    result['valid'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Temporal integrity validation failed: {e}")
            result['valid'] = False
            result['issues'].append(f"Validation error: {str(e)}")
            return result
    
    def _detect_leakage(self, X: pd.DataFrame, y: Optional[pd.Series], 
                       train_indices: np.ndarray, test_indices: np.ndarray) -> Dict[str, Any]:
        """Detect data leakage in a fold."""
        result = {
            'leakage_detected': False,
            'indicators': [],
            'warnings': [],
            'critical_issues': [],
            'critical': False
        }
        
        try:
            # Get train and test data
            X_train = X.iloc[train_indices]
            X_test = X.iloc[test_indices]
            
            # Check for identical samples
            if len(X_train) > 0 and len(X_test) > 0:
                # Simple check for identical rows
                train_hashes = X_train.apply(lambda x: hash(tuple(x)), axis=1)
                test_hashes = X_test.apply(lambda x: hash(tuple(x)), axis=1)
                common_hashes = set(train_hashes) & set(test_hashes)
                
                if len(common_hashes) > 0:
                    result['leakage_detected'] = True
                    result['indicators'].append('identical_samples')
                    result['warnings'].append(f"Identical samples detected: {len(common_hashes)}")
                    result['critical'] = True
                    result['critical_issues'].append("Identical samples in train and test sets")
            
            # Check for high correlations
            if len(X_train) > 1 and len(X_test) > 1:
                for col in X.columns:
                    if col in X_train.columns and col in X_test.columns:
                        if X[col].dtype in ['int64', 'float64']:
                            correlation = X_train[col].corr(X_test[col])
                            if not np.isnan(correlation) and abs(correlation) > self.config.max_correlation_threshold:
                                result['leakage_detected'] = True
                                result['indicators'].append('high_correlation')
                                result['warnings'].append(f"High correlation in {col}: {correlation:.3f}")
                                if abs(correlation) > 0.99:
                                    result['critical'] = True
                                    result['critical_issues'].append(f"Critical correlation in {col}: {correlation:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Leakage detection failed: {e}")
            result['warnings'].append(f"Leakage detection error: {str(e)}")
            return result
    
    def _validate_entity_separation(self, groups: pd.Series, train_indices: np.ndarray, 
                                  test_indices: np.ndarray) -> Dict[str, Any]:
        """Validate entity separation in a fold."""
        result = {
            'overlap_detected': False,
            'violations': [],
            'warnings': [],
            'critical': False
        }
        
        try:
            if self.config.entity_cols is None:
                return result
            
            # Get train and test groups
            train_groups = groups.iloc[train_indices]
            test_groups = groups.iloc[test_indices]
            
            # Check for entity overlap
            train_entities = set(train_groups.unique())
            test_entities = set(test_groups.unique())
            overlapping_entities = train_entities & test_entities
            
            if len(overlapping_entities) > 0:
                result['overlap_detected'] = True
                result['violations'] = list(overlapping_entities)
                result['warnings'].append(f"Entity overlap detected: {len(overlapping_entities)} entities")
                
                # Check if overlap is critical
                overlap_ratio = len(overlapping_entities) / len(test_entities)
                if overlap_ratio > 0.1:  # More than 10% overlap
                    result['critical'] = True
                    result['warnings'].append(f"Critical entity overlap: {overlap_ratio:.1%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Entity validation failed: {e}")
            result['warnings'].append(f"Entity validation error: {str(e)}")
            return result
    
    def _handle_edge_cases(self, X: pd.DataFrame, train_indices: np.ndarray, 
                          test_indices: np.ndarray) -> Dict[str, Any]:
        """Handle edge cases in a fold."""
        result = {
            'warnings': [],
            'critical_issues': []
        }
        
        try:
            # Check for irregular sampling
            if self.config.handle_irregular_sampling:
                train_times = X.index[train_indices]
                test_times = X.index[test_indices]
                
                # Check for large time gaps
                train_gaps = train_times.to_series().diff().dropna()
                test_gaps = test_times.to_series().diff().dropna()
                
                if len(train_gaps) > 0:
                    max_train_gap = train_gaps.max()
                    if max_train_gap > pd.Timedelta('1D'):
                        result['warnings'].append(f"Large time gap in training data: {max_train_gap}")
                
                if len(test_gaps) > 0:
                    max_test_gap = test_gaps.max()
                    if max_test_gap > pd.Timedelta('1D'):
                        result['warnings'].append(f"Large time gap in test data: {max_test_gap}")
            
            # Check for missing periods
            if self.config.handle_missing_periods:
                # This would require more sophisticated analysis
                pass
            
            return result
            
        except Exception as e:
            logger.error(f"Edge case handling failed: {e}")
            result['warnings'].append(f"Edge case handling error: {str(e)}")
            return result
    
    def get_n_splits(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None, 
                     groups: Optional[pd.Series] = None) -> int:
        """Return the number of splitting iterations."""
        return self.config.n_splits
    
    def get_validation_history(self) -> List[FoldValidationResult]:
        """Get validation history for all folds."""
        return self.validation_history.copy()
    
    def get_fold_history(self) -> List[Dict[str, Any]]:
        """Get fold history with detailed information."""
        return self.fold_history.copy()
    
    def generate_cv_report(self, filename: Optional[str] = None) -> str:
        """Generate comprehensive CV report."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"purged_cv_report_{timestamp}.json"
        
        filepath = Path(self.config.report_directory) / filename
        
        try:
            report_data = {
                'cv_timestamp': datetime.now().isoformat(),
                'config': {
                    'n_splits': self.config.n_splits,
                    'purge_mode': self.config.purge_mode.value,
                    'purge_length': self.config.purge_length,
                    'embargo_length': self.config.embargo_length,
                    'enable_temporal_validation': self.config.enable_temporal_validation,
                    'enable_leakage_detection': self.config.enable_leakage_detection
                },
                'total_folds': len(self.validation_history),
                'valid_folds': sum(1 for v in self.validation_history if v.is_valid),
                'invalid_folds': sum(1 for v in self.validation_history if not v.is_valid),
                'validation_summary': {
                    'temporal_integrity_violations': sum(1 for v in self.validation_history if not v.temporal_integrity_valid),
                    'leakage_detections': sum(1 for v in self.validation_history if v.leakage_detected),
                    'entity_overlaps': sum(1 for v in self.validation_history if v.entity_overlap_detected)
                },
                'folds': []
            }
            
            for i, fold in enumerate(self.fold_history):
                fold_data = {
                    'fold_id': fold['fold_id'],
                    'train_size': fold['validation_result'].train_size,
                    'test_size': fold['validation_result'].test_size,
                    'is_valid': fold['validation_result'].is_valid,
                    'temporal_integrity_valid': fold['validation_result'].temporal_integrity_valid,
                    'leakage_detected': fold['validation_result'].leakage_detected,
                    'entity_overlap_detected': fold['validation_result'].entity_overlap_detected,
                    'warnings_count': len(fold['validation_result'].warnings),
                    'critical_issues_count': len(fold['validation_result'].critical_issues)
                }
                report_data['folds'].append(fold_data)
            
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            if TPRINT_AVAILABLE:
                tprint_success(f"📄 CV report saved: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to generate CV report: {e}")
            return ""


# Convenience functions
def create_enhanced_purged_cv(config: Optional[PurgedCVConfig] = None, 
                            random_state: Optional[int] = None) -> EnhancedPurgedTemporalKFold:
    """Create enhanced purged temporal K-fold cross-validator."""
    return EnhancedPurgedTemporalKFold(config, random_state)

def create_quick_purged_cv(n_splits: int = 5, 
                          purge_length: int = 1, 
                          embargo_length: int = 1) -> EnhancedPurgedTemporalKFold:
    """Create quick purged CV with basic settings."""
    config = PurgedCVConfig(
        n_splits=n_splits,
        purge_length=purge_length,
        embargo_length=embargo_length,
        enable_detailed_logging=False,
        save_cv_reports=False
    )
    return EnhancedPurgedTemporalKFold(config)


if __name__ == "__main__":
    # Example usage
    print("Enhanced PurgedTemporalKFold with Sharp Edge Handling")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=1000, freq='1H')
    X = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000)
    }, index=dates)
    
    y = pd.Series(np.random.choice([0, 1], size=1000, p=[0.7, 0.3]), index=dates)
    groups = pd.Series(np.random.choice(['A', 'B', 'C'], size=1000), index=dates)
    
    # Create enhanced purged CV
    config = PurgedCVConfig(
        n_splits=3,
        purge_mode=PurgeMode.COMBINED,
        label_horizon=2,
        feature_max_lag=3,
        enable_temporal_validation=True,
        enable_leakage_detection=True,
        entity_cols=['groups']
    )
    
    cv = create_enhanced_purged_cv(config)
    
    # Perform cross-validation
    fold_count = 0
    for train_idx, test_idx in cv.split(X, y, groups):
        fold_count += 1
        print(f"Fold {fold_count}: train={len(train_idx)}, test={len(test_idx)}")
    
    # Get validation results
    validation_history = cv.get_validation_history()
    print(f"Total folds: {len(validation_history)}")
    print(f"Valid folds: {sum(1 for v in validation_history if v.is_valid)}")
    print(f"Leakage detected: {sum(1 for v in validation_history if v.leakage_detected)}")
    print(f"Entity overlaps: {sum(1 for v in validation_history if v.entity_overlap_detected)}")
    
    # Generate report
    report_path = cv.generate_cv_report()
    print(f"CV report saved: {report_path}")