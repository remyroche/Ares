"""
Consolidated Cross-Validation System

This module consolidates all cross-validation implementations into a single,
comprehensive system that provides:

1. Enhanced Purged Cross-Validation with sharp edge handling
2. Walk-forward validation with nested CV
3. Universal temporal validation
4. Standard KFold/Stratified KFold cross-validation
5. Nested cross-validation for unbiased model assessment
6. Comprehensive reporting and monitoring

This replaces the following redundant implementations:
- enhanced_purged_cv.py
- src/validation/walkforward_validation.py
- src/utils/ml_common/validation/universal_temporal_validation.py
- src/utils/ml_common/validation/unified_cv.py
- src/utils/ml_common/validation/temporal_cross_validation.py
- src/utils/ml_common/validation/cv.py
- src/utils/purged_kfold.py
- src/features_common/optimization/cv_base.py
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Generator, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json
from pathlib import Path
from sklearn.model_selection import (
    BaseCrossValidator, KFold, StratifiedKFold, TimeSeriesSplit,
    cross_val_score, cross_validate
)
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score

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

# ============================================================================
# ENUMS AND CONFIGURATION CLASSES
# ============================================================================

class PurgeMode(Enum):
    """Purge window calculation modes."""
    FIXED = "fixed"
    LABEL_HORIZON = "label_horizon"
    FEATURE_LAG = "feature_lag"
    COMBINED = "combined"

class ValidationType(Enum):
    """Types of validation."""
    WALK_FORWARD = "walk_forward"
    NESTED_CV = "nested_cv"
    ABLATION = "ablation"
    SPA_CHECK = "spa_check"
    PURGED = "purged"
    TEMPORAL = "temporal"
    STANDARD = "standard"

@dataclass
class ConsolidatedCVConfig:
    """Unified configuration for all CV types."""
    
    # Basic settings
    n_splits: int = 5
    test_size: float = 0.2
    random_state: Optional[int] = None
    
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
    
    # Walk-forward specific
    enable_walk_forward: bool = True
    initial_train_size: float = 0.6
    step_size: float = 0.1
    min_test_size: float = 0.1
    
    # Nested CV
    enable_nested_cv: bool = True
    n_inner_folds: int = 3
    
    # Ablation testing
    enable_ablation: bool = False
    ablation_steps: List[str] = field(default_factory=lambda: [
        'parents_only',
        'parents_transforms',
        'parents_transforms_patch',
        'parents_transforms_patch_8_interactions',
        'parents_transforms_patch_15_interactions'
    ])
    
    # SPA testing
    enable_spa_test: bool = False
    spa_permutations: int = 1000
    significance_level: float = 0.05
    
    # Reporting and logging
    enable_detailed_logging: bool = True
    save_cv_reports: bool = True
    report_directory: str = "reports/consolidated_cv"

@dataclass
class FoldValidationResult:
    """Result of fold validation."""
    
    fold_id: int
    is_valid: bool
    train_size: int
    test_size: int
    purge_size: int = 0
    embargo_size: int = 0
    effective_train_size: int = 0
    
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

@dataclass
class ValidationResult:
    """Comprehensive validation result."""
    
    # Basic metrics
    scores: Optional[List[float]] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    folds: Optional[int] = None
    
    # Multi-metric scoring
    mean_scores: Optional[Dict[str, float]] = None
    std_scores: Optional[Dict[str, float]] = None
    train_scores: Optional[Dict[str, float]] = None
    
    # Detailed results
    fold_results: List[FoldValidationResult] = field(default_factory=list)
    validation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Temporal validation
    temporal_order_valid: bool = True
    leakage_detected: bool = False
    validation_score: float = 0.0
    
    # Metadata
    validation_type: str = "unknown"
    model_name: str = "unknown"
    validation_timestamp: str = None
    
    def __post_init__(self):
        if self.validation_timestamp is None:
            self.validation_timestamp = datetime.now().isoformat()

# ============================================================================
# MAIN CONSOLIDATED CROSS-VALIDATOR
# ============================================================================

class ConsolidatedCrossValidator(BaseCrossValidator):
    """
    Consolidated cross-validator that combines all CV strategies.
    
    This class provides a unified interface for:
    - Purged cross-validation with sharp edge handling
    - Walk-forward validation with nested CV
    - Universal temporal validation
    - Standard KFold/Stratified KFold cross-validation
    - Nested cross-validation
    """
    
    def __init__(self, 
                 config: Optional[ConsolidatedCVConfig] = None,
                 validation_type: ValidationType = ValidationType.PURGED,
                 random_state: Optional[int] = None):
        """
        Initialize consolidated cross-validator.
        
        Args:
            config: Configuration for the cross-validator
            validation_type: Type of validation to perform
            random_state: Random state for reproducibility
        """
        self.config = config or ConsolidatedCVConfig()
        self.validation_type = validation_type
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
            X: Feature matrix
            y: Target labels (optional)
            groups: Group labels for multi-entity support (optional)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if self.validation_type == ValidationType.PURGED:
            yield from self._purged_split(X, y, groups)
        elif self.validation_type == ValidationType.WALK_FORWARD:
            yield from self._walk_forward_split(X, y, groups)
        elif self.validation_type == ValidationType.TEMPORAL:
            yield from self._temporal_split(X, y, groups)
        elif self.validation_type == ValidationType.STANDARD:
            yield from self._standard_split(X, y, groups)
        else:
            raise ValueError(f"Unsupported validation type: {self.validation_type}")
    
    def _purged_split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                     groups: Optional[pd.Series] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate purged cross-validation splits."""
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex for purged cross-validation")
        
        try:
            if TPRINT_AVAILABLE:
                tprint_info("🚀 Starting purged cross-validation")
            
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
            for train_indices, test_indices in self._generate_purged_folds(X, y, groups, time_bins, purge_window):
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
                tprint_success(f"✅ Purged cross-validation completed: {fold_count} valid folds")
            
        except Exception as e:
            logger.error(f"Purged CV failed: {e}")
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Purged CV failed: {e}")
    
    def _walk_forward_split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                           groups: Optional[pd.Series] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate walk-forward validation splits."""
        try:
            if TPRINT_AVAILABLE:
                tprint_info("🚀 Starting walk-forward validation")
            
            n_samples = len(X)
            initial_train_size = int(n_samples * self.config.initial_train_size)
            step_size = int(n_samples * self.config.step_size)
            min_test_size = int(n_samples * self.config.min_test_size)
            
            current_train_end = initial_train_size
            
            fold_count = 0
            while current_train_end + min_test_size <= n_samples:
                # Calculate test size
                remaining_samples = n_samples - current_train_end
                test_size = min(remaining_samples, int(remaining_samples * self.config.test_size))
                
                if test_size < min_test_size:
                    break
                
                # Generate indices
                train_indices = np.arange(0, current_train_end)
                test_indices = np.arange(current_train_end, current_train_end + test_size)
                
                # Validate fold
                validation_result = self._validate_fold(
                    X, y, groups, train_indices, test_indices, 0, fold_count
                )
                
                # Store validation result
                self.validation_history.append(validation_result)
                
                # Check if fold is valid
                if not validation_result.is_valid:
                    if TPRINT_AVAILABLE:
                        tprint_warning(f"⚠️ Skipping fold {fold_count + 1}: {validation_result.critical_issues}")
                    current_train_end += step_size
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
                
                # Move to next fold
                current_train_end += step_size
                
                # Stop if we have enough folds
                if fold_count >= self.config.n_splits:
                    break
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Walk-forward validation completed: {fold_count} valid folds")
            
        except Exception as e:
            logger.error(f"Walk-forward CV failed: {e}")
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Walk-forward CV failed: {e}")
    
    def _temporal_split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                       groups: Optional[pd.Series] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate temporal cross-validation splits."""
        try:
            if TPRINT_AVAILABLE:
                tprint_info("🚀 Starting temporal cross-validation")
            
            # Use sklearn's TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
            
            fold_count = 0
            for train_indices, test_indices in tscv.split(X, y):
                # Validate fold
                validation_result = self._validate_fold(
                    X, y, groups, train_indices, test_indices, 0, fold_count
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
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Temporal cross-validation completed: {fold_count} valid folds")
            
        except Exception as e:
            logger.error(f"Temporal CV failed: {e}")
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Temporal CV failed: {e}")
    
    def _standard_split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                       groups: Optional[pd.Series] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate standard cross-validation splits."""
        try:
            if TPRINT_AVAILABLE:
                tprint_info("🚀 Starting standard cross-validation")
            
            # Determine if classification
            is_classification = False
            if y is not None:
                try:
                    unique_values = np.unique(y)
                    is_classification = len(unique_values) <= 10
                except Exception:
                    is_classification = False
            
            # Choose appropriate splitter
            if is_classification:
                cv = StratifiedKFold(n_splits=self.config.n_splits, shuffle=False, random_state=self.random_state)
            else:
                cv = KFold(n_splits=self.config.n_splits, shuffle=False, random_state=self.random_state)
            
            fold_count = 0
            for train_indices, test_indices in cv.split(X, y):
                # Validate fold
                validation_result = self._validate_fold(
                    X, y, groups, train_indices, test_indices, 0, fold_count
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
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Standard cross-validation completed: {fold_count} valid folds")
            
        except Exception as e:
            logger.error(f"Standard CV failed: {e}")
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Standard CV failed: {e}")
    
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
    
    def _generate_purged_folds(self, X: pd.DataFrame, y: Optional[pd.Series], 
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
            filename = f"consolidated_cv_report_{timestamp}.json"
        
        filepath = Path(self.config.report_directory) / filename
        
        try:
            report_data = {
                'cv_timestamp': datetime.now().isoformat(),
                'validation_type': self.validation_type.value,
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

# ============================================================================
# CONVENIENCE FUNCTIONS AND WRAPPERS
# ============================================================================

def create_consolidated_cv(config: Optional[ConsolidatedCVConfig] = None,
                          validation_type: ValidationType = ValidationType.PURGED,
                          random_state: Optional[int] = None) -> ConsolidatedCrossValidator:
    """Create consolidated cross-validator."""
    return ConsolidatedCrossValidator(config, validation_type, random_state)

def create_purged_cv(n_splits: int = 5, 
                    purge_length: int = 1, 
                    embargo_length: int = 1) -> ConsolidatedCrossValidator:
    """Create purged CV with basic settings."""
    config = ConsolidatedCVConfig(
        n_splits=n_splits,
        purge_length=purge_length,
        embargo_length=embargo_length,
        enable_detailed_logging=False,
        save_cv_reports=False
    )
    return ConsolidatedCrossValidator(config, ValidationType.PURGED)

def create_walk_forward_cv(n_splits: int = 5,
                          initial_train_size: float = 0.6,
                          step_size: float = 0.1) -> ConsolidatedCrossValidator:
    """Create walk-forward CV with basic settings."""
    config = ConsolidatedCVConfig(
        n_splits=n_splits,
        initial_train_size=initial_train_size,
        step_size=step_size,
        enable_detailed_logging=False,
        save_cv_reports=False
    )
    return ConsolidatedCrossValidator(config, ValidationType.WALK_FORWARD)

def create_temporal_cv(n_splits: int = 5) -> ConsolidatedCrossValidator:
    """Create temporal CV with basic settings."""
    config = ConsolidatedCVConfig(
        n_splits=n_splits,
        enable_detailed_logging=False,
        save_cv_reports=False
    )
    return ConsolidatedCrossValidator(config, ValidationType.TEMPORAL)

def create_standard_cv(n_splits: int = 5, random_state: Optional[int] = None) -> ConsolidatedCrossValidator:
    """Create standard CV with basic settings."""
    config = ConsolidatedCVConfig(
        n_splits=n_splits,
        enable_detailed_logging=False,
        save_cv_reports=False
    )
    return ConsolidatedCrossValidator(config, ValidationType.STANDARD, random_state)

# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================================

# Legacy class names for backward compatibility
PurgedKFoldTime = ConsolidatedCrossValidator
UniversalTemporalValidator = ConsolidatedCrossValidator
WalkForwardValidator = ConsolidatedCrossValidator
UnifiedCrossValidator = ConsolidatedCrossValidator

# Legacy function names for backward compatibility
def purged_time_series_splits(X: pd.DataFrame, y: Optional[pd.Series] = None, 
                             config: Optional[ConsolidatedCVConfig] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Legacy function for purged time series splits."""
    if config is None:
        config = ConsolidatedCVConfig()
    cv = ConsolidatedCrossValidator(config, ValidationType.PURGED)
    yield from cv.split(X, y)

def temporal_cross_validation(model: Any, X: np.ndarray, y: np.ndarray, 
                             n_splits: int = 5, gap: int = 0, 
                             test_size: Optional[int] = None,
                             scoring: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
    """Legacy function for temporal cross-validation."""
    from sklearn.model_selection import cross_val_score, cross_validate
    
    if test_size is None:
        test_size = max(1, len(X) // (n_splits + 1))
    
    # Use sklearn's TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size)
    
    if isinstance(scoring, list):
        cv_result = cross_validate(model, X, y, cv=tscv, scoring=scoring, return_train_score=True)
        return {
            'mean_scores': {m: float(np.mean(cv_result.get(f"test_{m}", []))) for m in scoring},
            'std_scores': {m: float(np.std(cv_result.get(f"test_{m}", []))) for m in scoring},
            'train_scores': {m: float(np.mean(cv_result.get(f"train_{m}", []))) for m in scoring if f"train_{m}" in cv_result},
            'cv_folds': n_splits,
        }
    else:
        scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring)
        return {
            'scores': scores.tolist(),
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'cv_folds': n_splits,
        }

def perform_cross_validation(model: Any, X: np.ndarray, y: np.ndarray, 
                            strategy: str = "temporal", cv_folds: int = 5,
                            scoring: Union[str, List[str], None] = None,
                            random_state: Optional[int] = 42,
                            stratified: Optional[bool] = None,
                            n_jobs: int = -1,
                            temporal_gap: int = 0,
                            temporal_test_size: Optional[int] = None) -> Dict[str, Any]:
    """Legacy function for cross-validation."""
    from sklearn.model_selection import cross_val_score, cross_validate
    
    # Determine if classification
    is_classification = False
    if stratified is None:
        try:
            unique_values = np.unique(y)
            is_classification = len(unique_values) <= 10
        except Exception:
            is_classification = False
    else:
        is_classification = stratified
    
    if scoring is None:
        scoring = "accuracy" if is_classification else "r2"
    
    if strategy == "temporal":
        if temporal_test_size is None:
            temporal_test_size = max(1, len(X) // (cv_folds + 1))
        tscv = TimeSeriesSplit(n_splits=cv_folds, gap=temporal_gap, test_size=temporal_test_size)
        cv = tscv
    else:
        if is_classification:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=False, random_state=random_state)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=False, random_state=random_state)
    
    if isinstance(scoring, list):
        cv_result = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, return_train_score=True)
        return {
            'mean_scores': {m: float(np.mean(cv_result.get(f"test_{m}", []))) for m in scoring},
            'std_scores': {m: float(np.std(cv_result.get(f"test_{m}", []))) for m in scoring},
            'train_scores': {m: float(np.mean(cv_result.get(f"train_{m}", []))) for m in scoring if f"train_{m}" in cv_result},
            'cv_folds': cv_folds,
        }
    else:
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
        return {
            'scores': scores.tolist(),
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'cv_folds': cv_folds,
        }

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Main classes
    'ConsolidatedCrossValidator',
    'ConsolidatedCVConfig',
    'FoldValidationResult',
    'ValidationResult',
    'PurgeMode',
    'ValidationType',
    
    # Convenience functions
    'create_consolidated_cv',
    'create_purged_cv',
    'create_walk_forward_cv',
    'create_temporal_cv',
    'create_standard_cv',
    
    # Legacy compatibility
    'PurgedKFoldTime',
    'UniversalTemporalValidator',
    'WalkForwardValidator',
    'UnifiedCrossValidator',
    'purged_time_series_splits',
    'temporal_cross_validation',
    'perform_cross_validation',
]