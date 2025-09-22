"""
Temporal Validation and Walk-Forward Validation

Enhanced temporal validation with walk-forward validation for time series data.
Prevents lookahead bias and ensures proper temporal data handling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Generator
from dataclasses import dataclass
from sklearn.model_selection import BaseCrossValidator
import logging

logger = logging.getLogger(__name__)

@dataclass
class TemporalValidationConfig:
    """Configuration for temporal validation."""
    
    # Temporal validation settings
    enable_temporal_checks: bool = True
    strict_temporal_order: bool = True
    min_temporal_gap: int = 1  # Minimum gap between train and test
    
    # Walk-forward validation settings
    enable_walk_forward: bool = True
    initial_train_size: float = 0.6  # 60% for initial training
    step_size: float = 0.1  # 10% step size
    min_test_size: float = 0.1  # Minimum 10% for test
    
    # Data leakage detection
    enable_leakage_detection: bool = True
    max_correlation_threshold: float = 0.95
    temporal_consistency_threshold: float = 0.8
    
    # Validation reporting
    detailed_reporting: bool = True
    save_validation_plots: bool = False

class TemporalValidator:
    """Temporal validation utilities."""
    
    def __init__(self, config: TemporalValidationConfig):
        self.config = config
        self.validation_results = []
        
    def validate_temporal_order(self, 
                               train_data: np.ndarray, 
                               test_data: np.ndarray,
                               timestamps: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """
        Validate temporal order of train/test split.
        
        Args:
            train_data: Training data
            test_data: Test data
            timestamps: Optional timestamps for validation
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        if not self.config.enable_temporal_checks:
            return True, "Temporal checks disabled"
        
        try:
            # Basic size validation
            if len(train_data) == 0 or len(test_data) == 0:
                return False, "Empty train or test set"
            
            # If timestamps available, validate temporal order
            if timestamps is not None:
                if len(timestamps) != len(train_data) + len(test_data):
                    return False, "Timestamp length mismatch"
                
                train_timestamps = timestamps[:len(train_data)]
                test_timestamps = timestamps[len(train_data):]
                
                # Check temporal order
                if self.config.strict_temporal_order:
                    if np.max(train_timestamps) >= np.min(test_timestamps):
                        return False, f"Temporal leakage: train max ({np.max(train_timestamps)}) >= test min ({np.min(test_timestamps)})"
                
                # Check minimum gap
                gap = np.min(test_timestamps) - np.max(train_timestamps)
                if gap < self.config.min_temporal_gap:
                    return False, f"Insufficient temporal gap: {gap} < {self.config.min_temporal_gap}"
            
            return True, "Temporal order validation passed"
            
        except Exception as e:
            logger.error(f"Temporal validation error: {e}")
            return False, f"Temporal validation failed: {str(e)}"
    
    def detect_data_leakage(self, 
                           train_data: np.ndarray, 
                           test_data: np.ndarray,
                           feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect potential data leakage between train and test sets.
        
        Args:
            train_data: Training data
            test_data: Test data
            feature_names: Optional feature names
            
        Returns:
            Dict: Leakage detection results
        """
        if not self.config.enable_leakage_detection:
            return {'leakage_detected': False, 'message': 'Leakage detection disabled'}
        
        try:
            leakage_indicators = []
            warnings = []
            
            # 1. Check for identical samples
            train_set = set(map(tuple, train_data))
            test_set = set(map(tuple, test_data))
            common_samples = train_set.intersection(test_set)
            
            if len(common_samples) > 0:
                leakage_indicators.append('identical_samples')
                warnings.append(f"🚨 {len(common_samples)} identical samples found")
            
            # 2. Check for highly correlated features
            if train_data.shape[1] > 1:
                train_corr = np.corrcoef(train_data.T)
                test_corr = np.corrcoef(test_data.T)
                
                # Check for perfect correlation
                high_corr_train = np.sum(np.abs(train_corr - np.eye(train_corr.shape[0])) > self.config.max_correlation_threshold)
                high_corr_test = np.sum(np.abs(test_corr - np.eye(test_corr.shape[0])) > self.config.max_correlation_threshold)
                
                if high_corr_train > 0:
                    leakage_indicators.append('high_correlation_train')
                    warnings.append(f"⚠️ {high_corr_train} highly correlated features in training")
                
                if high_corr_test > 0:
                    leakage_indicators.append('high_correlation_test')
                    warnings.append(f"⚠️ {high_corr_test} highly correlated features in test")
            
            # 3. Check for temporal consistency
            if hasattr(train_data, 'index') and hasattr(test_data, 'index'):
                train_index = train_data.index
                test_index = test_data.index
                
                # Check for overlapping time periods
                if hasattr(train_index, 'min') and hasattr(test_index, 'min'):
                    if train_index.max() >= test_index.min():
                        leakage_indicators.append('temporal_overlap')
                        warnings.append("🚨 Temporal overlap between train and test")
            
            # 4. Statistical similarity check
            if len(train_data) > 10 and len(test_data) > 10:
                train_mean = np.mean(train_data, axis=0)
                test_mean = np.mean(test_data, axis=0)
                
                # Check if means are suspiciously similar
                mean_similarity = np.corrcoef(train_mean, test_mean)[0, 1]
                if mean_similarity > self.config.temporal_consistency_threshold:
                    leakage_indicators.append('statistical_similarity')
                    warnings.append(f"⚠️ High statistical similarity: {mean_similarity:.3f}")
            
            leakage_detected = len(leakage_indicators) > 0
            
            return {
                'leakage_detected': leakage_detected,
                'indicators': leakage_indicators,
                'warnings': warnings,
                'common_samples': len(common_samples) if 'common_samples' in locals() else 0,
                'message': f"Leakage detected: {len(leakage_indicators)} indicators" if leakage_detected else "No leakage detected"
            }
            
        except Exception as e:
            logger.error(f"Leakage detection error: {e}")
            return {
                'leakage_detected': False,
                'message': f"Leakage detection failed: {str(e)}"
            }

class WalkForwardValidator(BaseCrossValidator):
    """
    Walk-forward validation for time series data.
    
    Implements proper time series cross-validation without lookahead bias.
    """
    
    def __init__(self, 
                 initial_train_size: float = 0.6,
                 step_size: float = 0.1,
                 min_test_size: float = 0.1,
                 n_splits: Optional[int] = None):
        """
        Initialize walk-forward validator.
        
        Args:
            initial_train_size: Initial training set size (0.0 to 1.0)
            step_size: Step size for each fold (0.0 to 1.0)
            min_test_size: Minimum test set size (0.0 to 1.0)
            n_splits: Number of splits (auto-calculated if None)
        """
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.min_test_size = min_test_size
        self.n_splits = n_splits
        
        # Validate parameters
        if not 0 < initial_train_size < 1:
            raise ValueError(f"initial_train_size must be between 0 and 1, got {initial_train_size}")
        if not 0 < step_size < 1:
            raise ValueError(f"step_size must be between 0 and 1, got {step_size}")
        if not 0 < min_test_size < 1:
            raise ValueError(f"min_test_size must be between 0 and 1, got {min_test_size}")
        if initial_train_size + min_test_size > 1:
            raise ValueError("initial_train_size + min_test_size cannot exceed 1")
    
    def split(self, X, y=None, groups=None):
        """
        Generate train/test splits for walk-forward validation.
        
        Args:
            X: Input data
            y: Target data (optional)
            groups: Group data (optional)
            
        Yields:
            Tuple[np.ndarray, np.ndarray]: (train_indices, test_indices)
        """
        n_samples = len(X)
        
        # Calculate number of splits
        if self.n_splits is None:
            # Auto-calculate based on step size
            available_space = 1 - self.initial_train_size - self.min_test_size
            self.n_splits = max(1, int(available_space / self.step_size))
        
        # Generate splits
        for i in range(self.n_splits):
            # Calculate split points
            train_start = 0
            train_end = int(n_samples * (self.initial_train_size + i * self.step_size))
            test_start = train_end
            test_end = min(n_samples, int(n_samples * (self.initial_train_size + i * self.step_size + self.min_test_size)))
            
            # Ensure minimum test size
            if test_end - test_start < int(n_samples * self.min_test_size):
                continue
            
            # Generate indices
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Get number of splits."""
        return self.n_splits if self.n_splits is not None else 5

class TemporalCrossValidator:
    """Enhanced temporal cross-validation with comprehensive validation."""
    
    def __init__(self, config: TemporalValidationConfig):
        self.config = config
        self.validator = TemporalValidator(config)
        self.validation_history = []
        
    def validate_temporal_split(self, 
                               X_train: np.ndarray, 
                               X_test: np.ndarray,
                               y_train: np.ndarray, 
                               y_test: np.ndarray,
                               timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Comprehensive temporal split validation.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            timestamps: Optional timestamps
            
        Returns:
            Dict: Comprehensive validation results
        """
        results = {
            'temporal_order_valid': False,
            'leakage_detected': False,
            'warnings': [],
            'recommendations': [],
            'validation_score': 0.0
        }
        
        try:
            # 1. Temporal order validation
            temporal_valid, temporal_msg = self.validator.validate_temporal_order(
                X_train, X_test, timestamps
            )
            results['temporal_order_valid'] = temporal_valid
            results['temporal_message'] = temporal_msg
            
            if not temporal_valid:
                results['warnings'].append(f"🚨 Temporal order violation: {temporal_msg}")
                results['recommendations'].append("Fix temporal order in train/test split")
            
            # 2. Data leakage detection
            leakage_results = self.validator.detect_data_leakage(X_train, X_test)
            results['leakage_detected'] = leakage_results['leakage_detected']
            results['leakage_indicators'] = leakage_results.get('indicators', [])
            results['leakage_warnings'] = leakage_results.get('warnings', [])
            
            if leakage_results['leakage_detected']:
                results['warnings'].extend(leakage_results['warnings'])
                results['recommendations'].append("Investigate and fix data leakage")
            
            # 3. Calculate validation score
            score = 0.0
            if temporal_valid:
                score += 0.5
            if not leakage_results['leakage_detected']:
                score += 0.5
            
            results['validation_score'] = score
            
            # 4. Generate recommendations
            if score < 1.0:
                results['recommendations'].extend([
                    "Use walk-forward validation for time series",
                    "Enable strict temporal ordering",
                    "Implement proper data preprocessing"
                ])
            
            # Store validation history
            self.validation_history.append({
                'timestamp': len(self.validation_history),
                'temporal_valid': temporal_valid,
                'leakage_detected': leakage_results['leakage_detected'],
                'score': score
            })
            
        except Exception as e:
            logger.error(f"Temporal validation error: {e}")
            results['warnings'].append(f"❌ Validation failed: {str(e)}")
            results['validation_score'] = 0.0
        
        return results
    
    def create_walk_forward_validator(self, 
                                   initial_train_size: Optional[float] = None,
                                   step_size: Optional[float] = None,
                                   min_test_size: Optional[float] = None) -> WalkForwardValidator:
        """
        Create walk-forward validator with configuration.
        
        Args:
            initial_train_size: Initial training size (uses config if None)
            step_size: Step size (uses config if None)
            min_test_size: Minimum test size (uses config if None)
            
        Returns:
            WalkForwardValidator: Configured walk-forward validator
        """
        if not self.config.enable_walk_forward:
            raise ValueError("Walk-forward validation is disabled in configuration")
        
        return WalkForwardValidator(
            initial_train_size=initial_train_size or self.config.initial_train_size,
            step_size=step_size or self.config.step_size,
            min_test_size=min_test_size or self.config.min_test_size
        )

# Global instances for easy access
DEFAULT_TEMPORAL_CONFIG = TemporalValidationConfig()
DEFAULT_TEMPORAL_VALIDATOR = TemporalValidator(DEFAULT_TEMPORAL_CONFIG)
DEFAULT_TEMPORAL_CV = TemporalCrossValidator(DEFAULT_TEMPORAL_CONFIG)

def get_temporal_config() -> TemporalValidationConfig:
    """Get the default temporal validation configuration."""
    return DEFAULT_TEMPORAL_CONFIG

def get_temporal_validator() -> TemporalValidator:
    """Get the default temporal validator."""
    return DEFAULT_TEMPORAL_VALIDATOR

def get_temporal_cv() -> TemporalCrossValidator:
    """Get the default temporal cross-validator."""
    return DEFAULT_TEMPORAL_CV

def create_walk_forward_validator(**kwargs) -> WalkForwardValidator:
    """Create a walk-forward validator with custom parameters."""
    return WalkForwardValidator(**kwargs)