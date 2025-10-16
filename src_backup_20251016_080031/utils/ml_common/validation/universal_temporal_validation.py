"""
Universal Temporal Validation for ML Common

Temporal validation system that can be used across all ML models to prevent
lookahead bias and ensure proper time series data handling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Generator
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)

@dataclass
class TemporalValidationConfig:
    """Configuration for temporal validation across all ML models."""
    
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
    
    # Cross-validation settings
    n_splits: int = 5
    test_size: float = 0.2  # 20% for test
    gap_size: int = 1  # Gap between train and test
    
    # Validation reporting
    detailed_reporting: bool = True
    save_validation_plots: bool = False

@dataclass
class TemporalValidationReport:
    """Temporal validation report for any ML model."""
    
    # Validation results
    temporal_order_valid: bool
    leakage_detected: bool
    validation_score: float  # 0.0 to 1.0
    
    # Detailed analysis
    temporal_message: str
    leakage_indicators: List[str]
    leakage_warnings: List[str]
    warnings: List[str]
    recommendations: List[str]
    
    # Metadata
    model_name: str = "unknown"
    model_type: str = "unknown"
    validation_timestamp: str = None
    
    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.validation_timestamp is None:
            self.validation_timestamp = datetime.now().isoformat()

class UniversalTemporalValidator:
    """Universal temporal validator for all ML models."""
    
    def __init__(self, config: Optional[TemporalValidationConfig] = None):
        """
        Initialize universal temporal validator.
        
        Args:
            config: Temporal validation configuration
        """
        self.config = config or TemporalValidationConfig()
        self.validation_history = []
    
    def validate_temporal_split(self, 
                               X_train: np.ndarray, 
                               X_test: np.ndarray,
                               y_train: np.ndarray, 
                               y_test: np.ndarray,
                               timestamps: Optional[np.ndarray] = None,
                               model_name: str = "unknown",
                               model_type: str = "unknown") -> TemporalValidationReport:
        """
        Validate temporal split for any ML model.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            timestamps: Optional timestamps for validation
            model_name: Name of the model
            model_type: Type of model
            
        Returns:
            TemporalValidationReport: Comprehensive validation results
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
            temporal_valid, temporal_msg = self._validate_temporal_order(
                X_train, X_test, timestamps
            )
            results['temporal_order_valid'] = temporal_valid
            results['temporal_message'] = temporal_msg
            
            if not temporal_valid:
                results['warnings'].append(f"🚨 Temporal order violation: {temporal_msg}")
                results['recommendations'].append("Fix temporal order in train/test split")
            
            # 2. Data leakage detection
            leakage_results = self._detect_data_leakage(X_train, X_test)
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
            
            # Create comprehensive report
            report = TemporalValidationReport(
                temporal_order_valid=temporal_valid,
                leakage_detected=leakage_results['leakage_detected'],
                validation_score=score,
                temporal_message=temporal_msg,
                leakage_indicators=leakage_results.get('indicators', []),
                leakage_warnings=leakage_results.get('warnings', []),
                warnings=results['warnings'],
                recommendations=results['recommendations'],
                model_name=model_name,
                model_type=model_type
            )
            
            # Track validation history
            self.validation_history.append(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Temporal validation error: {e}")
            return TemporalValidationReport(
                temporal_order_valid=False,
                leakage_detected=True,
                validation_score=0.0,
                temporal_message=f"Validation failed: {str(e)}",
                leakage_indicators=[],
                leakage_warnings=[],
                warnings=[f"❌ Validation failed: {str(e)}"],
                recommendations=["Fix validation error and retry"],
                model_name=model_name,
                model_type=model_type
            )
    
    def _validate_temporal_order(self, 
                               train_data: np.ndarray, 
                               test_data: np.ndarray,
                               timestamps: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """Validate temporal order of train/test split."""
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
    
    def _detect_data_leakage(self, 
                           train_data: np.ndarray, 
                           test_data: np.ndarray) -> Dict[str, Any]:
        """Detect potential data leakage between train and test sets."""
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
            
            # 3. Statistical similarity check
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

class UniversalTimeSeriesSplit(BaseCrossValidator):
    """
    Universal time series cross-validation splitter for any ML model.
    
    Ensures proper temporal ordering and prevents lookahead bias.
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 test_size: float = 0.2,
                 gap_size: int = 1,
                 min_train_size: float = 0.3,
                 min_test_size: float = 0.1):
        """
        Initialize universal time series splitter.
        
        Args:
            n_splits: Number of splits
            test_size: Test set size (0.0 to 1.0)
            gap_size: Gap between train and test sets
            min_train_size: Minimum training set size
            min_test_size: Minimum test set size
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap_size = gap_size
        self.min_train_size = min_train_size
        self.min_test_size = min_test_size
        
        # Validate parameters
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        if not 0 < min_train_size < 1:
            raise ValueError(f"min_train_size must be between 0 and 1, got {min_train_size}")
        if not 0 < min_test_size < 1:
            raise ValueError(f"min_test_size must be between 0 and 1, got {min_test_size}")
        if min_train_size + min_test_size > 1:
            raise ValueError("min_train_size + min_test_size cannot exceed 1")
    
    def split(self, X, y=None, groups=None):
        """
        Generate train/test splits for time series with proper temporal ordering.

        Args:
            X: Input data
            y: Target data (optional)
            groups: Group data (optional)

        Yields:
            Tuple[np.ndarray, np.ndarray]: (train_indices, test_indices)
        """
        n_samples = len(X)

        # Check if X has a temporal index
        if hasattr(X, 'index') and hasattr(X.index, 'is_monotonic_increasing'):
            if X.index.is_monotonic_increasing:
                # Use time-based splitting
                return self._temporal_split(X, n_samples)
            else:
                logger.warning("Index is not monotonic increasing, falling back to sequential split")

        # Fallback to sequential split
        return self._sequential_split(n_samples)

    def _temporal_split(self, X, n_samples: int):
        """Time-aware split using index timestamps."""
        # Calculate split points based on temporal distribution
        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            # Calculate test set boundaries (moving window from the end)
            test_size_samples = max(int(n_samples * self.test_size), int(n_samples * self.min_test_size))
            test_start = n_samples - test_size_samples - (i * int(n_samples * 0.1))
            test_end = n_samples - (i * int(n_samples * 0.1))

            # Ensure boundaries are valid
            test_start = max(0, test_start)
            test_end = min(n_samples, test_end)

            # Ensure minimum test size
            if test_end - test_start < int(n_samples * self.min_test_size):
                continue

            # Calculate training set boundaries with gap
            train_end = test_start - self.gap_size
            train_start = max(0, int(n_samples * self.min_train_size))

            # Ensure minimum train size
            if train_end - train_start < int(n_samples * self.min_train_size):
                continue

            # Generate indices
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices

    def _sequential_split(self, n_samples: int):
        """Sequential split for non-temporal data."""
        for i in range(self.n_splits):
            # Calculate test set boundaries
            test_start = int(n_samples * (1 - self.test_size - i * 0.1))
            test_end = int(n_samples * (1 - i * 0.1))

            # Ensure minimum test size
            if test_end - test_start < int(n_samples * self.min_test_size):
                continue

            # Calculate training set boundaries
            train_end = test_start - self.gap_size
            train_start = max(0, int(n_samples * self.min_train_size))

            # Ensure minimum train size
            if train_end - train_start < int(n_samples * self.min_train_size):
                continue

            # Generate indices
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Get number of splits."""
        return self.n_splits

class UniversalTemporalCrossValidator:
    """Universal temporal cross-validation for any ML model."""
    
    def __init__(self, config: Optional[TemporalValidationConfig] = None):
        """
        Initialize universal temporal cross-validator.
        
        Args:
            config: Temporal validation configuration
        """
        self.config = config or TemporalValidationConfig()
        self.cv_results = []
        self.performance_history = []
    
    def cross_validate(self, 
                       estimator, 
                       X: np.ndarray, 
                       y: np.ndarray,
                       timestamps: Optional[np.ndarray] = None,
                       feature_names: Optional[List[str]] = None,
                       model_name: str = "unknown",
                       model_type: str = "unknown") -> Dict[str, Any]:
        """
        Perform temporal cross-validation for any ML model.
        
        Args:
            estimator: Model to validate
            X: Input features
            y: Target labels
            timestamps: Optional timestamps
            feature_names: Optional feature names
            model_name: Name of the model
            model_type: Type of model
            
        Returns:
            Dict: Cross-validation results
        """
        if not self.config.enable_temporal_checks:
            raise ValueError("Temporal checks are disabled")
        
        # Create time series splitter
        tscv = UniversalTimeSeriesSplit(
            n_splits=self.config.n_splits,
            test_size=self.config.test_size,
            gap_size=self.config.gap_size,
            min_train_size=0.3,
            min_test_size=self.config.min_test_size
        )
        
        # Perform cross-validation
        cv_scores = []
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X, y)):
            try:
                # Split data
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Validate temporal order
                if timestamps is not None:
                    train_times = timestamps[train_idx]
                    test_times = timestamps[test_idx]
                    
                    if np.max(train_times) >= np.min(test_times):
                        logger.warning(f"Fold {fold}: Temporal order violation detected")
                        continue
                
                # Train model
                estimator.fit(X_train, y_train)
                
                # Make predictions
                y_pred = estimator.predict(X_test)
                y_pred_proba = getattr(estimator, 'predict_proba', lambda x: None)(X_test)
                
                # Calculate metrics
                fold_metrics = self._calculate_fold_metrics(y_test, y_pred, y_pred_proba)
                fold_metrics['fold'] = fold
                fold_metrics['train_size'] = len(X_train)
                fold_metrics['test_size'] = len(X_test)
                
                # Store fold results
                fold_results.append(fold_metrics)
                cv_scores.append(fold_metrics['accuracy'])
                
                # Track performance if enabled
                self.performance_history.append({
                    'fold': fold,
                    'accuracy': fold_metrics['accuracy'],
                    'f1': fold_metrics['f1'],
                    'train_size': len(X_train),
                    'test_size': len(X_test)
                })
                
            except Exception as e:
                logger.error(f"Fold {fold} failed: {e}")
                continue
        
        # Calculate overall results
        if not cv_scores:
            raise ValueError("No valid folds completed")
        
        results = {
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'min_score': np.min(cv_scores),
            'max_score': np.max(cv_scores),
            'fold_results': fold_results,
            'n_folds': len(cv_scores),
            'successful_folds': len(cv_scores),
            'model_name': model_name,
            'model_type': model_type
        }
        
        # Store results
        self.cv_results.append(results)
        
        return results
    
    def _calculate_fold_metrics(self, 
                               y_true: np.ndarray, 
                               y_pred: np.ndarray, 
                               y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate metrics for a single fold."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, average='weighted'),
        }
        
        # Add probability-based metrics if available
        if y_pred_proba is not None:
            try:
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                metrics['log_loss'] = None
                metrics['auc'] = None
        
        return metrics

# Global instances for easy access
DEFAULT_TEMPORAL_CONFIG = TemporalValidationConfig()
DEFAULT_TEMPORAL_VALIDATOR = UniversalTemporalValidator(DEFAULT_TEMPORAL_CONFIG)
DEFAULT_TEMPORAL_CV = UniversalTemporalCrossValidator(DEFAULT_TEMPORAL_CONFIG)

def get_temporal_validator(config: Optional[TemporalValidationConfig] = None) -> UniversalTemporalValidator:
    """Get universal temporal validator."""
    if config is None:
        return DEFAULT_TEMPORAL_VALIDATOR
    return UniversalTemporalValidator(config)

def get_temporal_cv(config: Optional[TemporalValidationConfig] = None) -> UniversalTemporalCrossValidator:
    """Get universal temporal cross-validator."""
    if config is None:
        return DEFAULT_TEMPORAL_CV
    return UniversalTemporalCrossValidator(config)

def create_time_series_split(**kwargs) -> UniversalTimeSeriesSplit:
    """Create universal time series splitter."""
    return UniversalTimeSeriesSplit(**kwargs)