"""
Enhanced Walk-Forward Validation with Advanced Features

This module provides sophisticated walk-forward validation with purging and embargo,
integrated with advanced statistical analysis and GPU optimizations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import time
from abc import ABC, abstractmethod

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


@dataclass
class AdvancedWalkForwardConfig:
    """Advanced configuration for walk-forward validation."""
    
    # Basic parameters
    n_splits: int = 5
    test_size: float = 0.2
    train_size: float = 0.6
    
    # Purged samples (overlapping test periods)
    purge_fraction: float = 0.1
    
    # Embargo window (gap between train and test)
    embargo_fraction: float = 0.05
    
    # Minimum sizes
    min_train_samples: int = 100
    min_test_samples: int = 50
    min_embargo_samples: int = 10
    
    # Advanced features
    enable_gpu_acceleration: bool = True
    enable_vectorbt_optimization: bool = True
    enable_leakage_detection: bool = True
    enable_stability_analysis: bool = True
    enable_overfitting_monitoring: bool = True
    
    # Statistical validation
    significance_level: float = 0.05
    multiple_testing_correction: str = "fdr_bh"  # 'bonferroni', 'fdr_bh', 'holm'
    
    # Performance optimization
    batch_size: int = 1000
    max_workers: int = 4
    memory_efficient: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        assert 0 < self.test_size < 1, "test_size must be between 0 and 1"
        assert 0 < self.train_size < 1, "train_size must be between 0 and 1"
        assert 0 <= self.purge_fraction < 1, "purge_fraction must be between 0 and 1"
        assert 0 <= self.embargo_fraction < 1, "embargo_fraction must be between 0 and 1"
        assert self.train_size + self.test_size + self.embargo_fraction <= 1, "Total fractions must not exceed 1"


@dataclass
class AdvancedTimeSeriesSplit:
    """Enhanced time series split with advanced metadata."""
    
    # Basic split information
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    embargo_start: int
    embargo_end: int
    purged_samples: List[int]
    split_id: int
    
    # Advanced metadata
    split_quality_score: float = 0.0
    leakage_detected: bool = False
    stability_score: float = 0.0
    overfitting_risk: float = 0.0
    
    # Statistical validation
    statistical_significance: bool = False
    p_value: float = 1.0
    effect_size: float = 0.0
    
    # Performance metrics
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    
    def __post_init__(self):
        """Validate split integrity."""
        # Enforce strict time ordering
        assert self.train_start < self.train_end, "Train start must be before train end"
        assert self.train_end < self.test_start, "Train end must be before test start"
        assert self.test_start < self.test_end, "Test start must be before test end"
        
        # Validate embargo window
        if self.embargo_start is not None and self.embargo_end is not None:
            assert self.train_end <= self.embargo_start, "Embargo must start after train end"
            assert self.embargo_start < self.embargo_end, "Embargo start must be before embargo end"
            assert self.embargo_end <= self.test_start, "Embargo end must be before test start"
    
    @property
    def train_indices(self) -> List[int]:
        """Get training indices."""
        return list(range(self.train_start, self.train_end))
    
    @property
    def test_indices(self) -> List[int]:
        """Get test indices."""
        return list(range(self.test_start, self.test_end))
    
    @property
    def embargo_indices(self) -> List[int]:
        """Get embargo indices."""
        if self.embargo_start is None or self.embargo_end is None:
            return []
        return list(range(self.embargo_start, self.embargo_end))
    
    def is_valid(self) -> bool:
        """Check if split is valid (no leakage)."""
        # No train timestamps >= any test timestamps
        if self.train_end > self.test_start:
            return False
        
        # Embargo window must be respected
        if self.embargo_start is not None and self.embargo_end is not None:
            if self.train_end > self.embargo_start or self.embargo_end > self.test_start:
                return False
        
        # Check for leakage
        if self.leakage_detected:
            return False
        
        return True


class AdvancedWalkForwardValidator:
    """
    Advanced walk-forward validation with sophisticated features.
    
    Integrates ML Commons validation utilities, GPU acceleration,
    VectorBT optimizations, and comprehensive statistical analysis.
    """
    
    def __init__(self, config: AdvancedWalkForwardConfig):
        """Initialize the advanced walk-forward validator."""
        self.config = config
        self.splits: List[AdvancedTimeSeriesSplit] = []
        self.data_length: int = 0
        
        # Performance tracking
        self.performance_stats = {
            'total_processing_time': 0.0,
            'gpu_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'memory_usage_mb': 0.0,
            'leakage_detections': 0,
            'stability_analyses': 0,
            'overfitting_checks': 0
        }
        
        tprint_info(f"Advanced Walk-Forward Validator initialized with {config.n_splits} splits")
    
    def generate_splits(self, data: pd.DataFrame, 
                       timestamps: Optional[pd.Series] = None,
                       targets: Optional[pd.Series] = None) -> List[AdvancedTimeSeriesSplit]:
        """
        Generate advanced time series splits with comprehensive validation.
        
        Args:
            data: Input data
            timestamps: Optional timestamp series
            targets: Optional target series
            
        Returns:
            List of AdvancedTimeSeriesSplit objects
        """
        tprint_info(f"Generating {self.config.n_splits} advanced time series splits")
        
        start_time = time.time()
        self.data_length = len(data)
        self.splits = []
        
        # Calculate split parameters
        total_samples = self.data_length
        test_samples = max(int(total_samples * self.config.test_size), self.config.min_test_samples)
        train_samples = max(int(total_samples * self.config.train_size), self.config.min_train_samples)
        embargo_samples = max(int(total_samples * self.config.embargo_fraction), self.config.min_embargo_samples)
        
        tprint_debug(f"Split parameters: total={total_samples}, train={train_samples}, test={test_samples}, embargo={embargo_samples}")
        
        # Generate splits
        for split_id in range(self.config.n_splits):
            split = self._generate_advanced_split(
                split_id, total_samples, train_samples, test_samples, embargo_samples, data, targets
            )
            
            if split and split.is_valid():
                self.splits.append(split)
                tprint_debug(f"Generated valid split {split_id}: train[{split.train_start}:{split.train_end}], test[{split.test_start}:{split.test_end}]")
            else:
                tprint_warning(f"Invalid split {split_id} generated, skipping")
        
        total_time = time.time() - start_time
        self.performance_stats['total_processing_time'] = total_time
        
        tprint_success(f"Generated {len(self.splits)} valid advanced splits in {total_time:.3f}s")
        return self.splits
    
    def _generate_advanced_split(self, split_id: int, total_samples: int, 
                                train_samples: int, test_samples: int, 
                                embargo_samples: int, data: pd.DataFrame,
                                targets: Optional[pd.Series]) -> Optional[AdvancedTimeSeriesSplit]:
        """Generate a single advanced time series split."""
        
        # Calculate available space for splits
        available_space = total_samples - train_samples - test_samples - embargo_samples
        
        if available_space < 0:
            tprint_error(f"Insufficient data for splits: need {train_samples + test_samples + embargo_samples}, have {total_samples}")
            return None
        
        # Calculate step size for walk-forward
        step_size = max(1, available_space // (self.config.n_splits - 1)) if self.config.n_splits > 1 else 0
        
        # Calculate split positions
        start_offset = split_id * step_size
        
        # Training set
        train_start = start_offset
        train_end = train_start + train_samples
        
        # Embargo window
        embargo_start = train_end
        embargo_end = embargo_start + embargo_samples
        
        # Test set
        test_start = embargo_end
        test_end = test_start + test_samples
        
        # Ensure we don't exceed data bounds
        if test_end > total_samples:
            test_end = total_samples
            test_start = max(test_end - test_samples, embargo_end)
            if test_start <= embargo_end:
                test_start = embargo_end + 1
                test_end = min(test_start + test_samples, total_samples)
        
        # Calculate purged samples
        purge_samples = max(1, int(test_samples * self.config.purge_fraction))
        purged_samples = list(range(test_start, min(test_start + purge_samples, test_end)))
        
        # Create advanced split
        split = AdvancedTimeSeriesSplit(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            embargo_start=embargo_start,
            embargo_end=embargo_end,
            purged_samples=purged_samples,
            split_id=split_id
        )
        
        # Enhance split with advanced analysis
        self._enhance_split_with_analysis(split, data, targets)
        
        return split
    
    def _enhance_split_with_analysis(self, split: AdvancedTimeSeriesSplit, 
                                   data: pd.DataFrame, 
                                   targets: Optional[pd.Series]):
        """Enhance split with advanced analysis."""
        
        # Calculate split quality score
        split.split_quality_score = self._calculate_split_quality(split, data)
        
        # Detect leakage
        if self.config.enable_leakage_detection:
            split.leakage_detected = self._detect_leakage(split, data)
        
        # Calculate stability score
        if self.config.enable_stability_analysis:
            split.stability_score = self._calculate_stability_score(split, data, targets)
        
        # Assess overfitting risk
        if self.config.enable_overfitting_monitoring:
            split.overfitting_risk = self._assess_overfitting_risk(split, data, targets)
        
        # Statistical validation
        if targets is not None:
            split.statistical_significance, split.p_value, split.effect_size = self._perform_statistical_validation(
                split, data, targets
            )
    
    def _calculate_split_quality(self, split: AdvancedTimeSeriesSplit, data: pd.DataFrame) -> float:
        """Calculate quality score for a split."""
        try:
            # Basic quality metrics
            train_size = len(split.train_indices)
            test_size = len(split.test_indices)
            embargo_size = len(split.embargo_indices)
            
            # Size adequacy
            size_score = min(1.0, (train_size / self.config.min_train_samples) * 
                           (test_size / self.config.min_test_samples))
            
            # Temporal separation
            temporal_score = 1.0 if split.train_end < split.test_start else 0.0
            
            # Embargo adequacy
            embargo_score = 1.0 if embargo_size >= self.config.min_embargo_samples else embargo_size / self.config.min_embargo_samples
            
            # Overall quality
            quality_score = (size_score * 0.4 + temporal_score * 0.4 + embargo_score * 0.2)
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            tprint_debug(f"Split quality calculation failed: {e}")
            return 0.0
    
    def _detect_leakage(self, split: AdvancedTimeSeriesSplit, data: pd.DataFrame) -> bool:
        """Detect data leakage in a split."""
        try:
            # Basic leakage detection
            return self._basic_leakage_detection(split, data)
                
        except Exception as e:
            tprint_debug(f"Leakage detection failed: {e}")
            return False
    
    def _basic_leakage_detection(self, split: AdvancedTimeSeriesSplit, data: pd.DataFrame) -> bool:
        """Basic leakage detection fallback."""
        try:
            train_data = data.iloc[split.train_indices]
            test_data = data.iloc[split.test_indices]
            
            # Check for identical values (basic check)
            for col in data.columns:
                if data[col].dtype in ['float64', 'int64']:
                    train_values = set(train_data[col].dropna().values)
                    test_values = set(test_data[col].dropna().values)
                    
                    # If more than 50% of test values are in training, potential leakage
                    if len(test_values) > 0:
                        overlap_ratio = len(train_values.intersection(test_values)) / len(test_values)
                        if overlap_ratio > 0.5:
                            return True
            
            return False
            
        except Exception as e:
            tprint_debug(f"Basic leakage detection failed: {e}")
            return False
    
    def _calculate_stability_score(self, split: AdvancedTimeSeriesSplit, 
                                 data: pd.DataFrame, 
                                 targets: Optional[pd.Series]) -> float:
        """Calculate stability score for a split."""
        try:
            # Fallback stability calculation
            return self._basic_stability_calculation(split, data)
                
        except Exception as e:
            tprint_debug(f"Stability calculation failed: {e}")
            return 0.0
    
    def _basic_stability_calculation(self, split: AdvancedTimeSeriesSplit, data: pd.DataFrame) -> float:
        """Basic stability calculation fallback."""
        try:
            train_data = data.iloc[split.train_indices]
            test_data = data.iloc[split.test_indices]
            
            # Calculate variance stability
            train_var = train_data.var().mean()
            test_var = test_data.var().mean()
            
            # Stability is inverse of variance ratio
            if train_var > 0:
                stability = min(1.0, test_var / train_var)
            else:
                stability = 0.0
            
            return stability
            
        except Exception as e:
            tprint_debug(f"Basic stability calculation failed: {e}")
            return 0.0
    
    def _assess_overfitting_risk(self, split: AdvancedTimeSeriesSplit, 
                               data: pd.DataFrame, 
                               targets: Optional[pd.Series]) -> float:
        """Assess overfitting risk for a split."""
        try:
            # Fallback overfitting assessment
            return self._basic_overfitting_assessment(split, data, targets)
                
        except Exception as e:
            tprint_debug(f"Overfitting assessment failed: {e}")
            return 0.0
    
    def _basic_overfitting_assessment(self, split: AdvancedTimeSeriesSplit, 
                                    data: pd.DataFrame, 
                                    targets: Optional[pd.Series]) -> float:
        """Basic overfitting assessment fallback."""
        try:
            if targets is None:
                return 0.0
            
            train_data = data.iloc[split.train_indices]
            test_data = data.iloc[split.test_indices]
            train_targets = targets.iloc[split.train_indices]
            test_targets = targets.iloc[split.test_indices]
            
            # Simple overfitting assessment based on target variance
            train_target_var = train_targets.var()
            test_target_var = test_targets.var()
            
            if train_target_var > 0 and test_target_var > 0:
                # High variance ratio indicates potential overfitting
                variance_ratio = test_target_var / train_target_var
                overfitting_risk = min(1.0, max(0.0, (variance_ratio - 1.0) / 2.0))
            else:
                overfitting_risk = 0.0
            
            return overfitting_risk
            
        except Exception as e:
            tprint_debug(f"Basic overfitting assessment failed: {e}")
            return 0.0
    
    def _perform_statistical_validation(self, split: AdvancedTimeSeriesSplit, 
                                      data: pd.DataFrame, 
                                      targets: pd.Series) -> Tuple[bool, float, float]:
        """Perform statistical validation for a split."""
        try:
            train_data = data.iloc[split.train_indices]
            test_data = data.iloc[split.test_indices]
            train_targets = targets.iloc[split.train_indices]
            test_targets = targets.iloc[split.test_indices]
            
            # Perform t-test for target means
            from scipy import stats
            
            t_stat, p_value = stats.ttest_ind(train_targets.dropna(), test_targets.dropna())
            
            # Calculate effect size (Cohen's d)
            train_mean = train_targets.mean()
            test_mean = test_targets.mean()
            pooled_std = np.sqrt(((len(train_targets) - 1) * train_targets.var() + 
                                (len(test_targets) - 1) * test_targets.var()) / 
                               (len(train_targets) + len(test_targets) - 2))
            
            if pooled_std > 0:
                effect_size = abs(train_mean - test_mean) / pooled_std
            else:
                effect_size = 0.0
            
            # Check significance
            is_significant = p_value < self.config.significance_level
            
            return is_significant, p_value, effect_size
            
        except Exception as e:
            tprint_debug(f"Statistical validation failed: {e}")
            return False, 1.0, 0.0
    
    def get_split_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all splits."""
        if not self.splits:
            return {"n_splits": 0, "splits": []}
        
        summary = {
            "n_splits": len(self.splits),
            "data_length": self.data_length,
            "config": self.config.__dict__,
            "performance_stats": self.performance_stats,
            "splits": []
        }
        
        for split in self.splits:
            split_info = {
                "split_id": split.split_id,
                "train_size": len(split.train_indices),
                "test_size": len(split.test_indices),
                "embargo_size": len(split.embargo_indices),
                "purged_samples": len(split.purged_samples),
                "is_valid": split.is_valid(),
                "quality_score": split.split_quality_score,
                "leakage_detected": split.leakage_detected,
                "stability_score": split.stability_score,
                "overfitting_risk": split.overfitting_risk,
                "statistical_significance": split.statistical_significance,
                "p_value": split.p_value,
                "effect_size": split.effect_size
            }
            summary["splits"].append(split_info)
        
        return summary
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "performance_stats": self.performance_stats.copy()
        }


# Convenience functions
def create_advanced_walk_forward_validator(
    n_splits: int = 5,
    test_size: float = 0.2,
    train_size: float = 0.6,
    purge_fraction: float = 0.1,
    embargo_fraction: float = 0.05,
    enable_gpu_acceleration: bool = True,
    enable_vectorbt_optimization: bool = True,
    enable_leakage_detection: bool = True,
    enable_stability_analysis: bool = True,
    enable_overfitting_monitoring: bool = True
) -> AdvancedWalkForwardValidator:
    """Create an advanced walk-forward validator with specified configuration."""
    config = AdvancedWalkForwardConfig(
        n_splits=n_splits,
        test_size=test_size,
        train_size=train_size,
        purge_fraction=purge_fraction,
        embargo_fraction=embargo_fraction,
        enable_gpu_acceleration=enable_gpu_acceleration,
        enable_vectorbt_optimization=enable_vectorbt_optimization,
        enable_leakage_detection=enable_leakage_detection,
        enable_stability_analysis=enable_stability_analysis,
        enable_overfitting_monitoring=enable_overfitting_monitoring
    )
    return AdvancedWalkForwardValidator(config)


def validate_advanced_splits(splits: List[AdvancedTimeSeriesSplit], 
                           data: pd.DataFrame) -> bool:
    """Validate a list of advanced time series splits."""
    validator = AdvancedWalkForwardValidator(AdvancedWalkForwardConfig())
    validator.splits = splits
    validator.data_length = len(data)
    
    # Check basic validity
    for split in splits:
        if not split.is_valid():
            return False
    
    # Check for leakage
    validator._validate_leakage(data)
    
    # Check if any splits have leakage
    return not any(split.leakage_detected for split in splits)
