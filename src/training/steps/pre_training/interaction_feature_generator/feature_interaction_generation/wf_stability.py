"""
Stage 2: Walk-Forward Stability Testing with Purged Cross-Validation

This module implements the second stage of the lookback optimization system,
testing the stability of optimal lookback choices across time using purged
walk-forward validation to prevent data leakage.
"""

import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Import configuration and Stage 1 results
from .config import LookbackOptimizationConfig, FamilyType, CrossValidationConfig
from .ic_surface import ICSurfaceResult, ICSurfaceEstimator

# Import utilities
try:
    from src.utils.tprint import tprint, tprint_info, tprint_error, tprint_warning
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Result for a single CV fold."""
    fold_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    fold_optimal_lookback: float
    fold_optimal_ic: float
    global_optimal_lookback: float
    global_optimal_ic: float
    ic_penalty: float
    lookback_difference: float
    match_within_tolerance: bool
    execution_time: float = 0.0


@dataclass
class StabilityResult:
    """Result of walk-forward stability testing."""
    family: FamilyType
    global_optimal_lookback: float
    global_optimal_ic: float
    fold_results: List[FoldResult]
    match_rate: float
    average_ic_penalty: float
    average_lookback_difference: float
    stability_score: float
    recommendation: str  # "stable", "unstable", "blend_recommended"
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'family': self.family.value,
            'global_optimal_lookback': self.global_optimal_lookback,
            'global_optimal_ic': self.global_optimal_ic,
            'fold_results': [fold.to_dict() for fold in self.fold_results],
            'match_rate': self.match_rate,
            'average_ic_penalty': self.average_ic_penalty,
            'average_lookback_difference': self.average_lookback_difference,
            'stability_score': self.stability_score,
            'recommendation': self.recommendation,
            'execution_time': self.execution_time
        }


class PurgedTimeSeriesSplit:
    """Purged time series cross-validation to prevent data leakage."""
    
    def __init__(self, config: CrossValidationConfig):
        self.config = config
        self.n_folds = config.n_folds
        self.purging_period = config.purging_period
        self.embargo_period = config.embargo_period
        self.min_train_size = config.min_train_size
        self.min_test_size = config.min_test_size
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
              groups: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate purged time series splits."""
        n_samples = len(X)
        
        if n_samples < self.min_train_size + self.min_test_size:
            raise ValueError(f"Insufficient data: {n_samples} < {self.min_train_size + self.min_test_size}")
        
        # Calculate fold sizes
        available_size = n_samples - self.min_train_size - self.min_test_size
        fold_size = available_size // self.n_folds
        
        if fold_size < self.min_test_size:
            # Reduce number of folds if necessary
            self.n_folds = max(1, available_size // self.min_test_size)
            fold_size = available_size // self.n_folds
        
        splits = []
        
        for i in range(self.n_folds):
            # Calculate test period
            test_start = self.min_train_size + i * fold_size
            test_end = min(test_start + fold_size, n_samples)
            
            # Calculate train period (before test, with purging)
            train_end = test_start - self.purging_period
            train_start = max(0, train_end - self.min_train_size)
            
            # Apply embargo after test period
            embargo_end = min(test_end + self.embargo_period, n_samples)
            
            # Ensure we have enough data
            if train_end - train_start < self.min_train_size:
                continue
            if test_end - test_start < self.min_test_size:
                continue
            
            # Create indices
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
        
        return splits


class StabilityTester:
    """Main class for walk-forward stability testing."""
    
    def __init__(self, config: LookbackOptimizationConfig):
        self.config = config
        self.cv_splitter = PurgedTimeSeriesSplit(config.cv)
        self.ic_estimator = ICSurfaceEstimator(config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def test_stability(self, data: pd.DataFrame, target: np.ndarray, 
                      ic_surface_result: ICSurfaceResult, 
                      feature_name: str) -> StabilityResult:
        """Test stability of optimal lookback choice across time."""
        start_time = time.time()
        
        try:
            tprint_info(f"Testing stability for {ic_surface_result.family.value} family...")
            
            family = ic_surface_result.family
            global_optimal_lookback = ic_surface_result.optimal_lookback
            global_optimal_ic = ic_surface_result.optimal_ic
            
            # Generate purged CV splits
            X = np.arange(len(data))  # Dummy X for splitting
            splits = self.cv_splitter.split(X, target)
            
            if len(splits) < 2:
                raise ValueError(f"Insufficient splits for stability testing: {len(splits)}")
            
            fold_results = []
            ic_penalties = []
            lookback_differences = []
            matches_within_tolerance = []
            
            for fold_idx, (train_indices, test_indices) in enumerate(splits):
                try:
                    fold_start_time = time.time()
                    
                    # Get training data for this fold
                    train_data = data.iloc[train_indices].copy()
                    train_target = target[train_indices]
                    
                    # Get test data for this fold
                    test_data = data.iloc[test_indices].copy()
                    test_target = target[test_indices]
                    
                    # Estimate optimal lookback on training data
                    fold_ic_result = self.ic_estimator.estimate_surface(
                        train_data, train_target, family, feature_name
                    )
                    
                    fold_optimal_lookback = fold_ic_result.optimal_lookback
                    fold_optimal_ic = fold_ic_result.optimal_ic
                    
                    # Evaluate on test data
                    test_ic = self._evaluate_lookback_on_test(
                        test_data, test_target, family, feature_name, 
                        fold_optimal_lookback
                    )
                    global_test_ic = self._evaluate_lookback_on_test(
                        test_data, test_target, family, feature_name,
                        global_optimal_lookback
                    )
                    
                    # Calculate metrics
                    ic_penalty = global_test_ic - test_ic
                    lookback_difference = abs(fold_optimal_lookback - global_optimal_lookback)
                    
                    # Check if within tolerance (within 1 bar or 20% relative difference)
                    tolerance = max(1.0, 0.2 * global_optimal_lookback)
                    match_within_tolerance = lookback_difference <= tolerance
                    
                    fold_execution_time = time.time() - fold_start_time
                    
                    fold_result = FoldResult(
                        fold_idx=fold_idx,
                        train_start=train_indices[0],
                        train_end=train_indices[-1],
                        test_start=test_indices[0],
                        test_end=test_indices[-1],
                        fold_optimal_lookback=fold_optimal_lookback,
                        fold_optimal_ic=fold_optimal_ic,
                        global_optimal_lookback=global_optimal_lookback,
                        global_optimal_ic=global_optimal_ic,
                        ic_penalty=ic_penalty,
                        lookback_difference=lookback_difference,
                        match_within_tolerance=match_within_tolerance,
                        execution_time=fold_execution_time
                    )
                    
                    fold_results.append(fold_result)
                    ic_penalties.append(ic_penalty)
                    lookback_differences.append(lookback_difference)
                    matches_within_tolerance.append(match_within_tolerance)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process fold {fold_idx}: {e}")
                    continue
            
            if not fold_results:
                raise ValueError("No valid folds processed")
            
            # Calculate aggregate metrics
            match_rate = np.mean(matches_within_tolerance)
            average_ic_penalty = np.mean(ic_penalties)
            average_lookback_difference = np.mean(lookback_differences)
            
            # Calculate stability score (higher is better)
            stability_score = self._calculate_stability_score(
                match_rate, average_ic_penalty, average_lookback_difference
            )
            
            # Make recommendation
            recommendation = self._make_recommendation(
                match_rate, average_ic_penalty, stability_score
            )
            
            execution_time = time.time() - start_time
            
            result = StabilityResult(
                family=family,
                global_optimal_lookback=global_optimal_lookback,
                global_optimal_ic=global_optimal_ic,
                fold_results=fold_results,
                match_rate=match_rate,
                average_ic_penalty=average_ic_penalty,
                average_lookback_difference=average_lookback_difference,
                stability_score=stability_score,
                recommendation=recommendation,
                execution_time=execution_time
            )
            
            tprint_info(f"Stability testing completed in {execution_time:.3f}s")
            tprint_info(f"Match rate: {match_rate:.3f}, IC penalty: {average_ic_penalty:.4f}")
            tprint_info(f"Stability score: {stability_score:.3f}, Recommendation: {recommendation}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Stability testing failed: {e}")
            self.logger.error(f"Error details: {traceback.format_exc()}")
            
            # Return empty result
            return StabilityResult(
                family=ic_surface_result.family,
                global_optimal_lookback=ic_surface_result.optimal_lookback,
                global_optimal_ic=ic_surface_result.optimal_ic,
                fold_results=[],
                match_rate=0.0,
                average_ic_penalty=1.0,
                average_lookback_difference=float('inf'),
                stability_score=0.0,
                recommendation="unstable",
                execution_time=execution_time
            )
    
    def _evaluate_lookback_on_test(self, test_data: pd.DataFrame, test_target: np.ndarray,
                                  family: FamilyType, feature_name: str, 
                                  lookback: float) -> float:
        """Evaluate a specific lookback on test data."""
        try:
            # Round lookback to nearest integer
            lookback_int = int(round(lookback))
            
            # Generate feature for this lookback
            feature_values = self.ic_estimator._generate_feature(
                test_data, family, feature_name, lookback_int
            )
            
            # Remove NaN values
            valid_mask = np.isfinite(feature_values) & np.isfinite(test_target)
            if np.sum(valid_mask) < 5:
                return 0.0
            
            feature_clean = feature_values[valid_mask]
            target_clean = test_target[valid_mask]
            
            # Compute IC
            ic = np.corrcoef(feature_clean, target_clean)[0, 1]
            
            return float(ic) if not np.isnan(ic) else 0.0
            
        except Exception as e:
            self.logger.warning(f"Failed to evaluate lookback {lookback} on test data: {e}")
            return 0.0
    
    def _calculate_stability_score(self, match_rate: float, ic_penalty: float, 
                                  lookback_difference: float) -> float:
        """Calculate overall stability score."""
        # Normalize metrics to [0, 1] range
        match_score = match_rate  # Already in [0, 1]
        
        # IC penalty score (lower penalty is better)
        ic_score = max(0.0, 1.0 - abs(ic_penalty) / 0.5)  # Penalty of 0.5 = score of 0
        
        # Lookback difference score (smaller difference is better)
        lookback_score = max(0.0, 1.0 - lookback_difference / 10.0)  # Difference of 10 = score of 0
        
        # Weighted combination
        stability_score = (0.5 * match_score + 
                          0.3 * ic_score + 
                          0.2 * lookback_score)
        
        return stability_score
    
    def _make_recommendation(self, match_rate: float, ic_penalty: float, 
                           stability_score: float) -> str:
        """Make recommendation based on stability metrics."""
        if stability_score >= 0.8:
            return "stable"
        elif match_rate >= self.config.hysteresis.min_fold_match_rate:
            return "stable"
        elif abs(ic_penalty) <= 0.1:
            return "stable"
        elif stability_score >= 0.5:
            return "blend_recommended"
        else:
            return "unstable"


class MultiFamilyStabilityTester:
    """Test stability across multiple feature families."""
    
    def __init__(self, config: LookbackOptimizationConfig):
        self.config = config
        self.stability_tester = StabilityTester(config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def test_all_families(self, data: pd.DataFrame, target: np.ndarray,
                         ic_surface_results: Dict[FamilyType, ICSurfaceResult],
                         feature_names: Dict[FamilyType, str]) -> Dict[FamilyType, StabilityResult]:
        """Test stability for all feature families."""
        results = {}
        
        for family, ic_result in ic_surface_results.items():
            try:
                feature_name = feature_names.get(family, f"{family.value}_feature")
                
                tprint_info(f"Testing stability for {family.value} family...")
                
                stability_result = self.stability_tester.test_stability(
                    data, target, ic_result, feature_name
                )
                
                results[family] = stability_result
                
            except Exception as e:
                self.logger.error(f"Failed to test stability for {family.value}: {e}")
                continue
        
        return results
    
    def generate_stability_report(self, stability_results: Dict[FamilyType, StabilityResult]) -> Dict[str, Any]:
        """Generate comprehensive stability report."""
        report = {
            'summary': {
                'total_families': len(stability_results),
                'stable_families': 0,
                'blend_recommended_families': 0,
                'unstable_families': 0,
                'average_stability_score': 0.0,
                'average_match_rate': 0.0
            },
            'family_details': {},
            'recommendations': []
        }
        
        stability_scores = []
        match_rates = []
        
        for family, result in stability_results.items():
            # Update summary counts
            if result.recommendation == "stable":
                report['summary']['stable_families'] += 1
            elif result.recommendation == "blend_recommended":
                report['summary']['blend_recommended_families'] += 1
            else:
                report['summary']['unstable_families'] += 1
            
            stability_scores.append(result.stability_score)
            match_rates.append(result.match_rate)
            
            # Store family details
            report['family_details'][family.value] = {
                'stability_score': result.stability_score,
                'match_rate': result.match_rate,
                'average_ic_penalty': result.average_ic_penalty,
                'average_lookback_difference': result.average_lookback_difference,
                'recommendation': result.recommendation,
                'global_optimal_lookback': result.global_optimal_lookback,
                'global_optimal_ic': result.global_optimal_ic
            }
            
            # Generate recommendations
            if result.recommendation == "unstable":
                report['recommendations'].append(
                    f"{family.value}: Consider using default lookback or blend approach"
                )
            elif result.recommendation == "blend_recommended":
                report['recommendations'].append(
                    f"{family.value}: Use 2-3 window blend for robustness"
                )
        
        # Calculate averages
        if stability_scores:
            report['summary']['average_stability_score'] = np.mean(stability_scores)
            report['summary']['average_match_rate'] = np.mean(match_rates)
        
        return report