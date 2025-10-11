"""
VectorBT-Enhanced Bootstrap Validation for Feature Lookback Optimization.

This module provides efficient bootstrap validation using VectorBT's portfolio
analysis capabilities, enabling parallel processing of multiple lookback periods
with comprehensive financial metrics.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import random

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.portfolio.base import Portfolio
    from vectorbt.records.base import Records
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    Portfolio = None
    Records = None

from src.utils.tprint import tprint, tprint_error, tprint_success, tprint_warning, tprint_debug, tprint_info
from src.utils.logger import get_logger
from .utils.error_handling import safe_operation, get_error_handler
from .vectorbt_scoring import VectorBTScoringSystem, ScoringMethod, VectorBTScoringConfig

logger = get_logger('VectorBTBootstrap')


class BootstrapMethod(Enum):
    """Available bootstrap methods."""
    SIMPLE = "simple"
    BLOCK = "block"
    STATIONARY = "stationary"
    WILDCARD = "wildcard"


@dataclass
class VectorBTBootstrapConfig:
    """Configuration for VectorBT bootstrap validation."""
    n_bootstrap_samples: int = 100
    bootstrap_method: BootstrapMethod = BootstrapMethod.BLOCK
    block_size: int = 20
    train_ratio: float = 0.7
    validation_ratio: float = 0.2
    test_ratio: float = 0.1
    min_samples: int = 50
    max_samples: int = 1000
    random_seed: int = 42
    parallel_processing: bool = True
    max_workers: int = 4
    scoring_method: ScoringMethod = ScoringMethod.COMPOSITE
    confidence_level: float = 0.95
    use_vectorbt_portfolio: bool = True
    initial_capital: float = 100000.0
    fees: float = 0.001


@dataclass
class BootstrapResult:
    """Result from bootstrap validation."""
    mean_score: float
    std_score: float
    confidence_interval: Tuple[float, float]
    scores: List[float]
    execution_time: float
    n_samples: int
    is_valid: bool = True
    error_message: Optional[str] = None


class VectorBTBootstrapValidator:
    """
    High-performance bootstrap validator using VectorBT portfolio analysis.
    
    This class provides efficient bootstrap validation for feature lookback
    optimization using VectorBT's portfolio management capabilities.
    """
    
    def __init__(self, config: Optional[VectorBTBootstrapConfig] = None):
        """Initialize VectorBT bootstrap validator."""
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required but not available. Install with: pip install vectorbt")
        
        self.config = config or VectorBTBootstrapConfig()
        self.logger = get_logger('VectorBTBootstrapValidator')
        self.error_handler = get_error_handler()
        
        # Initialize scoring system
        scoring_config = VectorBTScoringConfig(
            initial_capital=self.config.initial_capital,
            fees=self.config.fees,
            scoring_method=self.config.scoring_method
        )
        self.scoring_system = VectorBTScoringSystem(scoring_config)
        
        # Set random seed
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
        tprint_success("✅ VectorBT Bootstrap Validator initialized")
    
    @safe_operation
    def validate_lookback_period(
        self,
        feature_values: np.ndarray,
        target_values: np.ndarray,
        lookback_period: int,
        n_samples: Optional[int] = None
    ) -> BootstrapResult:
        """
        Validate a single lookback period using bootstrap sampling.
        
        Args:
            feature_values: Feature values array
            target_values: Target values array
            lookback_period: Lookback period to validate
            n_samples: Number of bootstrap samples (uses config default if None)
            
        Returns:
            BootstrapResult with validation statistics
        """
        start_time = time.time()
        n_samples = n_samples or self.config.n_bootstrap_samples
        
        try:
            # Validate inputs
            if not self._validate_inputs(feature_values, target_values):
                return BootstrapResult(
                    mean_score=0.0,
                    std_score=0.0,
                    confidence_interval=(0.0, 0.0),
                    scores=[],
                    execution_time=time.time() - start_time,
                    n_samples=0,
                    is_valid=False,
                    error_message="Invalid inputs"
                )
            
            # Generate bootstrap samples
            bootstrap_samples = self._generate_bootstrap_samples(
                feature_values, target_values, n_samples
            )
            
            if not bootstrap_samples:
                return BootstrapResult(
                    mean_score=0.0,
                    std_score=0.0,
                    confidence_interval=(0.0, 0.0),
                    scores=[],
                    execution_time=time.time() - start_time,
                    n_samples=0,
                    is_valid=False,
                    error_message="Could not generate bootstrap samples"
                )
            
            # Score bootstrap samples
            scores = self._score_bootstrap_samples(bootstrap_samples, lookback_period)
            
            if not scores:
                return BootstrapResult(
                    mean_score=0.0,
                    std_score=0.0,
                    confidence_interval=(0.0, 0.0),
                    scores=[],
                    execution_time=time.time() - start_time,
                    n_samples=0,
                    is_valid=False,
                    error_message="Could not score bootstrap samples"
                )
            
            # Calculate statistics
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            confidence_interval = self._calculate_confidence_interval(scores)
            
            execution_time = time.time() - start_time
            
            return BootstrapResult(
                mean_score=mean_score,
                std_score=std_score,
                confidence_interval=confidence_interval,
                scores=scores,
                execution_time=execution_time,
                n_samples=len(scores),
                is_valid=True
            )
            
        except Exception as e:
            self.logger.error(f"Bootstrap validation failed: {e}")
            return BootstrapResult(
                mean_score=0.0,
                std_score=0.0,
                confidence_interval=(0.0, 0.0),
                scores=[],
                execution_time=time.time() - start_time,
                n_samples=0,
                is_valid=False,
                error_message=str(e)
            )
    
    def _validate_inputs(self, feature_values: np.ndarray, target_values: np.ndarray) -> bool:
        """Validate input arrays for bootstrap validation."""
        if feature_values is None or target_values is None:
            return False
        
        if len(feature_values) == 0 or len(target_values) == 0:
            return False
        
        if len(feature_values) != len(target_values):
            return False
        
        min_length = min(len(feature_values), len(target_values))
        if min_length < self.config.min_samples:
            return False
        
        return True
    
    def _generate_bootstrap_samples(
        self,
        feature_values: np.ndarray,
        target_values: np.ndarray,
        n_samples: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate bootstrap samples based on configured method."""
        try:
            if self.config.bootstrap_method == BootstrapMethod.SIMPLE:
                return self._generate_simple_bootstrap(feature_values, target_values, n_samples)
            elif self.config.bootstrap_method == BootstrapMethod.BLOCK:
                return self._generate_block_bootstrap(feature_values, target_values, n_samples)
            elif self.config.bootstrap_method == BootstrapMethod.STATIONARY:
                return self._generate_stationary_bootstrap(feature_values, target_values, n_samples)
            elif self.config.bootstrap_method == BootstrapMethod.WILDCARD:
                return self._generate_wildcard_bootstrap(feature_values, target_values, n_samples)
            else:
                return self._generate_simple_bootstrap(feature_values, target_values, n_samples)
                
        except Exception as e:
            self.logger.warning(f"Bootstrap sample generation failed: {e}")
            return []
    
    def _generate_simple_bootstrap(
        self,
        feature_values: np.ndarray,
        target_values: np.ndarray,
        n_samples: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate simple bootstrap samples with replacement."""
        samples = []
        n = len(feature_values)
        
        for _ in range(n_samples):
            # Random sampling with replacement
            indices = np.random.choice(n, size=n, replace=True)
            sample_features = feature_values[indices]
            sample_targets = target_values[indices]
            samples.append((sample_features, sample_targets))
        
        return samples
    
    def _generate_block_bootstrap(
        self,
        feature_values: np.ndarray,
        target_values: np.ndarray,
        n_samples: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate block bootstrap samples to preserve time series structure."""
        samples = []
        n = len(feature_values)
        block_size = self.config.block_size
        
        # Calculate number of blocks
        n_blocks = n // block_size
        if n_blocks == 0:
            return self._generate_simple_bootstrap(feature_values, target_values, n_samples)
        
        for _ in range(n_samples):
            # Sample blocks with replacement
            block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
            
            sample_features = []
            sample_targets = []
            
            for block_idx in block_indices:
                start_idx = block_idx * block_size
                end_idx = min(start_idx + block_size, n)
                
                sample_features.extend(feature_values[start_idx:end_idx])
                sample_targets.extend(target_values[start_idx:end_idx])
            
            # Truncate to original length
            sample_features = np.array(sample_features[:n])
            sample_targets = np.array(sample_targets[:n])
            
            samples.append((sample_features, sample_targets))
        
        return samples
    
    def _generate_stationary_bootstrap(
        self,
        feature_values: np.ndarray,
        target_values: np.ndarray,
        n_samples: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate stationary bootstrap samples with geometric block sizes."""
        samples = []
        n = len(feature_values)
        
        # Geometric distribution parameter (mean block size)
        p = 1.0 / self.config.block_size
        
        for _ in range(n_samples):
            sample_features = []
            sample_targets = []
            
            i = 0
            while i < n:
                # Generate block size from geometric distribution
                block_size = np.random.geometric(p)
                
                # Random starting point
                start_idx = np.random.randint(0, n)
                
                # Add block
                for j in range(block_size):
                    if i >= n:
                        break
                    
                    idx = (start_idx + j) % n
                    sample_features.append(feature_values[idx])
                    sample_targets.append(target_values[idx])
                    i += 1
            
            samples.append((np.array(sample_features), np.array(sample_targets)))
        
        return samples
    
    def _generate_wildcard_bootstrap(
        self,
        feature_values: np.ndarray,
        target_values: np.ndarray,
        n_samples: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate wildcard bootstrap with random block sizes."""
        samples = []
        n = len(feature_values)
        
        for _ in range(n_samples):
            sample_features = []
            sample_targets = []
            
            i = 0
            while i < n:
                # Random block size between 1 and block_size
                block_size = np.random.randint(1, self.config.block_size + 1)
                
                # Random starting point
                start_idx = np.random.randint(0, n)
                
                # Add block
                for j in range(block_size):
                    if i >= n:
                        break
                    
                    idx = (start_idx + j) % n
                    sample_features.append(feature_values[idx])
                    sample_targets.append(target_values[idx])
                    i += 1
            
            samples.append((np.array(sample_features), np.array(sample_targets)))
        
        return samples
    
    def _score_bootstrap_samples(
        self,
        bootstrap_samples: List[Tuple[np.ndarray, np.ndarray]],
        lookback_period: int
    ) -> List[float]:
        """Score bootstrap samples using VectorBT scoring system."""
        scores = []
        
        if self.config.parallel_processing and len(bootstrap_samples) > 1:
            scores = self._score_bootstrap_parallel(bootstrap_samples, lookback_period)
        else:
            scores = self._score_bootstrap_sequential(bootstrap_samples, lookback_period)
        
        return scores
    
    def _score_bootstrap_parallel(
        self,
        bootstrap_samples: List[Tuple[np.ndarray, np.ndarray]],
        lookback_period: int
    ) -> List[float]:
        """Score bootstrap samples in parallel."""
        scores = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit scoring tasks
            future_to_sample = {
                executor.submit(
                    self._score_single_bootstrap_sample,
                    sample_features, sample_targets, lookback_period
                ): i
                for i, (sample_features, sample_targets) in enumerate(bootstrap_samples)
            }
            
            # Collect results
            for future in as_completed(future_to_sample):
                try:
                    score = future.result()
                    if score is not None and not np.isnan(score):
                        scores.append(score)
                except Exception as e:
                    self.logger.warning(f"Bootstrap sample scoring failed: {e}")
        
        return scores
    
    def _score_bootstrap_sequential(
        self,
        bootstrap_samples: List[Tuple[np.ndarray, np.ndarray]],
        lookback_period: int
    ) -> List[float]:
        """Score bootstrap samples sequentially."""
        scores = []
        
        for sample_features, sample_targets in bootstrap_samples:
            try:
                score = self._score_single_bootstrap_sample(
                    sample_features, sample_targets, lookback_period
                )
                if score is not None and not np.isnan(score):
                    scores.append(score)
            except Exception as e:
                self.logger.warning(f"Bootstrap sample scoring failed: {e}")
        
        return scores
    
    def _score_single_bootstrap_sample(
        self,
        feature_values: np.ndarray,
        target_values: np.ndarray,
        lookback_period: int
    ) -> Optional[float]:
        """Score a single bootstrap sample."""
        try:
            # Use VectorBT scoring system
            result = self.scoring_system.score_feature_lookback(
                feature_values, target_values, lookback_period
            )
            
            if result.is_valid:
                return result.score
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Single bootstrap sample scoring failed: {e}")
            return None
    
    def _calculate_confidence_interval(self, scores: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for bootstrap scores."""
        if not scores:
            return (0.0, 0.0)
        
        alpha = 1 - self.config.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(scores, lower_percentile)
        upper_bound = np.percentile(scores, upper_percentile)
        
        return (lower_bound, upper_bound)
    
    def validate_multiple_lookbacks(
        self,
        feature_values: np.ndarray,
        target_values: np.ndarray,
        lookback_periods: List[int],
        n_samples: Optional[int] = None
    ) -> Dict[int, BootstrapResult]:
        """
        Validate multiple lookback periods using bootstrap sampling.
        
        Args:
            feature_values: Feature values array
            target_values: Target values array
            lookback_periods: List of lookback periods to validate
            n_samples: Number of bootstrap samples
            
        Returns:
            Dictionary mapping lookback periods to BootstrapResult objects
        """
        results = {}
        
        if self.config.parallel_processing and len(lookback_periods) > 1:
            results = self._validate_multiple_lookbacks_parallel(
                feature_values, target_values, lookback_periods, n_samples
            )
        else:
            results = self._validate_multiple_lookbacks_sequential(
                feature_values, target_values, lookback_periods, n_samples
            )
        
        return results
    
    def _validate_multiple_lookbacks_parallel(
        self,
        feature_values: np.ndarray,
        target_values: np.ndarray,
        lookback_periods: List[int],
        n_samples: Optional[int]
    ) -> Dict[int, BootstrapResult]:
        """Validate multiple lookbacks in parallel."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit validation tasks
            future_to_period = {
                executor.submit(
                    self.validate_lookback_period,
                    feature_values, target_values, period, n_samples
                ): period
                for period in lookback_periods
            }
            
            # Collect results
            for future in as_completed(future_to_period):
                period = future_to_period[future]
                try:
                    result = future.result()
                    results[period] = result
                except Exception as e:
                    self.logger.warning(f"Validation failed for period {period}: {e}")
                    results[period] = BootstrapResult(
                        mean_score=0.0,
                        std_score=0.0,
                        confidence_interval=(0.0, 0.0),
                        scores=[],
                        execution_time=0.0,
                        n_samples=0,
                        is_valid=False,
                        error_message=str(e)
                    )
        
        return results
    
    def _validate_multiple_lookbacks_sequential(
        self,
        feature_values: np.ndarray,
        target_values: np.ndarray,
        lookback_periods: List[int],
        n_samples: Optional[int]
    ) -> Dict[int, BootstrapResult]:
        """Validate multiple lookbacks sequentially."""
        results = {}
        
        for period in lookback_periods:
            try:
                result = self.validate_lookback_period(
                    feature_values, target_values, period, n_samples
                )
                results[period] = result
            except Exception as e:
                self.logger.warning(f"Validation failed for period {period}: {e}")
                results[period] = BootstrapResult(
                    mean_score=0.0,
                    std_score=0.0,
                    confidence_interval=(0.0, 0.0),
                    scores=[],
                    execution_time=0.0,
                    n_samples=0,
                    is_valid=False,
                    error_message=str(e)
                )
        
        return results


# Convenience functions
def create_vectorbt_bootstrap_validator(
    n_bootstrap_samples: int = 100,
    bootstrap_method: BootstrapMethod = BootstrapMethod.BLOCK,
    parallel_processing: bool = True
) -> VectorBTBootstrapValidator:
    """Create a VectorBT bootstrap validator with specified configuration."""
    config = VectorBTBootstrapConfig(
        n_bootstrap_samples=n_bootstrap_samples,
        bootstrap_method=bootstrap_method,
        parallel_processing=parallel_processing
    )
    return VectorBTBootstrapValidator(config)


def validate_lookback_with_vectorbt(
    feature_values: np.ndarray,
    target_values: np.ndarray,
    lookback_period: int,
    n_samples: int = 100
) -> BootstrapResult:
    """Convenience function to validate a single lookback period."""
    validator = create_vectorbt_bootstrap_validator(n_bootstrap_samples=n_samples)
    return validator.validate_lookback_period(feature_values, target_values, lookback_period)


# Test function
def test_vectorbt_bootstrap():
    """Test VectorBT bootstrap validation."""
    if not VECTORBT_AVAILABLE:
        tprint_error("❌ VectorBT not available for testing")
        return False
    
    tprint("🧪 Testing VectorBT Bootstrap Validation...")
    
    try:
        # Create test data
        np.random.seed(42)
        n_samples = 500
        
        # Create realistic feature and return data
        feature_values = np.cumsum(np.random.randn(n_samples) * 0.01)
        target_values = np.random.randn(n_samples) * 0.02
        
        # Test different bootstrap methods
        methods = [
            BootstrapMethod.SIMPLE,
            BootstrapMethod.BLOCK,
            BootstrapMethod.STATIONARY
        ]
        
        for method in methods:
            tprint_info(f"🔄 Testing {method.value} bootstrap...")
            
            validator = create_vectorbt_bootstrap_validator(
                n_bootstrap_samples=50,
                bootstrap_method=method
            )
            
            result = validator.validate_lookback_period(
                feature_values, target_values, 20
            )
            
            if result.is_valid:
                tprint_success(f"✅ {method.value}: mean={result.mean_score:.4f}, std={result.std_score:.4f}")
                tprint_info(f"📊 CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
            else:
                tprint_warning(f"⚠️ {method.value}: {result.error_message}")
        
        # Test multiple lookbacks
        lookback_periods = [10, 20, 30, 50]
        validator = create_vectorbt_bootstrap_validator(n_bootstrap_samples=30)
        
        results = validator.validate_multiple_lookbacks(
            feature_values, target_values, lookback_periods
        )
        
        tprint_success(f"✅ Validated {len(results)} lookback periods")
        
        # Find best lookback
        valid_results = {k: v for k, v in results.items() if v.is_valid}
        if valid_results:
            best_period = max(valid_results.keys(), key=lambda k: valid_results[k].mean_score)
            best_score = valid_results[best_period].mean_score
            tprint_info(f"🏆 Best lookback: {best_period} (score: {best_score:.4f})")
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ VectorBT bootstrap test failed: {e}")
        return False


if __name__ == "__main__":
    test_vectorbt_bootstrap()