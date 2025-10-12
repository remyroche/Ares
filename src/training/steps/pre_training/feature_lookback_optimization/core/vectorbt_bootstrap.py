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

# Import VectorBT Rolling Optimizer and Unified Vectorization Manager
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, 
        get_vectorbt_rolling_optimizer,
        optimized_rolling_mean,
        optimized_rolling_std,
        optimized_rolling_corr
    )
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager,
        get_unified_vectorization_manager,
        OperationType,
        OptimizationStrategy as UnifiedOptimizationStrategy
    )
    VECTORBT_UTILS_AVAILABLE = True
except ImportError:
    VECTORBT_UTILS_AVAILABLE = False
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None
    optimized_rolling_mean = None
    optimized_rolling_std = None
    optimized_rolling_corr = None
    UnifiedVectorizationManager = None
    get_unified_vectorization_manager = None
    OperationType = None
    UnifiedOptimizationStrategy = None

from src.utils.tprint import tprint, tprint_error, tprint_success, tprint_warning, tprint_debug, tprint_info
from src.utils.logger import get_logger
from ..error_handling.error_handler import safe_operation, get_error_handler
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
    
    # Enhanced optimization settings
    use_rolling_optimizer: bool = True
    use_unified_vectorization: bool = True
    rolling_optimizer_config: Optional[Dict[str, Any]] = None
    unified_vectorization_config: Optional[Dict[str, Any]] = None


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
        
        # Initialize enhanced optimization components
        self._initialize_enhanced_components()
        
        tprint_success("✅ VectorBT Bootstrap Validator initialized with enhanced optimizations")
    
    def _initialize_enhanced_components(self):
        """Initialize enhanced optimization components."""
        try:
            # Initialize VectorBT Rolling Optimizer
            if self.config.use_rolling_optimizer and VECTORBT_UTILS_AVAILABLE:
                rolling_config = self.config.rolling_optimizer_config or {}
                self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=rolling_config.get('enable_gpu', False),
                    enable_parallel=rolling_config.get('enable_parallel', True),
                    memory_efficient=rolling_config.get('memory_efficient', True)
                )
                self.logger.debug("VectorBT Rolling Optimizer initialized for bootstrap validation")
            else:
                self.rolling_optimizer = None
                self.logger.warning("VectorBT Rolling Optimizer not available for bootstrap validation")
            
            # Initialize Unified Vectorization Manager
            if self.config.use_unified_vectorization and VECTORBT_UTILS_AVAILABLE:
                unified_config = self.config.unified_vectorization_config or {}
                self.unified_manager = get_unified_vectorization_manager()
                self.logger.debug("Unified Vectorization Manager initialized for bootstrap validation")
            else:
                self.unified_manager = None
                self.logger.warning("Unified Vectorization Manager not available for bootstrap validation")
            
            self.logger.debug("Enhanced optimization components initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Enhanced component initialization failed: {e}")
            # Don't raise - these are optional enhancements
    
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
            
            # Use enhanced optimization if available
            if self.rolling_optimizer and self.unified_manager:
                return self._validate_lookback_period_enhanced(feature_values, target_values, lookback_period, n_samples)
            
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
    
    def _validate_lookback_period_enhanced(
        self,
        feature_values: np.ndarray,
        target_values: np.ndarray,
        lookback_period: int,
        n_samples: int
    ) -> BootstrapResult:
        """Validate lookback period using enhanced VectorBT optimizations."""
        start_time = time.time()
        
        try:
            # Use Unified Vectorization Manager for intelligent optimization
            if self.unified_manager:
                # Configure operation for bootstrap validation
                operation_config = self.unified_manager.create_operation_config(
                    operation_type=OperationType.CROSS_VALIDATION,
                    data_size=len(feature_values),
                    data_dimensions=(len(feature_values),),
                    memory_budget_mb=512.0,
                    time_budget_seconds=60.0
                )
                
                # Get optimal strategy
                strategy = self.unified_manager.select_optimization_strategy(operation_config)
                self.logger.debug(f"Selected bootstrap validation strategy: {strategy}")
            
            # Generate bootstrap samples using enhanced methods
            bootstrap_samples = self._generate_bootstrap_samples_enhanced(
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
                    error_message="Could not generate enhanced bootstrap samples"
                )
            
            # Score bootstrap samples using enhanced methods
            scores = self._score_bootstrap_samples_enhanced(bootstrap_samples, lookback_period)
            
            if not scores:
                return BootstrapResult(
                    mean_score=0.0,
                    std_score=0.0,
                    confidence_interval=(0.0, 0.0),
                    scores=[],
                    execution_time=time.time() - start_time,
                    n_samples=0,
                    is_valid=False,
                    error_message="Could not score enhanced bootstrap samples"
                )
            
            # Calculate statistics using enhanced methods
            mean_score = self._calculate_mean_enhanced(scores)
            std_score = self._calculate_std_enhanced(scores)
            confidence_interval = self._calculate_confidence_interval_enhanced(scores)
            
            result = BootstrapResult(
                mean_score=mean_score,
                std_score=std_score,
                confidence_interval=confidence_interval,
                scores=scores,
                execution_time=time.time() - start_time,
                n_samples=len(scores),
                is_valid=True
            )
            
            tprint_success(f"✅ Enhanced bootstrap validation completed: mean={mean_score:.4f}, std={std_score:.4f}")
            return result
            
        except Exception as e:
            self.logger.warning(f"Enhanced bootstrap validation failed: {e}")
            # Fallback to standard method
            return self._validate_lookback_period_standard(feature_values, target_values, lookback_period, n_samples)
    
    def _generate_bootstrap_samples_enhanced(
        self,
        feature_values: np.ndarray,
        target_values: np.ndarray,
        n_samples: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate bootstrap samples using enhanced VectorBT optimizations."""
        try:
            samples = []
            
            for i in range(n_samples):
                if self.config.bootstrap_method == BootstrapMethod.BLOCK:
                    # Use block bootstrap with VectorBTRollingOptimizer
                    sample = self._generate_block_bootstrap_sample_enhanced(feature_values, target_values)
                else:
                    # Use standard bootstrap
                    sample = self._generate_simple_bootstrap_sample(feature_values, target_values)
                
                if sample is not None:
                    samples.append(sample)
            
            return samples
            
        except Exception as e:
            self.logger.warning(f"Enhanced bootstrap sample generation failed: {e}")
            return self._generate_bootstrap_samples(feature_values, target_values, n_samples)
    
    def _generate_block_bootstrap_sample_enhanced(
        self,
        feature_values: np.ndarray,
        target_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate block bootstrap sample using VectorBTRollingOptimizer."""
        try:
            n = len(feature_values)
            block_size = self.config.block_size
            
            # Use VectorBTRollingOptimizer for block selection
            if self.rolling_optimizer and n > block_size * 2:
                # Calculate rolling statistics to identify stable blocks
                rolling_std = self.rolling_optimizer.rolling_std(
                    pd.Series(feature_values), window=block_size
                )
                
                # Select blocks with lower volatility for more stable samples
                stable_blocks = rolling_std.dropna()
                if len(stable_blocks) > 0:
                    # Select blocks with below-median volatility
                    threshold = stable_blocks.median()
                    stable_indices = stable_blocks[stable_blocks <= threshold].index
                    
                    if len(stable_indices) > 0:
                        # Randomly select from stable blocks
                        selected_blocks = np.random.choice(stable_indices, size=n // block_size, replace=True)
                        
                        # Reconstruct sample
                        sample_features = []
                        sample_targets = []
                        
                        for block_start in selected_blocks:
                            block_end = min(block_start + block_size, n)
                            sample_features.extend(feature_values[block_start:block_end])
                            sample_targets.extend(target_values[block_start:block_end])
                        
                        return np.array(sample_features[:n]), np.array(sample_targets[:n])
            
            # Fallback to standard block bootstrap
            return self._generate_block_bootstrap_sample(feature_values, target_values)
            
        except Exception as e:
            self.logger.warning(f"Enhanced block bootstrap sample generation failed: {e}")
            return self._generate_block_bootstrap_sample(feature_values, target_values)
    
    def _score_bootstrap_samples_enhanced(
        self,
        bootstrap_samples: List[Tuple[np.ndarray, np.ndarray]],
        lookback_period: int
    ) -> List[float]:
        """Score bootstrap samples using enhanced VectorBT optimizations."""
        try:
            scores = []
            
            for feature_values, target_values in bootstrap_samples:
                # Use enhanced scoring with rolling statistics
                if self.rolling_optimizer and len(feature_values) > 50:
                    # Calculate rolling correlation for more robust scoring
                    rolling_corr = self.rolling_optimizer.rolling_corr(
                        pd.Series(feature_values), 
                        pd.Series(target_values), 
                        window=min(20, len(feature_values) // 4)
                    )
                    
                    # Use mean of rolling correlations as score
                    valid_corr = rolling_corr.dropna()
                    if len(valid_corr) > 0:
                        score = abs(valid_corr.mean())
                    else:
                        # Fallback to simple correlation
                        score = abs(np.corrcoef(feature_values, target_values)[0, 1])
                else:
                    # Use standard scoring
                    score = abs(np.corrcoef(feature_values, target_values)[0, 1])
                
                scores.append(score)
            
            return scores
            
        except Exception as e:
            self.logger.warning(f"Enhanced bootstrap sample scoring failed: {e}")
            return self._score_bootstrap_samples(bootstrap_samples, lookback_period)
    
    def _calculate_mean_enhanced(self, scores: List[float]) -> float:
        """Calculate mean using enhanced VectorBT optimizations."""
        try:
            if self.rolling_optimizer and len(scores) > 20:
                # Use rolling mean for more robust statistics
                rolling_mean = self.rolling_optimizer.rolling_mean(
                    pd.Series(scores), window=min(10, len(scores) // 2)
                )
                
                # Use mean of rolling means
                valid_mean = rolling_mean.dropna()
                if len(valid_mean) > 0:
                    return valid_mean.mean()
            
            # Fallback to standard mean
            return np.mean(scores)
            
        except Exception as e:
            self.logger.warning(f"Enhanced mean calculation failed: {e}")
            return np.mean(scores)
    
    def _calculate_std_enhanced(self, scores: List[float]) -> float:
        """Calculate standard deviation using enhanced VectorBT optimizations."""
        try:
            if self.rolling_optimizer and len(scores) > 20:
                # Use rolling std for more robust statistics
                rolling_std = self.rolling_optimizer.rolling_std(
                    pd.Series(scores), window=min(10, len(scores) // 2)
                )
                
                # Use mean of rolling stds
                valid_std = rolling_std.dropna()
                if len(valid_std) > 0:
                    return valid_std.mean()
            
            # Fallback to standard std
            return np.std(scores)
            
        except Exception as e:
            self.logger.warning(f"Enhanced std calculation failed: {e}")
            return np.std(scores)
    
    def _calculate_confidence_interval_enhanced(self, scores: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval using enhanced VectorBT optimizations."""
        try:
            if self.rolling_optimizer and len(scores) > 20:
                # Use rolling quantiles for more robust confidence intervals
                rolling_q25 = self.rolling_optimizer.rolling_quantile(
                    pd.Series(scores), window=min(10, len(scores) // 2), q=0.25
                )
                rolling_q75 = self.rolling_optimizer.rolling_quantile(
                    pd.Series(scores), window=min(10, len(scores) // 2), q=0.75
                )
                
                # Use mean of rolling quantiles
                valid_q25 = rolling_q25.dropna()
                valid_q75 = rolling_q75.dropna()
                
                if len(valid_q25) > 0 and len(valid_q75) > 0:
                    return (valid_q25.mean(), valid_q75.mean())
            
            # Fallback to standard confidence interval
            return self._calculate_confidence_interval(scores)
            
        except Exception as e:
            self.logger.warning(f"Enhanced confidence interval calculation failed: {e}")
            return self._calculate_confidence_interval(scores)
    
    def _validate_lookback_period_standard(
        self,
        feature_values: np.ndarray,
        target_values: np.ndarray,
        lookback_period: int,
        n_samples: int
    ) -> BootstrapResult:
        """Validate lookback period using standard VectorBT methods."""
        start_time = time.time()
        
        try:
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
            
            result = BootstrapResult(
                mean_score=mean_score,
                std_score=std_score,
                confidence_interval=confidence_interval,
                scores=scores,
                execution_time=time.time() - start_time,
                n_samples=len(scores),
                is_valid=True
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Standard bootstrap validation failed: {e}")
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