"""
VectorBT-Enhanced Core Optimizer for Feature Lookback Optimization.

This module integrates all VectorBT optimizations into a unified optimizer that
replaces the existing core optimization logic with high-performance VectorBT-based
implementations.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import time
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.portfolio.base import Portfolio
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    Portfolio = None

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
from .utils.error_handling import safe_operation, get_error_handler
from .vectorbt_correlation import VectorBTCorrelationCalculator, VectorBTCorrelationConfig
from .vectorbt_scoring import VectorBTScoringSystem, ScoringMethod, VectorBTScoringConfig
from .vectorbt_feature_generation import VectorBTFeatureGenerator, FeatureType, VectorBTFeatureConfig
from .vectorbt_bootstrap import VectorBTBootstrapValidator, BootstrapMethod, VectorBTBootstrapConfig

logger = get_logger('VectorBTOptimizer')


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    VECTORBT_ONLY = "vectorbt_only"
    HYBRID = "hybrid"
    FALLBACK = "fallback"


@dataclass
class VectorBTOptimizationConfig:
    """Configuration for VectorBT optimization."""
    # Core settings
    strategy: OptimizationStrategy = OptimizationStrategy.VECTORBT_ONLY
    use_parallel_processing: bool = True
    max_workers: int = 4
    
    # Feature generation settings
    use_vectorbt_features: bool = True
    feature_cache_size: int = 1000
    
    # Correlation settings
    use_vectorbt_correlation: bool = True
    correlation_method: str = 'pearson'
    
    # Scoring settings
    use_vectorbt_scoring: bool = True
    scoring_method: ScoringMethod = ScoringMethod.COMPOSITE
    
    # Bootstrap settings
    use_vectorbt_bootstrap: bool = True
    n_bootstrap_samples: int = 100
    bootstrap_method: BootstrapMethod = BootstrapMethod.BLOCK
    
    # Performance settings
    early_termination_threshold: float = 0.02
    min_improvement_threshold: float = 0.01
    max_iterations: int = 50
    
    # Memory settings
    memory_efficient: bool = True
    batch_size: int = 1000
    
    # Enhanced optimization settings
    use_rolling_optimizer: bool = True
    use_unified_vectorization: bool = True
    rolling_optimizer_config: Optional[Dict[str, Any]] = None
    unified_vectorization_config: Optional[Dict[str, Any]] = None


@dataclass
class VectorBTOptimizationResult:
    """Result from VectorBT optimization."""
    best_lookback_period: int
    best_score: float
    optimization_method: str
    total_trials: int
    optimization_time: float
    convergence_achieved: bool
    feature_name: str
    metadata: Dict[str, Any]
    vectorbt_metrics: Optional[Dict[str, Any]] = None
    is_valid: bool = True
    error_message: Optional[str] = None


class VectorBTOptimizer:
    """
    High-performance optimizer using VectorBT for feature lookback optimization.
    
    This class integrates all VectorBT optimizations to provide a unified,
    high-performance optimization system that replaces the existing core optimizer.
    """
    
    def __init__(self, config: Optional[VectorBTOptimizationConfig] = None):
        """Initialize VectorBT optimizer."""
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required but not available. Install with: pip install vectorbt")
        
        self.config = config or VectorBTOptimizationConfig()
        self.logger = get_logger('VectorBTOptimizer')
        self.error_handler = get_error_handler()
        
        # Initialize VectorBT components
        self._initialize_components()
        
        # Initialize enhanced optimization components
        self._initialize_enhanced_components()
        
        tprint_success("✅ VectorBT Optimizer initialized with enhanced optimizations")
    
    def _initialize_components(self):
        """Initialize all VectorBT components."""
        try:
            # Initialize correlation calculator
            if self.config.use_vectorbt_correlation:
                correlation_config = VectorBTCorrelationConfig(
                    correlation_method=self.config.correlation_method,
                    memory_efficient=self.config.memory_efficient
                )
                self.correlation_calculator = VectorBTCorrelationCalculator(correlation_config)
            else:
                self.correlation_calculator = None
            
            # Initialize scoring system
            if self.config.use_vectorbt_scoring:
                scoring_config = VectorBTScoringConfig(
                    scoring_method=self.config.scoring_method
                )
                self.scoring_system = VectorBTScoringSystem(scoring_config)
            else:
                self.scoring_system = None
            
            # Initialize feature generator
            if self.config.use_vectorbt_features:
                feature_config = VectorBTFeatureConfig(
                    use_parallel=self.config.use_parallel_processing,
                    max_workers=self.config.max_workers,
                    cache_features=True,
                    cache_size=self.config.feature_cache_size
                )
                self.feature_generator = VectorBTFeatureGenerator(feature_config)
            else:
                self.feature_generator = None
            
            # Initialize bootstrap validator
            if self.config.use_vectorbt_bootstrap:
                bootstrap_config = VectorBTBootstrapConfig(
                    n_bootstrap_samples=self.config.n_bootstrap_samples,
                    bootstrap_method=self.config.bootstrap_method,
                    parallel_processing=self.config.use_parallel_processing,
                    max_workers=self.config.max_workers
                )
                self.bootstrap_validator = VectorBTBootstrapValidator(bootstrap_config)
            else:
                self.bootstrap_validator = None
            
            self.logger.debug("All VectorBT components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise
    
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
                self.logger.debug("VectorBT Rolling Optimizer initialized")
            else:
                self.rolling_optimizer = None
                self.logger.warning("VectorBT Rolling Optimizer not available")
            
            # Initialize Unified Vectorization Manager
            if self.config.use_unified_vectorization and VECTORBT_UTILS_AVAILABLE:
                unified_config = self.config.unified_vectorization_config or {}
                self.unified_manager = get_unified_vectorization_manager()
                self.logger.debug("Unified Vectorization Manager initialized")
            else:
                self.unified_manager = None
                self.logger.warning("Unified Vectorization Manager not available")
            
            self.logger.debug("Enhanced optimization components initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Enhanced component initialization failed: {e}")
            # Don't raise - these are optional enhancements
    
    @safe_operation
    def optimize_feature_lookback(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        lookback_range: Tuple[int, int],
        regularization_settings: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> VectorBTOptimizationResult:
        """
        Optimize feature lookback period using VectorBT.
        
        Args:
            data: Input data with features and target
            feature_name: Name of the feature to optimize
            target_column: Target column for optimization
            lookback_range: Min and max lookback periods to test
            regularization_settings: Regularization settings
            **kwargs: Additional parameters
            
        Returns:
            VectorBTOptimizationResult with optimization results
        """
        start_time = time.time()
        
        try:
            tprint_debug(f"🔄 Starting VectorBT optimization for {feature_name}")
            
            # Validate inputs
            if not self._validate_optimization_inputs(data, feature_name, target_column, lookback_range):
                return self._create_failed_result("validation_failed", 0.0, feature_name)
            
            # Generate lookback periods to test
            lookback_periods = self._generate_lookback_periods(lookback_range)
            
            # Generate features for all lookback periods
            features_dict = self._generate_features_for_lookbacks(
                data, feature_name, lookback_periods
            )
            
            if not features_dict:
                return self._create_failed_result("feature_generation_failed", 0.0, feature_name)
            
            # Get target values
            target_values = self._get_target_values(data, target_column)
            if target_values is None:
                return self._create_failed_result("target_extraction_failed", 0.0, feature_name)
            
            # Score all lookback periods
            scores = self._score_all_lookbacks(
                features_dict, target_values, lookback_periods
            )
            
            if not scores:
                return self._create_failed_result("scoring_failed", 0.0, feature_name)
            
            # Apply regularization if specified
            if regularization_settings:
                scores = self._apply_regularization(scores, regularization_settings)
            
            # Find best lookback period
            best_period, best_score = self._find_best_lookback(scores)
            
            # Perform bootstrap validation if enabled
            bootstrap_result = None
            if self.config.use_vectorbt_bootstrap and self.bootstrap_validator:
                bootstrap_result = self._perform_bootstrap_validation(
                    features_dict[best_period], target_values, best_period
                )
            
            # Create result
            optimization_time = time.time() - start_time
            result = VectorBTOptimizationResult(
                best_lookback_period=best_period,
                best_score=best_score,
                optimization_method="vectorbt_optimization",
                total_trials=len(scores),
                optimization_time=optimization_time,
                convergence_achieved=True,
                feature_name=feature_name,
                metadata=self._create_metadata(
                    feature_name, target_column, lookback_range, 
                    scores, bootstrap_result, optimization_time
                ),
                vectorbt_metrics=self._get_vectorbt_metrics(),
                is_valid=True
            )
            
            tprint_success(f"✅ VectorBT optimization completed: period={best_period}, score={best_score:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"VectorBT optimization failed: {e}")
            return VectorBTOptimizationResult(
                best_lookback_period=0,
                best_score=0.0,
                optimization_method="vectorbt_optimization",
                total_trials=0,
                optimization_time=time.time() - start_time,
                convergence_achieved=False,
                feature_name=feature_name,
                metadata={},
                is_valid=False,
                error_message=str(e)
            )
    
    def _validate_optimization_inputs(
        self, 
        data: pd.DataFrame, 
        feature_name: str, 
        target_column: str, 
        lookback_range: Tuple[int, int]
    ) -> bool:
        """Validate optimization inputs."""
        if data is None or len(data) == 0:
            return False
        
        if not feature_name or not target_column:
            return False
        
        if target_column not in data.columns:
            return False
        
        min_lookback, max_lookback = lookback_range
        if min_lookback >= max_lookback or min_lookback < 1:
            return False
        
        if len(data) < max_lookback + 10:  # Need sufficient data
            return False
        
        return True
    
    def _generate_lookback_periods(self, lookback_range: Tuple[int, int]) -> List[int]:
        """Generate lookback periods to test."""
        min_lookback, max_lookback = lookback_range
        
        # Generate periods with adaptive spacing
        if max_lookback - min_lookback <= 20:
            # Small range: test all periods
            periods = list(range(min_lookback, max_lookback + 1))
        else:
            # Large range: use adaptive sampling
            periods = []
            
            # Always include endpoints
            periods.extend([min_lookback, max_lookback])
            
            # Add intermediate points with logarithmic spacing
            n_points = min(20, max_lookback - min_lookback)
            if n_points > 2:
                log_space = np.logspace(
                    np.log10(min_lookback + 1), 
                    np.log10(max_lookback - 1), 
                    n_points - 2
                )
                periods.extend([int(round(x)) for x in log_space])
            
            # Remove duplicates and sort
            periods = sorted(list(set(periods)))
        
        return periods
    
    def _generate_features_for_lookbacks(
        self, 
        data: pd.DataFrame, 
        feature_name: str, 
        lookback_periods: List[int]
    ) -> Dict[int, np.ndarray]:
        """Generate features for all lookback periods."""
        # Use enhanced optimization if available
        if self.rolling_optimizer and self.unified_manager:
            return self._generate_features_enhanced(data, feature_name, lookback_periods)
        elif self.config.use_vectorbt_features and self.feature_generator:
            # Use VectorBT feature generation
            return self.feature_generator.generate_features_vectorbt(
                data, feature_name, lookback_periods
            )
        else:
            # Fallback to simple feature generation
            return self._generate_features_fallback(data, feature_name, lookback_periods)
    
    def _generate_features_fallback(
        self, 
        data: pd.DataFrame, 
        feature_name: str, 
        lookback_periods: List[int]
    ) -> Dict[int, np.ndarray]:
        """Fallback feature generation when VectorBT is not available."""
        features = {}
        
        for period in lookback_periods:
            try:
                # Simple moving average as fallback
                if 'close' in data.columns:
                    close_prices = data['close'].values
                    if len(close_prices) >= period:
                        sma = pd.Series(close_prices).rolling(window=period).mean()
                        features[period] = sma.values
            except Exception as e:
                self.logger.warning(f"Fallback feature generation failed for period {period}: {e}")
        
        return features
    
    def _generate_features_enhanced(
        self, 
        data: pd.DataFrame, 
        feature_name: str, 
        lookback_periods: List[int]
    ) -> Dict[int, np.ndarray]:
        """Generate features using enhanced VectorBT optimizations."""
        try:
            features = {}
            
            # Use Unified Vectorization Manager for intelligent optimization
            if self.unified_manager:
                # Configure operation for feature generation
                operation_config = self.unified_manager.create_operation_config(
                    operation_type=OperationType.FEATURE_ENGINEERING,
                    data_size=len(data),
                    data_dimensions=(len(data), len(data.columns)),
                    memory_budget_mb=1024.0,
                    time_budget_seconds=60.0
                )
                
                # Get optimal strategy
                strategy = self.unified_manager.select_optimization_strategy(operation_config)
                self.logger.debug(f"Selected optimization strategy: {strategy}")
            
            # Generate features using VectorBTRollingOptimizer
            for period in lookback_periods:
                try:
                    feature_values = self._generate_single_feature_enhanced(
                        data, feature_name, period
                    )
                    if feature_values is not None and len(feature_values) > 0:
                        features[period] = feature_values
                except Exception as e:
                    self.logger.warning(f"Enhanced feature generation failed for period {period}: {e}")
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Enhanced feature generation failed: {e}")
            # Fallback to standard method
            return self._generate_features_fallback(data, feature_name, lookback_periods)
    
    def _generate_single_feature_enhanced(
        self, 
        data: pd.DataFrame, 
        feature_name: str, 
        lookback_period: int
    ) -> Optional[np.ndarray]:
        """Generate a single feature using enhanced optimizations."""
        try:
            if 'close' not in data.columns:
                return None
            
            close_prices = data['close'].values
            
            # Use VectorBTRollingOptimizer for rolling operations
            if self.rolling_optimizer:
                if 'sma' in feature_name.lower() or 'simple' in feature_name.lower():
                    return self.rolling_optimizer.rolling_mean(
                        pd.Series(close_prices), lookback_period
                    ).values
                elif 'std' in feature_name.lower() or 'volatility' in feature_name.lower():
                    return self.rolling_optimizer.rolling_std(
                        pd.Series(close_prices), lookback_period
                    ).values
                elif 'rsi' in feature_name.lower():
                    # For RSI, we need to calculate price changes first
                    price_changes = np.diff(close_prices)
                    if len(price_changes) >= lookback_period:
                        # Use rolling operations for RSI calculation
                        gains = np.where(price_changes > 0, price_changes, 0)
                        losses = np.where(price_changes < 0, -price_changes, 0)
                        
                        avg_gains = self.rolling_optimizer.rolling_mean(
                            pd.Series(gains), lookback_period
                        ).values
                        avg_losses = self.rolling_optimizer.rolling_mean(
                            pd.Series(losses), lookback_period
                        ).values
                        
                        # Calculate RSI
                        rs = avg_gains / (avg_losses + 1e-10)
                        rsi = 100 - (100 / (1 + rs))
                        return rsi
                else:
                    # Default to rolling mean for unknown features
                    return self.rolling_optimizer.rolling_mean(
                        pd.Series(close_prices), lookback_period
                    ).values
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Enhanced single feature generation failed: {e}")
            return None
    
    def _get_target_values(self, data: pd.DataFrame, target_column: str) -> Optional[np.ndarray]:
        """Extract target values from data."""
        try:
            if target_column in data.columns:
                return data[target_column].values
            else:
                return None
        except Exception:
            return None
    
    def _score_all_lookbacks(
        self, 
        features_dict: Dict[int, np.ndarray], 
        target_values: np.ndarray, 
        lookback_periods: List[int]
    ) -> Dict[int, float]:
        """Score all lookback periods."""
        scores = {}
        
        # Use enhanced scoring if available
        if self.rolling_optimizer and self.unified_manager:
            scores = self._score_lookbacks_enhanced(features_dict, target_values, lookback_periods)
        elif self.config.use_vectorbt_scoring and self.scoring_system:
            # Use VectorBT scoring
            for period in lookback_periods:
                if period in features_dict:
                    feature_values = features_dict[period]
                    result = self.scoring_system.score_feature_lookback(
                        feature_values, target_values, period
                    )
                    if result.is_valid:
                        scores[period] = result.score
        else:
            # Fallback to correlation-based scoring
            scores = self._score_lookbacks_fallback(features_dict, target_values, lookback_periods)
        
        return scores
    
    def _score_lookbacks_fallback(
        self, 
        features_dict: Dict[int, np.ndarray], 
        target_values: np.ndarray, 
        lookback_periods: List[int]
    ) -> Dict[int, float]:
        """Fallback scoring when VectorBT is not available."""
        scores = {}
        
        for period in lookback_periods:
            if period in features_dict:
                feature_values = features_dict[period]
                try:
                    # Simple correlation-based scoring
                    min_length = min(len(feature_values), len(target_values))
                    if min_length > 10:
                        feature_aligned = feature_values[:min_length]
                        target_aligned = target_values[:min_length]
                        
                        # Remove NaN values
                        valid_mask = ~(np.isnan(feature_aligned) | np.isnan(target_aligned))
                        if np.any(valid_mask):
                            corr = np.corrcoef(
                                feature_aligned[valid_mask], 
                                target_aligned[valid_mask]
                            )[0, 1]
                            scores[period] = abs(corr) if not np.isnan(corr) else 0.0
                except Exception as e:
                    self.logger.warning(f"Fallback scoring failed for period {period}: {e}")
        
        return scores
    
    def _score_lookbacks_enhanced(
        self, 
        features_dict: Dict[int, np.ndarray], 
        target_values: np.ndarray, 
        lookback_periods: List[int]
    ) -> Dict[int, float]:
        """Score lookback periods using enhanced VectorBT optimizations."""
        scores = {}
        
        try:
            # Use Unified Vectorization Manager for intelligent optimization
            if self.unified_manager:
                operation_config = self.unified_manager.create_operation_config(
                    operation_type=OperationType.STATISTICAL_COMPUTATION,
                    data_size=len(target_values),
                    data_dimensions=(len(target_values),),
                    memory_budget_mb=512.0,
                    time_budget_seconds=30.0
                )
                
                strategy = self.unified_manager.select_optimization_strategy(operation_config)
                self.logger.debug(f"Selected scoring strategy: {strategy}")
            
            # Score each lookback period using VectorBTRollingOptimizer
            for period in lookback_periods:
                if period in features_dict:
                    feature_values = features_dict[period]
                    
                    # Calculate rolling correlation using VectorBTRollingOptimizer
                    if self.rolling_optimizer and len(feature_values) > 0:
                        try:
                            # Align feature and target values
                            min_length = min(len(feature_values), len(target_values))
                            if min_length > 10:
                                feature_aligned = feature_values[:min_length]
                                target_aligned = target_values[:min_length]
                                
                                # Remove NaN values
                                valid_mask = ~(np.isnan(feature_aligned) | np.isnan(target_aligned))
                                if np.any(valid_mask):
                                    feature_clean = feature_aligned[valid_mask]
                                    target_clean = target_aligned[valid_mask]
                                    
                                    # Use rolling correlation for more robust scoring
                                    if len(feature_clean) >= period:
                                        rolling_corr = self.rolling_optimizer.rolling_corr(
                                            pd.Series(feature_clean), 
                                            pd.Series(target_clean), 
                                            window=min(period, len(feature_clean) // 2)
                                        )
                                        
                                        # Use mean of rolling correlations as score
                                        valid_corr = rolling_corr.dropna()
                                        if len(valid_corr) > 0:
                                            scores[period] = abs(valid_corr.mean())
                                        else:
                                            # Fallback to simple correlation
                                            corr = np.corrcoef(feature_clean, target_clean)[0, 1]
                                            scores[period] = abs(corr) if not np.isnan(corr) else 0.0
                                    else:
                                        # Fallback to simple correlation
                                        corr = np.corrcoef(feature_clean, target_clean)[0, 1]
                                        scores[period] = abs(corr) if not np.isnan(corr) else 0.0
                        except Exception as e:
                            self.logger.warning(f"Enhanced scoring failed for period {period}: {e}")
                            # Fallback to simple correlation
                            try:
                                min_length = min(len(feature_values), len(target_values))
                                feature_aligned = feature_values[:min_length]
                                target_aligned = target_values[:min_length]
                                
                                valid_mask = ~(np.isnan(feature_aligned) | np.isnan(target_aligned))
                                if np.any(valid_mask):
                                    corr = np.corrcoef(
                                        feature_aligned[valid_mask], 
                                        target_aligned[valid_mask]
                                    )[0, 1]
                                    scores[period] = abs(corr) if not np.isnan(corr) else 0.0
                            except Exception:
                                scores[period] = 0.0
            
            return scores
            
        except Exception as e:
            self.logger.warning(f"Enhanced scoring failed: {e}")
            # Fallback to standard scoring
            return self._score_lookbacks_fallback(features_dict, target_values, lookback_periods)
    
    def _apply_regularization(
        self, 
        scores: Dict[int, float], 
        regularization_settings: Dict[str, float]
    ) -> Dict[int, float]:
        """Apply regularization to scores."""
        try:
            preferred_min = regularization_settings.get('preferred_min', 40.0)
            preferred_max = regularization_settings.get('preferred_max', 80.0)
            penalty_strength = regularization_settings.get('penalty_strength', 0.0)
            
            if penalty_strength <= 0:
                return scores  # No regularization
            
            regularized_scores = {}
            for period, score in scores.items():
                # Calculate penalty based on distance from preferred range
                if period < preferred_min:
                    penalty = penalty_strength * (preferred_min - period) ** 2
                elif period > preferred_max:
                    penalty = penalty_strength * (period - preferred_max) ** 2
                else:
                    penalty = 0.0
                
                regularized_scores[period] = max(0.0, score - penalty)
            
            return regularized_scores
            
        except Exception as e:
            self.logger.warning(f"Regularization failed: {e}")
            return scores
    
    def _find_best_lookback(self, scores: Dict[int, float]) -> Tuple[int, float]:
        """Find the best lookback period and score."""
        if not scores:
            return 0, 0.0
        
        best_period = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_period]
        
        return best_period, best_score
    
    def _perform_bootstrap_validation(
        self, 
        feature_values: np.ndarray, 
        target_values: np.ndarray, 
        lookback_period: int
    ) -> Optional[Dict[str, Any]]:
        """Perform bootstrap validation for the best lookback period."""
        try:
            result = self.bootstrap_validator.validate_lookback_period(
                feature_values, target_values, lookback_period
            )
            
            if result.is_valid:
                return {
                    'mean_score': result.mean_score,
                    'std_score': result.std_score,
                    'confidence_interval': result.confidence_interval,
                    'n_samples': result.n_samples
                }
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Bootstrap validation failed: {e}")
            return None
    
    def _create_metadata(
        self, 
        feature_name: str, 
        target_column: str, 
        lookback_range: Tuple[int, int], 
        scores: Dict[int, float], 
        bootstrap_result: Optional[Dict[str, Any]], 
        optimization_time: float
    ) -> Dict[str, Any]:
        """Create metadata for the optimization result."""
        metadata = {
            'feature_name': feature_name,
            'target_column': target_column,
            'lookback_range': lookback_range,
            'n_periods_tested': len(scores),
            'optimization_time': optimization_time,
            'scores': scores,
            'vectorbt_optimization': True
        }
        
        if bootstrap_result:
            metadata['bootstrap_validation'] = bootstrap_result
        
        return metadata
    
    def _get_vectorbt_metrics(self) -> Dict[str, Any]:
        """Get VectorBT-specific metrics."""
        metrics = {
            'vectorbt_available': VECTORBT_AVAILABLE,
            'correlation_calculator': self.correlation_calculator is not None,
            'scoring_system': self.scoring_system is not None,
            'feature_generator': self.feature_generator is not None,
            'bootstrap_validator': self.bootstrap_validator is not None,
            'vectorbt_utils_available': VECTORBT_UTILS_AVAILABLE,
            'rolling_optimizer': self.rolling_optimizer is not None,
            'unified_manager': self.unified_manager is not None
        }
        
        if self.feature_generator:
            metrics['feature_cache_stats'] = self.feature_generator.get_cache_stats()
        
        if self.rolling_optimizer:
            metrics['rolling_optimizer_stats'] = self.rolling_optimizer.get_performance_stats()
        
        if self.unified_manager:
            metrics['unified_manager_stats'] = self.unified_manager.get_performance_stats()
        
        return metrics
    
    def _create_failed_result(
        self, 
        reason: str, 
        score: float, 
        feature_name: str
    ) -> VectorBTOptimizationResult:
        """Create a failed optimization result."""
        return VectorBTOptimizationResult(
            best_lookback_period=0,
            best_score=score,
            optimization_method="vectorbt_optimization",
            total_trials=0,
            optimization_time=0.0,
            convergence_achieved=False,
            feature_name=feature_name,
            metadata={'failure_reason': reason},
            is_valid=False,
            error_message=reason
        )


# Convenience functions
def create_vectorbt_optimizer(
    strategy: OptimizationStrategy = OptimizationStrategy.VECTORBT_ONLY,
    use_parallel_processing: bool = True,
    scoring_method: ScoringMethod = ScoringMethod.COMPOSITE,
    use_rolling_optimizer: bool = True,
    use_unified_vectorization: bool = True
) -> VectorBTOptimizer:
    """Create a VectorBT optimizer with specified configuration."""
    config = VectorBTOptimizationConfig(
        strategy=strategy,
        use_parallel_processing=use_parallel_processing,
        scoring_method=scoring_method,
        use_rolling_optimizer=use_rolling_optimizer,
        use_unified_vectorization=use_unified_vectorization
    )
    return VectorBTOptimizer(config)


def optimize_feature_with_vectorbt(
    data: pd.DataFrame,
    feature_name: str,
    target_column: str,
    lookback_range: Tuple[int, int],
    **kwargs
) -> VectorBTOptimizationResult:
    """Convenience function to optimize a feature using VectorBT."""
    optimizer = create_vectorbt_optimizer()
    return optimizer.optimize_feature_lookback(
        data, feature_name, target_column, lookback_range, **kwargs
    )


# Test function
def test_vectorbt_optimizer():
    """Test VectorBT optimizer."""
    if not VECTORBT_AVAILABLE:
        tprint_error("❌ VectorBT not available for testing")
        return False
    
    tprint("🧪 Testing VectorBT Optimizer...")
    
    try:
        # Create test data
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'close': np.cumsum(np.random.randn(n_samples) * 0.01) + 100,
            'high': np.cumsum(np.random.randn(n_samples) * 0.01) + 105,
            'low': np.cumsum(np.random.randn(n_samples) * 0.01) + 95,
            'volume': np.random.randint(1000, 10000, n_samples),
            'returns': np.random.randn(n_samples) * 0.02
        })
        
        # Test different optimization strategies
        strategies = [
            OptimizationStrategy.VECTORBT_ONLY,
            OptimizationStrategy.HYBRID
        ]
        
        for strategy in strategies:
            tprint_info(f"🔄 Testing {strategy.value} strategy...")
            
            optimizer = create_vectorbt_optimizer(strategy=strategy)
            
            result = optimizer.optimize_feature_lookback(
                data, 'sma', 'returns', (10, 50)
            )
            
            if result.is_valid:
                tprint_success(f"✅ {strategy.value}: period={result.best_lookback_period}, score={result.best_score:.4f}")
                tprint_info(f"📊 Optimization time: {result.optimization_time:.3f}s")
                tprint_info(f"📊 Trials: {result.total_trials}")
            else:
                tprint_warning(f"⚠️ {strategy.value}: {result.error_message}")
        
        # Test different scoring methods
        scoring_methods = [
            ScoringMethod.SHARPE_RATIO,
            ScoringMethod.SORTINO_RATIO,
            ScoringMethod.COMPOSITE
        ]
        
        for method in scoring_methods:
            tprint_info(f"🔄 Testing {method.value} scoring...")
            
            optimizer = create_vectorbt_optimizer(scoring_method=method)
            
            result = optimizer.optimize_feature_lookback(
                data, 'rsi', 'returns', (10, 30)
            )
            
            if result.is_valid:
                tprint_success(f"✅ {method.value}: period={result.best_lookback_period}, score={result.best_score:.4f}")
            else:
                tprint_warning(f"⚠️ {method.value}: {result.error_message}")
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ VectorBT optimizer test failed: {e}")
        return False


if __name__ == "__main__":
    test_vectorbt_optimizer()