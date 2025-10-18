"""
Complementary Lookback Optimization System

This module implements the corrected approach to feature optimization:
1. Complementary scoring instead of alignment scoring
2. Regime-invariant optimization instead of regime-specific optimization
3. Multi-objective optimization for Tactician training
"""

from __future__ import annotations

import warnings
import logging
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np

from ....utils.tprint import tprint
from ....utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer, OptimizationConfig
from ....utils.ml_common.unified_vectorization_manager import get_unified_vectorization_manager
from ....utils.logger import system_logger

# Lazy import to avoid circular dependency
def _get_feature_generator_imports():
    try:
        from ...core.feature_generator import FeatureGenerator, FeatureConfig
        return FeatureGenerator, FeatureConfig
    except ImportError as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"FeatureGenerator import failed: {e}")
        return None, None

# Initialize logger
logger = system_logger.getChild("ComplementaryLookbackOptimizer")

# VectorBT detection (v0.28.1 compatible)
try:
    import vectorbt as vbt  # Do not import nested submodules; API paths differ across versions
    VECTORBT_AVAILABLE = True
    logger.info(f"✅ Detected VectorBT v{getattr(vbt, '__version__', 'unknown')}")
except Exception as e:
    VECTORBT_AVAILABLE = False
    vbt = None
    warnings.warn(f"VectorBT not available ({e}). Install with: pip install vectorbt for optimized performance")

logger = logging.getLogger(__name__)

class ComplementaryOptimizationMethod(Enum):
    """Optimization methods for complementary feature selection."""
    COMPLEMENTARY_CORRELATION = "complementary_correlation"
    INFORMATION_GAIN = "information_gain"
    MUTUAL_INFORMATION = "mutual_information"
    REGIME_INVARIANT = "regime_invariant"
    MULTI_OBJECTIVE = "multi_objective"

@dataclass
class ComplementaryOptimizationConfig:
    """Configuration for complementary feature optimization."""
    min_lookback: int = 5
    max_lookback: int = 252
    step_size: int = 1
    optimization_method: ComplementaryOptimizationMethod = ComplementaryOptimizationMethod.COMPLEMENTARY_CORRELATION
    cv_folds: int = 5
    stability_threshold: float = 0.8
    performance_threshold: float = 0.6
    regime_invariant: bool = True  # Always use regime-invariant optimization
    parallel_processing: bool = True
    max_workers: int = 4
    memory_efficient: bool = True
    chunk_size: int = 1000
    optimization_metric: str = "complementary_score"
    
    # Complementary optimization specific parameters
    analyst_alignment_penalty: float = 0.5  # Penalty for features that align with analyst
    complementary_bonus: float = 1.5  # Bonus for complementary information
    regime_consistency_weight: float = 0.3  # Weight for regime consistency
    temporal_stability_weight: float = 0.2  # Weight for temporal stability

@dataclass
class ComplementaryOptimizationResult:
    """Result of complementary feature optimization."""
    feature_name: str
    optimal_lookback: int
    complementary_score: float
    analyst_alignment_score: float
    regime_consistency_score: float
    temporal_stability_score: float
    overall_score: float
    confidence_interval: tuple
    optimization_method: str
    regime_performance: Dict[str, float]  # Performance across all regimes
    temporal_performance: List[float]  # Performance over time
    complementary_info_gain: float  # Information gain beyond analyst

class ComplementaryLookbackOptimizer:
    """
    Optimizer for feature lookback periods using complementary scoring.
    
    This optimizer focuses on finding features that provide complementary
    information beyond what the Analyst already knows, rather than aligning
    with Analyst signals.
    """

    def __init__(self, config: Optional[ComplementaryOptimizationConfig] = None):
        """
        Initialize the complementary lookback optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config or ComplementaryOptimizationConfig()
        self.logger = logger.getChild('ComplementaryLookbackOptimizer')

        # Cache for optimization results
        self._optimization_cache: Dict[str, ComplementaryOptimizationResult] = {}
        
        # Initialize vectorization manager for efficient computations
        try:
            self.vectorization_manager = get_unified_vectorization_manager()
            tprint("✅ UnifiedVectorizationManager initialized for complementary optimization")
        except ImportError:
            self.vectorization_manager = None
            tprint("⚠️ UnifiedVectorizationManager not available, using standard operations")
        
        # Initialize Bayesian TPE optimizer for advanced optimization
        try:
            tpe_config = OptimizationConfig(
                n_trials=50,
                timeout=300,
                enable_staged_optimization=True,
                coarse_grid_trials=10,
                fine_grid_trials=15,
                tpe_trials=25
            )
            self.tpe_optimizer = BayesianTPEOptimizer(tpe_config)
            tprint("✅ BayesianTPEOptimizer initialized for complementary optimization")
        except ImportError:
            self.tpe_optimizer = None
            tprint("⚠️ BayesianTPEOptimizer not available, using grid search")

        self.logger.info("✅ ComplementaryLookbackOptimizer initialized")
        tprint("🎯 ComplementaryLookbackOptimizer initialized with advanced optimization capabilities")

    def optimize_lookback(self,
                         generator: 'FeatureGenerator',
                         data: pd.DataFrame,
                         target_column: str,
                         analyst_signals: Optional[pd.Series] = None,
                         regime_series: Optional[pd.Series] = None) -> int:
        """
        Optimize lookback period for a feature generator using complementary scoring.

        Args:
            generator: Feature generator to optimize
            data: Input data
            target_column: Target column for optimization
            analyst_signals: Optional analyst signals for complementary scoring
            regime_series: Optional regime assignments as pd.Series for regime-invariant optimization

        Returns:
            Optimal lookback period
        """
        self.logger.info(f"Optimizing lookback for {generator.config.name} using complementary scoring")
        tprint(f"🔧 Optimizing lookback for {generator.config.name} using complementary scoring")

        # Check cache first with stable hash
        data_hash = self._get_stable_data_hash(data)
        analyst_hash = self._get_stable_analyst_hash(analyst_signals)
        cache_key = f"{generator.config.name}_{data_hash}_{analyst_hash}"
        if cache_key in self._optimization_cache:
            result = self._optimization_cache[cache_key]
            self.logger.info(f"Using cached optimization result: {result.optimal_lookback}")
            tprint(f"📋 Using cached optimization result: {result.optimal_lookback}")
            return result.optimal_lookback

        # Perform complementary optimization using advanced methods
        if self.tpe_optimizer is not None and self.config.optimization_method == ComplementaryOptimizationMethod.COMPLEMENTARY_REGIME_INVARIANT:
            tprint("🧠 Using Bayesian TPE optimization for efficient search")
            result = self._bayesian_tpe_optimization(
                generator, data, target_column, analyst_signals, regime_series
            )
        else:
            tprint("🔍 Using grid search optimization")
            result = self._complementary_optimization(
                generator, data, target_column, analyst_signals, regime_series
            )
        
        self._optimization_cache[cache_key] = result
        tprint(f"✅ Optimization completed: lookback={result.optimal_lookback}, score={result.complementary_score:.4f}")
        return result.optimal_lookback

    def _complementary_optimization(self,
                                   generator: 'FeatureGenerator',
                                   data: pd.DataFrame,
                                   target_column: str,
                                   analyst_signals: Optional[pd.Series] = None,
                                   regime_series: Optional[pd.Series] = None) -> ComplementaryOptimizationResult:
        """Perform complementary optimization."""
        self.logger.info(f"Using complementary optimization for {generator.config.name}")

        best_score = -np.inf
        best_lookback = self.config.min_lookback
        all_scores = []

        # Test different lookback periods
        for lookback in range(self.config.min_lookback,
                            self.config.max_lookback + 1,
                            self.config.step_size):
            try:
                # Generate feature with current lookback
                if generator.supports_lookback_optimization():
                    result = generator.generate_with_lookback(data, lookback)
                else:
                    result = generator.generate(data)

                if not result.success:
                    continue

                # Calculate complementary score
                complementary_score = self._calculate_complementary_score(
                    result.data, data[target_column], analyst_signals
                )
                
                # Calculate regime consistency (regime-invariant optimization)
                regime_consistency = self._calculate_regime_consistency(
                    result.data, data[target_column], regime_series
                )
                
                # Calculate temporal stability
                temporal_stability = self._calculate_temporal_stability(
                    result.data, data[target_column]
                )

                # Calculate overall score (complementary focus)
                overall_score = self._calculate_overall_score(
                    complementary_score, regime_consistency, temporal_stability
                )
                
                all_scores.append(overall_score)

                if overall_score > best_score:
                    best_score = overall_score
                    best_lookback = lookback
                    self.logger.info(f"New best lookback {lookback}: score={overall_score:.4f}, "
                                   f"complementary={complementary_score:.4f}, "
                                   f"regime={regime_consistency:.4f}, "
                                   f"temporal={temporal_stability:.4f}")

            except Exception as e:
                self.logger.warning(f"Error in optimization for lookback {lookback}: {e}")
                continue

        # Calculate final metrics for best lookback
        final_result = self._calculate_final_metrics(
            generator, data, target_column, best_lookback, 
            analyst_signals, regime_series
        )

        return ComplementaryOptimizationResult(
            feature_name=generator.config.name,
            optimal_lookback=best_lookback,
            complementary_score=final_result['complementary_score'],
            analyst_alignment_score=final_result['analyst_alignment_score'],
            regime_consistency_score=final_result['regime_consistency_score'],
            temporal_stability_score=final_result['temporal_stability_score'],
            overall_score=best_score,
            confidence_interval=self._calculate_confidence_interval(all_scores),
            optimization_method=self.config.optimization_method.value,
            regime_performance=final_result['regime_performance'],
            temporal_performance=final_result['temporal_performance'],
            complementary_info_gain=final_result['complementary_info_gain']
        )

    def _calculate_complementary_score(self,
                                     feature_values: pd.Series,
                                     target_values: pd.Series,
                                     analyst_signals: Optional[pd.Series] = None) -> float:
        """
        Calculate complementary score using partial correlation - how much information 
        the feature provides beyond what the analyst already knows.
        """
        # Get valid indices
        valid_indices = ~(feature_values.isna() | target_values.isna())
        if valid_indices.sum() < 10:
            return 0.0

        feature_clean = feature_values[valid_indices]
        target_clean = target_values[valid_indices]

        # Basic correlation with target
        target_correlation = abs(feature_clean.corr(target_clean))
        
        if analyst_signals is None:
            # No analyst signals - use direct correlation
            return target_correlation

        # Calculate complementary information using partial correlation
        analyst_clean = analyst_signals[valid_indices]
        
        # Ensure we have enough data for partial correlation
        if len(analyst_clean) < 10:
            return target_correlation

        # Calculate partial correlation: corr(feature residualized vs analyst, target residualized vs analyst)
        try:
            # Residualize feature and target vs analyst
            def residualize(y, x):
                """Calculate residuals of y regressed on x."""
                if len(x) < 2:
                    return y
                x_matrix = np.column_stack([np.ones(len(x)), x.values])
                try:
                    beta = np.linalg.lstsq(x_matrix, y.values, rcond=None)[0]
                    return y.values - x_matrix @ beta
                except np.linalg.LinAlgError:
                    return y.values

            # Residualize feature and target against analyst
            feature_residual = residualize(feature_clean, analyst_clean)
            target_residual = residualize(target_clean, analyst_clean)
            
            # Calculate partial correlation
            if len(feature_residual) > 1 and np.std(feature_residual) > 1e-10 and np.std(target_residual) > 1e-10:
                partial_correlation = np.corrcoef(feature_residual, target_residual)[0, 1]
                if not np.isnan(partial_correlation):
                    # Partial correlation measures information beyond analyst
                    complementary_score = abs(partial_correlation)
                else:
                    # Fallback to basic correlation
                    complementary_score = target_correlation
            else:
                # Fallback to basic correlation
                complementary_score = target_correlation

        except Exception as e:
            self.logger.warning(f"Partial correlation calculation failed: {e}, using basic correlation")
            complementary_score = target_correlation

        return min(1.0, max(0.0, complementary_score))

    def _bayesian_tpe_optimization(self,
                                  generator: 'FeatureGenerator',
                                  data: pd.DataFrame,
                                  target_column: str,
                                  analyst_signals: Optional[pd.Series] = None,
                                  regime_series: Optional[pd.Series] = None) -> ComplementaryOptimizationResult:
        """
        Perform Bayesian TPE optimization for efficient lookback search.
        """
        tprint("🧠 Starting Bayesian TPE optimization")
        
        def objective(trial):
            """Objective function for TPE optimization."""
            lookback = trial.suggest_int('lookback', self.config.min_lookback, self.config.max_lookback)
            
            try:
                # Generate feature with current lookback
                if generator.supports_lookback_optimization():
                    result = generator.generate_with_lookback(data, lookback)
                else:
                    result = generator.generate(data)
                
                if not result.success or result.data is None or result.data.empty:
                    return -np.inf
                
                # Calculate scores using vectorized operations if available
                if self.vectorization_manager is not None:
                    scores = self._calculate_scores_vectorized(
                        result.data, data[target_column], analyst_signals, regime_series
                    )
                else:
                    scores = self._calculate_scores_standard(
                        result.data, data[target_column], analyst_signals, regime_series
                    )
                
                return scores['overall_score']
                
            except Exception as e:
                self.logger.warning(f"Error in TPE trial for lookback {lookback}: {e}")
                return -np.inf
        
        # Run TPE optimization
        best_trial = self.tpe_optimizer.optimize(objective)
        best_lookback = best_trial.params['lookback']
        best_score = best_trial.value
        
        tprint(f"🧠 TPE optimization completed: lookback={best_lookback}, score={best_score:.4f}")
        
        # Calculate final metrics for best lookback
        final_result = self._calculate_final_metrics(
            generator, data, target_column, best_lookback, 
            analyst_signals, regime_series
        )
        
        return ComplementaryOptimizationResult(
            feature_name=generator.config.name,
            optimal_lookback=best_lookback,
            complementary_score=final_result['complementary_score'],
            regime_consistency=final_result['regime_consistency'],
            temporal_stability=final_result['temporal_stability'],
            overall_score=best_score,
            confidence_interval=final_result['confidence_interval'],
            optimization_method="bayesian_tpe",
            regime_performance=final_result['regime_performance'],
            temporal_performance=final_result['temporal_performance'],
            complementary_info_gain=final_result['complementary_info_gain']
        )

    def _calculate_scores_vectorized(self,
                                   feature_values: pd.Series,
                                   target_values: pd.Series,
                                   analyst_signals: Optional[pd.Series] = None,
                                   regime_series: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate scores using vectorized operations."""
        if self.vectorization_manager is None:
            return self._calculate_scores_standard(feature_values, target_values, analyst_signals, regime_series)
        
        try:
            # Use vectorization manager for efficient computations
            complementary_score = self._calculate_complementary_score_vectorized(
                feature_values, target_values, analyst_signals
            )
            regime_consistency = self._calculate_regime_consistency_vectorized(
                feature_values, target_values, regime_series
            )
            temporal_stability = self._calculate_temporal_stability_vectorized(
                feature_values, target_values
            )
            
            overall_score = self._calculate_overall_score(
                complementary_score, regime_consistency, temporal_stability
            )
            
            return {
                'complementary_score': complementary_score,
                'regime_consistency': regime_consistency,
                'temporal_stability': temporal_stability,
                'overall_score': overall_score
            }
        except Exception as e:
            self.logger.warning(f"Vectorized calculation failed: {e}, falling back to standard")
            return self._calculate_scores_standard(feature_values, target_values, analyst_signals, regime_series)

    def _calculate_scores_standard(self,
                                 feature_values: pd.Series,
                                 target_values: pd.Series,
                                 analyst_signals: Optional[pd.Series] = None,
                                 regime_series: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate scores using standard operations."""
        complementary_score = self._calculate_complementary_score(
            feature_values, target_values, analyst_signals
        )
        regime_consistency = self._calculate_regime_consistency(
            feature_values, target_values, regime_series
        )
        temporal_stability = self._calculate_temporal_stability(
            feature_values, target_values
        )
        
        overall_score = self._calculate_overall_score(
            complementary_score, regime_consistency, temporal_stability
        )
        
        return {
            'complementary_score': complementary_score,
            'regime_consistency': regime_consistency,
            'temporal_stability': temporal_stability,
            'overall_score': overall_score
        }

    def _calculate_complementary_score_vectorized(self,
                                                feature_values: pd.Series,
                                                target_values: pd.Series,
                                                analyst_signals: Optional[pd.Series] = None) -> float:
        """Calculate complementary score using vectorized operations."""
        if self.vectorization_manager is None:
            return self._calculate_complementary_score(feature_values, target_values, analyst_signals)
        
        try:
            # Use vectorization manager for efficient partial correlation calculation
            if analyst_signals is not None:
                # Align data
                valid_indices = ~(feature_values.isna() | target_values.isna() | analyst_signals.isna())
                if valid_indices.sum() < 10:
                    return 0.0
                
                feature_clean = feature_values[valid_indices]
                target_clean = target_values[valid_indices]
                analyst_clean = analyst_signals[valid_indices]
                
                # Use vectorization manager for efficient residualization
                feature_residual = self.vectorization_manager.residualize(feature_clean, analyst_clean)
                target_residual = self.vectorization_manager.residualize(target_clean, analyst_clean)
                
                # Calculate partial correlation
                if len(feature_residual) > 1 and np.std(feature_residual) > 1e-10 and np.std(target_residual) > 1e-10:
                    partial_correlation = np.corrcoef(feature_residual, target_residual)[0, 1]
                    if not np.isnan(partial_correlation):
                        return min(1.0, max(0.0, abs(partial_correlation)))
                
            # Fallback to basic correlation
            return min(1.0, max(0.0, abs(feature_values.corr(target_values))))
            
        except Exception as e:
            self.logger.warning(f"Vectorized complementary score calculation failed: {e}")
            return self._calculate_complementary_score(feature_values, target_values, analyst_signals)

    def _calculate_regime_consistency_vectorized(self,
                                               feature_values: pd.Series,
                                               target_values: pd.Series,
                                               regime_series: Optional[pd.Series] = None) -> float:
        """Calculate regime consistency using vectorized operations."""
        if self.vectorization_manager is None or regime_series is None:
            return self._calculate_regime_consistency(feature_values, target_values, regime_series)
        
        try:
            # Use vectorization manager for efficient regime analysis
            return self.vectorization_manager.calculate_regime_consistency(
                feature_values, target_values, regime_series
            )
        except Exception as e:
            self.logger.warning(f"Vectorized regime consistency calculation failed: {e}")
            return self._calculate_regime_consistency(feature_values, target_values, regime_series)

    def _calculate_temporal_stability_vectorized(self,
                                              feature_values: pd.Series,
                                              target_values: pd.Series) -> float:
        """Calculate temporal stability using vectorized operations."""
        if self.vectorization_manager is None:
            return self._calculate_temporal_stability(feature_values, target_values)
        
        try:
            # Use vectorization manager for efficient rolling correlation
            return self.vectorization_manager.calculate_temporal_stability(
                feature_values, target_values
            )
        except Exception as e:
            self.logger.warning(f"Vectorized temporal stability calculation failed: {e}")
            return self._calculate_temporal_stability(feature_values, target_values)

    def _calculate_regime_consistency(self,
                                    feature_values: pd.Series,
                                    target_values: pd.Series,
                                    regime_series: Optional[pd.Series] = None) -> float:
        """
        Calculate regime consistency - how well the feature works across all regimes.
        Uses regime-invariant optimization (single lookback for all regimes).
        
        Args:
            feature_values: Feature values
            target_values: Target values  
            regime_series: Regime assignments as pd.Series aligned to feature/target indices
        """
        if regime_series is None:
            return 1.0  # No regime information - assume consistent

        # Get valid indices
        valid_indices = ~(feature_values.isna() | target_values.isna())
        if valid_indices.sum() < 10:
            return 0.0

        feature_clean = feature_values[valid_indices]
        target_clean = target_values[valid_indices]
        regime_clean = regime_series[valid_indices]

        if len(regime_clean.unique()) < 2:
            return 1.0  # Single regime

        # Calculate correlation in each regime with sample size weighting
        regime_correlations = []
        regime_weights = []
        
        for regime in regime_clean.unique():
            regime_mask = regime_clean == regime
            regime_size = regime_mask.sum()
            
            if regime_size < 5:  # Need minimum samples per regime
                continue
                
            regime_corr = abs(feature_clean[regime_mask].corr(target_clean[regime_mask]))
            if not np.isnan(regime_corr):
                regime_correlations.append(regime_corr)
                regime_weights.append(regime_size)  # Weight by sample size

        if not regime_correlations:
            return 0.0

        # Weighted consistency: penalize high variance across regimes
        if len(regime_correlations) > 1:
            # Calculate weighted mean and penalize variance
            weights = np.array(regime_weights)
            weights = weights / weights.sum()  # Normalize weights
            
            weighted_mean = np.average(regime_correlations, weights=weights)
            weighted_std = np.sqrt(np.average((regime_correlations - weighted_mean)**2, weights=weights))
            
            # Consistency = 1 / (1 + coefficient of variation)
            if weighted_mean > 0:
                cv = weighted_std / weighted_mean
                consistency = 1 / (1 + cv)
            else:
                consistency = 0.0
        else:
            # Single regime - assume consistent
            consistency = 1.0

        return min(1.0, max(0.0, consistency))

    def _calculate_temporal_stability(self,
                                    feature_values: pd.Series,
                                    target_values: pd.Series) -> float:
        """Calculate temporal stability of the feature-target relationship."""
        # Get valid indices
        valid_indices = ~(feature_values.isna() | target_values.isna())
        if valid_indices.sum() < 20:
            return 0.0

        feature_clean = feature_values[valid_indices]
        target_clean = target_values[valid_indices]

        # Calculate rolling correlation over time
        window_size = min(50, len(feature_clean) // 4)
        if window_size < 10:
            return 0.0  # Not enough data for stability assessment

        rolling_correlations = []
        for i in range(window_size, len(feature_clean)):
            window_feature = feature_clean.iloc[i-window_size:i]
            window_target = target_clean.iloc[i-window_size:i]
            corr = abs(window_feature.corr(window_target))
            if not np.isnan(corr):
                rolling_correlations.append(corr)

        if not rolling_correlations:
            return 0.0

        # Stability = inverse of coefficient of variation
        mean_corr = np.mean(rolling_correlations)
        std_corr = np.std(rolling_correlations)
        
        if mean_corr == 0:
            return 0.0

        cv = std_corr / abs(mean_corr)
        stability = 1 / (1 + cv)
        
        return min(1.0, max(0.0, stability))

    def _get_stable_data_hash(self, data: pd.DataFrame) -> str:
        """Get stable hash for data based on shape and index."""
        try:
            # Use deterministic hash based on data characteristics
            index_min = data.index.min() if hasattr(data.index, 'min') else 0
            index_max = data.index.max() if hasattr(data.index, 'max') else 0
            return f"{len(data)}_{data.shape[1]}_{index_min}_{index_max}"
        except Exception:
            return f"{len(data)}_{data.shape[1]}"

    def _get_stable_analyst_hash(self, analyst_signals: Optional[pd.Series]) -> str:
        """Get stable hash for analyst signals."""
        if analyst_signals is None:
            return "none"
        try:
            # Use pandas hash for stable hashing
            return str(pd.util.hash_pandas_object(analyst_signals, index=True).sum())
        except Exception:
            return f"{len(analyst_signals)}_{analyst_signals.index.min()}_{analyst_signals.index.max()}"

    def _calculate_overall_score(self,
                               complementary_score: float,
                               regime_consistency: float,
                               temporal_stability: float) -> float:
        """Calculate overall optimization score."""
        # Weighted combination with emphasis on complementary information
        overall_score = (
            complementary_score * 0.5 +  # Primary: complementary information
            regime_consistency * self.config.regime_consistency_weight +
            temporal_stability * self.config.temporal_stability_weight
        )
        
        return min(1.0, max(0.0, overall_score))

    def _calculate_final_metrics(self,
                               generator: 'FeatureGenerator',
                               data: pd.DataFrame,
                               target_column: str,
                               optimal_lookback: int,
                               analyst_signals: Optional[pd.Series] = None,
                               regime_series: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate final metrics for the optimal lookback."""
        # Generate feature with optimal lookback
        if generator.supports_lookback_optimization():
            result = generator.generate_with_lookback(data, optimal_lookback)
        else:
            result = generator.generate(data)

        if not result.success:
            return {
                'complementary_score': 0.0,
                'analyst_alignment_score': 0.0,
                'regime_consistency_score': 0.0,
                'temporal_stability_score': 0.0,
                'regime_performance': {},
                'temporal_performance': [],
                'complementary_info_gain': 0.0
            }

        # Calculate all metrics
        complementary_score = self._calculate_complementary_score(
            result.data, data[target_column], analyst_signals
        )
        
        regime_consistency = self._calculate_regime_consistency(
            result.data, data[target_column], regime_series
        )
        
        temporal_stability = self._calculate_temporal_stability(
            result.data, data[target_column]
        )

        # Calculate analyst alignment (for reporting, not optimization)
        analyst_alignment = 0.0
        if analyst_signals is not None:
            valid_indices = ~(result.data.isna() | analyst_signals.isna())
            if valid_indices.sum() > 10:
                analyst_alignment = abs(result.data[valid_indices].corr(analyst_signals[valid_indices]))

        # Calculate regime performance
        regime_performance = {}
        if regime_series is not None:
            for regime in regime_series.unique():
                regime_mask = regime_series == regime
                if regime_mask.sum() > 5:
                    regime_corr = abs(result.data[regime_mask].corr(data[target_column][regime_mask]))
                    regime_performance[str(regime)] = regime_corr if not np.isnan(regime_corr) else 0.0

        # Calculate temporal performance
        temporal_performance = []
        window_size = min(50, len(result.data) // 4)
        if window_size >= 10:
            for i in range(window_size, len(result.data)):
                window_feature = result.data.iloc[i-window_size:i]
                window_target = data[target_column].iloc[i-window_size:i]
                corr = abs(window_feature.corr(window_target))
                if not np.isnan(corr):
                    temporal_performance.append(corr)

        # Calculate complementary info gain
        complementary_info_gain = complementary_score - analyst_alignment if analyst_signals is not None else complementary_score

        return {
            'complementary_score': complementary_score,
            'analyst_alignment_score': analyst_alignment,
            'regime_consistency_score': regime_consistency,
            'temporal_stability_score': temporal_stability,
            'regime_performance': regime_performance,
            'temporal_performance': temporal_performance,
            'complementary_info_gain': complementary_info_gain
        }

    def _calculate_confidence_interval(self, scores: List[float], confidence: float = 0.95) -> tuple:
        """Calculate confidence interval for scores."""
        if not scores or len(scores) < 2:
            return (0.0, 0.0)

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        n = len(scores)

        # Use t-distribution for small samples
        if n < 30:
            try:
                from scipy.stats import t
                t_val = t.ppf((1 + confidence) / 2, n - 1)
            except ImportError:
                t_val = 2.0  # Fallback
        else:
            try:
                from scipy.stats import norm
                t_val = norm.ppf((1 + confidence) / 2)
            except ImportError:
                t_val = 1.96  # Fallback

        margin_error = t_val * (std_score / np.sqrt(n))

        return (mean_score - margin_error, mean_score + margin_error)

    def optimize_multiple_features(self,
                                 generators: List['FeatureGenerator'],
                                 data: pd.DataFrame,
                                 target_column: str,
                                 analyst_signals: Optional[pd.Series] = None,
                                 regime_series: Optional[pd.Series] = None) -> Dict[str, int]:
        """
        Optimize lookback periods for multiple features using complementary scoring.

        Args:
            generators: List of feature generators
            data: Input data
            target_column: Target column
            analyst_signals: Optional analyst signals for complementary scoring
            regime_series: Optional regime assignments as pd.Series for regime-invariant optimization

        Returns:
            Dictionary mapping feature names to optimal lookback periods
        """
        self.logger.info(f"Optimizing {len(generators)} features using complementary scoring")
        tprint(f"🔧 Optimizing {len(generators)} features using complementary scoring")

        results = {}

        if self.config.parallel_processing and len(generators) > 1:
            # Parallel optimization
            tprint(f"⚡ Using parallel processing with {self.config.max_workers} workers")
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_generator = {
                    executor.submit(self.optimize_lookback, gen, data, target_column, analyst_signals, regime_series): gen
                    for gen in generators
                }

                for future in as_completed(future_to_generator):
                    generator = future_to_generator[future]
                    try:
                        optimal_lookback = future.result()
                        results[generator.config.name] = optimal_lookback
                    except Exception as e:
                        self.logger.error(f"Error optimizing {generator.config.name}: {e}")
                        results[generator.config.name] = generator.config.default_lookback
        else:
            # Sequential optimization
            tprint("🔄 Using sequential optimization")
            for generator in generators:
                try:
                    optimal_lookback = self.optimize_lookback(generator, data, target_column, analyst_signals, regime_series)
                    results[generator.config.name] = optimal_lookback
                except Exception as e:
                    self.logger.error(f"Error optimizing {generator.config.name}: {e}")
                    results[generator.config.name] = generator.config.default_lookback

        self.logger.info(f"Completed complementary optimization for {len(results)} features")
        tprint(f"✅ Completed complementary optimization for {len(results)} features")
        return results

    def get_optimization_summary(self, results: Dict[str, int]) -> Dict[str, Any]:
        """Generate a summary of complementary optimization results."""
        if not results:
            return {}

        lookbacks = list(results.values())

        summary = {
            'total_features': len(results),
            'lookback_distribution': {
                'mean': np.mean(lookbacks),
                'median': np.median(lookbacks),
                'std': np.std(lookbacks),
                'min': np.min(lookbacks),
                'max': np.max(lookbacks)
            },
            'optimization_approach': 'complementary_scoring',
            'regime_approach': 'regime_invariant',
            'recommendations': []
        }

        # Generate recommendations
        high_lookback_features = [name for name, lookback in results.items() if lookback > 50]
        low_lookback_features = [name for name, lookback in results.items() if lookback < 10]

        if high_lookback_features:
            summary['recommendations'].append(
                f"Features with high lookback periods (>50): {high_lookback_features}"
            )

        if low_lookback_features:
            summary['recommendations'].append(
                f"Features with low lookback periods (<10): {low_lookback_features}"
            )

        return summary

# Convenience functions
def optimize_complementary_lookbacks(generators: List['FeatureGenerator'],
                                   data: pd.DataFrame,
                                   target_column: str,
                                   analyst_signals: Optional[pd.Series] = None,
                                   config: Optional[ComplementaryOptimizationConfig] = None,
                                   regime_series: Optional[pd.Series] = None) -> Dict[str, int]:
    """
    Optimize lookback periods for multiple features using complementary scoring.

    Args:
        generators: List of feature generators
        data: Input data
        target_column: Target column
        analyst_signals: Optional analyst signals for complementary scoring
        config: Optimization configuration
        regime_series: Optional regime assignments as pd.Series for regime-invariant optimization

    Returns:
        Dictionary mapping feature names to optimal lookback periods
    """
    optimizer = ComplementaryLookbackOptimizer(config)
    return optimizer.optimize_multiple_features(generators, data, target_column, analyst_signals, regime_series)

def get_complementary_optimization_config(**kwargs) -> ComplementaryOptimizationConfig:
    """
    Create a complementary optimization configuration with the given parameters.

    Args:
        **kwargs: Configuration parameters

    Returns:
        Complementary optimization configuration
    """
    return ComplementaryOptimizationConfig(**kwargs)
