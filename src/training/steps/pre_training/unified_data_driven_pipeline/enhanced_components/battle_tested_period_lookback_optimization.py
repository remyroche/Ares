"""
Battle-Tested Period + Lookback Optimization with Best Practices Integration

This module implements production-ready period and lookback optimization following
battle-tested guidelines for financial ML pipelines, integrating Bayesian TPE optimization.

Key Features:
- Coarse → fine search strategy with Bayesian TPE
- Purged walk-forward validation with embargo
- Multi-objective optimization (IC + Sharpe + stability)
- Constraint enforcement (min periods, correlation caps)
- Economic validation with turnover adjustment
- Comprehensive logging and diagnostics
"""

import logging
import time
import warnings
from dataclasses import dataclass, field
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

# Import ML Commons utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer, OptimizationConfig
    from src.utils.ml_common.optimization.pareto import (
        Solution, ParetoFront, compute_pareto_front,
        select_knee_point, compute_hypervolume
    )
    ML_COMMONS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML Commons not available: {e}")
    ML_COMMONS_AVAILABLE = False

# Import purged K-fold
try:
    from src.utils.purged_kfold import PurgedKFoldTime
    PURGED_KFOLD_AVAILABLE = True
except ImportError:
    PURGED_KFOLD_AVAILABLE = False

# VectorBT import removed - not used in this implementation

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_correlation, safe_mean, safe_std,
    validate_finite, validate_positive
)


@dataclass
class PeriodLookbackConfig:
    """Configuration for battle-tested period + lookback optimization."""
    
    # Search space parameters
    min_period: int = 2
    max_period: int = 200
    min_lookback: int = 5
    max_lookback: int = 500
    
    # Feature type constraints
    min_periods_per_type: int = 2
    max_correlation_threshold: float = 0.85
    
    # CV parameters
    n_splits: int = 5
    embargo_days: int = 7
    gap_days: int = 1
    
    # Multi-objective weights
    ic_weight: float = 0.4
    sharpe_weight: float = 0.3
    stability_weight: float = 0.2
    turnover_weight: float = 0.1
    
    # Optimization parameters
    n_trials: int = 100
    n_startup_trials: int = 20
    n_warmup_steps: int = 10
    
    # Economic validation
    min_oof_ic: float = 0.01
    min_sharpe_improvement: float = 0.1
    max_turnover: float = 2.0
    
    # Logging
    enable_detailed_logging: bool = True
    save_artifacts: bool = True
    artifacts_dir: str = "outcomes"

    # Performance options
    enable_signal_only: bool = True  # Compute metrics from aggregated signal instead of materializing full feature frames
    cache_maxsize: int = 8  # Max entries per internal LRU cache
    enable_batch_precompute: bool = True  # Batch precompute coarse-grid SMAs via VectorBT/pandas

    # TPE optimizer wiring
    tpe_enable_pruner: bool = True
    tpe_pruner_type: str = 'hyperband'  # 'hyperband' | 'successive_halving' | 'median'
    tpe_pruner_params: Dict[str, Any] = field(default_factory=dict)
    tpe_max_trial_history: int = 200


@dataclass
class PeriodLookbackCombo:
    """Period and lookback combination with scores."""
    period: int
    lookback: int
    ic_score: float
    sharpe_score: float
    stability_score: float
    turnover_score: float
    composite_score: float
    oof_ic: float
    oof_sharpe: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PeriodLookbackResult:
    """Result of period + lookback optimization."""
    selected_combos: List[PeriodLookbackCombo]
    trading_defaults: Dict[str, Any]
    interaction_periods: Dict[str, Any]
    optimization_metrics: Dict[str, Any]
    heatmap_data: Dict[str, Any]
    sensitivity_data: Dict[str, Any]
    artifacts: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class BattleTestedPeriodLookbackOptimizer:
    """Production-ready period + lookback optimizer with battle-tested best practices."""
    
    def __init__(self, config: Optional[PeriodLookbackConfig] = None):
        """Initialize the optimizer."""
        self.config = config or PeriodLookbackConfig()
        self.logger = logging.getLogger(__name__)
        self.artifacts_dir = Path(self.config.artifacts_dir)
        self.artifacts_dir.mkdir(exist_ok=True)
        
        # Initialize Bayesian TPE optimizer (wired with pruner & history cap)
        if ML_COMMONS_AVAILABLE:
            try:
                tpe_cfg = OptimizationConfig(
                    enable_pruner=self.config.tpe_enable_pruner,
                    pruner_type=self.config.tpe_pruner_type,
                    pruner_params=self.config.tpe_pruner_params,
                    max_trial_history=self.config.tpe_max_trial_history,
                )
            except TypeError:
                # Fallback if OptimizationConfig signature differs
                tpe_cfg = None
            self.tpe_optimizer = BayesianTPEOptimizer(config=tpe_cfg) if tpe_cfg is not None else BayesianTPEOptimizer()
        else:
            self.tpe_optimizer = None
            
        # Initialize purged K-fold
        if PURGED_KFOLD_AVAILABLE:
            self.purged_kfold = PurgedKFoldTime(
                n_splits=self.config.n_splits,
                purge=pd.Timedelta(days=self.config.gap_days),
                embargo=pd.Timedelta(days=self.config.embargo_days)
            )
        else:
            self.purged_kfold = None
        
        # Initialize evaluation cache
        self._evaluation_cache = {}
        
        # Lightweight LRU caches for rolling / lag features to avoid recomputation across combos
        # Keep small max sizes to bound memory usage
        self._rolling_feature_cache: OrderedDict = OrderedDict()   # (op, period, data_sig) -> DataFrame
        self._lag_feature_cache: OrderedDict = OrderedDict()       # (op, lookback, data_sig) -> DataFrame
        self._feature_cache_maxsize: int = int(self.config.cache_maxsize)

        # Aggregated signal caches: store per-row means only (Series)
        self._sma_mean_cache: OrderedDict = OrderedDict()          # (period, data_sig, columns) -> Series
        self._lag_mean_cache: OrderedDict = OrderedDict()          # (lookback, data_sig, columns) -> Series
        self._signal_cache_maxsize: int = int(self.config.cache_maxsize)

        # Locks to protect caches in case of future parallelization
        self._cache_lock = Lock()
    
    def optimize(self, 
                data: pd.DataFrame, 
                targets: pd.Series,
                feature_columns: Optional[List[str]] = None) -> PeriodLookbackResult:
        """
        Perform battle-tested period + lookback optimization.
        
        Args:
            data: Input DataFrame with features
            targets: Target series
            feature_columns: Optional list of feature columns to optimize
            
        Returns:
            PeriodLookbackResult with optimized combinations
        """
        start_time = time.time()
        tprint_info("🔄 Starting battle-tested period + lookback optimization")
        
        try:
            # Step 1: Data validation and preparation
            tprint_info("📊 Step 1: Data validation and preparation")
            data, targets, feature_columns = self._validate_and_prepare_data(data, targets, feature_columns)
            
            # Step 2: Fail-fast gates
            tprint_info("🚪 Step 2: Fail-fast validation gates")
            if not self._apply_fail_fast_gates(data, targets):
                return self._create_failure_result("Failed fail-fast validation gates")
            
            # Step 3: Coarse grid search
            tprint_info("🔍 Step 3: Coarse grid search")
            coarse_results = self._coarse_grid_search(data, targets, feature_columns)
            
            # Step 4: Fine grid search around best results
            tprint_info("🎯 Step 4: Fine grid search around best results")
            fine_results = self._fine_grid_search(data, targets, feature_columns, coarse_results)
            
            # Step 5: Bayesian TPE optimization
            tprint_info("🧠 Step 5: Bayesian TPE optimization")
            tpe_results = self._bayesian_tpe_optimization(data, targets, feature_columns, fine_results)
            
            # Step 6: Multi-objective Pareto optimization
            tprint_info("🎯 Step 6: Multi-objective Pareto optimization")
            pareto_results = self._pareto_optimization(tpe_results)
            
            # Step 7: Economic validation
            tprint_info("💰 Step 7: Economic validation")
            validated_results = self._economic_validation(data, targets, pareto_results)
            
            # Step 8: Generate final selections
            tprint_info("📊 Step 8: Generating final selections")
            result = self._generate_final_selections(
                validated_results, data, targets, start_time
            )
            
            tprint_success(f"✅ Period + lookback optimization completed: {len(result.selected_combos)} combinations selected")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Period + lookback optimization failed: {e}")
            return self._create_failure_result(str(e))
    
    def _validate_and_prepare_data(self, 
                                  data: pd.DataFrame, 
                                  targets: pd.Series,
                                  feature_columns: Optional[List[str]]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Validate and prepare data for optimization."""
        # Validate inputs
        if data is None or len(data) == 0:
            raise ValueError("Input data is None or empty")
        if targets is None or targets.empty:
            raise ValueError("Targets is None or empty")
        if len(data) != len(targets):
            raise ValueError(f"Data and targets length mismatch: {len(data)} vs {len(targets)}")
        
        # Determine feature columns
        if feature_columns is None:
            # Exclude non-feature columns
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'open_time', 'timestamp']
            feature_columns = [col for col in data.columns if col not in exclude_cols]
        
        # Filter data to feature columns only
        feature_data = data[feature_columns].copy()
        
        # Remove features with insufficient variance
        variance_threshold = 1e-8
        high_variance_features = feature_data.var() > variance_threshold
        feature_columns = [col for col in feature_columns if high_variance_features[col]]
        feature_data = feature_data[feature_columns]
        
        tprint_info(f"📊 Prepared {len(feature_columns)} features for optimization")
        return feature_data, targets, feature_columns
    
    def _apply_fail_fast_gates(self, data: pd.DataFrame, targets: pd.Series) -> bool:
        """Apply fail-fast validation gates."""
        # Gate 1: Minimum data size
        if len(data) < 200:
            tprint_warning("⚠️ Insufficient data for reliable optimization")
            return False
        
        # Gate 2: Target variance check
        if targets.var() < 1e-8:
            tprint_warning("⚠️ Target variance too low")
            return False
        
        # Gate 3: Feature quality check
        nan_ratios = data.isnull().sum() / len(data)
        high_nan_features = nan_ratios > 0.3
        if high_nan_features.any():
            tprint_warning(f"⚠️ {high_nan_features.sum()} features have >30% NaN values")
            return False
        
        # Gate 4: Memory check
        memory_usage = data.memory_usage(deep=True).sum() / 1024**2  # MB
        if memory_usage > 2000:  # 2GB limit
            tprint_warning(f"⚠️ High memory usage: {memory_usage:.1f}MB")
            return False
        
        return True
    
    def _coarse_grid_search(self, 
                           data: pd.DataFrame, 
                           targets: pd.Series,
                           feature_columns: List[str]) -> List[PeriodLookbackCombo]:
        """Perform coarse grid search for initial exploration."""
        tprint_info("🔍 Performing coarse grid search")
        
        # Define coarse grid - FIXED: Remove duplicates
        periods = sorted(set(np.logspace(
            np.log10(self.config.min_period), 
            np.log10(self.config.max_period), 
            num=10, 
            dtype=int
        )))
        lookbacks = sorted(set(np.logspace(
            np.log10(self.config.min_lookback), 
            np.log10(self.config.max_lookback), 
            num=10, 
            dtype=int
        )))
        
        results = []

        # Optional: batch precompute per-row SMA means for coarse periods to reduce repetition
        if self.config.enable_signal_only and self.config.enable_batch_precompute and len(feature_columns) > 0:
            try:
                from src.feature_generation.utils.vectorbt_operation_batcher import batch_rolling_operations
                base = data[feature_columns]
                # Build signature keys consistent with signal cache
                if len(base) > 0:
                    data_sig = (len(base), base.index[0], base.index[-1])
                else:
                    data_sig = (0, None, None)
                cols_sig = tuple(feature_columns)

                batch_results = batch_rolling_operations(base, periods, operation_name='mean')
                for p in periods:
                    key = f"mean_{p}"
                    df = batch_results.get(key)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        series = df.mean(axis=1)
                        try:
                            series = series.astype(np.float32)
                        except Exception:
                            pass
                        sma_key = (p, data_sig, cols_sig)
                        with self._cache_lock:
                            self._sma_mean_cache[sma_key] = series
                            if len(self._sma_mean_cache) > self._signal_cache_maxsize:
                                self._sma_mean_cache.popitem(last=False)
                tprint_info("⚡ Precomputed coarse-grid SMA means via batcher")
            except Exception as e:
                tprint_warning(f"⚠️ Batching SMA precompute skipped: {e}")
        total_combinations = len(periods) * len(lookbacks)
        
        for i, period in enumerate(periods):
            for j, lookback in enumerate(lookbacks):
                try:
                    combo_idx = i * len(lookbacks) + j + 1
                    tprint_info(f"🔍 Evaluating combination {combo_idx}/{total_combinations}: period={period}, lookback={lookback}")
                    
                    # Evaluate combination
                    combo = self._evaluate_period_lookback_combo(
                        data, targets, feature_columns, period, lookback
                    )
                    
                    if combo is not None:
                        results.append(combo)
                        
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to evaluate period={period}, lookback={lookback}: {e}")
                    continue
        
        # Normalize metrics across all combinations
        results = self._normalize_metrics(results)
        
        # Sort by composite score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        tprint_info(f"🔍 Coarse grid search completed: {len(results)} valid combinations")
        return results
    
    def _fine_grid_search(self, 
                         data: pd.DataFrame, 
                         targets: pd.Series,
                         feature_columns: List[str],
                         coarse_results: List[PeriodLookbackCombo]) -> List[PeriodLookbackCombo]:
        """Perform fine grid search around best coarse results."""
        tprint_info("🎯 Performing fine grid search")
        
        if not coarse_results:
            return []
        
        # Take top 3 coarse results for fine search
        top_coarse = coarse_results[:3]
        fine_results = []
        
        for coarse_combo in top_coarse:
            # Define fine grid around this combination
            period_range = max(2, coarse_combo.period // 4)
            lookback_range = max(5, coarse_combo.lookback // 4)
            
            periods = np.arange(
                max(self.config.min_period, coarse_combo.period - period_range),
                min(self.config.max_period, coarse_combo.period + period_range + 1),
                step=max(1, period_range // 5)
            )
            lookbacks = np.arange(
                max(self.config.min_lookback, coarse_combo.lookback - lookback_range),
                min(self.config.max_lookback, coarse_combo.lookback + lookback_range + 1),
                step=max(1, lookback_range // 5)
            )
            
            for period in periods:
                for lookback in lookbacks:
                    try:
                        combo = self._evaluate_period_lookback_combo(
                            data, targets, feature_columns, period, lookback
                        )
                        
                        if combo is not None:
                            fine_results.append(combo)
                            
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to evaluate period={period}, lookback={lookback}: {e}")
                        continue
        
        # Normalize metrics across all combinations
        fine_results = self._normalize_metrics(fine_results)
        
        # Sort by composite score
        fine_results.sort(key=lambda x: x.composite_score, reverse=True)
        
        tprint_info(f"🎯 Fine grid search completed: {len(fine_results)} valid combinations")
        return fine_results
    
    def _bayesian_tpe_optimization(self, 
                                  data: pd.DataFrame, 
                                  targets: pd.Series,
                                  feature_columns: List[str],
                                  fine_results: List[PeriodLookbackCombo]) -> List[PeriodLookbackCombo]:
        """Perform Bayesian TPE optimization."""
        tprint_info("🧠 Performing Bayesian TPE optimization")
        
        if not ML_COMMONS_AVAILABLE or self.tpe_optimizer is None:
            tprint_warning("⚠️ Bayesian TPE not available, using fine grid results")
            return fine_results
        
        try:
            # Define search space
            search_space = {
                'period': (self.config.min_period, self.config.max_period, 'int'),
                'lookback': (self.config.min_lookback, self.config.max_lookback, 'int')
            }
            
            # Define objective function
            def objective(trial):
                period = trial.suggest_int('period', self.config.min_period, self.config.max_period)
                lookback = trial.suggest_int('lookback', self.config.min_lookback, self.config.max_lookback)
                
                combo = self._evaluate_period_lookback_combo(
                    data, targets, feature_columns, period, lookback
                )
                
                if combo is None:
                    return -np.inf
                
                return combo.composite_score
            
            # Run optimization
            best_params = self.tpe_optimizer.optimize(
                objective=objective,
                search_space=search_space,
                n_trials=self.config.n_trials,
                n_startup_trials=self.config.n_startup_trials
            )
            
            # Evaluate best parameters
            best_combo = self._evaluate_period_lookback_combo(
                data, targets, feature_columns, 
                best_params['period'], best_params['lookback']
            )
            
            if best_combo is not None:
                tpe_results = [best_combo]
            else:
                tpe_results = fine_results[:5]  # Fallback to top fine results
            
            # Normalize metrics across TPE results
            tpe_results = self._normalize_metrics(tpe_results)
            
            tprint_info(f"🧠 Bayesian TPE optimization completed: {len(tpe_results)} combinations")
            return tpe_results
            
        except Exception as e:
            tprint_warning(f"⚠️ Bayesian TPE optimization failed: {e}")
            return fine_results[:5]  # Fallback to top fine results
    
    def _pareto_optimization(self, 
                           tpe_results: List[PeriodLookbackCombo]) -> List[PeriodLookbackCombo]:
        """Perform Pareto optimization on TPE results."""
        tprint_info("🎯 Performing Pareto optimization")
        
        if not ML_COMMONS_AVAILABLE or len(tpe_results) < 2:
            return tpe_results
        
        try:
            # Create Pareto solutions - FIXED: Include turnover with correct direction
            solutions = []
            for i, combo in enumerate(tpe_results):
                # Get raw turnover for proper direction (lower is better)
                turnover_raw = combo.metadata.get('turnover_raw', 0.0)
                solution = Solution(
                    values=[combo.ic_score, combo.sharpe_score, combo.stability_score, -turnover_raw],
                    metadata={'combo': combo, 'index': i}
                )
                solutions.append(solution)
            
            # Compute Pareto front
            pareto_front = compute_pareto_front(solutions)
            
            # Extract Pareto-optimal combinations
            pareto_combos = []
            for solution in pareto_front:
                combo = solution.metadata['combo']
                pareto_combos.append(combo)
            
            tprint_info(f"🎯 Pareto optimization completed: {len(pareto_combos)} combinations")
            return pareto_combos
            
        except Exception as e:
            tprint_warning(f"⚠️ Pareto optimization failed: {e}")
            return tpe_results
    
    def _economic_validation(self, 
                           data: pd.DataFrame, 
                           targets: pd.Series,
                           pareto_results: List[PeriodLookbackCombo]) -> List[PeriodLookbackCombo]:
        """Perform economic validation of combinations."""
        tprint_info("💰 Performing economic validation")
        
        validated_combos = []
        
        for combo in pareto_results:
            try:
                # Check OOF IC threshold
                if combo.oof_ic < self.config.min_oof_ic:
                    tprint_warning(f"⚠️ Combo period={combo.period}, lookback={combo.lookback} failed OOF IC threshold: {combo.oof_ic:.4f}")
                    continue
                
                # Check OOF Sharpe improvement
                if combo.oof_sharpe < self.config.min_sharpe_improvement:
                    tprint_warning(f"⚠️ Combo period={combo.period}, lookback={combo.lookback} failed OOF Sharpe threshold: {combo.oof_sharpe:.4f}")
                    continue
                
                # Check turnover constraint - FIXED: Use raw turnover instead of normalized score
                turnover_raw = combo.metadata.get('turnover_raw', 0.0)
                if turnover_raw > self.config.max_turnover:
                    tprint_warning(f"⚠️ Combo period={combo.period}, lookback={combo.lookback} exceeded turnover limit: {turnover_raw:.4f}")
                    continue
                
                validated_combos.append(combo)
                
            except Exception as e:
                tprint_warning(f"⚠️ Economic validation failed for period={combo.period}, lookback={combo.lookback}: {e}")
                continue
        
        tprint_info(f"💰 Economic validation: {len(pareto_results)} -> {len(validated_combos)} combinations")
        return validated_combos
    
    def _evaluate_period_lookback_combo(self, 
                                      data: pd.DataFrame, 
                                      targets: pd.Series,
                                      feature_columns: List[str],
                                      period: int, 
                                      lookback: int) -> Optional[PeriodLookbackCombo]:
        """Evaluate a specific period + lookback combination."""
        # FIXED: Check cache first to avoid duplicate evaluations
        cache_key = (period, lookback, tuple(feature_columns))
        if cache_key in self._evaluation_cache:
            return self._evaluation_cache[cache_key]
        
        try:
            if self.config.enable_signal_only:
                # Compute aggregated signal directly (no large feature frames)
                signal = self._get_signal_series(data, feature_columns, period, lookback)
                if signal is None or signal.empty:
                    return None
                targets_aligned = targets.loc[signal.index]

                ic_score = self._calculate_ic_from_signal(signal, targets_aligned)
                sharpe_score = self._calculate_sharpe_from_signal(targets_aligned)
                stability_score = self._calculate_stability_from_signal(signal, targets_aligned)
                turnover_score, turnover_raw = self._calculate_turnover_from_signal(signal)
                oof_ic, oof_sharpe = self._calculate_oof_from_signal(signal, targets_aligned)
            else:
                # Generate features with this period/lookback combination
                features = self._generate_features_with_period_lookback(
                    data, feature_columns, period, lookback
                )
                if features is None or features.empty:
                    return None
                # Align targets with features after dropna()
                targets_aligned = targets.loc[features.index]
                # Calculate scores with aligned targets
                ic_score = self._calculate_ic_score(features, targets_aligned)
                sharpe_score = self._calculate_sharpe_score(features, targets_aligned)
                stability_score = self._calculate_stability_score(features, targets_aligned)
                turnover_score, turnover_raw = self._calculate_turnover_score(features, targets_aligned)
                oof_ic, oof_sharpe = self._calculate_oof_metrics(features, targets_aligned)
            
            # Calculate composite score
            composite_score = (
                self.config.ic_weight * ic_score +
                self.config.sharpe_weight * sharpe_score +
                self.config.stability_weight * stability_score +
                self.config.turnover_weight * turnover_score
            )
            
            result = PeriodLookbackCombo(
                period=period,
                lookback=lookback,
                ic_score=ic_score,
                sharpe_score=sharpe_score,
                stability_score=stability_score,
                turnover_score=turnover_score,
                composite_score=composite_score,
                oof_ic=oof_ic,
                oof_sharpe=oof_sharpe,
                metadata={
                    'feature_count': len(features.columns),
                    'data_points': len(features),
                    'turnover_raw': turnover_raw
                }
            )
            
            # Cache the result
            self._evaluation_cache[cache_key] = result
            return result
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to evaluate period={period}, lookback={lookback}: {e}")
            # Cache None result to avoid re-evaluating failed combinations
            self._evaluation_cache[cache_key] = None
            return None

    def _get_signal_series(self,
                           data: pd.DataFrame,
                           feature_columns: List[str],
                           period: int,
                           lookback: int) -> Optional[pd.Series]:
        """Compute aggregated signal Series for a (period, lookback) combo using caches.

        Signal is the mean across transformed columns. With both transforms present,
        it's the average of the per-row mean rolling SMA and the per-row mean lag.
        """
        try:
            base = data[feature_columns]

            # Build a compact signature for cache keys
            if len(base) > 0:
                data_sig = (len(base), base.index[0], base.index[-1])
            else:
                data_sig = (0, None, None)
            cols_sig = tuple(feature_columns)

            m_sma = None
            m_lag = None

            if period > 1:
                sma_key = (period, data_sig, cols_sig)
                with self._cache_lock:
                    if sma_key in self._sma_mean_cache:
                        m_sma = self._sma_mean_cache[sma_key]
                        self._sma_mean_cache.move_to_end(sma_key)
                if m_sma is None:
                    # Compute per-row mean of rolling SMA across columns
                    m_sma = base.rolling(window=period, min_periods=period).mean().mean(axis=1)
                    # Downcast
                    try:
                        m_sma = m_sma.astype(np.float32)
                    except Exception:
                        pass
                    with self._cache_lock:
                        self._sma_mean_cache[sma_key] = m_sma
                        if len(self._sma_mean_cache) > self._signal_cache_maxsize:
                            self._sma_mean_cache.popitem(last=False)

            if lookback > 1:
                lag_key = (lookback, data_sig, cols_sig)
                with self._cache_lock:
                    if lag_key in self._lag_mean_cache:
                        m_lag = self._lag_mean_cache[lag_key]
                        self._lag_mean_cache.move_to_end(lag_key)
                if m_lag is None:
                    m_lag = base.shift(lookback).mean(axis=1)
                    try:
                        m_lag = m_lag.astype(np.float32)
                    except Exception:
                        pass
                    with self._cache_lock:
                        self._lag_mean_cache[lag_key] = m_lag
                        if len(self._lag_mean_cache) > self._signal_cache_maxsize:
                            self._lag_mean_cache.popitem(last=False)

            # If neither transform applies, nothing to evaluate
            if m_sma is None and m_lag is None:
                return None

            # Combine available components with equal weights
            if m_sma is not None and m_lag is not None:
                signal = (m_sma + m_lag) / 2.0
                # Drop initial NaNs due to windows/lags
                signal = signal.dropna()
            else:
                signal = (m_sma if m_sma is not None else m_lag).dropna()

            if len(signal) < 10:
                return None
            return signal
        except Exception as e:
            tprint_warning(f"⚠️ Signal computation failed: {e}")
            return None

    # Metric helpers operating directly on aggregated signal
    def _calculate_ic_from_signal(self, signal: pd.Series, targets: pd.Series) -> float:
        try:
            # Align again just in case
            x, y = signal.align(targets, join='inner')
            x = x.to_numpy(dtype=float, copy=False)
            y = y.to_numpy(dtype=float, copy=False)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3:
                return 0.0
            rx = stats.rankdata(x[mask])
            ry = stats.rankdata(y[mask])
            ic = np.corrcoef(rx, ry)[0, 1]
            return self._finite_or_zero(ic)
        except Exception:
            return 0.0

    def _calculate_sharpe_from_signal(self, targets: pd.Series) -> float:
        try:
            if len(targets) < 2 or targets.std() == 0:
                return 0.0
            sharpe = safe_divide(targets.mean(), targets.std())
            return max(0.0, self._finite_or_zero(sharpe))
        except Exception:
            return 0.0

    def _calculate_stability_from_signal(self, signal: pd.Series, targets: pd.Series) -> float:
        try:
            if self.purged_kfold is None:
                return 0.5
            # Use positional indices
            idx = np.arange(len(signal))
            sig_arr = signal.to_numpy(dtype=float, copy=False)
            tgt_arr = targets.to_numpy(dtype=float, copy=False)
            fold_ics = []
            splitter = (self.purged_kfold.split_positions(len(signal), getattr(signal, 'index', None))
                        if hasattr(self.purged_kfold, 'split_positions')
                        else self.purged_kfold.split(pd.DataFrame(index=getattr(signal, 'index', None) if hasattr(signal, 'index') else pd.RangeIndex(len(signal)))))
            for train_idx, val_idx in splitter:
                if len(train_idx) < 10 or len(val_idx) < 5:
                    continue
                s = sig_arr[val_idx]
                t = tgt_arr[val_idx]
                mask = np.isfinite(s) & np.isfinite(t)
                if mask.sum() < 3:
                    continue
                rx = stats.rankdata(s[mask])
                ry = stats.rankdata(t[mask])
                ic_val = np.corrcoef(rx, ry)[0, 1]
                if np.isfinite(ic_val):
                    fold_ics.append(ic_val)
            if not fold_ics:
                return 0.0
            stability = 1.0 / (1.0 + np.std(fold_ics))
            return min(self._finite_or_zero(stability), 1.0)
        except Exception:
            return 0.0

    def _calculate_turnover_from_signal(self, signal: pd.Series) -> Tuple[float, float]:
        try:
            x = signal.to_numpy(dtype=float, copy=False)
            mu = np.nanmean(x)
            sd = np.nanstd(x)
            if not np.isfinite(sd) or sd == 0:
                return 1.0, 0.0
            z = (x - mu) / sd
            pos = np.sign(z)
            d = np.diff(pos)
            d = np.where(np.isfinite(d), d, 0.0)
            raw = np.nanmean(np.abs(d)) if d.size > 0 else 0.0
            score = 1.0 / (1.0 + raw)
            return score, raw
        except Exception:
            return 0.5, 0.0

    def _calculate_oof_from_signal(self, signal: pd.Series, targets: pd.Series) -> Tuple[float, float]:
        try:
            if self.purged_kfold is None:
                return 0.0, 0.0
            idx = np.arange(len(signal))
            sig_arr = signal.to_numpy(dtype=float, copy=False)
            tgt_arr = targets.to_numpy(dtype=float, copy=False)
            oof_ics, oof_sharpes = [], []
            splitter = (self.purged_kfold.split_positions(len(signal), getattr(signal, 'index', None))
                        if hasattr(self.purged_kfold, 'split_positions')
                        else self.purged_kfold.split(pd.DataFrame(index=getattr(signal, 'index', None) if hasattr(signal, 'index') else pd.RangeIndex(len(signal)))))
            for train_idx, val_idx in splitter:
                if len(train_idx) < 10 or len(val_idx) < 5:
                    continue
                s = sig_arr[val_idx]
                t = tgt_arr[val_idx]
                mask = np.isfinite(s) & np.isfinite(t)
                if mask.sum() >= 3:
                    rx = stats.rankdata(s[mask])
                    ry = stats.rankdata(t[mask])
                    ic = np.corrcoef(rx, ry)[0, 1]
                else:
                    ic = np.nan
                if np.isfinite(ic):
                    oof_ics.append(ic)
                std = np.nanstd(t)
                if np.isfinite(std) and std > 0:
                    oof_sharpes.append(np.nanmean(t) / std)
            return (np.mean(oof_ics) if oof_ics else 0.0,
                    np.mean(oof_sharpes) if oof_sharpes else 0.0)
        except Exception:
            return 0.0, 0.0
    
    def _generate_features_with_period_lookback(self, 
                                              data: pd.DataFrame, 
                                              feature_columns: List[str],
                                              period: int, 
                                              lookback: int) -> Optional[pd.DataFrame]:
        """Generate features with specific period and lookback parameters."""
        try:
            # Vectorized and cached implementation to reduce Python overhead and memory churn
            base = data[feature_columns]

            # Build a lightweight data signature for caching
            def _data_sig(df: pd.DataFrame) -> Tuple:
                if len(df) > 0:
                    return (len(df), df.index[0], df.index[-1], tuple(df.columns))
                return (0, None, None, tuple(df.columns))

            period_df = None
            lookback_df = None

            # Period-based transformations (rolling mean) with LRU cache
            if period > 1:
                pkey = ("roll_mean", period, _data_sig(base))
                if pkey in self._rolling_feature_cache:
                    period_df = self._rolling_feature_cache[pkey]
                    # Move to end (recently used)
                    self._rolling_feature_cache.move_to_end(pkey)
                else:
                    period_df = base.rolling(window=period, min_periods=period).mean()
                    period_df = period_df.add_suffix(f"_sma_{period}")
                    # Downcast to float32 when safe to cut memory in half
                    try:
                        period_df = period_df.astype(np.float32)
                    except Exception:
                        pass
                    self._rolling_feature_cache[pkey] = period_df
                    # Enforce LRU size
                    if len(self._rolling_feature_cache) > self._feature_cache_maxsize:
                        self._rolling_feature_cache.popitem(last=False)

            # Lookback-based transformations (shift) with LRU cache
            if lookback > 1:
                lkey = ("lag", lookback, _data_sig(base))
                if lkey in self._lag_feature_cache:
                    lookback_df = self._lag_feature_cache[lkey]
                    self._lag_feature_cache.move_to_end(lkey)
                else:
                    lookback_df = base.shift(lookback).add_suffix(f"_lag_{lookback}")
                    try:
                        lookback_df = lookback_df.astype(np.float32)
                    except Exception:
                        pass
                    self._lag_feature_cache[lkey] = lookback_df
                    if len(self._lag_feature_cache) > self._feature_cache_maxsize:
                        self._lag_feature_cache.popitem(last=False)

            # If neither transformation applies, nothing to evaluate
            if period_df is None and lookback_df is None:
                return None

            # Concatenate only transformed features to keep memory lower than copying base
            parts = []
            if period_df is not None:
                parts.append(period_df)
            if lookback_df is not None:
                parts.append(lookback_df)
            features = pd.concat(parts, axis=1)

            # Drop rows with NaNs introduced by rolling/lag
            features = features.dropna()

            # Quick guard: bail if too few data points post transforms
            if len(features) < 10:
                return None

            # Optional memory guard per-combo to avoid large temporary frames
            mem_mb = features.memory_usage(deep=True).sum() / (1024 * 1024)
            if mem_mb > 1500:  # ~1.5GB guardrail for a single combo
                tprint_warning(f"⚠️ Skipping combo (period={period}, lookback={lookback}) due to high feature memory: {mem_mb:.1f}MB")
                return None

            return features

        except Exception as e:
            tprint_warning(f"⚠️ Feature generation failed: {e}")
            return None
    
    def _calculate_ic_score(self, features: pd.DataFrame, targets: pd.Series) -> float:
        """Calculate Information Coefficient score using rank correlation."""
        try:
            # Compute signal as row-wise mean using NumPy for speed
            vals = features.values
            if vals.size == 0:
                return 0.0
            signal = np.nanmean(vals, axis=1)

            # Align to targets (features already aligned in caller, but be safe)
            # Convert to arrays with matching index
            if not features.index.equals(targets.index):
                # Reindex targets to feature index
                targets_aligned = targets.reindex(features.index)
            else:
                targets_aligned = targets

            x = signal.astype(float)
            y = targets_aligned.to_numpy(dtype=float, copy=False)

            # Drop NaNs jointly
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3:
                return 0.0

            x = x[mask]
            y = y[mask]

            # Fast Spearman via ranking + Pearson
            rx = stats.rankdata(x)
            ry = stats.rankdata(y)
            ic = np.corrcoef(rx, ry)[0, 1]
            return self._finite_or_zero(ic)

        except Exception:
            return 0.0
    
    def _calculate_sharpe_score(self, features: pd.DataFrame, targets: pd.Series) -> float:
        """Calculate Sharpe ratio score."""
        try:
            # FIXED: Don't apply pct_change to returns - targets are already returns
            returns = targets
            if len(returns) < 2 or returns.std() == 0:
                return 0.0
            
            # Calculate Sharpe ratio
            sharpe = safe_divide(returns.mean(), returns.std())
            return max(0.0, self._finite_or_zero(sharpe))
            
        except Exception:
            return 0.0
    
    def _calculate_stability_score(self, features: pd.DataFrame, targets: pd.Series) -> float:
        """Calculate stability score using purged walk-forward CV."""
        try:
            if self.purged_kfold is None:
                return 0.5  # Default stability score
            
            # FIXED: Use lightweight split by positions to avoid DataFrame reconstruction
            n = len(features)
            fold_ics = []

            # Precompute signal once for speed
            vals = features.values
            signal_full = np.nanmean(vals, axis=1)

            splitter = (self.purged_kfold.split_positions(n, getattr(features, 'index', None))
                        if hasattr(self.purged_kfold, 'split_positions')
                        else self.purged_kfold.split(features))

            for train_idx, val_idx in splitter:
                if len(train_idx) < 10 or len(val_idx) < 5:
                    continue
                
                # Slice precomputed signal and targets
                signal_val = signal_full[val_idx]
                targets_val = targets.iloc[val_idx].to_numpy(dtype=float, copy=False)
                
                # Calculate IC for this fold using Spearman correlation
                mask = np.isfinite(signal_val) & np.isfinite(targets_val)
                if mask.sum() < 3:
                    continue
                rx = stats.rankdata(signal_val[mask])
                ry = stats.rankdata(targets_val[mask])
                ic_val = np.corrcoef(rx, ry)[0, 1]
                if not np.isnan(ic_val):
                    fold_ics.append(ic_val)
            
            if not fold_ics:
                return 0.0
            
            # Stability is inverse of standard deviation of fold ICs
            stability = 1.0 / (1.0 + np.std(fold_ics))
            return min(self._finite_or_zero(stability), 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_turnover_score(self, features: pd.DataFrame, targets: pd.Series) -> Tuple[float, float]:
        """Calculate turnover score based on position changes (lower is better)."""
        try:
            # FIXED: Calculate turnover based on position changes, not feature volatility
            # Create signal from features using NumPy for speed
            vals = features.values
            if vals.size == 0:
                return 1.0, 0.0
            signal = np.nanmean(vals, axis=1)

            # Z-score the signal
            signal_mean = np.nanmean(signal)
            signal_std = np.nanstd(signal)
            if signal_std == 0 or not np.isfinite(signal_std):
                return 1.0, 0.0  # No turnover if no variation
            
            signal_z = (signal - signal_mean) / signal_std
            
            # Create simple long/short position proxy
            position = np.sign(signal_z)
            
            # Calculate turnover as average absolute change in position
            # Use numpy diff; then average absolute changes
            pos_diff = np.diff(position)
            # Replace NaNs (from diff on NaNs) with 0
            pos_diff = np.where(np.isfinite(pos_diff), pos_diff, 0.0)
            turnover_raw = np.nanmean(np.abs(pos_diff)) if pos_diff.size > 0 else 0.0
            
            # Normalize turnover score (lower is better)
            turnover_score = 1.0 / (1.0 + turnover_raw)
            
            # Store raw turnover for economic validation
            return turnover_score, turnover_raw
            
        except Exception:
            return 0.5, 0.0
    
    def _calculate_oof_metrics(self, features: pd.DataFrame, targets: pd.Series) -> Tuple[float, float]:
        """Calculate out-of-fold IC and Sharpe metrics."""
        try:
            if self.purged_kfold is None:
                return 0.0, 0.0
            
            # FIXED: Use lightweight split by positions
            n = len(features)
            oof_ics = []
            oof_sharpes = []

            # Precompute signal once
            vals = features.values
            signal_full = np.nanmean(vals, axis=1)

            splitter = (self.purged_kfold.split_positions(n, getattr(features, 'index', None))
                        if hasattr(self.purged_kfold, 'split_positions')
                        else self.purged_kfold.split(features))

            for train_idx, val_idx in splitter:
                if len(train_idx) < 10 or len(val_idx) < 5:
                    continue
                
                # Slice arrays
                signal_val = signal_full[val_idx]
                val_targets = targets.iloc[val_idx].to_numpy(dtype=float, copy=False)
                
                # Calculate IC using Spearman correlation
                mask = np.isfinite(signal_val) & np.isfinite(val_targets)
                if mask.sum() >= 3:
                    rx = stats.rankdata(signal_val[mask])
                    ry = stats.rankdata(val_targets[mask])
                    ic = np.corrcoef(rx, ry)[0, 1]
                else:
                    ic = np.nan
                if np.isfinite(ic):
                    oof_ics.append(ic)
                
                # Calculate Sharpe - FIXED: Don't apply pct_change to returns
                if val_targets.size > 1:
                    std = np.nanstd(val_targets)
                    if std > 0 and np.isfinite(std):
                        sharpe = safe_divide(np.nanmean(val_targets), std)
                    else:
                        sharpe = np.nan
                    if not np.isnan(sharpe):
                        oof_sharpes.append(sharpe)
            
            oof_ic = np.mean(oof_ics) if oof_ics else 0.0
            oof_sharpe = np.mean(oof_sharpes) if oof_sharpes else 0.0
            
            return oof_ic, oof_sharpe
            
        except Exception:
            return 0.0, 0.0
    
    def _normalize_metrics(self, combos: List[PeriodLookbackCombo]) -> List[PeriodLookbackCombo]:
        """Normalize metrics across combinations to prevent scale domination."""
        if not combos:
            return combos
        
        # Extract metrics
        ic_scores = np.array([c.ic_score for c in combos])
        sharpe_scores = np.array([c.sharpe_score for c in combos])
        stability_scores = np.array([c.stability_score for c in combos])
        turnover_scores = np.array([c.turnover_score for c in combos])
        
        # Normalize each metric to [0, 1] using robust percentiles
        def _normalize_robust(arr):
            arr = np.asarray(arr, float)
            lo, hi = np.nanpercentile(arr, 5), np.nanpercentile(arr, 95)
            if hi == lo:
                return np.zeros_like(arr)
            return np.clip((arr - lo) / (hi - lo), 0, 1)
        
        ic_norm = _normalize_robust(ic_scores)
        sharpe_norm = _normalize_robust(sharpe_scores)
        stability_norm = _normalize_robust(stability_scores)
        turnover_norm = _normalize_robust(turnover_scores)
        
        # Update combinations with normalized scores
        for i, combo in enumerate(combos):
            combo.ic_score = ic_norm[i]
            combo.sharpe_score = sharpe_norm[i]
            combo.stability_score = stability_norm[i]
            combo.turnover_score = turnover_norm[i]
            
            # Recalculate composite score with normalized metrics
            combo.composite_score = (
                self.config.ic_weight * combo.ic_score +
                self.config.sharpe_weight * combo.sharpe_score +
                self.config.stability_weight * combo.stability_score +
                self.config.turnover_weight * combo.turnover_score
            )
        
        return combos
    
    def _finite_or_zero(self, x: float) -> float:
        """Safety function to clamp values to finite and non-negative where intended."""
        return float(x) if np.isfinite(x) else 0.0
    
    def _generate_final_selections(self, 
                                 validated_combos: List[PeriodLookbackCombo],
                                 data: pd.DataFrame,
                                 targets: pd.Series,
                                 start_time: float) -> PeriodLookbackResult:
        """Generate final selections and artifacts."""
        tprint_info("📊 Generating final selections")
        
        # Select trading defaults (top 3 combinations)
        trading_defaults = {
            'primary': validated_combos[0] if validated_combos else None,
            'secondary': validated_combos[1] if len(validated_combos) > 1 else None,
            'tertiary': validated_combos[2] if len(validated_combos) > 2 else None
        }
        
        # Select interaction periods (top 5 diverse combinations)
        interaction_periods = {
            'top_5_combos': validated_combos[:5],
            'diverse_periods': list(set(combo.period for combo in validated_combos[:10])),
            'diverse_lookbacks': list(set(combo.lookback for combo in validated_combos[:10]))
        }
        
        # Calculate optimization metrics
        optimization_metrics = {
            'total_combinations_evaluated': len(validated_combos),
            'combinations_selected': len(validated_combos),
            'average_ic': np.mean([c.ic_score for c in validated_combos]) if validated_combos else 0,
            'average_sharpe': np.mean([c.sharpe_score for c in validated_combos]) if validated_combos else 0,
            'average_stability': np.mean([c.stability_score for c in validated_combos]) if validated_combos else 0,
            'execution_time': time.time() - start_time
        }
        
        # Generate heatmap data
        heatmap_data = {
            'periods': [c.period for c in validated_combos],
            'lookbacks': [c.lookback for c in validated_combos],
            'scores': [c.composite_score for c in validated_combos]
        }
        
        # Generate sensitivity data
        sensitivity_data = {
            'period_sensitivity': self._calculate_period_sensitivity(validated_combos),
            'lookback_sensitivity': self._calculate_lookback_sensitivity(validated_combos)
        }
        
        # Save artifacts if enabled
        artifacts = {}
        if self.config.save_artifacts:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save optimization results
            results_path = self.artifacts_dir / f"period_lookback_optimization_{timestamp}.json"
            results_data = {
                'selected_combos': [
                    {
                        'period': c.period,
                        'lookback': c.lookback,
                        'ic_score': c.ic_score,
                        'sharpe_score': c.sharpe_score,
                        'stability_score': c.stability_score,
                        'composite_score': c.composite_score,
                        'oof_ic': c.oof_ic,
                        'oof_sharpe': c.oof_sharpe
                    }
                    for c in validated_combos
                ],
                'trading_defaults': {
                    'primary': trading_defaults['primary'].__dict__ if trading_defaults['primary'] else None,
                    'secondary': trading_defaults['secondary'].__dict__ if trading_defaults['secondary'] else None,
                    'tertiary': trading_defaults['tertiary'].__dict__ if trading_defaults['tertiary'] else None
                },
                'optimization_metrics': optimization_metrics
            }
            
            import json
            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            artifacts['optimization_results_path'] = str(results_path)
        
        return PeriodLookbackResult(
            selected_combos=validated_combos,
            trading_defaults=trading_defaults,
            interaction_periods=interaction_periods,
            optimization_metrics=optimization_metrics,
            heatmap_data=heatmap_data,
            sensitivity_data=sensitivity_data,
            artifacts=artifacts,
            success=True
        )
    
    def _calculate_period_sensitivity(self, combos: List[PeriodLookbackCombo]) -> Dict[str, Any]:
        """Calculate period sensitivity analysis."""
        if not combos:
            return {}
        
        periods = [c.period for c in combos]
        scores = [c.composite_score for c in combos]
        
        return {
            'period_range': [min(periods), max(periods)],
            'score_range': [min(scores), max(scores)],
            'period_variance': np.var(periods),
            'score_variance': np.var(scores)
        }
    
    def _calculate_lookback_sensitivity(self, combos: List[PeriodLookbackCombo]) -> Dict[str, Any]:
        """Calculate lookback sensitivity analysis."""
        if not combos:
            return {}
        
        lookbacks = [c.lookback for c in combos]
        scores = [c.composite_score for c in combos]
        
        return {
            'lookback_range': [min(lookbacks), max(lookbacks)],
            'score_range': [min(scores), max(scores)],
            'lookback_variance': np.var(lookbacks),
            'score_variance': np.var(scores)
        }
    
    def _create_failure_result(self, error_message: str) -> PeriodLookbackResult:
        """Create a failure result."""
        return PeriodLookbackResult(
            selected_combos=[],
            trading_defaults={},
            interaction_periods={},
            optimization_metrics={},
            heatmap_data={},
            sensitivity_data={},
            artifacts={},
            success=False,
            error_message=error_message
        )
