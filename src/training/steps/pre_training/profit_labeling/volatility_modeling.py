"""
Volatility Modeling for Volatility-Aware Labeling (data-driven)

This module estimates volatility using multiple estimators on a *per-period* scale
and combines them with non-negative, sum-to-one data-driven weights that minimize
one-step-ahead absolute return prediction error, avoiding heuristics.

Key Features:
- Realized volatility (rolling std of returns)
- ATR-based volatility (True Range / close)
- EWMA volatility (EWMA of return variance)
- Strictly trailing windows (no look-ahead)
- Data-driven weight learning via projected gradient descent on the simplex
- Percentile-based flooring/capping to avoid blow-ups without arbitrary constants
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

# Optional imports
try:
    from src.utils.matrix_operations import UnifiedMatrixOperations
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False

# Enhanced utilities integration
try:
    from src.utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager, VectorizationConfig
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    from src.utils.hardware.hardware_optimizer import HardwareOptimizer
    from src.utils.hardware.enhanced_cpu_optimizer import EnhancedCPUOptimizer
    from src.utils.hardware.advanced_memory_optimizer import AdvancedMemoryOptimizer
    from src.utils.hardware.enhanced_caching_system import EnhancedCachingSystem
    from src.utils.serialization_utils import UniversalSerializer, safe_serialize, safe_deserialize
    from src.utils.ml_common.optimization.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
    ENHANCED_UTILS_AVAILABLE = True
except ImportError:
    ENHANCED_UTILS_AVAILABLE = False

from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success
)


class VolatilityMethod(Enum):
    REALIZED = "realized"
    ATR = "atr"
    EWMA = "ewma"
    COMBINED = "combined"  # data-driven combination


@dataclass
class VolatilityConfig:
    """
    Configuration for volatility modeling (all horizons in *bars*, not annualized).
    """

    # Method
    method: VolatilityMethod = VolatilityMethod.COMBINED

    # Realized volatility
    rv_window: int = 20
    rv_min_periods: int = 10

    # ATR
    atr_window: int = 14
    atr_min_periods: int = 7

    # EWMA (variance smoothing factor alpha in (0,1])
    ewma_alpha: float = 0.06
    ewma_min_periods: int = 10

    # Smoothing (trailing)
    enable_smoothing: bool = True
    smoothing_window: int = 5

    # Input checks
    min_volatility_samples: int = 50

    # Normalization (data-driven)
    use_percentile_floor_cap: bool = True
    floor_percentile: float = 0.5  # p0.5 to prevent zeros
    cap_percentile: float = 99.5   # p99.5 to cut extreme spikes
    absolute_floor: float = 1e-8   # hard lower bound if percentiles are degenerate

    # Combination training window (how far back to learn weights)
    combo_lookback: int = 252      # bars for weight estimation
    combo_max_iters: int = 800     # iterations for projected gradient
    combo_tol: float = 1e-8        # gradient stopping tolerance

    def _validate_config(self) -> None:
        if self.rv_window < 1:
            raise ValueError("rv_window must be >= 1")
        if self.atr_window < 1:
            raise ValueError("atr_window must be >= 1")
        if not (0 < self.ewma_alpha <= 1):
            raise ValueError("ewma_alpha must be in (0, 1]")
        if self.min_volatility_samples < 1:
            raise ValueError("min_volatility_samples must be >= 1")
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be >= 1")
        if not (0 < self.floor_percentile < 100):
            raise ValueError("floor_percentile must be in (0,100)")
        if not (0 < self.cap_percentile <= 100):
            raise ValueError("cap_percentile must be in (0,100]")
        if self.cap_percentile <= self.floor_percentile:
            raise ValueError("cap_percentile must be > floor_percentile")
        if self.combo_lookback < 20:
            raise ValueError("combo_lookback should be at least 20")


@dataclass
class VolatilityResult:
    volatility_series: pd.Series
    volatility_method: VolatilityMethod
    realized_volatility: Optional[pd.Series] = None
    atr_volatility: Optional[pd.Series] = None
    ewma_volatility: Optional[pd.Series] = None
    mean_volatility: float = 0.0
    volatility_std: float = 0.0
    volatility_percentiles: Dict[str, float] = field(default_factory=dict)
    volatility_consistency: float = 0.0
    volatility_stability: float = 0.0
    combo_weights: Optional[Dict[str, float]] = None
    config_used: VolatilityConfig = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class VolatilityModeler:
    """
    Volatility Modeler (data-driven)
    - Produces per-period volatility estimates (no annualization).
    - If method=COMBINED, learns non-negative, sum-to-one weights w
      that minimize MSE for predicting |r_{t+1}| from [rv_t, atr_t, ewma_t].
    """

    def __init__(self, config: Optional[VolatilityConfig] = None):
        self.config = config or VolatilityConfig()
        self.config._validate_config()
        self.logger = logging.getLogger("VolatilityModeler")

        if MATRIX_OPS_AVAILABLE:
            self.matrix_ops = UnifiedMatrixOperations()
            tprint_info("   → Matrix operations: Available")
        else:
            self.matrix_ops = None
            tprint_warning("   → Matrix operations: Not available, using pandas/numpy")

        # Initialize enhanced utilities
        if ENHANCED_UTILS_AVAILABLE:
            self.vectorization_manager = UnifiedVectorizationManager(
                VectorizationConfig(
                    enable_vectorization=True,
                    vectorization_method="numpy",
                    batch_size=1000,
                    enable_parallel_processing=True,
                    enable_optimization=True,
                    enable_gpu_acceleration=True,
                    memory_efficient=True
                )
            )
            
            self.tpe_optimizer = BayesianTPEOptimizer(
                n_trials=50,
                n_startup_trials=10,
                n_warmup_steps=25
            )
            
            self.hardware_optimizer = HardwareOptimizer()
            self.cpu_optimizer = EnhancedCPUOptimizer()
            self.memory_optimizer = AdvancedMemoryOptimizer()
            self.caching_system = EnhancedCachingSystem()
            self.vectorbt_optimizer = VectorBTRollingOptimizer()
            self.serializer = UniversalSerializer()
            
            tprint_info("   → Enhanced utilities: Available")
            tprint_info("   → VectorBTRollingOptimizer: Available")
            tprint_info("   → Hardware acceleration: Available")
        else:
            self.vectorization_manager = None
            self.tpe_optimizer = None
            self.hardware_optimizer = None
            self.cpu_optimizer = None
            self.memory_optimizer = None
            self.caching_system = None
            self.vectorbt_optimizer = None
            self.serializer = None
            tprint_warning("   → Enhanced utilities: Not available")

        # Performance tracking
        self._performance_metrics = {
            'vectorization_time': 0.0,
            'optimization_time': 0.0,
            'hardware_optimization_time': 0.0,
            'caching_time': 0.0,
            'gpu_acceleration_time': 0.0,
            'memory_optimization_time': 0.0
        }

        tprint_info("📈 Enhanced Volatility Modeler initialized")
        tprint_info(f"   → Method: {self.config.method.value}")
        tprint_info(f"   → Enhanced features: {ENHANCED_UTILS_AVAILABLE}")
    
    def model_volatility(self, bars: pd.DataFrame) -> VolatilityResult:
        start_time = datetime.now()
        tprint_info("📊 Modeling volatility with enhanced optimization")

        empty_series = pd.Series(dtype=float, index=bars.index if isinstance(bars, pd.DataFrame) else None)
        result = VolatilityResult(
            volatility_series=empty_series,
            volatility_method=self.config.method,
            config_used=self.config
        )

        try:
            if not self._validate_input_data(bars):
                return result

            # Apply hardware optimization to input data
            if self.hardware_optimizer:
                bars = self.hardware_optimizer.optimize_dataframe(bars)
                tprint_info("🔧 Applied hardware optimization to input data")

            # Compute close-to-close returns (per period) with vectorization
            close = bars["close"].astype(float)
            rets = close.pct_change().rename("returns")
            
            # Apply vectorization for better performance
            if self.vectorization_manager:
                rets = self.vectorization_manager.vectorize_data(rets)
                tprint_info("🚀 Applied vectorization to returns calculation")

            # 1) Component estimators (all on per-period scale, trailing)
            tprint_info("📈 Step 1: Calculating volatility components with optimization")
            rv = self._calculate_realized_volatility_enhanced(rets)
            atr = self._calculate_atr_volatility_enhanced(bars)
            ew = self._calculate_ewma_volatility_enhanced(rets)

            # Align components and keep common index
            comps = pd.concat({"rv": rv, "atr": atr, "ewma": ew}, axis=1).dropna(how="all")
            result.realized_volatility = comps["rv"]
            result.atr_volatility = comps["atr"]
            result.ewma_volatility = comps["ewma"]

            # 2) Choose / combine with enhanced optimization
            tprint_info("🔗 Step 2: Combining volatility estimates with TPE optimization")
            if self.config.method == VolatilityMethod.REALIZED:
                combined = comps["rv"]
                weights = {"rv": 1.0, "atr": 0.0, "ewma": 0.0}
            elif self.config.method == VolatilityMethod.ATR:
                combined = comps["atr"]
                weights = {"rv": 0.0, "atr": 1.0, "ewma": 0.0}
            elif self.config.method == VolatilityMethod.EWMA:
                combined = comps["ewma"]
                weights = {"rv": 0.0, "atr": 0.0, "ewma": 1.0}
            else:
                combined, weights = self._combine_data_driven_enhanced(comps, rets)

            # 3) Normalize (floor/cap) and optional smoothing (trailing)
            tprint_info("⚖️ Step 3: Normalizing scale (percentile floor/cap)")
            combined = self._normalize_volatility_units_enhanced(combined)

            if self.config.enable_smoothing and len(combined) >= self.config.smoothing_window:
                tprint_info("🔧 Step 4: Trailing smoothing with optimization")
                combined = self._apply_enhanced_smoothing(combined)

            # 4) Stats & quality with enhanced metrics
            tprint_info("📊 Step 5: Calculating enhanced statistics and quality metrics")
            stats = self._calculate_volatility_statistics_enhanced(combined)
            quality = self._calculate_volatility_quality_enhanced(combined)

            # Apply final hardware optimization
            if self.hardware_optimizer:
                combined = self.hardware_optimizer.optimize_series(combined)

            result.volatility_series = combined.astype(float)
            result.mean_volatility = float(stats["mean_volatility"])
            result.volatility_std = float(stats["volatility_std"])
            result.volatility_percentiles = stats["volatility_percentiles"]
            result.volatility_consistency = float(quality["consistency"])
            result.volatility_stability = float(quality["stability"])
            result.combo_weights = weights

        except Exception as e:
            tprint_error(f"❌ Enhanced volatility modeling failed: {e}")
            return result
        finally:
            result.processing_time = (datetime.now() - start_time).total_seconds()
            tprint_success("✅ Enhanced volatility modeling completed")
            tprint_info(f"   → Samples: {len(result.volatility_series)}")
            tprint_info(f"   → Mean: {result.mean_volatility:.6f}")
            tprint_info(f"   → Std: {result.volatility_std:.6f}")
            if result.combo_weights:
                tprint_info(f"   → Weights: {result.combo_weights}")
            tprint_info(f"   → Performance metrics: {self._performance_metrics}")

        return result
    
    def _validate_input_data(self, bars: pd.DataFrame) -> bool:
        try:
            if not isinstance(bars, pd.DataFrame) or bars.empty:
                tprint_warning("⚠️ Input bars are empty or not a DataFrame")
                return False

            required = {"open", "high", "low", "close"}
            missing = required - set(bars.columns)
            if missing:
                tprint_warning(f"⚠️ Missing required columns: {missing}")
                return False

            if len(bars) < self.config.min_volatility_samples:
                tprint_warning(
                    f"⚠️ Insufficient samples: {len(bars)} < {self.config.min_volatility_samples}"
                )
                return False

            if bars[list(required)].isnull().any().any():
                tprint_warning("⚠️ Data contains null values in OHLC")
                return False

            vals = bars[list(required)].to_numpy(dtype=float, copy=False)
            if not np.isfinite(vals).all():
                tprint_warning("⚠️ Data contains non-finite values")
                return False

            return True
        except Exception as e:
            tprint_error(f"❌ Data validation failed: {e}")
            return False
    
    @staticmethod
    def _rolling_std(x: pd.Series, window: int, min_periods: int) -> pd.Series:
        return x.rolling(window=window, min_periods=min_periods).std()

    def _calculate_realized_volatility(self, returns: pd.Series) -> pd.Series:
        """Per-period realized volatility: rolling std of close-to-close returns."""
        returns = returns.astype(float)
        if returns.dropna().shape[0] < self.config.rv_min_periods:
            return pd.Series(index=returns.index, dtype=float)
        rv = self._rolling_std(returns, self.config.rv_window, self.config.rv_min_periods)
        return rv.rename("rv")
    
    def _calculate_realized_volatility_enhanced(self, returns: pd.Series) -> pd.Series:
        """Enhanced realized volatility calculation with VectorBTRollingOptimizer."""
        returns = returns.astype(float)
        if returns.dropna().shape[0] < self.config.rv_min_periods:
            return pd.Series(index=returns.index, dtype=float)
        
        # Check cache first
        cache_key = f"rv_{len(returns)}_{self.config.rv_window}_{self.config.rv_min_periods}"
        if self.caching_system:
            cached_result = self.caching_system.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Use VectorBTRollingOptimizer for enhanced rolling calculations
        if self.vectorbt_optimizer:
            rv = self.vectorbt_optimizer.rolling_std(
                returns, 
                window=self.config.rv_window,
                min_periods=self.config.rv_min_periods,
                use_gpu=True
            )
        else:
            # Fallback to vectorized operations
            if self.vectorization_manager:
                returns = self.vectorization_manager.vectorize_data(returns)
            
            # Enhanced rolling std with better numerical stability
            rv = returns.rolling(
                window=self.config.rv_window, 
                min_periods=self.config.rv_min_periods
            ).std()
        
        # Apply additional smoothing for noise reduction
        if len(rv) > 10:
            rv = rv.ewm(alpha=0.1, min_periods=5).mean()
        
        # Cache the result
        if self.caching_system:
            self.caching_system.set(cache_key, rv)
        
        return rv.rename("rv")
    
    def _calculate_atr_volatility(self, bars: pd.DataFrame) -> pd.Series:
        """
        ATR-based per-period volatility: True Range divided by close (trailing mean).
        No np.roll to avoid wraparound bugs; uses pandas shift (no look-ahead).
        """
        high = bars["high"].astype(float)
        low = bars["low"].astype(float)
        close = bars["close"].astype(float)
        prev_close = close.shift(1)

        # True range components
        c1 = high - low
        c2 = (high - prev_close).abs()
        c3 = (low - prev_close).abs()
        tr = pd.concat([c1, c2, c3], axis=1).max(axis=1)

        if tr.dropna().shape[0] < self.config.atr_min_periods:
            return pd.Series(index=bars.index, dtype=float)

        atr = tr.rolling(self.config.atr_window, min_periods=self.config.atr_min_periods).mean()
        atr_vol = (atr / close).rename("atr")  # per-period magnitude in return units
        return atr_vol
    
    def _calculate_atr_volatility_enhanced(self, bars: pd.DataFrame) -> pd.Series:
        """Enhanced ATR-based volatility calculation with VectorBTRollingOptimizer."""
        # Check cache first
        cache_key = f"atr_{len(bars)}_{self.config.atr_window}_{self.config.atr_min_periods}"
        if self.caching_system:
            cached_result = self.caching_system.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        high = bars["high"].astype(float)
        low = bars["low"].astype(float)
        close = bars["close"].astype(float)
        prev_close = close.shift(1)

        # Use vectorized operations for better performance
        if self.vectorization_manager:
            high = self.vectorization_manager.vectorize_data(high)
            low = self.vectorization_manager.vectorize_data(low)
            close = self.vectorization_manager.vectorize_data(close)
            prev_close = self.vectorization_manager.vectorize_data(prev_close)

        # True range components with enhanced calculation using vectorized operations
        c1 = high - low
        c2 = (high - prev_close).abs()
        c3 = (low - prev_close).abs()
        tr = pd.concat([c1, c2, c3], axis=1).max(axis=1)

        if tr.dropna().shape[0] < self.config.atr_min_periods:
            return pd.Series(index=bars.index, dtype=float)

        # Use VectorBTRollingOptimizer for enhanced ATR calculation
        if self.vectorbt_optimizer:
            atr = self.vectorbt_optimizer.rolling_mean(
                tr, 
                window=self.config.atr_window,
                min_periods=self.config.atr_min_periods,
                use_gpu=True
            )
        else:
            # Enhanced ATR calculation with better smoothing
            atr = tr.rolling(
                window=self.config.atr_window, 
                min_periods=self.config.atr_min_periods
            ).mean()
        
        # Apply additional smoothing for noise reduction
        if len(atr) > 10:
            atr = atr.ewm(alpha=0.1, min_periods=5).mean()
        
        atr_vol = (atr / close).rename("atr")  # per-period magnitude in return units
        
        # Cache the result
        if self.caching_system:
            self.caching_system.set(cache_key, atr_vol)
        
        return atr_vol
    
    def _calculate_ewma_volatility(self, returns: pd.Series) -> pd.Series:
        """
        EWMA variance -> volatility on per-period scale.
        Uses pandas ewm (no manual padding that injects zeros).
        """
        r = returns.astype(float)
        if r.dropna().shape[0] < self.config.ewma_min_periods:
            return pd.Series(index=r.index, dtype=float)

        ew_var = r.ewm(alpha=self.config.ewma_alpha, min_periods=self.config.ewma_min_periods).var(bias=False)
        ew_vol = np.sqrt(ew_var).rename("ewma")
        return ew_vol
    
    def _calculate_ewma_volatility_enhanced(self, returns: pd.Series) -> pd.Series:
        """Enhanced EWMA volatility calculation with VectorBTRollingOptimizer."""
        # Check cache first
        cache_key = f"ewma_{len(returns)}_{self.config.ewma_alpha}_{self.config.ewma_min_periods}"
        if self.caching_system:
            cached_result = self.caching_system.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        r = returns.astype(float)
        if r.dropna().shape[0] < self.config.ewma_min_periods:
            return pd.Series(index=r.index, dtype=float)

        # Use vectorized operations for better performance
        if self.vectorization_manager:
            r = self.vectorization_manager.vectorize_data(r)

        # Use VectorBTRollingOptimizer for enhanced EWMA calculation
        if self.vectorbt_optimizer:
            ew_var = self.vectorbt_optimizer.ewm_var(
                r,
                alpha=self.config.ewma_alpha,
                min_periods=self.config.ewma_min_periods,
                bias=False,
                use_gpu=True
            )
        else:
            # Enhanced EWMA calculation with better numerical stability
            ew_var = r.ewm(
                alpha=self.config.ewma_alpha, 
                min_periods=self.config.ewma_min_periods
            ).var(bias=False)
        
        # Apply additional smoothing and clipping for stability
        ew_vol = np.sqrt(ew_var)
        
        # Clip extreme values to prevent numerical issues
        ew_vol = ew_vol.clip(lower=1e-8, upper=1.0)
        
        # Apply final smoothing
        if len(ew_vol) > 10:
            ew_vol = ew_vol.ewm(alpha=0.05, min_periods=5).mean()
        
        # Cache the result
        if self.caching_system:
            self.caching_system.set(cache_key, ew_vol)
        
        return ew_vol.rename("ewma")
    
    def _combine_data_driven(self, comps: pd.DataFrame, returns: pd.Series) -> tuple[pd.Series, Dict[str, float]]:
        """
        Learn weights w >= 0, sum(w)=1 to predict |r_{t+1}| from comps_t.
        Training uses trailing window (combo_lookback). No look-ahead.
        """
        # Regressors at t, target at t+1
        X_all = comps.dropna(how="any")
        if X_all.empty or X_all.shape[1] == 0:
            return pd.Series(index=comps.index, dtype=float), {"rv": 1/3, "atr": 1/3, "ewma": 1/3}

        y_all = returns.abs().reindex(X_all.index).shift(-1)  # |r_{t+1}|
        # Keep rows where y exists (drop tail to avoid look-ahead)
        mask = y_all.notna()
        X_all, y_all = X_all[mask], y_all[mask]

        if len(X_all) < max(30, self.config.combo_lookback // 4):
            # Not enough data to learn reliably
            w = np.ones(X_all.shape[1]) / X_all.shape[1]
        else:
            # Use last combo_lookback samples to train
            X = X_all.iloc[-self.config.combo_lookback:, :]
            y = y_all.iloc[-self.config.combo_lookback:]
            w = self._fit_simplex_pg(X.to_numpy(), y.to_numpy())

        w = self._project_to_simplex(w)  # safety

        # Build combined estimator on full timeline (no shift here; per-period vol estimate)
        combined = (X_all @ pd.Series(w, index=X_all.columns)).reindex(comps.index)

        weights = {col: float(w[i]) for i, col in enumerate(X_all.columns)}
        # If some component columns were fully NA earlier, pad their weights with 0
        for col in ["rv", "atr", "ewma"]:
            weights.setdefault(col, 0.0)

        return combined.astype(float), weights
    
    def _combine_data_driven_enhanced(self, comps: pd.DataFrame, returns: pd.Series) -> tuple[pd.Series, Dict[str, float]]:
        """Enhanced data-driven combination using TPE optimization with parallel processing."""
        # Check cache first
        cache_key = f"combo_{len(comps)}_{self.config.combo_lookback}"
        if self.caching_system:
            cached_result = self.caching_system.get(cache_key)
            if cached_result is not None:
                return cached_result['combined'], cached_result['weights']
        
        # Regressors at t, target at t+1
        X_all = comps.dropna(how="any")
        if X_all.empty or X_all.shape[1] == 0:
            return pd.Series(index=comps.index, dtype=float), {"rv": 1/3, "atr": 1/3, "ewma": 1/3}

        y_all = returns.abs().reindex(X_all.index).shift(-1)  # |r_{t+1}|
        # Keep rows where y exists (drop tail to avoid look-ahead)
        mask = y_all.notna()
        X_all, y_all = X_all[mask], y_all[mask]

        if len(X_all) < max(30, self.config.combo_lookback // 4):
            # Not enough data to learn reliably
            w = np.ones(X_all.shape[1]) / X_all.shape[1]
        else:
            # Use TPE optimization for better weight learning with parallel processing
            if self.tpe_optimizer and len(X_all) > 100:
                try:
                    # Define search space for TPE
                    search_space = {
                        'w_rv': (0.0, 1.0),
                        'w_atr': (0.0, 1.0),
                        'w_ewma': (0.0, 1.0)
                    }
                    
                    def objective(trial):
                        w = np.array([
                            trial.suggest_float('w_rv', 0.0, 1.0),
                            trial.suggest_float('w_atr', 0.0, 1.0),
                            trial.suggest_float('w_ewma', 0.0, 1.0)
                        ])
                        
                        # Normalize weights
                        w = w / (w.sum() + 1e-8)
                        
                        # Use last combo_lookback samples to train
                        X = X_all.iloc[-self.config.combo_lookback:, :]
                        y = y_all.iloc[-self.config.combo_lookback:]
                        
                        # Use vectorized operations for better performance
                        if self.vectorization_manager:
                            X = self.vectorization_manager.vectorize_data(X)
                            y = self.vectorization_manager.vectorize_data(y)
                        
                        # Calculate prediction error using vectorized operations
                        y_pred = X @ w
                        mse = np.mean((y_pred - y) ** 2)
                        return -mse  # Minimize MSE
                    
                    # Run TPE optimization with parallel processing
                    best_trial = self.tpe_optimizer.optimize(
                        objective=objective,
                        search_space=search_space,
                        n_trials=30,
                        timeout=60,
                        n_jobs=-1  # Use all available cores
                    )
                    
                    if best_trial and best_trial.value < 0:
                        w = np.array([
                            best_trial.params['w_rv'],
                            best_trial.params['w_atr'],
                            best_trial.params['w_ewma']
                        ])
                        w = w / (w.sum() + 1e-8)
                    else:
                        # Fallback to original method
                        X = X_all.iloc[-self.config.combo_lookback:, :]
                        y = y_all.iloc[-self.config.combo_lookback:]
                        w = self._fit_simplex_pg(X.to_numpy(), y.to_numpy())
                except Exception as e:
                    tprint_warning(f"⚠️ TPE optimization failed: {e}")
                    # Fallback to original method
                    X = X_all.iloc[-self.config.combo_lookback:, :]
                    y = y_all.iloc[-self.config.combo_lookback:]
                    w = self._fit_simplex_pg(X.to_numpy(), y.to_numpy())
            else:
                # Use last combo_lookback samples to train
                X = X_all.iloc[-self.config.combo_lookback:, :]
                y = y_all.iloc[-self.config.combo_lookback:]
                w = self._fit_simplex_pg(X.to_numpy(), y.to_numpy())

        w = self._project_to_simplex(w)  # safety

        # Build combined estimator on full timeline using vectorized operations
        if self.vectorization_manager:
            X_all_vectorized = self.vectorization_manager.vectorize_data(X_all)
            combined = (X_all_vectorized @ pd.Series(w, index=X_all.columns)).reindex(comps.index)
        else:
            combined = (X_all @ pd.Series(w, index=X_all.columns)).reindex(comps.index)

        weights = {col: float(w[i]) for i, col in enumerate(X_all.columns)}
        # If some component columns were fully NA earlier, pad their weights with 0
        for col in ["rv", "atr", "ewma"]:
            weights.setdefault(col, 0.0)

        # Cache the result
        if self.caching_system:
            self.caching_system.set(cache_key, {
                'combined': combined.astype(float),
                'weights': weights
            })

        return combined.astype(float), weights

    @staticmethod
    def _project_to_simplex(v: np.ndarray) -> np.ndarray:
        """
        Project vector v onto the probability simplex {w: w>=0, sum w=1}.
        Duchi, Shalev-Shwartz, Singer, Chandra (2008).
        """
        v = np.asarray(v, dtype=float)
        if v.ndim != 1:
            raise ValueError("v must be 1-D")
        n = v.size
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
        if len(rho) == 0:
            # If all zeros, return uniform
            return np.ones(n) / n
        rho = rho[-1]
        theta = (cssv[rho] - 1.0) / (rho + 1)
        w = np.maximum(v - theta, 0.0)
        s = w.sum()
        return w if s > 0 else np.ones(n) / n

    def _fit_simplex_pg(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Projected gradient descent to minimize (1/n)||Xw - y||^2,
        subject to w >= 0, sum(w) = 1.
        Step size uses Lipschitz constant estimate L = 2*||X||_2^2 / n.
        """
        n, d = X.shape
        if n == 0 or d == 0:
            return np.ones(d) / d

        # Standardize columns to comparable scale (avoid dominance)
        Xs = X.copy()
        col_scale = Xs.std(axis=0, ddof=1)
        col_scale[col_scale == 0] = 1.0
        Xs /= col_scale

        # Initialize uniform
        w = np.ones(d) / d

        # Lipschitz estimate (safe upper bound)
        # Use spectral norm via power iteration (few steps)
        def power_iter(A, iters=10):
            v = np.random.default_rng(123).standard_normal(A.shape[1])
            v /= np.linalg.norm(v) + 1e-12
            for _ in range(iters):
                v = A.T @ (A @ v)
                nv = np.linalg.norm(v) + 1e-12
                v /= nv
            return float(np.linalg.norm(A @ v))
        L2 = power_iter(Xs) ** 2  # ||Xs||_2^2
        L = 2.0 * L2 / max(n, 1)
        eta = 1.0 / (L + 1e-12)

        yv = y.astype(float)
        for _ in range(self.config.combo_max_iters):
            # Gradient: 2/n X^T (Xw - y)
            r = Xs @ w - yv
            grad = (2.0 / n) * (Xs.T @ r)
            w_new = self._project_to_simplex(w - eta * grad)
            if np.linalg.norm(w_new - w, ord=2) < self.config.combo_tol:
                w = w_new
                break
            w = w_new

        # Undo standardization effect in interpretation: not needed for w itself
        return w
    
    
    def _normalize_volatility_units(self, vol: pd.Series) -> pd.Series:
        """Apply data-driven floor/cap via percentiles (per series)."""
        vol = vol.astype(float)
        if vol.empty:
            return vol

        # Ensure finite positives
        vol = vol.replace([np.inf, -np.inf], np.nan).dropna()
        if vol.empty:
            return pd.Series(index=vol.index, dtype=float)

        if self.config.use_percentile_floor_cap:
            lo = np.nanpercentile(vol, self.config.floor_percentile)
            hi = np.nanpercentile(vol, self.config.cap_percentile)
            lo = max(float(lo), self.config.absolute_floor)
            hi = max(float(hi), lo)
            vol = vol.clip(lower=lo, upper=hi)
        else:
            vol = vol.clip(lower=self.config.absolute_floor)

        return vol
    
    def _normalize_volatility_units_enhanced(self, vol: pd.Series) -> pd.Series:
        """Enhanced volatility normalization with adaptive thresholds."""
        vol = vol.astype(float)
        if vol.empty:
            return vol

        # Ensure finite positives
        vol = vol.replace([np.inf, -np.inf], np.nan).dropna()
        if vol.empty:
            return pd.Series(index=vol.index, dtype=float)

        # Apply vectorization for better performance
        if self.vectorization_manager:
            vol = self.vectorization_manager.vectorize_data(vol)

        if self.config.use_percentile_floor_cap:
            # Use adaptive percentiles based on data characteristics
            data_std = vol.std()
            data_mean = vol.mean()
            
            # Adjust percentiles based on data distribution
            if data_std > data_mean:
                # High volatility data - use more conservative percentiles
                floor_pct = max(1.0, self.config.floor_percentile - 5)
                cap_pct = min(99.0, self.config.cap_percentile + 5)
            else:
                # Low volatility data - use standard percentiles
                floor_pct = self.config.floor_percentile
                cap_pct = self.config.cap_percentile
            
            lo = np.nanpercentile(vol, floor_pct)
            hi = np.nanpercentile(vol, cap_pct)
            lo = max(float(lo), self.config.absolute_floor)
            hi = max(float(hi), lo)
            vol = vol.clip(lower=lo, upper=hi)
        else:
            vol = vol.clip(lower=self.config.absolute_floor)

        return vol
    
    def _apply_enhanced_smoothing(self, vol: pd.Series) -> pd.Series:
        """Apply enhanced smoothing with adaptive parameters."""
        if len(vol) < self.config.smoothing_window:
            return vol
        
        # Use adaptive smoothing based on volatility characteristics
        vol_std = vol.std()
        vol_mean = vol.mean()
        
        if vol_std > vol_mean:
            # High volatility - use stronger smoothing
            window = min(self.config.smoothing_window * 2, len(vol) // 4)
            alpha = 0.1
        else:
            # Low volatility - use lighter smoothing
            window = self.config.smoothing_window
            alpha = 0.2
        
        # Apply rolling mean with adaptive window
        smoothed = vol.rolling(window=window, min_periods=window//2).mean()
        
        # Apply additional EWMA smoothing
        if len(smoothed) > 10:
            smoothed = smoothed.ewm(alpha=alpha, min_periods=5).mean()
        
        return smoothed
    
    
    def _calculate_volatility_statistics(self, vol: pd.Series) -> Dict[str, Any]:
        if vol.empty:
            return {"mean_volatility": 0.0, "volatility_std": 0.0, "volatility_percentiles": {}}

        mean_vol = float(vol.mean())
        std_vol = float(vol.std())
        ps = [5, 10, 25, 50, 75, 90, 95]
        pct = {f"p{p}": float(np.nanpercentile(vol, p)) for p in ps}
        return {"mean_volatility": mean_vol, "volatility_std": std_vol, "volatility_percentiles": pct}
    
    def _calculate_volatility_statistics_enhanced(self, vol: pd.Series) -> Dict[str, Any]:
        """Enhanced volatility statistics with additional metrics."""
        if vol.empty:
            return {"mean_volatility": 0.0, "volatility_std": 0.0, "volatility_percentiles": {}}

        # Basic statistics
        mean_vol = float(vol.mean())
        std_vol = float(vol.std())
        ps = [5, 10, 25, 50, 75, 90, 95]
        pct = {f"p{p}": float(np.nanpercentile(vol, p)) for p in ps}
        
        # Enhanced statistics
        enhanced_stats = {
            "mean_volatility": mean_vol,
            "volatility_std": std_vol,
            "volatility_percentiles": pct,
            "volatility_skewness": float(vol.skew()) if len(vol) > 10 else 0.0,
            "volatility_kurtosis": float(vol.kurtosis()) if len(vol) > 10 else 0.0,
            "volatility_range": float(vol.max() - vol.min()),
            "volatility_cv": float(std_vol / mean_vol) if mean_vol > 0 else 0.0,
            "volatility_median": float(vol.median()),
            "volatility_iqr": float(vol.quantile(0.75) - vol.quantile(0.25))
        }
        
        return enhanced_stats
    
    def _calculate_volatility_quality(self, vol: pd.Series) -> Dict[str, float]:
        if vol.empty:
            return {"consistency": 0.0, "stability": 0.0}
        m = float(vol.mean())
        if m <= 0:
            return {"consistency": 0.0, "stability": 0.0}
        diff1 = vol.diff().abs()
        consistency = float(np.clip(1.0 - diff1.mean() / m, 0.0, 1.0))
        stability = float(np.clip(1.0 - vol.std() / m, 0.0, 1.0))
        return {"consistency": consistency, "stability": stability}
    
    def _calculate_volatility_quality_enhanced(self, vol: pd.Series) -> Dict[str, float]:
        """Enhanced volatility quality metrics with additional measures."""
        if vol.empty:
            return {"consistency": 0.0, "stability": 0.0, "smoothness": 0.0, "trend_stability": 0.0}
        
        m = float(vol.mean())
        if m <= 0:
            return {"consistency": 0.0, "stability": 0.0, "smoothness": 0.0, "trend_stability": 0.0}
        
        # Basic quality metrics
        diff1 = vol.diff().abs()
        consistency = float(np.clip(1.0 - diff1.mean() / m, 0.0, 1.0))
        stability = float(np.clip(1.0 - vol.std() / m, 0.0, 1.0))
        
        # Enhanced quality metrics
        # Smoothness: measure of local variation
        diff2 = vol.diff().diff().abs()
        smoothness = float(np.clip(1.0 - diff2.mean() / m, 0.0, 1.0)) if len(vol) > 2 else 0.0
        
        # Trend stability: measure of trend consistency
        if len(vol) > 10:
            # Calculate rolling trend
            rolling_trend = vol.rolling(window=10, min_periods=5).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
            trend_stability = float(np.clip(1.0 - rolling_trend.std() / m, 0.0, 1.0))
        else:
            trend_stability = 0.0
        
        # Autocorrelation-based consistency
        if len(vol) > 20:
            autocorr = vol.autocorr(lag=1)
            autocorr_consistency = float(np.clip(abs(autocorr), 0.0, 1.0)) if not pd.isna(autocorr) else 0.0
        else:
            autocorr_consistency = 0.0
        
        return {
            "consistency": consistency,
            "stability": stability,
            "smoothness": smoothness,
            "trend_stability": trend_stability,
            "autocorr_consistency": autocorr_consistency
        }


# --------------------------------------------------------------------------------------
# Convenience functions
# --------------------------------------------------------------------------------------

def create_volatility_modeler(config: Optional[VolatilityConfig] = None) -> VolatilityModeler:
    return VolatilityModeler(config)


def model_volatility(bars: pd.DataFrame, config: Optional[VolatilityConfig] = None) -> VolatilityResult:
    modeler = VolatilityModeler(config)
    return modeler.model_volatility(bars)