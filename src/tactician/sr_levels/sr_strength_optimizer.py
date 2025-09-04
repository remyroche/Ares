"""S/R Strength Parameter Optimizer Module.

This module optimizes parameters for identifying strong S/R levels through comprehensive backtesting,
focusing on level strength characteristics rather than breakout predictions.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import optuna
import json
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from src.core.decorators import handles_errors, traced
from src.utils.logger import system_logger

# Try to import optional dependencies
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


@dataclass
class SRStrengthParameters:
    """Parameters for S/R strength calculation."""
    # Touch validation parameters
    min_touches: int = 3
    touch_proximity_threshold: float = 0.002
    touch_time_decay: float = 0.95
    
    # Bounce quality parameters
    min_bounce_ratio: float = 0.001
    bounce_strength_multiplier: float = 2.0
    failed_bounce_penalty: float = 0.7
    
    # Age importance parameters
    optimal_age_bars: int = 100
    age_decay_rate: float = 0.98
    max_age_bars: int = 500
    recent_bonus_bars: int = 20
    
    # Volume confirmation parameters
    volume_spike_threshold: float = 1.5
    volume_confirmation_weight: float = 0.3
    low_volume_penalty: float = 0.8
    
    # Price action parameters
    clean_bounce_threshold: float = 0.8
    rejection_wick_ratio: float = 0.6
    consolidation_bonus: float = 1.2
    
    # Multi-touch parameters
    consistent_bounce_bonus: float = 1.3
    increasing_strength_bonus: float = 1.2
    decreasing_strength_penalty: float = 0.8


@dataclass
class SRLevel:
    """Strong S/R level definition."""
    price: float
    strength: float
    type: str  # 'support' or 'resistance'
    touch_count: int
    first_touch_bar: int
    last_touch_bar: int
    age_bars: int
    avg_bounce_ratio: float
    max_bounce_ratio: float
    volume_confirmation_score: float
    consistency_score: float
    failure_count: int
    metadata: Dict[str, Any]


@dataclass
class OptimizationResult:
    """Results from S/R strength optimization."""
    best_parameters: SRStrengthParameters
    optimization_score: float
    strength_metrics: Dict[str, float]
    multi_timeframe_scores: Dict[str, float]
    feature_importance: Dict[str, float]
    n_trials: int
    optimization_time: float


class SRStrengthOptimizer:
    """Optimizes S/R strength calculation parameters with computational efficiency."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the S/R strength optimizer."""
        self.config = config
        self.logger = system_logger.getChild("SRStrengthOptimizer")
        
        # Optimization configuration
        self.optim_config = config.get("sr_strength_optimization", {})
        self.n_trials = self.optim_config.get("n_trials", 100)
        self.n_jobs = self.optim_config.get("n_jobs", -1)
        
        # Computational optimization settings
        self.use_parallel = JOBLIB_AVAILABLE and self.n_jobs != 1
        self.use_numba = NUMBA_AVAILABLE
        self.chunk_size = self.optim_config.get("chunk_size", 1000)
        
        # Parameter ranges
        self.param_ranges = self._get_parameter_ranges()
        
        # Cache for expensive calculations
        self.calculation_cache = {}
        self.cache_size = 1000
        
        # Best parameters storage
        self.best_parameters = SRStrengthParameters()
        self.optimization_history = []
        
    def _get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter ranges for optimization."""
        return {
            # Touch validation
            "min_touches": (2, 5),
            "touch_proximity_threshold": (0.001, 0.005),
            "touch_time_decay": (0.9, 0.99),
            
            # Bounce quality
            "min_bounce_ratio": (0.0005, 0.002),
            "bounce_strength_multiplier": (1.5, 3.0),
            "failed_bounce_penalty": (0.5, 0.9),
            
            # Age importance
            "optimal_age_bars": (50, 200),
            "age_decay_rate": (0.95, 0.995),
            "max_age_bars": (300, 1000),
            "recent_bonus_bars": (10, 50),
            
            # Volume confirmation
            "volume_spike_threshold": (1.2, 2.0),
            "volume_confirmation_weight": (0.2, 0.5),
            "low_volume_penalty": (0.6, 0.9),
            
            # Price action
            "clean_bounce_threshold": (0.6, 0.9),
            "rejection_wick_ratio": (0.5, 0.8),
            "consolidation_bonus": (1.1, 1.5),
            
            # Multi-touch
            "consistent_bounce_bonus": (1.1, 1.5),
            "increasing_strength_bonus": (1.1, 1.3),
            "decreasing_strength_penalty": (0.7, 0.9)
        }
    
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="optimize SR strength parameters"
    )
    @traced(span_name="SRStrength.optimize")
    async def optimize_parameters(
        self,
        market_data_dict: Dict[str, pd.DataFrame],
        validation_data_dict: Optional[Dict[str, pd.DataFrame]] = None
    ) -> OptimizationResult:
        """
        Optimize S/R strength parameters across multiple timeframes.
        
        Args:
            market_data_dict: Dictionary of market data by timeframe
            validation_data_dict: Optional validation data by timeframe
            
        Returns:
            OptimizationResult with optimized parameters
        """
        try:
            self.logger.info("🎯 Starting S/R strength parameter optimization...")
            start_time = datetime.now()
            
            # Precompute expensive calculations
            self._precompute_market_features(market_data_dict)
            
            # Create Optuna study with pruning for efficiency
            study = optuna.create_study(
                direction="maximize",
                study_name="sr_strength_optimization",
                pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
            )
            
            # Define objective function
            def objective(trial):
                return self._optimization_objective(
                    trial, market_data_dict
                )
            
            # Run optimization with parallel trials if available
            if self.use_parallel and self.n_jobs > 1:
                # Run parallel optimization
                study.optimize(
                    objective,
                    n_trials=self.n_trials,
                    n_jobs=self.n_jobs,
                    show_progress_bar=True
                )
            else:
                # Sequential optimization
                study.optimize(
                    objective,
                    n_trials=self.n_trials,
                    show_progress_bar=True
                )
            
            # Extract best parameters
            best_params = study.best_params
            best_value = study.best_value
            
            # Create parameter set
            optimized_params = self._create_parameter_set(best_params)
            
            # Calculate detailed metrics
            strength_metrics = await self._calculate_strength_metrics(
                optimized_params, market_data_dict
            )
            
            # Multi-timeframe analysis
            multi_tf_scores = await self._analyze_multi_timeframe_performance(
                optimized_params, market_data_dict
            )
            
            # Feature importance analysis
            feature_importance = await self._analyze_feature_importance(
                optimized_params, market_data_dict
            )
            
            # Validate if provided
            if validation_data_dict:
                validation_score = await self._validate_parameters(
                    optimized_params, validation_data_dict
                )
                self.logger.info(f"Validation score: {validation_score:.4f}")
            
            # Create result
            result = OptimizationResult(
                best_parameters=optimized_params,
                optimization_score=best_value,
                strength_metrics=strength_metrics,
                multi_timeframe_scores=multi_tf_scores,
                feature_importance=feature_importance,
                n_trials=len(study.trials),
                optimization_time=(datetime.now() - start_time).total_seconds()
            )
            
            # Store best parameters
            self.best_parameters = optimized_params
            self.optimization_history.append(result)
            
            # Log results
            self._log_optimization_results(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error optimizing S/R strength parameters: {e}")
            return None
    
    def _precompute_market_features(self, market_data_dict: Dict[str, pd.DataFrame]) -> None:
        """Precompute expensive features for all timeframes."""
        self.logger.info("Precomputing market features for efficiency...")
        
        for timeframe, data in market_data_dict.items():
            cache_key = f"features_{timeframe}"
            
            if cache_key not in self.calculation_cache:
                features = {}
                
                # Precompute volatility
                features['volatility'] = data['close'].pct_change().rolling(20).std()
                
                # Precompute volume metrics
                features['volume_ma'] = data['volume'].rolling(20).mean()
                features['volume_ratio'] = data['volume'] / features['volume_ma']
                
                # Precompute price metrics
                features['high_low_range'] = data['high'] - data['low']
                features['body_size'] = abs(data['close'] - data['open'])
                features['upper_wick'] = data['high'] - data[['open', 'close']].max(axis=1)
                features['lower_wick'] = data[['open', 'close']].min(axis=1) - data['low']
                
                self.calculation_cache[cache_key] = features
    
    def _optimization_objective(
        self,
        trial: optuna.Trial,
        market_data_dict: Dict[str, pd.DataFrame]
    ) -> float:
        """Objective function for parameter optimization."""
        
        # Sample parameters
        params = {}
        for param_name, (min_val, max_val) in self.param_ranges.items():
            if param_name in ["min_touches", "optimal_age_bars", "max_age_bars", "recent_bonus_bars"]:
                params[param_name] = trial.suggest_int(param_name, int(min_val), int(max_val))
            else:
                params[param_name] = trial.suggest_float(param_name, min_val, max_val)
        
        # Create parameter set
        param_set = self._create_parameter_set(params)
        
        # Evaluate across all timeframes
        scores = []
        for timeframe, market_data in market_data_dict.items():
            # Use cached features
            features = self.calculation_cache.get(f"features_{timeframe}", {})
            
            # Identify S/R levels with these parameters
            sr_levels = self._identify_sr_levels_fast(market_data, param_set, features)
            
            # Evaluate level quality
            score = self._evaluate_sr_quality(market_data, sr_levels, param_set)
            scores.append(score)
            
            # Prune if score is too low
            if score < 0.3:
                raise optuna.exceptions.TrialPruned()
        
        # Return average score across timeframes
        return np.mean(scores)
    
    @numba.jit(nopython=True) if NUMBA_AVAILABLE else lambda x: x
    def _find_local_extrema_fast(prices_high: np.ndarray, prices_low: np.ndarray, 
                                  window: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Fast local extrema detection using Numba if available."""
        n = len(prices_high)
        resistance_levels = []
        support_levels = []
        
        for i in range(window, n - window):
            # Check for local maximum (resistance)
            is_max = True
            for j in range(i - window, i + window + 1):
                if j != i and prices_high[j] >= prices_high[i]:
                    is_max = False
                    break
            if is_max:
                resistance_levels.append(i)
            
            # Check for local minimum (support)
            is_min = True
            for j in range(i - window, i + window + 1):
                if j != i and prices_low[j] <= prices_low[i]:
                    is_min = False
                    break
            if is_min:
                support_levels.append(i)
        
        return np.array(resistance_levels), np.array(support_levels)
    
    def _identify_sr_levels_fast(
        self,
        market_data: pd.DataFrame,
        params: SRStrengthParameters,
        features: Dict[str, pd.Series]
    ) -> List[SRLevel]:
        """Identify S/R levels using optimized algorithm."""
        
        if self.use_numba and NUMBA_AVAILABLE:
            # Use Numba-accelerated function
            resistance_idx, support_idx = self._find_local_extrema_fast(
                market_data['high'].values,
                market_data['low'].values
            )
        else:
            # Fallback to pandas
            resistance_idx = market_data['high'].rolling(20).apply(
                lambda x: x.argmax() == len(x) // 2
            ).fillna(False)
            resistance_idx = market_data.index[resistance_idx].tolist()
            
            support_idx = market_data['low'].rolling(20).apply(
                lambda x: x.argmin() == len(x) // 2
            ).fillna(False)
            support_idx = market_data.index[support_idx].tolist()
        
        # Process levels
        sr_levels = []
        
        # Process resistance levels
        for idx in resistance_idx:
            if isinstance(idx, (int, np.integer)):
                price = market_data['high'].iloc[idx]
                level = self._analyze_sr_level(
                    market_data, price, 'resistance', idx, params, features
                )
                if level and level.strength > 0.3:  # Minimum strength threshold
                    sr_levels.append(level)
        
        # Process support levels
        for idx in support_idx:
            if isinstance(idx, (int, np.integer)):
                price = market_data['low'].iloc[idx]
                level = self._analyze_sr_level(
                    market_data, price, 'support', idx, params, features
                )
                if level and level.strength > 0.3:
                    sr_levels.append(level)
        
        # Cluster nearby levels
        return self._cluster_sr_levels(sr_levels, params.touch_proximity_threshold)
    
    def _analyze_sr_level(
        self,
        market_data: pd.DataFrame,
        level_price: float,
        level_type: str,
        origin_idx: int,
        params: SRStrengthParameters,
        features: Dict[str, pd.Series]
    ) -> Optional[SRLevel]:
        """Analyze a potential S/R level with enhanced wick analysis and time weighting."""
        
        touches = []
        bounces = []
        failures = []
        volumes = []
        wick_touches = []
        body_touches = []
        
        # Get ATR for dynamic thresholds
        atr = features.get('atr_14', pd.Series())
        dynamic_threshold = params.touch_proximity_threshold
        if not atr.empty and len(atr) > origin_idx:
            # Use ATR-based dynamic threshold
            atr_value = atr.iloc[origin_idx] if origin_idx < len(atr) else atr.iloc[-1]
            dynamic_threshold = min(max(atr_value / level_price, params.touch_proximity_threshold * 0.5), params.touch_proximity_threshold * 2.0)
        
        # Analyze each bar after the origin
        for i in range(origin_idx + 1, len(market_data)):
            high = market_data['high'].iloc[i]
            low = market_data['low'].iloc[i]
            open_price = market_data['open'].iloc[i]
            close = market_data['close'].iloc[i]
            volume = market_data['volume'].iloc[i]
            
            # Calculate body and wick components
            body_high = max(open_price, close)
            body_low = min(open_price, close)
            upper_wick = high - body_high
            lower_wick = body_low - low
            
            # Check if price touched the level with enhanced analysis
            if level_type == 'resistance':
                if abs(high - level_price) / level_price < dynamic_threshold:
                    touches.append(i)
                    volumes.append(volume)
                    
                    # Analyze wick vs body touch
                    if abs(body_high - level_price) / level_price < dynamic_threshold:
                        body_touches.append(i)
                    else:
                        wick_touches.append(i)
                    
                    # Enhanced bounce detection with volume confirmation
                    if i < len(market_data) - 1:
                        next_close = market_data['close'].iloc[i + 1]
                        next_volume = market_data['volume'].iloc[i + 1]
                        
                        # Check for bounce with volume confirmation
                        if next_close < close:
                            bounce_ratio = (high - next_close) / high
                            # Require volume spike for valid bounce
                            volume_ma = features.get('volume_ma_20', pd.Series())
                            volume_spike = True
                            if not volume_ma.empty and i < len(volume_ma):
                                volume_spike = volume > volume_ma.iloc[i] * params.volume_spike_threshold
                            
                            if volume_spike:
                                bounces.append((i, bounce_ratio, volume_spike))
                            else:
                                failures.append(i)
                        else:
                            failures.append(i)
            else:  # support
                if abs(low - level_price) / level_price < dynamic_threshold:
                    touches.append(i)
                    volumes.append(volume)
                    
                    # Analyze wick vs body touch
                    if abs(body_low - level_price) / level_price < dynamic_threshold:
                        body_touches.append(i)
                    else:
                        wick_touches.append(i)
                    
                    # Enhanced bounce detection with volume confirmation
                    if i < len(market_data) - 1:
                        next_close = market_data['close'].iloc[i + 1]
                        next_volume = market_data['volume'].iloc[i + 1]
                        
                        # Check for bounce with volume confirmation
                        if next_close > close:
                            bounce_ratio = (next_close - low) / low
                            # Require volume spike for valid bounce
                            volume_ma = features.get('volume_ma_20', pd.Series())
                            volume_spike = True
                            if not volume_ma.empty and i < len(volume_ma):
                                volume_spike = volume > volume_ma.iloc[i] * params.volume_spike_threshold
                            
                            if volume_spike:
                                bounces.append((i, bounce_ratio, volume_spike))
                            else:
                                failures.append(i)
                        else:
                            failures.append(i)
        
        # Need minimum touches
        if len(touches) < params.min_touches:
            return None
        
        # Calculate enhanced strength metrics with time weighting
        strength = self._calculate_enhanced_level_strength(
            touches, bounces, failures, volumes, wick_touches, body_touches,
            origin_idx, len(market_data), params, features
        )
        
        if strength < 0.3:  # Minimum strength
            return None
        
        # Calculate additional metrics
        avg_bounce = np.mean([b[1] for b in bounces]) if bounces else 0
        max_bounce = max([b[1] for b in bounces]) if bounces else 0
        
        volume_score = self._calculate_volume_confirmation(
            volumes, features.get('volume_ma_20', pd.Series()), params
        )
        
        consistency_score = self._calculate_consistency_score(
            bounces, params
        )
        
        # Calculate wick vs body ratio
        wick_body_ratio = len(wick_touches) / max(len(body_touches), 1) if body_touches else 1.0
        
        return SRLevel(
            price=level_price,
            strength=strength,
            type=level_type,
            touch_count=len(touches),
            first_touch_bar=origin_idx,
            last_touch_bar=touches[-1] if touches else origin_idx,
            age_bars=len(market_data) - origin_idx,
            avg_bounce_ratio=avg_bounce,
            max_bounce_ratio=max_bounce,
            volume_confirmation_score=volume_score,
            consistency_score=consistency_score,
            failure_count=len(failures),
            metadata={
                'bounce_count': len(bounces),
                'touch_indices': touches[:10],  # Store first 10 for analysis
                'wick_touches': len(wick_touches),
                'body_touches': len(body_touches),
                'wick_body_ratio': wick_body_ratio,
                'dynamic_threshold_used': dynamic_threshold,
                'volume_spike_confirmations': sum(1 for b in bounces if len(b) > 2 and b[2])
            }
        )
    
    def _calculate_enhanced_level_strength(
        self,
        touches: List[int],
        bounces: List[Tuple[int, float, bool]],
        failures: List[int],
        volumes: List[float],
        wick_touches: List[int],
        body_touches: List[int],
        origin_idx: int,
        total_bars: int,
        params: SRStrengthParameters,
        features: Dict[str, pd.Series]
    ) -> float:
        """Calculate enhanced strength score with non-linear scoring and market regime adaptation."""
        
        if not touches:
            return 0.0
        
        # Get market regime indicators
        rsi = features.get('rsi_14', pd.Series())
        sma_20 = features.get('sma_20', pd.Series())
        sma_50 = features.get('sma_50', pd.Series())
        
        # Determine market regime
        market_regime = self._determine_market_regime(rsi, sma_20, sma_50)
        
        # Base score from touch count with time weighting
        time_weighted_touches = 0
        for i, touch_idx in enumerate(touches):
            # Recent touches get higher weight (exponential decay)
            time_factor = np.exp(-(total_bars - touch_idx) / 100.0)  # Decay over 100 bars
            time_weighted_touches += time_factor
        
        touch_score = min(time_weighted_touches / len(touches), 1.0)
        
        # Enhanced bounce quality score with non-linear scaling
        if bounces:
            bounce_ratios = [b[1] for b in bounces]
            avg_bounce = np.mean(bounce_ratios)
            
            # Non-linear bounce scoring (exponential for better discrimination)
            if avg_bounce > params.min_bounce_ratio:
                bounce_score = min(np.exp((avg_bounce - params.min_bounce_ratio) * 10), 1.0)
            else:
                bounce_score = 0.0
            
            # Apply multiplier for strong bounces
            if avg_bounce > params.min_bounce_ratio * 2:
                bounce_score *= params.bounce_strength_multiplier
        else:
            bounce_score = 0.0
        
        # Wick vs body analysis
        wick_body_score = 1.0
        if body_touches and wick_touches:
            # Body touches are more significant than wick touches
            body_ratio = len(body_touches) / len(touches)
            wick_ratio = len(wick_touches) / len(touches)
            wick_body_score = 0.7 * body_ratio + 0.3 * wick_ratio
        elif body_touches:
            wick_body_score = 1.0
        elif wick_touches:
            wick_body_score = 0.6
        
        # Failure penalty with non-linear scaling
        failure_rate = len(failures) / max(len(touches), 1)
        failure_penalty = np.exp(-failure_rate * 5)  # Exponential penalty
        
        # Age score with market regime adaptation
        age = total_bars - origin_idx
        if market_regime == 'trending':
            # In trending markets, recent levels are more important
            if age < params.recent_bonus_bars:
                age_score = 1.2
            elif age < params.optimal_age_bars:
                age_score = 1.0
            else:
                age_score = params.age_decay_rate ** (age - params.optimal_age_bars)
        else:  # ranging market
            # In ranging markets, older levels can be more reliable
            if age < params.optimal_age_bars:
                age_score = 1.0
            elif age < params.max_age_bars:
                age_score = 1.1  # Slight bonus for older levels in ranging markets
            else:
                age_score = params.age_decay_rate ** (age - params.max_age_bars)
        
        # Volume confirmation with regime adaptation
        if volumes and 'volume_ma_20' in features:
            volume_ma = features['volume_ma_20']
            volume_ratios = []
            for i, volume in enumerate(volumes):
                if i < len(touches) and touches[i] < len(volume_ma):
                    volume_ratios.append(volume / volume_ma.iloc[touches[i]])
            
            if volume_ratios:
                avg_volume_ratio = np.mean(volume_ratios)
                if market_regime == 'trending':
                    # In trending markets, volume spikes are more important
                    volume_score = min(avg_volume_ratio / params.volume_spike_threshold, 1.0)
                else:
                    # In ranging markets, consistent volume is more important
                    volume_score = min(avg_volume_ratio / (params.volume_spike_threshold * 0.8), 1.0)
            else:
                volume_score = 0.5
        else:
            volume_score = 0.5
        
        # Combine all factors with regime-adapted weights
        if market_regime == 'trending':
            weights = {
                'touch': 0.3,
                'bounce': 0.35,
                'wick_body': 0.15,
                'age': 0.1,
                'volume': 0.1
            }
        else:  # ranging
            weights = {
                'touch': 0.25,
                'bounce': 0.3,
                'wick_body': 0.2,
                'age': 0.15,
                'volume': 0.1
            }
        
        # Calculate final strength with non-linear combination
        strength = (
            weights['touch'] * touch_score +
            weights['bounce'] * bounce_score +
            weights['wick_body'] * wick_body_score +
            weights['age'] * age_score +
            weights['volume'] * volume_score
        ) * failure_penalty
        
        return min(strength, 1.0)
    
    def _determine_market_regime(self, rsi: pd.Series, sma_20: pd.Series, sma_50: pd.Series) -> str:
        """Determine market regime based on technical indicators."""
        try:
            if rsi.empty or sma_20.empty or sma_50.empty:
                return 'unknown'
            
            # Get recent values
            recent_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
            recent_sma_20 = sma_20.iloc[-1] if len(sma_20) > 0 else 0
            recent_sma_50 = sma_50.iloc[-1] if len(sma_50) > 0 else 0
            
            # Determine trend
            if recent_sma_20 > recent_sma_50 * 1.02:  # 2% above
                trend = 'uptrend'
            elif recent_sma_20 < recent_sma_50 * 0.98:  # 2% below
                trend = 'downtrend'
            else:
                trend = 'sideways'
            
            # Determine volatility/strength
            if recent_rsi > 70:
                strength = 'overbought'
            elif recent_rsi < 30:
                strength = 'oversold'
            else:
                strength = 'neutral'
            
            # Combine to determine regime
            if trend in ['uptrend', 'downtrend'] and strength == 'neutral':
                return 'trending'
            elif trend == 'sideways':
                return 'ranging'
            else:
                return 'transitional'
                
        except Exception:
            return 'unknown'
    
    def _calculate_level_strength(
        self,
        touches: List[int],
        bounces: List[Tuple[int, float]],
        failures: List[int],
        volumes: List[float],
        origin_idx: int,
        total_bars: int,
        params: SRStrengthParameters,
        features: Dict[str, pd.Series]
    ) -> float:
        """Calculate comprehensive strength score for S/R level."""
        
        if not touches:
            return 0.0
        
        # Base score from touch count
        touch_score = min(len(touches) / 10, 1.0)  # Normalize to max 1.0
        
        # Bounce quality score
        if bounces:
            avg_bounce = np.mean([b[1] for b in bounces])
            bounce_score = min(avg_bounce / params.min_bounce_ratio, 1.0)
            
            # Apply multiplier for strong bounces
            if avg_bounce > params.min_bounce_ratio * 2:
                bounce_score *= params.bounce_strength_multiplier
        else:
            bounce_score = 0.0
        
        # Failure penalty
        failure_rate = len(failures) / max(len(touches), 1)
        failure_penalty = params.failed_bounce_penalty ** failure_rate
        
        # Age score
        age = total_bars - origin_idx
        if age < params.recent_bonus_bars:
            age_score = 1.2  # Recent level bonus
        elif age < params.optimal_age_bars:
            age_score = 1.0
        elif age < params.max_age_bars:
            # Decay from optimal to max
            decay_factor = (age - params.optimal_age_bars) / (params.max_age_bars - params.optimal_age_bars)
            age_score = params.age_decay_rate ** decay_factor
        else:
            age_score = 0.5  # Old but still valid
        
        # Time decay for touches
        time_weighted_touches = 0
        for i, touch_idx in enumerate(touches):
            time_factor = params.touch_time_decay ** (total_bars - touch_idx)
            time_weighted_touches += time_factor
        touch_score *= time_weighted_touches / len(touches)
        
        # Volume confirmation
        if volumes and 'volume_ma' in features:
            volume_ratios = [v / features['volume_ma'].iloc[touches[i]] 
                           for i, v in enumerate(volumes) 
                           if touches[i] < len(features['volume_ma'])]
            if volume_ratios:
                volume_score = np.mean([min(r / params.volume_spike_threshold, 1.0) 
                                       for r in volume_ratios])
            else:
                volume_score = 0.5
        else:
            volume_score = 0.5
        
        # Combine all factors
        strength = (
            touch_score * 0.25 +
            bounce_score * 0.30 +
            age_score * 0.20 +
            volume_score * params.volume_confirmation_weight +
            (1 - params.volume_confirmation_weight) * 0.25
        ) * failure_penalty
        
        return min(strength, 1.0)
    
    def _calculate_volume_confirmation(
        self,
        volumes: List[float],
        volume_ma: pd.Series,
        params: SRStrengthParameters
    ) -> float:
        """Calculate volume confirmation score."""
        if not volumes or volume_ma.empty:
            return 0.5
        
        confirmations = 0
        for i, vol in enumerate(volumes):
            if i < len(volume_ma):
                ratio = vol / volume_ma.iloc[i]
                if ratio > params.volume_spike_threshold:
                    confirmations += 1
                elif ratio < 1 / params.volume_spike_threshold:
                    confirmations -= 0.5  # Low volume penalty
        
        return max(0, min(1, confirmations / len(volumes)))
    
    def _calculate_consistency_score(
        self,
        bounces: List[Tuple[int, float]],
        params: SRStrengthParameters
    ) -> float:
        """Calculate bounce consistency score."""
        if len(bounces) < 2:
            return 0.5
        
        bounce_ratios = [b[1] for b in bounces]
        
        # Check if bounces are getting stronger
        increasing = all(bounce_ratios[i] <= bounce_ratios[i+1] 
                        for i in range(len(bounce_ratios)-1))
        
        # Check if bounces are consistent
        std_dev = np.std(bounce_ratios)
        mean_bounce = np.mean(bounce_ratios)
        cv = std_dev / mean_bounce if mean_bounce > 0 else 1
        
        consistency = 1 - min(cv, 1)
        
        if increasing:
            consistency *= params.increasing_strength_bonus
        elif all(bounce_ratios[i] >= bounce_ratios[i+1] 
                for i in range(len(bounce_ratios)-1)):
            consistency *= params.decreasing_strength_penalty
        
        return min(consistency, 1.0)
    
    def _cluster_sr_levels(
        self,
        sr_levels: List[SRLevel],
        proximity_threshold: float
    ) -> List[SRLevel]:
        """Cluster nearby S/R levels."""
        if not sr_levels:
            return []
        
        # Sort by price
        sorted_levels = sorted(sr_levels, key=lambda x: x.price)
        
        clustered = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # Check if close to current cluster
            cluster_price = np.mean([l.price for l in current_cluster])
            if abs(level.price - cluster_price) / cluster_price < proximity_threshold:
                current_cluster.append(level)
            else:
                # Merge current cluster
                merged = self._merge_cluster(current_cluster)
                if merged:
                    clustered.append(merged)
                current_cluster = [level]
        
        # Don't forget last cluster
        if current_cluster:
            merged = self._merge_cluster(current_cluster)
            if merged:
                clustered.append(merged)
        
        return clustered
    
    def _merge_cluster(self, cluster: List[SRLevel]) -> Optional[SRLevel]:
        """Merge a cluster of S/R levels into one."""
        if not cluster:
            return None
        
        if len(cluster) == 1:
            return cluster[0]
        
        # Weight by strength
        total_strength = sum(l.strength for l in cluster)
        weighted_price = sum(l.price * l.strength for l in cluster) / total_strength
        
        # Combine metrics
        return SRLevel(
            price=weighted_price,
            strength=max(l.strength for l in cluster),  # Take strongest
            type=cluster[0].type,
            touch_count=sum(l.touch_count for l in cluster),
            first_touch_bar=min(l.first_touch_bar for l in cluster),
            last_touch_bar=max(l.last_touch_bar for l in cluster),
            age_bars=max(l.age_bars for l in cluster),
            avg_bounce_ratio=np.mean([l.avg_bounce_ratio for l in cluster]),
            max_bounce_ratio=max(l.max_bounce_ratio for l in cluster),
            volume_confirmation_score=np.mean([l.volume_confirmation_score for l in cluster]),
            consistency_score=np.mean([l.consistency_score for l in cluster]),
            failure_count=sum(l.failure_count for l in cluster),
            metadata={'merged_from': len(cluster)}
        )
    
    def _evaluate_sr_quality(
        self,
        market_data: pd.DataFrame,
        sr_levels: List[SRLevel],
        params: SRStrengthParameters
    ) -> float:
        """Evaluate quality of identified S/R levels."""
        if not sr_levels:
            return 0.0
        
        # Quality metrics
        metrics = {
            'level_count': len(sr_levels),
            'avg_strength': np.mean([l.strength for l in sr_levels]),
            'strong_levels': sum(1 for l in sr_levels if l.strength > 0.7),
            'avg_touches': np.mean([l.touch_count for l in sr_levels]),
            'avg_bounce_ratio': np.mean([l.avg_bounce_ratio for l in sr_levels if l.avg_bounce_ratio > 0]),
            'recent_levels': sum(1 for l in sr_levels if l.age_bars < params.recent_bonus_bars)
        }
        
        # Calculate quality score
        score = 0.0
        
        # Prefer reasonable number of levels
        if 5 <= metrics['level_count'] <= 20:
            score += 0.2
        elif metrics['level_count'] > 20:
            score += 0.1
        
        # Strength score
        score += metrics['avg_strength'] * 0.3
        
        # Strong levels bonus
        score += min(metrics['strong_levels'] / 5, 0.2)
        
        # Touch count score
        score += min(metrics['avg_touches'] / 10, 0.15)
        
        # Bounce quality
        if metrics['avg_bounce_ratio'] > params.min_bounce_ratio:
            score += 0.15
        
        return min(score, 1.0)
    
    async def _calculate_strength_metrics(
        self,
        params: SRStrengthParameters,
        market_data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Calculate detailed strength metrics."""
        metrics = {}
        
        for timeframe, data in market_data_dict.items():
            features = self.calculation_cache.get(f"features_{timeframe}", {})
            sr_levels = self._identify_sr_levels_fast(data, params, features)
            
            if sr_levels:
                metrics[f"{timeframe}_avg_strength"] = np.mean([l.strength for l in sr_levels])
                metrics[f"{timeframe}_level_count"] = len(sr_levels)
                metrics[f"{timeframe}_avg_age"] = np.mean([l.age_bars for l in sr_levels])
                metrics[f"{timeframe}_bounce_quality"] = np.mean([l.avg_bounce_ratio for l in sr_levels if l.avg_bounce_ratio > 0])
        
        return metrics
    
    async def _analyze_multi_timeframe_performance(
        self,
        params: SRStrengthParameters,
        market_data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Analyze performance across multiple timeframes."""
        scores = {}
        
        # Analyze each timeframe
        for timeframe, data in market_data_dict.items():
            features = self.calculation_cache.get(f"features_{timeframe}", {})
            sr_levels = self._identify_sr_levels_fast(data, params, features)
            score = self._evaluate_sr_quality(data, sr_levels, params)
            scores[timeframe] = score
        
        # Add correlation analysis
        if len(market_data_dict) > 1:
            # Find levels that appear in multiple timeframes
            all_levels = {}
            for timeframe, data in market_data_dict.items():
                features = self.calculation_cache.get(f"features_{timeframe}", {})
                sr_levels = self._identify_sr_levels_fast(data, params, features)
                all_levels[timeframe] = sr_levels
            
            # Calculate confluence score
            confluence_score = self._calculate_timeframe_confluence(all_levels, params)
            scores['confluence'] = confluence_score
        
        return scores
    
    def _calculate_timeframe_confluence(
        self,
        all_levels: Dict[str, List[SRLevel]],
        params: SRStrengthParameters
    ) -> float:
        """Calculate confluence score across timeframes."""
        if len(all_levels) < 2:
            return 0.5
        
        confluence_count = 0
        total_comparisons = 0
        
        timeframes = list(all_levels.keys())
        for i in range(len(timeframes)):
            for j in range(i + 1, len(timeframes)):
                tf1_levels = all_levels[timeframes[i]]
                tf2_levels = all_levels[timeframes[j]]
                
                # Check for matching levels
                for l1 in tf1_levels:
                    for l2 in tf2_levels:
                        if abs(l1.price - l2.price) / l1.price < params.touch_proximity_threshold * 2:
                            confluence_count += min(l1.strength, l2.strength)
                        total_comparisons += 1
        
        if total_comparisons > 0:
            return min(confluence_count / total_comparisons * 10, 1.0)
        return 0.5
    
    async def _analyze_feature_importance(
        self,
        params: SRStrengthParameters,
        market_data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Analyze importance of different features."""
        
        # Test impact of each parameter group
        importance = {}
        
        # Reference score with all parameters
        reference_score = 0
        for tf, data in market_data_dict.items():
            features = self.calculation_cache.get(f"features_{tf}", {})
            sr_levels = self._identify_sr_levels_fast(data, params, features)
            reference_score += self._evaluate_sr_quality(data, sr_levels, params)
        reference_score /= len(market_data_dict)
        
        # Test each parameter group
        param_groups = {
            'touch_validation': ['min_touches', 'touch_proximity_threshold', 'touch_time_decay'],
            'bounce_quality': ['min_bounce_ratio', 'bounce_strength_multiplier', 'failed_bounce_penalty'],
            'age_importance': ['optimal_age_bars', 'age_decay_rate', 'max_age_bars'],
            'volume_confirmation': ['volume_spike_threshold', 'volume_confirmation_weight'],
            'consistency': ['consistent_bounce_bonus', 'increasing_strength_bonus']
        }
        
        for group_name, group_params in param_groups.items():
            # Create modified parameters with defaults for this group
            modified_params = params.__dict__.copy()
            default_params = SRStrengthParameters()
            
            for param in group_params:
                modified_params[param] = getattr(default_params, param)
            
            test_params = SRStrengthParameters(**modified_params)
            
            # Calculate score without this group
            test_score = 0
            for tf, data in market_data_dict.items():
                features = self.calculation_cache.get(f"features_{tf}", {})
                sr_levels = self._identify_sr_levels_fast(data, test_params, features)
                test_score += self._evaluate_sr_quality(data, sr_levels, test_params)
            test_score /= len(market_data_dict)
            
            # Importance is the performance drop
            importance[group_name] = max(0, reference_score - test_score)
        
        # Normalize
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v/total_importance for k, v in importance.items()}
        
        return importance
    
    async def _validate_parameters(
        self,
        params: SRStrengthParameters,
        validation_data_dict: Dict[str, pd.DataFrame]
    ) -> float:
        """Validate parameters on out-of-sample data."""
        scores = []
        
        for timeframe, data in validation_data_dict.items():
            # Compute features for validation data
            features = {}
            features['volatility'] = data['close'].pct_change().rolling(20).std()
            features['volume_ma'] = data['volume'].rolling(20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_ma']
            
            # Identify levels
            sr_levels = self._identify_sr_levels_fast(data, params, features)
            
            # Evaluate
            score = self._evaluate_sr_quality(data, sr_levels, params)
            scores.append(score)
        
        return np.mean(scores)
    
    def _create_parameter_set(self, params: Dict[str, Any]) -> SRStrengthParameters:
        """Create parameter set from dictionary."""
        return SRStrengthParameters(**{
            k: v for k, v in params.items() 
            if k in SRStrengthParameters.__dataclass_fields__
        })
    
    def _log_optimization_results(self, result: OptimizationResult) -> None:
        """Log optimization results."""
        
        self.logger.info("📊 S/R Strength Optimization Results:")
        self.logger.info(f"  Optimization Score: {result.optimization_score:.4f}")
        self.logger.info(f"  Optimization Time: {result.optimization_time:.1f}s")
        
        self.logger.info("\n🎯 Optimized Parameters:")
        params = asdict(result.best_parameters)
        for param, value in params.items():
            self.logger.info(f"  {param}: {value}")
        
        self.logger.info("\n📈 Strength Metrics:")
        for metric, value in result.strength_metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        self.logger.info("\n🕐 Multi-Timeframe Scores:")
        for tf, score in result.multi_timeframe_scores.items():
            self.logger.info(f"  {tf}: {score:.4f}")
        
        self.logger.info("\n🎨 Feature Importance:")
        for feature, importance in result.feature_importance.items():
            self.logger.info(f"  {feature}: {importance:.2%}")
    
    def save_optimized_parameters(self, filepath: str) -> None:
        """Save optimized parameters to file."""
        params_dict = asdict(self.best_parameters)
        
        with open(filepath, 'w') as f:
            json.dump({
                "parameters": params_dict,
                "optimization_history": [
                    {
                        "score": r.optimization_score,
                        "metrics": r.strength_metrics,
                        "feature_importance": r.feature_importance,
                        "timestamp": datetime.now().isoformat()
                    }
                    for r in self.optimization_history[-10:]
                ]
            }, f, indent=2)
        
        self.logger.info(f"💾 Saved optimized S/R strength parameters to {filepath}")


class SRLevelIdentifier:
    """Uses optimized parameters to identify strong S/R levels."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("SRLevelIdentifier")
        self.parameters = self._load_parameters()
        
    def _load_parameters(self) -> SRStrengthParameters:
        """Load optimized parameters."""
        try:
            param_file = os.path.join(
                self.config.get("model_save_path", "models"),
                "optimized_sr_strength_parameters.json"
            )
            
            if os.path.exists(param_file):
                with open(param_file, 'r') as f:
                    data = json.load(f)
                    return SRStrengthParameters(**data["parameters"])
            
        except Exception as e:
            self.logger.error(f"Error loading parameters: {e}")
        
        return SRStrengthParameters()  # Default
    
    def identify_strong_sr_levels(
        self,
        market_data: pd.DataFrame,
        min_strength: float = 0.5
    ) -> List[SRLevel]:
        """Identify strong S/R levels using optimized parameters."""
        
        # Create optimizer instance for level identification
        optimizer = SRStrengthOptimizer(self.config)
        
        # Compute features
        features = {
            'volatility': market_data['close'].pct_change().rolling(20).std(),
            'volume_ma': market_data['volume'].rolling(20).mean(),
            'volume_ratio': market_data['volume'] / market_data['volume'].rolling(20).mean()
        }
        
        # Identify levels
        sr_levels = optimizer._identify_sr_levels_fast(
            market_data, self.parameters, features
        )
        
        # Filter by strength
        strong_levels = [l for l in sr_levels if l.strength >= min_strength]
        
        # Sort by strength
        return sorted(strong_levels, key=lambda x: x.strength, reverse=True)