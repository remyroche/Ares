"""
Continuous Predictor Geometry Generators

This module generates DENSE, CONTINUOUS predictors that naturally satisfy CI tests.
Unlike event-based triggers, these signals are:
- Dense (values at every timestamp)
- Smooth (not binary spikes)
- Causally grounded (derived from structural features)

These are the "gold" signals per De Prado (2026) - residualized predictors
that survive conditional independence tests because shared information is removed.

Signal Role: PREDICTOR (expected return magnitude)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

# Import role enum
try:
    from src.training.steps.labeling.causal_quality_assessment import SignalRole
except ImportError:
    class SignalRole(Enum):
        PREDICTOR = "predictor"
        TRIGGER = "trigger"
        INTERACTION = "interaction"
        CONTEXT = "context"

try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning
except ImportError:
    def tprint_info(msg): print(f"INFO: {msg}")
    def tprint_success(msg): print(f"SUCCESS: {msg}")
    def tprint_warning(msg): print(f"WARNING: {msg}")


@dataclass
class PredictorGeometry:
    """Container for a continuous predictor geometry."""
    name: str
    family: str
    values: pd.Series
    role: SignalRole = SignalRole.PREDICTOR
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def uuid(self) -> str:
        return f"{self.family}_{self.name}"


class ContinuousPredictorGenerator:
    """
    Generates continuous predictor geometries that pass CI tests.
    
    These predictors remain stationary and informative even when price is flat,
    making them ideal for low-volatility regime learning.
    """
    
    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 100,
        horizons: List[int] = [12, 24, 48, 96],
        verbose: bool = True
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.horizons = horizons
        self.verbose = verbose
        
    def generate_all_predictors(self, df: pd.DataFrame) -> List[PredictorGeometry]:
        """
        Generate all continuous predictor geometries.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            List of PredictorGeometry objects with role=PREDICTOR
        """
        if self.verbose:
            tprint_info("📊 Generating continuous predictor geometries...")
        
        predictors = []
        
        # 1. Surprise Intensity (continuous Z-scores)
        predictors.extend(self._generate_surprise_intensity(df))
        
        # 2. Relaxation / Mean-Reversion Geometry
        predictors.extend(self._generate_relaxation_features(df))
        
        # 3. Flow Pressure (signed, dense)
        predictors.extend(self._generate_flow_pressure(df))
        
        # 4. Multi-horizon Slope Predictors
        predictors.extend(self._generate_slope_predictors(df))
        
        # 5. Market Fragility / Resilience (NEW)
        predictors.extend(self._generate_fragility_resilience(df))
        
        if self.verbose:
            tprint_success(f"   ✅ Generated {len(predictors)} continuous predictors")
        
        return predictors
    
    def _generate_surprise_intensity(self, df: pd.DataFrame) -> List[PredictorGeometry]:
        """
        Generate continuous surprise intensity (NOT events).
        
        Instead of: surprise = abs(x - μ) > kσ  (boolean)
        We emit:    surprise_z = (x - μ) / σ     (continuous, signed)
        
        Properties: Dense, Signed, Mean-reverting capability
        """
        predictors = []
        close = df['close']
        returns = close.pct_change()
        volume = df['volume']
        
        for window in [self.short_window, self.long_window]:
            # Return surprise Z-score (Ranking Transformation)
            # Use shift(1) to ensure strict ex-ante mean/std (no leakage)
            ret_mean = returns.rolling(window).mean().shift(1)
            ret_std = returns.rolling(window).std().shift(1)

            # Stabilization: Ensure denominator is not effectively zero
            ret_std = ret_std.replace(0.0, 1e-9).fillna(1e-9)

            z_raw = (returns - ret_mean) / ret_std
            # Clip BEFORE ranking to handle extreme outliers
            z_clipped = z_raw.clip(-4, 4)
            # Rolling percentile (Rank)
            # Use pandas rank pct=True on rolling window
            # Note: rolling().rank() is O(N*W), can be slow.
            # Fallback: Just use tanh transformation of clipped Z to bound it safely without expensive ranking
            # tanh(z/2) maps -4..4 to -0.96..0.96 smoothly
            surprise_z = np.tanh(z_clipped / 2.0).fillna(0)
            
            predictors.append(PredictorGeometry(
                name=f"return_surprise_z_{window}",
                family="SURPRISE_Z_CONTINUOUS",
                values=surprise_z,
                metadata={"window": window, "source": "returns", "transform": "tanh_rank_proxy"}
            ))
            
            # Volume surprise Z-score
            # Use shift(1) to ensure strict ex-ante mean/std
            vol_mean = volume.rolling(window).mean().shift(1)
            vol_std = volume.rolling(window).std().shift(1)

            # Stabilization
            vol_std = vol_std.replace(0.0, 1e-9).fillna(1e-9)

            vol_z_raw = (volume - vol_mean) / vol_std
            vol_z_clipped = vol_z_raw.clip(-4, 4)
            vol_surprise_z = np.tanh(vol_z_clipped / 2.0).fillna(0)
            
            predictors.append(PredictorGeometry(
                name=f"volume_surprise_z_{window}",
                family="SURPRISE_Z_CONTINUOUS",
                values=vol_surprise_z,
                metadata={"window": window, "source": "volume", "transform": "tanh_rank_proxy"}
            ))
            
            # Volatility surprise Z-score
            vol_realized = returns.rolling(window).std()
            # Use shift(1) to ensure strict ex-ante mean/std
            vol_realized_mean = vol_realized.rolling(window * 2).mean().shift(1)
            vol_realized_std = vol_realized.rolling(window * 2).std().shift(1)

            # Stabilization
            vol_realized_std = vol_realized_std.replace(0.0, 1e-9).fillna(1e-9)

            vv_z_raw = (vol_realized - vol_realized_mean) / vol_realized_std
            vv_z_clipped = vv_z_raw.clip(-4, 4)
            vol_vol_z = np.tanh(vv_z_clipped / 2.0).fillna(0)
            
            predictors.append(PredictorGeometry(
                name=f"vol_surprise_z_{window}",
                family="SURPRISE_Z_CONTINUOUS",
                values=vol_vol_z,
                metadata={"window": window, "source": "volatility", "transform": "tanh_rank_proxy"}
            ))
        
        # Log-ratio surprise (alternative formulation)
        for window in [self.short_window]:
            median_price = close.rolling(window).median()
            log_ratio = np.log(close / median_price).clip(-0.5, 0.5).fillna(0)
            
            predictors.append(PredictorGeometry(
                name=f"surprise_log_ratio_{window}",
                family="SURPRISE_LOG_RATIO",
                values=log_ratio,
                metadata={"window": window}
            ))
        
        return predictors
    
    def _generate_relaxation_features(self, df: pd.DataFrame) -> List[PredictorGeometry]:
        """
        Generate time-to-mean / relaxation geometry.
        
        Markets don't just move — they relax.
        distance = (price - mean) / vol
        decay_rate = -Δdistance / Δt
        
        Positive decay_rate = reverting back to mean
        Negative decay_rate = accelerating away
        """
        predictors = []
        close = df['close']
        
        for window in [self.short_window, self.long_window]:
            rolling_mean = close.rolling(window).mean()
            rolling_vol = close.rolling(window).std()
            
            # Distance from mean (normalized)
            distance = ((close - rolling_mean) / (rolling_vol + 1e-9)).clip(-5, 5).fillna(0)
            
            predictors.append(PredictorGeometry(
                name=f"distance_from_mean_{window}",
                family="MEAN_REVERSION",
                values=distance,
                metadata={"window": window, "type": "distance"}
            ))
            
            # Decay rate (derivative of distance)
            decay_rate = -distance.diff().clip(-1, 1).fillna(0)
            
            predictors.append(PredictorGeometry(
                name=f"decay_rate_{window}",
                family="MEAN_REVERSION",
                values=decay_rate,
                metadata={"window": window, "type": "decay_rate"}
            ))
            
            # Acceleration (second derivative)
            acceleration = decay_rate.diff().clip(-0.5, 0.5).fillna(0)
            
            predictors.append(PredictorGeometry(
                name=f"relaxation_accel_{window}",
                family="MEAN_REVERSION",
                values=acceleration,
                metadata={"window": window, "type": "acceleration"}
            ))
        
        return predictors
    
    def _generate_flow_pressure(self, df: pd.DataFrame) -> List[PredictorGeometry]:
        """
        Generate flow imbalance as signed pressure (not events).
        
        Bad (trigger):  flow_imbalance > threshold
        Good (predictor): flow_pressure = (buy_vol - sell_vol) / total_vol
        
        Properties: Continuous, Cross-asset transferable, Survives CI tests
        """
        predictors = []
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Bar close position as buy/sell proxy
        bar_range = (high - low).replace(0, 1e-9)
        close_position = (close - low) / bar_range  # 0=low, 1=high
        
        # Buy volume estimate (close near high = buy pressure)
        buy_vol = close_position * volume
        sell_vol = (1 - close_position) * volume
        
        for window in [self.short_window, self.long_window]:
            total_vol = volume.rolling(window).sum()
            buy_rolling = buy_vol.rolling(window).sum()
            sell_rolling = sell_vol.rolling(window).sum()
            
            # Flow pressure (signed, -1 to +1)
            flow_pressure = ((buy_rolling - sell_rolling) / (total_vol + 1e-9)).clip(-1, 1).fillna(0)
            
            predictors.append(PredictorGeometry(
                name=f"flow_pressure_{window}",
                family="FLOW_PRESSURE_CONTINUOUS",
                values=flow_pressure,
                metadata={"window": window, "type": "pressure"}
            ))
            
            # Flow acceleration (change in pressure)
            flow_accel = flow_pressure.diff().clip(-0.5, 0.5).fillna(0)
            
            predictors.append(PredictorGeometry(
                name=f"flow_accel_{window}",
                family="FLOW_PRESSURE_CONTINUOUS",
                values=flow_accel,
                metadata={"window": window, "type": "acceleration"}
            ))
            
            # Flow momentum (pressure * volume)
            flow_momentum = (flow_pressure * volume / (volume.rolling(window).mean() + 1e-9)).clip(-5, 5).fillna(0)
            
            predictors.append(PredictorGeometry(
                name=f"flow_momentum_{window}",
                family="FLOW_PRESSURE_CONTINUOUS",
                values=flow_momentum,
                metadata={"window": window, "type": "momentum"}
            ))
        
        return predictors
    
    def _generate_slope_predictors(self, df: pd.DataFrame) -> List[PredictorGeometry]:
        """
        Generate multi-horizon slope predictors.
        
        Instead of return shocks, generate:
        slope_h = linreg_slope(log_price, window=h)
        curvature = slope_short - slope_long
        
        Properties: Dense, Smooth, Regime-sensitive
        """
        predictors = []
        log_price = np.log(df['close'])
        
        # Pre-compute slopes for each horizon
        slopes = {}
        for h in self.horizons:
            slope_series = pd.Series(0.0, index=df.index)
            for i in range(h, len(df)):
                y = log_price.iloc[i-h:i].values
                x = np.arange(h)
                if len(y) == h:
                    slope = np.polyfit(x, y, 1)[0]
                    slope_series.iloc[i] = slope
            slopes[h] = slope_series
            
            # Raw slope (normalized by typical magnitude)
            slope_normalized = (slopes[h] / 0.001).clip(-5, 5).fillna(0)  # 0.1% per bar typical
            
            predictors.append(PredictorGeometry(
                name=f"slope_h{h}",
                family="MULTI_HORIZON_SLOPE",
                values=slope_normalized,
                metadata={"horizon": h, "type": "raw"}
            ))
        
        # Curvature (difference between short and long slopes)
        if len(self.horizons) >= 2:
            short_h = min(self.horizons)
            long_h = max(self.horizons)
            
            curvature = ((slopes[short_h] - slopes[long_h]) / 0.0005).clip(-5, 5).fillna(0)
            
            predictors.append(PredictorGeometry(
                name=f"curvature_h{short_h}_h{long_h}",
                family="MULTI_HORIZON_SLOPE",
                values=curvature,
                metadata={"short_h": short_h, "long_h": long_h, "type": "curvature"}
            ))
            
            # Slope divergence (acceleration of slope difference)
            divergence = curvature.diff().clip(-2, 2).fillna(0)
            
            predictors.append(PredictorGeometry(
                name=f"slope_divergence_h{short_h}_h{long_h}",
                family="MULTI_HORIZON_SLOPE",
                values=divergence,
                metadata={"short_h": short_h, "long_h": long_h, "type": "divergence"}
            ))
        
        return predictors
    
    def _generate_fragility_resilience(self, df: pd.DataFrame) -> List[PredictorGeometry]:
        """
        Generate Market Fragility / Resilience predictors.
        
        Foundation:
            fragility = abs(return) / (abs(flow) + eps)
            fragility_z = zscore(fragility)
        
        Interpretation:
        - High fragility: Large price moves with small flow → Illiquid/toxic
        - Low fragility: Small price moves with large flow → Resilient market
        
        This captures market microstructure state that persists across regimes.
        """
        predictors = []
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Compute absolute return
        abs_return = close.pct_change().abs()
        
        # Compute flow estimate (signed volume via bar position)
        bar_range = (high - low).replace(0, 1e-9)
        close_position = (close - low) / bar_range  # 0=low, 1=high
        signed_flow = (close_position - 0.5) * 2 * volume  # -volume to +volume
        abs_flow = signed_flow.abs()
        
        for window in [self.short_window, self.long_window]:
            # Core fragility measure
            # Rolling sum to smooth noise
            rolling_abs_return = abs_return.rolling(window).sum()
            rolling_abs_flow = abs_flow.rolling(window).sum()
            
            fragility = rolling_abs_return / (rolling_abs_flow / volume.rolling(window).mean() + 1e-9)
            
            # Z-score the fragility
            fragility_mean = fragility.rolling(window * 5).mean()
            fragility_std = fragility.rolling(window * 5).std()
            fragility_z = ((fragility - fragility_mean) / (fragility_std + 1e-9)).clip(-5, 5).fillna(0)
            
            predictors.append(PredictorGeometry(
                name=f"fragility_z_{window}",
                family="MARKET_FRAGILITY",
                values=fragility_z,
                metadata={"window": window, "type": "fragility_z"}
            ))
            
            # Resilience is inverse of fragility (how much flow needed per unit return)
            resilience = 1.0 / (fragility + 1e-9)
            resilience_mean = resilience.rolling(window * 5).mean()
            resilience_std = resilience.rolling(window * 5).std()
            resilience_z = ((resilience - resilience_mean) / (resilience_std + 1e-9)).clip(-5, 5).fillna(0)
            
            predictors.append(PredictorGeometry(
                name=f"resilience_z_{window}",
                family="MARKET_FRAGILITY",
                values=resilience_z,
                metadata={"window": window, "type": "resilience_z"}
            ))
            
            # Fragility regime change (acceleration of fragility)
            fragility_accel = fragility_z.diff().clip(-2, 2).fillna(0)
            
            predictors.append(PredictorGeometry(
                name=f"fragility_accel_{window}",
                family="MARKET_FRAGILITY",
                values=fragility_accel,
                metadata={"window": window, "type": "fragility_acceleration"}
            ))
            
            # Kyle's Lambda proxy: Price impact per unit volume
            # lambda = abs(return) / volume
            kyle_lambda = abs_return / (volume / volume.rolling(window).mean() + 1e-9)
            kyle_lambda_mean = kyle_lambda.rolling(window * 3).mean()
            kyle_lambda_std = kyle_lambda.rolling(window * 3).std()
            kyle_lambda_z = ((kyle_lambda - kyle_lambda_mean) / (kyle_lambda_std + 1e-9)).clip(-5, 5).fillna(0)
            
            predictors.append(PredictorGeometry(
                name=f"kyle_lambda_z_{window}",
                family="MARKET_FRAGILITY",
                values=kyle_lambda_z,
                metadata={"window": window, "type": "price_impact"}
            ))
        
        return predictors


class CausalResidualGenerator:
    """
    Generate causal residual predictors by subtracting parent expectations.
    
    residual = target_feature - E[target_feature | parents]
    
    These yield:
    - Orthogonal predictors
    - Low CI penalty
    - High standalone validity
    
    This is De Prado gold.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def generate_residual_predictors(
        self,
        df: pd.DataFrame,
        target_features: pd.DataFrame,
        parent_features: pd.DataFrame
    ) -> List[PredictorGeometry]:
        """
        Generate residualized predictors by regressing out parent effects.
        
        Args:
            df: OHLCV DataFrame for context
            target_features: Features to residualize
            parent_features: Causal parent features to regress out
            
        Returns:
            List of PredictorGeometry with residualized values
        """
        from sklearn.linear_model import Ridge
        
        predictors = []
        
        if self.verbose:
            tprint_info(f"📊 Generating {len(target_features.columns)} causal residual predictors...")
        
        # Align indices
        common_idx = target_features.index.intersection(parent_features.index)
        if len(common_idx) < 100:
            tprint_warning(f"   ⚠️ Insufficient common indices: {len(common_idx)}")
            return []
        
        X_parents = parent_features.loc[common_idx].fillna(0)
        
        for col in target_features.columns:
            try:
                y_target = target_features.loc[common_idx, col].fillna(0).values
                
                # Fit Ridge regression
                model = Ridge(alpha=1.0)
                model.fit(X_parents, y_target)
                
                # Compute residual
                y_pred = model.predict(X_parents)
                residual = y_target - y_pred
                
                # Normalize residual
                std_residual = np.std(residual)
                if std_residual > 1e-9:
                    residual_z = np.clip(residual / std_residual, -5, 5)
                else:
                    residual_z = residual
                
                residual_series = pd.Series(residual_z, index=common_idx)
                
                predictors.append(PredictorGeometry(
                    name=f"residual_{col}",
                    family="CAUSAL_RESIDUAL",
                    values=residual_series,
                    metadata={
                        "original_feature": col,
                        "n_parents": len(parent_features.columns),
                        "explained_var": 1 - np.var(residual) / (np.var(y_target) + 1e-9)
                    }
                ))
                
            except Exception as e:
                if self.verbose:
                    tprint_warning(f"   ⚠️ Failed to residualize {col}: {e}")
        
        if self.verbose:
            tprint_success(f"   ✅ Generated {len(predictors)} causal residual predictors")
        
        return predictors


def generate_continuous_predictors(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Convenience function to generate all continuous predictors as a DataFrame.
    
    Args:
        df: OHLCV DataFrame
        verbose: Whether to print progress
        
    Returns:
        DataFrame with all continuous predictor columns
    """
    generator = ContinuousPredictorGenerator(verbose=verbose)
    predictors = generator.generate_all_predictors(df)
    
    # Combine into DataFrame
    result = pd.DataFrame(index=df.index)
    for pred in predictors:
        # Align to df index
        result[pred.uuid] = pred.values.reindex(df.index).fillna(0)
    
    return result
