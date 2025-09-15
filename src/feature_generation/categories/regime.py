"""
Comprehensive Regime Features

Market regimes represent different market states with distinct characteristics.
These features help identify and adapt to changing market conditions.
"""
import pandas as pd
import numpy as np
from typing import List
from ..core.feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory

# Core Regime Identification
class RegimeLabelGenerator(FeatureGenerator):
    """Generator for market regime labels (0-3)."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"regime_label_{window}",
            category=FeatureCategory.REGIME,
            description=f"Market regime label over {window} periods (0=low_vol, 1=high_vol, 2=bull, 3=bear)",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window
        )
        super().__init__(config)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # Enhanced regime detection
        returns = close.pct_change()
        volatility = returns.rolling(window=self.window).std()
        trend = close.rolling(window=self.window).apply(lambda x: (x[-1] - x[0]) / x[0])
        
        # Define regimes with thresholds
        regime = pd.Series(0, index=close.index)  # Default: low volatility
        
        # High volatility regime
        vol_threshold = volatility.quantile(0.75)
        regime[volatility > vol_threshold] = 1
        
        # Bull market regime (strong upward trend)
        trend_threshold = 0.02
        regime[(trend > trend_threshold) & (volatility <= vol_threshold)] = 2
        
        # Bear market regime (strong downward trend)
        regime[(trend < -trend_threshold) & (volatility <= vol_threshold)] = 3
        
        return regime

# Regime Probabilities
class RegimeProbabilityGenerator(FeatureGenerator):
    """Generator for regime probabilities."""
    
    def __init__(self, regime_id: int = 0, window: int = 20):
        config = FeatureConfig(
            name=f"regime_{regime_id}_probability_{window}",
            category=FeatureCategory.REGIME,
            description=f"Probability of being in regime {regime_id} over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window
        )
        super().__init__(config)
        self.regime_id = regime_id
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        returns = close.pct_change()
        volatility = returns.rolling(window=self.window).std()
        trend = close.rolling(window=self.window).apply(lambda x: (x[-1] - x[0]) / x[0])
        
        # Calculate regime probabilities using soft thresholds
        vol_threshold = volatility.quantile(0.75)
        trend_threshold = 0.02
        
        if self.regime_id == 0:  # Low volatility regime
            prob = np.exp(-volatility / vol_threshold)
        elif self.regime_id == 1:  # High volatility regime
            prob = 1 / (1 + np.exp(-(volatility - vol_threshold) / vol_threshold))
        elif self.regime_id == 2:  # Bull market regime
            prob = 1 / (1 + np.exp(-(trend - trend_threshold) / trend_threshold))
            prob = prob * (volatility <= vol_threshold).astype(float)
        else:  # Bear market regime
            prob = 1 / (1 + np.exp((trend + trend_threshold) / trend_threshold))
            prob = prob * (volatility <= vol_threshold).astype(float)
        
        return prob.clip(0, 1)

# Regime Transition Analysis
class RegimeTransitionProbabilityGenerator(FeatureGenerator):
    """Generator for regime transition probabilities."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"regime_transition_probability_{window}",
            category=FeatureCategory.REGIME,
            description=f"Probability of regime transition over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window
        )
        super().__init__(config)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        returns = close.pct_change()
        volatility = returns.rolling(window=self.window).std()
        trend = close.rolling(window=self.window).apply(lambda x: (x[-1] - x[0]) / x[0])
        
        # Calculate regime change indicators
        vol_change = volatility.pct_change().abs()
        trend_change = trend.pct_change().abs()
        
        # Transition probability based on changes in market characteristics
        transition_prob = (vol_change + trend_change) / 2
        transition_prob = transition_prob / transition_prob.quantile(0.9)
        
        return transition_prob.clip(0, 1).fillna(0)

class RegimeDurationGenerator(FeatureGenerator):
    """Generator for regime duration features."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"regime_duration_{window}",
            category=FeatureCategory.REGIME,
            description=f"Duration in current regime over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window
        )
        super().__init__(config)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        returns = close.pct_change()
        volatility = returns.rolling(window=self.window).std()
        
        # Detect regime changes
        vol_threshold = volatility.std()
        regime_changes = (volatility.diff().abs() > vol_threshold).astype(int)
        
        # Calculate duration since last regime change
        duration = regime_changes.cumsum().groupby(regime_changes.cumsum()).cumcount() + 1
        
        return duration

class RegimeStabilityGenerator(FeatureGenerator):
    """Generator for regime stability features."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"regime_stability_{window}",
            category=FeatureCategory.REGIME,
            description=f"Stability of current regime over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window
        )
        super().__init__(config)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        returns = close.pct_change()
        volatility = returns.rolling(window=self.window).std()
        
        # Calculate regime stability as inverse of volatility of volatility
        vol_of_vol = volatility.rolling(window=self.window).std()
        stability = 1 / (1 + vol_of_vol)
        
        return stability.fillna(0)

# Enhanced Regime Features
class RegimeVolatilityGenerator(FeatureGenerator):
    """Generator for regime-specific volatility features."""
    
    def __init__(self, regime_id: int = 0, window: int = 20):
        config = FeatureConfig(
            name=f"regime_{regime_id}_volatility_{window}",
            category=FeatureCategory.REGIME,
            description=f"Volatility characteristics of regime {regime_id} over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window
        )
        super().__init__(config)
        self.regime_id = regime_id
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        returns = close.pct_change()
        volatility = returns.rolling(window=self.window).std()
        
        # Get regime labels
        vol_threshold = volatility.quantile(0.75)
        trend = close.rolling(window=self.window).apply(lambda x: (x[-1] - x[0]) / x[0])
        trend_threshold = 0.02
        
        regime = pd.Series(0, index=close.index)
        regime[volatility > vol_threshold] = 1
        regime[(trend > trend_threshold) & (volatility <= vol_threshold)] = 2
        regime[(trend < -trend_threshold) & (volatility <= vol_threshold)] = 3
        
        # Calculate regime-specific volatility
        regime_vol = volatility.copy()
        regime_vol[regime != self.regime_id] = np.nan
        regime_vol = regime_vol.fillna(method='ffill')
        
        return regime_vol

class RegimeMomentumGenerator(FeatureGenerator):
    """Generator for regime-specific momentum features."""
    
    def __init__(self, regime_id: int = 0, window: int = 20):
        config = FeatureConfig(
            name=f"regime_{regime_id}_momentum_{window}",
            category=FeatureCategory.REGIME,
            description=f"Momentum characteristics of regime {regime_id} over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window
        )
        super().__init__(config)
        self.regime_id = regime_id
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        returns = close.pct_change()
        volatility = returns.rolling(window=self.window).std()
        trend = close.rolling(window=self.window).apply(lambda x: (x[-1] - x[0]) / x[0])
        
        # Get regime labels
        vol_threshold = volatility.quantile(0.75)
        trend_threshold = 0.02
        
        regime = pd.Series(0, index=close.index)
        regime[volatility > vol_threshold] = 1
        regime[(trend > trend_threshold) & (volatility <= vol_threshold)] = 2
        regime[(trend < -trend_threshold) & (volatility <= vol_threshold)] = 3
        
        # Calculate regime-specific momentum
        momentum = returns.rolling(window=self.window).sum()
        regime_momentum = momentum.copy()
        regime_momentum[regime != self.regime_id] = np.nan
        regime_momentum = regime_momentum.fillna(method='ffill')
        
        return regime_momentum

class RegimeTrendGenerator(FeatureGenerator):
    """Generator for regime-specific trend features."""
    
    def __init__(self, regime_id: int = 0, window: int = 20):
        config = FeatureConfig(
            name=f"regime_{regime_id}_trend_{window}",
            category=FeatureCategory.REGIME,
            description=f"Trend characteristics of regime {regime_id} over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window
        )
        super().__init__(config)
        self.regime_id = regime_id
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        returns = close.pct_change()
        volatility = returns.rolling(window=self.window).std()
        trend = close.rolling(window=self.window).apply(lambda x: (x[-1] - x[0]) / x[0])
        
        # Get regime labels
        vol_threshold = volatility.quantile(0.75)
        trend_threshold = 0.02
        
        regime = pd.Series(0, index=close.index)
        regime[volatility > vol_threshold] = 1
        regime[(trend > trend_threshold) & (volatility <= vol_threshold)] = 2
        regime[(trend < -trend_threshold) & (volatility <= vol_threshold)] = 3
        
        # Calculate regime-specific trend
        regime_trend = trend.copy()
        regime_trend[regime != self.regime_id] = np.nan
        regime_trend = regime_trend.fillna(method='ffill')
        
        return regime_trend

class RegimeVolumeGenerator(FeatureGenerator):
    """Generator for regime-specific volume features."""
    
    def __init__(self, regime_id: int = 0, window: int = 20):
        config = FeatureConfig(
            name=f"regime_{regime_id}_volume_{window}",
            category=FeatureCategory.REGIME,
            description=f"Volume characteristics of regime {regime_id} over {window} periods",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window
        )
        super().__init__(config)
        self.regime_id = regime_id
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        volume = data['volume']
        
        returns = close.pct_change()
        volatility = returns.rolling(window=self.window).std()
        trend = close.rolling(window=self.window).apply(lambda x: (x[-1] - x[0]) / x[0])
        
        # Get regime labels
        vol_threshold = volatility.quantile(0.75)
        trend_threshold = 0.02
        
        regime = pd.Series(0, index=close.index)
        regime[volatility > vol_threshold] = 1
        regime[(trend > trend_threshold) & (volatility <= vol_threshold)] = 2
        regime[(trend < -trend_threshold) & (volatility <= vol_threshold)] = 3
        
        # Calculate regime-specific volume
        volume_ma = volume.rolling(window=self.window).mean()
        regime_volume = volume_ma.copy()
        regime_volume[regime != self.regime_id] = np.nan
        regime_volume = regime_volume.fillna(method='ffill')
        
        return regime_volume

def create_default_regime_generators() -> List[FeatureGenerator]:
    """Create comprehensive regime feature generators."""
    generators = []
    windows = [10, 20, 50]
    
    for window in windows:
        # Core regime features
        generators.extend([
            RegimeLabelGenerator(window),
            RegimeTransitionProbabilityGenerator(window),
            RegimeDurationGenerator(window),
            RegimeStabilityGenerator(window),
        ])
        
        # Regime probabilities for each regime type (0-3)
        for regime_id in range(4):
            generators.extend([
                RegimeProbabilityGenerator(regime_id, window),
                RegimeVolatilityGenerator(regime_id, window),
                RegimeMomentumGenerator(regime_id, window),
                RegimeTrendGenerator(regime_id, window),
                RegimeVolumeGenerator(regime_id, window),
            ])
    
    return generators