"""Regime features"""
import pandas as pd
import numpy as np
from typing import List
from ..core.feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory

class RegimeLabelGenerator(FeatureGenerator):
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"regime_label_{window}",
            category=FeatureCategory.REGIME,
            description=f"Regime label over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window
        )
        super().__init__(config)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # Simple regime detection based on volatility and trend
        returns = close.pct_change()
        volatility = returns.rolling(window=self.window).std()
        trend = close.rolling(window=self.window).apply(lambda x: (x[-1] - x[0]) / x[0])
        
        # Define regimes: 0=low vol, 1=high vol, 2=trending up, 3=trending down
        regime = pd.Series(0, index=close.index)
        regime[volatility > volatility.quantile(0.7)] = 1
        regime[trend > 0.02] = 2
        regime[trend < -0.02] = 3
        
        return regime

class RegimeProbabilityGenerator(FeatureGenerator):
    def __init__(self, regime_id: int = 0, window: int = 20):
        config = FeatureConfig(
            name=f"regime_{regime_id}_probability_{window}",
            category=FeatureCategory.REGIME,
            description=f"Regime {regime_id} probability over {window} periods",
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
        
        # Simple regime detection
        returns = close.pct_change()
        volatility = returns.rolling(window=self.window).std()
        trend = close.rolling(window=self.window).apply(lambda x: (x[-1] - x[0]) / x[0])
        
        # Calculate probability of being in regime
        if self.regime_id == 0:  # Low volatility
            prob = 1 - (volatility / volatility.quantile(0.9))
        elif self.regime_id == 1:  # High volatility
            prob = volatility / volatility.quantile(0.9)
        elif self.regime_id == 2:  # Trending up
            prob = np.maximum(0, trend / 0.05)
        else:  # Trending down
            prob = np.maximum(0, -trend / 0.05)
        
        return prob.clip(0, 1)

class RegimeTransitionProbabilityGenerator(FeatureGenerator):
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"regime_transition_probability_{window}",
            category=FeatureCategory.REGIME,
            description=f"Regime transition probability over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window
        )
        super().__init__(config)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # Calculate regime changes
        returns = close.pct_change()
        volatility = returns.rolling(window=self.window).std()
        
        # Transition probability based on volatility changes
        vol_change = volatility.pct_change().abs()
        transition_prob = vol_change / vol_change.quantile(0.9)
        
        return transition_prob.clip(0, 1)

class RegimeDurationGenerator(FeatureGenerator):
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"regime_duration_{window}",
            category=FeatureCategory.REGIME,
            description=f"Regime duration over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window
        )
        super().__init__(config)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # Simple regime detection
        returns = close.pct_change()
        volatility = returns.rolling(window=self.window).std()
        
        # Calculate how long we've been in current regime
        regime_changes = (volatility.diff().abs() > volatility.std()).astype(int)
        duration = regime_changes.cumsum().groupby(regime_changes.cumsum()).cumcount() + 1
        
        return duration

class RegimeStabilityGenerator(FeatureGenerator):
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"regime_stability_{window}",
            category=FeatureCategory.REGIME,
            description=f"Regime stability over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window
        )
        super().__init__(config)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # Calculate regime stability based on consistency of market behavior
        returns = close.pct_change()
        volatility = returns.rolling(window=self.window).std()
        vol_stability = 1 - (volatility.rolling(window=self.window).std() / volatility)
        
        return vol_stability.fillna(0)

def create_default_regime_generators() -> List[FeatureGenerator]:
    generators = []
    windows = [10, 20, 50]
    
    for window in windows:
        generators.extend([
            RegimeLabelGenerator(window),
            RegimeTransitionProbabilityGenerator(window),
            RegimeDurationGenerator(window),
            RegimeStabilityGenerator(window),
        ])
        
        # Add regime probabilities for each regime type
        for regime_id in range(4):
            generators.append(RegimeProbabilityGenerator(regime_id, window))
    
    return generators