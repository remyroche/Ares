"""
Regime Tagging Step
-------------------
Calculates physics features and assigns regime tags using AdaptiveHunterRouter.
"""

import pandas as pd
import numpy as np
from src.training.steps.base_step import BaseStep
from src.utils.ml_common.regime.adaptive_hunter_router import AdaptiveHunterRouter
from src.utils.tprint import tprint_info, tprint_success, tprint_warning

class RegimeTaggingStep(BaseStep):
    """
    Tags data with market regimes (Quiet, Trending, Chaos).
    """

    def __init__(self, step_name: str = 'regime_tagging_step', **kwargs):
        super().__init__(step_name)
        self.router = AdaptiveHunterRouter()

    def _calculate_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate: [Vol_Intensity, Efficiency, MP_Dist, Wavelet_Entropy]
        """
        # 1. Volatility Intensity
        # Z-score of volatility relative to long-term
        if 'volatility_1d' in df.columns:
            vol = df['volatility_1d']
        else:
            vol = df['close'].pct_change().rolling(20).std()

        vol_long = vol.rolling(100).mean()
        vol_std = vol.rolling(100).std()
        vol_intensity = (vol - vol_long) / (vol_std + 1e-9)

        # 2. Efficiency Ratio
        close = df['close']
        window = 10
        change = (close - close.shift(window)).abs()
        path = close.diff().abs().rolling(window).sum()
        efficiency = change / (path + 1e-9)

        # 3. Market Profile Distance (Proxy: Dist from VWAP)
        if 'vwap' in df.columns:
            mp_dist = (close - df['vwap']).abs() / (close + 1e-9)
        else:
            # Fallback: Dist from SMA 20
            sma = close.rolling(20).mean()
            mp_dist = (close - sma).abs() / (close + 1e-9)

        # 4. Entropy (Proxy if wavelet not avail: Rolling Permutation Entropy or simpler)
        # Using simple log return distribution entropy on short window
        def roll_entropy(x):
            try:
                hist, _ = np.histogram(x, bins=5, density=True)
                hist = hist[hist > 0]
                return -np.sum(hist * np.log(hist))
            except:
                return 0.0

        entropy = df['close'].pct_change().rolling(20).apply(roll_entropy, raw=True)

        features = pd.DataFrame({
            'vol_intensity': vol_intensity,
            'efficiency': efficiency,
            'mp_dist': mp_dist,
            'entropy': entropy
        }, index=df.index).fillna(0.0)

        # Clip outliers
        return features.clip(-5, 5)

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        tprint_info("🏷️  Regime Tagging: Calculating physics features...")
        features = self._calculate_physics_features(df)

        tprint_info("🧠 Regime Tagging: Fitting Adaptive Hunter Router...")
        # Fit on first 20% or full history?
        # Ideally fit on a warm-up period. Here we fit on full for stability in backtest context.
        # For live, we would load a pretrained router.
        self.router.fit(features.values)

        tprint_info("🔮 Regime Tagging: Predicting regimes...")
        regime_df = self.router.predict_batch(features)

        # Join to DF
        result = df.copy()
        for col in regime_df.columns:
            result[col] = regime_df[col]

        # Summary
        counts = result['regime_label'].value_counts()
        tprint_success(f"✅ Regime Tagging Complete. Distribution:\n{counts}")

        return result

def register_regime_tagging_step() -> None:
    from src.training.steps.base_step import step_registry
    step_registry.register("regime_tagging_step", RegimeTaggingStep)
