"""
Dynamic market target builders for explaining market dynamics beyond price change.

This module defines a catalog of dynamic targets (classification/regression) that
capture different aspects of market behavior such as momentum continuation,
volatility state, breakout propensity, reversal violence, tail events, liquidity
stress, efficiency regime, and regime transition risk.

All targets are computed with strict lagging to avoid lookahead bias. Horizons
and lookbacks are configurable via the configuration dataclass.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import system_logger


@dataclass
class DynamicTargetConfig:
    """Configuration for building dynamic targets.

    Attributes:
        momentum_windows: Lookback windows for momentum signals.
        momentum_horizons: Horizons for continuation/reversal classification.
        volatility_window: Lookback for realized volatility.
        volatility_quantiles: Quantile bins for volatility state classification.
        breakout_lookback: Lookback for band/level computation.
        breakout_horizon: Horizon to check for breakout events.
        reversal_horizon: Horizon to measure reversal violence.
        tail_lookback: Lookback window for rolling VaR thresholds.
        tail_alpha: Tail probability (e.g., 0.05 for 95% VaR exceedance).
        efficiency_window: Window for efficiency ratio.
        transition_horizon: Horizon for regime transition risk proxy.
        feature_lag: Number of periods to lag features to prevent leakage.
    """

    momentum_windows: List[int] = None
    momentum_horizons: List[int] = None
    volatility_window: int = 20
    volatility_quantiles: Tuple[float, float, float] = (0.2, 0.5, 0.8)
    breakout_lookback: int = 20
    breakout_horizon: int = 5
    reversal_horizon: int = 10
    tail_lookback: int = 100
    tail_alpha: float = 0.05
    efficiency_window: int = 20
    transition_horizon: int = 10
    feature_lag: int = 1

    def __post_init__(self):
        if self.momentum_windows is None:
            self.momentum_windows = [5, 10, 20]
        if self.momentum_horizons is None:
            self.momentum_horizons = [5, 10]


class DynamicTargetsBuilder:
    """Builder for dynamic targets used in discovery pipelines."""

    def __init__(self, config: Optional[DynamicTargetConfig] = None):
        self.config = config or DynamicTargetConfig()
        self.logger = system_logger.getChild('DynamicTargetsBuilder')

    def build_all(self, market_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Build a comprehensive set of dynamic targets.

        The resulting targets are strictly aligned to use only past information
        at time t (features should later be lagged by `feature_lag`).
        """
        targets: Dict[str, pd.Series] = {}

        close = market_data['close'] if 'close' in market_data.columns else None
        high = market_data['high'] if 'high' in market_data.columns else None
        low = market_data['low'] if 'low' in market_data.columns else None
        volume = market_data['volume'] if 'volume' in market_data.columns else None

        if close is None:
            self.logger.warning('No close prices found. Dynamic targets will be limited.')
            return targets

        returns = close.pct_change().fillna(0)

        # 1) Momentum continuation / reversal (classification)
        for w in self.config.momentum_windows:
            signal = returns.rolling(w).mean()
            for h in self.config.momentum_horizons:
                future_signal = signal.shift(-h)
                # 1 if sign persists after h, else 0
                cont = ((signal > 0) & (future_signal > 0)) | ((signal < 0) & (future_signal < 0))
                targets[f'momentum_continuation_w{w}_h{h}'] = cont.astype(int).shift(self.config.feature_lag).fillna(0)
                # Reversal
                rev = ((signal > 0) & (future_signal < 0)) | ((signal < 0) & (future_signal > 0))
                targets[f'momentum_reversal_w{w}_h{h}'] = rev.astype(int).shift(self.config.feature_lag).fillna(0)

        # 2) Volatility state (classification)
        vol = returns.rolling(self.config.volatility_window).std().fillna(0)
        q1, q2, q3 = self.config.volatility_quantiles
        vol_state = pd.qcut(vol.rank(method='first'), q=[0.0, q1, q2, q3, 1.0], labels=False, duplicates='drop')
        targets['volatility_state_q'] = vol_state.astype(float).shift(self.config.feature_lag).fillna(0)

        # 3) Volatility of volatility (regression)
        vol_of_vol = vol.rolling(self.config.volatility_window).std().fillna(0)
        targets['vol_of_vol'] = vol_of_vol.shift(self.config.feature_lag).fillna(0)

        # 4) Breakout propensity (classification) using Bollinger Bands
        ma = close.rolling(self.config.breakout_lookback).mean()
        sd = close.rolling(self.config.breakout_lookback).std()
        upper = ma + 2 * sd
        lower = ma - 2 * sd
        future_max = close.shift(-1).rolling(self.config.breakout_horizon).max()
        future_min = close.shift(-1).rolling(self.config.breakout_horizon).min()
        near_upper = (abs(close - upper) / close < 0.01)
        near_lower = (abs(close - lower) / close < 0.01)
        upper_break = (future_max > upper)
        lower_break = (future_min < lower)
        breakout = ((near_upper & upper_break) | (near_lower & lower_break)).astype(int)
        targets['breakout_within_h'] = breakout.shift(self.config.feature_lag).fillna(0)

        # 5) Reversal violence (regression): magnitude × speed of reversion after extremes
        ma20 = close.rolling(20).mean()
        deviation = (close - ma20) / ma20
        future_close_h = close.shift(-self.config.reversal_horizon)
        reversion_amount = (close - future_close_h) / close
        violence = (deviation.abs() * reversion_amount.abs()).fillna(0)
        targets['reversal_violence'] = violence.shift(self.config.feature_lag).fillna(0)

        # 6) Tail event indicator (classification) using rolling VaR
        var = returns.rolling(self.config.tail_lookback).quantile(self.config.tail_alpha)
        tail_event = (returns <= var).astype(int)
        targets['tail_event'] = tail_event.shift(self.config.feature_lag).fillna(0)

        # 7) Liquidity/microstructure stress (regression): spread proxy, intensity
        if (high is not None) and (low is not None):
            spread_proxy = ((high - low) / close).replace([np.inf, -np.inf], np.nan).fillna(0)
            targets['liquidity_stress_proxy'] = spread_proxy.shift(self.config.feature_lag).fillna(0)
        if (volume is not None) and (high is not None) and (low is not None):
            trade_intensity_proxy = (volume / (high - low).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
            targets['trade_intensity_proxy'] = trade_intensity_proxy.shift(self.config.feature_lag).fillna(0)

        # 8) Efficiency regime (classification) via efficiency ratio
        total_movement = close.rolling(self.config.efficiency_window).apply(lambda x: np.sum(np.abs(np.diff(x))) if len(x) > 1 else 0)
        net_change = (close - close.shift(self.config.efficiency_window)).abs()
        efficiency_ratio = (net_change / total_movement.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
        eff_state = pd.qcut(efficiency_ratio.rank(method='first'), q=[0.0, 0.5, 1.0], labels=False, duplicates='drop')
        targets['efficiency_state'] = eff_state.astype(float).shift(self.config.feature_lag).fillna(0)

        # 9) Regime transition propensity proxy (classification): large changes in vol or returns
        vol_change = vol.pct_change().abs()
        ret_abs = returns.abs()
        trigger = ((vol_change > vol_change.rolling(50).quantile(0.9)) | (ret_abs > ret_abs.rolling(50).quantile(0.95))).astype(int)
        # Predict transition in next H
        transition_prob = trigger.shift(-self.config.transition_horizon).fillna(0)
        targets['regime_transition_trigger'] = transition_prob.shift(self.config.feature_lag).fillna(0).astype(int)

        # 10) Correlation/Autocorr regime (regression): rolling autocorr of returns at lag 1
        def _autocorr_lag1(x: pd.Series) -> float:
            return x.autocorr(1) if len(x) > 2 else 0.0
        ac_lag1 = returns.rolling(50).apply(_autocorr_lag1).fillna(0)
        targets['autocorr_lag1_state'] = ac_lag1.shift(self.config.feature_lag).fillna(0)

        # Drop leading NaNs consistently and align indices
        targets = {k: v.replace([np.inf, -np.inf], np.nan).fillna(0) for k, v in targets.items()}

        self.logger.info(f"Built {len(targets)} dynamic targets")
        return targets

