# src/analyst/market_health_analyzer.py
import pandas as pd
import numpy as np
import pandas_ta as ta

class GeneralMarketAnalystModule:
    """
    Provides a continuous "Market Health" score to the entire system
    based on volatility, trend, volume, and momentum.
    """
    def __init__(self, config):
        self.config = config.get("analyst", {}).get("market_health_analyzer", {})

    def get_market_health_score(self, klines_df: pd.DataFrame):
        """
        Calculates a composite "Market Health" score (0-100).
        Higher score indicates a healthier, more predictable market for trading.
        """
        if klines_df.empty or len(klines_df) < max(self.config.get("ma_periods", [20]), self.config.get("momentum_period", 14)):
            print("Insufficient data for Market Health Score calculation.")
            return 50.0 # Neutral score if not enough data

        # 1. Volatility Component (using ATR and Bollinger Bands)
        atr_period = self.config.get("atr_period", 14) # From main config
        bb_window = self.config.get("bollinger_bands", {}).get("window", 20) # From main config
        bb_std_dev = self.config.get("bollinger_bands", {}).get("num_std_dev", 2) # From main config

        klines_df.ta.atr(length=atr_period, append=True, col_names=('ATR'))
        bb = klines_df.ta.bbands(length=bb_window, std=bb_std_dev, append=False)
        
        current_atr = klines_df['ATR'].iloc[-1] if 'ATR' in klines_df.columns else 0
        current_bb_width = (bb[f'BBU_{bb_window}_{bb_std_dev:.1f}'].iloc[-1] - bb[f'BBL_{bb_window}_{bb_std_dev:.1f}'].iloc[-1])
        current_close = klines_df['close'].iloc[-1]

        # Normalize ATR (e.g., inverse of recent average ATR)
        avg_atr = klines_df['ATR'].mean() if 'ATR' in klines_df.columns else 0
        atr_score = 1 - np.clip(current_atr / (avg_atr * 2), 0, 1) if avg_atr > 0 else 0.5 # Lower ATR is better for health

        # Bollinger Band Squeeze (low width suggests consolidation, potential for future move)
        bb_width_norm = current_bb_width / current_close if current_close > 0 else 0
        bb_score = 1 - np.clip(bb_width_norm / 0.02, 0, 1) # 2% width is considered high, so score goes down

        volatility_component = (atr_score * 0.5 + bb_score * 0.5) # Sub-component weighting

        # 2. Trend Component (using Moving Average Clusters)
        ma_periods = self.config.get("ma_periods", [20, 50, 100])
        ma_df = pd.DataFrame(index=klines_df.index)
        for p in ma_periods:
            ma_df[f'SMA_{p}'] = klines_df['close'].rolling(window=p).mean()
        
        # Check for MA alignment (all MAs pointing same direction and ordered correctly)
        current_mas = [ma_df[f'SMA_{p}'].iloc[-1] for p in ma_periods if f'SMA_{p}' in ma_df.columns]
        if len(current_mas) == len(ma_periods):
            is_uptrend_aligned = all(current_mas[i] > current_mas[i+1] for i in range(len(current_mas) - 1))
            is_downtrend_aligned = all(current_mas[i] < current_mas[i+1] for i in range(len(current_mas) - 1))
            
            if is_uptrend_aligned:
                trend_component = 1.0 # Strong bullish trend
            elif is_downtrend_aligned:
                trend_component = 0.0 # Strong bearish trend (less healthy for general trading)
            else:
                trend_component = 0.5 # Sideways/choppy
        else:
            trend_component = 0.5

        # 3. Volume Component (using On-Balance Volume - OBV)
        obv = klines_df.ta.obv(close=klines_df['close'], volume=klines_df['volume'], append=False)
        obv_sma_period = self.config.get("obv_lookback", 20) # From main config's BEST_PARAMS
        obv_sma = obv.rolling(window=obv_sma_period).mean()

        if len(obv) > obv_sma_period:
            obv_slope = (obv.iloc[-1] - obv_sma.iloc[-1]) / obv_sma.iloc[-1] if obv_sma.iloc[-1] != 0 else 0
            volume_component = np.clip((obv_slope + 0.5) / 1.0, 0, 1) # Scale -0.5 to 0.5 range to 0-1
        else:
            volume_component = 0.5

        # 4. Momentum Component (using RSI - example)
        momentum_period = self.config.get("momentum_period", 14)
        rsi = klines_df.ta.rsi(length=momentum_period, append=False)
        current_rsi = rsi.iloc[-1] if not rsi.empty else 50
        
        # RSI between 40-60 is neutral/healthy, outside is overbought/oversold
        if 40 <= current_rsi <= 60:
            momentum_component = 1.0
        elif 30 <= current_rsi < 40 or 60 < current_rsi <= 70:
            momentum_component = 0.7
        else:
            momentum_component = 0.3

        # Combine all components with weights
        weights = {
            "volatility": self.config.get("atr_weight", 0.3),
            "trend": self.config.get("ma_cluster_weight", 0.3), # Using ma_cluster_weight for trend
            "volume": self.config.get("obv_weight", 0.2),
            "momentum": self.config.get("momentum_weight", 0.2)
        }
        total_weight = sum(weights.values())

        market_health_score_raw = (
            volatility_component * weights["volatility"] +
            trend_component * weights["trend"] +
            volume_component * weights["volume"] +
            momentum_component * weights["momentum"]
        )
        
        market_health_score = (market_health_score_raw / total_weight) * 100 if total_weight > 0 else 50

        print(f"Market Health Score: {market_health_score:.2f}")
        return market_health_score
