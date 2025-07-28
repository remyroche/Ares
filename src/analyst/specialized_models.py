# src/analyst/specialized_models.py
import pandas as pd
import numpy as np

class SpecializedModels:
    """
    Houses specialized models like S/R interaction and High-Impact Candle detection.
    """
    def __init__(self, config):
        self.config = config.get("analyst", {}).get("high_impact_candle_model", {})
        self.sr_proximity_pct = config.get("sr_proximity_pct", 0.005) # From main config

    def get_sr_interaction_signal(self, current_price: float, sr_levels: list, current_atr: float):
        """
        Checks if the current price is interacting with any significant S/R level.
        Returns the closest S/R level if found, otherwise None.
        """
        closest_sr = None
        min_distance_pct = float('inf')

        if not sr_levels or current_atr == 0:
            return None

        # Use the proximity multiplier from the main config's analyst section
        proximity_multiplier = self.config.get("proximity_multiplier", 0.25) # This should come from the main config's BEST_PARAMS or analyst section

        for level in sr_levels:
            if level["current_expectation"] in ["Very Strong", "Strong", "Moderate"]: # Only consider relevant levels
                level_price = level["level_price"]
                # Dynamic tolerance based on ATR
                tolerance_abs = current_atr * proximity_multiplier
                
                if abs(current_price - level_price) <= tolerance_abs:
                    distance_pct = abs(current_price - level_price) / current_price
                    if distance_pct < min_distance_pct:
                        min_distance_pct = distance_pct
                        closest_sr = level
        
        if closest_sr:
            print(f"S/R Interaction Signal: Price near {closest_sr['type']} at {closest_sr['level_price']:.2f}")
        return closest_sr

    def get_high_impact_candle_signal(self, klines_df: pd.DataFrame):
        """
        Detects if the most recent candle is a "High-Impact Candle"
        (exceptional volume and volatility).
        """
        if klines_df.empty or len(klines_df) < max(self.config["volume_sma_period"], self.config["atr_sma_period"]):
            return {"is_high_impact": False, "reason": "Insufficient data."}

        current_candle = klines_df.iloc[-1]
        prev_candles = klines_df.iloc[:-1]

        current_volume = current_candle['volume']
        current_high = current_candle['high']
        current_low = current_candle['low']
        current_close = current_candle['close']
        prev_close = prev_candles['close'].iloc[-1] if len(prev_candles) > 0 else current_close

        # Calculate average volume
        volume_sma_period = self.config.get("volume_sma_period", 20)
        avg_volume = prev_candles['volume'].rolling(window=volume_sma_period).mean().iloc[-1] if len(prev_candles) >= volume_sma_period else 0

        # Calculate average ATR
        atr_sma_period = self.config.get("atr_sma_period", 20)
        # Need to calculate ATR for previous candles to get average ATR
        high_low = prev_candles['high'] - prev_candles['low']
        high_prev_close = abs(prev_candles['high'] - prev_candles['close'].shift(1))
        low_prev_close = abs(prev_candles['low'] - prev_candles['close'].shift(1))
        true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        avg_atr = true_range.ewm(span=atr_sma_period, adjust=False).mean().iloc[-1] if len(true_range) >= atr_sma_period else 0

        # Calculate current candle's ATR (True Range)
        current_tr = max(current_high - current_low, 
                         abs(current_high - prev_close), 
                         abs(current_low - prev_close))

        volume_multiplier = self.config.get("volume_multiplier", 3.0)
        atr_multiplier = self.config.get("atr_multiplier", 2.0)

        is_huge_volume = (avg_volume > 0) and (current_volume > avg_volume * volume_multiplier)
        is_huge_volatility = (avg_atr > 0) and (current_tr > avg_atr * atr_multiplier)

        if is_huge_volume and is_huge_volatility:
            reason = f"Exceptional volume ({current_volume:.2f} vs avg {avg_volume:.2f}x{volume_multiplier}) " \
                     f"and volatility ({current_tr:.2f} vs avg {avg_atr:.2f}x{atr_multiplier})."
            print(f"High-Impact Candle Signal: {reason}")
            return {"is_high_impact": True, "reason": reason}
        else:
            return {"is_high_impact": False, "reason": "No high-impact candle detected."}
