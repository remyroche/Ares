import pandas as pd


class SpecializedModels:
    """
    Houses specialized models like S/R interaction and High-Impact Candle detection.
    """

    def __init__(self, config):
        self.config = config.get("analyst", {}).get("high_impact_candle_model", {})
        self.global_config = (
            config  # Store global config to access other sections like sr_proximity_pct
        )

    def get_sr_interaction_signal(
        self, current_price: float, sr_levels: list, current_atr: float
    ):
        """
        Checks if the current price is interacting with any significant S/R level.
        Returns the closest S/R level if found, otherwise None.
        """
        closest_sr = None
        min_distance_pct = float("inf")

        if not sr_levels or current_atr == 0:
            return None

        # Use the proximity multiplier from the main config's analyst section or BEST_PARAMS
        # Prioritize from BEST_PARAMS if available, otherwise from analyst.high_impact_candle_model
        proximity_multiplier = self.global_config["best_params"].get(
            "proximity_multiplier", self.config.get("proximity_multiplier", 0.25)
        )

        for level in sr_levels:
            if level["current_expectation"] in [
                "Very Strong",
                "Strong",
                "Moderate",
            ]:  # Only consider relevant levels
                level_price = level["level_price"]
                # Dynamic tolerance based on ATR
                tolerance_abs = current_atr * proximity_multiplier

                if abs(current_price - level_price) <= tolerance_abs:
                    distance_pct = (
                        abs(current_price - level_price) / current_price
                        if current_price > 0
                        else float("inf")
                    )
                    if distance_pct < min_distance_pct:
                        min_distance_pct = distance_pct
                        closest_sr = level

        if closest_sr:
            print(
                f"S/R Interaction Signal: Price near {closest_sr['type']} at {closest_sr['level_price']:.2f}"
            )
        return closest_sr

    def get_high_impact_candle_signal(self, klines_df: pd.DataFrame):
        """
        Detects if the most recent candle is a "High-Impact Candle"
        (exceptional volume and volatility).
        """
        # Ensure enough data for calculations
        volume_sma_period = self.config.get("volume_sma_period", 20)
        atr_sma_period = self.config.get("atr_sma_period", 20)
        min_data_required = (
            max(volume_sma_period, atr_sma_period) + 1
        )  # Need at least this many candles for SMA + current

        if klines_df.empty or len(klines_df) < min_data_required:
            return {
                "is_high_impact": False,
                "reason": f"Insufficient data ({len(klines_df)} candles). Need at least {min_data_required}.",
            }

        current_candle = klines_df.iloc[-1]
        prev_candles = klines_df.iloc[:-1]  # All candles except the very last one

        current_volume = current_candle["volume"]
        current_high = current_candle["high"]
        current_low = current_candle["low"]
        current_close = current_candle["close"]

        # Ensure prev_close is available if prev_candles is not empty
        prev_close = (
            prev_candles["close"].iloc[-1] if not prev_candles.empty else current_close
        )

        # Calculate average volume
        avg_volume = (
            prev_candles["volume"].rolling(window=volume_sma_period).mean().iloc[-1]
            if len(prev_candles) >= volume_sma_period
            else 0
        )

        # Calculate average ATR
        # Need to calculate ATR for previous candles to get average ATR
        avg_atr = 0
        if len(prev_candles) >= atr_sma_period:
            high_low = prev_candles["high"] - prev_candles["low"]
            high_prev_close = abs(prev_candles["high"] - prev_candles["close"].shift(1))
            low_prev_close = abs(prev_candles["low"] - prev_candles["close"].shift(1))
            true_range = pd.concat(
                [high_low, high_prev_close, low_prev_close], axis=1
            ).max(axis=1)
            atr_series = true_range.ewm(span=atr_sma_period, adjust=False).mean()
            avg_atr = atr_series.iloc[-1] if not atr_series.empty else 0

        # Calculate current candle's ATR (True Range)
        current_tr = max(
            current_high - current_low,
            abs(current_high - prev_close),
            abs(current_low - prev_close),
        )

        volume_multiplier = self.config.get("volume_multiplier", 3.0)
        atr_multiplier = self.config.get("atr_multiplier", 2.0)

        is_huge_volume = (avg_volume > 0) and (
            current_volume > avg_volume * volume_multiplier
        )
        is_huge_volatility = (avg_atr > 0) and (current_tr > avg_atr * atr_multiplier)

        if is_huge_volume and is_huge_volatility:
            reason = (
                f"Exceptional volume ({current_volume:.2f} vs avg {avg_volume:.2f}x{volume_multiplier}) "
                f"and volatility ({current_tr:.2f} vs avg {avg_atr:.2f}x{atr_multiplier})."
            )
            print(f"High-Impact Candle Signal: {reason}")
            return {"is_high_impact": True, "reason": reason}
        else:
            return {
                "is_high_impact": False,
                "reason": "No high-impact candle detected.",
            }
