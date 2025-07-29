# src/analyst/liquidation_risk_model.py
import pandas as pd
import numpy as np
from scipy.stats import norm # For normal distribution CDF

class ProbabilisticLiquidationRiskModel:
    """
    Calculates the Liquidation Safety Score (LSS), estimating the probability of liquidation
    at various price points based on volatility, order book depth, and current position.
    """
    def __init__(self, config):
        self.config = config.get("analyst", {}).get("liquidation_risk_model", {})

    def calculate_lss(self, current_price: float, current_position_notional: float, 
                      current_liquidation_price: float, historical_klines: pd.DataFrame, 
                      order_book_data: dict) -> tuple[float, str]:
        """
        Calculates the Liquidation Safety Score (LSS) (0-100).
        Higher score indicates the safety from liquidation.
        This enhanced version incorporates more probabilistic elements.

        :param current_price: The current market price.
        :param current_position_notional: The current open position's notional value (size * current_price).
        :param current_liquidation_price: The estimated liquidation price for the current position.
        :param historical_klines: DataFrame of recent historical k-lines, used for volatility.
        :param order_book_data: Dictionary of current order book (bids, asks).
        :return: A tuple containing the LSS (float) and a string explaining the reasons.
        """
        if current_position_notional == 0:
            return 100.0, "No open position, LSS is max." # Max safety if no position

        if current_liquidation_price is None or current_liquidation_price == 0:
            return 0.0, "Liquidation price unknown or zero, LSS is min."

        reasons = []

        # Determine if position is LONG or SHORT
        is_long = current_position_notional > 0
        
        # 1. Volatility Factor (Probability of touching liquidation price)
        # Using historical klines to estimate volatility (e.g., standard deviation of returns or ATR)
        vol_config = self.config.get("volatility_impact", 0.4)
        lookback_period_vol = self.config.get("lookback_periods_volatility", 240) # e.g., 240 periods for hourly data (10 days)

        probability_of_touch = 0.5 # Default neutral probability

        if historical_klines.empty or len(historical_klines) < lookback_period_vol:
            reasons.append("Insufficient historical data for volatility assessment.")
        else:
            klines_for_vol = historical_klines.tail(lookback_period_vol)
            
            # Calculate True Range and ATR
            high_prices = klines_for_vol['high']
            low_prices = klines_for_vol['low']
            close_prices = klines_for_vol['close']

            # Ensure there's enough data for ATR calculation (at least 1 period for TR, then atr_period for EMA)
            if len(close_prices) > 1:
                high_low = high_prices - low_prices
                high_prev_close = abs(high_prices - close_prices.shift(1))
                low_prev_close = abs(low_prices - close_prices.shift(1))
                true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
                
                atr_period_cfg = self.config.get("atr_period", 14)
                current_atr_series = true_range.ewm(span=atr_period_cfg, adjust=False).mean()
                current_atr = current_atr_series.iloc[-1] if not current_atr_series.empty else 0
            else:
                current_atr = 0

            if current_atr == 0:
                reasons.append("Current ATR is zero, volatility factor neutral.")
            else:
                # Convert ATR to a proxy for standard deviation of price movement
                # This factor is crucial and should be tuned.
                atr_to_std_factor = self.config.get("atr_to_std_factor", 2.5) 
                # Ensure period_std_dev_proxy is not zero
                period_std_dev_proxy = current_atr / atr_to_std_factor
                
                if period_std_dev_proxy == 0:
                    reasons.append("Effective volatility proxy is zero.")
                else:
                    # Z-score: How many standard deviations away is the liquidation price?
                    # This assumes a normal distribution of price changes.                 
                    # Adjust Z-score based on the direction of the trade
                    # For long, price needs to go down (negative deviation) to hit liq.
                    # For short, price needs to go up (positive deviation) to hit liq.
                    if (is_long and current_liquidation_price < current_price) or \
                       (not is_long and current_liquidation_price > current_price):
                        # This Z-score is for the distance to liquidation.
                        # We want the probability of price moving *towards* liquidation.
                        # If price needs to drop for long, we're interested in P(X <= liq_price)
                        # If price needs to rise for short, we're interested in P(X >= liq_price)
                        
                        # The sign of z_score_liq should reflect direction relative to current price
                        # If liq_price < current_price (long position) -> negative deviation
                        # If liq_price > current_price (short position) -> positive deviation
                        signed_distance_to_liq = current_liquidation_price - current_price
                        z_score_for_norm_cdf = signed_distance_to_liq / period_std_dev_proxy

                        if is_long: # Price needs to go down (negative deviation)
                            probability_of_touch = norm.cdf(z_score_for_norm_cdf) # P(X <= liq_price)
                        else: # Price needs to go up (positive deviation)
                            probability_of_touch = 1 - norm.cdf(z_score_for_norm_cdf) # P(X >= liq_price)
                        
                        # Clip to ensure it's within [0, 1] due to approximations
                        probability_of_touch = np.clip(probability_of_touch, 0.001, 0.999) # Avoid 0 or 1
                        reasons.append(f"Prob. of touching liq. price (volatility): {probability_of_touch:.2%}")
                    else:
                        reasons.append("Liquidation price is in favorable direction, volatility factor neutral.")
                        probability_of_touch = 0.001 # Very low probability if already profitable relative to liq price
            
        # 2. Order Book Depth Factor (Slippage/Absorption probability)
        # How much volume is needed to push price to liquidation, and is it available?
        ob_config = self.config.get("order_book_depth_impact", 0.3)
        
        bids = order_book_data.get('bids', [])
        asks = order_book_data.get('asks', [])

        order_book_resistance_score = 0.5 # Neutral by default (higher is more resistance to price move)

        # Calculate total notional volume within a certain percentage range towards liquidation
        depth_range_pct = self.config.get("ob_depth_range_pct", 0.005) # e.g., 0.5% range
        
        total_defensive_volume = 0
        if is_long: # Long position, looking for support (bids) below current price
            # Consider bids from current_price down to liquidation_price, plus a buffer
            lower_bound_depth = min(current_price * (1 - depth_range_pct), current_liquidation_price * (1 - self.config.get("liq_buffer_zone_pct", 0.001)))
            upper_bound_depth = current_price
            for p, q in bids:
                if lower_bound_depth <= p <= upper_bound_depth:
                    total_defensive_volume += p * q
        else: # Short position, looking for resistance (asks) above current price
            # Consider asks from current_price up to liquidation_price, plus a buffer
            lower_bound_depth = current_price
            upper_bound_depth = max(current_price * (1 + depth_range_pct), current_liquidation_price * (1 + self.config.get("liq_buffer_zone_pct", 0.001)))
            for p, q in asks:
                if lower_bound_depth <= p <= upper_bound_depth:
                    total_defensive_volume += p * q

        # Compare total defensive volume to current position notional
        if current_position_notional > 0:
            depth_ratio = total_defensive_volume / abs(current_position_notional)
            # Map depth_ratio to a score (e.g., 0-1, where higher is more resistance)
            # Use a sigmoid or tanh function to map to a bounded score
            # Example: score = tanh(depth_ratio / scaling_factor)
            order_book_resistance_score = np.tanh(depth_ratio / self.config.get("ob_depth_scaling_factor", 10.0)) # Scale depth_ratio
            reasons.append(f"Order book resistance: {depth_ratio:.2f}x position notional (score: {order_book_resistance_score:.2f}).")
        else:
            order_book_resistance_score = 0.5 # Neutral if no position or zero notional
            reasons.append("Order book depth not applicable or zero position notional.")


        # 3. Position Health Factor (how close is current price to liquidation price)
        # This is a direct measure of proximity, independent of volatility.
        pos_config = self.config.get("position_impact", 0.3)
        
        # Normalized distance from current price to liquidation price (0-1, higher is safer)
        # Max safety if price is far (e.g., 5% away). Min safety if very close.
        if current_price == 0:
            normalized_distance_to_liq = 0.0 # Max risk
        else:
            # Ensure distance is calculated correctly based on direction
            if is_long: # For long, distance is (current_price - liq_price)
                distance_abs_for_health = max(0, current_price - current_liquidation_price)
            else: # For short, distance is (liq_price - current_price)
                distance_abs_for_health = max(0, current_liquidation_price - current_price)

            normalized_distance_to_liq = distance_abs_for_health / current_price
            # Scale this distance. E.g., if 5% away is max safety (score 1.0), 0% away is 0.0.
            # Use a linear clip for simplicity, but could be non-linear.
            normalized_distance_to_liq = np.clip(normalized_distance_to_liq / self.config.get("max_safe_distance_pct", 0.05), 0, 1)
        
        position_health_score = normalized_distance_to_liq
        reasons.append(f"Proximity to liquidation: {normalized_distance_to_liq:.2%}.")


        # Combine factors into LSS (0-100)
        # Note: probability_of_touch is a risk (lower is better), so we use (1 - probability_of_touch)
        # order_book_resistance_score and position_health_score are safety scores (higher is better)
        
        # Convert probability_of_touch to a safety score (1 - prob_of_touch)
        volatility_safety_score = (1 - probability_of_touch)

        lss_raw = (volatility_safety_score * vol_config + 
                   order_book_resistance_score * ob_config + 
                   position_health_score * pos_config)
        
        # Normalize by sum of weights to keep it within 0-1, then scale to 0-100
        total_weight = vol_config + ob_config + pos_config
        lss = (lss_raw / total_weight) * 100 if total_weight > 0 else 0

        # Ensure LSS is within 0-100 bounds
        lss = np.clip(lss, 0, 100)

        return lss, " | ".join(reasons)
