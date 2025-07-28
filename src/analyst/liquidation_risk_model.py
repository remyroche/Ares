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
            return 0.0, "Liquidation price unknown, LSS is min."

        reasons = []

        # Determine if position is LONG or SHORT
        is_long = current_position_notional > 0
        
        # 1. Volatility Factor (Probability of touching liquidation price)
        # Using historical klines to estimate volatility (e.g., standard deviation of returns or ATR)
        vol_config = self.config.get("volatility_impact", 0.4)
        lookback_period_vol = self.config.get("lookback_periods_volatility", 240) # e.g., 240 periods for hourly data (10 days)

        probability_of_touch = 0.5 # Default neutral probability

        if len(historical_klines) < lookback_period_vol:
            reasons.append("Insufficient historical data for volatility assessment.")
        else:
            klines_for_vol = historical_klines['close'].tail(lookback_period_vol)
            returns = klines_for_vol.pct_change().dropna()
            
            if returns.empty or returns.std() == 0:
                reasons.append("Zero or no returns for volatility calculation.")
            else:
                period_volatility = returns.std()

                # Calculate True Range and ATR
                high_prices = historical_klines['high'].tail(lookback_period_vol)
                low_prices = historical_klines['low'].tail(lookback_period_vol)
                close_prices = historical_klines['close'].tail(lookback_period_vol)

                high_low = high_prices - low_prices
                high_prev_close = abs(high_prices - close_prices.shift(1))
                low_prev_close = abs(low_prices - close_prices.shift(1))
                true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
                current_atr = true_range.ewm(span=self.config.get("atr_period", 14), adjust=False).mean().iloc[-1] if not true_range.empty else 0

                if current_atr == 0:
                    reasons.append("Current ATR is zero, volatility factor neutral.")
                else:
                    # More sophisticated Z-score based on distance to liquidation and estimated volatility
                    # The 'period_std_dev_proxy' is a conversion from ATR to a typical standard deviation
                    period_std_dev_proxy = current_atr / self.config.get("atr_to_std_factor", 2.5) # Configurable factor
                    
                    if period_std_dev_proxy == 0:
                        reasons.append("Effective volatility is zero.")
                    else:
                        # Distance to liquidation price as a percentage of current price
                        distance_pct = (current_liquidation_price - current_price) / current_price
                        
                        # Z-score: How many standard deviations away is the liquidation price?
                        # Assuming a normal distribution of *percentage returns* over a short period.
                        # This is a simplification; actual price movements are not purely normal.
                        z_score_liq = distance_pct / period_volatility # Using period_volatility from returns.std()
                        
                        # Adjust Z-score based on the direction of the trade
                        if is_long: # Price needs to go down (negative deviation)
                            probability_of_touch = norm.cdf(z_score_liq) # P(X <= liq_price)
                        else: # Price needs to go up (positive deviation)
                            probability_of_touch = 1 - norm.cdf(z_score_liq) # P(X >= liq_price)
                        
                        # Clip to ensure it's within [0, 1] due to approximations
                        probability_of_touch = np.clip(probability_of_touch, 0.001, 0.999) # Avoid 0 or 1
                        reasons.append(f"Prob. of touching liq. price (volatility): {probability_of_touch:.2%}")

        # 2. Order Book Depth Factor (Slippage/Absorption probability)
        # How much volume is needed to push price to liquidation, and is it available?
        ob_config = self.config.get("order_book_depth_impact", 0.3)
        
        bids = order_book_data.get('bids', [])
        asks = order_book_data.get('asks', [])

        order_book_resistance_score = 0.5 # Neutral by default (higher is more resistance to price move)

        # Calculate total notional volume within a certain percentage range towards liquidation
        depth_range_pct = self.config.get("ob_depth_range_pct", 0.005) # e.g., 0.5% range
        
        relevant_depth_volume = 0
        if is_long: # Long position, looking for support (bids) below current price
            lower_bound = current_price * (1 - depth_range_pct)
            upper_bound = current_price # Up to current price
            for p, q in bids:
                if lower_bound <= p <= upper_bound:
                    relevant_depth_volume += p * q
        else: # Short position, looking for resistance (asks) above current price
            lower_bound = current_price
            upper_bound = current_price * (1 + depth_range_pct)
            for p, q in asks:
                if lower_bound <= p <= upper_bound:
                    relevant_depth_volume += p * q

        # Also consider volume directly at or very near liquidation price (if available in order book)
        # This is a heuristic for 'liquidation buffer'
        liquidation_buffer_volume = 0
        buffer_zone_pct = self.config.get("liq_buffer_zone_pct", 0.001) # e.g., 0.1% around liq price
        
        if is_long:
            for p, q in bids:
                if (current_liquidation_price * (1 - buffer_zone_pct)) <= p <= (current_liquidation_price * (1 + buffer_zone_pct)):
                    liquidation_buffer_volume += p * q
        else:
            for p, q in asks:
                if (current_liquidation_price * (1 - buffer_zone_pct)) <= p <= (current_liquidation_price * (1 + buffer_zone_pct)):
                    liquidation_buffer_volume += p * q

        # Combine relevant depth with liquidation buffer volume
        total_defensive_volume = relevant_depth_volume + liquidation_buffer_volume * self.config.get("liq_buffer_weight", 2.0) # Buffer volume weighted higher

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
        # Max safety if price is far (e.g., 10% away). Min safety if very close.
        if current_price == 0:
            normalized_distance_to_liq = 0.0 # Max risk
        else:
            normalized_distance_to_liq = abs(current_price - current_liquidation_price) / current_price
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

