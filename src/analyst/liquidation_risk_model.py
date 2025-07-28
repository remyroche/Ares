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

        if len(historical_klines) < lookback_period_vol:
            # Fallback to a neutral volatility assessment if not enough data
            probability_of_touch = 0.5 # Assume 50% chance if data is insufficient
            reasons.append("Insufficient historical data for volatility assessment.")
        else:
            klines_for_vol = historical_klines['close'].tail(lookback_period_vol)
            # Calculate daily (or per-period) returns for volatility
            returns = klines_for_vol.pct_change().dropna()
            
            if returns.empty or returns.std() == 0:
                probability_of_touch = 0.5
                reasons.append("Zero or no returns for volatility calculation.")
            else:
                # Annualized volatility (assuming 'interval' is known, e.g., 1m, 1h)
                # For simplicity, we'll use per-period std dev as a proxy for 'daily' vol
                # A more precise model would scale to a relevant timeframe (e.g., daily volatility)
                period_volatility = returns.std()

                # Distance to liquidation price in standard deviations
                price_diff = current_liquidation_price - current_price
                
                # For long position, price needs to go down. For short, price needs to go up.
                # We are interested in the probability of reaching the liquidation price.
                # Using a simplified Z-score approach with Normal CDF
                # Z = (target_price - current_price) / (volatility * sqrt(time_horizon))
                # Here, time_horizon is conceptual; we'll use a fixed 'risk horizon' for now.
                
                # A simple heuristic for 'distance in terms of volatility'
                # How many standard deviations away is the liquidation price?
                if period_volatility == 0:
                    z_score = np.inf * np.sign(price_diff) # Avoid division by zero
                else:
                    # Scale price difference by current price to get percentage difference
                    # This makes it more comparable to percentage volatility
                    normalized_price_diff = price_diff / current_price
                    # We assume price follows a random walk with volatility 'period_volatility'
                    # The number of 'steps' to reach liquidation is conceptual.
                    # Let's consider the distance in terms of 'ATR multiples' or 'std dev multiples'
                    
                    # Using ATR as a measure of recent range, and relating it to liquidation distance
                    # ATR calculation from historical_klines
                    high_prices = historical_klines['high'].tail(lookback_period_vol)
                    low_prices = historical_klines['low'].tail(lookback_period_vol)
                    close_prices = historical_klines['close'].tail(lookback_period_vol)

                    high_low = high_prices - low_prices
                    high_prev_close = abs(high_prices - close_prices.shift(1))
                    low_prev_close = abs(low_prices - close_prices.shift(1))
                    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
                    current_atr = true_range.ewm(span=self.config.get("atr_period", 14), adjust=False).mean().iloc[-1] if not true_range.empty else 0

                    if current_atr == 0:
                        probability_of_touch = 0.5 # Neutral
                        reasons.append("Current ATR is zero, volatility factor neutral.")
                    else:
                        distance_to_liq_abs = abs(current_price - current_liquidation_price)
                        atr_multiples = distance_to_liq_abs / current_atr
                        
                        # Map ATR multiples to probability of touch.
                        # Higher ATR multiples mean lower probability of touch.
                        # Using a sigmoid-like function or CDF to map:
                        # e.g., 1 ATR away -> higher prob, 5 ATRs away -> very low prob.
                        # Let's assume 3 ATRs is a significant distance.
                        # Probability of touch = 1 - CDF(distance / ATR_scale)
                        # We want P(price touches liq_price). This is complex.
                        # A simpler approach: P(touch) is inversely related to distance in ATRs.
                        
                        # Inverse relationship: closer = higher prob.
                        # Example: 1 / (1 + exp(k * atr_multiples))
                        # Or, directly use a scaled ATR multiple to influence a 'risk' score.
                        
                        # Let's use a normal CDF for a more probabilistic feel:
                        # Z-score based on distance to liquidation and volatility (ATR as proxy for std dev)
                        # Assume 1-period movement.
                        # Z = (liquidation_price - current_price) / (current_atr * factor)
                        # Factor to convert ATR to a "standard deviation equivalent" for a single step.
                        # A common rule of thumb is that ATR is roughly 2x-3x the standard deviation of returns.
                        # Let's say, 2.5. So, period_std_dev_proxy = current_atr / 2.5
                        
                        period_std_dev_proxy = current_atr / self.config.get("atr_to_std_factor", 2.5)
                        if period_std_dev_proxy == 0:
                            probability_of_touch = 0.5
                            reasons.append("Effective volatility is zero.")
                        else:
                            z_score_liq = (current_liquidation_price - current_price) / period_std_dev_proxy
                            
                            if is_long: # Price needs to go down to liquidation
                                probability_of_touch = norm.cdf(z_score_liq) # P(X <= liq_price)
                            else: # Price needs to go up to liquidation
                                probability_of_touch = 1 - norm.cdf(z_score_liq) # P(X >= liq_price)
                            
                            # Clip to ensure it's within [0, 1] due to approximations
                            probability_of_touch = np.clip(probability_of_touch, 0.01, 0.99)
                            reasons.append(f"Prob. of touching liq. price (volatility): {probability_of_touch:.2%}")

        # 2. Order Book Depth Factor (Slippage/Absorption probability)
        # How much volume is needed to push price to liquidation, and is it available?
        ob_config = self.config.get("order_book_depth_impact", 0.3)
        ob_depth_lookahead_pct = self.config.get("order_book_depth_lookahead_pct", 0.01) # Look 1% away from current price
        
        bids = order_book_data.get('bids', [])
        asks = order_book_data.get('asks', [])

        order_book_resistance_score = 0.5 # Neutral by default (higher is more resistance to price move)

        target_price_for_depth_analysis = current_liquidation_price # Focus on volume near liquidation
        
        if is_long: # Long position, looking for support (bids) below current price towards liquidation
            # Sum notional value of bids between current price and liquidation price
            relevant_depth_volume = sum(q * p for p, q in bids if p < current_price and p >= target_price_for_depth_analysis)
        else: # Short position, looking for resistance (asks) above current price towards liquidation
            # Sum notional value of asks between current price and liquidation price
            relevant_depth_volume = sum(q * p for p, q in asks if p > current_price and p <= target_price_for_depth_analysis)

        # Compare relevant depth volume to current position notional
        # A higher ratio means more resistance from the order book, thus safer.
        if current_position_notional > 0: # Ensure no division by zero
            depth_ratio = relevant_depth_volume / abs(current_position_notional)
            # Map ratio to a score (e.g., 0-1, where higher is better)
            # If depth is 5x position, very strong resistance.
            order_book_resistance_score = np.clip(depth_ratio / self.config.get("ob_depth_safety_multiplier", 5.0), 0, 1)
            reasons.append(f"Order book resistance: {depth_ratio:.2f}x position notional.")
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
