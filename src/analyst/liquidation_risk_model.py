# src/analyst/liquidation_risk_model.py
import pandas as pd
import numpy as np

class ProbabilisticLiquidationRiskModel:
    """
    Calculates the Liquidation Safety Score (LSS), estimating the probability of liquidation
    at various price points based on volatility, order book depth, and current position.
    """
    def __init__(self, config):
        self.config = config.get("analyst", {}).get("liquidation_risk_model", {})

    def calculate_lss(self, current_price: float, current_position_notional: float, 
                      current_liquidation_price: float, historical_klines: pd.DataFrame, 
                      order_book_data: dict):
        """
        Calculates the Liquidation Safety Score (LSS).
        LSS is a score (0-100) indicating the safety from liquidation, higher is safer.
        """
        if current_position_notional == 0:
            return 100.0, "No open position, LSS is max." # Max safety if no position

        if current_liquidation_price is None or current_liquidation_price == 0:
            return 0.0, "Liquidation price unknown, LSS is min."

        # 1. Volatility Factor
        # How far away is the liquidation price in terms of recent volatility (e.g., ATR multiples)?
        vol_config = self.config.get("volatility_impact", 0.4)
        atr_period = self.config.get("lookback_periods_volatility", 240) # Use a longer period for LSS
        
        if len(historical_klines) < atr_period:
            volatility_factor = 0.5 # Neutral if not enough data
            volatility_reason = "Insufficient historical data for volatility assessment."
        else:
            klines_for_atr = historical_klines.tail(atr_period)
            high_prices = klines_for_atr['high']
            low_prices = klines_for_atr['low']
            close_prices = klines_for_atr['close']

            high_low = high_prices - low_prices
            high_prev_close = abs(high_prices - close_prices.shift(1))
            low_prev_close = abs(low_prices - close_prices.shift(1))
            true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
            current_atr = true_range.ewm(span=atr_period, adjust=False).mean().iloc[-1]

            if current_atr == 0:
                volatility_factor = 0.5
                volatility_reason = "Current ATR is zero, volatility factor neutral."
            else:
                distance_to_liq = abs(current_price - current_liquidation_price)
                atr_multiples = distance_to_liq / current_atr
                # Scale atr_multiples to a 0-1 range (e.g., 0-5 ATRs away)
                volatility_factor = np.clip(atr_multiples / 5.0, 0, 1) # 5 ATRs away is considered very safe
                volatility_reason = f"Liquidation price is {atr_multiples:.2f} ATRs away."
        
        # 2. Order Book Depth Factor
        # Are there significant walls protecting the liquidation price?
        ob_config = self.config.get("order_book_depth_impact", 0.3)
        ob_depth_threshold = self.config.get("order_book_depth_threshold", 0.01) # 1% depth

        bids = order_book_data.get('bids', [])
        asks = order_book_data.get('asks', [])

        order_book_depth_factor = 0.5 # Neutral by default
        order_book_reason = "Order book depth not assessed."

        if current_position_notional > 0: # Long position, liquidation below current price
            # Look for strong bids (support) below current price, near liquidation price
            relevant_bids = [b for b in bids if b[0] < current_price and b[0] >= current_liquidation_price]
            if relevant_bids:
                total_depth = sum(b[0] * b[1] for b in relevant_bids) # Notional value of bids
                # Compare total depth to current position notional
                if current_position_notional > 0:
                    depth_ratio = total_depth / current_position_notional
                    order_book_depth_factor = np.clip(depth_ratio / 2.0, 0, 1) # If depth is 2x position, very safe
                    order_book_reason = f"Relevant bid depth is {depth_ratio:.2f}x current position."
        elif current_position_notional < 0: # Short position, liquidation above current price
            # Look for strong asks (resistance) above current price, near liquidation price
            relevant_asks = [a for a in asks if a[0] > current_price and a[0] <= current_liquidation_price]
            if relevant_asks:
                total_depth = sum(a[0] * a[1] for a in relevant_asks) # Notional value of asks
                if current_position_notional < 0:
                    depth_ratio = total_depth / abs(current_position_notional)
                    order_book_depth_factor = np.clip(depth_ratio / 2.0, 0, 1)
                    order_book_reason = f"Relevant ask depth is {depth_ratio:.2f}x current position."

        # 3. Position Health Factor (how close is current price to liquidation price)
        pos_config = self.config.get("position_impact", 0.3)
        price_diff_to_liq = abs(current_price - current_liquidation_price)
        
        # Normalize distance to liquidation price (e.g., as a percentage of current price)
        if current_price == 0:
            position_health_factor = 0.0
            position_reason = "Current price is zero, position health factor is min."
        else:
            normalized_distance = price_diff_to_liq / current_price
            # If normalized_distance is 0.05 (5%), factor is 1. If 0.005 (0.5%), factor is 0.1
            position_health_factor = np.clip(normalized_distance / 0.05, 0, 1) # 5% away is max safety
            position_reason = f"Distance to liquidation price is {normalized_distance*100:.2f}%."

        # Combine factors into LSS (0-100)
        lss_raw = (volatility_factor * vol_config + 
                   order_book_depth_factor * ob_config + 
                   position_health_factor * pos_config)
        
        # Normalize by sum of weights to keep it within 0-1, then scale to 0-100
        total_weight = vol_config + ob_config + pos_config
        lss = (lss_raw / total_weight) * 100 if total_weight > 0 else 0

        reasons = [volatility_reason, order_book_reason, position_reason]
        return lss, " | ".join(reasons)
