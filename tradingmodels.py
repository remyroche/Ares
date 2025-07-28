# tradingmodels.py
import pandas as pd
import numpy as np

class VolatilityModel:
    """
    Analyzes price volatility using Bollinger Bands and Average True Range (ATR).
    Provides insights and a confidence score based on volatility signals.
    """
    def __init__(self, config):
        self.config = config

    def calculate_bollinger_bands(self, prices: pd.Series):
        """Calculates Bollinger Bands (SMA, Upper Band, Lower Band)."""
        window = self.config["bollinger_bands"]["window"]
        num_std_dev = self.config["bollinger_bands"]["num_std_dev"]
        sma = prices.rolling(window=window).mean()
        std_dev = prices.rolling(window=window).std()
        upper_band = sma + (std_dev * num_std_dev)
        lower_band = sma - (std_dev * num_std_dev)
        return sma, upper_band, lower_band

    def calculate_atr(self, high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series):
        """
        Calculates the Average True Range (ATR).
        True Range (TR) = max[(High - Low), abs(High - Prev. Close), abs(Low - Prev. Close)]
        ATR is the exponential moving average of TR.
        """
        window = self.config["atr"]["window"]
        high_low = high_prices - low_prices
        high_prev_close = abs(high_prices - close_prices.shift(1))
        low_prev_close = abs(low_prices - close_prices.shift(1))
        true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        atr = true_range.ewm(span=window, adjust=False).mean()
        return atr

    def analyze(self, current_price: float, close_prices: pd.Series, high_prices: pd.Series, low_prices: pd.Series):
        """
        Analyzes volatility and generates insights with a confidence score.
        """
        insights = {
            "is_overbought_bb": False,
            "is_oversold_bb": False,
            "current_atr": np.nan,
            "model_confidence_score": 0.0, # 0-100, higher is more confident in signal
            "reasons": []
        }

        # Bollinger Bands Analysis
        if len(close_prices) >= self.config["bollinger_bands"]["window"]:
            sma, upper_bb, lower_bb = self.calculate_bollinger_bands(close_prices)
            current_upper_bb = upper_bb.iloc[-1]
            current_lower_bb = lower_bb.iloc[-1]

            if current_price > current_upper_bb:
                insights["is_overbought_bb"] = True
                insights["reasons"].append("Price above upper Bollinger Band (overbought).")
                insights["model_confidence_score"] += 30 # Initial score for an overbought signal
            if current_price < current_lower_bb:
                insights["is_oversold_bb"] = True
                insights["reasons"].append("Price below lower Bollinger Band (oversold).")
                insights["model_confidence_score"] += 30 # Initial score for an oversold signal

            # Adjust confidence based on how far price is outside the bands
            if insights["is_overbought_bb"]:
                diff_pct = (current_price - current_upper_bb) / current_upper_bb
            elif insights["is_oversold_bb"]:
                diff_pct = (current_lower_bb - current_price) / current_lower_bb
            else:
                diff_pct = 0 # Price is within bands

            if diff_pct > 0.01: # More than 1% outside the band
                insights["model_confidence_score"] += 20
            elif diff_pct > 0.005: # More than 0.5% outside
                insights["model_confidence_score"] += 10

        # ATR Analysis
        if len(close_prices) >= self.config["atr"]["window"]:
            atr_series = self.calculate_atr(high_prices, low_prices, close_prices)
            insights["current_atr"] = atr_series.iloc[-1]
            # ATR value itself doesn't directly generate a signal, but indicates market state.
            # Very high ATR might reduce confidence in range-bound strategies, increase for trend.
            # This model's confidence is primarily driven by BB signals.

        insights["model_confidence_score"] = min(100, insights["model_confidence_score"]) # Cap at 100
        return insights


class OrderBookModel:
    """
    Analyzes order book data to identify walls, spread, and potential deceptive activities.
    Provides insights and a confidence score based on order book dynamics.
    """
    def __init__(self, config):
        self.config = config
        # For tracking dynamic order book changes (requires continuous updates in a live system)
        self.previous_order_book_bids = {}
        self.previous_order_book_asks = {}

    def analyze(self, order_book_data: dict, current_price: float):
        """
        Analyzes order book data and generates insights with a confidence score.
        """
        insights = {
            "has_buy_wall": False,
            "buy_wall_price": None,
            "buy_wall_volume": 0,
            "has_sell_wall": False,
            "sell_wall_price": None,
            "sell_wall_volume": 0,
            "bid_ask_spread": 0.0,
            "spread_percentage": 0.0,
            "is_wide_spread": False,
            "is_narrow_spread": False,
            "potential_spoofing_or_pulling": False, # Requires real-time stream for accurate detection
            "model_confidence_score": 0.0, # 0-100, higher is more confident in signal
            "reasons": []
        }

        bids = order_book_data.get('bids', [])
        asks = order_book_data.get('asks', [])

        # Bid-Ask Spread Analysis
        if bids and asks:
            highest_bid = bids[0][0]
            lowest_ask = asks[0][0]
            insights["bid_ask_spread"] = lowest_ask - highest_bid
            insights["spread_percentage"] = (insights["bid_ask_spread"] / current_price) * 100

            if insights["spread_percentage"] > self.config["order_book"]["spread_wide_threshold_pct"]:
                insights["is_wide_spread"] = True
                insights["reasons"].append("Wide bid-ask spread (low liquidity).")
                insights["model_confidence_score"] -= 20 # Negative impact on confidence
            elif insights["spread_percentage"] < self.config["order_book"]["spread_narrow_threshold_pct"]:
                insights["is_narrow_spread"] = True
                insights["reasons"].append("Narrow bid-ask spread (high liquidity).")
                insights["model_confidence_score"] += 10 # Positive impact

        # Identify large buy/sell walls
        for price, quantity in bids:
            order_value_usd = price * quantity
            if order_value_usd >= self.config["order_book"]["large_order_threshold_usd"]:
                insights["has_buy_wall"] = True
                insights["buy_wall_price"] = price
                insights["buy_wall_volume"] = quantity
                insights["reasons"].append(f"Strong buy wall at {price:.2f}.")
                insights["model_confidence_score"] += 25 # Initial score for a wall
                break # Assuming the first large wall found is the most relevant closest one

        for price, quantity in asks:
            order_value_usd = price * quantity
            if order_value_usd >= self.config["order_book"]["large_order_threshold_usd"]:
                insights["has_sell_wall"] = True
                insights["sell_wall_price"] = price
                insights["sell_wall_volume"] = quantity
                insights["reasons"].append(f"Strong sell wall at {price:.2f}.")
                insights["model_confidence_score"] += 25 # Initial score for a wall
                break # Assuming the first large wall found is the most relevant closest one

        # Spoofing/Order Pulling detection (conceptual and simplified)
        # In a real system, this would involve tracking order IDs and their changes over time.
        # This is a basic check for a significant volume change in a previously identified wall.
        if insights["has_buy_wall"] and insights["buy_wall_price"] in self.previous_order_book_bids:
            prev_volume = self.previous_order_book_bids[insights["buy_wall_price"]]
            current_volume = insights["buy_wall_volume"]
            if abs(current_volume - prev_volume) / prev_volume > self.config["order_book"]["spoofing_pulling_volume_change_threshold_pct"]:
                insights["potential_spoofing_or_pulling"] = True
                insights["reasons"].append("Potential spoofing/order pulling detected (buy side).")
                insights["model_confidence_score"] -= 50 # Significant negative impact
        if insights["has_sell_wall"] and insights["sell_wall_price"] in self.previous_order_book_asks:
            prev_volume = self.previous_order_book_asks[insights["sell_wall_price"]]
            current_volume = insights["sell_wall_volume"]
            if abs(current_volume - prev_volume) / prev_volume > self.config["order_book"]["spoofing_pulling_volume_change_threshold_pct"]:
                insights["potential_spoofing_or_pulling"] = True
                insights["reasons"].append("Potential spoofing/order pulling detected (sell side).")
                insights["model_confidence_score"] -= 50 # Significant negative impact


        # Update previous order book for next iteration
        self.previous_order_book_bids = {p: q for p, q in bids}
        self.previous_order_book_asks = {p: q for p, q in asks}

        insights["model_confidence_score"] = max(0, min(100, insights["model_confidence_score"])) # Clamp score
        return insights

class VolumeModel:
    """
    Analyzes trading volume data and its relationship with price action.
    Provides insights and a confidence score based on volume signals.
    """
    def __init__(self, config):
        self.config = config

    def calculate_obv(self, prices: pd.Series, volumes: pd.Series):
        """
        Calculates On-Balance Volume (OBV).
        Adds volume if close is higher than previous, subtracts if lower.
        """
        obv = pd.Series(0.0, index=prices.index)
        for i in range(1, len(prices)):
            if prices.iloc[i] > prices.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volumes.iloc[i]
            elif prices.iloc[i] < prices.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volumes.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        return obv

    def analyze(self, current_price: float, prev_price: float,
                current_volume: float, volumes: pd.Series):
        """
        Analyzes volume and price relationship and generates insights with a confidence score.
        """
        insights = {
            "is_strong_uptrend": False,
            "is_strong_downtrend": False,
            "is_weak_rally": False,
            "is_weak_downtrend": False,
            "is_bearish_capitulation_signal": False,
            "is_bullish_exhaustion_signal": False,
            "model_confidence_score": 0.0, # 0-100, higher is more confident in signal
            "reasons": []
        }

        if len(volumes) < 2: # Need at least 2 data points for volume comparison
            return insights

        prev_volume = volumes.iloc[-2]
        avg_volume_recent = volumes.mean() if len(volumes) > 0 else current_volume

        price_change = current_price - prev_price
        volume_change = current_volume - prev_volume # Change from previous period

        # Basic Price-Volume Relationships
        if price_change > 0 and volume_change > 0:
            insights["is_strong_uptrend"] = True
            insights["reasons"].append("Price rising with increasing volume (strong uptrend).")
            insights["model_confidence_score"] += 30
        elif price_change < 0 and volume_change > 0:
            insights["is_strong_downtrend"] = True
            insights["reasons"].append("Price falling with increasing volume (strong downtrend).")
            insights["model_confidence_score"] += 30
        elif price_change > 0 and volume_change < 0:
            insights["is_weak_rally"] = True
            insights["reasons"].append("Price rising with decreasing volume (weak rally, potential reversal).")
            insights["model_confidence_score"] += 20
        elif price_change < 0 and volume_change < 0:
            insights["is_weak_downtrend"] = True
            insights["reasons"].append("Price falling with decreasing volume (weak downtrend, potential bottom).")
            insights["model_confidence_score"] += 20

        # Capitulation/Exhaustion (requires significant price movement and massive volume spike)
        if current_volume > avg_volume_recent * self.config["volume"]["capitulation_volume_spike_multiplier"]:
            if price_change < 0 and abs(price_change / prev_price) > 0.01: # Example: 1% sharp drop
                insights["is_bearish_capitulation_signal"] = True
                insights["reasons"].append("Sharp price drop with massive volume spike (potential bottom).")
                insights["model_confidence_score"] += 40 # High confidence signal
            elif price_change > 0 and abs(price_change / prev_price) > 0.01: # Example: 1% sharp surge
                insights["is_bullish_exhaustion_signal"] = True
                insights["reasons"].append("Sharp price surge with massive volume spike (potential top).")
                insights["model_confidence_score"] += 40

        # OBV divergence (conceptual - requires tracking OBV trend vs. price trend)
        # This would involve comparing the trend of self.calculate_obv(prices, volumes) with price trend.
        # For simplicity, not fully implemented here but a crucial part of OBV analysis.

        insights["model_confidence_score"] = min(100, insights["model_confidence_score"]) # Clamp score
        return insights
