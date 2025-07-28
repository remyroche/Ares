# trading_strategy.py
import pandas as pd
import numpy as np
import datetime

# Import modules
from config import CONFIG
from models import VolatilityModel, OrderBookModel, VolumeModel
from sr_analyzer import SRLevelAnalyzer # Assuming sr_analyzer.py is in the same directory

class TradingStrategy:
    """
    Main trading strategy combining multiple models, S/R levels, and confidence scores.
    """
    def __init__(self, config=CONFIG):
        self.config = config
        self.volatility_model = VolatilityModel(config)
        self.order_book_model = OrderBookModel(config)
        self.volume_model = VolumeModel(config)
        # Pass S/R specific config to SRLevelAnalyzer
        self.sr_analyzer = SRLevelAnalyzer(config["sr_analyzer"])

    def _is_price_near_sr(self, current_price: float, sr_levels: list):
        """
        Checks if the current price is close to any significant S/R level.
        Returns the closest S/R level if found, otherwise None.
        """
        closest_sr = None
        min_distance_pct = float('inf')

        for level in sr_levels:
            if level["current_expectation"] in ["Very Strong", "Strong", "Moderate"]: # Only consider relevant levels
                distance_pct = abs(current_price - level["level_price"]) / current_price
                if distance_pct <= self.config["sr_proximity_pct"] and distance_pct < min_distance_pct:
                    min_distance_pct = distance_pct
                    closest_sr = level
        return closest_sr

    def calculate_dynamic_stop_loss(self, entry_price: float, atr_value: float, is_long: bool):
        """
        Calculates a dynamic stop-loss price based on ATR.
        """
        if pd.isna(atr_value): return None
        stop_loss_distance = atr_value * self.config["atr"]["stop_loss_multiplier"]
        return entry_price - stop_loss_distance if is_long else entry_price + stop_loss_distance

    def calculate_position_size(self, capital: float, current_price: float, atr_value: float, leverage: float):
        """
        Calculates an adjusted position size based on ATR and max risk per trade.
        This aims to keep the absolute dollar risk consistent.
        """
        if pd.isna(atr_value) or atr_value == 0: return 0, "ATR not available or zero, cannot calculate dynamic position size."
        max_risk_per_trade_usd = capital * self.config["atr"]["max_risk_per_trade_pct"]
        stop_loss_distance_usd = atr_value * self.config["atr"]["stop_loss_multiplier"]
        if stop_loss_distance_usd == 0: return 0, "Stop loss distance is zero, cannot calculate position size."

        max_units = max_risk_per_trade_usd / stop_loss_distance_usd
        notional_value_usd = max_units * current_price
        required_margin = notional_value_usd / leverage

        if required_margin > capital:
            # If calculated required margin exceeds available capital, adjust units down
            notional_value_usd = capital * leverage
            max_units = notional_value_usd / current_price
            required_margin = capital
            return max_units, f"Adjusted position size due to capital limits. Notional: ${notional_value_usd:.2f}, Margin: ${required_margin:.2f}"
        return max_units, f"Recommended units: {max_units:.4f}, Notional Value: ${notional_value_usd:.2f}, Required Margin: ${required_margin:.2f}"

    def monitor_funding_rate(self, funding_rate: float):
        """
        Monitors funding rate for extreme sentiment.
        """
        if funding_rate > self.config["funding_rate"]["high_positive_threshold"]:
            return "High positive funding rate: Market potentially overheated, consider caution or contrarian strategy."
        elif funding_rate < -self.config["funding_rate"]["high_positive_threshold"]:
            return "High negative funding rate: Market potentially oversold, consider caution or contrarian strategy."
        return "Funding rate is neutral."

    def calculate_wrong_direction_confidence(self, current_price: float, atr_value: float,
                                             order_book_insights: dict, sr_levels: list,
                                             is_long_signal: bool):
        """
        Calculates confidence that price won't move in the wrong direction by X%.
        Returns a dictionary of confidence scores for each threshold.
        """
        confidences = {f"{int(p*1000)/10}%": 0 for p in self.config["confidence_wrong_direction_thresholds"]}
        if pd.isna(atr_value) or atr_value == 0:
            return confidences # Cannot calculate without ATR

        for threshold_pct in self.config["confidence_wrong_direction_thresholds"]:
            threshold_abs = current_price * threshold_pct
            confidence = 0

            # Factor 1: ATR
            # If threshold is significantly larger than ATR, higher confidence
            # If threshold is smaller than ATR, lower confidence
            if atr_value > 0:
                atr_ratio = threshold_abs / atr_value
                if atr_ratio >= 2.0: confidence += 40 # Threshold is 2x ATR or more
                elif atr_ratio >= 1.0: confidence += 20 # Threshold is 1x ATR or more
                elif atr_ratio < 0.5: confidence -= 20 # Threshold is less than 0.5x ATR

            # Factor 2: Order Book Walls
            if is_long_signal: # Wrong direction is down (for a long trade)
                # Check for a strong buy wall acting as support below the price, within the threshold
                if order_book_insights["has_buy_wall"] and \
                   (current_price - order_book_insights["buy_wall_price"]) > 0 and \
                   (current_price - order_book_insights["buy_wall_price"]) <= threshold_abs:
                    confidence += 30 # Buy wall within threshold adds confidence
                if order_book_insights["potential_spoofing_or_pulling"]:
                    confidence -= 40 # Spoofing reduces confidence significantly
            else: # Wrong direction is up (for a short trade)
                # Check for a strong sell wall acting as resistance above the price, within the threshold
                if order_book_insights["has_sell_wall"] and \
                   (order_book_insights["sell_wall_price"] - current_price) > 0 and \
                   (order_book_insights["sell_wall_price"] - current_price) <= threshold_abs:
                    confidence += 30 # Sell wall within threshold adds confidence
                if order_book_insights["potential_spoofing_or_pulling"]:
                    confidence -= 40

            # Factor 3: S/R Levels
            for sr_level in sr_levels:
                if sr_level["current_expectation"] in ["Very Strong", "Strong"]:
                    sr_distance = abs(current_price - sr_level["level_price"])
                    if is_long_signal and sr_level["type"] == "Support" and \
                       (current_price - sr_level["level_price"]) > 0 and sr_distance <= threshold_abs:
                        confidence += sr_level["strength_score"] * 3 # Strong support below adds confidence
                    elif not is_long_signal and sr_level["type"] == "Resistance" and \
                         (sr_level["level_price"] - current_price) > 0 and sr_distance <= threshold_abs:
                        confidence += sr_level["strength_score"] * 3 # Strong resistance above adds confidence

            # Clamp confidence between 0 and 100
            confidences[f"{int(threshold_pct*1000)/10}%"] = max(0, min(100, confidence))
        return confidences

    def calculate_right_direction_confidence(self, model_insights: dict, sr_levels: list, is_long_signal: bool):
        """
        Calculates confidence that price will move in the right direction.
        Combines model confidences and S/R level influence.
        """
        total_confidence = 0
        num_models = 0

        # Aggregate individual model confidences
        for model_name, insights in model_insights.items():
            if "model_confidence_score" in insights:
                total_confidence += insights["model_confidence_score"]
                num_models += 1

        avg_model_confidence = total_confidence / num_models if num_models > 0 else 0

        # Adjust based on S/R levels
        sr_influence_score = 0
        for sr_level in sr_levels:
            if sr_level["current_expectation"] in ["Very Strong", "Strong"]:
                # If a strong S/R level is in the right direction (e.g., strong support for long)
                # And price is currently above/below it, indicating a bounce or confirmation
                if is_long_signal and sr_level["type"] == "Support" and \
                   (current_price - sr_level["level_price"]) / current_price < self.config["sr_proximity_pct"] and \
                   current_price > sr_level["level_price"]: # Price is above support, potentially bouncing off
                    sr_influence_score += sr_level["strength_score"] * 5 # Boost for strong support below

                elif not is_long_signal and sr_level["type"] == "Resistance" and \
                     (sr_level["level_price"] - current_price) / current_price < self.config["sr_proximity_pct"] and \
                     current_price < sr_level["level_price"]: # Price is below resistance, potentially bouncing off
                    sr_influence_score += sr_level["strength_score"] * 5 # Boost for strong resistance above

                # If a strong S/R level is in the wrong direction (e.g., strong resistance for long)
                # And price is approaching it, indicating a potential barrier
                elif is_long_signal and sr_level["type"] == "Resistance" and \
                     (sr_level["level_price"] - current_price) / current_price < self.config["sr_proximity_pct"] and \
                     current_price < sr_level["level_price"]: # Price approaching resistance from below
                    sr_influence_score -= sr_level["strength_score"] * 5 # Penalty for resistance above

                elif not is_long_signal and sr_level["type"] == "Support" and \
                     (current_price - sr_level["level_price"]) / current_price < self.config["sr_proximity_pct"] and \
                     current_price > sr_level["level_price"]: # Price approaching support from above
                    sr_influence_score -= sr_level["strength_score"] * 5 # Penalty for support below

        # Combine average model confidence with S/R influence
        final_confidence = avg_model_confidence + sr_influence_score
        return max(0, min(100, final_confidence)) # Clamp between 0 and 100

    def generate_trading_signal(self, historical_data: pd.DataFrame, order_book_data: dict,
                                current_price: float, current_volume: float,
                                funding_rate: float, capital: float, leverage: float):
        """
        Generates a trading signal by combining insights from all models and S/R levels.
        :param historical_data: DataFrame with 'Close', 'High', 'Low', 'Volume' and a DateTime index.
        :param order_book_data: Dictionary with 'bids' and 'asks' lists.
        :param current_price: Current market price.
        :param current_volume: Current trading volume.
        :param funding_rate: Current funding rate.
        :param capital: Available trading capital.
        :param leverage: Desired leverage.
        :return: A tuple (signal: str, reasons: str, recommended_stop_loss: float,
                          recommended_position_units: float, wrong_direction_confidences: dict,
                          right_direction_confidence: float).
        """
        # Ensure sufficient historical data for indicator calculation
        min_data_points = max(self.config["bollinger_bands"]["window"], self.config["atr"]["window"])
        if historical_data.empty or len(historical_data) < min_data_points:
            return "HOLD", "Insufficient historical data for analysis.", None, 0, {}, 0

        close_prices = historical_data['Close']
        high_prices = historical_data['High']
        low_prices = historical_data['Low']
        volumes = historical_data['Volume']
        prev_price = close_prices.iloc[-2] if len(close_prices) >= 2 else current_price

        # 1. Analyze with individual models
        volatility_insights = self.volatility_model.analyze(current_price, close_prices, high_prices, low_prices)
        order_book_insights = self.order_book_model.analyze(order_book_data, current_price)
        volume_insights = self.volume_model.analyze(current_price, prev_price, current_volume, volumes)

        model_insights = {
            "volatility": volatility_insights,
            "order_book": order_book_insights,
            "volume": volume_insights
        }

        all_reasons = []
        for model_name, insights in model_insights.items():
            all_reasons.extend([f"{model_name.capitalize()}: {r}" for r in insights["reasons"]])

        # 2. Analyze S/R levels
        sr_levels = self.sr_analyzer.analyze(historical_data)
        closest_sr = self._is_price_near_sr(current_price, sr_levels)

        if closest_sr:
            all_reasons.append(f"S/R Analysis: Price is near a {closest_sr['current_expectation']} {closest_sr['type']} level at {closest_sr['level_price']:.2f}.")

        # 3. Determine main signal based on confluence
        signal = "HOLD"
        reason = "No strong directional signal."
        is_long_signal = None # True for BUY, False for SELL/SHORT

        # Confluence logic (examples, heavily customizable based on your strategy)
        # Strong BUY signal
        if (volatility_insights["is_oversold_bb"] or volume_insights["is_weak_downtrend"] or volume_insights["is_bearish_capitulation_signal"]) and \
           order_book_insights["has_buy_wall"] and \
           not order_book_insights["potential_spoofing_or_pulling"] and \
           not volatility_insights["is_overbought_bb"]: # Ensure not overbought simultaneously
            signal = "BUY"
            reason = "Confluence: Price oversold/weak downtrend/capitulation, strong buy wall, no spoofing, not overbought."
            is_long_signal = True
            if closest_sr and closest_sr["type"] == "Support" and current_price >= closest_sr["level_price"] * (1 - self.config["sr_proximity_pct"]):
                reason += f" Bouncing off {closest_sr['current_expectation']} support at {closest_sr['level_price']:.2f}."

        # Strong SELL/SHORT signal
        elif (volatility_insights["is_overbought_bb"] or volume_insights["is_weak_rally"] or volume_insights["is_bullish_exhaustion_signal"]) and \
             order_book_insights["has_sell_wall"] and \
             not order_book_insights["potential_spoofing_or_pulling"] and \
             not volatility_insights["is_oversold_bb"]: # Ensure not oversold simultaneously
            signal = "SELL" # Or SHORT
            reason = "Confluence: Price overbought/weak rally/bullish exhaustion, strong sell wall, no spoofing, not oversold."
            is_long_signal = False
            if closest_sr and closest_sr["type"] == "Resistance" and current_price <= closest_sr["level_price"] * (1 + self.config["sr_proximity_pct"]):
                reason += f" Bouncing off {closest_sr['current_expectation']} resistance at {closest_sr['level_price']:.2f}."

        # Caution due to order book issues takes precedence
        if order_book_insights["is_wide_spread"] or order_book_insights["potential_spoofing_or_pulling"]:
            signal = "CAUTION"
            reason = "Market conditions are unfavorable due to low liquidity or manipulation warnings. Overriding other signals."
            is_long_signal = None # No directional signal if caution is paramount

        # 4. Calculate Confidence Scores
        right_direction_confidence = 0
        wrong_direction_confidences = {f"{int(p*1000)/10}%": 0 for p in self.config["confidence_wrong_direction_thresholds"]}

        if is_long_signal is not None: # Only calculate if a directional signal exists
            right_direction_confidence = self.calculate_right_direction_confidence(
                model_insights, sr_levels, is_long_signal
            )
            wrong_direction_confidences = self.calculate_wrong_direction_confidence(
                current_price, volatility_insights["current_atr"], order_book_insights, sr_levels, is_long_signal
            )

        # 5. Risk Management
        recommended_stop_loss = self.calculate_dynamic_stop_loss(current_price, volatility_insights["current_atr"], is_long_signal) \
                                if is_long_signal is not None else None
        recommended_position_units, pos_reason = self.calculate_position_size(capital, current_price, volatility_insights["current_atr"], leverage) \
                                                if is_long_signal is not None else (0, "N/A")
        all_reasons.append(f"Risk Management: {pos_reason}")

        funding_rate_status = self.monitor_funding_rate(funding_rate)
        all_reasons.append(f"Funding Rate: {funding_rate_status}")

        return signal, "\n".join(all_reasons), recommended_stop_loss, recommended_position_units, wrong_direction_confidences, right_direction_confidence


# --- Example Usage (Simulated Data and Running the Strategy) ---
if __name__ == "__main__":
    # Simulate historical OHLCV data (e.g., 1-minute or 5-minute candles)
    data_points = 200 # More data points for better S/R detection and indicator calculation
    np.random.seed(42) # for reproducibility
    dates = pd.date_range(start='2024-01-01', periods=data_points, freq='H') # Hourly data for example

    # Generate synthetic price data with some trends and S/R levels
    base_price = np.cumsum(np.random.randn(data_points) * 0.5 + 10) + 2000
    # Add some specific patterns for testing rules
    # Simulate a strong uptrend
    base_price[50:60] = base_price[49] + np.cumsum(np.random.randn(10) * 2 + 5)
    # Simulate a sharp drop for capitulation
    base_price[150:155] = base_price[149] - np.cumsum(np.random.randn(5) * 5 + 15)

    close_prices = pd.Series(base_price + np.random.randn(data_points) * 2, index=dates, name='Close')
    high_prices = close_prices + np.random.rand(data_points) * 5
    low_prices = close_prices - np.random.rand(data_points) * 5
    volumes = pd.Series(np.random.randint(100, 10000, data_points), index=dates, name='Volume')

    # Introduce some S/R-like behavior for testing
    close_prices[70:75] = np.maximum(close_prices[70:75], 2080 + np.random.randn(5)*2) # Support touch
    close_prices[120:125] = np.minimum(close_prices[120:125], 2150 + np.random.randn(5)*2) # Resistance touch

    historical_df = pd.DataFrame({
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    })

    # Current market data snapshot
    current_price = close_prices.iloc[-1]
    current_volume = volumes.iloc[-1]

    # Simulate order book data
    simulated_bids = [
        [current_price - 0.1, 5],
        [current_price - 0.2, 10],
        [current_price - 0.5, 20],
        [current_price - 1.0, 100],
        [current_price - 2.0, 50000 / current_price], # Large buy wall (approx $50k)
        [current_price - 2.5, 15],
        [current_price - 3.0, 120000 / current_price] # Even larger buy wall (approx $120k)
    ]
    simulated_asks = [
        [current_price + 0.1, 7],
        [current_price + 0.2, 12],
        [current_price + 0.5, 25],
        [current_price + 1.0, 80],
        [current_price + 2.0, 60000 / current_price], # Large sell wall (approx $60k)
        [current_price + 2.5, 18],
        [current_price + 3.0, 110000 / current_price] # Even larger sell wall (approx $110k)
    ]
    simulated_order_book = {"bids": simulated_bids, "asks": simulated_asks}

    # Simulate funding rate
    simulated_funding_rate = 0.0006 # High positive funding rate

    # Initialize strategy
    strategy = TradingStrategy()

    # Generate trading signal
    signal, reasons, stop_loss, position_units, wrong_direction_confidences, right_direction_confidence = \
        strategy.generate_trading_signal(
            historical_data=historical_df,
            order_book_data=simulated_order_book,
            current_price=current_price,
            current_volume=current_volume,
            funding_rate=simulated_funding_rate,
            capital=10000, # Example capital
            leverage=20   # Example leverage
        )

    print(f"--- ETH/USDT Trading Signal ---")
    print(f"Current Price: {current_price:.2f}")
    print(f"Signal: {signal}")
    print(f"Overall Right Direction Confidence: {right_direction_confidence:.2f}%")
    print(f"Confidence Price Won't Move in Wrong Direction (by X%):")
    for pct, conf in wrong_direction_confidences.items():
        print(f"  - {pct}: {conf:.2f}%")
    print(f"Recommended Stop Loss: {stop_loss:.2f}" if stop_loss is not None else "Recommended Stop Loss: N/A")
    print(f"Recommended Position Units (ETH): {position_units:.4f}" if position_units > 0 else "Recommended Position Units: N/A")
    print(f"Reason(s):\n{reasons}")

    print("\n--- Individual Model Insights (for debugging/detailed view) ---")
    print(f"\nVolatility Model Insights:")
    print(strategy.volatility_model.analyze(current_price, close_prices, high_prices, low_prices))
    print(f"\nOrder Book Model Insights:")
    print(strategy.order_book_model.analyze(simulated_order_book, current_price))
    print(f"\nVolume Model Insights:")
    print(strategy.volume_model.analyze(current_price, close_prices.iloc[-2], current_volume, volumes))

    print("\n--- Identified Support and Resistance Levels ---")
    sr_levels_output = strategy.sr_analyzer.analyze(historical_df)
    if sr_levels_output:
        for level in sr_levels_output:
            print(f"Level: {level['level_price']:.2f} | Type: {level['type']} | "
                  f"Touches: {level['num_touches']} | Last Tested: {level['last_tested_timestamp'].strftime('%Y-%m-%d')} | "
                  f"Strength Score: {level['strength_score']:.2f} | Expectation: {level['current_expectation']}")
    else:
        print("No significant S/R levels identified with current configuration.")
