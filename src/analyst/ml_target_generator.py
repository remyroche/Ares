import logging

import numpy as np
import pandas as pd


class MLTargetGenerator:
    """
    Generates ML targets with a primary focus on risk management for high-leverage trading.
    This class defines trade opportunities based on a liquidation-aware risk/reward
    ratio and a maximum drawdown constraint.
    """

    def __init__(self, config):
        self.config = config.get("analyst", {}).get("ml_target_generator", {})
        self.logger = logging.getLogger(__name__)

        # Configuration for the new risk-first approach
        self.forward_window = self.config.get("forward_window", 60)
        self.atr_period = self.config.get("atr_period", 14)
        self.atr_multiplier_tp = self.config.get("atr_multiplier_tp", 2.0)
        self.required_rr_ratio = self.config.get("required_rr_ratio", 2.0)
        self.max_drawdown_pct_of_risk = self.config.get(
            "max_drawdown_pct_of_risk", 0.75
        )

    def generate_targets(self, features_df, leverage=50):
        """
        This method now orchestrates a liquidation-aware labeling process.
        It calculates ATR for dynamic take-profit levels and then calls the
        new core logic to generate survival-focused trade labels.

        Args:
            features_df (pd.DataFrame): DataFrame containing market data and features.
                                        Must include 'high', 'low', 'close', 'open'.
            leverage (int): The leverage to be used for calculating liquidation prices.

        Returns:
            pd.DataFrame: The input DataFrame with an added 'target' column.
        """
        self.logger.info(
            f"Generating ML targets with leverage-aware R:R of {self.required_rr_ratio}:1 "
            f"and max drawdown of {self.max_drawdown_pct_of_risk * 100}% of risk."
        )

        # Calculate ATR for dynamic take-profit levels
        high_low = features_df["high"] - features_df["low"]
        high_close = np.abs(features_df["high"] - features_df["close"].shift())
        low_close = np.abs(features_df["low"] - features_df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features_df["atr"] = tr.rolling(window=self.atr_period).mean()

        # Drop rows with NaN ATR
        data = features_df.dropna().copy()

        # Generate labels using the new risk-first logic
        labels = self._get_liquidation_aware_labels(data, leverage)
        data["target"] = labels

        self.logger.info(f"Generated labels: \n{data['target'].value_counts()}")
        return data

    def _get_liquidation_aware_labels(self, df, leverage):
        """
        This is the heart of the new risk-first approach. It iterates through each
        potential entry point and evaluates trades based on their reward vs. their
        distance to liquidation, subject to a strict drawdown constraint.
        """
        prices = df[["high", "low", "open", "close", "atr"]].to_numpy()
        n = len(prices)
        labels = np.full(n, "HOLD", dtype=object)

        for i in range(n - self.forward_window):
            entry_price = prices[i, 3]  # Current close as entry
            atr = prices[i, 4]
            future_highs = prices[i + 1 : i + 1 + self.forward_window, 0]
            future_lows = prices[i + 1 : i + 1 + self.forward_window, 1]

            # --- Evaluate potential LONG trade ---
            long_liq_price = entry_price * (1 - 1 / leverage)
            long_tp_price = entry_price + (atr * self.atr_multiplier_tp)
            long_risk = entry_price - long_liq_price
            long_reward = long_tp_price - entry_price

            if long_risk > 0 and (long_reward / long_risk) >= self.required_rr_ratio:
                tp_hit = False
                liq_hit = False
                max_drawdown_ok = True

                for j in range(self.forward_window):
                    future_low = future_lows[j]
                    future_high = future_highs[j]

                    # Check for liquidation
                    if future_low <= long_liq_price:
                        liq_hit = True
                        break

                    # Check for drawdown violation
                    drawdown = entry_price - future_low
                    if drawdown >= (long_risk * self.max_drawdown_pct_of_risk):
                        max_drawdown_ok = False
                        break

                    # Check for take-profit
                    if future_high >= long_tp_price:
                        tp_hit = True
                        break

                if tp_hit and not liq_hit and max_drawdown_ok:
                    labels[i] = "BUY"
                    continue  # Move to next entry point

            # --- Evaluate potential SHORT trade ---
            short_liq_price = entry_price * (1 + 1 / leverage)
            short_tp_price = entry_price - (atr * self.atr_multiplier_tp)
            short_risk = short_liq_price - entry_price
            short_reward = entry_price - short_tp_price

            if short_risk > 0 and (short_reward / short_risk) >= self.required_rr_ratio:
                tp_hit = False
                liq_hit = False
                max_drawdown_ok = True

                for j in range(self.forward_window):
                    future_low = future_lows[j]
                    future_high = future_highs[j]

                    # Check for liquidation
                    if future_high >= short_liq_price:
                        liq_hit = True
                        break

                    # Check for drawdown violation
                    drawdown = future_high - entry_price
                    if drawdown >= (short_risk * self.max_drawdown_pct_of_risk):
                        max_drawdown_ok = False
                        break

                    # Check for take-profit
                    if future_low <= short_tp_price:
                        tp_hit = True
                        break

                if tp_hit and not liq_hit and max_drawdown_ok:
                    labels[i] = "SELL"

        return labels
