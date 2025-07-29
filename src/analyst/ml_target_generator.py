import logging
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

class MLTargetGenerator:
    """
    Generates ML targets with a primary focus on risk management for high-leverage trading.
    """
    def __init__(self, config):
        self.config = config.get("analyst", {}).get("ml_target_generator", {})
        self.logger = logging.getLogger(self.__class__.__name__)
        self.forward_window = self.config.get("forward_window", 60)
        self.atr_period = self.config.get("atr_period", 14)
        self.atr_multiplier_tp = self.config.get("atr_multiplier_tp", 2.0)
        self.required_rr_ratio = self.config.get("required_rr_ratio", 2.0)
        self.max_drawdown_pct_of_risk = self.config.get("max_drawdown_pct_of_risk", 0.75)

    def generate_targets(self, features_df, leverage=50):
        """
        This method produces a DataFrame with separate columns for the categorical
        label, reward potential, and risk potential, ready for use by the new loss function.
        """
        self.logger.info("Generating liquidation-aware ML targets with PnL data...")
        
        # Calculate ATR
        high_low = features_df["high"] - features_df["low"]
        high_close = np.abs(features_df["high"] - features_df["close"].shift())
        low_close = np.abs(features_df["low"] - features_df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features_df["atr"] = tr.rolling(window=self.atr_period).mean()
        data = features_df.dropna().copy()

        # Generate labels and PnL data
        pnl_data = self._get_liquidation_aware_labels(data, leverage)
        
        # Combine with original data
        data[['target', 'reward', 'risk']] = pnl_data
        
        self.logger.info(f"Generated labels: \n{data['target'].value_counts()}")
        return data

    def _get_liquidation_aware_labels(self, df, leverage):
        prices = df[["high", "low", "open", "close", "atr"]].to_numpy()
        n = len(prices)
        # Initialize with [label, reward, risk]
        labels_pnl = np.array([["HOLD", 0.0, 0.0]] * n, dtype=object)

        for i in range(n - self.forward_window):
            entry_price = prices[i, 3]
            atr = prices[i, 4]
            future_highs = prices[i + 1 : i + 1 + self.forward_window, 0]
            future_lows = prices[i + 1 : i + 1 + self.forward_window, 1]

            # --- Evaluate LONG ---
            long_liq_price = entry_price * (1 - 1 / leverage)
            long_tp_price = entry_price + (atr * self.atr_multiplier_tp)
            long_risk = entry_price - long_liq_price
            long_reward = long_tp_price - entry_price

            if long_risk > 0 and (long_reward / long_risk) >= self.required_rr_ratio:
                tp_hit, liq_hit, max_drawdown_ok = False, False, True
                for j in range(self.forward_window):
                    if future_lows[j] <= long_liq_price: liq_hit = True; break
                    if (entry_price - future_lows[j]) >= (long_risk * self.max_drawdown_pct_of_risk): max_drawdown_ok = False; break
                    if future_highs[j] >= long_tp_price: tp_hit = True; break
                
                if tp_hit and not liq_hit and max_drawdown_ok:
                    labels_pnl[i] = ["BUY", long_reward, long_risk]
                    continue

            # --- Evaluate SHORT ---
            short_liq_price = entry_price * (1 + 1 / leverage)
            short_tp_price = entry_price - (atr * self.atr_multiplier_tp)
            short_risk = short_liq_price - entry_price
            short_reward = entry_price - short_tp_price

            if short_risk > 0 and (short_reward / short_risk) >= self.required_rr_ratio:
                tp_hit, liq_hit, max_drawdown_ok = False, False, True
                for j in range(self.forward_window):
                    if future_highs[j] >= short_liq_price: liq_hit = True; break
                    if (future_highs[j] - entry_price) >= (short_risk * self.max_drawdown_pct_of_risk): max_drawdown_ok = False; break
                    if future_lows[j] <= short_tp_price: tp_hit = True; break

                if tp_hit and not liq_hit and max_drawdown_ok:
                    labels_pnl[i] = ["SELL", short_reward, short_risk]
        
        return pd.DataFrame(labels_pnl, index=df.index, columns=['target', 'reward', 'risk'])
