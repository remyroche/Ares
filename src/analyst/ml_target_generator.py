import logging
import numpy as np
import pandas as pd

class MLTargetGenerator:
    """
    CHANGE: Implemented a new S/R-based target generation strategy.
    This class generates a dedicated 'target_sr' column with specific labels
    for S/R Fade (Reversal) and Breakout scenarios. This allows the SRZoneActionEnsemble
    to be trained to identify these precise, actionable trading setups.
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
        Main orchestration method for generating all target types.
        """
        self.logger.info("Generating all ML target types...")
        
        # Ensure necessary columns exist
        if 'ATR' not in features_df.columns:
            features_df['ATR'] = (features_df['high'] - features_df['low']).rolling(window=self.atr_period).mean()
        
        data = features_df.dropna(subset=['ATR', 'close', 'high', 'low']).copy()

        # 1. Generate standard liquidation-aware trend targets
        trend_pnl_data = self._get_liquidation_aware_labels(data, leverage)
        data[['target_trend', 'reward', 'risk']] = trend_pnl_data

        # 2. Generate specialized S/R strategy targets
        sr_labels = self._get_sr_strategy_labels(data)
        data['target_sr'] = sr_labels
        
        # For simplicity in the main training loop, we can create a final 'target' column.
        # The SRZoneActionEnsemble will be trained on 'target_sr', others on 'target_trend'.
        # This logic can be handled by the EnsembleOrchestrator.
        data.rename(columns={'target_trend': 'target'}, inplace=True)

        self.logger.info(f"Generated Trend labels: \n{data['target'].value_counts()}")
        self.logger.info(f"Generated S/R labels: \n{data['target_sr'].value_counts()}")
        return data

    def _get_sr_strategy_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Generates labels for S/R Fade and Breakout strategies.
        """
        labels = pd.Series("SR_HOLD", index=df.index)
        
        # Find all interacting candles to reduce search space
        interacting_indices = df[df['Is_SR_Interacting'] == 1].index
        
        for idx in interacting_indices:
            row = df.loc[idx]
            future_candles = df.loc[idx:].iloc[1:self.forward_window+1]
            if len(future_candles) < self.forward_window: continue

            atr = row['ATR']
            
            # --- Check for FADE (Reversal) ---
            if row['Is_SR_Support_Interacting'] == 1:
                # Potential Fade Long (Buy at support)
                entry_price = row['low']
                tp_price = entry_price + (atr * self.atr_multiplier_tp)
                sl_price = entry_price - atr # Tight stop below the support wick
                
                # Check if TP is hit before SL in the future
                tp_hit_time = future_candles[future_candles['high'] >= tp_price].index.min()
                sl_hit_time = future_candles[future_candles['low'] <= sl_price].index.min()

                if pd.notna(tp_hit_time) and (pd.isna(sl_hit_time) or tp_hit_time < sl_hit_time):
                    labels.loc[idx] = "SR_FADE_LONG"
                    continue # Move to next interaction

            if row['Is_SR_Resistance_Interacting'] == 1:
                # Potential Fade Short (Sell at resistance)
                entry_price = row['high']
                tp_price = entry_price - (atr * self.atr_multiplier_tp)
                sl_price = entry_price + atr
                
                tp_hit_time = future_candles[future_candles['low'] <= tp_price].index.min()
                sl_hit_time = future_candles[future_candles['high'] >= sl_price].index.min()

                if pd.notna(tp_hit_time) and (pd.isna(sl_hit_time) or tp_hit_time < sl_hit_time):
                    labels.loc[idx] = "SR_FADE_SHORT"
                    continue

            # --- Check for BREAKOUT ---
            breakout_threshold = atr * 0.5 # Price must close beyond level by this amount
            if row['Is_SR_Resistance_Interacting'] == 1:
                # Potential Breakout Long
                if row['close'] > (row['high'] + breakout_threshold): # High of interaction candle is proxy for level
                    entry_price = row['close']
                    tp_price = entry_price + (atr * self.atr_multiplier_tp)
                    sl_price = row['high'] # Stop loss is the level that was broken
                    
                    tp_hit_time = future_candles[future_candles['high'] >= tp_price].index.min()
                    sl_hit_time = future_candles[future_candles['low'] <= sl_price].index.min()

                    if pd.notna(tp_hit_time) and (pd.isna(sl_hit_time) or tp_hit_time < sl_hit_time):
                        labels.loc[idx] = "SR_BREAKOUT_LONG"
                        continue

            if row['Is_SR_Support_Interacting'] == 1:
                # Potential Breakout Short
                if row['close'] < (row['low'] - breakout_threshold):
                    entry_price = row['close']
                    tp_price = entry_price - (atr * self.atr_multiplier_tp)
                    sl_price = row['low']
                    
                    tp_hit_time = future_candles[future_candles['low'] <= tp_price].index.min()
                    sl_hit_time = future_candles[future_candles['high'] >= sl_price].index.min()

                    if pd.notna(tp_hit_time) and (pd.isna(sl_hit_time) or tp_hit_time < sl_hit_time):
                        labels.loc[idx] = "SR_BREAKOUT_SHORT"
                        continue
        return labels

    def _get_liquidation_aware_labels(self, df, leverage):
        prices = df[["high", "low", "open", "close", "ATR"]].to_numpy()
        n = len(prices)
        labels_pnl = np.array([["HOLD", 0.0, 0.0]] * n, dtype=object)
        for i in range(n - self.forward_window):
            entry_price, atr = prices[i, 3], prices[i, 4]
            future_highs = prices[i + 1 : i + 1 + self.forward_window, 0]
            future_lows = prices[i + 1 : i + 1 + self.forward_window, 1]
            # Long
            long_liq = entry_price * (1 - 1 / leverage); long_tp = entry_price + (atr * self.atr_multiplier_tp)
            long_risk = entry_price - long_liq; long_reward = long_tp - entry_price
            if long_risk > 0 and (long_reward / long_risk) >= self.required_rr_ratio:
                tp_hit, liq_hit, max_dd_ok = False, False, True
                for j in range(self.forward_window):
                    if future_lows[j] <= long_liq: liq_hit = True; break
                    if (entry_price - future_lows[j]) >= (long_risk * self.max_drawdown_pct_of_risk): max_dd_ok = False; break
                    if future_highs[j] >= long_tp: tp_hit = True; break
                if tp_hit and not liq_hit and max_dd_ok:
                    labels_pnl[i] = ["BUY", long_reward, long_risk]; continue
            # Short
            short_liq = entry_price * (1 + 1 / leverage); short_tp = entry_price - (atr * self.atr_multiplier_tp)
            short_risk = short_liq - entry_price; short_reward = entry_price - short_tp
            if short_risk > 0 and (short_reward / short_risk) >= self.required_rr_ratio:
                tp_hit, liq_hit, max_dd_ok = False, False, True
                for j in range(self.forward_window):
                    if future_highs[j] >= short_liq: liq_hit = True; break
                    if (future_highs[j] - entry_price) >= (short_risk * self.max_drawdown_pct_of_risk): max_dd_ok = False; break
                    if future_lows[j] <= short_tp: tp_hit = True; break
                if tp_hit and not liq_hit and max_dd_ok:
                    labels_pnl[i] = ["SELL", short_reward, short_risk]
        return pd.DataFrame(labels_pnl, index=df.index, columns=['target', 'reward', 'risk'])
