import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from src.utils.logger import system_logger
from src.config import CONFIG


class MLTargetGenerator:
    """
    CHANGE: Implemented a new S/R-based target generation strategy.
    This class generates a dedicated 'target_sr' column with specific labels
    for S/R Fade (Reversal) and Breakout scenarios. This allows the SRZoneActionEnsemble
    to be trained to identify these precise, actionable trading setups.
    ADDED: Barrier caps and a fixed risk-reward ratio for high-leverage trading.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = CONFIG
        self.config = config.get("analyst", {}).get("ml_target_generator", {})
        self.logger = system_logger.getChild(self.__class__.__name__)

        # --- Main Configuration ---
        self.forward_window = self.config.get("forward_window", 60)
        self.atr_period = self.config.get("atr_period", 14)
        self.atr_multiplier_tp = self.config.get("atr_multiplier_tp", 2.0)
        self.max_drawdown_pct_of_risk = self.config.get(
            "max_drawdown_pct_of_risk", 0.75
        )

        # --- New Barrier Constraints ---
        # The upper barrier (take profit) is capped at 1.25% of the price.
        self.max_upper_barrier_pct = self.config.get("max_upper_barrier_pct", 0.0125)
        # The lower barrier (stop loss) is now 50% of the upper barrier distance, enforcing a 2:1 RR.
        self.risk_reward_ratio = self.config.get("risk_reward_ratio", 2.0)

    def generate_targets(
        self, features_df: pd.DataFrame, leverage: int = 50
    ) -> pd.DataFrame:
        """
        Main orchestration method for generating all target types.
        """
        self.logger.info(
            "Generating all ML target types with new 2:1 RR barrier constraints..."
        )

        if "ATR" not in features_df.columns:
            features_df["ATR"] = (
                (features_df["high"] - features_df["low"])
                .rolling(window=self.atr_period)
                .mean()
            )

        data = features_df.dropna(subset=["ATR", "close", "high", "low"]).copy()

        # 1. Generate standard liquidation-aware trend targets
        trend_pnl_data = self._get_liquidation_aware_labels(data, leverage)
        data[["target_trend", "reward", "risk"]] = trend_pnl_data

        # 2. Generate specialized S/R strategy targets
        sr_labels = self._get_sr_strategy_labels(data)
        data["target_sr"] = sr_labels

        data.rename(columns={"target_trend": "target"}, inplace=True)

        self.logger.info(
            f"Generated Trend labels: \n{data['target'].value_counts(dropna=False)}"
        )
        self.logger.info(
            f"Generated S/R labels: \n{data['target_sr'].value_counts(dropna=False)}"
        )
        return data

    def _get_sr_strategy_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Generates labels for S/R Fade and Breakout strategies, applying new constraints.
        """
        labels = pd.Series("SR_HOLD", index=df.index)
        interacting_indices = df[df["Is_SR_Interacting"] == 1].index

        for idx in interacting_indices:
            row = df.loc[idx]
            future_candles = df.loc[idx:].iloc[1 : self.forward_window + 1]
            if len(future_candles) < self.forward_window:
                continue

            atr = row["ATR"]

            # --- FADE (Reversal) Check ---
            if row["Is_SR_Support_Interacting"] == 1:  # Fade Long
                entry_price = row["low"]
                reward_atr = atr * self.atr_multiplier_tp
                reward_cap = entry_price * self.max_upper_barrier_pct
                final_reward = min(reward_atr, reward_cap)
                final_risk = final_reward / self.risk_reward_ratio

                tp_price = entry_price + final_reward
                sl_price = entry_price - final_risk

                tp_hit_time = future_candles[
                    future_candles["high"] >= tp_price
                ].index.min()
                sl_hit_time = future_candles[
                    future_candles["low"] <= sl_price
                ].index.min()

                if pd.notna(tp_hit_time) and (
                    pd.isna(sl_hit_time) or tp_hit_time < sl_hit_time
                ):
                    labels.loc[idx] = "SR_FADE_LONG"
                    continue

            if row["Is_SR_Resistance_Interacting"] == 1:  # Fade Short
                entry_price = row["high"]
                reward_atr = atr * self.atr_multiplier_tp
                reward_cap = entry_price * self.max_upper_barrier_pct
                final_reward = min(reward_atr, reward_cap)
                final_risk = final_reward / self.risk_reward_ratio

                tp_price = entry_price - final_reward
                sl_price = entry_price + final_risk

                tp_hit_time = future_candles[
                    future_candles["low"] <= tp_price
                ].index.min()
                sl_hit_time = future_candles[
                    future_candles["high"] >= sl_price
                ].index.min()

                if pd.notna(tp_hit_time) and (
                    pd.isna(sl_hit_time) or tp_hit_time < sl_hit_time
                ):
                    labels.loc[idx] = "SR_FADE_SHORT"
                    continue

            # --- BREAKOUT Check ---
            breakout_threshold = atr * 0.5
            if row["Is_SR_Resistance_Interacting"] == 1 and row["close"] > (
                row["high"] + breakout_threshold
            ):  # Breakout Long
                entry_price = row["close"]
                reward_atr = atr * self.atr_multiplier_tp
                reward_cap = entry_price * self.max_upper_barrier_pct
                final_reward = min(reward_atr, reward_cap)
                final_risk = final_reward / self.risk_reward_ratio

                tp_price = entry_price + final_reward
                sl_price = entry_price - final_risk

                tp_hit_time = future_candles[
                    future_candles["high"] >= tp_price
                ].index.min()
                sl_hit_time = future_candles[
                    future_candles["low"] <= sl_price
                ].index.min()

                if pd.notna(tp_hit_time) and (
                    pd.isna(sl_hit_time) or tp_hit_time < sl_hit_time
                ):
                    labels.loc[idx] = "SR_BREAKOUT_LONG"
                    continue

            if row["Is_SR_Support_Interacting"] == 1 and row["close"] < (
                row["low"] - breakout_threshold
            ):  # Breakout Short
                entry_price = row["close"]
                reward_atr = atr * self.atr_multiplier_tp
                reward_cap = entry_price * self.max_upper_barrier_pct
                final_reward = min(reward_atr, reward_cap)
                final_risk = final_reward / self.risk_reward_ratio

                tp_price = entry_price - final_reward
                sl_price = entry_price + final_risk

                tp_hit_time = future_candles[
                    future_candles["low"] <= tp_price
                ].index.min()
                sl_hit_time = future_candles[
                    future_candles["high"] >= sl_price
                ].index.min()

                if pd.notna(tp_hit_time) and (
                    pd.isna(sl_hit_time) or tp_hit_time < sl_hit_time
                ):
                    labels.loc[idx] = "SR_BREAKOUT_SHORT"
                    continue
        return labels

    def _get_liquidation_aware_labels(
        self, df: pd.DataFrame, leverage: int
    ) -> pd.DataFrame:
        prices = df[["high", "low", "open", "close", "ATR"]].to_numpy()
        n = len(prices)
        labels_pnl = np.array([["HOLD", 0.0, 0.0]] * n, dtype=object)

        for i in range(n - self.forward_window):
            entry_price, atr = prices[i, 3], prices[i, 4]
            if entry_price == 0:
                continue  # Avoid division by zero

            future_highs = prices[i + 1 : i + 1 + self.forward_window, 0]
            future_lows = prices[i + 1 : i + 1 + self.forward_window, 1]

            # --- Long Position ---
            long_tp_atr = entry_price + (atr * self.atr_multiplier_tp)
            long_tp_capped = entry_price * (1 + self.max_upper_barrier_pct)
            long_tp = min(long_tp_atr, long_tp_capped)

            long_reward = long_tp - entry_price
            long_risk = long_reward / self.risk_reward_ratio
            long_sl = entry_price - long_risk

            # Liquidation Guardrail
            long_liq_price = entry_price * (1 - (1 / leverage) * 0.95)  # 0.95 buffer
            if long_sl <= long_liq_price:
                continue  # Skip if stop-loss is beyond liquidation point

            tp_hit, sl_hit, max_dd_ok = False, False, True
            for j in range(self.forward_window):
                if future_lows[j] <= long_sl:
                    sl_hit = True
                    break
                if (entry_price - future_lows[j]) >= (
                    long_risk * self.max_drawdown_pct_of_risk
                ):
                    max_dd_ok = False
                    break
                if future_highs[j] >= long_tp:
                    tp_hit = True
                    break
            if tp_hit and not sl_hit and max_dd_ok:
                labels_pnl[i] = ["BUY", long_reward, long_risk]
                continue

            # --- Short Position ---
            short_tp_atr = entry_price - (atr * self.atr_multiplier_tp)
            short_tp_capped = entry_price * (1 - self.max_upper_barrier_pct)
            short_tp = max(short_tp_atr, short_tp_capped)

            short_reward = entry_price - short_tp
            short_risk = short_reward / self.risk_reward_ratio
            short_sl = entry_price + short_risk

            # Liquidation Guardrail
            short_liq_price = entry_price * (1 + (1 / leverage) * 0.95)  # 0.95 buffer
            if short_sl >= short_liq_price:
                continue  # Skip if stop-loss is beyond liquidation point

            tp_hit, sl_hit, max_dd_ok = False, False, True
            for j in range(self.forward_window):
                if future_highs[j] >= short_sl:
                    sl_hit = True
                    break
                if (future_highs[j] - entry_price) >= (
                    short_risk * self.max_drawdown_pct_of_risk
                ):
                    max_dd_ok = False
                    break
                if future_lows[j] <= short_tp:
                    tp_hit = True
                    break
            if tp_hit and not sl_hit and max_dd_ok:
                labels_pnl[i] = ["SELL", short_reward, short_risk]

        return pd.DataFrame(
            labels_pnl, index=df.index, columns=["target", "reward", "risk"]
        )
