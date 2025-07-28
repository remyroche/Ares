import pandas as pd
import numpy as np
from loguru import logger

class MLTargetGenerator:
    """
    Implements the Triple-Barrier Method for creating sophisticated ML labels.

    This method labels each data point based on which of three barriers is hit first:
    1. Upper Barrier: Profit-taking level.
    2. Lower Barrier: Stop-loss level.
    3. Vertical Barrier: Maximum holding period for the trade.

    The final labels are:
    -  1: Profit-take barrier was hit.
    - -1: Stop-loss barrier was hit.
    -  0: Vertical barrier was hit (trade timed out).
    """
    def __init__(self, config: dict):
        self.config = config.get("ml_targets", {})
        self.logger = logger
        self.logger.info("MLTargetGenerator (Triple-Barrier) initialized.")

    def generate_labels(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates labels for the given price data using the Triple-Barrier Method.

        Args:
            price_data: DataFrame with 'High', 'Low', 'Close', and 'ATR' columns.

        Returns:
            A DataFrame with added 'label' and 'return' columns.
        """
        if price_data.empty:
            self.logger.warning("Input price_data is empty. Cannot generate labels.")
            return price_data

        # --- Get Parameters from Config ---
        pt_multiplier = self.config.get("profit_take_multiplier", 2.0)
        sl_multiplier = self.config.get("stop_loss_multiplier", 1.5)
        max_hold_periods = self.config.get("max_hold_periods", 10) # Vertical barrier

        # Calculate dynamic profit-take and stop-loss levels based on ATR
        # These barriers are set relative to the entry price (Close)
        atr = price_data['ATR']
        profit_take_levels = price_data['Close'] + (atr * pt_multiplier)
        stop_loss_levels = price_data['Close'] - (atr * sl_multiplier)

        labels = pd.Series(np.nan, index=price_data.index)
        outcomes = pd.DataFrame(index=price_data.index, columns=['return', 'hit_time'])

        self.logger.info(f"Generating triple-barrier labels for {len(price_data)} data points...")

        # --- Find Barrier Hit Time ---
        # This is a vectorized approach for efficiency
        for i in range(len(price_data) - max_hold_periods):
            entry_price = price_data['Close'].iloc[i]
            pt = profit_take_levels.iloc[i]
            sl = stop_loss_levels.iloc[i]

            # Look at the price path over the holding period
            path = price_data[['High', 'Low']].iloc[i+1 : i+1+max_hold_periods]

            # Find the first time the price hits the profit-take or stop-loss barriers
            pt_hits = path[path['High'] >= pt]
            sl_hits = path[path['Low'] <= sl]

            pt_hit_time = pt_hits.index.min() if not pt_hits.empty else pd.NaT
            sl_hit_time = sl_hits.index.min() if not sl_hits.empty else pd.NaT
            
            # Determine which barrier was hit first
            if pd.notna(pt_hit_time) and (pd.isna(sl_hit_time) or pt_hit_time <= sl_hit_time):
                labels.iloc[i] = 1  # Profit take
                outcomes['hit_time'].iloc[i] = pt_hit_time
                outcomes['return'].iloc[i] = (pt - entry_price) / entry_price
            elif pd.notna(sl_hit_time):
                labels.iloc[i] = -1 # Stop loss
                outcomes['hit_time'].iloc[i] = sl_hit_time
                outcomes['return'].iloc[i] = (sl - entry_price) / entry_price
            else:
                labels.iloc[i] = 0 # Vertical barrier (timeout)
                last_price = price_data['Close'].iloc[i + max_hold_periods]
                outcomes['hit_time'].iloc[i] = price_data.index[i + max_hold_periods]
                outcomes['return'].iloc[i] = (last_price - entry_price) / entry_price
        
        # Combine the results into the original DataFrame
        result_df = price_data.copy()
        result_df['label'] = labels.fillna(0).astype(int) # Fill any remaining NaNs
        result_df['return'] = outcomes['return']

        self.logger.success("Triple-barrier labels generated successfully.")
        self.logger.info(f"Label distribution:\n{result_df['label'].value_counts(normalize=True)}")
        
        return result_df
