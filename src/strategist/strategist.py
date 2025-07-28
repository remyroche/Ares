# src/strategist/strategist.py
import pandas as pd
import numpy as np
import pandas_ta as ta
import os
import sys

# Assume these are available in the same package or through sys.path
from config import CONFIG, KLINES_FILENAME
from sr_analyzer import SRLevelAnalyzer # Assuming sr_analyzer.py is at the root or accessible
from analyst.data_utils import load_klines_data, calculate_volume_profile # UPDATED: Import calculate_volume_profile

class Strategist:
    """
    The Strategist module defines the macro "playing field" for the current trading session.
    It operates on longer timeframes to establish overall trading range, leverage cap,
    and positional bias, now incorporating Volume Profile and VWAP.
    """
    def __init__(self, config=CONFIG):
        self.config = config.get("strategist", {})
        self.sr_analyzer = SRLevelAnalyzer(config["sr_analyzer"]) # Reuse S/R Analyzer
        self._historical_klines_htf = None # To store higher timeframe data

    def load_historical_data_htf(self):
        """
        Loads historical k-line data and resamples it to the Strategist's higher timeframe.
        """
        print(f"Strategist: Loading historical k-lines for {self.config['timeframe']} analysis...")
        raw_klines = load_klines_data(KLINES_FILENAME)
        if raw_klines.empty:
            print("Strategist: Failed to load raw k-lines data. Cannot perform macro analysis.")
            return False

        # Resample to higher timeframe
        # Ensure column names are correct for resampling
        raw_klines.columns = [col.lower() for col in raw_klines.columns] # Ensure lowercase for consistency
        
        # Aggregate to the specified higher timeframe
        self._historical_klines_htf = raw_klines.resample(self.config['timeframe']).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Rename columns for SRLevelAnalyzer compatibility and general use
        self._historical_klines_htf.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        }, inplace=True)

        if self._historical_klines_htf.empty:
            print(f"Strategist: No data after resampling to {self.config['timeframe']}.")
            return False
        
        print(f"Strategist: Loaded {len(self._historical_klines_htf)} {self.config['timeframe']} candles.")
        return True

    def _calculate_vwap(self, klines_df: pd.DataFrame) -> float:
        """
        Calculates the Volume-Weighted Average Price (VWAP) for the given DataFrame.
        """
        if klines_df.empty:
            return np.nan
        
        # Calculate typical price (High + Low + Close) / 3
        typical_price = (klines_df['High'] + klines_df['Low'] + klines_df['Close']) / 3
        
        # Calculate cumulative (Typical Price * Volume) and cumulative Volume
        tp_volume = typical_price * klines_df['Volume']
        
        # Ensure that the sum of volume is not zero to avoid division by zero
        if klines_df['Volume'].sum() == 0:
            return np.nan

        vwap = tp_volume.sum() / klines_df['Volume'].sum()
        return vwap

    def _calculate_anchored_vwap(self, klines_df: pd.DataFrame, anchor_index: pd.Timestamp) -> float:
        """
        Calculates the Anchored VWAP from a specific anchor point (timestamp).
        """
        if klines_df.empty or anchor_index not in klines_df.index:
            return np.nan
        
        anchored_df = klines_df.loc[anchor_index:]
        return self._calculate_vwap(anchored_df)

    def _analyze_positional_bias(self, klines_htf: pd.DataFrame, current_price: float):
        """
        Determines the overall positional bias (LONG, SHORT, NEUTRAL) based on HTF MAs, VWAP, and Volume Profile.
        """
        if klines_htf.empty or len(klines_htf) < max(self.config['ma_periods_for_bias']):
            print("Strategist: Insufficient HTF data for positional bias analysis.")
            return "NEUTRAL"

        close_prices = klines_htf['Close']
        ma_periods = self.config['ma_periods_for_bias']
        
        # Calculate MAs
        ma_values = {}
        for period in ma_periods:
            ma_values[period] = close_prices.rolling(window=period).mean().iloc[-1]
        
        # Simple bias logic: if shorter MA > longer MA, bullish; else bearish.
        # For multiple MAs, check alignment.
        ma_bias = "NEUTRAL"
        if len(ma_periods) >= 2:
            is_bullish_alignment = all(ma_values[ma_periods[i]] > ma_values[ma_periods[i+1]] for i in range(len(ma_periods) - 1))
            is_bearish_alignment = all(ma_values[ma_periods[i]] < ma_values[ma_periods[i+1]] for i in range(len(ma_periods) - 1))

            if is_bullish_alignment and close_prices.iloc[-1] > ma_values[min(ma_periods)]:
                ma_bias = "LONG"
            elif is_bearish_alignment and close_prices.iloc[-1] < ma_values[min(ma_periods)]:
                ma_bias = "SHORT"
        
        print(f"Strategist: MA-based bias: {ma_bias}")

        # --- Integrate VWAP for bias ---
        current_vwap = self._calculate_vwap(klines_htf)
        vwap_bias = "NEUTRAL"
        if not np.isnan(current_vwap):
            if current_price > current_vwap:
                vwap_bias = "LONG" # Price above VWAP suggests bullish bias
            elif current_price < current_vwap:
                vwap_bias = "SHORT" # Price below VWAP suggests bearish bias
        print(f"Strategist: VWAP-based bias: {vwap_bias} (Current VWAP: {current_vwap:.2f})")

        # --- Integrate Anchored VWAP for institutional sentiment ---
        # Find a significant anchor point (e.g., highest high or lowest low in last X periods)
        # For simplicity, let's anchor to the highest high or lowest low in the last 60 periods
        anchor_period = self.config.get("avwap_anchor_period", 60)
        recent_klines = klines_htf.tail(anchor_period)
        
        avwap_bias = "NEUTRAL"
        if not recent_klines.empty:
            high_idx = recent_klines['High'].idxmax()
            low_idx = recent_klines['Low'].idxmin()

            avwap_from_high = self._calculate_anchored_vwap(klines_htf, high_idx)
            avwap_from_low = self._calculate_anchored_vwap(klines_htf, low_idx)

            if not np.isnan(avwap_from_high) and current_price < avwap_from_high:
                # Price below AVWAP from a swing high suggests bearish sentiment
                avwap_bias = "SHORT"
            elif not np.isnan(avwap_from_low) and current_price > avwap_from_low:
                # Price above AVWAP from a swing low suggests bullish sentiment
                avwap_bias = "LONG"
        print(f"Strategist: Anchored VWAP bias: {avwap_bias}")

        # --- Combine biases ---
        # Simple majority vote or hierarchical decision
        biases = [ma_bias, vwap_bias, avwap_bias]
        long_votes = biases.count("LONG")
        short_votes = biases.count("SHORT")

        if long_votes > short_votes:
            final_bias = "LONG"
        elif short_votes > long_votes:
            final_bias = "SHORT"
        else:
            final_bias = "NEUTRAL" # Fallback if no clear majority or ties

        print(f"Strategist: Positional Bias: {final_bias} (Combined from MA, VWAP, AVWAP)")
        return final_bias

    def _determine_trading_range(self, klines_htf: pd.DataFrame, current_price: float):
        """
        Determines the operational trading range based on recent price action,
        significant S/R levels, and Volume Profile levels.
        """
        if klines_htf.empty:
            print("Strategist: Insufficient HTF data for trading range determination.")
            return {"low": 0.0, "high": float('inf')}

        # 1. Use SRLevelAnalyzer to get significant S/R levels
        sr_levels = self.sr_analyzer.analyze(klines_htf)
        relevant_sr_levels = [
            level for level in sr_levels 
            if level['strength_score'] >= self.config.get('sr_relevance_threshold', 5.0)
        ]
        
        # 2. Calculate Volume Profile levels
        volume_profile_data = calculate_volume_profile(klines_htf) # UPDATED: Call from data_utils
        poc = volume_profile_data["poc"]
        hvn_levels = volume_profile_data["hvn_levels"]
        lvn_levels = volume_profile_data["lvn_levels"]

        # Combine all relevant levels (S/R, HVNs, POC)
        all_relevant_levels = []
        for level in relevant_sr_levels:
            all_relevant_levels.append(level['level_price'])
        if not np.isnan(poc):
            all_relevant_levels.append(poc)
        all_relevant_levels.extend(hvn_levels)
        # LVNs can also act as temporary support/resistance, but might be less strong for range definition
        # all_relevant_levels.extend(lvn_levels) # Optional: include LVNs

        all_relevant_levels = sorted(list(set(all_relevant_levels))) # Remove duplicates and sort

        # Find the closest relevant support below and resistance above current price
        support_below = None
        resistance_above = None

        for level_price in all_relevant_levels:
            if level_price < current_price:
                support_below = level_price
            elif level_price > current_price:
                resistance_above = level_price
                break # Found the first resistance above, stop

        # Fallback to ATR-based range if no clear S/R or Volume Profile levels
        if support_below is None or resistance_above is None or support_below >= resistance_above:
            # Calculate ATR on the higher timeframe
            atr_htf = klines_htf.ta.atr(length=self.config.get("atr_period", 14), append=False).iloc[-1] # Use ATR period from main config
            atr_multiplier = self.config.get("trading_range_atr_multiplier", 3.0)
            
            range_low = current_price - (atr_htf * atr_multiplier)
            range_high = current_price + (atr_htf * atr_multiplier)
            print(f"Strategist: Using ATR-based range: [{range_low:.2f}, {range_high:.2f}] (No significant S/R/VP found)")
            return {"low": range_low, "high": range_high}
        
        print(f"Strategist: Trading Range: [{support_below:.2f}, {resistance_above:.2f}] (Based on S/R and Volume Profile levels)")
        return {"low": support_below, "high": resistance_above}

    def _set_max_leverage_cap(self, market_health_score: float):
        """
        Sets the absolute maximum leverage cap based on overall market health.
        """
        default_cap = self.config.get("max_leverage_cap_default", 100)
        
        # Example logic: Lower market health means lower max leverage
        # Scale cap from 25x to default_cap (e.g., 100x) based on health score (0-100)
        min_cap = 25 # Minimum leverage even in poor market health
        
        # Linear scaling: (market_health_score / 100) * (default_cap - min_cap) + min_cap
        scaled_cap = (market_health_score / 100.0) * (default_cap - min_cap) + min_cap
        
        final_cap = min(default_cap, max(min_cap, int(scaled_cap)))
        print(f"Strategist: Max Allowable Leverage Cap: {final_cap}x (Market Health: {market_health_score:.2f})")
        return final_cap

    def get_strategist_parameters(self, analyst_market_health_score: float, current_price: float):
        """
        Generates the macro parameters for the Tactician.
        This method should be called periodically (e.g., daily).
        :param analyst_market_health_score: The overall market health score from the Analyst.
        :param current_price: The current market price.
        :return: A dictionary with 'Trading_Range', 'Max_Allowable_Leverage_Cap', 'Positional_Bias'.
        """
        print("\n--- Strategist: Generating Macro Parameters ---")

        if self._historical_klines_htf is None:
            if not self.load_historical_data_htf():
                print("Strategist: Failed to load HTF data. Returning default parameters.")
                return {
                    "Trading_Range": {"low": 0.0, "high": float('inf')},
                    "Max_Allowable_Leverage_Cap": self.config.get("max_leverage_cap_default", 100),
                    "Positional_Bias": "NEUTRAL"
                }

        # 1. Positional Bias
        positional_bias = self._analyze_positional_bias(self._historical_klines_htf, current_price)

        # 2. Trading Range
        trading_range = self._determine_trading_range(self._historical_klines_htf, current_price)

        # 3. Max Allowable Leverage Cap
        max_leverage_cap = self._set_max_leverage_cap(analyst_market_health_score)

        strategist_output = {
            "Trading_Range": trading_range,
            "Max_Allowable_Leverage_Cap": max_leverage_cap,
            "Positional_Bias": positional_bias
        }
        print("--- Strategist: Macro Parameters Generated ---")
        return strategist_output

# --- Example Usage (Main execution block for demonstration) ---
if __name__ == "__main__":
    print("Running Strategist Module Demonstration...")

    # Ensure dummy klines data exists for loading
    from analyst.data_utils import create_dummy_data
    # Assuming KLINES_FILENAME is defined in config.py
    from config import KLINES_FILENAME
    create_dummy_data(KLINES_FILENAME, 'klines')

    strategist = Strategist()

    # Step 1: Load historical data (resampled to HTF)
    if not strategist.load_historical_data_htf():
        print("Strategist: Initial HTF data load failed. Exiting demo.")
        sys.exit(1)

    # Simulate Analyst's market health score and current price
    simulated_market_health = 75.0 # Good market health
    simulated_current_price = 2050.0

    # Step 2: Get Strategist parameters
    strategist_params = strategist.get_strategist_parameters(
        analyst_market_health_score=simulated_market_health,
        current_price=simulated_current_price
    )

    print("\n--- Final Strategist Parameters ---")
    for key, value in strategist_params.items():
        print(f"{key}: {value}")

    print("\n--- Scenario 2: Lower Market Health ---")
    simulated_market_health_low = 30.0 # Poor market health
    strategist_params_low_health = strategist.get_strategist_parameters(
        analyst_market_health_score=simulated_market_health_low,
        current_price=simulated_current_price
    )
    for key, value in strategist_params_low_health.items():
        print(f"{key}: {value}")

    print("\nStrategist Module Demonstration Complete.")

