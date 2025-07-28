import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import datetime

class SRLevelAnalyzer:
    """
    Analyzes historical price data to identify and assess Support and Resistance levels.
    """

    def __init__(self, config=None):
        # Default configuration for S/R analysis
        self.config = {
            "peak_prominence": 0.005,  # Minimum prominence for peak detection (as % of price)
            "peak_width": 5,           # Minimum width for peaks (in data points)
            "level_tolerance_pct": 0.002, # Tolerance for grouping nearby levels (0.2% of price)
            "min_touches": 2,          # Minimum number of touches for a level to be considered significant
            "volume_lookback_window": 10, # Number of periods to consider for volume at touch
            "max_age_days": 90         # Maximum age for a level to be considered relevant (in days)
        }
        if config:
            self.config.update(config)

    def _detect_peaks_and_troughs(self, prices: pd.Series):
        """
        Detects local maxima (resistance candidates) and minima (support candidates).
        """
        # Convert prominence to absolute value based on average price
        avg_price = prices.mean()
        prominence_abs = avg_price * self.config["peak_prominence"]

        # Find peaks (local maxima) for resistance
        resistance_indices, _ = find_peaks(prices, prominence=prominence_abs, width=self.config["peak_width"])
        resistance_levels = prices.iloc[resistance_indices].values
        resistance_timestamps = prices.index[resistance_indices].tolist()

        # Find troughs (local minima) for support by inverting the series
        support_indices, _ = find_peaks(-prices, prominence=prominence_abs, width=self.config["peak_width"])
        support_levels = prices.iloc[support_indices].values
        support_timestamps = prices.index[support_indices].tolist()

        return resistance_levels, resistance_timestamps, support_levels, support_timestamps

    def _group_levels(self, levels, timestamps, level_type: str, current_price: float):
        """
        Groups nearby price levels into significant S/R zones.
        """
        if not levels:
            return []

        # Sort levels by price
        sorted_levels_info = sorted(zip(levels, timestamps), key=lambda x: x[0])

        grouped_sr_levels = []
        current_group = []

        for level, ts in sorted_levels_info:
            if not current_group:
                current_group.append((level, ts))
            else:
                # Check if the current level is within tolerance of the last grouped level
                # Tolerance is dynamic based on the price of the last grouped level
                last_grouped_level_price = current_group[-1][0]
                tolerance_abs = last_grouped_level_price * self.config["level_tolerance_pct"]
                if abs(level - last_grouped_level_price) <= tolerance_abs:
                    current_group.append((level, ts))
                else:
                    # New group starts
                    grouped_sr_levels.append(current_group)
                    current_group = [(level, ts)]
        if current_group:
            grouped_sr_levels.append(current_group)

        # Consolidate groups into single S/R levels
        final_sr_levels = []
        for group in grouped_sr_levels:
            prices_in_group = [item[0] for item in group]
            timestamps_in_group = [item[1] for item in group]

            # Use the median price of the group as the S/R level
            level_price = np.median(prices_in_group)
            
            # Count distinct touches (can be refined to count distinct candles touching the zone)
            num_touches = len(group)
            
            # Find the most recent touch
            last_tested = max(timestamps_in_group)

            final_sr_levels.append({
                "level_price": level_price,
                "type": level_type,
                "touches": timestamps_in_group, # Store all touch timestamps
                "num_touches": num_touches,
                "last_tested_timestamp": last_tested,
                "strength_score": 0.0, # Will be calculated later
                "current_expectation": "Unknown" # Will be determined later
            })
        return final_sr_levels

    def _assess_strength(self, sr_levels: list, prices: pd.Series, volumes: pd.Series, current_timestamp):
        """
        Assesses the strength of each S/R level based on number of touches, volume, and recency.
        """
        assessed_levels = []
        for level_info in sr_levels:
            “ level_price = level_info["level_price"]
            “ num_touches = level_info["num_touches"]
            “ last_tested = level_info["last_tested_timestamp"]
            “ level_type = level_info["type"]

            # Calculate age of the level
            age_days = (current_timestamp - last_tested).days if current_timestamp and last_tested else 0

            # Volume at touches: Sum volume around the touch points
            total_volume_at_touches = 0
            for touch_ts in level_info["touches"]:
                # Find the index of the touch_ts in the prices/volumes index
                try:
                    idx = prices.index.get_loc(touch_ts, method='nearest')
                    # Sum volume for a small window around the touch
                    start_idx = max(0, idx - self.config["volume_lookback_window"] // 2)
                    end_idx = min(len(volumes) - 1, idx + self.config["volume_lookback_window"] // 2)
                    total_volume_at_touches += volumes.iloc[start_idx:end_idx+1].sum()
                except KeyError:
                    # Handle cases where timestamp might not exactly match index
                    pass

            # Normalize volume by average volume over the entire period
            avg_daily_volume = volumes.mean()
            volume_factor = (total_volume_at_touches / (num_touches * avg_daily_volume)) if num_touches > 0 and avg_daily_volume > 0 else 0

            # Strength Score Calculation (heuristic)
            # Factors: num_touches, volume_factor, recency (inverse of age)
            recency_factor = max(0, 1 - (age_days / self.config["max_age_days"])) # 1 for recent, 0 for old

            # Combine factors (weights can be adjusted)
            strength_score = (num_touches * 0.4) + (volume_factor * 0.3) + (recency_factor * 0.3)
            strength_score = min(10.0, strength_score) # Cap score for readability, adjust max as needed

            level_info["strength_score"] = strength_score

            # Determine current expectation based on age and strength
            if age_days > self.config["max_age_days"]:
                level_info["current_expectation"] = "Irrelevant (Too Old)"
            elif strength_score >= 7.0:
                level_info["current_expectation"] = "Very Strong"
            elif strength_score >= 5.0:
                level_info["current_expectation"] = "Strong"
            elif strength_score >= 3.0:
                level_info["current_expectation"] = "Moderate"
            else:
                level_info["current_expectation"] = "Weak"

            assessed_levels.append(level_info)
        return assessed_levels

    def analyze(self, historical_data: pd.DataFrame):
        """
        Main function to analyze historical data and identify S/R levels.
        :param historical_data: DataFrame with 'Close', 'High', 'Low', 'Volume' and a DateTime index.
        :return: List of dictionaries, each describing an S/R level.
        """
        if not all(col in historical_data.columns for col in ['Close', 'High', 'Low', 'Volume']):
            raise ValueError("Historical data must contain 'Close', 'High', 'Low', 'Volume' columns.")
        if not isinstance(historical_data.index, pd.DatetimeIndex):
            raise ValueError("Historical data index must be a DatetimeIndex.")

        close_prices = historical_data['Close']
        high_prices = historical_data['High']
        low_prices = historical_data['Low']
        volumes = historical_data['Volume']
        current_timestamp = historical_data.index[-1] # Use the latest timestamp as current

        # 1. Detect peaks and troughs
        resistance_levels, resistance_timestamps, support_levels, support_timestamps = \
            self._detect_peaks_and_troughs(close_prices)

        # 2. Group nearby levels
        grouped_resistances = self._group_levels(resistance_levels, resistance_timestamps, "Resistance", close_prices.iloc[-1])
        grouped_supports = self._group_levels(support_levels, support_timestamps, "Support", close_prices.iloc[-1])

        # 3. Assess strength
        assessed_resistances = self._assess_strength(grouped_resistances, close_prices, volumes, current_timestamp)
        assessed_supports = self._assess_strength(grouped_supports, close_prices, volumes, current_timestamp)

        # Combine and sort by price
        all_sr_levels = sorted(assessed_resistances + assessed_supports, key=lambda x: x["level_price"])

        return all_sr_levels

# --- Example Usage (Simulated Data) ---
if __name__ == "__main__":
    # Simulate historical OHLCV data for demonstration
    data_points = 200 # More data points for better S/R detection
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=data_points, freq='D')

    # Base price with some trend
    base_price = np.cumsum(np.random.randn(data_points) * 0.5 + 10) + 2000

    # Introduce some S/R-like behavior
    # Create a support level around 2050
    base_price[50:60] = np.maximum(base_price[50:60], 2050 + np.random.randn(10)*2)
    base_price[120:130] = np.maximum(base_price[120:130], 2050 + np.random.randn(10)*2)

    # Create a resistance level around 2150
    base_price[80:90] = np.minimum(base_price[80:90], 2150 + np.random.randn(10)*2)
    base_price[150:160] = np.minimum(base_price[150:160], 2150 + np.random.randn(10)*2)

    close_prices = pd.Series(base_price + np.random.randn(data_points) * 2, index=dates, name='Close')
    high_prices = close_prices + np.random.rand(data_points) * 5
    low_prices = close_prices - np.random.rand(data_points) * 5
    volumes = pd.Series(np.random.randint(1000, 50000, data_points), index=dates, name='Volume')

    historical_df = pd.DataFrame({
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    })

    # Initialize and run the S/R analyzer
    analyzer = SRLevelAnalyzer()
    sr_levels = analyzer.analyze(historical_df)

    print("--- Identified Support and Resistance Levels ---")
    if sr_levels:
        for level in sr_levels:
            print(f"Level: {level['level_price']:.2f} | Type: {level['type']} | "
                  f"Touches: {level['num_touches']} | Last Tested: {level['last_tested_timestamp'].strftime('%Y-%m-%d')} | "
                  f"Strength Score: {level['strength_score']:.2f} | Expectation: {level['current_expectation']}")
    else:
        print("No significant S/R levels identified with current configuration.")

    # Example of how to access a specific level's details
    if sr_levels:
        print(f"\nExample: Details of the first identified level:")
        first_level = sr_levels[0]
        print(f"  Price: {first_level['level_price']:.2f}")
        print(f"  Type: {first_level['type']}")
        print(f"  All Touch Timestamps: {[ts.strftime('%Y-%m-%d') for ts in first_level['touches']]}")
