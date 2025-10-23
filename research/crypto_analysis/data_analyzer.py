#!/usr/bin/env python3
"""
Cryptocurrency Price Movement Analyzer
Analyzes OHLCV data to calculate potential profits from different triple barrier methods
"""

import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import centralized logging with fallback
try:
    from centralized_logging import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

warnings.filterwarnings("ignore")

logger = get_logger(__name__)

class CryptoPriceAnalyzer:
    def __init__(self, data_file):
        """
        Initialize analyzer with data file

        Args:
            data_file (str): Path to Parquet data file
        """
        self.data_file = data_file
        self.df = None
        self.results = {}

    def load_data(self):
        """Load data from Parquet file"""
        try:
            self.df = pd.read_parquet(self.data_file)
            logger.info(f"Loaded {len(self.df):,} records for {self.df['symbol'].nunique()} assets")
            return True
        except Exception as e:
            logger.exception(f"Error loading data: {e}")
            return False

    def calculate_basic_metrics(self, symbol_data):
        """
        Calculate basic price movement metrics for a single asset

        Args:
            symbol_data (pd.DataFrame): Data for a single symbol

        Returns:
            dict: Dictionary of basic metrics
        """
        # Price metrics
        price_change = (symbol_data["close"].iloc[-1] - symbol_data["close"].iloc[0]) / symbol_data["close"].iloc[0]

        # Volatility metrics
        returns = symbol_data["close"].pct_change().dropna()
        volatility = returns.std() * np.sqrt(96)  # Annualized (96 15-min periods per day)

        # Volume metrics
        avg_volume = symbol_data["volume"].mean()
        volume_volatility = symbol_data["volume"].std() / symbol_data["volume"].mean()

        # Price range metrics
        daily_highs = symbol_data.groupby(symbol_data.index.date)["high"].max()
        daily_lows = symbol_data.groupby(symbol_data.index.date)["low"].min()
        avg_daily_range = ((daily_highs - daily_lows) / daily_lows).mean()

        # Intraday movement metrics
        intraday_highs = symbol_data.groupby(symbol_data.index.date)["high"].max()
        intraday_lows = symbol_data.groupby(symbol_data.index.date)["low"].min()
        avg_intraday_movement = ((intraday_highs - intraday_lows) / intraday_lows).mean()

        # Price movement frequency
        price_changes = symbol_data["close"].pct_change().abs()
        avg_price_change = price_changes.mean()
        price_change_std = price_changes.std()

        return {
            "total_return": price_change,
            "volatility": volatility,
            "avg_volume": avg_volume,
            "volume_volatility": volume_volatility,
            "avg_daily_range": avg_daily_range,
            "avg_intraday_movement": avg_intraday_movement,
            "avg_price_change": avg_price_change,
            "price_change_std": price_change_std,
            "total_volume": symbol_data["volume"].sum(),
            "avg_price": symbol_data["close"].mean(),
            "price_range": (symbol_data["high"].max() - symbol_data["low"].min()) / symbol_data["low"].min(),
        }

    def calculate_triple_barrier_profits(self, symbol_data, barrier_levels=None):
        """
        Calculate potential profits from triple barrier methods

        Args:
            symbol_data (pd.DataFrame): Data for a single symbol
            barrier_levels (list): List of barrier percentages to test (default: 0.3% to 1.5% in 0.1% increments)

        Returns:
            dict: Dictionary of triple barrier profit calculations
        """
        if barrier_levels is None:
            # Create barriers from 0.3% to 1.5% in 0.1% increments
            barrier_levels = [round(0.003 + i * 0.001, 4) for i in range(13)]  # 0.3% to 1.5%

        results = {}

        for barrier in barrier_levels:
            # Calculate potential profits for each barrier level
            barrier_results = self._calculate_single_barrier_profits(symbol_data, barrier)
            results[f"barrier_{int(barrier*1000)}bp"] = barrier_results

        return results

    def _calculate_single_barrier_profits(self, symbol_data, barrier_pct):
        """
        Calculate theoretical profits for a single barrier level
        Only captures successful movements at their peak (15-minute periods)

        Args:
            symbol_data (pd.DataFrame): Data for a single symbol
            barrier_pct (float): Barrier percentage (e.g., 0.01 for 1%)

        Returns:
            dict: Profit calculations for this barrier
        """
        # Calculate potential profits for each 15-minute period
        successful_trades = []
        long_trades = 0
        short_trades = 0

        for i in range(len(symbol_data)):
            current_data = symbol_data.iloc[i]
            current_time = symbol_data.index[i]
            open_price = current_data["open"]
            high_price = current_data["high"]
            low_price = current_data["low"]
            current_data["close"]

            # Calculate potential profit if we captured 100% of the movement
            # Long position: buy at open, sell at high
            long_profit = (high_price - open_price) / open_price

            # Short position: sell at open, buy at low
            short_profit = (open_price - low_price) / open_price

            # Only count trades that exceed the barrier
            if long_profit >= barrier_pct:
                successful_trades.append({
                    "time": current_time,
                    "profit": long_profit,
                    "position": "long",
                    "entry_price": open_price,
                    "exit_price": high_price,
                })
                long_trades += 1

            if short_profit >= barrier_pct:
                successful_trades.append({
                    "time": current_time,
                    "profit": short_profit,
                    "position": "short",
                    "entry_price": open_price,
                    "exit_price": low_price,
                })
                short_trades += 1

        if not successful_trades:
            return {
                "total_trades": 0,
                "avg_profit": 0,
                "long_trades": 0,
                "short_trades": 0,
                "max_profit": 0,
                "min_profit": 0,
                "profit_std": 0,
                "total_potential_profit": 0,
                "profit_frequency": 0,
            }

        # Convert to DataFrame for analysis
        trades_df = pd.DataFrame(successful_trades)

        return {
            "total_trades": len(successful_trades),
            "avg_profit": trades_df["profit"].mean(),
            "long_trades": long_trades,
            "short_trades": short_trades,
            "max_profit": trades_df["profit"].max(),
            "min_profit": trades_df["profit"].min(),
            "profit_std": trades_df["profit"].std(),
            "total_potential_profit": trades_df["profit"].sum(),
            "profit_frequency": len(successful_trades) / len(symbol_data),  # Successful trades per 15-min period
        }

    def calculate_intraday_patterns(self, symbol_data):
        """
        Calculate intraday price movement patterns

        Args:
            symbol_data (pd.DataFrame): Data for a single symbol

        Returns:
            dict: Dictionary of intraday patterns
        """
        # Add time components
        symbol_data = symbol_data.copy()
        symbol_data["hour"] = symbol_data.index.hour
        symbol_data["day_of_week"] = symbol_data.index.dayofweek

        # Hourly patterns
        hourly_volume = symbol_data.groupby("hour")["volume"].mean()
        hourly_volatility = symbol_data.groupby("hour")["close"].pct_change().std()
        hourly_price_changes = symbol_data.groupby("hour")["close"].pct_change().abs().mean()
        
        # Remove any NaN values
        hourly_volume = hourly_volume.dropna() if hasattr(hourly_volume, 'dropna') else hourly_volume
        hourly_volatility = hourly_volatility.dropna() if hasattr(hourly_volatility, 'dropna') else hourly_volatility
        hourly_price_changes = hourly_price_changes.dropna() if hasattr(hourly_price_changes, 'dropna') else hourly_price_changes

        # Peak hours (highest volume)
        peak_hours = hourly_volume.nlargest(min(3, len(hourly_volume))).index.tolist() if hasattr(hourly_volume, 'nlargest') and len(hourly_volume) > 0 else []

        # Day of week patterns
        daily_volume = symbol_data.groupby("day_of_week")["volume"].mean()
        daily_volatility = symbol_data.groupby("day_of_week")["close"].pct_change().std()
        daily_price_changes = symbol_data.groupby("day_of_week")["close"].pct_change().abs().mean()
        
        # Remove any NaN values
        daily_volume = daily_volume.dropna() if hasattr(daily_volume, 'dropna') else daily_volume
        daily_volatility = daily_volatility.dropna() if hasattr(daily_volatility, 'dropna') else daily_volatility
        daily_price_changes = daily_price_changes.dropna() if hasattr(daily_price_changes, 'dropna') else daily_price_changes

        # Best trading hours (highest price movements)
        best_hours = hourly_price_changes.nlargest(min(3, len(hourly_price_changes))).index.tolist() if hasattr(hourly_price_changes, 'nlargest') and len(hourly_price_changes) > 0 else []

        return {
            "peak_hours": peak_hours,
            "best_trading_hours": best_hours,
            "hourly_volume": hourly_volume.to_dict() if hasattr(hourly_volume, 'to_dict') else {},
            "hourly_volatility": hourly_volatility.to_dict() if hasattr(hourly_volatility, 'to_dict') else {},
            "hourly_price_changes": hourly_price_changes.to_dict() if hasattr(hourly_price_changes, 'to_dict') else {},
            "daily_volume": daily_volume.to_dict() if hasattr(daily_volume, 'to_dict') else {},
            "daily_volatility": daily_volatility.to_dict() if hasattr(daily_volatility, 'to_dict') else {},
            "daily_price_changes": daily_price_changes.to_dict() if hasattr(daily_price_changes, 'to_dict') else {},
        }

    def calculate_movement_statistics(self, symbol_data):
        """
        Calculate detailed price movement statistics

        Args:
            symbol_data (pd.DataFrame): Data for a single symbol

        Returns:
            dict: Movement statistics
        """
        # Calculate returns
        returns = symbol_data["close"].pct_change().dropna()

        # Movement size distribution
        movement_sizes = returns.abs()

        # Calculate percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        movement_percentiles = {}
        for p in percentiles:
            movement_percentiles[f"p{p}"] = movement_sizes.quantile(p/100)

        # Calculate movement frequency by size
        small_movements = (movement_sizes <= 0.001).sum() / len(movement_sizes)  # <= 0.1%
        medium_movements = ((movement_sizes > 0.001) & (movement_sizes <= 0.01)).sum() / len(movement_sizes)  # 0.1-1%
        large_movements = (movement_sizes > 0.01).sum() / len(movement_sizes)  # > 1%

        # Calculate consecutive movement patterns
        positive_runs = self._calculate_consecutive_runs(returns > 0)
        negative_runs = self._calculate_consecutive_runs(returns < 0)

        return {
            "avg_movement": movement_sizes.mean(),
            "median_movement": movement_sizes.median(),
            "movement_percentiles": movement_percentiles,
            "small_movements_pct": small_movements,
            "medium_movements_pct": medium_movements,
            "large_movements_pct": large_movements,
            "avg_positive_run": positive_runs["avg_length"],
            "avg_negative_run": negative_runs["avg_length"],
            "max_positive_run": positive_runs["max_length"],
            "max_negative_run": negative_runs["max_length"],
        }

    def _calculate_consecutive_runs(self, condition_series):
        """Calculate consecutive runs of True values"""
        runs = []
        current_run = 0

        for value in condition_series:
            if value:
                current_run += 1
            elif current_run > 0:
                runs.append(current_run)
                current_run = 0

        if current_run > 0:
            runs.append(current_run)

        if not runs:
            return {"avg_length": 0, "max_length": 0}

        return {
            "avg_length": np.mean(runs),
            "max_length": np.max(runs),
        }

    def calculate_volume_analysis(self, symbol_data):
        """
        Calculate comprehensive volume analysis for a single asset

        Args:
            symbol_data (pd.DataFrame): Data for a single symbol

        Returns:
            dict: Dictionary of volume analysis metrics
        """
        # Basic volume metrics - now value-weighted
        # Volume in USD = volume * close_price (represents actual trading value)
        volume_usd = symbol_data["volume"] * symbol_data["close"]
        total_volume_usd = volume_usd.sum()
        total_volume = symbol_data["volume"].sum()  # Keep raw volume for comparison
        avg_volume_usd = volume_usd.mean()
        avg_volume = symbol_data["volume"].mean()
        median_volume_usd = volume_usd.median()
        median_volume = symbol_data["volume"].median()
        volume_std_usd = volume_usd.std()
        volume_std = symbol_data["volume"].std()
        volume_cv = volume_std_usd / avg_volume_usd  # Coefficient of variation (value-weighted)
        
        # 30-day average volume (30 days * 96 15-min periods per day = 2880 periods)
        # Use min_periods to handle cases with insufficient data - extrapolate from available data
        min_periods_30d = min(len(symbol_data), 96)  # Use at least 1 day of data, or all available
        volume_30d_avg_usd = volume_usd.rolling(window=2880, min_periods=min_periods_30d).mean()
        avg_volume_30d_usd = volume_30d_avg_usd.mean()
        volume_30d_avg = symbol_data["volume"].rolling(window=2880, min_periods=min_periods_30d).mean()
        avg_volume_30d = volume_30d_avg.mean()
        
        # Current volume / 30-day average volume ratio (value-weighted)
        # Handle division by zero or very small values
        volume_ratio_30d_usd = volume_usd / volume_30d_avg_usd.replace(0, volume_usd.mean())
        avg_volume_ratio_30d_usd = volume_ratio_30d_usd.mean()
        volume_ratio_30d_std_usd = volume_ratio_30d_usd.std()
        volume_ratio_30d_median_usd = volume_ratio_30d_usd.median()
        
        # Keep raw volume ratios for comparison
        volume_ratio_30d = symbol_data["volume"] / volume_30d_avg.replace(0, symbol_data["volume"].mean())
        avg_volume_ratio_30d = volume_ratio_30d.mean()
        volume_ratio_30d_std = volume_ratio_30d.std()
        volume_ratio_30d_median = volume_ratio_30d.median()

        # Volume percentiles (both raw and value-weighted)
        volume_percentiles = {
            "p10": symbol_data["volume"].quantile(0.1),
            "p25": symbol_data["volume"].quantile(0.25),
            "p50": symbol_data["volume"].quantile(0.5),
            "p75": symbol_data["volume"].quantile(0.75),
            "p90": symbol_data["volume"].quantile(0.9),
            "p95": symbol_data["volume"].quantile(0.95),
            "p99": symbol_data["volume"].quantile(0.99),
        }
        
        volume_percentiles_usd = {
            "p10": volume_usd.quantile(0.1),
            "p25": volume_usd.quantile(0.25),
            "p50": volume_usd.quantile(0.5),
            "p75": volume_usd.quantile(0.75),
            "p90": volume_usd.quantile(0.9),
            "p95": volume_usd.quantile(0.95),
            "p99": volume_usd.quantile(0.99),
        }

        # Volume distribution analysis
        volume_bins = pd.cut(symbol_data["volume"], bins=10)
        volume_bins.value_counts().sort_index()

        # High volume periods (top 10% of volume) - using value-weighted volume
        high_volume_threshold_usd = volume_usd.quantile(0.9)
        high_volume_periods_usd = symbol_data[volume_usd >= high_volume_threshold_usd]
        high_volume_frequency_usd = len(high_volume_periods_usd) / len(symbol_data)

        # Low volume periods (bottom 10% of volume) - using value-weighted volume
        low_volume_threshold_usd = volume_usd.quantile(0.1)
        low_volume_periods_usd = symbol_data[volume_usd <= low_volume_threshold_usd]
        low_volume_frequency_usd = len(low_volume_periods_usd) / len(symbol_data)
        
        # Keep raw volume metrics for comparison
        high_volume_threshold = symbol_data["volume"].quantile(0.9)
        high_volume_periods = symbol_data[symbol_data["volume"] >= high_volume_threshold]
        high_volume_frequency = len(high_volume_periods) / len(symbol_data)

        low_volume_threshold = symbol_data["volume"].quantile(0.1)
        low_volume_periods = symbol_data[symbol_data["volume"] <= low_volume_threshold]
        low_volume_frequency = len(low_volume_periods) / len(symbol_data)

        # Volume-price relationship
        volume_price_corr = symbol_data["volume"].corr(symbol_data["close"])
        volume_returns_corr = symbol_data["volume"].corr(symbol_data["close"].pct_change())

        # Volume volatility (value-weighted)
        volume_volatility_usd = volume_usd.pct_change().std()
        volume_volatility = symbol_data["volume"].pct_change().std()

        # Volume trends (using rolling average) - value-weighted
        rolling_volume_usd = volume_usd.rolling(window=96).mean()  # 24 hours (96 15-min periods)
        volume_trend_usd = (rolling_volume_usd.iloc[-1] - rolling_volume_usd.iloc[0]) / rolling_volume_usd.iloc[0] if len(rolling_volume_usd.dropna()) > 0 else 0
        rolling_volume = symbol_data["volume"].rolling(window=96).mean()  # 24 hours (96 15-min periods)
        volume_trend = (rolling_volume.iloc[-1] - rolling_volume.iloc[0]) / rolling_volume.iloc[0] if len(rolling_volume.dropna()) > 0 else 0

        # Volume spikes (periods with volume > 2x average) - value-weighted
        volume_spikes_usd = symbol_data[volume_usd > 2 * avg_volume_usd]
        volume_spike_frequency_usd = len(volume_spikes_usd) / len(symbol_data)
        volume_spikes = symbol_data[symbol_data["volume"] > 2 * avg_volume]
        volume_spike_frequency = len(volume_spikes) / len(symbol_data)

        # Volume consistency (how often volume is within 50% of average) - value-weighted
        volume_consistency_usd = ((volume_usd >= 0.5 * avg_volume_usd) &
                                (volume_usd <= 1.5 * avg_volume_usd)).mean()
        volume_consistency = ((symbol_data["volume"] >= 0.5 * avg_volume) &
                            (symbol_data["volume"] <= 1.5 * avg_volume)).mean()

        return {
            # Value-weighted volume metrics (primary metrics)
            "total_volume_usd": total_volume_usd,
            "avg_volume_usd": avg_volume_usd,
            "median_volume_usd": median_volume_usd,
            "volume_std_usd": volume_std_usd,
            "volume_cv_usd": volume_cv,
            "avg_volume_30d_usd": avg_volume_30d_usd,
            "avg_volume_ratio_30d_usd": avg_volume_ratio_30d_usd,
            "volume_ratio_30d_std_usd": volume_ratio_30d_std_usd,
            "volume_ratio_30d_median_usd": volume_ratio_30d_median_usd,
            "volume_percentiles_usd": volume_percentiles_usd,
            "high_volume_frequency_usd": high_volume_frequency_usd,
            "low_volume_frequency_usd": low_volume_frequency_usd,
            "volume_volatility_usd": volume_volatility_usd,
            "volume_trend_usd": volume_trend_usd,
            "volume_spike_frequency_usd": volume_spike_frequency_usd,
            "volume_consistency_usd": volume_consistency_usd,
            "max_volume_usd": volume_usd.max(),
            "min_volume_usd": volume_usd.min(),
            "volume_range_usd": volume_usd.max() - volume_usd.min(),
            
            # Raw volume metrics (for comparison)
            "total_volume": total_volume,
            "avg_volume": avg_volume,
            "median_volume": median_volume,
            "volume_std": volume_std,
            "volume_cv": volume_std / avg_volume,  # Recalculate for consistency
            "avg_volume_30d": avg_volume_30d,
            "avg_volume_ratio_30d": avg_volume_ratio_30d,
            "volume_ratio_30d_std": volume_ratio_30d_std,
            "volume_ratio_30d_median": volume_ratio_30d_median,
            "volume_percentiles": volume_percentiles,
            "high_volume_frequency": high_volume_frequency,
            "low_volume_frequency": low_volume_frequency,
            "volume_price_correlation": volume_price_corr,
            "volume_returns_correlation": volume_returns_corr,
            "volume_volatility": volume_volatility,
            "volume_trend": volume_trend,
            "volume_spike_frequency": volume_spike_frequency,
            "volume_consistency": volume_consistency,
            "max_volume": symbol_data["volume"].max(),
            "min_volume": symbol_data["volume"].min(),
            "volume_range": symbol_data["volume"].max() - symbol_data["volume"].min(),
        }

    def calculate_volume_comparison(self):
        """
        Compare volume metrics across all assets

        Returns:
            dict: Volume comparison metrics
        """
        if not self.results:
            logger.error("No analysis results. Call analyze_all_assets() first.")
            return {}


        # Collect volume metrics from all assets
        volume_metrics = []
        for symbol, result in self.results.items():
            if "volume_analysis" in result:
                metrics = result["volume_analysis"].copy()
                metrics["symbol"] = symbol
                volume_metrics.append(metrics)

        if not volume_metrics:
            return {}

        volume_df = pd.DataFrame(volume_metrics)

        # Volume ranking (using value-weighted volume as primary)
        volume_ranking_usd = volume_df.sort_values("total_volume_usd", ascending=False)
        volume_ranking = volume_df.sort_values("total_volume", ascending=False)

        # Volume categories (using value-weighted volume as primary)
        volume_categories_usd = {
            "high_volume": volume_df[volume_df["total_volume_usd"] >= volume_df["total_volume_usd"].quantile(0.75)]["symbol"].tolist(),
            "medium_volume": volume_df[(volume_df["total_volume_usd"] >= volume_df["total_volume_usd"].quantile(0.25)) &
                                     (volume_df["total_volume_usd"] < volume_df["total_volume_usd"].quantile(0.75))]["symbol"].tolist(),
            "low_volume": volume_df[volume_df["total_volume_usd"] < volume_df["total_volume_usd"].quantile(0.25)]["symbol"].tolist(),
        }
        
        # Keep raw volume categories for comparison
        volume_categories = {
            "high_volume": volume_df[volume_df["total_volume"] >= volume_df["total_volume"].quantile(0.75)]["symbol"].tolist(),
            "medium_volume": volume_df[(volume_df["total_volume"] >= volume_df["total_volume"].quantile(0.25)) &
                                     (volume_df["total_volume"] < volume_df["total_volume"].quantile(0.75))]["symbol"].tolist(),
            "low_volume": volume_df[volume_df["total_volume"] < volume_df["total_volume"].quantile(0.25)]["symbol"].tolist(),
        }

        # Volume consistency ranking (using value-weighted metrics)
        consistency_ranking_usd = volume_df.sort_values("volume_consistency_usd", ascending=False)
        consistency_ranking = volume_df.sort_values("volume_consistency", ascending=False)

        # Volume volatility ranking (using value-weighted metrics)
        volatility_ranking_usd = volume_df.sort_values("volume_volatility_usd", ascending=False)
        volatility_ranking = volume_df.sort_values("volume_volatility", ascending=False)

        # Volume-price correlation ranking
        correlation_ranking = volume_df.sort_values("volume_price_correlation", ascending=False)

        # Volume trends ranking (using value-weighted metrics)
        trend_ranking_usd = volume_df.sort_values("volume_trend_usd", ascending=False)
        trend_ranking = volume_df.sort_values("volume_trend", ascending=False)

        # Volume spike frequency ranking (using value-weighted metrics)
        spike_ranking_usd = volume_df.sort_values("volume_spike_frequency_usd", ascending=False)
        spike_ranking = volume_df.sort_values("volume_spike_frequency", ascending=False)

        # Summary statistics (using value-weighted metrics as primary)
        summary_stats = {
            # Value-weighted volume statistics (primary)
            "total_volume_usd_mean": volume_df["total_volume_usd"].mean(),
            "total_volume_usd_median": volume_df["total_volume_usd"].median(),
            "total_volume_usd_std": volume_df["total_volume_usd"].std(),
            "avg_volume_usd_mean": volume_df["avg_volume_usd"].mean(),
            "avg_volume_usd_median": volume_df["avg_volume_usd"].median(),
            "volume_consistency_usd_mean": volume_df["volume_consistency_usd"].mean(),
            "volume_volatility_usd_mean": volume_df["volume_volatility_usd"].mean(),
            "volume_spike_frequency_usd_mean": volume_df["volume_spike_frequency_usd"].mean(),
            
            # Raw volume statistics (for comparison)
            "total_volume_mean": volume_df["total_volume"].mean(),
            "total_volume_median": volume_df["total_volume"].median(),
            "total_volume_std": volume_df["total_volume"].std(),
            "avg_volume_mean": volume_df["avg_volume"].mean(),
            "avg_volume_median": volume_df["avg_volume"].median(),
            "volume_consistency_mean": volume_df["volume_consistency"].mean(),
            "volume_volatility_mean": volume_df["volume_volatility"].mean(),
            "volume_price_correlation_mean": volume_df["volume_price_correlation"].mean(),
            "volume_spike_frequency_mean": volume_df["volume_spike_frequency"].mean(),
        }

        return {
            # Value-weighted volume rankings (primary)
            "volume_ranking_usd": volume_ranking_usd,
            "volume_categories_usd": volume_categories_usd,
            "consistency_ranking_usd": consistency_ranking_usd,
            "volatility_ranking_usd": volatility_ranking_usd,
            "trend_ranking_usd": trend_ranking_usd,
            "spike_ranking_usd": spike_ranking_usd,
            
            # Raw volume rankings (for comparison)
            "volume_ranking": volume_ranking,
            "volume_categories": volume_categories,
            "consistency_ranking": consistency_ranking,
            "volatility_ranking": volatility_ranking,
            "correlation_ranking": correlation_ranking,
            "trend_ranking": trend_ranking,
            "spike_ranking": spike_ranking,
            "summary_stats": summary_stats,
            "volume_dataframe": volume_df,
        }

    def calculate_volume_patterns(self, symbol_data):
        """
        Calculate volume patterns over time

        Args:
            symbol_data (pd.DataFrame): Data for a single symbol

        Returns:
            dict: Volume pattern analysis
        """
        # Add time components
        symbol_data = symbol_data.copy()
        symbol_data["hour"] = symbol_data.index.hour
        symbol_data["day_of_week"] = symbol_data.index.dayofweek
        symbol_data["date"] = symbol_data.index.date

        # Hourly volume patterns
        hourly_volume = symbol_data.groupby("hour")["volume"].agg(["mean", "std", "min", "max"])
        hourly_volume = hourly_volume.dropna() if hasattr(hourly_volume, 'dropna') else hourly_volume
        
        # Check if hourly_volume has data and "mean" column
        if len(hourly_volume) > 0 and "mean" in hourly_volume.columns:
            peak_hours = hourly_volume["mean"].nlargest(min(3, len(hourly_volume))).index.tolist()
            low_hours = hourly_volume["mean"].nsmallest(min(3, len(hourly_volume))).index.tolist()
        else:
            peak_hours = []
            low_hours = []

        # Daily volume patterns
        daily_volume = symbol_data.groupby("day_of_week")["volume"].agg(["mean", "std", "min", "max"])
        daily_volume = daily_volume.dropna() if hasattr(daily_volume, 'dropna') else daily_volume
        
        # Check if daily_volume has data and "mean" column
        if len(daily_volume) > 0 and "mean" in daily_volume.columns:
            peak_days = daily_volume["mean"].nlargest(min(3, len(daily_volume))).index.tolist()
            low_days = daily_volume["mean"].nsmallest(min(3, len(daily_volume))).index.tolist()
        else:
            peak_days = []
            low_days = []

        # Volume autocorrelation (lag 1)
        volume_autocorr = symbol_data["volume"].autocorr(lag=1)

        # Volume seasonality (using rolling averages)
        daily_rolling_volume = symbol_data.groupby("date")["volume"].mean().rolling(window=7).mean()
        weekly_seasonality = daily_rolling_volume.std() / daily_rolling_volume.mean() if len(daily_rolling_volume.dropna()) > 0 else 0

        # Volume clustering (consecutive high/low volume periods)
        high_volume_threshold = symbol_data["volume"].quantile(0.75)
        low_volume_threshold = symbol_data["volume"].quantile(0.25)

        high_volume_clusters = self._calculate_consecutive_runs(symbol_data["volume"] >= high_volume_threshold)
        low_volume_clusters = self._calculate_consecutive_runs(symbol_data["volume"] <= low_volume_threshold)

        return {
            "hourly_volume": hourly_volume.to_dict() if hasattr(hourly_volume, 'to_dict') else {},
            "daily_volume": daily_volume.to_dict() if hasattr(daily_volume, 'to_dict') else {},
            "peak_hours": peak_hours,
            "low_hours": low_hours,
            "peak_days": peak_days,
            "low_days": low_days,
            "volume_autocorrelation": volume_autocorr if not np.isnan(volume_autocorr) else 0.0,
            "weekly_seasonality": weekly_seasonality if not np.isnan(weekly_seasonality) else 0.0,
            "high_volume_clusters": high_volume_clusters,
            "low_volume_clusters": low_volume_clusters,
        }

    def analyze_all_assets(self):
        """Analyze all assets in the dataset"""
        if self.df is None:
            logger.error("No data loaded. Call load_data() first.")
            return

        symbols = self.df["symbol"].unique()
        logger.info(f"Analyzing {len(symbols)} assets")

        for symbol in symbols:
            logger.info(f"Analyzing {symbol}")
            symbol_data = self.df[self.df["symbol"] == symbol].copy()

            # Basic metrics
            basic_metrics = self.calculate_basic_metrics(symbol_data)

            # Triple barrier profits
            triple_barrier_profits = self.calculate_triple_barrier_profits(symbol_data)

            # Intraday patterns
            intraday_patterns = self.calculate_intraday_patterns(symbol_data)

            # Movement statistics
            movement_stats = self.calculate_movement_statistics(symbol_data)

            # Volume analysis
            volume_analysis = self.calculate_volume_analysis(symbol_data)

            # Volume patterns
            volume_patterns = self.calculate_volume_patterns(symbol_data)

            # Store results
            self.results[symbol] = {
                "basic_metrics": basic_metrics,
                "triple_barrier_profits": triple_barrier_profits,
                "intraday_patterns": intraday_patterns,
                "movement_statistics": movement_stats,
                "volume_analysis": volume_analysis,
                "volume_patterns": volume_patterns,
            }

    def _calculate_composite_scores(self):
        """Calculate composite trading opportunity scores for all assets"""
        if not self.results:
            return {}
        
        composite_scores = {}
        
        for symbol, data in self.results.items():
            barrier_results = data.get("triple_barrier_profits", {})
            
            if not barrier_results:
                composite_scores[symbol] = {
                    "profit_score": 0,
                    "frequency_score": 0,
                    "consistency_score": 0,
                    "composite_score": 0
                }
                continue
            
            # Calculate average profit across all barrier levels
            all_profits = []
            all_frequencies = []
            total_trades = 0
            
            for barrier_key, barrier_data in barrier_results.items():
                if isinstance(barrier_data, dict) and "avg_profit" in barrier_data:
                    all_profits.append(barrier_data["avg_profit"])
                    all_frequencies.append(barrier_data.get("profit_frequency", 0))
                    total_trades += barrier_data.get("total_trades", 0)
            
            if not all_profits:
                composite_scores[symbol] = {
                    "profit_score": 0,
                    "frequency_score": 0,
                    "consistency_score": 0,
                    "composite_score": 0
                }
                continue
            
            # Calculate component scores
            avg_profit = np.mean(all_profits)
            avg_frequency = np.mean(all_frequencies)
            
            # Profit Score (40% weight): avg_profit / 0.02 (normalized to max expected ~2%)
            profit_score = min(avg_profit / 0.02, 1.0)
            
            # Frequency Score (40% weight): avg_frequency / 0.30 (normalized to max expected ~30%)
            frequency_score = min(avg_frequency / 0.30, 1.0)
            
            # Consistency Score (20% weight): min(1.0, total_trades / 20,000)
            consistency_score = min(total_trades / 20000, 1.0)
            
            # Composite Score: weighted combination
            composite_score = (profit_score * 0.4) + (frequency_score * 0.4) + (consistency_score * 0.2)
            
            composite_scores[symbol] = {
                "profit_score": profit_score,
                "frequency_score": frequency_score,
                "consistency_score": consistency_score,
                "composite_score": composite_score
            }
        
        return composite_scores

    def generate_summary_report(self, save_to_file=True, output_dir="results"):
        """Generate a comprehensive summary report"""
        if not self.results:
            logger.error("No analysis results. Call analyze_all_assets() first.")
            return None

        # Create output directory if it doesn't exist
        if save_to_file:
            Path(output_dir).mkdir(exist_ok=True)
            Path(f"{output_dir}/reports").mkdir(exist_ok=True)

        report_lines = []
        def add_line(line=""):
            if save_to_file:
                report_lines.append(line)
            print(line)

        add_line("\n" + "="*80)
        add_line("CRYPTOCURRENCY PRICE MOVEMENT ANALYSIS REPORT")
        add_line("="*80)
        add_line()
        add_line("METHODOLOGY: COMPOSITE TRADING OPPORTUNITY SCORE")
        add_line("=" * 50)
        add_line("The Composite Score combines three weighted factors to rank trading opportunities:")
        add_line()
        add_line("1. PROFIT SCORE (40% weight):")
        add_line("   - Measures average profit per successful trade")
        add_line("   - Formula: avg_profit / 0.02 (normalized to max expected ~2%)")
        add_line("   - Higher profits = better score")
        add_line()
        add_line("2. FREQUENCY SCORE (40% weight):")
        add_line("   - Measures how often trading opportunities occur")
        add_line("   - Formula: success_rate / 0.30 (normalized to max expected ~30%)")
        add_line("   - More frequent opportunities = better score")
        add_line()
        add_line("3. CONSISTENCY SCORE (20% weight):")
        add_line("   - Measures reliability of opportunities")
        add_line("   - Formula: min(1.0, total_trades / 20,000)")
        add_line("   - More total trades = more consistent pattern")
        add_line()
        add_line("FINAL FORMULA: (Profit_Score × 0.4) + (Frequency_Score × 0.4) + (Consistency_Score × 0.2)")
        add_line()
        add_line("DAILY PROFIT POTENTIAL CALCULATION:")
        add_line("  Formula: Average Profit per Trade (%) × Number of Daily Opportunities")
        add_line("  Daily Opportunities = Total Opportunities ÷ Number of Days in Dataset")
        add_line("  Example: 18,544 total opportunities ÷ 730 days = 25.4 opportunities/day")
        add_line("  Daily ROI = 1.44% profit × 25.4 opportunities = 36.6% potential daily ROI")
        add_line("  Note: This represents the maximum theoretical daily ROI if all opportunities are captured")
        add_line()
        add_line("EXAMPLE - ALGOUSDT:")
        add_line("  Profit Score: 1.44% / 2% = 0.720")
        add_line("  Frequency Score: 26.5% / 30% = 0.883")
        add_line("  Consistency Score: min(1.0, 18,544/20,000) = 0.927")
        add_line("  Composite: (0.720×0.4) + (0.883×0.4) + (0.927×0.2) = 0.827")
        add_line("  Daily Opportunities: 18,544 total ÷ 730 days = 25.4 opportunities/day")
        add_line("  Daily ROI: 1.44% × 25.4 opportunities = 36.6% potential daily ROI")
        add_line()
        add_line("INTERPRETATION:")
        add_line("  Score 0.8+: Excellent trading opportunities")
        add_line("  Score 0.6-0.8: Good trading opportunities")
        add_line("  Score 0.4-0.6: Moderate trading opportunities")
        add_line("  Score <0.4: Limited trading opportunities")
        add_line()

        # Create summary DataFrames
        basic_summary = []
        barrier_summary = []

        for symbol, result in self.results.items():
            basic = result["basic_metrics"]

            basic_summary.append({
                "Symbol": symbol,
                "Total_Return": basic["total_return"],
                "Volatility": basic["volatility"],
                "Avg_Daily_Range": basic["avg_daily_range"],
                "Avg_Intraday_Movement": basic["avg_intraday_movement"],
                "Avg_Price_Change": basic["avg_price_change"],
                "Avg_Volume": basic["avg_volume"],
            })

            # Triple barrier results
            for barrier_name, barrier_data in result["triple_barrier_profits"].items():
                barrier_level = int(barrier_name.split("_")[1].replace("bp", "")) / 1000
                barrier_summary.append({
                    "Symbol": symbol,
                    "Barrier_Level": f"{barrier_level:.1%}",
                    "Total_Trades": barrier_data["total_trades"],
                    "Avg_Profit": barrier_data["avg_profit"],
                    "Long_Trades": barrier_data["long_trades"],
                    "Short_Trades": barrier_data["short_trades"],
                    "Profit_Frequency": barrier_data["profit_frequency"],
                    "Max_Profit": barrier_data["max_profit"],
                    "Total_Potential_Profit": barrier_data["total_potential_profit"],
                })

        basic_df = pd.DataFrame(basic_summary)
        barrier_df = pd.DataFrame(barrier_summary)

        # Print basic metrics
        add_line("\nBASIC PRICE MOVEMENT METRICS:")
        add_line("-" * 60)
        add_line(basic_df.round(4).to_string(index=False))

        # Print triple barrier results
        add_line("\nTRIPLE BARRIER PROFIT ANALYSIS:")
        add_line("-" * 60)
        add_line(barrier_df.round(4).to_string(index=False))

        # Top performers by barrier level
        add_line("\nTOP PERFORMERS BY BARRIER LEVEL:")
        add_line("-" * 60)

        barrier_levels = barrier_df["Barrier_Level"].unique()
        for barrier in sorted(barrier_levels):
            barrier_data = barrier_df[barrier_df["Barrier_Level"] == barrier]
            top_performers = barrier_data.nlargest(5, "Avg_Profit")[["Symbol", "Barrier_Level", "Avg_Profit", "Total_Trades", "Long_Trades", "Short_Trades"]]
            add_line(f"\nTop 5 for {barrier} barrier:")
            add_line(top_performers.round(4).to_string(index=False))

        # Best overall performers
        add_line("\nBEST OVERALL PERFORMERS:")
        add_line("-" * 60)

        # Average across all barriers
        avg_profits = barrier_df.groupby("Symbol")["Avg_Profit"].mean().sort_values(ascending=False)
        add_line("Average profit across all barriers:")
        for symbol, profit in avg_profits.head(10).items():
            add_line(f"  {symbol}: {profit:.4f}")

        # Most active assets
        total_trades = barrier_df.groupby("Symbol")["Total_Trades"].sum().sort_values(ascending=False)
        add_line("\nMost active assets (total trades across all barriers):")
        for symbol, trades in total_trades.head(10).items():
            add_line(f"  {symbol}: {trades:.0f} trades")

        # Long vs Short analysis
        total_longs = barrier_df.groupby("Symbol")["Long_Trades"].sum().sort_values(ascending=False)
        total_shorts = barrier_df.groupby("Symbol")["Short_Trades"].sum().sort_values(ascending=False)
        add_line("\nAssets with most long trades:")
        for symbol, trades in total_longs.head(5).items():
            add_line(f"  {symbol}: {trades:.0f} long trades")
        add_line("\nAssets with most short trades:")
        for symbol, trades in total_shorts.head(5).items():
            add_line(f"  {symbol}: {trades:.0f} short trades")

        # Add Composite Score Analysis
        add_line("\nCOMPOSITE TRADING OPPORTUNITY SCORES:")
        add_line("=" * 50)
        add_line("Ranking assets by overall trading opportunity quality")
        add_line()
        
        # Calculate composite scores for all assets
        composite_scores = {}
        for symbol, result in self.results.items():
            # Use average metrics across all barriers
            avg_profit = barrier_df[barrier_df["Symbol"] == symbol]["Avg_Profit"].mean()
            avg_frequency = barrier_df[barrier_df["Symbol"] == symbol]["Profit_Frequency"].mean()
            total_trades_sum = barrier_df[barrier_df["Symbol"] == symbol]["Total_Trades"].sum()
            
            # Calculate component scores
            profit_score = avg_profit / 0.02  # Normalized to max ~2%
            frequency_score = avg_frequency / 0.30  # Normalized to max ~30%
            consistency_score = min(1.0, total_trades_sum / 20000)  # Capped at 1.0
            
            # Weighted composite score
            composite_score = (profit_score * 0.4) + (frequency_score * 0.4) + (consistency_score * 0.2)
            
            composite_scores[symbol] = {
                'composite_score': composite_score,
                'profit_score': profit_score,
                'frequency_score': frequency_score,
                'consistency_score': consistency_score,
                'avg_profit': avg_profit,
                'avg_frequency': avg_frequency,
                'total_trades': total_trades_sum
            }
        
        # Sort by composite score
        sorted_composite = sorted(composite_scores.items(), key=lambda x: x[1]['composite_score'], reverse=True)
        
        add_line("RANK | SYMBOL   | COMPOSITE | PROFIT | FREQUENCY | CONSISTENCY | DAILY ROI | INTERPRETATION")
        add_line("-" * 85)
        
        for i, (symbol, scores) in enumerate(sorted_composite, 1):
            # Calculate daily profit potential
            # Formula: average profit per trade (%) * number of daily opportunities = potential daily ROI
            # Daily opportunities = total opportunities / number of days in dataset
            # Estimate days: total_trades / (profit_frequency * 96) where 96 = 15-min periods per day
            estimated_days = scores['total_trades'] / (scores['avg_frequency'] * 96) if scores['avg_frequency'] > 0 else 1
            opportunities_per_day = scores['total_trades'] / estimated_days if estimated_days > 0 else scores['total_trades']
            daily_profit_potential = scores['avg_profit'] * opportunities_per_day * 100
            
            # Determine interpretation
            if scores['composite_score'] >= 0.8:
                interpretation = "Excellent"
            elif scores['composite_score'] >= 0.6:
                interpretation = "Good"
            elif scores['composite_score'] >= 0.4:
                interpretation = "Moderate"
            else:
                interpretation = "Limited"
            
            add_line(f"{i:4d} | {symbol:8s} | {scores['composite_score']:9.3f} | {scores['profit_score']:6.3f} | {scores['frequency_score']:9.3f} | {scores['consistency_score']:11.3f} | {daily_profit_potential:7.2f}% | {interpretation}")
        
        add_line()
        add_line("DETAILED BREAKDOWN:")
        add_line("-" * 30)
        for i, (symbol, scores) in enumerate(sorted_composite[:5], 1):  # Top 5 detailed breakdown
            # Calculate daily profit potential
            # Formula: average profit per trade (%) * number of daily opportunities = potential daily ROI
            estimated_days = scores['total_trades'] / (scores['avg_frequency'] * 96) if scores['avg_frequency'] > 0 else 1
            opportunities_per_day = scores['total_trades'] / estimated_days if estimated_days > 0 else scores['total_trades']
            daily_profit_potential = scores['avg_profit'] * opportunities_per_day * 100
            
            add_line(f"{i}. {symbol}:")
            add_line(f"   Composite Score: {scores['composite_score']:.3f}")
            add_line(f"   Average Profit per Trade: {scores['avg_profit']*100:.2f}%")
            add_line(f"   Average Success Rate: {scores['avg_frequency']*100:.1f}%")
            add_line(f"   Total Trading Opportunities: {scores['total_trades']:,}")
            add_line(f"   Daily Opportunities: {opportunities_per_day:.1f} trades/day")
            add_line(f"   Daily Profit Potential: {daily_profit_potential:.2f}% ROI/day")
            add_line(f"   Component Scores: P={scores['profit_score']:.3f}, F={scores['frequency_score']:.3f}, C={scores['consistency_score']:.3f}")
            add_line()

        # Volume comparison summary
        volume_comparison_summary = self.calculate_volume_comparison()
        if volume_comparison_summary:
            add_line("\nVOLUME ANALYSIS SUMMARY (Value-Weighted USD):")
            add_line("-" * 60)
            add_line(f"Total Volume USD Mean: ${volume_comparison_summary['summary_stats']['total_volume_usd_mean']:,.2f}")
            add_line(f"Total Volume USD Median: ${volume_comparison_summary['summary_stats']['total_volume_usd_median']:,.2f}")
            add_line(f"Total Volume USD Std Dev: ${volume_comparison_summary['summary_stats']['total_volume_usd_std']:,.2f}")
            add_line(f"Avg Volume USD Mean: ${volume_comparison_summary['summary_stats']['avg_volume_usd_mean']:,.2f}")
            add_line(f"Avg Volume USD Median: ${volume_comparison_summary['summary_stats']['avg_volume_usd_median']:,.2f}")
            add_line(f"Volume Consistency USD Mean: {volume_comparison_summary['summary_stats']['volume_consistency_usd_mean']:.2f}")
            add_line(f"Volume Volatility USD Mean: {volume_comparison_summary['summary_stats']['volume_volatility_usd_mean']:.2f}")
            add_line(f"Volume Price Correlation Mean: {volume_comparison_summary['summary_stats']['volume_price_correlation_mean']:.2f}")
            add_line(f"Volume Spike Frequency USD Mean: {volume_comparison_summary['summary_stats']['volume_spike_frequency_usd_mean']:.2f}")

            add_line("\nVOLUME RANKINGS (Value-Weighted USD):")
            add_line("-" * 60)

            # Top 10 by total volume (value-weighted)
            add_line("Top 10 Assets by Total Volume (USD Value):")
            top_volume_usd = volume_comparison_summary["volume_ranking_usd"].head(10)
            for _, row in top_volume_usd.iterrows():
                add_line(f"  {row['symbol']}: ${row['total_volume_usd']:,.2f}")

            # Top 10 by volume consistency (value-weighted)
            add_line("\nTop 10 Assets by Volume Consistency (USD):")
            top_consistency_usd = volume_comparison_summary["consistency_ranking_usd"].head(10)
            for _, row in top_consistency_usd.iterrows():
                add_line(f"  {row['symbol']}: {row['volume_consistency_usd']:.4f}")

            # Top 10 by volume-price correlation
            add_line("\nTop 10 Assets by Volume-Price Correlation:")
            top_correlation = volume_comparison_summary["correlation_ranking"].head(10)
            for _, row in top_correlation.iterrows():
                add_line(f"  {row['symbol']}: {row['volume_price_correlation']:.4f}")

            # Volume categories (value-weighted)
            add_line("\nVOLUME CATEGORIES (Value-Weighted USD):")
            add_line("-" * 60)
            add_line(f"High Volume Assets (USD): {', '.join(volume_comparison_summary['volume_categories_usd']['high_volume'])}")
            add_line(f"Medium Volume Assets (USD): {', '.join(volume_comparison_summary['volume_categories_usd']['medium_volume'])}")
            add_line(f"Low Volume Assets (USD): {', '.join(volume_comparison_summary['volume_categories_usd']['low_volume'])}")

            # Volume patterns summary
            add_line("\nVOLUME PATTERNS SUMMARY:")
            add_line("-" * 60)
            for symbol in list(self.results.keys())[:5]:  # Show first 5 assets
                if "volume_patterns" in self.results[symbol]:
                    patterns = self.results[symbol]["volume_patterns"]
                    add_line(f"\n{symbol}:")
                    add_line(f"  Peak Hours: {patterns['peak_hours']}")
                    add_line(f"  Peak Days: {patterns['peak_days']}")
                    add_line(f"  Volume Autocorrelation: {patterns['volume_autocorrelation']:.4f}")
                    add_line(f"  Weekly Seasonality: {patterns['weekly_seasonality']:.4f}")

        # Save report to file
        if save_to_file and report_lines:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"{output_dir}/reports/comprehensive_crypto_analysis_{timestamp}.txt"
            with open(report_file, 'w') as f:
                f.write('\n'.join(report_lines))
            logger.info(f"Comprehensive report saved to {report_file}")

        # Save results to CSV
        if save_to_file:
            csv_dir = Path(output_dir) / "csv"
            csv_dir.mkdir(exist_ok=True)
            
            # Save basic summary
            basic_df.to_csv(csv_dir / "price_movement_metrics.csv", index=False)
            logger.info("Basic metrics saved to price_movement_metrics.csv")
            
            # Save barrier summary
            barrier_df.to_csv(csv_dir / "triple_barrier_profits.csv", index=False)
            logger.info("Triple barrier results saved to triple_barrier_profits.csv")
            
            # Save volume analysis results
            volume_df = volume_comparison_summary.get("volume_dataframe", pd.DataFrame()) if volume_comparison_summary else pd.DataFrame()
            if not volume_df.empty:
                volume_df.to_csv(csv_dir / "volume_analysis.csv", index=False)
                logger.info("Volume analysis results saved to volume_analysis.csv")
            
            # Create comprehensive consolidated CSV with all requested metrics
            consolidated_data = []
            
            # Get unique assets
            assets = basic_df['Symbol'].unique() if 'Symbol' in basic_df.columns else basic_df.index.unique()
            
            # Calculate composite scores for all assets
            composite_scores = self._calculate_composite_scores()
            
            for asset in assets:
                # Get basic metrics for this asset
                asset_basic = basic_df[basic_df['Symbol'] == asset] if 'Symbol' in basic_df.columns else basic_df.loc[asset]
                avg_volume = asset_basic['Avg_Volume'].iloc[0] if 'Avg_Volume' in asset_basic.columns else 0
                avg_intraday_movement = asset_basic['Avg_Intraday_Movement'].iloc[0] if 'Avg_Intraday_Movement' in asset_basic.columns else 0
                avg_price_change = asset_basic['Avg_Price_Change'].iloc[0] if 'Avg_Price_Change' in asset_basic.columns else 0
                avg_daily_range = asset_basic['Avg_Daily_Range'].iloc[0] if 'Avg_Daily_Range' in asset_basic.columns else 0
                
                # Get triple barrier profits for this asset
                asset_barriers = barrier_df[barrier_df['Symbol'] == asset] if 'Symbol' in barrier_df.columns else pd.DataFrame()
                
                # Calculate daily ROI potential for each barrier level (main outcome)
                daily_roi_potentials = {}
                avg_daily_roi = 0
                best_barrier_roi = 0
                best_barrier_level = "N/A"
                
                if not asset_barriers.empty:
                    # Estimate days in dataset (assuming 15-min intervals, 96 per day)
                    total_periods = asset_barriers['Total_Trades'].sum() / asset_barriers['Profit_Frequency'].mean() if asset_barriers['Profit_Frequency'].mean() > 0 else 1
                    estimated_days = total_periods / 96
                    
                    daily_rois = []
                    barrier_roi_pairs = []
                    
                    for _, row in asset_barriers.iterrows():
                        barrier_level = row['Barrier_Level']
                        avg_profit = row['Avg_Profit']
                        daily_opportunities = row['Total_Trades'] / estimated_days if estimated_days > 0 else row['Total_Trades']
                        daily_roi = avg_profit * daily_opportunities * 100  # Convert to percentage
                        daily_roi_potentials[f"Daily_ROI_{barrier_level}"] = daily_roi
                        daily_rois.append(daily_roi)
                        barrier_roi_pairs.append((barrier_level, daily_roi))
                    
                    # Calculate average daily ROI across all barrier levels
                    avg_daily_roi = np.mean(daily_rois) if daily_rois else 0
                    
                    # Find the best barrier level and its ROI
                    if barrier_roi_pairs:
                        best_barrier_level, best_barrier_roi = max(barrier_roi_pairs, key=lambda x: x[1])
                
                # Get composite scores for this asset
                asset_scores = composite_scores.get(asset, {})
                
                # Create row data with Daily ROI as main outcome
                row_data = {
                    'Symbol': asset,
                    'Best_Daily_ROI': best_barrier_roi,  # Main outcome - best daily ROI from optimal barrier
                    'Best_Barrier_Level': best_barrier_level,  # Which barrier level achieved the best ROI
                    'Average_Daily_ROI': avg_daily_roi,  # Average daily ROI across all barriers (for comparison)
                    'Average_Volume': avg_volume,
                    'Avg_Intraday_Movement': avg_intraday_movement,
                    'Avg_Price_Change': avg_price_change,
                    'Avg_Daily_Range': avg_daily_range,
                    'Profit_Score': asset_scores.get('profit_score', 0),
                    'Frequency_Score': asset_scores.get('frequency_score', 0),
                    'Consistency_Score': asset_scores.get('consistency_score', 0),
                    'Composite_Score': asset_scores.get('composite_score', 0),
                }
                
                # Add barrier-specific daily ROI potentials
                row_data.update(daily_roi_potentials)
                
                consolidated_data.append(row_data)
            
            # Create consolidated DataFrame
            consolidated_df = pd.DataFrame(consolidated_data)
            
            # Sort columns with Best Daily ROI as the main outcome (second column) and barrier info
            basic_cols = ['Symbol', 'Best_Daily_ROI', 'Best_Barrier_Level', 'Average_Daily_ROI', 'Average_Volume', 
                         'Avg_Intraday_Movement', 'Avg_Price_Change', 'Avg_Daily_Range', 'Profit_Score', 
                         'Frequency_Score', 'Consistency_Score', 'Composite_Score']
            barrier_cols = [col for col in consolidated_df.columns if col.startswith('Daily_ROI_')]
            barrier_cols.sort(key=lambda x: float(x.split('_')[-1].rstrip('%')))
            final_cols = basic_cols + barrier_cols
            
            # Reorder DataFrame
            consolidated_df = consolidated_df[final_cols]
            
            consolidated_df.to_csv(csv_dir / "comprehensive_asset_summary.csv", index=False)
            logger.info("Comprehensive asset summary saved to comprehensive_asset_summary.csv")
            
            logger.info(f"CSV files saved to {csv_dir}/")

        return {
            "basic_summary": basic_df,
            "barrier_summary": barrier_df,
            "volume_summary": volume_comparison_summary.get("volume_dataframe", pd.DataFrame()) if volume_comparison_summary else pd.DataFrame(),
        }

    def create_visualizations(self, output_dir="plots"):
        """Create visualizations for the analysis"""
        if not self.results:
            logger.error("No analysis results. Call analyze_all_assets() first.")
            return

        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # 1. Triple Barrier Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Triple Barrier Profit Analysis", fontsize=16)

        # Prepare data for plotting
        barrier_data = []
        for symbol, result in self.results.items():
            for barrier_name, barrier_result in result["triple_barrier_profits"].items():
                barrier_level = int(barrier_name.split("_")[1].replace("bp", "")) / 1000
                barrier_data.append({
                    "Symbol": symbol,
                    "Barrier_Level": barrier_level,
                    "Avg_Profit": barrier_result["avg_profit"],
                    "Total_Trades": barrier_result["total_trades"],
                    "Long_Trades": barrier_result["long_trades"],
                    "Short_Trades": barrier_result["short_trades"],
                })

        barrier_df = pd.DataFrame(barrier_data)

        # Plot 1: Average profit by barrier level
        for barrier in sorted(barrier_df["Barrier_Level"].unique()):
            data = barrier_df[barrier_df["Barrier_Level"] == barrier]
            axes[0, 0].scatter(data["Symbol"], data["Avg_Profit"],
                             label=f"{barrier:.1%}", alpha=0.7, s=50)

        axes[0, 0].set_xlabel("Assets")
        axes[0, 0].set_ylabel("Average Profit")
        axes[0, 0].set_title("Average Profit by Barrier Level")
        axes[0, 0].tick_params(axis="x", rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Total trades by barrier level
        for barrier in sorted(barrier_df["Barrier_Level"].unique()):
            data = barrier_df[barrier_df["Barrier_Level"] == barrier]
            axes[0, 1].scatter(data["Symbol"], data["Total_Trades"],
                             label=f"{barrier:.1%}", alpha=0.7, s=50)

        axes[0, 1].set_xlabel("Assets")
        axes[0, 1].set_ylabel("Total Trades")
        axes[0, 1].set_title("Total Trades by Barrier Level")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Average performance across all barriers
        avg_performance = barrier_df.groupby("Symbol").agg({
            "Avg_Profit": "mean",
            "Total_Trades": "sum",
            "Long_Trades": "sum",
            "Short_Trades": "sum",
        }).sort_values("Avg_Profit", ascending=False)

        axes[1, 0].bar(range(len(avg_performance)), avg_performance["Avg_Profit"])
        axes[1, 0].set_xlabel("Assets")
        axes[1, 0].set_ylabel("Average Profit (All Barriers)")
        axes[1, 0].set_title("Average Performance Across All Barriers")
        axes[1, 0].set_xticks(range(len(avg_performance)))
        axes[1, 0].set_xticklabels(avg_performance.index, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Long vs Short trades
        long_trades = avg_performance["Long_Trades"].values
        short_trades = avg_performance["Short_Trades"].values
        x_pos = np.arange(len(avg_performance))

        axes[1, 1].bar(x_pos - 0.2, long_trades, 0.4, label="Long Trades", alpha=0.8)
        axes[1, 1].bar(x_pos + 0.2, short_trades, 0.4, label="Short Trades", alpha=0.8)
        axes[1, 1].set_xlabel("Assets")
        axes[1, 1].set_ylabel("Number of Trades")
        axes[1, 1].set_title("Long vs Short Trades")
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(avg_performance.index, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)



        plt.tight_layout()
        plt.savefig(f"{output_dir}/triple_barrier_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Intraday Patterns (Movement Statistics Only)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Intraday Price Movement Patterns", fontsize=16)

        # Sample a few assets
        sample_symbols = list(self.results.keys())[:5]

        # Movement size distribution
        for symbol in sample_symbols:
            movement_stats = self.results[symbol]["movement_statistics"]
            percentiles = list(movement_stats["movement_percentiles"].keys())
            values = list(movement_stats["movement_percentiles"].values())
            axes[0, 0].plot(percentiles, values, label=symbol, marker="^", alpha=0.7)

        axes[0, 0].set_xlabel("Percentile")
        axes[0, 0].set_ylabel("Movement Size")
        axes[0, 0].set_title("Price Movement Distribution")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Average daily range vs volatility
        daily_ranges = [self.results[s]["basic_metrics"]["avg_daily_range"] for s in sample_symbols]
        volatilities = [self.results[s]["basic_metrics"]["volatility"] for s in sample_symbols]

        axes[0, 1].scatter(volatilities, daily_ranges, alpha=0.7, s=100)
        axes[0, 1].set_xlabel("Volatility")
        axes[0, 1].set_ylabel("Average Daily Range")
        axes[0, 1].set_title("Volatility vs Daily Range")
        axes[0, 1].grid(True, alpha=0.3)

        # Add labels
        for i, symbol in enumerate(sample_symbols):
            axes[0, 1].annotate(symbol, (volatilities[i], daily_ranges[i]),
                              xytext=(5, 5), textcoords="offset points", fontsize=8)

        # Average price change vs volatility
        avg_price_changes = [self.results[s]["basic_metrics"]["avg_price_change"] for s in sample_symbols]
        axes[1, 0].scatter(volatilities, avg_price_changes, alpha=0.7, s=100, color="orange")
        axes[1, 0].set_xlabel("Volatility")
        axes[1, 0].set_ylabel("Average Price Change")
        axes[1, 0].set_title("Volatility vs Average Price Change")
        axes[1, 0].grid(True, alpha=0.3)

        # Add labels
        for i, symbol in enumerate(sample_symbols):
            axes[1, 0].annotate(symbol, (volatilities[i], avg_price_changes[i]),
                              xytext=(5, 5), textcoords="offset points", fontsize=8)

        # Total return vs volatility
        total_returns = [self.results[s]["basic_metrics"]["total_return"] for s in sample_symbols]
        axes[1, 1].scatter(volatilities, total_returns, alpha=0.7, s=100, color="purple")
        axes[1, 1].set_xlabel("Volatility")
        axes[1, 1].set_ylabel("Total Return")
        axes[1, 1].set_title("Volatility vs Total Return")
        axes[1, 1].grid(True, alpha=0.3)

        # Add labels
        for i, symbol in enumerate(sample_symbols):
            axes[1, 1].annotate(symbol, (volatilities[i], total_returns[i]),
                              xytext=(5, 5), textcoords="offset points", fontsize=8)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/intraday_patterns.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 3. Volume Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Volume Analysis", fontsize=16)

        # Volume ranking (using value-weighted volume)
        volume_comparison = self.calculate_volume_comparison()
        if volume_comparison:
            volume_ranking_usd = volume_comparison["volume_ranking_usd"]
            symbols = volume_ranking_usd["symbol"].values
            total_volumes_usd = volume_ranking_usd["total_volume_usd"].values

            axes[0, 0].bar(range(len(symbols)), total_volumes_usd, alpha=0.7)
            axes[0, 0].set_xlabel("Assets")
            axes[0, 0].set_ylabel("Total Volume (USD)")
            axes[0, 0].set_title("Total Volume by Asset (Value-Weighted)")
            axes[0, 0].set_xticks(range(len(symbols)))
            axes[0, 0].set_xticklabels(symbols, rotation=45)
            axes[0, 0].grid(True, alpha=0.3)

            # Volume consistency (value-weighted)
            consistency_ranking_usd = volume_comparison["consistency_ranking_usd"]
            consistency_values_usd = consistency_ranking_usd["volume_consistency_usd"].values

            axes[0, 1].bar(range(len(symbols)), consistency_values_usd, alpha=0.7, color="green")
            axes[0, 1].set_xlabel("Assets")
            axes[0, 1].set_ylabel("Volume Consistency (USD)")
            axes[0, 1].set_title("Volume Consistency by Asset (Value-Weighted)")
            axes[0, 1].set_xticks(range(len(symbols)))
            axes[0, 1].set_xticklabels(symbols, rotation=45)
            axes[0, 1].grid(True, alpha=0.3)

            # Volume volatility (value-weighted)
            volatility_ranking_usd = volume_comparison["volatility_ranking_usd"]
            volatility_values_usd = volatility_ranking_usd["volume_volatility_usd"].values

            axes[1, 0].bar(range(len(symbols)), volatility_values_usd, alpha=0.7, color="red")
            axes[1, 0].set_xlabel("Assets")
            axes[1, 0].set_ylabel("Volume Volatility (USD)")
            axes[1, 0].set_title("Volume Volatility by Asset (Value-Weighted)")
            axes[1, 0].set_xticks(range(len(symbols)))
            axes[1, 0].set_xticklabels(symbols, rotation=45)
            axes[1, 0].grid(True, alpha=0.3)

            # Volume-price correlation
            correlation_ranking = volume_comparison["correlation_ranking"]
            correlation_values = correlation_ranking["volume_price_correlation"].values

            axes[1, 1].bar(range(len(symbols)), correlation_values, alpha=0.7, color="purple")
            axes[1, 1].set_xlabel("Assets")
            axes[1, 1].set_ylabel("Volume-Price Correlation")
            axes[1, 1].set_title("Volume-Price Correlation by Asset")
            axes[1, 1].set_xticks(range(len(symbols)))
            axes[1, 1].set_xticklabels(symbols, rotation=45)
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/volume_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 4. Volume Patterns
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Volume Patterns Over Time", fontsize=16)

        # Hourly volume patterns
        for symbol in sample_symbols:
            if "volume_patterns" in self.results[symbol]:
                hourly_volume = self.results[symbol]["volume_patterns"]["hourly_volume"]
                if hourly_volume and isinstance(hourly_volume, dict):
                    hours = list(hourly_volume.keys())
                    means = []
                    for h in hours:
                        if isinstance(hourly_volume[h], dict) and "mean" in hourly_volume[h]:
                            means.append(hourly_volume[h]["mean"])
                        elif isinstance(hourly_volume[h], (int, float)):
                            means.append(hourly_volume[h])
                    if means:
                        axes[0, 0].plot(hours[:len(means)], means, label=symbol, marker="o", alpha=0.7)

        axes[0, 0].set_xlabel("Hour of Day")
        axes[0, 0].set_ylabel("Average Volume")
        axes[0, 0].set_title("Hourly Volume Patterns")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Daily volume patterns
        for symbol in sample_symbols:
            if "volume_patterns" in self.results[symbol]:
                daily_volume = self.results[symbol]["volume_patterns"]["daily_volume"]
                if daily_volume and isinstance(daily_volume, dict):
                    days = list(daily_volume.keys())
                    means = []
                    for d in days:
                        if isinstance(daily_volume[d], dict) and "mean" in daily_volume[d]:
                            means.append(daily_volume[d]["mean"])
                        elif isinstance(daily_volume[d], (int, float)):
                            means.append(daily_volume[d])
                    if means:
                        axes[0, 1].plot(days[:len(means)], means, label=symbol, marker="s", alpha=0.7)

        axes[0, 1].set_xlabel("Day of Week")
        axes[0, 1].set_ylabel("Average Volume")
        axes[0, 1].set_title("Daily Volume Patterns")
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Volume autocorrelation
        autocorr_values = []
        autocorr_symbols = []
        for symbol in sample_symbols:
            if "volume_patterns" in self.results[symbol]:
                autocorr = self.results[symbol]["volume_patterns"]["volume_autocorrelation"]
                if not np.isnan(autocorr):
                    autocorr_values.append(autocorr)
                    autocorr_symbols.append(symbol)

        if autocorr_values:
            axes[1, 0].bar(range(len(autocorr_symbols)), autocorr_values, alpha=0.7, color="orange")
            axes[1, 0].set_xlabel("Assets")
            axes[1, 0].set_ylabel("Volume Autocorrelation")
            axes[1, 0].set_title("Volume Autocorrelation (Lag 1)")
            axes[1, 0].set_xticks(range(len(autocorr_symbols)))
            axes[1, 0].set_xticklabels(autocorr_symbols, rotation=45)
            axes[1, 0].grid(True, alpha=0.3)

        # Volume spike frequency (value-weighted)
        spike_values_usd = []
        spike_symbols = []
        for symbol in sample_symbols:
            spike_freq_usd = self.results[symbol]["volume_analysis"]["volume_spike_frequency_usd"]
            spike_values_usd.append(spike_freq_usd)
            spike_symbols.append(symbol)

        if spike_values_usd:
            axes[1, 1].bar(range(len(spike_symbols)), spike_values_usd, alpha=0.7, color="brown")
            axes[1, 1].set_xlabel("Assets")
            axes[1, 1].set_ylabel("Volume Spike Frequency (USD)")
            axes[1, 1].set_title("Volume Spike Frequency (>2x Average) - Value-Weighted")
            axes[1, 1].set_xticks(range(len(spike_symbols)))
            axes[1, 1].set_xticklabels(spike_symbols, rotation=45)
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/volume_patterns.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Visualizations saved to {output_dir}/")

def main():
    """Main function to run the analysis"""

    # Find the most recent data file
    data_dir = Path("data")
    if not data_dir.exists():
        logger.error("Data directory not found. Run data_downloader.py first.")
        return

    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        logger.error("No Parquet files found in data directory.")
        return

    # Use the most recent file
    latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Using data file: {latest_file}")

    # Create analyzer and run analysis
    analyzer = CryptoPriceAnalyzer(latest_file)

    if not analyzer.load_data():
        return

    analyzer.analyze_all_assets()
    summary = analyzer.generate_summary_report(save_to_file=True, output_dir="results")
    analyzer.create_visualizations("plots")

    # Save results to CSV
    output_dir = Path("results")
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(exist_ok=True)

    summary["basic_summary"].to_csv(csv_dir / "price_movement_metrics.csv", index=False)
    summary["barrier_summary"].to_csv(csv_dir / "triple_barrier_profits.csv", index=False)

    # Save volume analysis results
    if not summary["volume_summary"].empty:
        summary["volume_summary"].to_csv(csv_dir / "volume_analysis.csv", index=False)
        logger.info("Volume analysis results saved to volume_analysis.csv")

    # Save detailed results as JSON
    import json
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a comprehensive results dictionary
    comprehensive_results = {
        "timestamp": timestamp,
        "analysis_summary": {
            "total_assets": len(analyzer.results),
            "total_records": len(analyzer.df) if analyzer.df is not None else 0,
            "date_range": {
                "start": str(analyzer.df.index.min()) if analyzer.df is not None else None,
                "end": str(analyzer.df.index.max()) if analyzer.df is not None else None
            }
        },
        "detailed_results": analyzer.results
    }
    
    json_file = output_dir / f"comprehensive_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_dir}/")
    logger.info(f"CSV files saved to {csv_dir}/")
    logger.info(f"Comprehensive JSON results saved to {json_file}")

if __name__ == "__main__":
    main()
