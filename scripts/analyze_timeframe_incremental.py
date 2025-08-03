#!/usr/bin/env python3
"""
Price Action Timeframe Analysis Script - Incremental Version

This script analyzes historical aggtrades data to determine:
1. Average time for price to move 0.5-1.5% in one direction without reversing by more than 0.1-1%
2. Optimal timeframe for market labeling
3. Frequency analysis for defining best SL/TP levels
4. Price action characteristics for different market conditions

Uses incremental processing to handle large datasets without memory issues.

Usage:
    python scripts/analyze_timeframe_incremental.py --symbol ETHUSDT
    python scripts/analyze_timeframe_incremental.py --symbol BTCUSDT --timeframe 1m
"""

import argparse
import glob
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logging

# Ensure logging is set up
setup_logging()
logger = logging.getLogger(__name__)


def terminal_log(message: str, level: str = "INFO"):
    """Log to both terminal and logger"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {message}", flush=True)
    if level == "INFO":
        logger.info(message)
    elif level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)


class PriceActionAnalyzer:
    """
    Analyzes historical aggtrades data to determine optimal timeframes and SL/TP levels.
    Uses incremental processing to handle large datasets.
    """

    def __init__(self, symbol: str, timeframe: str = "1m"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_cache_dir = "data_cache"
        self.logger = logging.getLogger(__name__)

        # Analysis parameters - User-specified ranges for high leverage trading
        # Target ranges: 0.3% to 0.6% in 0.1% increments
        self.target_ranges = np.array([0.3, 0.4, 0.5, 0.6])

        # Stop ranges: 0.1% to 0.4% in 0.1% increments
        self.stop_ranges = np.array([0.1, 0.2, 0.3, 0.4])

        # Filter combinations to only test realistic risk-reward ratios
        self.valid_combinations = []
        for target in self.target_ranges:
            for stop in self.stop_ranges:
                # Only include combinations where target > stop and risk-reward ratio >= 1.5
                # Also ensure net profit after fees is at least 0.25%
                net_profit = target - 0.08  # Subtract round-trip fees
                if target > stop and (target / stop) >= 1.5 and net_profit >= 0.25:
                    self.valid_combinations.append((target, stop))

        self.round_trip_fee = 0.08  # 0.08% for Binance USDT-M Futures

    def load_aggtrades_data_incremental(
        self,
        test_mode: bool = False,
        days: int = None,
    ) -> pd.DataFrame:
        """
        Load all aggtrades files for the symbol from data_cache directory.
        Processes files incrementally to avoid memory issues.

        Args:
            test_mode: If True, only process first 5 files for testing
            days: Number of days to process (if None, process all files)

        Returns:
            DataFrame with timestamp and price columns
        """
        terminal_log("=" * 50, "INFO")
        terminal_log("📂 STARTING INCREMENTAL DATA LOADING PHASE", "INFO")
        terminal_log("=" * 50, "INFO")
        terminal_log(f"🔍 Looking for aggtrades data for {self.symbol}...", "INFO")

        # Find all aggtrades files for this symbol
        pattern = f"{self.data_cache_dir}/aggtrades_BINANCE_{self.symbol}_*.csv"
        aggtrades_files = glob.glob(pattern)

        if not aggtrades_files:
            terminal_log(f"❌ No aggtrades files found for {self.symbol}", "ERROR")
            terminal_log(f"🔍 Expected pattern: {pattern}", "ERROR")
            terminal_log(f"📁 Current directory: {os.getcwd()}", "ERROR")
            return pd.DataFrame()

        # Sort files by date for consistent processing
        aggtrades_files.sort()

        if test_mode:
            # Limit to first 5 files in test mode
            aggtrades_files = aggtrades_files[:5]
            terminal_log(
                f"🧪 TEST MODE: Processing only first {len(aggtrades_files)} files",
                "INFO",
            )
        elif days is not None and days > 0:
            # Limit to specified number of days
            aggtrades_files = aggtrades_files[:days]
            terminal_log(
                f"📅 DAYS MODE: Processing first {len(aggtrades_files)} days of data",
                "INFO",
            )
        else:
            # Process all files
            terminal_log(
                f"📅 FULL DATASET MODE: Processing all {len(aggtrades_files)} files",
                "INFO",
            )

        terminal_log(f"📦 Found {len(aggtrades_files)} files to process", "INFO")

        # Process files incrementally to avoid memory issues
        all_resampled_data = []
        total_rows = 0
        processed_files = 0

        for file_path in aggtrades_files:
            try:
                processed_files += 1
                terminal_log(f"📄 Loading {os.path.basename(file_path)}...", "INFO")

                # Load single file
                df = pd.read_csv(file_path)
                terminal_log(f"    📊 Raw file loaded: {len(df):,} rows", "INFO")

                # Convert timestamps - handle both millisecond and datetime formats
                try:
                    # First try to convert as milliseconds
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                except (ValueError, TypeError):
                    try:
                        # If that fails, try as datetime string
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                    except (ValueError, TypeError):
                        # If both fail, try to detect the format
                        sample_timestamp = str(df["timestamp"].iloc[0])
                        if sample_timestamp.isdigit():
                            # It's a number, try as milliseconds
                            df["timestamp"] = pd.to_datetime(
                                df["timestamp"].astype(float),
                                unit="ms",
                            )
                        else:
                            # It's a string, try as datetime
                            df["timestamp"] = pd.to_datetime(df["timestamp"])

                valid_timestamps = df["timestamp"].notna().sum()
                terminal_log(
                    f"    ⏰ Timestamps converted: {valid_timestamps:,} valid rows",
                    "INFO",
                )

                # Convert prices
                df["price"] = pd.to_numeric(df["price"], errors="coerce")
                valid_prices = df["price"].notna().sum()
                terminal_log(
                    f"    💰 Prices converted: {valid_prices:,} valid rows",
                    "INFO",
                )

                # Filter valid data
                df = df.dropna(subset=["timestamp", "price"])
                terminal_log(
                    f"    ✅ Loaded {len(df):,} rows from {os.path.basename(file_path)}",
                    "INFO",
                )

                if len(df) > 0:
                    # Resample this file's data immediately
                    resampled = self.resample_to_timeframe(df)
                    if not resampled.empty:
                        all_resampled_data.append(resampled)
                        total_rows += len(df)
                        terminal_log(
                            f"    🔄 Resampled to {len(resampled):,} {self.timeframe} candles",
                            "INFO",
                        )

                # Clear memory
                del df

                terminal_log(
                    f"    📁 Progress: {processed_files}/{len(aggtrades_files)} files processed",
                    "INFO",
                )

            except Exception as e:
                terminal_log(f"    ❌ Error processing {file_path}: {e}", "ERROR")
                continue

        terminal_log(f"📊 Total files processed: {processed_files}", "INFO")
        terminal_log(f"📊 Total rows processed: {total_rows:,}", "INFO")

        if not all_resampled_data:
            terminal_log("❌ No valid data found in any files", "ERROR")
            return pd.DataFrame()

        # Combine all resampled data
        terminal_log("🔗 Combining all resampled data...", "INFO")
        combined_df = pd.concat(all_resampled_data, ignore_index=True)

        # Sort by timestamp
        combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=["timestamp"]).reset_index(
            drop=True,
        )

        terminal_log(
            f"✅ Final combined dataset: {len(combined_df):,} {self.timeframe} candles",
            "INFO",
        )
        terminal_log(
            f"📅 Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}",
            "INFO",
        )

        return combined_df

    def resample_to_timeframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample price data to the target timeframe.

        Args:
            df: DataFrame with timestamp and price columns

        Returns:
            Resampled DataFrame with OHLC data
        """
        if df.empty:
            return pd.DataFrame()

        # Set timestamp as index for resampling
        df_resampled = df.set_index("timestamp")

        # Resample based on timeframe
        if self.timeframe == "1m":
            resampled = df_resampled["price"].resample("1min").ohlc()
        elif self.timeframe == "2m":
            resampled = df_resampled["price"].resample("2min").ohlc()
        elif self.timeframe == "3m":
            resampled = df_resampled["price"].resample("3min").ohlc()
        elif self.timeframe == "4m":
            resampled = df_resampled["price"].resample("4min").ohlc()
        elif self.timeframe == "5m":
            resampled = df_resampled["price"].resample("5min").ohlc()
        elif self.timeframe == "15m":
            resampled = df_resampled["price"].resample("15min").ohlc()
        elif self.timeframe == "1h":
            resampled = df_resampled["price"].resample("1H").ohlc()
        else:
            terminal_log(f"❌ Unsupported timeframe: {self.timeframe}", "ERROR")
            return pd.DataFrame()

        # Reset index to get timestamp as column
        resampled = resampled.reset_index()

        # Remove rows with NaN values (incomplete candles)
        resampled = resampled.dropna()

        return resampled

    def analyze_price_movement(
        self,
        df: pd.DataFrame,
        target_pct: float,
        stop_pct: float,
    ) -> dict:
        """
        Analyze price movements to find successful trades.

        Args:
            df: DataFrame with OHLC data
            target_pct: Target percentage for profit
            stop_pct: Stop loss percentage

        Returns:
            Dictionary with analysis results
        """
        if df.empty:
            return {
                "total_events": 0,
                "successful_events": 0,
                "success_rate": 0.0,
                "avg_duration": 0.0,
                "long_events": 0,
                "short_events": 0,
                "events": [],
            }

        events = []
        total_events = 0
        successful_events = 0
        long_events = 0
        short_events = 0
        total_duration = 0

        # Time barrier to prevent infinite loops (24 hours)
        time_barrier = pd.Timedelta(hours=24)

        terminal_log(f"📊 Starting analysis on {len(df):,} data points...", "INFO")
        terminal_log(f"⏰ Time barrier set to {time_barrier}", "INFO")

        # Process each price point
        for i in range(len(df) - 1):
            if i % max(1, len(df) // 20) == 0:  # Progress update every 5%
                progress = (i / len(df)) * 100
                terminal_log(
                    f"    📊 Progress: {i:,}/{len(df):,} points ({progress:.1f}%)",
                    "INFO",
                )

            start_price = df.iloc[i]["close"]
            start_time = df.iloc[i]["timestamp"]

            # Calculate target and stop prices
            long_target = start_price * (1 + target_pct / 100)
            long_stop = start_price * (1 - stop_pct / 100)
            short_target = start_price * (1 - target_pct / 100)
            short_stop = start_price * (1 + stop_pct / 100)

            # Look ahead for price movement
            for j in range(i + 1, len(df)):
                current_price = df.iloc[j]["close"]
                current_time = df.iloc[j]["timestamp"]
                duration = current_time - start_time

                # Check if time barrier exceeded
                if duration > time_barrier:
                    break

                # Check long position
                if current_price >= long_target:
                    events.append(
                        {
                            "type": "long",
                            "start_time": start_time,
                            "end_time": current_time,
                            "duration": duration,
                            "start_price": start_price,
                            "end_price": current_price,
                            "success": True,
                            "target_pct": target_pct,
                            "stop_pct": stop_pct,
                        },
                    )
                    successful_events += 1
                    long_events += 1
                    total_duration += (
                        duration.total_seconds() / 60
                    )  # Convert to minutes
                    break
                if current_price <= long_stop:
                    events.append(
                        {
                            "type": "long",
                            "start_time": start_time,
                            "end_time": current_time,
                            "duration": duration,
                            "start_price": start_price,
                            "end_price": current_price,
                            "success": False,
                            "target_pct": target_pct,
                            "stop_pct": stop_pct,
                        },
                    )
                    long_events += 1
                    total_duration += duration.total_seconds() / 60
                    break

                # Check short position
                if current_price <= short_target:
                    events.append(
                        {
                            "type": "short",
                            "start_time": start_time,
                            "end_time": current_time,
                            "duration": duration,
                            "start_price": start_price,
                            "end_price": current_price,
                            "success": True,
                            "target_pct": target_pct,
                            "stop_pct": stop_pct,
                        },
                    )
                    successful_events += 1
                    short_events += 1
                    total_duration += duration.total_seconds() / 60
                    break
                if current_price >= short_stop:
                    events.append(
                        {
                            "type": "short",
                            "start_time": start_time,
                            "end_time": current_time,
                            "duration": duration,
                            "start_price": start_price,
                            "end_price": current_price,
                            "success": False,
                            "target_pct": target_pct,
                            "stop_pct": stop_pct,
                        },
                    )
                    short_events += 1
                    total_duration += duration.total_seconds() / 60
                    break

            total_events += 1

        # Calculate metrics
        success_rate = (
            (successful_events / total_events * 100) if total_events > 0 else 0
        )
        avg_duration = total_duration / total_events if total_events > 0 else 0

        terminal_log(f"✅ Analysis completed: {total_events} events found", "INFO")
        terminal_log(f"📊 Success rate: {success_rate:.2f}%", "INFO")
        terminal_log(
            f"📈 Directional breakdown: {long_events} long vs {short_events} short",
            "INFO",
        )
        terminal_log(f"⏱️  Average duration: {avg_duration:.1f} minutes", "INFO")

        return {
            "total_events": total_events,
            "successful_events": successful_events,
            "success_rate": success_rate,
            "avg_duration": avg_duration,
            "long_events": long_events,
            "short_events": short_events,
            "events": events,
        }

    def calculate_frequency_score(self, result: dict) -> float:
        """
        Calculate a frequency score based on the number of events.

        Args:
            result: Analysis result dictionary

        Returns:
            Frequency score (0-100)
        """
        total_events = result["total_events"]

        # Score based on number of events (more events = higher score)
        if total_events == 0:
            return 0

        # Normalize to 0-100 scale
        # Consider 1000+ events as excellent (100 points)
        # 100-999 events as good (50-99 points)
        # 10-99 events as fair (10-49 points)
        # <10 events as poor (0-9 points)

        if total_events >= 1000:
            return 100
        if total_events >= 100:
            return 50 + (total_events - 100) / 9  # 50-99 points
        if total_events >= 10:
            return 10 + (total_events - 10) / 2.25  # 10-49 points
        return total_events  # 0-9 points

    def run_comprehensive_analysis(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run comprehensive analysis on all valid combinations.

        Args:
            df: Resampled DataFrame

        Returns:
            Tuple of (display_df, score_df)
        """
        terminal_log("🔍 Starting comprehensive analysis...", "INFO")
        terminal_log(f"📊 Testing {len(self.valid_combinations)} combinations", "INFO")

        results = []
        score_data = []

        for i, (target, stop) in enumerate(self.valid_combinations, 1):
            terminal_log(
                f"🔍 Testing combination {i}/{len(self.valid_combinations)}: Target {target}%, Stop {stop}%",
                "INFO",
            )

            start_time = datetime.now()
            result = self.analyze_price_movement(df, target, stop)
            duration = (datetime.now() - start_time).total_seconds()

            # Calculate additional metrics
            risk_reward_ratio = target / stop if stop > 0 else 0
            net_profit_pct = target - self.round_trip_fee
            frequency_score = self.calculate_frequency_score(result)

            # Store detailed results
            results.append(
                {
                    "target_pct": target,
                    "stop_pct": stop,
                    "risk_reward_ratio": risk_reward_ratio,
                    "total_events": result["total_events"],
                    "successful_events": result["successful_events"],
                    "success_rate": result["success_rate"],
                    "avg_duration_minutes": result["avg_duration"],
                    "long_events": result["long_events"],
                    "short_events": result["short_events"],
                    "frequency_score": frequency_score,
                    "net_profit_pct": net_profit_pct,
                    "analysis_time_seconds": duration,
                },
            )

            # Store scoring data
            score_data.append(
                {
                    "target_pct": target,
                    "stop_pct": stop,
                    "success_rate": result["success_rate"],
                    "frequency_score": frequency_score,
                    "avg_duration": result["avg_duration"],
                    "risk_reward_ratio": risk_reward_ratio,
                    "net_profit_pct": net_profit_pct,
                    "total_score": (
                        result["success_rate"] * 0.4
                        + frequency_score * 0.3
                        + (100 - result["avg_duration"] / 10) * 0.3
                    ),  # Lower duration = higher score
                },
            )

            terminal_log(f"✅ Combination {i} completed in {duration:.1f}s", "INFO")

        # Create DataFrames
        display_df = pd.DataFrame(results)
        score_df = pd.DataFrame(score_data)

        terminal_log("✅ Comprehensive analysis completed", "INFO")
        terminal_log(f"📊 Results generated for {len(results)} combinations", "INFO")

        return display_df, score_df

    def find_optimal_parameters(self, score_df: pd.DataFrame) -> dict:
        """
        Find optimal parameters based on scoring.

        Args:
            score_df: DataFrame with scoring data

        Returns:
            Dictionary with optimal parameters
        """
        if score_df.empty:
            return {}

        # Find the best combination based on total score
        best_idx = score_df["total_score"].idxmax()
        best_row = score_df.loc[best_idx]

        optimal_params = {
            "optimal_target": best_row["target_pct"],
            "optimal_stop": best_row["stop_pct"],
            "optimal_risk_reward": best_row["risk_reward_ratio"],
            "optimal_success_rate": best_row["success_rate"],
            "optimal_frequency_score": best_row["frequency_score"],
            "optimal_avg_duration": best_row["avg_duration"],
            "optimal_net_profit": best_row["net_profit_pct"],
            "optimal_total_score": best_row["total_score"],
        }

        return optimal_params

    def generate_recommendations(
        self,
        optimal_params: dict,
        display_df: pd.DataFrame,
    ) -> dict:
        """
        Generate trading recommendations based on analysis.

        Args:
            optimal_params: Optimal parameters dictionary
            display_df: Display DataFrame

        Returns:
            Dictionary with recommendations
        """
        if not optimal_params:
            return {}

        target = optimal_params["optimal_target"]
        stop = optimal_params["optimal_stop"]
        success_rate = optimal_params["optimal_success_rate"]
        frequency = optimal_params["optimal_frequency_score"]

        recommendations = {
            "primary_strategy": {
                "target_pct": target,
                "stop_pct": stop,
                "risk_reward_ratio": target / stop,
                "expected_success_rate": success_rate,
                "frequency_score": frequency,
            },
            "risk_management": {
                "max_position_size": "1-2% of portfolio per trade",
                "max_daily_loss": "5% of portfolio",
                "correlation_limit": "Max 3 correlated positions",
            },
            "execution_guidelines": {
                "entry_timing": "Enter on strong momentum confirmation",
                "exit_strategy": "Use trailing stops for winners",
                "position_sizing": "Scale in/out based on conviction",
            },
            "market_conditions": {
                "best_timeframes": [self.timeframe],
                "volatility_preference": "Medium to high volatility periods",
                "trend_following": "Strong directional moves preferred",
            },
        }

        return recommendations

    def save_results(
        self,
        display_df: pd.DataFrame,
        score_df: pd.DataFrame,
        optimal_params: dict,
        recommendations: dict,
        df_resampled: pd.DataFrame = None,
    ) -> None:
        """
        Save analysis results to files.

        Args:
            display_df: Display DataFrame
            score_df: Score DataFrame
            optimal_params: Optimal parameters
            recommendations: Recommendations
            df_resampled: Resampled data (optional)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create analysis_results directory if it doesn't exist
        os.makedirs("analysis_results", exist_ok=True)

        # Save main analysis results
        main_filename = (
            f"analysis_results/timeframe_analysis_{self.symbol}_{timestamp}.csv"
        )
        display_df.to_csv(main_filename, index=False)
        terminal_log(f"💾 Main analysis saved to: {main_filename}", "INFO")

        # Save scoring details
        score_filename = (
            f"analysis_results/scoring_details_{self.symbol}_{timestamp}.csv"
        )
        score_df.to_csv(score_filename, index=False)
        terminal_log(f"💾 Scoring details saved to: {score_filename}", "INFO")

        # Save summary report
        summary_filename = f"analysis_results/summary_{self.symbol}_{timestamp}.txt"
        with open(summary_filename, "w") as f:
            f.write("PRICE ACTION TIMEFRAME ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Symbol: {self.symbol}\n")
            f.write(f"Timeframe: {self.timeframe}\n")
            f.write(
                f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
            )

            if df_resampled is not None:
                f.write(
                    f"Data period: {df_resampled['timestamp'].min()} to {df_resampled['timestamp'].max()}\n",
                )
                f.write(f"Total candles: {len(df_resampled):,}\n\n")

            f.write("OPTIMAL PARAMETERS:\n")
            f.write("-" * 20 + "\n")
            for key, value in optimal_params.items():
                f.write(f"{key}: {value}\n")

            f.write("\nRECOMMENDATIONS:\n")
            f.write("-" * 15 + "\n")
            for category, items in recommendations.items():
                f.write(f"\n{category.upper()}:\n")
                for key, value in items.items():
                    f.write(f"  {key}: {value}\n")

        terminal_log(f"💾 Summary report saved to: {summary_filename}", "INFO")

    def print_summary(
        self,
        display_df: pd.DataFrame,
        optimal_params: dict,
        recommendations: dict,
    ) -> None:
        """
        Print analysis summary to console.

        Args:
            display_df: Display DataFrame
            optimal_params: Optimal parameters
            recommendations: Recommendations
        """
        terminal_log("=" * 60, "INFO")
        terminal_log("📋 ANALYSIS SUMMARY", "INFO")
        terminal_log("=" * 60, "INFO")

        if optimal_params:
            terminal_log("🎯 OPTIMAL PARAMETERS:", "INFO")
            terminal_log(
                f"   Target: {optimal_params.get('optimal_target', 'N/A')}%",
                "INFO",
            )
            terminal_log(
                f"   Stop: {optimal_params.get('optimal_stop', 'N/A')}%",
                "INFO",
            )
            terminal_log(
                f"   Risk-Reward: {optimal_params.get('optimal_risk_reward', 'N/A'):.2f}:1",
                "INFO",
            )
            terminal_log(
                f"   Success Rate: {optimal_params.get('optimal_success_rate', 'N/A'):.1f}%",
                "INFO",
            )
            terminal_log(
                f"   Frequency Score: {optimal_params.get('optimal_frequency_score', 'N/A'):.1f}",
                "INFO",
            )
            terminal_log(
                f"   Avg Duration: {optimal_params.get('optimal_avg_duration', 'N/A'):.1f} minutes",
                "INFO",
            )

        if not display_df.empty:
            terminal_log("\n📊 TOP 3 COMBINATIONS:", "INFO")
            top_3 = display_df.nlargest(3, "success_rate")
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                terminal_log(
                    f"   {i}. Target {row['target_pct']}%, Stop {row['stop_pct']}%: "
                    f"{row['success_rate']:.1f}% success rate",
                    "INFO",
                )

        terminal_log("=" * 60, "INFO")


def main():
    """Main function to run the price action analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze price action timeframes for optimal SL/TP levels (Incremental Version)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Trading symbol (e.g., ETHUSDT, BTCUSDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1m",
        help="Timeframe for analysis (default: 1m)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with limited data",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=10,
        help="Number of days of data to process (default: 10)",
    )

    args = parser.parse_args()

    # Enhanced startup logging
    start_time = datetime.now()
    terminal_log("=" * 60, "INFO")
    terminal_log("🚀 PRICE ACTION TIMEFRAME ANALYSIS (INCREMENTAL)", "INFO")
    terminal_log("=" * 60, "INFO")
    terminal_log(f"📊 Symbol: {args.symbol}", "INFO")
    terminal_log(f"⏰ Timeframe: {args.timeframe}", "INFO")
    terminal_log(f"🧪 Test Mode: {'Yes' if args.test_mode else 'No'}", "INFO")
    terminal_log(f"📅 Days to process: {args.days}", "INFO")
    terminal_log(f"🕐 Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}", "INFO")
    terminal_log("=" * 60, "INFO")

    try:
        # Initialize analyzer
        analyzer = PriceActionAnalyzer(args.symbol, args.timeframe)
        terminal_log(
            f"✅ Analyzer initialized with {len(analyzer.valid_combinations)} valid combinations",
            "INFO",
        )

        # Load data incrementally
        terminal_log("📂 Loading historical data incrementally...", "INFO")
        df = analyzer.load_aggtrades_data_incremental(
            test_mode=args.test_mode,
            days=args.days,
        )

        if df.empty:
            terminal_log("❌ No data loaded. Exiting.", "ERROR")
            return None

        terminal_log(f"✅ Data loaded: {len(df):,} rows", "INFO")
        terminal_log(
            f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}",
            "INFO",
        )

        # Run analysis
        terminal_log("🔍 Running comprehensive analysis...", "INFO")
        display_df, score_df = analyzer.run_comprehensive_analysis(df)

        if display_df.empty:
            terminal_log("❌ Analysis failed. No results generated.", "ERROR")
            return None

        # Find optimal parameters
        terminal_log("🎯 Finding optimal parameters...", "INFO")
        optimal_params = analyzer.find_optimal_parameters(score_df)

        # Generate recommendations
        terminal_log("💡 Generating recommendations...", "INFO")
        recommendations = analyzer.generate_recommendations(optimal_params, display_df)

        # Save results
        terminal_log("💾 Saving results...", "INFO")
        analyzer.save_results(
            display_df,
            score_df,
            optimal_params,
            recommendations,
            df,
        )

        # Print summary
        terminal_log("📋 Printing summary...", "INFO")
        analyzer.print_summary(display_df, optimal_params, recommendations)

        # Final summary
        total_time = (datetime.now() - start_time).total_seconds()
        terminal_log("=" * 60, "INFO")
        terminal_log("🎉 ANALYSIS COMPLETED SUCCESSFULLY!", "INFO")
        terminal_log(f"⏱️  Total time: {total_time:.1f} seconds", "INFO")
        terminal_log(f"📊 Combinations tested: {len(display_df)}", "INFO")
        terminal_log(
            f"🎯 Optimal target: {optimal_params.get('optimal_target', 'N/A')}%",
            "INFO",
        )
        terminal_log(
            f"🛑 Optimal stop: {optimal_params.get('optimal_stop', 'N/A')}%",
            "INFO",
        )
        terminal_log("=" * 60, "INFO")

    except KeyboardInterrupt:
        terminal_log("🛑 Analysis stopped by user", "INFO")
    except Exception as e:
        terminal_log(f"❌ Analysis failed: {e}", "ERROR")
        import traceback

        terminal_log(f"📋 Traceback: {traceback.format_exc()}", "ERROR")
        return 1

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        terminal_log("🛑 Analysis stopped by user", "INFO")
        sys.exit(0)
    except Exception as e:
        terminal_log(f"❌ Fatal error: {e}", "ERROR")
        sys.exit(1)
