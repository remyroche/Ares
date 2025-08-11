#!/usr/bin/env python3
"""
Price Action Timeframe Analysis Script

This script analyzes historical aggtrades data to determine:
1. Average time for price to move 0.5-1.5% in one direction without reversing by more than 0.1-1%
2. Optimal timeframe for market labeling
3. Frequency analysis for defining best SL/TP levels
4. Price action characteristics for different market conditions

Usage:
    python scripts/analyze_timeframe.py --symbol ETHUSDT
    python scripts/analyze_timeframe.py --symbol BTCUSDT --timeframe 1m
"""

import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import ensure_logging_setup, get_logger

# Ensure logging is set up
ensure_logging_setup()
logger = get_logger(__name__)


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
    """

    def __init__(self, symbol: str, timeframe: str = "1m"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_cache_dir = "data_cache"
        self.logger = get_logger(__name__)

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

    def load_aggtrades_data(self, test_mode: bool = False) -> pd.DataFrame:
        """
        Load all aggtrades files for the symbol from data_cache directory.
        Now processes files incrementally to avoid memory issues.

        Args:
            test_mode: If True, only process first 5 files for testing

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

                # Convert timestamps
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
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
                if len(all_resampled_data) > 0:
                    del all_resampled_data[-1]  # Remove from memory after processing

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
        Resample data to the specified timeframe.

        Args:
            df: DataFrame with timestamp and price columns

        Returns:
            Resampled DataFrame
        """
        terminal_log("=" * 50, "INFO")
        terminal_log("🔄 STARTING RESAMPLING PHASE", "INFO")
        terminal_log("=" * 50, "INFO")
        terminal_log(f"📊 Input data: {len(df):,} rows", "INFO")
        terminal_log(
            f"📅 Input date range: {df['timestamp'].min()} to {df['timestamp'].max()}",
            "INFO",
        )
        terminal_log(f"⏰ Target timeframe: {self.timeframe}", "INFO")

        # Set timestamp as index for resampling
        terminal_log("🔧 Setting timestamp as index...", "INFO")
        df_resampled = df.set_index("timestamp")
        terminal_log("✅ Index set successfully", "INFO")

        # Resample to the specified timeframe
        terminal_log("🔄 Starting resampling process...", "INFO")
        if self.timeframe == "1m":
            resampled = df_resampled["price"].resample("1min").ohlc()
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

        terminal_log(f"✅ Resampling completed: {len(resampled):,} candles", "INFO")

        # Calculate OHLC from the resampled data
        terminal_log("📊 Calculating OHLC data...", "INFO")
        ohlc_df = resampled.copy()
        ohlc_df.columns = ["open", "high", "low", "close"]

        # Use close price for analysis
        final_df = pd.DataFrame({"timestamp": ohlc_df.index, "price": ohlc_df["close"]})

        # Remove any NaN values
        final_df = final_df.dropna()

        terminal_log("✅ Final resampled data:", "INFO")
        terminal_log(f"   📊 Total candles: {len(final_df):,}")
        terminal_log(
            f"   📅 Date range: {final_df['timestamp'].min()} to {final_df['timestamp'].max()}",
            "INFO",
        )
        terminal_log(
            f"   💰 Price range: ${final_df['price'].min():.2f} to ${final_df['price'].max():.2f}",
            "INFO",
        )
        terminal_log("=" * 50, "INFO")

        return final_df

    def analyze_price_movement(
        self,
        df: pd.DataFrame,
        target_pct: float,
        stop_pct: float,
    ) -> dict:
        """
        Analyze price movements for a given target and stop percentage.

        Args:
            df: DataFrame with timestamp and price columns
            target_pct: Target percentage movement
            stop_pct: Stop loss percentage

        Returns:
            Dictionary with analysis results
        """
        terminal_log(
            f"🔍 Analyzing: Target {target_pct:.1f}%, Stop {stop_pct:.1f}%",
            "INFO",
        )

        if df.empty:
            terminal_log("❌ Cannot analyze empty DataFrame", "ERROR")
            return {
                "avg_duration_seconds": np.nan,
                "occurrences": 0,
                "total_attempts": 0,
                "success_rate": 0.0,
                "durations": [],
            }

        terminal_log(f"📊 Starting analysis on {len(df):,} data points...", "INFO")

        durations = []
        occurrences = 0
        total_attempts = 0
        long_successes = 0
        short_successes = 0

        prices = df["price"].values
        timestamps = df["timestamp"].values

        # Maximum time barrier (e.g., 24 hours)
        max_time_seconds = 24 * 3600  # 24 hours in seconds
        terminal_log(
            f"⏰ Time barrier set to {max_time_seconds/3600:.1f} hours",
            "INFO",
        )

        # Process each price point
        terminal_log("🔄 Processing price points...", "INFO")
        progress_interval = max(1, len(df) // 20)  # Show progress every 5%
        for i in range(len(df) - 1):
            if i % progress_interval == 0:
                progress_pct = (i / len(df)) * 100
                terminal_log(
                    f"    📊 Progress: {i:,}/{len(df):,} points ({progress_pct:.1f}%)",
                    "INFO",
                )
            start_price = prices[i]
            start_time = timestamps[i]

            # Calculate target and stop prices
            up_target = start_price * (1 + target_pct / 100)
            down_target = start_price * (1 - target_pct / 100)
            up_stop = start_price * (1 + stop_pct / 100)
            down_stop = start_price * (1 - stop_pct / 100)

            # Look ahead to find if we hit target before stop
            for j in range(i + 1, len(df)):
                current_price = prices[j]
                current_time = timestamps[j]

                # Check time barrier first
                time_diff = (
                    (current_time - start_time).astype("timedelta64[s]").astype(float)
                )
                if time_diff > max_time_seconds:
                    break  # Time barrier hit

                # Check if we hit target (up or down) - check targets BEFORE stops
                if current_price >= up_target:
                    # We hit the up target without hitting the down stop (LONG SUCCESS)
                    # Only check if we hit the down stop (for long trades)
                    hit_down_stop = False
                    for k in range(i + 1, j + 1):
                        if prices[k] <= down_stop:
                            hit_down_stop = True
                            break

                    if not hit_down_stop:
                        duration = time_diff
                        durations.append(duration)
                        occurrences += 1
                        long_successes += 1
                        break

                elif current_price <= down_target:
                    # We hit the down target without hitting the up stop (SHORT SUCCESS)
                    # Only check if we hit the up stop (for short trades)
                    hit_up_stop = False
                    for k in range(i + 1, j + 1):
                        if prices[k] >= up_stop:
                            hit_up_stop = True
                            break

                    if not hit_up_stop:
                        duration = time_diff
                        durations.append(duration)
                        occurrences += 1
                        short_successes += 1
                        break

            total_attempts += 1

        # Calculate results
        avg_duration = np.mean(durations) if durations else np.nan
        success_rate = occurrences / total_attempts if total_attempts > 0 else 0

        terminal_log(f"✅ Analysis completed: {occurrences} events found", "INFO")
        terminal_log(f"📊 Success rate: {success_rate:.2%}", "INFO")
        if occurrences > 0:
            long_pct = (long_successes / occurrences) * 100
            short_pct = (short_successes / occurrences) * 100
            terminal_log(
                f"📈 Directional breakdown: {long_successes} long ({long_pct:.1f}%) vs {short_successes} short ({short_pct:.1f}%)",
                "INFO",
            )
        if not np.isnan(avg_duration):
            terminal_log(f"⏱️  Average duration: {avg_duration/60:.1f} minutes", "INFO")

        return {
            "avg_duration_seconds": avg_duration,
            "occurrences": occurrences,
            "total_attempts": total_attempts,
            "success_rate": success_rate,
            "durations": durations,
            "long_successes": long_successes,
            "short_successes": short_successes,
        }

    def calculate_frequency_score(self, result: dict) -> float:
        """
        Calculate a frequency score based on the analysis results.

        Args:
            result: Analysis result dictionary

        Returns:
            Frequency score (higher is better)
        """
        if result["occurrences"] == 0:
            return 0.0

        # Score based on frequency and consistency
        frequency = result["occurrences"] / result["total_attempts"]
        avg_duration = result["avg_duration_seconds"]

        # Prefer more frequent events with reasonable duration
        if np.isnan(avg_duration) or avg_duration <= 0:
            return 0.0

        # Convert to minutes and prefer events that take 5-120 minutes
        duration_minutes = avg_duration / 60
        duration_score = 1.0 / (
            1.0 + abs(duration_minutes - 30) / 30
        )  # Peak at 30 minutes

        # Risk-reward ratio bonus
        target_pct = result["target_pct"]
        stop_pct = result["stop_pct"]
        risk_reward_ratio = target_pct / stop_pct

        # Bonus for better risk-reward ratios (2:1 or better gets bonus)
        rr_bonus = risk_reward_ratio / 2.0  # Remove the cap, let it scale naturally

        # Combine frequency, duration, and risk-reward scores
        return frequency * duration_score * rr_bonus

    def run_comprehensive_analysis(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run comprehensive analysis across all target and stop ranges.

        Args:
            df: DataFrame with timestamp and price columns

        Returns:
            Tuple of (display_df, score_df) DataFrames
        """
        start_time = datetime.now()
        terminal_log("🚀 Starting comprehensive analysis...", "INFO")
        terminal_log(f"📊 Testing {len(self.valid_combinations)} combinations", "INFO")
        terminal_log(f"📈 Data points: {len(df):,}")
        terminal_log(
            f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}",
            "INFO",
        )

        results = []
        scores = []

        # Analyze all combinations
        for i, (target_pct, stop_pct) in enumerate(self.valid_combinations, 1):
            combination_start = datetime.now()
            terminal_log(
                f"🔍 Testing combination {i}/{len(self.valid_combinations)}: Target {target_pct}%, Stop {stop_pct}%",
                "INFO",
            )

            try:
                result = self.analyze_price_movement(df, target_pct, stop_pct)

                # Calculate additional metrics
                result["target_pct"] = target_pct
                result["stop_pct"] = stop_pct
                result["risk_reward_ratio"] = target_pct / stop_pct
                result["net_profit_after_fees"] = target_pct - 0.08

                # Calculate frequency score
                frequency_score = self.calculate_frequency_score(result)
                result["frequency_score"] = frequency_score

                results.append(result)
                scores.append(
                    {
                        "target_pct": target_pct,
                        "stop_pct": stop_pct,
                        "frequency_score": frequency_score,
                        "risk_reward_ratio": result["risk_reward_ratio"],
                        "net_profit_after_fees": result["net_profit_after_fees"],
                        "occurrences": result["occurrences"],
                        "avg_duration_minutes": result["avg_duration_seconds"] / 60
                        if result["avg_duration_seconds"] > 0
                        else 0,
                    },
                )

                combination_time = (datetime.now() - combination_start).total_seconds()
                terminal_log(
                    f"✅ Combination {i} completed in {combination_time:.1f}s",
                    "INFO",
                )

            except Exception as e:
                terminal_log(f"❌ Error in combination {i}: {e}", "ERROR")
                continue

        # Create DataFrames
        display_df = pd.DataFrame(results)
        score_df = pd.DataFrame(scores)

        total_time = (datetime.now() - start_time).total_seconds()
        terminal_log(f"🎉 Analysis completed in {total_time:.1f} seconds", "INFO")
        terminal_log(f"📊 Processed {len(results)} combinations successfully", "INFO")

        return display_df, score_df

    def find_optimal_parameters(self, score_df: pd.DataFrame) -> dict:
        """
        Find optimal parameters based on scoring.

        Args:
            score_df: DataFrame with scoring results

        Returns:
            Dictionary with optimal parameters
        """
        if score_df.empty:
            terminal_log("⚠️  No valid scores found", "WARNING")
            return {
                "optimal_target": "N/A",
                "optimal_stop": "N/A",
                "optimal_score": 0.0,
                "total_combinations": 0,
            }

        # Find the combination with the highest frequency score
        best_idx = score_df["frequency_score"].idxmax()
        best_row = score_df.loc[best_idx]

        optimal_params = {
            "optimal_target": f"{best_row['target_pct']:.1f}%",
            "optimal_stop": f"{best_row['stop_pct']:.1f}%",
            "optimal_score": best_row["frequency_score"],
            "risk_reward_ratio": best_row["risk_reward_ratio"],
            "net_profit_after_fees": f"{best_row['net_profit_after_fees']:.2f}%",
            "occurrences": best_row["occurrences"],
            "avg_duration_minutes": f"{best_row['avg_duration_minutes']:.1f}",
            "total_combinations": len(score_df),
        }

        terminal_log("🎯 Optimal parameters found:", "INFO")
        terminal_log(f"   Target: {optimal_params['optimal_target']}", "INFO")
        terminal_log(f"   Stop: {optimal_params['optimal_stop']}", "INFO")
        terminal_log(f"   Score: {optimal_params['optimal_score']:.4f}", "INFO")
        terminal_log(
            f"   Risk-Reward: {optimal_params['risk_reward_ratio']:.2f}:1",
            "INFO",
        )
        terminal_log(
            f"   Net Profit: {optimal_params['net_profit_after_fees']}",
            "INFO",
        )

        return optimal_params

    def generate_recommendations(
        self,
        optimal_params: dict,
        display_df: pd.DataFrame,
    ) -> dict:
        """
        Generate trading recommendations based on analysis results.

        Args:
            optimal_params: Dictionary with optimal parameters
            display_df: DataFrame with analysis results

        Returns:
            Dictionary with recommendations
        """
        if not optimal_params or optimal_params.get("optimal_target") == "N/A":
            return {
                "strategy": "No valid strategy found",
                "risk_level": "Unknown",
                "timeframe": self.timeframe,
                "notes": "Analysis did not find profitable combinations",
            }

        # Extract target and stop from optimal parameters
        target_str = optimal_params.get("optimal_target", "N/A")
        stop_str = optimal_params.get("optimal_stop", "N/A")

        # Parse the percentage values
        try:
            target_pct = float(target_str.replace("%", ""))
            float(stop_str.replace("%", ""))
        except (ValueError, AttributeError):
            target_pct = 0.5  # Default values

        risk_reward = optimal_params.get("risk_reward_ratio", 2.0)
        net_profit = optimal_params.get("net_profit_after_fees", "0.00%")

        # Determine strategy type
        if target_pct <= 0.3:
            strategy_type = "Scalping"
            risk_level = "Low"
        elif target_pct <= 0.5:
            strategy_type = "Day Trading"
            risk_level = "Medium"
        else:
            strategy_type = "Swing Trading"
            risk_level = "High"

        # Generate recommendations
        recommendations = {
            "strategy": f"{strategy_type} with {target_str} target and {stop_str} stop",
            "risk_level": risk_level,
            "timeframe": self.timeframe,
            "risk_reward_ratio": f"{risk_reward:.2f}:1",
            "net_profit_after_fees": net_profit,
            "expected_frequency": f"{optimal_params.get('occurrences', 0)} events",
            "avg_duration": f"{optimal_params.get('avg_duration_minutes', '0')} minutes",
            "notes": f"Based on analysis of {optimal_params.get('total_combinations', 0)} combinations",
        }

        terminal_log("💡 Recommendations generated:", "INFO")
        terminal_log(f"   Strategy: {recommendations['strategy']}", "INFO")
        terminal_log(f"   Risk Level: {recommendations['risk_level']}", "INFO")
        terminal_log(f"   Risk-Reward: {recommendations['risk_reward_ratio']}", "INFO")
        terminal_log(
            f"   Net Profit: {recommendations['net_profit_after_fees']}",
            "INFO",
        )

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
        Save analysis results to CSV files with enhanced logging.

        Args:
            display_df: DataFrame with display results
            score_df: DataFrame with scoring results
            optimal_params: Dictionary with optimal parameters
            recommendations: Dictionary with recommendations
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create results directory if it doesn't exist
        results_dir = Path("analysis_results")
        results_dir.mkdir(exist_ok=True)

        # Save main results CSV
        main_results_file = (
            results_dir / f"timeframe_analysis_{self.symbol}_{timestamp}.csv"
        )
        display_df.to_csv(main_results_file, index=False)
        terminal_log(f"✅ Main results saved to: {main_results_file}", "INFO")

        # Save detailed scoring CSV
        scoring_file = results_dir / f"scoring_details_{self.symbol}_{timestamp}.csv"
        score_df.to_csv(scoring_file, index=False)
        terminal_log(f"✅ Scoring details saved to: {scoring_file}", "INFO")

        # Save optimal parameters and recommendations
        summary_file = results_dir / f"summary_{self.symbol}_{timestamp}.txt"
        with open(summary_file, "w") as f:
            f.write(f"Timeframe Analysis Summary for {self.symbol}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Timeframe: {self.timeframe}\n\n")

            f.write("OPTIMAL PARAMETERS:\n")
            f.write("=" * 50 + "\n")
            f.writelines(f"{key}: {value}\n" for key, value in optimal_params.items())

            f.write("\nRECOMMENDATIONS:\n")
            f.write("=" * 50 + "\n")
            f.writelines(f"{key}: {value}\n" for key, value in recommendations.items())

            f.write(f"\nTotal combinations tested: {len(display_df)}\n")
            if df_resampled is not None:
                f.write(
                    f"Data period: {df_resampled['timestamp'].min()} to {df_resampled['timestamp'].max()}\n",
                )

        terminal_log(f"✅ Summary saved to: {summary_file}", "INFO")

        # Print file locations
        terminal_log("📁 Results saved to:", "INFO")
        terminal_log(f"   📊 Main results: {main_results_file}", "INFO")
        terminal_log(f"   📈 Scoring details: {scoring_file}", "INFO")
        terminal_log(f"   📋 Summary: {summary_file}", "INFO")

    def print_summary(
        self,
        display_df: pd.DataFrame,
        optimal_params: dict,
        recommendations: dict,
    ) -> None:
        """
        Print a summary of the analysis results.

        Args:
            display_df: DataFrame with display results
            optimal_params: Dictionary with optimal parameters
            recommendations: Dictionary with recommendations
        """
        print("\n" + "=" * 80)
        print("PRICE ACTION ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"Symbol: {self.symbol}")
        print(f"Timeframe: {self.timeframe}")
        print(f"Data points analyzed: {len(display_df)}")
        print()

        print("Optimal Parameters:")
        print(f"  Target: {optimal_params.get('optimal_target', 'N/A')}")
        print(f"  Stop: {optimal_params.get('optimal_stop', 'N/A')}")
        print(f"  Score: {optimal_params.get('optimal_score', 0):.4f}")
        print(f"  Risk-Reward: {optimal_params.get('risk_reward_ratio', 0):.2f}:1")
        print(f"  Net Profit: {optimal_params.get('net_profit_after_fees', 'N/A')}")
        print()

        print("Recommendations:")
        print(f"  Strategy: {recommendations.get('strategy', 'N/A')}")
        print(f"  Risk Level: {recommendations.get('risk_level', 'N/A')}")
        print(
            f"  Expected Frequency: {recommendations.get('expected_frequency', 'N/A')}",
        )
        print(f"  Average Duration: {recommendations.get('avg_duration', 'N/A')}")
        print()

        if not display_df.empty:
            print("Top 3 Combinations:")
            for i, row in display_df.head(3).iterrows():
                print(
                    f"  {i+1}. Target {row['target_pct']:.1f}%, Stop {row['stop_pct']:.1f}% - Score: {row['frequency_score']:.4f}",
                )
        print("=" * 80)


def main():
    """Main function to run the price action analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze price action timeframes for optimal SL/TP levels",
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

    args = parser.parse_args()

    # Enhanced startup logging
    start_time = datetime.now()
    terminal_log("=" * 60, "INFO")
    terminal_log("🚀 PRICE ACTION TIMEFRAME ANALYSIS", "INFO")
    terminal_log("=" * 60, "INFO")
    terminal_log(f"📊 Symbol: {args.symbol}", "INFO")
    terminal_log(f"⏰ Timeframe: {args.timeframe}", "INFO")
    terminal_log(f"🧪 Test Mode: {'Yes' if args.test_mode else 'No'}", "INFO")
    terminal_log(f"🕐 Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}", "INFO")
    terminal_log("=" * 60, "INFO")

    try:
        # Initialize analyzer
        analyzer = PriceActionAnalyzer(args.symbol, args.timeframe)
        terminal_log(
            f"✅ Analyzer initialized with {len(analyzer.valid_combinations)} valid combinations",
            "INFO",
        )

        # Load data
        terminal_log("📂 Loading historical data...", "INFO")
        df = analyzer.load_aggtrades_data(test_mode=args.test_mode)

        if df.empty:
            terminal_log("❌ No data loaded. Exiting.", "ERROR")
            return None

        terminal_log(f"✅ Data loaded: {len(df):,} rows", "INFO")
        terminal_log(
            f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}",
            "INFO",
        )

        # Resample data
        terminal_log("🔄 Resampling data to target timeframe...", "INFO")
        df_resampled = analyzer.resample_to_timeframe(df)

        if df_resampled.empty:
            terminal_log("❌ Resampling failed. Exiting.", "ERROR")
            return None

        terminal_log(
            f"✅ Resampling complete: {len(df_resampled):,} data points",
            "INFO",
        )

        # Run analysis
        terminal_log("🔍 Running comprehensive analysis...", "INFO")
        display_df, score_df = analyzer.run_comprehensive_analysis(df_resampled)

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
            df_resampled,
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
