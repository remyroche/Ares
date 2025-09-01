#!/usr/bin/env python3
"""
Cryptocurrency Price Movement Analyzer
Analyzes OHLCV data to calculate potential profits from different triple barrier methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
            logger.error(f"Error loading data: {e}")
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
        price_change = (symbol_data['close'].iloc[-1] - symbol_data['close'].iloc[0]) / symbol_data['close'].iloc[0]

        # Volatility metrics
        returns = symbol_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(96)  # Annualized (96 15-min periods per day)

        # Volume metrics
        avg_volume = symbol_data['volume'].mean()
        volume_volatility = symbol_data['volume'].std() / symbol_data['volume'].mean()

        # Price range metrics
        daily_highs = symbol_data.groupby(symbol_data.index.date)['high'].max()
        daily_lows = symbol_data.groupby(symbol_data.index.date)['low'].min()
        avg_daily_range = ((daily_highs - daily_lows) / daily_lows).mean()

        # Intraday movement metrics
        intraday_highs = symbol_data.groupby(symbol_data.index.date)['high'].max()
        intraday_lows = symbol_data.groupby(symbol_data.index.date)['low'].min()
        avg_intraday_movement = ((intraday_highs - intraday_lows) / intraday_lows).mean()

        # Price movement frequency
        price_changes = symbol_data['close'].pct_change().abs()
        avg_price_change = price_changes.mean()
        price_change_std = price_changes.std()

        return {
            'total_return': price_change,
            'volatility': volatility,
            'avg_volume': avg_volume,
            'volume_volatility': volume_volatility,
            'avg_daily_range': avg_daily_range,
            'avg_intraday_movement': avg_intraday_movement,
            'avg_price_change': avg_price_change,
            'price_change_std': price_change_std,
            'total_volume': symbol_data['volume'].sum(),
            'avg_price': symbol_data['close'].mean(),
            'price_range': (symbol_data['high'].max() - symbol_data['low'].min()) / symbol_data['low'].min()
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
            results[f'barrier_{int(barrier*1000)}bp'] = barrier_results

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
            open_price = current_data['open']
            high_price = current_data['high']
            low_price = current_data['low']
            close_price = current_data['close']

            # Calculate potential profit if we captured 100% of the movement
            # Long position: buy at open, sell at high
            long_profit = (high_price - open_price) / open_price

            # Short position: sell at open, buy at low
            short_profit = (open_price - low_price) / open_price

            # Only count trades that exceed the barrier
            if long_profit >= barrier_pct:
                successful_trades.append({
                    'time': current_time,
                    'profit': long_profit,
                    'position': 'long',
                    'entry_price': open_price,
                    'exit_price': high_price
                })
                long_trades += 1

            if short_profit >= barrier_pct:
                successful_trades.append({
                    'time': current_time,
                    'profit': short_profit,
                    'position': 'short',
                    'entry_price': open_price,
                    'exit_price': low_price
                })
                short_trades += 1

        if not successful_trades:
            return {
                'total_trades': 0,
                'avg_profit': 0,
                'long_trades': 0,
                'short_trades': 0,
                'max_profit': 0,
                'min_profit': 0,
                'profit_std': 0,
                'total_potential_profit': 0,
                'profit_frequency': 0
            }

        # Convert to DataFrame for analysis
        trades_df = pd.DataFrame(successful_trades)

        return {
            'total_trades': len(successful_trades),
            'avg_profit': trades_df['profit'].mean(),
            'long_trades': long_trades,
            'short_trades': short_trades,
            'max_profit': trades_df['profit'].max(),
            'min_profit': trades_df['profit'].min(),
            'profit_std': trades_df['profit'].std(),
            'total_potential_profit': trades_df['profit'].sum(),
            'profit_frequency': len(successful_trades) / len(symbol_data)  # Successful trades per 15-min period
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
        symbol_data['hour'] = symbol_data.index.hour
        symbol_data['day_of_week'] = symbol_data.index.dayofweek

        # Hourly patterns
        hourly_volume = symbol_data.groupby('hour')['volume'].mean()
        hourly_volatility = symbol_data.groupby('hour')['close'].pct_change().std()
        hourly_price_changes = symbol_data.groupby('hour')['close'].pct_change().abs().mean()

        # Peak hours (highest volume)
        peak_hours = hourly_volume.nlargest(3).index.tolist()

        # Day of week patterns
        daily_volume = symbol_data.groupby('day_of_week')['volume'].mean()
        daily_volatility = symbol_data.groupby('day_of_week')['close'].pct_change().std()
        daily_price_changes = symbol_data.groupby('day_of_week')['close'].pct_change().abs().mean()

        # Best trading hours (highest price movements)
        best_hours = hourly_price_changes.nlargest(3).index.tolist()

        return {
            'peak_hours': peak_hours,
            'best_trading_hours': best_hours,
            'hourly_volume': hourly_volume.to_dict(),
            'hourly_volatility': hourly_volatility.to_dict(),
            'hourly_price_changes': hourly_price_changes.to_dict(),
            'daily_volume': daily_volume.to_dict(),
            'daily_volatility': daily_volatility.to_dict(),
            'daily_price_changes': daily_price_changes.to_dict()
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
        returns = symbol_data['close'].pct_change().dropna()

        # Movement size distribution
        movement_sizes = returns.abs()

        # Calculate percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        movement_percentiles = {}
        for p in percentiles:
            movement_percentiles[f'p{p}'] = movement_sizes.quantile(p/100)

        # Calculate movement frequency by size
        small_movements = (movement_sizes <= 0.001).sum() / len(movement_sizes)  # <= 0.1%
        medium_movements = ((movement_sizes > 0.001) & (movement_sizes <= 0.01)).sum() / len(movement_sizes)  # 0.1-1%
        large_movements = (movement_sizes > 0.01).sum() / len(movement_sizes)  # > 1%

        # Calculate consecutive movement patterns
        positive_runs = self._calculate_consecutive_runs(returns > 0)
        negative_runs = self._calculate_consecutive_runs(returns < 0)

        return {
            'avg_movement': movement_sizes.mean(),
            'median_movement': movement_sizes.median(),
            'movement_percentiles': movement_percentiles,
            'small_movements_pct': small_movements,
            'medium_movements_pct': medium_movements,
            'large_movements_pct': large_movements,
            'avg_positive_run': positive_runs['avg_length'],
            'avg_negative_run': negative_runs['avg_length'],
            'max_positive_run': positive_runs['max_length'],
            'max_negative_run': negative_runs['max_length']
        }

    def _calculate_consecutive_runs(self, condition_series):
        """Calculate consecutive runs of True values"""
        runs = []
        current_run = 0

        for value in condition_series:
            if value:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                    current_run = 0

        if current_run > 0:
            runs.append(current_run)

        if not runs:
            return {'avg_length': 0, 'max_length': 0}

        return {
            'avg_length': np.mean(runs),
            'max_length': np.max(runs)
        }

    def calculate_volume_analysis(self, symbol_data):
        """
        Calculate comprehensive volume analysis for a single asset

        Args:
            symbol_data (pd.DataFrame): Data for a single symbol

        Returns:
            dict: Dictionary of volume analysis metrics
        """
        # Basic volume metrics
        total_volume = symbol_data['volume'].sum()
        avg_volume = symbol_data['volume'].mean()
        median_volume = symbol_data['volume'].median()
        volume_std = symbol_data['volume'].std()
        volume_cv = volume_std / avg_volume  # Coefficient of variation

        # Volume percentiles
        volume_percentiles = {
            'p10': symbol_data['volume'].quantile(0.1),
            'p25': symbol_data['volume'].quantile(0.25),
            'p50': symbol_data['volume'].quantile(0.5),
            'p75': symbol_data['volume'].quantile(0.75),
            'p90': symbol_data['volume'].quantile(0.9),
            'p95': symbol_data['volume'].quantile(0.95),
            'p99': symbol_data['volume'].quantile(0.99)
        }

        # Volume distribution analysis
        volume_bins = pd.cut(symbol_data['volume'], bins=10)
        volume_distribution = volume_bins.value_counts().sort_index()

        # High volume periods (top 10% of volume)
        high_volume_threshold = symbol_data['volume'].quantile(0.9)
        high_volume_periods = symbol_data[symbol_data['volume'] >= high_volume_threshold]
        high_volume_frequency = len(high_volume_periods) / len(symbol_data)

        # Low volume periods (bottom 10% of volume)
        low_volume_threshold = symbol_data['volume'].quantile(0.1)
        low_volume_periods = symbol_data[symbol_data['volume'] <= low_volume_threshold]
        low_volume_frequency = len(low_volume_periods) / len(symbol_data)

        # Volume-price relationship
        volume_price_corr = symbol_data['volume'].corr(symbol_data['close'])
        volume_returns_corr = symbol_data['volume'].corr(symbol_data['close'].pct_change())

        # Volume volatility
        volume_volatility = symbol_data['volume'].pct_change().std()

        # Volume trends (using rolling average)
        rolling_volume = symbol_data['volume'].rolling(window=96).mean()  # 24 hours (96 15-min periods)
        volume_trend = (rolling_volume.iloc[-1] - rolling_volume.iloc[0]) / rolling_volume.iloc[0] if len(rolling_volume.dropna()) > 0 else 0

        # Volume spikes (periods with volume > 2x average)
        volume_spikes = symbol_data[symbol_data['volume'] > 2 * avg_volume]
        volume_spike_frequency = len(volume_spikes) / len(symbol_data)

        # Volume consistency (how often volume is within 50% of average)
        volume_consistency = ((symbol_data['volume'] >= 0.5 * avg_volume) &
                            (symbol_data['volume'] <= 1.5 * avg_volume)).mean()

        return {
            'total_volume': total_volume,
            'avg_volume': avg_volume,
            'median_volume': median_volume,
            'volume_std': volume_std,
            'volume_cv': volume_cv,
            'volume_percentiles': volume_percentiles,
            'high_volume_frequency': high_volume_frequency,
            'low_volume_frequency': low_volume_frequency,
            'volume_price_correlation': volume_price_corr,
            'volume_returns_correlation': volume_returns_corr,
            'volume_volatility': volume_volatility,
            'volume_trend': volume_trend,
            'volume_spike_frequency': volume_spike_frequency,
            'volume_consistency': volume_consistency,
            'max_volume': symbol_data['volume'].max(),
            'min_volume': symbol_data['volume'].min(),
            'volume_range': symbol_data['volume'].max() - symbol_data['volume'].min()
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

        volume_comparison = {}

        # Collect volume metrics from all assets
        volume_metrics = []
        for symbol, result in self.results.items():
            if 'volume_analysis' in result:
                metrics = result['volume_analysis'].copy()
                metrics['symbol'] = symbol
                volume_metrics.append(metrics)

        if not volume_metrics:
            return {}

        volume_df = pd.DataFrame(volume_metrics)

        # Volume ranking
        volume_ranking = volume_df.sort_values('total_volume', ascending=False)

        # Volume categories
        volume_categories = {
            'high_volume': volume_df[volume_df['total_volume'] >= volume_df['total_volume'].quantile(0.75)]['symbol'].tolist(),
            'medium_volume': volume_df[(volume_df['total_volume'] >= volume_df['total_volume'].quantile(0.25)) &
                                     (volume_df['total_volume'] < volume_df['total_volume'].quantile(0.75))]['symbol'].tolist(),
            'low_volume': volume_df[volume_df['total_volume'] < volume_df['total_volume'].quantile(0.25)]['symbol'].tolist()
        }

        # Volume consistency ranking
        consistency_ranking = volume_df.sort_values('volume_consistency', ascending=False)

        # Volume volatility ranking
        volatility_ranking = volume_df.sort_values('volume_volatility', ascending=False)

        # Volume-price correlation ranking
        correlation_ranking = volume_df.sort_values('volume_price_correlation', ascending=False)

        # Volume trends ranking
        trend_ranking = volume_df.sort_values('volume_trend', ascending=False)

        # Volume spike frequency ranking
        spike_ranking = volume_df.sort_values('volume_spike_frequency', ascending=False)

        # Summary statistics
        summary_stats = {
            'total_volume_mean': volume_df['total_volume'].mean(),
            'total_volume_median': volume_df['total_volume'].median(),
            'total_volume_std': volume_df['total_volume'].std(),
            'avg_volume_mean': volume_df['avg_volume'].mean(),
            'avg_volume_median': volume_df['avg_volume'].median(),
            'volume_consistency_mean': volume_df['volume_consistency'].mean(),
            'volume_volatility_mean': volume_df['volume_volatility'].mean(),
            'volume_price_correlation_mean': volume_df['volume_price_correlation'].mean(),
            'volume_spike_frequency_mean': volume_df['volume_spike_frequency'].mean()
        }

        return {
            'volume_ranking': volume_ranking,
            'volume_categories': volume_categories,
            'consistency_ranking': consistency_ranking,
            'volatility_ranking': volatility_ranking,
            'correlation_ranking': correlation_ranking,
            'trend_ranking': trend_ranking,
            'spike_ranking': spike_ranking,
            'summary_stats': summary_stats,
            'volume_dataframe': volume_df
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
        symbol_data['hour'] = symbol_data.index.hour
        symbol_data['day_of_week'] = symbol_data.index.dayofweek
        symbol_data['date'] = symbol_data.index.date

        # Hourly volume patterns
        hourly_volume = symbol_data.groupby('hour')['volume'].agg(['mean', 'std', 'min', 'max'])
        peak_hours = hourly_volume['mean'].nlargest(3).index.tolist()
        low_hours = hourly_volume['mean'].nsmallest(3).index.tolist()

        # Daily volume patterns
        daily_volume = symbol_data.groupby('day_of_week')['volume'].agg(['mean', 'std', 'min', 'max'])
        peak_days = daily_volume['mean'].nlargest(3).index.tolist()
        low_days = daily_volume['mean'].nsmallest(3).index.tolist()

        # Volume autocorrelation (lag 1)
        volume_autocorr = symbol_data['volume'].autocorr(lag=1)

        # Volume seasonality (using rolling averages)
        daily_rolling_volume = symbol_data.groupby('date')['volume'].mean().rolling(window=7).mean()
        weekly_seasonality = daily_rolling_volume.std() / daily_rolling_volume.mean() if len(daily_rolling_volume.dropna()) > 0 else 0

        # Volume clustering (consecutive high/low volume periods)
        high_volume_threshold = symbol_data['volume'].quantile(0.75)
        low_volume_threshold = symbol_data['volume'].quantile(0.25)

        high_volume_clusters = self._calculate_consecutive_runs(symbol_data['volume'] >= high_volume_threshold)
        low_volume_clusters = self._calculate_consecutive_runs(symbol_data['volume'] <= low_volume_threshold)

        return {
            'hourly_volume': hourly_volume.to_dict(),
            'daily_volume': daily_volume.to_dict(),
            'peak_hours': peak_hours,
            'low_hours': low_hours,
            'peak_days': peak_days,
            'low_days': low_days,
            'volume_autocorrelation': volume_autocorr,
            'weekly_seasonality': weekly_seasonality,
            'high_volume_clusters': high_volume_clusters,
            'low_volume_clusters': low_volume_clusters
        }

    def analyze_all_assets(self):
        """Analyze all assets in the dataset"""
        if self.df is None:
            logger.error("No data loaded. Call load_data() first.")
            return

        symbols = self.df['symbol'].unique()
        logger.info(f"Analyzing {len(symbols)} assets")

        for symbol in symbols:
            logger.info(f"Analyzing {symbol}")
            symbol_data = self.df[self.df['symbol'] == symbol].copy()

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
                'basic_metrics': basic_metrics,
                'triple_barrier_profits': triple_barrier_profits,
                'intraday_patterns': intraday_patterns,
                'movement_statistics': movement_stats,
                'volume_analysis': volume_analysis,
                'volume_patterns': volume_patterns
            }

    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        if not self.results:
            logger.error("No analysis results. Call analyze_all_assets() first.")
            return

        print("\n" + "="*80)
        print("CRYPTOCURRENCY PRICE MOVEMENT ANALYSIS REPORT")
        print("="*80)

        # Create summary DataFrames
        basic_summary = []
        barrier_summary = []

        for symbol, result in self.results.items():
            basic = result['basic_metrics']

            basic_summary.append({
                'Symbol': symbol,
                'Total_Return': basic['total_return'],
                'Volatility': basic['volatility'],
                'Avg_Daily_Range': basic['avg_daily_range'],
                'Avg_Intraday_Movement': basic['avg_intraday_movement'],
                'Avg_Price_Change': basic['avg_price_change'],
                'Avg_Volume': basic['avg_volume']
            })

            # Triple barrier results
            for barrier_name, barrier_data in result['triple_barrier_profits'].items():
                barrier_level = int(barrier_name.split('_')[1].replace('bp', '')) / 1000
                barrier_summary.append({
                    'Symbol': symbol,
                    'Barrier_Level': f"{barrier_level:.1%}",
                    'Total_Trades': barrier_data['total_trades'],
                    'Avg_Profit': barrier_data['avg_profit'],
                    'Long_Trades': barrier_data['long_trades'],
                    'Short_Trades': barrier_data['short_trades'],
                    'Profit_Frequency': barrier_data['profit_frequency'],
                    'Max_Profit': barrier_data['max_profit'],
                    'Total_Potential_Profit': barrier_data['total_potential_profit']
                })

        basic_df = pd.DataFrame(basic_summary)
        barrier_df = pd.DataFrame(barrier_summary)

        # Print basic metrics
        print("\nBASIC PRICE MOVEMENT METRICS:")
        print("-" * 60)
        print(basic_df.round(4).to_string(index=False))

        # Print triple barrier results
        print("\nTRIPLE BARRIER PROFIT ANALYSIS:")
        print("-" * 60)
        print(barrier_df.round(4).to_string(index=False))

        # Top performers by barrier level
        print("\nTOP PERFORMERS BY BARRIER LEVEL:")
        print("-" * 60)

        barrier_levels = barrier_df['Barrier_Level'].unique()
        for barrier in sorted(barrier_levels):
            barrier_data = barrier_df[barrier_df['Barrier_Level'] == barrier]
            top_performers = barrier_data.nlargest(5, 'Avg_Profit')[['Symbol', 'Barrier_Level', 'Avg_Profit', 'Total_Trades', 'Long_Trades', 'Short_Trades']]
            print(f"\nTop 5 for {barrier} barrier:")
            print(top_performers.round(4).to_string(index=False))

        # Best overall performers
        print("\nBEST OVERALL PERFORMERS:")
        print("-" * 60)

        # Average across all barriers
        avg_profits = barrier_df.groupby('Symbol')['Avg_Profit'].mean().sort_values(ascending=False)
        print("Average profit across all barriers:")
        for symbol, profit in avg_profits.head(10).items():
            print(f"  {symbol}: {profit:.4f}")

        # Most active assets
        total_trades = barrier_df.groupby('Symbol')['Total_Trades'].sum().sort_values(ascending=False)
        print(f"\nMost active assets (total trades across all barriers):")
        for symbol, trades in total_trades.head(10).items():
            print(f"  {symbol}: {trades:.0f} trades")

        # Long vs Short analysis
        total_longs = barrier_df.groupby('Symbol')['Long_Trades'].sum().sort_values(ascending=False)
        total_shorts = barrier_df.groupby('Symbol')['Short_Trades'].sum().sort_values(ascending=False)
        print(f"\nAssets with most long trades:")
        for symbol, trades in total_longs.head(5).items():
            print(f"  {symbol}: {trades:.0f} long trades")
        print(f"\nAssets with most short trades:")
        for symbol, trades in total_shorts.head(5).items():
            print(f"  {symbol}: {trades:.0f} short trades")

        # Volume comparison summary
        volume_comparison_summary = self.calculate_volume_comparison()
        if volume_comparison_summary:
            print("\nVOLUME ANALYSIS SUMMARY:")
            print("-" * 60)
            print(f"Total Volume Mean: {volume_comparison_summary['summary_stats']['total_volume_mean']:.2f}")
            print(f"Total Volume Median: {volume_comparison_summary['summary_stats']['total_volume_median']:.2f}")
            print(f"Total Volume Std Dev: {volume_comparison_summary['summary_stats']['total_volume_std']:.2f}")
            print(f"Avg Volume Mean: {volume_comparison_summary['summary_stats']['avg_volume_mean']:.2f}")
            print(f"Avg Volume Median: {volume_comparison_summary['summary_stats']['avg_volume_median']:.2f}")
            print(f"Volume Consistency Mean: {volume_comparison_summary['summary_stats']['volume_consistency_mean']:.2f}")
            print(f"Volume Volatility Mean: {volume_comparison_summary['summary_stats']['volume_volatility_mean']:.2f}")
            print(f"Volume Price Correlation Mean: {volume_comparison_summary['summary_stats']['volume_price_correlation_mean']:.2f}")
            print(f"Volume Spike Frequency Mean: {volume_comparison_summary['summary_stats']['volume_spike_frequency_mean']:.2f}")

            print("\nVOLUME RANKINGS:")
            print("-" * 60)

            # Top 10 by total volume
            print("Top 10 Assets by Total Volume:")
            top_volume = volume_comparison_summary['volume_ranking'].head(10)
            for _, row in top_volume.iterrows():
                print(f"  {row['symbol']}: {row['total_volume']:.2f}")

            # Top 10 by volume consistency
            print("\nTop 10 Assets by Volume Consistency:")
            top_consistency = volume_comparison_summary['consistency_ranking'].head(10)
            for _, row in top_consistency.iterrows():
                print(f"  {row['symbol']}: {row['volume_consistency']:.4f}")

            # Top 10 by volume-price correlation
            print("\nTop 10 Assets by Volume-Price Correlation:")
            top_correlation = volume_comparison_summary['correlation_ranking'].head(10)
            for _, row in top_correlation.iterrows():
                print(f"  {row['symbol']}: {row['volume_price_correlation']:.4f}")

            # Volume categories
            print("\nVOLUME CATEGORIES:")
            print("-" * 60)
            print(f"High Volume Assets: {', '.join(volume_comparison_summary['volume_categories']['high_volume'])}")
            print(f"Medium Volume Assets: {', '.join(volume_comparison_summary['volume_categories']['medium_volume'])}")
            print(f"Low Volume Assets: {', '.join(volume_comparison_summary['volume_categories']['low_volume'])}")

            # Volume patterns summary
            print("\nVOLUME PATTERNS SUMMARY:")
            print("-" * 60)
            for symbol in list(self.results.keys())[:5]:  # Show first 5 assets
                if 'volume_patterns' in self.results[symbol]:
                    patterns = self.results[symbol]['volume_patterns']
                    print(f"\n{symbol}:")
                    print(f"  Peak Hours: {patterns['peak_hours']}")
                    print(f"  Peak Days: {patterns['peak_days']}")
                    print(f"  Volume Autocorrelation: {patterns['volume_autocorrelation']:.4f}")
                    print(f"  Weekly Seasonality: {patterns['weekly_seasonality']:.4f}")

        return {
            'basic_summary': basic_df,
            'barrier_summary': barrier_df,
            'volume_summary': volume_comparison_summary.get('volume_dataframe', pd.DataFrame()) if volume_comparison_summary else pd.DataFrame()
        }

    def create_visualizations(self, output_dir="plots"):
        """Create visualizations for the analysis"""
        if not self.results:
            logger.error("No analysis results. Call analyze_all_assets() first.")
            return

        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. Triple Barrier Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Triple Barrier Profit Analysis', fontsize=16)

        # Prepare data for plotting
        barrier_data = []
        for symbol, result in self.results.items():
            for barrier_name, barrier_result in result['triple_barrier_profits'].items():
                barrier_level = int(barrier_name.split('_')[1].replace('bp', '')) / 1000
                barrier_data.append({
                    'Symbol': symbol,
                    'Barrier_Level': barrier_level,
                    'Avg_Profit': barrier_result['avg_profit'],
                    'Total_Trades': barrier_result['total_trades'],
                    'Long_Trades': barrier_result['long_trades'],
                    'Short_Trades': barrier_result['short_trades']
                })

        barrier_df = pd.DataFrame(barrier_data)

        # Plot 1: Average profit by barrier level
        for barrier in sorted(barrier_df['Barrier_Level'].unique()):
            data = barrier_df[barrier_df['Barrier_Level'] == barrier]
            axes[0, 0].scatter(data['Symbol'], data['Avg_Profit'],
                             label=f'{barrier:.1%}', alpha=0.7, s=50)

        axes[0, 0].set_xlabel('Assets')
        axes[0, 0].set_ylabel('Average Profit')
        axes[0, 0].set_title('Average Profit by Barrier Level')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Total trades by barrier level
        for barrier in sorted(barrier_df['Barrier_Level'].unique()):
            data = barrier_df[barrier_df['Barrier_Level'] == barrier]
            axes[0, 1].scatter(data['Symbol'], data['Total_Trades'],
                             label=f'{barrier:.1%}', alpha=0.7, s=50)

        axes[0, 1].set_xlabel('Assets')
        axes[0, 1].set_ylabel('Total Trades')
        axes[0, 1].set_title('Total Trades by Barrier Level')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Average performance across all barriers
        avg_performance = barrier_df.groupby('Symbol').agg({
            'Avg_Profit': 'mean',
            'Total_Trades': 'sum',
            'Long_Trades': 'sum',
            'Short_Trades': 'sum'
        }).sort_values('Avg_Profit', ascending=False)

        axes[1, 0].bar(range(len(avg_performance)), avg_performance['Avg_Profit'])
        axes[1, 0].set_xlabel('Assets')
        axes[1, 0].set_ylabel('Average Profit (All Barriers)')
        axes[1, 0].set_title('Average Performance Across All Barriers')
        axes[1, 0].set_xticks(range(len(avg_performance)))
        axes[1, 0].set_xticklabels(avg_performance.index, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Long vs Short trades
        long_trades = avg_performance['Long_Trades'].values
        short_trades = avg_performance['Short_Trades'].values
        x_pos = np.arange(len(avg_performance))

        axes[1, 1].bar(x_pos - 0.2, long_trades, 0.4, label='Long Trades', alpha=0.8)
        axes[1, 1].bar(x_pos + 0.2, short_trades, 0.4, label='Short Trades', alpha=0.8)
        axes[1, 1].set_xlabel('Assets')
        axes[1, 1].set_ylabel('Number of Trades')
        axes[1, 1].set_title('Long vs Short Trades')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(avg_performance.index, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)



        plt.tight_layout()
        plt.savefig(f"{output_dir}/triple_barrier_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Intraday Patterns
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Intraday Price Movement Patterns', fontsize=16)

        # Sample a few assets
        sample_symbols = list(self.results.keys())[:5]

        # Hourly price changes
        for symbol in sample_symbols:
            hourly_changes = self.results[symbol]['intraday_patterns']['hourly_price_changes']
            hours = list(hourly_changes.keys())
            changes = list(hourly_changes.values())
            axes[0, 0].plot(hours, changes, label=symbol, marker='o', alpha=0.7)

        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Average Price Change')
        axes[0, 0].set_title('Hourly Price Movement Patterns')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Daily price changes
        for symbol in sample_symbols:
            daily_changes = self.results[symbol]['intraday_patterns']['daily_price_changes']
            days = list(daily_changes.keys())
            changes = list(daily_changes.values())
            axes[0, 1].plot(days, changes, label=symbol, marker='s', alpha=0.7)

        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Average Price Change')
        axes[0, 1].set_title('Daily Price Movement Patterns')
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Movement size distribution
        for symbol in sample_symbols:
            movement_stats = self.results[symbol]['movement_statistics']
            percentiles = list(movement_stats['movement_percentiles'].keys())
            values = list(movement_stats['movement_percentiles'].values())
            axes[1, 0].plot(percentiles, values, label=symbol, marker='^', alpha=0.7)

        axes[1, 0].set_xlabel('Percentile')
        axes[1, 0].set_ylabel('Movement Size')
        axes[1, 0].set_title('Price Movement Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Average daily range vs volatility
        daily_ranges = [self.results[s]['basic_metrics']['avg_daily_range'] for s in sample_symbols]
        volatilities = [self.results[s]['basic_metrics']['volatility'] for s in sample_symbols]

        axes[1, 1].scatter(volatilities, daily_ranges, alpha=0.7, s=100)
        axes[1, 1].set_xlabel('Volatility')
        axes[1, 1].set_ylabel('Average Daily Range')
        axes[1, 1].set_title('Volatility vs Daily Range')
        axes[1, 1].grid(True, alpha=0.3)

        # Add labels
        for i, symbol in enumerate(sample_symbols):
            axes[1, 1].annotate(symbol, (volatilities[i], daily_ranges[i]),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/intraday_patterns.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Volume Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Volume Analysis', fontsize=16)

        # Volume ranking
        volume_comparison = self.calculate_volume_comparison()
        if volume_comparison:
            volume_ranking = volume_comparison['volume_ranking']
            symbols = volume_ranking['symbol'].values
            total_volumes = volume_ranking['total_volume'].values

            axes[0, 0].bar(range(len(symbols)), total_volumes, alpha=0.7)
            axes[0, 0].set_xlabel('Assets')
            axes[0, 0].set_ylabel('Total Volume')
            axes[0, 0].set_title('Total Volume by Asset')
            axes[0, 0].set_xticks(range(len(symbols)))
            axes[0, 0].set_xticklabels(symbols, rotation=45)
            axes[0, 0].grid(True, alpha=0.3)

            # Volume consistency
            consistency_ranking = volume_comparison['consistency_ranking']
            consistency_values = consistency_ranking['volume_consistency'].values

            axes[0, 1].bar(range(len(symbols)), consistency_values, alpha=0.7, color='green')
            axes[0, 1].set_xlabel('Assets')
            axes[0, 1].set_ylabel('Volume Consistency')
            axes[0, 1].set_title('Volume Consistency by Asset')
            axes[0, 1].set_xticks(range(len(symbols)))
            axes[0, 1].set_xticklabels(symbols, rotation=45)
            axes[0, 1].grid(True, alpha=0.3)

            # Volume volatility
            volatility_ranking = volume_comparison['volatility_ranking']
            volatility_values = volatility_ranking['volume_volatility'].values

            axes[1, 0].bar(range(len(symbols)), volatility_values, alpha=0.7, color='red')
            axes[1, 0].set_xlabel('Assets')
            axes[1, 0].set_ylabel('Volume Volatility')
            axes[1, 0].set_title('Volume Volatility by Asset')
            axes[1, 0].set_xticks(range(len(symbols)))
            axes[1, 0].set_xticklabels(symbols, rotation=45)
            axes[1, 0].grid(True, alpha=0.3)

            # Volume-price correlation
            correlation_ranking = volume_comparison['correlation_ranking']
            correlation_values = correlation_ranking['volume_price_correlation'].values

            axes[1, 1].bar(range(len(symbols)), correlation_values, alpha=0.7, color='purple')
            axes[1, 1].set_xlabel('Assets')
            axes[1, 1].set_ylabel('Volume-Price Correlation')
            axes[1, 1].set_title('Volume-Price Correlation by Asset')
            axes[1, 1].set_xticks(range(len(symbols)))
            axes[1, 1].set_xticklabels(symbols, rotation=45)
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/volume_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Volume Patterns
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Volume Patterns Over Time', fontsize=16)

        # Hourly volume patterns
        for symbol in sample_symbols:
            if 'volume_patterns' in self.results[symbol]:
                hourly_volume = self.results[symbol]['volume_patterns']['hourly_volume']
                hours = list(hourly_volume.keys())
                means = [hourly_volume[h]['mean'] for h in hours]
                axes[0, 0].plot(hours, means, label=symbol, marker='o', alpha=0.7)

        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Average Volume')
        axes[0, 0].set_title('Hourly Volume Patterns')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Daily volume patterns
        for symbol in sample_symbols:
            if 'volume_patterns' in self.results[symbol]:
                daily_volume = self.results[symbol]['volume_patterns']['daily_volume']
                days = list(daily_volume.keys())
                means = [daily_volume[d]['mean'] for d in days]
                axes[0, 1].plot(days, means, label=symbol, marker='s', alpha=0.7)

        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Average Volume')
        axes[0, 1].set_title('Daily Volume Patterns')
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Volume autocorrelation
        autocorr_values = []
        autocorr_symbols = []
        for symbol in sample_symbols:
            if 'volume_patterns' in self.results[symbol]:
                autocorr = self.results[symbol]['volume_patterns']['volume_autocorrelation']
                if not np.isnan(autocorr):
                    autocorr_values.append(autocorr)
                    autocorr_symbols.append(symbol)

        if autocorr_values:
            axes[1, 0].bar(range(len(autocorr_symbols)), autocorr_values, alpha=0.7, color='orange')
            axes[1, 0].set_xlabel('Assets')
            axes[1, 0].set_ylabel('Volume Autocorrelation')
            axes[1, 0].set_title('Volume Autocorrelation (Lag 1)')
            axes[1, 0].set_xticks(range(len(autocorr_symbols)))
            axes[1, 0].set_xticklabels(autocorr_symbols, rotation=45)
            axes[1, 0].grid(True, alpha=0.3)

        # Volume spike frequency
        spike_values = []
        spike_symbols = []
        for symbol in sample_symbols:
            spike_freq = self.results[symbol]['volume_analysis']['volume_spike_frequency']
            spike_values.append(spike_freq)
            spike_symbols.append(symbol)

        if spike_values:
            axes[1, 1].bar(range(len(spike_symbols)), spike_values, alpha=0.7, color='brown')
            axes[1, 1].set_xlabel('Assets')
            axes[1, 1].set_ylabel('Volume Spike Frequency')
            axes[1, 1].set_title('Volume Spike Frequency (>2x Average)')
            axes[1, 1].set_xticks(range(len(spike_symbols)))
            axes[1, 1].set_xticklabels(spike_symbols, rotation=45)
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/volume_patterns.png", dpi=300, bbox_inches='tight')
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
    summary = analyzer.generate_summary_report()
    analyzer.create_visualizations()

    # Save results to CSV
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    summary['basic_summary'].to_csv(output_dir / "price_movement_metrics.csv", index=False)
    summary['barrier_summary'].to_csv(output_dir / "triple_barrier_profits.csv", index=False)

    # Save volume analysis results
    if not summary['volume_summary'].empty:
        summary['volume_summary'].to_csv(output_dir / "volume_analysis.csv", index=False)
        logger.info("Volume analysis results saved to volume_analysis.csv")

    logger.info(f"Results saved to {output_dir}/")

if __name__ == "__main__":
    main()