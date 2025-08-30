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
    
    def calculate_triple_barrier_profits(self, symbol_data, barrier_levels=[0.005, 0.01, 0.02, 0.03, 0.05]):
        """
        Calculate potential profits from triple barrier methods
        
        Args:
            symbol_data (pd.DataFrame): Data for a single symbol
            barrier_levels (list): List of barrier percentages to test
            
        Returns:
            dict: Dictionary of triple barrier profit calculations
        """
        results = {}
        
        for barrier in barrier_levels:
            # Calculate potential profits for each barrier level
            barrier_results = self._calculate_single_barrier_profits(symbol_data, barrier)
            results[f'barrier_{int(barrier*1000)}bp'] = barrier_results
        
        return results
    
    def _calculate_single_barrier_profits(self, symbol_data, barrier_pct):
        """
        Calculate profits for a single barrier level
        
        Args:
            symbol_data (pd.DataFrame): Data for a single symbol
            barrier_pct (float): Barrier percentage (e.g., 0.01 for 1%)
            
        Returns:
            dict: Profit calculations for this barrier
        """
        # Group by day to calculate daily movements
        daily_data = symbol_data.groupby(symbol_data.index.date).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Calculate potential profits for each day
        daily_profits = []
        daily_movements = []
        
        for date, day_data in daily_data.iterrows():
            open_price = day_data['open']
            high_price = day_data['high']
            low_price = day_data['low']
            close_price = day_data['close']
            
            # Calculate potential profit if we captured 100% of the movement
            # Long position: buy at open, sell at high
            long_profit = (high_price - open_price) / open_price
            
            # Short position: sell at open, buy at low
            short_profit = (open_price - low_price) / open_price
            
            # Take the better of long or short
            best_profit = max(long_profit, short_profit)
            
            # Only count if it exceeds the barrier
            if best_profit >= barrier_pct:
                daily_profits.append(best_profit)
                daily_movements.append(best_profit)
            else:
                daily_movements.append(best_profit)
        
        if not daily_profits:
            return {
                'avg_daily_profit': 0,
                'total_days_with_profit': 0,
                'profit_frequency': 0,
                'max_daily_profit': 0,
                'min_daily_profit': 0,
                'profit_std': 0,
                'total_potential_profit': 0,
                'avg_all_movements': np.mean(daily_movements) if daily_movements else 0
            }
        
        return {
            'avg_daily_profit': np.mean(daily_profits),
            'total_days_with_profit': len(daily_profits),
            'profit_frequency': len(daily_profits) / len(daily_data),
            'max_daily_profit': np.max(daily_profits),
            'min_daily_profit': np.min(daily_profits),
            'profit_std': np.std(daily_profits),
            'total_potential_profit': np.sum(daily_profits),
            'avg_all_movements': np.mean(daily_movements)
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
            
            # Store results
            self.results[symbol] = {
                'basic_metrics': basic_metrics,
                'triple_barrier_profits': triple_barrier_profits,
                'intraday_patterns': intraday_patterns,
                'movement_statistics': movement_stats
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
                    'Avg_Daily_Profit': barrier_data['avg_daily_profit'],
                    'Profit_Frequency': barrier_data['profit_frequency'],
                    'Total_Days_With_Profit': barrier_data['total_days_with_profit'],
                    'Max_Daily_Profit': barrier_data['max_daily_profit'],
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
            top_performers = barrier_data.nlargest(5, 'Avg_Daily_Profit')[['Symbol', 'Avg_Daily_Profit', 'Profit_Frequency']]
            print(f"\nTop 5 for {barrier} barrier:")
            print(top_performers.round(4).to_string(index=False))
        
        # Best overall performers
        print("\nBEST OVERALL PERFORMERS:")
        print("-" * 60)
        
        # Average across all barriers
        avg_profits = barrier_df.groupby('Symbol')['Avg_Daily_Profit'].mean().sort_values(ascending=False)
        print("Average daily profit across all barriers:")
        for symbol, profit in avg_profits.head(10).items():
            print(f"  {symbol}: {profit:.4f}")
        
        # Highest frequency assets
        avg_frequency = barrier_df.groupby('Symbol')['Profit_Frequency'].mean().sort_values(ascending=False)
        print(f"\nHighest profit frequency across all barriers:")
        for symbol, freq in avg_frequency.head(10).items():
            print(f"  {symbol}: {freq:.4f}")
        
        return {
            'basic_summary': basic_df,
            'barrier_summary': barrier_df
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
                    'Avg_Daily_Profit': barrier_result['avg_daily_profit'],
                    'Profit_Frequency': barrier_result['profit_frequency']
                })
        
        barrier_df = pd.DataFrame(barrier_data)
        
        # Plot 1: Average daily profit by barrier level
        for barrier in sorted(barrier_df['Barrier_Level'].unique()):
            data = barrier_df[barrier_df['Barrier_Level'] == barrier]
            axes[0, 0].scatter(data['Symbol'], data['Avg_Daily_Profit'], 
                             label=f'{barrier:.1%}', alpha=0.7, s=50)
        
        axes[0, 0].set_xlabel('Assets')
        axes[0, 0].set_ylabel('Average Daily Profit')
        axes[0, 0].set_title('Average Daily Profit by Barrier Level')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Profit frequency by barrier level
        for barrier in sorted(barrier_df['Barrier_Level'].unique()):
            data = barrier_df[barrier_df['Barrier_Level'] == barrier]
            axes[0, 1].scatter(data['Symbol'], data['Profit_Frequency'], 
                             label=f'{barrier:.1%}', alpha=0.7, s=50)
        
        axes[0, 1].set_xlabel('Assets')
        axes[0, 1].set_ylabel('Profit Frequency')
        axes[0, 1].set_title('Profit Frequency by Barrier Level')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Average performance across all barriers
        avg_performance = barrier_df.groupby('Symbol').agg({
            'Avg_Daily_Profit': 'mean',
            'Profit_Frequency': 'mean'
        }).sort_values('Avg_Daily_Profit', ascending=False)
        
        axes[1, 0].bar(range(len(avg_performance)), avg_performance['Avg_Daily_Profit'])
        axes[1, 0].set_xlabel('Assets')
        axes[1, 0].set_ylabel('Average Daily Profit (All Barriers)')
        axes[1, 0].set_title('Average Performance Across All Barriers')
        axes[1, 0].set_xticks(range(len(avg_performance)))
        axes[1, 0].set_xticklabels(avg_performance.index, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Volatility vs Average Profit
        volatilities = [self.results[s]['basic_metrics']['volatility'] for s in avg_performance.index]
        avg_profits = avg_performance['Avg_Daily_Profit'].values
        
        axes[1, 1].scatter(volatilities, avg_profits, alpha=0.7, s=100)
        axes[1, 1].set_xlabel('Volatility')
        axes[1, 1].set_ylabel('Average Daily Profit')
        axes[1, 1].set_title('Volatility vs Average Profit')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add asset labels
        for i, symbol in enumerate(avg_performance.index):
            axes[1, 1].annotate(symbol, (volatilities[i], avg_profits[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
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
    
    logger.info(f"Results saved to {output_dir}/")

if __name__ == "__main__":
    main()