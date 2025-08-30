#!/usr/bin/env python3
"""
Cryptocurrency Data Analyzer for Scalping/Swinging Analysis
Analyzes OHLCV data to compare potential profits from different trading strategies
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

class CryptoDataAnalyzer:
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
        Calculate basic metrics for a single asset
        
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
        
        return {
            'total_return': price_change,
            'volatility': volatility,
            'avg_volume': avg_volume,
            'volume_volatility': volume_volatility,
            'avg_daily_range': avg_daily_range,
            'total_volume': symbol_data['volume'].sum(),
            'avg_price': symbol_data['close'].mean(),
            'price_range': (symbol_data['high'].max() - symbol_data['low'].min()) / symbol_data['low'].min()
        }
    
    def calculate_intraday_patterns(self, symbol_data):
        """
        Calculate intraday trading patterns
        
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
        
        # Peak hours (highest volume)
        peak_hours = hourly_volume.nlargest(3).index.tolist()
        
        # Day of week patterns
        daily_volume = symbol_data.groupby('day_of_week')['volume'].mean()
        daily_volatility = symbol_data.groupby('day_of_week')['close'].pct_change().std()
        
        return {
            'peak_hours': peak_hours,
            'hourly_volume': hourly_volume.to_dict(),
            'hourly_volatility': hourly_volatility.to_dict(),
            'daily_volume': daily_volume.to_dict(),
            'daily_volatility': daily_volatility.to_dict()
        }
    
    def simulate_scalping_strategy(self, symbol_data, take_profit_pct=0.005, stop_loss_pct=0.003):
        """
        Simulate a simple scalping strategy
        
        Args:
            symbol_data (pd.DataFrame): Data for a single symbol
            take_profit_pct (float): Take profit percentage
            stop_loss_pct (float): Stop loss percentage
            
        Returns:
            dict: Scalping strategy results
        """
        trades = []
        position = None
        entry_price = 0
        
        for i in range(1, len(symbol_data)):
            current_price = symbol_data['close'].iloc[i]
            current_high = symbol_data['high'].iloc[i]
            current_low = symbol_data['low'].iloc[i]
            
            if position is None:
                # Look for entry signal (simple: price increase > 0.5%)
                if symbol_data['close'].iloc[i] > symbol_data['close'].iloc[i-1] * 1.005:
                    position = 'long'
                    entry_price = current_price
                    entry_time = symbol_data.index[i]
            
            elif position == 'long':
                # Check for exit conditions
                profit_target = entry_price * (1 + take_profit_pct)
                stop_loss = entry_price * (1 - stop_loss_pct)
                
                if current_high >= profit_target:
                    # Take profit
                    exit_price = profit_target
                    exit_time = symbol_data.index[i]
                    profit = (exit_price - entry_price) / entry_price
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': profit,
                        'exit_reason': 'take_profit'
                    })
                    position = None
                    
                elif current_low <= stop_loss:
                    # Stop loss
                    exit_price = stop_loss
                    exit_time = symbol_data.index[i]
                    profit = (exit_price - entry_price) / entry_price
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': profit,
                        'exit_reason': 'stop_loss'
                    })
                    position = None
        
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        trades_df = pd.DataFrame(trades)
        winning_trades = trades_df[trades_df['profit'] > 0]
        
        # Calculate metrics
        total_trades = len(trades_df)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_profit = trades_df['profit'].mean()
        total_return = trades_df['profit'].sum()
        
        # Calculate drawdown
        cumulative_returns = (1 + trades_df['profit']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio (simplified)
        returns_std = trades_df['profit'].std()
        sharpe_ratio = avg_profit / returns_std if returns_std > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trades': trades_df
        }
    
    def simulate_swing_strategy(self, symbol_data, take_profit_pct=0.05, stop_loss_pct=0.03, hold_periods=96):
        """
        Simulate a simple swing trading strategy
        
        Args:
            symbol_data (pd.DataFrame): Data for a single symbol
            take_profit_pct (float): Take profit percentage
            stop_loss_pct (float): Stop loss percentage
            hold_periods (int): Maximum hold periods (96 = 1 day)
            
        Returns:
            dict: Swing strategy results
        """
        trades = []
        position = None
        entry_price = 0
        entry_time = None
        hold_count = 0
        
        for i in range(1, len(symbol_data)):
            current_price = symbol_data['close'].iloc[i]
            current_high = symbol_data['high'].iloc[i]
            current_low = symbol_data['low'].iloc[i]
            
            if position is None:
                # Look for entry signal (simple: price increase > 2% over 4 periods)
                if i >= 4:
                    price_change_4p = (symbol_data['close'].iloc[i] - symbol_data['close'].iloc[i-4]) / symbol_data['close'].iloc[i-4]
                    if price_change_4p > 0.02:
                        position = 'long'
                        entry_price = current_price
                        entry_time = symbol_data.index[i]
                        hold_count = 0
            
            elif position == 'long':
                hold_count += 1
                
                # Check for exit conditions
                profit_target = entry_price * (1 + take_profit_pct)
                stop_loss = entry_price * (1 - stop_loss_pct)
                
                if current_high >= profit_target:
                    # Take profit
                    exit_price = profit_target
                    exit_time = symbol_data.index[i]
                    profit = (exit_price - entry_price) / entry_price
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': profit,
                        'exit_reason': 'take_profit',
                        'hold_periods': hold_count
                    })
                    position = None
                    
                elif current_low <= stop_loss:
                    # Stop loss
                    exit_price = stop_loss
                    exit_time = symbol_data.index[i]
                    profit = (exit_price - entry_price) / entry_price
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': profit,
                        'exit_reason': 'stop_loss',
                        'hold_periods': hold_count
                    })
                    position = None
                    
                elif hold_count >= hold_periods:
                    # Time-based exit
                    exit_price = current_price
                    exit_time = symbol_data.index[i]
                    profit = (exit_price - entry_price) / entry_price
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': profit,
                        'exit_reason': 'time_exit',
                        'hold_periods': hold_count
                    })
                    position = None
        
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'avg_hold_periods': 0
            }
        
        trades_df = pd.DataFrame(trades)
        winning_trades = trades_df[trades_df['profit'] > 0]
        
        # Calculate metrics
        total_trades = len(trades_df)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_profit = trades_df['profit'].mean()
        total_return = trades_df['profit'].sum()
        avg_hold_periods = trades_df['hold_periods'].mean()
        
        # Calculate drawdown
        cumulative_returns = (1 + trades_df['profit']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio
        returns_std = trades_df['profit'].std()
        sharpe_ratio = avg_profit / returns_std if returns_std > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_hold_periods': avg_hold_periods,
            'trades': trades_df
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
            
            # Intraday patterns
            intraday_patterns = self.calculate_intraday_patterns(symbol_data)
            
            # Strategy simulations
            scalping_results = self.simulate_scalping_strategy(symbol_data)
            swing_results = self.simulate_swing_strategy(symbol_data)
            
            # Store results
            self.results[symbol] = {
                'basic_metrics': basic_metrics,
                'intraday_patterns': intraday_patterns,
                'scalping': scalping_results,
                'swing': swing_results
            }
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        if not self.results:
            logger.error("No analysis results. Call analyze_all_assets() first.")
            return
        
        print("\n" + "="*80)
        print("CRYPTOCURRENCY TRADING ANALYSIS REPORT")
        print("="*80)
        
        # Create summary DataFrames
        basic_summary = []
        scalping_summary = []
        swing_summary = []
        
        for symbol, result in self.results.items():
            basic = result['basic_metrics']
            scalping = result['scalping']
            swing = result['swing']
            
            basic_summary.append({
                'Symbol': symbol,
                'Total_Return': basic['total_return'],
                'Volatility': basic['volatility'],
                'Avg_Volume': basic['avg_volume'],
                'Daily_Range': basic['avg_daily_range'],
                'Price_Range': basic['price_range']
            })
            
            scalping_summary.append({
                'Symbol': symbol,
                'Total_Trades': scalping['total_trades'],
                'Win_Rate': scalping['win_rate'],
                'Avg_Profit': scalping['avg_profit'],
                'Total_Return': scalping['total_return'],
                'Max_Drawdown': scalping['max_drawdown'],
                'Sharpe_Ratio': scalping['sharpe_ratio']
            })
            
            swing_summary.append({
                'Symbol': symbol,
                'Total_Trades': swing['total_trades'],
                'Win_Rate': swing['win_rate'],
                'Avg_Profit': swing['avg_profit'],
                'Total_Return': swing['total_return'],
                'Max_Drawdown': swing['max_drawdown'],
                'Sharpe_Ratio': swing['sharpe_ratio'],
                'Avg_Hold_Periods': swing['avg_hold_periods']
            })
        
        basic_df = pd.DataFrame(basic_summary)
        scalping_df = pd.DataFrame(scalping_summary)
        swing_df = pd.DataFrame(swing_summary)
        
        # Print basic metrics
        print("\nBASIC METRICS SUMMARY:")
        print("-" * 50)
        print(basic_df.round(4).to_string(index=False))
        
        # Print scalping results
        print("\nSCALPING STRATEGY RESULTS:")
        print("-" * 50)
        print(scalping_df.round(4).to_string(index=False))
        
        # Print swing results
        print("\nSWING TRADING STRATEGY RESULTS:")
        print("-" * 50)
        print(swing_df.round(4).to_string(index=False))
        
        # Top performers
        print("\nTOP PERFORMERS BY STRATEGY:")
        print("-" * 50)
        
        # Top scalping performers
        top_scalping = scalping_df.nlargest(5, 'Total_Return')[['Symbol', 'Total_Return', 'Win_Rate', 'Sharpe_Ratio']]
        print("Top 5 Scalping Performers:")
        print(top_scalping.round(4).to_string(index=False))
        
        # Top swing performers
        top_swing = swing_df.nlargest(5, 'Total_Return')[['Symbol', 'Total_Return', 'Win_Rate', 'Sharpe_Ratio']]
        print("\nTop 5 Swing Trading Performers:")
        print(top_swing.round(4).to_string(index=False))
        
        # Risk analysis
        print("\nRISK ANALYSIS:")
        print("-" * 50)
        low_volatility = basic_df.nsmallest(5, 'Volatility')[['Symbol', 'Volatility', 'Total_Return']]
        print("Lowest Volatility Assets:")
        print(low_volatility.round(4).to_string(index=False))
        
        high_volume = basic_df.nlargest(5, 'Avg_Volume')[['Symbol', 'Avg_Volume', 'Total_Return']]
        print("\nHighest Volume Assets:")
        print(high_volume.round(4).to_string(index=False))
        
        return {
            'basic_summary': basic_df,
            'scalping_summary': scalping_df,
            'swing_summary': swing_df
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
        
        # 1. Strategy Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Strategy Performance Comparison', fontsize=16)
        
        scalping_returns = [self.results[s]['scalping']['total_return'] for s in self.results.keys()]
        swing_returns = [self.results[s]['swing']['total_return'] for s in self.results.keys()]
        symbols = list(self.results.keys())
        
        # Scalping vs Swing Returns
        x = np.arange(len(symbols))
        width = 0.35
        axes[0, 0].bar(x - width/2, scalping_returns, width, label='Scalping', alpha=0.8)
        axes[0, 0].bar(x + width/2, swing_returns, width, label='Swing', alpha=0.8)
        axes[0, 0].set_xlabel('Assets')
        axes[0, 0].set_ylabel('Total Return')
        axes[0, 0].set_title('Total Returns by Strategy')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(symbols, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Win Rates
        scalping_wins = [self.results[s]['scalping']['win_rate'] for s in self.results.keys()]
        swing_wins = [self.results[s]['swing']['win_rate'] for s in self.results.keys()]
        
        axes[0, 1].bar(x - width/2, scalping_wins, width, label='Scalping', alpha=0.8)
        axes[0, 1].bar(x + width/2, swing_wins, width, label='Swing', alpha=0.8)
        axes[0, 1].set_xlabel('Assets')
        axes[0, 1].set_ylabel('Win Rate')
        axes[0, 1].set_title('Win Rates by Strategy')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(symbols, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sharpe Ratios
        scalping_sharpe = [self.results[s]['scalping']['sharpe_ratio'] for s in self.results.keys()]
        swing_sharpe = [self.results[s]['swing']['sharpe_ratio'] for s in self.results.keys()]
        
        axes[1, 0].bar(x - width/2, scalping_sharpe, width, label='Scalping', alpha=0.8)
        axes[1, 0].bar(x + width/2, swing_sharpe, width, label='Swing', alpha=0.8)
        axes[1, 0].set_xlabel('Assets')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].set_title('Risk-Adjusted Returns (Sharpe Ratio)')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(symbols, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Volatility vs Returns
        volatilities = [self.results[s]['basic_metrics']['volatility'] for s in self.results.keys()]
        axes[1, 1].scatter(volatilities, scalping_returns, alpha=0.7, label='Scalping', s=100)
        axes[1, 1].scatter(volatilities, swing_returns, alpha=0.7, label='Swing', s=100)
        axes[1, 1].set_xlabel('Volatility')
        axes[1, 1].set_ylabel('Total Return')
        axes[1, 1].set_title('Risk vs Return')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add asset labels to scatter plot
        for i, symbol in enumerate(symbols):
            axes[1, 1].annotate(symbol, (volatilities[i], scalping_returns[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/strategy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Intraday Patterns
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Intraday Trading Patterns', fontsize=16)
        
        # Sample a few assets for intraday patterns
        sample_symbols = list(self.results.keys())[:5]
        
        for i, symbol in enumerate(sample_symbols):
            hourly_vol = self.results[symbol]['intraday_patterns']['hourly_volume']
            hours = list(hourly_vol.keys())
            volumes = list(hourly_vol.values())
            
            axes[0, 0].plot(hours, volumes, label=symbol, marker='o', alpha=0.7)
        
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Average Volume')
        axes[0, 0].set_title('Hourly Volume Patterns')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Daily patterns
        for i, symbol in enumerate(sample_symbols):
            daily_vol = self.results[symbol]['intraday_patterns']['daily_volume']
            days = list(daily_vol.keys())
            volumes = list(daily_vol.values())
            
            axes[0, 1].plot(days, volumes, label=symbol, marker='s', alpha=0.7)
        
        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Average Volume')
        axes[0, 1].set_title('Daily Volume Patterns')
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Volatility patterns
        for i, symbol in enumerate(sample_symbols):
            hourly_vol = self.results[symbol]['intraday_patterns']['hourly_volatility']
            hours = list(hourly_vol.keys())
            vols = list(hourly_vol.values())
            
            axes[1, 0].plot(hours, vols, label=symbol, marker='^', alpha=0.7)
        
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Volatility')
        axes[1, 0].set_title('Hourly Volatility Patterns')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Volume vs Volatility correlation
        avg_volumes = [self.results[s]['basic_metrics']['avg_volume'] for s in sample_symbols]
        volatilities = [self.results[s]['basic_metrics']['volatility'] for s in sample_symbols]
        
        axes[1, 1].scatter(avg_volumes, volatilities, alpha=0.7, s=100)
        axes[1, 1].set_xlabel('Average Volume')
        axes[1, 1].set_ylabel('Volatility')
        axes[1, 1].set_title('Volume vs Volatility')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add labels
        for i, symbol in enumerate(sample_symbols):
            axes[1, 1].annotate(symbol, (avg_volumes[i], volatilities[i]), 
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
    analyzer = CryptoDataAnalyzer(latest_file)
    
    if not analyzer.load_data():
        return
    
    analyzer.analyze_all_assets()
    summary = analyzer.generate_summary_report()
    analyzer.create_visualizations()
    
    # Save results to CSV
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    summary['basic_summary'].to_csv(output_dir / "basic_metrics.csv", index=False)
    summary['scalping_summary'].to_csv(output_dir / "scalping_results.csv", index=False)
    summary['swing_summary'].to_csv(output_dir / "swing_results.csv", index=False)
    
    logger.info(f"Results saved to {output_dir}/")

if __name__ == "__main__":
    main()