"""
SR Levels Mock Data Generator

This module provides comprehensive mock data generation for the Support/Resistance levels system.
It generates realistic market data, SR levels, and trading scenarios for testing and development.
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import os


@dataclass
class SRLevel:
    """Represents a Support/Resistance level."""
    price: float
    level_type: str  # 'support' or 'resistance'
    strength: float
    touch_count: int
    first_touch: datetime
    last_touch: datetime
    bounce_rate: float
    isolation_score: float
    volume_at_level: float
    age_days: int


@dataclass
class MarketDataPoint:
    """Represents a single market data point."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float


class SRMockDataGenerator:
    """Comprehensive mock data generator for SR levels system."""
    
    def __init__(self, seed: int = 42):
        """Initialize the mock data generator with a seed for reproducibility."""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_market_data(
        self, 
        symbol: str = "ETHUSDT",
        data_points: int = 1000,
        start_price: float = 3000.0,
        volatility: float = 0.02,
        trend_strength: float = 0.001
    ) -> pd.DataFrame:
        """
        Generate realistic market data with OHLCV and VWAP.
        
        Args:
            symbol: Trading symbol
            data_points: Number of data points to generate
            start_price: Starting price
            volatility: Price volatility (standard deviation)
            trend_strength: Overall trend strength
            
        Returns:
            DataFrame with market data
        """
        # Generate timestamps (15-minute intervals)
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=15 * data_points)
        timestamps = pd.date_range(start=start_time, end=end_time, freq='15T')[:data_points]
        
        # Generate price data with trend and volatility
        prices = [start_price]
        for i in range(1, data_points):
            # Add trend component
            trend = trend_strength * (i / data_points)
            # Add random walk component
            random_change = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + trend + random_change)
            prices.append(max(new_price, 0.01))  # Ensure positive prices
        
        # Generate OHLC data
        data = []
        for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
            # Generate realistic OHLC from close price
            high_low_range = abs(np.random.normal(0, volatility * 0.5))
            high = close * (1 + high_low_range)
            low = close * (1 - high_low_range)
            
            # Ensure OHLC consistency
            if i > 0:
                open_price = data[-1]['close']
            else:
                open_price = close
                
            # Ensure high >= max(open, close) and low <= min(open, close)
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Generate volume (log-normal distribution)
            volume = np.random.lognormal(10, 0.5)
            
            # Calculate VWAP
            vwap = (high + low + close) / 3
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'vwap': vwap
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def generate_sr_levels(
        self, 
        market_data: pd.DataFrame,
        num_levels: int = 20,
        min_strength: float = 0.3,
        max_strength: float = 0.95
    ) -> List[SRLevel]:
        """
        Generate realistic Support/Resistance levels from market data.
        
        Args:
            market_data: Market data DataFrame
            num_levels: Number of SR levels to generate
            min_strength: Minimum level strength
            max_strength: Maximum level strength
            
        Returns:
            List of SRLevel objects
        """
        levels = []
        price_range = market_data['close'].max() - market_data['close'].min()
        min_price = market_data['close'].min()
        
        # Generate support levels (below current price)
        current_price = market_data['close'].iloc[-1]
        support_levels = int(num_levels * 0.6)  # 60% support levels
        
        for i in range(support_levels):
            # Generate price between min and current price
            price = min_price + (current_price - min_price) * np.random.random()
            
            # Generate level properties
            strength = np.random.uniform(min_strength, max_strength)
            touch_count = np.random.randint(2, 10)
            
            # Generate touch timestamps
            first_touch = market_data.index[0] + timedelta(
                days=np.random.randint(0, (market_data.index[-1] - market_data.index[0]).days)
            )
            last_touch = first_touch + timedelta(
                days=np.random.randint(0, (market_data.index[-1] - first_touch).days)
            )
            
            # Calculate bounce rate and isolation score
            bounce_rate = np.random.uniform(0.3, 0.9)
            isolation_score = np.random.uniform(0.2, 0.8)
            
            # Calculate volume at level
            volume_at_level = np.random.uniform(1000, 10000)
            
            # Calculate age
            age_days = (market_data.index[-1] - first_touch).days
            
            level = SRLevel(
                price=price,
                level_type='support',
                strength=strength,
                touch_count=touch_count,
                first_touch=first_touch,
                last_touch=last_touch,
                bounce_rate=bounce_rate,
                isolation_score=isolation_score,
                volume_at_level=volume_at_level,
                age_days=age_days
            )
            levels.append(level)
        
        # Generate resistance levels (above current price)
        resistance_levels = num_levels - support_levels
        max_price = market_data['close'].max()
        
        for i in range(resistance_levels):
            # Generate price between current and max price
            price = current_price + (max_price - current_price) * np.random.random()
            
            # Generate level properties
            strength = np.random.uniform(min_strength, max_strength)
            touch_count = np.random.randint(2, 8)
            
            # Generate touch timestamps
            first_touch = market_data.index[0] + timedelta(
                days=np.random.randint(0, (market_data.index[-1] - market_data.index[0]).days)
            )
            last_touch = first_touch + timedelta(
                days=np.random.randint(0, (market_data.index[-1] - first_touch).days)
            )
            
            # Calculate bounce rate and isolation score
            bounce_rate = np.random.uniform(0.3, 0.9)
            isolation_score = np.random.uniform(0.2, 0.8)
            
            # Calculate volume at level
            volume_at_level = np.random.uniform(1000, 10000)
            
            # Calculate age
            age_days = (market_data.index[-1] - first_touch).days
            
            level = SRLevel(
                price=price,
                level_type='resistance',
                strength=strength,
                touch_count=touch_count,
                first_touch=first_touch,
                last_touch=last_touch,
                bounce_rate=bounce_rate,
                isolation_score=isolation_score,
                volume_at_level=volume_at_level,
                age_days=age_days
            )
            levels.append(level)
        
        return levels
    
    def generate_trading_scenarios(
        self,
        market_data: pd.DataFrame,
        sr_levels: List[SRLevel],
        num_scenarios: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Generate realistic trading scenarios based on market data and SR levels.
        
        Args:
            market_data: Market data DataFrame
            sr_levels: List of SR levels
            num_scenarios: Number of scenarios to generate
            
        Returns:
            List of trading scenario dictionaries
        """
        scenarios = []
        current_price = market_data['close'].iloc[-1]
        
        for i in range(num_scenarios):
            # Select a random SR level
            level = random.choice(sr_levels)
            
            # Generate scenario type
            scenario_type = random.choice([
                'breakout', 'bounce', 'false_breakout', 'consolidation'
            ])
            
            # Generate scenario properties
            confidence = np.random.uniform(0.4, 0.95)
            risk_reward_ratio = np.random.uniform(1.5, 4.0)
            expected_duration_hours = np.random.randint(1, 72)
            
            # Generate position details
            position_size = np.random.uniform(0.01, 0.1)
            stop_loss_pct = np.random.uniform(0.01, 0.03)
            take_profit_pct = stop_loss_pct * risk_reward_ratio
            
            scenario = {
                'scenario_id': f"scenario_{i+1}",
                'timestamp': datetime.now(),
                'symbol': 'ETHUSDT',
                'sr_level': {
                    'price': level.price,
                    'type': level.level_type,
                    'strength': level.strength
                },
                'current_price': current_price,
                'scenario_type': scenario_type,
                'confidence': confidence,
                'risk_reward_ratio': risk_reward_ratio,
                'expected_duration_hours': expected_duration_hours,
                'position_size': position_size,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                'expected_pnl': position_size * (take_profit_pct - stop_loss_pct),
                'market_conditions': {
                    'volatility': market_data['close'].pct_change().std(),
                    'trend': 'bullish' if market_data['close'].iloc[-1] > market_data['close'].iloc[-20] else 'bearish',
                    'volume_trend': 'increasing' if market_data['volume'].iloc[-1] > market_data['volume'].iloc[-5] else 'decreasing'
                }
            }
            scenarios.append(scenario)
        
        return scenarios
    
    def generate_performance_metrics(
        self,
        scenarios: List[Dict[str, Any]],
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate realistic performance metrics for the SR system.
        
        Args:
            scenarios: List of trading scenarios
            days: Number of days for metrics calculation
            
        Returns:
            Dictionary of performance metrics
        """
        # Calculate basic metrics
        total_scenarios = len(scenarios)
        successful_scenarios = sum(1 for s in scenarios if s['confidence'] > 0.7)
        success_rate = successful_scenarios / total_scenarios if total_scenarios > 0 else 0
        
        # Generate PnL data
        total_pnl = sum(s['expected_pnl'] for s in scenarios)
        avg_pnl_per_scenario = total_pnl / total_scenarios if total_scenarios > 0 else 0
        
        # Generate risk metrics
        max_drawdown = np.random.uniform(0.05, 0.15)
        sharpe_ratio = np.random.uniform(1.2, 2.5)
        
        # Generate trading frequency
        trades_per_day = np.random.uniform(2, 8)
        total_trades = int(trades_per_day * days)
        
        metrics = {
            'period_days': days,
            'total_scenarios': total_scenarios,
            'successful_scenarios': successful_scenarios,
            'success_rate': success_rate,
            'total_pnl': total_pnl,
            'avg_pnl_per_scenario': avg_pnl_per_scenario,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trades_per_day': trades_per_day,
            'total_trades': total_trades,
            'win_rate': np.random.uniform(0.55, 0.75),
            'avg_win': np.random.uniform(0.02, 0.05),
            'avg_loss': np.random.uniform(-0.015, -0.025),
            'profit_factor': np.random.uniform(1.5, 3.0),
            'recovery_factor': np.random.uniform(2.0, 5.0)
        }
        
        return metrics
    
    def save_mock_data(
        self,
        market_data: pd.DataFrame,
        sr_levels: List[SRLevel],
        scenarios: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        output_dir: str = "data/mock_sr_data"
    ) -> None:
        """
        Save mock data to files for testing and development.
        
        Args:
            market_data: Market data DataFrame
            sr_levels: List of SR levels
            scenarios: List of trading scenarios
            metrics: Performance metrics
            output_dir: Output directory for mock data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save market data
        market_data.to_csv(os.path.join(output_dir, "market_data.csv"))
        
        # Save SR levels
        sr_levels_data = []
        for level in sr_levels:
            sr_levels_data.append({
                'price': level.price,
                'level_type': level.level_type,
                'strength': level.strength,
                'touch_count': level.touch_count,
                'first_touch': level.first_touch.isoformat(),
                'last_touch': level.last_touch.isoformat(),
                'bounce_rate': level.bounce_rate,
                'isolation_score': level.isolation_score,
                'volume_at_level': level.volume_at_level,
                'age_days': level.age_days
            })
        
        with open(os.path.join(output_dir, "sr_levels.json"), 'w') as f:
            json.dump(sr_levels_data, f, indent=2)
        
        # Save scenarios
        with open(os.path.join(output_dir, "trading_scenarios.json"), 'w') as f:
            json.dump(scenarios, f, indent=2, default=str)
        
        # Save metrics
        with open(os.path.join(output_dir, "performance_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Mock data saved to {output_dir}")
    
    def generate_complete_mock_dataset(
        self,
        data_points: int = 1000,
        num_sr_levels: int = 20,
        num_scenarios: int = 50,
        output_dir: str = "data/mock_sr_data"
    ) -> Dict[str, Any]:
        """
        Generate a complete mock dataset for the SR levels system.
        
        Args:
            data_points: Number of market data points
            num_sr_levels: Number of SR levels
            num_scenarios: Number of trading scenarios
            output_dir: Output directory for mock data
            
        Returns:
            Dictionary containing all generated mock data
        """
        print("Generating mock market data...")
        market_data = self.generate_market_data(data_points=data_points)
        
        print("Generating SR levels...")
        sr_levels = self.generate_sr_levels(market_data, num_levels=num_sr_levels)
        
        print("Generating trading scenarios...")
        scenarios = self.generate_trading_scenarios(market_data, sr_levels, num_scenarios)
        
        print("Generating performance metrics...")
        metrics = self.generate_performance_metrics(scenarios)
        
        print("Saving mock data...")
        self.save_mock_data(market_data, sr_levels, scenarios, metrics, output_dir)
        
        return {
            'market_data': market_data,
            'sr_levels': sr_levels,
            'scenarios': scenarios,
            'metrics': metrics
        }


def create_mock_data_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create mock data based on configuration settings.
    
    Args:
        config: Configuration dictionary with mock data settings
        
    Returns:
        Dictionary containing generated mock data
    """
    # Extract configuration
    enable_mock_data = config.get('testing', {}).get('enable_mock_data', False)
    mock_data_points = config.get('testing', {}).get('mock_data_points', 1000)
    mock_data_seed = config.get('testing', {}).get('mock_data_seed', 42)
    
    if not enable_mock_data:
        raise ValueError("Mock data is disabled in configuration")
    
    # Create generator with configured seed
    generator = SRMockDataGenerator(seed=mock_data_seed)
    
    # Generate mock data
    return generator.generate_complete_mock_dataset(
        data_points=mock_data_points,
        num_sr_levels=20,
        num_scenarios=50
    )


if __name__ == "__main__":
    # Example usage
    generator = SRMockDataGenerator(seed=42)
    mock_data = generator.generate_complete_mock_dataset(
        data_points=1000,
        num_sr_levels=20,
        num_scenarios=50
    )
    
    print("Mock data generation completed!")
    print(f"Generated {len(mock_data['market_data'])} market data points")
    print(f"Generated {len(mock_data['sr_levels'])} SR levels")
    print(f"Generated {len(mock_data['scenarios'])} trading scenarios")