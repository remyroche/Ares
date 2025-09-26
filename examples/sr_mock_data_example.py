#!/usr/bin/env python3
"""
SR Levels Mock Data Example

This example demonstrates how to use the SR levels mock data functionality
for testing and development purposes.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.sr_mock_data_generator import SRMockDataGenerator
from config.sr_mock_data_config import SRMockDataConfig, create_mock_data_from_sr_config
from integration.sr_mock_data_integration import SRMockDataIntegration, SRMockDataManager


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def example_basic_usage():
    """Example of basic mock data generation."""
    print("=== Basic Mock Data Generation ===")
    
    # Create generator with seed for reproducibility
    generator = SRMockDataGenerator(seed=42)
    
    # Generate market data
    print("Generating market data...")
    market_data = generator.generate_market_data(
        symbol="ETHUSDT",
        data_points=500,
        start_price=3000.0,
        volatility=0.02
    )
    
    print(f"Generated {len(market_data)} market data points")
    print(f"Price range: ${market_data['close'].min():.2f} - ${market_data['close'].max():.2f}")
    print(f"Average volume: {market_data['volume'].mean():.2f}")
    
    # Generate SR levels
    print("\nGenerating SR levels...")
    sr_levels = generator.generate_sr_levels(market_data, num_levels=15)
    
    support_levels = [level for level in sr_levels if level.level_type == 'support']
    resistance_levels = [level for level in sr_levels if level.level_type == 'resistance']
    
    print(f"Generated {len(support_levels)} support levels")
    print(f"Generated {len(resistance_levels)} resistance levels")
    
    # Show strongest levels
    strongest_support = max(support_levels, key=lambda x: x.strength)
    strongest_resistance = max(resistance_levels, key=lambda x: x.strength)
    
    print(f"Strongest support: ${strongest_support.price:.2f} (strength: {strongest_support.strength:.2f})")
    print(f"Strongest resistance: ${strongest_resistance.price:.2f} (strength: {strongest_resistance.strength:.2f})")
    
    # Generate trading scenarios
    print("\nGenerating trading scenarios...")
    scenarios = generator.generate_trading_scenarios(market_data, sr_levels, num_scenarios=20)
    
    breakout_scenarios = [s for s in scenarios if s['scenario_type'] == 'breakout']
    bounce_scenarios = [s for s in scenarios if s['scenario_type'] == 'bounce']
    
    print(f"Generated {len(breakout_scenarios)} breakout scenarios")
    print(f"Generated {len(bounce_scenarios)} bounce scenarios")
    
    # Show high-confidence scenarios
    high_confidence = [s for s in scenarios if s['confidence'] > 0.8]
    print(f"High-confidence scenarios: {len(high_confidence)}")
    
    if high_confidence:
        best_scenario = max(high_confidence, key=lambda x: x['risk_reward_ratio'])
        print(f"Best scenario: {best_scenario['scenario_type']} with R:R {best_scenario['risk_reward_ratio']:.2f}")
    
    return market_data, sr_levels, scenarios


def example_configuration_usage():
    """Example of using mock data with configuration."""
    print("\n=== Configuration-Based Mock Data ===")
    
    # Create a temporary configuration
    config_data = {
        'testing': {
            'enable_mock_data': True,
            'mock_data_points': 1000,
            'mock_data_seed': 42,
            'mock_data_output_dir': 'data/mock_sr_data'
        },
        'sr_levels_manager': {
            'max_levels': 20,
            'min_strength': 0.3,
            'proximity_threshold': 0.005
        },
        'data_integration': {
            'symbol': 'ETHUSDT',
            'exchange': 'BINANCE',
            'timeframes': ['1m', '5m', '15m']
        }
    }
    
    # Generate mock data from configuration
    print("Generating mock data from configuration...")
    mock_data = create_mock_data_from_config(config_data)
    
    print(f"Generated dataset with:")
    print(f"  - {len(mock_data['market_data'])} market data points")
    print(f"  - {len(mock_data['sr_levels'])} SR levels")
    print(f"  - {len(mock_data['scenarios'])} trading scenarios")
    print(f"  - Performance metrics: {list(mock_data['metrics'].keys())}")
    
    return mock_data


def example_integration_usage():
    """Example of using mock data integration."""
    print("\n=== Mock Data Integration ===")
    
    # Create a temporary configuration file
    import tempfile
    import yaml
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_config:
        config_data = {
            'testing': {
                'enable_mock_data': True,
                'mock_data_points': 500,
                'mock_data_seed': 42,
                'mock_data_output_dir': 'data/mock_sr_data'
            },
            'sr_levels_manager': {
                'max_levels': 15,
                'min_strength': 0.3
            },
            'data_integration': {
                'symbol': 'ETHUSDT',
                'exchange': 'BINANCE'
            }
        }
        
        yaml.dump(config_data, temp_config, default_flow_style=False, indent=2)
        temp_config_path = temp_config.name
    
    try:
        # Create integration
        integration = SRMockDataIntegration(temp_config_path)
        
        # Initialize mock data
        print("Initializing mock data...")
        if integration.initialize_mock_data():
            print("Mock data initialized successfully")
            
            # Get data summary
            summary = integration.get_mock_data_summary()
            print(f"Data summary: {summary}")
            
            # Access different data types
            market_data = integration.get_market_data()
            sr_levels = integration.get_sr_levels()
            scenarios = integration.get_trading_scenarios()
            metrics = integration.get_performance_metrics()
            
            print(f"Market data shape: {market_data.shape}")
            print(f"SR levels count: {len(sr_levels)}")
            print(f"Scenarios count: {len(scenarios)}")
            print(f"Success rate: {metrics.get('success_rate', 0):.2%}")
            
            # Export data
            export_dir = "data/exported_mock_data"
            if integration.export_mock_data(export_dir):
                print(f"Mock data exported to {export_dir}")
            
        else:
            print("Failed to initialize mock data")
    
    finally:
        # Clean up temporary file
        os.unlink(temp_config_path)


def example_manager_usage():
    """Example of using mock data manager."""
    print("\n=== Mock Data Manager ===")
    
    # Create a temporary configuration file
    import tempfile
    import yaml
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_config:
        config_data = {
            'testing': {
                'enable_mock_data': True,
                'mock_data_points': 300,
                'mock_data_seed': 42,
                'mock_data_output_dir': 'data/mock_sr_data'
            },
            'sr_levels_manager': {
                'max_levels': 10
            },
            'data_integration': {
                'symbol': 'ETHUSDT',
                'exchange': 'BINANCE'
            }
        }
        
        yaml.dump(config_data, temp_config, default_flow_style=False, indent=2)
        temp_config_path = temp_config.name
    
    try:
        # Create manager
        manager = SRMockDataManager(temp_config_path)
        
        # Start service
        print("Starting mock data service...")
        if manager.start_mock_data_service():
            print("Service started successfully")
            
            # Get service status
            status = manager.get_service_status()
            print(f"Service status: {status['running']}")
            print(f"Data summary: {status['data_summary']}")
            
            # Export all data
            export_dir = "data/manager_export"
            if manager.export_all_mock_data(export_dir):
                print(f"All mock data exported to {export_dir}")
            
            # Stop service
            print("Stopping mock data service...")
            manager.stop_mock_data_service()
            print("Service stopped")
        
        else:
            print("Failed to start mock data service")
    
    finally:
        # Clean up temporary file
        os.unlink(temp_config_path)


def example_advanced_usage():
    """Example of advanced mock data usage."""
    print("\n=== Advanced Mock Data Usage ===")
    
    # Create generator with custom parameters
    generator = SRMockDataGenerator(seed=123)
    
    # Generate multiple datasets with different characteristics
    datasets = {}
    
    # Bullish market
    print("Generating bullish market data...")
    bullish_data = generator.generate_market_data(
        symbol="ETHUSDT",
        data_points=200,
        start_price=3000.0,
        volatility=0.015,
        trend_strength=0.002  # Slight upward trend
    )
    datasets['bullish'] = bullish_data
    
    # Bearish market
    print("Generating bearish market data...")
    bearish_data = generator.generate_market_data(
        symbol="ETHUSDT",
        data_points=200,
        start_price=3000.0,
        volatility=0.025,
        trend_strength=-0.001  # Slight downward trend
    )
    datasets['bearish'] = bearish_data
    
    # High volatility market
    print("Generating high volatility market data...")
    volatile_data = generator.generate_market_data(
        symbol="ETHUSDT",
        data_points=200,
        start_price=3000.0,
        volatility=0.05,  # High volatility
        trend_strength=0.0  # No trend
    )
    datasets['volatile'] = volatile_data
    
    # Analyze each dataset
    for name, data in datasets.items():
        print(f"\n{name.capitalize()} market analysis:")
        print(f"  Price change: {((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100:.2f}%")
        print(f"  Volatility: {data['close'].pct_change().std() * 100:.2f}%")
        print(f"  Volume trend: {'Increasing' if data['volume'].iloc[-5:].mean() > data['volume'].iloc[:5].mean() else 'Decreasing'}")
        
        # Generate SR levels for each dataset
        sr_levels = generator.generate_sr_levels(data, num_levels=8)
        support_count = len([l for l in sr_levels if l.level_type == 'support'])
        resistance_count = len([l for l in sr_levels if l.level_type == 'resistance'])
        
        print(f"  SR levels: {support_count} support, {resistance_count} resistance")
        
        # Generate scenarios
        scenarios = generator.generate_trading_scenarios(data, sr_levels, num_scenarios=15)
        high_confidence = len([s for s in scenarios if s['confidence'] > 0.7])
        print(f"  High-confidence scenarios: {high_confidence}")


def main():
    """Main function to run all examples."""
    setup_logging()
    
    print("SR Levels Mock Data Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_basic_usage()
        example_configuration_usage()
        example_integration_usage()
        example_manager_usage()
        example_advanced_usage()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()