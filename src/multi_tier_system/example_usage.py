"""
Example usage of the Multi-Tier Trading System

This script demonstrates how to use the HMM, Analyst, and Tactician systems
in a coordinated trading pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.multi_tier_system.trading_orchestrator import create_multi_tier_trading_orchestrator
from src.utils.tprint import tprint


def generate_sample_data(n_1h: int = 1000, n_5m: int = 12000, n_1m: int = 60000) -> tuple:
    """Generate sample market data for testing."""
    tprint("Generating sample market data...")
    
    # Generate timestamps
    end_time = datetime.now()
    
    # 1h data
    timestamps_1h = pd.date_range(end=end_time, periods=n_1h, freq='1H')
    # 5m data
    timestamps_5m = pd.date_range(end=end_time, periods=n_5m, freq='5T')
    # 1m data
    timestamps_1m = pd.date_range(end=end_time, periods=n_1m, freq='1T')
    
    # Generate price data with some realistic patterns
    np.random.seed(42)
    
    # Base price
    base_price = 2000.0
    
    # Generate 1h data
    returns_1h = np.random.normal(0, 0.02, n_1h)  # 2% hourly volatility
    prices_1h = base_price * np.exp(np.cumsum(returns_1h))
    
    data_1h = pd.DataFrame({
        'timestamp': timestamps_1h,
        'open': prices_1h * (1 + np.random.normal(0, 0.001, n_1h)),
        'high': prices_1h * (1 + np.abs(np.random.normal(0, 0.01, n_1h))),
        'low': prices_1h * (1 - np.abs(np.random.normal(0, 0.01, n_1h))),
        'close': prices_1h,
        'volume': np.random.lognormal(10, 1, n_1h)
    })
    data_1h.set_index('timestamp', inplace=True)
    
    # Generate 5m data (more volatile)
    returns_5m = np.random.normal(0, 0.005, n_5m)  # 0.5% 5-minute volatility
    prices_5m = base_price * np.exp(np.cumsum(returns_5m))
    
    data_5m = pd.DataFrame({
        'timestamp': timestamps_5m,
        'open': prices_5m * (1 + np.random.normal(0, 0.0005, n_5m)),
        'high': prices_5m * (1 + np.abs(np.random.normal(0, 0.005, n_5m))),
        'low': prices_5m * (1 - np.abs(np.random.normal(0, 0.005, n_5m))),
        'close': prices_5m,
        'volume': np.random.lognormal(8, 1, n_5m)
    })
    data_5m.set_index('timestamp', inplace=True)
    
    # Generate 1m data (most volatile)
    returns_1m = np.random.normal(0, 0.002, n_1m)  # 0.2% 1-minute volatility
    prices_1m = base_price * np.exp(np.cumsum(returns_1m))
    
    data_1m = pd.DataFrame({
        'timestamp': timestamps_1m,
        'open': prices_1m * (1 + np.random.normal(0, 0.0002, n_1m)),
        'high': prices_1m * (1 + np.abs(np.random.normal(0, 0.002, n_1m))),
        'low': prices_1m * (1 - np.abs(np.random.normal(0, 0.002, n_1m))),
        'close': prices_1m,
        'volume': np.random.lognormal(6, 1, n_1m)
    })
    data_1m.set_index('timestamp', inplace=True)
    
    tprint(f"Generated data: 1h={len(data_1h)}, 5m={len(data_5m)}, 1m={len(data_1m)}")
    
    return data_1h, data_5m, data_1m


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'multi_tier_config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def run_training_example():
    """Run training example."""
    tprint("=== Multi-Tier Trading System Training Example ===")
    
    # Load configuration
    config = load_config()
    
    # Generate sample data
    data_1h, data_5m, data_1m = generate_sample_data()
    
    # Create orchestrator
    orchestrator = create_multi_tier_trading_orchestrator(config)
    
    # Load data
    orchestrator.load_data(data_1h, data_5m, data_1m)
    
    # Train systems
    tprint("Training all systems...")
    training_results = orchestrator.train_systems()
    
    # Print training results
    tprint("\n=== Training Results ===")
    for system, results in training_results.items():
        tprint(f"\n{system.upper()} System:")
        if isinstance(results, dict):
            for key, value in results.items():
                if isinstance(value, dict):
                    tprint(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        tprint(f"    {sub_key}: {sub_value}")
                else:
                    tprint(f"  {key}: {value}")
        else:
            tprint(f"  {results}")
    
    return orchestrator


def run_live_trading_example():
    """Run live trading simulation example."""
    tprint("=== Multi-Tier Trading System Live Example ===")
    
    # Load configuration
    config = load_config()
    
    # Generate sample data
    data_1h, data_5m, data_1m = generate_sample_data()
    
    # Create orchestrator
    orchestrator = create_multi_tier_trading_orchestrator(config)
    
    # Load data
    orchestrator.load_data(data_1h, data_5m, data_1m)
    
    # Train systems
    tprint("Training all systems...")
    orchestrator.train_systems()
    
    # Start system
    tprint("Starting trading system...")
    orchestrator.start_system()
    
    # Run for a few cycles
    tprint("Running system for 10 cycles...")
    for i in range(10):
        decision = orchestrator.run_single_cycle()
        
        if decision:
            status = "TRADE" if decision.should_trade else "WAIT"
            tprint(f"Cycle {i+1}: {status} - {decision.decision_reasoning}")
        else:
            tprint(f"Cycle {i+1}: No decision yet")
        
        # Simulate time passing
        import time
        time.sleep(2)
    
    # Stop system
    orchestrator.stop_system()
    
    # Print final status
    status = orchestrator.get_system_status()
    tprint(f"\n=== Final System Status ===")
    for key, value in status.items():
        if isinstance(value, dict):
            tprint(f"{key}:")
            for sub_key, sub_value in value.items():
                tprint(f"  {sub_key}: {sub_value}")
        else:
            tprint(f"{key}: {value}")
    
    # Print recent decisions
    decisions = orchestrator.get_decision_history(5)
    tprint(f"\n=== Recent Decisions ===")
    for i, decision in enumerate(decisions):
        tprint(f"Decision {i+1}: {decision.timestamp}")
        tprint(f"  Should Trade: {decision.should_trade}")
        tprint(f"  Confidence: {decision.entry_confidence:.3f}")
        tprint(f"  Expected Return: {decision.expected_return:.3f}%")
        tprint(f"  Reasoning: {decision.decision_reasoning}")
        tprint()


def run_individual_system_examples():
    """Run examples for individual systems."""
    tprint("=== Individual System Examples ===")
    
    # Generate sample data
    data_1h, data_5m, data_1m = generate_sample_data()
    
    # HMM System Example
    tprint("\n--- HMM System Example ---")
    from src.hmm_system.hmm_regime_detector import create_hmm_regime_detector
    
    hmm_system = create_hmm_regime_detector()
    hmm_results = hmm_system.train_models(data_1h)
    tprint(f"HMM Training Results: {hmm_results}")
    
    hmm_prediction = hmm_system.predict_regime_probabilities(data_1h)
    tprint(f"HMM Prediction: Regime {hmm_prediction.dominant_regime}, "
          f"Confidence: {hmm_prediction.confidence:.3f}")
    
    # Analyst System Example
    tprint("\n--- Analyst System Example ---")
    from src.analyst_system.analyst_regime_predictor import create_analyst_regime_predictor
    
    analyst_system = create_analyst_regime_predictor()
    
    # Create dummy regime labels for training
    regime_labels = np.random.randint(0, 5, len(data_5m))
    analyst_results = analyst_system.train_regime_models(data_5m, regime_labels)
    tprint(f"Analyst Training Results: {analyst_results}")
    
    analyst_prediction = analyst_system.predict_trading_opportunity(data_5m, 0)
    tprint(f"Analyst Prediction: {'GREEN LIGHT' if analyst_prediction.should_trade else 'RED LIGHT'}, "
          f"Confidence: {analyst_prediction.confidence:.3f}")
    
    # Tactician System Example
    tprint("\n--- Tactician System Example ---")
    from src.tactician_system.tactician_timing_predictor import create_tactician_timing_predictor
    
    tactician_system = create_tactician_timing_predictor()
    
    # Create dummy green lights for training
    green_lights = np.random.random(len(data_1m)) < 0.2
    tactician_results = tactician_system.train_models(data_1m, green_lights)
    tprint(f"Tactician Training Results: {tactician_results}")
    
    tactician_prediction = tactician_system.predict_entry_timing(data_1m)
    tprint(f"Tactician Prediction: {'ENTER' if tactician_prediction.should_enter else 'WAIT'}, "
          f"Confidence: {tactician_prediction.entry_confidence:.3f}")


def main():
    """Main function to run examples."""
    tprint("Multi-Tier Trading System Examples")
    tprint("=" * 50)
    
    try:
        # Run individual system examples
        run_individual_system_examples()
        
        # Run training example
        run_training_example()
        
        # Run live trading example
        run_live_trading_example()
        
        tprint("\n=== All Examples Completed Successfully ===")
        
    except Exception as e:
        tprint(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()