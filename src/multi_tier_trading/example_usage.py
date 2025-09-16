"""
Example Usage of Multi-Tier Trading System

This script demonstrates how to use the complete multi-tier trading system
with proper scheduling, model configurations, and live execution.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import time

from src.utils.tprint import tprint
from src.multi_tier_trading import (
    create_live_execution_system,
    MultiTierModelConfigs,
    create_multi_tier_trading_orchestrator
)


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


def demonstrate_model_configurations():
    """Demonstrate the model configurations."""
    tprint("=" * 80)
    tprint("MULTI-TIER TRADING SYSTEM MODEL CONFIGURATIONS")
    tprint("=" * 80)
    
    # Print configuration summary
    MultiTierModelConfigs.print_config_summary()
    
    # Validate configurations
    configs = MultiTierModelConfigs.get_all_configs()
    for tier_name, config in configs.items():
        tprint(f"\nValidating {tier_name} configuration...")
        MultiTierModelConfigs.validate_config(config)


def demonstrate_orchestrator():
    """Demonstrate the orchestrator system."""
    tprint("\n" + "=" * 80)
    tprint("MULTI-TIER TRADING ORCHESTRATOR DEMONSTRATION")
    tprint("=" * 80)
    
    # Generate sample data
    data_1h, data_5m, data_1m = generate_sample_data()
    
    # Create orchestrator
    orchestrator = create_multi_tier_trading_orchestrator()
    
    # Load data
    orchestrator.load_data(data_1h, data_5m, data_1m)
    
    # Train systems
    tprint("Training all systems...")
    training_results = asyncio.run(orchestrator.train_systems())
    
    # Print training results
    tprint("\nTraining Results:")
    for system, results in training_results.items():
        tprint(f"  {system}: {results}")
    
    # Run single cycle
    tprint("\nRunning single analysis cycle...")
    decision = asyncio.run(orchestrator.run_single_cycle())
    
    if decision:
        tprint(f"Trading Decision: {decision.decision_reasoning}")
        tprint(f"  Should Trade: {decision.should_trade}")
        tprint(f"  Confidence: {decision.entry_confidence:.3f}")
        tprint(f"  Expected Return: {decision.expected_return:.3f}%")
    else:
        tprint("No trading decision available yet")
    
    # Get system status
    status = orchestrator.get_system_status()
    tprint(f"\nSystem Status: {status}")


def demonstrate_live_execution():
    """Demonstrate the live execution system."""
    tprint("\n" + "=" * 80)
    tprint("LIVE EXECUTION SYSTEM DEMONSTRATION")
    tprint("=" * 80)
    
    # Generate sample data
    data_1h, data_5m, data_1m = generate_sample_data()
    
    # Create live execution system
    live_system = create_live_execution_system()
    
    # Load data
    live_system.load_data(data_1h, data_5m, data_1m)
    
    # Start live execution
    tprint("Starting live execution system...")
    live_system.start_live_execution()
    
    # Run for a few cycles
    tprint("Running live execution for 10 cycles...")
    for i in range(10):
        time.sleep(2)  # Wait 2 seconds between cycles
        
        # Get latest decision
        decision = live_system.get_latest_decision()
        if decision:
            status = "TRADE" if decision.should_trade else "WAIT"
            tprint(f"Cycle {i+1}: {status} - {decision.decision_reasoning}")
        else:
            tprint(f"Cycle {i+1}: No decision yet")
        
        # Print status every 3 cycles
        if (i + 1) % 3 == 0:
            status = live_system.get_execution_status()
            tprint(f"  Status: {status['status']}")
            tprint(f"  HMM runs: {status['metrics']['hmm_runs']}")
            tprint(f"  Analyst runs: {status['metrics']['analyst_runs']}")
            tprint(f"  Tactician runs: {status['metrics']['tactician_runs']}")
            tprint(f"  Green lights: {status['metrics']['green_lights']}")
            tprint(f"  Trade signals: {status['metrics']['trade_signals']}")
    
    # Stop live execution
    tprint("Stopping live execution system...")
    live_system.stop_live_execution()
    
    # Get final status
    final_status = live_system.get_execution_status()
    tprint(f"\nFinal Status: {final_status}")
    
    # Get decision history
    decisions = live_system.get_decision_history(5)
    tprint(f"\nRecent Decisions ({len(decisions)}):")
    for i, decision in enumerate(decisions):
        tprint(f"  {i+1}. {decision.timestamp}: {decision.decision_reasoning}")


def demonstrate_feature_extraction():
    """Demonstrate the feature extraction systems."""
    tprint("\n" + "=" * 80)
    tprint("FEATURE EXTRACTION DEMONSTRATION")
    tprint("=" * 80)
    
    # Generate sample data
    data_1h, data_5m, data_1m = generate_sample_data()
    
    # HMM Feature Extraction
    tprint("\nHMM Feature Extraction (100 features, 1h base):")
    from src.multi_tier_trading import create_hmm_feature_extractor
    hmm_extractor = create_hmm_feature_extractor()
    hmm_features = hmm_extractor.extract_features(data_1h)
    tprint(f"  Extracted {len(hmm_features.columns)} features")
    tprint(f"  Feature names: {hmm_features.columns.tolist()[:10]}...")
    
    # Analyst Feature Extraction
    tprint("\nAnalyst Feature Extraction (300+ features, 5m base):")
    from src.multi_tier_trading import create_analyst_feature_extractor
    analyst_extractor = create_analyst_feature_extractor()
    
    # Mock HMM output
    hmm_output = {
        'regime_probs': np.random.dirichlet(np.ones(20)),
        'dominant_regime': 5,
        'confidence': 0.8,
        'regime_characteristics': {
            'mean_returns': 0.01,
            'volatility': 0.03,
            'mean_volume': 1.1
        }
    }
    
    analyst_features = analyst_extractor.extract_features(data_5m, hmm_output)
    tprint(f"  Extracted {len(analyst_features.columns)} features")
    tprint(f"  Feature names: {analyst_features.columns.tolist()[:10]}...")
    
    # Tactician Feature Extraction
    tprint("\nTactician Feature Extraction (50+ features, 1m base):")
    from src.multi_tier_trading import create_tactician_feature_extractor
    tactician_extractor = create_tactician_feature_extractor()
    
    # Mock Analyst output
    analyst_output = {
        'should_trade': True,
        'confidence': 0.75,
        'meta_learner_prediction': 0.3,
        'regime_id': 5
    }
    
    tactician_features = tactician_extractor.extract_features(data_1m, hmm_output, analyst_output)
    tprint(f"  Extracted {len(tactician_features.columns)} features")
    tprint(f"  Feature names: {tactician_features.columns.tolist()[:10]}...")


def main():
    """Main demonstration function."""
    tprint("MULTI-TIER TRADING SYSTEM DEMONSTRATION")
    tprint("=" * 80)
    
    try:
        # Demonstrate model configurations
        demonstrate_model_configurations()
        
        # Demonstrate feature extraction
        demonstrate_feature_extraction()
        
        # Demonstrate orchestrator
        demonstrate_orchestrator()
        
        # Demonstrate live execution
        demonstrate_live_execution()
        
        tprint("\n" + "=" * 80)
        tprint("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        tprint("=" * 80)
        
    except Exception as e:
        tprint(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()