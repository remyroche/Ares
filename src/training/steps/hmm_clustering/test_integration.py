#!/usr/bin/env python3
"""Test integration of enhanced HMM clustering with the pipeline."""

import asyncio
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.hmm_clustering import HMMRegimeDiscoveryStep
from src.utils.logger import system_logger

logger = system_logger.getChild("TestIntegration")


async def test_hmm_integration():
    """Test the HMM clustering integration."""
    logger.info("🧪 Testing HMM Clustering Integration")
    logger.info("=" * 80)
    
    # Test configuration
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'BINANCE',
        'timeframe': '1m',
        'data_dir': 'data_cache',
        'n_trials': 5,  # Reduced for testing
        'timeout_minutes': 5
    }
    
    # Initialize the step
    logger.info("1️⃣ Initializing HMM Regime Discovery Step...")
    step = HMMRegimeDiscoveryStep(config)
    
    try:
        await step.initialize()
        logger.info("✅ Initialization successful")
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return False
        
    # Create test data
    logger.info("\n2️⃣ Creating test data...")
    n_periods = 1000
    timestamps = pd.date_range(start='2024-01-01', periods=n_periods, freq='1min')
    
    # Simulate price data with regime changes
    regime_changes = [0, 300, 600, 800]
    regimes = np.zeros(n_periods)
    for i in range(len(regime_changes) - 1):
        regimes[regime_changes[i]:regime_changes[i+1]] = i
    regimes[regime_changes[-1]:] = len(regime_changes) - 1
    
    # Generate price data based on regimes
    prices = np.zeros(n_periods)
    price = 100
    for i in range(n_periods):
        if regimes[i] == 0:  # Trending up
            price += np.random.normal(0.1, 0.5)
        elif regimes[i] == 1:  # Trending down
            price -= np.random.normal(0.1, 0.5)
        elif regimes[i] == 2:  # High volatility
            price += np.random.normal(0, 2.0)
        else:  # Low volatility
            price += np.random.normal(0, 0.2)
        prices[i] = max(price, 1)  # Ensure positive prices
        
    # Create DataFrame
    test_data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices * np.random.uniform(0.99, 1.01, n_periods),
        'high': prices * np.random.uniform(1.0, 1.02, n_periods),
        'low': prices * np.random.uniform(0.98, 1.0, n_periods),
        'close': prices,
        'volume': np.random.exponential(1000, n_periods)
    })
    
    logger.info(f"✅ Created test data: {len(test_data)} periods")
    logger.info(f"   Price range: ${test_data['close'].min():.2f} - ${test_data['close'].max():.2f}")
    
    # Create training input and pipeline state
    training_input = {
        'symbol': config['symbol'],
        'exchange': config['exchange'],
        'timeframe': config['timeframe'],
        'data_dir': config['data_dir']
    }
    
    pipeline_state = {
        'data': test_data,
        'validated_data': test_data
    }
    
    # Test execution
    logger.info("\n3️⃣ Executing HMM Regime Discovery...")
    try:
        result = await step.execute(training_input, pipeline_state)
        
        if result.get('hmm_regime_discovery_completed', False):
            logger.info("✅ Execution successful")
            
            # Check results
            logger.info("\n4️⃣ Checking results...")
            
            # Check regime states
            if 'regime_states' in result or 'regime_labels' in result:
                regime_states = result.get('regime_states', result.get('regime_labels', []))
                logger.info(f"✅ Regime states found: {len(regime_states)} periods")
                
                unique_regimes = np.unique(regime_states)
                logger.info(f"✅ Unique regimes: {len(unique_regimes)} ({unique_regimes})")
                
                # Check regime distribution
                if 'regime_distribution' in result:
                    logger.info("✅ Regime distribution:")
                    for regime, count in result['regime_distribution'].items():
                        pct = (count / len(regime_states)) * 100
                        logger.info(f"   {regime}: {count} periods ({pct:.1f}%)")
            else:
                logger.error("❌ No regime states found in results")
                return False
                
            # Check transitions
            if 'regime_transitions' in result:
                transitions = result['regime_transitions']
                logger.info(f"✅ Transitions: {transitions.get('total_transitions', 0)}")
                logger.info(f"✅ Transition rate: {transitions.get('transition_rate', 0):.4f}")
            
            # Check enhanced features
            if 'enhanced_ml_transition_detection' in result:
                logger.info(f"✅ Enhanced ML transition detection: {result['enhanced_ml_transition_detection']}")
                
            if 'economic_significance' in result:
                logger.info(f"✅ Economic significance: {result['economic_significance']}")
                
            # Check quality metrics
            if 'overall_quality_score' in result:
                logger.info(f"✅ Overall quality score: {result['overall_quality_score']:.4f}")
                
            logger.info("\n🎉 All tests passed!")
            return True
            
        else:
            logger.error("❌ Execution failed")
            if 'regime_discovery_error' in result:
                logger.error(f"   Error: {result['regime_discovery_error']}")
            return False
            
    except Exception as e:
        logger.exception(f"❌ Execution error: {e}")
        return False


async def main():
    """Main test function."""
    success = await test_hmm_integration()
    
    logger.info("\n" + "=" * 80)
    if success:
        logger.info("✅ HMM CLUSTERING INTEGRATION TEST: PASSED")
    else:
        logger.info("❌ HMM CLUSTERING INTEGRATION TEST: FAILED")
    logger.info("=" * 80)
    
    return success


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(main())
    sys.exit(0 if success else 1)