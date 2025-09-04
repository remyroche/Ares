#!/usr/bin/env python3
"""Full integration test for enhanced HMM clustering."""

import asyncio
import sys
from pathlib import Path
import json
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.hmm_clustering import HMMRegimeDiscoveryStep, run_validator
from src.utils.logger import system_logger

logger = system_logger.getChild("FullIntegrationTest")


async def test_full_integration():
    """Test the complete HMM clustering integration."""
    logger.info("🧪 Testing Full HMM Clustering Integration")
    logger.info("=" * 80)
    
    # Clean up any existing test data
    test_data_dir = Path("data/hmm_regimes")
    if test_data_dir.exists():
        logger.info("🧹 Cleaning up existing test data...")
        shutil.rmtree(test_data_dir)
    
    # Test configuration
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'BINANCE', 
        'timeframe': '1m',
        'data_dir': 'data',
        'n_trials': 2,  # Minimal for testing
        'timeout_minutes': 5
    }
    
    # Initialize the step
    logger.info("\n1️⃣ Initializing HMM Regime Discovery Step...")
    step = HMMRegimeDiscoveryStep(config)
    
    try:
        await step.initialize()
        logger.info("✅ Initialization successful")
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return False
    
    # Create mock pipeline state with minimal test data
    logger.info("\n2️⃣ Creating mock pipeline state...")
    
    # Mock data that simulates different market regimes
    import numpy as np
    import pandas as pd
    
    n_periods = 500
    timestamps = pd.date_range(start='2024-01-01', periods=n_periods, freq='1min')
    
    # Create simple regime-based data
    regime_changes = [0, 150, 300, 400]
    prices = []
    price = 100
    
    for i in range(n_periods):
        if i < regime_changes[1]:  # Uptrend
            price += np.random.normal(0.05, 0.2)
        elif i < regime_changes[2]:  # Downtrend
            price -= np.random.normal(0.05, 0.2)
        elif i < regime_changes[3]:  # High volatility
            price += np.random.normal(0, 0.5)
        else:  # Low volatility
            price += np.random.normal(0, 0.1)
        prices.append(max(price, 10))  # Ensure positive
    
    test_data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': np.random.exponential(1000, n_periods)
    })
    
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
    
    logger.info(f"✅ Created test data: {len(test_data)} periods")
    
    # Execute HMM clustering
    logger.info("\n3️⃣ Executing HMM Regime Discovery...")
    
    try:
        result = await step.execute(training_input, pipeline_state)
        
        if result.get('hmm_regime_discovery_completed', False):
            logger.info("✅ Execution successful")
        else:
            logger.error("❌ Execution failed")
            if 'regime_discovery_error' in result:
                logger.error(f"   Error: {result['regime_discovery_error']}")
            return False
            
    except Exception as e:
        logger.exception(f"❌ Execution error: {e}")
        return False
    
    # Validate results
    logger.info("\n4️⃣ Validating results...")
    
    try:
        validation_passed = await run_validator(result, config)
        
        if validation_passed:
            logger.info("✅ Validation passed")
        else:
            logger.error("❌ Validation failed")
            return False
            
    except Exception as e:
        logger.exception(f"❌ Validation error: {e}")
        return False
    
    # Check saved files
    logger.info("\n5️⃣ Checking saved files...")
    
    saved_files = result.get('saved_files', {})
    if saved_files:
        logger.info(f"✅ Found {len(saved_files)} saved files:")
        
        all_exist = True
        for file_type, file_path in saved_files.items():
            exists = Path(file_path).exists()
            status = "✅" if exists else "❌"
            logger.info(f"   {status} {file_type}: {file_path}")
            
            if not exists:
                all_exist = False
                
        if not all_exist:
            logger.error("❌ Some files were not saved properly")
            return False
    else:
        logger.warning("⚠️ No saved files found in results")
    
    # Check specific outputs
    logger.info("\n6️⃣ Checking specific outputs...")
    
    checks = {
        'n_regimes': result.get('n_regimes', 0) > 1,
        'regime_states': len(result.get('regime_states', [])) == n_periods,
        'quality_score': 0 <= result.get('overall_quality_score', -1) <= 1,
        'transitions': result.get('regime_transitions', {}).get('total_transitions', 0) > 0,
        'distribution': len(result.get('regime_distribution', {})) == result.get('n_regimes', 0),
        'composite_df': 'composite_df' in result
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        logger.info(f"   {status} {check_name}")
        if not passed:
            all_passed = False
    
    if not all_passed:
        logger.error("❌ Some output checks failed")
        return False
    
    # Check file contents
    logger.info("\n7️⃣ Checking file contents...")
    
    # Check metadata file
    metadata_path = Path(saved_files.get('metadata', ''))
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            logger.info("✅ Metadata file is valid JSON")
            logger.info(f"   - Symbol: {metadata.get('symbol')}")
            logger.info(f"   - Regimes: {metadata.get('n_regimes')}")
            logger.info(f"   - Quality: {metadata.get('overall_quality_score', 0):.4f}")
        except Exception as e:
            logger.error(f"❌ Error reading metadata: {e}")
            return False
    
    # Performance summary
    logger.info("\n8️⃣ Performance Summary:")
    logger.info(f"   - Execution time: {result.get('execution_time', 0):.2f} seconds")
    logger.info(f"   - Regimes found: {result.get('n_regimes', 0)}")
    logger.info(f"   - Quality score: {result.get('overall_quality_score', 0):.4f}")
    logger.info(f"   - Economic significance: {result.get('economic_significance', False)}")
    
    if 'step_timings' in result:
        logger.info("   - Step timings:")
        for step, timing in result['step_timings'].items():
            logger.info(f"     • {step}: {timing:.2f}s")
    
    logger.info("\n🎉 All integration tests passed!")
    return True


async def main():
    """Main test function."""
    success = await test_full_integration()
    
    logger.info("\n" + "=" * 80)
    if success:
        logger.info("✅ FULL INTEGRATION TEST: PASSED")
        logger.info("The enhanced HMM clustering is fully functional!")
    else:
        logger.info("❌ FULL INTEGRATION TEST: FAILED")
    logger.info("=" * 80)
    
    return success


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(main())
    sys.exit(0 if success else 1)