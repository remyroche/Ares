"""
Example Usage of Optimized Data-Driven Lookback System

This example demonstrates how to use the optimized lookback optimization system
with matrix operations and hardware acceleration in the Ares pipeline.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import the optimized system
from .optimized_lookback_component import OptimizedLookbackComponent
from .config import create_production_config, create_development_config, FamilyType

# Import pipeline components
from ...components.component_factory import ComponentFactory
from ...sub_pipeline import SubPipelineConfig, ExecutionMode

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_market_data(n_days: int = 2000) -> pd.DataFrame:
    """Generate sample market data for demonstration."""
    logger.info(f"Generating sample market data for {n_days} days...")
    
    # Generate price data
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.02, n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    high_low_noise = np.random.uniform(0.001, 0.005, n_days)
    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, n_days)),
        'high': prices * (1 + high_low_noise),
        'low': prices * (1 - high_low_noise),
        'close': prices,
        'volume': np.random.uniform(1000000, 5000000, n_days)
    })
    
    # Add some technical indicators
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    return df


def generate_sample_targets(data: pd.DataFrame) -> np.ndarray:
    """Generate sample target variables."""
    # Generate future returns as targets
    future_returns = data['close'].pct_change(5).shift(-5)
    return future_returns.fillna(0).values


async def example_optimized_component_usage():
    """Example of using the optimized lookback component directly."""
    logger.info("🚀 Example: Direct Component Usage")
    
    # Generate sample data
    market_data = generate_sample_market_data(1000)
    targets = generate_sample_targets(market_data)
    
    # Create pipeline state
    pipeline_state = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'market_data': market_data,
        'multi_horizon_labeling_result': {
            'labeled_data': pd.DataFrame({'target': targets})
        }
    }
    
    # Create optimized component
    component = OptimizedLookbackComponent()
    
    # Execute component
    result = await component.execute(None, pipeline_state)
    
    if result.success:
        logger.info("✅ Optimized component execution successful!")
        logger.info(f"📊 Generated {len(result.artifacts['feature_names'])} features")
        logger.info(f"⚡ Matrix operations used: {result.artifacts['optimization_metrics']['matrix_ops_used']}")
        logger.info(f"🚀 Hardware accelerated ops: {result.artifacts['optimization_metrics']['hardware_accelerated_ops']}")
        
        # Access generated features
        feature_matrix = result.artifacts['feature_interaction_matrix']
        feature_names = result.artifacts['feature_names']
        
        logger.info(f"Feature matrix shape: {feature_matrix.shape}")
        logger.info(f"Feature names: {feature_names[:5]}...")  # Show first 5
        
    else:
        logger.error(f"❌ Component execution failed: {result.error_message}")
    
    return result


async def example_pipeline_integration():
    """Example of using the optimized system through the pipeline."""
    logger.info("🚀 Example: Pipeline Integration")
    
    # Create pipeline configuration
    config = SubPipelineConfig(
        symbol='ETHUSDT',
        exchange='binance',
        timeframe='15m',
        mode=ExecutionMode.FULL,
        parallel_processing=True,
        max_workers=4
    )
    
    # Create component using factory
    component = ComponentFactory.create_component('optimized_lookback_generation', config)
    
    # Generate sample data
    market_data = generate_sample_market_data(2000)
    targets = generate_sample_targets(market_data)
    
    # Create pipeline state
    pipeline_state = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'market_data': market_data,
        'multi_horizon_labeling_result': {
            'labeled_data': pd.DataFrame({'target': targets})
        }
    }
    
    # Execute component
    result = await component.execute(None, pipeline_state)
    
    if result.success:
        logger.info("✅ Pipeline integration successful!")
        
        # Access optimization results
        optimization_results = result.artifacts['optimized_lookback_results']
        logger.info(f"📊 Optimization completed in {optimization_results['execution_time']:.3f}s")
        logger.info(f"📈 Symbols processed: {len(optimization_results['ic_surface_results'])}")
        
        # Access feature generation results
        feature_metadata = result.artifacts['feature_generation_metadata']
        logger.info(f"⚙️ Features generated: {feature_metadata['total_features']}")
        logger.info(f"🔧 Base features: {feature_metadata['base_features']}")
        logger.info(f"🔗 Interaction features: {feature_metadata['interaction_features']}")
        
    else:
        logger.error(f"❌ Pipeline integration failed: {result.error_message}")
    
    return result


async def example_performance_comparison():
    """Example comparing optimized vs basic approaches."""
    logger.info("🚀 Example: Performance Comparison")
    
    # Generate larger dataset for comparison
    market_data = generate_sample_market_data(5000)
    targets = generate_sample_targets(market_data)
    
    # Test with development config (faster)
    from .config import create_development_config
    
    dev_config = create_development_config()
    dev_component = OptimizedLookbackComponent()
    dev_component.optimization_config = dev_config
    
    # Test with production config (more thorough)
    prod_config = create_production_config()
    prod_component = OptimizedLookbackComponent()
    prod_component.optimization_config = prod_config
    
    # Create pipeline state
    pipeline_state = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'market_data': market_data,
        'multi_horizon_labeling_result': {
            'labeled_data': pd.DataFrame({'target': targets})
        }
    }
    
    # Test development config
    logger.info("Testing development configuration...")
    start_time = datetime.now()
    dev_result = await dev_component.execute(None, pipeline_state)
    dev_time = (datetime.now() - start_time).total_seconds()
    
    # Test production config
    logger.info("Testing production configuration...")
    start_time = datetime.now()
    prod_result = await prod_component.execute(None, pipeline_state)
    prod_time = (datetime.now() - start_time).total_seconds()
    
    # Compare results
    logger.info("📊 Performance Comparison Results:")
    logger.info(f"Development config: {dev_time:.3f}s")
    logger.info(f"Production config: {prod_time:.3f}s")
    logger.info(f"Speed improvement: {prod_time/dev_time:.2f}x slower (but more thorough)")
    
    if dev_result.success and prod_result.success:
        dev_features = len(dev_result.artifacts['feature_names'])
        prod_features = len(prod_result.artifacts['feature_names'])
        logger.info(f"Development features: {dev_features}")
        logger.info(f"Production features: {prod_features}")
    
    return dev_result, prod_result


async def example_hardware_optimization():
    """Example demonstrating hardware optimization features."""
    logger.info("🚀 Example: Hardware Optimization")
    
    # Check hardware availability
    try:
        from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
        from src.utils.matrix_operations.hardware_integration import HardwareOptimizedMatrixProcessor
        
        hardware_manager = UnifiedHardwareManager()
        logger.info("✅ Hardware optimizations available")
        
        # Get hardware info
        hardware_info = hardware_manager.get_system_info()
        logger.info(f"CPU cores: {hardware_info.get('cpu_cores', 'Unknown')}")
        logger.info(f"Memory: {hardware_info.get('memory_gb', 'Unknown')} GB")
        logger.info(f"GPU available: {hardware_info.get('gpu_available', False)}")
        
    except ImportError:
        logger.warning("⚠️ Hardware optimizations not available")
    
    # Test matrix operations
    try:
        from src.utils.matrix_operations.unified_operations import get_unified_matrix_operations
        
        matrix_ops = get_unified_matrix_operations()
        logger.info("✅ Matrix operations available")
        
        # Test vectorized operations
        test_data = np.random.randn(1000, 10)
        test_target = np.random.randn(1000)
        
        # This would use optimized matrix operations
        logger.info("Testing vectorized operations...")
        
    except ImportError:
        logger.warning("⚠️ Matrix operations not available")
    
    return True


async def main():
    """Main example function."""
    print("Data-Driven Lookback Optimization System - Optimized Usage Examples")
    print("=" * 80)
    
    try:
        # Example 1: Direct component usage
        print("\n1. Direct Component Usage")
        print("-" * 40)
        await example_optimized_component_usage()
        
        # Example 2: Pipeline integration
        print("\n2. Pipeline Integration")
        print("-" * 40)
        await example_pipeline_integration()
        
        # Example 3: Performance comparison
        print("\n3. Performance Comparison")
        print("-" * 40)
        await example_performance_comparison()
        
        # Example 4: Hardware optimization
        print("\n4. Hardware Optimization")
        print("-" * 40)
        await example_hardware_optimization()
        
        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Example execution failed: {e}")
        print(f"\n❌ Example execution failed: {e}")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())