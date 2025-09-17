"""
Example usage of the improved regime data splitting module.

This example demonstrates how to use the standardized validation,
configuration management, and improved error handling.
"""

import asyncio
from typing import Dict, Any

from .enhanced import RegimeDataSplittingEnhanced
from .component import RegimeDataSplittingComponent
from .config_utils import RegimeDataSplittingConfig, get_config_manager, get_path_manager
from .validation_utils import get_validator, validate_training_input


async def example_enhanced_regime_splitting():
    """Example of using the enhanced regime data splitting."""
    
    # 1. Create configuration
    config = {
        'base_data_dir': 'data',
        'max_memory_gb': 4.0,
        'n_features': 50,
        'enable_hmm_models': True,
        'data_quality_threshold': 0.8
    }
    
    # 2. Initialize the enhanced splitter
    splitter = RegimeDataSplittingEnhanced(config)
    
    # 3. Prepare training input
    training_input = {
        'symbol': 'ETHUSDT',
        'exchange': 'BINANCE',
        'timeframe': '1m',
        'data_dir': 'data_cache'
    }
    
    # 4. Validate training input before processing
    validator = get_validator()
    validation_result = validate_training_input(training_input)
    
    if not validation_result.valid:
        print("❌ Training input validation failed:")
        for error in validation_result.errors:
            print(f"  - {error}")
        return
    
    print("✅ Training input validation passed")
    
    # 5. Execute regime data splitting
    pipeline_state = {}
    
    try:
        result = await splitter.execute(training_input, pipeline_state)
        
        if result.get('success', False):
            print("✅ Regime data splitting completed successfully")
            print(f"📊 Regimes detected: {result.get('regime_count', 'unknown')}")
            print(f"⏱️ Execution time: {result.get('execution_time_seconds', 0):.2f}s")
        else:
            print("❌ Regime data splitting failed")
            print(f"Error: {result.get('error_message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Exception during execution: {e}")


async def example_component_usage():
    """Example of using the regime data splitting component."""
    
    # 1. Create configuration using the config manager
    config_manager = get_config_manager({
        'min_regimes': 3,
        'max_regimes': 15,
        'data_quality_threshold': 0.75
    })
    
    # 2. Get path manager for consistent path handling
    path_manager = get_path_manager()
    
    # 3. Show some path examples
    symbol, exchange, timeframe = 'BTCUSDT', 'BINANCE', '5m'
    
    market_data_path = path_manager.get_market_data_path(exchange, symbol, timeframe)
    regime_tagged_path = path_manager.get_regime_tagged_data_path(exchange, symbol, timeframe)
    
    print(f"📂 Market data path: {market_data_path}")
    print(f"📂 Regime tagged data path: {regime_tagged_path}")
    
    # 4. Initialize component (this would normally be done by the framework)
    # component = RegimeDataSplittingComponent()
    # result = await component.execute(data, pipeline_state)
    
    print("✅ Component configuration example completed")


def example_configuration_management():
    """Example of configuration management features."""
    
    # 1. Create configuration from dictionary
    config_dict = {
        'base_data_dir': '/custom/data/path',
        'max_memory_gb': 16.0,
        'chunk_size': 200_000,
        'min_regimes': 2,
        'max_regimes': 25
    }
    
    config_manager = get_config_manager(config_dict)
    config = config_manager.get_config()
    
    print("📋 Configuration settings:")
    print(f"  Base data dir: {config.base_data_dir}")
    print(f"  Max memory: {config.max_memory_gb} GB")
    print(f"  Chunk size: {config.chunk_size:,}")
    print(f"  Regime range: {config.min_regimes}-{config.max_regimes}")
    
    # 2. Update configuration
    config_manager.update_config(
        max_memory_gb=8.0,
        data_quality_threshold=0.85
    )
    
    updated_config = config_manager.get_config()
    print(f"📋 Updated max memory: {updated_config.max_memory_gb} GB")
    print(f"📋 Updated quality threshold: {updated_config.data_quality_threshold}")
    
    # 3. Convert to dictionary for serialization
    config_dict_output = config_manager.to_dict()
    print("📋 Configuration as dictionary keys:")
    for key in sorted(config_dict_output.keys()):
        print(f"  - {key}")


def example_path_management():
    """Example of path management features."""
    
    # 1. Initialize path manager
    path_manager = get_path_manager()
    
    # 2. Example parameters
    symbol, exchange, timeframe = 'ADAUSDT', 'BINANCE', '15m'
    data_dir = '/custom/data'
    
    # 3. Get various paths
    paths = {
        'Market data': path_manager.get_market_data_path(exchange, symbol, timeframe, data_dir),
        'Regime tagged data': path_manager.get_regime_tagged_data_path(exchange, symbol, timeframe, data_dir),
        'Unified regime data': path_manager.get_unified_regime_data_path(exchange, symbol, timeframe, data_dir),
        'Composite clusters': path_manager.get_composite_clusters_path(exchange, symbol, timeframe, data_dir),
        'Regime statistics': path_manager.get_regime_statistics_path(exchange, symbol, data_dir),
        'HMM base model': path_manager.get_hmm_base_model_path('gaussian', exchange, symbol, timeframe, data_dir),
        'HMM ensemble model': path_manager.get_hmm_ensemble_model_path('voting', exchange, symbol, timeframe, data_dir),
        'Artifact path': path_manager.get_artifact_path('regime_splitting_report', symbol, exchange, timeframe)
    }
    
    print("📂 Path management examples:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    
    # 4. Ensure directories exist (this would create parent directories)
    # path_manager.ensure_directories_exist(*paths.values())
    print("✅ Path management example completed")


async def main():
    """Run all examples."""
    print("🚀 Regime Data Splitting Module Examples")
    print("=" * 50)
    
    print("\n1. Configuration Management Example:")
    example_configuration_management()
    
    print("\n2. Path Management Example:")
    example_path_management()
    
    print("\n3. Component Usage Example:")
    await example_component_usage()
    
    print("\n4. Enhanced Regime Splitting Example:")
    # Note: This would require actual data files to work
    # await example_enhanced_regime_splitting()
    print("📝 Enhanced regime splitting example requires actual data files")
    
    print("\n✅ All examples completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())