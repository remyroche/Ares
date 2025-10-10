"""
Exchange Adapters Example

This example demonstrates how to use the new standardized exchange adapters
for downloading and processing klines data from different exchanges.
"""

import asyncio
from datetime import datetime, timedelta

# Import all exchange adapters
from exchanges.binance import create_binance_klines_adapter
from exchanges.bingx import create_bingx_klines_adapter
from exchanges.mexc import create_mexc_klines_adapter
from exchanges.okx import create_okx_klines_adapter
from exchanges.gateio import create_gateio_klines_adapter
from exchanges.phemex import create_phemex_klines_adapter

# Import shared pipeline functions
from exchanges.shared import run_exchange_klines_pipeline


async def example_individual_adapters():
    """Example using individual exchange adapters."""
    print("🚀 Exchange Adapters Example - Individual Usage")
    print("=" * 60)
    
    # Create adapters for different exchanges
    exchanges = {
        'binance': create_binance_klines_adapter(),
        'bingx': create_bingx_klines_adapter(),
        'mexc': create_mexc_klines_adapter(),
        'okx': create_okx_klines_adapter(),
        'gateio': create_gateio_klines_adapter(),
        'phemex': create_phemex_klines_adapter()
    }
    
    symbol = "BTCUSDT"
    interval = "1m"
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    
    for exchange_name, adapter in exchanges.items():
        print(f"\n📊 Testing {exchange_name.upper()} adapter...")
        
        try:
            # Get klines data
            data = await adapter.get_klines_data(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
                limit=100
            )
            
            if not data.empty:
                print(f"✅ {exchange_name.upper()}: {len(data)} records")
                print(f"   Columns: {list(data.columns)}")
                print(f"   Date range: {data.index.min()} to {data.index.max()}")
            else:
                print(f"⚠️ {exchange_name.upper()}: No data returned")
                
        except Exception as e:
            print(f"❌ {exchange_name.upper()}: Error - {e}")


async def example_shared_pipeline():
    """Example using the shared pipeline for different exchanges."""
    print("\n🚀 Exchange Adapters Example - Shared Pipeline")
    print("=" * 60)
    
    exchanges = ['binance', 'bingx', 'mexc', 'okx', 'gateio', 'phemex']
    symbol = "ETHUSDT"
    interval = "1m"
    
    for exchange in exchanges:
        print(f"\n📊 Running {exchange.upper()} pipeline...")
        
        try:
            # Run the complete pipeline
            results = await run_exchange_klines_pipeline(
                exchange=exchange,
                symbol=symbol,
                interval=interval,
                years=1,  # Just 1 year for demo
                create_consolidated=True
            )
            
            print(f"✅ {exchange.upper()} pipeline completed")
            print(f"   Success: {results['pipeline_success']}")
            print(f"   Steps: {len(results['steps_completed'])}")
            print(f"   Errors: {len(results['errors'])}")
            print(f"   Warnings: {len(results['warnings'])}")
            
        except Exception as e:
            print(f"❌ {exchange.upper()} pipeline failed: {e}")


async def example_data_quality_validation():
    """Example of data quality validation across exchanges."""
    print("\n🚀 Exchange Adapters Example - Data Quality Validation")
    print("=" * 60)
    
    # Create a sample adapter
    adapter = create_binance_klines_adapter()
    
    # Create sample data for validation
    import pandas as pd
    import numpy as np
    
    sample_data = pd.DataFrame({
        'open': np.random.uniform(50000, 51000, 100),
        'high': np.random.uniform(51000, 52000, 100),
        'low': np.random.uniform(49000, 50000, 100),
        'close': np.random.uniform(50000, 51000, 100),
        'volume': np.random.uniform(100, 1000, 100)
    })
    
    # Validate data quality
    quality_result = adapter.validate_data_quality(sample_data, "Sample validation")
    
    print(f"📊 Data Quality Validation Results:")
    print(f"   Passed: {quality_result['passed']}")
    print(f"   Quality Score: {quality_result.get('quality_score', 'N/A')}")
    print(f"   Issues: {len(quality_result.get('issues', []))}")
    print(f"   Warnings: {len(quality_result.get('warnings', []))}")


async def main():
    """Main example function."""
    print("🎯 Exchange Adapters Standardization Example")
    print("=" * 80)
    print("This example demonstrates the new standardized exchange adapter pattern")
    print("where all exchanges use the same shared processing logic.")
    print()
    
    # Run examples
    await example_individual_adapters()
    await example_shared_pipeline()
    await example_data_quality_validation()
    
    print("\n🎉 Example completed!")
    print("\nKey Benefits of the New Pattern:")
    print("✅ Consistent API across all exchanges")
    print("✅ Shared data processing and quality validation")
    print("✅ Exchange-agnostic pipeline functions")
    print("✅ Minimal exchange-specific adapters")
    print("✅ Centralized data standardization")


if __name__ == "__main__":
    asyncio.run(main())