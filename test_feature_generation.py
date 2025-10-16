#!/usr/bin/env python3
"""
Simple test to verify feature generation works with our data cleaning
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_feature_generation():
    """Test feature generation with cleaned data"""
    
    print("🧪 Testing feature generation with data cleaning...")
    
    try:
        # Import the components we need
        from src.training.steps.data_collection.unified_data_loader import UnifiedDataLoader
        from src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_feature_generation_step import FeatureGenerationStep
        
        # Create data loader
        data_loader = UnifiedDataLoader()
        
        # Load data (this should apply our data cleaning)
        print("📊 Loading market data...")
        market_data = await data_loader.load_unified_data(
            symbol="ETHUSDT",
            exchange="binance", 
            timeframe="15m",
            start_date=None,
            end_date=None
        )
        
        if market_data is None:
            print("❌ Failed to load market data")
            return False
            
        print(f"✅ Data loaded: {market_data.shape}")
        
        # Check for non-finite values after loading
        non_finite = (~np.isfinite(market_data.select_dtypes(include=[np.number])).values).sum()
        print(f"🔍 Non-finite values after loading: {non_finite}")
        
        if non_finite > 0:
            print("❌ Data cleaning failed - non-finite values still present")
            return False
        
        # Test feature generation step
        print("🚀 Testing feature generation step...")
        step = FeatureGenerationStep()
        
        # Execute the step
        result = await step.execute(
            data=market_data,
            symbol="ETHUSDT",
            timeframe="15m", 
            direction="longs",
            custom_overrides={}
        )
        
        if result.success:
            print("✅ Feature generation completed successfully!")
            print(f"📊 Generated features: {result.generated_features.shape}")
            return True
        else:
            print(f"❌ Feature generation failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(test_feature_generation())
    if success:
        print("\n🎉 SUCCESS: Feature generation works with data cleaning!")
    else:
        print("\n💥 FAILED: Feature generation still has issues")
