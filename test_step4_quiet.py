#!/usr/bin/env python3
"""
Test Step 4 with reduced logging verbosity
"""

import asyncio
import logging
import sys
import os

# Set logging to WARNING level to reduce verbosity
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_step4():
    """Test Step 4 with minimal logging"""
    try:
        from src.training.steps.step4_analyst_labeling_feature_engineering import run_step
        
        print("🚀 Testing Step 4: Analyst Labeling & Feature Engineering...")
        result = await run_step('ETHUSDT', 'BINANCE', 'data/training')
        
        if result:
            print("✅ Step 4 completed successfully!")
        else:
            print("❌ Step 4 failed!")
            
        return result
        
    except Exception as e:
        print(f"❌ Error testing Step 4: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_step4()) 