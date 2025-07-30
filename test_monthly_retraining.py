#!/usr/bin/env python3
"""
Test script for monthly retraining functionality.
This script tests the Supervisor's monthly retraining capability.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.database.sqlite_manager import SQLiteManager
from src.utils.state_manager import StateManager
from src.supervisor.supervisor import Supervisor
from src.exchange.binance import exchange
from src.utils.logger import setup_logging

async def test_monthly_retraining():
    """Test the monthly retraining functionality."""
    print("🧪 Testing Monthly Retraining Functionality")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    
    try:
        # Initialize database manager
        db_manager = SQLiteManager()
        await db_manager.initialize()
        
        # Initialize state manager
        state_manager = StateManager()
        
        # Initialize supervisor
        supervisor = Supervisor(
            exchange_client=exchange,
            state_manager=state_manager,
            db_manager=db_manager
        )
        
        print("✅ Components initialized successfully")
        
        # Test the retraining check functionality
        print("\n🔍 Testing retraining check functionality...")
        await supervisor._check_for_retraining(retrain_interval_days=30)
        
        # Test the retraining trigger functionality
        print("\n🚀 Testing retraining trigger functionality...")
        await supervisor._trigger_retraining()
        
        print("\n✅ Monthly retraining test completed successfully!")
        
        # Cleanup
        await db_manager.close()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_monthly_retraining()) 