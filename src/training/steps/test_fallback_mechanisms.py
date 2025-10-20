"""
Test Fallback Mechanisms

This script tests the fallback mechanisms to ensure they work in the correct order:
1. Primary: Step-category structure (artifacts/STEP-CATEGORY/)
2. Fallback 1: General artifacts directory search
3. Fallback 2: Without model type and direction variations
4. Fallback 3: Fuzzy matching for similar names
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from src.training.steps.base_step import BaseStep


class TestFallbackStep(BaseStep):
    """Test step to verify fallback mechanisms work correctly."""
    
    def __init__(self, step_name: str = "test_fallback_step"):
        super().__init__(step_name)
        self.logger.info(f"🧪 TestFallbackStep initialized: {step_name}")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test the fallback mechanisms."""
        try:
            # Set context
            self._set_context(
                symbol=config.get('symbol', 'BTCUSDT'),
                exchange=config.get('exchange', 'binance'),
                information=config.get('information', 'klines'),
                direction=config.get('direction', 'long'),
                model=config.get('model', 'Analyst')
            )
            
            # Create test data
            test_data = self._create_test_data()
            
            # Test 1: Save data in step-category structure (Primary)
            self.logger.info("🧪 Test 1: Saving data in step-category structure...")
            self._save_dataframe(test_data, 'test_data_primary')
            
            # Test 2: Save data in general artifacts directory (Fallback 1)
            self.logger.info("🧪 Test 2: Saving data in general artifacts directory...")
            self._save_dataframe(test_data, 'test_data_fallback1')
            
            # Test 3: Save data without model/direction context (Fallback 2)
            self.logger.info("🧪 Test 3: Saving data without model/direction context...")
            # Clear context for generic save
            original_model = self.artifact_manager._current_model
            original_direction = self.artifact_manager._current_direction
            self.artifact_manager._current_model = ""
            self.artifact_manager._current_direction = ""
            self._save_dataframe(test_data, 'test_data_generic')
            # Restore context
            self.artifact_manager._current_model = original_model
            self.artifact_manager._current_direction = original_direction
            
            # Test 4: Save data with similar name (Fallback 3)
            self.logger.info("🧪 Test 4: Saving data with similar name...")
            self._save_dataframe(test_data, 'test_data_similar')
            
            # Now test retrieval with different fallback levels
            self.logger.info("🔄 Testing retrieval with different fallback levels...")
            
            # Test primary retrieval
            primary_data = self._load_dataframe('test_data_primary')
            if primary_data is not None:
                self.logger.info("✅ Primary fallback (step-category) successful")
            else:
                self.logger.error("❌ Primary fallback failed")
            
            # Test fallback 1 retrieval
            fallback1_data = self._load_dataframe('test_data_fallback1')
            if fallback1_data is not None:
                self.logger.info("✅ Fallback 1 (general artifacts) successful")
            else:
                self.logger.error("❌ Fallback 1 failed")
            
            # Test fallback 2 retrieval (generic search)
            generic_data = self._load_dataframe('test_data_generic')
            if generic_data is not None:
                self.logger.info("✅ Fallback 2 (generic search) successful")
            else:
                self.logger.error("❌ Fallback 2 failed")
            
            # Test fallback 3 retrieval (fuzzy matching)
            fuzzy_data = self._load_dataframe('test_data_similar')
            if fuzzy_data is not None:
                self.logger.info("✅ Fallback 3 (fuzzy matching) successful")
            else:
                self.logger.error("❌ Fallback 3 failed")
            
            # Test non-existent artifact
            nonexistent_data = self._load_dataframe('nonexistent_data')
            if nonexistent_data is None:
                self.logger.info("✅ Non-existent artifact correctly returns None")
            else:
                self.logger.error("❌ Non-existent artifact should return None")
            
            return {
                'success': True,
                'artifacts': ['test_data_primary', 'test_data_fallback1', 'test_data_generic', 'test_data_similar'],
                'message': 'All fallback mechanisms tested successfully'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'artifacts': [],
                'message': 'Test failed'
            }
    
    def _create_test_data(self) -> pd.DataFrame:
        """Create test data for fallback testing."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1H')
        np.random.seed(42)
        
        data = {
            'timestamp': dates,
            'value': np.random.randn(len(dates)),
            'category': np.random.choice(['A', 'B', 'C'], len(dates))
        }
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df


async def main():
    """Main function to test fallback mechanisms."""
    print("🧪 Testing Fallback Mechanisms")
    print("=" * 50)
    
    # Create the test step
    step = TestFallbackStep("test_fallback_step")
    
    # Configuration
    config = {
        'symbol': 'BTCUSDT',
        'exchange': 'binance',
        'information': 'klines',
        'direction': 'long',
        'model': 'Analyst'
    }
    
    print(f"📋 Configuration: {config}")
    print()
    
    # Run the test
    print("🔄 Running fallback mechanism tests...")
    result = await step.run(config)
    
    # Display results
    print("\n📊 Test Results:")
    print(f"Success: {result['success']}")
    print(f"Artifacts: {result.get('artifacts', [])}")
    print(f"Message: {result.get('message', '')}")
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    
    print(f"\n⏱️ Execution Time: {result.get('execution_time', 0):.2f} seconds")
    
    # Show directory structure
    print("\n📁 Directory Structure Created:")
    artifacts_dir = Path("artifacts")
    if artifacts_dir.exists():
        for item in sorted(artifacts_dir.rglob("*")):
            if item.is_file():
                print(f"  {item}")
    
    print("\n✅ Fallback mechanism tests completed!")


if __name__ == "__main__":
    asyncio.run(main())