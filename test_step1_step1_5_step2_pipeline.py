#!/usr/bin/env python3
"""
Test Script for Step1, Step1_5, and Step2 Pipeline with Mock Data

This script tests the complete pipeline using ares_launcher and enhanced_training_manager
with realistic mock data to ensure all steps work correctly together.

Usage:
    python test_step1_step1_5_step2_pipeline.py
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import required components
try:
    from src.training.step_orchestrator import StepOrchestrator
    from src.training.enhanced_training_manager import setup_enhanced_training_manager
    from src.config import CONFIG
    from src.utils.logger import system_logger, setup_logging
    from src.utils.error_handler import handle_errors
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running this from the project root directory")
    sys.exit(1)


class MockDataGenerator:
    """Generates realistic mock data for testing the pipeline."""
    
    def __init__(self, symbol: str = "ETHUSDT", exchange: str = "BINANCE"):
        self.symbol = symbol
        self.exchange = exchange
        self.logger = system_logger.getChild("MockDataGenerator")
        
    def generate_klines_data(self, days: int = 30) -> pd.DataFrame:
        """Generate realistic klines (OHLCV) data."""
        self.logger.info(f"📊 Generating {days} days of klines data for {self.symbol}")
        
        # Generate timestamps
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        timestamps = pd.date_range(start=start_date, end=end_date, freq='1min')
        
        # Generate realistic price data with some volatility
        np.random.seed(42)  # For reproducible results
        
        # Start with a realistic ETH price
        base_price = 3000.0
        price_changes = np.random.normal(0, 0.002, len(timestamps))  # 0.2% volatility per minute
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 100))  # Minimum price of $100
            
        prices = np.array(prices)
        
        # Generate OHLCV data
        data = []
        for i, timestamp in enumerate(timestamps):
            price = prices[i]
            volume = np.random.uniform(10, 1000)  # Realistic volume
            
            # Generate OHLC from base price with some spread
            spread = price * 0.001  # 0.1% spread
            open_price = price + np.random.uniform(-spread, spread)
            high_price = max(open_price, price + np.random.uniform(0, spread))
            low_price = min(open_price, price - np.random.uniform(0, spread))
            close_price = price + np.random.uniform(-spread, spread)
            
            data.append({
                'timestamp': timestamp,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': round(volume, 2),
                'quote_asset_volume': round(volume * close_price, 2),
                'number_of_trades': np.random.randint(1, 100),
                'taker_buy_base_asset_volume': round(volume * 0.6, 2),  # 60% taker volume
                'taker_buy_quote_asset_volume': round(volume * close_price * 0.6, 2),
            })
            
        df = pd.DataFrame(data)
        self.logger.info(f"✅ Generated {len(df)} klines records")
        return df
    
    def generate_aggtrades_data(self, days: int = 30) -> pd.DataFrame:
        """Generate realistic aggregated trades data."""
        self.logger.info(f"📊 Generating {days} days of aggtrades data for {self.symbol}")
        
        # Generate timestamps (less frequent than klines)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        timestamps = pd.date_range(start=start_date, end=end_date, freq='5min')  # Every 5 minutes
        
        np.random.seed(42)  # For reproducible results
        
        data = []
        base_price = 3000.0
        
        for timestamp in timestamps:
            # Generate multiple trades per timestamp
            num_trades = np.random.randint(1, 10)
            
            for _ in range(num_trades):
                price = base_price + np.random.normal(0, 50)  # Price variation
                quantity = np.random.uniform(0.1, 10.0)  # ETH quantity
                
                data.append({
                    'timestamp': timestamp,
                    'price': round(price, 2),
                    'quantity': round(quantity, 4),
                    'first_trade_id': np.random.randint(1000000, 9999999),
                    'last_trade_id': np.random.randint(1000000, 9999999),
                    'trade_time': int(timestamp.timestamp() * 1000),
                    'is_buyer_maker': np.random.choice([True, False]),
                })
                
        df = pd.DataFrame(data)
        self.logger.info(f"✅ Generated {len(df)} aggtrades records")
        return df
    
    def generate_futures_data(self, days: int = 30) -> pd.DataFrame:
        """Generate realistic futures data."""
        self.logger.info(f"📊 Generating {days} days of futures data for {self.symbol}")
        
        # Generate timestamps
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        timestamps = pd.date_range(start=start_date, end=end_date, freq='1min')
        
        np.random.seed(42)  # For reproducible results
        
        data = []
        base_price = 3000.0
        
        for timestamp in timestamps:
            price = base_price + np.random.normal(0, 30)
            funding_rate = np.random.uniform(-0.001, 0.001)  # -0.1% to 0.1%
            
            data.append({
                'timestamp': timestamp,
                'symbol': self.symbol,
                'mark_price': round(price, 2),
                'index_price': round(price + np.random.normal(0, 5), 2),
                'funding_rate': round(funding_rate, 6),
                'next_funding_time': int((timestamp + timedelta(hours=8)).timestamp() * 1000),
            })
            
        df = pd.DataFrame(data)
        self.logger.info(f"✅ Generated {len(df)} futures records")
        return df
    
    def create_mock_data_structure(self, data_dir: str = "data_cache") -> Dict[str, str]:
        """Create the complete mock data structure expected by the pipeline."""
        self.logger.info("🏗️ Creating mock data structure")
        
        # Create data directories
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Generate data
        klines_df = self.generate_klines_data(days=30)
        aggtrades_df = self.generate_aggtrades_data(days=30)
        futures_df = self.generate_futures_data(days=30)
        
        # Save data files
        files_created = {}
        
        # Klines data
        klines_file = data_path / f"klines_{self.exchange}_{self.symbol}_1m_consolidated.parquet"
        klines_df.to_parquet(klines_file, index=False)
        files_created['klines'] = str(klines_file)
        self.logger.info(f"💾 Saved klines data: {klines_file}")
        
        # Aggtrades data
        aggtrades_file = data_path / f"aggtrades_{self.exchange}_{self.symbol}_consolidated.parquet"
        aggtrades_df.to_parquet(aggtrades_file, index=False)
        files_created['aggtrades'] = str(aggtrades_file)
        self.logger.info(f"💾 Saved aggtrades data: {aggtrades_file}")
        
        # Futures data
        futures_file = data_path / f"futures_{self.exchange}_{self.symbol}_consolidated.parquet"
        futures_df.to_parquet(futures_file, index=False)
        files_created['futures'] = str(futures_file)
        self.logger.info(f"💾 Saved futures data: {futures_file}")
        
        self.logger.info("✅ Mock data structure created successfully")
        return files_created


class PipelineTester:
    """Tests the step1, step1_5, and step2 pipeline with mock data."""
    
    def __init__(self, symbol: str = "ETHUSDT", exchange: str = "BINANCE"):
        self.symbol = symbol
        self.exchange = exchange
        self.logger = system_logger.getChild("PipelineTester")
        self.mock_generator = MockDataGenerator(symbol, exchange)
        
    async def test_step1_data_collection(self) -> bool:
        """Test step1 data collection with mock data."""
        self.logger.info("🧪 Testing Step1: Data Collection")
        
        try:
            # Create mock data first
            mock_files = self.mock_generator.create_mock_data_structure()
            
            # Import step1
            from src.training.steps.step1_data_collection import run_step
            
            # Run step1
            success = await run_step(
                symbol=self.symbol,
                exchange=self.exchange,
                timeframe="1m",
                data_dir="data_cache",
                force_rerun=True
            )
            
            if success:
                self.logger.info("✅ Step1: Data Collection completed successfully")
                return True
            else:
                self.logger.error("❌ Step1: Data Collection failed")
                return False
                
        except Exception as e:
            self.logger.exception(f"❌ Step1 test failed: {e}")
            return False
    
    async def test_step1_5_data_converter(self) -> bool:
        """Test step1_5 data converter."""
        self.logger.info("🧪 Testing Step1.5: Data Converter")
        
        try:
            # Import step1_5
            from src.training.steps.step1_5_data_converter import run_step
            
            # Run step1_5
            success = await run_step(
                symbol=self.symbol,
                exchange=self.exchange,
                timeframe="1m",
                data_dir="data_cache",
                force_rerun=True
            )
            
            if success:
                self.logger.info("✅ Step1.5: Data Converter completed successfully")
                return True
            else:
                self.logger.error("❌ Step1.5: Data Converter failed")
                return False
                
        except Exception as e:
            self.logger.exception(f"❌ Step1.5 test failed: {e}")
            return False
    
    async def test_step2_feature_engineering(self) -> bool:
        """Test step2 feature engineering."""
        self.logger.info("🧪 Testing Step2: Feature Engineering")
        
        try:
            # Import step2
            from src.training.steps.step2_feature_engineering import run_step
            
            # Run step2
            success = await run_step(
                symbol=self.symbol,
                exchange=self.exchange,
                data_dir="data/training",
                timeframe="1m",
                force_rerun=True
            )
            
            if success:
                self.logger.info("✅ Step2: Feature Engineering completed successfully")
                return True
            else:
                self.logger.error("❌ Step2: Feature Engineering failed")
                return False
                
        except Exception as e:
            self.logger.exception(f"❌ Step2 test failed: {e}")
            return False
    
    async def test_with_step_orchestrator(self) -> bool:
        """Test the complete pipeline using StepOrchestrator."""
        self.logger.info("🧪 Testing complete pipeline with StepOrchestrator")
        
        try:
            # Create mock data first
            self.mock_generator.create_mock_data_structure()
            
            # Initialize orchestrator
            orchestrator = StepOrchestrator(self.symbol, self.exchange)
            
            # Test starting from step1
            success = await orchestrator.execute_from_step(
                start_step="step1_data_collection",
                config=CONFIG,
                force_rerun=True
            )
            
            if success:
                self.logger.info("✅ Complete pipeline test with StepOrchestrator successful")
                return True
            else:
                self.logger.error("❌ Complete pipeline test with StepOrchestrator failed")
                return False
                
        except Exception as e:
            self.logger.exception(f"❌ StepOrchestrator test failed: {e}")
            return False
    
    async def test_with_enhanced_training_manager(self) -> bool:
        """Test the complete pipeline using EnhancedTrainingManager directly."""
        self.logger.info("🧪 Testing complete pipeline with EnhancedTrainingManager")
        
        try:
            # Create mock data first
            self.mock_generator.create_mock_data_structure()
            
            # Setup enhanced training manager
            enhanced_manager = await setup_enhanced_training_manager(CONFIG)
            if not enhanced_manager:
                self.logger.error("❌ Failed to setup EnhancedTrainingManager")
                return False
            
            # Prepare training input
            training_input = {
                "symbol": self.symbol,
                "exchange": self.exchange,
                "timeframe": "1m",
                "data_dir": "data/training",
                "start_step": "step1_data_collection",
                "force_rerun": True,
                "lookback_days": 30,  # Use shorter period for testing
                "exclude_recent_days": 2,
            }
            
            # Execute pipeline
            success = await enhanced_manager.execute_enhanced_training(training_input)
            
            if success:
                self.logger.info("✅ Complete pipeline test with EnhancedTrainingManager successful")
                return True
            else:
                self.logger.error("❌ Complete pipeline test with EnhancedTrainingManager failed")
                return False
                
        except Exception as e:
            self.logger.exception(f"❌ EnhancedTrainingManager test failed: {e}")
            return False
    
    async def test_with_ares_launcher(self) -> bool:
        """Test using ares_launcher functionality."""
        self.logger.info("🧪 Testing with ares_launcher functionality")
        
        try:
            # Create mock data first
            self.mock_generator.create_mock_data_structure()
            
            # Import ares_launcher
            from ares_launcher import AresLauncher
            
            # Initialize launcher
            launcher = AresLauncher()
            
            # Test step-based training
            success = launcher._run_step_pipeline(
                symbol=self.symbol,
                exchange=self.exchange,
                start_step="step1_data_collection",
                force_rerun=True,
                with_gui=False,
                training_mode="blank"
            )
            
            if success:
                self.logger.info("✅ ares_launcher test successful")
                return True
            else:
                self.logger.error("❌ ares_launcher test failed")
                return False
                
        except Exception as e:
            self.logger.exception(f"❌ ares_launcher test failed: {e}")
            return False
    
    def validate_step_outputs(self) -> Dict[str, bool]:
        """Validate that each step produced the expected outputs."""
        self.logger.info("🔍 Validating step outputs")
        
        validation_results = {}
        
        # Check step1 outputs
        step1_files = [
            "data_cache/klines_BINANCE_ETHUSDT_1m_consolidated.parquet",
            "data_cache/aggtrades_BINANCE_ETHUSDT_consolidated.parquet",
        ]
        
        step1_valid = all(Path(f).exists() for f in step1_files)
        validation_results['step1'] = step1_valid
        self.logger.info(f"Step1 validation: {'✅' if step1_valid else '❌'}")
        
        # Check step1_5 outputs
        step1_5_files = [
            "data_cache/unified_BINANCE_ETHUSDT_1m.parquet",
            "data_cache/unified_BINANCE_ETHUSDT_1m_config.json",
        ]
        
        step1_5_valid = all(Path(f).exists() for f in step1_5_files)
        validation_results['step1_5'] = step1_5_valid
        self.logger.info(f"Step1.5 validation: {'✅' if step1_5_valid else '❌'}")
        
        # Check step2 outputs
        step2_files = [
            "data/training/features_BINANCE_ETHUSDT_train.parquet",
            "data/training/features_BINANCE_ETHUSDT_val.parquet",
            "data/training/features_BINANCE_ETHUSDT_test.parquet",
        ]
        
        step2_valid = all(Path(f).exists() for f in step2_files)
        validation_results['step2'] = step2_valid
        self.logger.info(f"Step2 validation: {'✅' if step2_valid else '❌'}")
        
        return validation_results


async def main():
    """Main test function."""
    print("🚀 Starting Step1, Step1_5, Step2 Pipeline Test")
    print("=" * 80)
    
    # Setup logging
    setup_logging()
    logger = system_logger.getChild("PipelineTest")
    
    # Initialize tester
    tester = PipelineTester("ETHUSDT", "BINANCE")
    
    # Test results
    results = {}
    
    try:
        # Test individual steps
        logger.info("🧪 Testing individual steps...")
        
        # Step1 test
        results['step1'] = await tester.test_step1_data_collection()
        
        # Step1_5 test
        results['step1_5'] = await tester.test_step1_5_data_converter()
        
        # Step2 test
        results['step2'] = await tester.test_step2_feature_engineering()
        
        # Test with orchestrators
        logger.info("🧪 Testing with orchestrators...")
        
        # StepOrchestrator test
        results['step_orchestrator'] = await tester.test_with_step_orchestrator()
        
        # EnhancedTrainingManager test
        results['enhanced_training_manager'] = await tester.test_with_enhanced_training_manager()
        
        # ares_launcher test
        results['ares_launcher'] = await tester.test_with_ares_launcher()
        
        # Validate outputs
        logger.info("🔍 Validating outputs...")
        validation_results = tester.validate_step_outputs()
        results['validation'] = validation_results
        
        # Print results
        print("\n" + "=" * 80)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 80)
        
        for test_name, result in results.items():
            if isinstance(result, dict):
                print(f"\n{test_name.upper()}:")
                for sub_test, sub_result in result.items():
                    status = "✅ PASS" if sub_result else "❌ FAIL"
                    print(f"  {sub_test}: {status}")
            else:
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{test_name}: {status}")
        
        # Overall success
        overall_success = all(
            result if isinstance(result, bool) else all(result.values())
            for result in results.values()
        )
        
        print("\n" + "=" * 80)
        if overall_success:
            print("🎉 ALL TESTS PASSED! Pipeline is working correctly.")
        else:
            print("💥 SOME TESTS FAILED! Check the logs for details.")
        print("=" * 80)
        
        return overall_success
        
    except Exception as e:
        logger.exception(f"❌ Test execution failed: {e}")
        print(f"❌ Test execution failed: {e}")
        return False


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(main())
    sys.exit(0 if success else 1)