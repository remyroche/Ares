#!/usr/bin/env python3
"""
Test Pipeline with Ares Launcher

This script uses ares_launcher to test the step1, step1_5, and step2 pipeline
with mock data to ensure the orchestration works correctly.

Usage:
    python test_pipeline_with_ares_launcher.py
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import required components
try:
    from ares_launcher import AresLauncher
    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
    from src.utils.logger import system_logger, setup_logging
import except ImportError as e:
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running this from the project root directory")
    sys.exit(1)


class MockDataCreator:
    """Creates mock data for testing the pipeline."""

    def __init__(self, symbol: str = "ETHUSDT", exchange: str = "BINANCE"):
    pass
    pass
        self.symbol = symbol
        self.exchange = exchange
        self.logger = system_logger.getChild("MockDataCreator")

    def create_mock_data(self) -> Dict[str, str]:
    pass
    pass
        """Create mock data files that the pipeline expects."""
        self.logger.info("🏗️ Creating mock data for pipeline testing")

        # Create data directories
        data_cache_path = Path("data_cache")
        data_cache_path.mkdir(parents=True, exist_ok=True)

        # Generate 30 days of realistic data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        # Generate klines data (1-minute intervals)
        klines_timestamps = pd.date_range(start=start_date, end=end_date, freq='1min')
        klines_data = []

        np.random.seed(42)  # For reproducible results
        base_price = 3000.0
        price = base_price

        for timestamp in klines_timestamps:
    pass
    pass
            # Simulate price movement
            price_change = np.random.normal(0, 0.001)  # 0.1% volatility
            price = max(price * (1 + price_change), 100)  # Minimum $100

            # Generate OHLCV
            spread = price * 0.0005  # 0.05% spread
            open_price = price + np.random.uniform(-spread, spread)
            high_price = max(open_price, price + np.random.uniform(0, spread))
            low_price = min(open_price, price - np.random.uniform(0, spread))
            close_price = price + np.random.uniform(-spread, spread)
            volume = np.random.uniform(10, 1000)

            klines_data.append({
                'timestamp': timestamp,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': round(volume, 2),
                'quote_asset_volume': round(volume * close_price, 2),
                'number_of_trades': np.random.randint(1, 100),
                'taker_buy_base_asset_volume': round(volume * 0.6, 2),
                'taker_buy_quote_asset_volume': round(volume * close_price * 0.6, 2),
            })

        # Generate aggtrades data (less frequent)
        aggtrades_timestamps = pd.date_range(start=start_date, end=end_date, freq='5min')
        aggtrades_data = []

        for timestamp in aggtrades_timestamps:
    pass
    pass
            num_trades = np.random.randint(1, 10)
            for _ in range(num_trades):
    pass
    pass
                trade_price = base_price + np.random.normal(0, 50)
                quantity = np.random.uniform(0.1, 10.0)

                aggtrades_data.append({
                    'timestamp': timestamp,
                    'price': round(trade_price, 2),
                    'quantity': round(quantity, 4),
                    'first_trade_id': np.random.randint(1000000, 9999999),
                    'last_trade_id': np.random.randint(1000000, 9999999),
                    'trade_time': int(timestamp.timestamp() * 1000),
                    'is_buyer_maker': np.random.choice([True, False]),
                })

        # Generate futures data
        futures_timestamps = pd.date_range(start=start_date, end=end_date, freq='1min')
        futures_data = []

        for timestamp in futures_timestamps:
    pass
    pass
            mark_price = base_price + np.random.normal(0, 30)
            funding_rate = np.random.uniform(-0.001, 0.001)

            futures_data.append({
                'timestamp': timestamp,
                'symbol': self.symbol,
                'mark_price': round(mark_price, 2),
                'index_price': round(mark_price + np.random.normal(0, 5), 2),
                'funding_rate': round(funding_rate, 6),
                'next_funding_time': int((timestamp + timedelta(hours=8)).timestamp() * 1000),
            })

        # Create DataFrames and save files
        files_created = {}

        # Klines
        klines_df = pd.DataFrame(klines_data)
        klines_file = data_cache_path / f"klines_{self.exchange}_{self.symbol}_1m_consolidated.parquet"
        klines_df.to_parquet(klines_file, index=False)
        files_created['klines'] = str(klines_file)
        self.logger.info(f"💾 Created klines data: {len(klines_df)} records")

        # Aggtrades
        aggtrades_df = pd.DataFrame(aggtrades_data)
        aggtrades_file = data_cache_path / f"aggtrades_{self.exchange}_{self.symbol}_consolidated.parquet"
        aggtrades_df.to_parquet(aggtrades_file, index=False)
        files_created['aggtrades'] = str(aggtrades_file)
        self.logger.info(f"💾 Created aggtrades data: {len(aggtrades_df)} records")

        # Futures
        futures_df = pd.DataFrame(futures_data)
        futures_file = data_cache_path / f"futures_{self.exchange}_{self.symbol}_consolidated.parquet"
        futures_df.to_parquet(futures_file, index=False)
        files_created['futures'] = str(futures_file)
        self.logger.info(f"💾 Created futures data: {len(futures_df)} records")

        self.logger.info("✅ Mock data creation completed")
        return files_created


class PipelineTester:
    """Tests the pipeline using ares_launcher."""

    def __init__(self, symbol: str = "ETHUSDT", exchange: str = "BINANCE"):
    pass
    pass
        self.symbol = symbol
        self.exchange = exchange
        self.logger = system_logger.getChild("PipelineTester")
        self.mock_creator = MockDataCreator(symbol, exchange)
        self.launcher = AresLauncher()

    def setup_environment(self):
    pass
    pass
        """Setup environment for testing."""
        self.logger.info("🔧 Setting up test environment")

        # Set environment variables for testing
        os.environ["BLANK_TRAINING_MODE"] = "1"
        os.environ["FULL_TRAINING_MODE"] = "0"
        os.environ["FORCE"] = "1"

        # Create necessary directories
        Path("data_cache").mkdir(exist_ok=True)
        Path("data/training").mkdir(parents=True, exist_ok=True)
        Path("log").mkdir(exist_ok=True)

        self.logger.info("✅ Test environment setup completed")

    def create_mock_data(self):
    pass
    pass
        """Create mock data for testing."""
        self.logger.info("📊 Creating mock data")
        return self.mock_creator.create_mock_data()

    def test_step1_data_collection(self) -> bool:
    pass
    pass
        """Test step1 using ares_launcher."""
        self.logger.info("🧪 Testing Step1: Data Collection")

        try:
            # Create mock data first
    except Exception as e:
        pass
    except Exception as e:
        pass
            self.create_mock_data()

            # Test using ares_launcher's step pipeline
            success = self.launcher._run_step_pipeline(
                symbol=self.symbol,
                exchange=self.exchange,
                start_step="step1_data_collection",
                force_rerun=True,
                with_gui=False,
                training_mode="blank"
            )

            if success:
    pass
    pass
                self.logger.info("✅ Step1 test completed successfully")
                return True
            else:
                self.logger.error("❌ Step1 test failed")
                return False

        except Exception as e:
            self.logger.exception(f"❌ Step1 test failed: {e}")
            return False

    def test_step1_5_data_converter(self) -> bool:
    pass
    pass
        """Test step1_5 using ares_launcher."""
        self.logger.info("🧪 Testing Step1.5: Data Converter")

        try:
            # Create mock data first
    except Exception as e:
        pass
    except Exception as e:
        pass
            self.create_mock_data()

            # Test using ares_launcher's step pipeline
            success = self.launcher._run_step_pipeline(
                symbol=self.symbol,
                exchange=self.exchange,
                start_step="step1_5_data_converter",
                force_rerun=True,
                with_gui=False,
                training_mode="blank"
            )

            if success:
    pass
    pass
                self.logger.info("✅ Step1.5 test completed successfully")
                return True
            else:
                self.logger.error("❌ Step1.5 test failed")
                return False

        except Exception as e:
            self.logger.exception(f"❌ Step1.5 test failed: {e}")
            return False

    def test_step2_feature_engineering(self) -> bool:
    pass
    pass
        """Test step2 using ares_launcher."""
        self.logger.info("🧪 Testing Step2: Feature Engineering")

        try:
            # Create mock data first
    except Exception as e:
        pass
    except Exception as e:
        pass
            self.create_mock_data()

            # Test using ares_launcher's step pipeline
            success = self.launcher._run_step_pipeline(
                symbol=self.symbol,
                exchange=self.exchange,
                start_step="step2_feature_engineering",
                force_rerun=True,
                with_gui=False,
                training_mode="blank"
            )

            if success:
    pass
    pass
                self.logger.info("✅ Step2 test completed successfully")
                return True
            else:
                self.logger.error("❌ Step2 test failed")
                return False

        except Exception as e:
            self.logger.exception(f"❌ Step2 test failed: {e}")
            return False

    def test_complete_pipeline(self) -> bool:
    pass
    pass
        """Test the complete pipeline from step1 to step2."""
        self.logger.info("🧪 Testing Complete Pipeline (Step1 -> Step1.5 -> Step2)")

        try:
            # Create mock data first
    except Exception as e:
        pass
    except Exception as e:
        pass
            self.create_mock_data()

            # Test using ares_launcher's step pipeline starting from step1
            success = self.launcher._run_step_pipeline(
                symbol=self.symbol,
                exchange=self.exchange,
                start_step="step1_data_collection",
                force_rerun=True,
                with_gui=False,
                training_mode="blank"
            )

            if success:
    pass
    pass
                self.logger.info("✅ Complete pipeline test completed successfully")
                return True
            else:
                self.logger.error("❌ Complete pipeline test failed")
                return False

        except Exception as e:
            self.logger.exception(f"❌ Complete pipeline test failed: {e}")
            return False

    def test_blank_training_mode(self) -> bool:
    pass
    pass
        """Test using ares_launcher's blank training mode."""
        self.logger.info("🧪 Testing Blank Training Mode")

        try:
            # Create mock data first
    except Exception as e:
        pass
    except Exception as e:
        pass
            self.create_mock_data()

            # Test using ares_launcher's blank training
            success = self.launcher.run_blank_training(
                symbol=self.symbol,
                exchange=self.exchange,
                force_rerun=True
            )

            if success:
    pass
    pass
                self.logger.info("✅ Blank training test completed successfully")
                return True
            else:
                self.logger.error("❌ Blank training test failed")
                return False

        except Exception as e:
            self.logger.exception(f"❌ Blank training test failed: {e}")
            return False

    def validate_outputs(self) -> Dict[str, bool]:
    pass
    pass
        """Validate that the pipeline produced expected outputs."""
        self.logger.info("🔍 Validating pipeline outputs")

        validation_results = {}

        # Check step1 outputs
        step1_files = [
            "data_cache/klines_BINANCE_ETHUSDT_1m_consolidated.parquet",
            "data_cache/aggtrades_BINANCE_ETHUSDT_consolidated.parquet",
        ]
        step1_valid = all(Path(f).exists() for f in step1_files)
        validation_results['step1'] = step1_valid
        self.logger.info(f"Step1 outputs: {'✅' if step1_valid else '❌'}")

        # Check step1_5 outputs
        step1_5_files = [
            "data_cache/unified_BINANCE_ETHUSDT_1m.parquet",
            "data_cache/unified_BINANCE_ETHUSDT_1m_config.json",
        ]
        step1_5_valid = all(Path(f).exists() for f in step1_5_files)
        validation_results['step1_5'] = step1_5_valid
        self.logger.info(f"Step1.5 outputs: {'✅' if step1_5_valid else '❌'}")

        # Check step2 outputs
        step2_files = [
            "data/training/features_BINANCE_ETHUSDT_train.parquet",
            "data/training/features_BINANCE_ETHUSDT_val.parquet",
            "data/training/features_BINANCE_ETHUSDT_test.parquet",
        ]
        step2_valid = all(Path(f).exists() for f in step2_files)
        validation_results['step2'] = step2_valid
        self.logger.info(f"Step2 outputs: {'✅' if step2_valid else '❌'}")

        return validation_results


def main():
    pass
    pass
    """Main test function."""
    print("🚀 Starting Pipeline Test with Ares Launcher")
    print("=" * 80)

    # Setup logging
    setup_logging()
    logger = system_logger.getChild("PipelineTest")

    # Initialize tester
    tester = PipelineTester("ETHUSDT", "BINANCE")

    # Setup environment
    tester.setup_environment()

    # Test results
    results = {}

    try:
        # Test individual steps
    except Exception as e:
        pass
    except Exception as e:
        pass
        logger.info("🧪 Testing individual steps...")

        # Step1 test
        results['step1'] = tester.test_step1_data_collection()

        # Step1_5 test
        results['step1_5'] = tester.test_step1_5_data_converter()

        # Step2 test
        results['step2'] = tester.test_step2_feature_engineering()

        # Test complete pipeline
        logger.info("🧪 Testing complete pipeline...")
        results['complete_pipeline'] = tester.test_complete_pipeline()

        # Test blank training mode
        logger.info("🧪 Testing blank training mode...")
        results['blank_training'] = tester.test_blank_training_mode()

        # Validate outputs
        logger.info("🔍 Validating outputs...")
        validation_results = tester.validate_outputs()
        results['validation'] = validation_results

        # Print results
        print("\\\n" + "=" * 80)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 80)

        for test_name, result in results.items():
    pass
    pass
            if isinstance(result, dict):
    pass
    pass
                print(f"\\\n{test_name.upper()}:")
                for sub_test, sub_result in result.items():
    pass
    pass
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

        print("\\\n" + "=" * 80)
        if overall_success:
    pass
    pass
            print("🎉 ALL TESTS PASSED! Pipeline is working correctly with ares_launcher.")
        else:
            print("💥 SOME TESTS FAILED! Check the logs for details.")
        print("=" * 80)

        return overall_success

    except Exception as e:
        logger.exception(f"❌ Test execution failed: {e}")
        print(f"❌ Test execution failed: {e}")
        return False


if __name__ == "__main__":
    pass
    pass
    # Run the test
    success = main()
    sys.exit(0 if success else 1)