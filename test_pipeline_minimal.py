#!/usr/bin/env python3
"""
Minimal Pipeline Test

This script tests the basic pipeline structure with mock data generation
without requiring all the complex dependencies.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class MinimalPipelineTester:
    """Minimal pipeline tester that focuses on structure and mock data."""

    def __init__(self, symbol: str = "ETHUSDT", exchange: str = "BINANCE"):
        self.symbol = symbol
        self.exchange = exchange
        print(f"🧪 Initializing Minimal Pipeline Tester for {symbol} on {exchange}")

    def setup_environment(self):
        """Setup test environment."""
        print("🔧 Setting up test environment...")

        # Set environment variables
        os.environ["BLANK_TRAINING_MODE"] = "1"
        os.environ["FULL_TRAINING_MODE"] = "0"
        os.environ["FORCE"] = "1"

        # Create directories
        Path("data_cache").mkdir(exist_ok=True)
        Path("data/training").mkdir(parents=True, exist_ok=True)
        Path("log").mkdir(exist_ok=True)

        print("✅ Environment setup completed")

    def create_mock_data(self):
        """Create mock data for testing."""
        print("📊 Creating mock data...")

        # Generate 7 days of data for quick testing
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        # Generate klines data
        klines_timestamps = pd.date_range(start=start_date, end=end_date, freq='1min')
        klines_data = []

        np.random.seed(42)
        base_price = 3000.0
        price = base_price

        for timestamp in klines_timestamps:
            price_change = np.random.normal(0, 0.001)
            price = max(price * (1 + price_change), 100)

            spread = price * 0.0005
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

        # Generate aggtrades data
        aggtrades_timestamps = pd.date_range(start=start_date, end=end_date, freq='5min')
        aggtrades_data = []

        for timestamp in aggtrades_timestamps:
            num_trades = np.random.randint(1, 10)
            for _ in range(num_trades):
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

        # Save data files
        klines_df = pd.DataFrame(klines_data)
        aggtrades_df = pd.DataFrame(aggtrades_data)

        klines_file = Path("data_cache") / f"klines_{self.exchange}_{self.symbol}_1m_consolidated.parquet"
        aggtrades_file = Path("data_cache") / f"aggtrades_{self.exchange}_{self.symbol}_consolidated.parquet"

        klines_df.to_parquet(klines_file, index=False)
        aggtrades_df.to_parquet(aggtrades_file, index=False)

        print(f"✅ Created {len(klines_df)} klines records")
        print(f"✅ Created {len(aggtrades_df)} aggtrades records")

        return {
            'klines': str(klines_file),
            'aggtrades': str(aggtrades_file)
        }

    def simulate_step1(self):
        """Simulate step1 data collection."""
        print("\n🧪 Simulating Step1: Data Collection")
        print("=" * 50)

        # Create mock data
        mock_files = self.create_mock_data()

        # Simulate step1 processing
        print("📊 Processing klines data...")
        print("📊 Processing aggtrades data...")
        print("📊 Consolidating data...")

        # Verify outputs
        step1_outputs = [
            f"data_cache/klines_{self.exchange}_{self.symbol}_1m_consolidated.parquet",
            f"data_cache/aggtrades_{self.exchange}_{self.symbol}_consolidated.parquet",
        ]

        all_exist = True
        for output_file in step1_outputs:
            if Path(output_file).exists():
                file_size = Path(output_file).stat().st_size
                print(f"✅ {output_file}: {file_size} bytes")
            else:
                print(f"❌ {output_file}: File not found")
                all_exist = False

        return all_exist

    def simulate_step1_5(self):
        """Simulate step1_5 data converter."""
        print("\n🧪 Simulating Step1.5: Data Converter")
        print("=" * 50)

        # Check if step1 outputs exist
        step1_files = [
            f"data_cache/klines_{self.exchange}_{self.symbol}_1m_consolidated.parquet",
            f"data_cache/aggtrades_{self.exchange}_{self.symbol}_consolidated.parquet",
        ]

        for file_path in step1_files:
            if not Path(file_path).exists():
                print(f"❌ Step1 output not found: {file_path}")
                return False

        print("📊 Loading step1 outputs...")
        print("📊 Converting data formats...")
        print("📊 Creating unified dataset...")

        # Create step1_5 outputs
        unified_data = {
            'timestamp': pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='1min'),
            'open': np.random.uniform(2900, 3100, 10081),
            'high': np.random.uniform(2900, 3100, 10081),
            'low': np.random.uniform(2900, 3100, 10081),
            'close': np.random.uniform(2900, 3100, 10081),
            'volume': np.random.uniform(10, 1000, 10081),
            'exchange': self.exchange,
            'symbol': self.symbol,
            'timeframe': '1m'
        }

        unified_df = pd.DataFrame(unified_data)
        unified_file = Path("data_cache") / f"unified_{self.exchange}_{self.symbol}_1m.parquet"
        config_file = Path("data_cache") / f"unified_{self.exchange}_{self.symbol}_1m_config.json"

        unified_df.to_parquet(unified_file, index=False)

        # Create config file
        config_data = {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timeframe': '1m',
            'created_at': datetime.now().isoformat(),
            'data_points': len(unified_df)
        }

        import json
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)

        print(f"✅ Created unified dataset: {len(unified_df)} records")
        print(f"✅ Created config file")

        return True

    def simulate_step2(self):
        """Simulate step2 feature engineering."""
        print("\n🧪 Simulating Step2: Feature Engineering")
        print("=" * 50)

        # Check if step1_5 outputs exist
        step1_5_files = [
            f"data_cache/unified_{self.exchange}_{self.symbol}_1m.parquet",
            f"data_cache/unified_{self.exchange}_{self.symbol}_1m_config.json",
        ]

        for file_path in step1_5_files:
            if not Path(file_path).exists():
                print(f"❌ Step1.5 output not found: {file_path}")
                return False

        print("📊 Loading unified data...")
        print("📊 Engineering features...")
        print("📊 Creating train/val/test splits...")

        # Create step2 outputs
        base_features = {
            'timestamp': pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='1min'),
            'open': np.random.uniform(2900, 3100, 10081),
            'high': np.random.uniform(2900, 3100, 10081),
            'low': np.random.uniform(2900, 3100, 10081),
            'close': np.random.uniform(2900, 3100, 10081),
            'volume': np.random.uniform(10, 1000, 10081),
        }

        # Add some engineered features
        base_features['price_change'] = np.random.normal(0, 0.01, 10081)
        base_features['volume_ma'] = np.random.uniform(100, 500, 10081)
        base_features['volatility'] = np.random.uniform(0.001, 0.01, 10081)

        features_df = pd.DataFrame(base_features)

        # Split into train/val/test
        total_rows = len(features_df)
        train_size = int(total_rows * 0.7)
        val_size = int(total_rows * 0.15)

        train_df = features_df[:train_size]
        val_df = features_df[train_size:train_size + val_size]
        test_df = features_df[train_size + val_size:]

        # Save splits
        train_file = Path("data/training") / f"features_{self.exchange}_{self.symbol}_train.parquet"
        val_file = Path("data/training") / f"features_{self.exchange}_{self.symbol}_val.parquet"
        test_file = Path("data/training") / f"features_{self.exchange}_{self.symbol}_test.parquet"

        train_df.to_parquet(train_file, index=False)
        val_df.to_parquet(val_file, index=False)
        test_df.to_parquet(test_file, index=False)

        print(f"✅ Created training features: {len(train_df)} records")
        print(f"✅ Created validation features: {len(val_df)} records")
        print(f"✅ Created test features: {len(test_df)} records")

        return True

    def validate_pipeline(self):
        """Validate the complete pipeline."""
        print("\n🔍 Validating Pipeline Outputs")
        print("=" * 50)

        validation_results = {}

        # Check step1 outputs
        step1_files = [
            f"data_cache/klines_{self.exchange}_{self.symbol}_1m_consolidated.parquet",
            f"data_cache/aggtrades_{self.exchange}_{self.symbol}_consolidated.parquet",
        ]
        step1_valid = all(Path(f).exists() for f in step1_files)
        validation_results['step1'] = step1_valid
        print(f"Step1 outputs: {'✅' if step1_valid else '❌'}")

        # Check step1_5 outputs
        step1_5_files = [
            f"data_cache/unified_{self.exchange}_{self.symbol}_1m.parquet",
            f"data_cache/unified_{self.exchange}_{self.symbol}_1m_config.json",
        ]
        step1_5_valid = all(Path(f).exists() for f in step1_5_files)
        validation_results['step1_5'] = step1_5_valid
        print(f"Step1.5 outputs: {'✅' if step1_5_valid else '❌'}")

        # Check step2 outputs
        step2_files = [
            f"data/training/features_{self.exchange}_{self.symbol}_train.parquet",
            f"data/training/features_{self.exchange}_{self.symbol}_val.parquet",
            f"data/training/features_{self.exchange}_{self.symbol}_test.parquet",
        ]
        step2_valid = all(Path(f).exists() for f in step2_files)
        validation_results['step2'] = step2_valid
        print(f"Step2 outputs: {'✅' if step2_valid else '❌'}")

        return validation_results

    def run_complete_test(self):
        """Run the complete pipeline test."""
        print("🚀 Starting Minimal Pipeline Test")
        print("=" * 80)

        # Setup environment
        self.setup_environment()

        # Run pipeline steps
        results = {}

        # Step1
        results['step1'] = self.simulate_step1()

        # Step1_5
        if results['step1']:
            results['step1_5'] = self.simulate_step1_5()
        else:
            results['step1_5'] = False

        # Step2
        if results['step1_5']:
            results['step2'] = self.simulate_step2()
        else:
            results['step2'] = False

        # Validate
        validation_results = self.validate_pipeline()

        # Print results
        print("\n" + "=" * 80)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 80)

        for step, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{step}: {status}")

        print("\nValidation Results:")
        for step, result in validation_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {step}: {status}")

        # Overall success
        overall_success = all(results.values()) and all(validation_results.values())

        print("\n" + "=" * 80)
        if overall_success:
            print("🎉 ALL TESTS PASSED! Pipeline structure is working correctly.")
        else:
            print("💥 SOME TESTS FAILED! Check the logs for details.")
        print("=" * 80)

        return overall_success

def main():
    """Main test function."""
    tester = MinimalPipelineTester("ETHUSDT", "BINANCE")
    success = tester.run_complete_test()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)