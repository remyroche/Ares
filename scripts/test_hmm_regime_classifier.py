#!/usr/bin/env python3
"""
Test script for HMM-based Market Regime Classifier

This script demonstrates the HMM regime classifier functionality:
1. Training the HMM classifier on historical data
2. Predicting market regimes using HMM
3. Comparing with traditional LightGBM approach
4. Analyzing HMM state characteristics

Usage:
    python scripts/test_hmm_regime_classifier.py --symbol ETHUSDT
    python scripts/test_hmm_regime_classifier.py --symbol BTCUSDT --use-real-data
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analyst.data_utils import create_dummy_data, load_klines_data
from src.analyst.hmm_regime_classifier import HMMRegimeClassifier
from src.analyst.regime_classifier import MarketRegimeClassifier
from src.analyst.sr_analyzer import SRLevelAnalyzer
from src.config import CONFIG


class HMMRegimeClassifierTester:
    """
    Test and demonstrate the HMM-based regime classifier.
    """

    def __init__(self, symbol: str, use_real_data: bool = False):
        self.symbol = symbol
        self.use_real_data = use_real_data
        self.config = CONFIG

        # Initialize classifiers
        self.hmm_classifier = HMMRegimeClassifier(self.config)

        # Initialize SR analyzer for traditional classifier
        sr_analyzer = SRLevelAnalyzer(self.config["analyst"]["sr_analyzer"])
        self.traditional_classifier = MarketRegimeClassifier(self.config, sr_analyzer)

    def generate_synthetic_data(self, num_records: int = 2000) -> pd.DataFrame:
        """
        Generate synthetic market data with different regimes.

        Args:
            num_records: Number of records to generate

        Returns:
            DataFrame with synthetic OHLCV data
        """
        print(f"Generating {num_records} synthetic market data points...")

        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate timestamps
        start_date = datetime.now() - timedelta(days=num_records)
        timestamps = pd.date_range(start=start_date, periods=num_records, freq="1H")

        # Initialize price
        initial_price = 100.0
        prices = [initial_price]

        # Generate different market regimes
        regime_lengths = [500, 400, 600, 500]  # Length of each regime
        regime_params = [
            {"mean_return": 0.001, "volatility": 0.02},  # Bull trend
            {"mean_return": -0.0008, "volatility": 0.025},  # Bear trend
            {"mean_return": 0.0001, "volatility": 0.035},  # High volatility sideways
            {"mean_return": 0.0002, "volatility": 0.015},  # Low volatility sideways
        ]

        current_idx = 0
        for regime_idx, (length, params) in enumerate(
            zip(regime_lengths, regime_params, strict=False),
        ):
            regime_prices = []
            current_price = prices[-1]

            for _ in range(length):
                # Generate price movement based on regime characteristics
                return_change = np.random.normal(
                    params["mean_return"],
                    params["volatility"],
                )
                current_price *= 1 + return_change
                regime_prices.append(current_price)

            prices.extend(regime_prices)
            current_idx += length

            print(f"Generated regime {regime_idx + 1}: {params}")

        # Create OHLCV data
        data = []
        for i, (timestamp, price) in enumerate(zip(timestamps, prices, strict=False)):
            # Generate OHLC from close price with some noise
            noise = np.random.normal(0, price * 0.001)
            open_price = price + noise
            high_price = max(open_price, price) + abs(
                np.random.normal(0, price * 0.002),
            )
            low_price = min(open_price, price) - abs(np.random.normal(0, price * 0.002))
            close_price = price

            # Generate volume
            volume = np.random.uniform(1000, 10000)

            data.append(
                {
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                },
            )

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)

        print(f"Generated synthetic data with {len(df)} records")
        print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")

        return df

    def load_real_data(self) -> pd.DataFrame:
        """
        Load real market data for testing.

        Returns:
            DataFrame with real OHLCV data
        """
        print(f"Loading real data for {self.symbol}...")

        # Create dummy data if it doesn't exist
        klines_filename = CONFIG["KLINES_FILENAME"]
        if not os.path.exists(klines_filename):
            print("Creating dummy data for testing...")
            create_dummy_data(klines_filename, "klines", num_records=2000)

        # Load data
        df = load_klines_data(klines_filename)

        if df.empty:
            print("Failed to load real data, falling back to synthetic data")
            return self.generate_synthetic_data()

        print(f"Loaded {len(df)} real data records")
        return df

    def train_classifiers(self, data: pd.DataFrame) -> bool:
        """
        Train both HMM and traditional classifiers.

        Args:
            data: Training data

        Returns:
            True if training successful
        """
        print("\n" + "=" * 60)
        print("TRAINING CLASSIFIERS")
        print("=" * 60)

        # Train HMM classifier
        print("\n1. Training HMM Classifier...")
        hmm_success = self.hmm_classifier.train_classifier(data)

        if hmm_success:
            print("✓ HMM classifier trained successfully")

            # Get HMM statistics
            stats = self.hmm_classifier.get_state_statistics()
            print(f"  - Number of states: {stats.get('n_states', 'N/A')}")
            print(
                f"  - State to regime mapping: {stats.get('state_to_regime_map', {})}",
            )
        else:
            print("✗ HMM classifier training failed")
            return False

        # Train traditional classifier (simplified for demo)
        print("\n2. Training Traditional Classifier...")
        try:
            # Create dummy features for traditional classifier
            dummy_features = pd.DataFrame(index=data.index)
            dummy_features["ADX"] = np.random.uniform(10, 50, len(data))
            dummy_features["ATR"] = data["close"] * 0.02
            dummy_features["volume_delta"] = np.random.normal(0, 1, len(data))
            dummy_features["autoencoder_reconstruction_error"] = np.random.uniform(
                0,
                0.1,
                len(data),
            )
            dummy_features["Is_SR_Interacting"] = np.random.choice([0, 1], len(data))

            # Add MACD histogram column
            macd_col = f"MACDh_{self.config['analyst']['market_regime_classifier'].get('macd_fast_period', 12)}_{self.config['analyst']['market_regime_classifier'].get('macd_slow_period', 26)}_{self.config['analyst']['market_regime_classifier'].get('macd_signal_period', 9)}"
            dummy_features[macd_col] = np.random.normal(0, 0.001, len(data))

            self.traditional_classifier.train_classifier(dummy_features, data)
            print("✓ Traditional classifier trained successfully")

        except Exception as e:
            print(f"✗ Traditional classifier training failed: {e}")
            # Continue with HMM only

        return True

    def compare_predictions(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compare predictions from both classifiers.

        Args:
            test_data: Test data

        Returns:
            DataFrame with comparison results
        """
        print("\n" + "=" * 60)
        print("COMPARING PREDICTIONS")
        print("=" * 60)

        results = []

        # Test on last 100 data points
        test_subset = test_data.tail(100)

        for i, (timestamp, row) in enumerate(test_subset.iterrows()):
            # Create DataFrame with sufficient historical data for prediction
            # Get the last 50 data points up to the current timestamp
            historical_data = test_data.loc[:timestamp].tail(50)
            current_data = pd.DataFrame([row]).set_index(pd.DatetimeIndex([timestamp]))

            # HMM prediction
            hmm_regime, hmm_confidence, hmm_info = self.hmm_classifier.predict_regime(
                historical_data,
            )

            # Traditional prediction (simplified)
            try:
                # Create dummy features for traditional classifier
                dummy_features = pd.DataFrame(index=current_data.index)
                dummy_features["ADX"] = np.random.uniform(20, 40)
                dummy_features["ATR"] = current_data["close"].iloc[0] * 0.02
                dummy_features["volume_delta"] = np.random.normal(0, 1)
                dummy_features["autoencoder_reconstruction_error"] = np.random.uniform(
                    0,
                    0.1,
                )
                dummy_features["Is_SR_Interacting"] = np.random.choice([0, 1])

                # Add MACD histogram column
                macd_col = f"MACDh_{self.config['analyst']['market_regime_classifier'].get('macd_fast_period', 12)}_{self.config['analyst']['market_regime_classifier'].get('macd_slow_period', 26)}_{self.config['analyst']['market_regime_classifier'].get('macd_signal_period', 9)}"
                dummy_features[macd_col] = np.random.normal(0, 0.001)

                # Create dummy SR levels
                dummy_sr_levels = []

                traditional_regime, trend_strength, adx = (
                    self.traditional_classifier.predict_regime(
                        dummy_features,
                        current_data,
                        dummy_sr_levels,
                    )
                )
            except Exception:
                traditional_regime = "UNKNOWN"
                trend_strength = 0.0
                adx = 0.0

            results.append(
                {
                    "timestamp": timestamp,
                    "price": current_data["close"].iloc[0],
                    "hmm_regime": hmm_regime,
                    "hmm_confidence": hmm_confidence,
                    "hmm_state": hmm_info.get("hmm_state", "N/A"),
                    "traditional_regime": traditional_regime,
                    "trend_strength": trend_strength,
                    "adx": adx,
                    "log_return": hmm_info.get("log_return", 0.0),
                    "volatility": hmm_info.get("volatility", 0.0),
                },
            )

            if i % 20 == 0:
                print(f"Processed {i+1}/{len(test_subset)} predictions...")

        results_df = pd.DataFrame(results)

        print(f"\nPrediction comparison completed for {len(results_df)} data points")
        return results_df

    def analyze_results(self, results_df: pd.DataFrame) -> None:
        """
        Analyze and display comparison results.

        Args:
            results_df: DataFrame with prediction results
        """
        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)

        # Regime distribution
        print("\n1. Regime Distribution:")
        print("HMM Classifier:")
        hmm_dist = results_df["hmm_regime"].value_counts()
        for regime, count in hmm_dist.items():
            percentage = (count / len(results_df)) * 100
            print(f"  {regime}: {count} ({percentage:.1f}%)")

        print("\nTraditional Classifier:")
        trad_dist = results_df["traditional_regime"].value_counts()
        for regime, count in trad_dist.items():
            percentage = (count / len(results_df)) * 100
            print(f"  {regime}: {count} ({percentage:.1f}%)")

        # Confidence analysis
        print("\n2. HMM Confidence Analysis:")
        print(f"  Average confidence: {results_df['hmm_confidence'].mean():.3f}")
        print(
            f"  Confidence range: {results_df['hmm_confidence'].min():.3f} - {results_df['hmm_confidence'].max():.3f}",
        )

        # State analysis
        print("\n3. HMM State Analysis:")
        state_dist = results_df["hmm_state"].value_counts()
        for state, count in state_dist.items():
            percentage = (count / len(results_df)) * 100
            print(f"  State {state}: {count} ({percentage:.1f}%)")

        # Agreement analysis
        agreement = (results_df["hmm_regime"] == results_df["traditional_regime"]).sum()
        agreement_pct = (agreement / len(results_df)) * 100
        print("\n4. Classifier Agreement:")
        print(f"  Agreement: {agreement}/{len(results_df)} ({agreement_pct:.1f}%)")

        # Feature analysis
        print("\n5. Feature Analysis:")
        print(f"  Average log return: {results_df['log_return'].mean():.6f}")
        print(f"  Average volatility: {results_df['volatility'].mean():.4f}")
        print(f"  Average trend strength: {results_df['trend_strength'].mean():.3f}")
        print(f"  Average ADX: {results_df['adx'].mean():.2f}")

    def save_results(self, results_df: pd.DataFrame) -> None:
        """
        Save results to file.

        Args:
            results_df: DataFrame with results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hmm_regime_test_results_{self.symbol}_{timestamp}.csv"

        results_df.to_csv(filename)
        print(f"\nResults saved to: {filename}")

    def run_test(self) -> None:
        """
        Run the complete HMM regime classifier test.
        """
        print("HMM Regime Classifier Test")
        print("=" * 60)
        print(f"Symbol: {self.symbol}")
        print(f"Data source: {'Real' if self.use_real_data else 'Synthetic'}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Load or generate data
        if self.use_real_data:
            data = self.load_real_data()
        else:
            data = self.generate_synthetic_data()

        # Train classifiers
        if not self.train_classifiers(data):
            print("Training failed. Exiting.")
            return

        # Compare predictions
        results_df = self.compare_predictions(data)

        # Analyze results
        self.analyze_results(results_df)

        # Save results
        self.save_results(results_df)

        print("\n" + "=" * 60)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)


def main():
    """Main entry point for the HMM regime classifier test."""
    parser = argparse.ArgumentParser(
        description="Test HMM-based Market Regime Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_hmm_regime_classifier.py --symbol ETHUSDT
  python scripts/test_hmm_regime_classifier.py --symbol BTCUSDT --use-real-data
        """,
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help="Trading symbol (e.g., ETHUSDT)",
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Use real data instead of synthetic data",
    )

    args = parser.parse_args()

    try:
        # Initialize and run tester
        tester = HMMRegimeClassifierTester(args.symbol, args.use_real_data)
        tester.run_test()

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
