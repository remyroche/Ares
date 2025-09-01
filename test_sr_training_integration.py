#!/usr/bin/env python3
"""
Test script to verify S/R integration across all training files.
This script checks that all training files can properly use the cleaned up S/R implementation.
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from src.utils.logger import system_logger

class SRTrainingIntegrationTester:
    """Test S/R integration across all training files."""

    def __init__(self):
        self.logger = system_logger.getChild("SRTrainingIntegrationTester")
        self.test_results = {}

    async def test_sr_breakout_predictor_import(self) -> bool:
        """Test that SRBreakoutPredictor can be imported and initialized."""
        try:
            from src.tactician.sr_breakout_predictor import SRBreakoutPredictor

            # Test configuration
            config = {
                "sr_breakout_predictor": {
                    "enable_sr_breakout_tactics": True,
                    "sr_proximity_threshold": 0.02,
                    "breakout_confidence_threshold": 0.6,
                    "sr_detection_method": "fractal",
                    "min_sr_strength": 0.3,
                    "max_sr_levels": 10,
                    "sr_lookback_periods": 100,
                }
            }

            # Initialize predictor
            predictor = SRBreakoutPredictor(config)
            init_success = await predictor.initialize()

            if not init_success:
                self.logger.error("Failed to initialize SRBreakoutPredictor")
                return False

            # Test basic functionality
            test_data = self._create_test_market_data()
            current_price = test_data['close'].iloc[-1]

            # Test get_sr_context
            sr_context = await predictor.get_sr_context(test_data, current_price)
            if not isinstance(sr_context, dict):
                self.logger.error("get_sr_context returned invalid type")
                return False

            # Test predict_sr_outcome
            sr_outcome = await predictor.predict_sr_outcome(test_data, current_price, sr_context)
            if not isinstance(sr_outcome, dict):
                self.logger.error("predict_sr_outcome returned invalid type")
                return False

            # Test calculate_sr_features
            sr_features = await predictor.calculate_sr_features(test_data)
            if not isinstance(sr_features, dict):
                self.logger.error("calculate_sr_features returned invalid type")
                return False

            # Test calculate_comprehensive_sr_features
            comprehensive_features = await predictor.calculate_comprehensive_sr_features(test_data)
            if not isinstance(comprehensive_features, dict):
                self.logger.error("calculate_comprehensive_sr_features returned invalid type")
                return False

            # Test is_near_sr_level
            is_near = predictor.is_near_sr_level(current_price, sr_context)
            if not isinstance(is_near, bool):
                self.logger.error("is_near_sr_level returned invalid type")
                return False

            # Cleanup
            await predictor.cleanup()

            self.logger.info("✅ SRBreakoutPredictor import and basic functionality test passed")
            return True

        except Exception as e:
            self.logger.error(f"❌ SRBreakoutPredictor import test failed: {e}")
            traceback.print_exc()
            return False

    async def test_step6_feature_engineering(self) -> bool:
        """Test step6_feature_engineering.py S/R integration."""
        try:
            # Import the function that uses S/R
            sys.path.insert(0, str(Path(__file__).parent / "src" / "training" / "steps"))

            # Test the _add_sr_features function
            test_data = self._create_test_market_data()
            features = pd.DataFrame({
                'feature1': np.random.randn(len(test_data)),
                'feature2': np.random.randn(len(test_data))
            })

            config = {
                "sr_breakout_predictor": {
                    "enable_sr_breakout_tactics": True,
                    "sr_proximity_threshold": 0.02,
                    "breakout_confidence_threshold": 0.6,
                    "sr_detection_method": "fractal",
                    "min_sr_strength": 0.3,
                    "max_sr_levels": 10,
                }
            }

            # Import and test the function
            from step6_feature_engineering import _add_sr_features

            enhanced_features = await _add_sr_features(features, test_data, config)

            if not isinstance(enhanced_features, pd.DataFrame):
                self.logger.error("_add_sr_features returned invalid type")
                return False

            # Check that S/R features were added
            sr_feature_cols = [col for col in enhanced_features.columns if col.startswith('sr_')]
            if len(sr_feature_cols) == 0:
                self.logger.error("No S/R features were added")
                return False

            self.logger.info(f"✅ Step6 feature engineering test passed - added {len(sr_feature_cols)} S/R features")
            return True

        except Exception as e:
            self.logger.error(f"❌ Step6 feature engineering test failed: {e}")
            traceback.print_exc()
            return False

    async def test_step10_unified_regime_intelligence(self) -> bool:
        """Test step10_unified_regime_intelligence.py S/R integration."""
        try:
            # Test that the class can be imported and initialized
            from step10_unified_regime_intelligence import UnifiedRegimeIntelligence

            config = {
                "sr_breakout_predictor": {
                    "enable_sr_breakout_tactics": True,
                    "sr_proximity_threshold": 0.02,
                    "breakout_confidence_threshold": 0.6,
                    "sr_detection_method": "fractal",
                    "min_sr_strength": 0.3,
                    "max_sr_levels": 10,
                }
            }

            # Initialize the class
            regime_intelligence = UnifiedRegimeIntelligence(config)

            # Test that sr_predictor was initialized
            if not hasattr(regime_intelligence, 'sr_predictor'):
                self.logger.error("UnifiedRegimeIntelligence missing sr_predictor")
                return False

            if regime_intelligence.sr_predictor is None:
                self.logger.error("UnifiedRegimeIntelligence sr_predictor is None")
                return False

            self.logger.info("✅ Step10 unified regime intelligence test passed")
            return True

        except Exception as e:
            self.logger.error(f"❌ Step10 unified regime intelligence test failed: {e}")
            traceback.print_exc()
            return False

    async def test_step15_tactician_specialist_training(self) -> bool:
        """Test step15_tactician_specialist_training.py S/R integration."""
        try:
            # Test that the class can be imported and initialized
            from step15_tactician_specialist_training import TacticianSpecialistTraining

            config = {
                "sr_breakout_predictor": {
                    "enable_sr_breakout_tactics": True,
                    "sr_proximity_threshold": 0.02,
                    "breakout_confidence_threshold": 0.6,
                    "sr_detection_method": "fractal",
                    "min_sr_strength": 0.3,
                    "max_sr_levels": 10,
                }
            }

            # Initialize the class
            tactician_training = TacticianSpecialistTraining(config)

            # Test that sr_predictor was initialized
            if not hasattr(tactician_training, 'sr_predictor'):
                self.logger.error("TacticianSpecialistTraining missing sr_predictor")
                return False

            if tactician_training.sr_predictor is None:
                self.logger.error("TacticianSpecialistTraining sr_predictor is None")
                return False

            self.logger.info("✅ Step15 tactician specialist training test passed")
            return True

        except Exception as e:
            self.logger.error(f"❌ Step15 tactician specialist training test failed: {e}")
            traceback.print_exc()
            return False

    async def test_sr_outcome_model_trainer(self) -> bool:
        """Test sr_outcome_model_trainer.py S/R integration."""
        try:
            # Test that the class can be imported and initialized
            from sr_outcome_model_trainer import SROutcomeModelTrainer

            config = {
                "sr_breakout_predictor": {
                    "enable_sr_breakout_tactics": True,
                    "sr_proximity_threshold": 0.02,
                    "breakout_confidence_threshold": 0.6,
                    "sr_detection_method": "fractal",
                    "min_sr_strength": 0.3,
                    "max_sr_levels": 10,
                }
            }

            # Initialize the class
            sr_trainer = SROutcomeModelTrainer(config)

            # Test that sr_predictor was initialized
            if not hasattr(sr_trainer, 'sr_predictor'):
                self.logger.error("SROutcomeModelTrainer missing sr_predictor")
                return False

            if sr_trainer.sr_predictor is None:
                self.logger.error("SROutcomeModelTrainer sr_predictor is None")
                return False

            self.logger.info("✅ SR outcome model trainer test passed")
            return True

        except Exception as e:
            self.logger.error(f"❌ SR outcome model trainer test failed: {e}")
            traceback.print_exc()
            return False

    async def test_step9_hmm_based_training(self) -> bool:
        """Test step9_hmm_based_training.py S/R integration."""
        try:
            # Test that the class can be imported and initialized
            from step9_hmm_based_training import HMMBasedTraining

            config = {
                "sr_breakout_predictor": {
                    "enable_sr_breakout_tactics": True,
                    "sr_proximity_threshold": 0.02,
                    "breakout_confidence_threshold": 0.6,
                    "sr_detection_method": "fractal",
                    "min_sr_strength": 0.3,
                    "max_sr_levels": 10,
                }
            }

            # Initialize the class
            hmm_training = HMMBasedTraining(config)

            # Test that sr_predictor was initialized
            if not hasattr(hmm_training, 'sr_predictor'):
                self.logger.error("HMMBasedTraining missing sr_predictor")
                return False

            if hmm_training.sr_predictor is None:
                self.logger.error("HMMBasedTraining sr_predictor is None")
                return False

            self.logger.info("✅ Step9 HMM based training test passed")
            return True

        except Exception as e:
            self.logger.error(f"❌ Step9 HMM based training test failed: {e}")
            traceback.print_exc()
            return False

    def _create_test_market_data(self) -> pd.DataFrame:
        """Create test market data for S/R testing."""
        np.random.seed(42)
        n_bars = 100

        # Create realistic market data
        base_price = 100.0
        prices = []
        volumes = []

        for i in range(n_bars):
            # Add some trend and noise
            trend = 0.001 * i  # Small upward trend
            noise = np.random.normal(0, 0.01)  # 1% volatility
            price_change = trend + noise

            if i == 0:
                close_price = base_price
            else:
                close_price = prices[-1]['close'] * (1 + price_change)

            # Create OHLC data
            high = close_price * (1 + abs(np.random.normal(0, 0.005)))
            low = close_price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = prices[-1]['close'] if i > 0 else close_price

            # Ensure OHLC relationship
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)

            # Volume with some correlation to price movement
            volume = np.random.randint(1000, 10000) * (1 + abs(price_change) * 10)

            prices.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
            volumes.append(volume)

        df = pd.DataFrame(prices)
        df.index = pd.date_range('2024-01-01', periods=n_bars, freq='1min')

        return df

    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all S/R integration tests."""
        self.logger.info("🚀 Starting S/R Training Integration Tests")
        self.logger.info("=" * 60)

        tests = [
            ("SRBreakoutPredictor Import", self.test_sr_breakout_predictor_import),
            ("Step6 Feature Engineering", self.test_step6_feature_engineering),
            ("Step10 Unified Regime Intelligence", self.test_step10_unified_regime_intelligence),
            ("Step15 Tactician Specialist Training", self.test_step15_tactician_specialist_training),
            ("SR Outcome Model Trainer", self.test_sr_outcome_model_trainer),
            ("Step9 HMM Based Training", self.test_step9_hmm_based_training),
        ]

        for test_name, test_func in tests:
            self.logger.info(f"\n🧪 Running {test_name} test...")
            try:
                result = await test_func()
                self.test_results[test_name] = result

                if result:
                    self.logger.info(f"✅ {test_name} test PASSED")
                else:
                    self.logger.error(f"❌ {test_name} test FAILED")

            except Exception as e:
                self.logger.error(f"❌ {test_name} test ERROR: {e}")
                self.test_results[test_name] = False
                traceback.print_exc()

        return self.test_results

    def print_summary(self):
        """Print test summary."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 S/R TRAINING INTEGRATION TEST SUMMARY")
        self.logger.info("=" * 60)

        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)

        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            self.logger.info(f"{test_name:<40} {status}")

        self.logger.info("-" * 60)
        self.logger.info(f"Total Tests: {total}")
        self.logger.info(f"Passed: {passed}")
        self.logger.info(f"Failed: {total - passed}")
        self.logger.info(f"Success Rate: {passed/total*100:.1f}%")

        if passed == total:
            self.logger.info("\n🎉 ALL S/R TRAINING INTEGRATION TESTS PASSED!")
            self.logger.info("The cleaned up S/R implementation is working correctly across all training files.")
        else:
            self.logger.error(f"\n⚠️ {total - passed} TESTS FAILED")
            self.logger.error("Some S/R integrations need attention.")

async def main():
    """Main test function."""
    tester = SRTrainingIntegrationTester()
    results = await tester.run_all_tests()
    tester.print_summary()

    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())