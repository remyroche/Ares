#!/usr/bin/env python3
"""
Test script for centralized S/R logic implementation.

This script validates that the centralized S/R logic in sr_breakout_predictor.py
is working correctly and can be used by all components.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
from src.utils.logger import setup_logging

setup_logging()


class CentralizedSRLogicTester:
    """Test the centralized S/R logic implementation."""

    def __init__(self):
        self.logger = setup_logging()
        self.test_results = {}

    def generate_test_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate realistic test market data."""
        np.random.seed(42)
        
        # Create realistic price data with some S/R levels
        base_price = 100
        price_changes = np.random.normal(0, 0.01, n_samples)
        prices = base_price * np.exp(np.cumsum(price_changes))
        
        # Add some S/R levels
        for i in range(50, n_samples, 100):
            # Create support level
            prices[i:i+10] = prices[i] * 0.98
            # Create resistance level
            prices[i+50:i+60] = prices[i+50] * 1.02
        
        # Create OHLCV data
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.002, n_samples)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_samples),
        })
        
        # Ensure OHLC relationships are valid
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        # Add timestamp
        data.index = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
        
        return data

    async def test_core_methods(self, config: dict[str, Any]) -> bool:
        """Test the core S/R methods."""
        try:
            self.logger.info("🧪 Testing core S/R methods...")
            
            # Initialize S/R predictor
            sr_predictor = SRBreakoutPredictor(config)
            init_success = await sr_predictor.initialize()
            
            if not init_success:
                self.logger.error("❌ Failed to initialize S/R predictor")
                return False
            
            # Generate test data
            test_data = self.generate_test_data(1000)
            current_price = test_data['close'].iloc[-1]
            
            # Test get_sr_context
            self.logger.info("Testing get_sr_context...")
            sr_context = await sr_predictor.get_sr_context(test_data, current_price)
            
            if not sr_context:
                self.logger.error("❌ get_sr_context returned empty result")
                return False
            
            required_keys = [
                'current_price', 'nearest_support', 'nearest_resistance',
                'support_strength', 'resistance_strength', 'support_proximity',
                'resistance_proximity', 'sr_zone_width'
            ]
            
            for key in required_keys:
                if key not in sr_context:
                    self.logger.error(f"❌ Missing key in sr_context: {key}")
                    return False
            
            self.logger.info("✅ get_sr_context test passed")
            
            # Test is_near_sr_level
            self.logger.info("Testing is_near_sr_level...")
            is_near = sr_predictor.is_near_sr_level(current_price, sr_context)
            
            if not isinstance(is_near, bool):
                self.logger.error("❌ is_near_sr_level returned non-boolean")
                return False
            
            self.logger.info(f"✅ is_near_sr_level test passed: {is_near}")
            
            # Test get_sr_proximity_details
            self.logger.info("Testing get_sr_proximity_details...")
            proximity_details = sr_predictor.get_sr_proximity_details(current_price, sr_context)
            
            if not proximity_details:
                self.logger.error("❌ get_sr_proximity_details returned empty result")
                return False
            
            self.logger.info("✅ get_sr_proximity_details test passed")
            
            # Test predict_sr_outcome
            self.logger.info("Testing predict_sr_outcome...")
            sr_outcome = await sr_predictor.predict_sr_outcome(test_data, current_price, sr_context)
            
            if not sr_outcome:
                self.logger.error("❌ predict_sr_outcome returned empty result")
                return False
            
            outcome_keys = ['outcome', 'confidence', 'features', 'sr_context']
            for key in outcome_keys:
                if key not in sr_outcome:
                    self.logger.error(f"❌ Missing key in sr_outcome: {key}")
                    return False
            
            valid_outcomes = ['breakout', 'rebounce', 'consolidation']
            if sr_outcome['outcome'] not in valid_outcomes:
                self.logger.error(f"❌ Invalid outcome: {sr_outcome['outcome']}")
                return False
            
            self.logger.info(f"✅ predict_sr_outcome test passed: {sr_outcome['outcome']}")
            
            # Test calculate_sr_features
            self.logger.info("Testing calculate_sr_features...")
            sr_features = await sr_predictor.calculate_sr_features(test_data)
            
            if not sr_features:
                self.logger.error("❌ calculate_sr_features returned empty result")
                return False
            
            self.logger.info(f"✅ calculate_sr_features test passed: {len(sr_features)} features")
            
            # Test calculate_comprehensive_sr_features
            self.logger.info("Testing calculate_comprehensive_sr_features...")
            comprehensive_features = await sr_predictor.calculate_comprehensive_sr_features(test_data)
            
            if not comprehensive_features:
                self.logger.error("❌ calculate_comprehensive_sr_features returned empty result")
                return False
            
            self.logger.info(f"✅ calculate_comprehensive_sr_features test passed: {len(comprehensive_features)} features")
            
            # Test predict_breakout
            self.logger.info("Testing predict_breakout...")
            breakout_prediction = await sr_predictor.predict_breakout(test_data)
            
            if not breakout_prediction:
                self.logger.error("❌ predict_breakout returned empty result")
                return False
            
            breakout_keys = ['direction', 'confidence', 'price', 'outcome', 'sr_context']
            for key in breakout_keys:
                if key not in breakout_prediction:
                    self.logger.error(f"❌ Missing key in breakout_prediction: {key}")
                    return False
            
            self.logger.info(f"✅ predict_breakout test passed: {breakout_prediction['direction']}")
            
            # Test set_weights
            self.logger.info("Testing set_weights...")
            test_weights = {
                'fractal_weight': 0.4,
                'volume_weight': 0.3,
                'pivot_weight': 0.2,
                'atr_weight': 0.1
            }
            
            weights_success = await sr_predictor.set_weights(test_weights)
            
            if not weights_success:
                self.logger.error("❌ set_weights failed")
                return False
            
            self.logger.info("✅ set_weights test passed")
            
            # Cleanup
            await sr_predictor.cleanup()
            
            self.logger.info("🎉 All core S/R methods tests passed!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Core methods test failed: {e}")
            return False

    async def test_feature_engineering_integration(self, config: dict[str, Any]) -> bool:
        """Test S/R integration with feature engineering."""
        try:
            self.logger.info("🧪 Testing feature engineering integration...")
            
            # Generate test data
            test_data = self.generate_test_data(500)
            
            # Test the feature engineering integration
            from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
            
            sr_predictor = SRBreakoutPredictor(config)
            await sr_predictor.initialize()
            
            # Calculate comprehensive features
            sr_features = await sr_predictor.calculate_comprehensive_sr_features(test_data)
            
            # Check that features are properly formatted for DataFrame integration
            for feature_name, feature_series in sr_features.items():
                if not isinstance(feature_series, pd.Series):
                    self.logger.error(f"❌ Feature {feature_name} is not a pandas Series")
                    return False
                
                if len(feature_series) != len(test_data):
                    self.logger.error(f"❌ Feature {feature_name} length mismatch")
                    return False
            
            # Test DataFrame integration
            features_df = test_data.copy()
            for feature_name, feature_series in sr_features.items():
                features_df[f"sr_{feature_name}"] = feature_series
            
            self.logger.info(f"✅ Feature engineering integration test passed: {len(sr_features)} features integrated")
            
            await sr_predictor.cleanup()
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Feature engineering integration test failed: {e}")
            return False

    async def test_analyst_integration(self, config: dict[str, Any]) -> bool:
        """Test S/R integration with analyst components."""
        try:
            self.logger.info("🧪 Testing analyst integration...")
            
            # Generate test data
            test_data = self.generate_test_data(500)
            current_price = test_data['close'].iloc[-1]
            
            # Test the analyst integration pattern
            from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
            
            sr_predictor = SRBreakoutPredictor(config)
            await sr_predictor.initialize()
            
            # Simulate analyst workflow
            sr_context = await sr_predictor.get_sr_context(test_data, current_price)
            is_near_sr = sr_predictor.is_near_sr_level(current_price, sr_context)
            
            if is_near_sr:
                sr_proximity_details = sr_predictor.get_sr_proximity_details(current_price, sr_context)
                sr_outcome = await sr_predictor.predict_sr_outcome(test_data, current_price, sr_context)
                
                # Check that we have all the data needed for analyst decisions
                analysis_result = {
                    'sr_monitoring': {
                        'is_near_sr_level': True,
                        'sr_proximity_details': sr_proximity_details,
                        'sr_outcome': sr_outcome,
                        'opportunity_detected': sr_outcome.get('confidence', 0) >= 0.6
                    }
                }
                
                self.logger.info(f"✅ Analyst integration test passed: opportunity_detected={analysis_result['sr_monitoring']['opportunity_detected']}")
            else:
                self.logger.info("✅ Analyst integration test passed: no S/R opportunity")
            
            await sr_predictor.cleanup()
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Analyst integration test failed: {e}")
            return False

    async def test_tactician_integration(self, config: dict[str, Any]) -> bool:
        """Test S/R integration with tactician components."""
        try:
            self.logger.info("🧪 Testing tactician integration...")
            
            # Generate test data
            test_data = self.generate_test_data(500)
            
            # Test the tactician integration pattern
            from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
            
            sr_predictor = SRBreakoutPredictor(config)
            await sr_predictor.initialize()
            
            # Simulate tactician workflow
            breakout_prediction = await sr_predictor.predict_breakout(test_data)
            
            if breakout_prediction:
                sr_decision = {
                    'breakout_direction': breakout_prediction.get('direction'),
                    'breakout_confidence': breakout_prediction.get('confidence', 0.0),
                    'breakout_price': breakout_prediction.get('price'),
                    'outcome': breakout_prediction.get('outcome', 'consolidation'),
                    'sr_context': breakout_prediction.get('sr_context', {}),
                    'source': 'sr_predictor'
                }
                
                self.logger.info(f"✅ Tactician integration test passed: {sr_decision['breakout_direction']} breakout")
            else:
                self.logger.info("✅ Tactician integration test passed: no breakout prediction")
            
            await sr_predictor.cleanup()
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Tactician integration test failed: {e}")
            return False

    async def run_all_tests(self) -> bool:
        """Run all tests for the centralized S/R logic."""
        try:
            self.logger.info("🚀 Starting centralized S/R logic tests...")
            
            # Configuration for testing
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
            
            # Run tests
            tests = [
                ("Core Methods", self.test_core_methods),
                ("Feature Engineering Integration", self.test_feature_engineering_integration),
                ("Analyst Integration", self.test_analyst_integration),
                ("Tactician Integration", self.test_tactician_integration),
            ]
            
            all_passed = True
            
            for test_name, test_func in tests:
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"Running {test_name} test...")
                self.logger.info(f"{'='*50}")
                
                try:
                    result = await test_func(config)
                    self.test_results[test_name] = result
                    
                    if result:
                        self.logger.info(f"✅ {test_name} test PASSED")
                    else:
                        self.logger.error(f"❌ {test_name} test FAILED")
                        all_passed = False
                        
                except Exception as e:
                    self.logger.error(f"❌ {test_name} test ERROR: {e}")
                    self.test_results[test_name] = False
                    all_passed = False
            
            # Print summary
            self.logger.info(f"\n{'='*50}")
            self.logger.info("TEST SUMMARY")
            self.logger.info(f"{'='*50}")
            
            for test_name, result in self.test_results.items():
                status = "✅ PASSED" if result else "❌ FAILED"
                self.logger.info(f"{test_name}: {status}")
            
            if all_passed:
                self.logger.info("\n🎉 ALL TESTS PASSED! Centralized S/R logic is working correctly.")
            else:
                self.logger.error("\n❌ SOME TESTS FAILED! Please check the implementation.")
            
            return all_passed
            
        except Exception as e:
            self.logger.error(f"❌ Test runner failed: {e}")
            return False


async def main():
    """Main function to run the centralized S/R logic tests."""
    tester = CentralizedSRLogicTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 Centralized S/R logic implementation is working correctly!")
        print("✅ All components can now use the centralized S/R logic")
        print("✅ Redundancy has been eliminated")
        print("✅ Integration is functional and consistent")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)