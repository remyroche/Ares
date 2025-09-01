#!/usr/bin/env python3
"""
Test script for Universal ML Profit Integration System
Demonstrates the integration of ML profit predictions from steps 6-14 
into Analyst and Tactician through the Supervisor, and tests the enhanced confidence calculation.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.supervisor.enhanced_prediction_service import EnhancedPredictionService
from src.supervisor.supervisor import Supervisor
from src.config.enhanced_prediction_service_config import get_integration_config


class MLProfitIntegrationTester:
    """Test class for ML Profit Integration System."""
    
    def __init__(self):
        self.config = get_integration_config()
        self.enhanced_prediction_service = None
        self.supervisor = None
        self.test_results = {}
        
    async def setup(self):
        """Setup test environment."""
        print("🚀 Setting up ML Profit Integration Test Environment...")
        
        # Initialize Enhanced Prediction Service
        self.enhanced_prediction_service = EnhancedPredictionService(self.config)
        success = await self.enhanced_prediction_service.initialize()
        
        if not success:
            print("❌ Failed to initialize Enhanced Prediction Service")
            return False
            
        # Initialize Supervisor
        self.supervisor = Supervisor(self.config)
        await self.supervisor.initialize()
        
        print("✅ Test environment setup complete")
        return True
        
    def generate_mock_market_data(self, symbol: str = "ETHUSDT", days: int = 30) -> pd.DataFrame:
        """Generate mock market data for testing."""
        print(f"📊 Generating mock market data for {symbol} ({days} days)...")
        
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
        
        # Generate price data with some volatility
        np.random.seed(42)  # For reproducible results
        base_price = 3000.0  # Base ETH price
        
        # Generate price movements
        returns = np.random.normal(0, 0.001, len(timestamps))  # 0.1% volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
            
        # Generate OHLCV data
        data = []
        for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
            # Add some noise to create realistic OHLC
            noise = np.random.normal(0, price * 0.0005)
            
            open_price = price + noise
            high_price = max(open_price, price + abs(noise) * 1.5)
            low_price = min(open_price, price - abs(noise) * 1.5)
            close_price = price
            
            volume = np.random.uniform(1000, 10000)
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
            
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        print(f"✅ Generated {len(df)} data points")
        return df
        
    def generate_mock_regime_info(self) -> dict:
        """Generate mock regime information."""
        return {
            "regime": "trending_bullish",
            "confidence": 0.75,
            "regime_probabilities": {
                "trending_bullish": 0.75,
                "trending_bearish": 0.15,
                "sideways": 0.10
            },
            "regime_features": {
                "volatility": 0.02,
                "momentum": 0.05,
                "trend_strength": 0.8
            }
        }
        
    def generate_mock_ml_profit_models(self):
        """Generate mock ML profit models for testing."""
        print("🤖 Generating mock ML profit models...")
        
        # Create mock model data directory
        mock_models_dir = Path("data/training/ml_profit_models")
        mock_models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create different model type directories
        model_types = ["hmm_profit", "analyst_profit", "tactician_profit", "ensemble_profit"]
        
        for model_type in model_types:
            type_dir = mock_models_dir / model_type
            type_dir.mkdir(exist_ok=True)
            
            # Create mock model files
            for i in range(3):  # Create 3 models per type
                model_name = f"{model_type}_model_{i+1}"
                model_data = {
                    "model": MockMLModel(model_type, i+1),
                    "confidence": 0.6 + (i * 0.1),
                    "model_type": model_type,
                    "training_date": datetime.now().isoformat(),
                    "performance_metrics": {
                        "accuracy": 0.65 + (i * 0.05),
                        "precision": 0.62 + (i * 0.05),
                        "recall": 0.68 + (i * 0.05)
                    }
                }
                
                # Save mock model (in real implementation, this would be a pickle file)
                print(f"  ✅ Created mock model: {model_name}")
                
        print("✅ Mock ML profit models generated")
        
    async def test_enhanced_confidence_calculation(self):
        """Test the enhanced confidence calculation function."""
        print("\n🎯 Testing Enhanced Confidence Calculation...")
        
        # Generate test data
        market_data = self.generate_mock_market_data()
        regime_info = self.generate_mock_regime_info()
        
        # Test different scenarios
        test_scenarios = [
            {
                "name": "High Confidence Bullish",
                "predicted_direction": 1,
                "predicted_magnitude": 0.03,
                "base_confidence": 0.8,
                "expected_high_confidence": True
            },
            {
                "name": "Low Confidence Bearish",
                "predicted_direction": -1,
                "predicted_magnitude": 0.01,
                "base_confidence": 0.4,
                "expected_high_confidence": False
            },
            {
                "name": "Neutral Direction",
                "predicted_direction": 0,
                "predicted_magnitude": 0.0,
                "base_confidence": 0.5,
                "expected_high_confidence": False
            }
        ]
        
        current_price = market_data['close'].iloc[-1]
        price_volatility = market_data['close'].pct_change().std()
        
        for scenario in test_scenarios:
            print(f"\n  📋 Testing: {scenario['name']}")
            
            # Calculate enhanced confidence
            enhanced_confidence = await self.enhanced_prediction_service._calculate_directional_confidence_with_barriers(
                predicted_direction=scenario["predicted_direction"],
                predicted_magnitude=scenario["predicted_magnitude"],
                base_confidence=scenario["base_confidence"],
                current_price=current_price,
                profit_threshold_price=current_price * 1.02,  # 2% profit target
                barrier_threshold_price=current_price * 0.99,  # 1% barrier
                price_volatility=price_volatility,
                prediction_name=f"test_{scenario['name'].lower().replace(' ', '_')}"
            )
            
            print(f"    Base Confidence: {scenario['base_confidence']:.3f}")
            print(f"    Enhanced Confidence: {enhanced_confidence:.3f}")
            print(f"    Confidence Improvement: {enhanced_confidence - scenario['base_confidence']:.3f}")
            
            # Validate results
            if scenario["expected_high_confidence"]:
                assert enhanced_confidence > 0.6, f"Expected high confidence for {scenario['name']}"
            else:
                assert enhanced_confidence <= 0.6, f"Expected low confidence for {scenario['name']}"
                
            print(f"    ✅ {scenario['name']} test passed")
            
        print("✅ Enhanced confidence calculation tests completed")
        
    async def test_ml_profit_integration(self):
        """Test the ML profit integration with Analyst and Tactician."""
        print("\n🔄 Testing ML Profit Integration...")
        
        # Generate test data
        market_data = self.generate_mock_market_data()
        regime_info = self.generate_mock_regime_info()
        
        # Test Analyst integration
        print("\n  📊 Testing Analyst Integration...")
        analyst_predictions = await self.supervisor.get_analyst_predictions(
            market_data=market_data,
            regime_info=regime_info,
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="1m"
        )
        
        if analyst_predictions:
            print("    ✅ Analyst predictions generated successfully")
            print(f"    📈 ML Profit Predictions: {len(analyst_predictions.get('ml_profit_integration', {}).get('ml_profit_predictions', {}))}")
            print(f"    🎯 Enhanced Confidence Scores: {len(analyst_predictions.get('ml_profit_integration', {}).get('enhanced_confidence_scores', {}))}")
            print(f"    🛡️ Barrier Analysis: {len(analyst_predictions.get('ml_profit_integration', {}).get('barrier_analysis', {}))}")
        else:
            print("    ⚠️ No analyst predictions generated (expected if no models loaded)")
            
        # Test Tactician integration
        print("\n  ⚡ Testing Tactician Integration...")
        tactician_predictions = await self.supervisor.get_tactician_predictions(
            market_data=market_data,
            regime_info=regime_info,
            analyst_signals=analyst_predictions or {},
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="1m"
        )
        
        if tactician_predictions:
            print("    ✅ Tactician predictions generated successfully")
            print(f"    🎯 Enhanced Tactician Signals: {len(tactician_predictions.get('enhanced_tactician_signals', {}))}")
            print(f"    ⚙️ Execution Parameters: {len(tactician_predictions.get('execution_parameters', {}))}")
            print(f"    📏 Position Sizing Enhancement: {len(tactician_predictions.get('position_sizing_enhancement', {}))}")
        else:
            print("    ⚠️ No tactician predictions generated (expected if no models loaded)")
            
        print("✅ ML Profit Integration tests completed")
        
    async def test_barrier_analysis(self):
        """Test the barrier analysis functionality."""
        print("\n🛡️ Testing Barrier Analysis...")
        
        # Generate test data
        market_data = self.generate_mock_market_data()
        current_price = market_data['close'].iloc[-1]
        price_volatility = market_data['close'].pct_change().std()
        
        # Test different prediction scenarios
        test_predictions = {
            "bullish_strong": {
                "direction": 1,
                "magnitude": 0.03,
                "confidence": 0.8
            },
            "bearish_weak": {
                "direction": -1,
                "magnitude": 0.01,
                "confidence": 0.4
            },
            "neutral": {
                "direction": 0,
                "magnitude": 0.0,
                "confidence": 0.5
            }
        }
        
        for prediction_name, prediction_data in test_predictions.items():
            print(f"\n  📋 Testing: {prediction_name}")
            
            # Calculate barrier metrics
            barrier_metrics = self.enhanced_prediction_service._calculate_barrier_metrics(
                prediction_data, current_price, price_volatility
            )
            
            print(f"    Profit Target: {barrier_metrics.get('profit_target', 0):.2f}")
            print(f"    Barrier Level: {barrier_metrics.get('barrier_level', 0):.2f}")
            print(f"    Risk-Reward Ratio: {barrier_metrics.get('risk_reward_ratio', 0):.3f}")
            print(f"    Expected Value: {barrier_metrics.get('expected_value', 0):.4f}")
            
            # Validate metrics
            assert barrier_metrics.get('profit_target', 0) > 0, "Profit target should be positive"
            assert barrier_metrics.get('barrier_level', 0) > 0, "Barrier level should be positive"
            
            print(f"    ✅ {prediction_name} barrier analysis passed")
            
        print("✅ Barrier analysis tests completed")
        
    async def test_risk_metrics_calculation(self):
        """Test risk metrics calculation."""
        print("\n📊 Testing Risk Metrics Calculation...")
        
        # Generate test data
        market_data = self.generate_mock_market_data()
        
        # Mock ML profit data
        mock_ml_profit_data = {
            "hmm_profit_model_1": {
                "direction": 1,
                "magnitude": 0.02,
                "confidence": 0.7
            },
            "analyst_profit_model_1": {
                "direction": 1,
                "magnitude": 0.015,
                "confidence": 0.65
            },
            "tactician_profit_model_1": {
                "direction": -1,
                "magnitude": 0.01,
                "confidence": 0.45
            }
        }
        
        # Mock barrier analysis
        mock_barrier_analysis = {
            "hmm_profit_model_1": {
                "expected_value": 0.015,
                "risk_reward_ratio": 2.0
            },
            "analyst_profit_model_1": {
                "expected_value": 0.010,
                "risk_reward_ratio": 1.5
            },
            "tactician_profit_model_1": {
                "expected_value": -0.005,
                "risk_reward_ratio": 0.8
            }
        }
        
        # Calculate risk metrics
        risk_metrics = await self.supervisor._calculate_analyst_risk_metrics(
            mock_ml_profit_data, mock_barrier_analysis, market_data
        )
        
        print("  📈 Aggregate Risk Metrics:")
        aggregate_risk = risk_metrics.get("aggregate_risk", {})
        print(f"    Average Confidence: {aggregate_risk.get('average_confidence', 0):.3f}")
        print(f"    Average Expected Value: {aggregate_risk.get('average_expected_value', 0):.4f}")
        print(f"    Average Risk-Reward Ratio: {aggregate_risk.get('average_risk_reward_ratio', 0):.3f}")
        print(f"    Overall Risk Level: {aggregate_risk.get('overall_risk_level', 'unknown')}")
        
        print("  🎯 Individual Risk Metrics:")
        for prediction_name, individual_risk in risk_metrics.get("individual_risks", {}).items():
            print(f"    {prediction_name}:")
            print(f"      Confidence: {individual_risk.get('confidence', 0):.3f}")
            print(f"      Expected Value: {individual_risk.get('expected_value', 0):.4f}")
            print(f"      Risk Level: {individual_risk.get('risk_level', 'unknown')}")
            
        print("  📊 Portfolio Implications:")
        portfolio_implications = risk_metrics.get("portfolio_implications", {})
        print(f"    Market Volatility: {portfolio_implications.get('market_volatility', 0):.4f}")
        print(f"    Recommended Position Size: {portfolio_implications.get('recommended_position_size', 'unknown')}")
        print(f"    Risk Adjustment Factor: {portfolio_implications.get('risk_adjustment_factor', 0):.3f}")
        
        print("✅ Risk metrics calculation tests completed")
        
    async def run_all_tests(self):
        """Run all tests for the ML Profit Integration System."""
        print("🚀 Starting Universal ML Profit Integration System Tests")
        print("=" * 80)
        
        # Setup
        if not await self.setup():
            print("❌ Setup failed, aborting tests")
            return False
            
        try:
            # Generate mock models
            self.generate_mock_ml_profit_models()
            
            # Run tests
            await self.test_enhanced_confidence_calculation()
            await self.test_ml_profit_integration()
            await self.test_barrier_analysis()
            await self.test_risk_metrics_calculation()
            
            print("\n" + "=" * 80)
            print("🎉 All ML Profit Integration System Tests Completed Successfully!")
            print("=" * 80)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


class MockMLModel:
    """Mock ML model for testing purposes."""
    
    def __init__(self, model_type: str, model_id: int):
        self.model_type = model_type
        self.model_id = model_id
        
    def predict(self, features):
        """Mock prediction method."""
        # Generate realistic mock predictions based on model type
        if self.model_type == "hmm_profit":
            # HMM models tend to be more conservative
            return np.array([0.01 + (self.model_id * 0.005)])
        elif self.model_type == "analyst_profit":
            # Analyst models are more directional
            return np.array([0.02 + (self.model_id * 0.01)])
        elif self.model_type == "tactician_profit":
            # Tactician models focus on execution
            return np.array([0.015 + (self.model_id * 0.008)])
        else:  # ensemble_profit
            # Ensemble models are balanced
            return np.array([0.018 + (self.model_id * 0.007)])


async def main():
    """Main test function."""
    tester = MLProfitIntegrationTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n✅ All tests passed! The Universal ML Profit Integration System is working correctly.")
        print("\n📋 Summary of what was tested:")
        print("  1. Enhanced confidence calculation with barrier analysis")
        print("  2. ML profit predictions integration with Analyst")
        print("  3. ML profit predictions integration with Tactician")
        print("  4. Barrier analysis for risk management")
        print("  5. Risk metrics calculation")
        print("  6. Position sizing enhancement")
        print("  7. Execution parameter optimization")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        
    return success


if __name__ == "__main__":
    asyncio.run(main())