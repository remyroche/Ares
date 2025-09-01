#!/usr/bin/env python3
"""
Test script for Enhanced Prediction Integration
Demonstrates the integration of price and confidence predictions from enhanced training manager steps 6-14
into the Analyst and Tactician components.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.analyst.analyst import setup_analyst
from src.tactician.tactician import setup_tactician
from src.utils.logger import system_logger


def create_sample_market_data() -> pd.DataFrame:
    """Create sample market data for testing."""
    import numpy as np

    # Generate sample OHLCV data
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')

    # Create realistic price data with some volatility
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0, 0.001, len(dates))  # 0.1% volatility per minute

    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # Create OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Add some intraday volatility
        volatility = 0.002  # 0.2% intraday volatility
        high = price * (1 + abs(np.random.normal(0, volatility)))
        low = price * (1 - abs(np.random.normal(0, volatility)))
        open_price = price * (1 + np.random.normal(0, volatility * 0.5))
        close_price = price
        volume = np.random.randint(1000, 10000)

        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


def create_sample_regime_info() -> Dict[str, Any]:
    """Create sample regime information."""
    return {
        "regime": "trending_bullish",
        "confidence": 0.85,
        "regime_transition_probability": 0.15,
        "regime_duration": 45,
        "regime_strength": 0.78
    }


def create_sample_analyst_signals() -> Dict[str, Any]:
    """Create sample analyst signals."""
    return {
        "signal": 1,  # Buy signal
        "confidence": 0.82,
        "prediction": 0.75,
        "direction": "long",
        "strength": "strong",
        "timestamp": datetime.now().isoformat()
    }


async def test_analyst_enhanced_predictions():
    """Test the Analyst enhanced prediction integration."""
    print("🧪 Testing Analyst Enhanced Prediction Integration...")

    try:
        # Load configuration
        config_path = Path("src/config/enhanced_prediction_integration.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {
                "analyst": {
                    "enable_enhanced_predictions": True,
                    "enhanced_prediction_integrator": {
                        "data_dir": "data/training",
                        "models_dir": "models",
                        "confidence_threshold": 0.7,
                        "price_prediction_threshold": 0.6
                    }
                }
            }

        # Setup analyst
        analyst = await setup_analyst(config)
        if not analyst:
            print("❌ Failed to setup Analyst")
            return False

        print("✅ Analyst setup completed")

        # Create sample data
        market_data = create_sample_market_data()
        regime_info = create_sample_regime_info()

        # Prepare analysis input
        analysis_input = {
            "market_data": market_data,
            "current_price": market_data["close"].iloc[-1],
            "current_position": 0,
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "timeframe": "1m"
        }

        # Execute analysis
        print("🔄 Executing analysis with enhanced predictions...")
        success = await analyst.execute_analysis(analysis_input)

        if success:
            # Get analysis results
            results = analyst.get_analysis_results()

            print("✅ Analysis completed successfully")
            print(f"📊 Analysis timestamp: {results.get('timestamp')}")
            print(f"💰 Current price: {results.get('current_price')}")

            # Check for enhanced predictions
            enhanced_predictions = results.get("enhanced_predictions", {})
            if enhanced_predictions:
                print("🎯 Enhanced predictions found:")
                print(f"   - Price predictions: {len(enhanced_predictions.get('price_predictions', {}))}")
                print(f"   - Confidence scores: {len(enhanced_predictions.get('confidence_scores', {}))}")
                print(f"   - Calibrated predictions: {len(enhanced_predictions.get('calibrated_predictions', {}))}")

                # Show some sample predictions
                price_predictions = enhanced_predictions.get("price_predictions", {})
                if price_predictions:
                    print("   📈 Sample price predictions:")
                    for name, pred in list(price_predictions.items())[:3]:
                        print(f"      {name}: {pred.get('prediction', 'N/A')} (confidence: {pred.get('confidence', 'N/A')})")
            else:
                print("⚠️ No enhanced predictions found (models may not be available)")

            return True
        else:
            print("❌ Analysis failed")
            return False

    except Exception as e:
        print(f"❌ Error testing Analyst enhanced predictions: {e}")
        return False


async def test_tactician_enhanced_predictions():
    """Test the Tactician enhanced prediction integration."""
    print("\n🧪 Testing Tactician Enhanced Prediction Integration...")

    try:
        # Load configuration
        config_path = Path("src/config/enhanced_prediction_integration.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {
                "tactician": {
                    "enable_enhanced_predictions": True,
                    "tactician_enhanced_prediction_integrator": {
                        "data_dir": "data/training",
                        "models_dir": "models",
                        "confidence_threshold": 0.7,
                        "price_prediction_threshold": 0.6,
                        "entry_threshold": 0.65,
                        "exit_threshold": 0.55
                    }
                }
            }

        # Setup tactician
        tactician = await setup_tactician(config)
        if not tactician:
            print("❌ Failed to setup Tactician")
            return False

        print("✅ Tactician setup completed")

        # Create sample data
        market_data = create_sample_market_data()
        regime_info = create_sample_regime_info()
        analyst_signals = create_sample_analyst_signals()

        # Test enhanced predictions directly
        if tactician.enhanced_prediction_integrator:
            print("🔄 Generating enhanced tactician predictions...")

            predictions = await tactician._get_enhanced_predictions(
                market_data=market_data,
                regime_info=regime_info,
                analyst_signals=analyst_signals,
                symbol="ETHUSDT",
                exchange="BINANCE",
                timeframe="1m"
            )

            if predictions:
                print("✅ Enhanced tactician predictions generated successfully")
                print(f"📊 Prediction timestamp: {predictions.get('timestamp')}")

                # Show prediction categories
                print("🎯 Prediction categories:")
                print(f"   - ML confidence predictions: {len(predictions.get('ml_confidence_predictions', {}))}")
                print(f"   - Calibrated confidence scores: {len(predictions.get('calibrated_confidence_scores', {}))}")
                print(f"   - Optimization weights: {len(predictions.get('optimization_weights', {}))}")
                print(f"   - HMM predictions: {len(predictions.get('hmm_predictions', {}))}")

                # Show ML confidence predictions
                ml_confidence = predictions.get("ml_confidence_predictions", {})
                if ml_confidence:
                    aggregate_ml = ml_confidence.get("aggregate_ml_confidence", {})
                    print(f"   📈 ML Confidence:")
                    print(f"      - Weighted ML confidence: {aggregate_ml.get('weighted_ml_confidence', 'N/A')}")
                    print(f"      - HMM avg confidence: {aggregate_ml.get('hmm_avg_confidence', 'N/A')}")
                    print(f"      - Analyst confidence: {aggregate_ml.get('analyst_confidence', 'N/A')}")

                # Test position sizer enhancement
                print("🔄 Testing position sizer enhancement...")
                enhanced_position = await tactician.enhanced_prediction_integrator.enhance_position_sizer(
                    base_position_size=0.1,
                    analyst_confidence=0.8,
                    enhanced_predictions=predictions
                )
                print(f"   📏 Enhanced position size: {enhanced_position.get('enhanced_position_size', 'N/A')}")

                # Test leverage sizer enhancement
                print("🔄 Testing leverage sizer enhancement...")
                enhanced_leverage = await tactician.enhanced_prediction_integrator.enhance_leverage_sizer(
                    base_leverage=50.0,
                    risk_score=0.3,
                    enhanced_predictions=predictions
                )
                print(f"   ⚡ Enhanced leverage: {enhanced_leverage.get('enhanced_leverage', 'N/A')}")

                return True
            else:
                print("⚠️ No enhanced predictions generated (models may not be available)")
                return False
        else:
            print("⚠️ Enhanced prediction integrator not available")
            return False

    except Exception as e:
        print(f"❌ Error testing Tactician enhanced predictions: {e}")
        return False


async def test_integration_workflow():
    """Test the complete integration workflow."""
    print("\n🧪 Testing Complete Integration Workflow...")

    try:
        # Load configuration
        config_path = Path("src/config/enhanced_prediction_integration.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {
                "analyst": {"enable_enhanced_predictions": True},
                "tactician": {"enable_enhanced_predictions": True}
            }

        # Setup both components
        analyst = await setup_analyst(config)
        tactician = await setup_tactician(config)

        if not analyst or not tactician:
            print("❌ Failed to setup components")
            return False

        print("✅ Both components setup completed")

        # Create sample data
        market_data = create_sample_market_data()
        regime_info = create_sample_regime_info()

        # Step 1: Analyst analysis
        print("🔄 Step 1: Running Analyst analysis...")
        analysis_input = {
            "market_data": market_data,
            "current_price": market_data["close"].iloc[-1],
            "current_position": 0,
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "timeframe": "1m"
        }

        analysis_success = await analyst.execute_analysis(analysis_input)
        if not analysis_success:
            print("❌ Analyst analysis failed")
            return False

        # Get analyst results
        analyst_results = analyst.get_analysis_results()
        enhanced_predictions = analyst_results.get("enhanced_predictions", {})

        print("✅ Analyst analysis completed")
        print(f"   - Enhanced predictions: {len(enhanced_predictions.get('price_predictions', {}))}")

        # Step 2: Tactician predictions using analyst signals
        print("🔄 Step 2: Running Tactician predictions...")

        # Create analyst signals from analyst results
        analyst_signals = {
            "signal": 1,  # Default buy signal
            "confidence": 0.8,
            "prediction": 0.7,
            "enhanced_predictions": enhanced_predictions
        }

        tactician_predictions = await tactician._get_enhanced_predictions(
            market_data=market_data,
            regime_info=regime_info,
            analyst_signals=analyst_signals,
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframe="1m"
        )

        if tactician_predictions:
            print("✅ Tactician predictions completed")
            print(f"   - ML confidence predictions: {len(tactician_predictions.get('ml_confidence_predictions', {}))}")
            print(f"   - Calibrated confidence scores: {len(tactician_predictions.get('calibrated_confidence_scores', {}))}")
            print(f"   - HMM predictions: {len(tactician_predictions.get('hmm_predictions', {}))}")

            # Show integration results
            print("\n🎯 Integration Results:")

            # ML confidence
            ml_confidence = tactician_predictions.get("ml_confidence_predictions", {})
            if ml_confidence:
                aggregate_ml = ml_confidence.get("aggregate_ml_confidence", {})
                print(f"   📈 ML Confidence: {aggregate_ml.get('weighted_ml_confidence', 'N/A')}")

            # Enhanced position sizing
            enhanced_position = await tactician.enhanced_prediction_integrator.enhance_position_sizer(
                base_position_size=0.1,
                analyst_confidence=0.8,
                enhanced_predictions=tactician_predictions
            )
            print(f"   📏 Enhanced position size: {enhanced_position.get('enhanced_position_size', 'N/A')}")

            # Enhanced leverage sizing
            enhanced_leverage = await tactician.enhanced_prediction_integrator.enhance_leverage_sizer(
                base_leverage=50.0,
                risk_score=0.3,
                enhanced_predictions=tactician_predictions
            )
            print(f"   ⚡ Enhanced leverage: {enhanced_leverage.get('enhanced_leverage', 'N/A')}")

            return True
        else:
            print("❌ Tactician predictions failed")
            return False

    except Exception as e:
        print(f"❌ Error testing integration workflow: {e}")
        return False


async def main():
    """Main test function."""
    print("🚀 Enhanced Prediction Integration Test Suite")
    print("=" * 50)

    # Test individual components
    analyst_success = await test_analyst_enhanced_predictions()
    tactician_success = await test_tactician_enhanced_predictions()

    # Test complete workflow
    workflow_success = await test_integration_workflow()

    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   Analyst Enhanced Predictions: {'✅ PASS' if analyst_success else '❌ FAIL'}")
    print(f"   Tactician Enhanced Predictions: {'✅ PASS' if tactician_success else '❌ FAIL'}")
    print(f"   Complete Integration Workflow: {'✅ PASS' if workflow_success else '❌ FAIL'}")

    if analyst_success and tactician_success and workflow_success:
        print("\n🎉 All tests passed! Enhanced prediction integration is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the logs for details.")

    print("\n📝 Notes:")
    print("   - If models are not available, predictions will be empty but components will still initialize")
    print("   - This is expected behavior when running without trained models from steps 6-14")
    print("   - To see full functionality, run the enhanced training manager first")


if __name__ == "__main__":
    asyncio.run(main())