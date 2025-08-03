#!/usr/bin/env python3
"""
Enhanced Ensemble Integration Demo

This script demonstrates how multi-timeframe training integrates into the existing
ensemble system, making each individual model (XGBoost, LSTM, etc.) a multi-timeframe ensemble.
"""

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from src.analyst.predictive_ensembles.enhanced_ensemble_orchestrator import (
    EnhancedRegimePredictiveEnsembles,
)
from src.analyst.predictive_ensembles.two_tier_integration import TwoTierIntegration
from src.config import CONFIG
from src.utils.logger import system_logger


def create_demo_data():
    """Create demo multi-timeframe data for testing."""
    logger = system_logger.getChild("DemoData")

    # Create synthetic data for different timeframes
    base_date = datetime.now() - timedelta(days=30)
    timeframes = ["1m", "5m", "15m", "1h"]

    prepared_data = {}

    for timeframe in timeframes:
        logger.info(f"Creating demo data for {timeframe}")

        # Create synthetic features
        n_samples = (
            1000
            if timeframe == "1m"
            else 500
            if timeframe == "5m"
            else 200
            if timeframe == "15m"
            else 100
        )

        data = {
            "timestamp": [base_date + timedelta(minutes=i) for i in range(n_samples)],
            "open": np.random.uniform(100, 200, n_samples),
            "high": np.random.uniform(100, 200, n_samples),
            "low": np.random.uniform(100, 200, n_samples),
            "close": np.random.uniform(100, 200, n_samples),
            "volume": np.random.uniform(1000, 10000, n_samples),
            "rsi": np.random.uniform(0, 100, n_samples),
            "macd": np.random.uniform(-10, 10, n_samples),
            "bollinger_upper": np.random.uniform(110, 210, n_samples),
            "bollinger_lower": np.random.uniform(90, 190, n_samples),
            "atr": np.random.uniform(1, 10, n_samples),
            "target": np.random.choice(["BUY", "SELL", "HOLD"], n_samples),
        }

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)

        prepared_data[timeframe] = df

    return prepared_data


def demo_enhanced_ensemble_training():
    """Demonstrate enhanced ensemble training."""
    logger = system_logger.getChild("EnhancedEnsembleDemo")
    logger.info("Starting Enhanced Ensemble Training Demo")

    # Create demo data
    prepared_data = create_demo_data()
    logger.info(f"Created demo data for timeframes: {list(prepared_data.keys())}")

    # Initialize enhanced ensemble
    config = CONFIG.get("ENHANCED_ENSEMBLE", {})
    enhanced_ensembles = EnhancedRegimePredictiveEnsembles(config)

    # Train enhanced ensembles
    logger.info("Training enhanced multi-timeframe ensembles...")
    success = enhanced_ensembles.train_all_models("ETHUSDT", prepared_data)

    if success:
        logger.info("✅ Enhanced ensemble training completed successfully!")
        return enhanced_ensembles
    logger.error("❌ Enhanced ensemble training failed!")
    return None


def demo_enhanced_predictions(enhanced_ensembles):
    """Demonstrate enhanced predictions."""
    logger = system_logger.getChild("EnhancedPredictionsDemo")
    logger.info("Starting Enhanced Predictions Demo")

    # Create demo current features
    current_features = pd.DataFrame(
        {
            "open": [150.0],
            "high": [155.0],
            "low": [148.0],
            "close": [152.0],
            "volume": [5000],
            "rsi": [65.0],
            "macd": [2.5],
            "bollinger_upper": [160.0],
            "bollinger_lower": [140.0],
            "atr": [5.0],
        },
    )

    # Get enhanced prediction
    prediction = enhanced_ensembles.get_all_predictions("ETHUSDT", current_features)

    # Display results
    logger.info("📊 Enhanced Prediction Results:")
    logger.info(f"Final Decision: {prediction['prediction']}")
    logger.info(f"Confidence: {prediction['confidence']:.2f}")
    logger.info(f"Regime: {prediction['regime']}")
    logger.info(
        f"Multi-timeframe Enabled: {prediction.get('multi_timeframe_enabled', False)}",
    )

    # Display enhanced predictions
    if "enhanced_predictions" in prediction:
        logger.info("\n🔍 Enhanced Model Predictions:")
        for ensemble_key, pred in prediction["enhanced_predictions"].items():
            conf = prediction["enhanced_confidences"].get(ensemble_key, 0.0)
            logger.info(f"  {ensemble_key}: {pred} (confidence: {conf:.2f})")

    # Display timeframe details
    if "timeframe_details" in prediction:
        logger.info("\n⏰ Timeframe Details:")
        for ensemble_key, details in prediction["timeframe_details"].items():
            logger.info(f"\n  {ensemble_key}:")
            for tf, pred in details["timeframe_predictions"].items():
                conf = details["timeframe_confidences"].get(tf, 0.0)
                logger.info(f"    {tf}: {pred} (confidence: {conf:.2f})")

    return prediction


def demo_two_tier_integration():
    """Demonstrate two-tier integration."""
    logger = system_logger.getChild("TwoTierDemo")
    logger.info("Starting Two-Tier Integration Demo")

    # Create demo ensemble prediction
    ensemble_prediction = {
        "prediction": "BUY",
        "confidence": 0.75,
        "regime": "BULL_TREND",
        "base_predictions": {
            "BULL_TREND_xgboost": "BUY",
            "BULL_TREND_lstm": "BUY",
            "BULL_TREND_random_forest": "HOLD",
        },
    }

    # Create demo current data
    current_data = {
        "price": 152.0,
        "sr_levels": [150.0, 160.0],
        "volume": 5000,
        "rsi": 65.0,
    }

    # Initialize two-tier integration
    two_tier = TwoTierIntegration()

    # Enhance ensemble prediction
    enhanced_prediction = two_tier.enhance_ensemble_prediction(
        ensemble_prediction,
        current_data,
    )

    # Display results
    logger.info("🎯 Two-Tier Integration Results:")
    logger.info(f"Original Prediction: {ensemble_prediction['prediction']}")
    logger.info(f"Original Confidence: {ensemble_prediction['confidence']:.2f}")
    logger.info(
        f"Tier 1 Direction: {enhanced_prediction.get('tier1_direction', 'N/A')}",
    )
    logger.info(f"Tier 1 Strategy: {enhanced_prediction.get('tier1_strategy', 'N/A')}")
    logger.info(
        f"Tier 2 Timing Signal: {enhanced_prediction.get('tier2_timing_signal', 'N/A')}",
    )
    logger.info(
        f"Tier 2 Should Execute: {enhanced_prediction.get('tier2_should_execute', 'N/A')}",
    )
    logger.info(f"Final Decision: {enhanced_prediction.get('final_decision', 'N/A')}")

    if "position_size_multiplier" in enhanced_prediction:
        logger.info(
            f"Position Size Multiplier: {enhanced_prediction['position_size_multiplier']:.2f}",
        )

    if "liquidation_risk_adjustment" in enhanced_prediction:
        risk_adj = enhanced_prediction["liquidation_risk_adjustment"]
        logger.info(f"Risk Multiplier: {risk_adj.get('risk_multiplier', 'N/A')}")
        logger.info(f"Confidence Level: {risk_adj.get('confidence_level', 'N/A')}")

    return enhanced_prediction


def demo_ensemble_info(enhanced_ensembles):
    """Demonstrate ensemble information."""
    logger = system_logger.getChild("EnsembleInfoDemo")
    logger.info("Starting Ensemble Information Demo")

    # Get ensemble info
    info = enhanced_ensembles.get_enhanced_ensemble_info()

    logger.info("📋 Enhanced Ensemble Information:")
    logger.info(f"Active Timeframes: {info['active_timeframes']}")
    logger.info(f"Timeframe Set: {info['timeframe_set']}")
    logger.info(f"Model Types: {info['model_types']}")

    logger.info("\n🔧 Enhanced Ensembles:")
    for regime_key, ensembles in info["enhanced_ensembles"].items():
        logger.info(f"\n  {regime_key}:")
        for ensemble_key, details in ensembles.items():
            trained = "✅" if details["trained"] else "❌"
            logger.info(f"    {trained} {ensemble_key}")
            logger.info(f"      Model: {details['model_name']}")
            logger.info(f"      Regime: {details['regime']}")
            logger.info(f"      Timeframes: {details['timeframe_models']}")


def main():
    """Run the enhanced ensemble integration demo."""
    logger = system_logger.getChild("EnhancedEnsembleDemo")
    logger.info("🚀 Starting Enhanced Ensemble Integration Demo")

    try:
        # Demo 1: Enhanced Ensemble Training
        logger.info("\n" + "=" * 50)
        logger.info("DEMO 1: Enhanced Ensemble Training")
        logger.info("=" * 50)

        enhanced_ensembles = demo_enhanced_ensemble_training()

        if enhanced_ensembles:
            # Demo 2: Enhanced Predictions
            logger.info("\n" + "=" * 50)
            logger.info("DEMO 2: Enhanced Predictions")
            logger.info("=" * 50)

            demo_enhanced_predictions(enhanced_ensembles)

            # Demo 3: Ensemble Information
            logger.info("\n" + "=" * 50)
            logger.info("DEMO 3: Ensemble Information")
            logger.info("=" * 50)

            demo_ensemble_info(enhanced_ensembles)

        # Demo 4: Two-Tier Integration
        logger.info("\n" + "=" * 50)
        logger.info("DEMO 4: Two-Tier Integration")
        logger.info("=" * 50)

        demo_two_tier_integration()

        logger.info("\n" + "=" * 50)
        logger.info("✅ Enhanced Ensemble Integration Demo Completed Successfully!")
        logger.info("=" * 50)

        logger.info("\n🎯 Key Benefits Demonstrated:")
        logger.info("  ✅ Multi-timeframe training for each model type")
        logger.info("  ✅ Enhanced predictions with timeframe details")
        logger.info("  ✅ Two-tier decision system integration")
        logger.info("  ✅ Preserved confidence levels and liquidation risk")
        logger.info("  ✅ Backward compatibility with existing ensemble system")

    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
