#!/usr/bin/env python3
"""
Test script to verify the Analyst probability logic is working correctly.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime

# Test the ML Confidence Predictor directly
async def test_ml_confidence_predictor():
    """Test the ML Confidence Predictor fallback predictions."""
    from src.analyst.ml_confidence_predictor import MLConfidencePredictor
    
    print("Testing ML Confidence Predictor...")
    print("="*60)
    
    # Create a simple config
    config = {
        "ml_confidence_predictor": {
            "model_path": "models/test_confidence_predictor.joblib",
            "min_samples_for_training": 500,
            "confidence_threshold": 0.6,
        }
    }
    
    # Initialize predictor
    predictor = MLConfidencePredictor(config)
    
    # Test fallback predictions directly
    current_price = 45000.0
    fallback_preds = predictor._generate_fallback_predictions(current_price)
    
    print("\nFallback Predictions Test:")
    print("-"*40)
    print(f"Current Price: ${current_price:,.2f}")
    print(f"Model Status: {fallback_preds.get('model_status', 'unknown')}")
    
    print("\nPrice Target Confidences:")
    targets = fallback_preds.get('price_target_confidences', {})
    for target, conf in sorted(targets.items(), key=lambda x: float(x[0].replace('%', ''))):
        print(f"  {target:>6}: {conf:.3f}")
    
    print("\nAdversarial Confidences:")
    adversarial = fallback_preds.get('adversarial_confidences', {})
    for target, conf in sorted(adversarial.items(), key=lambda x: float(x[0].replace('%', '')))[:5]:
        print(f"  {target:>6}: {conf:.3f}")
    
    print("\nDirectional Analysis:")
    direction = fallback_preds.get('directional_analysis', {})
    print(f"  Primary Direction: {direction.get('primary_direction', 'N/A')}")
    print(f"  Bullish Probability: {direction.get('bullish_probability', 0):.3f}")
    print(f"  Bearish Probability: {direction.get('bearish_probability', 0):.3f}")
    
    # Test directional analysis calculation
    print("\n" + "="*60)
    print("Testing Directional Analysis Calculation...")
    print("-"*40)
    
    # Create test price target confidences
    test_targets = {
        "0.1%": 0.8,  # High confidence for small move
        "0.2%": 0.7,
        "0.3%": 0.6,
        "0.5%": 0.5,
        "1.0%": 0.3,  # Lower confidence for larger moves
        "1.5%": 0.2,
        "2.0%": 0.1
    }
    
    test_adversarial = {
        "0.1%": 0.2,  # Low risk
        "0.5%": 0.3,
        "1.0%": 0.4,  # Moderate risk
        "2.0%": 0.5
    }
    
    # Calculate directional analysis
    directional = predictor._generate_directional_confidence_analysis(
        test_targets, test_adversarial, current_price
    )
    
    print("Test Scenario: Bullish bias with decreasing confidence")
    print(f"  Primary Direction: {directional.get('primary_direction')}")
    print(f"  Direction Confidence: {directional.get('direction_confidence'):.3f}")
    print(f"  Bullish Probability: {directional.get('bullish_probability'):.3f}")
    print(f"  Bearish Probability: {directional.get('bearish_probability'):.3f}")
    print(f"  Trend Strength: {directional.get('trend_strength'):.3f}")
    print(f"  Momentum Score: {directional.get('momentum_score'):.3f}")
    
    print("\n✅ ML Confidence Predictor test completed!")


async def test_analyst_integration():
    """Test the full Analyst integration."""
    from src.analyst.analyst import Analyst
    
    print("\n" + "="*60)
    print("Testing Analyst Integration...")
    print("="*60)
    
    # Create config
    config = {
        "analyst": {
            "enable_ml_confidence_predictor": True,
            "enable_dual_model_system": False,  # Disable to test ML predictor alone
            "enable_market_health_analysis": False,
            "enable_liquidation_risk_analysis": False,
            "enable_feature_engineering": False,
        }
    }
    
    # Initialize Analyst
    analyst = Analyst(config)
    
    # Test probability extraction
    print("\nTesting _extract_price_target_probabilities method...")
    
    # Test case 1: ML predictions available
    ml_predictions = {
        "price_target_confidences": {
            "0.5%": 0.6,
            "1.0%": 0.4,
            "1.5%": 0.3
        },
        "adversarial_confidences": {
            "0.5%": 0.3,
            "1.0%": 0.4
        },
        "directional_analysis": {
            "primary_direction": "BULLISH",
            "direction_confidence": 0.65,
            "bullish_probability": 0.65,
            "bearish_probability": 0.35
        }
    }
    
    result = analyst._extract_price_target_probabilities({}, ml_predictions, {})
    
    print("\nExtracted Probabilities:")
    print(f"  Price Targets: {result['price_targets']}")
    print(f"  Direction: {result['direction']['primary']}")
    print(f"  Summary: {result['summary']}")
    
    print("\n✅ Analyst integration test completed!")


async def main():
    """Run all tests."""
    await test_ml_confidence_predictor()
    await test_analyst_integration()
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())