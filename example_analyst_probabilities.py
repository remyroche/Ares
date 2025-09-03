#!/usr/bin/env python3
"""
Example script demonstrating how the Analyst module outputs probabilities
for hitting specific price targets.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from src.analyst.analyst import Analyst
from src.analyst.ml_confidence_predictor import MLConfidencePredictor


async def demonstrate_analyst_probabilities():
    """Demonstrate the Analyst's probability outputs."""
    
    # Create a simple configuration
    config = {
        "analyst": {
            "analysis_interval": 3600,
            "enable_technical_analysis": True,
            "enable_dual_model_system": True,
            "enable_market_health_analysis": True,
            "enable_liquidation_risk_analysis": True,
            "enable_feature_engineering": True,
            "enable_ml_confidence_predictor": True,
        },
        "ml_confidence_predictor": {
            "model_path": "models/confidence_predictor.joblib",
            "min_samples_for_training": 500,
            "confidence_threshold": 0.6,
        }
    }
    
    # Initialize the Analyst
    analyst = Analyst(config)
    await analyst.initialize()
    
    # Create sample market data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
    market_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(45000, 46000, 100),
        'high': np.random.uniform(45500, 46500, 100),
        'low': np.random.uniform(44500, 45500, 100),
        'close': np.random.uniform(45000, 46000, 100),
        'volume': np.random.uniform(100, 1000, 100),
    })
    current_price = 45500.0
    
    # Create analysis input
    analysis_input = {
        "market_data": market_data,
        "current_price": current_price,
        "current_position": None,
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "timeframe": "1h"
    }
    
    # Perform analysis
    print("Performing comprehensive analysis...")
    await analyst.execute_comprehensive_analysis(analysis_input)
    
    # Get analysis results
    results = analyst.get_analysis_results()
    
    # Display price target probabilities
    print("\n" + "="*60)
    print("ANALYST PROBABILITY OUTPUT DEMONSTRATION")
    print("="*60)
    
    if "price_target_probabilities" in results:
        probs = results["price_target_probabilities"]
        
        print(f"\nCurrent Price: ${current_price:,.2f}")
        print(f"Analysis Timestamp: {probs.get('timestamp', 'N/A')}")
        
        print("\n📊 PRICE TARGET PROBABILITIES:")
        print("-" * 40)
        price_targets = probs.get("price_targets", {})
        if price_targets:
            for target, probability in sorted(price_targets.items(), key=lambda x: float(x[0].replace('%', ''))):
                target_price = current_price * (1 + float(target.replace('%', '')) / 100)
                print(f"  {target:>6} movement (${target_price:,.2f}): {probability:.1%}")
        else:
            print("  No price targets available")
        
        print("\n⚠️  ADVERSARIAL RISK PROBABILITIES:")
        print("-" * 40)
        risks = probs.get("adversarial_risks", {})
        if risks:
            for risk, probability in sorted(risks.items(), key=lambda x: float(x[0].replace('%', ''))):
                risk_price = current_price * (1 - float(risk.replace('%', '')) / 100)
                print(f"  {risk:>6} downside (${risk_price:,.2f}): {probability:.1%}")
        else:
            print("  No adversarial risks available")
        
        print("\n🎯 DIRECTIONAL ANALYSIS:")
        print("-" * 40)
        direction = probs.get("direction", {})
        print(f"  Primary Direction: {direction.get('primary', 'UNKNOWN')}")
        print(f"  Direction Confidence: {direction.get('confidence', 0):.1%}")
        print(f"  Bullish Probability: {direction.get('bullish_probability', 0):.1%}")
        print(f"  Bearish Probability: {direction.get('bearish_probability', 0):.1%}")
        print(f"  Neutral Probability: {direction.get('neutral_probability', 0):.1%}")
        
        print(f"\n📝 SUMMARY: {probs.get('summary', 'N/A')}")
    
    # Also show ML predictions if available
    if "ml_predictions" in results and results["ml_predictions"]:
        ml_preds = results["ml_predictions"]
        print("\n" + "="*60)
        print("ML CONFIDENCE PREDICTIONS (Raw Output)")
        print("="*60)
        
        if "price_target_confidences" in ml_preds:
            print("\nPrice Target Confidences:")
            for target, conf in sorted(ml_preds["price_target_confidences"].items(), 
                                     key=lambda x: float(x[0].replace('%', ''))):
                print(f"  {target}: {conf:.3f}")
        
        print(f"\nModel Status: {ml_preds.get('model_status', 'unknown')}")
        print(f"Overall Confidence: {ml_preds.get('confidence', 0):.3f}")
    
    # Show trading decision if available
    if "trading_decision" in results and results["trading_decision"]:
        decision = results["trading_decision"]
        print("\n" + "="*60)
        print("TRADING DECISION")
        print("="*60)
        print(f"Action: {decision.get('action', 'UNKNOWN')}")
        print(f"Signal: {decision.get('signal', 'UNKNOWN')}")
        print(f"Direction: {decision.get('direction', 'UNKNOWN')}")
        print(f"Confidence: {decision.get('confidence', 0):.3f}")
        print(f"Reason: {decision.get('reason', 'N/A')}")
    
    print("\n" + "="*60)
    print("END OF DEMONSTRATION")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(demonstrate_analyst_probabilities())