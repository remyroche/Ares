#!/usr/bin/env python3
"""
Example script demonstrating directional prediction with adversarial analysis.
"""

from datetime import datetime , timedelta
from src.analyst.ml_confidence_predictor import MLConfidencePredictor
from src.analyst.ml_confidence_predictor import MLConfidencePredictor
import asyncio

import numpy as np
import pandas as pd






async def create_sample_market_data() -> pd.DataFrame:
    """
    Create sample market data for testing.

    Returns:
        Sample market data DataFrame
    """
    # Generate sample market data
    dates = pd.date_range(start, datetime.now() - timedelta(days=30), end=datetime.now(), freq='1H')

    # Create realistic price movements
    np.random.seed(42)  # For reproducible results
    base_price = 100.0
    returns = np.random.normal(0, 0.02, len(dates))  # 2% daily volatility
    prices = [base_price]

    for ret in returns[1:]:
    pass
    pass
    pass
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)

    # Create additional features
    data = {
        'timestamp': dates,
        'price': prices,
        'volume': np.random.uniform(1000, 10000, len(dates)),
        'rsi': np.random.uniform(20, 80, len(dates)),
        'macd': np.random.uniform(-2, 2, len(dates)),
        'bollinger_upper': [p * 1.02 for p in prices],
        'bollinger_lower': [p * 0.98 for p in prices],
        'sma_20': [p * 1.001 for p in prices],
        'sma_50': [p * 0.999 for p in prices],
    }

    return pd.DataFrame(data)

async def demonstrate_directional_analysis():
    """
    Demonstrate the directional prediction with adversarial analysis.
    """
    print("🚀 Starting Directional Prediction with Adversarial Analysis Demo")
    print("=" * 70)

    # Create sample market data
    print("📊 Creating sample market data...")
    market_data = await create_sample_market_data()
    current_price = market_data['price'].iloc[-1]

    print(f"Current price: ${current_price:.2f}")
    print(f"Data points: {len(market_data)}")
    print()

    # Initialize ML Confidence Predictor
    print("🤖 Initializing ML Confidence Predictor...")
    config = {
        "ml_confidence_predictor": {
            "model_path": "models/confidence_predictor.joblib",
            "confidence_threshold": 0.6,
            "max_prediction_levels": 20
        }
    }

    predictor = MLConfidencePredictor(config)
    await predictor.initialize()

    # Perform directional prediction with adversarial analysis
    print("🎯 Performing directional prediction with adversarial analysis...")
    result = await predictor.predict_directional_with_adversarial_analysis(
        market_data,
        current_price
    )

    try:
        result = await predictor.predict_directional_with_adversarial_analysis(
            market_data,
            current_price
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
        )

        print("✅ Analysis completed successfully!")
        print()

        # Display primary direction prediction
        primary = result["primary_direction"]
        print("📈 PRIMARY DIRECTION PREDICTION:")
        print(f"   Direction: {primary['direction'].upper()}")
        print(f"   Confidence: {primary['confidence']:.2%}")
        print(f"   Up Confidence: {primary['up_confidence']:.2%}")
        print(f"   Down Confidence: {primary['down_confidence']:.2%}")
        print(f"   Magnitude Levels: {primary['magnitude_levels']}")
        print()

        # Display adversarial analysis
        print(warning(" ADVERSARIAL ANALYSIS:"))
        adversarial = result["adversarial_analysis"]

        for magnitude, analysis in adversarial.items():
    pass
    pass
    pass
            print(f"\\\\n   For {magnitude} movement:")
            print(f"     Risk Score: {analysis['risk_score']:.2%}")
            print(f"     Recommended Stop Loss: {analysis['recommended_stop_loss']:.1f}%")
            print("     Adverse Probabilities:")

            for level, prob in analysis['adverse_probabilities'].items():
    pass
    pass
    pass
                if prob > 0.1:  # Only show significant probabilities
                    print(f"       {level}: {prob:.1%}")

        print()

        # Display risk assessment
        risk = result["risk_assessment"]
        print("🛡️  RISK ASSESSMENT:")
        print(f"   Overall Risk Score: {risk['overall_risk_score']:.2%}")
        print(f"   Risk Category: {risk['risk_category']}")
        print(f"   Recommendation: {risk['recommendation']}")
        print()

        # Display risk levels breakdown
        print("📊 RISK LEVELS BREAKDOWN:")
        for level_info in risk['risk_levels']:
    pass
    pass
    pass
            print(f"   {level_info['magnitude']}: Risk {level_info['risk_score']:.2%}, Stop Loss {level_info['stop_loss']:.1f}%")

        print()
        print("=" * 70)
        print("✅ Demo completed successfully!")

    except ValueError as e:
        print(failed("Analysis failed: {str(e)}"))
        print("   This is expected when the model is not trained or data is invalid.")
        print("   The system correctly refuses to provide fallback predictions.")
    except Exception as e:
        print(warning("Unexpected error: {str(e)}"))

    # Cleanup
    await predictor.stop()

def print_usage_example():
    pass
    pass
    pass
    """
    Print usage example for the new functionality.
    """
    print("""
USAGE EXAMPLE:

```python

# Initialize predictor
config = {
    "ml_confidence_predictor": {
        "model_path": "models/confidence_predictor.joblib",
        "confidence_threshold": 0.6,
        "max_prediction_levels": 20
    }
}

predictor = MLConfidencePredictor(config)
await predictor.initialize()

# Perform directional prediction with adversarial analysis
result = await predictor.predict_directional_with_adversarial_analysis(
    market_data = current_price
)

# Access results
primary_direction = result["primary_direction"]
adversarial_analysis = result["adversarial_analysis"]
risk_assessment = result["risk_assessment"]

# Example: Check if UP direction is predicted with low risk
if (primary_direction["direction"] == "up" and
    risk_assessment["risk_category"] == "LOW"):
    print("Good opportunity for UP position")
```
""")

if __name__ == "__main__":
    pass
    pass
    pass
    print("🎯 ML Directional Prediction with Adversarial Analysis")
    print("=" * 70)
    print()

    # Run the demonstration
    asyncio.run(demonstrate_directional_analysis())

    print()
    print_usage_example()
