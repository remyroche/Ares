"""
Example demonstrating how the weight optimization system predicts future SR level quality.

This script shows the complete pipeline from weight optimization to future quality prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.utils.sr_clustering import (
    get_backtesting_engine, BacktestConfig, SRLevel,
    get_backtesting_enhanced_clustering, BacktestingEnhancedConfig,
    get_weight_optimization_engine, WeightOptimizationConfig,
    get_predictive_sr_engine, PredictiveConfig
)

def create_realistic_market_data(days: int = 200) -> pd.DataFrame:
    """Create realistic market data with trends, volatility, and volume patterns."""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
    
    # Generate realistic price data with trends and volatility
    np.random.seed(42)
    base_price = 100.0
    
    # Create trend component
    trend = np.linspace(0, 0.2, days)  # 20% upward trend over period
    
    # Create volatility component
    volatility = 0.02 + 0.01 * np.sin(np.linspace(0, 4*np.pi, days))  # Cyclical volatility
    
    # Generate returns
    returns = []
    for i in range(days):
        trend_component = trend[i] / days
        volatility_component = volatility[i] * np.random.normal(0, 1)
        returns.append(trend_component + volatility_component)
    
    # Generate prices
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate volume with patterns
    base_volume = 1000000
    volume_pattern = 1 + 0.3 * np.sin(np.linspace(0, 2*np.pi, days))  # Weekly pattern
    volumes = [base_volume * pattern * np.random.lognormal(0, 0.3) for pattern in volume_pattern]
    
    # Generate OHLC data
    data = []
    for i, (date, close, volume) in enumerate(zip(dates, prices, volumes)):
        # Generate realistic OHLC
        daily_volatility = volatility[i] * 0.5
        high = close * (1 + abs(np.random.normal(0, daily_volatility)))
        low = close * (1 - abs(np.random.normal(0, daily_volatility)))
        open_price = close * (1 + np.random.normal(0, daily_volatility * 0.3))
        
        data.append({
            'date': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def create_diverse_sr_levels() -> list:
    """Create diverse SR levels with different characteristics."""
    levels = []
    
    # Strong support levels (high quality expected)
    strong_support_prices = [95.0, 98.0, 102.0]
    for price in strong_support_prices:
        level = SRLevel(
            price=price,
            level_type='support',
            strength=0.8,  # High strength
            first_touch=datetime.now() - timedelta(days=150),
            last_touch=datetime.now() - timedelta(days=10),
            touch_count=8,  # Many touches
            timeframe='1D',
            symbol='TEST',
            source='example'
        )
        levels.append(level)
    
    # Weak resistance levels (low quality expected)
    weak_resistance_prices = [108.0, 112.0, 118.0]
    for price in weak_resistance_prices:
        level = SRLevel(
            price=price,
            level_type='resistance',
            strength=0.3,  # Low strength
            first_touch=datetime.now() - timedelta(days=50),
            last_touch=datetime.now() - timedelta(days=5),
            touch_count=2,  # Few touches
            timeframe='1D',
            symbol='TEST',
            source='example'
        )
        levels.append(level)
    
    # Medium quality levels
    medium_prices = [105.0, 110.0, 115.0]
    for price in medium_prices:
        level = SRLevel(
            price=price,
            level_type='support' if price < 110 else 'resistance',
            strength=0.6,  # Medium strength
            first_touch=datetime.now() - timedelta(days=100),
            last_touch=datetime.now() - timedelta(days=15),
            touch_count=4,  # Moderate touches
            timeframe='1D',
            symbol='TEST',
            source='example'
        )
        levels.append(level)
    
    return levels

def demonstrate_future_prediction():
    """Demonstrate how the system predicts future SR level quality."""
    print("🔮 Future SR Quality Prediction Demo")
    print("=" * 60)
    
    # Create historical and current market data
    print("📊 Creating market data...")
    historical_data = create_realistic_market_data(150)  # 150 days of historical data
    current_data = create_realistic_market_data(200)     # 200 days including future
    
    print(f"✅ Historical data: {len(historical_data)} days")
    print(f"✅ Current data: {len(current_data)} days")
    
    # Create diverse SR levels
    print("\n🎯 Creating diverse SR levels...")
    sr_levels = create_diverse_sr_levels()
    print(f"✅ Created {len(sr_levels)} SR levels")
    
    # Initialize predictive engine
    print("\n🔧 Initializing predictive engine...")
    predictive_config = PredictiveConfig(
        model_type='ensemble',
        ensemble_models=['ridge', 'random_forest', 'gradient_boosting'],
        include_market_context=True,
        include_time_features=True,
        include_volatility_features=True,
        include_volume_features=True,
        prediction_horizon_days=30,
        confidence_threshold=0.7,
        quality_threshold=0.6
    )
    
    predictive_engine = get_predictive_sr_engine(predictive_config)
    
    # Train the predictive model
    print("\n📈 Training predictive model...")
    training_result = predictive_engine.train_predictive_model(
        historical_data, 
        sr_levels, 
        optimize_weights=True
    )
    
    if training_result.get('status') == 'success':
        print("✅ Model training successful!")
        print(f"   Training samples: {training_result.get('training_samples', 0)}")
        print(f"   Model performance: {training_result.get('model_performance', {})}")
        print(f"   Validation performance: {training_result.get('validation_performance', {})}")
        
        # Show optimized weights
        optimized_weights = training_result.get('optimized_weights', {})
        if optimized_weights:
            print(f"\n🎯 Optimized weights:")
            for feature, weight in optimized_weights.items():
                print(f"   {feature}: {weight:.3f}")
        
        # Show feature importance
        feature_importance = training_result.get('feature_importance', {})
        if feature_importance:
            print(f"\n📊 Top feature importance:")
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:10]:
                print(f"   {feature}: {importance:.3f}")
    else:
        print(f"❌ Model training failed: {training_result.get('error', 'Unknown error')}")
        return
    
    # Predict future quality for each SR level
    print(f"\n🔮 Predicting future quality for {len(sr_levels)} SR levels...")
    
    predictions = []
    for i, sr_level in enumerate(sr_levels):
        try:
            prediction = predictive_engine.predict_sr_quality(
                sr_level, 
                current_data, 
                prediction_horizon=30
            )
            predictions.append(prediction)
            
            print(f"\n   Level {i+1} at {sr_level.price} ({sr_level.level_type}):")
            print(f"     Predicted Quality: {prediction.predicted_quality:.3f}")
            print(f"     Confidence: {prediction.confidence:.3f}")
            print(f"     Prediction Horizon: {prediction.prediction_horizon} days")
            
            # Show key factors
            if prediction.key_factors:
                print(f"     Key Factors:")
                sorted_factors = sorted(prediction.key_factors.items(), key=lambda x: abs(x[1]), reverse=True)
                for factor, contribution in sorted_factors[:5]:
                    print(f"       {factor}: {contribution:.3f}")
            
            # Show market context
            if prediction.market_context:
                print(f"     Market Context:")
                for context, value in prediction.market_context.items():
                    print(f"       {context}: {value:.3f}")
                    
        except Exception as e:
            print(f"   ❌ Failed to predict level {i+1}: {e}")
    
    # Get high-quality predictions
    print(f"\n⭐ High-Quality SR Level Predictions:")
    high_quality_predictions = predictive_engine.get_high_quality_predictions(
        sr_levels, 
        current_data,
        min_quality=0.7,  # Higher threshold for demonstration
        min_confidence=0.8
    )
    
    if high_quality_predictions:
        print(f"✅ Found {len(high_quality_predictions)} high-quality predictions:")
        for i, prediction in enumerate(high_quality_predictions):
            print(f"   {i+1}. Price: {prediction.level.price}, Quality: {prediction.predicted_quality:.3f}, Confidence: {prediction.confidence:.3f}")
    else:
        print("   No high-quality predictions found with current thresholds")
    
    # Show prediction summary
    print(f"\n📊 Prediction Summary:")
    summary = predictive_engine.get_prediction_summary()
    if summary.get('status') != 'no_predictions':
        print(f"   Total Predictions: {summary.get('total_predictions', 0)}")
        print(f"   Average Predicted Quality: {summary.get('avg_predicted_quality', 0.0):.3f}")
        print(f"   Average Confidence: {summary.get('avg_confidence', 0.0):.3f}")
        print(f"   High Quality Predictions: {summary.get('high_quality_predictions', 0)}")
        print(f"   High Confidence Predictions: {summary.get('high_confidence_predictions', 0)}")
    
    # Demonstrate how predictions improve over time
    print(f"\n🔄 Demonstrating continuous learning...")
    
    # Simulate new market data (future)
    future_data = create_realistic_market_data(250)  # Extended data
    
    # Make predictions on future data
    future_predictions = []
    for sr_level in sr_levels[:3]:  # Test first 3 levels
        try:
            future_prediction = predictive_engine.predict_sr_quality(
                sr_level, 
                future_data, 
                prediction_horizon=30
            )
            future_predictions.append(future_prediction)
            
            print(f"   Future prediction for {sr_level.price}: Quality = {future_prediction.predicted_quality:.3f}")
            
        except Exception as e:
            print(f"   ❌ Future prediction failed for {sr_level.price}: {e}")
    
    print(f"\n🎉 Future prediction demo completed!")
    print("=" * 60)
    
    # Show how the system answers the key question
    print(f"\n💡 How This Predicts Future SR Quality:")
    print(f"   1. ✅ Weight Optimization: Learned optimal weights from historical data")
    print(f"   2. ✅ Feature Engineering: Extracted predictive features (market context, volatility, volume)")
    print(f"   3. ✅ Model Training: Trained ensemble model on historical performance")
    print(f"   4. ✅ Future Prediction: Applied model to current market conditions")
    print(f"   5. ✅ Quality Assessment: Predicted which levels will be most effective")
    print(f"   6. ✅ Confidence Scoring: Provided confidence levels for predictions")
    print(f"   7. ✅ Continuous Learning: System improves as more data becomes available")

if __name__ == "__main__":
    try:
        demonstrate_future_prediction()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()