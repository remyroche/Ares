from src.utils.tprint import tprint

"""
Example demonstrating how the weight optimization system predicts future SR level quality.

This script shows the complete pipeline from weight optimization to future quality prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import logging

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import logging system
from src.utils.logger import system_logger

# Initialize logger for this example
logger = system_logger.getChild('predictive_example')

logger.info("Starting Predictive SR Example")

try:
    from src.utils.sr_clustering import (
        get_backtesting_engine, BacktestConfig, SRLevel,
        get_backtesting_enhanced_clustering, BacktestingEnhancedConfig,
        get_weight_optimization_engine, WeightOptimizationConfig,
        get_predictive_sr_engine, PredictiveConfig
    )
    logger.info("✅ Successfully imported SR clustering components")
except ImportError as e:
    logger.error(f"❌ Failed to import SR clustering components: {e}")
    raise

def create_realistic_market_data(days: int = 200) -> pd.DataFrame:
    """Create realistic market data with trends, volatility, and volume patterns."""
    logger.info(f"📊 Creating realistic market data for {days} days")
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
    logger.debug(f"Generated date range: {dates[0]} to {dates[-1]}")
    
    # Generate realistic price data with trends and volatility
    np.random.seed(42)
    base_price = 100.0
    logger.debug(f"Base price: {base_price}, random seed: 42")
    
    # Create trend component
    trend = np.linspace(0, 0.2, days)  # 20% upward trend over period
    logger.debug(f"Created trend component: {trend[0]:.4f} to {trend[-1]:.4f}")
    
    # Create volatility component
    volatility = 0.02 + 0.01 * np.sin(np.linspace(0, 4*np.pi, days))  # Cyclical volatility
    logger.debug(f"Created volatility component: min={volatility.min():.4f}, max={volatility.max():.4f}")
    
    # Generate returns
    logger.info("Generating returns with trend and volatility components")
    returns = []
    for i in range(days):
        trend_component = trend[i] / days
        volatility_component = volatility[i] * np.random.normal(0, 1)
        returns.append(trend_component + volatility_component)
    
    logger.debug(f"Generated {len(returns)} returns: mean={np.mean(returns):.4f}, std={np.std(returns):.4f}")
    
    # Generate prices
    logger.info("Generating price series from returns")
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    logger.info(f"Generated {len(prices)} price points: {prices[0]:.2f} to {prices[-1]:.2f}")
    
    # Generate volume with patterns
    logger.info("Generating volume data with weekly patterns")
    base_volume = 1000000
    volume_pattern = 1 + 0.3 * np.sin(np.linspace(0, 2*np.pi, days))  # Weekly pattern
    volumes = [base_volume * pattern * np.random.lognormal(0, 0.3) for pattern in volume_pattern]
    
    logger.debug(f"Generated {len(volumes)} volume points: min={min(volumes):.0f}, max={max(volumes):.0f}")
    
    # Generate OHLC data
    logger.info("Generating OHLC data with realistic patterns")
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
        
        # Log progress every 50 days
        if i % 50 == 0:
            logger.debug(f"Generated OHLC data for day {i+1}/{days}")
    
    df = pd.DataFrame(data)
    logger.info(f"✅ Created realistic market data: {len(df)} rows")
    logger.info(f"Data columns: {list(df.columns)}")
    logger.debug(f"Data shape: {df.shape}")
    
    return df

def create_diverse_sr_levels() -> list:
    """Create diverse SR levels with different characteristics."""
    logger.info("🎯 Creating diverse SR levels with different characteristics")
    levels = []
    
    # Strong support levels (high quality expected)
    strong_support_prices = [95.0, 98.0, 102.0]
    logger.info(f"Creating {len(strong_support_prices)} strong support levels")
    
    for i, price in enumerate(strong_support_prices):
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
        logger.debug(f"Created strong support level {i+1}: price={price}, strength=0.8, touches=8")
    
    # Weak resistance levels (low quality expected)
    weak_resistance_prices = [108.0, 112.0, 118.0]
    logger.info(f"Creating {len(weak_resistance_prices)} weak resistance levels")
    
    for i, price in enumerate(weak_resistance_prices):
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
        logger.debug(f"Created weak resistance level {i+1}: price={price}, strength=0.3, touches=2")
    
    # Medium quality levels
    medium_prices = [105.0, 110.0, 115.0]
    logger.info(f"Creating {len(medium_prices)} medium quality levels")
    
    for i, price in enumerate(medium_prices):
        level_type = 'support' if price < 110 else 'resistance'
        level = SRLevel(
            price=price,
            level_type=level_type,
            strength=0.6,  # Medium strength
            first_touch=datetime.now() - timedelta(days=100),
            last_touch=datetime.now() - timedelta(days=15),
            touch_count=4,  # Moderate touches
            timeframe='1D',
            symbol='TEST',
            source='example'
        )
        levels.append(level)
        logger.debug(f"Created medium quality level {i+1}: price={price}, type={level_type}, strength=0.6, touches=4")
    
    logger.info(f"✅ Created {len(levels)} diverse SR levels total")
    
    # Log level distribution
    support_count = len([l for l in levels if l.level_type == 'support'])
    resistance_count = len([l for l in levels if l.level_type == 'resistance'])
    logger.info(f"Level distribution: {support_count} support, {resistance_count} resistance")
    
    return levels

def demonstrate_future_prediction():
    """Demonstrate how the system predicts future SR level quality."""
    logger.info("🔮 Starting Future SR Quality Prediction Demo")
    tprint("🔮 Future SR Quality Prediction Demo")
    tprint("=" * 60)
    
    # Create historical and current market data
    logger.info("📊 Creating market data for prediction demonstration")
    tprint("📊 Creating market data...")
    
    historical_data = create_realistic_market_data(150)  # 150 days of historical data
    current_data = create_realistic_market_data(200)     # 200 days including future
    
    logger.info(f"✅ Created historical data: {len(historical_data)} days")
    logger.info(f"✅ Created current data: {len(current_data)} days")
    tprint(f"✅ Historical data: {len(historical_data)} days")
    tprint(f"✅ Current data: {len(current_data)} days")
    
    # Create diverse SR levels
    logger.info("🎯 Creating diverse SR levels for prediction")
    tprint("\n🎯 Creating diverse SR levels...")
    
    sr_levels = create_diverse_sr_levels()
    logger.info(f"✅ Created {len(sr_levels)} SR levels for prediction")
    tprint(f"✅ Created {len(sr_levels)} SR levels")
    
    # Initialize predictive engine
    logger.info("🔧 Initializing predictive engine")
    tprint("\n🔧 Initializing predictive engine...")
    
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
    
    logger.info(f"Predictive config: model_type={predictive_config.model_type}, ensemble_models={predictive_config.ensemble_models}")
    logger.info(f"Feature flags: market_context={predictive_config.include_market_context}, time={predictive_config.include_time_features}")
    logger.info(f"Prediction settings: horizon={predictive_config.prediction_horizon_days} days, confidence_threshold={predictive_config.confidence_threshold}")
    
    predictive_engine = get_predictive_sr_engine(predictive_config)
    logger.info("✅ Predictive engine initialized successfully")
    
    # Train the predictive model
    logger.info("📈 Training predictive model with historical data")
    tprint("\n📈 Training predictive model...")
    
    training_result = predictive_engine.train_predictive_model(
        historical_data, 
        sr_levels, 
        optimize_weights=True
    )
    
    if training_result.get('status') == 'success':
        logger.info("✅ Model training successful")
        tprint("✅ Model training successful!")
        
        training_samples = training_result.get('training_samples', 0)
        model_perf = training_result.get('model_performance', {})
        validation_perf = training_result.get('validation_performance', {})
        
        logger.info(f"Training samples: {training_samples}")
        logger.info(f"Model performance: {model_perf}")
        logger.info(f"Validation performance: {validation_perf}")
        
        tprint(f"   Training samples: {training_samples}")
        tprint(f"   Model performance: {model_perf}")
        tprint(f"   Validation performance: {validation_perf}")
        
        # Show optimized weights
        optimized_weights = training_result.get('optimized_weights', {})
        if optimized_weights:
            logger.info(f"Optimized weights available for {len(optimized_weights)} features")
            tprint(f"\n🎯 Optimized weights:")
            for feature, weight in optimized_weights.items():
                tprint(f"   {feature}: {weight:.3f}")
                logger.debug(f"Optimized weight: {feature}={weight:.3f}")
        else:
            logger.warning("No optimized weights available")
        
        # Show feature importance
        feature_importance = training_result.get('feature_importance', {})
        if feature_importance:
            logger.info(f"Feature importance available for {len(feature_importance)} features")
            tprint(f"\n📊 Top feature importance:")
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:10]:
                tprint(f"   {feature}: {importance:.3f}")
                logger.debug(f"Feature importance: {feature}={importance:.3f}")
        else:
            logger.warning("No feature importance data available")
    else:
        error_msg = training_result.get('error', 'Unknown error')
        logger.error(f"❌ Model training failed: {error_msg}")
        tprint(f"❌ Model training failed: {error_msg}")
        return
    
    # Predict future quality for each SR level
    logger.info(f"🔮 Predicting future quality for {len(sr_levels)} SR levels")
    tprint(f"\n🔮 Predicting future quality for {len(sr_levels)} SR levels...")
    
    predictions = []
    prediction_errors = 0
    
    for i, sr_level in enumerate(sr_levels):
        try:
            logger.debug(f"Predicting quality for level {i+1}: price={sr_level.price}, type={sr_level.level_type}")
            
            prediction = predictive_engine.predict_sr_quality(
                sr_level, 
                current_data, 
                prediction_horizon=30
            )
            predictions.append(prediction)
            
            logger.info(f"Level {i+1} prediction: quality={prediction.predicted_quality:.3f}, confidence={prediction.confidence:.3f}")
            
            tprint(f"\n   Level {i+1} at {sr_level.price} ({sr_level.level_type}):")
            tprint(f"     Predicted Quality: {prediction.predicted_quality:.3f}")
            tprint(f"     Confidence: {prediction.confidence:.3f}")
            tprint(f"     Prediction Horizon: {prediction.prediction_horizon} days")
            
            # Show key factors
            if prediction.key_factors:
                logger.debug(f"Key factors for level {i+1}: {len(prediction.key_factors)} factors")
                tprint(f"     Key Factors:")
                sorted_factors = sorted(prediction.key_factors.items(), key=lambda x: abs(x[1]), reverse=True)
                for factor, contribution in sorted_factors[:5]:
                    tprint(f"       {factor}: {contribution:.3f}")
                    logger.debug(f"Key factor: {factor}={contribution:.3f}")
            else:
                logger.warning(f"No key factors available for level {i+1}")
            
            # Show market context
            if prediction.market_context:
                logger.debug(f"Market context for level {i+1}: {len(prediction.market_context)} context features")
                tprint(f"     Market Context:")
                for context, value in prediction.market_context.items():
                    tprint(f"       {context}: {value:.3f}")
                    logger.debug(f"Market context: {context}={value:.3f}")
            else:
                logger.warning(f"No market context available for level {i+1}")
                    
        except Exception as e:
            prediction_errors += 1
            logger.error(f"❌ Failed to predict level {i+1}: {e}")
            tprint(f"   ❌ Failed to predict level {i+1}: {e}")
    
    logger.info(f"✅ Prediction completed: {len(predictions)} successful, {prediction_errors} errors")
    
    # Get high-quality predictions
    logger.info("⭐ Identifying high-quality SR level predictions")
    tprint(f"\n⭐ High-Quality SR Level Predictions:")
    
    high_quality_predictions = predictive_engine.get_high_quality_predictions(
        sr_levels, 
        current_data,
        min_quality=0.7,  # Higher threshold for demonstration
        min_confidence=0.8
    )
    
    if high_quality_predictions:
        logger.info(f"✅ Found {len(high_quality_predictions)} high-quality predictions")
        tprint(f"✅ Found {len(high_quality_predictions)} high-quality predictions:")
        for i, prediction in enumerate(high_quality_predictions):
            tprint(f"   {i+1}. Price: {prediction.level.price}, Quality: {prediction.predicted_quality:.3f}, Confidence: {prediction.confidence:.3f}")
            logger.debug(f"High-quality prediction {i+1}: price={prediction.level.price}, quality={prediction.predicted_quality:.3f}, confidence={prediction.confidence:.3f}")
    else:
        logger.warning("No high-quality predictions found with current thresholds")
        tprint("   No high-quality predictions found with current thresholds")
    
    # Show prediction summary
    logger.info("📊 Generating prediction summary")
    tprint(f"\n📊 Prediction Summary:")
    
    summary = predictive_engine.get_prediction_summary()
    logger.info(f"Prediction summary: {summary}")
    
    if summary.get('status') != 'no_predictions':
        total_predictions = summary.get('total_predictions', 0)
        avg_quality = summary.get('avg_predicted_quality', 0.0)
        avg_confidence = summary.get('avg_confidence', 0.0)
        high_quality_count = summary.get('high_quality_predictions', 0)
        high_confidence_count = summary.get('high_confidence_predictions', 0)
        
        logger.info(f"Summary stats: total={total_predictions}, avg_quality={avg_quality:.3f}, avg_confidence={avg_confidence:.3f}")
        
        tprint(f"   Total Predictions: {total_predictions}")
        tprint(f"   Average Predicted Quality: {avg_quality:.3f}")
        tprint(f"   Average Confidence: {avg_confidence:.3f}")
        tprint(f"   High Quality Predictions: {high_quality_count}")
        tprint(f"   High Confidence Predictions: {high_confidence_count}")
    else:
        logger.warning("No predictions available for summary")
    
    # Demonstrate how predictions improve over time
    logger.info("🔄 Demonstrating continuous learning with extended data")
    tprint(f"\n🔄 Demonstrating continuous learning...")
    
    # Simulate new market data (future)
    future_data = create_realistic_market_data(250)  # Extended data
    logger.info(f"Created extended future data: {len(future_data)} days")
    
    # Make predictions on future data
    future_predictions = []
    future_errors = 0
    
    for i, sr_level in enumerate(sr_levels[:3]):  # Test first 3 levels
        try:
            logger.debug(f"Making future prediction for level {i+1}: price={sr_level.price}")
            
            future_prediction = predictive_engine.predict_sr_quality(
                sr_level, 
                future_data, 
                prediction_horizon=30
            )
            future_predictions.append(future_prediction)
            
            logger.info(f"Future prediction for {sr_level.price}: Quality = {future_prediction.predicted_quality:.3f}")
            tprint(f"   Future prediction for {sr_level.price}: Quality = {future_prediction.predicted_quality:.3f}")
            
        except Exception as e:
            future_errors += 1
            logger.error(f"❌ Future prediction failed for {sr_level.price}: {e}")
            tprint(f"   ❌ Future prediction failed for {sr_level.price}: {e}")
    
    logger.info(f"Future predictions completed: {len(future_predictions)} successful, {future_errors} errors")
    
    logger.info("🎉 Future prediction demo completed successfully")
    tprint(f"\n🎉 Future prediction demo completed!")
    tprint("=" * 60)
    
    # Show how the system answers the key question
    logger.info("💡 Demonstrating how the system predicts future SR quality")
    tprint(f"\n💡 How This Predicts Future SR Quality:")
    tprint(f"   1. ✅ Weight Optimization: Learned optimal weights from historical data")
    tprint(f"   2. ✅ Feature Engineering: Extracted predictive features (market context, volatility, volume)")
    tprint(f"   3. ✅ Model Training: Trained ensemble model on historical performance")
    tprint(f"   4. ✅ Future Prediction: Applied model to current market conditions")
    tprint(f"   5. ✅ Quality Assessment: Predicted which levels will be most effective")
    tprint(f"   6. ✅ Confidence Scoring: Provided confidence levels for predictions")
    tprint(f"   7. ✅ Continuous Learning: System improves as more data becomes available")

if __name__ == "__main__":
    try:
        logger.info("Starting future prediction demonstration")
        demonstrate_future_prediction()
        logger.info("✅ Future prediction demonstration completed successfully")
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        tprint(f"❌ Demo failed: {e}")
        traceback.print_exc()