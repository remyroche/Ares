"""
Complete Trading Pipeline Example - From SR Quality to Trading Decisions.

This example demonstrates the complete pipeline:
1. Weight optimization for SR quality assessment
2. Enhanced training data creation
3. ML model training for trading decisions
4. Trading signal generation
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
logger = system_logger.getChild('complete_trading_pipeline_example')

logger.info("Starting Complete Trading Pipeline Example")

try:
    from src.utils.sr_clustering import (
        get_backtesting_engine, BacktestConfig, SRLevel,
        get_predictive_sr_engine, PredictiveConfig,
        get_trading_ml_integration, TradingMLConfig
    )
    logger.info("✅ Successfully imported SR clustering components")
except ImportError as e:
    logger.error(f"❌ Failed to import SR clustering components: {e}")
    raise

def create_comprehensive_market_data(days: int = 300) -> pd.DataFrame:
    """Create comprehensive market data with trends, volatility, and volume patterns."""
    logger.info(f"📊 Creating comprehensive market data for {days} days")
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
    logger.debug(f"Generated date range: {dates[0]} to {dates[-1]}")
    
    # Generate realistic price data with multiple trends
    np.random.seed(42)
    base_price = 100.0
    logger.debug(f"Base price: {base_price}, random seed: 42")
    
    # Create multiple trend periods
    trend_periods = [
        (0, 100, 0.15),    # Bull market: 15% gain
        (100, 150, -0.10), # Bear market: 10% loss
        (150, 200, 0.20),  # Bull market: 20% gain
        (200, 250, -0.05), # Correction: 5% loss
        (250, 300, 0.12)   # Recovery: 12% gain
    ]
    
    logger.info(f"Defined {len(trend_periods)} trend periods")
    for i, (start, end, trend) in enumerate(trend_periods):
        logger.debug(f"Period {i+1}: days {start}-{end}, trend: {trend:.1%}")
    
    # Generate prices with trends
    logger.info("Generating price data with trend periods")
    prices = [base_price]
    trend_changes = 0
    
    for i in range(1, days):
        # Find current trend period
        current_trend = 0.0
        for start, end, trend in trend_periods:
            if start <= i < end:
                current_trend = trend / (end - start)  # Daily trend
                break
        
        # Add trend and random noise
        daily_return = current_trend + np.random.normal(0, 0.02)
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)
        
        # Log trend changes
        if i in [100, 150, 200, 250]:
            trend_changes += 1
            logger.debug(f"Trend change at day {i}: price={new_price:.2f}, trend={current_trend:.4f}")
    
    logger.info(f"Generated {len(prices)} price points with {trend_changes} trend changes")
    logger.debug(f"Price range: {min(prices):.2f} - {max(prices):.2f}")
    
    # Generate volume with patterns
    logger.info("Generating volume data with trend-based patterns")
    base_volume = 1000000
    volumes = []
    
    for i in range(days):
        # Volume increases during trends
        trend_multiplier = 1.0
        for start, end, trend in trend_periods:
            if start <= i < end:
                trend_multiplier = 1.0 + abs(trend) * 0.5
                break
        
        # Add random variation
        volume = base_volume * trend_multiplier * np.random.lognormal(0, 0.3)
        volumes.append(volume)
    
    logger.info(f"Generated {len(volumes)} volume points")
    logger.debug(f"Volume range: {min(volumes):.0f} - {max(volumes):.0f}")
    
    # Generate OHLC data
    logger.info("Generating OHLC data with realistic patterns")
    data = []
    
    for i, (date, close, volume) in enumerate(zip(dates, prices, volumes)):
        # Generate realistic OHLC
        daily_volatility = 0.01 + 0.005 * np.sin(i / 10)  # Cyclical volatility
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
    logger.info(f"✅ Created comprehensive market data: {len(df)} rows")
    logger.info(f"Data columns: {list(df.columns)}")
    logger.debug(f"Data shape: {df.shape}")
    
    return df

def create_historical_performance_data(market_data: pd.DataFrame, sr_levels: list) -> pd.DataFrame:
    """Create historical performance data for training."""
    logger.info(f"📈 Creating historical performance data for {len(sr_levels)} SR levels")
    logger.info(f"Market data shape: {market_data.shape}")
    
    performance_data = []
    
    processed_days = 0
    skipped_days = 0
    
    for i, row in market_data.iterrows():
        if i < 30:  # Skip first 30 days for stability
            skipped_days += 1
            continue
        
        # Find nearby SR levels
        current_price = row['close']
        nearby_levels = []
        
        for sr_level in sr_levels:
            if abs(sr_level.price - current_price) / current_price < 0.05:  # Within 5%
                nearby_levels.append(sr_level)
        
        if not nearby_levels:
            continue
        
        processed_days += 1
        
        # Calculate future return (30 days ahead)
        future_idx = min(i + 30, len(market_data) - 1)
        future_price = market_data.iloc[future_idx]['close']
        future_return = (future_price - current_price) / current_price
        
        # Determine trade success
        trade_success = 1 if future_return > 0.02 else 0  # 2% threshold
        
        # Calculate risk level
        volatility = market_data.iloc[max(0, i-20):i]['close'].pct_change().std()
        risk_level = volatility * np.sqrt(252)  # Annualized
        
        # Create performance record
        for sr_level in nearby_levels:
            performance_data.append({
                'symbol': 'TEST',
                'price': sr_level.price,
                'level_type': sr_level.level_type,
                'timestamp': row['date'],
                'future_return': future_return,
                'trade_success': trade_success,
                'risk_level': risk_level,
                'time_of_day': row['date'].hour,
                'day_of_week': row['date'].weekday(),
                'month': row['date'].month
            })
        
        # Log progress every 50 days
        if processed_days % 50 == 0:
            logger.debug(f"Processed {processed_days} days, created {len(performance_data)} performance records")
    
    df = pd.DataFrame(performance_data)
    logger.info(f"✅ Created historical performance data: {len(df)} records")
    logger.info(f"Processed {processed_days} days, skipped {skipped_days} days")
    logger.info(f"Performance data columns: {list(df.columns)}")
    
    if len(df) > 0:
        success_rate = df['trade_success'].mean()
        avg_return = df['future_return'].mean()
        logger.info(f"Performance stats: success_rate={success_rate:.3f}, avg_return={avg_return:.3f}")
    
    return df

def create_diverse_sr_levels() -> list:
    """Create diverse SR levels for testing."""
    logger.info("🎯 Creating diverse SR levels for testing")
    levels = []
    
    # Strong support levels
    strong_support_prices = [95.0, 98.0, 102.0, 105.0]
    logger.info(f"Creating {len(strong_support_prices)} strong support levels")
    
    for i, price in enumerate(strong_support_prices):
        level = SRLevel(
            price=price,
            level_type='support',
            strength=0.8,
            first_touch=datetime.now() - timedelta(days=200),
            last_touch=datetime.now() - timedelta(days=10),
            touch_count=8,
            timeframe='1D',
            symbol='TEST',
            source='example'
        )
        levels.append(level)
        logger.debug(f"Created strong support level {i+1}: price={price}, strength=0.8")
    
    # Weak resistance levels
    weak_resistance_prices = [108.0, 112.0, 118.0, 122.0]
    logger.info(f"Creating {len(weak_resistance_prices)} weak resistance levels")
    
    for i, price in enumerate(weak_resistance_prices):
        level = SRLevel(
            price=price,
            level_type='resistance',
            strength=0.3,
            first_touch=datetime.now() - timedelta(days=50),
            last_touch=datetime.now() - timedelta(days=5),
            touch_count=2,
            timeframe='1D',
            symbol='TEST',
            source='example'
        )
        levels.append(level)
        logger.debug(f"Created weak resistance level {i+1}: price={price}, strength=0.3")
    
    # Medium quality levels
    medium_prices = [110.0, 115.0, 120.0]
    logger.info(f"Creating {len(medium_prices)} medium quality levels")
    
    for i, price in enumerate(medium_prices):
        level_type = 'support' if price < 115 else 'resistance'
        level = SRLevel(
            price=price,
            level_type=level_type,
            strength=0.6,
            first_touch=datetime.now() - timedelta(days=100),
            last_touch=datetime.now() - timedelta(days=15),
            touch_count=4,
            timeframe='1D',
            symbol='TEST',
            source='example'
        )
        levels.append(level)
        logger.debug(f"Created medium quality level {i+1}: price={price}, type={level_type}, strength=0.6")
    
    logger.info(f"✅ Created {len(levels)} diverse SR levels total")
    
    # Log level distribution
    support_count = len([l for l in levels if l.level_type == 'support'])
    resistance_count = len([l for l in levels if l.level_type == 'resistance'])
    logger.info(f"Level distribution: {support_count} support, {resistance_count} resistance")
    
    return levels

def demonstrate_complete_trading_pipeline():
    """Demonstrate the complete pipeline from SR quality to trading decisions."""
    print("🚀 Complete Trading Pipeline Demo")
    print("=" * 70)
    
    # Step 1: Create comprehensive market data
    print("📊 Step 1: Creating comprehensive market data...")
    market_data = create_comprehensive_market_data(300)
    print(f"✅ Created {len(market_data)} days of market data")
    
    # Step 2: Create diverse SR levels
    print("\n🎯 Step 2: Creating diverse SR levels...")
    sr_levels = create_diverse_sr_levels()
    print(f"✅ Created {len(sr_levels)} SR levels")
    
    # Step 3: Create historical performance data
    print("\n📈 Step 3: Creating historical performance data...")
    historical_performance = create_historical_performance_data(market_data, sr_levels)
    print(f"✅ Created {len(historical_performance)} historical performance records")
    
    # Step 4: Initialize trading ML integration
    print("\n🔧 Step 4: Initializing trading ML integration...")
    trading_config = TradingMLConfig(
        classification_model='random_forest',
        regression_model='ridge',
        include_sr_quality=True,
        include_momentum=True,
        include_volatility=True,
        include_volume=True,
        include_market_regime=True,
        quality_threshold=0.6,
        confidence_threshold=0.7,
        prediction_horizon=30
    )
    
    trading_ml = get_trading_ml_integration(trading_config)
    
    # Step 5: Prepare enhanced training data
    print("\n🧠 Step 5: Preparing enhanced training data with SR quality predictions...")
    enhanced_data = trading_ml.prepare_enhanced_training_data(
        market_data, sr_levels, historical_performance
    )
    
    if len(enhanced_data) > 0:
        print(f"✅ Created enhanced training dataset with {len(enhanced_data)} samples")
        print(f"   Features: {list(enhanced_data.columns)}")
        
        # Show sample of enhanced data
        print(f"\n📊 Sample of enhanced training data:")
        sample_features = ['sr_quality', 'sr_confidence', 'momentum_score', 'volatility_score', 'volume_score', 'trade_success', 'future_return']
        print(enhanced_data[sample_features].head())
    else:
        print("❌ Failed to create enhanced training data")
        return
    
    # Step 6: Train trading models
    print(f"\n🎯 Step 6: Training ML models for trading decisions...")
    training_result = trading_ml.train_trading_models(enhanced_data)
    
    if training_result.get('status') == 'success':
        print("✅ Trading models trained successfully!")
        
        # Show model performance
        classification_perf = training_result.get('classification_performance', {})
        regression_perf = training_result.get('regression_performance', {})
        
        print(f"\n📊 Model Performance:")
        print(f"   Classification Accuracy: {classification_perf.get('accuracy', 0.0):.3f}")
        print(f"   Classification F1 Score: {classification_perf.get('f1_score', 0.0):.3f}")
        print(f"   Classification ROC AUC: {classification_perf.get('roc_auc', 0.0):.3f}")
        print(f"   Regression R² Score: {regression_perf.get('r2_score', 0.0):.3f}")
        print(f"   Regression MSE: {regression_perf.get('mse', 0.0):.4f}")
        
        # Show feature importance
        feature_importance = training_result.get('feature_importance', {})
        if feature_importance:
            print(f"\n🎯 Top Feature Importance:")
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:10]:
                print(f"   {feature}: {importance:.3f}")
    else:
        print(f"❌ Model training failed: {training_result.get('error', 'Unknown error')}")
        return
    
    # Step 7: Generate trading signals
    print(f"\n📡 Step 7: Generating trading signals...")
    
    # Use recent market data for signal generation
    recent_market_data = market_data.tail(50)
    current_sr_levels = sr_levels
    
    trading_signals = trading_ml.generate_trading_signals(recent_market_data, current_sr_levels)
    
    if trading_signals:
        print(f"✅ Generated {len(trading_signals)} trading signals")
        
        # Show trading signals
        print(f"\n📊 Trading Signals:")
        for i, signal in enumerate(trading_signals[:5]):  # Show top 5
            print(f"   {i+1}. {signal.symbol} at {signal.sr_quality:.3f} quality:")
            print(f"      Signal: {signal.signal_type.upper()}")
            print(f"      Confidence: {signal.confidence:.3f}")
            print(f"      Expected Return: {signal.expected_return:.3f}")
            print(f"      Market Regime: {signal.market_regime}")
            print(f"      Risk Score: {signal.risk_score:.3f}")
            print()
    else:
        print("❌ No trading signals generated")
    
    # Step 8: Show trading summary
    print(f"\n📈 Step 8: Trading Summary...")
    trading_summary = trading_ml.get_trading_summary()
    
    print(f"✅ Trading Summary:")
    print(f"   Total Signals: {trading_summary.get('total_signals', 0)}")
    print(f"   Buy Signals: {trading_summary.get('buy_signals', 0)}")
    print(f"   Sell Signals: {trading_summary.get('sell_signals', 0)}")
    print(f"   Hold Signals: {trading_summary.get('hold_signals', 0)}")
    print(f"   Average Confidence: {trading_summary.get('avg_confidence', 0.0):.3f}")
    print(f"   Average Expected Return: {trading_summary.get('avg_expected_return', 0.0):.3f}")
    
    logger.info("🎉 Complete trading pipeline demo completed successfully")
    print(f"\n🎉 Complete trading pipeline demo completed!")
    print("=" * 70)
    
    # Show how this answers the key question
    logger.info("💡 Demonstrating how the pipeline answers the key question")
    print(f"\n💡 How This Answers 'What Makes a Strong SR Level for Trading?':")
    print(f"   1. ✅ Weight Optimization: Learned optimal weights from historical data")
    print(f"   2. ✅ SR Quality Prediction: Predicted quality of each SR level")
    print(f"   3. ✅ Enhanced Training Data: Combined SR quality with market features")
    print(f"   4. ✅ ML Model Training: Trained models on enhanced data")
    print(f"   5. ✅ Trading Signal Generation: Generated actionable trading signals")
    print(f"   6. ✅ Risk Assessment: Provided confidence and risk scores")
    print(f"   7. ✅ Market Context: Considered momentum, volatility, volume, regime")
    
    logger.info("🎯 Providing key insights from the demonstration")
    print(f"\n🎯 Key Insights:")
    print(f"   • SR quality is now a quantifiable feature in trading models")
    print(f"   • Models learn which SR levels work best in different market conditions")
    print(f"   • Trading signals combine SR quality with market context")
    print(f"   • System provides confidence scores for risk management")
    print(f"   • Continuous learning improves predictions over time")

if __name__ == "__main__":
    try:
        logger.info("Starting complete trading pipeline demonstration")
        demonstrate_complete_trading_pipeline()
        logger.info("✅ Complete trading pipeline demonstration completed successfully")
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"❌ Demo failed: {e}")
        traceback.print_exc()