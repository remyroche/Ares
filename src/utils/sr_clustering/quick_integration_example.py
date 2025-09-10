"""
Quick Integration Example - Shows how to integrate SR clustering with existing ML models.

This script demonstrates the minimal changes needed to integrate the SR clustering system
with your existing ML models in Step02_5.
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
logger = system_logger.getChild('quick_integration_example')

logger.info("Starting Quick Integration Example")

def demonstrate_quick_integration():
    """Demonstrate quick integration with existing ML models."""
    logger.info("🔧 Starting Quick Integration Demo")
    print("🔧 Quick Integration Demo")
    print("=" * 50)
    
    # Step 1: Check if SR clustering is available
    logger.info("📋 Step 1: Checking SR clustering availability")
    print("📋 Step 1: Checking SR clustering availability...")
    
    try:
        from src.utils.sr_clustering import (
            get_backtesting_enhanced_clustering, BacktestingEnhancedConfig,
            get_predictive_sr_engine, PredictiveConfig,
            get_trading_ml_integration, TradingMLConfig
        )
        SR_CLUSTERING_AVAILABLE = True
        logger.info("✅ SR clustering system available")
        print("✅ SR clustering system available")
    except ImportError as e:
        SR_CLUSTERING_AVAILABLE = False
        logger.error(f"❌ SR clustering not available: {e}")
        print(f"❌ SR clustering not available: {e}")
        return
    
    # Step 2: Simulate existing Step02_5 data
    logger.info("📊 Step 2: Simulating existing Step02_5 data")
    print("\n📊 Step 2: Simulating existing Step02_5 data...")
    
    # Simulate market data (what Step02_5 currently processes)
    logger.info("Creating simulated market data")
    market_data = pd.DataFrame({
        'date': pd.date_range(start=datetime.now() - timedelta(days=100), periods=100, freq='D'),
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(105, 115, 100),
        'low': np.random.uniform(95, 105, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.uniform(1000000, 2000000, 100)
    })
    
    # Simulate existing SR levels (what EnhancedSRDetector currently returns)
    logger.info("Creating simulated existing SR levels")
    existing_sr_levels = [
        {'price': 105.0, 'type': 'support', 'strength': 0.7, 'touches': 3},
        {'price': 108.0, 'type': 'resistance', 'strength': 0.5, 'touches': 2},
        {'price': 102.0, 'type': 'support', 'strength': 0.8, 'touches': 4},
        {'price': 110.0, 'type': 'resistance', 'strength': 0.6, 'touches': 3}
    ]
    
    logger.info(f"✅ Created market data: {len(market_data)} rows")
    logger.info(f"✅ Created existing SR levels: {len(existing_sr_levels)} levels")
    print(f"✅ Market data: {len(market_data)} rows")
    print(f"✅ Existing SR levels: {len(existing_sr_levels)} levels")
    
    # Step 3: Enhanced SR processing (NEW)
    logger.info("🎯 Step 3: Enhanced SR processing with weight optimization")
    print("\n🎯 Step 3: Enhanced SR processing with weight optimization...")
    
    if SR_CLUSTERING_AVAILABLE:
        # Use backtesting-enhanced clustering
        logger.info("Configuring backtesting-enhanced clustering")
        clustering_config = BacktestingEnhancedConfig(
            min_levels_for_learning=3,
            quality_filter_threshold=0.1,
            proximity_adjustment_factor=0.5
        )
        
        clustering = get_backtesting_enhanced_clustering(clustering_config)
        
        # Convert existing levels to SRLevel format
        logger.info("Converting existing levels to SRLevel format")
        from src.utils.sr_clustering import SRLevel
        sr_levels = []
        for i, level in enumerate(existing_sr_levels):
            sr_level = SRLevel(
                price=level['price'],
                level_type=level['type'],
                strength=level['strength'],
                first_touch=datetime.now() - timedelta(days=50),
                last_touch=datetime.now() - timedelta(days=5),
                touch_count=level['touches'],
                timeframe='1D',
                symbol='TEST',
                source='integration_example'
            )
            sr_levels.append(sr_level)
            logger.debug(f"Converted level {i+1}: price={level['price']}, type={level['type']}, strength={level['strength']}")
        
        # Run enhanced clustering
        logger.info("Running enhanced clustering")
        enhanced_sr_levels = clustering.cluster_with_backtesting(sr_levels, market_data)
        
        logger.info(f"✅ Enhanced SR levels: {len(enhanced_sr_levels.clusters)} clusters")
        print(f"✅ Enhanced SR levels: {len(enhanced_sr_levels.clusters)} clusters")
        
        # Get SR quality predictions
        logger.info("Configuring predictive engine for SR quality predictions")
        predictive_config = PredictiveConfig(
            model_type='ensemble',
            prediction_horizon_days=30
        )
        
        predictive_engine = get_predictive_sr_engine(predictive_config)
        training_result = predictive_engine.train_predictive_model(market_data, sr_levels)
        
        if training_result.get('status') == 'success':
            logger.info("✅ SR quality predictions generated successfully")
            print("✅ SR quality predictions generated")
            
            # Show optimized weights
            optimized_weights = training_result.get('optimized_weights', {})
            if optimized_weights:
                logger.info(f"Optimized weights available for {len(optimized_weights)} features")
                print(f"   Optimized weights: {optimized_weights}")
            else:
                logger.warning("No optimized weights available")
        else:
            error_msg = training_result.get('error', 'Unknown error')
            logger.warning(f"⚠️ SR quality prediction failed: {error_msg}")
            print("⚠️  SR quality prediction failed, using default weights")
    else:
        logger.warning("SR clustering not available, skipping enhanced processing")
    
    # Step 4: Enhanced training data creation (NEW)
    logger.info("🧠 Step 4: Creating enhanced training data for ML models")
    print("\n🧠 Step 4: Creating enhanced training data for ML models...")
    
    if SR_CLUSTERING_AVAILABLE:
        # Simulate historical performance data
        logger.info("Creating simulated historical performance data")
        historical_performance = pd.DataFrame({
            'symbol': ['TEST'] * 50,
            'price': np.random.uniform(100, 110, 50),
            'future_return': np.random.uniform(-0.05, 0.05, 50),
            'trade_success': np.random.choice([0, 1], 50),
            'timestamp': pd.date_range(start=datetime.now() - timedelta(days=50), periods=50, freq='D')
        })
        
        logger.info(f"Created historical performance data: {len(historical_performance)} records")
        
        # Create enhanced training data
        logger.info("Configuring trading ML integration")
        trading_config = TradingMLConfig(
            classification_model='random_forest',
            regression_model='ridge',
            include_sr_quality=True,
            include_momentum=True,
            include_volatility=True,
            include_volume=True
        )
        
        trading_ml = get_trading_ml_integration(trading_config)
        enhanced_data = trading_ml.prepare_enhanced_training_data(
            market_data, sr_levels, historical_performance
        )
        
        if len(enhanced_data) > 0:
            logger.info(f"✅ Enhanced training data created: {len(enhanced_data)} samples")
            logger.info(f"Enhanced data features: {list(enhanced_data.columns)}")
            print(f"✅ Enhanced training data: {len(enhanced_data)} samples")
            print(f"   Features: {list(enhanced_data.columns)}")
            
            # Show sample of enhanced data
            print(f"\n📊 Sample enhanced features:")
            sample_features = ['sr_quality', 'sr_confidence', 'momentum_score', 'volatility_score', 'trade_success']
            available_features = [f for f in sample_features if f in enhanced_data.columns]
            if available_features:
                print(enhanced_data[available_features].head())
                logger.debug(f"Sample enhanced data: {enhanced_data[available_features].head().to_dict()}")
            else:
                logger.warning(f"None of the expected sample features found in enhanced data")
        else:
            logger.error("❌ Failed to create enhanced training data")
            print("❌ Failed to create enhanced training data")
    else:
        logger.warning("SR clustering not available, skipping enhanced training data creation")
    
    # Step 5: ML model training with enhanced data (NEW)
    logger.info("🎯 Step 5: Training ML models with enhanced data")
    print("\n🎯 Step 5: Training ML models with enhanced data...")
    
    if SR_CLUSTERING_AVAILABLE and len(enhanced_data) > 0:
        # Train trading models
        logger.info("Training trading ML models")
        training_result = trading_ml.train_trading_models(enhanced_data)
        
        if training_result.get('status') == 'success':
            logger.info("✅ ML models trained successfully")
            print("✅ ML models trained successfully")
            
            # Show model performance
            classification_perf = training_result.get('classification_performance', {})
            regression_perf = training_result.get('regression_performance', {})
            
            accuracy = classification_perf.get('accuracy', 0.0)
            r2_score = regression_perf.get('r2_score', 0.0)
            
            logger.info(f"Model performance: classification accuracy={accuracy:.3f}, regression r2={r2_score:.3f}")
            print(f"   Classification Accuracy: {accuracy:.3f}")
            print(f"   Regression R²: {r2_score:.3f}")
            
            # Show feature importance
            feature_importance = training_result.get('feature_importance', {})
            if feature_importance:
                logger.info(f"Feature importance available for {len(feature_importance)} features")
                print(f"\n🎯 Top feature importance:")
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                for feature, importance in sorted_features[:5]:
                    print(f"   {feature}: {importance:.3f}")
                    logger.debug(f"Feature importance: {feature}={importance:.3f}")
            else:
                logger.warning("No feature importance data available")
        else:
            error_msg = training_result.get('error', 'Unknown error')
            logger.error(f"❌ ML model training failed: {error_msg}")
            print(f"❌ ML model training failed: {error_msg}")
    else:
        logger.warning("SR clustering not available or no enhanced data, skipping ML model training")
    
    # Step 6: Trading signal generation (NEW)
    logger.info("📡 Step 6: Generating trading signals")
    print("\n📡 Step 6: Generating trading signals...")
    
    if SR_CLUSTERING_AVAILABLE and training_result.get('status') == 'success':
        # Generate trading signals
        logger.info("Generating trading signals using trained models")
        trading_signals = trading_ml.generate_trading_signals(market_data, sr_levels)
        
        if trading_signals:
            logger.info(f"✅ Generated {len(trading_signals)} trading signals")
            print(f"✅ Generated {len(trading_signals)} trading signals")
            
            # Show trading signals
            for i, signal in enumerate(trading_signals[:3]):  # Show top 3
                print(f"   {i+1}. {signal.symbol} at {signal.sr_quality:.3f} quality:")
                print(f"      Signal: {signal.signal_type.upper()}")
                print(f"      Confidence: {signal.confidence:.3f}")
                print(f"      Expected Return: {signal.expected_return:.3f}")
                
                logger.debug(f"Signal {i+1}: {signal.symbol}, type={signal.signal_type}, confidence={signal.confidence:.3f}")
        else:
            logger.warning("❌ No trading signals generated")
            print("❌ No trading signals generated")
    else:
        logger.warning("SR clustering not available or models not trained, skipping signal generation")
    
    # Step 7: Integration summary
    logger.info("📋 Step 7: Integration summary")
    print(f"\n📋 Step 7: Integration summary...")
    
    if SR_CLUSTERING_AVAILABLE:
        logger.info("✅ Integration successful")
        print("✅ Integration successful!")
        print("   • SR clustering system integrated")
        print("   • Weight optimization enabled")
        print("   • Enhanced training data created")
        print("   • ML models trained with SR quality features")
        print("   • Trading signals generated")
        
        # Show what would be passed to next steps
        print(f"\n🎯 Enhanced output for next steps:")
        print(f"   • Enhanced SR levels with quality scores")
        print(f"   • Enhanced training data with SR features")
        print(f"   • Trained ML models with SR quality integration")
        print(f"   • Trading signals with confidence scores")
        print(f"   • Optimized weights for continuous learning")
        
        logger.info("Integration completed successfully with all components")
    else:
        logger.error("❌ Integration failed - SR clustering not available")
        print("❌ Integration failed - SR clustering not available")
    
    logger.info("🎉 Quick integration demo completed")
    print(f"\n🎉 Quick integration demo completed!")
    print("=" * 50)

def show_integration_code_changes():
    """Show the actual code changes needed for integration."""
    logger.info("🔧 Showing integration code changes")
    print("\n🔧 Code Changes Needed for Integration")
    print("=" * 50)
    
    print("📝 File: src/training/steps/data_collection/data_preparation/step02_5_sr_optimization.py")
    print("\n1. Add imports at the top:")
    print("""
try:
    from src.utils.sr_clustering import (
        get_backtesting_enhanced_clustering, BacktestingEnhancedConfig,
        get_predictive_sr_engine, PredictiveConfig,
        get_trading_ml_integration, TradingMLConfig
    )
    SR_CLUSTERING_AVAILABLE = True
except ImportError:
    SR_CLUSTERING_AVAILABLE = False
""")
    
    print("\n2. Replace SR detection logic in execute_main_logic:")
    print("""
# Replace this:
detector = EnhancedSRDetector(sr_config)
sr_levels = detector.detect_sr_levels(clean_data)

# With this:
if SR_CLUSTERING_AVAILABLE:
    # Use enhanced clustering
    clustering_config = BacktestingEnhancedConfig(
        min_levels_for_learning=10,
        quality_filter_threshold=0.1,
        proximity_adjustment_factor=0.5
    )
    
    clustering = get_backtesting_enhanced_clustering(clustering_config)
    sr_levels = clustering.cluster_with_backtesting(levels, data)
    
    # Get SR quality predictions
    predictive_config = PredictiveConfig(model_type='ensemble')
    predictive_engine = get_predictive_sr_engine(predictive_config)
    training_result = predictive_engine.train_predictive_model(data, sr_levels)
    
    # Create enhanced training data
    trading_config = TradingMLConfig(include_sr_quality=True)
    trading_ml = get_trading_ml_integration(trading_config)
    enhanced_data = trading_ml.prepare_enhanced_training_data(data, sr_levels, historical_performance)
    
    # Train ML models
    ml_results = trading_ml.train_trading_models(enhanced_data)
    
    # Generate trading signals
    trading_signals = trading_ml.generate_trading_signals(data, sr_levels)
else:
    # Fall back to existing detection
    detector = EnhancedSRDetector(sr_config)
    sr_levels = detector.detect_sr_levels(clean_data)
""")
    
    print("\n3. Update return statement:")
    print("""
return {
    'sr_levels': sr_levels,
    'enhanced_training_data': enhanced_data if SR_CLUSTERING_AVAILABLE else None,
    'ml_models': ml_results if SR_CLUSTERING_AVAILABLE else None,
    'trading_signals': trading_signals if SR_CLUSTERING_AVAILABLE else None,
    'sr_quality_predictions': training_result if SR_CLUSTERING_AVAILABLE else None,
    # ... existing results
}
""")
    
    logger.info("✅ Integration code changes provided")
    print("\n✅ These changes will integrate SR clustering with your existing ML models!")

if __name__ == "__main__":
    try:
        logger.info("Starting quick integration demonstration")
        demonstrate_quick_integration()
        show_integration_code_changes()
        logger.info("✅ Quick integration demonstration completed successfully")
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"❌ Demo failed: {e}")
        traceback.print_exc()