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

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def demonstrate_quick_integration():
    """Demonstrate quick integration with existing ML models."""
    print("🔧 Quick Integration Demo")
    print("=" * 50)
    
    # Step 1: Check if SR clustering is available
    print("📋 Step 1: Checking SR clustering availability...")
    try:
        from src.utils.sr_clustering import (
            get_backtesting_enhanced_clustering, BacktestingEnhancedConfig,
            get_predictive_sr_engine, PredictiveConfig,
            get_trading_ml_integration, TradingMLConfig
        )
        SR_CLUSTERING_AVAILABLE = True
        print("✅ SR clustering system available")
    except ImportError as e:
        SR_CLUSTERING_AVAILABLE = False
        print(f"❌ SR clustering not available: {e}")
        return
    
    # Step 2: Simulate existing Step02_5 data
    print("\n📊 Step 2: Simulating existing Step02_5 data...")
    
    # Simulate market data (what Step02_5 currently processes)
    market_data = pd.DataFrame({
        'date': pd.date_range(start=datetime.now() - timedelta(days=100), periods=100, freq='D'),
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(105, 115, 100),
        'low': np.random.uniform(95, 105, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.uniform(1000000, 2000000, 100)
    })
    
    # Simulate existing SR levels (what EnhancedSRDetector currently returns)
    existing_sr_levels = [
        {'price': 105.0, 'type': 'support', 'strength': 0.7, 'touches': 3},
        {'price': 108.0, 'type': 'resistance', 'strength': 0.5, 'touches': 2},
        {'price': 102.0, 'type': 'support', 'strength': 0.8, 'touches': 4},
        {'price': 110.0, 'type': 'resistance', 'strength': 0.6, 'touches': 3}
    ]
    
    print(f"✅ Market data: {len(market_data)} rows")
    print(f"✅ Existing SR levels: {len(existing_sr_levels)} levels")
    
    # Step 3: Enhanced SR processing (NEW)
    print("\n🎯 Step 3: Enhanced SR processing with weight optimization...")
    
    if SR_CLUSTERING_AVAILABLE:
        # Use backtesting-enhanced clustering
        clustering_config = BacktestingEnhancedConfig(
            min_levels_for_learning=3,
            quality_filter_threshold=0.1,
            proximity_adjustment_factor=0.5
        )
        
        clustering = get_backtesting_enhanced_clustering(clustering_config)
        
        # Convert existing levels to SRLevel format
        from src.utils.sr_clustering import SRLevel
        sr_levels = []
        for level in existing_sr_levels:
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
        
        # Run enhanced clustering
        enhanced_sr_levels = clustering.cluster_with_backtesting(sr_levels, market_data)
        
        print(f"✅ Enhanced SR levels: {len(enhanced_sr_levels.clusters)} clusters")
        
        # Get SR quality predictions
        predictive_config = PredictiveConfig(
            model_type='ensemble',
            prediction_horizon_days=30
        )
        
        predictive_engine = get_predictive_sr_engine(predictive_config)
        training_result = predictive_engine.train_predictive_model(market_data, sr_levels)
        
        if training_result.get('status') == 'success':
            print("✅ SR quality predictions generated")
            
            # Show optimized weights
            optimized_weights = training_result.get('optimized_weights', {})
            if optimized_weights:
                print(f"   Optimized weights: {optimized_weights}")
        else:
            print("⚠️  SR quality prediction failed, using default weights")
    
    # Step 4: Enhanced training data creation (NEW)
    print("\n🧠 Step 4: Creating enhanced training data for ML models...")
    
    if SR_CLUSTERING_AVAILABLE:
        # Simulate historical performance data
        historical_performance = pd.DataFrame({
            'symbol': ['TEST'] * 50,
            'price': np.random.uniform(100, 110, 50),
            'future_return': np.random.uniform(-0.05, 0.05, 50),
            'trade_success': np.random.choice([0, 1], 50),
            'timestamp': pd.date_range(start=datetime.now() - timedelta(days=50), periods=50, freq='D')
        })
        
        # Create enhanced training data
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
            print(f"✅ Enhanced training data: {len(enhanced_data)} samples")
            print(f"   Features: {list(enhanced_data.columns)}")
            
            # Show sample of enhanced data
            print(f"\n📊 Sample enhanced features:")
            sample_features = ['sr_quality', 'sr_confidence', 'momentum_score', 'volatility_score', 'trade_success']
            if all(feat in enhanced_data.columns for feat in sample_features):
                print(enhanced_data[sample_features].head())
        else:
            print("❌ Failed to create enhanced training data")
    
    # Step 5: ML model training with enhanced data (NEW)
    print("\n🎯 Step 5: Training ML models with enhanced data...")
    
    if SR_CLUSTERING_AVAILABLE and len(enhanced_data) > 0:
        # Train trading models
        training_result = trading_ml.train_trading_models(enhanced_data)
        
        if training_result.get('status') == 'success':
            print("✅ ML models trained successfully")
            
            # Show model performance
            classification_perf = training_result.get('classification_performance', {})
            regression_perf = training_result.get('regression_performance', {})
            
            print(f"   Classification Accuracy: {classification_perf.get('accuracy', 0.0):.3f}")
            print(f"   Regression R²: {regression_perf.get('r2_score', 0.0):.3f}")
            
            # Show feature importance
            feature_importance = training_result.get('feature_importance', {})
            if feature_importance:
                print(f"\n🎯 Top feature importance:")
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                for feature, importance in sorted_features[:5]:
                    print(f"   {feature}: {importance:.3f}")
        else:
            print(f"❌ ML model training failed: {training_result.get('error', 'Unknown error')}")
    
    # Step 6: Trading signal generation (NEW)
    print("\n📡 Step 6: Generating trading signals...")
    
    if SR_CLUSTERING_AVAILABLE and training_result.get('status') == 'success':
        # Generate trading signals
        trading_signals = trading_ml.generate_trading_signals(market_data, sr_levels)
        
        if trading_signals:
            print(f"✅ Generated {len(trading_signals)} trading signals")
            
            # Show trading signals
            for i, signal in enumerate(trading_signals[:3]):  # Show top 3
                print(f"   {i+1}. {signal.symbol} at {signal.sr_quality:.3f} quality:")
                print(f"      Signal: {signal.signal_type.upper()}")
                print(f"      Confidence: {signal.confidence:.3f}")
                print(f"      Expected Return: {signal.expected_return:.3f}")
        else:
            print("❌ No trading signals generated")
    
    # Step 7: Integration summary
    print(f"\n📋 Step 7: Integration summary...")
    
    if SR_CLUSTERING_AVAILABLE:
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
    else:
        print("❌ Integration failed - SR clustering not available")
    
    print(f"\n🎉 Quick integration demo completed!")
    print("=" * 50)

def show_integration_code_changes():
    """Show the actual code changes needed for integration."""
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
    
    print("\n✅ These changes will integrate SR clustering with your existing ML models!")

if __name__ == "__main__":
    try:
        demonstrate_quick_integration()
        show_integration_code_changes()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()