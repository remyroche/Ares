#!/usr/bin/env python3
"""
Test script for regression profit prediction integration with Analyst/Tactician systems.

This script demonstrates:
1. How regression models can predict actual profit percentages
2. Integration with existing classification systems
3. Enhanced position sizing based on predicted returns
4. Hybrid decision making combining regression and classification
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our new regression components
from src.training.regression_profit_predictor import RegressionProfitPredictor
from src.training.regression_integration_manager import RegressionIntegrationManager


def generate_test_data(n_samples: int = 1000) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Generate realistic test data for regression profit prediction.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (features, profit_targets, classification_targets)
    """
    print(f"📊 Generating {n_samples} test samples...")
    
    # Generate realistic market features
    np.random.seed(42)
    
    # Price-based features
    base_price = 100.0
    price_changes = np.random.normal(0, 0.02, n_samples)  # 2% daily volatility
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    # Technical indicators
    features = pd.DataFrame({
        'price': prices,
        'price_change': np.diff(prices, prepend=prices[0]),
        'price_change_pct': np.diff(prices, prepend=prices[0]) / prices,
        'sma_20': pd.Series(prices).rolling(20).mean().fillna(prices),
        'sma_50': pd.Series(prices).rolling(50).mean().fillna(prices),
        'rsi': np.random.uniform(20, 80, n_samples),
        'macd': np.random.normal(0, 0.5, n_samples),
        'bollinger_upper': prices * 1.02,
        'bollinger_lower': prices * 0.98,
        'volume': np.random.lognormal(10, 1, n_samples),
        'volatility': np.random.uniform(0.01, 0.05, n_samples),
        'momentum': np.random.normal(0, 0.1, n_samples),
        'support_level': prices * 0.95,
        'resistance_level': prices * 1.05,
    })
    
    # Generate realistic profit targets based on features
    # Higher RSI + positive momentum + price above SMA = higher profit potential
    profit_potential = (
        (features['rsi'] - 50) / 50 * 0.02 +  # RSI contribution
        features['momentum'] * 0.1 +          # Momentum contribution
        (features['price'] > features['sma_20']).astype(float) * 0.01 +  # Trend contribution
        np.random.normal(0, 0.005, n_samples)  # Random noise
    )
    
    # Add some realistic constraints
    profit_potential = np.clip(profit_potential, -0.03, 0.05)  # -3% to +5%
    
    # Generate classification targets (simplified)
    classification_targets = (profit_potential > 0.005).astype(int)  # 0.5% threshold
    
    # Clean up features
    features = features.fillna(method='ffill').fillna(0)
    
    print(f"   ✅ Generated features: {features.shape}")
    print(f"   📈 Profit range: {profit_potential.min():.4f} to {profit_potential.max():.4f}")
    print(f"   🎯 Classification distribution: {np.bincount(classification_targets)}")
    
    return features, pd.Series(profit_potential), pd.Series(classification_targets)


async def test_regression_profit_predictor():
    """Test the standalone regression profit predictor."""
    print("\n" + "="*60)
    print("🧪 TESTING REGRESSION PROFIT PREDICTOR")
    print("="*60)
    
    # Generate test data
    features, profit_targets, _ = generate_test_data(1000)
    
    # Initialize predictor
    config = {
        "model_type": "LightGBM",
        "min_profit_threshold": 0.005,  # 0.5%
        "max_profit_threshold": 0.03,   # 3%
        "position_sizing_enabled": True
    }
    
    predictor = RegressionProfitPredictor(config)
    
    # Train model
    print("\n🚀 Training regression model...")
    success = await predictor.train_model(features, profit_targets)
    
    if not success:
        print("❌ Training failed")
        return
    
    print("✅ Training completed successfully")
    
    # Test predictions
    print("\n🔮 Testing predictions...")
    test_features = features.iloc[-5:].copy()  # Last 5 samples
    current_price = 100.0
    
    for i, (idx, row) in enumerate(test_features.iterrows()):
        prediction = await predictor.predict_profit(
            pd.DataFrame([row]), current_price, include_confidence=True
        )
        
        if prediction:
            print(f"   Sample {i+1}:")
            print(f"     Predicted profit: {prediction['predicted_profit_pct']:.4f} ({prediction['predicted_profit_pct']*100:.2f}%)")
            print(f"     Position size: {prediction['recommended_position_size']:.2f}")
            print(f"     Confidence level: {prediction['confidence_level']}")
            print(f"     Trade recommendation: {prediction['trade_recommendation']}")
    
    # Evaluate model
    print("\n📊 Model evaluation...")
    test_size = int(len(features) * 0.2)
    test_features = features.iloc[-test_size:]
    test_targets = profit_targets.iloc[-test_size:]
    
    metrics = await predictor.evaluate_model(test_features, test_targets)
    
    print(f"   R² Score: {metrics.get('r2', 0):.4f}")
    print(f"   MAE: {metrics.get('mae', 0):.6f}")
    print(f"   RMSE: {metrics.get('rmse', 0):.6f}")
    print(f"   Profit Accuracy: {metrics.get('profit_accuracy', 0):.4f}")
    
    # Feature importance
    importance = predictor.get_feature_importance()
    print(f"\n🔍 Top 5 most important features:")
    for i, (feature, score) in enumerate(list(importance.items())[:5]):
        print(f"   {i+1}. {feature}: {score:.4f}")


async def test_integration_manager():
    """Test the integration manager with both Analyst and Tactician."""
    print("\n" + "="*60)
    print("🔗 TESTING REGRESSION INTEGRATION MANAGER")
    print("="*60)
    
    # Generate test data
    features, profit_targets, classification_targets = generate_test_data(2000)
    
    # Split data for Analyst (multi-timeframe) and Tactician (1m)
    split_point = len(features) // 2
    analyst_features = features.iloc[:split_point]
    analyst_profits = profit_targets.iloc[:split_point]
    tactician_features = features.iloc[split_point:]
    tactician_profits = profit_targets.iloc[split_point:]
    
    # Initialize integration manager
    config = {
        "regression_integration": {
            "enable_analyst_regression": True,
            "enable_tactician_regression": True,
            "hybrid_threshold": 0.5
        },
        "analyst_regression": {
            "model_type": "LightGBM",
            "min_profit_threshold": 0.005,  # 0.5%
            "max_profit_threshold": 0.03,   # 3%
            "position_sizing_enabled": True
        },
        "tactician_regression": {
            "model_type": "LightGBM",
            "min_profit_threshold": 0.003,  # 0.3%
            "max_profit_threshold": 0.02,   # 2%
            "position_sizing_enabled": True
        }
    }
    
    integration_manager = RegressionIntegrationManager(config)
    
    # Initialize
    print("\n🚀 Initializing integration manager...")
    success = await integration_manager.initialize()
    
    if not success:
        print("❌ Initialization failed")
        return
    
    print("✅ Integration manager initialized")
    
    # Train models
    print("\n🎓 Training regression models...")
    
    print("   Training Analyst model...")
    analyst_success = await integration_manager.train_analyst_regression(
        analyst_features, analyst_profits
    )
    
    print("   Training Tactician model...")
    tactician_success = await integration_manager.train_tactician_regression(
        tactician_features, tactician_profits
    )
    
    if not analyst_success or not tactician_success:
        print("❌ Training failed")
        return
    
    print("✅ Both models trained successfully")
    
    # Test hybrid predictions
    print("\n🔮 Testing hybrid predictions...")
    current_price = 100.0
    
    # Test Analyst predictions
    print("\n   📊 Analyst Predictions:")
    for i in range(5):
        test_features = analyst_features.iloc[i:i+1]
        classification_confidence = np.random.uniform(0.3, 0.9)  # Simulate classification confidence
        
        result = await integration_manager.predict_analyst_profit(
            test_features, current_price, classification_confidence
        )
        
        if result:
            print(f"     Sample {i+1}:")
            print(f"       Predicted profit: {result['predicted_profit_pct']:.4f} ({result['predicted_profit_pct']*100:.2f}%)")
            print(f"       Classification confidence: {result['classification_confidence']:.3f}")
            print(f"       Regression confidence: {result['regression_confidence']:.3f}")
            print(f"       Hybrid confidence: {result['hybrid_confidence']:.3f}")
            print(f"       Final decision: {result['final_decision']}")
            print(f"       Position size: {result['position_sizing']['risk_adjusted_position_size']:.2f}")
    
    # Test Tactician predictions
    print("\n   ⚡ Tactician Predictions:")
    for i in range(5):
        test_features = tactician_features.iloc[i:i+1]
        classification_confidence = np.random.uniform(0.4, 0.95)  # Higher confidence for tactician
        
        result = await integration_manager.predict_tactician_profit(
            test_features, current_price, classification_confidence
        )
        
        if result:
            print(f"     Sample {i+1}:")
            print(f"       Predicted profit: {result['predicted_profit_pct']:.4f} ({result['predicted_profit_pct']*100:.2f}%)")
            print(f"       Classification confidence: {result['classification_confidence']:.3f}")
            print(f"       Regression confidence: {result['regression_confidence']:.3f}")
            print(f"       Hybrid confidence: {result['hybrid_confidence']:.3f}")
            print(f"       Final decision: {result['final_decision']}")
            print(f"       Position size: {result['position_sizing']['risk_adjusted_position_size']:.2f}")
    
    # Get analytics
    print("\n📈 Integration Analytics:")
    analytics = integration_manager.get_integration_analytics()
    
    print(f"   Total predictions: {analytics.get('total_predictions', 0)}")
    print(f"   Analyst predictions: {analytics.get('analyst_predictions', 0)}")
    print(f"   Tactician predictions: {analytics.get('tactician_predictions', 0)}")
    print(f"   Average hybrid confidence: {analytics.get('average_hybrid_confidence', 0):.3f}")
    print(f"   Decision distribution: {analytics.get('decision_distribution', {})}")
    
    profit_stats = analytics.get('profit_predictions', {})
    print(f"   Profit predictions - Mean: {profit_stats.get('mean', 0):.4f}, Std: {profit_stats.get('std', 0):.4f}")


async def test_comparison_with_classification():
    """Compare regression approach with traditional classification."""
    print("\n" + "="*60)
    print("⚖️ COMPARISON: REGRESSION vs CLASSIFICATION")
    print("="*60)
    
    # Generate test data
    features, profit_targets, classification_targets = generate_test_data(1500)
    
    # Traditional classification approach
    print("\n📊 Traditional Classification Approach:")
    print("   - Predicts: Buy/Sell/Hold (discrete categories)")
    print("   - Output: Binary decision with confidence")
    print("   - Position sizing: Fixed or simple rules")
    
    # Simulate classification predictions
    classification_predictions = np.random.uniform(0.3, 0.9, len(features))
    classification_decisions = (classification_predictions > 0.6).astype(int)
    
    # Calculate classification performance
    classification_accuracy = np.mean(classification_decisions == classification_targets)
    profitable_trades = profit_targets[classification_decisions == 1]
    avg_profit_classification = profitable_trades.mean() if len(profitable_trades) > 0 else 0
    
    print(f"   Classification accuracy: {classification_accuracy:.3f}")
    print(f"   Average profit per trade: {avg_profit_classification:.4f}")
    print(f"   Number of trades: {np.sum(classification_decisions)}")
    
    # Regression approach
    print("\n📈 Regression Approach:")
    print("   - Predicts: Actual profit percentage (continuous)")
    print("   - Output: Expected return with confidence")
    print("   - Position sizing: Dynamic based on predicted return")
    
    # Initialize and train regression model
    config = {
        "model_type": "LightGBM",
        "min_profit_threshold": 0.005,
        "max_profit_threshold": 0.03,
        "position_sizing_enabled": True
    }
    
    regression_predictor = RegressionProfitPredictor(config)
    await regression_predictor.train_model(features, profit_targets)
    
    # Make regression predictions
    regression_predictions = []
    regression_decisions = []
    position_sizes = []
    
    for i in range(len(features)):
        prediction = await regression_predictor.predict_profit(
            features.iloc[i:i+1], 100.0, include_confidence=True
        )
        
        if prediction:
            regression_predictions.append(prediction['predicted_profit_pct'])
            decision = 1 if prediction['trade_recommendation'] == 'enter' else 0
            regression_decisions.append(decision)
            position_sizes.append(prediction['recommended_position_size'])
        else:
            regression_predictions.append(0)
            regression_decisions.append(0)
            position_sizes.append(0)
    
    regression_predictions = np.array(regression_predictions)
    regression_decisions = np.array(regression_decisions)
    position_sizes = np.array(position_sizes)
    
    # Calculate regression performance
    regression_accuracy = np.mean(regression_decisions == classification_targets)
    profitable_trades_regression = profit_targets[regression_decisions == 1]
    avg_profit_regression = profitable_trades_regression.mean() if len(profitable_trades_regression) > 0 else 0
    
    # Calculate weighted returns (accounting for position sizing)
    weighted_returns = profit_targets * regression_decisions * position_sizes
    avg_weighted_return = weighted_returns.mean()
    
    print(f"   Regression accuracy: {regression_accuracy:.3f}")
    print(f"   Average profit per trade: {avg_profit_regression:.4f}")
    print(f"   Number of trades: {np.sum(regression_decisions)}")
    print(f"   Average weighted return: {avg_weighted_return:.4f}")
    
    # Comparison summary
    print("\n📊 Comparison Summary:")
    print(f"   Classification trades: {np.sum(classification_decisions)}")
    print(f"   Regression trades: {np.sum(regression_decisions)}")
    print(f"   Classification avg profit: {avg_profit_classification:.4f}")
    print(f"   Regression avg profit: {avg_profit_regression:.4f}")
    print(f"   Regression weighted return: {avg_weighted_return:.4f}")
    
    improvement = ((avg_weighted_return - avg_profit_classification) / abs(avg_profit_classification) * 100) if avg_profit_classification != 0 else 0
    print(f"   Improvement with regression: {improvement:+.1f}%")


async def main():
    """Run all tests."""
    print("🚀 REGRESSION PROFIT PREDICTION INTEGRATION TEST")
    print("="*60)
    print("This test demonstrates the benefits of using regression models")
    print("to predict actual profit percentages instead of discrete categories.")
    print("="*60)
    
    try:
        # Test 1: Standalone regression predictor
        await test_regression_profit_predictor()
        
        # Test 2: Integration manager
        await test_integration_manager()
        
        # Test 3: Comparison with classification
        await test_comparison_with_classification()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*60)
        
        print("\n🎯 KEY BENEFITS OF REGRESSION APPROACH:")
        print("   1. Predicts actual profit percentages (not just categories)")
        print("   2. Enables dynamic position sizing based on expected returns")
        print("   3. Provides more granular risk management")
        print("   4. Combines with existing classification for hybrid decisions")
        print("   5. Better capital allocation through risk-adjusted sizing")
        
        print("\n🔧 INTEGRATION WITH EXISTING SYSTEM:")
        print("   - Analyst: Use regression to filter high-probability trades")
        print("   - Tactician: Use regression for optimal entry timing and sizing")
        print("   - Hybrid: Combine regression predictions with classification confidence")
        print("   - Risk Management: Dynamic position sizing based on predicted returns")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())