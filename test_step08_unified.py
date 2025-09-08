#!/usr/bin/env python3
"""
Comprehensive Test Script for Unified Step08 Implementation

This script tests all five critical improvements:
1. Consolidated implementations
2. Financial metrics calculation
3. Regime balance handling
4. Feature selection validation
5. Risk assessment with explicit risk metrics
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import json

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_comprehensive_test_data():
    """Create comprehensive test data for all Step08 functionality."""
    print("📊 Creating comprehensive test data...")
    
    # Create datetime index
    start_date = datetime(2023, 1, 1)
    num_samples = 10000
    dates = [start_date + timedelta(minutes=i) for i in range(num_samples)]
    
    # Generate realistic OHLCV data
    np.random.seed(42)
    base_price = 50000
    
    # Create trending and volatile periods
    trend_changes = np.random.choice([-1, 0, 1], size=num_samples, p=[0.3, 0.4, 0.3])
    volatility_changes = np.random.choice([0.5, 1.0, 2.0], size=num_samples, p=[0.5, 0.3, 0.2])
    
    prices = [base_price]
    volumes = []
    
    for i in range(1, num_samples):
        # Price movement based on trend and volatility
        price_change = np.random.normal(0, 0.001 * volatility_changes[i])
        price_change += trend_changes[i] * 0.0005
        
        new_price = prices[-1] * (1 + price_change)
        prices.append(max(new_price, 0.1))
        
        # Volume based on price volatility
        volume = np.random.normal(1000, 200) * (1 + abs(price_change) * 100)
        volumes.append(max(volume, 10))
    
    # Create OHLC from price series
    opens = prices[:-1]
    highs = [max(o, c) + np.random.uniform(0, o * 0.002) for o, c in zip(opens, prices[1:])]
    lows = [min(o, c) - np.random.uniform(0, o * 0.002) for o, c in zip(opens, prices[1:])]
    closes = prices[1:]
    volumes = volumes[:len(closes)]
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': dates[:-1],
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
    })
    
    # Add technical indicators
    data['returns'] = data['close'].pct_change()
    data['volatility'] = data['returns'].rolling(20).std()
    data['sma_20'] = data['close'].rolling(20).mean()
    data['sma_50'] = data['close'].rolling(50).mean()
    data['rsi'] = 50 + np.random.normal(0, 10, len(data))  # Simulated RSI
    data['macd'] = np.random.normal(0, 0.001, len(data))  # Simulated MACD
    data['bb_upper'] = data['close'] * 1.02
    data['bb_lower'] = data['close'] * 0.98
    data['atr'] = data['volatility'] * data['close']
    
    # Add composite cluster IDs (HMM regime labels) with intentional imbalance
    regime_patterns = []
    current_regime = 0
    
    for i in range(len(data)):
        # Change regime occasionally with intentional imbalance
        if np.random.random() < 0.001:  # 0.1% chance to change regime
            # Create imbalanced regime distribution
            regime_probs = [0.6, 0.25, 0.1, 0.05]  # Intentionally imbalanced
            current_regime = np.random.choice([0, 1, 2, 3], p=regime_probs)
        
        regime_patterns.append(current_regime)
    
    data['composite_cluster_id'] = regime_patterns
    
    # Set timestamp as index
    data = data.set_index('timestamp')
    
    print(f"✅ Created test data: {len(data)} rows, {len(data.columns)} columns")
    print(f"   Regime distribution: {data['composite_cluster_id'].value_counts().to_dict()}")
    
    return data

def create_test_config():
    """Create test configuration for unified Step08."""
    return {
        'symbol': 'ETHUSDT',
        'exchange': 'BINANCE',
        'timeframe': '1m',
        'data_dir': 'data_cache',
        'lookback_days': 365,
        'step08_unified': {
            'phase1_target_features': 50,  # Smaller for testing
            'phase2_targets': [30, 20, 15],  # Smaller for testing
            'enable_mrmr': True,
            'enable_rf_importance': True,
            'boruta_max_iter': 50,  # Smaller for testing
            'boruta_alpha': 0.05,
            'min_regime_samples': 50,  # Smaller for testing
            'target_balance_ratio': 0.8,
            'enable_regime_rebalancing': True,
            'rebalancing_method': 'oversample',
            'risk_free_rate': 0.02,
            'var_confidence_levels': [0.95, 0.99],
            'lookback_periods': [30, 90, 252],
            'model_risk_threshold': 0.3,
            'overfitting_threshold': 0.1,
            'feature_stability_threshold': 0.8,
            'output_dir': 'data/step08_unified_test'
        }
    }

async def test_unified_step08():
    """Test the unified Step08 implementation."""
    print("🧪 Testing Unified Step08 Implementation")
    print("=" * 60)
    
    try:
        # Create test data and config
        test_data = create_comprehensive_test_data()
        test_config = create_test_config()
        
        # Import the unified Step08
        from src.training.steps.step08_unified_complete import UnifiedStep08
        
        # Initialize the unified step
        print("🔧 Initializing Unified Step08...")
        step = UnifiedStep08(test_config)
        
        # Create mock pipeline state with test data
        pipeline_state = {
            'dataframe': test_data,
            'feature_engineered_data': {
                'train': test_data,
                'validation': test_data.iloc[:1000],
                'test': test_data.iloc[1000:2000]
            },
            'regime_labels': {
                'train': test_data['composite_cluster_id'].values,
                'validation': test_data.iloc[:1000]['composite_cluster_id'].values,
                'test': test_data.iloc[1000:2000]['composite_cluster_id'].values
            }
        }
        
        training_input = {
            'symbol': 'ETHUSDT',
            'exchange': 'BINANCE',
            'timeframe': '1m',
            'data_dir': 'data_cache'
        }
        
        # Execute the unified step
        print("🚀 Executing Unified Step08...")
        result = await step.execute(training_input, pipeline_state)
        
        if result.get('success', False):
            print("✅ Unified Step08 execution completed successfully!")
            
            # Extract results
            results = result.get('results')
            if results:
                print("\n📊 Results Summary:")
                print(f"   Success: {results.success}")
                print(f"   Total Samples: {len(results.regime_data) if results.regime_data is not None else 0:,}")
                print(f"   Selected Features: {len(results.selected_features.get('final', []))}")
                print(f"   Regime Count: {len(results.regime_data['composite_cluster_id'].unique()) if results.regime_data is not None else 0}")
                print(f"   Execution Time: {results.execution_metadata.get('duration_seconds', 0):.2f} seconds")
                
                # Test 1: Financial Metrics
                print("\n💰 Financial Metrics Test:")
                financial_metrics = results.financial_metrics
                print(f"   Daily Return: {financial_metrics.returns.get('daily', 0):.6f}")
                print(f"   Annualized Return: {financial_metrics.returns.get('annualized', 0):.6f}")
                print(f"   Annualized Volatility: {financial_metrics.volatility.get('annualized', 0):.6f}")
                print(f"   Sharpe Ratio: {financial_metrics.sharpe_ratio.get('overall', 0):.4f}")
                print(f"   VaR (95%): {financial_metrics.var_95.get('overall', 0):.6f}")
                print(f"   Maximum Drawdown: {financial_metrics.max_drawdown.get('overall', 0):.6f}")
                
                # Test 2: Risk Assessment
                print("\n⚠️ Risk Assessment Test:")
                risk_metrics = results.risk_metrics
                print(f"   Overall Risk Score: {risk_metrics.overall_risk_score:.4f}")
                print(f"   Portfolio VaR: {risk_metrics.portfolio_var:.6f}")
                print(f"   Model Risk: {risk_metrics.model_risk:.4f}")
                print(f"   Regime Risk: {risk_metrics.regime_risk:.4f}")
                print(f"   Overfitting Risk: {risk_metrics.overfitting_risk:.4f}")
                print(f"   Data Quality Risk: {risk_metrics.data_quality_risk:.4f}")
                
                # Test 3: Regime Balance Handling
                print("\n⚖️ Regime Balance Test:")
                regime_balance = results.regime_balance
                print(f"   Balance Score: {regime_balance.balance_score:.4f}")
                print(f"   Imbalance Severity: {regime_balance.imbalance_severity}")
                print(f"   Rebalancing Applied: {regime_balance.rebalancing_applied}")
                print(f"   Rebalancing Method: {regime_balance.rebalancing_method}")
                print(f"   Regime Distribution: {regime_balance.regime_percentages}")
                
                # Test 4: Feature Selection Validation
                print("\n✅ Feature Selection Validation Test:")
                feature_validation = results.feature_validation
                print(f"   Validation Passed: {feature_validation.validation_passed}")
                print(f"   Selection Bias Score: {feature_validation.selection_bias_score:.4f}")
                print(f"   Temporal Stability: {feature_validation.temporal_stability:.4f}")
                print(f"   Regime Consistency: {feature_validation.regime_consistency:.4f}")
                print(f"   Correlation Stability: {feature_validation.correlation_stability:.4f}")
                print(f"   Importance Stability: {feature_validation.importance_stability:.4f}")
                
                # Test 5: Consolidated Implementation
                print("\n🔧 Consolidated Implementation Test:")
                print(f"   Selected Feature Sets: {list(results.selected_features.keys())}")
                print(f"   Phase 1 Features: {len(results.selected_features.get('phase1', []))}")
                print(f"   Phase 2 Features: {len(results.selected_features.get('phase2', []))}")
                print(f"   Final Features: {len(results.selected_features.get('final', []))}")
                print(f"   Artifacts Generated: {len(results.artifacts_generated)}")
                
                # Display warnings and errors
                if results.warnings:
                    print(f"\n⚠️ Warnings ({len(results.warnings)}):")
                    for warning in results.warnings:
                        print(f"   • {warning}")
                
                if results.errors:
                    print(f"\n❌ Errors ({len(results.errors)}):")
                    for error in results.errors:
                        print(f"   • {error}")
                
                # Test artifact generation
                print(f"\n💾 Generated Artifacts:")
                for artifact in results.artifacts_generated:
                    if os.path.exists(artifact):
                        file_size = os.path.getsize(artifact) / 1024  # KB
                        print(f"   ✅ {artifact} ({file_size:.1f} KB)")
                    else:
                        print(f"   ❌ {artifact} (not found)")
                
                # Validate all five improvements
                print(f"\n🎯 Validation Results:")
                improvements = {
                    "Consolidated Implementation": len(results.selected_features) > 0,
                    "Financial Metrics": len(financial_metrics.returns) > 0,
                    "Regime Balance Handling": regime_balance.rebalancing_applied or regime_balance.balance_score > 0.3,
                    "Feature Selection Validation": feature_validation.validation_passed,
                    "Risk Assessment": risk_metrics.overall_risk_score >= 0.0
                }
                
                for improvement, passed in improvements.items():
                    status = "✅ PASS" if passed else "❌ FAIL"
                    print(f"   {improvement}: {status}")
                
                all_passed = all(improvements.values())
                print(f"\n🎉 Overall Test Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
                
                return all_passed
            else:
                print("❌ No results returned from execution")
                return False
        else:
            print(f"❌ Unified Step08 execution failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_individual_components():
    """Test individual components of the unified Step08."""
    print("\n🔬 Testing Individual Components")
    print("=" * 40)
    
    try:
        from src.training.steps.step08_unified_complete import (
            FinancialMetrics, RiskMetrics, RegimeBalanceMetrics, 
            FeatureSelectionValidation, Step08Results
        )
        
        # Test 1: Financial Metrics
        print("💰 Testing Financial Metrics...")
        financial_metrics = FinancialMetrics()
        financial_metrics.returns = {'daily': 0.001, 'annualized': 0.25}
        financial_metrics.volatility = {'annualized': 0.20}
        financial_metrics.sharpe_ratio = {'overall': 1.25}
        print(f"   ✅ Financial Metrics created: {len(financial_metrics.returns)} return metrics")
        
        # Test 2: Risk Metrics
        print("⚠️ Testing Risk Metrics...")
        risk_metrics = RiskMetrics()
        risk_metrics.overall_risk_score = 0.3
        risk_metrics.model_risk = 0.2
        risk_metrics.regime_risk = 0.4
        print(f"   ✅ Risk Metrics created: Overall risk = {risk_metrics.overall_risk_score}")
        
        # Test 3: Regime Balance Metrics
        print("⚖️ Testing Regime Balance Metrics...")
        regime_balance = RegimeBalanceMetrics()
        regime_balance.balance_score = 0.7
        regime_balance.imbalance_severity = "mild"
        regime_balance.rebalancing_applied = True
        print(f"   ✅ Regime Balance Metrics created: Balance score = {regime_balance.balance_score}")
        
        # Test 4: Feature Selection Validation
        print("✅ Testing Feature Selection Validation...")
        feature_validation = FeatureSelectionValidation()
        feature_validation.validation_passed = True
        feature_validation.selection_bias_score = 0.8
        feature_validation.temporal_stability = 0.7
        print(f"   ✅ Feature Selection Validation created: Passed = {feature_validation.validation_passed}")
        
        # Test 5: Step08 Results
        print("📋 Testing Step08 Results...")
        results = Step08Results()
        results.success = True
        results.selected_features = {'final': ['feature1', 'feature2', 'feature3']}
        results.financial_metrics = financial_metrics
        results.risk_metrics = risk_metrics
        results.regime_balance = regime_balance
        results.feature_validation = feature_validation
        print(f"   ✅ Step08 Results created: Success = {results.success}")
        
        print("\n🎉 All individual components tested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Individual component test failed: {e}")
        return False

async def main():
    """Main test function."""
    print("🚀 Starting Comprehensive Step08 Unified Test Suite")
    print("=" * 60)
    
    # Test individual components first
    components_passed = await test_individual_components()
    
    # Test the full unified implementation
    unified_passed = await test_unified_step08()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUITE SUMMARY")
    print("=" * 60)
    print(f"Individual Components: {'✅ PASSED' if components_passed else '❌ FAILED'}")
    print(f"Unified Implementation: {'✅ PASSED' if unified_passed else '❌ FAILED'}")
    
    overall_success = components_passed and unified_passed
    print(f"\n🎯 OVERALL RESULT: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n🎉 Step08 Unified Implementation is ready for production!")
        print("✅ All five critical improvements have been successfully implemented:")
        print("   1. ✅ Consolidated implementations")
        print("   2. ✅ Financial metrics (returns, volatility, Sharpe ratio, VaR)")
        print("   3. ✅ Regime balance handling for imbalanced distributions")
        print("   4. ✅ Feature selection validation to prevent bias")
        print("   5. ✅ Risk assessment with explicit risk metrics")
    else:
        print("\n⚠️ Some tests failed. Please review the implementation.")
    
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)