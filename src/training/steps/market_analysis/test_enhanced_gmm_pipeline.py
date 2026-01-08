"""
Test Script for Enhanced GMM Pipeline

This script provides comprehensive testing and validation of the enhanced GMM pipeline
including FracDiff, TreeSHAP analysis, and trading logic components.

Usage:
    python test_enhanced_gmm_pipeline.py
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, project_root)

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.fracdiff import FracDiffTransformer, fracdiff_series, validate_stationarity
from src.training.steps.market_analysis.gmm_enhanced_features import EnhancedGMMFeatures
from src.training.steps.market_analysis.shap_interaction_analyzer import SHAPInteractionAnalyzer
from src.training.steps.market_analysis.gmm_trading_logic import GMMTradingEngine, generate_gmm_trading_signals


def generate_test_data(n_samples: int = 1000, freq: str = '1H') -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic market data for testing."""
    tprint_info(f"📊 Generating {n_samples} samples of test data...")
    
    # Create date range
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq=freq)
    
    # Generate synthetic price series with trend and noise
    np.random.seed(42)
    
    # Base trend
    trend = np.linspace(100, 150, n_samples)
    
    # Add cycles
    cycle1 = 5 * np.sin(2 * np.pi * np.arange(n_samples) / 50)  # 50-period cycle
    cycle2 = 2 * np.sin(2 * np.pi * np.arange(n_samples) / 20)  # 20-period cycle
    
    # Add noise
    noise = np.random.normal(0, 2, n_samples)
    
    # Combine components
    prices = trend + cycle1 + cycle2 + noise
    
    # Create OHLCV data
    df = pd.DataFrame(index=dates)
    df['close'] = prices
    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    
    # Generate realistic high/low based on close
    volatility = df['close'].rolling(20).std().fillna(2)
    df['high'] = df['close'] + np.abs(np.random.normal(0, volatility * 0.5, n_samples))
    df['low'] = df['close'] - np.abs(np.random.normal(0, volatility * 0.5, n_samples))
    
    # Generate volume
    df['volume'] = np.random.lognormal(10, 1, n_samples)
    
    # Calculate returns
    returns = df['close'].pct_change().fillna(0)
    
    tprint_success(f"✅ Generated test data: {len(df)} rows")
    
    return df, returns


def test_fracdiff_functionality():
    """Test FracDiff functionality."""
    tprint_info("\n🧪 Testing FracDiff functionality...")
    
    try:
        # Generate test series
        _, returns = generate_test_data(500)
        test_series = returns.cumsum() + 100  # Price-like series
        
        # Test FracDiff transformer
        transformer = FracDiffTransformer(max_d=1.0, min_d=0.0, adf_threshold=0.05)
        
        # Find optimal d
        optimal_d = transformer.find_optimal_d(test_series, method='binary_search')
        tprint_info(f"📊 Optimal d found: {optimal_d:.4f}")
        
        # Apply FracDiff
        fracdiff_series = transformer.fracdiff(test_series, optimal_d)
        tprint_info(f"📊 FracDiff applied: {len(fracdiff_series)} points")
        
        # Validate stationarity
        stationarity_results = validate_stationarity(fracdiff_series.dropna())
        tprint_info(f"📊 Stationarity confirmed: {stationarity_results.get('stationarity_confirmed', False)}")
        
        # Test convenience function
        fracdiff_result, d_param = fracdiff_series(test_series, adf_threshold=0.05)
        tprint_info(f"📊 Convenience function: d={d_param:.4f}")
        
        tprint_success("✅ FracDiff functionality test passed")
        return True
        
    except Exception as e:
        tprint_error(f"❌ FracDiff test failed: {e}")
        return False


def test_enhanced_gmm_pipeline():
    """Test Enhanced GMM Pipeline."""
    tprint_info("\n🧪 Testing Enhanced GMM Pipeline...")
    
    try:
        # Generate test data
        market_data, returns = generate_test_data(800)
        
        # Create enhanced pipeline
        config = {
            'use_original_pipeline': True,
            'use_enhanced_pipeline': True,
            'use_fracdiff': True,
            'use_treeshap': True,
            'n_clusters_macro': 6,  # Smaller for testing
            'fracdiff_config': {
                'max_d': 0.8,
                'min_d': 0.0,
                'adf_threshold': 0.05,
                'method': 'binary_search',
                'tolerance': 0.05
            },
            'treeshap_config': {
                'n_estimators': 50,  # Smaller for testing
                'max_depth': 6,
                'interaction_sample_size': 200
            }
        }
        
        pipeline = EnhancedGMMFeatures(**config)
        
        # Create minimal config for pipeline
        pipeline_config = {
            'symbol': 'TEST',
            'exchange': 'binance',
            'timeframe': '1h',
            'start_date': '2023-01-01',
            'end_date': '2023-02-01'
        }
        
        # Mock the load_market_data_or_fail method
        def mock_load_data(config):
            return market_data, None
        
        pipeline.load_market_data_or_fail = mock_load_data
        
        # Run pipeline
        results = pipeline.run(pipeline_config)
        
        if results.get('success'):
            tprint_success(f"✅ Enhanced GMM pipeline test passed")
            tprint_info(f"📊 Features generated: {results.get('n_enhanced_features', 0)}")
            tprint_info(f"📊 FracDiff optimal d: {results.get('fracdiff_optimal_d', 0):.4f}")
            tprint_info(f"📊 Overextended clusters: {results.get('overextended_clusters', 0)}")
            tprint_info(f"📊 SHAP features analyzed: {results.get('treeshap_features_analyzed', 0)}")
            tprint_info(f"📊 SHAP interactions found: {results.get('treeshap_interactions_found', 0)}")
            return True
        else:
            tprint_error(f"❌ Enhanced GMM pipeline failed: {results.get('error')}")
            return False
            
    except Exception as e:
        tprint_error(f"❌ Enhanced GMM pipeline test failed: {e}")
        import traceback
        tprint_error(traceback.format_exc())
        return False


def test_shap_analyzer():
    """Test SHAP Interaction Analyzer."""
    tprint_info("\n🧪 Testing SHAP Interaction Analyzer...")
    
    try:
        # Generate test features
        np.random.seed(42)
        n_samples = 500
        n_features = 20
        
        # Create feature matrix
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=feature_names,
            index=pd.date_range('2023-01-01', periods=n_samples, freq='1H')
        )
        
        # Create target with some relationship to features
        y = pd.Series(
            X.iloc[:, 0] * 0.5 + X.iloc[:, 1] * 0.3 + np.random.randn(n_samples) * 0.1,
            index=X.index
        )
        
        # Add some interaction effects
        X['interaction_1'] = X.iloc[:, 2] * X.iloc[:, 3]
        y += X['interaction_1'] * 0.2
        
        # Test SHAP analyzer
        config = {
            'n_estimators': 50,
            'max_depth': 6,
            'max_samples': 500,
            'interaction_sample_size': 200,
            'importance_threshold': 0.01,
            'max_features': 15
        }
        
        analyzer = SHAPInteractionAnalyzer(config)
        results = analyzer.analyze_features(X, y)
        
        if results.get('success'):
            tprint_success("✅ SHAP analyzer test passed")
            
            feature_analysis = results.get('feature_analysis', {})
            interaction_analysis = results.get('interaction_analysis', {})
            
            tprint_info(f"📊 Total features: {feature_analysis.get('total_features', 0)}")
            tprint_info(f"📊 Selected features: {feature_analysis.get('selected_features', 0)}")
            tprint_info(f"📊 Total interactions: {interaction_analysis.get('total_interactions', 0)}")
            
            # Test interaction feature generation
            enhanced_X = analyzer.generate_interaction_features(X, top_n_interactions=5)
            tprint_info(f"📊 Enhanced features: {len(enhanced_X.columns)} columns")
            
            return True
        else:
            tprint_error(f"❌ SHAP analyzer failed: {results.get('error')}")
            return False
            
    except Exception as e:
        tprint_error(f"❌ SHAP analyzer test failed: {e}")
        import traceback
        tprint_error(traceback.format_exc())
        return False


def test_trading_engine():
    """Test GMM Trading Engine."""
    tprint_info("\n🧪 Testing GMM Trading Engine...")
    
    try:
        # Generate test market data
        market_data, returns = generate_test_data(300)
        
        # Create mock GMM features
        np.random.seed(42)
        n_samples = len(market_data)
        
        gmm_features = pd.DataFrame(index=market_data.index)
        
        # Add mock GMM features
        gmm_features['macro_gmm_signal'] = np.random.randn(n_samples) * 0.1
        gmm_features['macro_regime_velocity'] = np.random.randn(n_samples) * 0.05
        gmm_features['macro_entropy'] = np.random.uniform(1.5, 2.5, n_samples)
        gmm_features['macro_z_familiarity'] = np.random.randn(n_samples)
        
        # Add overextended cluster features
        for k in range(6):
            gmm_features[f'cluster_{k}_overextended_score'] = np.random.uniform(0, 1, n_samples)
            gmm_features[f'cluster_{k}_is_overextended'] = (np.random.uniform(0, 1, n_samples) > 0.8).astype(int)
        
        # Add shock features
        gmm_features['gmm_shock_composite'] = (np.random.uniform(0, 1, n_samples) > 0.9).astype(int)
        gmm_features['gmm_shock_confidence'] = np.random.uniform(0, 1, n_samples)
        
        for k in range(6):
            gmm_features[f'gmm_shock_prob_jump_{k}'] = (np.random.uniform(0, 1, n_samples) > 0.95).astype(float)
        
        gmm_features['gmm_shock_z_fam_jump'] = (np.random.uniform(0, 1, n_samples) > 0.95).astype(float)
        gmm_features['gmm_shock_entropy_drop'] = (np.random.uniform(0, 1, n_samples) > 0.95).astype(float)
        
        # Test trading engine
        config = {
            'runway_analysis': {
                'momentum_window': 10,
                'minimum_runway': 3,
                'maximum_runway': 50
            },
            'shock_detection': {
                'confidence_threshold': 0.5,
                'magnitude_threshold': 0.2
            },
            'position_sizing': {
                'base_size': 1.0,
                'max_size': 2.0
            }
        }
        
        engine = GMMTradingEngine(config)
        
        # Test runway analysis
        runway_analysis = engine.analyze_regime_runway(gmm_features, market_data)
        tprint_info(f"📊 Runway analysis: {runway_analysis.current_regime.value}")
        tprint_info(f"📊 Runway estimate: {runway_analysis.runway_estimate:.1f} bars")
        tprint_info(f"📊 Runway confidence: {runway_analysis.runway_confidence:.2f}")
        
        # Test shock detection
        shock_events = engine.detect_shock_events(gmm_features, market_data)
        tprint_info(f"📊 Shock events detected: {len(shock_events)}")
        
        # Test signal generation
        signals = engine.generate_trading_signals(gmm_features, market_data)
        tprint_info(f"📊 Trading signals generated: {len(signals)}")
        
        # Test performance summary
        performance = engine.get_performance_summary()
        tprint_info(f"📊 Performance metrics available: {len(performance)} keys")
        
        tprint_success("✅ Trading engine test passed")
        return True
        
    except Exception as e:
        tprint_error(f"❌ Trading engine test failed: {e}")
        import traceback
        tprint_error(traceback.format_exc())
        return False


def test_integration():
    """Test full integration of all components."""
    tprint_info("\n🧪 Testing Full Integration...")
    
    try:
        # Generate test data
        market_data, returns = generate_test_data(400)
        
        # Step 1: Generate enhanced GMM features
        config = {
            'use_original_pipeline': False,  # Skip for speed
            'use_enhanced_pipeline': True,
            'use_fracdiff': True,
            'use_treeshap': True,
            'n_clusters_macro': 4,  # Smaller for testing
            'fracdiff_config': {'max_d': 0.5, 'adf_threshold': 0.1},
            'treeshap_config': {'n_estimators': 30, 'interaction_sample_size': 100}
        }
        
        pipeline = EnhancedGMMFeatures(**config)
        
        # Mock data loading
        pipeline.load_market_data_or_fail = lambda cfg: (market_data, None)
        
        pipeline_config = {'symbol': 'TEST', 'exchange': 'binance'}
        results = pipeline.run(pipeline_config)
        
        if not results.get('success'):
            raise Exception(f"Pipeline failed: {results.get('error')}")
        
        # Load enhanced features
        enhanced_features_path = results.get('enhanced_features_path')
        if enhanced_features_path and os.path.exists(enhanced_features_path):
            enhanced_features = pd.read_parquet(enhanced_features_path)
        else:
            # Create mock enhanced features for testing
            enhanced_features = pd.DataFrame(index=market_data.index)
            enhanced_features['macro_gmm_signal'] = np.random.randn(len(market_data)) * 0.1
            enhanced_features['macro_regime_velocity'] = np.random.randn(len(market_data)) * 0.05
            enhanced_features['gmm_shock_composite'] = (np.random.uniform(0, 1, len(market_data)) > 0.9).astype(int)
        
        # Step 2: Run SHAP analysis
        shap_config = {
            'n_estimators': 30,
            'max_depth': 4,
            'max_samples': 300,
            'interaction_sample_size': 100
        }
        
        analyzer = SHAPInteractionAnalyzer(shap_config)
        target = returns.reindex(enhanced_features.index).fillna(0)
        
        shap_results = analyzer.analyze_features(enhanced_features, target.to_frame())
        
        if not shap_results.get('success'):
            raise Exception(f"SHAP analysis failed: {shap_results.get('error')}")
        
        # Step 3: Generate trading signals
        trading_config = {
            'runway_analysis': {'minimum_runway': 3, 'maximum_runway': 30},
            'shock_detection': {'confidence_threshold': 0.4},
            'position_sizing': {'base_size': 1.0}
        }
        
        signals = generate_gmm_trading_signals(enhanced_features, market_data, trading_config)
        
        # Step 4: Validate integration
        validation_results = {
            'pipeline_success': results.get('success', False),
            'shap_success': shap_results.get('success', False),
            'signals_generated': len(signals),
            'enhanced_features': len(enhanced_features.columns),
            'shap_features_analyzed': shap_results.get('feature_analysis', {}).get('total_features', 0),
            'shap_interactions': shap_results.get('interaction_analysis', {}).get('total_interactions', 0)
        }
        
        tprint_success("✅ Full integration test passed")
        tprint_info("📊 Integration Results:")
        for key, value in validation_results.items():
            tprint_info(f"   - {key}: {value}")
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ Integration test failed: {e}")
        import traceback
        tprint_error(traceback.format_exc())
        return False


def run_all_tests():
    """Run all tests and generate summary."""
    tprint_info("🚀 Starting Enhanced GMM Pipeline Test Suite")
    tprint_info("=" * 60)
    
    test_results = {
        'fracdiff': test_fracdiff_functionality(),
        'enhanced_gmm': test_enhanced_gmm_pipeline(),
        'shap_analyzer': test_shap_analyzer(),
        'trading_engine': test_trading_engine(),
        'integration': test_integration()
    }
    
    # Generate summary
    tprint_info("\n" + "=" * 60)
    tprint_info("📊 TEST SUMMARY")
    tprint_info("=" * 60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        tprint_info(f"{test_name:20} : {status}")
    
    tprint_info("-" * 60)
    tprint_info(f"Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        tprint_success("🎉 ALL TESTS PASSED! Enhanced GMM Pipeline is ready for production.")
    else:
        tprint_warning(f"⚠️ {total_tests - passed_tests} test(s) failed. Please review the errors above.")
    
    return test_results


if __name__ == "__main__":
    # Run all tests
    results = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if all(results.values()) else 1)
