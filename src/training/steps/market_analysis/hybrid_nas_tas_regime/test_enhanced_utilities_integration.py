"""
Test script to verify that enhanced utility integrations are properly wired and used.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_utility_integration():
    """Test that enhanced utility integrations are properly wired and used."""
    try:
        # Import the enhanced hybrid orchestrator
        from .enhanced_hybrid_orchestrator import EnhancedHybridOrchestrator
        from .config.hybrid_regime_config import HybridRegimeConfig
        
        logger.info("🧪 Testing Enhanced Utility Integration...")
        
        # Create a simple configuration
        config = HybridRegimeConfig(
            symbol="BTCUSDT",
            timeframe="15m",
            start_date="2023-01-01",
            end_date="2023-12-31",
            n_regimes=3,
            combination_strategy="weighted_average"
        )
        
        # Initialize the orchestrator
        orchestrator = EnhancedHybridOrchestrator(config)
        
        # Check that enhanced utilities are initialized
        assert hasattr(orchestrator, 'utility_integration'), "Utility integration not found"
        assert hasattr(orchestrator, 'data_integration'), "Data integration not found"
        assert hasattr(orchestrator, 'ml_integration'), "ML integration not found"
        
        logger.info("✅ Enhanced utility integrations are properly initialized")
        
        # Test utility integration methods
        logger.info("🔧 Testing utility integration methods...")
        
        # Test data operations
        test_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        # Test data validation
        is_valid = orchestrator.utility_integration.validate_dataframe_columns(
            test_data, ['open', 'high', 'low', 'close', 'volume']
        )
        assert is_valid, "Data validation failed"
        logger.info("✅ Data validation working")
        
        # Test data quality metrics
        quality_metrics = orchestrator.utility_integration.calculate_data_quality_metrics(test_data)
        assert isinstance(quality_metrics, dict), "Quality metrics not returned as dict"
        logger.info("✅ Data quality metrics working")
        
        # Test math operations
        result = orchestrator.utility_integration.safe_divide(10, 2, default=0.0)
        assert result == 5.0, f"Safe divide failed: {result}"
        logger.info("✅ Safe math operations working")
        
        # Test data integration
        logger.info("📊 Testing data integration methods...")
        
        # Test data processing
        processed_data = orchestrator.data_integration.process_market_data(test_data, "BTCUSDT", "15m")
        assert isinstance(processed_data, pd.DataFrame), "Data processing failed"
        logger.info("✅ Data processing working")
        
        # Test data quality calculation
        quality_report = orchestrator.data_integration.calculate_data_quality_metrics(processed_data)
        assert isinstance(quality_report, dict), "Data quality calculation failed"
        logger.info("✅ Data quality calculation working")
        
        # Test ML integration
        logger.info("🤖 Testing ML integration methods...")
        
        # Test feature selection
        X = test_data.select_dtypes(include=[np.number]).values
        y = np.random.randint(0, 3, len(X))
        
        X_selected, selected_features = orchestrator.ml_integration.select_features(
            X, y, method="mutual_info", n_features=min(3, X.shape[1])
        )
        assert X_selected.shape[1] <= X.shape[1], "Feature selection failed"
        logger.info("✅ Feature selection working")
        
        # Test cross-validation
        from sklearn.ensemble import RandomForestClassifier
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        cv_results = orchestrator.ml_integration.cross_validate_model(
            estimator, X_selected, y, cv=3, scoring="accuracy"
        )
        assert isinstance(cv_results, dict), "Cross-validation failed"
        logger.info("✅ Cross-validation working")
        
        # Test regime detection
        regime_results = orchestrator.ml_integration.detect_regimes_hmm(
            test_data, n_regimes=2, features=['open', 'high', 'low', 'close']
        )
        assert isinstance(regime_results, dict), "Regime detection failed"
        logger.info("✅ Regime detection working")
        
        # Test enhanced orchestrator methods
        logger.info("🎯 Testing enhanced orchestrator methods...")
        
        # Test enhanced preprocessing
        processed_data_enhanced = orchestrator._preprocess_market_data_enhanced(test_data)
        assert isinstance(processed_data_enhanced, pd.DataFrame), "Enhanced preprocessing failed"
        logger.info("✅ Enhanced preprocessing working")
        
        # Test enhanced TAS analysis
        tas_result = orchestrator._run_tas_analysis_enhanced(processed_data_enhanced)
        assert isinstance(tas_result, dict), "Enhanced TAS analysis failed"
        assert 'enhanced_analysis' in tas_result, "Enhanced TAS analysis not marked as enhanced"
        logger.info("✅ Enhanced TAS analysis working")
        
        # Test enhanced NAS analysis
        nas_result = orchestrator._run_nas_analysis_enhanced(processed_data_enhanced)
        assert isinstance(nas_result, dict), "Enhanced NAS analysis failed"
        assert 'enhanced_analysis' in nas_result, "Enhanced NAS analysis not marked as enhanced"
        logger.info("✅ Enhanced NAS analysis working")
        
        # Test enhanced output analysis
        hybrid_analysis = orchestrator._analyze_tas_nas_outputs_enhanced(tas_result, nas_result, processed_data_enhanced)
        assert isinstance(hybrid_analysis, dict), "Enhanced output analysis failed"
        assert 'enhanced_analysis' in hybrid_analysis, "Enhanced output analysis not marked as enhanced"
        logger.info("✅ Enhanced output analysis working")
        
        # Test enhanced clustering
        hybrid_regimes = orchestrator._create_hybrid_regime_clusters_enhanced(
            tas_result, nas_result, hybrid_analysis, processed_data_enhanced
        )
        assert isinstance(hybrid_regimes, dict), "Enhanced clustering failed"
        assert 'enhanced_clustering' in hybrid_regimes, "Enhanced clustering not marked as enhanced"
        logger.info("✅ Enhanced clustering working")
        
        # Test enhanced cross-validation
        cv_results_enhanced = orchestrator._perform_hybrid_cross_validation_enhanced(
            hybrid_regimes, processed_data_enhanced
        )
        assert isinstance(cv_results_enhanced, dict), "Enhanced cross-validation failed"
        assert 'enhanced_cv' in cv_results_enhanced, "Enhanced cross-validation not marked as enhanced"
        logger.info("✅ Enhanced cross-validation working")
        
        # Test enhanced weight optimization
        weight_optimization = orchestrator._optimize_ensemble_weights_enhanced(0.7, 0.8, 0.75)
        assert isinstance(weight_optimization, dict), "Enhanced weight optimization failed"
        assert 'enhanced_optimization' in weight_optimization, "Enhanced weight optimization not marked as enhanced"
        logger.info("✅ Enhanced weight optimization working")
        
        # Test full analysis pipeline
        logger.info("🚀 Testing full enhanced analysis pipeline...")
        
        # Create sample market data
        dates = pd.date_range('2023-01-01', periods=100, freq='15T')
        market_data = pd.DataFrame({
            'timestamp': dates,
            'open': 100 + np.cumsum(np.random.randn(100) * 0.01),
            'high': 100 + np.cumsum(np.random.randn(100) * 0.01) + np.random.rand(100) * 2,
            'low': 100 + np.cumsum(np.random.randn(100) * 0.01) - np.random.rand(100) * 2,
            'close': 100 + np.cumsum(np.random.randn(100) * 0.01),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Ensure high >= low
        market_data['high'] = np.maximum(market_data['high'], market_data['low'])
        market_data['low'] = np.minimum(market_data['high'], market_data['low'])
        
        # Run full analysis
        result = orchestrator.analyze_market_regimes(market_data, enable_multi_timeframe=False)
        
        # Verify result structure
        assert hasattr(result, 'regime_predictions'), "Result missing regime_predictions"
        assert hasattr(result, 'regime_probabilities'), "Result missing regime_probabilities"
        assert hasattr(result, 'metadata'), "Result missing metadata"
        
        # Check metadata for enhanced utilities usage
        metadata = result.metadata
        assert metadata.get('enhanced_utilities_used', False), "Enhanced utilities not marked as used"
        assert metadata.get('utility_integration_used', False), "Utility integration not marked as used"
        assert metadata.get('data_integration_used', False), "Data integration not marked as used"
        assert metadata.get('ml_integration_used', False), "ML integration not marked as used"
        
        logger.info("✅ Full enhanced analysis pipeline working")
        
        # Test integration status
        logger.info("📊 Testing integration status...")
        
        utility_status = orchestrator.utility_integration.get_integration_status()
        data_status = orchestrator.data_integration.get_integration_status()
        ml_status = orchestrator.ml_integration.get_integration_status()
        
        assert isinstance(utility_status, dict), "Utility integration status not returned as dict"
        assert isinstance(data_status, dict), "Data integration status not returned as dict"
        assert isinstance(ml_status, dict), "ML integration status not returned as dict"
        
        logger.info("✅ Integration status reporting working")
        
        # Test available utilities
        available_utilities = orchestrator.utility_integration.get_available_utilities()
        available_data_utilities = orchestrator.data_integration.get_available_data_utilities()
        available_ml_utilities = orchestrator.ml_integration.get_available_ml_utilities()
        
        assert isinstance(available_utilities, list), "Available utilities not returned as list"
        assert isinstance(available_data_utilities, list), "Available data utilities not returned as list"
        assert isinstance(available_ml_utilities, list), "Available ML utilities not returned as list"
        
        logger.info(f"✅ Available utilities: {len(available_utilities)} utility, {len(available_data_utilities)} data, {len(available_ml_utilities)} ML")
        
        # Test cleanup
        logger.info("🧹 Testing resource cleanup...")
        
        utility_cleanup = orchestrator.utility_integration.cleanup_resources()
        data_cleanup = orchestrator.data_integration.cleanup_data_resources()
        ml_cleanup = orchestrator.ml_integration.cleanup_ml_resources()
        
        assert utility_cleanup, "Utility cleanup failed"
        assert data_cleanup, "Data cleanup failed"
        assert ml_cleanup, "ML cleanup failed"
        
        logger.info("✅ Resource cleanup working")
        
        logger.info("🎉 All enhanced utility integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced utility integration test failed: {e}")
        raise


def main():
    """Main test function."""
    try:
        success = test_enhanced_utility_integration()
        if success:
            print("\n" + "="*80)
            print("🎉 ENHANCED UTILITY INTEGRATION TESTS PASSED!")
            print("✅ All enhanced utilities are properly wired and used")
            print("✅ Utility integration is working")
            print("✅ Data integration is working")
            print("✅ ML integration is working")
            print("✅ Enhanced orchestrator methods are working")
            print("✅ Full analysis pipeline is working")
            print("="*80)
        else:
            print("\n❌ Tests failed")
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        raise


if __name__ == "__main__":
    main()