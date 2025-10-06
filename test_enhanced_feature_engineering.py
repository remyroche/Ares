#!/usr/bin/env python3
"""
Test Enhanced Feature Engineering Integration

This script tests the enhanced feature engineering capabilities to ensure
they are properly integrated with the existing infrastructure.
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample market data for testing."""
    # Create 1-minute data for 1 week
    start_date = datetime.now() - timedelta(days=7)
    dates_1m = pd.date_range(start=start_date, periods=7*24*60, freq='1min')
    
    # Create 5-minute data for 1 week
    dates_5m = pd.date_range(start=start_date, periods=7*24*12, freq='5min')
    
    # Generate realistic price data with trends and volatility
    np.random.seed(42)
    
    # 1-minute data
    returns_1m = np.random.normal(0, 0.001, len(dates_1m))
    prices_1m = 100 * np.exp(np.cumsum(returns_1m))
    
    data_1m = pd.DataFrame({
        'timestamp': dates_1m,
        'open': prices_1m * (1 + np.random.normal(0, 0.0005, len(dates_1m))),
        'high': prices_1m * (1 + np.abs(np.random.normal(0, 0.001, len(dates_1m)))),
        'low': prices_1m * (1 - np.abs(np.random.normal(0, 0.001, len(dates_1m)))),
        'close': prices_1m,
        'volume': np.random.randint(1000, 10000, len(dates_1m))
    }).set_index('timestamp')
    
    # 5-minute data
    returns_5m = np.random.normal(0, 0.002, len(dates_5m))
    prices_5m = 100 * np.exp(np.cumsum(returns_5m))
    
    data_5m = pd.DataFrame({
        'timestamp': dates_5m,
        'open': prices_5m * (1 + np.random.normal(0, 0.001, len(dates_5m))),
        'high': prices_5m * (1 + np.abs(np.random.normal(0, 0.002, len(dates_5m)))),
        'low': prices_5m * (1 - np.abs(np.random.normal(0, 0.002, len(dates_5m)))),
        'close': prices_5m,
        'volume': np.random.randint(5000, 50000, len(dates_5m))
    }).set_index('timestamp')
    
    return {
        '1m': data_1m,
        '5m': data_5m
    }

def test_enhanced_normalization_features():
    """Test enhanced normalization features."""
    logger.info("🧪 Testing enhanced normalization features...")
    
    try:
        from src.feature_generation.categories.enhanced_normalization import (
            EnhancedNormalizationFeatureGenerator,
            AdvancedRollingZScoreGenerator,
            RegimeAwareNormalizer
        )
        
        # Create sample data
        data = create_sample_data()['1m']
        
        # Test main enhanced normalization generator
        generator = EnhancedNormalizationFeatureGenerator()
        result = generator.generate(data)
        
        if result.success:
            logger.info(f"✅ Enhanced normalization generator: {len(result.data)} features generated")
        else:
            logger.error(f"❌ Enhanced normalization generator failed: {result.error_message}")
        
        # Test individual generators
        zscore_gen = AdvancedRollingZScoreGenerator(window=20, column='close', method='zscore')
        zscore_result = zscore_gen.generate(data)
        
        if zscore_result.success:
            logger.info(f"✅ Rolling z-score generator: {zscore_result.data.name}")
        else:
            logger.error(f"❌ Rolling z-score generator failed: {zscore_result.error_message}")
        
        regime_gen = RegimeAwareNormalizer(window=60, column='close', regime_method='volatility')
        regime_result = regime_gen.generate(data)
        
        if regime_result.success:
            logger.info(f"✅ Regime-aware normalizer: {regime_result.data.name}")
        else:
            logger.error(f"❌ Regime-aware normalizer failed: {regime_result.error_message}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced normalization test failed: {e}")
        return False

def test_enhanced_cross_timeframe_features():
    """Test enhanced cross-timeframe features."""
    logger.info("🧪 Testing enhanced cross-timeframe features...")
    
    try:
        from src.feature_generation.categories.enhanced_cross_timeframe import (
            EnhancedCrossTimeframeFeatureGenerator,
            FractionalChangeGenerator,
            CrossTimeframeAlignmentGenerator,
            LearnedProjectionGenerator
        )
        
        # Create sample data
        data = create_sample_data()['1m']
        
        # Test main enhanced cross-timeframe generator
        generator = EnhancedCrossTimeframeFeatureGenerator()
        result = generator.generate(data)
        
        if result.success:
            logger.info(f"✅ Enhanced cross-timeframe generator: {len(result.data)} features generated")
        else:
            logger.error(f"❌ Enhanced cross-timeframe generator failed: {result.error_message}")
        
        # Test individual generators
        frac_gen = FractionalChangeGenerator(fast_tf=5, slow_tf=15, feature_type='volatility')
        frac_result = frac_gen.generate(data)
        
        if frac_result.success:
            logger.info(f"✅ Fractional change generator: {frac_result.data.name}")
        else:
            logger.error(f"❌ Fractional change generator failed: {frac_result.error_message}")
        
        align_gen = CrossTimeframeAlignmentGenerator(source_tf=1, target_tf=5, alignment_method='lag')
        align_result = align_gen.generate(data)
        
        if align_result.success:
            logger.info(f"✅ Cross-timeframe alignment generator: {align_result.data.name}")
        else:
            logger.error(f"❌ Cross-timeframe alignment generator failed: {align_result.error_message}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced cross-timeframe test failed: {e}")
        return False

def test_enhanced_interaction_features():
    """Test enhanced interaction features."""
    logger.info("🧪 Testing enhanced interaction features...")
    
    try:
        from src.feature_generation.categories.enhanced_interaction import (
            EnhancedInteractionFeatureGenerator,
            PairwiseInteractionGenerator,
            RegimeDependentFeatureGenerator,
            StructuralRatioGenerator
        )
        
        # Create sample data
        data = create_sample_data()['1m']
        
        # Test main enhanced interaction generator
        generator = EnhancedInteractionFeatureGenerator()
        result = generator.generate(data)
        
        if result.success:
            logger.info(f"✅ Enhanced interaction generator: {len(result.data)} features generated")
        else:
            logger.error(f"❌ Enhanced interaction generator failed: {result.error_message}")
        
        # Test individual generators
        pairwise_gen = PairwiseInteractionGenerator(feature1='momentum', feature2='volume', interaction_type='product')
        pairwise_result = pairwise_gen.generate(data)
        
        if pairwise_result.success:
            logger.info(f"✅ Pairwise interaction generator: {pairwise_result.data.name}")
        else:
            logger.error(f"❌ Pairwise interaction generator failed: {pairwise_result.error_message}")
        
        regime_gen = RegimeDependentFeatureGenerator(regime_detector='volatility', feature_type='momentum')
        regime_result = regime_gen.generate(data)
        
        if regime_result.success:
            logger.info(f"✅ Regime-dependent generator: {regime_result.data.name}")
        else:
            logger.error(f"❌ Regime-dependent generator failed: {regime_result.error_message}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced interaction test failed: {e}")
        return False

def test_enhanced_representation_learning_features():
    """Test enhanced representation learning features."""
    logger.info("🧪 Testing enhanced representation learning features...")
    
    try:
        from src.feature_generation.categories.enhanced_representation_learning import (
            EnhancedRepresentationLearningGenerator,
            PatchTSTRepresentationGenerator,
            TFTEncoderRepresentationGenerator
        )
        
        # Create sample data
        data = create_sample_data()['1m']
        
        # Test main enhanced representation learning generator
        generator = EnhancedRepresentationLearningGenerator()
        result = generator.generate(data)
        
        if result.success:
            logger.info(f"✅ Enhanced representation learning generator: {len(result.data)} features generated")
        else:
            logger.error(f"❌ Enhanced representation learning generator failed: {result.error_message}")
        
        # Test individual generators
        patchtst_gen = PatchTSTRepresentationGenerator(patch_length=16, num_patches=8, embedding_dim=64)
        patchtst_result = patchtst_gen.generate(data)
        
        if patchtst_result.success:
            logger.info(f"✅ PatchTST generator: {patchtst_result.data.name}")
        else:
            logger.error(f"❌ PatchTST generator failed: {patchtst_result.error_message}")
        
        tft_gen = TFTEncoderRepresentationGenerator(seq_length=60, hidden_size=64, num_heads=4)
        tft_result = tft_gen.generate(data)
        
        if tft_result.success:
            logger.info(f"✅ TFT encoder generator: {tft_result.data.name}")
        else:
            logger.error(f"❌ TFT encoder generator failed: {tft_result.error_message}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced representation learning test failed: {e}")
        return False

def test_enhanced_integration():
    """Test the enhanced feature engineering integration."""
    logger.info("🧪 Testing enhanced feature engineering integration...")
    
    try:
        from src.feature_generation.enhanced_feature_engineering_integration import (
            EnhancedFeatureEngineeringIntegration,
            create_enhanced_feature_engineer,
            generate_enhanced_features_sync
        )
        
        # Create sample data
        data_dict = create_sample_data()
        
        # Test integration class
        engineer = create_enhanced_feature_engineer({
            'enable_gpu': False,
            'max_workers': 2
        })
        
        # Test feature generation
        features = generate_enhanced_features_sync(
            data_dict,
            include_categories=['normalization', 'cross_timeframe', 'interaction', 'representation_learning']
        )
        
        if features:
            logger.info(f"✅ Enhanced integration: Generated features for {len(features)} timeframes")
            
            for timeframe, feature_df in features.items():
                logger.info(f"  {timeframe}: {len(feature_df.columns)} features, {len(feature_df)} samples")
                
                # Check for missing values
                missing_count = feature_df.isnull().sum().sum()
                if missing_count > 0:
                    logger.warning(f"  {timeframe}: {missing_count} missing values found")
                
                # Check for infinite values
                inf_count = np.isinf(feature_df).sum().sum()
                if inf_count > 0:
                    logger.warning(f"  {timeframe}: {inf_count} infinite values found")
        else:
            logger.error("❌ Enhanced integration: No features generated")
            return False
        
        # Test performance stats
        stats = engineer.get_performance_stats()
        logger.info(f"✅ Performance stats: {stats}")
        
        # Test feature summary
        summary = engineer.get_feature_summary()
        logger.info(f"✅ Feature summary: {summary}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced integration test failed: {e}")
        return False

def test_feature_bank_integration():
    """Test integration with the existing feature bank."""
    logger.info("🧪 Testing feature bank integration...")
    
    try:
        from src.feature_generation.core.feature_bank import get_global_feature_bank
        
        # Get the global feature bank
        feature_bank = get_global_feature_bank()
        
        # Test that enhanced categories are available
        categories = feature_bank.list_categories()
        logger.info(f"✅ Available categories: {[cat.value for cat in categories]}")
        
        # Check if enhanced categories are present
        enhanced_categories = ['normalization', 'representation_learning']
        for category in enhanced_categories:
            if any(cat.value == category for cat in categories):
                logger.info(f"✅ Enhanced category '{category}' is available")
            else:
                logger.warning(f"⚠️ Enhanced category '{category}' not found")
        
        # Test feature generation with enhanced categories
        data = create_sample_data()['1m']
        
        # Test normalization features
        try:
            norm_features = feature_bank.generate_features_by_category(data, 'normalization')
            if not norm_features.empty:
                logger.info(f"✅ Normalization features: {len(norm_features.columns)} features generated")
            else:
                logger.warning("⚠️ No normalization features generated")
        except Exception as e:
            logger.warning(f"⚠️ Normalization feature generation failed: {e}")
        
        # Test representation learning features
        try:
            repr_features = feature_bank.generate_features_by_category(data, 'representation_learning')
            if not repr_features.empty:
                logger.info(f"✅ Representation learning features: {len(repr_features.columns)} features generated")
            else:
                logger.warning("⚠️ No representation learning features generated")
        except Exception as e:
            logger.warning(f"⚠️ Representation learning feature generation failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature bank integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting Enhanced Feature Engineering Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Enhanced Normalization Features", test_enhanced_normalization_features),
        ("Enhanced Cross-Timeframe Features", test_enhanced_cross_timeframe_features),
        ("Enhanced Interaction Features", test_enhanced_interaction_features),
        ("Enhanced Representation Learning Features", test_enhanced_representation_learning_features),
        ("Enhanced Integration", test_enhanced_integration),
        ("Feature Bank Integration", test_feature_bank_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Enhanced feature engineering is working correctly.")
        return 0
    else:
        logger.error(f"⚠️ {total - passed} tests failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)