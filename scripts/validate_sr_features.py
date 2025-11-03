#!/usr/bin/env python3
"""
SR Feature Validation Script

Quick validation script to verify that new features are working correctly.
Tests feature extraction, counts features, and validates consistency.

Usage:
    python scripts/validate_sr_features.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger

logger = system_logger.getChild('FeatureValidation')


def validate_feature_extraction():
    """Test feature extraction with dummy data."""
    logger.info("="*80)
    logger.info("🧪 TESTING FEATURE EXTRACTION")
    logger.info("="*80)
    
    try:
        from src.tactician.sr_levels.ml_quality.sr_quality_data_collector import SRQualityDataCollector
        
        collector = SRQualityDataCollector()
        
        # Create dummy SR level
        class DummyLevel:
            def __init__(self):
                self.price = 50000.0
                self.strength = 0.75
                self.prominence_score = 0.8
                self.width_score = 10.0
                self.volume_confirmation_score = 0.6
                self.consistency_score = 0.7
                self.touch_count = 5
                self.age_bars = 50
                self.failure_count = 1
                self.avg_bounce_ratio = 0.02
                self.max_bounce_ratio = 0.05
                self.median_bounce_ratio = 0.015
                self.bounce_consistency = 0.8
                self.volume_weighted_bounce = 0.03
                self.strong_bounce_count = 3
                self.avg_touch_volume_ratio = 1.5
                self.approach_velocity = 0.01
                self.rejection_velocity = 0.02
                self.cluster_density = 0.3
                self.recency_weighted_strength = 0.8
                self.dwell_time = 5
                self.multi_tf_score = 0.6
                self.confirmation_count = 2
                self.type = 'support'
                self.method = 'fractal'
                self.metadata = {}
        
        # Create dummy market data
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
        data = pd.DataFrame({
            'open': np.random.randn(200).cumsum() + 50000,
            'high': np.random.randn(200).cumsum() + 50050,
            'low': np.random.randn(200).cumsum() + 49950,
            'close': np.random.randn(200).cumsum() + 50000,
            'volume': np.random.rand(200) * 1000000
        }, index=dates)
        
        # Extract features
        level = DummyLevel()
        features = collector._extract_all_features(level, data)
        
        # Count features
        feature_names = [k for k in features.keys() if k.startswith('feature_')]
        
        logger.info(f"✅ Feature extraction successful!")
        logger.info(f"   Total features extracted: {len(feature_names)}")
        logger.info(f"   Feature types:")
        
        # Categorize features
        categories = {
            'Basic SR': [],
            'Bounce': [],
            'Volume': [],
            'Temporal': [],
            'Market Context': [],
            'Interaction': [],
            'Regime': [],
            'Statistical': [],
            'Method': [],
            'Other': []
        }
        
        for fname in feature_names:
            fname_lower = fname.lower()
            if any(kw in fname_lower for kw in ['strength', 'prominence', 'width', 'consistency', 'touch']):
                categories['Basic SR'].append(fname)
            elif 'bounce' in fname_lower:
                categories['Bounce'].append(fname)
            elif 'volume' in fname_lower:
                categories['Volume'].append(fname)
            elif any(kw in fname_lower for kw in ['age', 'recency', 'time', 'decay', 'frequency']):
                categories['Temporal'].append(fname)
            elif any(kw in fname_lower for kw in ['market', 'regime', 'trend', 'volatility']):
                categories['Market Context'].append(fname)
            elif '_x_' in fname_lower or 'composite' in fname_lower:
                categories['Interaction'].append(fname)
            elif 'method' in fname_lower or 'confluence' in fname_lower:
                categories['Method'].append(fname)
            elif any(kw in fname_lower for kw in ['spike', 'reaction', 'profile', 'quality']):
                categories['Statistical'].append(fname)
            else:
                categories['Other'].append(fname)
        
        for category, feats in categories.items():
            if feats:
                logger.info(f"      {category}: {len(feats)} features")
        
        # Validate new features exist
        expected_new_features = [
            'feature_touch_frequency',
            'feature_avg_time_between_touches',
            'feature_regime_volatility',
            'feature_regime_trend_strength',
            'feature_distance_to_price_atr',
            'feature_volume_spike_ratio',
            'feature_price_reaction_strength',
            'feature_touches_x_recency',
            'feature_quality_composite',
            'feature_strength_percentile',
            'feature_is_top_10_pct',
            'feature_quality_tier'
        ]
        
        missing_features = [f for f in expected_new_features if f not in feature_names]
        
        if missing_features:
            logger.warning(f"\n⚠️  Missing expected features:")
            for f in missing_features:
                logger.warning(f"      - {f}")
        else:
            logger.info(f"\n✅ All expected new features present!")
        
        # Check for NaN values
        nan_features = [k for k, v in features.items() if pd.isna(v)]
        if nan_features:
            logger.warning(f"\n⚠️  Features with NaN values: {len(nan_features)}")
            for f in nan_features[:5]:  # Show first 5
                logger.warning(f"      - {f}")
        else:
            logger.info(f"✅ No NaN values in features")
        
        return True, len(feature_names)
        
    except Exception as e:
        logger.error(f"❌ Feature extraction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, 0


def validate_detection_features():
    """Test detection feature extraction."""
    logger.info("\n" + "="*80)
    logger.info("🧪 TESTING DETECTION FEATURE EXTRACTION")
    logger.info("="*80)
    
    try:
        from src.tactician.sr_levels.enhanced_sr_detection import EnhancedSRDetector, SRLevel
        
        detector = EnhancedSRDetector({'min_touches': 2})
        
        # Create dummy data
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
        data = pd.DataFrame({
            'open': np.random.randn(200).cumsum() + 50000,
            'high': np.random.randn(200).cumsum() + 50050,
            'low': np.random.randn(200).cumsum() + 49950,
            'close': np.random.randn(200).cumsum() + 50000,
            'volume': np.random.rand(200) * 1000000
        }, index=dates)
        
        # Create dummy level using correct SRLevel dataclass constructor
        level = SRLevel(
            price=50000.0,
            strength=0.75,
            type='support',
            touch_count=5,
            first_touch_time=dates[0],
            last_touch_time=dates[-1],
            age_bars=50,
            avg_bounce_ratio=0.02,
            max_bounce_ratio=0.05,
            median_bounce_ratio=0.015,
            bounce_consistency=0.8,
            touches=[],
            method='fractal',
            metadata={}
        )
        level.prominence_score = 0.8
        level.width_score = 10.0
        level.volume_confirmation_score = 0.6
        level.consistency_score = 0.7
        level.failure_count = 1
        
        # Extract features
        features = detector._extract_all_ml_features(level, data)
        
        feature_names = [k for k in features.keys() if k.startswith('feature_')]
        
        logger.info(f"✅ Detection feature extraction successful!")
        logger.info(f"   Total features: {len(feature_names)}")
        
        # Check for NaN
        nan_features = [k for k, v in features.items() if pd.isna(v) or np.isinf(v)]
        if nan_features:
            logger.warning(f"⚠️  Features with NaN/Inf: {len(nan_features)}")
        else:
            logger.info(f"✅ No NaN/Inf values")
        
        return True, len(feature_names)
        
    except Exception as e:
        logger.error(f"❌ Detection feature extraction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, 0


def compare_feature_counts(training_count, detection_count):
    """Compare feature counts between training and detection."""
    logger.info("\n" + "="*80)
    logger.info("📊 FEATURE COUNT COMPARISON")
    logger.info("="*80)
    
    logger.info(f"   Training features:  {training_count}")
    logger.info(f"   Detection features: {detection_count}")
    
    if training_count == detection_count:
        logger.info(f"✅ Feature counts match!")
        return True
    else:
        diff = abs(training_count - detection_count)
        logger.warning(f"⚠️  Feature count mismatch! Difference: {diff}")
        
        if training_count > detection_count:
            logger.warning(f"   Training has {diff} more features than detection")
        else:
            logger.warning(f"   Detection has {diff} more features than training")
        
        return False


def main():
    """Main validation."""
    logger.info("\n" + "="*80)
    logger.info("🚀 SR FEATURE VALIDATION")
    logger.info("="*80)
    logger.info(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    # Test training feature extraction
    training_success, training_count = validate_feature_extraction()
    
    # Test detection feature extraction  
    detection_success, detection_count = validate_detection_features()
    
    # Compare counts
    counts_match = compare_feature_counts(training_count, detection_count)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("📋 VALIDATION SUMMARY")
    logger.info("="*80)
    
    results = []
    results.append(("Training feature extraction", "✅ PASS" if training_success else "❌ FAIL"))
    results.append(("Detection feature extraction", "✅ PASS" if detection_success else "❌ FAIL"))
    results.append(("Feature count consistency", "✅ PASS" if counts_match else "⚠️  MISMATCH"))
    
    for test, result in results:
        logger.info(f"   {test:<35} {result}")
    
    all_pass = training_success and detection_success and counts_match
    
    if all_pass:
        logger.info("\n" + "="*80)
        logger.info("✅ ALL VALIDATIONS PASSED!")
        logger.info("="*80)
        logger.info("\n📚 Next steps:")
        logger.info("   1. Run feature investigation:")
        logger.info("      python scripts/investigate_sr_features.py --training-data <path> --analyze-missing")
        logger.info("\n   2. Retrain model with new features:")
        logger.info("      python scripts/run_sr_workflow.py --symbol BTCUSDT --timeframe 1h")
        logger.info("\n   3. Compare performance:")
        logger.info("      Check outcomes/ for before/after metrics")
        logger.info("="*80)
        return 0
    else:
        logger.error("\n" + "="*80)
        logger.error("❌ VALIDATION FAILED")
        logger.error("="*80)
        logger.error("\n🔧 Troubleshooting:")
        if not training_success:
            logger.error("   - Check sr_quality_data_collector.py::_extract_all_features()")
        if not detection_success:
            logger.error("   - Check enhanced_sr_detection.py::_extract_all_ml_features()")
        if not counts_match:
            logger.error("   - Ensure both methods extract the same features")
        logger.error("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())

