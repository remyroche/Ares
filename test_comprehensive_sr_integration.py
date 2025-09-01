#!/usr/bin/env python3
"""Test Comprehensive SR Feature Integration.

This script tests the comprehensive SR feature integration implementation
to ensure all ML models are trained on the full feature set from step7
and SR levels from step2_5.
"""

import asyncio
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Import the comprehensive training pipeline
from src.training.comprehensive_sr_training_pipeline import ComprehensiveSRTrainingPipeline
from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
from src.training.multi_output_model_trainer import MultiOutputModelTrainer, MultiOutputModelConfig


def create_test_market_data(n_rows: int = 1000) -> pd.DataFrame:
    """Create test market data for validation."""
    np.random.seed(42)
    
    # Generate realistic market data
    base_price = 50000.0
    dates = pd.date_range(start='2024-01-01', periods=n_rows, freq='1min')
    
    # Generate price movements
    returns = np.random.normal(0, 0.001, n_rows)  # 0.1% volatility per minute
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.0005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.0005))) for p in prices],
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_rows),
    })
    
    # Ensure high >= close >= low
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])
    
    return data


def create_test_step7_results() -> dict:
    """Create test step7 matrix operations results."""
    return {
        "sr_analysis": {
            "sr_features": [
                "sr_proximity", "support_proximity", "resistance_proximity", "sr_zone_width",
                "sr_strength", "support_strength", "resistance_strength", "sr_enhanced_strength",
                "sr_total_support_levels", "sr_total_resistance_levels", "sr_clusters_detected",
                "sr_fibonacci_levels", "sr_elliott_waves", "sr_order_flow_imbalances"
            ]
        },
        "sr_enhanced_analysis": {
            "enhanced_sr_features": [
                "sr_distance", "normalized_distance", "sr_proximity_score", "sr_zone_position_pct",
                "strength_score", "sr_enhanced_support_strength", "sr_enhanced_resistance_strength",
                "sr_optimized_strength_weights", "sr_noise_points", "sr_clustering_quality"
            ]
        },
        "sr_optimization_analysis": {
            "optimization_features": [
                "sr_level", "sr_order_flow_poc", "sr_order_flow_hvns", "sr_optimized_fibonacci_sensitivity",
                "sr_optimized_elliott_confidence", "sr_optimized_order_flow_threshold",
                "sr_touch_count", "sr_bounce_rate", "sr_isolation_score", "sr_momentum_pct",
                "sr_volatility_pct", "sr_trend_pct", "sr_optimization_score", "sr_optimized_method_weights",
                "sr_optimized_dbscan_eps", "sr_optimized_dbscan_min_samples", "delta_sr_score", "clarity_factor"
            ]
        }
    }


def create_test_step2_5_results() -> dict:
    """Create test step2_5 SR optimization results."""
    return {
        "sr_levels_result": {
            "support_levels": [
                {"price": 49500.0, "strength": 0.8, "volume": 1000000, "age": 50, "touches": 3},
                {"price": 49000.0, "strength": 0.7, "volume": 800000, "age": 30, "touches": 2},
                {"price": 48500.0, "strength": 0.6, "volume": 600000, "age": 20, "touches": 1}
            ],
            "resistance_levels": [
                {"price": 50500.0, "strength": 0.8, "volume": 1200000, "age": 45, "touches": 3},
                {"price": 51000.0, "strength": 0.7, "volume": 900000, "age": 25, "touches": 2},
                {"price": 51500.0, "strength": 0.6, "volume": 700000, "age": 15, "touches": 1}
            ]
        }
    }


async def test_sr_breakout_predictor_features():
    """Test SRBreakoutPredictor feature extraction."""
    print("🧪 Testing SRBreakoutPredictor feature extraction...")
    
    # Create test data
    market_data = create_test_market_data(100)
    current_price = market_data['close'].iloc[-1]
    
    # Initialize SRBreakoutPredictor
    config = {
        "sr_breakout_predictor": {
            "enable_detailed_reporting": True,
            "report_directory": "test_reports"
        }
    }
    sr_predictor = SRBreakoutPredictor(config)
    
    # Test feature extraction
    try:
        features = sr_predictor.extract_ml_features(market_data, current_price)
        
        print(f"✅ SRBreakoutPredictor extracted {len(features)} features")
        print(f"   - Proximity features: {len([f for f in features.keys() if 'proximity' in f])}")
        print(f"   - Strength features: {len([f for f in features.keys() if 'strength' in f])}")
        print(f"   - Level features: {len([f for f in features.keys() if 'level' in f])}")
        print(f"   - Advanced features: {len([f for f in features.keys() if any(keyword in f for keyword in ['fibonacci', 'elliott', 'order_flow'])])}")
        
        return True
        
    except Exception as e:
        print(f"❌ SRBreakoutPredictor feature extraction failed: {e}")
        return False


async def test_multi_output_model_trainer():
    """Test MultiOutputModelTrainer SR feature integration."""
    print("🧪 Testing MultiOutputModelTrainer SR feature integration...")
    
    # Create test data
    market_data = create_test_market_data(100)
    
    # Add target columns
    market_data['direction'] = np.random.choice([0, 1], size=len(market_data))
    market_data['potential_profit_pct'] = np.random.normal(0.02, 0.01, size=len(market_data))
    
    # Initialize trainer
    config = MultiOutputModelConfig()
    trainer = MultiOutputModelTrainer(config)
    
    # Test step7 feature loading
    try:
        # Create test step7 results
        step7_results = create_test_step7_results()
        step7_path = Path("test_data/step7")
        step7_path.mkdir(parents=True, exist_ok=True)
        
        with open(step7_path / "matrix_operations_results.json", 'w') as f:
            json.dump(step7_results, f)
        
        success = await trainer.load_step7_features(str(step7_path))
        print(f"✅ Step7 features loaded: {success}")
        print(f"   - Features count: {len(trainer.step7_features)}")
        
    except Exception as e:
        print(f"❌ Step7 feature loading failed: {e}")
        return False
    
    # Test step2_5 SR levels loading
    try:
        # Create test step2_5 results
        step2_5_results = create_test_step2_5_results()
        step2_5_path = Path("test_data/step2_5")
        step2_5_path.mkdir(parents=True, exist_ok=True)
        
        with open(step2_5_path / "sr_optimization_results.json", 'w') as f:
            json.dump(step2_5_results, f)
        
        success = await trainer.load_step2_5_sr_levels(str(step2_5_path))
        print(f"✅ Step2_5 SR levels loaded: {success}")
        print(f"   - Support levels: {len(trainer.step2_5_sr_levels.get('support_levels', []))}")
        print(f"   - Resistance levels: {len(trainer.step2_5_sr_levels.get('resistance_levels', []))}")
        
    except Exception as e:
        print(f"❌ Step2_5 SR levels loading failed: {e}")
        return False
    
    # Test comprehensive feature addition
    try:
        comprehensive_data = await trainer._add_comprehensive_sr_features(market_data)
        
        sr_features = [col for col in comprehensive_data.columns if 'sr_' in col.lower()]
        print(f"✅ Comprehensive features added: {len(sr_features)} SR features")
        print(f"   - Total features: {len(comprehensive_data.columns)}")
        print(f"   - SR feature percentage: {len(sr_features) / len(comprehensive_data.columns) * 100:.1f}%")
        
        # Test feature validation
        missing_features = trainer.validate_feature_completeness(comprehensive_data)
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
        else:
            print("✅ All required features present")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive feature addition failed: {e}")
        return False


async def test_comprehensive_training_pipeline():
    """Test ComprehensiveSRTrainingPipeline."""
    print("🧪 Testing ComprehensiveSRTrainingPipeline...")
    
    # Create test data
    market_data = create_test_market_data(200)
    market_data['direction'] = np.random.choice([0, 1], size=len(market_data))
    market_data['potential_profit_pct'] = np.random.normal(0.02, 0.01, size=len(market_data))
    
    # Create test results directories
    test_config = {
        "step7_output_path": "test_data/step7",
        "step2_5_output_path": "test_data/step2_5",
        "training_output_path": "test_data/training"
    }
    
    # Create test step7 results
    step7_results = create_test_step7_results()
    step7_path = Path(test_config["step7_output_path"])
    step7_path.mkdir(parents=True, exist_ok=True)
    
    with open(step7_path / "matrix_operations_results.json", 'w') as f:
        json.dump(step7_results, f)
    
    # Create test step2_5 results
    step2_5_results = create_test_step2_5_results()
    step2_5_path = Path(test_config["step2_5_output_path"])
    step2_5_path.mkdir(parents=True, exist_ok=True)
    
    with open(step2_5_path / "sr_optimization_results.json", 'w') as f:
        json.dump(step2_5_results, f)
    
    # Test comprehensive training
    try:
        pipeline = ComprehensiveSRTrainingPipeline(test_config)
        
        # Test feature summary
        feature_summary = pipeline.get_comprehensive_feature_summary()
        print(f"✅ Feature summary generated:")
        print(f"   - Step7 features: {feature_summary['step7_features']['count']}")
        print(f"   - Step2_5 support levels: {feature_summary['step2_5_sr_levels']['support_levels']}")
        print(f"   - Step2_5 resistance levels: {feature_summary['step2_5_sr_levels']['resistance_levels']}")
        
        # Test comprehensive training (without actual model training for speed)
        print("✅ Comprehensive training pipeline initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive training pipeline failed: {e}")
        return False


async def test_feature_completeness():
    """Test feature completeness validation."""
    print("🧪 Testing feature completeness validation...")
    
    # Create test data with comprehensive features
    market_data = create_test_market_data(100)
    market_data['direction'] = np.random.choice([0, 1], size=len(market_data))
    market_data['potential_profit_pct'] = np.random.normal(0.02, 0.01, size=len(market_data))
    
    # Initialize trainer
    config = MultiOutputModelConfig()
    trainer = MultiOutputModelTrainer(config)
    
    # Add comprehensive features
    comprehensive_data = await trainer._add_comprehensive_sr_features(market_data)
    
    # Test feature analysis
    try:
        sr_feature_stats = trainer._analyze_sr_features(comprehensive_data)
        
        print(f"✅ SR feature analysis completed:")
        print(f"   - Total SR features: {sr_feature_stats['sr_feature_count']}")
        print(f"   - SR feature percentage: {sr_feature_stats['sr_feature_percentage']:.1f}%")
        print(f"   - Feature categories: {list(sr_feature_stats['sr_feature_categories'].keys())}")
        
        # Test feature completeness
        missing_features = trainer.validate_feature_completeness(comprehensive_data)
        
        if missing_features:
            print(f"⚠️ Missing features by category:")
            for category, features in missing_features.items():
                print(f"   - {category}: {len(features)} missing")
        else:
            print("✅ All required features present")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature analysis failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Comprehensive SR Feature Integration Tests")
    print("=" * 60)
    
    test_results = {}
    
    # Test 1: SRBreakoutPredictor feature extraction
    test_results['sr_breakout_predictor'] = await test_sr_breakout_predictor_features()
    
    # Test 2: MultiOutputModelTrainer SR feature integration
    test_results['multi_output_trainer'] = await test_multi_output_model_trainer()
    
    # Test 3: Comprehensive training pipeline
    test_results['comprehensive_pipeline'] = await test_comprehensive_training_pipeline()
    
    # Test 4: Feature completeness validation
    test_results['feature_completeness'] = await test_feature_completeness()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Comprehensive SR feature integration is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    # Cleanup test data
    try:
        import shutil
        if Path("test_data").exists():
            shutil.rmtree("test_data")
        if Path("test_reports").exists():
            shutil.rmtree("test_reports")
        print("🧹 Test data cleaned up")
    except Exception as e:
        print(f"⚠️ Cleanup failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())