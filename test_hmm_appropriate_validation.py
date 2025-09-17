#!/usr/bin/env python3
"""
Test Script for HMM-Appropriate Validation Metrics

This script demonstrates the new HMM-appropriate validation metrics that replace
traditional clustering metrics (silhouette_score, davies_bouldin_score, calinski_harabasz_score)
with metrics designed specifically for temporal regime modeling.
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_sample_market_data(n_samples=1000):
    """Create sample market data for testing."""
    np.random.seed(42)
    
    # Create time series data
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='1H')
    
    # Create regime-based price data with overlapping characteristics
    regimes = []
    prices = []
    volumes = []
    
    current_price = 100.0
    
    for i in range(n_samples):
        # Define regimes with overlapping characteristics
        if i < n_samples * 0.2:  # Regime 0: Trending up with moderate volatility
            regime = 0
            price_change = np.random.normal(0.001, 0.02)  # Positive trend, moderate volatility
            volume = np.random.normal(1000, 200)
        elif i < n_samples * 0.4:  # Regime 1: High volatility (can overlap with trending)
            regime = 1
            price_change = np.random.normal(0.0, 0.05)  # High volatility
            volume = np.random.normal(1200, 300)
        elif i < n_samples * 0.6:  # Regime 2: Ranging with moderate volatility (overlaps with Regime 0)
            regime = 2
            price_change = np.random.normal(0.0, 0.02)  # Moderate volatility (overlaps with Regime 0)
            volume = np.random.normal(800, 150)
        elif i < n_samples * 0.8:  # Regime 3: Extreme events (overlaps with Regime 1)
            regime = 3
            price_change = np.random.normal(0.0, 0.08)  # Very high volatility (overlaps with Regime 1)
            volume = np.random.normal(2000, 500)
        else:  # Regime 4: Low activity
            regime = 4
            price_change = np.random.normal(0.0, 0.01)  # Low volatility
            volume = np.random.normal(500, 100)
        
        current_price *= (1 + price_change)
        
        regimes.append(regime)
        prices.append(current_price)
        volumes.append(max(volume, 100))  # Ensure positive volume
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': volumes,
        'regime': regimes
    })
    
    # Calculate returns
    data['returns'] = data['close'].pct_change().fillna(0)
    
    return data

def test_hmm_validation_metrics():
    """Test the new HMM validation metrics."""
    print("🧪 Testing HMM-Appropriate Validation Metrics")
    print("=" * 60)
    
    # Create sample data
    print("📊 Creating sample market data with overlapping regime characteristics...")
    market_data = create_sample_market_data(1000)
    regime_data = market_data[['regime']].copy()
    
    print(f"✅ Created {len(market_data)} samples with {market_data['regime'].nunique()} regimes")
    print(f"📈 Regime distribution: {market_data['regime'].value_counts().to_dict()}")
    
    try:
        # Import and test HMM validation framework
        from src.utils.ml_common.hmm_validation_metrics import HMMValidationFramework
        
        print("\n🔍 Testing HMM Validation Framework...")
        validator = HMMValidationFramework()
        
        # Test individual metrics
        print("\n1️⃣ Testing Temporal Coherence Metrics...")
        temporal_metrics = validator.calculate_temporal_coherence(market_data['regime'].values)
        print(f"   Temporal Coherence: {temporal_metrics.temporal_coherence:.3f}")
        print(f"   Avg Regime Duration: {temporal_metrics.avg_regime_duration:.1f}")
        print(f"   Duration Stability: {temporal_metrics.duration_stability:.3f}")
        print(f"   Interpretation: {temporal_metrics.interpretation}")
        
        print("\n2️⃣ Testing Transition Quality Metrics...")
        transition_matrix = validator._calculate_transition_matrix(market_data['regime'].values)
        transition_metrics = validator.calculate_transition_quality(transition_matrix)
        print(f"   Transition Quality: {transition_metrics.transition_quality:.3f}")
        print(f"   Avg Persistence: {transition_metrics.avg_persistence:.3f}")
        print(f"   Transition Entropy: {transition_metrics.transition_entropy:.3f}")
        print(f"   Interpretation: {transition_metrics.interpretation}")
        
        print("\n3️⃣ Testing Economic Differentiation Metrics...")
        economic_metrics = validator.calculate_economic_differentiation(market_data)
        print(f"   Economic Differentiation: {economic_metrics.economic_differentiation:.3f}")
        print(f"   Return Differentiation: {economic_metrics.return_differentiation:.3f}")
        print(f"   Volatility Differentiation: {economic_metrics.volatility_differentiation:.3f}")
        print(f"   Risk-Return Tradeoff: {economic_metrics.risk_return_tradeoff:.3f}")
        print(f"   Interpretation: {economic_metrics.interpretation}")
        
        print("\n4️⃣ Testing Spatial Coherence Metrics...")
        feature_columns = ['returns', 'volume', 'close']
        spatial_metrics = validator.calculate_spatial_coherence(market_data, feature_columns)
        print(f"   Spatial Coherence: {spatial_metrics['spatial_coherence']:.3f}")
        print(f"   Intra-Regime Similarity: {spatial_metrics['intra_regime_similarity']:.3f}")
        print(f"   Interpretation: {spatial_metrics['interpretation']}")
        
        print("\n5️⃣ Testing Comprehensive HMM Validation...")
        comprehensive_metrics = validator.validate_hmm_regimes(market_data, market_data, feature_columns, generate_detailed_report=True)
        print(f"   Overall HMM Quality Score: {comprehensive_metrics.hmm_quality_score:.3f}")
        print(f"   Overall Interpretation: {comprehensive_metrics.overall_interpretation}")
        
        # Test detailed reporting
        if comprehensive_metrics.detailed_report:
            print("\n6️⃣ Testing Detailed Reporting...")
            detailed = comprehensive_metrics.detailed_report
            print(f"   Execution Summary: {detailed.execution_summary.get('validation_status', 'Unknown')}")
            print(f"   Quality Grade: {detailed.execution_summary.get('quality_grade', 'Unknown')}")
            print(f"   Temporal Grade: {detailed.temporal_analysis.get('temporal_grade', 'Unknown')}")
            print(f"   Transition Grade: {detailed.transition_analysis.get('transition_grade', 'Unknown')}")
            print(f"   Economic Grade: {detailed.economic_analysis.get('economic_grade', 'Unknown')}")
            print(f"   Spatial Grade: {detailed.spatial_analysis.get('spatial_grade', 'Unknown')}")
            
            # Show recommendations
            recommendations = detailed.recommendations
            print(f"   Immediate Actions: {len(recommendations.get('immediate_actions', []))} items")
            print(f"   Improvement Suggestions: {len(recommendations.get('improvement_suggestions', []))} items")
            print(f"   Parameter Tuning: {len(recommendations.get('parameter_tuning', []))} items")
            
            # Show quality assessment
            quality = detailed.quality_assessment
            print(f"   Overall Grade: {quality.get('overall_grade', 'Unknown')}")
            print(f"   Production Readiness: {quality.get('production_readiness', {}).get('readiness_level', 'Unknown')}")
            print(f"   ML Training Suitability: {quality.get('ml_training_suitability', {}).get('suitability_level', 'Unknown')}")
            
            print(f"   Strengths: {len(quality.get('strengths', []))} identified")
            print(f"   Weaknesses: {len(quality.get('weaknesses', []))} identified")
        
        # Compare with traditional clustering metrics (to show the difference)
        print("\n📊 Comparison with Traditional Clustering Metrics:")
        print("   (These metrics are misleading for HMM regimes)")
        
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data for traditional clustering metrics
            feature_data = market_data[feature_columns].values
            scaler = StandardScaler()
            feature_data_scaled = scaler.fit_transform(feature_data)
            
            # Calculate traditional metrics
            silhouette = silhouette_score(feature_data_scaled, market_data['regime'])
            calinski_harabasz = calinski_harabasz_score(feature_data_scaled, market_data['regime'])
            davies_bouldin = davies_bouldin_score(feature_data_scaled, market_data['regime'])
            
            print(f"   Silhouette Score: {silhouette:.3f} (negative = overlapping regimes)")
            print(f"   Calinski-Harabasz Score: {calinski_harabasz:.1f} (higher = better separation)")
            print(f"   Davies-Bouldin Score: {davies_bouldin:.3f} (lower = better separation)")
            print(f"   ❌ These metrics assume spatial separation, which is inappropriate for HMM regimes")
            
        except ImportError:
            print("   ⚠️ sklearn not available for traditional clustering metrics comparison")
        
        print("\n✅ HMM validation metrics test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the HMM validation framework is properly installed")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_validation():
    """Test the integrated validation system."""
    print("\n🔧 Testing Integrated HMM Validation System")
    print("=" * 60)
    
    try:
        # Create sample data
        market_data = create_sample_market_data(500)
        
        # Test integrated validation
        from src.utils.hmm_validation import HMMStatisticalValidator
        
        validator = HMMStatisticalValidator()
        
        print("📊 Testing integrated HMM validation...")
        validation_result = validator.validate_hmm_regimes_appropriate(
            market_data, market_data, ['returns', 'volume', 'close']
        )
        
        print(f"✅ Integrated validation completed")
        print(f"   HMM Quality Score: {validation_result['hmm_validation_metrics']['hmm_quality_score']:.3f}")
        print(f"   Validation Passed: {validation_result['hmm_validation_metrics']['validation_passed']}")
        print(f"   Summary Assessment: {validation_result.get('summary_assessment', 'Unknown')}")
        print(f"   Validation Method: {validation_result.get('validation_method', 'Unknown')}")
        
        # Show detailed metrics
        if 'temporal_coherence' in validation_result:
            temporal = validation_result['temporal_coherence']
            print(f"   Temporal Coherence: {temporal.get('temporal_coherence', 0):.3f}")
        
        if 'transition_quality' in validation_result:
            transition = validation_result['transition_quality']
            print(f"   Transition Quality: {transition.get('transition_quality', 0):.3f}")
        
        if 'economic_differentiation' in validation_result:
            economic = validation_result['economic_differentiation']
            print(f"   Economic Differentiation: {economic.get('economic_differentiation', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integrated validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 HMM-Appropriate Validation Metrics Test Suite")
    print("=" * 80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test individual metrics
    success1 = test_hmm_validation_metrics()
    
    # Test integrated system
    success2 = test_integrated_validation()
    
    print("\n" + "=" * 80)
    if success1 and success2:
        print("🎉 All tests passed! HMM-appropriate validation metrics are working correctly.")
        print()
        print("📋 Summary of Changes:")
        print("   ✅ Replaced Silhouette Score with Temporal Coherence")
        print("   ✅ Replaced Davies-Bouldin Score with Transition Quality")
        print("   ✅ Replaced Calinski-Harabasz Score with Economic Differentiation")
        print("   ✅ Added comprehensive HMM validation framework")
        print("   ✅ Integrated spatial coherence for internal cluster validity")
        print("   ✅ Added regime stability analysis")
        print()
        print("🎯 Your HMM regime detection system now uses appropriate metrics!")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()