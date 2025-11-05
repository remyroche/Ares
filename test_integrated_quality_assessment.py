#!/usr/bin/env python3
"""
Test Integrated ClusterQualityAssessor in Sticky Finite HMM Components

This script demonstrates the comprehensive quality assessment integration across:
1. StickyFiniteHMMRegimeDiscoveryStep
2. StickyFiniteHMMAutoTuner  
3. EnhancedStickyFiniteHMMClusteringIntegration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import asyncio

# Import tprint utilities
from src.utils.tprint import (
    tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_structured, tprint_timer
)

def generate_sample_market_data(years: int = 1) -> pd.DataFrame:
    """
    Generate realistic sample market data for testing.
    
    Args:
        years: Number of years of data to generate
        
    Returns:
        DataFrame with OHLCV data
    """
    tprint_info(f"📊 Generating {years} year(s) of sample market data...")
    
    # Generate date range
    start_date = datetime.now() - timedelta(days=365 * years)
    dates = pd.date_range(start=start_date, periods=365 * 24 * years, freq='1h')
    
    # Generate realistic price data with regime-like behavior
    np.random.seed(42)
    base_price = 50000
    
    # Create different regime parameters
    regime_params = [
        {'vol': 0.01, 'trend': 0.0002, 'duration': 720},   # Stable upward
        {'vol': 0.03, 'trend': -0.0001, 'duration': 480}, # Volatile downward
        {'vol': 0.015, 'trend': 0.0000, 'duration': 600}, # Sideways
        {'vol': 0.025, 'trend': 0.0003, 'duration': 360}, # Trending upward
        {'vol': 0.04, 'trend': -0.0002, 'duration': 240}  # High volatility
    ]
    
    prices = [base_price]
    current_regime = 0
    regime_counter = 0
    
    for i in range(1, len(dates)):
        # Switch regimes periodically
        if regime_counter >= regime_params[current_regime]['duration']:
            current_regime = (current_regime + 1) % len(regime_params)
            regime_counter = 0
        
        params = regime_params[current_regime]
        regime_counter += 1
        
        # Generate return with regime-specific parameters
        ret = np.random.normal(params['trend'], params['vol'])
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1000))  # Ensure price stays positive
    
    prices = prices[1:]
    
    # Create OHLCV DataFrame
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,  # Use same prices for close to avoid length mismatch
        'volume': np.random.lognormal(10, 1, len(prices))
    }, index=dates[:len(prices)])
    
    tprint_success(f"✅ Generated {len(data)} samples with realistic regime behavior")
    return data

async def test_regime_discovery_step():
    """
    Test StickyFiniteHMMRegimeDiscoveryStep with integrated quality assessment.
    """
    tprint_info("\n" + "="*80)
    tprint_info("🧪 Testing StickyFiniteHMMRegimeDiscoveryStep with Quality Assessment")
    tprint_info("="*80)
    
    try:
        # Import the regime discovery step
        from src.training.steps.market_analysis.sticky_finite_hmm_clustering.sticky_finite_hmm_regime_discovery_step import (
            StickyFiniteHMMRegimeDiscoveryStep
        )
        
        # Initialize the step
        step = StickyFiniteHMMRegimeDiscoveryStep()
        
        # Generate test data
        market_data = generate_sample_market_data(years=1)
        
        # Configure for testing with reduced parameters for speed
        config = {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1h',
            'regime_timeframe': '1h',
            'execution_mode': 'light',  # Use light mode for faster testing
            'enable_auto_tuning': False,  # Skip auto-tuning for this test
            'sticky_finite_hmm_params': {
                'K': 5,
                'n_mixtures': 1,
                'base_alpha': 0.5,
                'kappa': 10.0,
                'num_iters': 50,  # Reduced for faster testing
                'lr': 1e-2,
                'min_features': 30,
                'max_features': 50,
                'enable_pca': True,
                'pca_components': 10
            }
        }
        
        # Execute the step
        with tprint_timer("Regime Discovery Step with Quality Assessment", level="PERFORMANCE"):
            result = await step.execute(config)
        
        # Display results
        if result['success']:
            tprint_success("✅ Regime Discovery Step completed successfully!")
            
            # Show comprehensive quality metrics
            comprehensive_metrics = result.get('comprehensive_quality_metrics', {})
            if comprehensive_metrics:
                tprint_info("\n📊 Comprehensive Quality Assessment Results:")
                tprint_structured({
                    "Overall Quality Score": comprehensive_metrics.get('quality_score', 'N/A'),
                    "Silhouette Score": comprehensive_metrics.get('silhouette_score', 'N/A'),
                    "Davies-Bouldin Index": comprehensive_metrics.get('davies_bouldin_score', 'N/A'),
                    "Calinski-Harabasz Index": comprehensive_metrics.get('calinski_harabasz_score', 'N/A'),
                    "Temporal Smoothness": comprehensive_metrics.get('temporal_smoothness', 'N/A'),
                    "Regime Persistence": comprehensive_metrics.get('regime_persistence', 'N/A'),
                    "Number of Regimes": comprehensive_metrics.get('n_regimes', 'N/A'),
                    "Balance Score": comprehensive_metrics.get('balance_score', 'N/A')
                }, level="INFO")
                
                # Show per-regime metrics
                per_regime = comprehensive_metrics.get('per_regime_metrics', {})
                if per_regime:
                    tprint_info("\n🎯 Per-Regime Analysis:")
                    for regime_id, metrics in per_regime.items():
                        if isinstance(metrics, dict):
                            tprint_info(f"   Regime {regime_id}: "
                                      f"Size={metrics.get('size', 0)}, "
                                      f"Type={metrics.get('regime_type', 'unknown')}, "
                                      f"Sharpe={metrics.get('sharpe', 0):.3f}")
            else:
                tprint_warning("⚠️ No comprehensive quality metrics available")
            
            return result
        else:
            tprint_error(f"❌ Regime Discovery Step failed: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        tprint_error(f"❌ Test failed: {e}")
        return None

def test_auto_tuner_objective():
    """
    Test StickyFiniteHMMAutoTuner objective function with integrated quality assessment.
    """
    tprint_info("\n" + "="*80)
    tprint_info("🧪 Testing StickyFiniteHMMAutoTuner with Quality Assessment")
    tprint_info("="*80)
    
    try:
        # Import the objective function
        from src.training.steps.market_analysis.sticky_finite_hmm_clustering.sticky_finite_hmm_auto_tuner import (
            sticky_finite_hmm_objective_function
        )
        
        # Generate test data
        market_data = generate_sample_market_data(years=1)
        
        # Test parameters
        test_params = {
            'K': 5,
            'n_mixtures': 1,
            'base_alpha': 0.5,
            'kappa': 10.0,
            'lr': 1e-2,
            'pca_components': 10
        }
        
        tprint_info("🧪 Testing objective function with sample parameters...")
        tprint_structured(test_params, level="INFO")
        
        # Execute objective function
        with tprint_timer("Auto-Tuner Objective Function", level="PERFORMANCE"):
            score = sticky_finite_hmm_objective_function(
                params=test_params,
                X_train=np.random.random((100, 10)),  # Dummy data
                y_train=np.random.random(100),
                market_data=market_data,
                symbol='BTCUSDT',
                exchange='binance',
                timeframe='1h',
                scoring_metric='composite_score'
            )
        
        tprint_success(f"✅ Auto-tuner objective function completed!")
        tprint_info(f"📊 Composite Quality Score: {score:.4f}")
        
        return score
        
    except Exception as e:
        tprint_error(f"❌ Auto-tuner test failed: {e}")
        return None

def test_enhanced_integration():
    """
    Test EnhancedStickyFiniteHMMClusteringIntegration with quality assessment.
    """
    tprint_info("\n" + "="*80)
    tprint_info("🧪 Testing EnhancedStickyFiniteHMMClusteringIntegration with Quality Assessment")
    tprint_info("="*80)
    
    try:
        # Import the enhanced integration
        from src.feature_generation.integration.enhanced_sticky_finite_hmm_clustering_integration import (
            EnhancedStickyFiniteHMMClusteringIntegration
        )
        
        # Initialize the integration
        integration = EnhancedStickyFiniteHMMClusteringIntegration(
            K=5,
            n_mixtures=1,
            base_alpha=0.5,
            kappa=10.0,
            num_iters=50,  # Reduced for faster testing
            lr=1e-2,
            min_features=30,
            max_features=50,
            enable_pca_reduction=True,
            pca_components=10
        )
        
        # Generate test data
        market_data = generate_sample_market_data(years=1)
        
        tprint_info("🧪 Running enhanced clustering with comprehensive quality assessment...")
        
        # Execute clustering
        with tprint_timer("Enhanced Clustering Integration", level="PERFORMANCE"):
            result = integration.cluster_with_sticky_finite_hmm(
                data=market_data,
                compute_posteriors=True
            )
        
        tprint_success("✅ Enhanced clustering integration completed!")
        
        # Show comprehensive quality metrics
        comprehensive_metrics = result.get('comprehensive_quality_metrics', {})
        if comprehensive_metrics:
            tprint_info("\n📊 Enhanced Integration Quality Assessment:")
            tprint_structured({
                "Enhanced Quality Score": result.get('enhanced_quality_score', 'N/A'),
                "Basic Composite Score": result['quality_metrics'].get('composite_score', 'N/A'),
                "Silhouette Score": comprehensive_metrics.get('silhouette_score', 'N/A'),
                "Temporal Smoothness": comprehensive_metrics.get('temporal_smoothness', 'N/A'),
                "Regime Persistence": comprehensive_metrics.get('regime_persistence', 'N/A'),
                "Number of Regimes": comprehensive_metrics.get('n_regimes', 'N/A'),
                "Balance Score": comprehensive_metrics.get('balance_score', 'N/A')
            }, level="INFO")
        else:
            tprint_warning("⚠️ No comprehensive quality metrics available in enhanced integration")
        
        return result
        
    except Exception as e:
        tprint_error(f"❌ Enhanced integration test failed: {e}")
        return None

async def main():
    """
    Main test function to run all integrated quality assessment tests.
    """
    tprint_info("🚀 Starting Integrated ClusterQualityAssessor Tests")
    tprint_info("="*80)
    
    # Test 1: Regime Discovery Step
    regime_result = await test_regime_discovery_step()
    
    # Test 2: Auto-Tuner Objective Function
    tuner_score = test_auto_tuner_objective()
    
    # Test 3: Enhanced Integration
    integration_result = test_enhanced_integration()
    
    # Summary
    tprint_info("\n" + "="*80)
    tprint_info("📋 INTEGRATED QUALITY ASSESSMENT TEST SUMMARY")
    tprint_info("="*80)
    
    summary = {
        "Regime Discovery Step": "✅ PASSED" if regime_result and regime_result.get('success') else "❌ FAILED",
        "Auto-Tuner Integration": "✅ PASSED" if tuner_score is not None else "❌ FAILED", 
        "Enhanced Integration": "✅ PASSED" if integration_result else "❌ FAILED"
    }
    
    tprint_structured(summary, level="INFO")
    
    # Check for generated CSV reports
    tprint_info("\n📄 Checking for generated CSV reports...")
    
    # Check artifacts directory
    artifacts_dir = Path("artifacts")
    if artifacts_dir.exists():
        quality_reports = list(artifacts_dir.glob("**/quality_reports/**/*.csv"))
        if quality_reports:
            tprint_success(f"✅ Found {len(quality_reports)} quality report CSVs in artifacts:")
            for report in quality_reports[:5]:  # Show first 5
                tprint_info(f"   📄 {report.name}")
        else:
            tprint_warning("⚠️ No quality reports found in artifacts directory")
    
    # Check outcomes directory  
    outcomes_dir = Path("outcomes")
    if outcomes_dir.exists():
        enhanced_reports = list(outcomes_dir.glob("**/enhanced_sticky_finite_hmm_quality_reports/**/*.csv"))
        if enhanced_reports:
            tprint_success(f"✅ Found {len(enhanced_reports)} enhanced quality report CSVs in outcomes:")
            for report in enhanced_reports[:5]:  # Show first 5
                tprint_info(f"   📄 {report.name}")
        else:
            tprint_warning("⚠️ No enhanced quality reports found in outcomes directory")
    
    # Overall result
    all_passed = all("PASSED" in status for status in summary.values())
    
    if all_passed:
        tprint_success("\n🎉 ALL INTEGRATED QUALITY ASSESSMENT TESTS PASSED!")
        tprint_success("✅ ClusterQualityAssessor successfully integrated into all components")
    else:
        tprint_error("\n❌ SOME TESTS FAILED - Check integration")
    
    return all_passed

if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    exit(0 if success else 1)
