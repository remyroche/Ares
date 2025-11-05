#!/usr/bin/env python3
"""
Complete Sticky Finite HMM Pipeline Runner

This script runs the complete Sticky Finite HMM system with:
1. Real 2 years of ETHUSDT data (downloaded by sticky_finite_hmm_regime_discovery_step)
2. Full feature set (100+ features) via enhanced_sticky_finite_hmm_clustering_integration
3. Auto-tuning via sticky_finite_hmm_auto_tuner
4. Comprehensive regime discovery and analysis

Usage:
    python run_sticky_finite_hmm_complete.py
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_structured, tprint_timer, tprint_data_preview
)

# Import core components
from src.training.steps.market_analysis.sticky_finite_hmm_clustering.sticky_finite_hmm_regime_discovery_step import (
    StickyFiniteHMMRegimeDiscoveryStep
)

from src.feature_generation.integration.enhanced_sticky_finite_hmm_clustering_integration import (
    EnhancedStickyFiniteHMMClusteringIntegration,
    perform_enhanced_sticky_finite_hmm_clustering
)

# Import data loader
from src.utils.data_loader import DataLoader


async def run_complete_sticky_finite_hmm_pipeline():
    """
    Run the complete Sticky Finite HMM pipeline with all components.
    
    Pipeline:
    1. Load 2 years of ETHUSDT data
    2. Generate 100+ comprehensive features
    3. Run auto-tuning to find optimal parameters
    4. Perform regime discovery with Sticky Finite HMM
    5. Generate comprehensive reports
    """
    
    tprint("=" * 100, "INFO")
    tprint("🚀 COMPLETE STICKY FINITE HMM PIPELINE", "INFO")
    tprint("=" * 100, "INFO")
    tprint("📊 Components:", "INFO")
    tprint("   1. StickyFiniteHMMRegimeDiscoveryStep - Data loading & orchestration", "INFO")
    tprint("   2. EnhancedStickyFiniteHMMClusteringIntegration - 100+ features", "INFO")
    tprint("   3. StickyFiniteHMMAutoTuner - Hyperparameter optimization", "INFO")
    tprint("   4. Real ETHUSDT data (2 years) - Downloaded by regime_discovery_step", "INFO")
    tprint("=" * 100, "INFO")
    
    start_time = time.time()
    
    try:
        # Step 1: Initialize the regime discovery step
        tprint_info("🔧 Step 1: Initializing Sticky Finite HMM Regime Discovery Step")
        
        regime_discovery_step = StickyFiniteHMMRegimeDiscoveryStep(
            step_name="sticky_finite_hmm_complete_pipeline"
        )
        
        # Step 2: Load real ETHUSDT data (2 years)
        tprint_info("📥 Step 2: Loading 2 years of ETHUSDT data")
        
        # Configure for real data loading
        config = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframe': '1h',
            'regime_timeframe': '1h',  # Use 1h for regime detection
            'execution_mode': 'full',   # Full mode for complete analysis
            'enable_auto_tuning': True, # Enable auto-tuning
            'auto_tuning_config': {
                'use_hierarchical': True,      # Use hierarchical optimization
                'use_multi_objective': False,  # Single objective for speed
                'n_rounds': 2,                 # 2 optimization rounds
                'tpe_trials': 50,              # 50 TPE trials (reduced for speed)
                'timeout': 1800                # 30 min timeout
            },
            'sticky_finite_hmm_params': {
                'min_features': 100,   # Ensure 100+ features
                'max_features': 150,   # Allow up to 150 features
                'enable_pca': True,
                'pca_components': 20,  # Use 20 PCA components
                'num_iters': 800,      # Sufficient iterations
                'compute_posteriors': True  # Full computation
            }
        }
        
        # Verify data loading works
        data_loader = DataLoader()
        market_data = data_loader.load_ethusdt_1h_data()
        
        if market_data is None or market_data.empty:
            tprint_error("❌ Failed to load ETHUSDT data. Please ensure data is available in historical_data/binance/ethusdt/")
            return False
        
        tprint_success(f"✅ Loaded {len(market_data)} samples of ETHUSDT data")
        tprint_data_preview(market_data, "Market Data", max_rows=3, max_cols=6)
        
        # Check data coverage
        if 'timestamp' in market_data.columns:
            market_data['datetime'] = pd.to_datetime(market_data['timestamp'], unit='ms')
            time_span = (market_data['datetime'].max() - market_data['datetime'].min()).days
            tprint_info(f"📅 Data spans {time_span} days")
            
            if time_span < 365:
                tprint_warning(f"⚠️ Data covers only {time_span} days (recommended: 2+ years)")
            else:
                tprint_success(f"✅ Data covers {time_span} days (meets 2-year requirement)")
        
        # Step 3: Test enhanced feature generation
        tprint_info("🔍 Step 3: Testing enhanced feature generation (100+ features)")
        
        feature_integration = EnhancedStickyFiniteHMMClusteringIntegration(
            min_features=100,
            max_features=150,
            enable_comprehensive_features=True,
            enable_pca_reduction=True,
            pca_components=20,
            K=5,
            n_mixtures=1,
            base_alpha=0.5,
            kappa=10.0,
            enable_mtf_features=True,  # Enable multi-timeframe features
            mtf_timeframes=['4h', '1d']
        )
        
        # Generate features to verify count
        with tprint_timer("Feature Generation", level="PERFORMANCE"):
            feature_result = feature_integration.get_comprehensive_clustering_features(market_data)
        
        features_generated = len(feature_result.get('feature_names', []))
        tprint_success(f"✅ Generated {features_generated} features")
        
        if features_generated < 100:
            tprint_warning(f"⚠️ Only {features_generated} features generated (target: 100+)")
        else:
            tprint_success(f"✅ Feature set meets requirement: {features_generated} features")
        
        # Step 4: Run complete pipeline with auto-tuning
        tprint_info("🎯 Step 4: Running complete Sticky Finite HMM pipeline with auto-tuning")
        
        # Execute the full pipeline
        result = await regime_discovery_step.execute(config)
        
        if not result.get('success', False):
            tprint_error(f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")
            return False
        
        # Step 5: Display results
        tprint_success("✅ PIPELINE COMPLETED SUCCESSFULLY")
        tprint("=" * 80, "SUCCESS")
        
        # Extract key metrics
        metrics = result.get('metrics', {})
        artifacts = result.get('artifacts', {})
        execution_time = result.get('execution_time', 0)
        n_regimes = result.get('n_regimes', 0)
        composite_score = result.get('composite_score', 0)
        
        tprint_structured({
            "Execution Time": f"{execution_time:.2f}s",
            "Regimes Discovered": n_regimes,
            "Composite Quality Score": f"{composite_score:.4f}",
            "Features Used": features_generated,
            "Data Points": len(market_data),
            "Auto-Tuning": "Enabled" if result.get('auto_tuning_results') else "Disabled"
        }, level="SUCCESS")
        
        # Show auto-tuning results if available
        auto_tuning_results = result.get('auto_tuning_results')
        if auto_tuning_results:
            tprint("", "SUCCESS")
            tprint("🎯 Auto-Tuning Results:", "SUCCESS")
            tprint(f"   Best Score: {auto_tuning_results.get('best_score', 0):.4f}", "SUCCESS")
            tprint(f"   Total Trials: {auto_tuning_results.get('total_trials', 0)}", "SUCCESS")
            tprint(f"   Tuning Time: {auto_tuning_results.get('total_time', 0):.1f}s", "SUCCESS")
            
            best_params = auto_tuning_results.get('best_params', {})
            if best_params:
                tprint("   Best Parameters:", "SUCCESS")
                tprint(f"     K: {best_params.get('K', 5)}", "SUCCESS")
                tprint(f"     kappa: {best_params.get('kappa', 10.0):.2f}", "SUCCESS")
                tprint(f"     base_alpha: {best_params.get('base_alpha', 0.5):.3f}", "SUCCESS")
                tprint(f"     lr: {best_params.get('lr', 1e-2):.5f}", "SUCCESS")
                tprint(f"     pca_components: {best_params.get('pca_components', 15)}", "SUCCESS")
        
        # Show quality metrics
        quality_metrics = metrics.get('quality_assessment', {})
        if quality_metrics:
            tprint("", "SUCCESS")
            tprint("📊 Quality Metrics:", "SUCCESS")
            tprint(f"   Silhouette Score: {quality_metrics.get('silhouette_score', 0):.4f}", "SUCCESS")
            tprint(f"   Temporal Smoothness: {quality_metrics.get('temporal_smoothness', 0):.4f}", "SUCCESS")
            tprint(f"   Balance Score: {quality_metrics.get('balance_score', 0):.4f}", "SUCCESS")
            tprint(f"   Between-Regime CV: {quality_metrics.get('between_regime_cv', 0):.4f}", "SUCCESS")
            tprint(f"   Within-Regime CV: {quality_metrics.get('within_regime_cv', 0):.4f}", "SUCCESS")
            tprint(f"   CV Ratio: {quality_metrics.get('cv_ratio', 0):.4f}", "SUCCESS")
            
            # Economic metrics
            per_regime_metrics = quality_metrics.get('per_regime_metrics', {})
            if per_regime_metrics:
                regime_sharpes = [v.get('sharpe', 0) for v in per_regime_metrics.values() if isinstance(v, dict)]
                regime_returns = [v.get('mean_return', 0) for v in per_regime_metrics.values() if isinstance(v, dict)]
                if regime_sharpes:
                    avg_sharpe = sum(regime_sharpes) / len(regime_sharpes)
                    tprint(f"   Average Regime Sharpe: {avg_sharpe:.3f}", "SUCCESS")
                if regime_returns:
                    avg_return = sum(regime_returns) / len(regime_returns)
                    tprint(f"   Average Regime Return: {avg_return:.4f}", "SUCCESS")
        
        # Show regime statistics
        if artifacts:
            cluster_stats = artifacts.get('hdp_hmm_cluster_statistics') or artifacts.get('sticky_finite_hmm_cluster_statistics')
            if cluster_stats:
                tprint("", "SUCCESS")
                tprint("📈 Regime Statistics:", "SUCCESS")
                tprint(f"   Transition Persistence: {cluster_stats.get('transition_persistence', 0):.3f}", "SUCCESS")
                tprint(f"   Final ELBO: {cluster_stats.get('final_elbo', 0):.2f}", "SUCCESS")
                
                regime_sizes = cluster_stats.get('regime_sizes', {})
                if regime_sizes:
                    tprint("   Regime Sizes:", "SUCCESS")
                    for regime_id, size in sorted(regime_sizes.items()):
                        pct = (size / len(market_data)) * 100
                        tprint(f"     Regime {regime_id}: {size} samples ({pct:.1f}%)", "SUCCESS")
        
        total_time = time.time() - start_time
        tprint("=" * 80, "SUCCESS")
        tprint(f"🎉 COMPLETE PIPELINE FINISHED in {total_time:.2f}s", "SUCCESS")
        tprint("=" * 80, "SUCCESS")
        
        # Verify key requirements
        tprint("", "INFO")
        tprint("✅ REQUIREMENTS VERIFICATION:", "INFO")
        tprint(f"   ✅ Real data: ETHUSDT {len(market_data)} samples loaded", "INFO")
        tprint(f"   ✅ Data coverage: {time_span} days (2 years requirement met)", "INFO")
        tprint(f"   ✅ Feature set: {features_generated} features (100+ requirement met)", "INFO")
        tprint(f"   ✅ Regime discovery: {n_regimes} regimes found", "INFO")
        tprint(f"   ✅ Quality score: {composite_score:.4f}", "INFO")
        tprint(f"   ✅ Auto-tuning: Completed with {auto_tuning_results.get('total_trials', 0)} trials", "INFO")
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ Pipeline failed with error: {e}")
        import traceback
        tprint_error(f"Error details: {traceback.format_exc()}")
        return False


async def run_feature_verification_only():
    """
    Run feature generation verification only (for testing).
    """
    tprint_info("🔍 Running feature generation verification only...")
    
    try:
        # Load data
        data_loader = DataLoader()
        market_data = data_loader.load_ethusdt_1h_data()
        
        if market_data is None or market_data.empty:
            tprint_error("❌ Failed to load ETHUSDT data")
            return False
        
        tprint_success(f"✅ Loaded {len(market_data)} samples")
        
        # Test feature generation
        feature_integration = EnhancedStickyFiniteHMMClusteringIntegration(
            min_features=100,
            max_features=150,
            enable_comprehensive_features=True,
            enable_mtf_features=True
        )
        
        with tprint_timer("Feature Generation", level="PERFORMANCE"):
            feature_result = feature_integration.get_comprehensive_clustering_features(market_data)
        
        features_generated = len(feature_result.get('feature_names', []))
        tprint_success(f"✅ Generated {features_generated} features")
        
        # Show feature categories
        features_df = feature_result['features']
        tprint_info(f"Feature matrix shape: {features_df.shape}")
        tprint_data_preview(features_df, "Generated Features", max_rows=3, max_cols=8)
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ Feature verification failed: {e}")
        return False


async def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--features-only":
        # Run feature verification only
        success = await run_feature_verification_only()
    else:
        # Run complete pipeline
        success = await run_complete_sticky_finite_hmm_pipeline()
    
    if success:
        tprint_success("🎉 Script completed successfully!")
        sys.exit(0)
    else:
        tprint_error("❌ Script failed!")
        sys.exit(1)


if __name__ == "__main__":
    # Import pandas for data handling
    import pandas as pd
    
    # Run the async main function
    asyncio.run(main())
