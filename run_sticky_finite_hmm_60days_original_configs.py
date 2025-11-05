#!/usr/bin/env python3
"""
Complete Sticky Finite HMM Pipeline Runner - 60 Days Version

This script runs the complete Sticky Finite HMM system with:
1. 60 days of ETHUSDT data (instead of 2 years)
2. Full feature set (100+ features) via enhanced_sticky_finite_hmm_clustering_integration
3. Auto-tuning via sticky_finite_hmm_auto_tuner
4. Comprehensive regime discovery and analysis

All other configs remain the same as the original complete script.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
from datetime import datetime, timedelta
import pandas as pd

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


def filter_data_by_days(data: pd.DataFrame, days: int = 60) -> pd.DataFrame:
    """Filter DataFrame to last N days."""
    if data is None or data.empty:
        return data
    
    # Ensure timestamp column exists and is datetime
    if 'timestamp' not in data.columns:
        return data
    
    # Convert timestamp to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Sort by timestamp
    data = data.sort_values('timestamp').reset_index(drop=True)
    
    # Filter to last N days
    cutoff_date = data['timestamp'].max() - timedelta(days=days)
    filtered_data = data[data['timestamp'] >= cutoff_date].reset_index(drop=True)
    
    print(f"📊 Filtered to {days} days: {len(filtered_data)} rows (from {len(data)} total)")
    return filtered_data


async def run_complete_sticky_finite_hmm_pipeline_60_days():
    """
    Run the complete Sticky Finite HMM pipeline with all components, but only 60 days of data.
    
    Pipeline:
    1. Load 60 days of ETHUSDT data
    2. Generate 100+ comprehensive features
    3. Run auto-tuning to find optimal parameters
    4. Perform regime discovery with Sticky Finite HMM
    5. Generate comprehensive reports
    """
    
    tprint("=" * 100, "INFO")
    tprint("🚀 COMPLETE STICKY FINITE HMM PIPELINE - 60 DAYS", "INFO")
    tprint("=" * 100, "INFO")
    tprint("📊 Components:", "INFO")
    tprint("   1. StickyFiniteHMMRegimeDiscoveryStep - Data loading & orchestration", "INFO")
    tprint("   2. EnhancedStickyFiniteHMMClusteringIntegration - 100+ features", "INFO")
    tprint("   3. StickyFiniteHMMAutoTuner - Hyperparameter optimization", "INFO")
    tprint("   4. Real ETHUSDT data (60 days) - Filtered from full dataset", "INFO")
    tprint("=" * 100, "INFO")
    
    start_time = time.time()
    
    try:
        # Step 1: Initialize the regime discovery step
        tprint_info("🔧 Step 1: Initializing Sticky Finite HMM Regime Discovery Step")
        
        regime_discovery_step = StickyFiniteHMMRegimeDiscoveryStep(
            step_name="sticky_finite_hmm_60_days_pipeline"
        )
        
        # Step 2: Load and filter ETHUSDT data (60 days)
        tprint_info("📥 Step 2: Loading and filtering 60 days of ETHUSDT data")
        
        # Load full data first
        data_loader = DataLoader()
        market_data = data_loader.load_ethusdt_1h_data()
        
        if market_data is None or market_data.empty:
            tprint_error("❌ Failed to load ETHUSDT data. Please ensure data is available in historical_data/binance/ethusdt/")
            return False
        
        # Filter to 60 days
        market_data = filter_data_by_days(market_data, days=60)
        
        tprint_success(f"✅ Loaded {len(market_data)} samples of ETHUSDT data (60 days)")
        tprint_data_preview(market_data, "Market Data", max_rows=3, max_cols=6)
        
        # Check data coverage
        if 'timestamp' in market_data.columns:
            market_data['datetime'] = pd.to_datetime(market_data['timestamp'], unit='ms')
            time_span = (market_data['datetime'].max() - market_data['datetime'].min()).days
            tprint_info(f"📅 Data spans {time_span} days")
            
            if time_span < 50:
                tprint_warning(f"⚠️ Data covers only {time_span} days (expected: ~60 days)")
            else:
                tprint_success(f"✅ Data covers {time_span} days (meets 60-day requirement)")
        
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
        
        # Step 4: Configure with all original settings but override data loading
        tprint_info("🎯 Step 4: Running complete Sticky Finite HMM pipeline with auto-tuning")
        
        # Use the exact same config as the original complete script
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
                'tpe_trials': 50,              # 50 TPE trials (same as original)
                'timeout': 1800                # 30 min timeout (same as original)
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
        
        # Override the _load_market_data method to use our filtered 60-day data
        original_load = regime_discovery_step._load_market_data
        regime_discovery_step._load_market_data = lambda symbol, exchange, timeframe, cfg: market_data
        
        # Execute the full pipeline
        result = await regime_discovery_step.execute(config)
        
        # Restore original method
        regime_discovery_step._load_market_data = original_load
        
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
            "Data Duration": "60 days",
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
                for key, value in list(best_params.items())[:5]:  # Show first 5 params
                    tprint(f"     {key}: {value}", "SUCCESS")
        
        tprint("", "SUCCESS")
        tprint("📊 Pipeline completed with all original configurations (except 60-day data)", "SUCCESS")
        tprint("=" * 80, "SUCCESS")
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ Pipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main entry point."""
    try:
        success = await run_complete_sticky_finite_hmm_pipeline_60_days()
        if success:
            tprint_success("🎉 60-day Sticky Finite HMM pipeline completed successfully!")
            sys.exit(0)
        else:
            tprint_error("❌ 60-day Sticky Finite HMM pipeline failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        tprint_warning("⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        tprint_error(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
