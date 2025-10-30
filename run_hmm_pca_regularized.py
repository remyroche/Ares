#!/usr/bin/env python3
"""
Run HMM Regime Discovery with PCA + Covariance Regularization
Implements user's recommendations for production-ready regime discovery
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.steps.market_analysis.hmm_clustering.hmm_regime_discovery_step import create_hmm_regime_discovery_step
from src.utils.tprint import tprint

async def main():
    """Run HMM regime discovery with PCA + regularization."""
    
    tprint("=" * 80, "INFO")
    tprint("HMM WITH PCA + COVARIANCE REGULARIZATION", "INFO")
    tprint("=" * 80, "INFO")
    
    # Configuration
    symbol = 'ETHUSDT'
    exchange = 'binance'
    timeframe = '1h'
    
    tprint(f"📊 Target: {symbol} ({exchange}) - {timeframe}", "INFO")
    tprint(f"🎯 Improvements:", "INFO")
    tprint(f"  ✅ PCA enabled (target: 10 components, ≥60% variance)", "INFO")
    tprint(f"  ✅ Covariance regularization (diag instead of full)", "INFO")
    tprint(f"  ✅ Tiny regime merging (Mahalanobis distance)", "INFO")
    tprint(f"  ✅ Block bootstrap CIs (500 iterations)", "INFO")
    
    # Create HMM step with REGULARIZED parameters
    hmm_step = create_hmm_regime_discovery_step(
        n_states=4,  # 4 regimes for interpretability
        correlation_threshold=0.85,
        random_state=42,
        covariance_type='diag',  # REGULARIZED (was 'full')
        n_iter=100,
        min_regime_pct=0.05,  # Min 5% of samples per regime
        min_regime_samples=50,  # Min 50 samples per regime
        merge_tiny_regimes=True,
        bootstrap_iterations=500,  # 500 bootstrap iterations
        confidence_level=0.95
    )
    
    # Execute discovery
    tprint("\n🚀 Starting HMM regime discovery...", "INFO")
    start_time = datetime.now()
    
    context = {
        'symbol': symbol,
        'exchange': exchange,
        'timeframe': timeframe,
        'execution_mode': 'blank'  # Use blank mode to load all data
    }
    
    results = await hmm_step.execute(context)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Display results
    tprint("\n" + "=" * 80, "INFO")
    tprint("EXECUTION COMPLETE", "SUCCESS")
    tprint("=" * 80, "INFO")
    
    if results.get('success', False):
        tprint(f"✅ HMM Discovery Successful", "SUCCESS")
        tprint(f"⏱️  Total Time: {duration:.2f} seconds", "INFO")
        tprint(f"📊 Number of Regimes: {results.get('n_regimes', 'N/A')}", "INFO")
        tprint(f"📈 Number of Samples: {len(results.get('regime_labels', []))}", "INFO")
        
        quality = results.get('quality_metrics')
        if quality:
            tprint(f"🎯 Quality Score: {quality.quality_score:.3f}", "INFO")
            tprint(f"📊 Silhouette Score: {quality.silhouette_score:.3f}", "INFO")
            tprint(f"⚖️  Balance Score: {quality.balance_score:.3f}", "INFO")
        
        # Show regime distribution
        import numpy as np
        regime_labels = results.get('regime_labels', [])
        if len(regime_labels) > 0:
            unique, counts = np.unique(regime_labels, return_counts=True)
            tprint("\n📊 Regime Distribution:", "INFO")
            for label, count in zip(unique, counts):
                pct = (count / len(regime_labels)) * 100
                tprint(f"   Regime {label}: {count:,} samples ({pct:.1f}%)", "INFO")
        
        # Show economic metrics
        economic = results.get('economic_metrics', {})
        if economic:
            tprint("\n💰 Economic Performance:", "INFO")
            for regime_id, metrics in economic.items():
                sharpe = metrics.get('sharpe', 0.0)
                win_rate = metrics.get('win_rate', 0.0) * 100
                n_samples = metrics.get('n_samples', 0)
                bootstrap = metrics.get('bootstrap_ci', {})
                reliable = bootstrap.get('reliable', False)
                
                status = "✅ RELIABLE" if reliable else "⚠️  UNRELIABLE"
                tprint(f"   Regime {regime_id} ({status}): Sharpe={sharpe:.2f}, WinRate={win_rate:.1f}%, N={n_samples}", 
                       "SUCCESS" if reliable and sharpe > 0 else "WARNING")
        
        # Show tradeable regimes
        tradeable = results.get('tradeable_regimes', {})
        if tradeable:
            tprint("\n🎯 Tradeable Regimes:", "INFO")
            for regime_id, status in tradeable.items():
                emoji = "🟢" if status == 'LONG' else "🟡" if status == 'FLAT' else "🔴"
                tprint(f"   {emoji} Regime {regime_id}: {status}", "SUCCESS" if status == 'LONG' else "WARNING")
        
        tprint("\n📄 Report generated in outcomes/hmm_regime_discovery_ETHUSDT/", "SUCCESS")
        
    else:
        tprint(f"❌ HMM Discovery Failed: {results.get('error', 'Unknown error')}", "ERROR")
    
    tprint("\n" + "=" * 80, "INFO")

if __name__ == "__main__":
    asyncio.run(main())

