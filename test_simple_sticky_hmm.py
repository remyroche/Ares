"""
Simple test for Sticky Finite HMM with all enhancements.
"""

import asyncio
from src.training.steps.market_analysis.sticky_finite_hmm_clustering import (
    StickyFiniteHMMRegimeDiscoveryStep
)
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error

async def test_with_enhancements():
    """Test with all enhancements enabled."""
    
    tprint("=" * 80, "INFO")
    tprint("🧪 Testing Sticky Finite HMM with All Enhancements", "INFO")
    tprint("=" * 80, "INFO")
    
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'regime_timeframe': '1h',
        'execution_mode': 'blank',
        'enable_auto_tuning': True,  # Enable auto-tuning to find optimal parameters!
        
        # Auto-tuner configuration
        'auto_tuning_params': {
            'n_trials': 20,  # Number of parameter combinations to try
            'timeout': 3600,  # 1 hour timeout
            'n_jobs': 1,  # Sequential (set to -1 for parallel if needed)
            'verbose': True,
            'search_strategy': 'tpe',  # Tree-structured Parzen Estimator (smart search)
        },
        
        # Search space for auto-tuner (will explore these ranges)
        'sticky_finite_hmm_params': {
            'K_min': 4,
            'K_max': 7,
            'n_mixtures': 1,  # Keep fixed at 1 for now
            'kappa_min': 5.0,  # Lower bound for stickiness
            'kappa_max': 30.0,  # Upper bound for stickiness
            'base_alpha_min': 0.2,  # Lower bound for transition sparsity
            'base_alpha_max': 0.8,  # Upper bound for transition sparsity
            'lr_min': 1e-3,
            'lr_max': 1e-1,
            'pca_components_min': 10,
            'pca_components_max': 20,
            'num_iters': 500,
            'min_features': 50,
            'max_features': 100
        }
    }
    
    tprint("", "INFO")
    tprint("🎯 AUTO-TUNING ENABLED - Will search for optimal parameters!", "INFO")
    tprint("", "INFO")
    tprint("Search Space:", "INFO")
    tprint(f"  K: {config['sticky_finite_hmm_params']['K_min']}-{config['sticky_finite_hmm_params']['K_max']} regimes", "INFO")
    tprint(f"  kappa: {config['sticky_finite_hmm_params']['kappa_min']}-{config['sticky_finite_hmm_params']['kappa_max']} (stickiness)", "INFO")
    tprint(f"  base_alpha: {config['sticky_finite_hmm_params']['base_alpha_min']}-{config['sticky_finite_hmm_params']['base_alpha_max']} (transition sparsity)", "INFO")
    tprint(f"  pca_components: {config['sticky_finite_hmm_params']['pca_components_min']}-{config['sticky_finite_hmm_params']['pca_components_max']}", "INFO")
    tprint(f"  Trials: {config['auto_tuning_params']['n_trials']}", "INFO")
    tprint("", "INFO")
    tprint("Enhancements:", "INFO")
    tprint("  ✅ MTF features (4h, 1d context)", "INFO")
    tprint("  ✅ Economic metrics (Sharpe, returns, drawdown)", "INFO")
    tprint("  ✅ Microstructure features (8% weight)", "INFO")
    tprint("  ✅ Volatility quantile differentiation", "INFO")
    tprint("  ✅ Feature Bank integration (50-100 features)", "INFO")
    tprint("", "INFO")
    
    step = StickyFiniteHMMRegimeDiscoveryStep()
    result = await step.execute(config)
    
    if result['success']:
        tprint("", "SUCCESS")
        tprint("=" * 80, "SUCCESS")
        tprint("✅ TEST PASSED - All Enhancements Working!", "SUCCESS")
        tprint("=" * 80, "SUCCESS")
        tprint(f"Regimes Found: {result['n_regimes']}", "SUCCESS")
        tprint(f"Composite Score: {result['composite_score']:.4f}", "SUCCESS")
        
        # Show best parameters if auto-tuning was used
        best_params = result.get('best_params', {})
        if best_params:
            tprint("", "SUCCESS")
            tprint("🏆 Best Parameters Found by Auto-Tuner:", "SUCCESS")
            tprint(f"  K: {best_params.get('K', 'N/A')}", "SUCCESS")
            tprint(f"  kappa: {best_params.get('kappa', 'N/A')}", "SUCCESS")
            tprint(f"  base_alpha: {best_params.get('base_alpha', 'N/A')}", "SUCCESS")
            tprint(f"  lr: {best_params.get('lr', 'N/A')}", "SUCCESS")
            tprint(f"  pca_components: {best_params.get('pca_components', 'N/A')}", "SUCCESS")
        
        # Show regime size distribution
        regime_sizes = result.get('regime_sizes', {})
        if regime_sizes:
            tprint("", "SUCCESS")
            tprint("📊 Regime Distribution:", "SUCCESS")
            total_samples = sum(regime_sizes.values())
            for regime_id, count in sorted(regime_sizes.items()):
                pct = 100.0 * count / total_samples if total_samples > 0 else 0
                tprint(f"  Regime {regime_id}: {count:5d} samples ({pct:5.1f}%)", "SUCCESS")
        
        # Check for economic metrics
        metrics = result.get('metrics', {})
        quality_assessment = metrics.get('quality_assessment', {})
        per_regime = quality_assessment.get('per_regime_metrics', {})
        
        if per_regime:
            tprint("", "SUCCESS")
            tprint("Economic Metrics Per Regime:", "SUCCESS")
            for regime_id, regime_data in sorted(per_regime.items()):  # Show all regimes
                sharpe = regime_data.get('sharpe', 'N/A')
                mean_return = regime_data.get('mean_return', 'N/A')
                max_dd = regime_data.get('max_drawdown', 'N/A')
                tprint(f"  Regime {regime_id}:", "SUCCESS")
                tprint(f"    - Sharpe: {sharpe}", "SUCCESS")
                tprint(f"    - Mean Return: {mean_return}", "SUCCESS")
                tprint(f"    - Max Drawdown: {max_dd}", "SUCCESS")
        
        return True
    else:
        tprint_error(f"❌ TEST FAILED: {result.get('error', 'Unknown error')}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_with_enhancements())
    exit(0 if result else 1)


