"""
MARKET_ANALYSIS Module

This module provides comprehensive market analysis functionality including:
- Multi-horizon profit labeling for enhanced ML training
- HMM regime detection and clustering
- Performance optimization and validation
- Integration with existing market analysis pipeline

Key Components:
- multi_horizon_profit_labeler: Core multi-horizon implementation
- multi_horizon_sub_pipeline_adapter: Sub-pipeline integration
- gradient_flow_analysis: Performance analysis framework
- feature_lookback_optimization: Enhanced optimization system
"""

# Import core multi-horizon components (NEW SYSTEM)
from ..pre_training.multi_horizon_profit_labeler import (
    MultiHorizonProfitLabeler,
    MultiHorizonConfig
)

# Import sub-pipeline adapter
from .multi_horizon_sub_pipeline_adapter import (
    MultiHorizonSubPipelineAdapter,
    execute_multi_horizon_labeling_step
)

# Import gradient flow analysis
from .gradient_flow_analysis import (
    GradientFlowAnalyzer,
    GradientFlowAnalysis,
    analyze_gradient_flow_benefits
)

# Legacy compatibility - DEPRECATED but maintained for backward compatibility
try:
    from ..pre_training.multi_horizon_profit_labeler import (
        MultiHorizonProfitLabeler,
        MultiHorizonConfig
    )
    LEGACY_TRIPLE_BARRIER_AVAILABLE = True
except ImportError:
    LEGACY_TRIPLE_BARRIER_AVAILABLE = False

# Legacy compatibility mapping
if LEGACY_TRIPLE_BARRIER_AVAILABLE:
    MarketAnalysisTripleBarrierLabeling = MultiHorizonProfitLabeler

# Import regime-aware optimizer
from .regime_aware_triple_barrier_optimizer import (
    RegimeAwareTripleBarrierOptimizer,
    RegimeBarrierParams,
    RegimePerformanceMetrics,
    optimize_regime_barriers,
    apply_optimized_regime_labeling
)

# Import validation framework
from .triple_barrier_validator import (
    TripleBarrierValidator,
    ValidationResult,
    ValidationReport,
    validate_triple_barrier_implementation,
    quick_validate_triple_barrier
)

# Import enhanced pipeline
from .enhanced_market_analysis_with_triple_barrier import (
    EnhancedMarketAnalysisWithTripleBarrier,
    MarketAnalysisTripleBarrierConfig,
    run_enhanced_market_analysis_with_triple_barrier,
    quick_triple_barrier_analysis
)

# Import Interactive Feature Generation (replaces legacy PID-based system)
# Deferred import to avoid circular dependency - import when needed instead of at module level
InteractiveFeatureGenerationComponent = None  # Lazy-loaded when needed

# Hardware optimizations are now integrated into the main triple_barrier_labeling module

# Version information
__version__ = "1.0.0"
__author__ = "Market Analysis Team"
__email__ = "market-analysis@example.com"

# Module-level constants
MODULE_NAME = "MARKET_ANALYSIS"
COMPONENTS = [
    "triple_barrier_labeling",
    "regime_aware_triple_barrier_optimizer", 
    "triple_barrier_validator",
    "enhanced_market_analysis_with_triple_barrier",
    "interactive_feature_generation"
]

# Convenience function to get module information
def get_module_info():
    """Get information about the MARKET_ANALYSIS module."""
    return {
        "name": MODULE_NAME,
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "components": COMPONENTS,
        "description": "Comprehensive market analysis with triple barrier labeling",
        "features": [
            "Triple barrier labeling with regime awareness",
            "HMM regime detection and clustering",
            "Performance optimization with Numba acceleration",
            "Integrated hardware optimizations for M1/M2/M3 Macs",
            "Advanced memory management and optimization",
            "GPU acceleration support (MPS)",
            "Comprehensive validation framework",
            "Seamless pipeline integration",
            "Transaction cost modeling",
            "Binary and ternary classification support",
            "Interactive feature generation with optimized lookbacks and cross-timeframe coverage",
            "Data-driven Bayesian lookback optimization",
            "Matrix operations for hardware-optimized computations"
        ]
    }

# Quick start function
def quick_start_example():
    """Provide a quick start example for the MARKET_ANALYSIS module."""
    import pandas as pd
    import numpy as np
    
    print("🚀 MARKET_ANALYSIS Quick Start Example")
    print("=" * 50)
    
    # Create sample data with valid OHLC relationships
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    
    # Generate realistic OHLC data
    base_price = 100
    prices = []
    current_price = base_price
    
    for i in range(1000):
        # Random walk with some trend
        change = np.random.normal(0, 0.001)  # 0.1% volatility
        current_price *= (1 + change)
        
        # Generate OHLC with proper relationships
        open_price = current_price
        close_price = open_price * (1 + np.random.normal(0, 0.0005))
        
        # High is always >= max(open, close)
        high_price = max(open_price, close_price) * (1 + abs(np.random.uniform(0, 0.002)))
        
        # Low is always <= min(open, close)
        low_price = min(open_price, close_price) * (1 - abs(np.random.uniform(0, 0.002)))
        
        prices.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': np.random.uniform(1000, 10000),
            'hmm_regime': np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        })
    
    data = pd.DataFrame(prices, index=dates)
    
    print(f"📊 Created sample data with {len(data)} samples")
    print(f"   → Time range: {data.index[0]} to {data.index[-1]}")
    print(f"   → Regimes: {data['hmm_regime'].value_counts().to_dict()}")
    
    # Apply multi-horizon labeling
    print("\n🎯 Applying multi-horizon profit labeling...")
    result = apply_multi_horizon_labeling(data)
    
    if isinstance(result, pd.DataFrame) and len(result) > 0:
        labeled_data = result
        print(f"✅ Labeling completed:")
        print(f"   → Labeled samples: {len(labeled_data)}")
        print(f"   → Total features: {labeled_data.shape[1]}")
        print(f"   → New probability targets: {len([c for c in labeled_data.columns if c.endswith('_prob')])}")
        
        # Show sample opportunities
        if 'overall_opportunity' in labeled_data.columns:
            overall_opp = labeled_data['overall_opportunity'].dropna()
            print(f"   → Average opportunity score: {overall_opp.mean():.3f}")
            print(f"   → High opportunity samples: {(overall_opp > 0.7).sum()} ({(overall_opp > 0.7).sum()/len(overall_opp)*100:.1f}%)")
    else:
        print(f"❌ Labeling failed or returned empty result")
        return None
    
    # Calculate multi-horizon metrics
    probability_columns = [col for col in labeled_data.columns if col.endswith('_prob')]
    if probability_columns:
        print(f"💰 Multi-horizon metrics:")
        
        # Average probabilities by target type
        for target_type in ['micro', 'small', 'medium', 'good']:
            target_cols = [col for col in probability_columns if col.startswith(f'{target_type}_')]
            if target_cols:
                avg_prob = labeled_data[target_cols].mean().mean()
                print(f"   → {target_type.capitalize()} target avg probability: {avg_prob:.3f}")
        
        # Composite scores
        if 'leverage_adjusted_score' in labeled_data.columns:
            leverage_score = labeled_data['leverage_adjusted_score'].mean()
            print(f"   → Leverage-adjusted score: {leverage_score:.3f}")
        
        if 'reversal_capture_score' in labeled_data.columns:
            reversal_score = labeled_data['reversal_capture_score'].mean()
            print(f"   → Reversal capture score: {reversal_score:.3f}")
    
    # Validate results
    print("\n🔍 Validating results...")
    has_probabilities = len(probability_columns) > 0
    has_opportunities = 'overall_opportunity' in labeled_data.columns
    print(f"✅ Validation result: {'PASSED' if has_probabilities and has_opportunities else 'FAILED'}")
    if has_probabilities:
        print(f"   → Probability targets: {len(probability_columns)}")
    if has_opportunities:
        print(f"   → Opportunity scoring: Available")
    
    print("\n🎉 Quick start example completed successfully!")
    print("\nFor more advanced usage, see:")
    print("   → multi_horizon_profit_labeler.py")
    print("   → gradient_flow_analysis.py")
    print("   → multi_horizon_sub_pipeline_adapter.py")
    
    return labeled_data

# Export all public components
__all__ = [
    # Core multi-horizon components (NEW SYSTEM)
    "MultiHorizonProfitLabeler",
    "MultiHorizonConfig", 
    
    # Sub-pipeline integration
    "MultiHorizonSubPipelineAdapter",
    "execute_multi_horizon_labeling_step",
    
    # Analysis components
    "GradientFlowAnalyzer",
    "GradientFlowAnalysis", 
    "analyze_gradient_flow_benefits",
    
    # Legacy triple barrier components (DEPRECATED - for backward compatibility only)
    "MultiHorizonProfitLabeler",
    "MultiHorizonConfig",
    "MarketAnalysisTripleBarrierLabeling",
    
    # Regime-aware optimizer
    "RegimeAwareTripleBarrierOptimizer",
    "RegimeBarrierParams",
    "RegimePerformanceMetrics",
    "optimize_regime_barriers",
    "apply_optimized_regime_labeling",
    
    # Validation framework
    "TripleBarrierValidator",
    "ValidationResult",
    "ValidationReport", 
    "validate_triple_barrier_implementation",
    "quick_validate_triple_barrier",
    
    # Enhanced pipeline
    "EnhancedMarketAnalysisWithTripleBarrier",
    "MarketAnalysisTripleBarrierConfig",
    "run_enhanced_market_analysis_with_triple_barrier",
    "quick_triple_barrier_analysis",
    
    # Interactive feature generation (replaces legacy PID system)
    "InteractiveFeatureGenerationComponent",
    
    # Utility functions
    "get_module_info",
    "quick_start_example"
]