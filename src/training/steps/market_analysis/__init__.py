"""
MARKET_ANALYSIS Module

This module provides comprehensive market analysis functionality including:
- Triple barrier labeling with regime awareness
- HMM regime detection and clustering
- Performance optimization and validation
- Integration with existing market analysis pipeline

Key Components:
- triple_barrier_labeling: Core triple barrier implementation
- regime_aware_triple_barrier_optimizer: Regime-specific optimization
- triple_barrier_validator: Comprehensive validation framework
- enhanced_market_analysis_with_triple_barrier: Integrated pipeline
"""

# Import core triple barrier components from new package
from .triple_barrier_labeling import (
    UnifiedTripleBarrierLabeler,
    TripleBarrierConfig,
    TripleBarrierResult,
    create_triple_barrier_labeler,
    apply_triple_barrier_labeling,
    # Exception classes
    TripleBarrierError,
    ValidationError,
    ConfigurationError,
    HardwareOptimizationError,
    DataQualityError
)

# Legacy compatibility - map old names to new implementation
MarketAnalysisTripleBarrierLabeling = UnifiedTripleBarrierLabeler

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

# Import PID-based feature generation
from .pid_based_feature_generation import (
    PIDBasedFeatureOrchestrator,
    OrchestratorConfig,
    InteractionFeatureGenerator,
    InteractionConfig,
    PolynomialFeatureGenerator,
    PolynomialConfig,
    CrossTimeframeFeatureGenerator,
    CrossTimeframeConfig,
    OptimizedLookbackIntegration
)

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
    "pid_based_feature_generation"
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
            "PID-based feature generation with interaction, polynomial, and cross-timeframe features",
            "Optimized lookback period integration",
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
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 1000),
        'high': np.random.uniform(105, 115, 1000),
        'low': np.random.uniform(95, 105, 1000),
        'close': np.random.uniform(100, 110, 1000),
        'volume': np.random.uniform(1000, 10000, 1000),
        'hmm_regime': np.random.choice([0, 1, 2], 1000, p=[0.4, 0.4, 0.2])
    }, index=dates)
    
    print(f"📊 Created sample data with {len(data)} samples")
    print(f"   → Time range: {data.index[0]} to {data.index[-1]}")
    print(f"   → Regimes: {data['hmm_regime'].value_counts().to_dict()}")
    
    # Apply triple barrier labeling
    print("\n🏷️ Applying triple barrier labeling...")
    result = apply_triple_barrier_labeling(data)
    
    if result.success:
        labeled_data = result.labeled_data
        print(f"✅ Labeling completed:")
        print(f"   → Labeled samples: {result.total_labels_generated}")
        print(f"   → Label distribution: {result.label_distribution}")
        print(f"   → Quality score: {result.data_quality_score:.2%}")
        print(f"   → Execution time: {result.execution_duration:.2f}s")
    else:
        print(f"❌ Labeling failed: {result.error_message}")
        return None
    
    # Calculate basic metrics
    if 'net_profit_pct' in labeled_data.columns:
        profits = labeled_data['net_profit_pct']
        win_rate = (profits > 0).mean()
        avg_profit = profits.mean()
        sharpe_ratio = profits.mean() / profits.std() * np.sqrt(252) if profits.std() > 0 else 0
        
        print(f"💰 Performance metrics:")
        print(f"   → Win rate: {win_rate:.3f}")
        print(f"   → Average profit: {avg_profit:.4f}")
        print(f"   → Sharpe ratio: {sharpe_ratio:.3f}")
    
    # Validate results
    print("\n🔍 Validating results...")
    is_valid = result.validation_passed
    print(f"✅ Validation result: {'PASSED' if is_valid else 'FAILED'}")
    if result.validation_warnings:
        print(f"   → Warnings: {len(result.validation_warnings)}")
    
    print("\n🎉 Quick start example completed successfully!")
    print("\nFor more advanced usage, see:")
    print("   → TRIPLE_BARRIER_DOCUMENTATION.md")
    print("   → triple_barrier_labeling/test_unified_labeler.py")
    
    return labeled_data

# Export all public components
__all__ = [
    # Core triple barrier components (new unified implementation)
    "UnifiedTripleBarrierLabeler",
    "TripleBarrierConfig",
    "TripleBarrierResult",
    "create_triple_barrier_labeler",
    "apply_triple_barrier_labeling",
    
    # Exception classes
    "TripleBarrierError",
    "ValidationError",
    "ConfigurationError",
    "HardwareOptimizationError",
    "DataQualityError",
    
    # Legacy compatibility
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
    
    # PID-based feature generation
    "PIDBasedFeatureOrchestrator",
    "OrchestratorConfig",
    "InteractionFeatureGenerator",
    "InteractionConfig",
    "PolynomialFeatureGenerator",
    "PolynomialConfig",
    "CrossTimeframeFeatureGenerator",
    "CrossTimeframeConfig",
    "OptimizedLookbackIntegration",
    
    # Utility functions
    "get_module_info",
    "quick_start_example"
]