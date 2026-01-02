"""
Enhanced Modern De Prado Framework Configuration

Complete configuration template for the fully enhanced modern De Prado framework
with comprehensive metrics, regime/liquidity features, uncertainty quantification,
and temporal conflict detection.

Usage:
    from enhanced_de_prado_config import ENHANCED_CONFIG
    config.update(ENHANCED_CONFIG)
"""

# =============================================================================
# ENHANCED MODERN DE PRADO FRAMEWORK CONFIGURATION
# =============================================================================

ENHANCED_CONFIG = {
    # =============================================================================
    # MASTER SWITCHES
    =============================================================================
    'enable_causal_framework': True,  # Enable entire modern De Prado framework
    
    # =============================================================================
    # COMPREHENSIVE METRICS
    =============================================================================
    'comprehensive_metrics_enabled': True,  # Enable comprehensive metrics reporting
    'layer2_metrics_enabled': True,        # Layer 2 discovery and engineering metrics
    'layer25_metrics_enabled': True,       # Layer 2.5 Chaser metrics
    'layer3_metrics_enabled': True,        # Layer 3 meta-learner metrics
    
    # =============================================================================
    # REGIME & LIQUIDITY FEATURES
    =============================================================================
    'regime_features_enabled': True,        # Enable market regime features
    'liquidity_features_enabled': True,     # Enable liquidity dynamics features
    'cross_asset_features_enabled': True,   # Enable cross-asset correlation features
    
    # Regime Feature Configuration
    'regime_volatility_windows': [10, 20, 50],
    'regime_trend_windows': [20, 50, 100],
    'regime_volume_windows': [10, 20, 50],
    
    # Liquidity Feature Configuration
    'liquidity_windows': [5, 10, 20],
    'liquidity_spread_enabled': True,
    'liquidity_vwap_enabled': True,
    'liquidity_impact_enabled': True,
    
    # =============================================================================
    # BAYESIAN CAUSAL DISCOVERY
    =============================================================================
    'use_bayesian_discovery': True,         # Enable Bayesian causal discovery
    'bayesian_n_bootstrap': 50,             # Number of bootstrap samples
    'bayesian_confidence_level': 0.95,      # Confidence level for intervals
    'bayesian_fallback_to_deterministic': True,  # Fallback if Bayesian fails
    
    # =============================================================================
    # UNCERTAINTY QUANTIFICATION
    =============================================================================
    'uncertainty_quantification_enabled': True,  # Enable uncertainty quantification
    'treatment_uncertainty_enabled': True,       # Treatment effect uncertainty
    'specialist_uncertainty_enabled': True,     # Specialist prediction uncertainty
    'uncertainty_n_bootstrap': 100,               # Bootstrap samples for uncertainty
    
    # =============================================================================
    # TEMPORAL CONFLICT DETECTION
    =============================================================================
    'temporal_conflict_detection_enabled': True,  # Enable temporal conflict analysis
    'conflict_rolling_windows': [5, 10, 20],      # Rolling window sizes
    'conflict_regime_analysis_enabled': True,     # Regime-based conflict analysis
    'conflict_forecasting_enabled': True,         # Conflict predictability analysis
    
    # =============================================================================
    # ENHANCED LAYER 2.5 CHASER
    =============================================================================
    'layer25_chaser_enabled': True,              # Enable Layer 2.5 Chaser
    'layer25_enhanced_features_enabled': True,     # Enable regime/liquidity features
    'layer25_uncertainty_enabled': True,          # Enable uncertainty in Chaser
    'layer25_temporal_conflicts_enabled': True,   # Enable temporal conflict detection
    
    # Chaser Feature Configuration
    'layer25_max_regime_features': 33,           # Maximum regime features to use
    'layer25_max_liquidity_features': 45,        # Maximum liquidity features to use
    'layer25_feature_selection_method': 'importance',  # 'importance', 'correlation', 'all'
    
    # =============================================================================
    # ENHANCED LAYER 3 META-LEARNER
    =============================================================================
    'layer3_enhanced_features_enabled': True,    # Enable enhanced features in Layer 3
    'layer3_temporal_conflict_features': True,   # Add temporal conflict as features
    'layer3_regime_liquidity_features': True,     # Add regime/liquidity to meta-learner
    'layer3_comprehensive_metrics': True,         # Enable comprehensive Layer 3 metrics
    
    # =============================================================================
    # FEATURE COUNTS AND LIMITS
    =============================================================================
    # Layer 2.5 Chaser Feature Limits
    'layer25_max_total_features': 100,           # Maximum total features for Chaser
    'layer25_min_feature_importance': 0.001,    # Minimum importance threshold
    
    # Layer 3 Meta-Learner Feature Limits
    'layer3_max_total_features': 150,           # Maximum total features for meta-learner
    'layer3_feature_expansion_limit': 5.0,       # Maximum feature expansion ratio
    
    # =============================================================================
    # PERFORMANCE AND EFFICIENCY
    =============================================================================
    'enhanced_framework_fast_mode': False,       # Fast mode for testing (reduced features)
    'enhanced_framework_parallel_processing': True,  # Enable parallel processing
    'enhanced_framework_caching_enabled': True,   # Enable feature caching
    
    # =============================================================================
    # REPORTING AND LOGGING
    =============================================================================
    'enhanced_framework_verbose': True,           # Enhanced verbose logging
    'comprehensive_metrics_report_enabled': True,  # Generate comprehensive reports
    'uncertainty_report_enabled': True,           # Include uncertainty in reports
    'temporal_analysis_report_enabled': True,    # Include temporal analysis in reports
    
    # =============================================================================
    # QUALITY CONTROL
    =============================================================================
    'feature_quality_validation_enabled': True,   # Validate feature quality
    'uncertainty_quality_threshold': 0.5,         # Minimum uncertainty quality
    'conflict_quality_threshold': 0.3,            # Minimum conflict detection quality
    
    # =============================================================================
    # EXPERIMENTAL FEATURES
    =============================================================================
    'experimental_cross_asset_learning': False,  # Cross-asset learning (experimental)
    'experimental_adaptive_causal_learning': False,  # Adaptive causal graphs (experimental)
    'experimental_ensemble_uncertainty': False,   # Advanced ensemble uncertainty (experimental)
}

# =============================================================================
# PRODUCTION CONFIGURATION (Optimized for Speed and Stability)
# =============================================================================
PRODUCTION_CONFIG = {
    **ENHANCED_CONFIG,
    
    # Optimized for production
    'bayesian_n_bootstrap': 25,              # Reduced for speed
    'uncertainty_n_bootstrap': 50,           # Reduced for speed
    'layer25_max_total_features': 75,        # Reduced for speed
    'layer3_max_total_features': 100,        # Reduced for speed
    'enhanced_framework_fast_mode': True,     # Enable fast optimizations
    'comprehensive_metrics_report_enabled': False,  # Reduce reporting overhead
}

# =============================================================================
# RESEARCH CONFIGURATION (Maximum Features and Analysis)
# =============================================================================
RESEARCH_CONFIG = {
    **ENHANCED_CONFIG,
    
    # Maximum features for research
    'bayesian_n_bootstrap': 200,             # Maximum bootstrap samples
    'uncertainty_n_bootstrap': 200,           # Maximum uncertainty samples
    'layer25_max_total_features': 150,       # Maximum Chaser features
    'layer3_max_total_features': 200,        # Maximum meta-learner features
    'enhanced_framework_fast_mode': False,    # Full feature set
    'experimental_cross_asset_learning': True,  # Enable experimental features
    'experimental_adaptive_causal_learning': True,  # Enable experimental features
}

# =============================================================================
# MINIMAL CONFIGURATION (Fastest Deployment)
# =============================================================================
MINIMAL_CONFIG = {
    **ENHANCED_CONFIG,
    
    # Minimal configuration for speed
    'bayesian_n_bootstrap': 10,              # Minimal bootstrap
    'uncertainty_n_bootstrap': 25,           # Minimal uncertainty samples
    'layer25_max_total_features': 50,        # Reduced features
    'layer3_max_total_features': 75,         # Reduced features
    'regime_features_enabled': False,        # Disable regime features
    'liquidity_features_enabled': False,     # Disable liquidity features
    'temporal_conflict_detection_enabled': False,  # Disable temporal analysis
    'comprehensive_metrics_enabled': False,   # Disable comprehensive metrics
    'enhanced_framework_fast_mode': True,     # Enable fast mode
}

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def get_enhanced_config(config_type: str = "enhanced"):
    """
    Get enhanced configuration by type.
    
    Args:
        config_type: Type of configuration ('enhanced', 'production', 'research', 'minimal')
    
    Returns:
        Configuration dictionary
    """
    configs = {
        "enhanced": ENHANCED_CONFIG,
        "production": PRODUCTION_CONFIG,
        "research": RESEARCH_CONFIG,
        "minimal": MINIMAL_CONFIG
    }
    
    return configs.get(config_type, ENHANCED_CONFIG)

def print_enhanced_feature_counts():
    """Print expected feature counts for enhanced configuration."""
    print("=== Enhanced Modern De Prado Framework Feature Counts ===")
    print()
    
    print("ENHANCED CONFIG:")
    print("  Layer 2.5 Chaser: Up to 100 features")
    print("    - Regime features: Up to 33")
    print("    - Liquidity features: Up to 45")
    print("    - Original features: ~30")
    print("    - Cross-asset features: ~15")
    print("  Layer 3 Meta-Learner: Up to 150 features")
    print("    - Base features: ~30")
    print("    - Chaser features: ~20")
    print("    - Streamlined features: 8")
    print("    - Minimal features: 3-4")
    print("    - Temporal conflict features: ~10")
    print("  Total: Up to 250 enhanced features")
    print()
    
    print("PRODUCTION CONFIG:")
    print("  Layer 2.5 Chaser: Up to 75 features")
    print("  Layer 3 Meta-Learner: Up to 100 features")
    print("  Total: Up to 175 enhanced features")
    print()
    
    print("RESEARCH CONFIG:")
    print("  Layer 2.5 Chaser: Up to 150 features")
    print("  Layer 3 Meta-Learner: Up to 200 features")
    print("  Total: Up to 350 enhanced features")
    print()
    
    print("MINIMAL CONFIG:")
    print("  Layer 2.5 Chaser: Up to 50 features")
    print("  Layer 3 Meta-Learner: Up to 75 features")
    print("  Total: Up to 125 enhanced features")

if __name__ == "__main__":
    print_enhanced_feature_counts()
    
    print("\n=== Enhanced Configuration Usage ===")
    print("from enhanced_de_prado_config import ENHANCED_CONFIG")
    print("config.update(ENHANCED_CONFIG)")
    print()
    print("# Or use get_enhanced_config function:")
    print("from enhanced_de_prado_config import get_enhanced_config")
    print("config = get_enhanced_config('production')")
