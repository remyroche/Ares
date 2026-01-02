"""
Layer 4 Feature Registry - Single Source of Truth

Centralized definition of all Layer 4 feature patterns for consistency
across all Layer 4 implementations including position sizing and risk management.
"""

from typing import Dict, List

# Centralized Layer 4 feature patterns
LAYER4_FEATURE_PATTERNS = {
    'performance': [
        'perf_bayesian_psr', 'perf_psr_trend', 'perf_entropy',
        'past_precision', 'avg_prob_product'
    ],
    'regime': [
        'regime_sadf', 'vol_long', 'vol_ratio', 'sadf_score_norm', 
        'cusum_score_norm', 'volatility_zscore', 'volatility_regime'
    ],
    'market': [
        'market_stretch', 'noise_persistence', 'vwap_distance', 'vwap_ratio',
        'relative_strength_ma', 'relative_strength_short', 'efficiency_ratio'
    ],
    'technical': [
        'adx_proxy', 'choppiness_index', 'variance_ratio', 'slope_short'
    ],
    'structural': [
        'drawdown_from_peak', 'distance_from_trough', 'is_near_peak', 'is_near_trough',
        'drawdown_regime_severe', 'drawdown_regime_moderate', 'drawdown_regime_mild', 'drawdown_regime_none'
    ],
    'model': [
        'prediction_dispersion', 'confidence_gap', 'uncertainty', 'prediction_range',
        'avg_divergence', 'max_confidence', 'disagreement_rate', 'snr_internal', 'snr_consensus'
    ],
    'time': [
        'hour_of_day', 'day_of_week', 'hour_sin', 'hour_cos', 
        'is_session_start', 'is_session_end', 'is_weekend'
    ],
    'layer3_inputs': [
        'meta_prob_', 'ensemble_prob', 'max_base_prob', 'min_base_prob',
        'base_prob_range', 'logit_prob', 'logit_momentum_'
    ],
    'contextual': [
        'residual_', 'contextual_', 'harmonized_'
    ],
    'causal': [
        'causal_effect_estimate', 'causal_effect_ci_low', 'causal_effect_ci_high',
        'causal_refutation_score', 'causal_residuals', 'cate_estimates',
        'heterogeneity_score', 'treatment_residuals'
    ]
}

# Flatten patterns for easy iteration
ALL_LAYER4_FEATURE_PATTERNS = []
for patterns in LAYER4_FEATURE_PATTERNS.values():
    ALL_LAYER4_FEATURE_PATTERNS.extend(patterns)

# Core features that should always be included
CORE_LAYER4_FEATURES = [
    'close', 'high', 'low', 'volume',  # OHLCV data
    'volatility_1d', 'primary_ret',    # Essential volatility and returns
    'meta_prob', 'realized_return'     # Layer 3 inputs and targets
]

def get_layer4_feature_patterns() -> List[str]:
    """Get all Layer 4 feature patterns."""
    return ALL_LAYER4_FEATURE_PATTERNS.copy()

def get_layer4_feature_patterns_by_category() -> Dict[str, List[str]]:
    """Get Layer 4 feature patterns organized by category."""
    return {k: v.copy() for k, v in LAYER4_FEATURE_PATTERNS.items()}

def get_core_layer4_features() -> List[str]:
    """Get core Layer 4 features that should always be included."""
    return CORE_LAYER4_FEATURES.copy()

def validate_layer4_features(df, feature_patterns: List[str] = None) -> Dict[str, any]:
    """
    Validate Layer 4 features against available dataframe columns.
    
    Args:
        df: DataFrame with features
        feature_patterns: List of feature patterns to check (uses default if None)
    
    Returns:
        Dictionary with validation results
    """
    if feature_patterns is None:
        feature_patterns = get_layer4_feature_patterns()
    
    available_features = []
    missing_patterns = []
    pattern_counts = {}
    
    for pattern in feature_patterns:
        matching_cols = [c for c in df.columns if pattern in c]
        if matching_cols:
            available_features.extend(matching_cols)
            pattern_counts[pattern] = len(matching_cols)
        else:
            missing_patterns.append(pattern)
    
    # Add core features
    for core_feat in get_core_layer4_features():
        if core_feat in df.columns:
            available_features.append(core_feat)
    
    # Remove duplicates while preserving order
    unique_features = list(dict.fromkeys(available_features))
    
    return {
        'available_features': unique_features,
        'missing_patterns': missing_patterns,
        'pattern_counts': pattern_counts,
        'total_available': len(unique_features),
        'total_patterns': len(feature_patterns),
        'coverage_rate': len(unique_features) / len(feature_patterns) if feature_patterns else 0
    }

def get_layer4_features_from_dataframe(df) -> List[str]:
    """
    Extract all available Layer 4 features from a dataframe.
    
    Args:
        df: DataFrame with features
    
    Returns:
        List of available Layer 4 feature names
    """
    validation_result = validate_layer4_features(df)
    return validation_result['available_features']
