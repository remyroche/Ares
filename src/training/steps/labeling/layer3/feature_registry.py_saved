"""
Layer 3 Feature Registry - Single Source of Truth

Centralized definition of all Layer 3 feature patterns for consistency
across all Layer 3 implementations.
"""

from typing import Dict, List

# Centralized Layer 3 feature patterns
LAYER3_FEATURE_PATTERNS = {
    'logit': [
        'logit_prob', 'logit_momentum_5', 'logit_momentum_1'
    ],
    'price_action': [
        'vol_at_signal', 'candle_shape', 'candle_shape_4'
    ],
    'momentum': [
        'momentum_agreement', 'momentum_agreement_abs', 'trend_consistency_12'
    ],
    'technical': [
        'slope_short', 'adx_proxy', 'momentum_short', 'snr'
    ],
    'temporal': [
        'time_since_last_vol_spike', 'time_since_last_large_candle',
        'choppiness_index', 'variance_ratio', 'permutation_entropy'
    ],
    'time_features': [
        'hour', 'day_of_week', 'hour_sin', 'hour_cos', 'is_weekend'
    ],
    'efficiency': [
        'efficiency_ratio', 'price_position_in_range'
    ],
    'layer0': [
        'unified_price_', 'adaptive_filter_', 'noise_reduction_', 'filter_consensus_'
    ],
    'layer1': [
        'layer1_weight_'
    ]
}

# Flatten patterns for easy iteration
ALL_LAYER3_FEATURE_PATTERNS = []
for patterns in LAYER3_FEATURE_PATTERNS.values():
    ALL_LAYER3_FEATURE_PATTERNS.extend(patterns)

# Core features that should always be included
CORE_LAYER3_FEATURES = [
    'volatility_1d'  # Essential for volatility-based calculations
]

def get_layer3_feature_patterns() -> List[str]:
    """Get all Layer 3 feature patterns."""
    return ALL_LAYER3_FEATURE_PATTERNS.copy()

def get_layer3_feature_patterns_by_category() -> Dict[str, List[str]]:
    """Get Layer 3 feature patterns organized by category."""
    return {k: v.copy() for k, v in LAYER3_FEATURE_PATTERNS.items()}

def get_core_layer3_features() -> List[str]:
    """Get core Layer 3 features that should always be included."""
    return CORE_LAYER3_FEATURES.copy()

def validate_layer3_features(df, feature_patterns: List[str] = None) -> Dict[str, any]:
    """
    Validate Layer 3 features against available dataframe columns.
    
    Args:
        df: DataFrame with features
        feature_patterns: List of feature patterns to check (uses default if None)
    
    Returns:
        Dictionary with validation results
    """
    if feature_patterns is None:
        feature_patterns = get_layer3_feature_patterns()
    
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
    for core_feat in get_core_layer3_features():
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

def get_layer3_features_from_dataframe(df) -> List[str]:
    """
    Extract all available Layer 3 features from a dataframe.
    
    Args:
        df: DataFrame with features
    
    Returns:
        List of available Layer 3 feature names
    """
    validation_result = validate_layer3_features(df)
    return validation_result['available_features']
