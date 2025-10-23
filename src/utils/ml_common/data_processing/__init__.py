
"""
Lightweight exports for data_processing. Heavy imports are deferred to call sites
to avoid circular imports and import-time side effects.
"""

# Defer heavy modules: expose import helpers instead of importing at package init

def get_regime_processor():
    from .regime_processing import RegimeProcessor
    return RegimeProcessor

def get_feature_preparator():
    from .feature_preparation import FeaturePreparator
    return FeaturePreparator

def get_enhanced_data_labeler():
    from .data_labeling import EnhancedDataLabeler
    return EnhancedDataLabeler

def get_labeling_config():
    try:
        from .data_labeling import TripleBarrierConfig as LabelingConfig
        return LabelingConfig
    except Exception:
        return None

__all__ = [
    'get_regime_processor',
    'get_feature_preparator',
    'get_enhanced_data_labeler',
    'get_labeling_config'
]
