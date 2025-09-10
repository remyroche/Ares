"""
Compatibility module for step08_advanced_feature_selection imports in data collection.

This module provides backwards compatibility for imports that were previously
available in the deleted step08_advanced_feature_selection.py file in data collection.

All functionality has been moved to ml_commons.
"""

import warnings
import logging
from typing import Any, Dict, List, Optional, Union

# Setup logging
logger = logging.getLogger(__name__)

logger.info('🔧 Loading step08_advanced_feature_selection compatibility module...')

# Import the actual functionality from the new locations
try:
    logger.info('📦 Attempting to import UnifiedFeatureSelectionManager...')
    from src.utils.ml_common.feature_selection import (
        UnifiedFeatureSelectionManager,
        AdvancedFeatureSelector
    )
    FUNCTIONALITY_AVAILABLE = True
    logger.info('✅ Successfully imported UnifiedFeatureSelectionManager')
except ImportError as e:
    FUNCTIONALITY_AVAILABLE = False
    logger.error(f'❌ Failed to import Step08 functionality: {e}')
    logger.warning(f'⚠️ Step08 functionality not available: {e}')
    warnings.warn(f"Step08 functionality not available: {e}")


class Step08AdvancedFeatureSelection:
    """
    Compatibility class for Step08AdvancedFeatureSelection in data collection.
    
    This class provides backwards compatibility for the Step08AdvancedFeatureSelection
    that was previously available in data collection step08_advanced_feature_selection.py.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the advanced feature selection."""
        logger.info('🔧 Initializing Step08AdvancedFeatureSelection compatibility class...')
        logger.info(f'📊 Configuration provided: {config is not None}')
        
        if not FUNCTIONALITY_AVAILABLE:
            logger.error('❌ Step08 functionality not available')
            raise ImportError("Step08 functionality not available. Please ensure ml_commons is properly installed.")
        
        self.config = config or {}
        logger.info(f'📊 Configuration keys: {list(self.config.keys()) if self.config else "None"}')
        
        logger.info('🔧 Creating UnifiedFeatureSelectionManager instance...')
        self.feature_selector = UnifiedFeatureSelectionManager(self.config)
        logger.info('✅ UnifiedFeatureSelectionManager instance created successfully')
        
        logger.warning('⚠️ Step08AdvancedFeatureSelection is deprecated - use UnifiedFeatureSelectionManager from src.utils.ml_common.feature_selection instead')
        warnings.warn(
            "Step08AdvancedFeatureSelection is deprecated. Use UnifiedFeatureSelectionManager from src.utils.ml_common.feature_selection instead.",
            DeprecationWarning,
            stacklevel=2
        )
    
    def select_features(self, X: Any, y: Any, **kwargs) -> Any:
        """Select features using the unified feature selection manager."""
        logger.info('🚀 Executing advanced feature selection...')
        logger.info(f'📊 X data type: {type(X)}')
        logger.info(f'📊 y data type: {type(y)}')
        logger.info(f'📊 Additional kwargs: {list(kwargs.keys()) if kwargs else "None"}')
        
        if hasattr(X, 'shape'):
            logger.info(f'📊 X shape: {X.shape}')
        elif hasattr(X, '__len__'):
            logger.info(f'📊 X length: {len(X)}')
            
        if hasattr(y, 'shape'):
            logger.info(f'📊 y shape: {y.shape}')
        elif hasattr(y, '__len__'):
            logger.info(f'📊 y length: {len(y)}')
        
        try:
            result = self.feature_selector.select_features(X, y, **kwargs)
            logger.info('✅ Advanced feature selection completed successfully')
            logger.info(f'📊 Result type: {type(result)}')
            
            if hasattr(result, 'shape'):
                logger.info(f'📊 Result shape: {result.shape}')
            elif hasattr(result, '__len__'):
                logger.info(f'📊 Result length: {len(result)}')
                
            return result
        except Exception as e:
            logger.error(f'❌ Advanced feature selection failed: {e}')
            logger.error(f'📊 Error type: {type(e).__name__}')
            raise
    
    def rank_features(self, X: Any, y: Any, **kwargs) -> Any:
        """Rank features using the unified feature selection manager."""
        return self.feature_selector.rank_features(X, y, **kwargs)
    
    def filter_features(self, X: Any, y: Any, **kwargs) -> Any:
        """Filter features using the unified feature selection manager."""
        return self.feature_selector.filter_features(X, y, **kwargs)


# Backwards compatibility exports
__all__ = [
    'Step08AdvancedFeatureSelection',
    'FUNCTIONALITY_AVAILABLE'
]