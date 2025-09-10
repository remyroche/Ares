"""
Compatibility module for step06_feature_engineering imports in data collection.

This module provides backwards compatibility for imports that were previously
available in the deleted step06_feature_engineering.py file in data collection.

All functionality has been moved to ml_commons and step06_utilities.
"""

import warnings
import logging
from typing import Any, Dict, List, Optional, Union

# Setup logging
logger = logging.getLogger(__name__)

logger.info('🔧 Loading step06_feature_engineering compatibility module...')

# Import the actual functionality from the new locations
try:
    logger.info('📦 Attempting to import EnhancedFeatureEngineeringStep...')
    from src.feature_engineering.step06_enhanced_feature_engineering import (
        EnhancedFeatureEngineering
    )
    from src.feature_engineering.step06_enhanced_feature_engineering_step import (
        EnhancedFeatureEngineeringStep
    )
    FUNCTIONALITY_AVAILABLE = True
    logger.info('✅ Successfully imported EnhancedFeatureEngineeringStep')
except ImportError as e:
    FUNCTIONALITY_AVAILABLE = False
    logger.error(f'❌ Failed to import Step06 functionality: {e}')
    logger.warning(f'⚠️ Step06 functionality not available: {e}')
    warnings.warn(f"Step06 functionality not available: {e}")


class FeatureEngineeringStep:
    """
    Compatibility class for FeatureEngineeringStep in data collection.
    
    This class provides backwards compatibility for the FeatureEngineeringStep
    that was previously available in data collection step06_feature_engineering.py.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the feature engineering step."""
        logger.info('🔧 Initializing FeatureEngineeringStep compatibility class...')
        logger.info(f'📊 Configuration provided: {config is not None}')
        
        if not FUNCTIONALITY_AVAILABLE:
            logger.error('❌ Step06 functionality not available')
            raise ImportError("Step06 functionality not available. Please ensure ml_commons and step06_utilities are properly installed.")
        
        self.config = config or {}
        logger.info(f'📊 Configuration keys: {list(self.config.keys()) if self.config else "None"}')
        
        logger.info('🔧 Creating EnhancedFeatureEngineeringStep instance...')
        self.enhanced_engine = EnhancedFeatureEngineeringStep(self.config)
        logger.info('✅ EnhancedFeatureEngineeringStep instance created successfully')
        
        logger.warning('⚠️ FeatureEngineeringStep is deprecated - use EnhancedFeatureEngineeringStep from src.feature_engineering instead')
        warnings.warn(
            "FeatureEngineeringStep is deprecated. Use EnhancedFeatureEngineeringStep from src.feature_engineering instead.",
            DeprecationWarning,
            stacklevel=2
        )
    
    def execute(self, data: Any, **kwargs) -> Any:
        """Execute feature engineering using the enhanced feature engineering."""
        logger.info('🚀 Executing feature engineering step...')
        logger.info(f'📊 Data type: {type(data)}')
        logger.info(f'📊 Additional kwargs: {list(kwargs.keys()) if kwargs else "None"}')
        
        if hasattr(data, 'shape'):
            logger.info(f'📊 Data shape: {data.shape}')
        elif hasattr(data, '__len__'):
            logger.info(f'📊 Data length: {len(data)}')
        
        try:
            result = self.enhanced_engine.execute(data, **kwargs)
            logger.info('✅ Feature engineering execution completed successfully')
            logger.info(f'📊 Result type: {type(result)}')
            
            if hasattr(result, 'shape'):
                logger.info(f'📊 Result shape: {result.shape}')
            elif hasattr(result, '__len__'):
                logger.info(f'📊 Result length: {len(result)}')
                
            return result
        except Exception as e:
            logger.error(f'❌ Feature engineering execution failed: {e}')
            logger.error(f'📊 Error type: {type(e).__name__}')
            raise
    
    def create_features(self, data: Any, **kwargs) -> Any:
        """Create features using the enhanced feature engineering."""
        return self.enhanced_engine.create_features(data, **kwargs)
    
    def extract_basic_features(self, data: Any, **kwargs) -> Any:
        """Extract basic features using the enhanced feature engineering."""
        return self.enhanced_engine.extract_basic_features(data, **kwargs)


# Backwards compatibility exports
__all__ = [
    'FeatureEngineeringStep',
    'FUNCTIONALITY_AVAILABLE'
]