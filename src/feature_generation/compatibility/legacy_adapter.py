"""
Simplified Legacy Feature Adapter

This module provides a minimal compatibility layer for legacy feature generation code.
Most functionality has been moved to the unified systems.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class LegacyFeatureAdapter:
    """
    Simplified legacy feature adapter.
    
    This provides minimal compatibility for legacy code while encouraging
    migration to the unified feature generation and optimization systems.
    """
    
    def __init__(self):
        """Initialize the simplified legacy adapter."""
        self.logger = logger.getChild('LegacyFeatureAdapter')
        self.logger.info("✅ Simplified legacy adapter initialized")
        self.logger.warning("⚠️ Legacy adapter is deprecated. Please migrate to unified systems:")
        self.logger.warning("  - Feature generation: src.feature_generation")
        self.logger.warning("  - Feature optimization: src.feature_generation.utils.optimization")
    
    def migrate_legacy_features(self, legacy_config: Dict[str, Any]) -> List[Any]:
        """
        Migrate legacy feature configurations.
        
        Args:
            legacy_config: Legacy configuration dictionary
            
        Returns:
            Empty list with migration guidance
        """
        self.logger.warning("⚠️ migrate_legacy_features is deprecated")
        self.logger.info("📋 Migration guidance:")
        self.logger.info("  1. Use FeatureBank from src.feature_generation")
        self.logger.info("  2. Use category-based feature generation")
        self.logger.info("  3. Use unified optimization from src.feature_generation.utils.optimization")
        
        return []
    
    def get_legacy_adapter(self):
        """Get legacy adapter (deprecated)."""
        self.logger.warning("⚠️ get_legacy_adapter is deprecated")
        return self
    
    def enable_legacy_compatibility(self, enabled: bool = True):
        """Enable/disable legacy compatibility (deprecated)."""
        self.logger.warning("⚠️ enable_legacy_compatibility is deprecated")
        self.logger.info("💡 Use unified systems instead of legacy compatibility")

# Backward compatibility functions
def migrate_legacy_features(legacy_config: Dict[str, Any]) -> List[Any]:
    """Migrate legacy features (deprecated)."""
    adapter = LegacyFeatureAdapter()
    return adapter.migrate_legacy_features(legacy_config)

def get_legacy_adapter() -> LegacyFeatureAdapter:
    """Get legacy adapter (deprecated)."""
    return LegacyFeatureAdapter()

def enable_legacy_compatibility(enabled: bool = True):
    """Enable legacy compatibility (deprecated)."""
    adapter = LegacyFeatureAdapter()
    adapter.enable_legacy_compatibility(enabled)

__all__ = [
    'LegacyFeatureAdapter',
    'migrate_legacy_features', 
    'get_legacy_adapter',
    'enable_legacy_compatibility'
]
