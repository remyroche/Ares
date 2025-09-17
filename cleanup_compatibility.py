#!/usr/bin/env python3
"""
Cleanup Script: Consolidate and simplify compatibility layers

This script removes redundant compatibility layers and consolidates HMM compatibility.
"""

import os
import shutil
from pathlib import Path
from typing import List

def backup_file(file_path: Path) -> None:
    """Create a backup of the file."""
    backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
    shutil.copy2(file_path, backup_path)
    print(f"📦 Backed up {file_path} to {backup_path}")

def remove_redundant_files():
    """Remove redundant compatibility files."""
    print("🧹 Cleaning up redundant compatibility layers")
    print("=" * 50)
    
    # Files to remove (redundant compatibility layers)
    redundant_files = [
        "src/feature_generation/compatibility/simple_hmm_compatibility.py",
        "src/feature_engineering/standalone_hmm_compatibility.py",
        "src/hmm_feature_compatibility.py"  # Keep this as it's referenced
    ]
    
    for file_path_str in redundant_files:
        file_path = Path(file_path_str)
        if file_path.exists():
            # Check if it's the main hmm_feature_compatibility.py
            if file_path.name == "hmm_feature_compatibility.py" and "src/hmm_feature_compatibility.py" in str(file_path):
                print(f"⚠️ Keeping {file_path} (referenced by training pipeline)")
                continue
                
            backup_file(file_path)
            file_path.unlink()
            print(f"🗑️ Removed redundant file: {file_path}")
        else:
            print(f"⚠️ File not found: {file_path}")
    
    # Update the main HMM compatibility to use the unified system
    main_hmm_compat = Path("src/feature_generation/compatibility/hmm_compatibility.py")
    if main_hmm_compat.exists():
        print(f"\n📝 Updating main HMM compatibility: {main_hmm_compat}")
        
        with open(main_hmm_compat, 'r') as f:
            content = f.read()
        
        # Update imports to use unified optimization
        updated_content = content.replace(
            'from ...feature_engineering.feature_generators import FeatureGenerators',
            'from ...feature_engineering.feature_generators import FeatureGenerators'
        )
        
        # Add reference to unified optimization
        if 'unified_optimizer' not in updated_content:
            import_section = '''
# Try to use unified optimization system
try:
    from ...feature_engineering.optimization import get_feature_optimizer
    UNIFIED_OPTIMIZATION_AVAILABLE = True
except ImportError:
    UNIFIED_OPTIMIZATION_AVAILABLE = False
'''
            # Insert after existing imports
            lines = updated_content.split('\n')
            insert_index = -1
            for i, line in enumerate(lines):
                if line.startswith('logger = logging.getLogger'):
                    insert_index = i
                    break
            
            if insert_index > 0:
                lines.insert(insert_index, import_section)
                updated_content = '\n'.join(lines)
        
        with open(main_hmm_compat, 'w') as f:
            f.write(updated_content)
        
        print(f"✅ Updated {main_hmm_compat}")

def simplify_legacy_adapter():
    """Simplify the legacy adapter."""
    print("\n🔧 Simplifying legacy adapter...")
    
    legacy_adapter = Path("src/feature_generation/compatibility/legacy_adapter.py")
    if legacy_adapter.exists():
        backup_file(legacy_adapter)
        
        # Create a simplified version
        simplified_content = '''"""
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
        self.logger.warning("  - Feature optimization: src.feature_engineering.optimization")
    
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
        self.logger.info("  3. Use unified optimization from src.feature_engineering.optimization")
        
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
'''
        
        with open(legacy_adapter, 'w') as f:
            f.write(simplified_content)
        
        print(f"✅ Simplified {legacy_adapter}")

def update_compatibility_init():
    """Update compatibility package __init__.py."""
    print("\n📝 Updating compatibility package __init__.py...")
    
    compat_init = Path("src/feature_generation/compatibility/__init__.py")
    if compat_init.exists():
        backup_file(compat_init)
        
        updated_content = '''"""
Feature Generation Compatibility Package

This package provides compatibility layers for integrating with legacy systems
and external components that expect older interfaces.

Main Components:
- hmm_compatibility.py: HMM process compatibility (primary)
- legacy_adapter.py: Legacy feature generation compatibility (simplified)
"""

# Primary HMM compatibility (main interface)
from .hmm_compatibility import (
    HMMCompatibleFeatureGenerators,
    FeatureGenerators,  # Alias for backward compatibility
    get_hmm_compatible_generators
)

# Simplified legacy adapter (deprecated)
from .legacy_adapter import (
    LegacyFeatureAdapter,
    migrate_legacy_features,
    get_legacy_adapter,
    enable_legacy_compatibility
)

__all__ = [
    # HMM compatibility (primary)
    'HMMCompatibleFeatureGenerators',
    'FeatureGenerators',
    'get_hmm_compatible_generators',
    
    # Legacy adapter (deprecated)
    'LegacyFeatureAdapter',
    'migrate_legacy_features',
    'get_legacy_adapter', 
    'enable_legacy_compatibility'
]
'''
        
        with open(compat_init, 'w') as f:
            f.write(updated_content)
        
        print(f"✅ Updated {compat_init}")

def cleanup_compatibility_layers():
    """Main cleanup function."""
    print("🚀 Starting Compatibility Layer Cleanup")
    print("=" * 50)
    
    remove_redundant_files()
    simplify_legacy_adapter()
    update_compatibility_init()
    
    print("\n🎉 Compatibility cleanup completed!")
    print("📋 Summary:")
    print("  ✅ Removed redundant compatibility files")
    print("  ✅ Simplified legacy adapter")
    print("  ✅ Updated compatibility package")
    print("  ✅ Maintained HMM compatibility as primary interface")
    print("\n📋 Remaining compatibility structure:")
    print("  - src/feature_generation/compatibility/hmm_compatibility.py (primary)")
    print("  - src/feature_generation/compatibility/legacy_adapter.py (simplified)")
    print("  - src/hmm_feature_compatibility.py (standalone, kept for training pipeline)")

if __name__ == "__main__":
    cleanup_compatibility_layers()