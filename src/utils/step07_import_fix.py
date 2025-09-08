"""
Step07 Import Fix Module

This module provides safe imports with proper fallback handling
to resolve the import chain issues identified in the audit.
"""

import sys
import warnings
from typing import Optional, Any, Dict
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

class SafeImporter:
    """Safe import utility with fallback handling."""
    
    def __init__(self):
        self.import_cache = {}
        self.fallback_modules = {}
    
    def safe_import(self, module_name: str, fallback: Any = None) -> Any:
        """Safely import a module with fallback."""
        if module_name in self.import_cache:
            return self.import_cache[module_name]
        
        try:
            module = __import__(module_name)
            self.import_cache[module_name] = module
            return module
        except ImportError as e:
            if fallback is not None:
                self.fallback_modules[module_name] = fallback
                return fallback
            else:
                warnings.warn(f"Failed to import {module_name}: {e}")
                return None
    
    def get_import_status(self) -> Dict[str, bool]:
        """Get status of all imports."""
        status = {}
        for module_name in self.import_cache:
            status[module_name] = True
        for module_name in self.fallback_modules:
            status[module_name] = False
        return status

# Global safe importer instance
safe_importer = SafeImporter()

# Core scientific computing imports
numpy = safe_importer.safe_import('numpy')
pandas = safe_importer.safe_import('pandas')
scipy = safe_importer.safe_import('scipy')
sklearn = safe_importer.safe_import('sklearn')
torch = safe_importer.safe_import('torch')
numba = safe_importer.safe_import('numba')
psutil = safe_importer.safe_import('psutil')

# Optional ML libraries
lightgbm = safe_importer.safe_import('lightgbm')
xgboost = safe_importer.safe_import('xgboost')

# Project-specific imports with fallbacks
def get_system_logger():
    """Get system logger with fallback."""
    try:
        from src.utils.logger import system_logger
        return system_logger
    except ImportError:
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger('step07_fallback')

def get_handles_errors_decorator():
    """Get handles_errors decorator with fallback."""
    try:
        from src.core.decorators import handles_errors
        return handles_errors
    except ImportError:
        def fallback_decorator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        return fallback_decorator

def get_base_step():
    """Get BaseStep class with fallback."""
    try:
        from src.training.base_step import BaseStep
        return BaseStep
    except ImportError:
        class FallbackBaseStep:
            def __init__(self, config, step_id, step_name):
                self.config = config
                self.step_id = step_id
                self.step_name = step_name
        return FallbackBaseStep

# Initialize fallback components
system_logger = get_system_logger()
handles_errors = get_handles_errors_decorator()
BaseStep = get_base_step()

def check_dependencies():
    """Check if all required dependencies are available."""
    required_modules = ['numpy', 'pandas', 'sklearn', 'torch', 'numba', 'psutil']
    missing_modules = []
    
    for module_name in required_modules:
        if safe_importer.safe_import(module_name) is None:
            missing_modules.append(module_name)
    
    if missing_modules:
        print(f"❌ Missing required modules: {missing_modules}")
        return False
    else:
        print("✅ All required modules available")
        return True

def get_import_summary():
    """Get summary of import status."""
    status = safe_importer.get_import_status()
    print("📊 Import Status Summary:")
    for module, available in status.items():
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {module}")
    return status

if __name__ == "__main__":
    print("🔍 Step07 Import Fix Module")
    print("=" * 40)
    check_dependencies()
    get_import_summary()
