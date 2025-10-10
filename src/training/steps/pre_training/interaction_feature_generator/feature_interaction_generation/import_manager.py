"""
Import Manager for Interactive Feature Generation

This module provides a centralized way to handle optional imports with consistent
error handling and logging, reducing code duplication across the codebase.

Key Features:
- Consistent error handling for optional imports
- Centralized logging for import failures
- Support for both required and optional dependencies
- Caching of import results
- Type hints for better IDE support
"""

import sys
import importlib
from typing import Any, Optional, Dict, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
)


class ImportStatus(Enum):
    """Status of an import attempt."""
    SUCCESS = "success"
    FAILED = "failed"
    NOT_ATTEMPTED = "not_attempted"


@dataclass
class ImportResult:
    """Result of an import attempt."""
    module: Optional[Any]
    status: ImportStatus
    error: Optional[Exception] = None
    module_name: str = ""


class ImportManager:
    """
    Centralized import manager for handling optional dependencies.
    
    This class provides a consistent way to handle imports across the
    interactive feature generation system, reducing code duplication
    and improving maintainability.
    """
    
    def __init__(self):
        """Initialize the import manager."""
        self._import_cache: Dict[str, ImportResult] = {}
        self._required_modules: set = set()
        self._optional_modules: set = set()
        
        tprint_debug("🔧 ImportManager initialized")
    
    def register_required_module(self, module_name: str) -> None:
        """Register a module as required."""
        self._required_modules.add(module_name)
        tprint_debug(f"📝 Registered required module: {module_name}")
    
    def register_optional_module(self, module_name: str) -> None:
        """Register a module as optional."""
        self._optional_modules.add(module_name)
        tprint_debug(f"📝 Registered optional module: {module_name}")
    
    def safe_import(
        self, 
        module_name: str, 
        required: bool = False,
        from_list: Optional[list] = None,
        alias: Optional[str] = None
    ) -> ImportResult:
        """
        Safely import a module with consistent error handling.
        
        Args:
            module_name: Name of the module to import
            required: Whether the module is required (raises exception if not found)
            from_list: List of items to import from the module
            alias: Optional alias for the module
            
        Returns:
            ImportResult with the imported module or None
        """
        # Check cache first
        cache_key = f"{module_name}_{bool(from_list)}_{alias or ''}"
        if cache_key in self._import_cache:
            cached_result = self._import_cache[cache_key]
            if cached_result.status == ImportStatus.SUCCESS:
                tprint_debug(f"📦 Using cached import: {module_name}")
                return cached_result
        
        try:
            # Import the module
            if from_list:
                # Import specific items from the module
                module = importlib.import_module(module_name)
                imported_items = {}
                for item in from_list:
                    if hasattr(module, item):
                        imported_items[item] = getattr(module, item)
                    else:
                        tprint_warning(f"⚠️ Item '{item}' not found in module '{module_name}'")
                
                result = ImportResult(
                    module=imported_items,
                    status=ImportStatus.SUCCESS,
                    module_name=module_name
                )
            else:
                # Import the entire module
                module = importlib.import_module(module_name)
                result = ImportResult(
                    module=module,
                    status=ImportStatus.SUCCESS,
                    module_name=module_name
                )
            
            # Cache the successful result
            self._import_cache[cache_key] = result
            
            # Log success
            if required:
                tprint_success(f"✅ Required module imported: {module_name}")
            else:
                tprint_debug(f"📦 Optional module imported: {module_name}")
            
            return result
            
        except ImportError as e:
            error_msg = f"Module '{module_name}' not available: {e}"
            
            result = ImportResult(
                module=None,
                status=ImportStatus.FAILED,
                error=e,
                module_name=module_name
            )
            
            # Cache the failed result
            self._import_cache[cache_key] = result
            
            if required:
                tprint_error(f"❌ {error_msg}")
                raise ImportError(f"Required module {module_name} not available: {e}")
            else:
                tprint_warning(f"⚠️ {error_msg}")
            
            return result
        
        except Exception as e:
            error_msg = f"Unexpected error importing '{module_name}': {e}"
            tprint_error(f"❌ {error_msg}")
            
            result = ImportResult(
                module=None,
                status=ImportStatus.FAILED,
                error=e,
                module_name=module_name
            )
            
            self._import_cache[cache_key] = result
            
            if required:
                raise
            else:
                return result
    
    def import_common_operations(self) -> ImportResult:
        """Import common operations utilities."""
        return self.safe_import(
            "src.utils.common_operations",
            required=True,
            from_list=[
                "safe_divide", "safe_log", "safe_sqrt", "safe_power", "validate_finite",
                "get_m1_gpu_manager", "get_m1_memory_optimizer", "get_m1_cpu_optimizer",
                "optimize_memory_usage", "parallel_processing_optimizer"
            ]
        )
    
    def import_matrix_operations(self) -> ImportResult:
        """Import matrix operations utilities."""
        return self.safe_import(
            "src.utils.matrix_operations",
            required=True,
            from_list=[
                "get_unified_matrix_operations", "get_vectorized_processing_core",
                "get_batch_matrix_processor", "safe_matrix_multiply",
                "vectorized_rolling_features", "parallel_feature_engineering",
                "optimize_dataframe", "get_hardware_performance_report"
            ]
        )
    
    def import_ml_common(self) -> ImportResult:
        """Import ML common utilities."""
        return self.safe_import(
            "src.utils.ml_common.optimization.bayesian_tpe_optimizer",
            required=True,
            from_list=["BayesianTPEOptimizer", "OptimizationConfig"]
        )
    
    def import_data_utils(self) -> ImportResult:
        """Import data utilities."""
        return self.safe_import(
            "src.utils.data.klines_parquet",
            required=True,
            from_list=["KlinesParquetManager"]
        )
    
    def import_math_validation(self) -> ImportResult:
        """Import math validation utilities."""
        return self.safe_import(
            "src.utils.math_validation",
            required=True,
            from_list=[
                "safe_divide", "safe_log", "safe_sqrt", "validate_finite"
            ]
        )
    
    def import_tprint(self) -> ImportResult:
        """Import tprint utilities."""
        return self.safe_import(
            "src.utils.tprint",
            required=True,
            from_list=[
                "tprint", "tprint_info", "tprint_success", "tprint_warning", 
                "tprint_error", "tprint_debug", "tprint_performance", "tprint_progress"
            ]
        )
    
    def import_purged_kfold(self) -> ImportResult:
        """Import purged K-fold utilities."""
        return self.safe_import(
            "src.utils.purged_kfold",
            required=True,
            from_list=["PurgedKFoldTime"]
        )
    
    def import_feature_selection(self) -> ImportResult:
        """Import feature selection utilities."""
        return self.safe_import(
            "src.feature_selection",
            required=True,
            from_list=["select_features"]
        )
    
    def get_import_status(self, module_name: str) -> ImportStatus:
        """Get the status of a module import."""
        for result in self._import_cache.values():
            if result.module_name == module_name:
                return result.status
        return ImportStatus.NOT_ATTEMPTED
    
    def is_module_available(self, module_name: str) -> bool:
        """Check if a module is available."""
        return self.get_import_status(module_name) == ImportStatus.SUCCESS
    
    def get_imported_module(self, module_name: str) -> Optional[Any]:
        """Get the imported module if available."""
        for result in self._import_cache.values():
            if result.module_name == module_name and result.status == ImportStatus.SUCCESS:
                return result.module
        return None
    
    def clear_cache(self) -> None:
        """Clear the import cache."""
        self._import_cache.clear()
        tprint_debug("🧹 Import cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the import cache."""
        stats = {
            "total_imports": len(self._import_cache),
            "successful_imports": sum(1 for r in self._import_cache.values() if r.status == ImportStatus.SUCCESS),
            "failed_imports": sum(1 for r in self._import_cache.values() if r.status == ImportStatus.FAILED),
        }
        return stats


# Global import manager instance
_import_manager = ImportManager()


def get_import_manager() -> ImportManager:
    """Get the global import manager instance."""
    return _import_manager


# Convenience functions for common imports
def import_common_operations():
    """Import common operations utilities."""
    return _import_manager.import_common_operations()


def import_matrix_operations():
    """Import matrix operations utilities."""
    return _import_manager.import_matrix_operations()


def import_ml_common():
    """Import ML common utilities."""
    return _import_manager.import_ml_common()


def import_data_utils():
    """Import data utilities."""
    return _import_manager.import_data_utils()


def import_math_validation():
    """Import math validation utilities."""
    return _import_manager.import_math_validation()


def import_tprint():
    """Import tprint utilities."""
    return _import_manager.import_tprint()


def import_purged_kfold():
    """Import purged K-fold utilities."""
    return _import_manager.import_purged_kfold()


def import_feature_selection():
    """Import feature selection utilities."""
    return _import_manager.import_feature_selection()


# Example usage
if __name__ == "__main__":
    # Test the import manager
    manager = get_import_manager()
    
    # Test required import
    try:
        tprint_result = manager.import_tprint()
        if tprint_result.status == ImportStatus.SUCCESS:
            print("✅ tprint imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import tprint: {e}")
    
    # Test optional import
    optional_result = manager.safe_import("nonexistent.module", required=False)
    print(f"Optional import status: {optional_result.status}")
    
    # Print cache stats
    stats = manager.get_cache_stats()
    print(f"Cache stats: {stats}")