from ..standardized_parquet_handler import standardized_parquet_handler
"""Step 10 Safe Imports Management.

This module provides safe import management for all Step 10 dependencies,
with fallback handling and validation.
"""

from typing import Any, Dict, List, Optional, Tuple
import importlib
from src.utils.logger import system_logger

logger = system_logger.getChild('Step10Imports')

# Required modules for Step 10
REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "torch",
    "sklearn",
    "src.tactician.sr_breakout_predictor",
    "src.utils.error_handler",
    "src.utils.logger",
    "src.utils.warning_symbols",
    "src.training.enhanced_lm_optimizer",
]

# Optional modules
OPTIONAL_MODULES = [
    "optuna",
    "shap",
    "lightgbm",
    "xgboost",
]


class SafeImportManager:
    """Safe import manager for Step 10 dependencies."""

    def __init__(self):
        self.imports = {}
        self.failed_imports = []
        self.logger = logger

    def safe_import(self, module_name: str, fallback: Any = None) -> Any:
        """Safely import a module with fallback.

        Args:
            module_name: Name of the module to import
            fallback: Fallback value if import fails

        Returns:
            Imported module or fallback
        """
        if module_name in self.imports:
            return self.imports[module_name]

        try:
            module = importlib.import_module(module_name)
            self.imports[module_name] = module
            self.logger.debug(f"✅ Imported: {module_name}")
            return module

        except ImportError as e:
            self.failed_imports.append((module_name, str(e)))
            self.logger.warning(f"⚠️ Failed to import {module_name}: {e}")
            self.imports[module_name] = fallback
            return fallback

        except Exception as e:
            self.failed_imports.append((module_name, str(e)))
            self.logger.error(f"❌ Error importing {module_name}: {e}")
            self.imports[module_name] = fallback
            return fallback

    def validate_environment(self) -> Dict[str, Any]:
        """Validate the environment for Step 10 execution.

        Returns:
            Validation results dictionary
        """
        results = {
            "valid": True,
            "required_modules": {},
            "optional_modules": {},
            "failed_imports": self.failed_imports.copy(),
            "warnings": []
        }

        # Check required modules
        for module_name in REQUIRED_MODULES:
            module = self.safe_import(module_name)
            results["required_modules"][module_name] = module is not None

            if module is None:
                results["valid"] = False
                results["warnings"].append(f"Required module missing: {module_name}")

        # Check optional modules
        for module_name in OPTIONAL_MODULES:
            module = self.safe_import(module_name)
            results["optional_modules"][module_name] = module is not None

            if module is None:
                results["warnings"].append(f"Optional module missing: {module_name}")

        # Critical validation
        torch_available = results["required_modules"].get("torch", False)
        sklearn_available = results["required_modules"].get("sklearn", False)

        if not torch_available:
            results["valid"] = False
            results["warnings"].append("PyTorch is required but not available")

        if not sklearn_available:
            results["valid"] = False
            results["warnings"].append("Scikit-learn is required but not available")

        return results

    def get_import_status(self) -> Dict[str, Any]:
        """Get current import status.

        Returns:
            Dictionary with import status information
        """
        return {
            "total_imports": len(self.imports),
            "successful_imports": len([m for m in self.imports.values() if m is not None]),
            "failed_imports": len(self.failed_imports),
            "failed_modules": [name for name, _ in self.failed_imports]
        }


# Global import manager instance
safe_import_manager = SafeImportManager()

# Initialize safe imports
numpy = safe_import_manager.safe_import("numpy")
pandas = safe_import_manager.safe_import("pandas")
torch = safe_import_manager.safe_import("torch")
sklearn = safe_import_manager.safe_import("sklearn")

# Specialized imports
sr_breakout_predictor = safe_import_manager.safe_import("src.tactician.sr_breakout_predictor")
error_handler = safe_import_manager.safe_import("src.utils.error_handler")
system_logger_import = safe_import_manager.safe_import("src.utils.logger")
warning_symbols = safe_import_manager.safe_import("src.utils.warning_symbols")
enhanced_lm_optimizer = safe_import_manager.safe_import("src.training.enhanced_lm_optimizer")

# Optional ML libraries
optuna = safe_import_manager.safe_import("optuna")
shap = safe_import_manager.safe_import("shap")
lightgbm = safe_import_manager.safe_import("lightgbm")
xgboost = safe_import_manager.safe_import("xgboost")

# sklearn submodules
if sklearn is not None:
    try:
        from sklearn.preprocessing import StandardScaler, LabelEncoder
    except ImportError:
        StandardScaler = None
        LabelEncoder = None
else:
    StandardScaler = None
    LabelEncoder = None

# torch submodules
if torch is not None:
    try:
        from torch.utils.data import DataLoader, TensorDataset
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.nn.utils.prune as prune
    except ImportError:
        nn = None
        F = None
        DataLoader = None
        TensorDataset = None
        prune = None
else:
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None
    prune = None


def validate_step10_imports() -> bool:
    """Validate all Step 10 imports and dependencies.

    Returns:
        True if all critical imports are available
    """
    validation = safe_import_manager.validate_environment()

    if validation["valid"]:
        logger.info("✅ Step 10 imports validation passed")
        return True
    else:
        logger.error("❌ Step 10 imports validation failed")
        for warning in validation["warnings"]:
            logger.warning(f"  {warning}")
        return False


def get_import_summary() -> Dict[str, Any]:
    """Get summary of import status.

    Returns:
        Dictionary with import summary
    """
    validation = safe_import_manager.validate_environment()
    status = safe_import_manager.get_import_status()

    return {
        "validation": validation,
        "status": status,
        "ready_for_execution": validation["valid"]
    }
