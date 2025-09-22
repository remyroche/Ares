"""
Clean Import Template for Model Training Modules

This template provides:
- Clean, organized imports with proper error handling
- Dependency validation using centralized manager
- No circular dependencies
- Consistent import patterns across all modules
- Proper fallback mechanisms
"""

"""
Instructions for using this template:

1. Copy the imports you need from the sections below
2. Remove any imports you don't need
3. Replace the complex try/except blocks with simple imports
4. Use the dependency manager for optional dependencies
5. Import only what you need - don't import entire modules
"""

# =============================================================================
# CORE DEPENDENCIES (Required - fail fast if missing)
# =============================================================================

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# =============================================================================
# LOCAL DEPENDENCIES (Required - use centralized imports)
# =============================================================================

# Core system imports
from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_debug, tprint_progress, tprint_performance, tprint_structured,
    tprint_timer, LogLevel
)

# Centralized dependency and configuration management
from .dependency_manager import get_dependency_manager, validate_training_environment
from .config_manager import get_config_manager, get_model_config, get_training_mode_config

# ML common utilities (required)
from src.utils.ml_common.config import BaseTrainingConfig
from src.utils.ml_common.training import BaseTrainingStep

# Common operations (required)
from src.utils.common_operations import (
    safe_divide, safe_mean, safe_std, validate_finite, validate_positive,
    ensure_directory, safe_json_dump, safe_json_load, sanitize_string,
    get_memory_usage, check_disk_space, timed_operation
)

# Math validation (required)
from src.utils.math_validation import (
    safe_divide as math_safe_divide, validate_finite as math_validate_finite,
    validate_positive as math_validate_positive, safe_mean as math_safe_mean,
    safe_std as math_safe_std, MathValidationError
)

# =============================================================================
# OPTIONAL DEPENDENCIES (Use dependency manager for these)
# =============================================================================

def get_optional_dependencies():
    """Get optional dependencies with proper fallback handling."""
    manager = get_dependency_manager()

    deps = {}

    # Hardware optimization (optional)
    try:
        deps['hardware'] = {
            'gpu_manager': manager.require_critical('hardware_optimizers').get_m1_gpu_manager(),
            'memory_optimizer': manager.require_critical('hardware_optimizers').get_m1_memory_optimizer(),
            'cpu_optimizer': manager.require_critical('hardware_optimizers').get_m1_cpu_optimizer()
        }
        tprint_info("✅ Hardware optimizers loaded")
    except ImportError:
        deps['hardware'] = None
        tprint_warning("⚠️ Hardware optimizers not available, using standard processing")

    # ML libraries (optional with fallbacks)
    deps['lightgbm'] = manager.get_module('lightgbm')
    deps['catboost'] = manager.get_module('catboost')
    deps['xgboost'] = manager.get_module('xgboost')

    # Advanced training utilities (optional)
    try:
        from src.utils.ml_common.training.enhanced_training_utils import EnhancedTrainingUtils
        from src.utils.ml_common.training.training_integration import TrainingStepEnhancer
        deps['enhanced_training'] = {
            'utils': EnhancedTrainingUtils,
            'enhancer': TrainingStepEnhancer
        }
        tprint_info("✅ Enhanced training utilities loaded")
    except ImportError:
        deps['enhanced_training'] = None
        tprint_info("ℹ️ Enhanced training utilities not available")

    return deps

# =============================================================================
# MODEL-SPECIFIC IMPORTS (Import only what you need)
# =============================================================================

def get_model_imports(model_types: List[str]):
    """Get model-specific imports based on required model types."""
    imports = {}

    if 'lightgbm' in model_types:
        try:
            from lightgbm import LGBMRegressor
            imports['lightgbm'] = LGBMRegressor
            tprint_info("✅ LightGBM imported")
        except ImportError:
            tprint_warning("⚠️ LightGBM not available")

    if 'catboost' in model_types:
        try:
            from catboost import CatBoostRegressor
            imports['catboost'] = CatBoostRegressor
            tprint_info("✅ CatBoost imported")
        except ImportError:
            tprint_warning("⚠️ CatBoost not available")

    if 'xgboost' in model_types:
        try:
            from xgboost import XGBRegressor
            imports['xgboost'] = XGBRegressor
            tprint_info("✅ XGBoost imported")
        except ImportError:
            tprint_warning("⚠️ XGBoost not available")

    if 'tcn' in model_types:
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv1D, Dense, Dropout, LayerNormalization
            imports['tcn'] = {
                'Sequential': Sequential,
                'Conv1D': Conv1D,
                'Dense': Dense,
                'Dropout': Dropout,
                'LayerNormalization': LayerNormalization
            }
            tprint_info("✅ TCN dependencies imported")
        except ImportError:
            tprint_warning("⚠️ TCN dependencies not available")

    return imports

# =============================================================================
# UTILITY FUNCTIONS FOR CLEAN IMPORTS
# =============================================================================

def setup_logging(module_name: str) -> logging.Logger:
    """Setup logging for a module."""
    return system_logger.getChild(module_name)

def validate_imports(required_modules: List[str]) -> bool:
    """Validate that required imports are available."""
    manager = get_dependency_manager()

    missing = []
    for module in required_modules:
        if not manager.is_available(module):
            missing.append(module)

    if missing:
        tprint_error(f"❌ Missing required modules: {', '.join(missing)}")
        return False

    tprint_success("✅ All required imports validated")
    return True

def get_fallback_model(original_model_type: str):
    """Get fallback model type if original is not available."""
    manager = get_dependency_manager()
    config_manager = get_config_manager()

    # Get the model config to check fallback
    try:
        model_config = config_manager.get_model_config(original_model_type)
        if model_config.fallback_model:
            return model_config.fallback_model
    except:
        pass

    # Default fallbacks
    fallbacks = {
        'tcn': 'lightgbm',
        'catboost': 'lightgbm',
        'xgboost': 'lightgbm',
        'node': 'xgboost',
        'random_forest': 'linear_regression'
    }

    return fallbacks.get(original_model_type, 'lightgbm')

# =============================================================================
# EXAMPLE USAGE PATTERNS
# =============================================================================

"""
Example of how to use this clean import system:

# 1. Import core dependencies
import numpy as np
import pandas as pd
from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_error
from .dependency_manager import validate_training_environment
from .config_manager import get_config_manager

# 2. Validate environment early
if not validate_training_environment():
    raise RuntimeError("Training environment validation failed")

# 3. Get optional dependencies cleanly
deps = get_optional_dependencies()

# 4. Get model-specific imports
model_imports = get_model_imports(['lightgbm', 'catboost'])

# 5. Use configuration manager instead of hardcoded values
config_manager = get_config_manager()
analyst_models = config_manager.get_models_by_priority('analyst', 'full')

# 6. Setup logging cleanly
logger = setup_logging('MyTrainingModule')

# This replaces 100+ lines of complex try/except blocks with clean, maintainable code!
"""