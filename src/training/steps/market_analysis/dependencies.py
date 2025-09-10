from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
import numpy as np

"""Dependency Management for Step05 Labeling.

This module provides centralized dependency management and validation
for the labeling step, ensuring robust operation even when dependencies are missing.
"""
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import system_logger
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Required modules for the labeling step
REQUIRED_MODULES = [
    'pandas', 'numpy', 'psutil', 
    'src.utils.centralized_decorators', 
    'src.utils.logger', 
    'src.utils.enhanced_mlflow_integration', 
    'src.analyst.meta_labeling_system',
    'threading', 'multiprocessing', 'concurrent.futures',
    'collections', 'gc', 'warnings', 're', 'os'
]

# Optional modules that enhance functionality
OPTIONAL_MODULES = [
    'src.utils.pipeline_standards',
    'src.utils.common_operations',
    'src.utils.regime_data_access',
    'src.training.steps.step06_labeling_components.regime_specific_triple_barrier_optimizer',
    'src.training.steps.step06_labeling_components.regime_aware_triple_barrier_labeling',
]

class DependencyManager:
    """Manages dependencies and provides fallback mechanisms."""
    @log_important_calls
    
    def __init__(self, logger: Any = None):
        self.logger = logger or logging.getLogger(__name__)
        self.dependency_status: Dict[str, bool] = {}
        self.imported_modules: Dict[str, Any] = {}
        self._validate_dependencies()
    @log_all_calls
    
    def _validate_dependencies(self) -> None:
        """Validate all dependencies and record their status."""
        self.logger.info('🔍 Validating dependencies...')
        
        # Check required modules
        for module in REQUIRED_MODULES:
            try:
                __import__(module)
                self.dependency_status[module] = True
                self.logger.debug(f'✅ {module} - available')
            except ImportError:
                self.dependency_status[module] = False
                self.logger.warning(f'⚠️ {module} - not available')
        
        # Check optional modules
        for module in OPTIONAL_MODULES:
            try:
                __import__(module)
                self.dependency_status[module] = True
                self.logger.debug(f'✅ {module} - available (optional)')
            except ImportError:
                self.dependency_status[module] = False
                self.logger.debug(f'⚠️ {module} - not available (optional)')
        
        # Log summary
        required_available = sum(1 for m in REQUIRED_MODULES if self.dependency_status.get(m, False))
        optional_available = sum(1 for m in OPTIONAL_MODULES if self.dependency_status.get(m, False))
        
        self.logger.info(f'📊 Dependency Status:')
        self.logger.info(f'   Required modules: {required_available}/{len(REQUIRED_MODULES)} available')
        self.logger.info(f'   Optional modules: {optional_available}/{len(OPTIONAL_MODULES)} available')
        
        if required_available < len(REQUIRED_MODULES):
            missing_required = [m for m in REQUIRED_MODULES if not self.dependency_status.get(m, False)]
            self.logger.warning(f'⚠️ Missing required modules: {missing_required}')
    
    def safe_import(self, module_name: str, default: Any = None) -> Any:
        """Safely import a module with fallback."""
        if module_name in self.imported_modules:
            return self.imported_modules[module_name]
        
        try:
            module = __import__(module_name)
            self.imported_modules[module_name] = module
            return module
        except ImportError:
            self.logger.debug(f'⚠️ Could not import {module_name}, using fallback')
            return default
    
    def get_dependency_status(self) -> Dict[str, bool]:
        """Get the current dependency status."""
        return self.dependency_status.copy()
    
    def is_available(self, module_name: str) -> bool:
        """Check if a module is available."""
        return self.dependency_status.get(module_name, False)

# Global dependency manager instance
dependency_manager = DependencyManager()

# Safe imports with fallbacks
try:
    from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
    pipeline_standards_available = True
except ImportError:
    pipeline_standards_available = False
    PipelineStandards = None
    pipeline_standards = None

try:
    from src.utils.logger import system_logger
    system_logger_available = True
except ImportError:
    system_logger_available = False
    system_logger = logging.getLogger(__name__)

try:
    from src.utils.common_operations import ensure_directory, safe_json_dump
    common_operations_available = True
except ImportError:
    common_operations_available = False
    # Fallback implementations
    def ensure_directory(path: Path | str) -> Path:
        p = Path(path)
        p.mkdir(parents = True, exist_ok = True)
        return p
    
    def safe_json_dump(data: Any, path: Path | str, **kwargs) -> None:
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, **kwargs)

try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_available = False
    psutil = None

# Validate environment dependencies
if pipeline_standards_available:
    dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)
else:
    dependency_status = dependency_manager.get_dependency_status()

def get_system_logger() -> Any:
    """Get the system logger with fallback."""
    if system_logger_available:
        return system_logger
    else:
        logging.basicConfig(level = logging.INFO)
        return logging.getLogger("System")

def get_pipeline_standards() -> Any:
    """Get pipeline standards with fallback."""
    if pipeline_standards_available:
        return pipeline_standards
    else:
        return None

def get_ensure_directory() -> callable:
    """Get ensure_directory function with fallback."""
    if common_operations_available:
        return ensure_directory
    else:
        return ensure_directory  # Use fallback implementation

def get_safe_json_dump() -> callable:
    """Get safe_json_dump function with fallback."""
    if common_operations_available:
        return safe_json_dump
    else:
        return safe_json_dump  # Use fallback implementation

def get_psutil() -> Optional[Any]:
    """Get psutil module with fallback."""
    if psutil_available:
        return psutil
    else:
        return None

# Export key dependencies
__all__ = [
    "DependencyManager",
    "dependency_manager",
    "dependency_status",
    "get_system_logger",
    "get_pipeline_standards", 
    "get_ensure_directory",
    "get_safe_json_dump",
    "get_psutil",
    "REQUIRED_MODULES",
    "OPTIONAL_MODULES",
]