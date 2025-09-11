from src.utils.tprint import tprint

"""
Dependency management for optional packages.
"""

import importlib
import logging
from typing import Any, Dict, List, Optional, Set, Type, Union
from functools import wraps
import warnings
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DependencyManager:
    """
    Manages optional dependencies and provides graceful fallbacks.
    """
    
    def __init__(self):
        self.available_packages: Set[str] = set()
        self.optional_packages: Dict[str, Dict[str, Any]] = {}
        self.fallback_implementations: Dict[str, Type] = {}
        self._check_available_packages()
    
    def _check_available_packages(self) -> None:
        """Check which optional packages are available."""
        optional_packages = {
            'numpy': 'numpy',
            'pandas': 'pandas',
            'scipy': 'scipy',
            'sklearn': 'sklearn',
            'tensorflow': 'tensorflow',
            'torch': 'torch',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'plotly': 'plotly',
            'requests': 'requests',
            'aiohttp': 'aiohttp',
            'pydantic': 'pydantic',
            'fastapi': 'fastapi',
            'flask': 'flask',
            'sqlalchemy': 'sqlalchemy',
            'redis': 'redis',
            'celery': 'celery',
            'pytest': 'pytest',
            'black': 'black',
            'flake8': 'flake8',
            'mypy': 'mypy',
            'isort': 'isort',
            'pyyaml': 'yaml',
            'toml': 'toml',
            'python_dotenv': 'dotenv',
            'structlog': 'structlog',
            'loguru': 'loguru',
            'prometheus_client': 'prometheus_client',
            'mlflow': 'mlflow',
            'wandb': 'wandb',
            'optuna': 'optuna',
            'hyperopt': 'hyperopt',
            'xgboost': 'xgboost',
            'lightgbm': 'lightgbm',
            'catboost': 'catboost',
            'shap': 'shap',
            'lime': 'lime',
            'eli5': 'eli5',
            'boruta': 'boruta',
            'featuretools': 'featuretools',
            'ta': 'ta',
            'pandas_ta': 'pandas_ta',
            'yfinance': 'yfinance',
            'ccxt': 'ccxt',
            'binance': 'binance',
            'kraken': 'kraken',
            'coinbase': 'coinbase',
            'alpaca': 'alpaca',
            'ibapi': 'ibapi',
            'quantlib': 'quantlib',
            'zipline': 'zipline',
            'backtrader': 'backtrader',
            'vectorbt': 'vectorbt',
            'empyrical': 'empyrical',
            'pyfolio': 'pyfolio',
            'bt': 'bt',
            'ffn': 'ffn',
            'ta_lib': 'talib'
        }
        
        for package_name, import_name in optional_packages.items():
            try:
                importlib.import_module(import_name)
                self.available_packages.add(package_name)
                logger.debug(f"Package {package_name} is available")
            except ImportError:
                logger.debug(f"Package {package_name} is not available")
    
    def is_available(self, package_name: str) -> bool:
        """Check if a package is available."""
        return package_name in self.available_packages
    
    def get_available_packages(self) -> Set[str]:
        """Get set of available packages."""
        return self.available_packages.copy()
    
    def get_missing_packages(self, required_packages: List[str]) -> List[str]:
        """Get list of missing required packages."""
        return [pkg for pkg in required_packages if not self.is_available(pkg)]
    
    def register_optional_package(self, package_name: str, 
                                import_name: str | None = None,
                                fallback_class: Type | None = None,
                                description: str = "") -> None:
        """
        Register an optional package with fallback.
        
        Args:
            package_name: Name of the package
            import_name: Import name (defaults to package_name)
            fallback_class: Fallback class to use if package is not available
            description: Description of the package
        """
        if import_name is None:
            import_name = package_name
        
        self.optional_packages[package_name] = {
            'import_name': import_name,
            'fallback_class': fallback_class,
            'description': description,
            'available': self.is_available(package_name)
        }
        
        if fallback_class:
            self.fallback_implementations[package_name] = fallback_class
    
    def get_package_info(self, package_name: str) -> Dict[str, Any]:
        """Get information about a package."""
        return self.optional_packages.get(package_name, {})
    
    def create_fallback_instance(self, package_name: str, *args, **kwargs) -> Any:
        """Create a fallback instance for a missing package."""
        if package_name in self.fallback_implementations:
            fallback_class = self.fallback_implementations[package_name]
            return fallback_class(*args, **kwargs)
        else:
            raise ImportError(f"No fallback available for package {package_name}")
    
    def safe_import(self, package_name: str, import_name: str | None = None) -> Any:
        """
        Safely import a package with fallback.
        
        Args:
            package_name: Name of the package
            import_name: Import name (defaults to package_name)
            
        Returns:
            Imported module or fallback instance
        """
        if import_name is None:
            import_name = package_name
        
        try:
            return importlib.import_module(import_name)
        except ImportError:
            if package_name in self.fallback_implementations:
                logger.warning(f"Package {package_name} not available, using fallback")
                return self.create_fallback_instance(package_name)
            else:
                raise ImportError(f"Package {package_name} not available and no fallback provided")
    
    def require_packages(self, *package_names: str) -> None:
        """
        Require packages to be available.
        
        Args:
            *package_names: Names of required packages
            
        Raises:
            ImportError: If any required package is not available
        """
        missing = self.get_missing_packages(list(package_names))
        if missing:
            raise ImportError(f"Required packages not available: {', '.join(missing)}")
    
    def warn_missing_packages(self, *package_names: str) -> None:
        """
        Warn about missing packages.
        
        Args:
            *package_names: Names of packages to check
        """
        missing = self.get_missing_packages(list(package_names))
        if missing:
            warnings.warn(f"Optional packages not available: {', '.join(missing)}", 
                         UserWarning, stacklevel=2)


# Global dependency manager instance
_global_dependency_manager: Optional[DependencyManager] = None


def get_dependency_manager() -> DependencyManager:
    """Get the global dependency manager instance."""
    global _global_dependency_manager
    if _global_dependency_manager is None:
        _global_dependency_manager = DependencyManager()
    return _global_dependency_manager


def requires_package(*package_names: str):
    """
    Decorator to require packages for a function.
    
    Args:
        *package_names: Names of required packages
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_dependency_manager()
            manager.require_packages(*package_names)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def optional_package(*package_names: str):
    """
    Decorator to warn about missing optional packages.
    
    Args:
        *package_names: Names of optional packages
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_dependency_manager()
            manager.warn_missing_packages(*package_names)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def safe_import(package_name: str, import_name: str | None = None) -> Any:
    """Safely import a package with fallback."""
    manager = get_dependency_manager()
    return manager.safe_import(package_name, import_name)


def is_package_available(package_name: str) -> bool:
    """Check if a package is available."""
    manager = get_dependency_manager()
    return manager.is_available(package_name)


# Example fallback implementations
class NumpyFallback:
    """Fallback implementation for numpy."""
    
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    def __getattr__(self, name):
        raise ImportError("numpy is not available")


class PandasFallback:
    """Fallback implementation for pandas."""
    
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    def __getattr__(self, name):
        raise ImportError("pandas is not available")


# Register fallback implementations
def _register_fallback_implementations():
    """Register fallback implementations for common packages."""
    manager = get_dependency_manager()
    
    manager.register_optional_package('numpy', fallback_class=NumpyFallback)
    manager.register_optional_package('pandas', fallback_class=PandasFallback)


# Initialize fallback implementations
_register_fallback_implementations()


# Example usage
if __name__ == "__main__":
    manager = get_dependency_manager()
    
    tprint("Available packages:", manager.get_available_packages())
    
    # Check if numpy is available
    if manager.is_available('numpy'):
        tprint("numpy is available")
    else:
        tprint("numpy is not available")
    
    # Try to import with fallback
    try:
        np = safe_import('numpy')
        tprint("Successfully imported numpy or fallback")
    except ImportError as e:
        tprint(f"Failed to import numpy: {e}")
