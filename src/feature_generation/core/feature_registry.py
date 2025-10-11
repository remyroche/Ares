"""
Feature Registry

This module provides the FeatureRegistry class for managing and organizing
feature generators by category and name.
"""

import logging
import traceback
import inspect
from typing import Dict, List, Optional, Set
from collections import defaultdict
from datetime import datetime

from .feature_generator import FeatureGenerator, FeatureCategory

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)

class FeatureRegistry:
    """
    Registry for managing feature generators by category and name.
    
    The FeatureRegistry provides a centralized way to organize and access
    feature generators, making it easy to find generators by category
    or specific feature name.
    """
    
    def __init__(self):
        """Initialize the feature registry."""
        self.logger = logger.getChild('FeatureRegistry')
        
        # Registry storage
        self._generators_by_name: Dict[str, FeatureGenerator] = {}
        self._generators_by_category: Dict[FeatureCategory, List[FeatureGenerator]] = defaultdict(list)
        self._category_names: Set[str] = set()
        
        # Registration tracking for enhanced warnings
        self._registration_info: Dict[str, Dict[str, any]] = {}
        self._registration_count: int = 0
        
        self.logger.info("FeatureRegistry initialized")
    
    def register(self, generator: FeatureGenerator) -> None:
        """
        Register a feature generator.
        
        Args:
            generator: Feature generator to register
        """
        name = generator.config.name
        category = generator.config.category
        self._registration_count += 1
        
        # Capture registration context
        registration_context = self._capture_registration_context()
        
        # Check for duplicate names and provide enhanced warning
        if name in self._generators_by_name:
            old_info = self._registration_info.get(name, {})
            self._log_overwrite_warning(name, old_info, registration_context)
        else:
            # First registration - log with context
            self.logger.debug(f"📝 First registration of generator: {name} ({category.value})")
        
        # Store registration info
        self._registration_info[name] = {
            'registration_count': self._registration_count,
            'timestamp': datetime.now().isoformat(),
            'category': category.value,
            'generator_class': generator.__class__.__name__,
            'config': {
                'period': getattr(generator.config, 'period', None),
                'base_calculation': str(getattr(generator.config, 'base_calculation', 'unknown')),
                'dependencies': getattr(generator.config, 'dependencies', [])
            },
            'call_stack': registration_context
        }
        
        # Register by name
        self._generators_by_name[name] = generator
        
        # Register by category
        if generator not in self._generators_by_category[category]:
            self._generators_by_category[category].append(generator)
        
        # Track category names
        self._category_names.add(category.value)
        
        self.logger.debug(f"✅ Registered generator: {name} ({category.value})")
    
    def _capture_registration_context(self) -> Dict[str, str]:
        """Capture the call stack context for registration tracking."""
        try:
            # Get the current call stack
            stack = traceback.extract_stack()
            
            # Find the caller that's not part of this registry
            caller_info = None
            for frame in reversed(stack):
                if 'feature_registry.py' not in frame.filename:
                    caller_info = {
                        'filename': frame.filename,
                        'function': frame.name,
                        'line': frame.lineno,
                        'code': frame.line.strip() if frame.line else 'N/A'
                    }
                    break
            
            # Fallback to the immediate caller
            if not caller_info and len(stack) > 1:
                frame = stack[-2]
                caller_info = {
                    'filename': frame.filename,
                    'function': frame.name,
                    'line': frame.lineno,
                    'code': frame.line.strip() if frame.line else 'N/A'
                }
            
            return caller_info or {'filename': 'unknown', 'function': 'unknown', 'line': 0, 'code': 'N/A'}
            
        except Exception as e:
            return {'error': f'Failed to capture context: {e}'}
    
    def _log_overwrite_warning(self, name: str, old_info: Dict[str, any], new_context: Dict[str, str]) -> None:
        """Log enhanced warning for overwriting existing generators."""
        # Extract key information
        old_timestamp = old_info.get('timestamp', 'unknown')
        old_class = old_info.get('generator_class', 'unknown')
        old_config = old_info.get('config', {})
        old_period = old_config.get('period', 'unknown')
        old_base_calc = old_config.get('base_calculation', 'unknown')
        
        new_filename = new_context.get('filename', 'unknown')
        new_function = new_context.get('function', 'unknown')
        new_line = new_context.get('line', 0)
        
        # Extract filename for cleaner logging
        if '/' in new_filename:
            new_filename = new_filename.split('/')[-1]
        elif '\\' in new_filename:
            new_filename = new_filename.split('\\')[-1]
        
        # Enhanced warning message
        warning_msg = (
            f"⚠️ Overwriting existing generator: {name}\n"
            f"   📊 Previous registration:\n"
            f"      - Class: {old_class}\n"
            f"      - Period: {old_period}\n"
            f"      - Base Calculation: {old_base_calc}\n"
            f"      - Timestamp: {old_timestamp}\n"
            f"   🔄 New registration from:\n"
            f"      - File: {new_filename}\n"
            f"      - Function: {new_function}\n"
            f"      - Line: {new_line}"
        )
        
        self.logger.warning(warning_msg)
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a feature generator by name.
        
        Args:
            name: Name of the generator to unregister
            
        Returns:
            True if generator was found and unregistered
        """
        if name not in self._generators_by_name:
            self.logger.warning(f"Generator not found: {name}")
            return False
        
        generator = self._generators_by_name[name]
        category = generator.config.category
        
        # Remove from name registry
        del self._generators_by_name[name]
        
        # Remove from category registry
        if generator in self._generators_by_category[category]:
            self._generators_by_category[category].remove(generator)
        
        # Clean up empty categories
        if not self._generators_by_category[category]:
            del self._generators_by_category[category]
            self._category_names.discard(category.value)
        
        self.logger.debug(f"Unregistered generator: {name}")
        return True
    
    def get_by_name(self, name: str) -> Optional[FeatureGenerator]:
        """
        Get a generator by name.
        
        Args:
            name: Generator name
            
        Returns:
            Generator or None if not found
        """
        return self._generators_by_name.get(name)
    
    def get_by_category(self, category: FeatureCategory) -> List[FeatureGenerator]:
        """
        Get all generators for a category.
        
        Args:
            category: Feature category
            
        Returns:
            List of generators for the category
        """
        return self._generators_by_category.get(category, []).copy()
    
    def get_all(self) -> List[FeatureGenerator]:
        """
        Get all registered generators.
        
        Returns:
            List of all generators
        """
        return list(self._generators_by_name.values())
    
    def list_names(self) -> List[str]:
        """
        List all registered generator names.
        
        Returns:
            List of generator names
        """
        return list(self._generators_by_name.keys())
    
    def list_categories(self) -> List[FeatureCategory]:
        """
        List all categories that have registered generators.
        
        Returns:
            List of categories
        """
        return list(self._generators_by_category.keys())
    
    def list_features(self, category: Optional[FeatureCategory] = None) -> List[str]:
        """
        List feature names, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of feature names
        """
        if category is None:
            return self.list_names()
        
        generators = self.get_by_category(category)
        return [gen.config.name for gen in generators]
    
    def get_category_stats(self) -> Dict[str, int]:
        """
        Get statistics about generators per category.
        
        Returns:
            Dictionary mapping category names to generator counts
        """
        stats = {}
        for category, generators in self._generators_by_category.items():
            stats[category.value] = len(generators)
        return stats
    
    def search_generators(self, 
                         name_pattern: Optional[str] = None,
                         category: Optional[FeatureCategory] = None,
                         description_pattern: Optional[str] = None) -> List[FeatureGenerator]:
        """
        Search for generators based on various criteria.
        
        Args:
            name_pattern: Pattern to match in generator names
            category: Category filter
            description_pattern: Pattern to match in descriptions
            
        Returns:
            List of matching generators
        """
        generators = self.get_all()
        
        # Filter by category
        if category is not None:
            generators = [g for g in generators if g.config.category == category]
        
        # Filter by name pattern
        if name_pattern is not None:
            generators = [g for g in generators if name_pattern.lower() in g.config.name.lower()]
        
        # Filter by description pattern
        if description_pattern is not None:
            generators = [g for g in generators 
                         if description_pattern.lower() in g.config.description.lower()]
        
        return generators
    
    def get_dependencies(self, name: str) -> List[str]:
        """
        Get dependencies for a generator.
        
        Args:
            name: Generator name
            
        Returns:
            List of dependency names
        """
        generator = self.get_by_name(name)
        if generator is None:
            return []
        
        return generator.config.dependencies.copy()
    
    def validate_dependencies(self) -> Dict[str, List[str]]:
        """
        Validate that all dependencies are satisfied.
        
        Returns:
            Dictionary mapping generator names to missing dependencies
        """
        missing_deps = {}
        
        for name, generator in self._generators_by_name.items():
            missing = []
            for dep in generator.config.dependencies:
                if dep not in self._generators_by_name:
                    missing.append(dep)
            
            if missing:
                missing_deps[name] = missing
        
        return missing_deps
    
    def get_generators_with_dependencies(self, name: str) -> List[FeatureGenerator]:
        """
        Get a generator and all its dependencies in dependency order.
        
        Args:
            name: Generator name
            
        Returns:
            List of generators in dependency order
        """
        generator = self.get_by_name(name)
        if generator is None:
            return []
        
        # Simple dependency resolution (no circular dependency detection)
        resolved = []
        to_resolve = [name]
        resolved_names = set()
        
        while to_resolve:
            current_name = to_resolve.pop(0)
            if current_name in resolved_names:
                continue
            
            current_generator = self.get_by_name(current_name)
            if current_generator is None:
                self.logger.warning(f"Dependency not found: {current_name}")
                continue
            
            # Add dependencies to resolution queue
            for dep in current_generator.config.dependencies:
                if dep not in resolved_names:
                    to_resolve.append(dep)
            
            resolved.append(current_generator)
            resolved_names.add(current_name)
        
        return resolved
    
    def clear(self) -> None:
        """Clear all registered generators."""
        self._generators_by_name.clear()
        self._generators_by_category.clear()
        self._category_names.clear()
        self.logger.info("Registry cleared")
    
    def get_summary(self) -> Dict[str, any]:
        """
        Get a summary of the registry.
        
        Returns:
            Dictionary with registry summary
        """
        return {
            'total_generators': len(self._generators_by_name),
            'categories': list(self._category_names),
            'category_stats': self.get_category_stats(),
            'missing_dependencies': self.validate_dependencies()
        }
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
