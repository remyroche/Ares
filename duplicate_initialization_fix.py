"""
Duplicate Initialization Fix
============================

This file provides solutions for the duplicate DataCleaner initialization issue.
"""

from typing import Dict, Any, Optional
import logging
from functools import lru_cache
import threading

# Solution 1: Singleton Pattern with Thread Safety
class SingletonDataCleaner:
    """Thread-safe singleton DataCleaner to prevent duplicate initialization."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, data_type: str = 'klines'):
        # Only initialize once
        if not hasattr(self, '_initialized'):
            self.logger = logging.getLogger('DataCleaner')
            self.data_type = data_type
            self._initialized = True
            self.logger.info(f"🔧 DataCleaner singleton initialized with data_type='{data_type}'")

# Solution 2: Factory Pattern with Registry
class DataCleanerFactory:
    """Factory for creating and managing DataCleaner instances."""
    
    _instances: Dict[str, Any] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_cleaner(cls, data_type: str = 'klines', **kwargs):
        """Get or create DataCleaner instance for specific data type."""
        key = f"cleaner_{data_type}"
        
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    # Import here to avoid circular imports
                    from src.utils.data.quality.data_cleaning import DataCleaner
                    cls._instances[key] = DataCleaner(data_type=data_type, **kwargs)
                    logging.getLogger('DataCleanerFactory').info(
                        f"🏭 Created DataCleaner for data_type='{data_type}'"
                    )
        
        return cls._instances[key]
    
    @classmethod
    def get_all_instances(cls) -> Dict[str, Any]:
        """Get all created instances."""
        return cls._instances.copy()
    
    @classmethod
    def clear_instances(cls):
        """Clear all instances (for testing)."""
        with cls._lock:
            cls._instances.clear()

# Solution 3: Cached Factory Function
@lru_cache(maxsize=10)
def get_data_cleaner(data_type: str = 'klines', **kwargs):
    """Cached function to get DataCleaner instances."""
    from src.utils.data.quality.data_cleaning import DataCleaner
    return DataCleaner(data_type=data_type, **kwargs)

# Solution 4: Module-level Singleton
class ModuleDataCleaner:
    """Module-level singleton for DataCleaner."""
    
    _instance: Optional[Any] = None
    
    @classmethod
    def get_instance(cls, data_type: str = 'klines', **kwargs):
        """Get singleton instance."""
        if cls._instance is None:
            from src.utils.data.quality.data_cleaning import DataCleaner
            cls._instance = DataCleaner(data_type=data_type, **kwargs)
            logging.getLogger('ModuleDataCleaner').info(
                f"📦 Module-level DataCleaner created for data_type='{data_type}'"
            )
        return cls._instance

# Recommended Implementation: Centralized DataCleaner Manager
class DataCleanerManager:
    """Centralized manager for DataCleaner instances."""
    
    def __init__(self):
        self.logger = logging.getLogger('DataCleanerManager')
        self._cleaners: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def get_cleaner(self, data_type: str = 'klines', **kwargs):
        """Get or create DataCleaner for specific data type."""
        if data_type not in self._cleaners:
            with self._lock:
                if data_type not in self._cleaners:
                    from src.utils.data.quality.data_cleaning import DataCleaner
                    self._cleaners[data_type] = DataCleaner(data_type=data_type, **kwargs)
                    self.logger.info(f"🔧 Created DataCleaner for data_type='{data_type}'")
        
        return self._cleaners[data_type]
    
    def get_all_cleaners(self) -> Dict[str, Any]:
        """Get all created cleaners."""
        return self._cleaners.copy()
    
    def clear_cleaners(self):
        """Clear all cleaners (for testing)."""
        with self._lock:
            self._cleaners.clear()

# Global manager instance
data_cleaner_manager = DataCleanerManager()

# Convenience function for modules to use
def get_data_cleaner(data_type: str = 'klines', **kwargs):
    """Get DataCleaner instance through centralized manager."""
    return data_cleaner_manager.get_cleaner(data_type, **kwargs)

# Migration guide for existing code
MIGRATION_GUIDE = """
Migration Guide for Duplicate DataCleaner Initialization
========================================================

BEFORE (causing duplicates):
```python
# In multiple files:
data_cleaner = DataCleaner(data_type='klines')
```

AFTER (using centralized manager):
```python
# Option 1: Use centralized manager
from src.utils.data.quality.data_cleaning import get_data_cleaner
data_cleaner = get_data_cleaner(data_type='klines')

# Option 2: Use factory pattern
from src.utils.data.quality.data_cleaning import DataCleanerFactory
data_cleaner = DataCleanerFactory.get_cleaner(data_type='klines')

# Option 3: Use singleton pattern
from src.utils.data.quality.data_cleaning import SingletonDataCleaner
data_cleaner = SingletonDataCleaner(data_type='klines')
```

Files that need to be updated:
1. src/training/steps/market_analysis/regime_data_splitting/main.py:1292
2. src/training/steps/market_analysis/enhanced_validation_framework.py:65
3. src/training/steps/data_collection/data_preparation/enhanced_data_quality_manager.py:52
"""

# Performance impact analysis
PERFORMANCE_ANALYSIS = """
Performance Impact Analysis
==========================

Current Issue:
- Multiple DataCleaner instances created
- Each instance logs initialization message
- Memory overhead from duplicate objects
- Potential configuration inconsistencies

Benefits of Fix:
- Single instance per data type
- Reduced memory usage
- Consistent configuration
- Cleaner log output
- Better performance monitoring

Estimated Impact:
- Memory reduction: ~50-80% for DataCleaner objects
- Log reduction: Eliminate duplicate messages
- Initialization time: Slight improvement due to singleton pattern
"""