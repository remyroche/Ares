"""
Lazy Feature Specifications

This module provides lazy loading of feature specifications to reduce
startup time and memory usage.
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

class LazyFeatureSpecs:
    """
    Lazy loader for feature specifications that only loads specs when needed.
    """
    
    def __init__(self):
        self.logger = logger.getChild('LazyFeatureSpecs')
        self._spec_cache = {}
        self._load_times = {}
        
        # Common feature specifications
        self._common_specs = {
            'basic_ohlcv': self._get_basic_ohlcv_specs,
            'technical_indicators': self._get_technical_indicators_specs,
            'rolling_features': self._get_rolling_features_specs,
            'statistical_features': self._get_statistical_features_specs,
            'volume_features': self._get_volume_features_specs,
            'volatility_features': self._get_volatility_features_specs
        }
    
    def get_specs(self, spec_name: str, **kwargs) -> Dict[str, Any]:
        """
        Get feature specifications lazily.
        
        Args:
            spec_name: Name of the specification set
            **kwargs: Additional parameters for the specification
            
        Returns:
            Feature specifications dictionary
        """
        cache_key = f"{spec_name}_{hash(str(sorted(kwargs.items())))}"
        
        if cache_key in self._spec_cache:
            return self._spec_cache[cache_key]
        
        start_time = time.time()
        
        try:
            if spec_name in self._common_specs:
                specs = self._common_specs[spec_name](**kwargs)
            else:
                specs = self._get_custom_specs(spec_name, **kwargs)
            
            self._spec_cache[cache_key] = specs
            self._load_times[cache_key] = time.time() - start_time
            
            self.logger.debug(f"📋 Loaded specs '{spec_name}' in {self._load_times[cache_key]:.3f}s")
            
            return specs
            
        except Exception as e:
            self.logger.error(f"Failed to load specs '{spec_name}': {e}")
            return {}
    
    @lru_cache(maxsize=128)
    def _get_basic_ohlcv_specs(self, windows: str = "20,50,100") -> Dict[str, Any]:
        """Get basic OHLCV feature specifications."""
        window_list = [int(w.strip()) for w in windows.split(',')]
        
        return {
            'close': {
                'rolling': {
                    'windows': window_list,
                    'functions': ['mean', 'std', 'min', 'max', 'median']
                },
                'technical': {
                    'functions': ['rsi', 'macd', 'bollinger']
                }
            },
            'volume': {
                'rolling': {
                    'windows': window_list,
                    'functions': ['mean', 'std', 'sum']
                },
                'statistical': {
                    'functions': ['skew', 'kurt']
                }
            },
            'high': {
                'rolling': {
                    'windows': window_list,
                    'functions': ['max', 'mean']
                }
            },
            'low': {
                'rolling': {
                    'windows': window_list,
                    'functions': ['min', 'mean']
                }
            },
            'open': {
                'rolling': {
                    'windows': window_list,
                    'functions': ['mean', 'std']
                }
            }
        }
    
    @lru_cache(maxsize=64)
    def _get_technical_indicators_specs(self, windows: str = "14,21") -> Dict[str, Any]:
        """Get technical indicators specifications."""
        window_list = [int(w.strip()) for w in windows.split(',')]
        
        return {
            'close': {
                'technical': {
                    'functions': ['rsi', 'macd', 'bollinger', 'stochastic', 'williams_r'],
                    'windows': window_list
                }
            }
        }
    
    @lru_cache(maxsize=64)
    def _get_rolling_features_specs(self, windows: str = "5,10,20,50,100") -> Dict[str, Any]:
        """Get rolling features specifications."""
        window_list = [int(w.strip()) for w in windows.split(',')]
        
        return {
            'close': {
                'rolling': {
                    'windows': window_list,
                    'functions': ['mean', 'std', 'min', 'max', 'median', 'sum', 'count']
                }
            },
            'volume': {
                'rolling': {
                    'windows': window_list,
                    'functions': ['mean', 'std', 'sum', 'max']
                }
            }
        }
    
    @lru_cache(maxsize=64)
    def _get_statistical_features_specs(self) -> Dict[str, Any]:
        """Get statistical features specifications."""
        return {
            'close': {
                'statistical': {
                    'functions': ['skew', 'kurt', 'quantile'],
                    'quantiles': [0.25, 0.75, 0.9, 0.95, 0.99]
                }
            },
            'volume': {
                'statistical': {
                    'functions': ['skew', 'kurt']
                }
            }
        }
    
    @lru_cache(maxsize=64)
    def _get_volume_features_specs(self, windows: str = "10,20,50") -> Dict[str, Any]:
        """Get volume-specific features specifications."""
        window_list = [int(w.strip()) for w in windows.split(',')]
        
        return {
            'volume': {
                'rolling': {
                    'windows': window_list,
                    'functions': ['mean', 'std', 'max', 'sum']
                },
                'statistical': {
                    'functions': ['skew', 'kurt']
                }
            },
            'close': {
                'rolling': {
                    'windows': window_list,
                    'functions': ['mean']
                }
            }
        }
    
    @lru_cache(maxsize=64)
    def _get_volatility_features_specs(self, windows: str = "10,20,30") -> Dict[str, Any]:
        """Get volatility features specifications."""
        window_list = [int(w.strip()) for w in windows.split(',')]
        
        return {
            'close': {
                'rolling': {
                    'windows': window_list,
                    'functions': ['std']
                },
                'statistical': {
                    'functions': ['skew', 'kurt']
                }
            }
        }
    
    def _get_custom_specs(self, spec_name: str, **kwargs) -> Dict[str, Any]:
        """Get custom feature specifications."""
        # This could be extended to load from files, databases, etc.
        self.logger.warning(f"Custom specs '{spec_name}' not found, returning empty specs")
        return {}
    
    def get_combined_specs(self, spec_names: List[str], **kwargs) -> Dict[str, Any]:
        """
        Get combined specifications from multiple spec sets.
        
        Args:
            spec_names: List of specification names to combine
            **kwargs: Additional parameters
            
        Returns:
            Combined feature specifications
        """
        combined_specs = {}
        
        for spec_name in spec_names:
            specs = self.get_specs(spec_name, **kwargs)
            
            # Merge specifications
            for column, column_specs in specs.items():
                if column not in combined_specs:
                    combined_specs[column] = {}
                
                for feature_type, features in column_specs.items():
                    if feature_type not in combined_specs[column]:
                        combined_specs[column][feature_type] = {}
                    
                    if isinstance(features, dict):
                        # Merge dictionaries
                        combined_specs[column][feature_type].update(features)
                    else:
                        # Replace lists
                        combined_specs[column][feature_type] = features
        
        return combined_specs
    
    def clear_cache(self):
        """Clear the specification cache."""
        self._spec_cache.clear()
        self._load_times.clear()
        self.logger.info("🧹 Feature specification cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cached_specs': len(self._spec_cache),
            'total_load_time': sum(self._load_times.values()),
            'average_load_time': sum(self._load_times.values()) / len(self._load_times) if self._load_times else 0,
            'cache_hit_rate': len(self._spec_cache) / max(1, len(self._load_times))
        }

# Global instance
_lazy_feature_specs: Optional[LazyFeatureSpecs] = None

def get_lazy_feature_specs() -> LazyFeatureSpecs:
    """Get the global lazy feature specs instance."""
    global _lazy_feature_specs
    if _lazy_feature_specs is None:
        _lazy_feature_specs = LazyFeatureSpecs()
    return _lazy_feature_specs

def get_feature_specs(spec_name: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to get feature specifications."""
    return get_lazy_feature_specs().get_specs(spec_name, **kwargs)
