"""
Incremental Computation for Feature Generation

This module implements incremental computation to maintain rolling features
efficiently by updating state incrementally instead of recalculating from scratch.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Deque
from dataclasses import dataclass, field
from collections import deque
import threading
import time

# Import existing vectorization components for integration
from .vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
from .unified_vectorization_manager import get_unified_vectorization_manager

logger = logging.getLogger(__name__)

@dataclass
class IncrementalConfig:
    """Configuration for incremental processing."""
    max_window_size: int = 1000
    memory_efficient: bool = True
    enable_thread_safety: bool = True
    cleanup_frequency: int = 1000  # Cleanup every N updates
    persist_state: bool = True
    use_vectorbt_optimizer: bool = True
    use_unified_vectorization: bool = True

@dataclass
class RollingState:
    """State for rolling calculations."""
    buffer: Deque[float] = field(default_factory=deque)
    sum: float = 0.0
    sum_squares: float = 0.0
    count: int = 0
    last_update: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if self.buffer is None:
            self.buffer = deque(maxlen=self.maxlen)

class IncrementalRollingMean:
    """Incremental rolling mean calculator."""
    
    def __init__(self, window: int, config: Optional[IncrementalConfig] = None):
        self.window = window
        self.config = config or IncrementalConfig()
        self.state = RollingState()
        self.state.buffer = deque(maxlen=window)
        self.lock = threading.Lock() if self.config.enable_thread_safety else None
        
        self.stats = {
            'updates': 0,
            'computations_saved': 0,
            'last_cleanup': time.time()
        }
    
    def update(self, new_value: float) -> float:
        """Update with new value and return current mean."""
        if self.lock:
            with self.lock:
                return self._update_unsafe(new_value)
        else:
            return self._update_unsafe(new_value)
    
    def _update_unsafe(self, new_value: float) -> float:
        """Update without thread safety."""
        # Remove oldest value if buffer is full
        if len(self.state.buffer) == self.window:
            old_value = self.state.buffer[0]
            self.state.sum -= old_value
            self.state.sum_squares -= old_value * old_value
        
        # Add new value
        self.state.buffer.append(new_value)
        self.state.sum += new_value
        self.state.sum_squares += new_value * new_value
        self.state.count = len(self.state.buffer)
        self.state.last_update = time.time()
        
        self.stats['updates'] += 1
        
        # Periodic cleanup
        if self.stats['updates'] % self.config.cleanup_frequency == 0:
            self._cleanup()
        
        return self.get_current_mean()
    
    def get_current_mean(self) -> float:
        """Get current rolling mean."""
        if self.state.count == 0:
            return 0.0
        return self.state.sum / self.state.count
    
    def _cleanup(self):
        """Cleanup and optimize state."""
        current_time = time.time()
        if current_time - self.stats['last_cleanup'] > 3600:  # 1 hour
            # Only trigger GC if system memory usage is high
            try:
                import psutil  # Local import to avoid hard dependency at module import
                vm = psutil.virtual_memory()
                if vm.percent >= 80:
                    import gc
                    gc.collect()
            except Exception:
                pass
            self.stats['last_cleanup'] = current_time

class IncrementalRollingStd:
    """Incremental rolling standard deviation calculator."""
    
    def __init__(self, window: int, config: Optional[IncrementalConfig] = None):
        self.window = window
        self.config = config or IncrementalConfig()
        self.state = RollingState()
        self.state.buffer = deque(maxlen=window)
        self.lock = threading.Lock() if self.config.enable_thread_safety else None
        
        self.stats = {
            'updates': 0,
            'computations_saved': 0
        }
    
    def update(self, new_value: float) -> float:
        """Update with new value and return current std."""
        if self.lock:
            with self.lock:
                return self._update_unsafe(new_value)
        else:
            return self._update_unsafe(new_value)
    
    def _update_unsafe(self, new_value: float) -> float:
        """Update without thread safety."""
        # Remove oldest value if buffer is full
        if len(self.state.buffer) == self.window:
            old_value = self.state.buffer[0]
            self.state.sum -= old_value
            self.state.sum_squares -= old_value * old_value
        
        # Add new value
        self.state.buffer.append(new_value)
        self.state.sum += new_value
        self.state.sum_squares += new_value * new_value
        self.state.count = len(self.state.buffer)
        self.state.last_update = time.time()
        
        self.stats['updates'] += 1
        
        return self.get_current_std()
    
    def get_current_std(self) -> float:
        """Get current rolling standard deviation."""
        if self.state.count <= 1:
            return 0.0
        
        mean = self.state.sum / self.state.count
        variance = (self.state.sum_squares / self.state.count) - (mean * mean)
        return np.sqrt(max(0, variance))

class IncrementalRSI:
    """Incremental RSI calculator."""
    
    def __init__(self, window: int = 14, config: Optional[IncrementalConfig] = None):
        self.window = window
        self.config = config or IncrementalConfig()
        
        self.gain_state = RollingState()
        self.loss_state = RollingState()
        self.gain_state.buffer = deque(maxlen=window)
        self.loss_state.buffer = deque(maxlen=window)
        
        self.last_price = None
        self.lock = threading.Lock() if self.config.enable_thread_safety else None
        
        self.stats = {
            'updates': 0,
            'computations_saved': 0
        }
    
    def update(self, new_price: float) -> float:
        """Update with new price and return current RSI."""
        if self.lock:
            with self.lock:
                return self._update_unsafe(new_price)
        else:
            return self._update_unsafe(new_price)
    
    def _update_unsafe(self, new_price: float) -> float:
        """Update without thread safety."""
        if self.last_price is None:
            self.last_price = new_price
            return 50.0  # Neutral RSI for first value
        
        # Calculate price change
        change = new_price - self.last_price
        gain = max(0, change)
        loss = max(0, -change)
        
        # Update gain state
        self._update_rolling_state(self.gain_state, gain)
        
        # Update loss state
        self._update_rolling_state(self.loss_state, loss)
        
        self.last_price = new_price
        self.stats['updates'] += 1
        
        return self.get_current_rsi()
    
    def _update_rolling_state(self, state: RollingState, new_value: float):
        """Update a rolling state."""
        # Remove oldest value if buffer is full
        if len(state.buffer) == self.window:
            old_value = state.buffer[0]
            state.sum -= old_value
        
        # Add new value
        state.buffer.append(new_value)
        state.sum += new_value
        state.count = len(state.buffer)
        state.last_update = time.time()
    
    def get_current_rsi(self) -> float:
        """Get current RSI value."""
        if self.gain_state.count == 0 or self.loss_state.count == 0:
            return 50.0
        
        avg_gain = self.gain_state.sum / self.gain_state.count
        avg_loss = self.loss_state.sum / self.loss_state.count
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class IncrementalFeatureProcessor:
    """
    Incremental feature processor that maintains rolling features efficiently.
    """
    
    def __init__(self, config: Optional[IncrementalConfig] = None):
        """Initialize the incremental processor."""
        self.config = config or IncrementalConfig()
        self.logger = logger.getChild('IncrementalFeatureProcessor')
        
        # Store incremental calculators for each feature
        self.calculators: Dict[str, Dict[str, Any]] = {}
        
        # Initialize vectorization components
        self.vectorbt_optimizer = None
        self.unified_vectorization_manager = None
        
        if self.config.use_vectorbt_optimizer:
            try:
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
                self.logger.info("✅ VectorBT Rolling Optimizer integrated for incremental processing")
            except Exception as e:
                self.logger.warning(f"VectorBT Rolling Optimizer not available: {e}")
        
        if self.config.use_unified_vectorization:
            try:
                self.unified_vectorization_manager = get_unified_vectorization_manager()
                self.logger.info("✅ Unified Vectorization Manager integrated for incremental processing")
            except Exception as e:
                self.logger.warning(f"Unified Vectorization Manager not available: {e}")
        
        # Performance tracking
        self.stats = {
            'features_tracked': 0,
            'total_updates': 0,
            'computations_saved': 0,
            'memory_usage_mb': 0.0,
            'vectorbt_operations': 0,
            'unified_operations': 0
        }
        
        self.logger.info("🚀 IncrementalFeatureProcessor initialized")
    
    def add_feature(self, feature_name: str, feature_type: str, **kwargs) -> bool:
        """
        Add a feature to track incrementally.
        
        Args:
            feature_name: Name of the feature
            feature_type: Type of feature ('rolling_mean', 'rolling_std', 'rsi', etc.)
            **kwargs: Additional parameters for the feature
            
        Returns:
            True if feature was added successfully
        """
        try:
            if feature_type == 'rolling_mean':
                window = kwargs.get('window', 20)
                calculator = IncrementalRollingMean(window, self.config)
            elif feature_type == 'rolling_std':
                window = kwargs.get('window', 20)
                calculator = IncrementalRollingStd(window, self.config)
            elif feature_type == 'rsi':
                window = kwargs.get('window', 14)
                calculator = IncrementalRSI(window, self.config)
            else:
                self.logger.error(f"Unknown feature type: {feature_type}")
                return False
            
            if feature_name not in self.calculators:
                self.calculators[feature_name] = {}
            
            self.calculators[feature_name][feature_type] = calculator
            self.stats['features_tracked'] = len(self.calculators)
            
            self.logger.debug(f"✅ Added incremental feature: {feature_name} ({feature_type})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add feature {feature_name}: {e}")
            return False
    
    def update_features(self, data: Dict[str, float]) -> Dict[str, float]:
        """
        Update all tracked features with new data.
        
        Args:
            data: Dictionary of column_name -> value
            
        Returns:
            Dictionary of feature_name -> current_value
        """
        results = {}
        
        for feature_name, feature_calculators in self.calculators.items():
            for feature_type, calculator in feature_calculators.items():
                try:
                    # Extract the appropriate data column for this feature
                    column_name = self._get_column_for_feature(feature_name, feature_type)
                    
                    if column_name in data:
                        current_value = calculator.update(data[column_name])
                        results[f"{feature_name}_{feature_type}"] = current_value
                        self.stats['total_updates'] += 1
                    
                except Exception as e:
                    self.logger.error(f"Error updating {feature_name}: {e}")
        
        return results
    
    def get_current_features(self) -> Dict[str, float]:
        """Get current values of all tracked features."""
        results = {}
        
        for feature_name, feature_calculators in self.calculators.items():
            for feature_type, calculator in feature_calculators.items():
                try:
                    if feature_type in ['rolling_mean', 'rolling_std']:
                        current_value = calculator.get_current_mean() if hasattr(calculator, 'get_current_mean') else calculator.get_current_std()
                    elif feature_type == 'rsi':
                        current_value = calculator.get_current_rsi()
                    else:
                        current_value = 0.0
                    
                    results[f"{feature_name}_{feature_type}"] = current_value
                    
                except Exception as e:
                    self.logger.error(f"Error getting current value for {feature_name}: {e}")
        
        return results
    
    def _get_column_for_feature(self, feature_name: str, feature_type: str) -> str:
        """Determine which data column to use for a feature."""
        # Simple heuristic - could be made more sophisticated
        if 'close' in feature_name.lower():
            return 'close'
        elif 'volume' in feature_name.lower():
            return 'volume'
        elif 'high' in feature_name.lower():
            return 'high'
        elif 'low' in feature_name.lower():
            return 'low'
        else:
            return 'close'  # Default to close price
    
    def process_dataframe_incremental(self, data: pd.DataFrame, feature_specs: Dict[str, Any]) -> pd.DataFrame:
        """
        Process a DataFrame using hybrid incremental/vectorized computation.
        
        Args:
            data: Input DataFrame
            feature_specs: Feature specifications
            
        Returns:
            DataFrame with incremental features
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"🔄 Starting hybrid incremental processing for {len(data)} rows")
        
        # Determine processing strategy based on data size and available components
        use_hybrid = (len(data) > 1000 and 
                     (self.vectorbt_optimizer or self.unified_vectorization_manager))
        
        if use_hybrid:
            self.logger.info("🚀 Using hybrid incremental/vectorized processing")
            return self._process_hybrid_incremental(data, feature_specs, start_time)
        else:
            self.logger.info("🔄 Using pure incremental processing")
            return self._process_pure_incremental(data, feature_specs, start_time)
    
    def _process_hybrid_incremental(self, data: pd.DataFrame, feature_specs: Dict[str, Any], start_time: float) -> pd.DataFrame:
        """Process using hybrid incremental and vectorized approach."""
        # Use vectorized operations for initial computation
        vectorized_features = {}
        
        for column, specs in feature_specs.items():
            if column in data.columns:
                column_data = data[column]
                
                # Try VectorBT optimizer first
                if self.vectorbt_optimizer and len(column_data) > 1000:
                    try:
                        column_features = self._compute_vectorized_features_vectorbt(column, column_data, specs)
                        vectorized_features.update(column_features)
                        self.stats['vectorbt_operations'] += 1
                        continue
                    except Exception as e:
                        self.logger.debug(f"VectorBT hybrid failed for {column}: {e}")
                
                # Try unified vectorization manager
                if self.unified_vectorization_manager and len(column_data) > 500:
                    try:
                        column_features = self._compute_vectorized_features_unified(column, column_data, specs)
                        vectorized_features.update(column_features)
                        self.stats['unified_operations'] += 1
                        continue
                    except Exception as e:
                        self.logger.debug(f"Unified hybrid failed for {column}: {e}")
                
                # Fallback to incremental for smaller datasets or when vectorization fails
                column_features = self._compute_incremental_features(column, column_data, specs)
                vectorized_features.update(column_features)
        
        # Combine with original data
        result_df = pd.concat([data, pd.DataFrame(vectorized_features, index=data.index)], axis=1)
        
        processing_time = time.time() - start_time
        self.logger.info(f"✅ Hybrid processing completed: {len(result_df.columns)} features in {processing_time:.2f}s")
        
        return result_df
    
    def _process_pure_incremental(self, data: pd.DataFrame, feature_specs: Dict[str, Any], start_time: float) -> pd.DataFrame:
        """Process using pure incremental approach."""
        # Add features based on specifications
        for column, specs in feature_specs.items():
            if column in data.columns:
                if 'rolling_mean' in specs:
                    for window in specs['rolling_mean'].get('windows', [20]):
                        feature_name = f"{column}_sma_{window}"
                        self.add_feature(feature_name, 'rolling_mean', window=window)
                
                if 'rolling_std' in specs:
                    for window in specs['rolling_std'].get('windows', [20]):
                        feature_name = f"{column}_std_{window}"
                        self.add_feature(feature_name, 'rolling_std', window=window)
                
                if 'rsi' in specs:
                    for window in specs['rsi'].get('windows', [14]):
                        feature_name = f"{column}_rsi_{window}"
                        self.add_feature(feature_name, 'rsi', window=window)
        
        # Process data incrementally
        results = []
        
        for idx, row in data.iterrows():
            # Convert row to dictionary
            row_data = row.to_dict()
            
            # Update features
            feature_values = self.update_features(row_data)
            
            # Add original data
            result_row = {**row_data, **feature_values}
            results.append(result_row)
        
        # Convert to DataFrame
        result_df = pd.DataFrame(results, index=data.index)
        
        processing_time = time.time() - start_time
        self.logger.info(f"✅ Pure incremental processing completed: {len(result_df.columns)} features in {processing_time:.2f}s")
        
        return result_df
    
    def _compute_vectorized_features_vectorbt(self, column_name: str, data: pd.Series, specs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute features using VectorBT optimizer."""
        features = {}

        try:
            if 'rolling_mean' in specs:
                for window in specs['rolling_mean'].get('windows', [20]):
                    s = self.vectorbt_optimizer.rolling_mean(data, window)
                    features[f"{column_name}_sma_{window}"] = s.astype(np.float32, copy=False) if hasattr(s, 'astype') else s
            
            if 'rolling_std' in specs:
                for window in specs['rolling_std'].get('windows', [20]):
                    s = self.vectorbt_optimizer.rolling_std(data, window)
                    features[f"{column_name}_std_{window}"] = s.astype(np.float32, copy=False) if hasattr(s, 'astype') else s
            
            if 'rsi' in specs:
                for window in specs['rsi'].get('windows', [14]):
                    s = self.vectorbt_optimizer.rsi(data, window)
                    features[f"{column_name}_rsi_{window}"] = s.astype(np.float32, copy=False) if hasattr(s, 'astype') else s

        except Exception as e:
            self.logger.warning(f"VectorBT feature computation failed for {column_name}: {e}")
            raise
        
        return features
    
    def _compute_vectorized_features_unified(self, column_name: str, data: pd.Series, specs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute features using unified vectorization manager."""
        features = {}
        
        try:
            # Prepare operations for batch processing
            operations = []
            
            if 'rolling_mean' in specs:
                for window in specs['rolling_mean'].get('windows', [20]):
                    operations.append({
                        'operation': 'rolling_mean',
                        'data': data,
                        'window': window,
                        'feature_name': f"{column_name}_sma_{window}"
                    })
            
            if 'rolling_std' in specs:
                for window in specs['rolling_std'].get('windows', [20]):
                    operations.append({
                        'operation': 'rolling_std',
                        'data': data,
                        'window': window,
                        'feature_name': f"{column_name}_std_{window}"
                    })
            
            if 'rsi' in specs:
                for window in specs['rsi'].get('windows', [14]):
                    operations.append({
                        'operation': 'rsi',
                        'data': data,
                        'window': window,
                        'feature_name': f"{column_name}_rsi_{window}"
                    })
            
            # Execute batch operations
            results = self.unified_vectorization_manager.batch_operations(operations)
            
            # Extract results
            for result in results:
                if result.get('success'):
                    features[result['feature_name']] = result['result']
        
        except Exception as e:
            self.logger.warning(f"Unified vectorization feature computation failed for {column_name}: {e}")
            raise
        
        return features
    
    def _compute_incremental_features(self, column_name: str, data: pd.Series, specs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute features using incremental approach."""
        features = {}
        
        # Add features to incremental tracking
        if 'rolling_mean' in specs:
            for window in specs['rolling_mean'].get('windows', [20]):
                feature_name = f"{column_name}_sma_{window}"
                self.add_feature(feature_name, 'rolling_mean', window=window)
        
        if 'rolling_std' in specs:
            for window in specs['rolling_std'].get('windows', [20]):
                feature_name = f"{column_name}_std_{window}"
                self.add_feature(feature_name, 'rolling_std', window=window)
        
        if 'rsi' in specs:
            for window in specs['rsi'].get('windows', [14]):
                feature_name = f"{column_name}_rsi_{window}"
                self.add_feature(feature_name, 'rsi', window=window)
        
        # Compute incrementally
        for idx, value in data.items():
            row_data = {column_name: value}
            feature_values = self.update_features(row_data)
            for key, val in feature_values.items():
                if key not in features:
                    features[key] = []
                features[key].append(val)
        
        # Convert lists to pandas Series
        for key in features:
            features[key] = pd.Series(features[key], index=data.index)
        
        return features
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        
        # Add calculator-specific stats
        for feature_name, calculators in self.calculators.items():
            for feature_type, calculator in calculators.items():
                if hasattr(calculator, 'stats'):
                    stats[f"{feature_name}_{feature_type}_updates"] = calculator.stats.get('updates', 0)
        
        return stats
    
    def clear_state(self):
        """Clear all incremental state."""
        self.calculators.clear()
        self.stats['features_tracked'] = 0
        self.stats['total_updates'] = 0
        self.logger.info("🧹 Incremental processor state cleared")

# Global instance
_incremental_processor: Optional[IncrementalFeatureProcessor] = None

def get_incremental_processor(config: Optional[IncrementalConfig] = None) -> IncrementalFeatureProcessor:
    """Get the global incremental processor instance."""
    global _incremental_processor
    if _incremental_processor is None:
        _incremental_processor = IncrementalFeatureProcessor(config)
    return _incremental_processor

def process_features_incremental(data: pd.DataFrame, 
                               feature_specs: Dict[str, Any],
                               config: Optional[IncrementalConfig] = None) -> pd.DataFrame:
    """
    Convenience function to process features using incremental computation.
    
    Args:
        data: Input DataFrame
        feature_specs: Feature specifications
        config: Optional configuration
        
    Returns:
        DataFrame with incremental features
    """
    processor = get_incremental_processor(config)
    return processor.process_dataframe_incremental(data, feature_specs)
