"""
Temporal Window Handler

Creates overlapping rolling windows for smooth regime transitions
with effective sample size tracking and meta-feature generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta

# Import existing utilities
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug, tprint_performance, tprint_progress, tprint_timer, tprint_logged, LogLevel
from src.utils.common_operations import safe_dataframe_operation
from src.utils.common_utilities import safe_dataframe_operation as safe_df_op
from src.utils.math_validation import validate_finite, safe_divide, safe_log, safe_sqrt
from src.utils.matrix_operations import get_unified_matrix_operations

from ..config.regime_discovery_config import RegimeDiscoveryConfig

logger = logging.getLogger(__name__)


class TemporalWindowHandler:
    """
    Handles temporal windows for regime discovery with smooth transitions.
    
    Features:
    - Overlapping rolling windows for smooth regime transitions
    - Effective sample size tracking (N_eff after overlap)
    - Meta-feature generation (time since last regime change)
    - Hardware-optimized window operations
    """
    
    def __init__(self, config: RegimeDiscoveryConfig):
        """Initialize the temporal window handler."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hardware optimization
        self._initialize_hardware_optimization()
        
        # Window parameters
        self.window_size = config.window_size
        self.overlap_pct = config.window_overlap_pct
        self.step_size = int(self.window_size * (1 - self.overlap_pct))
        
        # Effective sample size tracking
        self.n_effective_samples = 0
        self.window_metadata = {}
        
        tprint("🚀 TemporalWindowHandler initialized", "SUCCESS")
        self.logger.info("TemporalWindowHandler initialized successfully")
    
    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization utilities."""
        try:
            # Initialize unified matrix operations
            self.matrix_ops = get_unified_matrix_operations()
            
            if self.matrix_ops:
                tprint("✅ Matrix operations available for window processing", "SUCCESS")
            else:
                tprint("⚠️ Matrix operations not available, using standard operations", "WARNING")
                
        except Exception as e:
            tprint(f"❌ Hardware optimization initialization failed: {e}", "ERROR")
            self.matrix_ops = None
    
    def create_windows(self, data: pd.DataFrame, 
                      feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create overlapping rolling windows from time series data.
        
        Args:
            data: Time series data with datetime index
            feature_columns: Optional list of feature columns to include
            
        Returns:
            Tuple of (windowed_features, metadata)
        """
        try:
            tprint(f"🔄 Creating temporal windows: {len(data)} samples, window_size={self.window_size}, overlap={self.overlap_pct:.1%}", "INFO")
            
            # Validate input
            self._validate_input_data(data)
            
            # Select feature columns
            if feature_columns is None:
                feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Calculate number of windows
            n_windows = self._calculate_number_of_windows(len(data))
            self.n_effective_samples = n_windows
            
            tprint(f"📊 Effective sample size: {self.n_effective_samples} windows", "INFO")
            
            # Create windows
            if self.matrix_ops:
                windowed_features = self._create_windows_optimized(data[feature_columns].values, n_windows)
            else:
                windowed_features = self._create_windows_standard(data[feature_columns].values, n_windows)
            
            # Generate meta-features
            meta_features = self._generate_meta_features(data, n_windows)
            
            # Combine features and meta-features
            if meta_features is not None:
                combined_features = np.hstack([windowed_features, meta_features])
            else:
                combined_features = windowed_features
            
            # Create metadata
            metadata = {
                'n_original_samples': len(data),
                'n_windows': n_windows,
                'n_effective_samples': self.n_effective_samples,
                'window_size': self.window_size,
                'overlap_pct': self.overlap_pct,
                'step_size': self.step_size,
                'feature_columns': feature_columns,
                'meta_features_included': meta_features is not None,
                'output_shape': combined_features.shape,
                'window_metadata': self.window_metadata
            }
            
            tprint(f"✅ Windows created: {combined_features.shape[0]} windows × {combined_features.shape[1]} features", "SUCCESS")
            
            return combined_features, metadata
            
        except Exception as e:
            tprint(f"❌ Window creation failed: {e}", "ERROR")
            raise
    
    def _validate_input_data(self, data: pd.DataFrame):
        """Validate input data."""
        if data is None or len(data) == 0:
            raise ValueError("Data cannot be None or empty")
        
        if len(data) < self.window_size:
            raise ValueError(f"Insufficient data: {len(data)} < {self.window_size}")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            tprint("⚠️ Data index is not datetime, assuming sequential order", "WARNING")
    
    def _calculate_number_of_windows(self, n_samples: int) -> int:
        """Calculate number of windows that can be created."""
        if n_samples < self.window_size:
            return 0
        
        # Number of windows = (n_samples - window_size) // step_size + 1
        n_windows = (n_samples - self.window_size) // self.step_size + 1
        
        # Ensure we don't exceed available data
        n_windows = min(n_windows, n_samples - self.window_size + 1)
        
        return max(0, n_windows)
    
    def _create_windows_optimized(self, data: np.ndarray, n_windows: int) -> np.ndarray:
        """Create windows using hardware-optimized operations."""
        try:
            if self.matrix_ops and hasattr(self.matrix_ops, 'create_rolling_windows'):
                # Use optimized rolling window creation
                windowed_features = self.matrix_ops.create_rolling_windows(
                    data, 
                    window_size=self.window_size,
                    step_size=self.step_size
                )
            else:
                # Fallback to standard method
                windowed_features = self._create_windows_standard(data, n_windows)
            
            return windowed_features
            
        except Exception as e:
            tprint(f"⚠️ Optimized window creation failed: {e}, using standard method", "WARNING")
            return self._create_windows_standard(data, n_windows)
    
    def _create_windows_standard(self, data: np.ndarray, n_windows: int) -> np.ndarray:
        """Create windows using standard numpy operations."""
        try:
            # Initialize output array
            n_features = data.shape[1]
            windowed_features = np.zeros((n_windows, self.window_size * n_features))
            
            # Create windows
            for i in range(n_windows):
                start_idx = i * self.step_size
                end_idx = start_idx + self.window_size
                
                if end_idx <= len(data):
                    # Flatten window into single row
                    window = data[start_idx:end_idx].flatten()
                    windowed_features[i] = window
                    
                    # Store window metadata
                    self.window_metadata[i] = {
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'window_size': self.window_size
                    }
            
            return windowed_features
            
        except Exception as e:
            tprint(f"❌ Standard window creation failed: {e}", "ERROR")
            raise
    
    def _generate_meta_features(self, data: pd.DataFrame, n_windows: int) -> Optional[np.ndarray]:
        """Generate meta-features for temporal context."""
        try:
            meta_features = []
            
            # Time since last regime change (calculated after clustering)
            time_since_change = np.zeros((n_windows, 1))
            meta_features.append(time_since_change)
            
            # Time-based features if datetime index available
            if isinstance(data.index, pd.DatetimeIndex):
                # Hour of day (cyclical)
                hour_sin = np.sin(2 * np.pi * data.index.hour / 24)
                hour_cos = np.cos(2 * np.pi * data.index.hour / 24)
                
                # Day of week (cyclical)
                dow_sin = np.sin(2 * np.pi * data.index.dayofweek / 7)
                dow_cos = np.cos(2 * np.pi * data.index.dayofweek / 7)
                
                # Create windowed versions
                for i in range(n_windows):
                    start_idx = i * self.step_size
                    end_idx = start_idx + self.window_size
                    
                    if end_idx <= len(data):
                        # Use the last timestamp in the window
                        window_hour_sin = hour_sin[end_idx - 1]
                        window_hour_cos = hour_cos[end_idx - 1]
                        window_dow_sin = dow_sin[end_idx - 1]
                        window_dow_cos = dow_cos[end_idx - 1]
                        
                        meta_features.append(np.array([[window_hour_sin, window_hour_cos, 
                                                       window_dow_sin, window_dow_cos]]))
            
            if meta_features:
                return np.vstack(meta_features)
            else:
                return None
                
        except Exception as e:
            tprint(f"⚠️ Meta-feature generation failed: {e}", "WARNING")
            return None
    
    def update_time_since_change(self, windowed_features: np.ndarray, 
                                regime_labels: np.ndarray) -> np.ndarray:
        """
        Update time since last regime change meta-feature.
        
        Args:
            windowed_features: Windowed features array
            regime_labels: Regime labels for each window
            
        Returns:
            Updated windowed features with time since change
        """
        try:
            if len(regime_labels) != windowed_features.shape[0]:
                tprint("⚠️ Regime labels length doesn't match windowed features", "WARNING")
                return windowed_features
            
            # Calculate time since last regime change
            time_since_change = np.zeros(len(regime_labels))
            
            for i in range(1, len(regime_labels)):
                if regime_labels[i] != regime_labels[i-1]:
                    time_since_change[i] = 0
                else:
                    time_since_change[i] = time_since_change[i-1] + 1
            
            # Update the first column (time since change)
            windowed_features[:, 0] = time_since_change
            
            tprint(f"✅ Updated time since change meta-feature", "SUCCESS")
            
            return windowed_features
            
        except Exception as e:
            tprint(f"❌ Time since change update failed: {e}", "ERROR")
            return windowed_features
    
    def get_effective_sample_size(self) -> int:
        """Get effective sample size after windowing."""
        return self.n_effective_samples
    
    def get_window_info(self, window_idx: int) -> Dict[str, Any]:
        """Get information about a specific window."""
        if window_idx in self.window_metadata:
            return self.window_metadata[window_idx]
        else:
            return {'error': f'Window {window_idx} not found'}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of window handler state."""
        return {
            'window_size': self.window_size,
            'overlap_pct': self.overlap_pct,
            'step_size': self.step_size,
            'n_effective_samples': self.n_effective_samples,
            'n_windows_created': len(self.window_metadata),
            'matrix_ops_available': self.matrix_ops is not None
        }
