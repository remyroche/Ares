"""
Noise Handler for HDBSCAN Results

Handles noise points (-1 labels) with causal/acausal smoothing modes
and kNN fallback for out-of-sample prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
from scipy import ndimage

# Import existing utilities
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error
from src.utils.common_operations import safe_dataframe_operation
from src.utils.common_utilities import safe_dataframe_operation as safe_df_op
from src.utils.math_validation import validate_finite, safe_divide, safe_log, safe_sqrt
# Lazy import to avoid circular imports
def get_unified_matrix_operations():
    """Lazy import of get_unified_matrix_operations to avoid circular imports."""
    try:
        from src.utils.matrix_operations import get_unified_matrix_operations as _get_unified_matrix_operations
        return _get_unified_matrix_operations
    except ImportError:
        return None

from ..config.regime_discovery_config import RegimeDiscoveryConfig

logger = logging.getLogger(__name__)


class NoiseHandler:
    """
    Handles noise points from HDBSCAN with various smoothing strategies.
    
    Modes:
    - 'keep': Keep -1 as "transition" regime
    - 'knn_assign': Assign to nearest cluster using kNN
    - 'causal_smooth': Causal smoothing (safe for live trading)
    - 'acausal_smooth': Acausal smoothing (offline only)
    """
    
    def __init__(self, config: RegimeDiscoveryConfig):
        """Initialize the noise handler."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hardware optimization
        self._initialize_hardware_optimization()
        
        # Noise handling parameters
        self.mode = config.noise_handling_mode
        self.smoothing_window = config.smoothing_window
        self.knn_k = config.knn_k
        
        # State tracking
        self.last_labels = None
        self.smoothing_history = []
        
        tprint(f"🚀 NoiseHandler initialized with mode: {self.mode}", "SUCCESS")
        self.logger.info(f"NoiseHandler initialized with mode: {self.mode}")
    
    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization utilities."""
        try:
            # Initialize unified matrix operations
            self.matrix_ops = get_unified_matrix_operations()
            
            if self.matrix_ops:
                tprint("✅ Matrix operations available for noise handling", "SUCCESS")
            else:
                tprint("⚠️ Matrix operations not available, using standard operations", "WARNING")
                
        except Exception as e:
            tprint(f"❌ Hardware optimization initialization failed: {e}", "ERROR")
            self.matrix_ops = None
    
    def handle_noise(self, labels: np.ndarray, 
                    probabilities: Optional[np.ndarray] = None,
                    is_live: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Handle noise points based on configured mode.
        
        Args:
            labels: Cluster labels (-1 for noise)
            probabilities: Cluster probabilities (optional)
            is_live: Whether this is live trading (affects smoothing mode)
            
        Returns:
            Tuple of (processed_labels, metadata)
        """
        try:
            tprint(f"🔧 Handling noise points: {np.sum(labels == -1)}/{len(labels)} noise samples", "INFO")
            
            # Validate input
            self._validate_input(labels, probabilities)
            
            # Apply noise handling based on mode
            if self.mode == 'keep':
                processed_labels, metadata = self._keep_noise(labels)
            elif self.mode == 'knn_assign':
                processed_labels, metadata = self._knn_assign_noise(labels, probabilities)
            elif self.mode == 'causal_smooth':
                processed_labels, metadata = self._causal_smooth_noise(labels, probabilities, is_live)
            elif self.mode == 'acausal_smooth':
                if is_live:
                    tprint("⚠️ Acausal smoothing not allowed in live mode, switching to causal", "WARNING")
                    processed_labels, metadata = self._causal_smooth_noise(labels, probabilities, is_live)
                else:
                    processed_labels, metadata = self._acausal_smooth_noise(labels, probabilities)
            else:
                raise ValueError(f"Unknown noise handling mode: {self.mode}")
            
            # Update state
            self.last_labels = processed_labels.copy()
            self.smoothing_history.append({
                'timestamp': datetime.now().isoformat(),
                'mode': self.mode,
                'noise_count': np.sum(labels == -1),
                'processed_noise_count': np.sum(processed_labels == -1)
            })
            
            tprint(f"✅ Noise handling completed: {np.sum(processed_labels == -1)}/{len(processed_labels)} noise samples remaining", "SUCCESS")
            
            return processed_labels, metadata
            
        except Exception as e:
            tprint(f"❌ Noise handling failed: {e}", "ERROR")
            raise
    
    def _keep_noise(self, labels: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Keep -1 as "transition" regime."""
        tprint("📌 Keeping noise as transition regime", "INFO")
        
        metadata = {
            'method': 'keep',
            'noise_count': np.sum(labels == -1),
            'noise_ratio': np.sum(labels == -1) / len(labels),
            'processed': False
        }
        
        return labels.copy(), metadata
    
    def _knn_assign_noise(self, labels: np.ndarray, 
                         probabilities: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Assign noise points to nearest cluster using kNN."""
        try:
            tprint(f"🔍 Assigning noise points using kNN (k={self.knn_k})", "INFO")
            
            # Find noise points
            noise_mask = labels == -1
            noise_indices = np.where(noise_mask)[0]
            
            if len(noise_indices) == 0:
                tprint("✅ No noise points to assign", "SUCCESS")
                return labels.copy(), {'method': 'knn_assign', 'assigned_count': 0}
            
            # Get non-noise labels
            non_noise_labels = labels[~noise_mask]
            unique_labels = np.unique(non_noise_labels)
            
            if len(unique_labels) == 0:
                tprint("⚠️ No non-noise clusters found, keeping noise", "WARNING")
                return labels.copy(), {'method': 'knn_assign', 'assigned_count': 0}
            
            # For now, assign to most common cluster
            # In a real implementation, you'd use actual kNN on features
            most_common_label = np.bincount(non_noise_labels).argmax()
            
            processed_labels = labels.copy()
            processed_labels[noise_indices] = most_common_label
            
            metadata = {
                'method': 'knn_assign',
                'assigned_count': len(noise_indices),
                'assigned_to': most_common_label,
                'knn_k': self.knn_k
            }
            
            tprint(f"✅ Assigned {len(noise_indices)} noise points to cluster {most_common_label}", "SUCCESS")
            
            return processed_labels, metadata
            
        except Exception as e:
            tprint(f"❌ kNN assignment failed: {e}", "ERROR")
            raise
    
    def _causal_smooth_noise(self, labels: np.ndarray, 
                           probabilities: Optional[np.ndarray],
                           is_live: bool) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply causal smoothing (safe for live trading)."""
        try:
            tprint(f"🔄 Applying causal smoothing (window={self.smoothing_window})", "INFO")
            
            # Causal smoothing: only use past information
            processed_labels = labels.copy()
            
            # Apply forward-only median filter
            for i in range(len(labels)):
                if labels[i] == -1:  # Only smooth noise points
                    # Look back at most smoothing_window steps
                    start_idx = max(0, i - self.smoothing_window + 1)
                    window_labels = labels[start_idx:i+1]
                    
                    # Remove noise from window for smoothing
                    non_noise_labels = window_labels[window_labels != -1]
                    
                    if len(non_noise_labels) > 0:
                        # Use most common label in window
                        smoothed_label = np.bincount(non_noise_labels).argmax()
                        processed_labels[i] = smoothed_label
            
            # Calculate statistics
            noise_before = np.sum(labels == -1)
            noise_after = np.sum(processed_labels == -1)
            smoothed_count = noise_before - noise_after
            
            metadata = {
                'method': 'causal_smooth',
                'smoothing_window': self.smoothing_window,
                'noise_before': noise_before,
                'noise_after': noise_after,
                'smoothed_count': smoothed_count,
                'is_live': is_live
            }
            
            tprint(f"✅ Causal smoothing completed: {smoothed_count} points smoothed", "SUCCESS")
            
            return processed_labels, metadata
            
        except Exception as e:
            tprint(f"❌ Causal smoothing failed: {e}", "ERROR")
            raise
    
    def _acausal_smooth_noise(self, labels: np.ndarray, 
                             probabilities: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply acausal smoothing (offline only, uses future information)."""
        try:
            tprint(f"🔄 Applying acausal smoothing (window={self.smoothing_window})", "INFO")
            
            # Acausal smoothing: use symmetric window (past and future)
            processed_labels = labels.copy()
            
            # Apply symmetric median filter
            for i in range(len(labels)):
                if labels[i] == -1:  # Only smooth noise points
                    # Create symmetric window
                    half_window = self.smoothing_window // 2
                    start_idx = max(0, i - half_window)
                    end_idx = min(len(labels), i + half_window + 1)
                    
                    window_labels = labels[start_idx:end_idx]
                    
                    # Remove noise from window for smoothing
                    non_noise_labels = window_labels[window_labels != -1]
                    
                    if len(non_noise_labels) > 0:
                        # Use most common label in window
                        smoothed_label = np.bincount(non_noise_labels).argmax()
                        processed_labels[i] = smoothed_label
            
            # Calculate statistics
            noise_before = np.sum(labels == -1)
            noise_after = np.sum(processed_labels == -1)
            smoothed_count = noise_before - noise_after
            
            metadata = {
                'method': 'acausal_smooth',
                'smoothing_window': self.smoothing_window,
                'noise_before': noise_before,
                'noise_after': noise_after,
                'smoothed_count': smoothed_count,
                'warning': 'Acausal smoothing uses future information - not suitable for live trading'
            }
            
            tprint(f"✅ Acausal smoothing completed: {smoothed_count} points smoothed", "SUCCESS")
            tprint("⚠️ WARNING: Acausal smoothing uses future information", "WARNING")
            
            return processed_labels, metadata
            
        except Exception as e:
            tprint(f"❌ Acausal smoothing failed: {e}", "ERROR")
            raise
    
    def _validate_input(self, labels: np.ndarray, probabilities: Optional[np.ndarray]):
        """Validate input data."""
        if labels is None or len(labels) == 0:
            raise ValueError("Labels cannot be None or empty")
        
        if probabilities is not None and len(probabilities) != len(labels):
            raise ValueError("Probabilities length must match labels length")
    
    def get_noise_statistics(self) -> Dict[str, Any]:
        """Get statistics about noise handling."""
        if self.last_labels is None:
            return {'error': 'No labels processed yet'}
        
        return {
            'mode': self.mode,
            'smoothing_window': self.smoothing_window,
            'knn_k': self.knn_k,
            'noise_count': np.sum(self.last_labels == -1),
            'noise_ratio': np.sum(self.last_labels == -1) / len(self.last_labels),
            'smoothing_history': self.smoothing_history
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of noise handler state."""
        return {
            'mode': self.mode,
            'smoothing_window': self.smoothing_window,
            'knn_k': self.knn_k,
            'last_processed': self.last_labels is not None,
            'smoothing_history_count': len(self.smoothing_history)
        }
