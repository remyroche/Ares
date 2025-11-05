"""
Covariance Stabilization for Robust Clustering

This module implements covariance matrix stabilization techniques to handle
noisy financial data and improve clustering robustness.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union
from dataclasses import dataclass

# Import sklearn for covariance estimation
try:
    from sklearn.covariance import LedoitWolf, ShrunkCovariance, EmpiricalCovariance
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import utilities
try:
    from src.utils.tprint import (
        tprint_info, tprint_success, tprint_warning, tprint_error
    )
except ImportError:
    def tprint_info(msg): print(f'ℹ️  {msg}')
    def tprint_success(msg): print(f'✅ {msg}')
    def tprint_warning(msg): print(f'⚠️  {msg}')
    def tprint_error(msg): print(f'❌ {msg}')


@dataclass
class CovarianceConfig:
    """Configuration for covariance stabilization."""
    method: str = 'ledoit_wolf'  # 'ledoit_wolf', 'exponential', 'shrunk'
    shrinkage: float = 0.1  # For shrunk covariance
    exponential_decay: float = 0.94  # For exponential weighting
    min_periods: int = 20  # Minimum periods for estimation
    
    # Regularization
    enable_regularization: bool = True
    regularization_strength: float = 1e-6


class CovarianceStabilizer:
    """
    Covariance stabilizer using multiple techniques for robust estimation.
    
    Implements Ledoit-Wolf shrinkage, exponential weighting, and other
    stabilization techniques to handle noisy financial data.
    """
    
    def __init__(self, config: Optional[CovarianceConfig] = None):
        """
        Initialize covariance stabilizer.
        
        Args:
            config: Configuration for covariance stabilization
        """
        self.config = config or CovarianceConfig()
        
        tprint_info(f"🔧 Initialized Covariance Stabilizer (method: {self.config.method})")
        
        if not SKLEARN_AVAILABLE and self.config.method in ['ledoit_wolf', 'shrunk']:
            tprint_warning("⚠️ sklearn not available, falling back to empirical covariance")
            self.config.method = 'empirical'
    
    def stabilize_covariance(self, 
                         returns: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stabilize covariance matrix using configured method.
        
        Args:
            returns: Asset returns matrix (T x N)
            
        Returns:
            Tuple of (stabilized_covariance, correlation_matrix)
        """
        tprint_info(f"🔍 Stabilizing covariance using {self.config.method} method")
        
        try:
            # Convert to numpy if needed
            if isinstance(returns, pd.DataFrame):
                returns_array = returns.values
            else:
                returns_array = returns
            
            # Remove NaN values
            valid_mask = ~np.isnan(returns_array).any(axis=1)
            clean_returns = returns_array[valid_mask]
            
            if len(clean_returns) < self.config.min_periods:
                tprint_warning(f"⚠️ Insufficient data: {len(clean_returns)} < {self.config.min_periods}")
                # Fallback to simple covariance
                cov_matrix = np.cov(clean_returns.T)
                if self.config.enable_regularization:
                    cov_matrix += np.eye(cov_matrix.shape[0]) * self.config.regularization_strength
                corr_matrix = self._covariance_to_correlation(cov_matrix)
                return cov_matrix, corr_matrix
            
            # Apply stabilization method
            if self.config.method == 'ledoit_wolf':
                cov_matrix = self._ledoit_wolf_shrinkage(clean_returns)
            elif self.config.method == 'exponential':
                cov_matrix = self._exponential_weighted_covariance(clean_returns)
            elif self.config.method == 'shrunk':
                cov_matrix = self._shrunk_covariance(clean_returns)
            else:  # empirical
                cov_matrix = np.cov(clean_returns.T)
            
            # Apply regularization
            if self.config.enable_regularization:
                cov_matrix += np.eye(cov_matrix.shape[0]) * self.config.regularization_strength
            
            # Convert to correlation
            corr_matrix = self._covariance_to_correlation(cov_matrix)
            
            tprint_success(f"✅ Covariance stabilization complete: {cov_matrix.shape}")
            return cov_matrix, corr_matrix
            
        except Exception as e:
            tprint_error(f"❌ Covariance stabilization failed: {e}")
            # Fallback to simple covariance
            cov_matrix = np.cov(returns.T)
            corr_matrix = self._covariance_to_correlation(cov_matrix)
            return cov_matrix, corr_matrix
    
    def _ledoit_wolf_shrinkage(self, returns: np.ndarray) -> np.ndarray:
        """Apply Ledoit-Wolf shrinkage to covariance matrix."""
        tprint_info("🔍 Applying Ledoit-Wolf shrinkage to covariance matrix")
        
        if not SKLEARN_AVAILABLE:
            tprint_warning("⚠️ sklearn not available, using fallback implementation")
            return self._simple_shrinkage(returns)
        
        try:
            tprint_info("📊 Fitting Ledoit-Wolf shrinkage model")
            lw = LedoitWolf().fit(returns)
            tprint_success("✅ Ledoit-Wolf shrinkage applied successfully")
            return lw.covariance_
        except Exception as e:
            tprint_warning(f"⚠️ Ledoit-Wolf failed: {e}, using fallback")
            return self._simple_shrinkage(returns)
    
    def _exponential_weighted_covariance(self, returns: np.ndarray) -> np.ndarray:
        """Calculate exponentially weighted covariance matrix."""
        tprint_info("📊 Calculating exponentially weighted covariance matrix")
        
        T, N = returns.shape
        tprint_info(f"📈 Input shape: {T} time periods × {N} assets")
        
        cov_matrix = np.zeros((N, N))
        
        # Calculate exponentially weighted mean
        tprint_info(f"🔢 Calculating exponential weights with decay factor {self.config.exponential_decay}")
        weights = np.zeros(T)
        weights[0] = 1.0
        for t in range(1, T):
            weights[t] = weights[t-1] * self.config.exponential_decay
        
        weights = weights / weights.sum()  # Normalize weights
        tprint_info(f"📊 Normalized weights: min={weights.min():.4f}, max={weights.max():.4f}")
        
        # Calculate weighted covariance
        weighted_mean = np.sum(returns * weights[:, np.newaxis], axis=0)
        
        tprint_info("🔄 Computing weighted covariance matrix")
        for i in range(N):
            for j in range(N):
                cov_matrix[i, j] = np.sum(
                    weights * (returns[:, i] - weighted_mean[i]) * (returns[:, j] - weighted_mean[j])
                )
        
        tprint_success("✅ Exponential weighted covariance calculated")
        return cov_matrix
    
    def _shrunk_covariance(self, returns: np.ndarray) -> np.ndarray:
        """Apply shrinkage to covariance matrix."""
        tprint_info(f"🔍 Applying shrinkage to covariance matrix (shrinkage={self.config.shrinkage})")
        
        if not SKLEARN_AVAILABLE:
            tprint_warning("⚠️ sklearn not available, using fallback implementation")
            return self._simple_shrinkage(returns)
        
        try:
            tprint_info("📊 Fitting shrunk covariance model")
            sc = ShrunkCovariance(shrinkage=self.config.shrinkage).fit(returns)
            tprint_success("✅ Shrunk covariance applied successfully")
            return sc.covariance_
        except Exception as e:
            tprint_warning(f"⚠️ Shrunk covariance failed: {e}, using fallback")
            return self._simple_shrinkage(returns)
    
    def _simple_shrinkage(self, returns: np.ndarray) -> np.ndarray:
        """Simple shrinkage implementation as fallback."""
        tprint_info("🔍 Applying simple shrinkage as fallback")
        
        # Calculate sample covariance
        tprint_info("📊 Calculating sample covariance")
        sample_cov = np.cov(returns.T)
        
        # Calculate target (diagonal matrix)
        tprint_info("🎯 Creating diagonal target matrix")
        target = np.diag(np.diag(sample_cov))
        
        # Apply shrinkage
        tprint_info(f"🔄 Applying shrinkage factor: {self.config.shrinkage}")
        shrunk_cov = (1 - self.config.shrinkage) * sample_cov + self.config.shrinkage * target
        
        tprint_success("✅ Simple shrinkage applied successfully")
        return shrunk_cov
    
    def _covariance_to_correlation(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix."""
        tprint_info("🔄 Converting covariance matrix to correlation matrix")
        
        # Calculate standard deviations
        tprint_info("📊 Calculating standard deviations from diagonal")
        std_devs = np.sqrt(np.diag(cov_matrix))
        
        # Avoid division by zero
        zero_count = np.sum(std_devs == 0)
        if zero_count > 0:
            tprint_warning(f"⚠️ Found {zero_count} zero standard deviations, replacing with 1")
        std_devs = np.where(std_devs == 0, 1, std_devs)
        
        # Calculate correlation matrix
        tprint_info("🔄 Computing correlation matrix")
        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
        
        # Ensure diagonal is exactly 1
        np.fill_diagonal(corr_matrix, 1.0)
        
        tprint_success("✅ Covariance to correlation conversion complete")
        return corr_matrix
    
    def calculate_distance_matrix(self,
                             correlation_matrix: np.ndarray,
                             distance_type: str = 'angular') -> np.ndarray:
        """
        Calculate distance matrix from correlation matrix.
        
        Args:
            correlation_matrix: Correlation matrix
            distance_type: Type of distance ('angular', 'euclidean', 'absolute')
            
        Returns:
            Distance matrix
        """
        tprint_info(f"📏 Calculating {distance_type} distance matrix from correlation matrix")
        
        if distance_type == 'angular':
            # Angular distance: sqrt(2 * (1 - correlation))
            tprint_info("📐 Using angular distance formula")
            distance_matrix = np.sqrt(2 * (1 - correlation_matrix))
        elif distance_type == 'euclidean':
            # Euclidean distance: sqrt(2 * (1 - correlation))
            tprint_info("📐 Using euclidean distance formula")
            distance_matrix = np.sqrt(2 * (1 - correlation_matrix))
        elif distance_type == 'absolute':
            # Absolute distance: 1 - |correlation|
            tprint_info("📐 Using absolute distance formula")
            distance_matrix = 1 - np.abs(correlation_matrix)
        else:
            tprint_error(f"❌ Unknown distance type: {distance_type}")
            raise ValueError(f"Unknown distance type: {distance_type}")
        
        tprint_success(f"✅ {distance_type} distance matrix calculated: {distance_matrix.shape}")
        return distance_matrix


def create_covariance_stabilizer(
    method: str = 'ledoit_wolf',
    shrinkage: float = 0.1,
    exponential_decay: float = 0.94,
    enable_regularization: bool = True
) -> CovarianceStabilizer:
    """
    Factory function to create covariance stabilizer.
    
    Args:
        method: Stabilization method
        shrinkage: Shrinkage intensity
        exponential_decay: Decay factor for exponential weighting
        enable_regularization: Enable additional regularization
        
    Returns:
        CovarianceStabilizer instance
    """
    tprint_info("🏭 Creating Covariance Stabilizer with factory function")
    
    config = CovarianceConfig(
        method=method,
        shrinkage=shrinkage,
        exponential_decay=exponential_decay,
        enable_regularization=enable_regularization
    )
    
    tprint_info(f"📊 Configuration: method={method}, shrinkage={shrinkage}, decay={exponential_decay}")
    stabilizer = CovarianceStabilizer(config)
    tprint_success("✅ Covariance Stabilizer created successfully")
    return stabilizer