"""
Variance Ratio Filter for Random Walk Detection

Implements Lo-MacKinlay variance ratio tests to identify and filter
features that follow random walks before expensive feature selection.

Usage:
- Pre-filter features before Titan RFE in Layer 2
- Pre-filter features before De Prado selection in Layer 3
- Reduces computational cost and improves feature quality
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
import warnings

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

def variance_ratio_test(returns: pd.Series, q: int = 2) -> float:
    """
    Fast Lo-MacKinlay variance ratio test.
    
    Tests H0: Random walk (VR = 1) vs H1: Not random walk (VR ≠ 1)
    
    Args:
        returns: Return series
        q: Aggregation period (default: 2)
        
    Returns:
        p-value of the test
    """
    n = len(returns)
    
    # Need sufficient data
    if n < q * 2:
        return 1.0  # Cannot reject H0 with insufficient data
    
    # Handle problematic values
    returns_clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns_clean) < n * 0.5:  # Lost too much data
        return 1.0
    
    # Calculate variances with protection against zero variance
    var_1 = np.var(returns_clean, ddof=1)
    if var_1 == 0 or np.isnan(var_1) or np.isinf(var_1):
        return 1.0  # No variance or invalid variance, cannot test
    
    # Create q-period returns
    q_returns = returns_clean.rolling(q).sum().dropna()
    if len(q_returns) < 10:
        return 1.0  # Insufficient q-period returns
    
    var_q = np.var(q_returns, ddof=1)
    if var_q == 0 or np.isnan(var_q) or np.isinf(var_q):
        return 1.0  # Invalid q-period variance
    
    # Variance ratio
    vr = var_q / (q * var_1)
    
    # Test statistic under H0 (asymptotically normal)
    delta = (q - 1) * (2 * q - 1) / (3 * q)
    if delta <= 0 or n <= 0:
        return 1.0
    
    stat = (vr - 1) / np.sqrt(delta / n)
    
    # Two-sided p-value
    p_value = 2 * (1 - norm.cdf(abs(stat)))
    
    return min(p_value, 1.0)  # Cap at 1.0

def prefilter_features_vr(
    X: pd.DataFrame, 
    p_threshold: float = 0.05, 
    min_samples: int = 100,
    q_values: List[int] = [12, 48],
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Remove random walk features using variance ratio tests.
    
    Tests H0: Random walk (VR = 1) vs H1: Not random walk (VR ≠ 1)
    Uses aggregation periods that match label horizons (12, 48 bars).
    
    Args:
        X: Feature matrix
        p_threshold: p-value threshold for rejecting random walk H0
        min_samples: Minimum samples required for testing
        q_values: List of aggregation periods to test (default: [12, 48] for label horizons)
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (filtered features, variance ratio results)
    """
    if verbose:
        tprint_info(f"🔍 Variance Ratio Pre-filter: Testing {len(X.columns)} features...")
    
    good_features = []
    vr_results = {}
    rejected_count = 0
    
    for col in X.columns:
        series = X[col].dropna()
        
        # Skip insufficient data
        if len(series) < min_samples:
            if verbose:
                tprint_warning(f"⚠️ Skipping {col}: insufficient data ({len(series)} < {min_samples})")
            vr_results[col] = 1.0  # Conservative: keep
            good_features.append(col)
            continue
        
        # Calculate returns (Assume features are already stationary/returns-like)
        # We use the series as input to the VR test directly.
        returns = series
        if len(returns) < 50:
            if verbose:
                tprint_warning(f"⚠️ Skipping {col}: insufficient data points ({len(returns)} < 50)")
            vr_results[col] = 1.0  # Conservative: keep
            good_features.append(col)
            continue
        
        # Test multiple horizons
        p_values = []
        for q in q_values:
            p_val = variance_ratio_test(returns, q)
            p_values.append(p_val)
        
        # Average p-value across horizons
        avg_p_value = np.mean(p_values)
        vr_results[col] = avg_p_value
        
        # Keep feature if it deviates from random walk
        if avg_p_value < p_threshold:
            good_features.append(col)
            if verbose and len(good_features) % 10 == 0:
                tprint_info(f"✅ Kept {len(good_features)} features so far...")
        else:
            rejected_count += 1
            if verbose and rejected_count <= 5:  # Show first few rejections
                tprint_info(f"❌ Rejected {col}: p={avg_p_value:.4f} (random walk)")
    
    # Create filtered DataFrame
    filtered_X = X[good_features].copy()
    
    if verbose:
        tprint_success(f"✅ VR Pre-filter complete:")
        tprint_info(f"   - Original features: {len(X.columns)}")
        tprint_info(f"   - Rejected (random walk): {rejected_count}")
        tprint_info(f"   - Remaining features: {len(filtered_X.columns)}")
        tprint_info(f"   - Reduction: {rejected_count/len(X.columns)*100:.1f}%")
        
        # Show some statistics
        p_values = list(vr_results.values())
        tprint_info(f"   - Mean p-value: {np.mean(p_values):.4f}")
        tprint_info(f"   - Features with p < 0.01: {sum(1 for p in p_values if p < 0.01)}")
        tprint_info(f"   - Features with p < 0.05: {sum(1 for p in p_values if p < 0.05)}")
        tprint_info(f"   - Features with p < 0.10: {sum(1 for p in p_values if p < 0.10)}")
    
    return filtered_X, vr_results

def analyze_vr_results(vr_results: Dict[str, float], save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Analyze variance ratio test results.
    
    Args:
        vr_results: Dictionary of feature -> p-value
        save_path: Optional path to save analysis
        
    Returns:
        DataFrame with analysis results
    """
    df = pd.DataFrame([
        {'feature': feature, 'p_value': p_value}
        for feature, p_value in vr_results.items()
    ]).sort_values('p_value')
    
    # Add categories
    df['category'] = df['p_value'].apply(lambda p: 
        'strong_rejection' if p < 0.01 else
        'moderate_rejection' if p < 0.05 else
        'weak_rejection' if p < 0.10 else
        'no_rejection'
    )
    
    if save_path:
        df.to_csv(save_path, index=False)
        tprint_success(f"💾 VR analysis saved to {save_path}")
    
    return df

class VarianceRatioFilter:
    """
    Configurable variance ratio filter for consistent usage across layers.
    """
    
    def __init__(
        self,
        p_threshold: float = 0.05,
        min_samples: int = 100,
        q_values: List[int] = [12, 48],
        enabled: bool = True,
        verbose: bool = True
    ):
        self.p_threshold = p_threshold
        self.min_samples = min_samples
        self.q_values = q_values
        self.enabled = enabled
        self.verbose = verbose
        self.results_ = None
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit filter and transform features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Filtered feature matrix
        """
        if not self.enabled:
            if self.verbose:
                tprint_info("⏭️ Variance Ratio Filter disabled")
            return X.copy()
        
        filtered_X, vr_results = prefilter_features_vr(
            X, 
            p_threshold=self.p_threshold,
            min_samples=self.min_samples,
            q_values=self.q_values,
            verbose=self.verbose
        )
        
        self.results_ = vr_results
        return filtered_X
    
    def get_results(self) -> Optional[Dict[str, float]]:
        """Get variance ratio test results."""
        return self.results_
    
    def analyze_results(self, save_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Analyze and optionally save results."""
        if self.results_ is None:
            tprint_warning("⚠️ No results available. Run fit_transform first.")
            return None
        
        return analyze_vr_results(self.results_, save_path)

# Convenience function for quick usage
def quick_vr_filter(X: pd.DataFrame, p_threshold: float = 0.05, q_values: List[int] = [12, 48]) -> pd.DataFrame:
    """
    Quick variance ratio filtering with default settings.
    
    Args:
        X: Feature matrix
        p_threshold: p-value threshold
        q_values: List of aggregation periods (default: [12, 48] for label horizons)
        
    Returns:
        Filtered feature matrix
    """
    filter = VarianceRatioFilter(p_threshold=p_threshold, q_values=q_values, verbose=False)
    return filter.fit_transform(X)
