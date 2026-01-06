"""
Causal Residual Computation for Layer 2.5 Chaser

Computes the target for the Chaser: y~ = y_actual - y_causal_anchor

This ensures the Chaser only learns "unexplained alpha" and doesn't
waste capacity on simple linear relationships already captured by
the Causal Anchor.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Tuple
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

def compute_causal_residuals(
    y_actual: Union[pd.Series, np.ndarray],
    y_causal_anchor: Union[pd.Series, np.ndarray],
    min_residual_threshold: float = 1e-8,
    max_residual_threshold: float = 10.0,
    clip_residuals: bool = True,
    verbose: bool = True
) -> pd.Series:
    """
    Compute causal residuals for Chaser targeting.
    
    Formula: y~ = y_actual - y_causal_anchor
    
    Args:
        y_actual: Actual returns/targets
        y_causal_anchor: Causal Anchor predictions
        min_residual_threshold: Minimum residual threshold (prevents tiny values)
        max_residual_threshold: Maximum residual threshold (prevents outliers)
        clip_residuals: Whether to clip extreme residuals
        verbose: Whether to print statistics
        
    Returns:
        Causal residuals (unexplained alpha)
    """
    try:
        # Convert to Series if needed
        if isinstance(y_actual, np.ndarray):
            y_actual = pd.Series(y_actual)
        if isinstance(y_causal_anchor, np.ndarray):
            y_causal_anchor = pd.Series(y_causal_anchor)
        
        # Align indices
        if len(y_actual) != len(y_causal_anchor):
            if isinstance(y_actual, pd.Series) and isinstance(y_causal_anchor, pd.Series):
                y_actual, y_causal_anchor = y_actual.align(y_causal_anchor, join='inner')
            else:
                raise ValueError("Length mismatch between actual and anchor predictions")
        
        # Compute residuals
        residuals = y_actual - y_causal_anchor
        
        # Handle NaN values
        nan_mask = residuals.isna()
        if nan_mask.any():
            if verbose:
                tprint_warning(f"⚠️ Found {nan_mask.sum()} NaN residuals, setting to 0")
            residuals = residuals.fillna(0)
        
        # Optional clipping to prevent extreme values
        if clip_residuals:
            residual_std = residuals.std()
            if residual_std > 0:
                # Clip at ±3 standard deviations or max_threshold
                clip_value = min(max_residual_threshold * residual_std, 3 * residual_std)
                residuals_clipped = residuals.clip(lower=-clip_value, upper=clip_value)
                
                if verbose:
                    n_clipped = ((residuals != residuals_clipped).sum())
                    if n_clipped > 0:
                        tprint_info(f"📊 Clipped {n_clipped} extreme residuals (±{clip_value:.6f})")
                
                residuals = residuals_clipped
        
        # Filter tiny residuals (optional)
        tiny_mask = np.abs(residuals) < min_residual_threshold
        if tiny_mask.any() and verbose:
            tprint_info(f"📊 Found {tiny_mask.sum()} tiny residuals (< {min_residual_threshold})")
        
        if verbose:
            tprint_success("✅ Causal residuals computed:")
            tprint_info(f"   - Samples: {len(residuals)}")
            tprint_info(f"   - Mean: {residuals.mean():.6f}")
            tprint_info(f"   - Std: {residuals.std():.6f}")
            tprint_info(f"   - Min: {residuals.min():.6f}")
            tprint_info(f"   - Max: {residuals.max():.6f}")
            tprint_info(f"   - Positive ratio: {(residuals > 0).mean():.2%}")
            
            # Correlation analysis
            if len(y_actual) > 10:
                correlation = np.corrcoef(y_actual, y_causal_anchor)[0, 1]
                tprint_info(f"   - Actual vs Anchor correlation: {correlation:.4f}")
                
                residual_correlation = np.corrcoef(residuals, y_actual)[0, 1]
                tprint_info(f"   - Residual vs Actual correlation: {residual_correlation:.4f}")
        
        return residuals
        
    except Exception as e:
        if verbose:
            tprint_error(f"❌ Causal residual computation failed: {e}")
        raise

def analyze_residual_quality(
    y_actual: Union[pd.Series, np.ndarray],
    y_causal_anchor: Union[pd.Series, np.ndarray],
    residuals: Optional[pd.Series] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Analyze the quality of causal residuals.
    
    Args:
        y_actual: Actual returns
        y_causal_anchor: Causal Anchor predictions
        residuals: Pre-computed residuals (optional)
        verbose: Whether to print analysis
        
    Returns:
        Dictionary with quality metrics
    """
    try:
        # Compute residuals if not provided
        if residuals is None:
            residuals = compute_causal_residuals(y_actual, y_causal_anchor, verbose=False)
        
        # Basic statistics
        residual_mean = residuals.mean()
        residual_std = residuals.std()
        residual_skew = residuals.skew()
        residual_kurt = residuals.kurtosis()
        
        # Correlation analysis
        actual_anchor_corr = np.corrcoef(y_actual, y_causal_anchor)[0, 1]
        residual_actual_corr = np.corrcoef(residuals, y_actual)[0, 1]
        residual_anchor_corr = np.corrcoef(residuals, y_causal_anchor)[0, 1]
        
        # Signal quality metrics
        signal_to_noise = abs(residual_mean) / residual_std if residual_std > 0 else 0
        predictability = abs(residual_actual_corr)
        
        # Anchor effectiveness
        anchor_explained_variance = actual_anchor_corr ** 2
        residual_variance_ratio = 1 - anchor_explained_variance
        
        # Quality score (higher is better)
        # Combines: low correlation with anchor (orthogonal), 
        # high correlation with actual (predictable),
        # reasonable signal-to-noise
        orthogonality = 1 - abs(residual_anchor_corr)
        quality_score = orthogonality * predictability * min(signal_to_noise, 1.0)
        
        metrics = {
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'residual_skew': residual_skew,
            'residual_kurtosis': residual_kurt,
            'actual_anchor_correlation': actual_anchor_corr,
            'residual_actual_correlation': residual_actual_corr,
            'residual_anchor_correlation': residual_anchor_corr,
            'signal_to_noise': signal_to_noise,
            'predictability': predictability,
            'anchor_explained_variance': anchor_explained_variance,
            'residual_variance_ratio': residual_variance_ratio,
            'orthogonality': orthogonality,
            'quality_score': quality_score
        }
        
        if verbose:
            tprint_info("📊 Residual Quality Analysis:")
            tprint_info(f"   - Residual mean: {residual_mean:.6f}")
            tprint_info(f"   - Residual std: {residual_std:.6f}")
            tprint_info(f"   - Signal-to-noise: {signal_to_noise:.4f}")
            tprint_info(f"   - Predictability: {predictability:.4f}")
            tprint_info(f"   - Orthogonality: {orthogonality:.4f}")
            tprint_info(f"   - Quality score: {quality_score:.4f}")
            tprint_info(f"   - Anchor explained variance: {anchor_explained_variance:.2%}")
            tprint_info(f"   - Residual variance ratio: {residual_variance_ratio:.2%}")
        
        return metrics
        
    except Exception as e:
        if verbose:
            tprint_error(f"❌ Residual quality analysis failed: {e}")
        raise

def validate_residual_targets(
    residuals: pd.Series,
    min_samples: int = 100,
    min_variance: float = 1e-6,
    max_skewness: float = 10.0,
    max_kurtosis: float = 50.0,
    verbose: bool = True
) -> Dict[str, bool]:
    """
    Validate that residuals are suitable for Chaser training.
    
    Args:
        residuals: Causal residuals
        min_samples: Minimum number of samples
        min_variance: Minimum variance threshold
        max_skewness: Maximum acceptable skewness
        max_kurtosis: Maximum acceptable kurtosis
        verbose: Whether to print validation results
        
    Returns:
        Dictionary with validation results
    """
    try:
        validation_results = {}
        
        # Sample size check
        validation_results['sufficient_samples'] = len(residuals) >= min_samples
        
        # Variance check
        residual_variance = residuals.var()
        validation_results['sufficient_variance'] = residual_variance >= min_variance
        
        # Distribution checks
        residual_skew = residuals.skew()
        residual_kurt = residuals.kurtosis()
        validation_results['reasonable_skewness'] = abs(residual_skew) <= max_skewness
        validation_results['reasonable_kurtosis'] = abs(residual_kurt) <= max_kurtosis
        
        # Overall validity
        validation_results['valid_for_training'] = all(validation_results.values())
        
        if verbose:
            tprint_info("🔍 Residual Validation:")
            for check, passed in validation_results.items():
                status = "✅" if passed else "❌"
                tprint_info(f"   {status} {check}: {passed}")
            
            if not validation_results['valid_for_training']:
                tprint_warning("⚠️ Residuals may not be suitable for Chaser training")
            else:
                tprint_success("✅ Residuals validated for Chaser training")
        
        return validation_results
        
    except Exception as e:
        if verbose:
            tprint_error(f"❌ Residual validation failed: {e}")
        raise

# Convenience functions
def quick_residuals(y_actual, y_causal_anchor, **kwargs):
    """Quick residual computation with defaults."""
    return compute_causal_residuals(y_actual, y_causal_anchor, **kwargs)

def residual_pipeline(
    y_actual: Union[pd.Series, np.ndarray],
    y_causal_anchor: Union[pd.Series, np.ndarray],
    validate: bool = True,
    analyze: bool = True,
    **kwargs
) -> Tuple[pd.Series, Optional[Dict[str, float]], Optional[Dict[str, bool]]]:
    """
    Complete residual pipeline: compute, analyze, and validate.
    
    Args:
        y_actual: Actual returns
        y_causal_anchor: Causal Anchor predictions
        validate: Whether to validate residuals
        analyze: Whether to analyze quality
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (residuals, quality_metrics, validation_results)
    """
    # Compute residuals
    residuals = compute_causal_residuals(y_actual, y_causal_anchor, **kwargs)
    
    # Analyze quality
    quality_metrics = None
    if analyze:
        quality_metrics = analyze_residual_quality(y_actual, y_causal_anchor, residuals)
    
    # Validate
    validation_results = None
    if validate:
        validation_results = validate_residual_targets(residuals)
    
    return residuals, quality_metrics, validation_results
