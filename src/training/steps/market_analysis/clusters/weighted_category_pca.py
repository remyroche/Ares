"""
Weighted Per-Category PCA for Enhanced Clustering Performance.

This module implements weighted Principal Component Analysis (PCA) separately on
feature categories to improve regime separation and clustering quality.

🎯 ENHANCEMENT SUGGESTION: Advanced PCA Improvements

Based on the iterative optimization review, here are suggested enhancements for the PCA implementation:

1. DYNAMIC WEIGHT ADAPTATION:
   - Implement adaptive weighting based on clustering performance
   - Use CV ratio and silhouette scores to automatically adjust category weights
   - Example: If returns features show poor regime separation, reduce their weight

2. HIERARCHICAL PCA APPROACH:
   - Apply PCA within each category first (as currently done)
   - Then apply a second-level PCA on the concatenated category PCs
   - This captures both within-category and cross-category relationships

3. TEMPORAL-AWARE PCA:
   - Use rolling windows for PCA to capture changing feature relationships
   - Apply different PCA models for different market regimes
   - This addresses temporal non-stationarity in feature correlations

4. FEATURE INTERACTION PCA:
   - Create interaction features between categories before PCA
   - Example: returns × volatility features to capture regime-specific risk
   - This can improve regime separation in complex market conditions

5. ROBUST PCA VARIANTS:
   - Implement robust PCA methods that handle outliers better
   - Use methods like RPCA (Robust PCA) for noisy financial data
   - This improves clustering stability in volatile market conditions

6. ADAPTIVE VARIANCE THRESHOLDS:
   - Dynamically adjust variance retention based on clustering performance
   - Use higher variance retention for categories that contribute more to CV ratio
   - This optimizes the dimensionality reduction for clustering objectives

7. CATEGORY-SPECIFIC NORMALIZATION:
   - Apply different normalization strategies per category
   - Example: RobustScaler for volatility features, StandardScaler for returns
   - This ensures each category contributes equally despite different scales

8. ONLINE/INCREMENTAL PCA:
   - Implement incremental PCA for large datasets and real-time updates
   - This allows continuous model adaptation as new data arrives
   - Critical for live trading applications

Implementation Priority:
1. Dynamic weight adaptation (highest impact, easiest to implement)
2. Hierarchical PCA (medium impact, moderate complexity)
3. Temporal-aware PCA (high impact, higher complexity)
4. Adaptive variance thresholds (medium impact, easy to implement)

These enhancements should significantly improve CV ratio, temporal smoothness,
and silhouette scores in the clustering optimization process.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
import pickle
from dataclasses import dataclass
from enum import Enum

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning


def robust_pca_rpca(X: np.ndarray, lambda_val: float = 0.1, max_iter: int = 100, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust PCA using Principal Component Pursuit (PCP) algorithm.

    Decomposes X into low-rank matrix L and sparse matrix S:
    X = L + S

    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, n_features)
        Input data matrix
    lambda_val : float
        Regularization parameter for sparse component
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance

    Returns:
    --------
    L : np.ndarray
        Low-rank component
    S : np.ndarray
        Sparse component (outliers)
    """
    try:
        from scipy.optimize import minimize_scalar
    except ImportError:
        # Fallback to simple SVD-based approximation if scipy not available
        tprint("⚠️  Scipy not available, using SVD-based robust PCA approximation", "WARNING")
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        # Keep top components for low-rank approximation
        rank = min(5, min(X.shape) // 2)
        L = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
        S = X - L
        return L, S

    m, n = X.shape
    mu = np.prod(X.shape) / (4 * np.linalg.norm(X, ord='nuc'))  # Initial step size

    # Initialize L and S
    L = np.zeros_like(X)
    S = np.zeros_like(X)
    Y = np.zeros_like(X)  # Lagrange multiplier

    for iteration in range(max_iter):
        # Update L (low-rank component)
        L_old = L.copy()

        # Solve: L = argmin_L ||L||_* + (mu/2)||L - (X - S - Y/mu)||_F^2
        M = X - S + Y / mu
        U, s, Vt = np.linalg.svd(M, full_matrices=False)

        # Soft thresholding for singular values
        s_thresholded = np.maximum(s - 1/mu, 0)
        rank = np.sum(s_thresholded > 0)
        if rank > 0:
            L = U[:, :rank] @ np.diag(s_thresholded[:rank]) @ Vt[:rank, :]
        else:
            L = np.zeros_like(X)

        # Update S (sparse component)
        S_old = S.copy()

        # Solve: S = argmin_S lambda||S||_1 + (mu/2)||S - (X - L - Y/mu)||_F^2
        residual = X - L + Y / mu
        S = np.sign(residual) * np.maximum(np.abs(residual) - lambda_val / mu, 0)

        # Update Lagrange multiplier
        Y += mu * (X - L - S)

        # Check convergence
        error = np.linalg.norm(X - L - S, 'fro') / np.linalg.norm(X, 'fro')
        if error < tol:
            break

        # Update step size
        if np.linalg.norm(L - L_old, 'fro') > 10 * np.linalg.norm(S - S_old, 'fro'):
            mu *= 1.1
        elif np.linalg.norm(S - S_old, 'fro') > 10 * np.linalg.norm(L - L_old, 'fro'):
            mu *= 0.9

    return L, S


class NormalizationType(Enum):
    """Different normalization strategies for feature categories."""
    STANDARD = "standard"      # StandardScaler - mean=0, std=1
    ROBUST = "robust"          # RobustScaler - median and IQR-based
    MINMAX = "minmax"          # MinMaxScaler - scale to [0,1]
    POWER = "power"            # PowerTransformer - Yeo-Johnson transformation
    NONE = "none"              # No normalization


class RobustPCAType(Enum):
    """Different robust PCA variants for outlier handling."""
    STANDARD = "standard"      # Standard PCA
    RPCA = "rpca"              # Robust PCA (low-rank + sparse decomposition)
    HUBER = "huber"            # Huberized PCA (robust to outliers)


@dataclass
class CategoryConfig:
    """Configuration for a feature category."""
    description: str
    weight: float  # Category importance weight (0-1)
    variance_threshold: float  # Variance to retain (0-1)
    features: List[str]  # Feature names in this category
    normalization_type: NormalizationType = NormalizationType.STANDARD  # Type of normalization to apply
    robust_pca_type: RobustPCAType = RobustPCAType.STANDARD  # Type of PCA to use for robustness


# Default feature categorization based on actual feature_engineer.py output
# NOTE: Momentum features (momentum_10, momentum_20, roc_10, roc_20, vwap_momentum_*)
# are EXCLUDED from returns as they measure acceleration rather than trend
DEFAULT_FEATURE_CATEGORIES = {
    'returns': CategoryConfig(
        description='Return-based features (price/volume changes)',
        weight=0.40,  # Highest weight - primary regime driver
        variance_threshold=0.95,  # Retain 95% variance
        features=[
            # Price returns (NOT momentum which measures acceleration)
            'close_return', 'close_log_return',
            # Volume returns
            'volume_return', 'volume_log_return',
            # Price patterns
            'body_size_pct', 'price_range_pct',
            'upper_shadow', 'lower_shadow'
        ]
    ),
    'volatility': CategoryConfig(
        description='Volatility and risk measures',
        weight=0.30,  # Second highest - regime state indicator
        variance_threshold=0.90,  # Retain 90% variance
        features=[
            # Realized volatility
            'volatility_20', 'volatility_5',
            # Bollinger bands (volatility proxy)
            'bb_width', 'bb_position',
            # Price range features
            'price_range', 'price_range_pct',
            # ATR (Average True Range)
            'atr'
        ]
    ),
    'volume': CategoryConfig(
        description='Volume and liquidity metrics',
        weight=0.15,  # Moderate weight - market participation
        variance_threshold=0.85,  # Retain 85% variance
        features=[
            # Raw volume
            'volume', 'volume_sma_20', 'volume_ratio',
            # Volume-based indicators
            'obv',  # On-Balance Volume
            'cmf',  # Chaikin Money Flow
            'pvt',  # Price Volume Trend
            # VWAP features
            'vwap', 'vwap_price_ratio'
        ]
    ),
    'technical': CategoryConfig(
        description='Technical indicators (momentum in separate category)',
        weight=0.10,  # Reduced weight - supplementary indicators
        variance_threshold=0.85,  # Retain 85% variance
        features=[
            # Oscillators
            'rsi_14', 'stoch_k', 'stoch_d', 'williams_r', 'cci',
            # Trend indicators  
            'macd', 'macd_signal', 'macd_histogram', 'adx',
            # Moving averages
            'close_sma_5', 'close_sma_20', 'close_ema_12', 'close_ema_26',
            # Bollinger bands levels
            'bb_upper', 'bb_middle', 'bb_lower'
        ]
    ),
    'momentum': CategoryConfig(
        description='Momentum/acceleration indicators (separated from returns)',
        weight=0.05,  # Low weight - acceleration is noisy
        variance_threshold=0.80,  # Retain 80% variance
        features=[
            # Price momentum (measures acceleration, not trend)
            'momentum_10', 'momentum_20',
            # Rate of change
            'roc_10', 'roc_20',
            # VWAP momentum
            'vwap_momentum_5', 'vwap_momentum_10', 'vwap_momentum_20'
        ]
    )
}


class WeightedCategoryPCA:
    """
    Apply weighted PCA separately to feature categories for enhanced clustering.
    
    This approach:
    1. Divides features into meaningful categories (returns, volatility, etc.)
    2. Applies PCA within each category to extract principal components
    3. Weights components by category importance
    4. Combines weighted components into final feature vector
    
    Benefits:
    - Better regime separation through focused dimensionality reduction
    - Noise reduction via PCA filtering
    - Interpretable components within categories
    - Computational efficiency (fewer dimensions)
    """
    
    def __init__(self, categories_config: Optional[Dict[str, CategoryConfig]] = None):
        """
        Initialize weighted category PCA transformer.
        
        Parameters:
        -----------
        categories_config : dict, optional
            Dictionary mapping category names to CategoryConfig objects.
            If None, uses DEFAULT_FEATURE_CATEGORIES.
        """
        self.categories_config = categories_config or DEFAULT_FEATURE_CATEGORIES
        self.pca_transformers: Dict[str, PCA] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_indices: Dict[str, List[int]] = {}
        self.original_feature_names: Optional[List[str]] = None
        self.transformed_feature_names: Optional[List[str]] = None
        self.is_fitted = False
        
    def fit(self, features: np.ndarray, feature_names: List[str]) -> 'WeightedCategoryPCA':
        """
        Fit PCA transformers for each category.
        
        Parameters:
        -----------
        features : np.ndarray, shape (n_samples, n_features)
            Input feature matrix
        feature_names : list of str
            List of feature names corresponding to columns
            
        Returns:
        --------
        self : WeightedCategoryPCA
            Fitted transformer
        """
        if features is None or features.size == 0:
            raise ValueError("Features array is None or empty")
        
        if len(feature_names) != features.shape[1]:
            raise ValueError(f"Feature names length ({len(feature_names)}) doesn't match "
                           f"features shape[1] ({features.shape[1]})")
        
        self.original_feature_names = feature_names
        
        # Map feature names to indices (case-insensitive partial matching)
        name_to_idx = {name.lower(): idx for idx, name in enumerate(feature_names)}
        
        tprint("\n🔧 Fitting Weighted Per-Category PCA...", "INFO")
        tprint("="*80, "INFO")
        
        transformed_feature_names = []
        total_original_features = 0
        total_pca_components = 0
        
        for cat_name, cat_config in self.categories_config.items():
            # Find feature indices for this category (fuzzy matching)
            cat_feature_names = cat_config.features
            cat_indices = []
            
            for cat_feat in cat_feature_names:
                cat_feat_lower = cat_feat.lower()
                # Try exact match first
                if cat_feat_lower in name_to_idx:
                    cat_indices.append(name_to_idx[cat_feat_lower])
                else:
                    # Try partial match (feature name contains category feature)
                    for feat_name, idx in name_to_idx.items():
                        if cat_feat_lower in feat_name or feat_name in cat_feat_lower:
                            if idx not in cat_indices:  # Avoid duplicates
                                cat_indices.append(idx)
                                break
            
            if not cat_indices:
                tprint(f"⚠️  Warning: No features found for category '{cat_name}', skipping", "WARNING")
                continue
            
            self.feature_indices[cat_name] = cat_indices
            total_original_features += len(cat_indices)
            
            # Extract category features
            cat_features = features[:, cat_indices]
            
            # Check for constant features
            feature_stds = np.std(cat_features, axis=0)
            constant_mask = feature_stds > 1e-10
            
            if not np.any(constant_mask):
                tprint(f"⚠️  Warning: All features in category '{cat_name}' are constant, skipping", "WARNING")
                continue
            
            # Remove constant features
            if not np.all(constant_mask):
                cat_features = cat_features[:, constant_mask]
                cat_indices = [cat_indices[i] for i in range(len(cat_indices)) if constant_mask[i]]
                self.feature_indices[cat_name] = cat_indices
            
            # Apply category-specific normalization
            cat_config = self.categories_config[cat_name]
            scaler = self._get_scaler_for_category(cat_config.normalization_type)
            cat_features_scaled = scaler.fit_transform(cat_features)
            self.scalers[cat_name] = scaler
            
            # Determine number of components
            if variance_threshold >= 1.0:
                # Treat as number of components
                n_components = min(int(variance_threshold), cat_features_scaled.shape[1])
            else:
                # Treat as variance threshold
                n_components = min(cat_features_scaled.shape[1], cat_features_scaled.shape[0])

            # Apply robust PCA variant
            pca = self._get_pca_for_category(cat_config.robust_pca_type, n_components)
            pca.fit(cat_features_scaled)

            # Adjust to meet variance threshold for standard PCA
            if (cat_config.robust_pca_type == RobustPCAType.STANDARD and
                variance_threshold < 1.0 and hasattr(pca, 'explained_variance_ratio_')):
                cumsum_var = np.cumsum(pca.explained_variance_ratio_)
                n_components = int(np.searchsorted(cumsum_var, variance_threshold) + 1)
                n_components = min(n_components, pca.n_components_)

                # Refit with optimal number of components
                if n_components < pca.n_components_:
                    pca = self._get_pca_for_category(cat_config.robust_pca_type, n_components)
                    pca.fit(cat_features_scaled)
            
            self.pca_transformers[cat_name] = pca
            total_pca_components += pca.n_components_
            
            # Create component names
            for i in range(pca.n_components_):
                comp_name = f"{cat_name}_pc{i+1}"
                transformed_feature_names.append(comp_name)
            
            # Log results
            explained_var = pca.explained_variance_ratio_.sum() if hasattr(pca, 'explained_variance_ratio_') else 0.0
            norm_type = cat_config.normalization_type.value
            pca_type = cat_config.robust_pca_type.value
            tprint(f"✅ {cat_name:12s}: {len(cat_indices):3d} features → "
                  f"{pca.n_components_:3d} components ({explained_var:6.2%} variance, "
                  f"norm={norm_type:8s}, pca={pca_type:5s}, weight={cat_config.weight:.2f})", "SUCCESS")
        
        self.transformed_feature_names = transformed_feature_names
        self.is_fitted = True
        
        # Summary
        tprint("="*80, "INFO")
        tprint(f"📊 PCA Summary:", "SUCCESS")
        tprint(f"   Original dimensions: {features.shape[1]}", "INFO")
        tprint(f"   Features used: {total_original_features}", "INFO")
        tprint(f"   Transformed dimensions: {total_pca_components}", "INFO")
        reduction = (1 - total_pca_components / features.shape[1]) * 100
        tprint(f"   Dimensionality reduction: {reduction:.1f}%", "SUCCESS")
        tprint("="*80, "INFO")
        
        return self

    def _get_scaler_for_category(self, normalization_type: NormalizationType):
        """Get appropriate scaler for a category based on normalization type."""
        if normalization_type == NormalizationType.STANDARD:
            return StandardScaler()
        elif normalization_type == NormalizationType.ROBUST:
            return RobustScaler()
        elif normalization_type == NormalizationType.MINMAX:
            return MinMaxScaler()
        elif normalization_type == NormalizationType.POWER:
            return PowerTransformer(method='yeo-johnson')
        elif normalization_type == NormalizationType.NONE:
            # Return identity scaler (no transformation)
            class IdentityScaler:
                def fit_transform(self, X):
                    return X
                def transform(self, X):
                    return X
                def fit(self, X):
                    return self
            return IdentityScaler()
        else:
            tprint(f"⚠️  Unknown normalization type {normalization_type}, using StandardScaler", "WARNING")
            return StandardScaler()

    def _get_pca_for_category(self, robust_pca_type: RobustPCAType, n_components: int):
        """Get appropriate PCA variant for a category."""
        if robust_pca_type == RobustPCAType.STANDARD:
            return PCA(n_components=n_components, svd_solver='full')
        elif robust_pca_type == RobustPCAType.RPCA:
            # For RPCA, we'll use our custom implementation
            # We'll apply it as a preprocessing step before standard PCA
            class RobustPCAWrapper:
                def __init__(self, n_components, lambda_val=0.1):
                    self.n_components = n_components
                    self.lambda_val = lambda_val
                    self.pca = PCA(n_components=n_components, svd_solver='full')

                def fit(self, X):
                    # Apply RPCA first
                    L, S = robust_pca_rpca(X, lambda_val=self.lambda_val)
                    # Then apply standard PCA to the low-rank component
                    self.pca.fit(L)
                    return self

                def transform(self, X):
                    # Apply RPCA transform
                    L, S = robust_pca_rpca(X, lambda_val=self.lambda_val)
                    # Transform using standard PCA
                    return self.pca.transform(L)

                def fit_transform(self, X):
                    L, S = robust_pca_rpca(X, lambda_val=self.lambda_val)
                    return self.pca.fit_transform(L)

            return RobustPCAWrapper(n_components)
        elif robust_pca_type == RobustPCAType.HUBER:
            # For Huber PCA, we'll use a simple approximation with robust preprocessing
            class HuberPCAWrapper:
                def __init__(self, n_components):
                    self.n_components = n_components
                    self.pca = PCA(n_components=n_components, svd_solver='full')

                def fit(self, X):
                    # Apply Huber-like robust preprocessing (clip extreme values)
                    # This is a simplified version of Huber PCA
                    median = np.median(X, axis=0)
                    mad = np.median(np.abs(X - median), axis=0)
                    # Huber-like clipping: values beyond 1.5*MAD from median
                    lower_bound = median - 1.5 * mad
                    upper_bound = median + 1.5 * mad
                    X_robust = np.clip(X, lower_bound, upper_bound)

                    self.pca.fit(X_robust)
                    return self

                def transform(self, X):
                    # Apply same preprocessing to new data
                    median = np.median(X, axis=0)
                    mad = np.median(np.abs(X - median), axis=0)
                    lower_bound = median - 1.5 * mad
                    upper_bound = median + 1.5 * mad
                    X_robust = np.clip(X, lower_bound, upper_bound)

                    return self.pca.transform(X_robust)

            return HuberPCAWrapper(n_components)
        else:
            tprint(f"⚠️  Unknown robust PCA type {robust_pca_type}, using standard PCA", "WARNING")
            return PCA(n_components=n_components, svd_solver='full')

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted PCA transformers with category weights.
        
        Parameters:
        -----------
        features : np.ndarray, shape (n_samples, n_features)
            Input feature matrix
            
        Returns:
        --------
        transformed_features : np.ndarray, shape (n_samples, total_pca_components)
            Weighted PCA-transformed features
        """
        if not self.is_fitted:
            raise ValueError("WeightedCategoryPCA must be fitted before transform")
        
        if features.shape[1] != len(self.original_feature_names):
            raise ValueError(f"Feature dimension mismatch: expected {len(self.original_feature_names)}, "
                           f"got {features.shape[1]}")
        
        transformed_parts = []
        
        for cat_name, cat_config in self.categories_config.items():
            if cat_name not in self.feature_indices:
                continue
            
            # Extract category features
            cat_indices = self.feature_indices[cat_name]
            cat_features = features[:, cat_indices]
            
            # Standardize
            cat_features_scaled = self.scalers[cat_name].transform(cat_features)
            
            # Apply PCA
            cat_pca = self.pca_transformers[cat_name].transform(cat_features_scaled)
            
            # Apply category weight (use sqrt for variance weighting)
            category_weight = cat_config.weight
            weighted_pca = cat_pca * np.sqrt(category_weight)
            
            transformed_parts.append(weighted_pca)
        
        # Concatenate all weighted PCA components
        final_features = np.hstack(transformed_parts)
        
        # L2 normalization for unit scale (per sample)
        norms = np.linalg.norm(final_features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        final_features = final_features / norms
        
        return final_features
    
    def fit_transform(self, features: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(features, feature_names)
        return self.transform(features)
    
    def get_component_summary(self) -> Dict[str, Dict]:
        """
        Get summary of PCA components per category.
        
        Returns:
        --------
        summary : dict
            Dictionary with component information per category
        """
        if not self.is_fitted:
            raise ValueError("WeightedCategoryPCA must be fitted before getting summary")
        
        summary = {}
        for cat_name, pca in self.pca_transformers.items():
            cat_config = self.categories_config[cat_name]
            summary[cat_name] = {
                'n_original_features': len(self.feature_indices[cat_name]),
                'n_components': pca.n_components_,
                'category_weight': cat_config.weight,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': pca.explained_variance_ratio_.cumsum().tolist(),
                'total_variance_explained': float(pca.explained_variance_ratio_.sum())
            }
        return summary
    
    def get_feature_names_out(self) -> List[str]:
        """Get transformed feature names."""
        if not self.is_fitted:
            raise ValueError("WeightedCategoryPCA must be fitted first")
        return self.transformed_feature_names
    
    def save(self, filepath: str):
        """Save fitted transformer to disk."""
        if not self.is_fitted:
            raise ValueError("WeightedCategoryPCA must be fitted before saving")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        tprint(f"✅ Saved WeightedCategoryPCA to {filepath}", "SUCCESS")
    
    @staticmethod
    def load(filepath: str) -> 'WeightedCategoryPCA':
        """Load fitted transformer from disk."""
        with open(filepath, 'rb') as f:
            transformer = pickle.load(f)
        
        if not transformer.is_fitted:
            raise ValueError("Loaded transformer is not fitted")
        
        tprint(f"✅ Loaded WeightedCategoryPCA from {filepath}", "SUCCESS")
        return transformer


def create_feature_categories_from_names(feature_names: List[str]) -> Dict[str, CategoryConfig]:
    """
    Automatically create feature categories based on feature names.
    
    This is a fallback when specific feature lists are not available.
    Uses keyword matching to categorize features.
    
    Parameters:
    -----------
    feature_names : list of str
        List of all feature names
        
    Returns:
    --------
    categories : dict
        Dictionary mapping category names to CategoryConfig objects
    """
    # Keywords for each category
    returns_keywords = ['return', 'momentum', 'trend', 'sharpe', 'sortino']
    volatility_keywords = ['volatility', 'vol', 'variance', 'std', 'atr', 'bollinger', 'garch']
    volume_keywords = ['volume', 'turnover', 'liquidity', 'dollar_volume', 'obv', 'vwap']
    technical_keywords = ['rsi', 'macd', 'stochastic', 'adx', 'cci', 'ma_', 'sma', 'ema']
    
    # Categorize features
    returns_features = []
    volatility_features = []
    volume_features = []
    technical_features = []
    uncategorized_features = []
    
    for feat_name in feature_names:
        feat_lower = feat_name.lower()
        
        if any(kw in feat_lower for kw in returns_keywords):
            returns_features.append(feat_name)
        elif any(kw in feat_lower for kw in volatility_keywords):
            volatility_features.append(feat_name)
        elif any(kw in feat_lower for kw in volume_keywords):
            volume_features.append(feat_name)
        elif any(kw in feat_lower for kw in technical_keywords):
            technical_features.append(feat_name)
        else:
            uncategorized_features.append(feat_name)
    
    # Distribute uncategorized features (split evenly)
    if uncategorized_features:
        n_per_category = len(uncategorized_features) // 4
        returns_features.extend(uncategorized_features[:n_per_category])
        volatility_features.extend(uncategorized_features[n_per_category:2*n_per_category])
        volume_features.extend(uncategorized_features[2*n_per_category:3*n_per_category])
        technical_features.extend(uncategorized_features[3*n_per_category:])
    
    # Create category configs
    categories = {}
    
    if returns_features:
        categories['returns'] = CategoryConfig(
            description='Return-based features (auto-detected)',
            weight=0.40,
            variance_threshold=0.95,
            features=returns_features,
            normalization_type=NormalizationType.STANDARD,  # Returns are usually well-behaved
            robust_pca_type=RobustPCAType.STANDARD  # Standard PCA for clean return data
        )
    
    if volatility_features:
        categories['volatility'] = CategoryConfig(
            description='Volatility features (auto-detected)',
            weight=0.30,
            variance_threshold=0.90,
            features=volatility_features,
            normalization_type=NormalizationType.ROBUST,  # Volatility can have outliers
            robust_pca_type=RobustPCAType.RPCA  # Robust PCA for noisy volatility data
        )
    
    if volume_features:
        categories['volume'] = CategoryConfig(
            description='Volume features (auto-detected)',
            weight=0.15,
            variance_threshold=0.85,
            features=volume_features,
            normalization_type=NormalizationType.ROBUST,  # Volume can have extreme values
            robust_pca_type=RobustPCAType.RPCA  # Robust PCA for extreme volume data
        )
    
    if technical_features:
        categories['technical'] = CategoryConfig(
            description='Technical indicators (auto-detected)',
            weight=0.15,
            variance_threshold=0.85,
            features=technical_features,
            normalization_type=NormalizationType.POWER,  # Technical indicators often need power transformation
            robust_pca_type=RobustPCAType.STANDARD  # Standard PCA works well for most technical indicators
        )
    
    tprint(f"\n📊 Auto-detected feature categories:", "INFO")
    tprint(f"   Returns: {len(returns_features)} features", "INFO")
    tprint(f"   Volatility: {len(volatility_features)} features", "INFO")
    tprint(f"   Volume: {len(volume_features)} features", "INFO")
    tprint(f"   Technical: {len(technical_features)} features", "INFO")
    
    return categories
