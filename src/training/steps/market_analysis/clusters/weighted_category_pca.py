"""
Weighted Per-Category PCA for Enhanced Clustering Performance.

This module implements weighted Principal Component Analysis (PCA) separately on
feature categories to improve regime separation and clustering quality.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
from dataclasses import dataclass

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning


@dataclass
class CategoryConfig:
    """Configuration for a feature category."""
    description: str
    weight: float  # Category importance weight (0-1)
    variance_threshold: float  # Variance to retain (0-1)
    features: List[str]  # Feature names in this category


# Default feature categorization for financial time series
DEFAULT_FEATURE_CATEGORIES = {
    'returns': CategoryConfig(
        description='Return-based features (momentum, trends)',
        weight=0.40,  # Highest weight - primary regime driver
        variance_threshold=0.95,  # Retain 95% variance
        features=[
            'log_returns_1d', 'log_returns_5d', 'log_returns_20d',
            'forward_returns_1d', 'forward_returns_5d', 'forward_returns_20d',
            'momentum_10d', 'momentum_20d', 'momentum_60d',
            'return_volatility_ratio', 'sharpe_ratio_20d'
        ]
    ),
    'volatility': CategoryConfig(
        description='Volatility and risk measures',
        weight=0.30,  # Second highest - regime state indicator
        variance_threshold=0.90,  # Retain 90% variance
        features=[
            'realized_volatility_5d', 'realized_volatility_20d', 'realized_volatility_60d',
            'garch_volatility', 'parkinson_volatility', 'garman_klass_volatility',
            'atr_14', 'bollinger_width', 'volatility_regime_indicator',
            'vol_of_vol_20d'
        ]
    ),
    'volume': CategoryConfig(
        description='Volume and liquidity metrics',
        weight=0.15,  # Moderate weight - market participation
        variance_threshold=0.85,  # Retain 85% variance
        features=[
            'volume_normalized', 'turnover_ratio', 'dollar_volume',
            'bid_ask_spread', 'volume_ma_ratio_20', 'volume_trend_strength',
            'volume_volatility_20d', 'volume_price_correlation'
        ]
    ),
    'technical': CategoryConfig(
        description='Technical indicators',
        weight=0.15,  # Moderate weight - market sentiment
        variance_threshold=0.85,  # Retain 85% variance
        features=[
            'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
            'ma_cross_20_50', 'ma_cross_50_200',
            'stochastic_k', 'stochastic_d',
            'adx_14', 'cci_20'
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
            
            # Standardize features within category
            scaler = StandardScaler()
            cat_features_scaled = scaler.fit_transform(cat_features)
            self.scalers[cat_name] = scaler
            
            # Apply PCA
            variance_threshold = cat_config.variance_threshold
            
            # Determine number of components
            if variance_threshold >= 1.0:
                # Treat as number of components
                n_components = min(int(variance_threshold), cat_features_scaled.shape[1])
            else:
                # Treat as variance threshold
                n_components = min(cat_features_scaled.shape[1], cat_features_scaled.shape[0])
            
            pca = PCA(n_components=n_components, svd_solver='full')
            pca.fit(cat_features_scaled)
            
            # Adjust to meet variance threshold
            if variance_threshold < 1.0:
                cumsum_var = np.cumsum(pca.explained_variance_ratio_)
                n_components = int(np.searchsorted(cumsum_var, variance_threshold) + 1)
                n_components = min(n_components, pca.n_components_)
                
                # Refit with optimal number of components
                if n_components < pca.n_components_:
                    pca = PCA(n_components=n_components, svd_solver='full')
                    pca.fit(cat_features_scaled)
            
            self.pca_transformers[cat_name] = pca
            total_pca_components += pca.n_components_
            
            # Create component names
            for i in range(pca.n_components_):
                comp_name = f"{cat_name}_pc{i+1}"
                transformed_feature_names.append(comp_name)
            
            # Log results
            explained_var = pca.explained_variance_ratio_.sum()
            tprint(f"✅ {cat_name:12s}: {len(cat_indices):3d} features → "
                  f"{pca.n_components_:3d} components ({explained_var:6.2%} variance, "
                  f"weight={cat_config.weight:.2f})", "SUCCESS")
        
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
            features=returns_features
        )
    
    if volatility_features:
        categories['volatility'] = CategoryConfig(
            description='Volatility features (auto-detected)',
            weight=0.30,
            variance_threshold=0.90,
            features=volatility_features
        )
    
    if volume_features:
        categories['volume'] = CategoryConfig(
            description='Volume features (auto-detected)',
            weight=0.15,
            variance_threshold=0.85,
            features=volume_features
        )
    
    if technical_features:
        categories['technical'] = CategoryConfig(
            description='Technical indicators (auto-detected)',
            weight=0.15,
            variance_threshold=0.85,
            features=technical_features
        )
    
    tprint(f"\n📊 Auto-detected feature categories:", "INFO")
    tprint(f"   Returns: {len(returns_features)} features", "INFO")
    tprint(f"   Volatility: {len(volatility_features)} features", "INFO")
    tprint(f"   Volume: {len(volume_features)} features", "INFO")
    tprint(f"   Technical: {len(technical_features)} features", "INFO")
    
    return categories
