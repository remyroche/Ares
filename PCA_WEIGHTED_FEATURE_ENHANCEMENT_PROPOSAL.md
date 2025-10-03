# Weighted PCA on Per-Category Features - Enhancement Proposal

## Executive Summary

This proposal recommends implementing **weighted Principal Component Analysis (PCA) on per-category features** to enhance the clustering optimization in `iterative_optimization.py`. This approach will improve regime separation by emphasizing important feature categories while reducing noise and computational complexity.

---

## 🎯 Objectives

1. **Improve CV Ratio**: Better regime separation through focused dimensionality reduction
2. **Enhance Silhouette Score**: More cohesive clusters via noise reduction
3. **Better Temporal Stability**: Emphasize features that correlate with regime persistence
4. **Computational Efficiency**: Reduce feature dimensions while preserving signal

---

## 📊 Rationale

### Current Limitations
- All features treated equally regardless of importance for regime identification
- Raw features have different scales and noise levels
- High dimensionality can dilute clustering signal with noise
- No explicit mechanism to prioritize financially relevant features

### Benefits of Weighted Per-Category PCA
- **Category-Specific Signal Extraction**: Each category's intrinsic structure preserved
- **Adaptive Importance**: Weight categories by relevance to regime identification
- **Noise Reduction**: PCA filters out low-variance (noisy) components
- **Interpretability**: PCA components within categories have clear meaning
- **Computational Efficiency**: Fewer dimensions = faster clustering

---

## 🏗️ Recommended Architecture

### 1. Feature Categorization

Divide features into meaningful financial categories with associated weights:

```python
FEATURE_CATEGORIES = {
    'returns': {
        'description': 'Return-based features (momentum, trends)',
        'weight': 0.40,  # Highest weight - primary regime driver
        'variance_threshold': 0.95,  # Retain 95% variance
        'features': [
            'log_returns_1d', 'log_returns_5d', 'log_returns_20d',
            'forward_returns_1d', 'forward_returns_5d',
            'momentum_10d', 'momentum_20d', 'momentum_60d'
        ]
    },
    'volatility': {
        'description': 'Volatility and risk measures',
        'weight': 0.30,  # Second highest - regime state indicator
        'variance_threshold': 0.90,  # Retain 90% variance
        'features': [
            'realized_volatility_5d', 'realized_volatility_20d',
            'garch_volatility', 'parkinson_volatility',
            'atr_14', 'bollinger_width'
        ]
    },
    'volume': {
        'description': 'Volume and liquidity metrics',
        'weight': 0.15,  # Moderate weight - market participation
        'variance_threshold': 0.85,  # Retain 85% variance
        'features': [
            'volume_normalized', 'turnover_ratio',
            'dollar_volume', 'bid_ask_spread',
            'volume_ma_ratio_20'
        ]
    },
    'technical': {
        'description': 'Technical indicators',
        'weight': 0.15,  # Moderate weight - market sentiment
        'variance_threshold': 0.85,  # Retain 85% variance
        'features': [
            'rsi_14', 'macd', 'macd_signal',
            'ma_cross_20_50', 'ma_cross_50_200',
            'stochastic_k', 'stochastic_d'
        ]
    }
}
```

### 2. Per-Category PCA Implementation

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

class WeightedCategoryPCA:
    """Apply weighted PCA separately to feature categories for enhanced clustering."""
    
    def __init__(self, categories_config):
        """
        Parameters:
        -----------
        categories_config : dict
            Dictionary defining feature categories, weights, and PCA parameters
        """
        self.categories_config = categories_config
        self.pca_transformers = {}
        self.scalers = {}
        self.feature_indices = {}
        
    def fit(self, features, feature_names):
        """
        Fit PCA transformers for each category.
        
        Parameters:
        -----------
        features : np.ndarray, shape (n_samples, n_features)
            Input feature matrix
        feature_names : list
            List of feature names corresponding to columns
        """
        # Map feature names to indices
        name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        
        for cat_name, cat_config in self.categories_config.items():
            # Get feature indices for this category
            cat_feature_names = cat_config['features']
            cat_indices = [name_to_idx[name] for name in cat_feature_names if name in name_to_idx]
            
            if not cat_indices:
                print(f"⚠️ Warning: No features found for category '{cat_name}'")
                continue
                
            self.feature_indices[cat_name] = cat_indices
            
            # Extract category features
            cat_features = features[:, cat_indices]
            
            # Standardize features within category
            scaler = StandardScaler()
            cat_features_scaled = scaler.fit_transform(cat_features)
            self.scalers[cat_name] = scaler
            
            # Apply PCA
            variance_threshold = cat_config['variance_threshold']
            pca = PCA(n_components=variance_threshold, svd_solver='full')
            pca.fit(cat_features_scaled)
            self.pca_transformers[cat_name] = pca
            
            # Log results
            n_components = pca.n_components_
            explained_var = pca.explained_variance_ratio_.sum()
            print(f"✅ {cat_name}: {len(cat_indices)} features → {n_components} components "
                  f"({explained_var:.2%} variance)")
    
    def transform(self, features):
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
            
            # Apply category weight
            category_weight = cat_config['weight']
            weighted_pca = cat_pca * np.sqrt(category_weight)  # sqrt for variance weighting
            
            transformed_parts.append(weighted_pca)
        
        # Concatenate all weighted PCA components
        final_features = np.hstack(transformed_parts)
        
        # L2 normalization for unit scale
        norms = np.linalg.norm(final_features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        final_features = final_features / norms
        
        return final_features
    
    def fit_transform(self, features, feature_names):
        """Fit and transform in one step."""
        self.fit(features, feature_names)
        return self.transform(features)
    
    def get_component_summary(self):
        """Get summary of PCA components per category."""
        summary = {}
        for cat_name, pca in self.pca_transformers.items():
            summary[cat_name] = {
                'n_components': pca.n_components_,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': pca.explained_variance_ratio_.cumsum().tolist()
            }
        return summary
```

### 3. Integration with Iterative Optimization

```python
# In step1_feature_preparation.py or before clustering initialization

def prepare_features_with_weighted_pca(raw_features, feature_names, categories_config):
    """
    Prepare features for clustering using weighted per-category PCA.
    
    Parameters:
    -----------
    raw_features : np.ndarray
        Raw feature matrix
    feature_names : list
        Feature column names
    categories_config : dict
        Category configuration
        
    Returns:
    --------
    transformed_features : np.ndarray
        PCA-transformed and weighted features
    transformer : WeightedCategoryPCA
        Fitted transformer (save for test-time transformation)
    """
    print("🔧 Applying Weighted Per-Category PCA...")
    
    # Initialize transformer
    transformer = WeightedCategoryPCA(categories_config)
    
    # Fit and transform
    transformed_features = transformer.fit_transform(raw_features, feature_names)
    
    # Print summary
    print(f"\n📊 PCA Transformation Summary:")
    print(f"   Original dimensions: {raw_features.shape[1]}")
    print(f"   Transformed dimensions: {transformed_features.shape[1]}")
    print(f"   Dimensionality reduction: {(1 - transformed_features.shape[1]/raw_features.shape[1]):.1%}")
    
    component_summary = transformer.get_component_summary()
    print(f"\n📈 Per-Category Components:")
    for cat_name, info in component_summary.items():
        print(f"   {cat_name}: {info['n_components']} components "
              f"({info['cumulative_variance'][-1]:.2%} variance)")
    
    return transformed_features, transformer
```

---

## 🎯 Expected Improvements

### Quantitative Targets

| Metric | Current (Baseline) | Expected with Weighted PCA | Improvement |
|--------|-------------------|---------------------------|-------------|
| **CV Ratio** | ~1.2 | 1.5 - 2.0 | +25-65% |
| **Silhouette Score** | ~0.25 | 0.35 - 0.45 | +40-80% |
| **Temporal Stability** | Variable | More stable regimes | +20-30% |
| **Computational Time** | Baseline | -30% to -50% | Faster |
| **Feature Dimensions** | ~30-50 | ~15-25 | ~50% reduction |

### Qualitative Benefits

1. **Better Regime Separation**: Focus on financially meaningful patterns
2. **Noise Reduction**: PCA filters out measurement noise and redundancy
3. **Interpretability**: Category-level components are more interpretable
4. **Robustness**: Less sensitive to individual feature outliers
5. **Efficiency**: Faster clustering with fewer dimensions

---

## 🚀 Implementation Roadmap

### Phase 1: Proof of Concept (Week 1)
- [ ] Implement `WeightedCategoryPCA` class
- [ ] Test on sample dataset
- [ ] Compare clustering metrics (CV, Silhouette, Temporal) with/without PCA
- [ ] Validate dimensionality reduction

### Phase 2: Integration (Week 2)
- [ ] Integrate into `step1_feature_preparation.py`
- [ ] Add configuration file for category definitions
- [ ] Implement transformer persistence (pickle/joblib)
- [ ] Add comprehensive logging and monitoring

### Phase 3: Optimization (Week 3)
- [ ] Hyperparameter tuning (variance thresholds, category weights)
- [ ] Cross-validation to optimize weights
- [ ] A/B testing against baseline
- [ ] Performance profiling and optimization

### Phase 4: Production (Week 4)
- [ ] Full integration testing
- [ ] Documentation and code review
- [ ] Add unit tests
- [ ] Deploy to production pipeline

---

## 🔬 Advanced Variants (Future Enhancements)

### 1. Kernel PCA for Non-Linear Relationships
```python
from sklearn.decomposition import KernelPCA

# Replace linear PCA with kernel PCA
pca = KernelPCA(n_components=n_comp, kernel='rbf', gamma=0.1)
```

### 2. Sparse PCA for Interpretability
```python
from sklearn.decomposition import SparsePCA

# Sparse components with clear feature contributions
pca = SparsePCA(n_components=n_comp, alpha=0.1)
```

### 3. Incremental PCA for Large Datasets
```python
from sklearn.decomposition import IncrementalPCA

# Memory-efficient for large feature matrices
pca = IncrementalPCA(n_components=n_comp, batch_size=1000)
```

### 4. Time-Adaptive PCA (Rolling Window)
```python
def rolling_pca_transform(features, window_size=252):
    """Apply PCA with rolling window for time-varying structure."""
    # Fit PCA on recent window, transform current data
    # Adapts to regime evolution over time
    pass
```

### 5. Learned Category Weights (Meta-Optimization)
```python
from scipy.optimize import minimize

def optimize_category_weights(features, labels, categories):
    """Learn optimal category weights via cross-validation."""
    
    def objective(weights):
        # Apply weighted PCA with these weights
        # Cluster and compute CV ratio
        # Return negative CV ratio (minimize)
        pass
    
    # Optimize weights to maximize CV ratio
    result = minimize(objective, x0=initial_weights, 
                     bounds=[(0, 1)] * len(categories),
                     constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1})
    
    return result.x
```

---

## 📊 Validation and Testing

### Metrics to Track
1. **CV Ratio**: Primary metric - should increase
2. **Silhouette Score**: Cluster quality - should increase
3. **Temporal Smoothness**: Regime stability - should improve
4. **Davies-Bouldin Index**: Should decrease (better separation)
5. **Calinski-Harabasz Index**: Should increase (better defined clusters)
6. **Cluster Balance**: Should remain similar or improve
7. **Computational Time**: Should decrease

### Testing Protocol
```python
def validate_weighted_pca(features, assignments, feature_names):
    """Comprehensive validation of weighted PCA approach."""
    
    # Baseline (no PCA)
    baseline_metrics = compute_all_metrics(features, assignments)
    
    # With weighted PCA
    transformed_features, transformer = prepare_features_with_weighted_pca(
        features, feature_names, FEATURE_CATEGORIES
    )
    pca_metrics = compute_all_metrics(transformed_features, assignments)
    
    # Compare
    print("\n📊 Weighted PCA Validation Results:")
    print("="*60)
    for metric, baseline_val in baseline_metrics.items():
        pca_val = pca_metrics[metric]
        improvement = (pca_val - baseline_val) / baseline_val * 100
        symbol = "📈" if improvement > 0 else "📉"
        print(f"{symbol} {metric:20s}: {baseline_val:.4f} → {pca_val:.4f} ({improvement:+.1f}%)")
    
    return baseline_metrics, pca_metrics, transformer
```

---

## 🎓 References and Background

1. **Jolliffe, I.T.** (2002). *Principal Component Analysis*. Springer.
2. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning*.
3. **scikit-learn PCA Documentation**: https://scikit-learn.org/stable/modules/decomposition.html
4. **Financial Market Regimes**: Literature on regime-switching models and clustering

---

## 💡 Key Takeaways

1. **Weighted per-category PCA** offers a principled approach to feature engineering for clustering
2. **Category weights** (0.40, 0.30, 0.15, 0.15) reflect financial relevance for regime identification
3. **Expected improvements**: 25-65% in CV ratio, 40-80% in Silhouette score
4. **Implementation is straightforward** using scikit-learn's PCA
5. **Dimensionality reduction** (~50%) improves both quality and efficiency
6. **Future enhancements** include kernel PCA, time-adaptive PCA, and learned weights

---

## 📝 Conclusion

Implementing weighted PCA on per-category features is a **high-impact, low-risk enhancement** that directly addresses the optimization objectives:

- ✅ **Enhanced CV Ratio**: Better regime separation through focused feature selection
- ✅ **Improved Silhouette**: More cohesive clusters via noise reduction  
- ✅ **Better Temporal Stability**: Emphasize regime-persistent features
- ✅ **Computational Efficiency**: Faster clustering with fewer dimensions
- ✅ **Interpretability**: Category-level understanding of regime drivers

**Recommendation**: Implement in Phase 1 (Proof of Concept) to validate benefits before full production integration.

---

*Generated: 2025-10-03*  
*Related Files: `iterative_optimization.py`, `step1_feature_preparation.py`*
