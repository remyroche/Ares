# PCA-Based Feature Combination Implementation

## Overview

This document describes the implementation of a PCA-based feature combination approach that replaces simple removal of highly collinear features in the Ares trading system. Instead of deleting correlated features, this approach combines them using Principal Component Analysis (PCA) to create uncorrelated "meta-features" that retain the collective information while solving multicollinearity problems.

## Problem Statement

Traditional VIF-based feature selection simply removes features with high Variance Inflation Factor (VIF), which can lead to:
- Loss of valuable information from removed features
- Reduced model performance due to discarded features
- Inefficient use of available data

## Solution: PCA-Based Feature Combination

### Key Features

1. **High VIF Threshold**: Only features with VIF > 20.0 are considered for combination (configurable)
2. **Correlation-Based Clustering**: Features are grouped using hierarchical clustering based on correlation
3. **PCA Transformation**: Each cluster of highly correlated features is transformed into uncorrelated principal components
4. **Variance Preservation**: PCA components explain 95% of the original variance (configurable)

### Implementation Details

#### 1. VIF Calculation and Feature Identification

```python
def _combine_high_vif_features_with_pca(self, data: pd.DataFrame) -> pd.DataFrame:
    # Calculate VIF scores for all features
    vif_scores = {}
    for col in data_imputed.columns:
        # Use other features to predict this feature
        other_cols = [c for c in data_imputed.columns if c != col]
        X = data_imputed[other_cols]
        y = data_imputed[col]
        
        reg = LinearRegression()
        reg.fit(X, y)
        
        # Calculate VIF
        y_pred = reg.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        vif = 1 / (1 - r_squared) if r_squared != 1 else np.inf
        vif_scores[col] = vif
    
    # Identify features with high VIF
    high_vif_features = [col for col, vif in vif_scores.items() if vif > self.pca_high_vif_threshold]
```

#### 2. Correlation-Based Clustering

```python
# Create correlation matrix for high VIF features
high_vif_data = data_imputed[high_vif_features]
correlation_matrix = high_vif_data.corr().abs()

# Use hierarchical clustering to group highly correlated features
distance_matrix = 1 - correlation_matrix.values

clustering = AgglomerativeClustering(
    n_clusters=None,  # Let it determine optimal number
    distance_threshold=1 - self.pca_correlation_threshold,  # Features with correlation > threshold will be grouped
    linkage='complete'
)

cluster_labels = clustering.fit_predict(distance_matrix)
```

#### 3. Feature Scaling and PCA Transformation

**Critical: Feature Scaling for PCA**

PCA is highly sensitive to the scale of input features. Features with larger numerical ranges can disproportionately influence the principal components, leading to suboptimal results. Therefore, all features within a cluster are scaled before applying PCA.

```python
for cluster_id, features in multi_feature_clusters.items():
    # Prepare data for PCA
    cluster_data = data_imputed[features]
    
    # Log original feature statistics to understand scaling impact
    self.logger.info(f"📊 Cluster {cluster_id} - Original feature statistics:")
    for feature in features:
        mean_val = cluster_data[feature].mean()
        std_val = cluster_data[feature].std()
        range_val = cluster_data[feature].max() - cluster_data[feature].min()
        self.logger.info(f"   {feature}: mean={mean_val:.4f}, std={std_val:.4f}, range={range_val:.4f}")
    
    # Scale the data - crucial for PCA to work correctly
    # This ensures features with larger numerical ranges don't disproportionately influence PCA
    if self.pca_scaling_method == "standard":
        scaler = StandardScaler()  # Z-score normalization (mean=0, std=1)
    elif self.pca_scaling_method == "robust":
        scaler = RobustScaler()    # Robust to outliers
    elif self.pca_scaling_method == "minmax":
        scaler = MinMaxScaler()    # Scale to [0, 1] range
    
    cluster_data_scaled = scaler.fit_transform(cluster_data)
    
    # Validate scaling results
    cluster_data_scaled_df = pd.DataFrame(cluster_data_scaled, columns=features, index=cluster_data.index)
    self.logger.info(f"📊 Cluster {cluster_id} - Scaled feature statistics:")
    for feature in features:
        mean_val = cluster_data_scaled_df[feature].mean()
        std_val = cluster_data_scaled_df[feature].std()
        min_val = cluster_data_scaled_df[feature].min()
        max_val = cluster_data_scaled_df[feature].max()
        self.logger.info(f"   {feature}: mean={mean_val:.6f}, std={std_val:.6f}, range=[{min_val:.6f}, {max_val:.6f}]")
    
    # Apply PCA - keep components that explain configured percentage of variance
    pca = PCA(n_components=self.pca_variance_explained_threshold)
    pca_components = pca.fit_transform(cluster_data_scaled)
    
    # Create new feature names
    cluster_name = f"pca_cluster_{cluster_id}"
    for i in range(n_components):
        component_name = f"{cluster_name}_pc{i+1}"
        result_data[component_name] = pca_components[:, i]
```

**Why Scaling is Critical for PCA:**

1. **Equal Influence**: Without scaling, features with larger ranges dominate PCA
2. **Covariance Matrix**: PCA is based on the covariance matrix, which is scale-dependent
3. **Interpretability**: Scaled features produce more interpretable principal components
4. **Numerical Stability**: Prevents numerical issues in eigenvalue decomposition

**Demonstration Results:**

Testing with features at different scales (range 1 vs 100 vs 10,000) shows:

- **Without Scaling**: PCA produces only 1 component, dominated by the largest-scale feature
- **With Scaling**: PCA produces 3 components, capturing patterns from all features equally

This demonstrates that proper scaling is essential for PCA to work correctly and capture meaningful patterns across all features.

### Configuration

The PCA combination approach is configured in `src/config/feature_selection_config.yaml`:

```yaml
feature_selection:
  # PCA combination settings for highly collinear features
  pca_combination:
    # High VIF threshold for PCA combination (only features above this will be combined)
    high_vif_threshold: 20.0
    # Correlation threshold for clustering features (features with correlation > 0.7 will be grouped)
    correlation_threshold: 0.7
    # Variance explained threshold for PCA components (95% of variance)
    variance_explained_threshold: 0.95
    # Scaling method for PCA preprocessing (standard, robust, minmax)
    # Standard scaling (z-score) is recommended for PCA as it ensures features have zero mean and unit variance
    scaling_method: "standard"
```

### Integration with Feature Selection Pipeline

The PCA combination is integrated into the feature selection pipeline in `src/training/steps/vectorized_labelling_orchestrator.py`:

```python
# Combine high VIF features using PCA instead of removal
if (self.enable_vif_removal and 
    len(data.columns) > self.min_features_to_keep and 
    len(data.columns) > 2):
    
    self.logger.info("🔄 Applying PCA-based feature combination for highly collinear features...")
    data = self._combine_high_vif_features_with_pca(data)
    
    # After PCA combination, also remove any remaining extremely high VIF features
    remaining_high_vif_features = self._remove_high_vif_features_vectorized(data)
    # ... handle remaining high VIF features
```

## Benefits

### 1. Information Preservation
- **Before**: Removing 7 highly collinear features loses all their information
- **After**: Combining them into 3 PCA components preserves 95% of the variance

### 2. Multicollinearity Resolution
- **Before**: High VIF features cause model instability
- **After**: PCA components are uncorrelated by design

### 3. Feature Reduction
- **Before**: 10 features → 3 features (70% reduction)
- **After**: 10 features → 6 features (40% reduction) with better information retention

### 4. Model Performance
- Retains collective information from correlated features
- Reduces dimensionality while preserving variance
- Improves model stability and interpretability

## Test Results

Running the test script `test_pca_feature_combination.py` demonstrates:

```
📋 Summary:
   Original features: 10
   Final features: 6
   High VIF features before: 7
   High VIF features after: 0
   PCA components added: 3
```

The test shows that:
- 7 highly collinear features were identified (VIF > 20.0)
- 3 clusters were formed based on correlation
- Each cluster was transformed into 1 PCA component
- All high VIF issues were resolved
- Information was preserved through PCA transformation

## Usage

The PCA-based feature combination is automatically applied during the feature selection step in the training pipeline. No additional configuration is required beyond the settings in `feature_selection_config.yaml`.

### Monitoring and Validation

The implementation provides comprehensive logging and validation:

#### Feature Scaling Validation
- **Original Statistics**: Logs mean, std, and range for each feature before scaling
- **Scaled Statistics**: Logs mean, std, and range for each feature after scaling
- **Validation Checks**: Verifies scaling was successful based on the chosen method:
  - **StandardScaler**: Validates mean ≈ 0 and std ≈ 1
  - **RobustScaler**: Validates median ≈ 0
  - **MinMaxScaler**: Validates range within [0, 1]

#### General Monitoring
- VIF scores for all features
- Identification of highly collinear features
- Clustering results and cluster composition
- PCA variance explanation for each component
- Feature count changes and net reduction

Example log output:
```
🔍 VIF Analysis: Found 7 features with VIF > 20.0 for PCA combination
📊 High VIF feature - feature7: 90.34
📊 High VIF feature - feature2: 87.50
🔍 Found 3 clusters of highly collinear features for PCA combination
📊 Processing cluster 0: ['feature1', 'feature2', 'feature3']
📊 Cluster 0 - Original feature statistics:
   feature1: mean=0.0123, std=1.0456, range=6.2345
   feature2: mean=0.0246, std=2.0912, range=12.4690
   feature3: mean=0.0061, std=0.5228, range=3.1172
📊 Using StandardScaler (z-score normalization) for cluster 0
📊 Cluster 0 - Scaled feature statistics:
   feature1: mean=0.000000, std=1.000000, range=[-2.9876, 2.9876]
   feature2: mean=0.000000, std=1.000000, range=[-2.9876, 2.9876]
   feature3: mean=0.000000, std=1.000000, range=[-2.9876, 2.9876]
✅ Scaling validation passed for cluster 0
📊 Cluster 0 PCA: 1 components explain 0.988 of variance
🔄 Removed 7 highly collinear features
🔄 Added 3 PCA meta-features
🔄 Net change: -4 features
```

## Future Enhancements

1. **Adaptive Thresholds**: Dynamic VIF thresholds based on dataset characteristics
2. **Feature Importance Integration**: Consider feature importance when forming clusters
3. **Cross-Validation**: Validate PCA components on holdout data
4. **Interpretability**: Provide feature contribution analysis for PCA components
