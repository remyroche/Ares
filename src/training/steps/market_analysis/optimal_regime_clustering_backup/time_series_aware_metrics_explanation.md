# 📊 Time-Series Aware Metrics: Comprehensive Explanation

## 🎯 **Time-Series Aware Metrics Overview**

Time-series aware metrics consider the temporal nature of financial data, recognizing that:
- **Temporal dependencies** exist between observations
- **Trends and patterns** evolve over time
- **Seasonal effects** influence data characteristics
- **Concept drift** can occur as market conditions change
- **Temporal stability** is crucial for reliable clustering

## 🔍 **What Are Time-Series Aware Metrics?**

### **1. Temporal Consistency Analysis**
**Definition**: Measures how stable cluster assignments remain over time periods.

**Why Important**:
- **Market regimes evolve**: Economic conditions change over time
- **Stationarity assumptions**: Financial time series are often non-stationary
- **Predictive stability**: Clusters should remain relevant for future predictions

**Implementation**:
```python
def calculate_temporal_consistency(features, labels, timestamps):
    # Sort data by time
    sorted_indices = np.argsort(timestamps)
    sorted_features = features[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Create time windows
    n_windows = min(10, len(features) // 20)
    window_size = len(features) // n_windows

    consistency_scores = []

    for i in range(n_windows - 1):
        window1_labels = sorted_labels[i*window_size:(i+1)*window_size]
        window2_labels = sorted_labels[(i+1)*window_size:(i+2)*window_size]

        # Calculate Jaccard similarity between consecutive windows
        unique_labels = np.unique(np.concatenate([window1_labels, window2_labels]))
        overlap_matrix = np.zeros((len(unique_labels), 2))

        for j, label in enumerate(unique_labels):
            overlap_matrix[j, 0] = np.sum(window1_labels == label)
            overlap_matrix[j, 1] = np.sum(window2_labels == label)

        # Jaccard similarity = intersection / union
        intersection = np.sum(np.minimum(overlap_matrix[:, 0], overlap_matrix[:, 1]))
        union = np.sum(np.maximum(overlap_matrix[:, 0], overlap_matrix[:, 1]))
        jaccard = intersection / union if union > 0 else 0.0

        consistency_scores.append(jaccard)

    return np.mean(consistency_scores)
```

**Key Benefits**:
- **Detects concept drift**: Identifies when cluster characteristics change
- **Measures stability**: Quantifies how consistent clustering is over time
- **Early warning system**: Alerts when clusters become unstable

### **2. Trend Consistency Within Clusters**
**Definition**: Analyzes whether data points within clusters follow similar trends over time.

**Why Important**:
- **Homogeneous behavior**: Points in same cluster should behave similarly
- **Predictive power**: Similar trends enable better forecasting
- **Market regime identification**: Different regimes have different trend characteristics

**Implementation**:
```python
def calculate_trend_consistency(features, labels, timestamps):
    unique_labels = np.unique(labels)
    consistency_scores = []

    for label in unique_labels:
        cluster_mask = labels == label
        cluster_features = features[cluster_mask]
        cluster_timestamps = timestamps[cluster_mask]

        if len(cluster_features) < 10:
            continue

        # Sort by time within cluster
        sort_indices = np.argsort(cluster_timestamps)
        sorted_features = cluster_features[sort_indices]

        # Calculate trend consistency using multiple methods:

        # 1. Autocorrelation analysis
        if sorted_features.shape[1] > 2:  # Assuming trend feature exists
            trend_feature = sorted_features[:, 2]
            # Calculate autocorrelation for lag 1
            autocorr = np.corrcoef(trend_feature[:-1], trend_feature[1:])[0, 1]

        # 2. Linear trend fitting consistency
        time_points = np.arange(len(sorted_features))
        trend_consistency = []

        for feature_idx in range(min(4, sorted_features.shape[1])):
            feature_values = sorted_features[:, feature_idx]

            # Fit linear trend
            slope, intercept = np.polyfit(time_points, feature_values, 1)

            # Calculate R-squared
            predicted = slope * time_points + intercept
            ss_res = np.sum((feature_values - predicted) ** 2)
            ss_tot = np.sum((feature_values - np.mean(feature_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            trend_consistency.append(r_squared)

        # Average consistency across features
        consistency_scores.append(np.mean(trend_consistency))

    return np.mean(consistency_scores) if consistency_scores else 0.0
```

### **3. Seasonal Pattern Analysis**
**Definition**: Identifies and quantifies seasonal patterns within clusters.

**Why Important**:
- **Market cycles**: Financial markets exhibit seasonal patterns
- **Calendar effects**: Month-end, quarter-end effects
- **Holiday impacts**: Trading behavior changes around holidays

**Implementation**:
```python
def calculate_seasonal_consistency(features, labels, timestamps):
    # Extract seasonal components (simplified approach)
    unique_labels = np.unique(labels)
    seasonal_scores = []

    for label in unique_labels:
        cluster_mask = labels == label
        cluster_timestamps = timestamps[cluster_mask]
        cluster_features = features[cluster_mask]

        if len(cluster_features) < 30:  # Need sufficient data for seasonal analysis
            continue

        # Convert timestamps to seasonal indicators
        seasonal_patterns = []

        for feature_idx in range(min(4, cluster_features.shape[1])):
            feature_values = cluster_features[:, feature_idx]

            # Simple seasonal decomposition (would use more sophisticated methods in practice)
            # Group by month
            monthly_values = {}

            for i, timestamp in enumerate(cluster_timestamps):
                month = pd.to_datetime(timestamp).month if hasattr(timestamp, '__iter__') else timestamp.month
                if month not in monthly_values:
                    monthly_values[month] = []
                monthly_values[month].append(feature_values[i])

            # Calculate seasonal consistency
            if len(monthly_values) > 1:
                monthly_means = [np.mean(values) for values in monthly_values.values()]
                monthly_std = np.std(monthly_means)
                overall_mean = np.mean(feature_values)

                # Lower variation in monthly means = higher seasonal consistency
                seasonal_consistency = 1.0 / (1.0 + monthly_std / (overall_mean + 1e-6))
                seasonal_patterns.append(seasonal_consistency)

        if seasonal_patterns:
            seasonal_scores.append(np.mean(seasonal_patterns))

    return np.mean(seasonal_scores) if seasonal_scores else 0.0
```

### **4. Volatility Regime Stability**
**Definition**: Measures how consistent volatility patterns are within clusters over time.

**Why Important**:
- **Volatility clustering**: High volatility periods tend to cluster
- **Risk management**: Understanding volatility regime stability is crucial
- **Option pricing**: Volatility affects derivative valuations

**Implementation**:
```python
def calculate_volatility_regime_stability(features, labels, timestamps):
    unique_labels = np.unique(labels)
    stability_scores = []

    for label in unique_labels:
        cluster_mask = labels == label
        cluster_timestamps = timestamps[cluster_mask]
        cluster_features = features[cluster_mask]

        if len(cluster_features) < 20:
            continue

        # Calculate rolling volatility
        window_size = min(10, len(cluster_features) // 3)
        volatility_series = []

        for i in range(window_size, len(cluster_features)):
            window_data = cluster_features[i-window_size:i]
            # Calculate volatility as standard deviation
            volatility = np.mean(np.std(window_data, axis=0))
            volatility_series.append(volatility)

        if len(volatility_series) > 1:
            # Measure stability of volatility regime
            volatility_changes = np.abs(np.diff(volatility_series))
            stability = 1.0 / (1.0 + np.mean(volatility_changes))

            # Also consider autocorrelation of volatility
            if len(volatility_series) > 5:
                autocorr = np.corrcoef(volatility_series[:-1], volatility_series[1:])[0, 1]
                stability = stability * (1.0 + autocorr) / 2  # Bonus for persistent volatility

            stability_scores.append(stability)

    return np.mean(stability_scores) if stability_scores else 0.0
```

## 🎯 **Interpretability: How to Implement**

### **1. Feature Importance Analysis**
**Definition**: Quantifies which features contribute most to cluster separation.

**Implementation**:
```python
def calculate_feature_importance(features, labels):
    """Calculate importance of each feature in clustering."""
    n_features = features.shape[1]
    importance_scores = []

    for feature_idx in range(n_features):
        feature_values = features[:, feature_idx]

        # Calculate between-cluster and within-cluster variance
        unique_labels = np.unique(labels)
        between_variance = 0
        within_variance = 0

        for label in unique_labels:
            cluster_mask = labels == label
            cluster_values = feature_values[cluster_mask]

            # Between-cluster variance
            cluster_mean = np.mean(cluster_values)
            between_variance += len(cluster_values) * (cluster_mean - np.mean(feature_values))**2

            # Within-cluster variance
            within_variance += np.sum((cluster_values - cluster_mean)**2)

        # Feature importance = between / total variance
        total_variance = between_variance + within_variance
        importance = between_variance / total_variance if total_variance > 0 else 0
        importance_scores.append(importance)

    return np.array(importance_scores)
```

### **2. Cluster Explainability Score**
**Definition**: Measures how well cluster characteristics can be explained.

**Implementation**:
```python
def calculate_cluster_explainability(features, labels, feature_names=None):
    """Calculate how explainable each cluster is."""
    unique_labels = np.unique(labels)
    explainability_scores = []

    for label in unique_labels:
        cluster_mask = labels == label
        cluster_features = features[cluster_mask]

        if len(cluster_features) < 5:
            continue

        # 1. Calculate feature statistics for this cluster
        feature_stats = []
        for feature_idx in range(cluster_features.shape[1]):
            values = cluster_features[:, feature_idx]
            stats = {
                'mean': np.mean(values),
                'std': np.std(values),
                'skewness': scipy.stats.skew(values),
                'kurtosis': scipy.stats.kurtosis(values)
            }
            feature_stats.append(stats)

        # 2. Calculate cluster separation from other clusters
        other_mask = labels != label
        other_features = features[other_mask]

        separation_scores = []
        for feature_idx in range(cluster_features.shape[1]):
            cluster_values = cluster_features[:, feature_idx]
            other_values = other_features[:, feature_idx]

            # Effect size (Cohen's d)
            mean_diff = abs(np.mean(cluster_values) - np.mean(other_values))
            pooled_std = np.sqrt((np.var(cluster_values) + np.var(other_values)) / 2)
            effect_size = mean_diff / (pooled_std + 1e-6)

            separation_scores.append(effect_size)

        # 3. Combine into explainability score
        avg_separation = np.mean(separation_scores)
        feature_consistency = 1.0 / (1.0 + np.std(separation_scores))

        explainability = 0.7 * avg_separation + 0.3 * feature_consistency
        explainability_scores.append(explainability)

    return np.mean(explainability_scores) if explainability_scores else 0.0
```

### **3. Concept Drift Detection**
**Definition**: Identifies when cluster characteristics change over time.

**Implementation**:
```python
def detect_concept_drift(features, labels, timestamps, window_size=50):
    """Detect concept drift in clustering structure over time."""
    if len(features) < window_size * 2:
        return 0.0  # Not enough data

    # Sort by time
    sorted_indices = np.argsort(timestamps)
    sorted_features = features[sorted_indices]
    sorted_labels = labels[sorted_indices]

    drift_scores = []

    # Sliding window analysis
    for i in range(window_size, len(sorted_features) - window_size, window_size // 2):
        window1_end = i
        window2_start = i

        window1_features = sorted_features[i-window_size:i]
        window1_labels = sorted_labels[i-window_size:i]

        window2_features = sorted_features[i:i+window_size]
        window2_labels = sorted_labels[i:i+window_size]

        # Compare cluster characteristics between windows
        window1_centers = calculate_cluster_centers(window1_features, window1_labels)
        window2_centers = calculate_cluster_centers(window2_features, window2_labels)

        # Calculate drift as distance between cluster centers
        if len(window1_centers) == len(window2_centers):
            total_drift = 0
            for center1, center2 in zip(window1_centers, window2_centers):
                distance = np.linalg.norm(center1 - center2)
                total_drift += distance

            avg_drift = total_drift / len(window1_centers)
            drift_scores.append(avg_drift)

    if not drift_scores:
        return 0.0

    # Lower drift = higher stability
    return 1.0 / (1.0 + np.mean(drift_scores))
```

## 🎯 **Domain Fitness: How to Implement**

### **1. Volatility Clustering Constraints**
**Definition**: Ensures clusters respect market-specific volatility patterns.

**Implementation**:
```python
def check_volatility_constraints(features, labels, domain_constraints):
    """Check if clusters satisfy volatility-based domain constraints."""
    volatility_bounds = domain_constraints.get('volatility_bounds', (0.1, 5.0))

    unique_labels = np.unique(labels)
    constraint_satisfaction = []

    for label in unique_labels:
        cluster_mask = labels == label
        cluster_features = features[cluster_mask]

        if len(cluster_features) < 2:
            continue

        # Calculate cluster volatility (coefficient of variation)
        cluster_volatility = 0
        valid_calculations = 0

        for feature_idx in range(min(4, cluster_features.shape[1])):
            feature_values = cluster_features[:, feature_idx]
            feature_values = feature_values[np.isfinite(feature_values)]

            if len(feature_values) > 1:
                mean_val = np.mean(feature_values)
                std_val = np.std(feature_values)

                if mean_val > 0:
                    cv = std_val / mean_val
                    cluster_volatility += cv
                    valid_calculations += 1

        if valid_calculations > 0:
            avg_volatility = cluster_volatility / valid_calculations

            # Check against domain constraints
            if volatility_bounds[0] <= avg_volatility <= volatility_bounds[1]:
                constraint_satisfaction.append(1.0)
            else:
                constraint_satisfaction.append(0.0)

    return np.mean(constraint_satisfaction) if constraint_satisfaction else 0.0
```

### **2. Volume Distribution Analysis**
**Definition**: Ensures clusters have appropriate trading volume characteristics.

**Implementation**:
```python
def check_volume_constraints(features, labels, domain_constraints):
    """Check if clusters satisfy volume-based constraints."""
    volume_ranges = domain_constraints.get('volume_ranges', (0.1, 10.0))

    # Assuming volume is the first feature
    volume_feature = features[:, 0] if features.shape[1] > 0 else np.ones(len(features))

    unique_labels = np.unique(labels)
    constraint_satisfaction = []

    for label in unique_labels:
        cluster_mask = labels == label
        cluster_volumes = volume_feature[cluster_mask]

        if len(cluster_volumes) < 2:
            continue

        # Calculate normalized volume metrics
        mean_volume = np.mean(cluster_volumes)
        max_volume = np.max(cluster_volumes)
        normalized_volume = mean_volume / (max_volume + 1e-6)

        # Check against domain constraints
        if volume_ranges[0] <= normalized_volume <= volume_ranges[1]:
            constraint_satisfaction.append(1.0)
        else:
            constraint_satisfaction.append(0.0)

    return np.mean(constraint_satisfaction) if constraint_satisfaction else 0.0
```

### **3. Sharpe Ratio Optimization**
**Definition**: Evaluates clusters based on risk-adjusted returns.

**Implementation**:
```python
def calculate_sharpe_ratio_fitness(features, labels):
    """Calculate Sharpe ratio-based fitness for financial clusters."""
    unique_labels = np.unique(labels)
    sharpe_scores = []

    for label in unique_labels:
        cluster_mask = labels == label
        cluster_features = features[cluster_mask]

        if len(cluster_features) < 5:
            continue

        # Calculate returns (simplified approach)
        if cluster_features.shape[1] > 1:
            # Assume feature 1 represents price-like data
            price_series = cluster_features[:, 1]

            if len(price_series) > 1:
                returns = np.diff(price_series) / price_series[:-1]

                if len(returns) > 0:
                    mean_return = np.mean(returns)
                    std_return = np.std(returns)

                    if std_return > 0:
                        # Annualized Sharpe ratio (simplified)
                        sharpe_ratio = mean_return / std_return

                        # Normalize to [0, 1] range
                        normalized_sharpe = np.tanh(sharpe_ratio)
                        sharpe_scores.append(normalized_sharpe)

    return np.mean(sharpe_scores) if sharpe_scores else 0.5
```

### **4. Maximum Drawdown Constraints**
**Definition**: Ensures clusters don't violate risk management constraints.

**Implementation**:
```python
def calculate_drawdown_fitness(features, labels):
    """Calculate maximum drawdown-based fitness."""
    unique_labels = np.unique(labels)
    drawdown_scores = []

    for label in unique_labels:
        cluster_mask = labels == label
        cluster_features = features[cluster_mask]

        if len(cluster_features) < 10:
            continue

        # Calculate cumulative returns
        if cluster_features.shape[1] > 1:
            price_series = cluster_features[:, 1]

            # Calculate drawdown
            cumulative = np.cumprod(1 + np.diff(price_series) / price_series[:-1])
            peak = np.maximum.accumulate(cumulative)
            drawdown = np.min((cumulative / peak) - 1)

            # Convert to fitness score (lower drawdown = higher fitness)
            fitness_score = 1.0 / (1.0 + abs(drawdown))
            drawdown_scores.append(fitness_score)

    return np.mean(drawdown_scores) if drawdown_scores else 0.5
```

## 🔄 **Batch Transfer Processing: Clarification**

### **Batch Size Calculation**
- **Current Setting**: 10% batch size (`batch_size_ratio = 0.1`)
- **Maximum Transfers**: Not exactly 5 iterations × 10% = 50 transfers

### **Actual Processing Logic**
```python
def process_transfers_with_stability(features, labels, transfer_candidates, quality_evaluator):
    # Sort candidates by benefit (highest first)
    sorted_candidates = sorted(transfer_candidates, key=lambda x: x['transfer_benefit'], reverse=True)

    # Calculate batch size (10% of candidates)
    batch_size = max(1, int(len(sorted_candidates) * 0.1))

    current_iteration = 0
    while current_iteration < max_iterations and not converged:
        # Get current batch
        start_idx = current_iteration * batch_size
        end_idx = min((current_iteration + 1) * batch_size, len(sorted_candidates))
        current_batch = sorted_candidates[start_idx:end_idx]

        # Apply transfers in this batch
        # ... transfer logic ...

        current_iteration += 1
```

### **Why Not Exactly 50 Transfers?**
1. **Batch Overlap**: Each iteration processes 10% of remaining candidates
2. **Dynamic Sizing**: Batch size depends on remaining candidates
3. **Early Convergence**: Process may stop before 5 iterations if quality doesn't improve
4. **Transfer Validation**: Some transfers may be rejected due to constraints

### **Example Processing**
- **Initial candidates**: 200 regimes to potentially transfer
- **Iteration 1**: Process 20 candidates (10% of 200)
- **Iteration 2**: Process 18 candidates (10% of remaining 180)
- **Iteration 3**: Process 16 candidates (10% of remaining 162)
- **Total transfers**: 20 + 18 + 16 = 54 transfers (not exactly 50)

### **Benefits of This Approach**
- **Gradual optimization**: Prevents sudden quality degradation
- **Adaptive processing**: Adjusts to data characteristics
- **Quality monitoring**: Can stop early if no improvement
- **Memory efficiency**: Processes manageable chunks

This approach ensures **stable, controlled optimization** rather than attempting all transfers simultaneously, which could cause **system instability** or **poor clustering quality**.