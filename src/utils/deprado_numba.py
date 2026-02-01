"""
Numba-optimized utilities for De Prado Feature Engine.
Provides high-performance JIT-compiled functions for tree traversal, entropy calculation, and clustering metrics.
"""

import numpy as np
from numba import njit, prange

@njit(fastmath=True)
def get_node_depths_numba(children_left, children_right, feature, max_nodes):
    """
    Calculate the depth of the first occurrence of each feature in a decision tree.
    Uses an iterative stack-based approach to avoid recursion limits and improve performance.
    
    Args:
        children_left: Array of left children indices
        children_right: Array of right children indices
        feature: Array of feature indices (-2 for leaf nodes)
        max_nodes: Maximum number of nodes in the tree (for stack sizing)
        
    Returns:
        Two arrays (feature_indices, depths) representing every feature encounter.
        The caller must process these to find the minimum depth per feature.
    """
    # Initialize stack for DFS: (node_idx, depth)
    stack_node = np.zeros(max_nodes, dtype=np.int64)
    stack_depth = np.zeros(max_nodes, dtype=np.int64)
    
    stack_ptr = 0
    stack_node[0] = 0  # Start at root
    stack_depth[0] = 0
    
    # We return arrays of (feature_idx, depth) for every non-leaf node visited
    node_feats = np.empty(max_nodes, dtype=np.int64)
    node_depths = np.empty(max_nodes, dtype=np.int64)
    count = 0
    
    while stack_ptr >= 0:
        node = stack_node[stack_ptr]
        depth = stack_depth[stack_ptr]
        stack_ptr -= 1
        
        feat = feature[node]
        if feat != -2:  # Not a leaf
            # Record occurrence
            node_feats[count] = feat
            node_depths[count] = depth
            count += 1
            
            # Push children (Right first so Left is processed first)
            if children_right[node] != -1:
                stack_ptr += 1
                stack_node[stack_ptr] = children_right[node]
                stack_depth[stack_ptr] = depth + 1
                
            if children_left[node] != -1:
                stack_ptr += 1
                stack_node[stack_ptr] = children_left[node]
                stack_depth[stack_ptr] = depth + 1
                
    return node_feats[:count], node_depths[:count]


@njit(parallel=True, fastmath=True)
def calculate_entropy_numba(X_values, bin_edges):
    """
    Calculate Shannon Entropy for each column in X using pre-computed bin edges.
    Parallelized over features (columns).
    
    Args:
        X_values: Input matrix (n_samples, n_features)
        bin_edges: Matrix of bin edges (n_bins+1, n_features)
        
    Returns:
        Array of entropy values for each feature
    """
    n_samples, n_features = X_values.shape
    entropies = np.zeros(n_features, dtype=np.float64)
    n_edges = bin_edges.shape[0]
    
    for idx in prange(n_features):
        edges = bin_edges[:, idx]
        # Check if edges are valid (e.g. not all identical)
        if edges[0] == edges[-1]:
            entropies[idx] = 0.0
            continue
            
        # Numba's searchsorted works on 1D arrays
        # edges[1:-1] to define internal boundaries
        bins = np.searchsorted(edges[1:-1], X_values[:, idx], side='right')
        
        # bins will range from 0 to len(edges)-2 + 1 = len(edges)-1
        n_bins_actual = n_edges - 1
        counts = np.bincount(bins, minlength=n_bins_actual)
        
        # Filter zero counts for log
        valid_counts = counts[counts > 0]
        if len(valid_counts) == 0:
            entropies[idx] = 0.0
            continue
            
        probs = valid_counts.astype(np.float64) / n_samples
        entropies[idx] = -np.sum(probs * np.log(probs + 1e-12))
        
    return entropies


@njit(parallel=True, fastmath=True)
def spearman_corr_numba(X_ranked, y_ranked):
    """
    Vectorized Spearman correlation calculation.
    Assumes inputs are already rank-transformed.
    
    Args:
        X_ranked: Ranked feature matrix (n_samples, n_features)
        y_ranked: Ranked target vector (n_samples,)
        
    Returns:
        Array of correlation coefficients
    """
    n_samples, n_features = X_ranked.shape
    ics = np.zeros(n_features, dtype=np.float64)
    
    y_mean = np.mean(y_ranked)
    y_centered = y_ranked - y_mean
    y_ss = np.sum(y_centered ** 2)
    y_norm = np.sqrt(y_ss)
    
    if y_norm < 1e-12:
        return ics # Zero correlation if target is constant
    
    for i in prange(n_features):
        col = X_ranked[:, i]
        x_mean = np.mean(col)
        x_centered = col - x_mean
        x_ss = np.sum(x_centered ** 2)
        x_norm = np.sqrt(x_ss)
        
        if x_norm < 1e-12:
            ics[i] = 0.0
        else:
            # Pearson on ranks
            cov = np.sum(x_centered * y_centered)
            ics[i] = cov / (x_norm * y_norm)
            
    return ics

@njit(fastmath=True)
def calculate_cv_ratio_numba(values, labels):
    """
    Calculate CV Ratio (Between-Cluster Variance / Within-Cluster Variance).
    Optimized to avoid large intermediate allocations.
    """
    n_features, n_samples = values.shape
    n_clusters = np.max(labels) + 1
    
    # Global Mean (Centroid of all features)
    global_mean = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_features):
        for j in range(n_samples):
            global_mean[j] += values[i, j]
    global_mean /= n_features
    
    # Cluster Means
    cluster_counts = np.zeros(n_clusters, dtype=np.float64)
    cluster_sums = np.zeros((n_clusters, n_samples), dtype=np.float64)
    
    for i in range(n_features):
        c_idx = labels[i]
        cluster_counts[c_idx] += 1.0
        for j in range(n_samples):
            cluster_sums[c_idx, j] += values[i, j]
            
    # Compute WCSS and BCSS
    wcss = 0.0
    bcss = 0.0
    
    for c in range(n_clusters):
        count = cluster_counts[c]
        if count > 0:
            centroid = cluster_sums[c] / count
            
            # BCSS contribution
            dist_sq_b = 0.0
            for j in range(n_samples):
                d = centroid[j] - global_mean[j]
                dist_sq_b += d * d
            bcss += count * dist_sq_b
    
    # Second pass for WCSS
    for i in range(n_features):
        c_idx = labels[i]
        if cluster_counts[c_idx] > 0:
            centroid = cluster_sums[c_idx] / cluster_counts[c_idx]
            dist_sq_w = 0.0
            for j in range(n_samples):
                d = values[i, j] - centroid[j]
                dist_sq_w += d * d
            wcss += dist_sq_w
            
    if wcss > 1e-12:
        return bcss / wcss
    return 0.0
