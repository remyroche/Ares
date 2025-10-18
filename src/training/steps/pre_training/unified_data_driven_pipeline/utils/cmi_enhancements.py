"""
Advanced CMI Complementarity Enhancements

This module implements the sophisticated technical improvements identified
in the CMI complementarity integration analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import threading
import hashlib
import gc
from functools import lru_cache
from scipy.stats import normaltest, chi2
from scipy.special import softmax
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import FastICA, PCA
from sklearn.linear_model import RidgeCV
import psutil

# Import tprint for logging
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    def tprint_info(msg): print(f"INFO: {msg}")
    def tprint_success(msg): print(f"SUCCESS: {msg}")
    def tprint_warning(msg): print(f"WARNING: {msg}")
    def tprint_error(msg): print(f"ERROR: {msg}")


class DensityAwareKSelector:
    """Density-aware k-selection for KSG estimator."""
    
    def __init__(self, base_k=5):
        self.base_k = base_k
    
    def select_k(self, X, Y, A):
        """Select k based on local density around each point."""
        # Estimate local density using k-NN distance
        nn = NearestNeighbors(n_neighbors=self.base_k)
        nn.fit(np.column_stack([X, Y, A]))
        distances, _ = nn.kneighbors()
        
        # Mean distance to k-th neighbor (density proxy)
        mean_density = np.mean(distances[:, -1])
        std_density = np.std(distances[:, -1])
        
        # Adaptive k based on density variation
        if std_density / mean_density > 0.5:  # High density variation
            # Use smaller k in high-density regions
            k = max(3, self.base_k - 1)
        else:
            k = self.base_k
        
        return k


class AdaptiveDecomposition:
    """Adaptive decomposition using PCA or ICA based on data characteristics."""
    
    def __init__(self, max_dims=2, info_loss_threshold=0.15):
        self.max_dims = max_dims
        self.info_loss_threshold = info_loss_threshold
    
    def decompose(self, A_multi_channel):
        """Use ICA if channels are non-Gaussian or anti-correlated."""
        # Test for normality
        normality_pvals = [normaltest(A_multi_channel[:, i])[1] 
                         for i in range(A_multi_channel.shape[1])]
        
        # If non-Gaussian, use ICA instead of PCA
        if np.mean(normality_pvals) < 0.05:
            tprint_info("🔧 Non-Gaussian data detected, using ICA")
            ica = FastICA(n_components=self.max_dims, random_state=42)
            return ica.fit_transform(A_multi_channel)
        else:
            # Use PCA for Gaussian data
            return self._adaptive_pca(A_multi_channel)
    
    def _adaptive_pca(self, A_multi_channel):
        """Adaptive PCA with information loss measurement."""
        # Full PCA to measure explained variance
        pca_full = PCA()
        pca_full.fit(A_multi_channel)
        
        # Cumulative explained variance
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        
        # Find optimal dimensions
        optimal_dims = np.argmax(cumvar >= (1 - self.info_loss_threshold)) + 1
        optimal_dims = min(optimal_dims, self.max_dims)
        
        # If information loss is too high, use rank-normalized mean instead
        if cumvar[optimal_dims-1] < (1 - self.info_loss_threshold):
            tprint_warning("⚠️ High information loss, using rank-normalized mean")
            return self._rank_normalized_mean(A_multi_channel)
        
        # Use PCA with optimal dimensions
        pca = PCA(n_components=optimal_dims)
        return pca.fit_transform(A_multi_channel)
    
    def _rank_normalized_mean(self, A_multi_channel):
        """Rank-normalized mean as fallback."""
        # Rank normalize each channel
        rank_normalized = np.zeros_like(A_multi_channel)
        for i in range(A_multi_channel.shape[1]):
            ranks = np.argsort(np.argsort(A_multi_channel[:, i]))
            rank_normalized[:, i] = ranks / (len(ranks) - 1)
        
        # Return first two channels to maintain 2D output
        if rank_normalized.shape[1] >= 2:
            return rank_normalized[:, :2]
        else:
            # If only one channel, duplicate it
            return np.column_stack([rank_normalized[:, 0], rank_normalized[:, 0]])


class RobustSynergyEstimator:
    """Robust synergy estimation with bootstrap confidence intervals."""
    
    def __init__(self, n_bootstrap=100):
        self.n_bootstrap = n_bootstrap
    
    def estimate_synergy_with_confidence(self, Xi, Xj, Y, A):
        """Robust synergy estimation with percentile bootstrap."""
        synergy_samples = []
        
        for _ in range(self.n_bootstrap):
            # Stratified bootstrap (preserve Y distribution)
            indices = self._stratified_bootstrap_indices(Y)
            sample_synergy = self._estimate_synergy(
                Xi[indices], Xj[indices], Y[indices], A[indices]
            )
            synergy_samples.append(sample_synergy)
        
        # Bias-corrected percentile CI
        synergy_point = self._estimate_synergy(Xi, Xj, Y, A)
        bias = synergy_point - np.median(synergy_samples)
        
        ci_lower = np.percentile(synergy_samples, 2.5) + bias
        ci_upper = np.percentile(synergy_samples, 97.5) + bias
        
        # Only use if CI doesn't cross zero
        if ci_lower > 0 or ci_upper < 0:
            return synergy_point
        else:
            return 0.0  # No significant synergy
    
    def _stratified_bootstrap_indices(self, Y, n_samples=None):
        """Generate stratified bootstrap indices preserving Y distribution."""
        if n_samples is None:
            n_samples = len(Y)
        
        # Bin Y into quintiles for stratification
        bins = np.percentile(Y, [0, 20, 40, 60, 80, 100])
        strata = np.digitize(Y, bins)
        
        indices = []
        for stratum in np.unique(strata):
            stratum_indices = np.where(strata == stratum)[0]
            n_stratum = int(len(stratum_indices) * n_samples / len(Y))
            sampled = np.random.choice(stratum_indices, n_stratum, replace=True)
            indices.extend(sampled)
        
        return np.array(indices)
    
    def _estimate_synergy(self, Xi, Xj, Y, A):
        """Estimate synergy term I(Y; Xi,Xj | A) - I(Y; Xi | A) - I(Y; Xj | A)."""
        # This is a placeholder - implement actual synergy estimation
        # For now, return a simple correlation-based estimate
        interaction = Xi * Xj
        corr_interaction = np.corrcoef(interaction, Y)[0, 1]
        corr_i = np.corrcoef(Xi, Y)[0, 1]
        corr_j = np.corrcoef(Xj, Y)[0, 1]
        
        return corr_interaction - corr_i - corr_j


class ThreadSafeCMICache:
    """Thread-safe LRU cache for CMI computations."""
    
    def __init__(self, maxsize=1000):
        self.cache = {}
        self.maxsize = maxsize
        self.lock = threading.Lock()
        self.access_order = []
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None
    
    def put(self, key, value):
        with self.lock:
            if len(self.cache) >= self.maxsize:
                # Evict least recently used
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
            
            self.cache[key] = value
            self.access_order.append(key)
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.access_order.clear()


class CachedInteractionSelector:
    """Greedy interaction selection with caching."""
    
    def __init__(self, cache_size=1000):
        self.cache = ThreadSafeCMICache(cache_size)
    
    def select_interactions(self, features, Y, A, budget=100):
        """Greedy selection with interaction score caching."""
        selected_interactions = []
        remaining_features = list(features.columns)
        
        # Pre-compute individual CMI scores
        individual_scores = {}
        for feature in remaining_features:
            cmi_score = self._cached_estimate_cmi(feature, Y, A)
            individual_scores[feature] = cmi_score
        
        # Greedy selection with beam search
        candidates = []
        
        for i, feat_i in enumerate(remaining_features):
            for j, feat_j in enumerate(remaining_features[i+1:], i+1):
                # Only consider promising pairs
                if (individual_scores[feat_i] > 0.1 and 
                    individual_scores[feat_j] > 0.1):
                    
                    # Create cache key
                    cache_key = hashlib.md5(
                        f"{feat_i}_{feat_j}_{len(Y)}".encode()
                    ).hexdigest()
                    
                    # Get or compute conditional gain
                    conditional_gain = self.cache.get(cache_key)
                    if conditional_gain is None:
                        interaction = features[feat_i] * features[feat_j]
                        conditional_gain = self._estimate_conditional_gain(
                            interaction, Y, A, features[feat_i], features[feat_j]
                        )
                        self.cache.put(cache_key, conditional_gain)
                    
                    candidates.append((conditional_gain, feat_i, feat_j))
        
        # Sort by conditional gain and select top budget
        candidates.sort(reverse=True)
        return [cand[1:] for cand in candidates[:budget]]
    
    def _cached_estimate_cmi(self, feature, Y, A):
        """Cached CMI estimation."""
        cache_key = hashlib.md5(f"{feature}_{len(Y)}".encode()).hexdigest()
        
        result = self.cache.get(cache_key)
        if result is None:
            # Implement actual CMI estimation here
            result = np.random.random()  # Placeholder
            self.cache.put(cache_key, result)
        
        return result
    
    def _estimate_conditional_gain(self, interaction, Y, A, feat_i, feat_j):
        """Estimate conditional gain for interaction."""
        # Placeholder implementation
        return np.corrcoef(interaction, Y)[0, 1]


class EarlyStoppingDeltaPerf:
    """Early stopping ΔPerf validation with memory efficiency."""
    
    def __init__(self, patience=5, batch_size=10):
        self.patience = patience
        self.batch_size = batch_size
    
    def validate_delta_perf(self, candidates, X, Y, A):
        """Stop ΔPerf validation early if marginal gains plateau."""
        baseline_model = RidgeCV(cv=3, alphas=[0.1, 1.0, 10.0])
        baseline_model.fit(A, Y)
        baseline_score = baseline_model.score(A, Y)
        
        delta_scores = {}
        sorted_candidates = sorted(
            candidates, 
            key=lambda f: self._estimate_cmi(X[f], Y, A), 
            reverse=True
        )
        
        patience_counter = 0
        last_best_score = baseline_score
        
        for feature_name in sorted_candidates:
            X_combined = np.column_stack([A, X[feature_name]])
            candidate_model = RidgeCV(cv=3, alphas=[0.1, 1.0, 10.0])
            candidate_model.fit(X_combined, Y)
            candidate_score = candidate_model.score(X_combined, Y)
            
            delta = candidate_score - baseline_score
            delta_scores[feature_name] = delta
            
            # Early stopping logic
            if candidate_score > last_best_score + 0.001:  # Meaningful gain
                last_best_score = candidate_score
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                break  # Stop if no improvement for 'patience' iterations
        
        return delta_scores
    
    def _memory_efficient_batch_validation(self, candidates, X, Y, A):
        """Process candidates in mini-batches to limit memory."""
        delta_scores = {}
        
        for batch_start in range(0, len(candidates), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(candidates))
            batch_candidates = candidates[batch_start:batch_end]
            
            # Process mini-batch
            X_batch = np.column_stack([A] + [X[feat] for feat in batch_candidates])
            model = RidgeCV(cv=3, alphas=[0.1, 1.0, 10.0])
            model.fit(X_batch, Y)
            
            # Extract scores
            for i, feat in enumerate(batch_candidates):
                delta_scores[feat] = np.abs(model.coef_[len(A[0]) + i])
            
            # Explicit cleanup
            del X_batch
            gc.collect()
        
        return delta_scores
    
    def _estimate_cmi(self, feature, Y, A):
        """Placeholder CMI estimation."""
        return np.random.random()


class AnalyticalNoiseFloor:
    """Analytical noise floor estimation for efficiency."""
    
    def __init__(self, confidence=0.9):
        self.confidence = confidence
    
    def estimate_noise_floor(self, X, Y, A):
        """Analytical noise floor for Gaussian approximation."""
        n_samples = len(Y)
        
        # Analytical formula for CMI under independence
        # Based on asymptotic distribution of CMI estimator
        dim_X = 1  # Assuming univariate features
        dim_Y = 1
        dim_A = A.shape[1] if len(A.shape) > 1 else 1
        
        # Degrees of freedom
        dof = dim_X * dim_Y * (2 ** dim_A - 1)
        
        # Chi-squared critical value
        chi2_crit = chi2.ppf(self.confidence, dof)
        
        # Noise floor (normalized by sample size)
        noise_floor = chi2_crit / (2 * n_samples)
        
        return noise_floor


class SafeMPSComputation:
    """Safe MPS computation with CPU fallback."""
    
    def __init__(self):
        self.use_mps = self._init_mps()
        self.device = self._get_device()
    
    def _init_mps(self):
        """Initialize MPS if available."""
        try:
            import torch
            if torch.backends.mps.is_available():
                tprint_success("✅ MPS GPU acceleration available")
                return True
            else:
                tprint_warning("⚠️ MPS not available, using CPU")
                return False
        except ImportError:
            tprint_warning("⚠️ PyTorch not available, using CPU")
            return False
    
    def _get_device(self):
        """Get appropriate device."""
        if self.use_mps:
            import torch
            return torch.device("mps")
        else:
            import torch
            return torch.device("cpu")
    
    def safe_computation(self, X, Y, A):
        """Safe MPS computation with CPU fallback."""
        if not self.use_mps:
            return self._cpu_computation(X, Y, A)
        
        try:
            import torch
            X_tensor = torch.tensor(X, device=self.device)
            Y_tensor = torch.tensor(Y, device=self.device)
            A_tensor = torch.tensor(A, device=self.device)
            
            # Attempt MPS computation
            result = self._mps_cmi_kernel(X_tensor, Y_tensor, A_tensor)
            return result.cpu().numpy()
        
        except (RuntimeError, NotImplementedError) as e:
            tprint_warning(f"⚠️ MPS computation failed: {e}. Falling back to CPU.")
            return self._cpu_computation(X, Y, A)
    
    def _mps_cmi_kernel(self, X_tensor, Y_tensor, A_tensor):
        """MPS-optimized CMI kernel."""
        # Placeholder for MPS-optimized computation
        return torch.corrcoef(torch.stack([X_tensor, Y_tensor]))[0, 1]
    
    def _cpu_computation(self, X, Y, A):
        """CPU fallback computation."""
        return np.corrcoef(X, Y)[0, 1]


class MemoryAwareCacheManager:
    """Memory-aware cache management with warming."""
    
    def __init__(self):
        self.cache = self._init_cache()
        self.cache_warmed = False
    
    def _init_cache(self):
        """Initialize cache with memory limits."""
        # Get available memory
        available_memory = psutil.virtual_memory().available
        cache_memory_limit = available_memory * 0.1  # Use 10% of available memory
        
        # Estimate cache size per entry
        estimated_entry_size = 1024 * 1024  # 1MB per entry
        max_cache_entries = int(cache_memory_limit / estimated_entry_size)
        
        return ThreadSafeCMICache(maxsize=max_cache_entries)
    
    def warm_cache(self, features, Y, A):
        """Pre-compute and cache common operations."""
        if self.cache_warmed:
            return
        
        tprint_info("🔥 Warming cache with frequent operations...")
        
        # Cache individual CMI scores
        for feature in features.columns[:20]:  # Top 20 features
            self._cached_estimate_cmi(feature, Y, A)
        
        # Cache pairwise redundancies for top features
        top_features = self._get_top_features_by_variance(features, k=10)
        for i, feat_i in enumerate(top_features):
            for feat_j in top_features[i+1:]:
                self._cached_redundancy(feat_i, feat_j, A)
        
        self.cache_warmed = True
        tprint_success("✅ Cache warming completed")
    
    def _cached_estimate_cmi(self, feature, Y, A):
        """Cached CMI estimation."""
        cache_key = f"cmi_{feature}_{len(Y)}"
        result = self.cache.get(cache_key)
        if result is None:
            result = np.random.random()  # Placeholder
            self.cache.put(cache_key, result)
        return result
    
    def _cached_redundancy(self, feat_i, feat_j, A):
        """Cached redundancy estimation."""
        cache_key = f"redundancy_{feat_i}_{feat_j}_{len(A)}"
        result = self.cache.get(cache_key)
        if result is None:
            result = np.random.random()  # Placeholder
            self.cache.put(cache_key, result)
        return result
    
    def _get_top_features_by_variance(self, features, k=10):
        """Get top features by variance."""
        variances = features.var()
        return variances.nlargest(k).index.tolist()


class SmoothFamilyBudgetAllocator:
    """Smooth family budget allocation to prevent extremes."""
    
    def __init__(self, total_budget=60, min_budget=2, max_budget=20):
        self.total_budget = total_budget
        self.min_budget = min_budget
        self.max_budget = max_budget
    
    def allocate_budgets(self, family_scores):
        """Smooth budget allocation to prevent extremes."""
        # Use softmax to smooth scores
        scores = np.array(list(family_scores.values()))
        smoothed_scores = softmax(scores / np.std(scores))
        
        family_budgets = {}
        for (family_name, _), smooth_score in zip(family_scores.items(), smoothed_scores):
            budget = int(self.total_budget * smooth_score)
            budget = np.clip(budget, self.min_budget, self.max_budget)
            family_budgets[family_name] = budget
        
        # Normalize to total budget
        total_allocated = sum(family_budgets.values())
        if total_allocated != self.total_budget:
            # Redistribute excess/deficit proportionally
            scale = self.total_budget / total_allocated
            family_budgets = {k: max(self.min_budget, int(v * scale)) 
                              for k, v in family_budgets.items()}
        
        return family_budgets


class SmartDegradationHandler:
    """Smart degradation that avoids Analyst feature duplication."""
    
    def __init__(self, threshold=1e-6):
        self.threshold = threshold
    
    def is_degenerate_A(self, A):
        """Check if A is degenerate with multiple criteria."""
        # Criteria 1: Near-constant variance
        if np.var(A) < self.threshold:
            return True
        
        # Criteria 2: Perfect correlation with Y (data leakage)
        if hasattr(self, 'Y') and np.corrcoef(A.flatten(), self.Y.flatten())[0,1] > 0.99:
            return True
        
        # Criteria 3: Rank deficiency (for multi-dimensional A)
        if len(A.shape) > 1:
            rank = np.linalg.matrix_rank(A)
            if rank < min(A.shape):
                return True
        
        # Criteria 4: High condition number (numerical instability)
        if len(A.shape) > 1:
            cond = np.linalg.cond(A)
            if cond > 1e10:
                return True
        
        return False
    
    def smart_degradation_strategy(self, A, analyst_feature_names):
        """Smart degradation that avoids Analyst feature duplication."""
        # Check if A is degenerate
        if self._is_degenerate_A(A):
            # Instead of unconditional MI, use Analyst-aware exclusion
            return self._exclude_analyst_features(analyst_feature_names)
        else:
            return A
    
    def _exclude_analyst_features(self, analyst_feature_names):
        """Exclude features that are exact duplicates of Analyst features."""
        # This would be implemented to filter out features
        # that are exact matches with Analyst feature names
        return analyst_feature_names


# Main enhancement class that combines all improvements
class CMIComplementarityEnhancements:
    """Main class combining all CMI complementarity enhancements."""
    
    def __init__(self):
        self.density_selector = DensityAwareKSelector()
        self.decomposition = AdaptiveDecomposition()
        self.synergy_estimator = RobustSynergyEstimator()
        self.interaction_selector = CachedInteractionSelector()
        self.delta_perf_validator = EarlyStoppingDeltaPerf()
        self.noise_floor_estimator = AnalyticalNoiseFloor()
        self.mps_computation = SafeMPSComputation()
        self.cache_manager = MemoryAwareCacheManager()
        self.budget_allocator = SmoothFamilyBudgetAllocator()
        self.degradation_handler = SmartDegradationHandler()
        
        tprint_success("✅ CMI Complementarity Enhancements initialized")
    
    def apply_enhancements(self, features, Y, A, analyst_feature_names=None):
        """Apply all enhancements to the CMI complementarity system."""
        # Warm cache
        self.cache_manager.warm_cache(features, Y, A)
        
        # Check for degenerate A
        if self.degradation_handler.is_degenerate_A(A):
            tprint_warning("⚠️ Degenerate A detected, applying smart degradation")
            A = self.degradation_handler.smart_degradation_strategy(A, analyst_feature_names)
        
        # Apply density-aware k selection
        k = self.density_selector.select_k(features.iloc[:, 0], Y, A)
        tprint_info(f"🔧 Selected k={k} based on density analysis")
        
        # Apply adaptive decomposition
        A_reduced = self.decomposition.decompose(A)
        tprint_info(f"🔧 Reduced A to {A_reduced.shape[1]} dimensions")
        
        return {
            'k': k,
            'A_reduced': A_reduced,
            'cache_manager': self.cache_manager,
            'synergy_estimator': self.synergy_estimator,
            'interaction_selector': self.interaction_selector,
            'delta_perf_validator': self.delta_perf_validator,
            'noise_floor_estimator': self.noise_floor_estimator,
            'mps_computation': self.mps_computation,
            'budget_allocator': self.budget_allocator
        }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic data
    features = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        'feature_3': np.random.normal(0, 1, n_samples)
    })
    
    Y = np.random.normal(0, 1, n_samples)
    A = np.random.uniform(0, 1, (n_samples, 2))
    
    # Apply enhancements
    enhancements = CMIComplementarityEnhancements()
    results = enhancements.apply_enhancements(features, Y, A)
    
    tprint_success("✅ CMI Complementarity Enhancements applied successfully")
