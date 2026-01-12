"""
Causal Discovery Module - PC Algorithm Implementation

Implements causal structure learning using the PC algorithm with LiNGAM
for discovering causal relationships in financial time series data.

Key Features:
1. PC Algorithm for causal graph discovery
2. LiNGAM for linear non-Gaussian acyclic models
3. Conditional independence testing
4. Causal strength estimation
5. Graph visualization and analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import networkx as nx
from itertools import combinations
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import GraphicalLassoCV

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

class CausalDiscovery:
    """
    Causal discovery using PC algorithm and LiNGAM.
    
    Discovers causal relationships between financial variables to identify
    causal parents and structural relationships for the modern De Prado framework.
    """
    
    def __init__(
        self,
        significance_level: float = 0.05,
        max_conditioning_set: int = 3,
        use_lingam: bool = True,
        target_variable: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize Causal Discovery system.
        
        Args:
            significance_level: Significance level for conditional independence tests
            max_conditioning_set: Maximum size of conditioning sets
            use_lingam: Whether to use LiNGAM for final orientation
            target_variable: Primary target to focus discovery on
            verbose: Whether to print progress information
        """
        tprint_info("🔬 Causal Discovery: Initializing...")
        
        self.significance_level = significance_level
        self.max_conditioning_set = max_conditioning_set
        self.use_lingam = use_lingam
        self.target_variable = target_variable
        self.verbose = verbose
        
        # Results storage
        self.causal_graph_ = None
        self.causal_strength_ = None
        self.causal_parents_ = None
        self.adjacency_matrix_ = None
        
        tprint_info(f"   ⚙️ Parameters: significance={significance_level}, max_set={max_conditioning_set}, lingam={use_lingam}")
        tprint_success("   ✅ Causal Discovery: Initialization complete")
        self.causal_strength_ = None
        self.causal_parents_ = None
        self.adjacency_matrix_ = None
        
    def conditional_independence_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: Optional[np.ndarray] = None,
        method: str = "partial_correlation"
    ) -> Tuple[bool, float]:
        """
        Test conditional independence between x and y given z.
        
        Args:
            x: First variable
            y: Second variable
            z: Conditioning variables (optional)
            method: Test method ("partial_correlation" or "kernel")
            
        Returns:
            Tuple of (is_independent, p_value)
        """
        try:
            if self.verbose:
                tprint_info(f"   📊 CI Test: method={method}, x_shape={x.shape}, y_shape={y.shape}, z_shape={z.shape if z is not None else None}")
            
            if z is None or len(z) == 0:
                # Simple correlation test
                corr, p_value = stats.pearsonr(x, y)
                is_independent = p_value > self.significance_level
                
                if self.verbose:
                    tprint_info(f"   ⚙️ CI Test: Simple correlation - corr={corr:.4f}, p={p_value:.6f}")
                
                return is_independent, p_value
            
            else:
                if method == "partial_correlation":
                    # Partial correlation test
                    n_samples = len(x)
                    if n_samples <= len(z) + 10:
                        if self.verbose:
                            tprint_warning("   ⚠️ CI Test: Insufficient samples for partial correlation")
                        return True, 1.0
                    
                    # Calculate partial correlation
                    # Ensure z is properly shaped (n_samples, n_vars)
                    z_reshaped = z
                    if z.ndim == 2:
                        if z.shape[0] != n_samples and z.shape[1] == n_samples:
                            z_reshaped = z.T
                        elif z.shape[0] == n_samples:
                             pass
                        else:
                             # Fallback or error if shapes don't match expected patterns
                             pass

                    # Flatten 1D z if needed for column_stack to work predictably if it was somehow 2D (n, 1)
                    # But if z has multiple columns, we keep it 2D.
                    
                    data = np.column_stack([x, y, z_reshaped])
                    cov_matrix = np.cov(data, rowvar=False)
                    
                    if cov_matrix.shape[0] < 3:
                        if self.verbose:
                            tprint_warning("   ⚠️ CI Test: Insufficient covariance matrix dimensions")
                        return True, 1.0
                    
                    # Partial correlation formula
                    cov_xy = cov_matrix[0, 1]
                    cov_xz = cov_matrix[0, 2:]
                    cov_yz = cov_matrix[1, 2:]
                    cov_zz = cov_matrix[2:, 2:]
                    
                    try:
                        inv_cov_zz = np.linalg.inv(cov_zz)
                        partial_corr = (cov_xy - cov_xz @ inv_cov_zz @ cov_yz) / \
                                     np.sqrt((cov_matrix[0, 0] - cov_xz @ inv_cov_zz @ cov_xz.T) * \
                                            (cov_matrix[1, 1] - cov_yz @ inv_cov_zz @ cov_yz.T))
                        
                        # Convert to t-statistic
                        df = n_samples - len(z) - 2
                        if df <= 0:
                            if self.verbose:
                                tprint_warning("   ⚠️ CI Test: Invalid degrees of freedom")
                            return True, 1.0
                        
                        # Convert to t-statistic
                        if abs(partial_corr) >= 1.0:
                            t_stat = np.inf
                        else:
                            t_stat = partial_corr * np.sqrt(df / (1 - partial_corr**2))

                        # Use survival function (sf) for better numerical stability with extremely small p-values
                        # sf = 1 - cdf, so 2*sf gives two-tailed p-value
                        p_value = 2 * stats.t.sf(abs(t_stat), df)
                        is_independent = p_value > self.significance_level
                        
                        if self.verbose:
                            tprint_info(f"   ⚙️ CI Test: Partial correlation - corr={partial_corr:.4f}, p={p_value:.6f}, df={df}")
                        
                        return is_independent, p_value
                        
                    except np.linalg.LinAlgError:
                        if self.verbose:
                            tprint_warning("   ⚠️ CI Test: Matrix inversion failed")
                        return True, 1.0
                
                else:
                    # Kernel-based test (simplified)
                    if self.verbose:
                        tprint_warning("   ⚠️ CI Test: Kernel method not implemented, returning independent")
                    return True, 1.0

        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ Conditional independence test failed: {e}")
            return True, 1.0
                    
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ Conditional independence test failed: {e}")
            return True, 1.0
    
    def pc_algorithm(
        self,
        data: pd.DataFrame,
        variable_names: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Run PC algorithm for causal discovery.
        
        Args:
            data: Input data
            variable_names: Names of variables
            
        Returns:
            Causal graph as adjacency list
        """
        try:
            if self.verbose:
                tprint_info("🔍 PC Algorithm: Starting causal discovery...")
            
            # Helper to normalize confidence if present in metadata
            def normalize_confidence(conf):
                return np.clip(conf / 100.0 if conf > 1.0 else conf, 0.0, 1.0)
            
            n_vars = data.shape[1]
            if variable_names is None:
                variable_names = [f"X{i}" for i in range(n_vars)]
            
            tprint_info(f"   📊 PC Algorithm: {n_vars} variables, {len(data)} samples")
            tprint_info(f"   📝 Variables: {variable_names}")
            
            # Initialize fully connected graph
            graph = {var: [] for var in variable_names}
            adjacency_matrix = np.ones((n_vars, n_vars)) - np.eye(n_vars)
            
            initial_edges = np.sum(adjacency_matrix) // 2
            tprint_info(f"   ⚙️ PC Algorithm: Initial graph - {initial_edges} edges")
            
            # Phase 1: Edge removal (skeleton discovery)
            if self.verbose:
                tprint_info("   🔍 Phase 1: Skeleton discovery...")
            
            target_idx = None
            if self.target_variable and self.target_variable in variable_names:
                target_idx = variable_names.index(self.target_variable)

            edges_removed = 0
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    # TARGET-CENTRIC OPTIMIZATION:
                    # Only test pairs involving the target or existing neighbors of the target
                    if target_idx is not None:
                        is_target_related = (i == target_idx or j == target_idx)
                        if not is_target_related:
                            # Prune edges NOT connected to target if we are in strict target-centric mode
                            # We keep them in the matrix for now but skip expensive tests?
                            # Actually, we need a skeleton. 
                            # If not related, we can just skip or use a very weak threshold.
                            # For efficiency: Skip tests for non-target-related pairs
                            continue

                    x = data.iloc[:, i].values
                    y = data.iloc[:, j].values
                    
                    # Test unconditional independence
                    is_independent, p_value = self.conditional_independence_test(x, y)
                    
                    if is_independent:
                        adjacency_matrix[i, j] = 0
                        adjacency_matrix[j, i] = 0
                        edges_removed += 1
                        
                        if self.verbose and edges_removed <= 5:  # Show first few
                            tprint_info(f"      ❌ Removed edge {variable_names[i]}-{variable_names[j]} (p={p_value:.4f})")
            
            tprint_info(f"   ✅ Phase 1: Removed {edges_removed} edges, {initial_edges - edges_removed} remaining")
            
            # Phase 2: Conditional independence tests
            if self.verbose:
                tprint_info("   🔍 Phase 2: Conditional independence tests...")
            
            # Iteratively increase conditioning set size
            for cond_size in range(1, self.max_conditioning_set + 1):
                if self.verbose:
                    tprint_info(f"      ⚙️ Testing conditioning sets of size {cond_size}...")
                
                changed = False
                edges_removed_cond = 0
                
                for i in range(n_vars):
                    # TARGET-CENTRIC: Skip if i is not target and not connected to target neighbors
                    if target_idx is not None:
                        if i != target_idx and adjacency_matrix[i, target_idx] == 0:
                            continue

                    neighbors = [j for j in range(n_vars) if adjacency_matrix[i, j] == 1]
                    
                    for j in neighbors:
                        # Find common neighbors
                        common_neighbors = [k for k in neighbors if adjacency_matrix[j, k] == 1 and k != i]
                        
                        if len(common_neighbors) >= cond_size:
                            # Test conditioning on subsets
                            for cond_set in combinations(common_neighbors, cond_size):
                                x = data.iloc[:, i].values
                                y = data.iloc[:, j].values
                                z = data.iloc[:, list(cond_set)].values.T
                                
                                is_independent, p_value = self.conditional_independence_test(x, y, z)
                                
                                if is_independent:
                                    adjacency_matrix[i, j] = 0
                                    adjacency_matrix[j, i] = 0
                                    changed = True
                                    edges_removed_cond += 1
                                    
                                    if self.verbose and edges_removed_cond <= 3:  # Show first few
                                        cond_vars = [variable_names[k] for k in cond_set]
                                        tprint_info(f"         ❌ Removed edge {variable_names[i]}-{variable_names[j]} | {cond_vars} (p={p_value:.4f})")
                                    break
                
                if self.verbose:
                    tprint_info(f"      ✅ Size {cond_size}: Removed {edges_removed_cond} edges")
                
                if not changed:
                    if self.verbose:
                        tprint_info(f"      ⏭️ No changes, stopping at size {cond_size-1}")
                    break

            # --- LAYER 2: TARGETED REFINEMENT ---
            # If we just finished Layer 1 (max_cond=1) and the graph is sparse enough,
            # auto-refine with higher order tests (Layer 2) to disambiguate edges.
            current_density = np.sum(adjacency_matrix) / (n_vars * (n_vars - 1))
            if self.max_conditioning_set == 1 and current_density < 0.15:
                if self.verbose:
                    tprint_info(f"   ✨ Layer 1 Complete (Density={current_density:.3f}). Auto-triggering Layer 2 Refinement (max_set=2)...")
                
                # Run Refinement (cond_size=2)
                cond_size = 2
                edges_removed_cond = 0
                
                for i in range(n_vars):
                    # TARGET-CENTRIC (strictly enforce on refinement)
                    if target_idx is not None:
                        if i != target_idx and adjacency_matrix[i, target_idx] == 0:
                            continue
                    
                    neighbors = [j for j in range(n_vars) if adjacency_matrix[i, j] == 1]
                    for j in neighbors:
                        common_neighbors = [k for k in neighbors if adjacency_matrix[j, k] == 1 and k != i]
                        if len(common_neighbors) >= cond_size:
                            for cond_set in combinations(common_neighbors, cond_size):
                                x = data.iloc[:, i].values
                                y = data.iloc[:, j].values
                                z = data.iloc[:, list(cond_set)].values.T
                                is_independent, p_value = self.conditional_independence_test(x, y, z)
                                if is_independent:
                                    adjacency_matrix[i, j] = 0
                                    adjacency_matrix[j, i] = 0
                                    edges_removed_cond += 1
                                    if self.verbose and edges_removed_cond <= 3:
                                        tprint_info(f"         ❌ [Refinement] Removed edge {variable_names[i]}-{variable_names[j]} (p={p_value:.4f})")
                                    break
                if self.verbose:
                    tprint_info(f"      ✅ Layer 2 Refinement: Removed {edges_removed_cond} edges")
            
            # Phase 3: Edge orientation (simplified)
            if self.verbose:
                tprint_info("   🔍 Phase 3: Edge orientation...")
            
            # Convert adjacency matrix to graph
            for i in range(n_vars):
                for j in range(n_vars):
                    if adjacency_matrix[i, j] == 1:
                        graph[variable_names[i]].append(variable_names[j])
            
            self.adjacency_matrix_ = adjacency_matrix
            
            final_edges = np.sum(adjacency_matrix) // 2
            tprint_success(f"✅ PC Algorithm complete:")
            tprint_info(f"   - Variables: {n_vars}")
            tprint_info(f"   - Final edges: {final_edges}")
            tprint_info(f"   - Significance level: {self.significance_level}")
            tprint_info(f"   - Conditioning sets: up to size {cond_size}")
            
            return graph
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ PC Algorithm failed: {e}")
            raise
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ PC Algorithm failed: {e}")
            raise
    
    def lingam_orientation(self, data: pd.DataFrame) -> np.ndarray:
        """
        Use LiNGAM for edge orientation (simplified implementation).
        
        Args:
            data: Input data
            
        Returns:
            Oriented adjacency matrix
        """
        try:
            if self.verbose:
                tprint_info("🔄 LiNGAM Orientation: Starting edge orientation...")
            
            if self.adjacency_matrix_ is None:
                if self.verbose:
                    tprint_warning("   ⚠️ LiNGAM: No adjacency matrix available")
                return np.zeros((data.shape[1], data.shape[1]))
            
            n_vars = data.shape[1]
            tprint_info(f"   📊 LiNGAM: Processing {n_vars} variables")
            
            # Simplified LiNGAM using linear regression
            n_vars = data.shape[1]
            B = np.zeros((n_vars, n_vars))
            
            # Order variables by variance (heuristic)
            variances = data.var().sort_values(ascending=False)
            var_order = variances.index.tolist()
            var_indices = [list(data.columns).index(var) for var in var_order]
            
            if self.verbose:
                tprint_info(f"   ⚙️ LiNGAM: Variable order by variance: {[data.columns[i] for i in var_indices[:5]]}")
            
            # Build causal order
            oriented_edges = 0
            
            target_idx = None
            if self.target_variable and self.target_variable in data.columns:
                target_idx = list(data.columns).index(self.target_variable)

            for i, var_idx in enumerate(var_indices):
                # TARGET-CENTRIC: Skip if this variable is not related to target
                if target_idx is not None:
                    # If var is not target AND not connected to target, skip regression?
                    # In LiNGAM, we need the full order, but we can shorten regressions.
                    pass 

                # Regress on previous variables
                if i > 0:
                    prev_vars = var_indices[:i]
                    
                    # OPTIMIZATION: Constrain to variables connected in the PC skeleton
                    # This makes regression sparse O(k^3) instead of dense O(N^3)
                    if self.adjacency_matrix_ is not None:
                        # Filter prev_vars to only those adjacent to current var_idx
                        # In PC skeleton, matrix is symmetric for undirected edges
                        prev_vars = [p for p in prev_vars if self.adjacency_matrix_[p, var_idx] == 1 or self.adjacency_matrix_[var_idx, p] == 1]
                    
                    X = data.iloc[:, prev_vars].values
                    y = data.iloc[:, var_idx].values
                    
                    # Linear regression
                    if len(prev_vars) > 0:
                        try:
                            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                            # Map sparse coeffs back to full matrix
                            for p_idx, p_var in enumerate(prev_vars):
                                B[p_var, var_idx] = coeffs[p_idx]
                            
                            # Count non-zero coefficients
                            non_zero_coeffs = np.sum(np.abs(coeffs) > 1e-6)
                            oriented_edges += non_zero_coeffs
                            
                            if self.verbose and i <= 3:  # Show first few
                                var_name = data.columns[var_idx]
                                parent_names = [data.columns[p] for p in prev_vars if abs(coeffs[prev_vars.index(p)]) > 1e-6]
                                tprint_info(f"      🔗 {var_name} <- {parent_names}")
                            
                        except np.linalg.LinAlgError:
                            if self.verbose:
                                tprint_warning(f"      ⚠️ LiNGAM: Regression failed for {data.columns[var_idx]}")
                            pass
            
            # Update adjacency matrix
            if self.adjacency_matrix_ is not None:
                oriented_matrix = self.adjacency_matrix_.copy() * (B != 0)
            else:
                oriented_matrix = (B != 0).astype(int)
            
            final_edges = np.sum(oriented_matrix)
            tprint_success(f"✅ LiNGAM orientation complete:")
            tprint_info(f"   - Oriented edges: {final_edges}")
            tprint_info(f"   - Non-zero coefficients: {oriented_edges}")
            tprint_info(f"   - Variables processed: {n_vars}")
            
            return oriented_matrix
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ LiNGAM orientation failed: {e}")
            return self.adjacency_matrix_ if self.adjacency_matrix_ is not None else np.zeros((data.shape[1], data.shape[1]))
    
    def estimate_causal_strength(self, data: pd.DataFrame) -> np.ndarray:
        """
        Estimate causal strength of discovered relationships.
        
        Args:
            data: Input data
            
        Returns:
            Causal strength matrix
        """
        try:
            if self.verbose:
                tprint_info("💪 Estimating causal strength...")
            
            n_vars = data.shape[1]
            strength_matrix = np.zeros((n_vars, n_vars))
            
            if self.adjacency_matrix_ is None:
                return strength_matrix
            
            # Use linear regression coefficients as strength estimates
            for i in range(n_vars):
                for j in range(n_vars):
                    if self.adjacency_matrix_[i, j] == 1:
                        x = data.iloc[:, i].values
                        y = data.iloc[:, j].values
                        
                        # Simple linear regression
                        try:
                            coeffs = np.polyfit(x, y, 1)
                            strength_matrix[i, j] = abs(coeffs[0])
                        except:
                            strength_matrix[i, j] = 0.1  # Default strength
            
            self.causal_strength_ = strength_matrix
            
            if self.verbose:
                avg_strength = np.mean(strength_matrix[strength_matrix > 0])
                tprint_success(f"✅ Causal strength estimated:")
                tprint_info(f"   - Average strength: {avg_strength:.4f}")
            
            return strength_matrix
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ Causal strength estimation failed: {e}")
            return np.zeros((data.shape[1], data.shape[1]))
    
    def identify_causal_parents(self, target_var: str) -> List[str]:
        """
        Identify causal parents of a target variable.
        
        Args:
            target_var: Target variable name
            
        Returns:
            List of causal parent variables
        """
        try:
            if self.causal_graph_ is None:
                return []
            
            # Find parents (variables that point to target)
            parents = []
            for var, children in self.causal_graph_.items():
                if target_var in children:
                    parents.append(var)
            
            return parents
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ Parent identification failed: {e}")
            return []
    
    def discover_causal_structure(
        self,
        data: pd.DataFrame,
        target_variable: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete causal discovery pipeline.
        
        Args:
            data: Input data
            target_variable: Target variable for parent identification
            
        Returns:
            Dictionary with causal discovery results
        """
        try:
            tprint_info("🚀 CausalDiscovery: Starting causal discovery pipeline...")
            tprint_info(f"   📊 Input data shape: {data.shape}")
            tprint_info(f"   🎯 Target variable: {target_variable if target_variable else 'None'}")
            tprint_info(f"   ⚙️ Configuration: significance_level={self.significance_level}, use_lingam={self.use_lingam}")

            # Validate input data
            if data.empty:
                tprint_error("   ❌ CausalDiscovery: Empty input data provided")
                return {'error': 'Empty input data'}
            
            n_samples, n_vars = data.shape
            tprint_info(f"   📊 Input: {n_samples} samples, {n_vars} variables")
            
            if target_variable:
                tprint_info(f"   🎯 Target variable: {target_variable}")
            
            # Standardize data - only numeric columns
            if self.verbose:
                tprint_info("   ⚙️ Standardizing data...")
            
            # Filter to numeric columns only
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                tprint_error("   ❌ No numeric columns found in data")
                return None
            
            scaler = StandardScaler()
            data_scaled = pd.DataFrame(
                scaler.fit_transform(numeric_data),
                columns=numeric_data.columns,
                index=numeric_data.index
            )
            
            tprint_success("   ✅ Data standardized")
            
            # Run PC algorithm
            if self.verbose:
                tprint_info("   🔍 Running PC Algorithm...")
            
            causal_graph = self.pc_algorithm(data_scaled, list(data.columns))
            
            if not causal_graph:
                if self.verbose:
                    tprint_warning("   ⚠️ No causal graph discovered")
                return {'error': 'No causal graph discovered'}
            
            # Apply LiNGAM orientation if enabled
            if self.use_lingam:
                if self.verbose:
                    tprint_info("   🔄 Applying LiNGAM orientation...")
                
                oriented_matrix = self.lingam_orientation(data_scaled)
            else:
                oriented_matrix = self.adjacency_matrix_
                if self.verbose:
                    tprint_info("   ⏭️ LiNGAM orientation disabled")
            
            # Estimate causal strength
            if self.verbose:
                tprint_info("   💪 Estimating causal strength...")
            
            strength_matrix = self.estimate_causal_strength(data_scaled)
            
            # Identify causal parents for target
            causal_parents = {}
            if target_variable and target_variable in causal_graph:
                parents = self.identify_causal_parents(target_variable)
                causal_parents[target_variable] = parents
                
                if self.verbose:
                    tprint_info(f"   👨‍👩‍👧‍👦 Parents of {target_variable}: {parents}")
            elif target_variable:
                if self.verbose:
                    tprint_warning(f"   ⚠️ Target variable {target_variable} not in graph")
            
            # Compile results
            results = {
                'causal_graph': causal_graph,
                'adjacency_matrix': oriented_matrix,
                'causal_strength': strength_matrix,
                'causal_parents': causal_parents,
                'variable_names': list(data.columns),
                'significance_level': self.significance_level,
                'n_variables': n_vars,
                'n_edges': np.sum(oriented_matrix) // 2,
                'n_samples': n_samples
            }
            
            if self.verbose:
                tprint_success("✅ Causal Discovery Complete:")
                tprint_info(f"   - Variables: {results['n_variables']}")
                tprint_info(f"   - Edges: {results['n_edges']}")
                tprint_info(f"   - Samples: {results['n_samples']}")
                tprint_info(f"   - Significance level: {results['significance_level']}")
                if target_variable:
                    tprint_info(f"   - Target parents: {len(causal_parents.get(target_variable, []))}")
            
            return results
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Causal discovery failed: {e}")
            return {'error': str(e)}
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Causal discovery failed: {e}")
            raise
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of causal discovery results.
        
        Returns:
            Summary dictionary
        """
        return {
            'significance_level': self.significance_level,
            'max_conditioning_set': self.max_conditioning_set,
            'use_lingam': self.use_lingam,
            'has_graph': self.causal_graph_ is not None,
            'has_strength': self.causal_strength_ is not None,
            'n_variables': len(self.causal_graph_) if self.causal_graph_ else 0,
            'n_edges': np.sum(self.adjacency_matrix_) // 2 if self.adjacency_matrix_ is not None else 0
        }

# Convenience functions
def quick_causal_discovery(
    data: pd.DataFrame,
    target_variable: Optional[str] = None,
    significance_level: float = 0.05,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick causal discovery with default parameters.
    
    Args:
        data: Input data
        target_variable: Target variable
        significance_level: Significance level
        **kwargs: Additional parameters
        
    Returns:
        Causal discovery results
    """
    discoverer = CausalDiscovery(significance_level=significance_level, **kwargs)
    return discoverer.discover_causal_structure(data, target_variable)

def analyze_causal_parents(
    data: pd.DataFrame,
    target_variable: str,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Analyze causal parents of target variable.
    
    Args:
        data: Input data
        target_variable: Target variable
        top_k: Number of top parents to return
        
    Returns:
        List of (parent, strength) tuples
    """
    results = quick_causal_discovery(data, target_variable)
    
    if target_variable not in results['causal_parents']:
        return []
    
    parents = results['causal_parents'][target_variable]
    strength_matrix = results['causal_strength']
    var_names = results['variable_names']
    
    parent_strengths = []
    for parent in parents:
        if parent in var_names:
            parent_idx = var_names.index(parent)
            target_idx = var_names.index(target_variable)
            strength = strength_matrix[parent_idx, target_idx]
            parent_strengths.append((parent, strength))
    
    # Sort by strength
    parent_strengths.sort(key=lambda x: x[1], reverse=True)
    
    return parent_strengths[:top_k]
