"""
Causal Denoising & Enhancement Engine

Implements advanced denoising techniques using causal relationships:
1. Sparse covariance matrix optimization
2. Causal denoising using structural constraints
3. Graph-based noise filtering
4. Causal imputation of missing values
5. Feature enhancement using causal information

Key Features:
- Sparse precision matrix estimation
- Remove spurious correlations using causal structure
- Graph-based signal processing
- Causal constraint-based imputation
- Feature enhancement through causal relationships
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import time
from sklearn.covariance import GraphicalLassoCV, LedoitWolf
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.sparse import csr_matrix
import networkx as nx

# Import existing components
from .structural_causal_model import StructuralCausalModel

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")


class SparseCovarianceDenoiser:
    """
    Sparse covariance matrix optimization for efficient causal discovery.
    
    Uses GraphicalLasso to estimate sparse precision matrices that
    better reflect the underlying causal structure.
    """
    
    def __init__(
        self,
        alpha: Optional[float] = None,
        cv_folds: int = 3,
        max_iter: int = 100,
        verbose: bool = True
    ):
        """
        Initialize Sparse Covariance Denoiser.
        
        Args:
            alpha: Regularization parameter (None for CV selection)
            cv_folds: Number of CV folds for alpha selection
            max_iter: Maximum iterations for optimization
            verbose: Whether to print progress information
        """
        self.alpha = alpha
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        self.verbose = verbose
        
        # Storage for results
        self.precision_matrix_ = None
        self.covariance_matrix_ = None
        self.sparse_correlation_ = None
        
    def fit_sparse_covariance(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Fit sparse covariance matrix using GraphicalLasso.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Dictionary with covariance, precision, and correlation matrices
        """
        if self.verbose:
            tprint_info(f"🔍 Sparse Covariance: Fitting on {X.shape[1]} features")
        
        start_time = time.time()
        
        try:
            # Standardize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.fillna(X.mean()))
            
            # Fit GraphicalLasso
            if self.alpha is None:
                gl = GraphicalLassoCV(
                    cv=self.cv_folds,
                    max_iter=self.max_iter,
                    n_jobs=-1,
                    verbose=self.verbose
                )
            else:
                gl = GraphicalLasso(
                    alpha=self.alpha,
                    max_iter=self.max_iter,
                    verbose=self.verbose
                )
            
            gl.fit(X_scaled)
            
            # Store results
            self.covariance_matrix_ = gl.covariance_
            self.precision_matrix_ = gl.precision_
            
            # Compute sparse correlation matrix
            self.sparse_correlation_ = self._covariance_to_correlation(self.covariance_matrix_)
            
            fitting_time = time.time() - start_time
            
            if self.verbose:
                n_nonzero = np.count_nonzero(self.precision_matrix_)
                total_elements = self.precision_matrix_.size
                sparsity = 1 - (n_nonzero / total_elements)
                tprint_success(f"✅ Sparse Covariance: Complete!")
                tprint_info(f"   📊 Sparsity: {sparsity:.3f}")
                tprint_info(f"   📊 Non-zero elements: {n_nonzero}/{total_elements}")
                tprint_info(f"   ⏱️  Time: {fitting_time:.2f}s")
            
            return {
                'covariance': self.covariance_matrix_,
                'precision': self.precision_matrix_,
                'correlation': self.sparse_correlation_
            }
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Sparse covariance fitting failed: {e}")
            
            # Fallback to Ledoit-Wolf shrinkage
            try:
                lw = LedoitWolf()
                lw.fit(X_scaled.fillna(X_scaled.mean()))
                
                self.covariance_matrix_ = lw.covariance_
                self.precision_matrix_ = np.linalg.inv(self.covariance_matrix_)
                self.sparse_correlation_ = self._covariance_to_correlation(self.covariance_matrix_)
                
                if self.verbose:
                    tprint_warning("⚠️ Used Ledoit-Wolf fallback")
                
                return {
                    'covariance': self.covariance_matrix_,
                    'precision': self.precision_matrix_,
                    'correlation': self.sparse_correlation_
                }
                
            except Exception as e2:
                if self.verbose:
                    tprint_error(f"❌ Fallback also failed: {e2}")
                return {}
    
    def _covariance_to_correlation(self, covariance: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix."""
        d = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(d, d)
        return correlation
    
    def get_sparse_edges(self, threshold: float = 0.1) -> List[Tuple[str, str, float]]:
        """
        Get sparse edges from precision matrix.
        
        Args:
            threshold: Threshold for edge selection
            
        Returns:
            List of tuples (source, target, weight)
        """
        if self.precision_matrix_ is None:
            return []
        
        edges = []
        n_features = self.precision_matrix_.shape[0]
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                weight = abs(self.precision_matrix_[i, j])
                if weight > threshold:
                    edges.append((f"var_{i}", f"var_{j}", weight))
        
        return edges


class CausalDenoisingEngine:
    """
    Advanced causal denoising using structural relationships.
    
    Removes noise from features using causal constraints and
    enhances signal through causal information propagation.
    """
    
    def __init__(
        self,
        causal_graph: Optional[Dict[str, List[str]]] = None,
        scm: Optional[StructuralCausalModel] = None,
        denoising_methods: List[str] = None,
        verbose: bool = True
    ):
        """
        Initialize Causal Denoising Engine.
        
        Args:
            causal_graph: Causal graph from discovery
            scm: Fitted structural causal models
            denoising_methods: List of denoising methods to apply
            verbose: Whether to print progress information
        """
        self.causal_graph = causal_graph or {}
        self.scm = scm
        self.verbose = verbose
        
        # Default denoising methods
        if denoising_methods is None:
            self.denoising_methods = [
                'sparse_covariance',
                'causal_constraints',
                'graph_filtering',
                'causal_imputation',
                'signal_enhancement'
            ]
        else:
            self.denoising_methods = denoising_methods
        
        # Storage for denoising components
        self.sparse_denoiser_ = None
        self.denoised_features_ = {}
        self.denoising_metadata_ = {}
        
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit denoising models and apply all denoising methods.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Denoised feature matrix
        """
        if self.verbose:
            tprint_info("🧹 Causal Denoising Engine: Starting denoising...")
        
        start_time = time.time()
        
        # Initialize SCM if not provided
        if self.scm is None and self.causal_graph:
            self.scm = StructuralCausalModel(verbose=False)
            try:
                self.scm.fit_structural_equations(X, self.causal_graph)
            except Exception as e:
                if self.verbose:
                    tprint_warning(f"⚠️ Failed to fit SCM: {e}")
        
        # Apply denoising methods
        denoised_dfs = []
        
        for method in self.denoising_methods:
            try:
                if method == 'sparse_covariance':
                    denoised_df = self._apply_sparse_covariance_denoising(X)
                elif method == 'causal_constraints':
                    denoised_df = self._apply_causal_constraints_denoising(X)
                elif method == 'graph_filtering':
                    denoised_df = self._apply_graph_filtering(X)
                elif method == 'causal_imputation':
                    denoised_df = self._apply_causal_imputation(X)
                elif method == 'signal_enhancement':
                    denoised_df = self._apply_signal_enhancement(X)
                else:
                    if self.verbose:
                        tprint_warning(f"⚠️ Unknown denoising method: {method}")
                    continue
                
                if denoised_df is not None and not denoised_df.empty:
                    # Add method prefix to column names
                    denoised_df.columns = [f"{method}_{col}" for col in denoised_df.columns]
                    denoised_dfs.append(denoised_df)
                    
                    if self.verbose:
                        tprint_info(f"   ✅ {method}: {denoised_df.shape[1]} features")
                
            except Exception as e:
                if self.verbose:
                    tprint_warning(f"⚠️ Failed {method} denoising: {e}")
                continue
        
        # Combine all denoised features
        if denoised_dfs:
            final_features = pd.concat(denoised_dfs, axis=1)
        else:
            final_features = pd.DataFrame(index=X.index)
        
        denoising_time = time.time() - start_time
        
        if self.verbose:
            tprint_success(f"✅ Causal Denoising: Complete!")
            tprint_info(f"   📊 Original features: {X.shape[1]}")
            tprint_info(f"   📊 Denoised features: {final_features.shape[1]}")
            tprint_info(f"   ⏱️  Time: {denoising_time:.2f}s")
        
        self.denoised_features_ = final_features
        return final_features
    
    def _apply_sparse_covariance_denoising(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply sparse covariance-based denoising.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Denoised features
        """
        # Initialize sparse denoiser
        self.sparse_denoiser_ = SparseCovarianceDenoiser(verbose=False)
        
        # Fit sparse covariance
        cov_results = self.sparse_denoiser_.fit_sparse_covariance(X)
        
        if not cov_results:
            return X.copy()
        
        precision_matrix = cov_results['precision']
        
        # Use precision matrix for denoising
        # Remove correlations that are not supported by precision matrix
        denoised_X = X.copy()
        
        # Apply precision-based filtering
        for i, col1 in enumerate(X.columns):
            for j, col2 in enumerate(X.columns):
                if i >= j:  # Only upper triangle
                    continue
                
                # Check if edge exists in precision matrix
                if abs(precision_matrix[i, j]) < 0.01:  # Threshold for edge
                    # Remove correlation between these variables
                    try:
                        # Simple decorrelation: make one variable orthogonal to the other
                        values1 = X[col1].values
                        values2 = X[col2].values
                        
                        # Remove linear component
                        if np.std(values1) > 1e-8 and np.std(values2) > 1e-8:
                            corr = np.corrcoef(values1, values2)[0, 1]
                            if abs(corr) > 0.3:  # Only decorrelate if highly correlated
                                # Orthogonalize col2 with respect to col1
                                coeff = np.cov(values1, values2)[0, 1] / np.var(values1)
                                denoised_values2 = values2 - coeff * values1
                                denoised_X[col2] = denoised_values2
                    
                    except Exception:
                        continue
        
        return denoised_X
    
    def _apply_causal_constraints_denoising(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply causal constraints for denoising.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Denoised features using causal constraints
        """
        if not self.causal_graph or not self.scm:
            return X.copy()
        
        denoised_X = X.copy()
        
        # Apply structural equation constraints
        for target, parents in self.causal_graph.items():
            if target not in X.columns:
                continue
            
            valid_parents = [p for p in parents if p in X.columns]
            if not valid_parents:
                continue
            
            try:
                # Get fitted SEM for this target
                if target not in self.scm.structural_models_:
                    continue
                
                model = self.scm.structural_models_[target]
                
                # Predict target based on parents (causal component)
                X_parents = X[valid_parents].values
                causal_prediction = model.predict(X_parents)
                
                # Residuals represent noise/non-causal component
                actual_values = X[target].values
                
                # Smooth by reducing non-causal component
                smoothing_factor = 0.7  # Keep 70% of causal component
                denoised_values = smoothing_factor * causal_prediction + (1 - smoothing_factor) * actual_values
                
                denoised_X[target] = denoised_values
                
            except Exception:
                continue
        
        return denoised_X
    
    def _apply_graph_filtering(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply graph-based filtering using causal structure.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Graph-filtered features
        """
        if not self.causal_graph:
            return X.copy()
        
        # Build causal graph
        G = nx.DiGraph()
        for target, parents in self.causal_graph.items():
            for parent in parents:
                if parent in X.columns and target in X.columns:
                    G.add_edge(parent, target)
        
        if not G.nodes():
            return X.copy()
        
        denoised_X = X.copy()
        
        try:
            # Apply graph-based smoothing
            # For each node, smooth with its neighbors
            for node in G.nodes():
                if node not in X.columns:
                    continue
                
                # Get neighbors (parents and children)
                neighbors = list(G.predecessors(node)) + list(G.successors(node))
                valid_neighbors = [n for n in neighbors if n in X.columns]
                
                if not valid_neighbors:
                    continue
                
                try:
                    # Compute weighted average with neighbors
                    node_values = X[node].values
                    neighbor_values = X[valid_neighbors].values
                    
                    # Simple graph smoothing
                    neighbor_mean = np.mean(neighbor_values, axis=1)
                    
                    # Smooth with neighbors
                    alpha = 0.3  # Smoothing parameter
                    smoothed_values = (1 - alpha) * node_values + alpha * neighbor_mean
                    
                    denoised_X[node] = smoothed_values
                
                except Exception:
                    continue
        
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ Graph filtering failed: {e}")
        
        return denoised_X
    
    def _apply_causal_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply causal constraint-based imputation for missing values.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Features with causal imputation
        """
        if not self.causal_graph or not self.scm:
            # Fallback to standard KNN imputation
            imputer = KNNImputer(n_neighbors=5)
            return pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        
        imputed_X = X.copy()
        
        # Impute missing values using causal relationships
        for target, parents in self.causal_graph.items():
            if target not in X.columns:
                continue
            
            valid_parents = [p for p in parents if p in X.columns]
            if not valid_parents:
                continue
            
            try:
                # Get fitted SEM for this target
                if target not in self.scm.structural_models_:
                    continue
                
                model = self.scm.structural_models_[target]
                
                # Find missing values in target
                missing_mask = X[target].isna()
                if not missing_mask.any():
                    continue
                
                # Use parents to impute missing values
                X_parents = X[valid_parents]
                
                # Check if parents have sufficient data
                parent_missing = X_parents[missing_mask].isna().any(axis=1)
                if parent_missing.all():
                    continue
                
                # Predict missing values
                valid_imputation_mask = missing_mask & ~parent_missing
                
                if valid_imputation_mask.any():
                    X_parents_valid = X_parents[valid_imputation_mask]
                    predicted_values = model.predict(X_parents_valid)
                    
                    imputed_X.loc[valid_imputation_mask, target] = predicted_values
                
            except Exception:
                continue
        
        # For remaining missing values, use KNN imputation
        if imputed_X.isna().any().any():
            imputer = KNNImputer(n_neighbors=5)
            imputed_values = imputer.fit_transform(imputed_X)
            imputed_X = pd.DataFrame(imputed_values, columns=X.columns, index=X.index)
        
        return imputed_X
    
    def _apply_signal_enhancement(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply signal enhancement using causal information.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Signal-enhanced features
        """
        if not self.causal_graph or not self.scm:
            return X.copy()
        
        enhanced_X = X.copy()
        
        try:
            # Enhance signals along causal pathways
            for target, parents in self.causal_graph.items():
                if target not in X.columns:
                    continue
                
                valid_parents = [p for p in parents if p in X.columns]
                if not valid_parents:
                    continue
                
                try:
                    # Get fitted SEM for this target
                    if target not in self.scm.structural_models_:
                        continue
                    
                    model = self.scm.structural_models_[target]
                    
                    # Get model diagnostics
                    if target in self.scm.model_diagnostics_:
                        diagnostics = self.scm.model_diagnostics_[target]
                        r2 = diagnostics.get('r2', 0.0)
                        
                        # Enhance signal if model is good
                        if r2 > 0.3:  # Threshold for enhancement
                            # Predict causal component
                            X_parents = X[valid_parents].values
                            causal_component = model.predict(X_parents)
                            
                            # Enhance by amplifying causal component
                            enhancement_factor = 1.0 + r2  # Scale by model quality
                            original_values = X[target].values
                            
                            # Blend original with enhanced causal component
                            enhanced_values = (
                                0.7 * original_values + 
                                0.3 * enhancement_factor * causal_component
                            )
                            
                            enhanced_X[target] = enhanced_values
                
                except Exception:
                    continue
        
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ Signal enhancement failed: {e}")
        
        return enhanced_X
    
    def get_denoising_summary(self) -> Dict[str, Any]:
        """Get summary of applied denoising methods."""
        if not self.denoised_features_:
            return {'error': 'No denoising applied'}
        
        summary = {
            'original_features': len(self.causal_graph) if self.causal_graph else 0,
            'denoised_features': len(self.denoised_features_.columns),
            'denoising_methods': self.denoising_methods,
            'feature_types': {}
        }
        
        # Categorize features by denoising method
        for col in self.denoised_features_.columns:
            for method in self.denoising_methods:
                if col.startswith(f"{method}_"):
                    if method not in summary['feature_types']:
                        summary['feature_types'][method] = []
                    summary['feature_types'][method].append(col)
                    break
        
        # Add sparse covariance information if available
        if self.sparse_denoiser_ and self.sparse_denoiser_.precision_matrix_ is not None:
            precision = self.sparse_denoiser_.precision_matrix_
            n_nonzero = np.count_nonzero(precision)
            total_elements = precision.size
            summary['sparse_covariance'] = {
                'sparsity': 1 - (n_nonzero / total_elements),
                'non_zero_elements': n_nonzero,
                'total_elements': total_elements
            }
        
        return summary


# Convenience function for quick usage
def denoise_causal_features(
    X: pd.DataFrame,
    causal_graph: Dict[str, List[str]],
    denoising_methods: List[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Quick function for causal feature denoising.
    
    Args:
        X: Input feature matrix
        causal_graph: Causal graph from discovery
        denoising_methods: List of denoising methods
        verbose: Whether to print progress information
        
    Returns:
        Denoised feature matrix
    """
    denoiser = CausalDenoisingEngine(
        causal_graph=causal_graph,
        denoising_methods=denoising_methods,
        verbose=verbose
    )
    
    return denoiser.fit_transform(X)
