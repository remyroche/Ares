from ..standardized_parquet_handler import standardized_parquet_handler
#!/usr/bin/env python3
"""Advanced Dimensionality Reduction for Feature Matrices.

from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

This module provides comprehensive solutions for the curse of dimensionality
in regime discovery, including automated dimensionality reduction, feature
selection, and manifold learning techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, TruncatedSVD, KernelPCA
from sklearn.manifold import TSNE, UMAP, Isomap, LocallyLinearEmbedding
from sklearn.feature_selection import SelectKBest, SelectPercentile, RFE, RFECV
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Import centralized systems
from .step03_imports import get_import_manager, safe_import, check_feature_availability
from .step03_config import Step03Config
from .step03_memory_manager import get_memory_manager, memory_aware_processing

# Import optional dependencies
umap = safe_import('umap')
sklearn = safe_import('sklearn')


@dataclass
class DimensionalityReductionConfig:
    """Configuration for dimensionality reduction."""
    
    # General settings
    max_features: int = 100
    min_features: int = 10
    target_variance_explained: float = 0.95
    correlation_threshold: float = 0.95
    
    # PCA settings
    pca_components: Optional[int] = None
    pca_whiten: bool = True
    
    # ICA settings
    ica_components: Optional[int] = None
    ica_max_iter: int = 1000
    
    # Factor Analysis settings
    fa_components: Optional[int] = None
    fa_max_iter: int = 1000
    
    # Manifold Learning settings
    enable_manifold_learning: bool = True
    tsne_components: int = 2
    tsne_perplexity: float = 30.0
    umap_components: int = 10
    umap_n_neighbors: int = 15
    
    # Feature Selection settings
    feature_selection_method: str = 'auto'  # 'auto', 'univariate', 'recursive', 'embedded'
    univariate_k: int = 50
    recursive_cv_folds: int = 5
    
    # Embedded methods
    lasso_alpha: float = 0.01
    elastic_net_alpha: float = 0.01
    elastic_net_l1_ratio: float = 0.5
    
    # Ensemble feature selection
    enable_ensemble_selection: bool = True
    ensemble_methods: List[str] = None
    @log_all_calls
    
    def __post_init__(self):
        if self.ensemble_methods is None:
            self.ensemble_methods = ['univariate', 'recursive', 'embedded']


class AdvancedDimensionalityReducer:
    """Advanced dimensionality reduction with multiple strategies."""
    @log_important_calls
    
    def __init__(self, config: Step03Config):
        self.config = config
        self.dim_config = DimensionalityReductionConfig()
        self.logger = logging.getLogger('AdvancedDimensionalityReducer')
        self.memory_manager = get_memory_manager(config.memory.__dict__)
        
        # Results storage
        self.reduction_results = {}
        self.feature_importance = {}
        self.selected_features = None
        self.reduced_features = None
    @log_all_calls
        
    def _remove_correlated_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features."""
        try:
            # Calculate correlation matrix
            corr_matrix = features.corr().abs()
            
            # Find pairs of highly correlated features
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.dim_config.correlation_threshold)]
            
            # Remove features with highest average correlation
            if len(to_drop) > 0:
                self.logger.info(f"Removing {len(to_drop)} highly correlated features")
                features_cleaned = features.drop(columns = to_drop)
            else:
                features_cleaned = features.copy()
            
            return features_cleaned
            
        except Exception as e:
            self.logger.warning(f"Correlation removal failed: {e}")
            return features
    @log_all_calls
    
    def _univariate_feature_selection(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Univariate feature selection using statistical tests."""
        try:
            # Use mutual information for better performance with non-linear relationships
            selector = SelectKBest(score_func = mutual_info_classif, k = min(self.dim_config.univariate_k, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            
            # Store feature importance
            self.feature_importance['univariate'] = selector.scores_
            
            self.logger.info(f"Univariate selection: {X.shape[1]} -> {X_selected.shape[1]} features")
            return X_selected, selector.get_support(indices = True)
            
        except Exception as e:
            self.logger.warning(f"Univariate feature selection failed: {e}")
            return X, np.arange(X.shape[1])
    @log_all_calls
    
    def _recursive_feature_elimination(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Recursive feature elimination with cross-validation."""
        try:
            # Use Random Forest as base estimator
            estimator = RandomForestClassifier(n_estimators = 50, random_state = 42, n_jobs=-1)
            
            # RFE with cross-validation
            selector = RFECV(
                estimator = estimator,
                step = 1,
                cv = self.dim_config.recursive_cv_folds,
                scoring='f1',
                min_features_to_select = self.dim_config.min_features,
                n_jobs=-1
            )
            
            X_selected = selector.fit_transform(X, y)
            
            # Store feature importance
            self.feature_importance['recursive'] = selector.ranking_
            
            self.logger.info(f"Recursive elimination: {X.shape[1]} -> {X_selected.shape[1]} features")
            return X_selected, selector.get_support(indices = True)
            
        except Exception as e:
            self.logger.warning(f"Recursive feature elimination failed: {e}")
            return X, np.arange(X.shape[1])
    @log_all_calls
    
    def _embedded_feature_selection(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Embedded feature selection using L1 regularization."""
        try:
            # Use ElasticNet for feature selection
            selector = ElasticNetCV(
                l1_ratio = self.dim_config.elastic_net_l1_ratio,
                cv = 5,
                random_state = 42,
                n_jobs=-1
            )
            
            selector.fit(X, y)
            
            # Select features with non-zero coefficients
            feature_mask = np.abs(selector.coef_) > 1e-6
            X_selected = X[:, feature_mask]
            
            # Store feature importance
            self.feature_importance['embedded'] = np.abs(selector.coef_)
            
            self.logger.info(f"Embedded selection: {X.shape[1]} -> {X_selected.shape[1]} features")
            return X_selected, np.where(feature_mask)[0]
            
        except Exception as e:
            self.logger.warning(f"Embedded feature selection failed: {e}")
            return X, np.arange(X.shape[1])
    @log_all_calls
    
    def _ensemble_feature_selection(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Ensemble feature selection combining multiple methods."""
        try:
            feature_scores = np.zeros(X.shape[1])
            method_weights = {'univariate': 0.4, 'recursive': 0.3, 'embedded': 0.3}
            
            # Univariate selection
            if 'univariate' in self.dim_config.ensemble_methods:
                try:
                    _, univariate_indices = self._univariate_feature_selection(X, y)
                    feature_scores[univariate_indices] += method_weights['univariate']
                except:
                    pass
            
            # Recursive elimination
            if 'recursive' in self.dim_config.ensemble_methods:
                try:
                    _, recursive_indices = self._recursive_feature_elimination(X, y)
                    feature_scores[recursive_indices] += method_weights['recursive']
                except:
                    pass
            
            # Embedded selection
            if 'embedded' in self.dim_config.ensemble_methods:
                try:
                    _, embedded_indices = self._embedded_feature_selection(X, y)
                    feature_scores[embedded_indices] += method_weights['embedded']
                except:
                    pass
            
            # Select top features
            n_features = min(self.dim_config.max_features, X.shape[1])
            top_features = np.argsort(feature_scores)[-n_features:]
            X_selected = X[:, top_features]
            
            self.logger.info(f"Ensemble selection: {X.shape[1]} -> {X_selected.shape[1]} features")
            return X_selected, top_features
            
        except Exception as e:
            self.logger.warning(f"Ensemble feature selection failed: {e}")
            return X, np.arange(X.shape[1])
    @log_all_calls
    
    def _linear_dimensionality_reduction(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply linear dimensionality reduction methods."""
        results = {}
        
        try:
            # PCA
            if self.dim_config.pca_components is None:
                # Determine components to explain target variance
                pca_temp = PCA()
                pca_temp.fit(X)
                cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
                n_components = np.argmax(cumsum_variance >= self.dim_config.target_variance_explained) + 1
                n_components = min(n_components, self.dim_config.max_features)
            else:
                n_components = min(self.dim_config.pca_components, X.shape[1])
            
            pca = PCA(n_components = n_components, whiten = self.dim_config.pca_whiten)
            results['pca'] = pca.fit_transform(X)
            
            self.logger.info(f"PCA: {X.shape[1]} -> {results['pca'].shape[1]} components "
                           f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})")
            
        except Exception as e:
            self.logger.warning(f"PCA failed: {e}")
        
        try:
            # FastICA
            if self.dim_config.ica_components is None:
                ica_components = min(50, X.shape[1] // 2)
            else:
                ica_components = min(self.dim_config.ica_components, X.shape[1])
            
            ica = FastICA(n_components = ica_components, max_iter = self.dim_config.ica_max_iter, random_state = 42)
            results['ica'] = ica.fit_transform(X)
            
            self.logger.info(f"FastICA: {X.shape[1]} -> {results['ica'].shape[1]} components")
            
        except Exception as e:
            self.logger.warning(f"FastICA failed: {e}")
        
        try:
            # Factor Analysis
            if self.dim_config.fa_components is None:
                fa_components = min(30, X.shape[1] // 3)
            else:
                fa_components = min(self.dim_config.fa_components, X.shape[1])
            
            fa = FactorAnalysis(n_components = fa_components, max_iter = self.dim_config.fa_max_iter, random_state = 42)
            results['factor_analysis'] = fa.fit_transform(X)
            
            self.logger.info(f"Factor Analysis: {X.shape[1]} -> {results['factor_analysis'].shape[1]} components")
            
        except Exception as e:
            self.logger.warning(f"Factor Analysis failed: {e}")
        
        try:
            # Truncated SVD
            svd_components = min(50, X.shape[1] // 2)
            svd = TruncatedSVD(n_components = svd_components, random_state = 42)
            results['svd'] = svd.fit_transform(X)
            
            self.logger.info(f"Truncated SVD: {X.shape[1]} -> {results['svd'].shape[1]} components")
            
        except Exception as e:
            self.logger.warning(f"Truncated SVD failed: {e}")
        
        return results
    @log_all_calls
    
    def _manifold_learning(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply manifold learning methods."""
        results = {}
        
        if not self.dim_config.enable_manifold_learning:
            return results
        
        try:
            # t-SNE (for visualization and low-dimensional representation)
            if X.shape[0] > 1000:  # Subsample for t-SNE
                indices = np.random.choice(X.shape[0], 1000, replace = False)
                X_subset = X[indices]
            else:
                X_subset = X
            
            tsne = TSNE(
                n_components = self.dim_config.tsne_components,
                perplexity = self.dim_config.tsne_perplexity,
                random_state = 42,
                n_jobs=-1
            )
            results['tsne'] = tsne.fit_transform(X_subset)
            
            self.logger.info(f"t-SNE: {X_subset.shape[1]} -> {results['tsne'].shape[1]} components")
            
        except Exception as e:
            self.logger.warning(f"t-SNE failed: {e}")
        
        try:
            # UMAP (if available)
            if umap and X.shape[0] > 100:
                umap_reducer = umap.UMAP(
                    n_components = self.dim_config.umap_components,
                    n_neighbors = self.dim_config.umap_n_neighbors,
                    random_state = 42,
                    n_jobs=-1
                )
                results['umap'] = umap_reducer.fit_transform(X)
                
                self.logger.info(f"UMAP: {X.shape[1]} -> {results['umap'].shape[1]} components")
            
        except Exception as e:
            self.logger.warning(f"UMAP failed: {e}")
        
        try:
            # Isomap
            if X.shape[0] > 100 and X.shape[1] > 10:
                isomap = Isomap(n_components = min(10, X.shape[1] - 1), n_neighbors = 10, n_jobs=-1)
                results['isomap'] = isomap.fit_transform(X)
                
                self.logger.info(f"Isomap: {X.shape[1]} -> {results['isomap'].shape[1]} components")
            
        except Exception as e:
            self.logger.warning(f"Isomap failed: {e}")
        
        try:
            # Locally Linear Embedding
            if X.shape[0] > 100 and X.shape[1] > 10:
                lle = LocallyLinearEmbedding(
                    n_components = min(10, X.shape[1] - 1),
                    n_neighbors = 10,
                    random_state = 42,
                    n_jobs=-1
                )
                results['lle'] = lle.fit_transform(X)
                
                self.logger.info(f"LLE: {X.shape[1]} -> {results['lle'].shape[1]} components")
            
        except Exception as e:
            self.logger.warning(f"LLE failed: {e}")
        
        return results
    @log_all_calls
    
    def _select_best_reduction_method(self, X: np.ndarray, y: np.ndarray, 
                                    reduction_results: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray]:
        """Select the best dimensionality reduction method based on clustering quality."""
        try:
            best_method = None
            best_score = -np.inf
            best_features = None
            
            for method, features in reduction_results.items():
                if features.shape[1] < 2:
                    continue
                
                try:
                    # Use K-means clustering to evaluate the reduced features
                    from sklearn.cluster import KMeans
                    
                    # Determine optimal number of clusters
                    n_clusters = min(8, max(2, features.shape[0] // 100))
                    kmeans = KMeans(n_clusters = n_clusters, random_state = 42, n_init = 10)
                    labels = kmeans.fit_predict(features)
                    
                    # Calculate silhouette score
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(features, labels)
                        
                        if score > best_score:
                            best_score = score
                            best_method = method
                            best_features = features
                    
                except Exception as e:
                    self.logger.warning(f"Evaluation failed for {method}: {e}")
                    continue
            
            if best_method is None:
                # Fallback to PCA
                best_method = 'pca'
                best_features = reduction_results.get('pca', X)
            
            self.logger.info(f"Best reduction method: {best_method} (silhouette score: {best_score:.3f})")
            return best_method, best_features
            
        except Exception as e:
            self.logger.warning(f"Method selection failed: {e}")
            return 'pca', X
    
    def reduce_dimensionality(self, features: pd.DataFrame, target: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Comprehensive dimensionality reduction pipeline."""
        self.logger.info("🚀 Starting comprehensive dimensionality reduction...")
        
        with memory_aware_processing("dimensionality_reduction", self.config.memory.__dict__):
            # Step 1: Remove correlated features
            self.logger.info("🔗 Removing highly correlated features...")
            features_cleaned = self._remove_correlated_features(features)
            
            # Step 2: Feature selection (if target is provided)
            if target is not None and len(features_cleaned.columns) > self.dim_config.max_features:
                self.logger.info("🎯 Performing feature selection...")
                
                X = features_cleaned.values
                
                if self.dim_config.feature_selection_method == 'auto':
                    if self.dim_config.enable_ensemble_selection:
                        X_selected, selected_indices = self._ensemble_feature_selection(X, target)
                    else:
                        X_selected, selected_indices = self._univariate_feature_selection(X, target)
                elif self.dim_config.feature_selection_method == 'univariate':
                    X_selected, selected_indices = self._univariate_feature_selection(X, target)
                elif self.dim_config.feature_selection_method == 'recursive':
                    X_selected, selected_indices = self._recursive_feature_elimination(X, target)
                elif self.dim_config.feature_selection_method == 'embedded':
                    X_selected, selected_indices = self._embedded_feature_selection(X, target)
                else:
                    X_selected, selected_indices = X, np.arange(X.shape[1])
                
                # Update features with selected ones
                selected_columns = features_cleaned.columns[selected_indices]
                features_cleaned = pd.DataFrame(X_selected, index = features_cleaned.index, columns = selected_columns)
            
            # Step 3: Linear dimensionality reduction
            self.logger.info("📐 Applying linear dimensionality reduction...")
            X = features_cleaned.values
            linear_results = self._linear_dimensionality_reduction(X)
            
            # Step 4: Manifold learning (optional)
            manifold_results = {}
            if self.dim_config.enable_manifold_learning and X.shape[0] > 100:
                self.logger.info("🌊 Applying manifold learning...")
                manifold_results = self._manifold_learning(X)
            
            # Step 5: Select best method
            all_results = {**linear_results, **manifold_results}
            if all_results:
                best_method, best_features = self._select_best_reduction_method(X, target, all_results)
                
                # Create final features DataFrame
                feature_names = [f"{best_method}_component_{i}" for i in range(best_features.shape[1])]
                final_features = pd.DataFrame(best_features, index = features.index, columns = feature_names)
                
                self.reduction_results = {
                    'method': best_method,
                    'original_features': len(features.columns),
                    'final_features': len(final_features.columns),
                    'reduction_ratio': len(final_features.columns) / len(features.columns),
                    'all_results': all_results
                }
                
                self.logger.info(f"✅ Dimensionality reduction completed")
                self.logger.info(f"   Original features: {len(features.columns)}")
                self.logger.info(f"   Final features: {len(final_features.columns)}")
                self.logger.info(f"   Reduction ratio: {self.reduction_results['reduction_ratio']:.3f}")
                self.logger.info(f"   Best method: {best_method}")
                
                return final_features
            else:
                self.logger.warning("All dimensionality reduction methods failed, returning original features")
                return features_cleaned.iloc[:, :self.dim_config.max_features]
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from different selection methods."""
        return self.feature_importance
    
    def get_reduction_summary(self) -> Dict[str, Any]:
        """Get summary of dimensionality reduction results."""
        return self.reduction_results

"""
Advanced Dimensionality Reduction for Feature Matrices.

This module provides comprehensive solutions for the curse of dimensionality
in regime discovery, including automated dimensionality reduction, feature
selection, and manifold learning techniques.
"""