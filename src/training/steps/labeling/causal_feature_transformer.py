"""
Causal Feature Transformation Module

Implements advanced causal feature engineering techniques:
1. Parent-adjusted features using structural equation models
2. Counterfactual feature generation
3. Causal pathway activation features
4. Treatment effect heterogeneity features
5. Causal embedding generation

Key Features:
- Remove parental influence using learned SEMs
- Generate what-if scenarios for key variables
- Create causal pathway features
- Compute heterogeneous treatment effects
- Generate causal embeddings for ML models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import time
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
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


class CausalFeatureTransformer:
    """
    Advanced causal feature transformation using structural equation models.
    
    Transforms features based on their causal relationships to improve
    predictive power and ensure causal consistency.
    """
    
    def __init__(
        self,
        causal_graph: Optional[Dict[str, List[str]]] = None,
        scm: Optional[StructuralCausalModel] = None,
        transformation_methods: List[str] = None,
        verbose: bool = True
    ):
        """
        Initialize Causal Feature Transformer.
        
        Args:
            causal_graph: Causal graph from discovery
            scm: Fitted structural causal models
            transformation_methods: List of transformation methods to apply
            verbose: Whether to print progress information
        """
        self.causal_graph = causal_graph or {}
        self.scm = scm
        self.verbose = verbose
        
        # Default transformation methods
        if transformation_methods is None:
            self.transformation_methods = [
                'parent_adjusted',
                'counterfactual',
                'pathway_activation',
                'treatment_heterogeneity',
                'causal_embeddings'
            ]
        else:
            self.transformation_methods = transformation_methods
        
        # Storage for transformation results
        self.transformed_features_ = {}
        self.transformation_metadata_ = {}
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit transformer and apply all causal transformations.
        
        Args:
            X: Input feature matrix
            y: Target variable (optional, for some transformations)
            
        Returns:
            Transformed feature matrix
        """
        if self.verbose:
            tprint_info("🔄 Causal Feature Transformer: Starting transformations...")
        
        start_time = time.time()
        
        # Validate inputs
        if not self.causal_graph:
            if self.verbose:
                tprint_warning("⚠️ No causal graph provided, skipping transformations")
            return X.copy()
        
        # Initialize SCM if not provided
        if self.scm is None:
            self.scm = StructuralCausalModel(verbose=False)
            try:
                self.scm.fit_structural_equations(X, self.causal_graph)
            except Exception as e:
                if self.verbose:
                    tprint_warning(f"⚠️ Failed to fit SCM: {e}")
                return X.copy()
        
        # Apply transformations
        transformed_dfs = []
        
        for method in self.transformation_methods:
            try:
                if method == 'parent_adjusted':
                    transformed_df = self._compute_parent_adjusted_features(X)
                elif method == 'counterfactual':
                    transformed_df = self._generate_counterfactual_features(X)
                elif method == 'pathway_activation':
                    transformed_df = self._compute_pathway_activation_features(X)
                elif method == 'treatment_heterogeneity':
                    transformed_df = self._compute_treatment_heterogeneity_features(X, y)
                elif method == 'causal_embeddings':
                    transformed_df = self._generate_causal_embeddings(X)
                else:
                    if self.verbose:
                        tprint_warning(f"⚠️ Unknown transformation method: {method}")
                    continue
                
                if transformed_df is not None and not transformed_df.empty:
                    # Add method prefix to column names
                    transformed_df.columns = [f"{method}_{col}" for col in transformed_df.columns]
                    transformed_dfs.append(transformed_df)
                    
                    if self.verbose:
                        tprint_info(f"   ✅ {method}: {transformed_df.shape[1]} features")
                
            except Exception as e:
                if self.verbose:
                    tprint_warning(f"⚠️ Failed {method} transformation: {e}")
                continue
        
        # Combine all transformations
        if transformed_dfs:
            final_features = pd.concat(transformed_dfs, axis=1)
        else:
            final_features = pd.DataFrame(index=X.index)
        
        transformation_time = time.time() - start_time
        
        if self.verbose:
            tprint_success(f"✅ Causal Transformations: Complete!")
            tprint_info(f"   📊 Original features: {X.shape[1]}")
            tprint_info(f"   📊 Transformed features: {final_features.shape[1]}")
            tprint_info(f"   ⏱️  Time: {transformation_time:.2f}s")
        
        self.transformed_features_ = final_features
        return final_features
    
    def _compute_parent_adjusted_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute parent-adjusted features by removing parental influence.
        
        Args:
            X: Input feature matrix
            
        Returns:
            DataFrame with parent-adjusted features
        """
        adjusted_features = {}
        
        for target, parents in self.causal_graph.items():
            if target not in X.columns:
                continue
            
            valid_parents = [p for p in parents if p in X.columns]
            if not valid_parents:
                continue
            
            try:
                # Get fitted model for this target
                if target not in self.scm.structural_models_:
                    continue
                
                model = self.scm.structural_models_[target]
                
                # Predict target based on parents
                X_parents = X[valid_parents].values
                y_pred = model.predict(X_parents)
                
                # Compute residuals (parent-adjusted features)
                y_actual = X[target].values
                residuals = y_actual - y_pred
                
                adjusted_features[f"{target}_parent_adjusted"] = residuals
                
            except Exception:
                continue
        
        return pd.DataFrame(adjusted_features, index=X.index)
    
    def _generate_counterfactual_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate counterfactual features for key causal variables.
        
        Args:
            X: Input feature matrix
            
        Returns:
            DataFrame with counterfactual features
        """
        counterfactual_features = {}
        
        # Identify key variables (high degree nodes)
        node_degrees = {}
        for target, parents in self.causal_graph.items():
            node_degrees[target] = len(parents)
        
        # Select top degree nodes for counterfactuals
        if node_degrees:
            top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            key_variables = [node for node, _ in top_nodes]
        else:
            key_variables = list(X.columns)[:5]
        
        for var in key_variables:
            if var not in X.columns:
                continue
            
            try:
                # Generate counterfactual by setting variable to different values
                original_values = X[var].values
                
                # Counterfactual 1: Set to median
                median_val = np.median(original_values)
                X_cf_median = X.copy()
                X_cf_median[var] = median_val
                
                # Counterfactual 2: Set to mean +/- std
                mean_val = np.mean(original_values)
                std_val = np.std(original_values)
                X_cf_high = X.copy()
                X_cf_high[var] = mean_val + std_val
                X_cf_low = X.copy()
                X_cf_low[var] = mean_val - std_val
                
                # Compute effects on descendants
                descendants = self._get_descendants(var)
                
                for descendant in descendants:
                    if descendant not in X.columns or descendant not in self.scm.structural_models_:
                        continue
                    
                    # Get model for descendant
                    model = self.scm.structural_models_[descendant]
                    parents = self.causal_graph.get(descendant, [])
                    valid_parents = [p for p in parents if p in X.columns]
                    
                    if not valid_parents:
                        continue
                    
                    # Predict under different counterfactuals
                    X_parents_orig = X[valid_parents].values
                    X_parents_median = X_cf_median[valid_parents].values
                    X_parents_high = X_cf_high[valid_parents].values
                    X_parents_low = X_cf_low[valid_parents].values
                    
                    pred_orig = model.predict(X_parents_orig)
                    pred_median = model.predict(X_parents_median)
                    pred_high = model.predict(X_parents_high)
                    pred_low = model.predict(X_parents_low)
                    
                    # Store counterfactual effects
                    counterfactual_features[f"{descendant}_cf_median_{var}"] = pred_median - pred_orig
                    counterfactual_features[f"{descendant}_cf_high_{var}"] = pred_high - pred_orig
                    counterfactual_features[f"{descendant}_cf_low_{var}"] = pred_low - pred_orig
                
            except Exception:
                continue
        
        return pd.DataFrame(counterfactual_features, index=X.index)
    
    def _compute_pathway_activation_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute causal pathway activation features.
        
        Args:
            X: Input feature matrix
            
        Returns:
            DataFrame with pathway activation features
        """
        pathway_features = {}
        
        # Find important pathways (short causal chains)
        pathways = self._find_important_pathways(max_length=3)
        
        for i, pathway in enumerate(pathways):
            try:
                # Compute pathway activation score
                activation_score = self._compute_pathway_activation(X, pathway)
                pathway_features[f"pathway_{i}_activation"] = activation_score
                
                # Compute pathway strength
                pathway_strength = self._compute_pathway_strength(X, pathway)
                pathway_features[f"pathway_{i}_strength"] = pathway_strength
                
            except Exception:
                continue
        
        return pd.DataFrame(pathway_features, index=X.index)
    
    def _compute_treatment_heterogeneity_features(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Compute treatment effect heterogeneity features.
        
        Args:
            X: Input feature matrix
            y: Target variable
            
        Returns:
            DataFrame with treatment heterogeneity features
        """
        heterogeneity_features = {}
        
        if y is None:
            if self.verbose:
                tprint_warning("⚠️ No target variable provided for treatment heterogeneity")
            return pd.DataFrame()
        
        # Identify potential treatments (root nodes)
        root_nodes = self._find_root_nodes()
        
        for treatment in root_nodes[:3]:  # Limit to top 3 treatments
            if treatment not in X.columns:
                continue
            
            try:
                # Estimate heterogeneous treatment effects using causal forest approach
                hete_effects = self._estimate_heterogeneous_effects(X, treatment, y)
                
                for effect_name, effect_values in hete_effects.items():
                    heterogeneity_features[f"{treatment}_{effect_name}"] = effect_values
                
            except Exception:
                continue
        
        return pd.DataFrame(heterogeneity_features, index=X.index)
    
    def _generate_causal_embeddings(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate causal embeddings using structural information.
        
        Args:
            X: Input feature matrix
            
        Returns:
            DataFrame with causal embeddings
        """
        embedding_features = {}
        
        try:
            # Create causal graph for embedding
            G = nx.DiGraph()
            for target, parents in self.causal_graph.items():
                for parent in parents:
                    if parent in X.columns and target in X.columns:
                        G.add_edge(parent, target)
            
            if not G.nodes():
                return pd.DataFrame()
            
            # Compute graph-based features
            # 1. Node centrality features
            centrality_measures = ['degree', 'betweenness', 'closeness', 'pagerank']
            
            for measure in centrality_measures:
                try:
                    if measure == 'degree':
                        centrality = dict(G.degree())
                    elif measure == 'betweenness':
                        centrality = nx.betweenness_centrality(G)
                    elif measure == 'closeness':
                        centrality = nx.closeness_centrality(G)
                    elif measure == 'pagerank':
                        centrality = nx.pagerank(G)
                    
                    # Apply centrality as weights to features
                    for node, centrality_score in centrality.items():
                        if node in X.columns:
                            weighted_feature = X[node] * centrality_score
                            embedding_features[f"{node}_{measure}_weighted"] = weighted_feature
                
                except Exception:
                    continue
            
            # 2. Path-based features
            for node in G.nodes():
                if node not in X.columns:
                    continue
                
                # Shortest path lengths to other nodes
                try:
                    path_lengths = dict(nx.single_source_shortest_path_length(G, node))
                    avg_path_length = np.mean(list(path_lengths.values())) if path_lengths else 0
                    
                    # Create path length weighted feature
                    embedding_features[f"{node}_avg_path_length"] = np.full(len(X), avg_path_length)
                    
                except Exception:
                    continue
            
            # 3. Causal layer features (topological ordering)
            try:
                layers = list(nx.topological_generations(G))
                for i, layer in enumerate(layers):
                    for node in layer:
                        if node in X.columns:
                            embedding_features[f"{node}_causal_layer"] = np.full(len(X), i)
                
            except Exception:
                pass
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ Failed to generate causal embeddings: {e}")
        
        return pd.DataFrame(embedding_features, index=X.index)
    
    def _get_descendants(self, node: str) -> List[str]:
        """Get all descendants of a node in the causal graph."""
        descendants = []
        
        # Build directed graph
        G = nx.DiGraph()
        for target, parents in self.causal_graph.items():
            for parent in parents:
                G.add_edge(parent, target)
        
        try:
            if node in G:
                descendants = list(nx.descendants(G, node))
        except Exception:
            pass
        
        return descendants
    
    def _find_important_pathways(self, max_length: int = 3) -> List[List[str]]:
        """Find important causal pathways in the graph."""
        pathways = []
        
        # Build directed graph
        G = nx.DiGraph()
        for target, parents in self.causal_graph.items():
            for parent in parents:
                if parent in self.causal_graph or target in self.causal_graph:
                    G.add_edge(parent, target)
        
        try:
            # Find all simple paths up to max_length
            for source in G.nodes():
                for target in G.nodes():
                    if source != target:
                        try:
                            paths = list(nx.all_simple_paths(G, source, target, cutoff=max_length))
                            # Take a subset of paths to avoid explosion
                            pathways.extend(paths[:10])
                        except Exception:
                            continue
        except Exception:
            pass
        
        # Return top pathways by length (longer paths might be more important)
        pathways.sort(key=len, reverse=True)
        return pathways[:20]  # Limit to top 20 pathways
    
    def _compute_pathway_activation(self, X: pd.DataFrame, pathway: List[str]) -> np.ndarray:
        """Compute activation score for a causal pathway."""
        if len(pathway) < 2:
            return np.zeros(len(X))
        
        try:
            # Compute product of standardized features along pathway
            activation = np.ones(len(X))
            
            for node in pathway:
                if node in X.columns:
                    node_values = X[node].values
                    # Standardize
                    node_std = (node_values - np.mean(node_values)) / (np.std(node_values) + 1e-8)
                    activation *= node_std
            
            return activation
            
        except Exception:
            return np.zeros(len(X))
    
    def _compute_pathway_strength(self, X: pd.DataFrame, pathway: List[str]) -> np.ndarray:
        """Compute strength score for a causal pathway."""
        if len(pathway) < 2:
            return np.zeros(len(X))
        
        try:
            # Compute average correlation along pathway
            correlations = []
            
            for i in range(len(pathway) - 1):
                node1, node2 = pathway[i], pathway[i + 1]
                
                if node1 in X.columns and node2 in X.columns:
                    corr, _ = stats.pearsonr(X[node1], X[node2])
                    correlations.append(abs(corr))
            
            if correlations:
                avg_correlation = np.mean(correlations)
                return np.full(len(X), avg_correlation)
            else:
                return np.zeros(len(X))
                
        except Exception:
            return np.zeros(len(X))
    
    def _find_root_nodes(self) -> List[str]:
        """Find root nodes (nodes with no parents) in the causal graph."""
        all_nodes = set(self.causal_graph.keys())
        parent_nodes = set()
        
        for parents in self.causal_graph.values():
            parent_nodes.update(parents)
        
        root_nodes = list(all_nodes - parent_nodes)
        return root_nodes
    
    def _estimate_heterogeneous_effects(self, X: pd.DataFrame, treatment: str, y: pd.Series) -> Dict[str, np.ndarray]:
        """Estimate heterogeneous treatment effects using a simple approach."""
        effects = {}
        
        try:
            treatment_values = X[treatment].values
            
            # Simple heterogeneity: treatment * other features interactions
            for feature in X.columns:
                if feature == treatment:
                    continue
                
                feature_values = X[feature].values
                
                # Treatment-feature interaction
                interaction = treatment_values * feature_values
                effects[f"interaction_{feature}"] = interaction
                
                # Conditional treatment effects (simplified)
                # High vs low feature values
                feature_median = np.median(feature_values)
                high_mask = feature_values > feature_median
                low_mask = feature_values <= feature_median
                
                if np.sum(high_mask) > 10 and np.sum(low_mask) > 10:
                    # Treatment effect in high vs low feature groups
                    high_effect = np.corrcoef(treatment_values[high_mask], y[high_mask])[0, 1]
                    low_effect = np.corrcoef(treatment_values[low_mask], y[low_mask])[0, 1]
                    
                    effect_diff = high_effect - low_effect
                    effects[f"effect_diff_{feature}"] = np.full(len(X), effect_diff)
        
        except Exception:
            pass
        
        return effects
    
    def get_transformation_summary(self) -> Dict[str, Any]:
        """Get summary of applied transformations."""
        if not self.transformed_features_:
            return {'error': 'No transformations applied'}
        
        summary = {
            'original_features': len(self.causal_graph) if self.causal_graph else 0,
            'transformed_features': len(self.transformed_features_.columns),
            'transformation_methods': self.transformation_methods,
            'feature_types': {}
        }
        
        # Categorize features by transformation method
        for col in self.transformed_features_.columns:
            for method in self.transformation_methods:
                if col.startswith(f"{method}_"):
                    if method not in summary['feature_types']:
                        summary['feature_types'][method] = []
                    summary['feature_types'][method].append(col)
                    break
        
        return summary


# Convenience function for quick usage
def transform_causal_features(
    X: pd.DataFrame,
    causal_graph: Dict[str, List[str]],
    y: Optional[pd.Series] = None,
    transformation_methods: List[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Quick function for causal feature transformation.
    
    Args:
        X: Input feature matrix
        causal_graph: Causal graph from discovery
        y: Target variable (optional)
        transformation_methods: List of transformation methods
        verbose: Whether to print progress information
        
    Returns:
        Transformed feature matrix
    """
    transformer = CausalFeatureTransformer(
        causal_graph=causal_graph,
        transformation_methods=transformation_methods,
        verbose=verbose
    )
    
    return transformer.fit_transform(X, y)
