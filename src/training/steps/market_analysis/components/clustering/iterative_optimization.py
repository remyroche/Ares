"""
Iterative Optimization for NAS-TAS Clustering.

This module handles the iterative optimization loop that includes:
- Cluster splitting decisions
- Iterative convergence
- Neighborhood analysis
- Sample reallocation
- Regime balance optimization
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error
)

from ..shared_utils import get_logger


class IterativeOptimization:
    """Iterative optimization loop for clustering refinement."""
    
    def __init__(self, verbose: bool = True):
        """Initialize the iterative optimization."""
        self.verbose = verbose
        self.logger = get_logger('IterativeOptimization')
        self.max_iterations = 100
        self.tolerance = 1e-5
        
    async def execute_optimization_loop(
        self, 
        context: ClusteringContext, 
        config: Any, 
        max_iterations: int = 100
    ) -> ClusteringContext:
        """Execute the iterative optimization loop."""
        try:
            tprint("Starting iterative optimization loop...", "INFO")
            
            current_assignments = context.initial_assignments.copy()
            current_k = len(np.unique(current_assignments))
            
            for iteration in range(max_iterations):
                tprint(f"Iteration {iteration + 1}/{max_iterations}", "INFO")
                
                # Step 1: Cluster splitting (if needed)
                current_assignments, current_k = await self._apply_cluster_splitting(
                    current_assignments, context.optimized_features, current_k, iteration
                )
                
                # Step 2: Iterative convergence
                current_assignments = await self._run_iterative_convergence(
                    context.optimized_features, current_assignments, current_k
                )
                
                # Step 3: Neighborhood analysis
                neighborhood_results = await self._perform_neighborhood_analysis(
                    context.optimized_features, current_assignments
                )
                
                # Step 4: Sample reallocation
                current_assignments = await self._perform_sample_reallocation(
                    context.optimized_features, current_assignments, neighborhood_results
                )
                
                # Step 5: Regime balance optimization
                current_assignments = await self._optimize_regime_balance(
                    context.optimized_features, current_assignments, current_k
                )
                
                # Check convergence
                if self._check_convergence(current_assignments, iteration):
                    tprint(f"Convergence reached at iteration {iteration + 1}", "SUCCESS")
                    break
            
            context.optimized_assignments = current_assignments
            context.final_k = len(np.unique(current_assignments))
            
            tprint("Iterative optimization loop completed", "SUCCESS")
            return context
            
        except Exception as e:
            tprint(f"Iterative optimization failed: {e}", "ERROR")
            raise ValueError(f"Iterative optimization failed: {e}")
    
    async def _apply_cluster_splitting(
        self, 
        assignments: np.ndarray, 
        features: np.ndarray, 
        current_k: int, 
        iteration: int
    ) -> Tuple[np.ndarray, int]:
        """Apply smart cluster splitting decisions."""
        try:
            # Analyze cluster quality
            cluster_metrics = self._analyze_cluster_quality(assignments, features)
            
            # Check if splitting is needed
            if self._should_split_clusters(cluster_metrics, iteration):
                tprint("Applying cluster splitting...", "INFO")
                
                # Find clusters to split
                clusters_to_split = self._identify_clusters_for_splitting(
                    assignments, features, cluster_metrics
                )
                
                # Perform splitting
                new_assignments, new_k = self._perform_cluster_splitting(
                    assignments, features, clusters_to_split, current_k
                )
                
                return new_assignments, new_k
            else:
                return assignments, current_k
                
        except Exception as e:
            tprint(f"Cluster splitting failed: {e}", "ERROR")
            return assignments, current_k
    
    async def _run_iterative_convergence(
        self, 
        features: np.ndarray, 
        assignments: np.ndarray, 
        k: int
    ) -> np.ndarray:
        """Run enhanced iterative convergence."""
        try:
            tprint("Running iterative convergence...", "INFO")
            
            current_assignments = assignments.copy()
            max_iterations = 50
            tolerance = 1e-4
            
            for iteration in range(max_iterations):
                # Calculate current objective
                current_score = self._calculate_composite_score(features, current_assignments)
                
                # Try to improve assignments
                improved_assignments = self._improve_assignments(
                    features, current_assignments, k
                )
                
                # Calculate new objective
                new_score = self._calculate_composite_score(features, improved_assignments)
                
                # Check for improvement
                if new_score > current_score + tolerance:
                    current_assignments = improved_assignments
                    tprint(f"Convergence iteration {iteration + 1}: score improved to {new_score:.4f}", "INFO")
                else:
                    tprint(f"Convergence reached at iteration {iteration + 1}", "SUCCESS")
                    break
            
            return current_assignments
            
        except Exception as e:
            tprint(f"Iterative convergence failed: {e}", "ERROR")
            return assignments
    
    async def _perform_neighborhood_analysis(
        self, 
        features: np.ndarray, 
        assignments: np.ndarray
    ) -> Dict[str, Any]:
        """Perform neighborhood analysis for local structure insights."""
        try:
            tprint("Performing neighborhood analysis...", "INFO")
            
            k = min(15, len(features) - 1)
            nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
            nn.fit(features)
            distances, indices = nn.kneighbors(features)
            
            # Analyze neighborhood consistency
            consistency_scores = []
            misclustered_points = []
            
            for i in range(len(features)):
                # Get neighbor assignments (excluding self)
                neighbor_assignments = assignments[indices[i][1:]]
                
                # Check consistency
                unique, counts = np.unique(neighbor_assignments, return_counts=True)
                majority_cluster = unique[np.argmax(counts)]
                majority_count = counts[np.argmax(counts)]
                
                consistency_score = majority_count / k
                consistency_scores.append(consistency_score)
                
                # Identify misclustered points
                if consistency_score < 0.6 and assignments[i] != majority_cluster:
                    misclustered_points.append(i)
            
            results = {
                'consistency_scores': consistency_scores,
                'misclustered_points': misclustered_points,
                'overall_consistency': np.mean(consistency_scores),
                'k_used': k
            }
            
            tprint(f"Neighborhood analysis: {len(misclustered_points)} misclustered points", "INFO")
            return results
            
        except Exception as e:
            tprint(f"Neighborhood analysis failed: {e}", "ERROR")
            return {'error': str(e)}
    
    async def _perform_sample_reallocation(
        self, 
        features: np.ndarray, 
        assignments: np.ndarray, 
        neighborhood_results: Dict[str, Any]
    ) -> np.ndarray:
        """Perform intelligent sample reallocation."""
        try:
            tprint("Performing sample reallocation...", "INFO")
            
            new_assignments = assignments.copy()
            misclustered_points = neighborhood_results.get('misclustered_points', [])
            
            if not misclustered_points:
                tprint("No misclustered points found for reallocation", "INFO")
                return assignments
            
            # Reallocate misclustered points
            for point_idx in misclustered_points:
                # Find best target cluster
                target_cluster = self._find_best_target_cluster(
                    features, assignments, point_idx
                )
                
                if target_cluster is not None:
                    new_assignments[point_idx] = target_cluster
            
            tprint(f"Reallocated {len(misclustered_points)} samples", "SUCCESS")
            return new_assignments
            
        except Exception as e:
            tprint(f"Sample reallocation failed: {e}", "ERROR")
            return assignments
    
    async def _optimize_regime_balance(
        self, 
        features: np.ndarray, 
        assignments: np.ndarray, 
        k: int
    ) -> np.ndarray:
        """Optimize regime balance."""
        try:
            tprint("Optimizing regime balance...", "INFO")
            
            # Calculate current balance
            current_balance = self._calculate_regime_balance(assignments)
            
            # Try to improve balance
            improved_assignments = self._improve_regime_balance(
                features, assignments, k
            )
            
            # Check if improvement was made
            new_balance = self._calculate_regime_balance(improved_assignments)
            if new_balance > current_balance:
                tprint(f"Regime balance improved: {current_balance:.3f} -> {new_balance:.3f}", "SUCCESS")
                return improved_assignments
            else:
                return assignments
            
        except Exception as e:
            tprint(f"Regime balance optimization failed: {e}", "ERROR")
            return assignments
    
    def _analyze_cluster_quality(self, assignments: np.ndarray, features: np.ndarray) -> Dict[str, Any]:
        """Analyze cluster quality metrics."""
        try:
            unique_clusters = np.unique(assignments)
            cluster_stats = {}
            
            for cluster in unique_clusters:
                cluster_mask = assignments == cluster
                cluster_features = features[cluster_mask]
                
                cluster_stats[cluster] = {
                    'size': len(cluster_features),
                    'mean_intra_distance': np.mean([
                        np.linalg.norm(cluster_features - centroid)
                        for centroid in cluster_features
                    ]),
                    'silhouette_score': silhouette_score(features, assignments)
                }
            
            return cluster_stats
            
        except Exception as e:
            tprint(f"Cluster quality analysis failed: {e}", "ERROR")
            return {}
    
    def _should_split_clusters(self, cluster_metrics: Dict[str, Any], iteration: int) -> bool:
        """Determine if clusters should be split."""
        try:
            # Simple heuristic: split if we have too few clusters and it's early in the process
            if iteration < 10 and len(cluster_metrics) < 8:
                return True
            return False
        except Exception:
            return False
    
    def _identify_clusters_for_splitting(
        self, 
        assignments: np.ndarray, 
        features: np.ndarray, 
        cluster_metrics: Dict[str, Any]
    ) -> List[int]:
        """Identify clusters that should be split."""
        try:
            clusters_to_split = []
            
            for cluster, stats in cluster_metrics.items():
                # Split large clusters with poor quality
                if stats['size'] > 50 and stats['silhouette_score'] < 0.3:
                    clusters_to_split.append(cluster)
            
            return clusters_to_split
            
        except Exception as e:
            tprint(f"Cluster identification failed: {e}", "ERROR")
            return []
    
    def _perform_cluster_splitting(
        self, 
        assignments: np.ndarray, 
        features: np.ndarray, 
        clusters_to_split: List[int], 
        current_k: int
    ) -> Tuple[np.ndarray, int]:
        """Perform actual cluster splitting."""
        try:
            new_assignments = assignments.copy()
            new_k = current_k
            
            for cluster in clusters_to_split:
                # Find samples in this cluster
                cluster_mask = assignments == cluster
                cluster_features = features[cluster_mask]
                cluster_indices = np.where(cluster_mask)[0]
                
                if len(cluster_features) > 10:  # Only split if cluster is large enough
                    # Use K-means to split the cluster
                    kmeans = KMeans(n_clusters=2, random_state=42)
                    sub_assignments = kmeans.fit_predict(cluster_features)
                    
                    # Update assignments
                    for i, idx in enumerate(cluster_indices):
                        if sub_assignments[i] == 1:
                            new_assignments[idx] = new_k
                            new_k += 1
            
            return new_assignments, new_k
            
        except Exception as e:
            tprint(f"Cluster splitting failed: {e}", "ERROR")
            return assignments, current_k
    
    def _improve_assignments(self, features: np.ndarray, assignments: np.ndarray, k: int) -> np.ndarray:
        """Try to improve assignments by reassigning samples."""
        try:
            # Simple improvement: reassign samples to nearest centroid
            centroids = self._calculate_centroids(features, assignments, k)
            
            new_assignments = assignments.copy()
            for i in range(len(features)):
                distances = [np.linalg.norm(features[i] - centroid) for centroid in centroids]
                new_assignments[i] = np.argmin(distances)
            
            return new_assignments
            
        except Exception as e:
            tprint(f"Assignment improvement failed: {e}", "ERROR")
            return assignments
    
    def _find_best_target_cluster(
        self, 
        features: np.ndarray, 
        assignments: np.ndarray, 
        point_idx: int
    ) -> Optional[int]:
        """Find the best target cluster for a point."""
        try:
            point_features = features[point_idx]
            unique_clusters = np.unique(assignments)
            
            best_cluster = None
            best_distance = float('inf')
            
            for cluster in unique_clusters:
                if cluster != assignments[point_idx]:
                    cluster_mask = assignments == cluster
                    cluster_features = features[cluster_mask]
                    
                    if len(cluster_features) > 0:
                        centroid = np.mean(cluster_features, axis=0)
                        distance = np.linalg.norm(point_features - centroid)
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_cluster = cluster
            
            return best_cluster
            
        except Exception as e:
            tprint(f"Target cluster finding failed: {e}", "ERROR")
            return None
    
    def _improve_regime_balance(self, features: np.ndarray, assignments: np.ndarray, k: int) -> np.ndarray:
        """Try to improve regime balance."""
        try:
            # Simple balance improvement: reassign samples to balance cluster sizes
            cluster_sizes = [np.sum(assignments == i) for i in range(k)]
            target_size = len(assignments) // k
            
            new_assignments = assignments.copy()
            
            # Reassign samples from oversized clusters to undersized ones
            for i in range(k):
                if cluster_sizes[i] > target_size * 1.5:  # Oversized cluster
                    cluster_mask = assignments == i
                    cluster_indices = np.where(cluster_mask)[0]
                    
                    # Find undersized clusters
                    undersized_clusters = [j for j in range(k) if cluster_sizes[j] < target_size * 0.8]
                    
                    if undersized_clusters:
                        # Move some samples to undersized clusters
                        n_to_move = min(len(cluster_indices) // 2, len(undersized_clusters))
                        for j in range(n_to_move):
                            new_assignments[cluster_indices[j]] = undersized_clusters[j % len(undersized_clusters)]
            
            return new_assignments
            
        except Exception as e:
            tprint(f"Regime balance improvement failed: {e}", "ERROR")
            return assignments
    
    def _calculate_composite_score(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate composite clustering score."""
        try:
            if len(np.unique(assignments)) < 2:
                return 0.0
            
            # Calculate silhouette score
            silhouette = silhouette_score(features, assignments)
            
            # Calculate Davies-Bouldin score (inverted)
            dbi = davies_bouldin_score(features, assignments)
            dbi_normalized = max(0, 1 - (dbi / 5.0))  # Normalize to 0-1 range
            
            # Composite score
            composite_score = 0.7 * silhouette + 0.3 * dbi_normalized
            
            return composite_score
            
        except Exception as e:
            tprint(f"Composite score calculation failed: {e}", "ERROR")
            return 0.0
    
    def _calculate_centroids(self, features: np.ndarray, assignments: np.ndarray, k: int) -> List[np.ndarray]:
        """Calculate cluster centroids."""
        try:
            centroids = []
            for i in range(k):
                cluster_mask = assignments == i
                if np.any(cluster_mask):
                    centroids.append(np.mean(features[cluster_mask], axis=0))
                else:
                    centroids.append(np.zeros(features.shape[1]))
            return centroids
        except Exception as e:
            tprint(f"Centroid calculation failed: {e}", "ERROR")
            return [np.zeros(features.shape[1]) for _ in range(k)]
    
    def _calculate_regime_balance(self, assignments: np.ndarray) -> float:
        """Calculate regime balance score."""
        try:
            unique, counts = np.unique(assignments, return_counts=True)
            if len(unique) < 2:
                return 0.0
            
            # Calculate balance as 1 - coefficient of variation
            mean_size = np.mean(counts)
            std_size = np.std(counts)
            cv = std_size / mean_size if mean_size > 0 else 1.0
            
            balance = max(0, 1 - cv)
            return balance
            
        except Exception as e:
            tprint(f"Regime balance calculation failed: {e}", "ERROR")
            return 0.0
    
    def _check_convergence(self, assignments: np.ndarray, iteration: int) -> bool:
        """Check if convergence has been reached."""
        try:
            # Simple convergence check: stop after max iterations or if we have good quality
            if iteration >= self.max_iterations - 1:
                return True
            
            # Additional convergence criteria can be added here
            return False
            
        except Exception:
            return False