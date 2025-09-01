#!/usr/bin/env python3
"""
Enhanced Regime Clustering with Quality-Driven DBSCAN + Bayesian Optimization + Hybrid Refinement

This module implements a sophisticated clustering approach that:
1. Uses DBSCAN with Bayesian optimization to find natural clusters
2. Applies hybrid refinement to reach target clusters while maintaining quality
3. Handles noise points intelligently
4. Provides comprehensive reporting and analysis
"""

import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import itertools
import logging

# Clustering imports
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

# Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False
    logging.warning("scikit-optimize not available, using grid search fallback")

# Visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False
    logging.warning("matplotlib/seaborn not available, skipping visualizations")


class EnhancedRegimeClustering:
    """Enhanced regime clustering with quality-driven optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.results = {}
        self.clustering_history = []
        
        # Quality thresholds
        self.min_quality_threshold = config.get("min_quality_threshold", 0.3)
        self.quality_drop_threshold = config.get("quality_drop_threshold", 0.8)
        self.max_iterations = config.get("max_iterations", 50)
        self.no_improvement_limit = config.get("no_improvement_limit", 10)
        self.min_coverage_threshold = config.get("min_coverage_threshold", 0.98)
        
        # Target clusters
        self.target_clusters = config.get("target_clusters", 20)
        self.min_clusters = config.get("min_clusters", 5)
        self.max_clusters = config.get("max_clusters", 30)
        
        # Bayesian optimization parameters
        self.bayesian_calls = config.get("bayesian_calls", 100)
        self.eps_range = config.get("eps_range", (0.01, 2.0))
        self.min_samples_range = config.get("min_samples_range", (2, 50))
        
    def calculate_composite_score(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive quality metrics and composite score."""
        try:
            # Handle noise points (-1 labels)
            valid_mask = labels != -1
            if sum(valid_mask) < 10:
                return {
                    "composite_score": -1000,
                    "silhouette": -1,
                    "calinski_harabasz": -1,
                    "davies_bouldin": -1,
                    "skew_penalty": 1,
                    "volatility_penalty": 1,
                    "n_clusters": 0,
                    "coverage": 0
                }
            
            valid_labels = labels[valid_mask]
            valid_features = features[valid_mask]
            
            n_clusters = len(set(valid_labels))
            if n_clusters < 2:
                return {
                    "composite_score": -1000,
                    "silhouette": -1,
                    "calinski_harabasz": -1,
                    "davies_bouldin": -1,
                    "skew_penalty": 1,
                    "volatility_penalty": 1,
                    "n_clusters": n_clusters,
                    "coverage": sum(valid_mask) / len(labels)
                }
            
            # Calculate quality metrics
            sil_score = silhouette_score(valid_features, valid_labels)
            cal_score = calinski_harabasz_score(valid_features, valid_labels)
            dav_score = davies_bouldin_score(valid_features, valid_labels)
            
            # Calculate cluster size statistics
            cluster_sizes = [sum(valid_labels == i) for i in set(valid_labels)]
            mean_size = np.mean(cluster_sizes)
            std_size = np.std(cluster_sizes)
            
            # Skew penalty (size variation)
            skew_penalty = std_size / mean_size if mean_size > 0 else 1
            
            # Volatility penalty (small clusters)
            small_clusters = len([s for s in cluster_sizes if s < 5])
            volatility_penalty = small_clusters / len(cluster_sizes)
            
            # Coverage (percentage of non-noise points)
            coverage = sum(valid_mask) / len(labels)
            
            # Composite score
            composite_score = (
                0.4 * sil_score + 
                0.2 * cal_score - 
                0.2 * dav_score - 
                0.1 * skew_penalty - 
                0.1 * volatility_penalty
            )
            
            return {
                "composite_score": composite_score,
                "silhouette": sil_score,
                "calinski_harabasz": cal_score,
                "davies_bouldin": dav_score,
                "skew_penalty": skew_penalty,
                "volatility_penalty": volatility_penalty,
                "n_clusters": n_clusters,
                "coverage": coverage,
                "cluster_sizes": cluster_sizes,
                "mean_cluster_size": mean_size,
                "std_cluster_size": std_size
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating composite score: {e}")
            return {
                "composite_score": -1000,
                "silhouette": -1,
                "calinski_harabasz": -1,
                "davies_bouldin": -1,
                "skew_penalty": 1,
                "volatility_penalty": 1,
                "n_clusters": 0,
                "coverage": 0
            }
    
    def objective_function(self, params: Tuple[float, int]) -> float:
        """Objective function for Bayesian optimization."""
        eps, min_samples = params
        
        try:
            dbscan = DBSCAN(eps=eps, min_samples=int(min_samples))
            labels = dbscan.fit_predict(self.features_scaled)
            
            score_dict = self.calculate_composite_score(self.features_scaled, labels)
            return -score_dict["composite_score"]  # Minimize negative score
            
        except Exception as e:
            self.logger.error(f"Error in objective function: {e}")
            return 1000  # High penalty for errors
    
    def find_optimal_dbscan_params(self, features: np.ndarray) -> Tuple[float, int]:
        """Find optimal DBSCAN parameters using Bayesian optimization."""
        self.features_scaled = features
        self.logger.info("🔍 Starting Bayesian optimization for DBSCAN parameters...")
        
        if not BAYESIAN_OPT_AVAILABLE:
            self.logger.warning("Bayesian optimization not available, using grid search")
            return self._grid_search_dbscan_params(features)
        
        # Define parameter space
        space = [
            Real(self.eps_range[0], self.eps_range[1], name='eps'),
            Integer(self.min_samples_range[0], self.min_samples_range[1], name='min_samples')
        ]
        
        # Run Bayesian optimization
        result = gp_minimize(
            self.objective_function, 
            space, 
            n_calls=self.bayesian_calls, 
            random_state=42,
            verbose=True
        )
        
        best_eps, best_min_samples = result.x
        best_score = -result.fun
        
        self.logger.info(f"✅ Bayesian optimization completed:")
        self.logger.info(f"   Best eps: {best_eps:.4f}")
        self.logger.info(f"   Best min_samples: {int(best_min_samples)}")
        self.logger.info(f"   Best score: {best_score:.4f}")
        
        return best_eps, int(best_min_samples)
    
    def _grid_search_dbscan_params(self, features: np.ndarray) -> Tuple[float, int]:
        """Fallback grid search for DBSCAN parameters."""
        self.logger.info("🔍 Running grid search for DBSCAN parameters...")
        
        best_score = -1000
        best_params = (0.5, 5)
        
        eps_values = np.linspace(self.eps_range[0], self.eps_range[1], 20)
        min_samples_values = range(self.min_samples_range[0], self.min_samples_range[1], 2)
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(features)
                    
                    score_dict = self.calculate_composite_score(features, labels)
                    score = score_dict["composite_score"]
                    
                    if score > best_score:
                        best_score = score
                        best_params = (eps, min_samples)
                        
                except Exception:
                    continue
        
        self.logger.info(f"✅ Grid search completed:")
        self.logger.info(f"   Best eps: {best_params[0]:.4f}")
        self.logger.info(f"   Best min_samples: {best_params[1]}")
        self.logger.info(f"   Best score: {best_score:.4f}")
        
        return best_params
    
    def handle_noise_points(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Intelligently handle noise points from DBSCAN."""
        noise_mask = labels == -1
        noise_count = sum(noise_mask)
        
        if noise_count == 0:
            return labels
        
        self.logger.info(f"🔧 Handling {noise_count} noise points...")
        
        # Get existing clusters
        existing_clusters = set(labels) - {-1}
        if not existing_clusters:
            return labels
        
        noise_features = features[noise_mask]
        
        # Strategy 1: Try to cluster noise points if there are enough
        if noise_count > 50:
            try:
                # Use K-means on noise points
                n_noise_clusters = min(5, noise_count // 10)
                kmeans = KMeans(n_clusters=n_noise_clusters, random_state=42)
                noise_labels = kmeans.fit_predict(noise_features)
                
                # Assign new cluster IDs
                new_cluster_start = max(existing_clusters) + 1
                noise_labels += new_cluster_start
                
                # Update labels
                new_labels = labels.copy()
                new_labels[noise_mask] = noise_labels
                
                self.logger.info(f"   ✅ Created {n_noise_clusters} clusters from noise points")
                return new_labels
                
            except Exception as e:
                self.logger.warning(f"   ⚠️ Failed to cluster noise points: {e}")
        
        # Strategy 2: Assign noise points to nearest clusters
        self.logger.info("   📍 Assigning noise points to nearest clusters...")
        
        # Find cluster centroids
        centroids = []
        cluster_ids = []
        for cluster_id in existing_clusters:
            cluster_mask = labels == cluster_id
            centroid = np.mean(features[cluster_mask], axis=0)
            centroids.append(centroid)
            cluster_ids.append(cluster_id)
        
        centroids = np.array(centroids)
        
        # Find nearest cluster for each noise point
        new_labels = labels.copy()
        for i, is_noise in enumerate(noise_mask):
            if is_noise:
                distances = np.linalg.norm(centroids - features[i], axis=1)
                nearest_cluster_idx = np.argmin(distances)
                new_labels[i] = cluster_ids[nearest_cluster_idx]
        
        self.logger.info(f"   ✅ Assigned {noise_count} noise points to nearest clusters")
        return new_labels
    
    def hybrid_refinement(self, features: np.ndarray, initial_labels: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply hybrid refinement to reach target clusters while maintaining quality."""
        self.logger.info("🔄 Starting hybrid refinement...")
        
        current_labels = initial_labels.copy()
        current_score_dict = self.calculate_composite_score(features, current_labels)
        current_score = current_score_dict["composite_score"]
        initial_score = current_score
        
        self.logger.info(f"   Initial clusters: {current_score_dict['n_clusters']}")
        self.logger.info(f"   Initial score: {current_score:.4f}")
        
        # Store history
        self.clustering_history = [{
            "iteration": 0,
            "n_clusters": current_score_dict["n_clusters"],
            "score": current_score,
            "action": "initial",
            "details": current_score_dict
        }]
        
        iteration = 0
        no_improvement_count = 0
        best_score = current_score
        
        while (current_score_dict["n_clusters"] != self.target_clusters and 
               iteration < self.max_iterations and 
               no_improvement_count < self.no_improvement_limit):
            
            iteration += 1
            best_change = None
            best_new_score = current_score
            best_action = None
            best_details = None
            
            # Early stopping conditions
            if current_score < initial_score * self.quality_drop_threshold:
                self.logger.warning(f"   ⚠️ Quality dropped below threshold, stopping")
                break
            
            if current_score_dict["coverage"] >= self.min_coverage_threshold:
                self.logger.info(f"   ✅ Coverage threshold reached ({current_score_dict['coverage']:.3f})")
                break
            
            # Try splits
            clusters = set(current_labels)
            for cluster_id in clusters:
                cluster_mask = current_labels == cluster_id
                cluster_size = sum(cluster_mask)
                
                if cluster_size > 20:  # Only split large clusters
                    # Try splitting this cluster
                    cluster_features = features[cluster_mask]
                    
                    try:
                        kmeans = KMeans(n_clusters=2, random_state=42)
                        split_labels = kmeans.fit_predict(cluster_features)
                        
                        # Create new labels
                        new_labels = current_labels.copy()
                        new_cluster_start = max(clusters) + 1
                        new_labels[cluster_mask] = split_labels + new_cluster_start
                        
                        new_score_dict = self.calculate_composite_score(features, new_labels)
                        new_score = new_score_dict["composite_score"]
                        
                        if new_score > best_new_score:
                            best_change = new_labels
                            best_new_score = new_score
                            best_action = f"split_cluster_{cluster_id}"
                            best_details = new_score_dict
                            
                    except Exception as e:
                        self.logger.debug(f"   Failed to split cluster {cluster_id}: {e}")
            
            # Try merges
            for i, j in itertools.combinations(clusters, 2):
                cluster_i_mask = current_labels == i
                cluster_j_mask = current_labels == j
                
                # Only merge small clusters
                if sum(cluster_i_mask) < 10 or sum(cluster_j_mask) < 10:
                    # Check if clusters are similar
                    centroid_i = np.mean(features[cluster_i_mask], axis=0)
                    centroid_j = np.mean(features[cluster_j_mask], axis=0)
                    distance = np.linalg.norm(centroid_i - centroid_j)
                    
                    # Merge if clusters are close
                    if distance < np.std(features) * 0.5:
                        new_labels = current_labels.copy()
                        new_labels[cluster_j_mask] = i
                        
                        new_score_dict = self.calculate_composite_score(features, new_labels)
                        new_score = new_score_dict["composite_score"]
                        
                        if new_score > best_new_score:
                            best_change = new_labels
                            best_new_score = new_score
                            best_action = f"merge_clusters_{i}_{j}"
                            best_details = new_score_dict
            
            # Apply best change if found
            if best_change is not None:
                current_labels = best_change
                current_score = best_new_score
                current_score_dict = best_details
                
                if current_score > best_score:
                    best_score = current_score
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                self.logger.info(f"   Iteration {iteration}: {best_action}")
                self.logger.info(f"     Clusters: {current_score_dict['n_clusters']}, Score: {current_score:.4f}")
                
                # Store history
                self.clustering_history.append({
                    "iteration": iteration,
                    "n_clusters": current_score_dict["n_clusters"],
                    "score": current_score,
                    "action": best_action,
                    "details": current_score_dict
                })
            else:
                no_improvement_count += 1
                self.logger.info(f"   Iteration {iteration}: No improvement found")
        
        # Final assessment
        final_score_dict = self.calculate_composite_score(features, current_labels)
        
        self.logger.info(f"✅ Hybrid refinement completed:")
        self.logger.info(f"   Final clusters: {final_score_dict['n_clusters']}")
        self.logger.info(f"   Final score: {final_score_dict['composite_score']:.4f}")
        self.logger.info(f"   Coverage: {final_score_dict['coverage']:.3f}")
        self.logger.info(f"   Iterations: {iteration}")
        
        return current_labels, {
            "initial_score": initial_score,
            "final_score": final_score_dict["composite_score"],
            "initial_clusters": self.clustering_history[0]["n_clusters"],
            "final_clusters": final_score_dict["n_clusters"],
            "iterations": iteration,
            "improvements": len(self.clustering_history) - 1,
            "coverage": final_score_dict["coverage"],
            "quality_improvement": final_score_dict["composite_score"] - initial_score
        }
    
    def analyze_cluster_characteristics(self, features: np.ndarray, labels: np.ndarray, 
                                      feature_names: List[str]) -> Dict[str, Any]:
        """Analyze characteristics of each cluster."""
        self.logger.info("🔍 Analyzing cluster characteristics...")
        
        analysis = {
            "clusters": {},
            "overall_statistics": {},
            "feature_importance": {}
        }
        
        # Calculate overall statistics
        valid_mask = labels != -1
        valid_labels = labels[valid_mask]
        valid_features = features[valid_mask]
        
        unique_clusters = sorted(set(valid_labels))
        
        # Analyze each cluster
        for cluster_id in unique_clusters:
            cluster_mask = valid_labels == cluster_id
            cluster_features = valid_features[cluster_mask]
            
            cluster_stats = {
                "size": len(cluster_features),
                "percentage": len(cluster_features) / len(valid_features) * 100,
                "features": {}
            }
            
            # Analyze each feature
            for i, feature_name in enumerate(feature_names):
                feature_values = cluster_features[:, i]
                overall_values = valid_features[:, i]
                
                cluster_stats["features"][feature_name] = {
                    "mean": float(np.mean(feature_values)),
                    "std": float(np.std(feature_values)),
                    "min": float(np.min(feature_values)),
                    "max": float(np.max(feature_values)),
                    "percentile_25": float(np.percentile(feature_values, 25)),
                    "percentile_75": float(np.percentile(feature_values, 75)),
                    "z_score_vs_overall": float((np.mean(feature_values) - np.mean(overall_values)) / np.std(overall_values))
                }
            
            analysis["clusters"][f"cluster_{cluster_id}"] = cluster_stats
        
        # Calculate feature importance across clusters
        for i, feature_name in enumerate(feature_names):
            feature_values = valid_features[:, i]
            
            # Calculate feature variance across clusters
            cluster_means = []
            for cluster_id in unique_clusters:
                cluster_mask = valid_labels == cluster_id
                cluster_mean = np.mean(feature_values[cluster_mask])
                cluster_means.append(cluster_mean)
            
            # Feature importance based on inter-cluster variance
            feature_importance = np.var(cluster_means) / np.var(feature_values)
            
            analysis["feature_importance"][feature_name] = {
                "importance": float(feature_importance),
                "inter_cluster_variance": float(np.var(cluster_means)),
                "total_variance": float(np.var(feature_values))
            }
        
        # Overall statistics
        analysis["overall_statistics"] = {
            "total_points": len(valid_features),
            "n_clusters": len(unique_clusters),
            "coverage": len(valid_features) / len(features),
            "cluster_size_stats": {
                "mean": float(np.mean([len(valid_features[valid_labels == c]) for c in unique_clusters])),
                "std": float(np.std([len(valid_features[valid_labels == c]) for c in unique_clusters])),
                "min": int(np.min([len(valid_features[valid_labels == c]) for c in unique_clusters])),
                "max": int(np.max([len(valid_features[valid_labels == c]) for c in unique_clusters]))
            }
        }
        
        return analysis
    
    def generate_comprehensive_report(self, features: np.ndarray, labels: np.ndarray, 
                                    feature_names: List[str], refinement_results: Dict[str, Any]) -> str:
        """Generate a comprehensive human-readable report."""
        self.logger.info("📊 Generating comprehensive report...")
        
        # Calculate final metrics
        final_score_dict = self.calculate_composite_score(features, labels)
        cluster_analysis = self.analyze_cluster_characteristics(features, labels, feature_names)
        
        # Create report
        report = []
        report.append("=" * 80)
        report.append("🎯 ENHANCED REGIME CLUSTERING REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("📋 EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"• Target Clusters: {self.target_clusters}")
        report.append(f"• Final Clusters: {final_score_dict['n_clusters']}")
        report.append(f"• Data Points: {len(features):,}")
        report.append(f"• Coverage: {final_score_dict['coverage']:.1%}")
        report.append(f"• Composite Score: {final_score_dict['composite_score']:.4f}")
        report.append(f"• Quality Improvement: {refinement_results['quality_improvement']:.4f}")
        report.append("")
        
        # Clustering Process
        report.append("🔄 CLUSTERING PROCESS")
        report.append("-" * 40)
        report.append(f"• Initial Clusters: {refinement_results['initial_clusters']}")
        report.append(f"• Final Clusters: {refinement_results['final_clusters']}")
        report.append(f"• Iterations: {refinement_results['iterations']}")
        report.append(f"• Improvements: {refinement_results['improvements']}")
        report.append(f"• Initial Score: {refinement_results['initial_score']:.4f}")
        report.append(f"• Final Score: {refinement_results['final_score']:.4f}")
        report.append("")
        
        # Quality Metrics
        report.append("📊 QUALITY METRICS")
        report.append("-" * 40)
        report.append(f"• Silhouette Score: {final_score_dict['silhouette']:.4f}")
        report.append(f"• Calinski-Harabasz Score: {final_score_dict['calinski_harabasz']:.4f}")
        report.append(f"• Davies-Bouldin Score: {final_score_dict['davies_bouldin']:.4f}")
        report.append(f"• Skew Penalty: {final_score_dict['skew_penalty']:.4f}")
        report.append(f"• Volatility Penalty: {final_score_dict['volatility_penalty']:.4f}")
        report.append("")
        
        # Cluster Analysis
        report.append("🎯 CLUSTER ANALYSIS")
        report.append("-" * 40)
        
        # Sort clusters by size
        cluster_sizes = []
        for cluster_id in cluster_analysis["clusters"]:
            size = cluster_analysis["clusters"][cluster_id]["size"]
            cluster_sizes.append((cluster_id, size))
        
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        
        report.append("Top 10 Clusters by Size:")
        for i, (cluster_id, size) in enumerate(cluster_sizes[:10]):
            percentage = cluster_analysis["clusters"][cluster_id]["percentage"]
            report.append(f"  {i+1:2d}. {cluster_id}: {size:,} points ({percentage:.1f}%)")
        
        report.append("")
        
        # Feature Importance
        report.append("🔍 FEATURE IMPORTANCE")
        report.append("-" * 40)
        
        # Sort features by importance
        feature_importance = []
        for feature_name, importance_data in cluster_analysis["feature_importance"].items():
            feature_importance.append((feature_name, importance_data["importance"]))
        
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        report.append("Top Features by Importance:")
        for i, (feature_name, importance) in enumerate(feature_importance[:10]):
            report.append(f"  {i+1:2d}. {feature_name}: {importance:.4f}")
        
        report.append("")
        
        # Cluster Characteristics
        report.append("📈 CLUSTER CHARACTERISTICS")
        report.append("-" * 40)
        
        for cluster_id, size in cluster_sizes[:5]:  # Top 5 clusters
            cluster_data = cluster_analysis["clusters"][cluster_id]
            report.append(f"\n{cluster_id} ({size:,} points, {cluster_data['percentage']:.1f}%):")
            
            # Show top 3 most distinctive features
            feature_scores = []
            for feature_name, feature_data in cluster_data["features"].items():
                z_score = abs(feature_data["z_score_vs_overall"])
                feature_scores.append((feature_name, z_score))
            
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feature_name, z_score) in enumerate(feature_scores[:3]):
                feature_data = cluster_data["features"][feature_name]
                direction = "↑" if feature_data["z_score_vs_overall"] > 0 else "↓"
                report.append(f"  • {feature_name}: {feature_data['mean']:.4f} {direction} (z={z_score:.2f})")
        
        report.append("")
        
        # Iteration History
        if len(self.clustering_history) > 1:
            report.append("🔄 ITERATION HISTORY")
            report.append("-" * 40)
            
            for i, history in enumerate(self.clustering_history[:10]):  # Show first 10 iterations
                report.append(f"  {i:2d}. Clusters: {history['n_clusters']:2d}, Score: {history['score']:.4f}, Action: {history['action']}")
            
            if len(self.clustering_history) > 10:
                report.append(f"  ... and {len(self.clustering_history) - 10} more iterations")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run_enhanced_clustering(self, features: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Run the complete enhanced clustering pipeline."""
        self.logger.info("🚀 Starting enhanced regime clustering pipeline...")
        start_time = time.time()
        
        # Step 1: Find optimal DBSCAN parameters
        self.logger.info("Step 1: Finding optimal DBSCAN parameters...")
        best_eps, best_min_samples = self.find_optimal_dbscan_params(features)
        
        # Step 2: Apply DBSCAN with optimal parameters
        self.logger.info("Step 2: Applying DBSCAN clustering...")
        dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
        initial_labels = dbscan.fit_predict(features)
        
        initial_score_dict = self.calculate_composite_score(features, initial_labels)
        self.logger.info(f"   Initial DBSCAN clusters: {initial_score_dict['n_clusters']}")
        self.logger.info(f"   Initial score: {initial_score_dict['composite_score']:.4f}")
        
        # Step 3: Handle noise points
        self.logger.info("Step 3: Handling noise points...")
        refined_labels = self.handle_noise_points(features, initial_labels)
        
        # Step 4: Hybrid refinement
        self.logger.info("Step 4: Applying hybrid refinement...")
        final_labels, refinement_results = self.hybrid_refinement(features, refined_labels)
        
        # Step 5: Generate comprehensive report
        self.logger.info("Step 5: Generating comprehensive report...")
        report = self.generate_comprehensive_report(features, final_labels, feature_names, refinement_results)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Store results
        self.results = {
            "final_labels": final_labels,
            "final_score_dict": self.calculate_composite_score(features, final_labels),
            "refinement_results": refinement_results,
            "clustering_history": self.clustering_history,
            "execution_time": execution_time,
            "report": report,
            "parameters": {
                "best_eps": best_eps,
                "best_min_samples": best_min_samples,
                "target_clusters": self.target_clusters
            }
        }
        
        self.logger.info(f"✅ Enhanced clustering completed in {execution_time:.2f}s")
        self.logger.info(f"   Final clusters: {self.results['final_score_dict']['n_clusters']}")
        self.logger.info(f"   Final score: {self.results['final_score_dict']['composite_score']:.4f}")
        
        return self.results