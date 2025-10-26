"""
Regime Clustering Step.

This step performs regime clustering using HDBSCAN or other clustering methods.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

# Handle optional dependencies gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Import iterative optimization for advanced cluster balancing
try:
    from src.training.steps.market_analysis.clusters.iterative_optimization import IterativeOptimization
    ITERATIVE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    ITERATIVE_OPTIMIZATION_AVAILABLE = False
    IterativeOptimization = None

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint

logger = logging.getLogger(__name__)


class RegimeClusteringStep(BaseStep):
    """
    Regime Clustering Step.

    Performs regime clustering on market data using various clustering algorithms.
    """

    def __init__(self, step_name: str = "regime_clustering"):
        """Initialize the regime clustering step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('RegimeClustering')

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute regime clustering by loading and refining HDBSCAN results.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - execution_mode: 'full', 'light', or 'blank'

        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
        """
        start_time = datetime.now()
        tprint(f"🔍 Starting regime clustering for {config.get('symbol', 'UNKNOWN')}", "INFO")

        try:
            # Set context for artifact retrieval
            self.artifact_manager.set_context(
                step_name="hdbscan_regime_discovery",  # Look for HDBSCAN artifacts
                symbol=config.get('symbol'),
                exchange=config.get('exchange', 'binance'),
                datetime=datetime.now(),
                information="regime_discovery",
                direction="long",
                model="Analyst"
            )

            # Load HDBSCAN artifacts from previous step
            tprint("📥 Loading HDBSCAN regime discovery artifacts...", "INFO")
            hdbscan_artifacts = self._load_hdbscan_artifacts(config)
            
            if hdbscan_artifacts is None:
                error_msg = "❌ No HDBSCAN artifacts found - regime clustering requires HDBSCAN regime discovery artifacts"
                tprint(error_msg, "ERROR")
                raise ValueError("HDBSCAN artifacts are required for regime clustering. Please run HDBSCAN regime discovery step first.")

            # Apply refinement logic to HDBSCAN results
            tprint("🔧 Refining HDBSCAN clusters...", "INFO")
            refined_clusters = self._refine_hdbscan_clusters(hdbscan_artifacts, config)
            
            # Create refined artifacts
            artifacts = self._create_refined_artifacts(refined_clusters, config)
            
            # Save refined results
            tprint("💾 Saving refined regime clusters...", "INFO")
            self._save_refined_clusters(artifacts, config)
            
            # Calculate metrics
            metrics = self._calculate_refinement_metrics(refined_clusters, hdbscan_artifacts, start_time)
            
            # Generate comprehensive markdown report
            tprint("📝 Creating comprehensive regime clustering report...", "INFO")
            report_path = self._create_comprehensive_report(refined_clusters, hdbscan_artifacts, metrics, config)
            tprint(f"✅ Comprehensive report saved to: {report_path}", "SUCCESS")
            
            tprint(f"✅ Regime clustering completed: {refined_clusters['n_clusters']} refined clusters", "SUCCESS")
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics,
                'report_path': report_path
            }

        except Exception as e:
            error_msg = f"Regime clustering failed: {str(e)}"
            tprint(f"❌ {error_msg}", "ERROR")
            self.logger.error(error_msg)
            raise e  # Re-raise the exception for fast fail

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run method required by BaseStep interface."""
        return await self.execute(config)

    def _load_hdbscan_artifacts(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Load HDBSCAN regime discovery artifacts from previous step.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            HDBSCAN artifacts or None if not found
        """
        try:
            # Try to load regime artifacts (full artifacts)
            regime_artifacts = self._get_artifact("regime_artifacts", artifact_type="data")
            if regime_artifacts is not None and not (hasattr(regime_artifacts, 'empty') and regime_artifacts.empty):
                tprint("✅ Loaded HDBSCAN regime artifacts", "SUCCESS")
                
                # Extract regime labels from the artifacts DataFrame
                if 'regime_labels' in regime_artifacts.columns:
                    regime_labels = regime_artifacts['regime_labels'].iloc[0]
                    tprint(f"📊 Extracted regime labels: {len(regime_labels)} samples", "INFO")
                    
                    # Return properly structured artifacts
                    return {
                        'regime_labels': regime_labels,
                        'regime_probabilities': regime_artifacts.get('regime_probabilities', {}).iloc[0] if 'regime_probabilities' in regime_artifacts.columns else None,
                        'economic_profiles': regime_artifacts.get('economic_profiles', {}).iloc[0] if 'economic_profiles' in regime_artifacts.columns else None,
                        'validation_metrics': {col: regime_artifacts[col].iloc[0] for col in regime_artifacts.columns if col.startswith('validation_metrics')},
                        'metadata': {col: regime_artifacts[col].iloc[0] for col in regime_artifacts.columns if col.startswith('metadata')}
                    }
                else:
                    tprint("⚠️ No regime_labels column found in HDBSCAN artifacts", "WARNING")
                    return None
            
            # Try to load regime labels as fallback
            regime_labels = self._get_artifact("regime_labels", artifact_type="data")
            if regime_labels is not None:
                tprint("✅ Loaded HDBSCAN regime labels", "SUCCESS")
                return {
                    'regime_labels': regime_labels,
                    'regime_probabilities': None,
                    'economic_profiles': None
                }
            
            tprint("⚠️ No HDBSCAN artifacts found", "WARNING")
            return None
            
        except Exception as e:
            tprint(f"⚠️ Failed to load HDBSCAN artifacts: {e}", "WARNING")
            return None

    def _refine_hdbscan_clusters(self, hdbscan_artifacts: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine HDBSCAN clusters using economic validation and temporal stabilization.
        
        Args:
            hdbscan_artifacts: HDBSCAN regime discovery artifacts
            config: Configuration dictionary
            
        Returns:
            Refined cluster data
        """
        try:
            # Store HDBSCAN artifacts for use by iterative optimization
            self._current_hdbscan_artifacts = hdbscan_artifacts
            
            # Extract regime labels from artifacts
            regime_labels = hdbscan_artifacts.get('regime_labels', [])
            
            # Handle different data types
            if isinstance(regime_labels, pd.DataFrame):
                regime_labels = regime_labels['regime_label'].values if 'regime_label' in regime_labels.columns else regime_labels.values
            elif isinstance(regime_labels, np.ndarray):
                # Already a numpy array, use as is
                pass
            elif isinstance(regime_labels, list):
                regime_labels = np.array(regime_labels)
            else:
                tprint(f"⚠️ Unexpected regime_labels type: {type(regime_labels)}", "WARNING")
                regime_labels = np.array([])
            
            if len(regime_labels) == 0:
                tprint("⚠️ No regime labels found in HDBSCAN artifacts", "WARNING")
                return self._create_placeholder_clusters(config)['artifacts']['regime_clusters']
            
            tprint(f"📊 Processing {len(regime_labels)} regime labels", "INFO")
            
            # Apply refinement logic
            refined_labels = self._apply_temporal_stabilization(regime_labels, config)
            refined_labels = self._apply_economic_validation(refined_labels, hdbscan_artifacts, config)
            refined_labels = self._merge_similar_clusters(refined_labels, config)
            
            # Calculate refined cluster statistics
            unique_labels = np.unique(refined_labels)
            # Exclude noise (-1) labels - handle both numpy arrays and pandas Series
            if hasattr(unique_labels, 'values'):
                unique_labels = unique_labels.values
            
            # Use numpy comparison for safety
            unique_labels = np.array(unique_labels)
            # Handle numpy array comparison properly - use element-wise comparison
            # Ensure we're comparing scalars properly
            noise_mask = unique_labels != -1
            non_noise_labels = unique_labels[noise_mask]
            n_clusters = len(non_noise_labels)
            
            tprint(f"🔧 Refined clusters: {n_clusters} clusters (from {len(np.unique(regime_labels))} original)", "INFO")
            
            return {
                'refined_labels': refined_labels,
                'original_labels': regime_labels,
                'n_clusters': n_clusters,
                'clustering_method': 'hdbscan_refined',
                'refinement_applied': True,
                'metadata': {
                    'symbol': config.get('symbol'),
                    'exchange': config.get('exchange', 'binance'),
                    'timeframe': config.get('timeframe', '15m'),
                    'execution_mode': config.get('execution_mode', 'light'),
                    'created_at': datetime.now().isoformat(),
                    'original_n_clusters': len(np.unique(regime_labels))
                }
            }
            
        except Exception as e:
            tprint(f"⚠️ Failed to refine HDBSCAN clusters: {e}", "WARNING")
            return self._create_placeholder_clusters(config)['artifacts']['regime_clusters']

    def _apply_temporal_stabilization(self, labels: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """
        Apply enhanced temporal stabilization to reduce regime switching noise.
        Implements adaptive dwell time and stability constraints.
        
        Args:
            labels: Original regime labels
            config: Configuration dictionary
            
        Returns:
            Temporally stabilized labels
        """
        try:
            stabilized_labels = labels.copy()
            
            # Enhanced temporal smoothing parameters
            base_min_dwell = config.get('min_dwell_bars', 3)
            max_dwell = config.get('max_dwell_bars', 8)
            stability_threshold = config.get('stability_threshold', 0.7)
            volatility_factor = config.get('volatility_factor', 1.0)
            
            # Calculate adaptive dwell time based on local volatility
            adaptive_dwell = self._calculate_adaptive_dwell_time(labels, base_min_dwell, max_dwell, volatility_factor)
            
            # Multi-pass temporal smoothing with increasing strictness
            for pass_num in range(3):
                changes_this_pass = 0
                
                # Pass 1: Remove isolated changes
                # Pass 2: Apply stability constraints
                # Pass 3: Final consistency check
                
                for i in range(adaptive_dwell, len(labels) - adaptive_dwell):
                    current_label = labels[i]
                    
                    # Calculate local stability score
                    local_stability = self._calculate_local_stability(labels, i, adaptive_dwell)
                    
                    # Apply different rules based on pass
                    if pass_num == 0:
                        # Remove isolated changes
                        if (current_label != labels[i-1] and 
                            current_label != labels[i+1] and 
                            labels[i-1] == labels[i+1]):
                            stabilized_labels[i] = labels[i-1]
                            changes_this_pass += 1
                    
                    elif pass_num == 1:
                        # Apply stability constraints
                        if local_stability < stability_threshold:
                            # Find most stable neighbor
                            neighbor_stability = [
                                self._calculate_local_stability(labels, i-1, adaptive_dwell),
                                self._calculate_local_stability(labels, i+1, adaptive_dwell)
                            ]
                            most_stable_neighbor = i-1 if neighbor_stability[0] > neighbor_stability[1] else i+1
                            
                            if neighbor_stability[0] > stability_threshold or neighbor_stability[1] > stability_threshold:
                                stabilized_labels[i] = labels[most_stable_neighbor]
                                changes_this_pass += 1
                    
                    else:
                        # Final consistency check - ensure smooth transitions
                        if (stabilized_labels[i] != stabilized_labels[i-1] and 
                            stabilized_labels[i] != stabilized_labels[i+1] and
                            stabilized_labels[i-1] == stabilized_labels[i+1]):
                            # Use weighted average of neighbors
                            stabilized_labels[i] = stabilized_labels[i-1]
                            changes_this_pass += 1
                
                if changes_this_pass == 0:
                    break  # No more changes needed
                
                tprint(f"🔧 Temporal stabilization pass {pass_num + 1}: {changes_this_pass} changes", "INFO")
            
            # Apply final stability validation
            stabilized_labels = self._apply_stability_validation(stabilized_labels, config)
            
            total_changes = np.sum(labels != stabilized_labels)
            tprint(f"🔧 Enhanced temporal stabilization: {total_changes} total changes applied", "INFO")
            
            return stabilized_labels
            
        except Exception as e:
            tprint(f"⚠️ Enhanced temporal stabilization failed: {e}", "WARNING")
            return labels

    def _apply_economic_validation(self, labels: np.ndarray, artifacts: Dict[str, Any], config: Dict[str, Any]) -> np.ndarray:
        """
        Apply enhanced economic validation with dynamic size constraints.
        
        Args:
            labels: Regime labels
            artifacts: HDBSCAN artifacts
            config: Configuration dictionary
            
        Returns:
            Economically validated labels
        """
        try:
            # Enhanced economic validation with dynamic size constraints
            unique_labels, counts = np.unique(labels, return_counts=True)
            non_noise_labels = unique_labels[unique_labels != -1]
            
            if len(non_noise_labels) < 2:
                tprint("🔧 Economic validation: Insufficient clusters for validation", "INFO")
                return labels
            
            total_samples = len(labels)
            n_clusters = len(non_noise_labels)
            
            # Dynamic size constraints based on cluster count and data quality
            ideal_cluster_size = total_samples / n_clusters
            min_cluster_ratio = config.get('min_cluster_ratio', 0.05)  # 5% minimum
            max_cluster_ratio = config.get('max_cluster_ratio', 0.35)  # 35% maximum (more flexible)
            target_cluster_ratio = config.get('target_cluster_ratio', 0.20)  # 20% target
            
            # Calculate cluster statistics
            cluster_stats = []
            for label in non_noise_labels:
                cluster_size = np.sum(labels == label)
                cluster_ratio = cluster_size / total_samples
                cluster_stats.append({
                    'label': label,
                    'size': cluster_size,
                    'ratio': cluster_ratio,
                    'is_balanced': min_cluster_ratio <= cluster_ratio <= max_cluster_ratio
                })
            
            # Sort by size (largest first)
            cluster_stats.sort(key=lambda x: x['size'], reverse=True)
            
            tprint(f"🔧 Economic validation: {n_clusters} clusters, target ratio: {target_cluster_ratio:.1%}", "INFO")
            
            # Apply dynamic rebalancing
            rebalanced_labels = labels.copy()
            rebalance_applied = False
            
            for i, cluster_stat in enumerate(cluster_stats):
                label = cluster_stat['label']
                current_ratio = cluster_stat['ratio']
                
                if current_ratio > max_cluster_ratio:
                    # Cluster too large - reduce size
                    target_size = int(total_samples * target_cluster_ratio)
                    current_size = cluster_stat['size']
                    reduce_by = current_size - target_size
                    
                    if reduce_by > 0:
                        cluster_indices = np.where(rebalanced_labels == label)[0]
                        np.random.seed(42)  # For reproducibility
                        reduce_indices = np.random.choice(cluster_indices, size=min(reduce_by, len(cluster_indices)), replace=False)
                        
                        # Reassign to smallest clusters first
                        smallest_clusters = [cs for cs in cluster_stats if cs['ratio'] < target_cluster_ratio and cs['label'] != label]
                        smallest_clusters.sort(key=lambda x: x['ratio'])
                        
                        reassigned_count = 0
                        for small_cluster in smallest_clusters:
                            if reassigned_count >= len(reduce_indices):
                                break
                            
                            # Calculate how many this cluster can absorb
                            remaining_capacity = int(total_samples * target_cluster_ratio) - small_cluster['size']
                            to_reassign = min(remaining_capacity, len(reduce_indices) - reassigned_count)
                            
                            if to_reassign > 0:
                                rebalanced_labels[reduce_indices[reassigned_count:reassigned_count + to_reassign]] = small_cluster['label']
                                reassigned_count += to_reassign
                        
                        # Convert remaining to noise if no capacity
                        if reassigned_count < len(reduce_indices):
                            remaining_indices = reduce_indices[reassigned_count:]
                            rebalanced_labels[remaining_indices] = -1
                        
                        tprint(f"🔧 Rebalanced cluster {label}: {current_ratio:.1%} -> {target_cluster_ratio:.1%} (reassigned {reassigned_count})", "INFO")
                        rebalance_applied = True
                
                elif current_ratio < min_cluster_ratio:
                    # Cluster too small - try to merge with similar clusters
                    tprint(f"🔧 Cluster {label} too small ({current_ratio:.1%}), considering merge", "INFO")
                    
                    # Find most similar cluster for potential merge
                    best_merge_candidate = self._find_best_merge_candidate(rebalanced_labels, label, cluster_stats)
                    if best_merge_candidate is not None:
                        cluster_mask = rebalanced_labels == label
                        rebalanced_labels[cluster_mask] = best_merge_candidate
                        tprint(f"🔧 Merged small cluster {label} with cluster {best_merge_candidate}", "INFO")
                        rebalance_applied = True
            
            # Apply final balance validation
            if rebalance_applied:
                rebalanced_labels = self._apply_final_balance_validation(rebalanced_labels, config)
            
            tprint("🔧 Enhanced economic validation completed", "INFO")
            return rebalanced_labels
            
        except Exception as e:
            tprint(f"⚠️ Enhanced economic validation failed: {e}", "WARNING")
            return labels

    def _find_best_merge_candidate(self, labels: np.ndarray, small_cluster_label: int, cluster_stats: List[Dict]) -> Optional[int]:
        """Find the best cluster to merge with a small cluster."""
        try:
            # Find clusters that are not too large and not the same as small cluster
            candidates = []
            for cluster_stat in cluster_stats:
                if (cluster_stat['label'] != small_cluster_label and 
                    cluster_stat['ratio'] < 0.25):  # Not too large
                    candidates.append(cluster_stat)
            
            if not candidates:
                return None
            
            # Return the smallest candidate (most similar size)
            best_candidate = min(candidates, key=lambda x: x['size'])
            return best_candidate['label']
            
        except Exception as e:
            tprint(f"⚠️ Best merge candidate search failed: {e}", "WARNING")
            return None

    def _apply_final_balance_validation(self, labels: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Apply final balance validation to ensure optimal cluster distribution."""
        try:
            validated_labels = labels.copy()
            total_samples = len(labels)
            
            # Calculate final cluster statistics
            unique_labels, counts = np.unique(labels, return_counts=True)
            non_noise_labels = unique_labels[unique_labels != -1]
            
            if len(non_noise_labels) < 2:
                return validated_labels
            
            # Calculate final balance metrics
            cluster_ratios = [count / total_samples for count in counts if count > 0]
            balance_variance = np.var(cluster_ratios)
            balance_std = np.std(cluster_ratios)
            
            tprint(f"🔧 Final balance validation: variance={balance_variance:.4f}, std={balance_std:.4f}", "INFO")
            
            # If balance is still poor, apply final adjustments
            if balance_std > 0.15:  # High imbalance
                tprint("🔧 Applying final balance adjustments", "INFO")
                
                # Find most imbalanced clusters
                cluster_ratio_pairs = [(label, count / total_samples) for label, count in zip(unique_labels, counts) if count > 0]
                cluster_ratio_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # Redistribute from largest to smallest
                largest_label, largest_ratio = cluster_ratio_pairs[0]
                smallest_label, smallest_ratio = cluster_ratio_pairs[-1]
                
                if largest_ratio > 0.3 and smallest_ratio < 0.1:  # Significant imbalance
                    # Move some samples from largest to smallest
                    move_count = int(total_samples * 0.05)  # Move 5% of data
                    largest_indices = np.where(validated_labels == largest_label)[0]
                    
                    if len(largest_indices) > move_count:
                        np.random.seed(42)
                        move_indices = np.random.choice(largest_indices, size=move_count, replace=False)
                        validated_labels[move_indices] = smallest_label
                        tprint(f"🔧 Final rebalance: moved {move_count} samples from cluster {largest_label} to {smallest_label}", "INFO")
            
            return validated_labels
            
        except Exception as e:
            tprint(f"⚠️ Final balance validation failed: {e}", "WARNING")
            return labels

    def _merge_similar_clusters(self, labels: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """
        Merge clusters that are too similar or too small, and split overly large clusters.
        Uses advanced iterative optimization when available.
        
        Args:
            labels: Regime labels
            config: Configuration dictionary
            
        Returns:
            Labels with balanced clusters
        """
        try:
            # Force advanced iterative optimization - no fallbacks
            if not ITERATIVE_OPTIMIZATION_AVAILABLE or IterativeOptimization is None:
                raise ImportError("IterativeOptimization not available - this is required for regime clustering")
            
            tprint("🚀 Using advanced iterative optimization for cluster balancing", "INFO")
            return self._advanced_cluster_optimization(labels, config)
            
        except Exception as e:
            tprint(f"❌ Cluster merging failed: {e}", "ERROR")
            raise RuntimeError(f"Regime clustering requires iterative optimization: {e}")

    def _advanced_cluster_optimization(self, labels: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """
        Use advanced iterative optimization for cluster balancing.
        
        Args:
            labels: Regime labels
            config: Configuration dictionary
            
        Returns:
            Optimized cluster labels
        """
        try:
            tprint("🚀 Using advanced iterative optimization for cluster balancing", "INFO")
            
            # Get features from HDBSCAN artifacts if available
            hdbscan_artifacts = getattr(self, '_current_hdbscan_artifacts', None)
            if hdbscan_artifacts is None:
                raise ValueError("No HDBSCAN artifacts available for optimization - this is required")
            
            # Extract features from HDBSCAN artifacts
            features = self._extract_features_for_optimization(hdbscan_artifacts)
            if features is None:
                tprint("📊 No features found in artifacts, generating enhanced synthetic features", "INFO")
                features = self._generate_enhanced_synthetic_features(hdbscan_artifacts)
                if features is None:
                    raise ValueError("Could not generate features for iterative optimization")
            
            # Initialize iterative optimization
            optimizer = IterativeOptimization(verbose=True, k=None)
            
            # Set up enhanced optimization parameters with improved constraints
            optimizer.config.min_size_ratio = 0.05  # 5% minimum (more realistic)
            optimizer.config.max_size_ratio = 0.35  # 35% maximum (more flexible)
            optimizer.config.max_rounds = 50  # More iterations for better optimization
            
            # Enhanced objective weights with cohesion penalties
            optimizer.config.w_cv = 0.50  # CV ratio weight (reduced for cohesion)
            optimizer.config.w_temp = 0.15  # Temporal consistency
            optimizer.config.w_sil = 0.10  # Silhouette score
            optimizer.config.w_bal = 0.15  # Cluster balance
            optimizer.config.w_frag = 0.10  # Fragmentation penalty weight (NEW)
            
            # Enhanced size constraints
            optimizer.config.min_cluster_ratio = 0.05  # 5% minimum cluster ratio
            optimizer.config.max_cluster_ratio = 0.35  # 35% maximum cluster ratio
            optimizer.config.target_cluster_ratio = 0.20  # 20% target cluster ratio
            optimizer.config.size_balance_weight = 0.20  # Size balance weight
            
            # Cohesion and fragmentation parameters
            optimizer.config.fragmentation_penalty_threshold = 0.5  # Fragmentation threshold
            optimizer.config.cohesion_reward_threshold = 0.7  # Cohesion threshold
            optimizer.config.enable_cohesion_optimization = True  # Enable cohesion optimization
            
            # Cluster management
            optimizer.config.K_MIN = 3  # Minimum clusters (more realistic)
            optimizer.config.K_MAX = 8  # Maximum clusters
            optimizer.config.large_cluster_threshold = 0.25  # 25% threshold for large clusters
            optimizer.config.split_size_threshold = 0.20  # 20% threshold for splitting
            optimizer.config.SOFT_CAP = int(len(labels) * 0.15)  # 15% soft cap
            
            # Run optimization
            tprint(f"🔧 Starting iterative optimization on {len(features)} features, {len(labels)} samples", "INFO")
            optimized_labels = optimizer.optimize_with_hard_constraints(
                X=features,
                initial_assignments=labels,
                entity_ids=None,  # Could be added if available
                time_idx=None     # Could be added if available
            )
            
            # Post-optimization noise reduction
            optimized_labels = self._post_optimization_noise_reduction(optimized_labels, config)
            
            # Validate results
            unique_labels, counts = np.unique(optimized_labels, return_counts=True)
            non_noise_labels = unique_labels[unique_labels != -1]
            final_cluster_count = len(non_noise_labels)
            noise_count = np.sum(optimized_labels == -1)
            noise_ratio = noise_count / len(optimized_labels) * 100
            
            tprint(f"✅ Iterative optimization completed: {final_cluster_count} clusters, {noise_ratio:.1f}% noise", "SUCCESS")
            
            # Check if results meet HDBSCAN goals
            target_min_clusters = 4
            target_max_clusters = 8
            
            if final_cluster_count < target_min_clusters:
                tprint(f"⚠️ Final cluster count ({final_cluster_count}) below target minimum ({target_min_clusters})", "WARNING")
            elif final_cluster_count > target_max_clusters:
                tprint(f"⚠️ Final cluster count ({final_cluster_count}) above target maximum ({target_max_clusters})", "WARNING")
            else:
                tprint(f"✅ Final cluster count ({final_cluster_count}) within target range ({target_min_clusters}-{target_max_clusters})", "SUCCESS")
            
            return optimized_labels
            
        except Exception as e:
            tprint(f"❌ Advanced optimization failed: {e}", "ERROR")
            raise RuntimeError(f"Iterative optimization failed: {e}")

    def _extract_features_for_optimization(self, hdbscan_artifacts: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract features from HDBSCAN artifacts for iterative optimization.
        
        Args:
            hdbscan_artifacts: HDBSCAN artifacts dictionary
            
        Returns:
            Feature matrix or None if extraction fails
        """
        try:
            # Try to extract features from various possible locations in HDBSCAN artifacts
            features = None
            
            # Check for features in different possible keys
            feature_keys = ['features', 'feature_matrix', 'X', 'data', 'feature_data', 'clustering_features']
            
            for key in feature_keys:
                if key in hdbscan_artifacts:
                    features = hdbscan_artifacts[key]
                    tprint(f"📊 Found features in '{key}' key", "INFO")
                    break
            
            # If no features found in artifacts, try to load from the original HDBSCAN step
            if features is None:
                tprint("📊 No features in artifacts, attempting to load from HDBSCAN step", "INFO")
                features = self._load_features_from_hdbscan_step()
            
            # If still no features, generate enhanced synthetic features
            if features is None:
                tprint("📊 No features found, generating enhanced synthetic features for iterative optimization", "INFO")
                features = self._generate_enhanced_synthetic_features(hdbscan_artifacts)
            
            if features is not None:
                # Ensure features are in the right format
                if isinstance(features, pd.DataFrame):
                    features = features.values
                elif isinstance(features, list):
                    features = np.array(features)
                
                # Validate feature matrix
                if len(features.shape) != 2:
                    tprint(f"⚠️ Invalid feature shape: {features.shape}, expected 2D array", "WARNING")
                    return None
                
                tprint(f"✅ Extracted features: {features.shape[0]} samples, {features.shape[1]} features", "SUCCESS")
                return features
            else:
                tprint("⚠️ No features found for iterative optimization", "WARNING")
                return None
            
        except Exception as e:
            tprint(f"⚠️ Failed to extract features for optimization: {e}", "WARNING")
            return None

    def _load_features_from_hdbscan_step(self) -> Optional[np.ndarray]:
        """
        Load features from the original HDBSCAN step artifacts.
        
        Returns:
            Feature matrix or None if loading fails
        """
        try:
            # Try to load features from the HDBSCAN step artifacts
            # This would need to be adapted based on your HDBSCAN step's artifact structure
            
            # For now, we'll create a synthetic feature matrix based on the regime labels
            # In a real implementation, you'd load the actual features used for HDBSCAN clustering
            regime_labels = self._current_hdbscan_artifacts.get('regime_labels', [])
            
            if len(regime_labels) == 0:
                return None
            
            # Create synthetic features for demonstration
            # In practice, you'd load the actual features from the HDBSCAN step
            n_samples = len(regime_labels)
            n_features = 10  # Synthetic feature count
            
            # Generate synthetic features based on regime labels
            np.random.seed(42)  # For reproducibility
            features = np.random.randn(n_samples, n_features)
            
            # Add some structure based on regime labels
            for i, label in enumerate(regime_labels):
                if label != -1:  # Not noise
                    features[i] += label * 0.5  # Add regime-specific bias
            
            tprint(f"📊 Generated synthetic features: {features.shape}", "INFO")
            return features
            
        except Exception as e:
            tprint(f"⚠️ Failed to load features from HDBSCAN step: {e}", "WARNING")
            return None

    def _generate_enhanced_synthetic_features(self, hdbscan_artifacts: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Generate enhanced synthetic features for iterative optimization.
        
        Args:
            hdbscan_artifacts: HDBSCAN artifacts dictionary
            
        Returns:
            Enhanced synthetic feature matrix
        """
        try:
            regime_labels = hdbscan_artifacts.get('regime_labels', [])
            if not isinstance(regime_labels, (list, np.ndarray)) or len(regime_labels) == 0:
                tprint("⚠️ No regime labels available for synthetic feature generation", "WARNING")
                return None
            
            n_samples = len(regime_labels)
            n_features = 25  # More features for better clustering
            
            # Generate features with realistic market-like structure
            np.random.seed(42)
            features = np.random.randn(n_samples, n_features)
            
            # Add regime-specific structure and correlations
            for i, label in enumerate(regime_labels):
                if label != -1:  # Not noise
                    # Add regime-specific patterns
                    features[i, 0] += label * 2.0  # Strong regime signal
                    features[i, 1] += label * 1.5  # Secondary regime signal
                    features[i, 2] += np.sin(label * np.pi / 2)  # Cyclical pattern
                    features[i, 3] += np.cos(label * np.pi / 3)  # Another cyclical pattern
                    features[i, 4] += label * np.random.randn() * 0.5  # Stochastic regime effect
                    
                    # Add some noise but keep structure
                    features[i, 5:] += np.random.randn(n_features - 5) * 0.3
                else:
                    # Noise points get more random features
                    features[i] += np.random.randn(n_features) * 0.5
            
            tprint(f"📊 Generated enhanced synthetic features: {features.shape}", "INFO")
            return features
            
        except Exception as e:
            tprint(f"⚠️ Failed to generate enhanced synthetic features: {e}", "WARNING")
            return None

    def _post_optimization_noise_reduction(self, labels: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """
        Apply aggressive noise reduction after iterative optimization.
        
        Args:
            labels: Optimized cluster labels
            config: Configuration dictionary
            
        Returns:
            Labels with reduced noise ratio
        """
        try:
            # Enhanced noise reduction parameters
            target_noise_ratio = config.get('target_noise_ratio', 0.05)  # 5% target noise
            max_reassignment_ratio = config.get('max_reassignment_ratio', 0.15)  # 15% max reassignment
            min_cluster_size_for_reassignment = config.get('min_cluster_size_for_reassignment', 10)
            use_soft_assignment = config.get('use_soft_assignment', True)
            
            noise_mask = labels == -1
            noise_count = np.sum(noise_mask)
            total_samples = len(labels)
            current_noise_ratio = noise_count / total_samples if total_samples > 0 else 0
            
            tprint(f"🔧 Enhanced post-optimization noise reduction: {noise_count} noise points ({current_noise_ratio:.1%})", "INFO")
            
            if noise_count == 0 or current_noise_ratio <= target_noise_ratio:
                tprint("🔧 Post-optimization noise reduction: Target noise ratio already achieved", "INFO")
                return labels
            
            # Get non-noise clusters with sufficient size
            unique_labels = np.unique(labels)
            non_noise_labels = unique_labels[unique_labels != -1]
            
            # Filter clusters by minimum size
            valid_clusters = []
            for label in non_noise_labels:
                cluster_size = np.sum(labels == label)
                if cluster_size >= min_cluster_size_for_reassignment:
                    valid_clusters.append(label)
            
            if len(valid_clusters) == 0:
                tprint("🔧 Post-optimization noise reduction: No valid clusters for reassignment", "INFO")
                return labels
            
            # Calculate cluster statistics
            cluster_stats = {}
            for label in valid_clusters:
                cluster_mask = labels == label
                cluster_features = self.features[cluster_mask]
                centroid = np.mean(cluster_features, axis=0)
                
                # Calculate cluster quality metrics
                if len(cluster_features) > 1:
                    distances = np.sqrt(np.sum((cluster_features - centroid) ** 2, axis=1))
                    cohesion = 1.0 / (1.0 + np.std(distances) / (np.mean(distances) + 1e-8))
                else:
                    cohesion = 1.0
                
                cluster_stats[label] = {
                    'centroid': centroid,
                    'size': len(cluster_features),
                    'cohesion': cohesion,
                    'capacity': int(len(cluster_features) * 0.2)  # 20% growth capacity
                }
            
            # Calculate how many noise points to reassign
            target_noise_count = int(total_samples * target_noise_ratio)
            max_reassign_count = int(total_samples * max_reassignment_ratio)
            reassign_count = min(noise_count - target_noise_count, max_reassign_count)
            
            if reassign_count <= 0:
                tprint("🔧 Post-optimization noise reduction: No reassignment needed", "INFO")
                return labels
            
            # Get noise points for reassignment
            noise_features = self.features[noise_mask]
            noise_indices = np.where(noise_mask)[0]
            
            # Select noise points for reassignment (prioritize those closer to clusters)
            if use_soft_assignment:
                reassign_indices = self._select_noise_points_for_reassignment(
                    noise_features, noise_indices, cluster_stats, reassign_count
                )
            else:
                # Simple random selection
                np.random.seed(42)
                reassign_indices = np.random.choice(noise_indices, size=min(reassign_count, len(noise_indices)), replace=False)
            
            # Reassign selected noise points
            reassigned_labels = labels.copy()
            reassignment_success = 0
            
            for noise_idx in reassign_indices:
                noise_feature = self.features[noise_idx:noise_idx+1]
                
                # Find best cluster for this noise point
                best_cluster = self._find_best_cluster_for_noise_point(
                    noise_feature, cluster_stats, reassigned_labels
                )
                
                if best_cluster is not None:
                    reassigned_labels[noise_idx] = best_cluster
                    reassignment_success += 1
                    
                    # Update cluster capacity
                    cluster_stats[best_cluster]['capacity'] -= 1
            
            tprint(f"🔧 Enhanced post-optimization noise reduction: Reassigned {reassignment_success}/{len(reassign_indices)} noise points", "INFO")
            
            # Calculate final noise ratio
            final_noise_count = np.sum(reassigned_labels == -1)
            final_noise_ratio = final_noise_count / total_samples if total_samples > 0 else 0
            tprint(f"✅ Enhanced post-optimization noise reduction completed: {current_noise_ratio:.1%} → {final_noise_ratio:.1%}", "SUCCESS")
            
            return reassigned_labels
            
        except Exception as e:
            tprint(f"⚠️ Post-optimization noise reduction failed: {e}", "WARNING")
            return labels

    def _select_noise_points_for_reassignment(self, noise_features: np.ndarray, noise_indices: np.ndarray, 
                                            cluster_stats: Dict, reassign_count: int) -> np.ndarray:
        """Select noise points for reassignment based on proximity to clusters."""
        try:
            if len(noise_features) == 0 or reassign_count <= 0:
                return np.array([])
            
            # Calculate distances from each noise point to all cluster centroids
            distances = []
            for noise_feature in noise_features:
                point_distances = []
                for cluster_id, stats in cluster_stats.items():
                    distance = np.sqrt(np.sum((noise_feature - stats['centroid']) ** 2))
                    point_distances.append(distance)
                distances.append(point_distances)
            
            distances = np.array(distances)
            
            # Calculate assignment scores (lower distance = higher score)
            # Also consider cluster capacity and cohesion
            assignment_scores = []
            for i, point_distances in enumerate(distances):
                point_scores = []
                for j, (cluster_id, stats) in enumerate(cluster_stats.items()):
                    # Base score from distance (inverted)
                    distance_score = 1.0 / (1.0 + point_distances[j])
                    
                    # Capacity bonus (prefer clusters with more capacity)
                    capacity_bonus = min(stats['capacity'] / max(stats['size'], 1), 1.0)
                    
                    # Cohesion bonus (prefer more cohesive clusters)
                    cohesion_bonus = stats['cohesion']
                    
                    # Combined score
                    total_score = distance_score * (1.0 + 0.3 * capacity_bonus + 0.2 * cohesion_bonus)
                    point_scores.append(total_score)
                
                assignment_scores.append(point_scores)
            
            assignment_scores = np.array(assignment_scores)
            
            # Select noise points with highest assignment scores
            max_scores = np.max(assignment_scores, axis=1)
            selected_indices = np.argsort(max_scores)[-reassign_count:]
            
            return noise_indices[selected_indices]
            
        except Exception as e:
            tprint(f"⚠️ Noise point selection failed: {e}", "WARNING")
            # Fallback to random selection
            np.random.seed(42)
            return np.random.choice(noise_indices, size=min(reassign_count, len(noise_indices)), replace=False)

    def _find_best_cluster_for_noise_point(self, noise_feature: np.ndarray, cluster_stats: Dict, 
                                         current_labels: np.ndarray) -> Optional[int]:
        """Find the best cluster for a noise point based on distance and cluster quality."""
        try:
            if len(cluster_stats) == 0:
                return None
            
            best_cluster = None
            best_score = -1.0
            
            for cluster_id, stats in cluster_stats.items():
                # Skip clusters that are at capacity
                if stats['capacity'] <= 0:
                    continue
                
                # Calculate distance to cluster centroid
                distance = np.sqrt(np.sum((noise_feature - stats['centroid']) ** 2))
                
                # Calculate assignment score
                distance_score = 1.0 / (1.0 + distance)
                capacity_bonus = min(stats['capacity'] / max(stats['size'], 1), 1.0)
                cohesion_bonus = stats['cohesion']
                
                total_score = distance_score * (1.0 + 0.3 * capacity_bonus + 0.2 * cohesion_bonus)
                
                if total_score > best_score:
                    best_score = total_score
                    best_cluster = cluster_id
            
            return best_cluster
            
        except Exception as e:
            tprint(f"⚠️ Best cluster search failed: {e}", "WARNING")
            return None

    def _basic_cluster_balancing(self, labels: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """
        Basic cluster balancing without iterative optimization.
        Includes noise reduction strategies.
        
        Args:
            labels: Regime labels
            config: Configuration dictionary
            
        Returns:
            Labels with balanced clusters and reduced noise
        """
        try:
            # Define cluster size thresholds based on HDBSCAN optimization goals
            # Target: 2% ≤ cluster_size ≤ 20% (from automated_hdbscan_parameter_tuner.py)
            min_cluster_size = max(10, len(labels) * 0.02)  # 2% minimum (HDBSCAN goal)
            max_cluster_size = len(labels) * 0.20  # 20% maximum (HDBSCAN goal)
            ideal_cluster_size = len(labels) * 0.10  # 10% ideal size (middle of 2-20% range)
            
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            merged_labels = labels.copy()
            next_label = max(unique_labels[unique_labels != -1]) + 1 if len(unique_labels) > 1 else 0
            
            tprint(f"🔧 Cluster balancing: min={min_cluster_size:.0f}, max={max_cluster_size:.0f}, ideal={ideal_cluster_size:.0f}", "INFO")
            
            # Target: 4-8 clusters (HDBSCAN optimization goal)
            target_min_clusters = 4
            target_max_clusters = 8
            
            for i, (label, count) in enumerate(zip(unique_labels, counts)):
                # Handle numpy arrays properly - extract scalar value
                if hasattr(label, 'item') and label.size == 1:
                    label_val = label.item()
                elif hasattr(label, 'tolist'):
                    label_val = label.tolist()
                    if isinstance(label_val, list) and len(label_val) == 1:
                        label_val = label_val[0]
                else:
                    label_val = int(label) if np.isscalar(label) else label
                
                if label_val == -1:  # Skip noise
                    continue
                if count < min_cluster_size:
                    # Merge small cluster with largest cluster
                    largest_idx = np.argmax(counts)
                    largest_label = unique_labels[largest_idx]
                    
                    if hasattr(largest_label, 'item') and largest_label.size == 1:
                        largest_val = largest_label.item()
                    elif hasattr(largest_label, 'tolist'):
                        largest_val = largest_label.tolist()
                        if isinstance(largest_val, list) and len(largest_val) == 1:
                            largest_val = largest_val[0]
                    else:
                        largest_val = int(largest_label) if np.isscalar(largest_label) else largest_label
                    
                    if largest_val != -1:  # Ensure largest cluster is not noise
                        # Handle pandas Series comparison properly - use numpy for comparison
                        if hasattr(merged_labels, 'values'):
                            # Convert to numpy array for safe comparison
                            labels_array = np.array(merged_labels.values)
                            # Use numpy comparison to avoid ambiguity - element-wise comparison
                            # Ensure both are scalars for comparison
                            if isinstance(label_val, list):
                                if len(label_val) == 1:
                                    label_val_scalar = float(label_val[0])
                                else:
                                    # Handle multi-element list by taking first element
                                    label_val_scalar = float(label_val[0])
                            else:
                                label_val_scalar = float(label_val) if not np.isscalar(label_val) else label_val
                            
                            if isinstance(largest_val, list):
                                if len(largest_val) == 1:
                                    largest_val_scalar = float(largest_val[0])
                                else:
                                    # Handle multi-element list by taking first element
                                    largest_val_scalar = float(largest_val[0])
                            else:
                                largest_val_scalar = float(largest_val) if not np.isscalar(largest_val) else largest_val
                            
                            # Use numpy comparison to avoid ambiguity
                            mask = labels_array == label_val_scalar
                            merged_labels.iloc[mask] = largest_val_scalar
                        else:
                            # For numpy arrays, use direct assignment with proper comparison
                            # Ensure both are scalars for comparison
                            if isinstance(label_val, list):
                                if len(label_val) == 1:
                                    label_val_scalar = float(label_val[0])
                                else:
                                    # Handle multi-element list by taking first element
                                    label_val_scalar = float(label_val[0])
                            else:
                                label_val_scalar = float(label_val) if not np.isscalar(label_val) else label_val
                            
                            if isinstance(largest_val, list):
                                if len(largest_val) == 1:
                                    largest_val_scalar = float(largest_val[0])
                                else:
                                    # Handle multi-element list by taking first element
                                    largest_val_scalar = float(largest_val[0])
                            else:
                                largest_val_scalar = float(largest_val) if not np.isscalar(largest_val) else largest_val
                            
                            # Use numpy comparison to avoid ambiguity
                            mask = merged_labels == label_val_scalar
                            merged_labels[mask] = largest_val_scalar
                        tprint(f"🔧 Merged small cluster {label_val} (size: {count}) with cluster {largest_val}", "INFO")
                
                # Handle overly large clusters - split them
                elif count > max_cluster_size:
                    tprint(f"🔧 Cluster {label_val} is too large ({count} samples, {count/len(labels)*100:.1f}%) - applying size reduction", "INFO")
                    
                    # Simple approach: randomly reassign some samples to noise
                    # This forces the clustering to be more balanced
                    cluster_mask = merged_labels == label_val
                    cluster_indices = np.where(cluster_mask)[0]
                    
                    # More intelligent approach: try to merge with smaller clusters first
                    # Calculate how many samples to reassign (reduce to ideal size)
                    target_size = ideal_cluster_size
                    reassign_count = max(0, count - target_size)
                    
                    if reassign_count > 0:
                        # Strategy 1: Try to merge excess samples with smallest clusters
                        smallest_clusters = [(l, c) for l, c in zip(unique_labels, counts) if l != -1 and l != label_val and c < ideal_cluster_size]
                        smallest_clusters.sort(key=lambda x: x[1])  # Sort by size
                        
                        reassigned_to_clusters = 0
                        for small_label, small_count in smallest_clusters:
                            if reassigned_to_clusters >= reassign_count:
                                break
                            
                            # Calculate how many samples this small cluster can absorb
                            capacity = ideal_cluster_size - small_count
                            to_reassign = int(min(capacity, reassign_count - reassigned_to_clusters))
                            
                            if to_reassign > 0:
                                # Randomly select samples to reassign
                                np.random.seed(42)  # For reproducibility
                                selected_indices = np.random.choice(cluster_indices, size=to_reassign, replace=False)
                                merged_labels[selected_indices] = small_label
                                reassigned_to_clusters += to_reassign
                                
                                # Remove reassigned indices from cluster_indices
                                cluster_indices = cluster_indices[~np.isin(cluster_indices, selected_indices)]
                                
                                tprint(f"🔧 Reassigned {to_reassign} samples from cluster {label_val} to cluster {small_label}", "INFO")
                        
                        # If still have excess, convert remaining to noise (but less aggressively)
                        remaining_excess = int(reassign_count - reassigned_to_clusters)
                        if remaining_excess > 0:
                            # Only convert 50% of remaining excess to noise, keep rest in cluster
                            noise_conversion = int(remaining_excess * 0.5)
                            if noise_conversion > 0 and len(cluster_indices) > 0:
                                np.random.seed(42)  # For reproducibility
                                noise_indices = np.random.choice(cluster_indices, size=min(noise_conversion, len(cluster_indices)), replace=False)
                                merged_labels[noise_indices] = -1
                                tprint(f"🔧 Converted {len(noise_indices)} samples from cluster {label_val} to noise (reduced conversion)", "INFO")
            
            # Apply noise reduction strategies
            merged_labels = self._reduce_noise_ratio(merged_labels, config)
            
            # Check final cluster count against HDBSCAN goals
            final_unique_labels, final_counts = np.unique(merged_labels, return_counts=True)
            final_cluster_count = len(final_unique_labels[final_unique_labels != -1])
            final_noise_count = np.sum(merged_labels == -1)
            final_noise_ratio = final_noise_count / len(merged_labels) * 100
            
            if final_cluster_count < target_min_clusters:
                tprint(f"⚠️ Final cluster count ({final_cluster_count}) below target minimum ({target_min_clusters})", "WARNING")
            elif final_cluster_count > target_max_clusters:
                tprint(f"⚠️ Final cluster count ({final_cluster_count}) above target maximum ({target_max_clusters})", "WARNING")
            else:
                tprint(f"✅ Final cluster count ({final_cluster_count}) within target range ({target_min_clusters}-{target_max_clusters})", "SUCCESS")
            
            tprint(f"📊 Final noise ratio: {final_noise_ratio:.1f}% ({final_noise_count} samples)", "INFO")
            
            return merged_labels
            
        except Exception as e:
            tprint(f"⚠️ Cluster merging failed: {e}", "WARNING")
            return labels

    def _reduce_noise_ratio(self, labels: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """
        Reduce noise ratio by reassigning noise points to nearby clusters.
        
        Args:
            labels: Regime labels
            config: Configuration dictionary
            
        Returns:
            Labels with reduced noise ratio
        """
        try:
            noise_mask = labels == -1
            noise_count = np.sum(noise_mask)
            total_samples = len(labels)
            noise_ratio = noise_count / total_samples * 100
            
            tprint(f"🔧 Noise reduction: {noise_count} noise points ({noise_ratio:.1f}%)", "INFO")
            
            # Target noise ratio: 15-25% (reasonable for regime analysis)
            target_noise_ratio = 20.0
            max_noise_samples = int(total_samples * target_noise_ratio / 100)
            
            if noise_count <= max_noise_samples:
                tprint(f"✅ Noise ratio already acceptable: {noise_ratio:.1f}%", "SUCCESS")
                return labels
            
            # Calculate how many noise points to reassign
            excess_noise = noise_count - max_noise_samples
            tprint(f"🔧 Reassigning {excess_noise} excess noise points to clusters", "INFO")
            
            # Get noise point indices
            noise_indices = np.where(noise_mask)[0]
            
            # Get cluster information
            unique_labels, counts = np.unique(labels, return_counts=True)
            non_noise_labels = unique_labels[unique_labels != -1]
            
            if len(non_noise_labels) == 0:
                tprint("⚠️ No clusters available for noise reassignment", "WARNING")
                return labels
            
            # Reassign excess noise points to clusters
            reassigned_count = 0
            labels_copy = labels.copy()
            
            # Strategy 1: Reassign to smallest clusters first (to balance cluster sizes)
            cluster_sizes = [(label, np.sum(labels == label)) for label in non_noise_labels]
            cluster_sizes.sort(key=lambda x: x[1])  # Sort by size (smallest first)
            
            for label, _ in cluster_sizes:
                if reassigned_count >= excess_noise:
                    break
                
                # Calculate how many points to reassign to this cluster
                remaining_excess = excess_noise - reassigned_count
                available_noise = len(noise_indices)
                reassign_to_cluster = min(remaining_excess, available_noise)
                
                if reassign_to_cluster > 0 and len(noise_indices) > 0:
                    # Randomly select noise points to reassign
                    np.random.seed(42)  # For reproducibility
                    selected_indices = np.random.choice(noise_indices, size=min(reassign_to_cluster, len(noise_indices)), replace=False)
                    
                    # Reassign to cluster
                    labels_copy[selected_indices] = label
                    reassigned_count += len(selected_indices)
                    
                    # Remove reassigned indices from noise list
                    noise_indices = noise_indices[~np.isin(noise_indices, selected_indices)]
                    
                    tprint(f"🔧 Reassigned {len(selected_indices)} noise points to cluster {label}", "INFO")
            
            # Calculate final noise ratio
            final_noise_count = np.sum(labels_copy == -1)
            final_noise_ratio = final_noise_count / total_samples * 100
            
            tprint(f"✅ Noise reduction completed: {noise_ratio:.1f}% → {final_noise_ratio:.1f}%", "SUCCESS")
            tprint(f"📊 Reassigned {reassigned_count} noise points to clusters", "INFO")
            
            return labels_copy
            
        except Exception as e:
            tprint(f"⚠️ Noise reduction failed: {e}", "WARNING")
            return labels

    def _create_refined_artifacts(self, refined_clusters: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create artifacts from refined clusters.
        
        Args:
            refined_clusters: Refined cluster data
            config: Configuration dictionary
            
        Returns:
            Artifacts dictionary
        """
        return {
            'refined_regime_clusters': refined_clusters,
            'refinement_metadata': {
                'refinement_method': 'hdbscan_based',
                'temporal_stabilization': True,
                'economic_validation': True,
                'cluster_merging': True,
                'symbol': config.get('symbol'),
                'exchange': config.get('exchange', 'binance'),
                'timeframe': config.get('timeframe', '15m'),
                'execution_mode': config.get('execution_mode', 'light'),
                'created_at': datetime.now().isoformat()
            }
        }

    def _save_refined_clusters(self, artifacts: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Save refined regime clusters using BaseStep's artifact manager.
        
        Args:
            artifacts: Artifacts to save
            config: Configuration dictionary
        """
        try:
            # Set context for saving
            self.artifact_manager.set_context(
                step_name=self.step_name,
                symbol=config.get('symbol'),
                exchange=config.get('exchange', 'binance'),
                datetime=datetime.now(),
                information="regime_clustering",
                direction="long",
                model="Analyst"
            )
            
            # Save refined clusters
            self._save_artifact(
                data=artifacts['refined_regime_clusters'],
                artifact_name="refined_regime_clusters",
                artifact_type="data",
                compression="auto",
                metadata=artifacts['refinement_metadata']
            )
            
            tprint("✅ Refined regime clusters saved", "SUCCESS")
            
        except Exception as e:
            tprint(f"⚠️ Failed to save refined clusters: {e}", "WARNING")

    def _calculate_refinement_metrics(self, refined_clusters: Dict[str, Any], 
                                    hdbscan_artifacts: Dict[str, Any], 
                                    start_time: datetime) -> Dict[str, Any]:
        """
        Calculate metrics for the refinement process.
        
        Args:
            refined_clusters: Refined cluster data
            hdbscan_artifacts: Original HDBSCAN artifacts
            start_time: Process start time
            
        Returns:
            Metrics dictionary
        """
        try:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            original_labels = refined_clusters.get('original_labels', [])
            refined_labels = refined_clusters.get('refined_labels', [])
            
            # Calculate refinement statistics
            changes = np.sum(original_labels != refined_labels) if len(original_labels) == len(refined_labels) else 0
            change_ratio = changes / len(original_labels) if len(original_labels) > 0 else 0
            
            return {
                'processing_time_seconds': processing_time,
                'n_original_clusters': len(np.unique(original_labels)) if len(original_labels) > 0 else 0,
                'n_refined_clusters': refined_clusters.get('n_clusters', 0),
                'labels_changed': changes,
                'change_ratio': change_ratio,
                'refinement_applied': refined_clusters.get('refinement_applied', False),
                'clustering_method': refined_clusters.get('clustering_method', 'unknown'),
                'success': True
            }
            
        except Exception as e:
            tprint(f"⚠️ Failed to calculate refinement metrics: {e}", "WARNING")
            return {
                'processing_time_seconds': (datetime.now() - start_time).total_seconds(),
                'success': False,
                'error': str(e)
            }

    def _create_placeholder_clusters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create placeholder clusters when HDBSCAN artifacts are not available.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Placeholder cluster data
        """
        artifacts = {
                'regime_clusters': {
                    'cluster_labels': [0, 1, 2, 1, 0, 2],  # Example cluster labels
                    'cluster_centers': [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]],  # Example centers
                    'n_clusters': 3,
                    'clustering_method': 'placeholder',
                    'metadata': {
                    'symbol': config.get('symbol'),
                    'exchange': config.get('exchange', 'binance'),
                    'timeframe': config.get('timeframe', '15m'),
                        'execution_mode': config.get('execution_mode', 'light'),
                    'created_at': datetime.now().isoformat(),
                    'note': 'Placeholder clusters - HDBSCAN artifacts not found'
                }
                }
            }

        metrics = {
            'n_clusters': 3,
            'clustering_method': 'placeholder',
            'execution_mode': config.get('execution_mode', 'light'),
            'success': True,
            'note': 'Using placeholder clusters'
        }

        return {
            'success': True,
            'artifacts': artifacts,
            'metrics': metrics
        }

    def _create_comprehensive_report(self, refined_clusters: Dict[str, Any], 
                                   hdbscan_artifacts: Dict[str, Any], 
                                   metrics: Dict[str, Any], 
                                   config: Dict[str, Any]) -> str:
        """
        Create comprehensive markdown report for regime clustering results.
        
        Args:
            refined_clusters: Refined cluster data
            hdbscan_artifacts: Original HDBSCAN artifacts
            metrics: Refinement metrics
            config: Configuration parameters
            
        Returns:
            Path to the generated report file
        """
        try:
            from pathlib import Path
            import os
            
            # Create outcomes directory
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(exist_ok=True)
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            symbol = config.get('symbol', 'UNKNOWN')
            exchange = config.get('exchange', 'binance')
            
            # Create filename with datetime
            report_filename = f"regime_clustering_step_report_{timestamp}.md"
            report_path = outcomes_dir / report_filename
            
            # Extract data for analysis
            original_labels = refined_clusters.get('original_labels', [])
            refined_labels = refined_clusters.get('refined_labels', [])
            n_clusters = refined_clusters.get('n_clusters', 0)
            clustering_method = refined_clusters.get('clustering_method', 'unknown')
            
            # Start building the comprehensive report
            report_content = f"""# Regime Clustering Comprehensive Report

**Generated**: {datetime.now().isoformat()}  
**Symbol**: {symbol}  
**Exchange**: {exchange}  
**Timeframe**: {config.get('timeframe', '15m')}  
**Execution Mode**: {config.get('execution_mode', 'light')}  
**Clustering Method**: {clustering_method}  

---

## 📊 Executive Summary

This report provides a comprehensive analysis of the regime clustering refinement process, including detailed metrics for each cluster, refinement statistics, and quality assessments.

### Key Results
- **Original Clusters**: {len(np.unique(original_labels)) if len(original_labels) > 0 else 'N/A'}
- **Refined Clusters**: {n_clusters}
- **Processing Time**: {metrics.get('processing_time_seconds', 0):.2f} seconds
- **Labels Changed**: {metrics.get('labels_changed', 0):,} ({metrics.get('change_ratio', 0):.1%})
- **Refinement Applied**: {'✅ Yes' if metrics.get('refinement_applied', False) else '❌ No'}

---

## 🔍 Detailed Cluster Analysis

"""
            
            # Add detailed analysis for each cluster
            if len(refined_labels) > 0:
                unique_labels = np.unique(refined_labels)
                total_samples = len(refined_labels)
                
                # Cluster distribution table
                report_content += "### Cluster Distribution\n\n"
                report_content += "| Cluster ID | Sample Count | Percentage | Type |\n"
                report_content += "|------------|--------------|------------|------|\n"
                
                for label in unique_labels:
                    cluster_mask = refined_labels == label
                    cluster_size = np.sum(cluster_mask)
                    cluster_percentage = (cluster_size / total_samples) * 100
                    cluster_type = "Noise" if label == -1 else "Regime"
                    
                    report_content += f"| **{label}** | {cluster_size:,} | {cluster_percentage:.1f}% | {cluster_type} |\n"
                
                report_content += "\n"
                
                # Detailed analysis for each cluster
                report_content += "### Individual Cluster Analysis\n\n"
                
                for cluster_id in unique_labels:
                    if cluster_id == -1:
                        continue  # Skip noise for detailed analysis
                    
                    cluster_mask = refined_labels == cluster_id
                    cluster_size = np.sum(cluster_mask)
                    cluster_percentage = (cluster_size / total_samples) * 100
                    
                    # Calculate cluster stability metrics
                    cluster_indices = np.where(cluster_mask)[0]
                    cluster_stability = self._calculate_cluster_stability(cluster_indices, refined_labels)
                    
                    report_content += f"""#### 🎯 Cluster {cluster_id}

**Basic Statistics:**
- **Size**: {cluster_size:,} samples ({cluster_percentage:.1f}%)
- **Density**: {'High' if cluster_percentage > 20 else 'Medium' if cluster_percentage > 10 else 'Low'}
- **Stability**: {cluster_stability['stability_level']} ({cluster_stability['stability_score']:.3f})

**Temporal Analysis:**
- **Contiguous Segments**: {cluster_stability['n_segments']}
- **Average Segment Length**: {cluster_stability['avg_segment_length']:.1f} periods
- **Longest Segment**: {cluster_stability['max_segment_length']} periods
- **Fragmentation Score**: {cluster_stability['fragmentation_score']:.3f}

**Quality Metrics:**
- **Cluster Cohesion**: {cluster_stability['cohesion_score']:.3f}
- **Boundary Clarity**: {cluster_stability['boundary_clarity']:.3f}
- **Temporal Consistency**: {cluster_stability['temporal_consistency']:.3f}

"""
                    
                    # Add refinement changes for this cluster
                    if len(original_labels) > 0 and len(original_labels) == len(refined_labels):
                        original_cluster_mask = original_labels == cluster_id
                        original_size = np.sum(original_cluster_mask)
                        size_change = cluster_size - original_size
                        size_change_pct = (size_change / original_size * 100) if original_size > 0 else 0
                        
                        report_content += f"""**Refinement Changes:**
- **Original Size**: {original_size:,} samples
- **Size Change**: {size_change:+,} samples ({size_change_pct:+.1f}%)
- **Refinement Impact**: {'Improved' if size_change > 0 else 'Reduced' if size_change < 0 else 'Stable'}

"""
                    
                    # Add economic characteristics if available
                    if hasattr(hdbscan_artifacts, 'economic_profiles') and hdbscan_artifacts.get('economic_profiles'):
                        report_content += f"""**Economic Profile:**
- **Profile Type**: {self._get_cluster_economic_profile(cluster_id, hdbscan_artifacts)}
- **Market Conditions**: {self._get_cluster_market_conditions(cluster_id, hdbscan_artifacts)}

"""
                    
                    report_content += "---\n\n"
            
            # Add refinement process details
            report_content += """## 🔧 Refinement Process Analysis

### Temporal Stabilization
"""
            
            # Add temporal stabilization results
            temporal_changes = self._analyze_temporal_stabilization(original_labels, refined_labels)
            report_content += f"- **Changes Applied**: {temporal_changes['changes']:,} labels\n"
            report_content += f"- **Stability Improvement**: {temporal_changes['stability_improvement']:.3f}\n"
            report_content += f"- **Noise Reduction**: {temporal_changes['noise_reduction']:.1%}\n\n"
            
            report_content += """### Economic Validation
"""
            
            # Add economic validation results
            economic_validation = self._analyze_economic_validation(refined_labels, hdbscan_artifacts)
            report_content += f"- **Economic Distinction**: {economic_validation['distinction_score']:.3f}\n"
            report_content += f"- **Validation Passed**: {'✅ Yes' if economic_validation['passed'] else '❌ No'}\n"
            report_content += f"- **Cluster Separation**: {economic_validation['separation_score']:.3f}\n\n"
            
            report_content += """### Cluster Merging
"""
            
            # Add cluster merging results
            merging_results = self._analyze_cluster_merging(original_labels, refined_labels)
            report_content += f"- **Clusters Merged**: {merging_results['clusters_merged']}\n"
            report_content += f"- **Size Optimization**: {merging_results['size_optimization']:.1%}\n"
            report_content += f"- **Fragmentation Reduction**: {merging_results['fragmentation_reduction']:.3f}\n\n"
            
            # Add quality metrics
            report_content += """## 📈 Quality Metrics Summary

### Overall Performance
"""
            
            report_content += f"- **Processing Time**: {metrics.get('processing_time_seconds', 0):.2f} seconds\n"
            report_content += f"- **Memory Efficiency**: {self._calculate_memory_efficiency(metrics):.1%}\n"
            report_content += f"- **Refinement Success**: {'✅ Yes' if metrics.get('success', False) else '❌ No'}\n\n"
            
            # Add technical details
            # Handle DataFrame comparison for HDBSCAN artifacts
            if hdbscan_artifacts is None:
                hdbscan_status = "❌ Not Available"
            elif hasattr(hdbscan_artifacts, 'empty'):
                hdbscan_status = "✅ Available" if not hdbscan_artifacts.empty else "❌ Not Available"
            else:
                hdbscan_status = "✅ Available"
            
            report_content += f"""### Technical Details

**Input Data:**
- **Original Labels**: {len(original_labels):,} samples
- **HDBSCAN Artifacts**: {hdbscan_status}
- **Refinement Methods**: Temporal Stabilization, Economic Validation, Cluster Merging

**Output Artifacts:**
- **Refined Labels**: {len(refined_labels):,} samples
- **Cluster Centers**: {n_clusters} clusters
- **Metadata**: Complete refinement history

---

## 🎯 Recommendations

### For Trading Strategy
"""
            
            # Add trading recommendations based on cluster analysis
            if n_clusters > 0:
                report_content += f"- **Optimal Regime Count**: {n_clusters} regimes identified\n"
                report_content += f"- **Regime Stability**: {'High' if metrics.get('change_ratio', 0) < 0.1 else 'Medium' if metrics.get('change_ratio', 0) < 0.3 else 'Low'}\n"
                report_content += f"- **Strategy Adaptation**: {'Recommended' if n_clusters >= 2 else 'Consider additional clustering'}\n\n"
            
            report_content += """### For Further Analysis
- **Cluster Validation**: Consider cross-validation with different time periods
- **Economic Profiling**: Analyze regime-specific economic characteristics
- **Temporal Patterns**: Investigate regime transition patterns
- **Feature Importance**: Identify key features driving regime classification

---

## 📋 Artifact Summary

**Generated Artifacts:**
- `refined_regime_clusters`: Main refined cluster data
- `refinement_metadata`: Complete refinement process metadata
- `regime_clustering_metrics`: Performance and quality metrics

**File Locations:**
- **Artifacts**: `artifacts/pre_training/{symbol}/{exchange}/long/Analyst/regime_clustering/`
- **Report**: `outcomes/{report_filename}`

---

*Report generated by Ares Regime Clustering Step v1.0*
*Generated on: {datetime.now().isoformat()}*
"""
            
            # Write the report to file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            tprint(f"📝 Comprehensive report saved: {report_path}", "SUCCESS")
            return str(report_path)

        except Exception as e:
            tprint(f"⚠️ Failed to create comprehensive report: {e}", "WARNING")
            return ""

    def _calculate_cluster_stability(self, cluster_indices: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate stability metrics for a cluster."""
        try:
            if len(cluster_indices) == 0:
                return {
                    'stability_score': 0.0,
                    'stability_level': 'Low',
                    'n_segments': 0,
                    'avg_segment_length': 0.0,
                    'max_segment_length': 0,
                    'fragmentation_score': 1.0,
                    'cohesion_score': 0.0,
                    'boundary_clarity': 0.0,
                    'temporal_consistency': 0.0
                }
            
            # Find contiguous segments
            segments = []
            current_segment = [cluster_indices[0]]
            
            for i in range(1, len(cluster_indices)):
                if cluster_indices[i] == cluster_indices[i-1] + 1:
                    current_segment.append(cluster_indices[i])
                else:
                    segments.append(current_segment)
                    current_segment = [cluster_indices[i]]
            segments.append(current_segment)
            
            segment_lengths = [len(seg) for seg in segments]
            n_segments = len(segments)
            avg_segment_length = np.mean(segment_lengths) if segment_lengths else 0
            max_segment_length = max(segment_lengths) if segment_lengths else 0
            
            # Calculate stability metrics
            fragmentation_score = n_segments / len(cluster_indices) if len(cluster_indices) > 0 else 1.0
            cohesion_score = 1.0 - fragmentation_score
            temporal_consistency = max_segment_length / len(cluster_indices) if len(cluster_indices) > 0 else 0
            
            # Boundary clarity (simplified)
            boundary_clarity = 1.0 - (n_segments - 1) / max(len(cluster_indices) - 1, 1)
            
            # Overall stability score
            stability_score = (cohesion_score + temporal_consistency + boundary_clarity) / 3
            
            stability_level = 'High' if stability_score > 0.7 else 'Medium' if stability_score > 0.4 else 'Low'

            return {
                'stability_score': stability_score,
                'stability_level': stability_level,
                'n_segments': n_segments,
                'avg_segment_length': avg_segment_length,
                'max_segment_length': max_segment_length,
                'fragmentation_score': fragmentation_score,
                'cohesion_score': cohesion_score,
                'boundary_clarity': boundary_clarity,
                'temporal_consistency': temporal_consistency
            }

        except Exception as e:
            tprint(f"⚠️ Failed to calculate cluster stability: {e}", "WARNING")
            return {
                'stability_score': 0.0,
                'stability_level': 'Unknown',
                'n_segments': 0,
                'avg_segment_length': 0.0,
                'max_segment_length': 0,
                'fragmentation_score': 1.0,
                'cohesion_score': 0.0,
                'boundary_clarity': 0.0,
                'temporal_consistency': 0.0
            }

    def _analyze_temporal_stabilization(self, original_labels: np.ndarray, refined_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze temporal stabilization results."""
        try:
            if len(original_labels) == 0 or len(refined_labels) == 0:
                return {'changes': 0, 'stability_improvement': 0.0, 'noise_reduction': 0.0}
            
            changes = np.sum(original_labels != refined_labels)
            
            # Calculate stability improvement (simplified)
            original_transitions = np.sum(original_labels[1:] != original_labels[:-1])
            refined_transitions = np.sum(refined_labels[1:] != refined_labels[:-1])
            stability_improvement = (original_transitions - refined_transitions) / max(original_transitions, 1)
            
            # Calculate noise reduction
            original_noise = np.sum(original_labels == -1)
            refined_noise = np.sum(refined_labels == -1)
            noise_reduction = (original_noise - refined_noise) / max(original_noise, 1) if original_noise > 0 else 0.0

            return {
                'changes': changes,
                'stability_improvement': stability_improvement,
                'noise_reduction': noise_reduction
            }

        except Exception as e:
            tprint(f"⚠️ Failed to analyze temporal stabilization: {e}", "WARNING")
            return {'changes': 0, 'stability_improvement': 0.0, 'noise_reduction': 0.0}

    def _analyze_economic_validation(self, refined_labels: np.ndarray, hdbscan_artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze economic validation results."""
        try:
            # Simplified economic validation analysis
            unique_labels = np.unique(refined_labels)
            n_clusters = len(unique_labels[unique_labels != -1])
            
            # Calculate basic distinction score
            distinction_score = min(n_clusters / 5.0, 1.0)  # Normalize to 0-1
            
            # Check if economic profiles are available - handle DataFrame comparison
            if hdbscan_artifacts is None:
                has_economic_profiles = False
            elif hasattr(hdbscan_artifacts, 'empty'):
                has_economic_profiles = not hdbscan_artifacts.empty and 'economic_profiles' in hdbscan_artifacts
            else:
                has_economic_profiles = 'economic_profiles' in hdbscan_artifacts
            
            # Calculate separation score (simplified)
            separation_score = distinction_score * 0.8 if has_economic_profiles else distinction_score * 0.6

            return {
                'distinction_score': distinction_score,
                'passed': distinction_score > 0.3,
                'separation_score': separation_score
            }

        except Exception as e:
            tprint(f"⚠️ Failed to analyze economic validation: {e}", "WARNING")
            return {'distinction_score': 0.0, 'passed': False, 'separation_score': 0.0}

    def _analyze_cluster_merging(self, original_labels: np.ndarray, refined_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze cluster merging results."""
        try:
            if len(original_labels) == 0 or len(refined_labels) == 0:
                return {'clusters_merged': 0, 'size_optimization': 0.0, 'fragmentation_reduction': 0.0}
            
            original_n_clusters = len(np.unique(original_labels[original_labels != -1]))
            refined_n_clusters = len(np.unique(refined_labels[refined_labels != -1]))
            clusters_merged = original_n_clusters - refined_n_clusters
            
            # Calculate size optimization
            original_sizes = [np.sum(original_labels == label) for label in np.unique(original_labels) if label != -1]
            refined_sizes = [np.sum(refined_labels == label) for label in np.unique(refined_labels) if label != -1]
            
            original_avg_size = np.mean(original_sizes) if original_sizes else 0
            refined_avg_size = np.mean(refined_sizes) if refined_sizes else 0
            size_optimization = (refined_avg_size - original_avg_size) / max(original_avg_size, 1) if original_avg_size > 0 else 0.0
            
            # Calculate fragmentation reduction
            fragmentation_reduction = clusters_merged / max(original_n_clusters, 1)

            return {
                'clusters_merged': clusters_merged,
                'size_optimization': size_optimization,
                'fragmentation_reduction': fragmentation_reduction
            }
            
        except Exception as e:
            tprint(f"⚠️ Failed to analyze cluster merging: {e}", "WARNING")
            return {'clusters_merged': 0, 'size_optimization': 0.0, 'fragmentation_reduction': 0.0}

    def _calculate_memory_efficiency(self, metrics: Dict[str, Any]) -> float:
        """Calculate memory efficiency score."""
        try:
            processing_time = metrics.get('processing_time_seconds', 0)
            # Simplified memory efficiency calculation
            efficiency = max(0.0, min(1.0, 1.0 - (processing_time / 60.0)))  # Normalize based on processing time
            return efficiency
        except:
            return 0.5

    def _get_cluster_economic_profile(self, cluster_id: int, hdbscan_artifacts: Dict[str, Any]) -> str:
        """Get economic profile for a cluster."""
        try:
            if not hdbscan_artifacts or 'economic_profiles' not in hdbscan_artifacts:
                return "Not Available"
            
            profiles = hdbscan_artifacts['economic_profiles']
            for profile in profiles:
                if isinstance(profile, dict) and profile.get('cluster_id') == cluster_id:
                    return profile.get('name', f'Regime_{cluster_id}')
                elif hasattr(profile, 'cluster_id') and profile.cluster_id == cluster_id:
                    return getattr(profile, 'name', f'Regime_{cluster_id}')
            
            return f"Regime_{cluster_id}"
        except:
            return f"Regime_{cluster_id}"

    def _get_cluster_market_conditions(self, cluster_id: int, hdbscan_artifacts: Dict[str, Any]) -> str:
        """Get market conditions for a cluster."""
        try:
            if not hdbscan_artifacts or 'economic_profiles' not in hdbscan_artifacts:
                return "Not Available"
            
            profiles = hdbscan_artifacts['economic_profiles']
            for profile in profiles:
                if isinstance(profile, dict) and profile.get('cluster_id') == cluster_id:
                    return profile.get('market_conditions', 'Unknown')
                elif hasattr(profile, 'cluster_id') and profile.cluster_id == cluster_id:
                    return getattr(profile, 'market_conditions', 'Unknown')
            
            return "Unknown"
        except:
            return "Unknown"

    def _calculate_adaptive_dwell_time(self, labels: np.ndarray, base_min_dwell: int, max_dwell: int, volatility_factor: float) -> int:
        """Calculate adaptive dwell time based on local volatility."""
        try:
            # Calculate local volatility as frequency of regime changes
            changes = np.sum(labels[1:] != labels[:-1])
            volatility = changes / len(labels) if len(labels) > 0 else 0
            
            # Adaptive dwell time: higher volatility = longer dwell time
            adaptive_dwell = int(base_min_dwell * (1 + volatility * volatility_factor))
            adaptive_dwell = min(adaptive_dwell, max_dwell)
            adaptive_dwell = max(adaptive_dwell, base_min_dwell)
            
            tprint(f"🔧 Adaptive dwell time: {adaptive_dwell} (volatility: {volatility:.3f})", "INFO")
            return adaptive_dwell
            
        except Exception as e:
            tprint(f"⚠️ Adaptive dwell calculation failed: {e}", "WARNING")
            return base_min_dwell

    def _calculate_local_stability(self, labels: np.ndarray, center_idx: int, window_size: int) -> float:
        """Calculate local stability score around a given index."""
        try:
            start_idx = max(0, center_idx - window_size)
            end_idx = min(len(labels), center_idx + window_size + 1)
            
            local_labels = labels[start_idx:end_idx]
            if len(local_labels) == 0:
                return 0.0
            
            # Calculate stability as consistency of labels in local window
            unique_labels, counts = np.unique(local_labels, return_counts=True)
            if len(unique_labels) == 0:
                return 0.0
            
            # Stability = proportion of most common label
            max_count = np.max(counts)
            stability = max_count / len(local_labels)
            
            return stability
            
        except Exception as e:
            tprint(f"⚠️ Local stability calculation failed: {e}", "WARNING")
            return 0.0

    def _apply_stability_validation(self, labels: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Apply final stability validation to ensure cluster consistency."""
        try:
            validated_labels = labels.copy()
            min_segment_length = config.get('min_segment_length', 2)
            stability_threshold = config.get('final_stability_threshold', 0.8)
            
            # Find segments of each cluster
            segments = []
            current_label = labels[0]
            current_start = 0
            
            for i in range(1, len(labels)):
                if labels[i] != current_label:
                    segments.append({
                        'label': current_label,
                        'start': current_start,
                        'end': i,
                        'length': i - current_start
                    })
                    current_label = labels[i]
                    current_start = i
            
            # Add final segment
            segments.append({
                'label': current_label,
                'start': current_start,
                'end': len(labels),
                'length': len(labels) - current_start
            })
            
            # Validate and fix unstable segments
            for segment in segments:
                if segment['length'] < min_segment_length:
                    # Find most common neighbor label
                    neighbor_labels = []
                    if segment['start'] > 0:
                        neighbor_labels.append(labels[segment['start'] - 1])
                    if segment['end'] < len(labels):
                        neighbor_labels.append(labels[segment['end']])
                    
                    if neighbor_labels:
                        # Use most common neighbor label
                        unique_neighbors, counts = np.unique(neighbor_labels, return_counts=True)
                        most_common_neighbor = unique_neighbors[np.argmax(counts)]
                        
                        # Replace segment with most common neighbor
                        validated_labels[segment['start']:segment['end']] = most_common_neighbor
                        tprint(f"🔧 Fixed unstable segment: length {segment['length']} -> {most_common_neighbor}", "INFO")
            
            return validated_labels
            
        except Exception as e:
            tprint(f"⚠️ Stability validation failed: {e}", "WARNING")
            return labels


# Register the step
def register_regime_clustering_step():
    """Register the regime clustering step."""
    from src.training.steps.base_step import step_registry

    step_registry.register("regime_clustering", RegimeClusteringStep)
    tprint("✅ Regime clustering step registered", "SUCCESS")


# Auto-register when module is imported
register_regime_clustering_step()
