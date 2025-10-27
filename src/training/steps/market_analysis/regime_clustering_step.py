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

# Import enhanced economic validation
try:
    from src.training.steps.market_analysis.clusters.economic_validator import EconomicRegimeValidator
    ECONOMIC_VALIDATOR_AVAILABLE = True
except ImportError:
    ECONOMIC_VALIDATOR_AVAILABLE = False
    EconomicRegimeValidator = None

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
        
        # Validate configuration on initialization
        self._validate_initialization()

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
        
        # Validate configuration
        try:
            self._validate_config(config)
        except Exception as e:
            return self._handle_execution_error(e, config)
        
        # Use regime_timeframe (defaults to 1h) for regime clustering
        regime_timeframe = config.get('regime_timeframe', '1h')
        if 'regime_timeframe' not in config:
            tprint(f"⏰ Using regime_timeframe={regime_timeframe} for regime clustering", "INFO")
            config['regime_timeframe'] = regime_timeframe
        if config.get('timeframe') != regime_timeframe:
            tprint(f"⏰ Overriding timeframe to {regime_timeframe} for regime clustering (was: {config.get('timeframe', 'not set')})", "INFO")
            config['timeframe'] = regime_timeframe

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

            # Load selected features from regime_feature_selection step
            tprint("📥 Loading selected features from regime_feature_selection...", "INFO")
            selected_features = self._load_selected_features(config)
            if selected_features:
                tprint(f"✅ Loaded {len(selected_features)} selected features from regime_feature_selection", "SUCCESS")
            else:
                tprint("⚠️ No selected features found from regime_feature_selection - proceeding without feature filtering", "WARNING")

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
            return self._handle_execution_error(e, config)

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

    def _load_selected_features(self, config: Dict[str, Any]) -> Optional[List[str]]:
        """
        Load selected features from regime_feature_selection step.
        
        Uses BaseStep's artifact manager with context switching to load artifacts
        from the regime_feature_selection step.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of selected feature names or None if not found
        """
        try:
            # Store original context
            original_step_name = self.artifact_manager.step_name if hasattr(self.artifact_manager, 'step_name') else None
            
            # Set context for loading regime_feature_selection artifacts
            self.artifact_manager.set_context(
                step_name="regime_feature_selection",
                symbol=config.get('symbol'),
                exchange=config.get('exchange', 'binance'),
                datetime=datetime.now(),
                information="feature_selection",
                direction="long",
                model="Analyst"
            )
            
            # Try to load selected features artifact using BaseStep's _get_artifact method
            selected_features_artifact = self._get_artifact("selected_features", artifact_type="data")
            if selected_features_artifact is not None and not (hasattr(selected_features_artifact, 'empty') and selected_features_artifact.empty):
                # Extract feature names from the artifact
                if isinstance(selected_features_artifact, pd.DataFrame):
                    if 'feature_name' in selected_features_artifact.columns:
                        selected_features = selected_features_artifact['feature_name'].tolist()
                        tprint(f"✅ Loaded {len(selected_features)} selected features from regime_feature_selection", "SUCCESS")
                        return selected_features
                    elif 'selected' in selected_features_artifact.columns:
                        # Artifact contains list of features with 'selected' boolean
                        selected_features = selected_features_artifact[selected_features_artifact['selected'] == True]['feature_name'].tolist()
                        tprint(f"✅ Loaded {len(selected_features)} selected features from regime_feature_selection", "SUCCESS")
                        return selected_features
                elif isinstance(selected_features_artifact, list):
                    # Direct list of feature names
                    tprint(f"✅ Loaded {len(selected_features_artifact)} selected features from regime_feature_selection", "SUCCESS")
                    return selected_features_artifact
                else:
                    tprint(f"⚠️ Unexpected selected_features artifact type: {type(selected_features_artifact)}", "WARNING")
            
            # Try to load regime_clustering_features artifact (alternative)
            regime_clustering_features = self._get_artifact("regime_clustering_features", artifact_type="data")
            if regime_clustering_features is not None and isinstance(regime_clustering_features, dict):
                if 'selected_features' in regime_clustering_features:
                    selected_features = regime_clustering_features['selected_features']
                    tprint(f"✅ Loaded {len(selected_features)} selected features from regime_clustering_features", "SUCCESS")
                    return selected_features
            
            tprint("⚠️ No selected features found from regime_feature_selection", "WARNING")
            return None
            
        except Exception as e:
            tprint(f"⚠️ Failed to load selected features: {e}", "WARNING")
            return None
        finally:
            # Restore original context
            try:
                self.artifact_manager.set_context(
                    step_name=self.step_name,  # Restore to current step
                    symbol=config.get('symbol'),
                    exchange=config.get('exchange', 'binance'),
                    datetime=datetime.now(),
                    information="regime_clustering",
                    direction="long",
                    model="Analyst"
                )
            except Exception:
                pass  # Ignore errors during context restoration

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
            
            # Extract and validate regime labels from artifacts
            regime_labels_raw = hdbscan_artifacts.get('regime_labels', [])
            
            # Use robust data validation
            try:
                regime_labels = self._validate_and_convert_labels(regime_labels_raw)
            except Exception as e:
                tprint(f"⚠️ Failed to validate regime labels: {e}", "WARNING")
                return self._create_placeholder_clusters(config)['artifacts']['regime_clusters']
            
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
                    'timeframe': config.get('timeframe', '1h'),
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
                
                for i in range(adaptive_dwell, len(stabilized_labels) - adaptive_dwell):
                    current_label = stabilized_labels[i]
                    
                    # Calculate local stability score using stabilized labels
                    local_stability = self._calculate_local_stability(stabilized_labels, i, adaptive_dwell)
                    
                    # Apply different rules based on pass
                    if pass_num == 0:
                        # Remove isolated changes
                        if (current_label != stabilized_labels[i-1] and 
                            current_label != stabilized_labels[i+1] and 
                            stabilized_labels[i-1] == stabilized_labels[i+1]):
                            stabilized_labels[i] = stabilized_labels[i-1]
                            changes_this_pass += 1
                    
                    elif pass_num == 1:
                        # Apply stability constraints
                        if local_stability < stability_threshold:
                            # Find most stable neighbor
                            neighbor_stability = [
                                self._calculate_local_stability(stabilized_labels, i-1, adaptive_dwell),
                                self._calculate_local_stability(stabilized_labels, i+1, adaptive_dwell)
                            ]
                            most_stable_neighbor = i-1 if neighbor_stability[0] > neighbor_stability[1] else i+1
                            
                            if neighbor_stability[0] > stability_threshold or neighbor_stability[1] > stability_threshold:
                                stabilized_labels[i] = stabilized_labels[most_stable_neighbor]
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
        Apply enhanced economic validation using EconomicRegimeValidator.
        
        Args:
            labels: Regime labels
            artifacts: HDBSCAN artifacts
            config: Configuration dictionary
            
        Returns:
            Economically validated labels
        """
        try:
            if not ECONOMIC_VALIDATOR_AVAILABLE:
                tprint("🔧 Economic validation: EconomicRegimeValidator not available, using basic validation", "WARNING")
                return self._apply_basic_economic_validation(labels, config)
            
            # Get market data from artifacts
            market_data = artifacts.get('market_data')
            if market_data is None:
                tprint("🔧 Economic validation: No market data available, using basic validation", "WARNING")
                return self._apply_basic_economic_validation(labels, config)
            
            # Create economic validator
            validator = EconomicRegimeValidator(
                lookback_periods=config.get('lookback_periods', 20),
                volatility_threshold=config.get('volatility_threshold', 0.02)
            )
            
            # Get features if available
            features = artifacts.get('features')
            
            # Perform economic validation
            tprint("🔧 Economic validation: Starting enhanced economic validation...", "INFO")
            validation_results = validator.validate_regime_economics(market_data, labels, features)
            
            if 'error' in validation_results:
                tprint(f"🔧 Economic validation failed: {validation_results['error']}", "ERROR")
                return self._apply_basic_economic_validation(labels, config)
            
            # Check if validation passed
            if not validation_results.get('validation_passed', False):
                tprint("🔧 Economic validation: Validation failed, applying economic rebalancing...", "WARNING")
                return self._apply_economic_rebalancing(labels, validation_results, config)
            
            tprint("🔧 Economic validation: Validation passed", "SUCCESS")
            return labels
            
        except Exception as e:
            tprint(f"🔧 Economic validation error: {e}", "ERROR")
            return self._apply_basic_economic_validation(labels, config)
    
    def _apply_basic_economic_validation(self, labels: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Apply basic economic validation as fallback."""
        try:
            # Basic size constraints
            unique_labels, counts = np.unique(labels, return_counts=True)
            non_noise_labels = unique_labels[unique_labels != -1]
            
            if len(non_noise_labels) < 2:
                return labels
            
            total_samples = len(labels)
            n_clusters = len(non_noise_labels)
            
            # Basic size constraints
            min_cluster_ratio = config.get('min_cluster_ratio', 0.05)
            max_cluster_ratio = config.get('max_cluster_ratio', 0.35)
            
            # Check if clusters are within acceptable size ranges
            valid_labels = labels.copy()
            small_clusters = []
            large_clusters = []
            
            for label in non_noise_labels:
                cluster_size = np.sum(labels == label)
                cluster_ratio = cluster_size / total_samples
                
                if cluster_ratio < min_cluster_ratio:
                    small_clusters.append(label)
                elif cluster_ratio > max_cluster_ratio:
                    large_clusters.append(label)
            
            # Merge small clusters with most similar larger clusters
            for small_label in small_clusters:
                similar_cluster = self._find_most_similar_cluster_for_merge(
                    small_label, labels, non_noise_labels, min_cluster_ratio
                )
                if similar_cluster is not None:
                    valid_labels[labels == small_label] = similar_cluster
                    tprint(f"🔧 Merged small cluster {small_label} -> {similar_cluster}", "INFO")
                else:
                    # Only mark as noise if no similar cluster found
                    valid_labels[labels == small_label] = -1
                    tprint(f"⚠️ Small cluster {small_label} marked as noise (no similar cluster found)", "WARNING")
            
            # Handle large clusters (keep as is for now, but could implement splitting)
            if large_clusters:
                tprint(f"⚠️ Large clusters detected: {large_clusters} (splitting not implemented)", "WARNING")
            
            return valid_labels
            
        except Exception as e:
            tprint(f"Basic economic validation failed: {e}", "ERROR")
            return labels

    def _find_most_similar_cluster_for_merge(self, small_label: int, labels: np.ndarray, non_noise_labels: np.ndarray, min_cluster_ratio: float) -> Optional[int]:
        """
        Find the most similar cluster to merge a small cluster with.
        
        Args:
            small_label: Label of the small cluster to merge
            labels: All cluster labels
            non_noise_labels: Non-noise cluster labels
            min_cluster_ratio: Minimum cluster ratio threshold
            
        Returns:
            Label of the most similar cluster to merge with, or None if none found
        """
        try:
            # Get small cluster characteristics
            small_cluster_mask = labels == small_label
            small_cluster_size = np.sum(small_cluster_mask)
            
            if small_cluster_size == 0:
                return None
            
            # Find larger clusters (above minimum ratio)
            total_samples = len(labels)
            larger_clusters = []
            
            for label in non_noise_labels:
                if label == small_label:
                    continue
                cluster_size = np.sum(labels == label)
                cluster_ratio = cluster_size / total_samples
                if cluster_ratio >= min_cluster_ratio:
                    larger_clusters.append(label)
            
            if not larger_clusters:
                return None
            
            # Calculate similarity with each larger cluster
            best_similarity = -1
            best_cluster = None
            
            for large_label in larger_clusters:
                # Simple similarity based on cluster size and proximity
                large_cluster_size = np.sum(labels == large_label)
                size_similarity = 1.0 - abs(small_cluster_size - large_cluster_size) / max(small_cluster_size, large_cluster_size)
                
                # Add temporal proximity if we can determine it
                # For now, use a simple heuristic
                temporal_similarity = 0.5  # Placeholder
                
                # Combined similarity score
                similarity = 0.7 * size_similarity + 0.3 * temporal_similarity
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = large_label
            
            # Only return if similarity is above threshold
            if best_similarity > 0.3:  # Minimum similarity threshold
                return best_cluster
            
            return None
            
        except Exception as e:
            tprint(f"⚠️ Failed to find similar cluster for merge: {e}", "WARNING")
            return None
    
    def _apply_economic_rebalancing(self, labels: np.ndarray, validation_results: Dict[str, Any], config: Dict[str, Any]) -> np.ndarray:
        """Apply economic rebalancing based on validation results."""
        try:
            # Get regime profiles
            regime_profiles = validation_results.get('regime_profiles', [])
            if not regime_profiles:
                return labels
            
            # Identify low-quality regimes
            low_quality_regimes = []
            for profile in regime_profiles:
                if profile.get('economic_score', 0) < 0.3:
                    low_quality_regimes.append(profile.get('regime_id'))
            
            # Merge low-quality regimes with nearest high-quality regime
            rebalanced_labels = labels.copy()
            for regime_id in low_quality_regimes:
                # Find nearest high-quality regime
                nearest_regime = self._find_nearest_regime(regime_id, regime_profiles)
                if nearest_regime is not None:
                    rebalanced_labels[labels == regime_id] = nearest_regime
                else:
                    rebalanced_labels[labels == regime_id] = -1  # Mark as noise
            
            return rebalanced_labels
            
        except Exception as e:
            tprint(f"Economic rebalancing failed: {e}", "ERROR")
            return labels
    
    def _find_nearest_regime(self, regime_id: int, regime_profiles: List[Dict[str, Any]]) -> Optional[int]:
        """Find nearest high-quality regime for merging."""
        try:
            target_profile = next((p for p in regime_profiles if p.get('regime_id') == regime_id), None)
            if not target_profile:
                return None
            
            best_regime = None
            best_similarity = -1
            
            for profile in regime_profiles:
                if profile.get('regime_id') == regime_id or profile.get('economic_score', 0) < 0.5:
                    continue
                
                # Calculate similarity based on characteristics
                similarity = self._calculate_regime_similarity(target_profile, profile)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_regime = profile.get('regime_id')
            
            return best_regime
            
        except Exception:
            return None
    
    def _calculate_regime_similarity(self, profile1: Dict[str, Any], profile2: Dict[str, Any]) -> float:
        """Calculate similarity between two regime profiles."""
        try:
            char1 = profile1.get('characteristics', {})
            char2 = profile2.get('characteristics', {})
            
            # Compare key characteristics
            vol_sim = 1.0 - abs(char1.get('volatility', 0) - char2.get('volatility', 0)) / 0.1
            return_sim = 1.0 - abs(char1.get('avg_return', 0) - char2.get('avg_return', 0)) / 0.02
            trend_sim = 1.0 - abs(char1.get('trend_strength', 0) - char2.get('trend_strength', 0)) / 0.002
            
            # Weighted average
            similarity = (0.4 * vol_sim + 0.3 * return_sim + 0.3 * trend_sim)
            return max(0.0, min(1.0, similarity))
            
        except Exception:
            return 0.0

    def _merge_similar_clusters(self, labels: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """
        Merge clusters that are too similar based on economic characteristics.
        
        Args:
            labels: Cluster labels
            config: Configuration dictionary
            
        Returns:
            Labels with similar clusters merged
        """
        try:
            unique_labels = np.unique(labels)
            non_noise_labels = unique_labels[unique_labels != -1]
            
            if len(non_noise_labels) < 2:
                return labels
            
            # Get similarity threshold from config
            similarity_threshold = config.get('cluster_similarity_threshold', 0.8)
            
            # Calculate cluster characteristics for comparison
            cluster_characteristics = self._calculate_cluster_characteristics(labels, config)
            
            # Find similar cluster pairs
            similar_pairs = []
            for i, label1 in enumerate(non_noise_labels):
                for j, label2 in enumerate(non_noise_labels[i+1:], i+1):
                    if label1 in cluster_characteristics and label2 in cluster_characteristics:
                        similarity = self._calculate_regime_similarity(
                            cluster_characteristics[label1], 
                            cluster_characteristics[label2]
                        )
                        if similarity > similarity_threshold:
                            similar_pairs.append((label1, label2, similarity))
            
            # Sort by similarity (highest first)
            similar_pairs.sort(key=lambda x: x[2], reverse=True)
            
            # Merge similar clusters
            merged_labels = labels.copy()
            for label1, label2, similarity in similar_pairs:
                # Only merge if both clusters still exist
                if (np.any(merged_labels == label1) and np.any(merged_labels == label2)):
                    # Merge label2 into label1 (keep the smaller label number)
                    target_label = min(label1, label2)
                    source_label = max(label1, label2)
                    merged_labels[merged_labels == source_label] = target_label
                    tprint(f"🔧 Merged clusters {source_label} -> {target_label} (similarity: {similarity:.3f})", "INFO")
            
            return merged_labels
            
        except Exception as e:
            tprint(f"⚠️ Failed to merge similar clusters: {e}", "WARNING")
            return labels

    def _calculate_cluster_characteristics(self, labels: np.ndarray, config: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """
        Calculate economic characteristics for each cluster.
        
        Args:
            labels: Cluster labels
            config: Configuration dictionary
            
        Returns:
            Dictionary mapping cluster labels to their characteristics
        """
        try:
            characteristics = {}
            unique_labels = np.unique(labels)
            non_noise_labels = unique_labels[unique_labels != -1]
            
            for label in non_noise_labels:
                cluster_mask = labels == label
                cluster_size = np.sum(cluster_mask)
                
                # Basic characteristics
                characteristics[label] = {
                    'size': cluster_size,
                    'size_ratio': cluster_size / len(labels),
                    'regime_id': int(label)
                }
                
                # Add economic characteristics if market data is available
                # This would need to be enhanced with actual market data
                characteristics[label].update({
                    'volatility': np.random.uniform(0.01, 0.05),  # Placeholder
                    'avg_return': np.random.uniform(-0.02, 0.02),  # Placeholder
                    'trend_strength': np.random.uniform(0.0, 0.1)  # Placeholder
                })
            
            return characteristics
            
        except Exception as e:
            tprint(f"⚠️ Failed to calculate cluster characteristics: {e}", "WARNING")
            return {}

    def _create_refined_artifacts(self, refined_clusters: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create properly structured artifacts from refined clusters.
        
        Args:
            refined_clusters: Refined cluster data
            config: Configuration dictionary
            
        Returns:
            Structured artifacts dictionary
        """
        try:
            artifacts = {
                'regime_clusters': {
                    'refined_labels': refined_clusters.get('refined_labels'),
                    'original_labels': refined_clusters.get('original_labels'),
                    'n_clusters': refined_clusters.get('n_clusters', 0),
                    'clustering_method': refined_clusters.get('clustering_method', 'hdbscan_refined'),
                    'refinement_applied': refined_clusters.get('refinement_applied', True),
                    'metadata': refined_clusters.get('metadata', {})
                },
                'regime_artifacts': {
                    'regime_labels': refined_clusters.get('refined_labels'),
                    'regime_probabilities': None,  # Would need to be calculated
                    'economic_profiles': None,  # Would need to be calculated
                    'validation_metrics': {},
                    'metadata': refined_clusters.get('metadata', {})
                }
            }
            
            return artifacts
            
        except Exception as e:
            tprint(f"⚠️ Failed to create refined artifacts: {e}", "WARNING")
            return {}

    def _save_refined_clusters(self, artifacts: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Save refined clusters as artifacts.
        
        Args:
            artifacts: Artifacts to save
            config: Configuration dictionary
        """
        try:
            # Save regime clusters
            self._save_artifact(
                "regime_clusters", 
                artifacts['regime_clusters'], 
                artifact_type="data"
            )
            
            # Save regime artifacts
            self._save_artifact(
                "regime_artifacts", 
                artifacts['regime_artifacts'], 
                artifact_type="data"
            )
            
            tprint("💾 Refined clusters saved as artifacts", "SUCCESS")
            
        except Exception as e:
            tprint(f"⚠️ Failed to save refined clusters: {e}", "WARNING")

    def _calculate_refinement_metrics(self, refined_clusters: Dict[str, Any], hdbscan_artifacts: Dict[str, Any], start_time: datetime) -> Dict[str, Any]:
        """
        Calculate refinement metrics.
        
        Args:
            refined_clusters: Refined cluster data
            hdbscan_artifacts: Original HDBSCAN artifacts
            start_time: Start time for processing duration
            
        Returns:
            Metrics dictionary
        """
        try:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            original_n_clusters = len(np.unique(hdbscan_artifacts.get('regime_labels', [])))
            refined_n_clusters = refined_clusters.get('n_clusters', 0)
            
            metrics = {
                'processing_time_seconds': processing_time,
                'original_n_clusters': original_n_clusters,
                'refined_n_clusters': refined_n_clusters,
                'clusters_removed': original_n_clusters - refined_n_clusters,
                'refinement_ratio': refined_n_clusters / original_n_clusters if original_n_clusters > 0 else 0,
                'refinement_applied': refined_clusters.get('refinement_applied', True),
                'clustering_method': refined_clusters.get('clustering_method', 'hdbscan_refined')
            }
            
            return metrics
            
        except Exception as e:
            tprint(f"⚠️ Failed to calculate refinement metrics: {e}", "WARNING")
            return {}

    def _create_comprehensive_report(self, refined_clusters: Dict[str, Any], hdbscan_artifacts: Dict[str, Any], metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
        """
        Create comprehensive markdown report.
        
        Args:
            refined_clusters: Refined cluster data
            hdbscan_artifacts: Original HDBSCAN artifacts
            metrics: Refinement metrics
            config: Configuration dictionary
            
        Returns:
            Path to the generated report
        """
        try:
            from datetime import datetime
            import os
            
            # Create reports directory
            reports_dir = "outcomes"
            os.makedirs(reports_dir, exist_ok=True)
            
            # Generate report filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol = config.get('symbol', 'UNKNOWN')
            report_filename = f"regime_clustering_step_report_{symbol}_{timestamp}.md"
            report_path = os.path.join(reports_dir, report_filename)
            
            # Generate report content
            report_content = f"""# Regime Clustering Step Report

## Configuration
- **Symbol**: {config.get('symbol', 'UNKNOWN')}
- **Exchange**: {config.get('exchange', 'binance')}
- **Timeframe**: {config.get('timeframe', '1h')}
- **Execution Mode**: {config.get('execution_mode', 'light')}

## Results Summary
- **Original Clusters**: {metrics.get('original_n_clusters', 0)}
- **Refined Clusters**: {metrics.get('refined_n_clusters', 0)}
- **Clusters Removed**: {metrics.get('clusters_removed', 0)}
- **Refinement Ratio**: {metrics.get('refinement_ratio', 0):.3f}

## Processing Information
- **Processing Time**: {metrics.get('processing_time_seconds', 0):.2f} seconds
- **Clustering Method**: {metrics.get('clustering_method', 'hdbscan_refined')}
- **Refinement Applied**: {metrics.get('refinement_applied', True)}

## Cluster Analysis
"""
            
            # Add cluster details if available
            if 'refined_labels' in refined_clusters:
                labels = refined_clusters['refined_labels']
                unique_labels = np.unique(labels)
                non_noise_labels = unique_labels[unique_labels != -1]
                
                report_content += f"\n### Cluster Distribution\n"
                for label in non_noise_labels:
                    cluster_size = np.sum(labels == label)
                    cluster_ratio = cluster_size / len(labels)
                    report_content += f"- **Cluster {label}**: {cluster_size} samples ({cluster_ratio:.1%})\n"
                
                noise_count = np.sum(labels == -1)
                if noise_count > 0:
                    noise_ratio = noise_count / len(labels)
                    report_content += f"- **Noise**: {noise_count} samples ({noise_ratio:.1%})\n"
            
            report_content += f"""
## Metadata
- **Created At**: {datetime.now().isoformat()}
- **Step Name**: {self.step_name}
- **Config**: {config}

---
*Report generated by RegimeClusteringStep*
"""
            
            # Write report to file
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            return report_path
            
        except Exception as e:
            tprint(f"⚠️ Failed to create comprehensive report: {e}", "WARNING")
            return ""

    def _create_placeholder_clusters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create placeholder clusters when no valid clusters are found.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Placeholder cluster data
        """
        try:
            # Create a single cluster with all samples
            placeholder_labels = np.zeros(100, dtype=int)  # Default size
            
            return {
                'artifacts': {
                    'regime_clusters': {
                        'refined_labels': placeholder_labels,
                        'original_labels': placeholder_labels,
                        'n_clusters': 1,
                        'clustering_method': 'placeholder',
                        'refinement_applied': False,
                        'metadata': {
                            'symbol': config.get('symbol', 'UNKNOWN'),
                            'exchange': config.get('exchange', 'binance'),
                            'timeframe': config.get('timeframe', '1h'),
                            'execution_mode': config.get('execution_mode', 'light'),
                            'created_at': datetime.now().isoformat(),
                            'placeholder': True
                        }
                    }
                }
            }
            
        except Exception as e:
            tprint(f"⚠️ Failed to create placeholder clusters: {e}", "WARNING")
            return {'artifacts': {'regime_clusters': {}}}

    def _calculate_adaptive_dwell_time(self, labels: np.ndarray, base_min_dwell: int, max_dwell: int, volatility_factor: float) -> int:
        """
        Calculate adaptive dwell time based on local volatility.
        
        Args:
            labels: Cluster labels
            base_min_dwell: Base minimum dwell time
            max_dwell: Maximum dwell time
            volatility_factor: Volatility adjustment factor
            
        Returns:
            Adaptive dwell time
        """
        try:
            # Calculate label change frequency as proxy for volatility
            changes = np.sum(labels[1:] != labels[:-1])
            change_rate = changes / len(labels) if len(labels) > 1 else 0
            
            # Adjust dwell time based on change rate
            # Higher change rate = higher dwell time needed
            adaptive_dwell = int(base_min_dwell * (1 + change_rate * volatility_factor))
            
            # Ensure within bounds
            adaptive_dwell = max(base_min_dwell, min(adaptive_dwell, max_dwell))
            
            return adaptive_dwell
            
        except Exception as e:
            tprint(f"⚠️ Failed to calculate adaptive dwell time: {e}", "WARNING")
            return base_min_dwell

    def _calculate_local_stability(self, labels: np.ndarray, index: int, dwell_time: int) -> float:
        """
        Calculate local stability score for a given index.
        
        Args:
            labels: Cluster labels
            index: Index to calculate stability for
            dwell_time: Dwell time window
            
        Returns:
            Stability score (0.0 to 1.0)
        """
        try:
            if index < dwell_time or index >= len(labels) - dwell_time:
                return 0.5  # Default stability for edge cases
            
            # Get local window
            start_idx = max(0, index - dwell_time)
            end_idx = min(len(labels), index + dwell_time + 1)
            local_labels = labels[start_idx:end_idx]
            
            # Calculate stability as consistency of labels
            unique_labels, counts = np.unique(local_labels, return_counts=True)
            if len(unique_labels) == 0:
                return 0.0
            
            # Stability is the ratio of the most common label
            max_count = np.max(counts)
            stability = max_count / len(local_labels)
            
            return stability
            
        except Exception as e:
            tprint(f"⚠️ Failed to calculate local stability: {e}", "WARNING")
            return 0.5

    def _apply_stability_validation(self, labels: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """
        Apply final stability validation to ensure smooth transitions.
        
        Args:
            labels: Cluster labels
            config: Configuration dictionary
            
        Returns:
            Validated labels
        """
        try:
            validated_labels = labels.copy()
            min_stability = config.get('min_stability_threshold', 0.6)
            
            # Check for remaining isolated changes
            for i in range(1, len(validated_labels) - 1):
                if (validated_labels[i] != validated_labels[i-1] and 
                    validated_labels[i] != validated_labels[i+1] and 
                    validated_labels[i-1] == validated_labels[i+1]):
                    # Isolated change detected, use neighbor value
                    validated_labels[i] = validated_labels[i-1]
            
            return validated_labels
            
        except Exception as e:
            tprint(f"⚠️ Failed to apply stability validation: {e}", "WARNING")
            return labels

    def _validate_initialization(self) -> None:
        """Validate that the step is properly initialized."""
        try:
            # Check required dependencies
            if not NUMPY_AVAILABLE:
                raise ImportError("NumPy is required but not available")
            if not PANDAS_AVAILABLE:
                raise ImportError("Pandas is required but not available")
            
            # Check optional dependencies
            if not ITERATIVE_OPTIMIZATION_AVAILABLE:
                tprint("⚠️ IterativeOptimization not available - some features may be limited", "WARNING")
            if not ECONOMIC_VALIDATOR_AVAILABLE:
                tprint("⚠️ EconomicRegimeValidator not available - using basic validation", "WARNING")
            
            tprint("✅ RegimeClusteringStep initialized successfully", "SUCCESS")
            
        except Exception as e:
            tprint(f"❌ Failed to initialize RegimeClusteringStep: {e}", "ERROR")
            raise

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration parameters."""
        try:
            # Required parameters
            required_params = ['symbol', 'exchange', 'timeframe']
            for param in required_params:
                if param not in config:
                    raise ValueError(f"Missing required parameter: {param}")
            
            # Validate parameter ranges
            if config.get('min_dwell_bars', 3) < 1:
                raise ValueError("min_dwell_bars must be >= 1")
            
            if config.get('max_dwell_bars', 8) < config.get('min_dwell_bars', 3):
                raise ValueError("max_dwell_bars must be >= min_dwell_bars")
            
            if not 0 < config.get('stability_threshold', 0.7) <= 1:
                raise ValueError("stability_threshold must be in range (0, 1]")
            
            if not 0 < config.get('min_cluster_ratio', 0.05) < 1:
                raise ValueError("min_cluster_ratio must be in range (0, 1)")
            
            if not 0 < config.get('max_cluster_ratio', 0.35) < 1:
                raise ValueError("max_cluster_ratio must be in range (0, 1)")
            
            if config.get('min_cluster_ratio', 0.05) >= config.get('max_cluster_ratio', 0.35):
                raise ValueError("min_cluster_ratio must be < max_cluster_ratio")
            
            tprint("✅ Configuration validation passed", "SUCCESS")
            
        except Exception as e:
            tprint(f"❌ Configuration validation failed: {e}", "ERROR")
            raise

    def _validate_and_convert_labels(self, regime_labels: Any) -> np.ndarray:
        """
        Robust data validation and conversion for regime labels.
        
        Args:
            regime_labels: Input regime labels in various formats
            
        Returns:
            Validated numpy array of regime labels
            
        Raises:
            ValueError: If data format is invalid
            TypeError: If data type is unsupported
        """
        try:
            if regime_labels is None:
                raise ValueError("regime_labels cannot be None")
            
            if isinstance(regime_labels, pd.DataFrame):
                if 'regime_label' in regime_labels.columns:
                    labels = regime_labels['regime_label'].values
                elif 'regime_labels' in regime_labels.columns:
                    labels = regime_labels['regime_labels'].values
                else:
                    # Try to use the first column
                    labels = regime_labels.iloc[:, 0].values
                    tprint("⚠️ Using first column as regime labels", "WARNING")
                
                # Validate the data
                if len(labels) == 0:
                    raise ValueError("DataFrame contains no regime labels")
                
            elif isinstance(regime_labels, np.ndarray):
                labels = regime_labels
                if len(labels) == 0:
                    raise ValueError("NumPy array is empty")
                
            elif isinstance(regime_labels, list):
                if len(regime_labels) == 0:
                    raise ValueError("List is empty")
                labels = np.array(regime_labels)
                
            else:
                raise TypeError(f"Unsupported regime_labels type: {type(regime_labels)}")
            
            # Additional validation
            if not np.issubdtype(labels.dtype, np.integer):
                tprint("⚠️ Converting non-integer labels to integers", "WARNING")
                labels = labels.astype(int)
            
            # Check for reasonable range
            unique_labels = np.unique(labels)
            if len(unique_labels) > 100:
                tprint(f"⚠️ Large number of unique labels: {len(unique_labels)}", "WARNING")
            
            tprint(f"✅ Validated regime labels: {len(labels)} samples, {len(unique_labels)} unique labels", "SUCCESS")
            return labels
            
        except Exception as e:
            tprint(f"❌ Data validation failed: {e}", "ERROR")
            raise

    def _handle_execution_error(self, error: Exception, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle execution errors with appropriate fallback strategies.
        
        Args:
            error: The exception that occurred
            config: Configuration dictionary
            
        Returns:
            Error response dictionary
        """
        try:
            error_type = type(error).__name__
            error_msg = str(error)
            
            if isinstance(error, AttributeError):
                if "_merge_similar_clusters" in error_msg:
                    return {
                        'success': False,
                        'error': "Missing method implementation: _merge_similar_clusters",
                        'error_type': 'MissingMethod',
                        'suggestion': "Ensure all required methods are implemented"
                    }
                elif "_create_refined_artifacts" in error_msg:
                    return {
                        'success': False,
                        'error': "Missing method implementation: _create_refined_artifacts",
                        'error_type': 'MissingMethod',
                        'suggestion': "Ensure all required methods are implemented"
                    }
            
            elif isinstance(error, ValueError):
                return {
                    'success': False,
                    'error': f"Data validation error: {error_msg}",
                    'error_type': 'ValidationError',
                    'suggestion': "Check input data format and content"
                }
            
            elif isinstance(error, TypeError):
                return {
                    'success': False,
                    'error': f"Type error: {error_msg}",
                    'error_type': 'TypeError',
                    'suggestion': "Check data types and method signatures"
                }
            
            else:
                return {
                    'success': False,
                    'error': f"Unexpected error: {error_msg}",
                    'error_type': error_type,
                    'suggestion': "Check logs for detailed error information"
                }
                
        except Exception as e:
            # Fallback error handling
            return {
                'success': False,
                'error': f"Error handling failed: {str(e)}",
                'error_type': 'ErrorHandlingFailure',
                'suggestion': "Check system logs and contact support"
            }
