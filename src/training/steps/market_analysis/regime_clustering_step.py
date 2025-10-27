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
            for label in non_noise_labels:
                cluster_size = np.sum(labels == label)
                cluster_ratio = cluster_size / total_samples
                
                if cluster_ratio < min_cluster_ratio:
                    # Merge small clusters with nearest neighbor
                    valid_labels[labels == label] = -1  # Mark as noise
                elif cluster_ratio > max_cluster_ratio:
                    # Split large clusters (simplified approach)
                    pass  # Keep as is for now
            
            return valid_labels
            
        except Exception as e:
            tprint(f"Basic economic validation failed: {e}", "ERROR")
            return labels
    
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
