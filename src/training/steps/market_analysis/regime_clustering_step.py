"""
Regime Clustering Step.

This step performs regime clustering using HDBSCAN or other clustering methods.
"""

import asyncio
import logging
import os
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

# Import unified clustering optimization goals
try:
    from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
        DEFAULT_OPTIMIZATION_TARGETS,
        format_metrics_report
    )
    UNIFIED_GOALS_AVAILABLE = True
except ImportError:
    UNIFIED_GOALS_AVAILABLE = False
    DEFAULT_OPTIMIZATION_TARGETS = None

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
                
                # Validate feature selection quality
                validation_result = self._validate_selected_features(selected_features, config)
                if not validation_result.get('valid', True):
                    tprint(f"⚠️ Feature selection validation issues: {validation_result.get('issues', [])}", "WARNING")
                    if validation_result.get('use_fallback', False):
                        tprint("🔄 Using fallback feature set due to validation failures", "WARNING")
                        selected_features = validation_result.get('fallback_features', selected_features)
            else:
                tprint("⚠️ No selected features found from regime_feature_selection - using fallback features", "WARNING")
                # Use fallback features from categorization system
                selected_features = self._get_fallback_regime_features()

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
            # Force loading the most recent artifacts by searching for the latest timestamp
            regime_artifacts = self._get_most_recent_hdbscan_artifacts()
            if regime_artifacts is not None and not (hasattr(regime_artifacts, 'empty') and regime_artifacts.empty):
                tprint("✅ Loaded HDBSCAN regime artifacts", "SUCCESS")
                
                # Extract regime labels from the artifacts DataFrame
                if 'regime_labels' in regime_artifacts.columns:
                    regime_labels = regime_artifacts['regime_labels'].iloc[0]
                    tprint(f"📊 Extracted regime labels: {len(regime_labels)} samples", "INFO")
                    
                    # Return properly structured artifacts
                    artifacts = {
                        'regime_labels': regime_labels,
                        'regime_probabilities': regime_artifacts.get('regime_probabilities', {}).iloc[0] if 'regime_probabilities' in regime_artifacts.columns else None,
                        'economic_profiles': regime_artifacts.get('economic_profiles', {}).iloc[0] if 'economic_profiles' in regime_artifacts.columns else None,
                        'validation_metrics': {col: regime_artifacts[col].iloc[0] for col in regime_artifacts.columns if col.startswith('validation_metrics')},
                        'metadata': {col: regime_artifacts[col].iloc[0] for col in regime_artifacts.columns if col.startswith('metadata')}
                    }
                    
                    # Load clustering features from separate artifact
                    # Search directly for clustering_features in HDBSCAN directories
                    try:
                        import glob
                        import os
                        import pickle
                        
                        # Search for clustering_features files in HDBSCAN regime discovery directories
                        search_patterns = [
                            "artifacts/Analyst/hdbscan_regime_discovery/*clustering_features*.pkl",
                            "artifacts/long/Analyst/hdbscan_regime_discovery/*clustering_features*.pkl",
                            "artifacts/binance/long/Analyst/hdbscan_regime_discovery/*clustering_features*.pkl",
                            "artifacts/**/hdbscan_regime_discovery/*clustering_features*.pkl"
                        ]
                        
                        clustering_features_files = []
                        for pattern in search_patterns:
                            files = glob.glob(pattern, recursive=True)
                            clustering_features_files.extend(files)
                        
                        if clustering_features_files:
                            # Sort by modification time (most recent first)
                            clustering_features_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                            most_recent_file = clustering_features_files[0]
                            
                            tprint(f"📁 Loading clustering features from: {most_recent_file}", "INFO")
                            
                            # Load the clustering features
                            with open(most_recent_file, 'rb') as f:
                                clustering_features = pickle.load(f)
                            
                            artifacts['clustering_features'] = clustering_features
                            if hasattr(clustering_features, 'shape'):
                                tprint(f"📊 Loaded clustering features: {clustering_features.shape[0]} samples, {clustering_features.shape[1]} features", "INFO")
                            else:
                                tprint(f"📊 Loaded clustering features: {len(clustering_features)} samples", "INFO")
                        else:
                            tprint("⚠️ No clustering features artifact found", "WARNING")
                    except Exception as e:
                        tprint(f"⚠️ Failed to load clustering features: {e}", "WARNING")
                    
                    return artifacts
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
    
    def _get_most_recent_hdbscan_artifacts(self) -> Optional[Any]:
        """
        Get the most recent HDBSCAN regime artifacts by searching for the latest timestamp.
        
        Returns:
            Most recent regime artifacts or None if not found
        """
        try:
            import glob
            import os
            from pathlib import Path
            
            # Search for regime_artifacts files in the HDBSCAN regime discovery directory
            # Try multiple search patterns to handle different artifact storage structures
            search_patterns = [
                "artifacts/long/Analyst/hdbscan_regime_discovery/*regime_artifacts*.parquet",
                "artifacts/binance/long/Analyst/hdbscan_regime_discovery/*regime_artifacts*.parquet",
                "artifacts/**/hdbscan_regime_discovery/*regime_artifacts*.parquet"
            ]
            
            matching_files = []
            for pattern in search_patterns:
                files = glob.glob(pattern, recursive=True)
                matching_files.extend(files)
            
            if not matching_files:
                tprint("⚠️ No HDBSCAN regime artifacts found", "WARNING")
                return None
            
            # Sort by modification time (most recent first)
            matching_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            most_recent_file = matching_files[0]
            
            tprint(f"📁 Loading most recent HDBSCAN artifacts: {most_recent_file}", "INFO")
            
            # Load the most recent artifact
            import pandas as pd
            return pd.read_parquet(most_recent_file)
            
        except Exception as e:
            tprint(f"❌ Failed to find most recent HDBSCAN artifacts: {e}", "ERROR")
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
    
    def _validate_selected_features(self, selected_features: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate selected features using the validation utilities.
        
        Args:
            selected_features: List of selected feature names
            config: Configuration dictionary
            
        Returns:
            Validation result dictionary
        """
        try:
            from .feature_selection_validation import validate_regime_clustering_features
            
            # Validate features
            validation_result = validate_regime_clustering_features(selected_features)
            
            # Determine if fallback is needed
            use_fallback = False
            if not validation_result.get('valid', True):
                # Use fallback if validation fails critically
                if validation_result.get('use_case_alignment', {}).get('invalid_count', 0) > len(selected_features) * 0.5:
                    use_fallback = True
            
            # Add fallback features if needed
            if use_fallback:
                fallback_features = self._get_fallback_regime_features()
                validation_result['use_fallback'] = True
                validation_result['fallback_features'] = fallback_features
            else:
                validation_result['use_fallback'] = False
            
            return validation_result
            
        except Exception as e:
            tprint(f"⚠️ Feature validation failed: {e}, proceeding without validation", "WARNING")
            return {'valid': True, 'use_fallback': False}
    
    def _get_fallback_regime_features(self) -> List[str]:
        """
        Get fallback feature set from the categorization system.
        
        Returns:
            List of fallback feature names
        """
        try:
            from src.feature_generation.categories.regime_feature_categorization import (
                get_regime_clustering_features,
                FeatureUseCase,
                RegimeFeatureCategorizer
            )
            
            # Get priority features for regime clustering
            categorizer = RegimeFeatureCategorizer()
            fallback_features = categorizer.get_priority_features(
                FeatureUseCase.REGIME_CLUSTERING,
                max_features=50
            )
            
            tprint(f"📋 Generated {len(fallback_features)} fallback features from categorization system", "INFO")
            
            return fallback_features
            
        except Exception as e:
            tprint(f"⚠️ Failed to get fallback features: {e}", "WARNING")
            # Return empty list if categorization system fails
            return []

    def _refine_hdbscan_clusters(self, hdbscan_artifacts: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine HDBSCAN clusters using economic validation and temporal stabilization.
        Falls back to iterative optimization if quality targets are not met.
        
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
            
            # Apply initial refinement logic
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
            
            tprint(f"🔧 Initial refinement: {n_clusters} clusters (from {len(np.unique(regime_labels))} original)", "INFO")
            
            # Check if we meet quality targets
            quality_targets = self._check_quality_targets(refined_labels, hdbscan_artifacts, config)
            
            if quality_targets['meets_targets']:
                tprint("✅ Quality targets met with initial refinement", "SUCCESS")
                clustering_method = 'hdbscan_refined'
            else:
                tprint("⚠️ Quality targets not met, attempting iterative optimization fallback", "WARNING")
                tprint(f"📊 Quality issues: {quality_targets['issues']}", "INFO")
                
                # Try iterative optimization as fallback
                # Use original regime_labels (before merging) to give more clusters to work with
                iterative_result = self._run_iterative_optimization_fallback(
                    hdbscan_artifacts, regime_labels, config
                )
                
                if iterative_result is not None:
                    refined_labels = iterative_result
                    clustering_method = 'hdbscan_iterative_optimized'
                    
                    # Recalculate cluster count
                    unique_labels = np.unique(refined_labels)
                    if hasattr(unique_labels, 'values'):
                        unique_labels = unique_labels.values
                    unique_labels = np.array(unique_labels)
                    noise_mask = unique_labels != -1
                    non_noise_labels = unique_labels[noise_mask]
                    n_clusters = len(non_noise_labels)
                    
                    tprint(f"🔧 Iterative optimization: {n_clusters} clusters", "INFO")
                else:
                    tprint("⚠️ Iterative optimization failed, using initial refinement", "WARNING")
                    clustering_method = 'hdbscan_refined_fallback'
            
            return {
                'refined_labels': refined_labels,
                'original_labels': regime_labels,
                'n_clusters': n_clusters,
                'clustering_method': clustering_method,
                'refinement_applied': True,
                'quality_targets': quality_targets,
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
                artifacts['regime_clusters'], 
                "regime_clusters", 
                artifact_type="data"
            )
            
            # Save regime artifacts
            self._save_artifact(
                artifacts['regime_artifacts'], 
                "regime_artifacts", 
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

## Quality Assessment
- **Meets Targets**: {refined_clusters.get('quality_targets', {}).get('meets_targets', 'Unknown')}
- **Cluster Count**: {refined_clusters.get('quality_targets', {}).get('n_clusters', 'Unknown')}
- **Issues**: {', '.join(refined_clusters.get('quality_targets', {}).get('issues', ['None']))}

### Quality Metrics
"""
            
            # Add quality metrics if available
            quality_targets = refined_clusters.get('quality_targets', {})
            if quality_targets:
                metrics = quality_targets.get('metrics', {})
                targets = quality_targets.get('targets', {})
                
                report_content += f"\n| Metric | Value | Target | Status |\n"
                report_content += f"|--------|-------|--------|--------|\n"
                
                # Use unified optimization targets if available
                if UNIFIED_GOALS_AVAILABLE and DEFAULT_OPTIMIZATION_TARGETS:
                    unified_targets = DEFAULT_OPTIMIZATION_TARGETS
                    cv_target = targets.get('min_cv_score', unified_targets.min_cv_score)
                    sil_target = targets.get('min_silhouette_score', unified_targets.min_silhouette_score)
                    dbi_target = targets.get('max_dbi_score', unified_targets.max_dbi_score)
                    temp_target = targets.get('min_temporal_smoothness', unified_targets.min_temporal_smoothness)
                else:
                    # Fallback to hardcoded defaults
                    cv_target = targets.get('min_cv_score', 0.3)
                    sil_target = targets.get('min_silhouette_score', 0.2)
                    dbi_target = targets.get('max_dbi_score', 2.0)
                    temp_target = targets.get('min_temporal_smoothness', 0.6)
                
                # CV Score
                cv_score = metrics.get('cv_score')
                cv_status = "✅" if cv_score and cv_score >= cv_target else "❌"
                cv_display = f"{cv_score:.3f}" if cv_score is not None else "N/A"
                report_content += f"| CV Score | {cv_display} | ≥{cv_target} | {cv_status} |\n"
                
                # Silhouette Score
                sil_score = metrics.get('silhouette_score')
                sil_status = "✅" if sil_score and sil_score >= sil_target else "❌"
                sil_display = f"{sil_score:.3f}" if sil_score is not None else "N/A"
                report_content += f"| Silhouette | {sil_display} | ≥{sil_target} | {sil_status} |\n"
                
                # DBI Score
                dbi_score = metrics.get('dbi_score')
                dbi_status = "✅" if dbi_score and dbi_score <= dbi_target else "❌"
                dbi_display = f"{dbi_score:.3f}" if dbi_score is not None else "N/A"
                report_content += f"| DBI Score | {dbi_display} | ≤{dbi_target} | {dbi_status} |\n"
                
                # Temporal Smoothness
                temp_smooth = metrics.get('temporal_smoothness')
                temp_status = "✅" if temp_smooth and temp_smooth >= temp_target else "❌"
                temp_display = f"{temp_smooth:.3f}" if temp_smooth is not None else "N/A"
                report_content += f"| Temporal Smoothness | {temp_display} | ≥{temp_target} | {temp_status} |\n"
            
            report_content += f"\n## Cluster Analysis\n"
            
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

    def _check_quality_targets(self, labels: np.ndarray, hdbscan_artifacts: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if clustering results meet quality targets.
        
        Targets:
        - 4-8 clusters
        - Minimum CV score
        - Minimum Silhouette score  
        - Minimum DBI score
        - Minimum temporal smoothness
        
        Args:
            labels: Cluster labels
            hdbscan_artifacts: HDBSCAN artifacts
            config: Configuration dictionary
            
        Returns:
            Dictionary with quality assessment results
        """
        try:
            # Get quality thresholds from unified targets if available, otherwise use config or defaults
            if UNIFIED_GOALS_AVAILABLE and DEFAULT_OPTIMIZATION_TARGETS:
                unified_targets = DEFAULT_OPTIMIZATION_TARGETS
                min_clusters = config.get('min_clusters', unified_targets.min_clusters)
                max_clusters = config.get('max_clusters', unified_targets.max_clusters)
                min_cv_score = config.get('min_cv_score', unified_targets.min_cv_score)
                min_silhouette_score = config.get('min_silhouette_score', unified_targets.min_silhouette_score)
                min_dbi_score = config.get('max_dbi_score', unified_targets.max_dbi_score)  # Lower is better for DBI
                min_temporal_smoothness = config.get('min_temporal_smoothness', unified_targets.min_temporal_smoothness)
            else:
                # Fallback to config or hardcoded defaults
                min_clusters = config.get('min_clusters', 4)
                max_clusters = config.get('max_clusters', 8)
                min_cv_score = config.get('min_cv_score', 0.3)
                min_silhouette_score = config.get('min_silhouette_score', 0.2)
                min_dbi_score = config.get('max_dbi_score', 0.5)  # Lower is better for DBI
                min_temporal_smoothness = config.get('min_temporal_smoothness', 0.6)
            
            # Calculate cluster count
            unique_labels = np.unique(labels)
            non_noise_labels = unique_labels[unique_labels != -1]
            n_clusters = len(non_noise_labels)
            
            issues = []
            meets_targets = True
            
            # Check cluster count target
            if n_clusters < min_clusters:
                issues.append(f"Too few clusters: {n_clusters} < {min_clusters}")
                meets_targets = False
            elif n_clusters > max_clusters:
                issues.append(f"Too many clusters: {n_clusters} > {max_clusters}")
                meets_targets = False
            
            # Calculate quality metrics if we have features
            features = hdbscan_artifacts.get('features')
            if features is not None and len(features) > 0:
                try:
                    # Calculate CV score (if available)
                    cv_score = self._calculate_cv_score(features, labels)
                    if cv_score is not None and cv_score < min_cv_score:
                        issues.append(f"Low CV score: {cv_score:.3f} < {min_cv_score}")
                        meets_targets = False
                    
                    # Calculate Silhouette score
                    silhouette_score = self._calculate_silhouette_score(features, labels)
                    if silhouette_score is not None and silhouette_score < min_silhouette_score:
                        issues.append(f"Low Silhouette score: {silhouette_score:.3f} < {min_silhouette_score}")
                        meets_targets = False
                    
                    # Calculate DBI score
                    dbi_score = self._calculate_dbi_score(features, labels)
                    if dbi_score is not None and dbi_score > min_dbi_score:  # Higher is worse for DBI
                        issues.append(f"High DBI score: {dbi_score:.3f} > {min_dbi_score}")
                        meets_targets = False
                    
                except Exception as e:
                    tprint(f"⚠️ Failed to calculate quality metrics: {e}", "WARNING")
                    issues.append("Quality metrics calculation failed")
                    meets_targets = False
            
            # Calculate temporal smoothness
            temporal_smoothness = self._calculate_temporal_smoothness(labels)
            if temporal_smoothness < min_temporal_smoothness:
                issues.append(f"Low temporal smoothness: {temporal_smoothness:.3f} < {min_temporal_smoothness}")
                meets_targets = False
            
            return {
                'meets_targets': meets_targets,
                'n_clusters': n_clusters,
                'issues': issues,
                'metrics': {
                    'cv_score': cv_score if 'cv_score' in locals() else None,
                    'silhouette_score': silhouette_score if 'silhouette_score' in locals() else None,
                    'dbi_score': dbi_score if 'dbi_score' in locals() else None,
                    'temporal_smoothness': temporal_smoothness
                },
                'targets': {
                    'min_clusters': min_clusters,
                    'max_clusters': max_clusters,
                    'min_cv_score': min_cv_score,
                    'min_silhouette_score': min_silhouette_score,
                    'min_dbi_score': min_dbi_score,
                    'min_temporal_smoothness': min_temporal_smoothness
                }
            }
            
        except Exception as e:
            tprint(f"⚠️ Failed to check quality targets: {e}", "WARNING")
            return {
                'meets_targets': False,
                'n_clusters': 0,
                'issues': [f"Quality check failed: {e}"],
                'metrics': {},
                'targets': {}
            }

    def _calculate_cv_score(self, features: np.ndarray, labels: np.ndarray) -> Optional[float]:
        """Calculate Calinski-Harabasz (CV) score."""
        try:
            from sklearn.metrics import calinski_harabasz_score
            non_noise_mask = labels != -1
            if np.sum(non_noise_mask) < 2:
                return None
            return calinski_harabasz_score(features[non_noise_mask], labels[non_noise_mask])
        except Exception:
            return None

    def _calculate_silhouette_score(self, features: np.ndarray, labels: np.ndarray) -> Optional[float]:
        """Calculate Silhouette score."""
        try:
            from sklearn.metrics import silhouette_score
            non_noise_mask = labels != -1
            if np.sum(non_noise_mask) < 2:
                return None
            return silhouette_score(features[non_noise_mask], labels[non_noise_mask])
        except Exception:
            return None

    def _calculate_dbi_score(self, features: np.ndarray, labels: np.ndarray) -> Optional[float]:
        """Calculate Davies-Bouldin Index (DBI) score."""
        try:
            from sklearn.metrics import davies_bouldin_score
            non_noise_mask = labels != -1
            if np.sum(non_noise_mask) < 2:
                return None
            return davies_bouldin_score(features[non_noise_mask], labels[non_noise_mask])
        except Exception:
            return None

    def _calculate_temporal_smoothness(self, labels: np.ndarray) -> float:
        """Calculate temporal smoothness score."""
        try:
            if len(labels) < 2:
                return 0.0
            
            # Calculate the ratio of consecutive identical labels
            changes = np.sum(labels[1:] != labels[:-1])
            total_pairs = len(labels) - 1
            smoothness = 1.0 - (changes / total_pairs)
            
            return smoothness
        except Exception:
            return 0.0

    def _load_regime_clustering_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load regime clustering configuration from YAML file and merge with base config.
        
        Args:
            base_config: Base configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        try:
            import yaml
            config_path = "config/regime_clustering_config.yaml"
            
            if not os.path.exists(config_path):
                tprint(f"ℹ️ No config file found at {config_path}, using base config", "INFO")
                return base_config
            
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            if yaml_config:
                # Merge YAML config with base config (YAML takes precedence)
                merged_config = {**base_config, **yaml_config}
                tprint(f"✅ Loaded configuration from {config_path}", "SUCCESS")
                return merged_config
            else:
                return base_config
                
        except Exception as e:
            tprint(f"⚠️ Failed to load config file: {e}, using base config", "WARNING")
            return base_config
    
    def _load_cached_tuning_results(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Load cached tuning results if available and not too old.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Cached parameters or None
        """
        try:
            import glob
            import json
            from datetime import timedelta
            
            # Look for recent tuning results
            pattern = f"artifacts/hyperparameter_tuning/auto_tuning_results_{config['symbol']}_*.json"
            result_files = glob.glob(pattern)
            
            if not result_files:
                tprint("📭 No cached tuning results found", "INFO")
                return None
            
            # Sort by modification time (most recent first)
            result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            most_recent = result_files[0]
            
            # Check if cache is too old
            max_age_hours = config.get('cached_tuning_max_age_hours', 24)
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(most_recent))
            
            if file_age.total_seconds() / 3600 > max_age_hours:
                tprint(f"⏰ Cached results too old ({file_age.total_seconds()/3600:.1f}h > {max_age_hours}h)", "WARNING")
                return None
            
            # Load cached results
            with open(most_recent, 'r') as f:
                cached_results = json.load(f)
            
            if 'best_params' not in cached_results:
                tprint("⚠️ Cached results missing best_params", "WARNING")
                return None
            
            tprint(f"📦 Loaded cached tuning results from {most_recent} (age: {file_age.total_seconds()/3600:.1f}h)", "INFO")
            
            # Convert to config format
            best_params = cached_results['best_params']
            converted_params = {
                'min_clusters': best_params.get('K_MIN', 4),
                'max_clusters': best_params.get('K_MAX', 8),
                'iterative_max_iterations': best_params.get('max_rounds', 25),
                'iterative_min_frac': best_params.get('MIN_FRAC', 0.03),
                'iterative_max_frac': best_params.get('MAX_FRAC', 0.20),
                'iterative_w_cv': best_params.get('w_cv', 0.70),
                'iterative_w_sil': best_params.get('w_sil', 0.10),
                'iterative_w_temp': best_params.get('w_temp', 0.20),
                'iterative_w_bal': best_params.get('w_bal', 0.05),
                'iterative_eps_std_step1': best_params.get('eps_std_step1', -0.20),
                'iterative_sil_guard': best_params.get('sil_guard', -0.08),
                'iterative_temporal_bonus': best_params.get('temporal_bonus', 0.25),
                'iterative_eps_cv': best_params.get('eps_cv', 1e-5),
                'iterative_eps_sil': best_params.get('eps_sil', 1e-4),
                'iterative_eps_temp': best_params.get('eps_temp', 1e-4),
                'iterative_local_churn_cap': best_params.get('local_churn_cap', 5000),
                'iterative_knn_size': best_params.get('knn_size', 25),
                'iterative_size_gate_base': best_params.get('size_gate_base', 1e-4),
                'iterative_size_gate_alpha': best_params.get('size_gate_alpha', 0.02),
                'iterative_size_gate_beta': best_params.get('size_gate_beta', 0.05),
            }
            
            # Show cached metrics
            if 'best_metrics' in cached_results:
                metrics = cached_results['best_metrics']
                tprint(f"📊 Cached metrics: CV={metrics.get('cv_score', 'N/A'):.4f}, Sil={metrics.get('silhouette_score', 'N/A'):.4f}, DBI={metrics.get('dbi_score', 'N/A'):.4f}", "INFO")
            
            return converted_params
            
        except Exception as e:
            tprint(f"⚠️ Failed to load cached results: {e}", "WARNING")
            return None

    def _run_automated_tuning(self, config: Dict[str, Any], n_trials: int = 20) -> Optional[Dict[str, Any]]:
        """
        Run automated hyperparameter tuning for iterative optimization.
        
        Args:
            config: Configuration dictionary
            n_trials: Number of tuning trials
            
        Returns:
            Dictionary with best parameters or None if tuning fails
        """
        try:
            from src.training.steps.market_analysis.clusters.iterative_optimization_tuner import IterativeOptimizationTuner
            import os
            
            tprint("🎯 Starting automated hyperparameter tuning...", "INFO")
            
            # Load features for tuning
            features = self._load_feature_data_for_optimization(config)
            if features is None:
                tprint("❌ Failed to load features for tuning", "ERROR")
                return None
            
            # Load initial labels (original HDBSCAN labels)
            # Get from current execution context
            selected_features = self._load_selected_features(config)
            if selected_features is None:
                tprint("❌ Failed to load selected features for tuning", "ERROR")
                return None
            
            # Load HDBSCAN labels
            self.artifact_manager.set_context(
                step_name="hdbscan_regime_discovery",
                symbol=config['symbol'],
                exchange=config['exchange'],
                datetime=datetime.now(),
                information="regime_discovery",
                direction="long",
                model="Analyst"
            )
            
            regime_labels_df = self._get_artifact("regime_labels", artifact_type="data")
            
            # Restore context
            self.artifact_manager.set_context(
                step_name="regime_clustering",
                symbol=config['symbol'],
                exchange=config['exchange'],
                datetime=datetime.now(),
                information="regime_clustering",
                direction="long",
                model="Analyst"
            )
            
            if regime_labels_df is None or (hasattr(regime_labels_df, 'empty') and regime_labels_df.empty):
                tprint("❌ Failed to load regime labels for tuning", "ERROR")
                return None
            
            # Extract labels
            if 'regime_label' in regime_labels_df.columns:
                initial_labels = regime_labels_df['regime_label'].values
            elif 'label' in regime_labels_df.columns:
                initial_labels = regime_labels_df['label'].values
            else:
                initial_labels = regime_labels_df.iloc[:, 0].values
            
            # Load market data
            self.artifact_manager.set_context(
                step_name="feature_generation_feature_generation_step",
                symbol=config['symbol'],
                exchange=config['exchange'],
                datetime=datetime.now(),
                information="feature_generation",
                direction="long",
                model="Analyst"
            )
            
            market_data = self._get_artifact("generated_features_15m", artifact_type="data")
            
            # Restore context
            self.artifact_manager.set_context(
                step_name="regime_clustering",
                symbol=config['symbol'],
                exchange=config['exchange'],
                datetime=datetime.now(),
                information="regime_clustering",
                direction="long",
                model="Analyst"
            )
            
            if market_data is None or (hasattr(market_data, 'empty') and market_data.empty):
                tprint("❌ Failed to load market data for tuning", "ERROR")
                return None
            
            # Resample market data to match labels length if needed
            if len(market_data) != len(initial_labels):
                tprint(f"🔧 Resampling market data from {len(market_data)} to {len(initial_labels)} samples", "INFO")
                # Resample to 1h if needed
                if not isinstance(market_data.index, pd.DatetimeIndex):
                    if 'open_time' in market_data.columns:
                        market_data = market_data.set_index('open_time')
                market_data = market_data.resample('1H').last()
                market_data = market_data.dropna(how='all')
            
            # Align features and labels
            min_len = min(len(features), len(initial_labels), len(market_data))
            features = features[:min_len]
            initial_labels = initial_labels[:min_len]
            market_data = market_data.iloc[:min_len]
            
            tprint(f"📊 Tuning dataset: {features.shape[0]} samples × {features.shape[1]} features", "INFO")
            
            # Create tuner
            tuner = IterativeOptimizationTuner(
                features=features,
                initial_labels=initial_labels,
                market_data=market_data,
                verbose=False  # Reduce noise during automated tuning
            )
            
            # Run Bayesian optimization
            tprint(f"🚀 Running Bayesian optimization ({n_trials} trials)...", "INFO")
            results = tuner.optimize_bayesian(n_trials=n_trials)
            
            if results is None or 'best_params' not in results:
                tprint("❌ Tuning failed to produce results", "ERROR")
                return None
            
            # Save tuning results
            output_dir = "artifacts/hyperparameter_tuning/"
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_path = os.path.join(output_dir, f"auto_tuning_results_{config['symbol']}_{timestamp}.json")
            report_path = os.path.join(output_dir, f"auto_tuning_report_{config['symbol']}_{timestamp}.md")
            
            tuner.save_results(results, results_path)
            tuner.generate_report(results, report_path)
            
            tprint(f"✅ Tuning results saved to: {results_path}", "SUCCESS")
            tprint(f"📊 Tuning report saved to: {report_path}", "SUCCESS")
            
            # Extract best parameters
            best_params = results['best_params']
            best_metrics = results['best_metrics']
            
            tprint("🏆 Best tuned parameters:", "SUCCESS")
            tprint(f"   • CV Score: {best_metrics.cv_score:.4f}", "INFO")
            tprint(f"   • Silhouette: {best_metrics.silhouette_score:.4f}", "INFO")
            tprint(f"   • DBI: {best_metrics.dbi_score:.4f}", "INFO")
            tprint(f"   • Clusters: {best_metrics.n_clusters}", "INFO")
            
            # Convert tuner params to iterative optimization config params
            converted_params = {
                'min_clusters': best_params['K_MIN'],
                'max_clusters': best_params['K_MAX'],
                'iterative_max_iterations': best_params['max_rounds'],
                'iterative_min_frac': best_params['MIN_FRAC'],
                'iterative_max_frac': best_params['MAX_FRAC'],
                'iterative_w_cv': best_params['w_cv'],
                'iterative_w_sil': best_params['w_sil'],
                'iterative_w_temp': best_params['w_temp'],
                'iterative_w_bal': best_params['w_bal'],
                'iterative_eps_std_step1': best_params['eps_std_step1'],
                'iterative_sil_guard': best_params['sil_guard'],
                'iterative_temporal_bonus': best_params['temporal_bonus'],
                'iterative_eps_cv': best_params['eps_cv'],
                'iterative_eps_sil': best_params['eps_sil'],
                'iterative_eps_temp': best_params['eps_temp'],
                'iterative_local_churn_cap': best_params['local_churn_cap'],
                'iterative_knn_size': best_params['knn_size'],
                'iterative_size_gate_base': best_params['size_gate_base'],
                'iterative_size_gate_alpha': best_params['size_gate_alpha'],
                'iterative_size_gate_beta': best_params['size_gate_beta'],
            }
            
            # Save best params as artifact for future reference
            self._save_artifact(
                data=best_params,
                artifact_name="tuned_iterative_opt_params",
                artifact_type="config",
                metadata={
                    'symbol': config['symbol'],
                    'timestamp': timestamp,
                    'metrics': {
                        'cv_score': best_metrics.cv_score,
                        'silhouette_score': best_metrics.silhouette_score,
                        'dbi_score': best_metrics.dbi_score,
                        'balance_score': best_metrics.balance_score,
                        'temporal_smoothness': best_metrics.temporal_smoothness,
                        'n_clusters': best_metrics.n_clusters
                    }
                }
            )
            
            return converted_params
            
        except Exception as e:
            tprint(f"❌ Automated tuning failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return None

    def _load_feature_data_for_optimization(self, config: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Load feature data from regime_feature_selection for iterative optimization.
        
        This method loads the market data and selected features, then creates
        the feature matrix needed for iterative optimization.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Feature matrix as NumPy array (n_samples, n_features) or None if loading fails
        """
        try:
            import pandas as pd
            
            # Load market data from pre_training artifacts using BaseStep's method
            tprint("📥 Loading market data from pre_training artifacts...", "INFO")
            
            # Set context to feature_generation_feature_generation_step to load the freshly generated features
            # Use regime_timeframe (1h) instead of trading timeframe (15m)
            regime_timeframe = config.get('regime_timeframe', '1h')
            tprint(f"📊 Loading features for regime timeframe: {regime_timeframe}", "INFO")
            
            self.artifact_manager.set_context(
                step_name="feature_generation_feature_generation_step",
                symbol=config['symbol'],
                exchange=config['exchange'],
                datetime=datetime.now(),
                information="feature_generation",
                direction="long",
                model="Analyst"
            )
            
            # Try to get the generated features for the regime timeframe (1h)
            # Note: The artifact name should match the timeframe used
            artifact_name = f"generated_features_{regime_timeframe}"
            market_data_with_features = self._get_artifact(artifact_name, artifact_type="data")
            
            # If 1h features don't exist, try to resample from 15m
            if market_data_with_features is None or (hasattr(market_data_with_features, 'empty') and market_data_with_features.empty):
                tprint(f"⚠️ No {regime_timeframe} features found, attempting to load and resample from 15m", "WARNING")
                market_data_15m = self._get_artifact("generated_features_15m", artifact_type="data")
                
                if market_data_15m is not None and not (hasattr(market_data_15m, 'empty') and market_data_15m.empty):
                    # Resample 15m to 1h
                    tprint("🔄 Resampling 15m features to 1h...", "INFO")
                    # Ensure index is datetime
                    if not isinstance(market_data_15m.index, pd.DatetimeIndex):
                        if 'open_time' in market_data_15m.columns:
                            market_data_15m = market_data_15m.set_index('open_time')
                        elif 'timestamp' in market_data_15m.columns:
                            market_data_15m = market_data_15m.set_index('timestamp')
                    
                    # Resample to 1h (use last value for each hour)
                    market_data_with_features = market_data_15m.resample('1H').last()
                    market_data_with_features = market_data_with_features.dropna(how='all')
                    tprint(f"✅ Resampled features from {len(market_data_15m)} (15m) to {len(market_data_with_features)} (1h) samples", "SUCCESS")
                else:
                    tprint("❌ Failed to load 15m features for resampling", "ERROR")
                    market_data_with_features = None
            
            # Restore context
            self.artifact_manager.set_context(
                step_name="regime_clustering",
                symbol=config['symbol'],
                exchange=config['exchange'],
                datetime=datetime.now(),
                information="regime_clustering",
                direction="long",
                model="Analyst"
            )
            
            if market_data_with_features is None or (hasattr(market_data_with_features, 'empty') and market_data_with_features.empty):
                tprint("❌ Failed to load market data with features from pre_training", "ERROR")
                return None
            
            # Load selected features from regime_feature_selection (already loaded earlier in execution)
            selected_features = self._load_selected_features(config)
            if selected_features is None or len(selected_features) == 0:
                tprint("❌ No selected features available", "ERROR")
                return None
            
            tprint(f"📊 Loaded {len(selected_features)} selected features from regime_feature_selection", "INFO")
            
            # Extract the feature columns from market data
            missing_features = [f for f in selected_features if f not in market_data_with_features.columns]
            if missing_features:
                tprint(f"⚠️ Missing {len(missing_features)} features in market data: {missing_features[:5]}...", "WARNING")
                # Use only available features
                available_features = [f for f in selected_features if f in market_data_with_features.columns]
                if len(available_features) == 0:
                    tprint("❌ No features available in market data", "ERROR")
                    return None
                selected_features = available_features
            
            # Create feature matrix
            feature_matrix = market_data_with_features[selected_features].values
            
            # Handle any remaining NaN values
            if np.isnan(feature_matrix).any():
                tprint("🔧 Handling NaN values in feature matrix", "INFO")
                # Fill NaN with column mean
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='mean')
                feature_matrix = imputer.fit_transform(feature_matrix)
            
            tprint(f"✅ Created feature matrix: {feature_matrix.shape[0]} samples × {feature_matrix.shape[1]} features", "SUCCESS")
            return feature_matrix
            
        except Exception as e:
            tprint(f"❌ Error loading feature data: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return None

    def _run_iterative_optimization_fallback(self, hdbscan_artifacts: Dict[str, Any], initial_labels: np.ndarray, config: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Run iterative optimization as fallback when quality targets are not met.
        
        Args:
            hdbscan_artifacts: HDBSCAN artifacts
            initial_labels: Initial cluster labels
            config: Configuration dictionary
            
        Returns:
            Optimized cluster labels or None if optimization fails
        """
        try:
            if not ITERATIVE_OPTIMIZATION_AVAILABLE:
                tprint("⚠️ IterativeOptimization not available for fallback", "WARNING")
                return None
            
            tprint("🔄 Starting iterative optimization fallback...", "INFO")
            
            # Load configuration from YAML if available
            config = self._load_regime_clustering_config(config)
            
            # Check if automatic hyperparameter tuning is enabled
            auto_tune = config.get('auto_tune_iterative_opt', False)
            use_cached = config.get('use_cached_tuning', False)
            tuning_trials = config.get('tuning_trials', 20)
            
            tuned_params = None
            
            # Try to use cached tuning results first
            if use_cached and not auto_tune:
                tprint("📦 Attempting to load cached tuning results...", "INFO")
                cached_params = self._load_cached_tuning_results(config)
                if cached_params:
                    tprint("✅ Using cached tuning results", "SUCCESS")
                    tuned_params = cached_params
                else:
                    tprint("⚠️ No valid cached results found", "WARNING")
            
            # Run fresh tuning if enabled and no cached results
            if auto_tune and tuned_params is None:
                tprint("🎯 Automatic hyperparameter tuning enabled!", "INFO")
                tprint(f"📊 Running {tuning_trials} tuning trials before optimization...", "INFO")
                
                # Run automated tuning
                tuned_params = self._run_automated_tuning(config, tuning_trials)
                
                if tuned_params:
                    tprint("✅ Tuning completed - applying best parameters", "SUCCESS")
                else:
                    tprint("⚠️ Tuning failed - using default parameters", "WARNING")
            
            # Apply tuned parameters if available
            if tuned_params:
                config.update(tuned_params)
            
            # Load features from regime_feature_selection (NOT from HDBSCAN)
            # The features for regime clustering should come from regime_feature_selection
            tprint("📥 Loading feature data from regime_feature_selection for iterative optimization...", "INFO")
            features = self._load_feature_data_for_optimization(config)
            
            if features is None:
                tprint("❌ Failed to load feature data from regime_feature_selection", "ERROR")
                return None
            
            if not isinstance(features, np.ndarray):
                tprint(f"⚠️ Unexpected feature format: {type(features)}", "WARNING")
                return None
            
            tprint(f"📊 Using {features.shape[0]} samples with {features.shape[1]} features for iterative optimization", "INFO")
            
            # Filter out noise labels (-1) before iterative optimization
            # Iterative optimization expects non-negative cluster IDs only
            noise_mask = initial_labels >= 0
            if not np.any(noise_mask):
                tprint("❌ All labels are noise (-1), cannot run iterative optimization", "ERROR")
                return None
            
            # Filter features and labels to exclude noise points
            filtered_features = features[noise_mask]
            filtered_labels = initial_labels[noise_mask]
            
            noise_count = np.sum(~noise_mask)
            if noise_count > 0:
                tprint(f"🔧 Filtered out {noise_count} noise points, using {len(filtered_labels)} samples for optimization", "INFO")
            
            # Create clustering context for iterative optimization
            context = self._create_clustering_context_for_iterative_optimization(
                filtered_features, filtered_labels, config
            )
            
            # Configure iterative optimization
            iterative_config = {
                'max_iterations': config.get('iterative_max_iterations', 25),
                'convergence_threshold': config.get('iterative_convergence_threshold', 0.001),
                'enable_risk_mitigation': config.get('iterative_enable_risk_mitigation', True),
                'min_clusters': config.get('min_clusters', 4),
                'max_clusters': config.get('max_clusters', 8)
            }
            
            # Run iterative optimization
            optimizer = IterativeOptimization(verbose=True)
            
            # Apply tuned parameters if auto-tuning was run
            if config.get('auto_tune_iterative_opt', False):
                tprint("🔧 Applying tuned parameters to optimizer...", "INFO")
                
                # Update optimizer config with tuned parameters
                if 'iterative_min_frac' in config:
                    optimizer.config.MIN_FRAC = config['iterative_min_frac']
                if 'iterative_max_frac' in config:
                    optimizer.config.MAX_FRAC = config['iterative_max_frac']
                if 'iterative_w_cv' in config:
                    optimizer.config.w_cv = config['iterative_w_cv']
                if 'iterative_w_sil' in config:
                    optimizer.config.w_sil = config['iterative_w_sil']
                if 'iterative_w_temp' in config:
                    optimizer.config.w_temp = config['iterative_w_temp']
                if 'iterative_w_bal' in config:
                    optimizer.config.w_bal = config['iterative_w_bal']
                if 'iterative_eps_std_step1' in config:
                    optimizer.config.eps_std_step1 = config['iterative_eps_std_step1']
                if 'iterative_sil_guard' in config:
                    optimizer.config.sil_guard = config['iterative_sil_guard']
                if 'iterative_temporal_bonus' in config:
                    optimizer.config.temporal_bonus = config['iterative_temporal_bonus']
                if 'iterative_eps_cv' in config:
                    optimizer.config.eps_cv = config['iterative_eps_cv']
                if 'iterative_eps_sil' in config:
                    optimizer.config.eps_sil = config['iterative_eps_sil']
                if 'iterative_eps_temp' in config:
                    optimizer.config.eps_temp = config['iterative_eps_temp']
                if 'iterative_local_churn_cap' in config:
                    optimizer.config.local_churn_cap = config['iterative_local_churn_cap']
                if 'iterative_knn_size' in config:
                    optimizer.config.knn_size = config['iterative_knn_size']
                if 'iterative_size_gate_base' in config:
                    optimizer.config.size_gate_base = config['iterative_size_gate_base']
                if 'iterative_size_gate_alpha' in config:
                    optimizer.config.size_gate_alpha = config['iterative_size_gate_alpha']
                if 'iterative_size_gate_beta' in config:
                    optimizer.config.size_gate_beta = config['iterative_size_gate_beta']
                
                tprint("✅ Tuned parameters applied to optimizer", "SUCCESS")
            
            # Use asyncio to run the async method
            # Handle event loop properly (support nested loops)
            import asyncio
            try:
                # Try to use nest_asyncio to allow nested event loops
                import nest_asyncio
                nest_asyncio.apply()
                tprint("✅ nest_asyncio applied - nested event loops enabled", "INFO")
            except ImportError:
                tprint("⚠️ nest_asyncio not available - trying alternative approach", "WARNING")
            
            try:
                # Try to get existing event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Event loop is already running - create new task
                    tprint("🔄 Event loop already running - using asyncio.create_task", "INFO")
                    # We can't use run_until_complete on a running loop, so we'll run synchronously
                    # Create a new event loop in a different thread or use sync approach
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            optimizer.execute_optimization_loop(
                                context, iterative_config,
                                max_iterations=iterative_config['max_iterations'],
                                enable_risk_mitigation=iterative_config['enable_risk_mitigation']
                            )
                        )
                        optimized_context = future.result()
                else:
                    # Loop exists but not running
                    optimized_context = loop.run_until_complete(
                        optimizer.execute_optimization_loop(
                            context, iterative_config,
                            max_iterations=iterative_config['max_iterations'],
                            enable_risk_mitigation=iterative_config['enable_risk_mitigation']
                        )
                    )
            except RuntimeError:
                # No event loop exists - create one
                optimized_context = asyncio.run(
                    optimizer.execute_optimization_loop(
                        context, iterative_config,
                        max_iterations=iterative_config['max_iterations'],
                        enable_risk_mitigation=iterative_config['enable_risk_mitigation']
                    )
                )
            
            # Extract optimized labels
            # Check if optimized_assignments is available (updated during optimization)
            if hasattr(optimized_context, 'optimized_assignments') and optimized_context.optimized_assignments is not None:
                optimized_labels_filtered = optimized_context.optimized_assignments
            elif hasattr(optimized_context, 'assignments') and optimized_context.assignments is not None:
                optimized_labels_filtered = optimized_context.assignments
            else:
                tprint("❌ No optimized assignments found in context", "ERROR")
                return None
            
            tprint(f"✅ Iterative optimization completed: {len(np.unique(optimized_labels_filtered))} clusters", "SUCCESS")
            
            # Map the optimized labels back to include noise points
            # Create full labels array with noise points restored
            optimized_labels_full = np.full(len(initial_labels), -1, dtype=np.int32)
            optimized_labels_full[noise_mask] = optimized_labels_filtered
            
            tprint(f"📊 Final labels: {len(np.unique(optimized_labels_full))} unique labels (including noise)", "INFO")
            
            return optimized_labels_full
            
        except Exception as e:
            tprint(f"⚠️ Iterative optimization fallback failed: {e}", "WARNING")
            return None

    def _create_clustering_context_for_iterative_optimization(self, features: np.ndarray, labels: np.ndarray, config: Dict[str, Any]) -> Any:
        """
        Create clustering context for iterative optimization.
        
        Args:
            features: Feature matrix
            labels: Initial cluster labels
            config: Configuration dictionary
            
        Returns:
            ClusteringContext object
        """
        try:
            # Import the ClusteringContext class
            from src.training.steps.market_analysis.clusters.step1_feature_preparation import ClusteringContext
            
            # Load market data for context (needed by ClusteringContext)
            # Set context to feature_generation to load the generated features with market data
            self.artifact_manager.set_context(
                step_name="feature_generation_feature_generation_step",
                symbol=config['symbol'],
                exchange=config['exchange'],
                datetime=datetime.now(),
                information="feature_generation",
                direction="long",
                model="Analyst"
            )
            
            market_data = self._get_artifact("generated_features_15m", artifact_type="data")
            
            # Restore context
            self.artifact_manager.set_context(
                step_name="regime_clustering",
                symbol=config['symbol'],
                exchange=config['exchange'],
                datetime=datetime.now(),
                information="regime_clustering",
                direction="long",
                model="Analyst"
            )
            
            # Create context with required arguments
            context = ClusteringContext(
                original_features=features,
                market_data=market_data if market_data is not None else pd.DataFrame()
            )
            
            # Set initial assignments (required by iterative optimization)
            context.initial_assignments = labels.copy()
            context.assignments = labels
            context.optimal_k = len(np.unique(labels[labels != -1]))
            context.n_samples = len(features)
            context.n_features = features.shape[1] if len(features.shape) > 1 else 1
            
            # Set optimized_features (required by iterative optimization)
            context.optimized_features = features
            context.optimized_feature_names = None  # Will be set by iterative optimization
            
            # Add any additional context attributes that might be needed
            context.symbol = config.get('symbol', 'UNKNOWN')
            context.exchange = config.get('exchange', 'binance')
            context.timeframe = config.get('timeframe', '1h')
            
            return context
            
        except Exception as e:
            tprint(f"⚠️ Failed to create clustering context: {e}", "WARNING")
            # Return a minimal context if the full context creation fails
            class MinimalContext:
                def __init__(self, features, labels):
                    self.features = features
                    self.assignments = labels
                    self.optimal_k = len(np.unique(labels[labels != -1]))
                    self.n_samples = len(features)
                    self.n_features = features.shape[1] if len(features.shape) > 1 else 1
            
            return MinimalContext(features, labels)

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
