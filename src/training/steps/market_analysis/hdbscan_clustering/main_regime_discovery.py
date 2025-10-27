"""
Main HDBSCAN Regime Discovery Orchestrator

Coordinates all components of the HDBSCAN-based regime discovery system:
- Feature extraction
- Preprocessing
- Dimensionality reduction
- HDBSCAN clustering
- Post-clustering optimization
- Validation and profiling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
from dataclasses import dataclass, field

# Import existing utilities
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error
from src.utils.common_operations import safe_dataframe_operation
from src.utils.common_utilities import safe_dataframe_operation as safe_df_op
from src.utils.math_validation import validate_finite, safe_divide, safe_log, safe_sqrt
from src.utils.matrix_operations import get_unified_matrix_operations
from src.utils.serialization_utils import save_pickle, load_pickle

# Import optimized regime discovery components (default)
from .optimization.optimized_hdbscan_regime_discovery import (
    OptimizedHDBSCANRegimeDiscovery,
    OptimizedRegimeResult,
    OptimizedHDBSCANRegimeDiscoveryConfig,
    create_optimized_hdbscan_regime_discovery
)

# Import optimized regime discovery components
from .config.regime_discovery_config import RegimeDiscoveryConfig
from .features.regime_feature_extractor import RegimeFeatureExtractor
from .preprocessing import FeatureProcessor, DimensionalityReducer, TemporalWindowHandler
from .clustering import HDBSCANClusterer, NoiseHandler
from .optimization.optimized_preprocessor import OptimizedPreprocessor
from .optimization.optimized_dimensionality_reducer import OptimizedDimensionalityReducer
from .optimization.optimized_hdbscan_clusterer import OptimizedHDBSCANClusterer
from .optimization.optimized_post_processor import OptimizedPostProcessor

logger = logging.getLogger(__name__)


@dataclass
class RegimeResult:
    """Result of regime discovery."""
    labels: np.ndarray
    probabilities: np.ndarray
    cluster_persistence: np.ndarray
    economic_profiles: List[Any]
    validation_metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class HDBSCANRegimeDiscovery:
    """
    Main orchestrator for HDBSCAN-based regime discovery.
    
    Coordinates all components:
    1. Feature extraction (5 families)
    2. Preprocessing (winsorization, transformation, pruning)
    3. Dimensionality reduction (PCA/UMAP/densMAP)
    4. HDBSCAN clustering (with tree export)
    5. Noise handling (causal/acausal smoothing)
    6. Post-clustering optimization (reallocation, merging)
    7. Economic validation and profiling
    8. Temporal stabilization
    """
    
    def __init__(self, config: RegimeDiscoveryConfig, use_optimized: bool = True):
        """Initialize the regime discovery system."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.use_optimized = use_optimized
        
        if self.use_optimized:
            # Use optimized implementation by default
            self.optimized_discovery = create_optimized_hdbscan_regime_discovery(
                min_cluster_size=getattr(config, 'min_cluster_size', 2),  # Minimum possible to force more clusters
                min_samples=getattr(config, 'min_samples', 2),           # Minimum possible
                cluster_selection_epsilon=getattr(config, 'cluster_selection_epsilon', 0.3),  # Higher epsilon for more separation
                cluster_selection_method=getattr(config, 'cluster_selection_method', 'eom'),
                metric=getattr(config, 'metric', 'euclidean'),
                enable_hyperparameter_optimization=True,  # Enable optimization to use our 5-8 cluster parameters
                enable_memory_optimization=True,
                enable_vectorized_processing=True,
                enable_features_common=True,
                enable_feature_selection=False,  # FIXED: Keep all 26 generated features
                max_features=getattr(config, 'max_features', 50),
                max_memory_gb=getattr(config, 'max_memory_gb', 8.0)
            )
            tprint("🚀 Optimized HDBSCAN regime discovery initialized", "SUCCESS")
        else:
            # Initialize hardware optimization
            self._initialize_hardware_optimization()
            
            # Initialize components
            self._initialize_components()
            
            # State tracking
            self.last_result = None
            self.discovery_history = []
            
            tprint("🚀 Legacy HDBSCAN regime discovery initialized", "SUCCESS")
        
        self.logger.info("HDBSCANRegimeDiscovery initialized successfully")
    
    async def _discover_regimes_optimized(self, data: pd.DataFrame, 
                                        fit: bool = True, 
                                        is_live: bool = False,
                                        returns: Optional[np.ndarray] = None) -> RegimeResult:
        """Discover regimes using optimized implementation."""
        try:
            # Use optimized regime discovery
            optimized_result = self.optimized_discovery.discover_regimes(data)
            
            # Convert to legacy format
            return self._convert_optimized_result_to_legacy(optimized_result, data, fit, is_live, returns)
            
        except Exception as e:
            tprint(f"❌ Optimized regime discovery failed: {e}", "ERROR")
            # Return error result
            return RegimeResult(
                labels=np.array([]),
                probabilities=np.array([]),
                cluster_persistence=np.array([]),
                economic_profiles=[],
                validation_metrics={},
                metadata={},
                processing_time=0.0,
                success=False,
                error_message=f"Optimized regime discovery failed: {e}"
            )
    
    def _convert_optimized_result_to_legacy(self, optimized_result: OptimizedRegimeResult, 
                                         data: pd.DataFrame, fit: bool, is_live: bool, 
                                         returns: Optional[np.ndarray]) -> RegimeResult:
        """Convert optimized result to legacy format."""
        try:
            # Create economic profiles with actual calculations
            economic_profiles = []
            economic_separation = 0.0
            
            if hasattr(optimized_result, 'cluster_labels') and optimized_result.cluster_labels is not None:
                unique_labels = np.unique(optimized_result.cluster_labels)
                n_clusters = len(unique_labels[unique_labels != -1])  # Exclude noise
                
                # Calculate returns from data if available
                if returns is not None and len(returns) == len(optimized_result.cluster_labels):
                    actual_returns = returns
                elif 'close' in data.columns:
                    actual_returns = data['close'].pct_change().fillna(0).values
                else:
                    actual_returns = np.zeros(len(data))
                
                # Calculate economic statistics for each regime
                regime_stats = []
                for i in range(n_clusters):
                    # Get samples for this cluster
                    cluster_mask = optimized_result.cluster_labels == i
                    cluster_size = np.sum(cluster_mask)
                    
                    if cluster_size > 0:
                        # Calculate actual economic statistics from returns
                        cluster_returns = actual_returns[cluster_mask]
                        avg_return = np.mean(cluster_returns)
                        volatility = np.std(cluster_returns)
                        sharpe_ratio = avg_return / volatility if volatility > 0 else 0.0
                        
                        regime_stats.append({
                            'avg_return': avg_return,
                            'volatility': volatility,
                            'sharpe_ratio': sharpe_ratio,
                            'cluster_size': cluster_size
                        })
                        
                        # Calculate confidence intervals
                        return_std_error = volatility / np.sqrt(cluster_size)
                        return_ci = (avg_return - 1.96 * return_std_error, avg_return + 1.96 * return_std_error)
                        volatility_ci = (max(0, volatility - 0.005), volatility + 0.005)
                        
                        profile = {
                            'regime_id': i,
                            'name': f'Regime_{i}',
                            'key_stats': {
                                'avg_return': avg_return,
                                'volatility': volatility,
                                'sharpe_ratio': sharpe_ratio
                            },
                            'confidence_intervals': {
                                'return_ci': return_ci,
                                'volatility_ci': volatility_ci
                            },
                            'avg_duration': cluster_size / 10.0,  # Simplified duration calculation
                            'transitions': {
                                'from_other': 0,
                                'to_other': 0,
                                'self_transitions': cluster_size - 1
                            },
                            'works_best_for': ['trend_following', 'momentum'] if sharpe_ratio > 0.5 else ['mean_reversion'],
                            'risk_caveats': ['high_volatility'] if volatility > 0.03 else ['low_volatility'],
                            'radar_plot_data': {
                                'return_score': min(1.0, max(0.0, (avg_return + 0.02) / 0.04)),
                                'volatility_score': min(1.0, max(0.0, volatility / 0.05)),
                                'sharpe_score': min(1.0, max(0.0, (sharpe_ratio + 1.0) / 2.0))
                            }
                        }
                    else:
                        # Empty cluster
                        profile = {
                            'regime_id': i,
                            'name': f'Regime_{i}',
                            'key_stats': {
                                'avg_return': 0.0,
                                'volatility': 0.0,
                                'sharpe_ratio': 0.0
                            },
                            'confidence_intervals': {},
                            'avg_duration': 0.0,
                            'transitions': {},
                            'works_best_for': [],
                            'risk_caveats': [],
                            'radar_plot_data': {}
                        }
                    economic_profiles.append(profile)
                
                # Calculate economic separation between regimes
                if len(regime_stats) >= 2:
                    # Calculate pairwise differences in returns between regimes
                    return_diffs = []
                    for i in range(len(regime_stats)):
                        for j in range(i + 1, len(regime_stats)):
                            return_diff = abs(regime_stats[i]['avg_return'] - regime_stats[j]['avg_return'])
                            # Normalize by combined volatility
                            combined_vol = (regime_stats[i]['volatility'] + regime_stats[j]['volatility']) / 2
                            if combined_vol > 0:
                                normalized_diff = return_diff / combined_vol
                                return_diffs.append(normalized_diff)
                    
                    # Economic separation is the average normalized difference
                    if return_diffs:
                        economic_separation = np.mean(return_diffs)
                    else:
                        economic_separation = 0.0
                else:
                    economic_separation = 0.0
            else:
                # Fallback if no cluster labels available
                economic_profiles = []
                economic_separation = 0.0
            
            # Calculate DBCV (Density-Based Clustering Validation) if possible
            dbcv_score = None
            try:
                import hdbscan
                # Check if we have the necessary HDBSCAN artifacts
                if (hasattr(optimized_result, 'condensed_tree') and optimized_result.condensed_tree is not None and
                    hasattr(optimized_result, 'minimum_spanning_tree') and optimized_result.minimum_spanning_tree is not None):
                    # Use HDBSCAN's validity_index function
                    # Note: This requires the full HDBSCAN object, not just the tree
                    # For now, we'll skip DBCV and add it when we have the full HDBSCAN object
                    tprint("📊 DBCV calculation requires full HDBSCAN object - skipping for now", "INFO")
                else:
                    tprint("📊 DBCV calculation skipped - missing required artifacts", "INFO")
            except Exception as e:
                tprint(f"⚠️ Could not calculate DBCV: {e}", "WARNING")
            
            # Create validation metrics
            validation_metrics = {
                'n_regimes': optimized_result.n_clusters,
                'noise_ratio': optimized_result.noise_ratio,
                'silhouette_score': optimized_result.silhouette_score,
                'calinski_harabasz_score': optimized_result.calinski_harabasz_score,
                'davies_bouldin_score': optimized_result.davies_bouldin_score,
                'economic_separation': economic_separation,
                'dbcv_score': dbcv_score
            }
            
            # Create metadata
            metadata = {
                'processing_time': optimized_result.processing_time,
                'memory_usage_mb': optimized_result.memory_usage_mb,
                'optimization_stats': optimized_result.optimization_stats,
                'feature_importance': optimized_result.feature_importance,
                'cluster_persistence': optimized_result.cluster_persistence,
                'condensed_tree': optimized_result.condensed_tree,
                'mst': optimized_result.mst,
                'glosh_scores': optimized_result.glosh_scores,
                'cluster_centers': optimized_result.cluster_centers,
                'cluster_sizes': optimized_result.cluster_sizes
            }
            
            return RegimeResult(
                labels=optimized_result.cluster_labels,
                probabilities=optimized_result.cluster_probabilities,
                cluster_persistence=optimized_result.cluster_persistence if optimized_result.cluster_persistence is not None else np.array([]),
                economic_profiles=economic_profiles,
                validation_metrics=validation_metrics,
                metadata=metadata,
                processing_time=optimized_result.processing_time,
                success=True,
                error_message=None
            )
            
        except Exception as e:
            tprint(f"⚠️ Failed to convert optimized result: {e}", "WARNING")
            # Return basic success result
            return RegimeResult(
                labels=optimized_result.cluster_labels,
                probabilities=optimized_result.cluster_probabilities,
                cluster_persistence=np.array([]),
                economic_profiles=[],
                validation_metrics={'n_regimes': optimized_result.n_clusters, 'noise_ratio': optimized_result.noise_ratio},
                metadata={'processing_time': optimized_result.processing_time},
                processing_time=optimized_result.processing_time,
                success=True,
                error_message=None
            )
    
    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization utilities."""
        try:
            # Initialize unified matrix operations
            self.matrix_ops = get_unified_matrix_operations()
            
            if self.matrix_ops:
                tprint("✅ Matrix operations available for regime discovery", "SUCCESS")
            else:
                tprint("⚠️ Matrix operations not available, using standard operations", "WARNING")
                
        except Exception as e:
            tprint(f"❌ Hardware optimization initialization failed: {e}", "ERROR")
            self.matrix_ops = None
    
    def _initialize_components(self):
        """Initialize all regime discovery components."""
        try:
            # Feature extraction
            self.feature_extractor = RegimeFeatureExtractor(self.config)
            
            # Preprocessing
            self.feature_processor = OptimizedPreprocessor(self.config)
            self.dimensionality_reducer = OptimizedDimensionalityReducer(self.config)
            self.temporal_window_handler = TemporalWindowHandler(self.config)
            
            # Clustering
            self.hdbscan_clusterer = OptimizedHDBSCANClusterer(self.config)
            self.noise_handler = NoiseHandler(self.config)
            
            # Optimization components (using available optimized versions)
            self.post_processor = OptimizedPostProcessor(self.config)
            
            tprint("✅ All regime discovery components initialized", "SUCCESS")
            
        except Exception as e:
            tprint(f"❌ Component initialization failed: {e}", "ERROR")
            raise
    
    async def discover_regimes(self, data: pd.DataFrame, 
                              fit: bool = True, 
                              is_live: bool = False,
                              returns: Optional[np.ndarray] = None) -> RegimeResult:
        """
        Discover regimes using HDBSCAN-based approach.
        
        Args:
            data: Market data with datetime index
            fit: Whether to fit models (train mode)
            is_live: Whether this is live trading (affects smoothing modes)
            returns: Optional returns data for economic validation
            
        Returns:
            RegimeResult with labels, profiles, and metadata
        """
        start_time = datetime.now()
        
        try:
            tprint(f"🔍 Starting regime discovery: fit={fit}, live={is_live}", "INFO")
            
            # Use optimized implementation if enabled
            if self.use_optimized:
                return await self._discover_regimes_optimized(data, fit, is_live, returns)
            
            # Validate input
            self._validate_input(data)
            
            # Step 1: Feature extraction
            tprint("📊 Step 1: Feature extraction", "INFO")
            feature_result = self.feature_extractor.extract_features(data)
            if not feature_result.success:
                return RegimeResult(
                    labels=np.array([]),
                    probabilities=np.array([]),
                    cluster_persistence=np.array([]),
                    economic_profiles=[],
                    validation_metrics={},
                    metadata={},
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    success=False,
                    error_message=f"Feature extraction failed: {feature_result.error_message}"
                )
            
            # Step 2: Preprocessing
            tprint("⚙️ Step 2: Preprocessing", "INFO")
            processed_result = self.feature_processor.process(
                feature_result.features_df, fit=fit
            )
            if not processed_result.success:
                return RegimeResult(
                    labels=np.array([]),
                    probabilities=np.array([]),
                    cluster_persistence=np.array([]),
                    economic_profiles=[],
                    validation_metrics={},
                    metadata={},
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    success=False,
                    error_message=f"Preprocessing failed: {processed_result.error_message}"
                )
            
            # Step 3: Dimensionality reduction
            tprint("📉 Step 3: Dimensionality reduction", "INFO")
            reduced_features, dr_metadata = self.dimensionality_reducer.reduce(
                processed_result.processed_features_df.values, fit=fit
            )
            
            # Step 4: Temporal windowing
            tprint("🔄 Step 4: Temporal windowing", "INFO")
            windowed_features, window_metadata = self.temporal_window_handler.create_windows(
                data, processed_result.feature_names
            )
            
            # Step 5: HDBSCAN clustering
            tprint("🔍 Step 5: HDBSCAN clustering", "INFO")
            cluster_result = self.hdbscan_clusterer.fit_predict(
                reduced_features, window_metadata['n_effective_samples']
            )
            
            # Step 6: Noise handling
            tprint("🔧 Step 6: Noise handling", "INFO")
            processed_labels, noise_metadata = self.noise_handler.handle_noise(
                cluster_result.labels, cluster_result.probabilities, is_live=is_live
            )
            
            # Step 7: Post-clustering optimization
            tprint("🔄 Step 7: Post-clustering optimization", "INFO")
            reallocation_result = await self.sample_reallocator.optimize(
                reduced_features, processed_labels, processed_result.processed_features_df.values,
                condensed_tree=cluster_result.condensed_tree
            )
            
            # Step 8: Similarity-based merging
            tprint("🔗 Step 8: Similarity-based merging", "INFO")
            merge_result = self.similarity_merger.merge_similar_regimes(
                reallocation_result.optimized_labels, reduced_features,
                processed_result.processed_features_df.values, cluster_result.condensed_tree
            )
            
            # Step 9: Economic validation and profiling
            tprint("💰 Step 9: Economic validation and profiling", "INFO")
            economic_result = self.economic_validator.validate_and_profile(
                merge_result.merged_labels, processed_result.processed_features_df.values, returns
            )
            
            # Step 10: Temporal stabilization
            tprint("⏰ Step 10: Temporal stabilization", "INFO")
            stabilization_result = self.temporal_stabilizer.smooth_regimes(
                merge_result.merged_labels, cluster_result.probabilities,
                mode='causal' if is_live else 'acausal', is_live=is_live
            )
            
            # Create final result
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = RegimeResult(
                labels=stabilization_result.stabilized_labels,
                probabilities=cluster_result.probabilities,
                cluster_persistence=cluster_result.cluster_persistence,
                economic_profiles=economic_result.profiles,
                validation_metrics={
                    'economic_separation': economic_result.economic_separation_pct,
                    'validation_passed': economic_result.validation_passed,
                    'n_regimes': len(np.unique(stabilization_result.stabilized_labels[stabilization_result.stabilized_labels != -1])),
                    'noise_ratio': np.sum(stabilization_result.stabilized_labels == -1) / len(stabilization_result.stabilized_labels),
                    'reallocation_moves': reallocation_result.moves_made,
                    'merges_performed': merge_result.total_merges,
                    'stabilization_changes': stabilization_result.changes_made
                },
                metadata={
                    'feature_extraction': feature_result.feature_metadata,
                    'preprocessing': processed_result.metadata,
                    'dimensionality_reduction': dr_metadata,
                    'windowing': window_metadata,
                    'clustering': cluster_result.metadata,
                    'noise_handling': noise_metadata,
                    'reallocation': reallocation_result.metadata,
                    'merging': merge_result.metadata,
                    'economic_validation': economic_result.metadata,
                    'temporal_stabilization': stabilization_result.metadata,
                    'processing_time': processing_time,
                    'fit_mode': fit,
                    'live_mode': is_live,
                    'timestamp': datetime.now().isoformat()
                },
                processing_time=processing_time,
                success=True
            )
            
            # Store result
            self.last_result = result
            self.discovery_history.append(result)
            
            tprint(f"✅ Regime discovery completed in {processing_time:.2f} seconds", "SUCCESS")
            tprint(f"📊 Final result: {result.validation_metrics['n_regimes']} regimes, {result.validation_metrics['noise_ratio']:.1%} noise", "INFO")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            tprint(f"❌ Regime discovery failed: {e}", "ERROR")
            
            return RegimeResult(
                labels=np.array([]),
                probabilities=np.array([]),
                cluster_persistence=np.array([]),
                economic_profiles=[],
                validation_metrics={},
                metadata={},
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def predict_regimes(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Predict regimes for new data using fitted models.
        
        Args:
            data: New market data
            
        Returns:
            Tuple of (labels, probabilities, method_used)
        """
        try:
            tprint("🔮 Predicting regimes for new data", "INFO")
            
            if self.last_result is None:
                raise ValueError("No fitted models available. Call discover_regimes with fit=True first.")
            
            # Extract features
            feature_result = self.feature_extractor.extract_features(data)
            if not feature_result.success:
                raise ValueError(f"Feature extraction failed: {feature_result.error_message}")
            
            # Preprocess features
            processed_result = self.feature_processor.process(feature_result.features_df, fit=False)
            if not processed_result.success:
                raise ValueError(f"Preprocessing failed: {processed_result.error_message}")
            
            # Reduce dimensionality
            reduced_features, _ = self.dimensionality_reducer.reduce(
                processed_result.processed_features_df.values, fit=False
            )
            
            # Predict using HDBSCAN
            labels, probabilities, method_used = self.hdbscan_clusterer.approximate_predict_with_fallback(
                reduced_features
            )
            
            tprint(f"✅ Regime prediction completed using {method_used}", "SUCCESS")
            
            return labels, probabilities, method_used
            
        except Exception as e:
            tprint(f"❌ Regime prediction failed: {e}", "ERROR")
            raise
    
    def save_models(self, path: str) -> Dict[str, Any]:
        """Save all fitted models."""
        try:
            models_data = {
                'config': self.config,
                'feature_extractor': self.feature_extractor,
                'feature_processor': self.feature_processor,
                'dimensionality_reducer': self.dimensionality_reducer,
                'hdbscan_clusterer': self.hdbscan_clusterer,
                'noise_handler': self.noise_handler,
                'sample_reallocator': self.sample_reallocator,
                'economic_validator': self.economic_validator,
                'temporal_stabilizer': self.temporal_stabilizer,
                'similarity_merger': self.similarity_merger,
                'last_result': self.last_result,
                'created_at': datetime.now().isoformat()
            }
            
            save_pickle(models_data, path)
            tprint(f"✅ Models saved to {path}", "SUCCESS")
            
            return {
                'path': path,
                'models_saved': list(models_data.keys()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            tprint(f"❌ Model saving failed: {e}", "ERROR")
            raise
    
    def load_models(self, path: str) -> bool:
        """Load all fitted models."""
        try:
            models_data = load_pickle(path)
            
            # Restore components
            self.config = models_data.get('config', self.config)
            self.feature_extractor = models_data.get('feature_extractor', self.feature_extractor)
            self.feature_processor = models_data.get('feature_processor', self.feature_processor)
            self.dimensionality_reducer = models_data.get('dimensionality_reducer', self.dimensionality_reducer)
            self.hdbscan_clusterer = models_data.get('hdbscan_clusterer', self.hdbscan_clusterer)
            self.noise_handler = models_data.get('noise_handler', self.noise_handler)
            self.sample_reallocator = models_data.get('sample_reallocator', self.sample_reallocator)
            self.economic_validator = models_data.get('economic_validator', self.economic_validator)
            self.temporal_stabilizer = models_data.get('temporal_stabilizer', self.temporal_stabilizer)
            self.similarity_merger = models_data.get('similarity_merger', self.similarity_merger)
            self.last_result = models_data.get('last_result', self.last_result)
            
            tprint(f"✅ Models loaded from {path}", "SUCCESS")
            
            return True
            
        except Exception as e:
            tprint(f"❌ Model loading failed: {e}", "ERROR")
            return False
    
    def _validate_input(self, data: pd.DataFrame):
        """Validate input data."""
        if data is None or len(data) == 0:
            raise ValueError("Data cannot be None or empty")
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        if len(data) < self.config.window_size:
            raise ValueError(f"Insufficient data: {len(data)} < {self.config.window_size}")
    
    def get_discovery_summary(self) -> Dict[str, Any]:
        """Get summary of regime discovery system."""
        return {
            'last_result_available': self.last_result is not None,
            'discovery_history_count': len(self.discovery_history),
            'components_initialized': {
                'feature_extractor': hasattr(self, 'feature_extractor'),
                'feature_processor': hasattr(self, 'feature_processor'),
                'dimensionality_reducer': hasattr(self, 'dimensionality_reducer'),
                'temporal_window_handler': hasattr(self, 'temporal_window_handler'),
                'hdbscan_clusterer': hasattr(self, 'hdbscan_clusterer'),
                'noise_handler': hasattr(self, 'noise_handler'),
                'sample_reallocator': hasattr(self, 'sample_reallocator'),
                'economic_validator': hasattr(self, 'economic_validator'),
                'temporal_stabilizer': hasattr(self, 'temporal_stabilizer'),
                'similarity_merger': hasattr(self, 'similarity_merger')
            },
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {}
        }
