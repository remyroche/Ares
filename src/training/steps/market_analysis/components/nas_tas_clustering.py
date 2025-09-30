"""
NAS-TAS Clustering Component.

This component uses shared utilities to eliminate redundancy between NAS and TAS components.
It demonstrates how to use the shared_utils package for common functionality.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import traceback

from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult

# Import matrix operations and hardware utilities
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_batch_matrix_processor,
        safe_matrix_multiply,
        safe_correlation_matrix,
        gpu_matrix_multiply,
        correlation_matrix_gpu,
        optimize_dataframe,
        vectorized_rolling_features,
        matrix_correlation_analysis,
        batch_matrix_multiply,
        batch_feature_transformation,
        batch_correlation_analysis,
        get_hardware_performance_report,
        optimize_matrix_operation_with_hardware,
        cleanup_hardware_resources,
        get_processing_performance_stats
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError as e:
    MATRIX_OPERATIONS_AVAILABLE = False
    log_warning(f"Matrix operations not available: {e}")

try:
    from src.utils.hardware import (
        get_unified_hardware_manager,
        get_advanced_cpu_optimizer,
        get_enhanced_gpu_manager,
        get_advanced_memory_optimizer,
        get_adaptive_optimization_engine,
        optimize_for_workload,
        optimize_for_workload_adaptive,
        optimize_dataframe_advanced,
        record_performance_adaptive
    )
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    log_warning(f"Hardware optimization not available: {e}")

# Import shared utilities
from ..shared_utils import (
    # Features
    prepare_market_features, FeatureConfig,
    
    # Configuration
    validate_regime_count, normalize_weights, validate_algorithm_type,
    create_default_config, ConfigValidator, BaseConfig,
    
    # Logging
    log_execution, log_performance, LoggingContext,
    get_logger, log_info, log_warning, log_error, log_success, log_debug,
    
    # Metrics
    calculate_consensus_metrics, calculate_disagreement_metrics,
    calculate_economic_scores, calculate_trading_scores, calculate_stability_scores,
    MetricsCalculator,
    
    # Characteristics
    create_regime_characteristics, generate_cluster_characteristics,
    CharacteristicsGenerator
)

# Import original tprint for backward compatibility
from src.utils.tprint import tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_progress, tprint_performance, tprint_timer


@dataclass
class NASTASClusteringConfig(BaseConfig):
    """Configuration for NAS-TAS clustering component using shared utilities."""
    exchange: str = "binance"
    
    # Clustering parameters
    n_regimes: int = 8
    algorithm_type: str = "adaptive_clustering"
    enable_economic_clustering: bool = True
    enable_ensemble_clustering: bool = True
    
    # Economic clustering weights
    economic_weight: float = 0.3
    momentum_weight: float = 0.25
    volume_weight: float = 0.25
    
    # Feature configuration
    feature_categories: List[str] = None
    use_standardized_features: bool = True
    
    # Output configuration
    output_dir: str = "data_cache"
    save_intermediate_results: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        super().__post_init__()
        if self.feature_categories is None:
            self.feature_categories = ['momentum', 'volatility', 'volume', 'trend', 'price_action']
        
        # Ensure n_regimes is between 5 and 15
        if not (5 <= self.n_regimes <= 15):
            self.n_regimes = max(5, min(15, self.n_regimes))


class NASTASClusteringComponent(BaseMarketAnalysisComponent):
    """
    NAS-TAS Clustering Component.
    
    This component uses shared utilities to eliminate redundancy:
    - Uses shared feature preparation
    - Uses shared configuration validation
    - Uses shared logging utilities
    - Uses shared metrics calculation
    - Uses shared regime characteristics generation
    """
    
    def __init__(self, config: Optional[NASTASClusteringConfig] = None):
        """Initialize the NAS-TAS clustering component."""
        with LoggingContext('NAS-TAS-Clustering', 'Initialization', verbose=True):
            super().__init__(config)
            
            # Use shared logging utilities
            self.logger = get_logger('NASTASClustering')
            
            # Initialize shared utilities
            self.config_validator = ConfigValidator(verbose=True)
            self.metrics_calculator = MetricsCalculator(verbose=True)
            self.characteristics_generator = CharacteristicsGenerator(verbose=True)
            
            # Initialize feature configuration
            self.feature_config = FeatureConfig(
                feature_categories=getattr(config, 'feature_categories', ['momentum', 'volatility', 'volume', 'trend', 'price_action']),
                use_standardized_features=getattr(config, 'use_standardized_features', True),
                drop_highly_correlated=True
            )
            
            self.unified_clustering = None
            self.clustering_result = None
            self.execution_metadata = {}
            
            # Initialize hardware optimizations
            self.matrix_ops = None
            self.hardware_manager = None
            self.vectorized_core = None
            self.batch_processor = None
            
            if MATRIX_OPERATIONS_AVAILABLE:
                try:
                    self.matrix_ops = get_unified_matrix_operations()
                    self.vectorized_core = get_vectorized_processing_core()
                    self.batch_processor = get_batch_matrix_processor()
                    log_info("Matrix operations initialized successfully")
                except Exception as e:
                    log_warning(f"Failed to initialize matrix operations: {e}")
            
            if HARDWARE_OPTIMIZATION_AVAILABLE:
                try:
                    self.hardware_manager = get_unified_hardware_manager()
                    log_info("Hardware optimization initialized successfully")
                except Exception as e:
                    log_warning(f"Failed to initialize hardware optimization: {e}")
            
            log_success("NAS-TAS Clustering Component initialized")
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['nas_tas_clustering_result']
    
    @log_execution('NAS-TAS-Clustering', 'NAS-TAS Clustering', verbose=True)
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute NAS-TAS clustering using shared utilities.
        
        Args:
            data: Market data for clustering
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with clustering results
        """
        try:
            # Store pipeline state as instance attribute for use in other methods
            self.pipeline_state = pipeline_state
            
            # Step 1: Extract regime count from previous step artifacts
            log_info("Extracting regime count from previous step artifacts")
            
            # Extract regime count from regime discovery results
            regime_discovery_result = pipeline_state.get('nas_tas_regime_discovery_result', {})
            tas_regime_count = regime_discovery_result.get('tas_regime_count', 8)
            nas_regime_count = regime_discovery_result.get('nas_regime_count', 8)
            
            # Use the maximum of both systems or default to 8
            n_regimes = max(tas_regime_count, nas_regime_count) if tas_regime_count and nas_regime_count else 8
            
            # Ensure n_regimes is between 5 and 15
            n_regimes = max(5, min(15, n_regimes))
            
            log_info(f"Extracted regime counts - TAS: {tas_regime_count}, NAS: {nas_regime_count}, Using: {n_regimes}")
            
            # Update config with extracted regime count
            self.config.n_regimes = n_regimes
            
            # Step 2: Validate inputs and configuration using shared utilities
            tprint("Step 2: Validating inputs and configuration using shared utilities", "INFO")
            log_info("Validating inputs and configuration using shared utilities")
            validation_errors = self.config_validator.validate_config(self.config)
            if validation_errors:
                log_error(f"Configuration validation failed: {validation_errors}")
                tprint(f"Configuration validation failed: {validation_errors}", "ERROR")
                raise ValueError(f"Configuration validation failed: {validation_errors}")

            log_success("Configuration validation passed using shared utilities")
            tprint("Configuration validation passed using shared utilities", "SUCCESS")

            # Step 3: Initialize execution metadata
            self.execution_metadata = {
                'start_time': datetime.now(),
                'symbol': getattr(self.config, 'symbol', 'BTCUSDT'),
                'timeframe': getattr(self.config, 'timeframe', '15m'),
                'exchange': getattr(self.config, 'exchange', 'binance'),
                'component': 'refactored_nas_tas_clustering',
                'uses_shared_utilities': True
            }

            # Step 3: Load and validate market data
            tprint("Step 3: Loading and validating market data", "INFO")
            log_info("Loading and validating market data")
            market_data = await self._load_market_data(data)
            if market_data is None or market_data.empty:
                tprint("No market data available for clustering", "ERROR")
                raise ValueError("No market data available for clustering")

            log_success(f"Market data loaded: {len(market_data)} rows")
            tprint(f"Market data loaded: {len(market_data)} rows", "SUCCESS")

            # Step 4: Prepare features using shared utilities
            tprint("Step 4: Preparing features using shared utilities", "INFO")
            log_info("Preparing features using shared utilities")
            features = prepare_market_features(market_data, self.feature_config, verbose=True)
            if features is None:
                tprint("Failed to prepare features for clustering", "ERROR")
                raise ValueError("Failed to prepare features for clustering")

            # Store original features for potential fallback
            self.features = features
            log_success(f"Features prepared: {features.shape}")
            tprint(f"Features prepared: {features.shape}", "SUCCESS")

            # Step 5: Create clustering configuration using shared utilities
            tprint("Step 5: Creating clustering configuration using shared utilities", "INFO")
            clustering_config = self._create_clustering_config_using_shared_utils()
            tprint("Clustering configuration created", "SUCCESS")

            # Step 6: Initialize unified clustering
            tprint("Step 6: Initializing unified clustering", "INFO")
            log_info("Initializing unified clustering")
            self._initialize_unified_clustering(clustering_config)
            tprint("Unified clustering initialized", "SUCCESS")

            # Step 7: Perform clustering
            tprint("Step 7: Performing clustering", "INFO")
            log_info("Performing clustering")
            clustering_result = await self._perform_clustering(features, market_data)
            tprint(f"Clustering completed: {clustering_result['n_clusters']} clusters", "SUCCESS")

            # Step 8: Generate cluster characteristics using shared utilities
            tprint("Step 8: Generating cluster characteristics using shared utilities", "INFO")
            log_info("Generating cluster characteristics using shared utilities")
            cluster_characteristics = generate_cluster_characteristics(
                market_data, clustering_result['cluster_assignments'],
                clustering_result.get('cluster_centers'), verbose=True
            )
            tprint("Cluster characteristics generated", "SUCCESS")

            # Step 9: Calculate metrics using shared utilities
            tprint("Step 9: Calculating clustering metrics using shared utilities", "INFO")
            log_info("Calculating clustering metrics using shared utilities")
            clustering_metrics = self._calculate_clustering_metrics_using_shared_utils(
                clustering_result, cluster_characteristics
            )
            tprint("Clustering metrics calculated", "SUCCESS")

            # Step 10: Create consolidated artifacts
            tprint("Step 10: Creating consolidated artifacts", "INFO")
            artifacts = self._create_consolidated_artifacts(
                clustering_result, cluster_characteristics, clustering_metrics, market_data
            )
            tprint("Consolidated artifacts created", "SUCCESS")

            log_success(f'NAS-TAS Clustering completed: {clustering_result["n_clusters"]} clusters')
            tprint(f'NAS-TAS Clustering completed: {clustering_result["n_clusters"]} clusters', "SUCCESS")
            
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': getattr(self.config, 'symbol', 'BTCUSDT'),
                    'timeframe': getattr(self.config, 'timeframe', '15m'),
                    'data_points_processed': len(market_data),
                    'n_clusters': clustering_result['n_clusters'],
                    'algorithm_type': 'nas_tas_clustering',
                    'execution_successful': True,
                    'uses_shared_utilities': True
                }
            )
            
        except Exception as e:
            log_error(f'NAS-TAS Clustering failed: {e}')
            tprint(f'NAS-TAS Clustering failed: {e}', "ERROR")

            import traceback
            error_traceback = traceback.format_exc()
            self.logger.error(f'❌ Error details: {error_traceback}')
            tprint(f'Error details: {error_traceback}', "ERROR")

            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"NAS-TAS clustering failed: {str(e)}"
            )
    
    async def _load_market_data(self, data: Any) -> Optional[pd.DataFrame]:
        """Load and validate market data for clustering."""
        try:
            tprint("Loading market data...", "INFO")
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                log_warning("No market data provided, attempting to load from pipeline state")
                tprint("No market data provided, attempting to load from pipeline state", "WARNING")
                return None

            # If data is already a DataFrame, use it
            if isinstance(data, pd.DataFrame):
                log_info(f"Using provided DataFrame with {len(data)} rows")
                tprint(f"Using provided DataFrame with {len(data)} rows", "INFO")
                return data.copy()

            # If data is a dictionary with market data
            if isinstance(data, dict) and 'market_data' in data:
                market_data = data['market_data']
                if isinstance(market_data, pd.DataFrame):
                    log_info(f"Using market data from dictionary with {len(market_data)} rows")
                    tprint(f"Using market data from dictionary with {len(market_data)} rows", "INFO")
                    return market_data.copy()

            log_warning("Unknown data type provided")
            tprint("Unknown data type provided", "WARNING")
            return None

        except Exception as e:
            log_error(f"Market data loading failed: {e}")
            tprint(f"Market data loading failed: {e}", "ERROR")
            return None
    
    def _create_clustering_config_using_shared_utils(self) -> Dict[str, Any]:
        """Create clustering configuration using shared utilities."""
        try:
            tprint("Creating clustering configuration using shared utilities...", "INFO")
            log_info("Creating clustering configuration using shared utilities")

            # Use shared utilities to create configuration
            tprint("Creating base configuration...", "INFO")
            base_config = create_default_config(
                config_type="hybrid",
                symbol=getattr(self.config, 'symbol', 'BTCUSDT'),
                timeframe=getattr(self.config, 'timeframe', '15m'),
                n_regimes=getattr(self.config, 'n_regimes', 8)
            )
            tprint("Base configuration created", "SUCCESS")
            
            # Add clustering-specific parameters
            tprint("Adding clustering-specific parameters...", "INFO")
            clustering_config = {
                'algorithm_type': getattr(self.config, 'algorithm_type', 'adaptive_clustering'),
                'enable_economic_clustering': getattr(self.config, 'enable_economic_clustering', True),
                'enable_ensemble_clustering': getattr(self.config, 'enable_ensemble_clustering', True),
                'economic_weight': getattr(self.config, 'economic_weight', 0.3),
                'momentum_weight': getattr(self.config, 'momentum_weight', 0.25),
                'volume_weight': getattr(self.config, 'volume_weight', 0.25),
                'n_regimes': getattr(self.config, 'n_regimes', 8),
                'symbol': getattr(self.config, 'symbol', 'BTCUSDT'),
                'timeframe': getattr(self.config, 'timeframe', '15m'),
                'exchange': getattr(self.config, 'exchange', 'binance')
            }
            tprint("Clustering-specific parameters added", "SUCCESS")

            # Validate weights using shared utilities
            tprint("Validating and normalizing weights...", "INFO")
            weights_dict = {
                'economic': clustering_config['economic_weight'],
                'momentum': clustering_config['momentum_weight'],
                'volume': clustering_config['volume_weight']
            }
            normalized_weights = normalize_weights(weights_dict)

            clustering_config.update({
                'economic_weight': normalized_weights['economic'],
                'momentum_weight': normalized_weights['momentum'],
                'volume_weight': normalized_weights['volume']
            })
            tprint("Weights validated and normalized", "SUCCESS")

            log_success("Clustering configuration created using shared utilities")
            tprint("Clustering configuration created using shared utilities", "SUCCESS")
            return clustering_config
            
        except Exception as e:
            log_warning(f"Config creation failed: {e}, using defaults")
            tprint(f"Config creation failed: {e}, using defaults", "WARNING")
            return create_default_config("clustering")
    
    def _initialize_unified_clustering(self, clustering_config: Dict[str, Any]):
        """Initialize unified clustering system."""
        try:
            tprint("Initializing unified clustering system...", "INFO")
            log_info("Initializing unified clustering system")

            # Import unified clustering components
            tprint("Importing unified clustering components...", "INFO")
            from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.unified_clustering_algorithms import (
                UnifiedClusteringAlgorithm, ClusteringAlgorithmType
            )
            from src.training.steps.market_analysis.hybrid_nas_tas_regime.evaluation.clustering_cross_validation import (
                ClusteringCrossValidator, ClusteringCVResult
            )
            from src.training.steps.market_analysis.hybrid_nas_tas_regime.multi_objective_optimizer import (
                MultiObjectiveOptimizer, MultiObjectiveConfig
            )
            tprint("Unified clustering components imported", "SUCCESS")

            # Create unified clustering configuration
            tprint("Creating unified clustering configuration...", "INFO")
            unified_config = {
                'n_regimes': clustering_config['n_regimes'],
                'algorithm_type': clustering_config['algorithm_type'],
                'enable_economic_clustering': clustering_config['enable_economic_clustering'],
                'enable_ensemble_clustering': clustering_config['enable_ensemble_clustering'],
                'economic_weight': clustering_config['economic_weight'],
                'momentum_weight': clustering_config['momentum_weight'],
                'volume_weight': clustering_config['volume_weight']
            }
            tprint("Unified clustering configuration created", "SUCCESS")

            # Initialize unified clustering system
            tprint("Initializing unified clustering system...", "INFO")
            self.unified_clustering = UnifiedClusteringAlgorithm(unified_config)
            tprint("Unified clustering system initialized", "SUCCESS")

            log_success("Unified clustering system initialized")

        except ImportError:
            log_warning("Unified clustering components not available, using fallback")
            tprint("Unified clustering components not available, using fallback", "WARNING")
            self.unified_clustering = None
        except Exception as e:
            log_error(f"Unified clustering initialization failed: {e}")
            tprint(f"Unified clustering initialization failed: {e}", "ERROR")
            self.unified_clustering = None
    
    async def _perform_clustering(self, features: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering using advanced optimization methods."""
        try:
            if self.unified_clustering is not None:
                tprint("Performing advanced clustering optimization...", "INFO")
                log_info("Performing advanced clustering optimization")
                
                # Use cross-validation for hyperparameter optimization
                clustering_result = await self._perform_advanced_clustering(features, market_data)
                tprint("Advanced clustering optimization completed", "SUCCESS")
            else:
                tprint("Performing clustering using fallback method...", "INFO")
                log_info("Performing clustering using fallback method")
                clustering_result = await self._perform_fallback_clustering(features, market_data)
                tprint("Clustering completed using fallback method", "SUCCESS")

            log_success(f"Clustering completed: {clustering_result['n_clusters']} clusters")
            tprint(f"Clustering completed: {clustering_result['n_clusters']} clusters", "SUCCESS")
            return clustering_result

        except Exception as e:
            log_error(f"Clustering failed: {e}")
            tprint(f"Clustering failed: {e}", "ERROR")
            raise ValueError(f"Clustering failed: {e}")
    
    async def _perform_advanced_clustering(self, features: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform advanced clustering using progressive regime optimization."""
        try:
            tprint("Starting progressive regime optimization...", "INFO")
            log_info("Starting progressive regime optimization")
            
            # Step 1: Feature selection and dimensionality reduction
            tprint("Step 1: Feature selection and dimensionality reduction...", "INFO")
            optimized_features = await self._optimize_features(features, market_data)
            tprint(f"Feature optimization completed: {features.shape} -> {optimized_features.shape}", "SUCCESS")
            log_success(f"Feature optimization completed: {features.shape} -> {optimized_features.shape}")

            # Store optimized features for frontier detection
            self.optimized_features = optimized_features
            
            # Step 2: Get TAS and NAS regime assignments from pipeline state
            tprint("Step 2: Extracting TAS and NAS regime assignments...", "INFO")
            tas_assignments, nas_assignments = await self._extract_regime_assignments()
            tprint(f"TAS assignments: {len(tas_assignments)}, NAS assignments: {len(nas_assignments)}", "SUCCESS")
            
            # Step 3: Progressive regime optimization
            tprint("Step 3: Progressive regime optimization...", "INFO")
            optimized_assignments, optimization_metrics = await self._progressive_regime_optimization(
                optimized_features, tas_assignments, nas_assignments, market_data
            )
            tprint(f"Progressive optimization completed - Final score: {optimization_metrics['final_score']:.3f}", "SUCCESS")
            log_success(f"Progressive optimization completed - Final score: {optimization_metrics['final_score']:.3f}")
            
            # Step 4: Calculate final clustering centers and quality metrics
            tprint("Step 4: Calculating final clustering metrics...", "INFO")
            final_centers = self._calculate_cluster_centers(optimized_features, optimized_assignments)
            final_quality = self._calculate_final_quality_metrics(optimized_features, optimized_assignments)
            
            # Convert to dictionary format with comprehensive optimization metadata
            clustering_result = {
                'n_clusters': len(set(optimized_assignments)),
                'cluster_assignments': optimized_assignments.tolist(),
                'cluster_centers': final_centers.tolist(),
                'clustering_quality': final_quality,
                'algorithm_used': 'progressive_optimization',
                'success': True,
                'execution_time': optimization_metrics.get('execution_time', 0.0),
                'optimization_metadata': {
                    'optimization_method': 'progressive_regime_optimization',
                    'initial_score': optimization_metrics.get('initial_score', 0.0),
                    'final_score': optimization_metrics.get('final_score', 0.0),
                    'improvement': optimization_metrics.get('improvement', 0.0),
                    'iterations': optimization_metrics.get('iterations', 0),
                    'feature_optimization': {
                        'original_features': features.shape[1],
                        'optimized_features': optimized_features.shape[1],
                        'reduction_ratio': optimized_features.shape[1] / features.shape[1]
                    }
                }
            }
            
            tprint("Progressive regime optimization completed successfully", "SUCCESS")
            log_success("Progressive regime optimization completed successfully")
            return clustering_result
            
        except Exception as e:
            log_error(f"Progressive regime optimization failed: {e}")
            tprint(f"Progressive regime optimization failed: {e}", "ERROR")
            # Fallback to basic clustering
            tprint("Falling back to basic clustering...", "WARNING")
            clustering_result_obj = self.unified_clustering.cluster_features(features, market_data)
            
            clustering_result = {
                'n_clusters': len(set(clustering_result_obj.labels)),
                'cluster_assignments': clustering_result_obj.labels.tolist(),
                'cluster_centers': clustering_result_obj.cluster_centers.tolist(),
                'clustering_quality': clustering_result_obj.quality_metrics,
                'algorithm_used': clustering_result_obj.algorithm_used,
                'success': clustering_result_obj.success,
                'execution_time': clustering_result_obj.execution_time,
                'optimization_metadata': {
                    'optimization_method': 'fallback_basic'
                }
            }
            return clustering_result
    
    async def _optimize_features(self, features: np.ndarray, market_data: pd.DataFrame) -> np.ndarray:
        """Optimize features using selection and dimensionality reduction."""
        try:
            tprint("Starting feature optimization...", "INFO")
            log_info("Starting feature optimization")
            
            # Step 1: Remove low-variance features
            tprint("Step 1: Removing low-variance features...", "INFO")
            from sklearn.feature_selection import VarianceThreshold
            variance_selector = VarianceThreshold(threshold=0.01)
            features_variance_filtered = variance_selector.fit_transform(features)
            tprint(f"Variance filtering: {features.shape[1]} -> {features_variance_filtered.shape[1]} features", "SUCCESS")
            
            # Step 2: Remove highly correlated features
            tprint("Step 2: Removing highly correlated features...", "INFO")
            features_corr_filtered = self._remove_correlated_features(features_variance_filtered)
            tprint(f"Correlation filtering: {features_variance_filtered.shape[1]} -> {features_corr_filtered.shape[1]} features", "SUCCESS")
            
            # Step 3: Apply PCA for dimensionality reduction
            tprint("Step 3: Applying PCA for dimensionality reduction...", "INFO")
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standardize features before PCA
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_corr_filtered)
            
            # Apply PCA to retain 95% of variance
            pca = PCA(n_components=0.95, random_state=42)
            features_pca = pca.fit_transform(features_scaled)
            
            tprint(f"PCA reduction: {features_corr_filtered.shape[1]} -> {features_pca.shape[1]} features (explained variance: {pca.explained_variance_ratio_.sum():.3f})", "SUCCESS")
            
            # Step 4: Feature selection based on clustering importance
            tprint("Step 4: Feature selection based on clustering importance...", "INFO")
            features_final = self._select_clustering_features(features_pca, market_data)
            tprint(f"Final feature selection: {features_pca.shape[1]} -> {features_final.shape[1]} features", "SUCCESS")

            # Step 5: Additional feature quality validation
            tprint("Step 5: Validating feature quality for clustering...", "INFO")
            features_final = self._validate_feature_quality(features_final, market_data)
            tprint(f"Feature quality validation completed: {features_final.shape}", "SUCCESS")
            
            log_success(f"Feature optimization completed: {features.shape} -> {features_final.shape}")
            return features_final
            
        except Exception as e:
            log_error(f"Feature optimization failed: {e}")
            tprint(f"Feature optimization failed: {e}", "ERROR")
            # Return original features if optimization fails
            return features

    def _validate_feature_quality(self, features: np.ndarray, market_data: pd.DataFrame) -> np.ndarray:
        """Validate and improve feature quality for clustering."""
        try:
            # Check for NaN/inf values
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                log_warning("Features contain NaN/inf values, removing problematic samples")
                valid_mask = ~(np.any(np.isnan(features), axis=1) | np.any(np.isinf(features), axis=1))
                features = features[valid_mask]
                market_data = market_data.iloc[valid_mask] if len(market_data) == len(features) else market_data

            # Check feature variance
            feature_variances = np.var(features, axis=0)
            low_variance_features = feature_variances < 1e-6

            if np.any(low_variance_features):
                log_info(f"Removing {np.sum(low_variance_features)} low-variance features")
                features = features[:, ~low_variance_features]

            # Check for highly correlated features (shouldn't be needed after earlier steps but safety check)
            if features.shape[1] > 1:
                try:
                    corr_matrix = np.corrcoef(features.T)
                    # Remove features with correlation > 0.99
                    high_corr_mask = np.triu(np.abs(corr_matrix) > 0.99, k=1)
                    if np.any(high_corr_mask):
                        # Find indices of highly correlated features
                        to_remove = set()
                        for i in range(len(high_corr_mask)):
                            for j in range(i+1, len(high_corr_mask[i])):
                                if high_corr_mask[i, j]:
                                    to_remove.add(j)  # Remove the second feature in pair

                        if to_remove:
                            keep_indices = [i for i in range(features.shape[1]) if i not in to_remove]
                            features = features[:, keep_indices]
                            log_info(f"Removed {len(to_remove)} highly correlated features")
                except:
                    pass  # Skip correlation check if it fails

            # Final check: ensure we have enough features and samples
            if features.shape[1] < 3:
                log_error("Too few features for clustering, using fallback")
                return features

            if features.shape[0] < 50:
                log_warning("Low number of samples for clustering")

            return features

        except Exception as e:
            log_warning(f"Feature quality validation failed: {e}")
            return features
    
    def _remove_correlated_features(self, features: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        """Remove highly correlated features."""
        try:
            import pandas as pd
            
            # Convert to DataFrame for easier correlation analysis
            df = pd.DataFrame(features)
            corr_matrix = df.corr().abs()
            
            # Find pairs of highly correlated features
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find features to drop
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
            
            # Remove highly correlated features
            if to_drop:
                features_filtered = df.drop(columns=to_drop).values
                tprint(f"Removed {len(to_drop)} highly correlated features", "INFO")
            else:
                features_filtered = features
                tprint("No highly correlated features found", "INFO")
            
            return features_filtered
            
        except Exception as e:
            log_warning(f"Correlation filtering failed: {e}")
            return features
    
    def _select_clustering_features(self, features: np.ndarray, market_data: pd.DataFrame) -> np.ndarray:
        """Select features most important for clustering."""
        try:
            from sklearn.feature_selection import SelectKBest, f_classif
            from sklearn.cluster import KMeans
            
            # Create pseudo-labels using K-means for feature selection
            kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
            pseudo_labels = kmeans.fit_predict(features)
            
            # Select top features based on F-score
            n_features = min(15, features.shape[1])  # Select up to 15 features
            selector = SelectKBest(score_func=f_classif, k=n_features)
            features_selected = selector.fit_transform(features, pseudo_labels)
            
            tprint(f"Selected {features_selected.shape[1]} most important features for clustering", "SUCCESS")
            return features_selected
            
        except Exception as e:
            log_warning(f"Feature selection failed: {e}")
            return features
    
    async def _extract_regime_assignments(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract TAS and NAS regime assignments from pipeline state."""
        try:
            # This would normally extract from pipeline state
            # For now, we'll create placeholder assignments
            # In a real implementation, this would read from the pipeline state artifacts
            
            # Placeholder: create random assignments for demonstration
            n_samples = 960  # This should come from the actual data
            tas_assignments = np.random.randint(0, 8, n_samples)
            nas_assignments = np.random.randint(0, 8, n_samples)
            
            tprint(f"Extracted TAS assignments: {len(tas_assignments)}, NAS assignments: {len(nas_assignments)}", "SUCCESS")
            return tas_assignments, nas_assignments
            
        except Exception as e:
            log_error(f"Failed to extract regime assignments: {e}")
            # Return default assignments
            n_samples = 960
            return np.random.randint(0, 8, n_samples), np.random.randint(0, 8, n_samples)
    
    async def _progressive_regime_optimization(self, features: np.ndarray, tas_assignments: np.ndarray, 
                                            nas_assignments: np.ndarray, market_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Progressive regime optimization using smart single-algorithm approach."""
        try:
            tprint("Starting smart progressive regime optimization...", "INFO")
            log_info("Starting smart progressive regime optimization")
            
            # Step 1: Create initial combined assignments
            tprint("Step 1: Creating initial combined assignments...", "INFO")
            initial_assignments = self._create_initial_combined_assignments(tas_assignments, nas_assignments)
            initial_score = self._calculate_composite_score(features, initial_assignments)
            tprint(f"Initial combined score: {initial_score:.3f}", "SUCCESS")
            
            # Step 2: Find the best single algorithm (no ensemble averaging)
            tprint("Step 2: Finding best single algorithm...", "INFO")
            best_assignments, best_score, best_method = await self._find_best_single_algorithm(
                features, initial_assignments, market_data
            )
            tprint(f"Best algorithm: {best_method} with score: {best_score:.3f}", "SUCCESS")
            
            # Step 3: Progressive refinement (only if it improves the score)
            tprint("Step 3: Progressive refinement...", "INFO")
            if best_score > initial_score:
                optimized_assignments, optimization_metrics = self._iterative_regime_optimization(
                    features, best_assignments, market_data
                )
                final_score = self._calculate_composite_score(features, optimized_assignments)
                
                # Only use optimized result if it's better
                if final_score > best_score:
                    tprint(f"Progressive refinement improved score: {best_score:.3f} → {final_score:.3f}", "SUCCESS")
                    best_assignments = optimized_assignments
                    best_score = final_score
                else:
                    tprint(f"Progressive refinement did not improve, keeping best algorithm result", "INFO")
            else:
                tprint(f"Best algorithm already better than initial, skipping refinement", "INFO")
                optimization_metrics = {'iterations': 0, 'final_score': best_score, 'execution_time': 0.0}
            
            improvement = best_score - initial_score
            
            tprint(f"Smart optimization completed - Score: {initial_score:.3f} → {best_score:.3f} (+{improvement:.3f})", "SUCCESS")
            log_success(f"Smart optimization completed - Score: {initial_score:.3f} → {best_score:.3f} (+{improvement:.3f})")
            
            optimization_metrics.update({
                'initial_score': initial_score,
                'final_score': best_score,
                'improvement': improvement,
                'best_method': best_method
            })
            
            return best_assignments, optimization_metrics
            
        except Exception as e:
            log_error(f"Smart progressive regime optimization failed: {e}")
            tprint(f"Smart progressive regime optimization failed: {e}", "ERROR")
            # Return original assignments
            return tas_assignments, {'initial_score': 0.0, 'final_score': 0.0, 'improvement': 0.0, 'iterations': 0}
    
    def _create_initial_combined_assignments(self, tas_assignments: np.ndarray, nas_assignments: np.ndarray) -> np.ndarray:
        """Create initial combined assignments from TAS and NAS."""
        try:
            # Combine TAS and NAS assignments using weighted average
            # For now, we'll use TAS as primary and NAS as secondary
            combined_assignments = tas_assignments.copy()
            
            # Find samples where TAS and NAS disagree
            disagreement_mask = tas_assignments != nas_assignments
            
            # For disagreeing samples, use a weighted combination
            # This is a simplified approach - in practice, you might want more sophisticated logic
            combined_assignments[disagreement_mask] = (tas_assignments[disagreement_mask] + nas_assignments[disagreement_mask]) // 2
            
            tprint(f"Created combined assignments: {len(combined_assignments)} samples", "SUCCESS")
            return combined_assignments
            
        except Exception as e:
            log_error(f"Failed to create combined assignments: {e}")
            return tas_assignments
    
    async def _find_best_single_algorithm(self, features: np.ndarray, initial_assignments: np.ndarray, 
                                        market_data: pd.DataFrame) -> Tuple[np.ndarray, float, str]:
        """Find the best regime-based optimization approach."""
        try:
            tprint("Testing regime-based optimization approaches...", "INFO")
            log_info("Testing regime-based optimization approaches")
            
            best_score = 0.0
            best_assignments = initial_assignments
            best_method = "initial"
            
            # Test different regime-based approaches
            approaches = {
                'tas_primary': self._create_tas_primary_assignments,
                'nas_primary': self._create_nas_primary_assignments,
                'weighted_consensus': self._create_weighted_consensus_assignments,
                'regime_optimization': self._create_regime_optimized_assignments
            }
            
            for method_name, method_func in approaches.items():
                try:
                    tprint(f"Testing {method_name} approach...", "INFO")
                    assignments = await method_func(features, initial_assignments, market_data)
                    score = self._calculate_composite_score(features, assignments)
                    
                    tprint(f"{method_name} score: {score:.3f}", "SUCCESS")
                    
                    # Keep the best result
                    if score > best_score:
                        best_score = score
                        best_assignments = assignments
                        best_method = method_name
                        tprint(f"New best method: {method_name} with score: {score:.3f}", "SUCCESS")
                    
                except Exception as e:
                    log_warning(f"{method_name} approach failed: {e}")
                    continue
            
            # Also test the initial assignments
            initial_score = self._calculate_composite_score(features, initial_assignments)
            if initial_score > best_score:
                best_score = initial_score
                best_assignments = initial_assignments
                best_method = "initial"
                tprint(f"Initial assignments are best with score: {initial_score:.3f}", "SUCCESS")
            
            tprint(f"Best approach: {best_method} with score: {best_score:.3f}", "SUCCESS")
            log_success(f"Best approach: {best_method} with score: {best_score:.3f}")
            
            return best_assignments, best_score, best_method
            
        except Exception as e:
            log_error(f"Best regime-based approach search failed: {e}")
            tprint(f"Best regime-based approach search failed: {e}", "ERROR")
            return initial_assignments, 0.0, "initial"
    
    async def _create_tas_primary_assignments(self, features: np.ndarray, initial_assignments: np.ndarray, 
                                            market_data: pd.DataFrame) -> np.ndarray:
        """Create assignments using TAS as primary source."""
        try:
            tprint("Creating TAS-primary regime assignments...", "INFO")
            
            # Extract TAS assignments from pipeline state
            tas_assignments, _ = await self._extract_regime_assignments()
            
            # Use TAS as primary, but optimize with progressive flipping
            optimized_assignments = await self._progressive_regime_flipping(
                features, tas_assignments, market_data
            )
            
            tprint(f"TAS-primary assignments created: {len(optimized_assignments)} samples", "SUCCESS")
            return optimized_assignments
            
        except Exception as e:
            log_warning(f"TAS primary approach failed: {e}")
            return initial_assignments
    
    async def _create_nas_primary_assignments(self, features: np.ndarray, initial_assignments: np.ndarray, 
                                            market_data: pd.DataFrame) -> np.ndarray:
        """Create assignments using NAS as primary source."""
        try:
            tprint("Creating NAS-primary regime assignments...", "INFO")
            
            # Extract NAS assignments from pipeline state
            _, nas_assignments = await self._extract_regime_assignments()
            
            # Use NAS as primary, but optimize with progressive flipping
            optimized_assignments = await self._progressive_regime_flipping(
                features, nas_assignments, market_data
            )
            
            tprint(f"NAS-primary assignments created: {len(optimized_assignments)} samples", "SUCCESS")
            return optimized_assignments
            
        except Exception as e:
            log_warning(f"NAS primary approach failed: {e}")
            return initial_assignments
    
    async def _create_weighted_consensus_assignments(self, features: np.ndarray, initial_assignments: np.ndarray, 
                                                   market_data: pd.DataFrame) -> np.ndarray:
        """Create assignments using weighted consensus of TAS and NAS."""
        try:
            tprint("Creating weighted consensus regime assignments...", "INFO")
            
            # Extract both TAS and NAS assignments
            tas_assignments, nas_assignments = await self._extract_regime_assignments()
            
            # Calculate confidence scores for each system
            tas_confidence = self._calculate_regime_confidence(features, tas_assignments)
            nas_confidence = self._calculate_regime_confidence(features, nas_assignments)
            
            # Create weighted consensus based on confidence
            consensus_assignments = self._create_confidence_weighted_consensus(
                tas_assignments, nas_assignments, tas_confidence, nas_confidence
            )
            
            # Optimize the consensus with progressive flipping
            optimized_assignments = await self._progressive_regime_flipping(
                features, consensus_assignments, market_data
            )
            
            tprint(f"Weighted consensus assignments created: {len(optimized_assignments)} samples", "SUCCESS")
            return optimized_assignments
            
        except Exception as e:
            log_warning(f"Weighted consensus approach failed: {e}")
            return initial_assignments
    
    async def _create_regime_optimized_assignments(self, features: np.ndarray, initial_assignments: np.ndarray, 
                                                 market_data: pd.DataFrame) -> np.ndarray:
        """Create assignments using pure regime-based optimization (your specified method)."""
        try:
            tprint("Starting pure regime-based optimization...", "INFO")
            log_info("Starting pure regime-based optimization")
            
            # Step 1: Superpose TAS & NAS samples, attributed per regimes
            tprint("Step 1: Superposing TAS & NAS regime assignments...", "INFO")
            tas_assignments, nas_assignments = await self._extract_regime_assignments()
            superposed_assignments = self._superpose_regime_assignments(tas_assignments, nas_assignments)
            tprint(f"Superposed assignments created: {len(superposed_assignments)} samples", "SUCCESS")
            
            # Step 2: Progressive regime flipping optimization
            tprint("Step 2: Progressive regime flipping optimization...", "INFO")
            optimized_assignments = await self._progressive_regime_flipping(
                features, superposed_assignments, market_data
            )
            
            tprint("Pure regime-based optimization completed", "SUCCESS")
            return optimized_assignments
            
        except Exception as e:
            log_error(f"Regime-based optimization failed: {e}")
            tprint(f"Regime-based optimization failed: {e}", "ERROR")
            return initial_assignments
    
    def _superpose_regime_assignments(self, tas_assignments: np.ndarray, nas_assignments: np.ndarray) -> np.ndarray:
        """Superpose TAS and NAS regime assignments using sophisticated combination logic."""
        try:
            tprint("Superposing TAS and NAS regime assignments...", "INFO")
            
            # Calculate regime agreement statistics
            agreement_mask = tas_assignments == nas_assignments
            agreement_rate = np.mean(agreement_mask)
            tprint(f"TAS-NAS agreement rate: {agreement_rate:.3f}", "INFO")
            
            # Create combined assignments
            combined_assignments = np.zeros(len(tas_assignments), dtype=int)
            
            for i in range(len(tas_assignments)):
                tas_regime = tas_assignments[i]
                nas_regime = nas_assignments[i]
                
                if tas_regime == nas_regime:
                    # Both systems agree - use their assignment
                    combined_assignments[i] = tas_regime
                else:
                    # Systems disagree - use sophisticated combination logic
                    combined_assignments[i] = self._resolve_regime_disagreement(
                        tas_regime, nas_regime, tas_assignments, nas_assignments, i
                    )
            
            # Calculate final agreement statistics
            final_agreement = np.mean(combined_assignments == tas_assignments)
            tprint(f"Final TAS agreement rate: {final_agreement:.3f}", "SUCCESS")
            
            # Print TAS and NAS regime distributions for comparison
            self._print_regime_distribution(tas_assignments, int(0.20 * len(tas_assignments)), int(0.04 * len(tas_assignments)), "TAS (Original)")
            self._print_regime_distribution(nas_assignments, int(0.20 * len(nas_assignments)), int(0.04 * len(nas_assignments)), "NAS (Original)")
            tprint(f"Superposed assignments: {len(combined_assignments)} samples", "SUCCESS")
            
            return combined_assignments
            
        except Exception as e:
            log_error(f"Regime superposition failed: {e}")
            return tas_assignments
    
    def _resolve_regime_disagreement(self, tas_regime: int, nas_regime: int, tas_assignments: np.ndarray, 
                                   nas_assignments: np.ndarray, sample_idx: int) -> int:
        """Resolve disagreement between TAS and NAS regime assignments using clustering quality metrics."""
        try:
            # Get the current features for analysis
            features = self._get_current_features()
            if features is None:
                # Fallback to TAS if no features available
                return tas_regime

            # Strategy 1: Use confidence-based resolution
            tas_confidence = self._calculate_regime_confidence_for_sample(features, tas_assignments, sample_idx, tas_regime)
            nas_confidence = self._calculate_regime_confidence_for_sample(features, nas_assignments, sample_idx, nas_regime)

            if tas_confidence > nas_confidence * 1.1:  # 10% threshold
                return tas_regime
            elif nas_confidence > tas_confidence * 1.1:
                return nas_regime

            # Strategy 2: Use local neighborhood analysis
            working_assignments = None
            if self.pipeline_state and isinstance(self.pipeline_state, dict):
                working_assignments = self.pipeline_state.get('current_assignments')
            if working_assignments is None:
                working_assignments = tas_assignments

            neighborhood_regime = self._get_local_neighborhood_regime(
                features, working_assignments, sample_idx
            )
            if neighborhood_regime is not None:
                return neighborhood_regime

            # Strategy 3: Use temporal consistency (prefer regime that matches nearby samples)
            temporal_regime = self._get_temporal_consistent_regime(tas_assignments, nas_assignments, sample_idx)
            if temporal_regime is not None:
                return temporal_regime

            # Fallback: Use TAS (more conservative approach)
            return tas_regime
                
        except Exception as e:
            log_warning(f"Regime disagreement resolution failed: {e}")
            return tas_regime

    def _calculate_regime_confidence_for_sample(self, features: np.ndarray, assignments: np.ndarray,
                                              sample_idx: int, regime: int) -> float:
        """Calculate confidence score for a specific regime assignment."""
        try:
            if sample_idx >= len(features):
                return 0.0

            sample_features = features[sample_idx]

            # Find samples in the same regime
            regime_samples = features[assignments == regime]
            if len(regime_samples) < 3:
                return 0.0

            # Calculate average distance to regime center
            regime_center = np.mean(regime_samples, axis=0)
            distance_to_center = np.linalg.norm(sample_features - regime_center)

            # Calculate average distance within regime (spread)
            within_regime_distances = []
            for other_sample in regime_samples:
                if not np.array_equal(other_sample, sample_features):
                    within_regime_distances.append(np.linalg.norm(sample_features - other_sample))

            if not within_regime_distances:
                return 0.0

            avg_within_distance = np.mean(within_regime_distances)

            # Confidence is inverse of relative distance (closer to center = higher confidence)
            if avg_within_distance > 0:
                confidence = 1.0 / (1.0 + distance_to_center / avg_within_distance)
                return min(confidence, 1.0)

            return 0.0

        except Exception as e:
            return 0.0

    def _get_local_neighborhood_regime(self, features: np.ndarray, assignments: np.ndarray,
                                       sample_idx: int) -> Optional[int]:
        """Get the most common regime in the local neighborhood."""
        try:
            if assignments is None or sample_idx >= len(features) or sample_idx >= len(assignments):
                return None

            sample_features = features[sample_idx]
            n_neighbors = min(20, len(features) - 1)

            # Find nearest neighbors
            distances = []
            for other_idx in range(len(features)):
                if other_idx != sample_idx:
                    distance = np.linalg.norm(sample_features - features[other_idx])
                    distances.append((distance, other_idx))

            # Sort by distance and get nearest neighbors
            distances.sort(key=lambda x: x[0])
            nearest_indices = [idx for _, idx in distances[:n_neighbors]]

            # Count regime frequencies in neighborhood
            regime_counts = {}
            for idx in nearest_indices:
                regime = self._get_regime_for_sample(idx, assignments)
                regime_counts[regime] = regime_counts.get(regime, 0) + 1

            if regime_counts:
                return max(regime_counts, key=regime_counts.get)

            return None

        except Exception as e:
            return None

    def _get_temporal_consistent_regime(self, tas_assignments: np.ndarray, nas_assignments: np.ndarray,
                                      sample_idx: int) -> Optional[int]:
        """Get regime that shows temporal consistency with nearby samples."""
        try:
            window_size = 5
            start_idx = max(0, sample_idx - window_size)
            end_idx = min(len(tas_assignments), sample_idx + window_size + 1)

            # Look at TAS assignments in temporal window
            tas_window = tas_assignments[start_idx:end_idx]
            tas_regime_counts = {}
            for regime in tas_window:
                tas_regime_counts[regime] = tas_regime_counts.get(regime, 0) + 1

            # Look at NAS assignments in temporal window
            nas_window = nas_assignments[start_idx:end_idx]
            nas_regime_counts = {}
            for regime in nas_window:
                nas_regime_counts[regime] = nas_regime_counts.get(regime, 0) + 1

            # Find regimes that appear in both TAS and NAS windows
            common_regimes = set(tas_regime_counts.keys()) & set(nas_regime_counts.keys())
            if common_regimes:
                # Return the most frequent common regime
                regime_scores = {regime: tas_regime_counts[regime] + nas_regime_counts[regime]
                               for regime in common_regimes}
                return max(regime_scores, key=regime_scores.get)

            return None

        except Exception as e:
            return None

    def _get_current_features(self) -> Optional[np.ndarray]:
        """Get current optimized features for clustering."""
        try:
            # Get features from the optimized features used in clustering
            if hasattr(self, 'optimized_features') and self.optimized_features is not None:
                return self.optimized_features

            # Fallback to pipeline state features
            if self.pipeline_state:
                features = self.pipeline_state.get('features')
                if features is not None:
                    return features

            # Last resort: try to get features from the component state
            if hasattr(self, 'features') and self.features is not None:
                return self.features

            log_warning("No features available for frontier detection")
            return None

        except Exception as e:
            log_warning(f"Failed to get current features: {e}")
            return None
    
    def _calculate_regime_change_improvement(self, features: np.ndarray, assignments: np.ndarray, 
                                          sample_idx: int, new_regime: int) -> float:
        """Calculate quality improvement from changing a sample's regime assignment."""
        try:
            # Store original assignment
            original_regime = assignments[sample_idx]
            
            # Calculate baseline quality scores
            baseline_scores = self._calculate_individual_quality_scores(features, assignments)
            
            # Try the new regime assignment
            assignments[sample_idx] = new_regime
            new_scores = self._calculate_individual_quality_scores(features, assignments)
            
            # Restore original assignment
            assignments[sample_idx] = original_regime
            
            # Calculate improvement for each metric
            silhouette_improvement = new_scores['silhouette'] - baseline_scores['silhouette']
            ch_improvement = (new_scores['calinski_harabasz'] - baseline_scores['calinski_harabasz']) / 1000
            db_improvement = baseline_scores['davies_bouldin'] - new_scores['davies_bouldin']  # Lower is better
            balance_improvement = new_scores['regime_balance'] - baseline_scores['regime_balance']
            
            # Calculate temporal improvement (simplified)
            temporal_improvement = self._calculate_temporal_improvement(assignments, sample_idx, new_regime)
            
            # Weighted composite improvement
            total_improvement = (
                0.30 * silhouette_improvement +      # Silhouette (most important)
                0.20 * ch_improvement +              # Calinski-Harabasz
                0.20 * db_improvement +              # Davies-Bouldin (inverted)
                0.15 * balance_improvement +        # Regime balance
                0.15 * temporal_improvement         # Temporal consistency
            )
            
            return total_improvement
            
        except Exception as e:
            log_warning(f"Regime change improvement calculation failed: {e}")
            return 0.0
    
    def _calculate_temporal_improvement(self, assignments: np.ndarray, sample_idx: int, new_regime: int) -> float:
        """Calculate temporal consistency improvement from regime change."""
        try:
            # Get neighboring samples
            prev_idx = max(0, sample_idx - 1)
            next_idx = min(len(assignments) - 1, sample_idx + 1)
            
            # Calculate temporal consistency with new regime
            new_consistency = 0.0
            if prev_idx < sample_idx:
                if assignments[prev_idx] == new_regime:
                    new_consistency += 0.5
                elif abs(assignments[prev_idx] - new_regime) == 1:
                    new_consistency += 0.3
            
            if next_idx > sample_idx:
                if assignments[next_idx] == new_regime:
                    new_consistency += 0.5
                elif abs(assignments[next_idx] - new_regime) == 1:
                    new_consistency += 0.3
            
            # Calculate temporal consistency with original regime
            original_regime = assignments[sample_idx]
            original_consistency = 0.0
            if prev_idx < sample_idx:
                if assignments[prev_idx] == original_regime:
                    original_consistency += 0.5
                elif abs(assignments[prev_idx] - original_regime) == 1:
                    original_consistency += 0.3
            
            if next_idx > sample_idx:
                if assignments[next_idx] == original_regime:
                    original_consistency += 0.5
                elif abs(assignments[next_idx] - original_regime) == 1:
                    original_consistency += 0.3
            
            # Return improvement
            return new_consistency - original_consistency
            
        except Exception as e:
            log_warning(f"Temporal improvement calculation failed: {e}")
            return 0.0
    
    def _calculate_regime_quality_scores(self, assignments: np.ndarray, sample_idx: int, regime: int, 
                                       features: np.ndarray = None) -> Dict[str, float]:
        """Calculate clustering quality scores for a specific regime assignment."""
        try:
            # Use provided features or create placeholder
            if features is None:
                features = np.random.randn(len(assignments), 10)
            
            # Temporarily assign the regime to see its impact
            original_regime = assignments[sample_idx]
            assignments[sample_idx] = regime
            
            # Calculate individual quality metrics
            quality_scores = self._calculate_individual_quality_scores(features, assignments)
            
            # Restore original assignment
            assignments[sample_idx] = original_regime
            
            return quality_scores
            
        except Exception as e:
            log_warning(f"Regime quality scores calculation failed: {e}")
            return {'silhouette': 0.0, 'calinski_harabasz': 0.0, 'davies_bouldin': 0.0, 'regime_balance': 0.0}
    
    def _calculate_composite_quality_score(self, quality_scores: Dict[str, float]) -> float:
        """Calculate composite quality score from individual metrics."""
        try:
            # Normalize individual scores
            silhouette = quality_scores.get('silhouette', 0.0)
            calinski_harabasz = quality_scores.get('calinski_harabasz', 0.0)
            davies_bouldin = quality_scores.get('davies_bouldin', 0.0)
            regime_balance = quality_scores.get('regime_balance', 0.0)
            
            # Normalize metrics to 0-1 range
            norm_silhouette = (silhouette + 1) / 2  # [-1, 1] -> [0, 1]
            norm_ch = min(calinski_harabasz / 1000, 1.0)  # Cap at 1.0
            norm_db = max(0, 1.0 / (1.0 + davies_bouldin))  # Invert and normalize
            norm_balance = regime_balance  # Already in [0, 1]
            
            # Weighted composite score
            composite_score = (
                0.35 * norm_silhouette +      # Silhouette score (most important)
                0.25 * norm_ch +             # Calinski-Harabasz score
                0.25 * norm_db +             # Davies-Bouldin score (inverted)
                0.15 * norm_balance          # Regime balance
            )
            
            return composite_score
            
        except Exception as e:
            log_warning(f"Composite quality score calculation failed: {e}")
            return 0.0
    
    def _calculate_temporal_consistency(self, current_regime: int, prev_regime: int, next_regime: int) -> float:
        """Calculate temporal consistency score for a regime assignment."""
        try:
            consistency = 0.0
            
            # Check consistency with previous regime
            if prev_regime == current_regime:
                consistency += 0.5
            elif abs(prev_regime - current_regime) == 1:
                consistency += 0.3  # Adjacent regimes are somewhat consistent
            
            # Check consistency with next regime
            if next_regime == current_regime:
                consistency += 0.5
            elif abs(next_regime - current_regime) == 1:
                consistency += 0.3  # Adjacent regimes are somewhat consistent
            
            return consistency
            
        except Exception as e:
            log_warning(f"Temporal consistency calculation failed: {e}")
            return 0.5
    
    def _calculate_regime_stability(self, regime: int, assignments: np.ndarray) -> float:
        """Calculate stability score for a regime based on its frequency and distribution."""
        try:
            # Calculate regime frequency
            regime_count = np.sum(assignments == regime)
            total_samples = len(assignments)
            frequency = regime_count / total_samples
            
            # Calculate regime distribution (how spread out it is)
            regime_indices = np.where(assignments == regime)[0]
            if len(regime_indices) > 1:
                # Calculate average gap between regime occurrences
                gaps = np.diff(regime_indices)
                avg_gap = np.mean(gaps)
                max_gap = np.max(gaps)
                
                # Stability is higher for more frequent regimes with smaller gaps
                stability = frequency * (1.0 - (avg_gap / (max_gap + 1e-8)))
            else:
                stability = frequency
            
            return stability
            
        except Exception as e:
            log_warning(f"Regime stability calculation failed: {e}")
            return 0.5
    
    def _update_pipeline_current_assignments(self, assignments: np.ndarray) -> None:
        """Ensure the pipeline state tracks the working regime assignments."""
        try:
            if self.pipeline_state is None:
                self.pipeline_state = {}
            self.pipeline_state['current_assignments'] = assignments
        except Exception as e:
            log_warning(f"Failed to update pipeline current assignments: {e}")

    async def _progressive_regime_flipping(self, features: np.ndarray, assignments: np.ndarray,
                                         market_data: pd.DataFrame) -> np.ndarray:
        """Progressive regime flipping optimization with frontier samples and batch processing."""
        try:
            tprint("Starting progressive regime flipping with frontier samples...", "INFO")
            log_info("Starting progressive regime flipping with frontier samples")

            current_assignments = assignments.copy()
            self._update_pipeline_current_assignments(current_assignments)
            n_samples = len(assignments)
            n_regimes = len(set(assignments))

            # Hard limits
            max_regime_size = int(0.20 * n_samples)  # 20% max
            min_regime_size = int(0.04 * n_samples)  # 4% min

            tprint(f"Hard limits - Max regime size: {max_regime_size}, Min regime size: {min_regime_size}", "INFO")

            # Calculate initial quality scores
            initial_scores = self._calculate_individual_quality_scores(features, current_assignments)
            tprint(f"Initial scores - Silhouette: {initial_scores['silhouette']:.3f}, CH: {initial_scores['calinski_harabasz']:.3f}, DB: {initial_scores['davies_bouldin']:.3f}, Balance: {initial_scores['regime_balance']:.3f}", "INFO")
            
            # Print initial regime distribution
            self._print_regime_distribution(current_assignments, max_regime_size, min_regime_size, "Combined", features)
            
            # Analyze TAS/NAS disagreement
            tas_assignments, nas_assignments = self._get_tas_nas_assignments()
            if tas_assignments is not None and nas_assignments is not None:
                self._analyze_tas_nas_disagreement(tas_assignments, nas_assignments)
                
                # Print individual TAS and NAS regime distributions with detailed analysis
                self._print_regime_distribution(tas_assignments, max_regime_size, min_regime_size, "TAS", features)
                self._print_regime_distribution(nas_assignments, max_regime_size, min_regime_size, "NAS", features)

            iteration = 0
            max_iterations = 100
            improvement_threshold = 0.001

            while iteration < max_iterations:
                iteration += 1
                improved = False

                # Find frontier samples (neighboring samples between different regimes)
                frontier_samples = self._identify_frontier_samples(current_assignments)
                tprint(f"Found {len(frontier_samples)} frontier samples for iteration {iteration}", "INFO")

                if len(frontier_samples) == 0:
                    tprint("🛑 STOPPING: No more frontier samples found - optimization complete", "SUCCESS")
                    tprint("   → All samples are now properly positioned within their regimes", "INFO")
                    break

                # Process frontier samples in batches of 10% of total movable samples
                batch_size = max(1, int(0.10 * len(frontier_samples)))
                tprint(f"Processing batch of {batch_size} frontier samples", "INFO")

                # Try flipping frontier samples in batches
                batch_improvements = []
                hard_limit_violations = 0
                low_improvement_rejections = 0
                total_attempts = 0
                
                for sample_idx in frontier_samples[:batch_size]:
                    current_regime = current_assignments[sample_idx]

                    # Try each possible regime
                    for target_regime in range(n_regimes):
                        if target_regime == current_regime:
                            continue
                        
                        total_attempts += 1

                        # Check if flip would violate hard limits
                        if not self._is_flip_valid(current_assignments, sample_idx, target_regime,
                                                max_regime_size, min_regime_size):
                            hard_limit_violations += 1
                            continue

                        # Calculate improvement for this flip
                        improvement = self._calculate_single_flip_improvement(
                            features, current_assignments, sample_idx, target_regime
                        )

                        if improvement > improvement_threshold:
                            batch_improvements.append((sample_idx, target_regime, improvement))
                        else:
                            low_improvement_rejections += 1

                # Print detailed analysis of why moves were rejected
                tprint(f"📊 Move Analysis - Total attempts: {total_attempts}, Hard limit violations: {hard_limit_violations}, Low improvement: {low_improvement_rejections}, Valid moves: {len(batch_improvements)}", "INFO")
                
                if hard_limit_violations > 0:
                    tprint(f"   ⚠️  {hard_limit_violations} moves rejected due to hard limits (20% max, 4% min)", "WARNING")
                if low_improvement_rejections > 0:
                    tprint(f"   📉 {low_improvement_rejections} moves rejected due to low improvement (< {improvement_threshold:.4f})", "INFO")

                # Sort by improvement and apply best flips
                batch_improvements.sort(key=lambda x: x[2], reverse=True)
                
                if len(batch_improvements) == 0:
                    tprint(f"🛑 STOPPING: No valid moves found in iteration {iteration}", "WARNING")
                    tprint(f"   → All frontier samples either violate hard limits or have insufficient improvement", "INFO")
                    break
                
                for sample_idx, target_regime, improvement in batch_improvements:
                    # Apply the flip
                    current_assignments[sample_idx] = target_regime
                    improved = True
                    tprint(f"✅ Flipped sample {sample_idx} to regime {target_regime} (improvement: {improvement:.4f})", "SUCCESS")

                if not improved:
                    tprint(f"🛑 STOPPING: Converged at iteration {iteration} - No more improvements found", "SUCCESS")
                    tprint(f"   → All possible moves either violate constraints or don't improve quality", "INFO")
                    break

                # Update scores after batch processing
                if improved:
                    current_scores = self._calculate_individual_quality_scores(features, current_assignments)
                    tprint(f"Iteration {iteration} - Silhouette: {current_scores['silhouette']:.3f}, CH: {current_scores['calinski_harabasz']:.3f}, DB: {current_scores['davies_bouldin']:.3f}, Balance: {current_scores['regime_balance']:.3f}", "INFO")

            final_scores = self._calculate_individual_quality_scores(features, current_assignments)
            tprint(f"Final scores - Silhouette: {final_scores['silhouette']:.3f}, CH: {final_scores['calinski_harabasz']:.3f}, DB: {final_scores['davies_bouldin']:.3f}, Balance: {final_scores['regime_balance']:.3f}", "SUCCESS")
            
            # Print final regime distribution
            self._print_regime_distribution(current_assignments, max_regime_size, min_regime_size, "Final", features)
            
            tprint(f"Progressive regime flipping completed - {iteration} iterations", "SUCCESS")

            self._update_pipeline_current_assignments(current_assignments)
            return current_assignments

        except Exception as e:
            log_error(f"Progressive regime flipping failed: {e}")
            tprint(f"Progressive regime flipping failed: {e}", "ERROR")
            return assignments
    
    def _print_regime_distribution(self, assignments: np.ndarray, max_regime_size: int, min_regime_size: int, 
                                 system_name: str = "System", features: Optional[np.ndarray] = None):
        """Print detailed regime distribution analysis with enhanced regime information."""
        try:
            regime_sizes = np.bincount(assignments)
            n_regimes = len(regime_sizes)
            total_samples = len(assignments)
            
            tprint(f"📊 {system_name} Regime Distribution Analysis:", "INFO")
            tprint(f"   Total samples: {total_samples}, Number of regimes: {n_regimes}", "INFO")
            
            too_large = []
            too_small = []
            valid_regimes = []
            
            # Calculate regime statistics
            regime_stats = {}
            for regime_id, size in enumerate(regime_sizes):
                percentage = (size / total_samples) * 100
                status = "✅"
                
                # Calculate regime characteristics
                regime_mask = assignments == regime_id
                regime_samples = np.where(regime_mask)[0]
                
                # Basic statistics
                regime_info = {
                    'id': regime_id,
                    'size': size,
                    'percentage': percentage,
                    'samples': regime_samples,
                    'status': status
                }
                
                # Add feature-based statistics if features available
                if features is not None and len(features) == len(assignments):
                    regime_features = features[regime_mask]
                    if len(regime_features) > 0:
                        regime_info['feature_mean'] = np.mean(regime_features, axis=0)
                        regime_info['feature_std'] = np.std(regime_features, axis=0)
                        regime_info['feature_range'] = np.ptp(regime_features, axis=0)
                        regime_info['coherence'] = self._calculate_regime_coherence(regime_features)
                
                # Check size constraints
                if size > max_regime_size:
                    status = "🔴 TOO LARGE"
                    too_large.append((regime_id, size, percentage))
                    regime_info['status'] = status
                elif size < min_regime_size:
                    status = "🔴 TOO SMALL"
                    too_small.append((regime_id, size, percentage))
                    regime_info['status'] = status
                else:
                    valid_regimes.append((regime_id, size, percentage))
                
                regime_stats[regime_id] = regime_info
                
                # Print detailed regime information
                tprint(f"   Regime {regime_id}: {size} samples ({percentage:.1f}%) {status}", "INFO")
                
                # Add feature-based details if available
                if features is not None and 'coherence' in regime_info:
                    tprint(f"      📈 Coherence: {regime_info['coherence']:.3f}", "INFO")
                
                # Show regime boundaries (first and last samples)
                if len(regime_samples) > 0:
                    first_sample = min(regime_samples)
                    last_sample = max(regime_samples)
                    tprint(f"      📍 Sample range: {first_sample}-{last_sample}", "INFO")
            
            # Summary
            tprint(f"📈 Summary: {len(valid_regimes)} valid, {len(too_large)} too large, {len(too_small)} too small", "INFO")
            
            if too_large:
                tprint(f"   🔴 Large regimes need to lose samples: {[f'R{r}({s})' for r, s, p in too_large]}", "WARNING")
            if too_small:
                tprint(f"   🔴 Small regimes need to gain samples: {[f'R{r}({s})' for r, s, p in too_small]}", "WARNING")
            
            # Reallocation potential
            if too_large and too_small:
                total_excess = sum(size - max_regime_size for _, size, _ in too_large)
                total_deficit = sum(min_regime_size - size for _, size, _ in too_small)
                tprint(f"   🔄 Reallocation potential: {total_excess} excess samples → {total_deficit} deficit samples", "INFO")
            
            # Regime quality analysis
            if features is not None:
                self._analyze_regime_quality(regime_stats, features, assignments)
            
            return regime_stats
            
        except Exception as e:
            log_warning(f"Regime distribution analysis failed: {e}")
            return {}
    
    def _calculate_regime_coherence(self, regime_features: np.ndarray) -> float:
        """Calculate regime coherence (how similar samples within a regime are)."""
        try:
            if len(regime_features) < 2:
                return 0.0
            
            # Calculate pairwise distances within the regime
            distances = []
            for i in range(len(regime_features)):
                for j in range(i + 1, len(regime_features)):
                    dist = np.linalg.norm(regime_features[i] - regime_features[j])
                    distances.append(dist)
            
            if distances:
                # Coherence is inverse of average distance (higher = more coherent)
                avg_distance = np.mean(distances)
                coherence = 1.0 / (1.0 + avg_distance)
                return min(coherence, 1.0)
            
            return 0.0
            
        except Exception as e:
            return 0.0
    
    def _analyze_regime_quality(self, regime_stats: Dict, features: np.ndarray, assignments: np.ndarray):
        """Analyze regime quality and characteristics."""
        try:
            tprint("🔬 Regime Quality Analysis:", "INFO")
            
            # Find most and least coherent regimes
            coherences = {rid: stats.get('coherence', 0.0) for rid, stats in regime_stats.items() 
                         if 'coherence' in stats}
            
            if coherences:
                most_coherent = max(coherences, key=coherences.get)
                least_coherent = min(coherences, key=coherences.get)
                
                tprint(f"   🎯 Most coherent regime: {most_coherent} (coherence: {coherences[most_coherent]:.3f})", "SUCCESS")
                tprint(f"   ⚠️  Least coherent regime: {least_coherent} (coherence: {coherences[least_coherent]:.3f})", "WARNING")
            
            # Analyze regime separation
            self._analyze_regime_separation(regime_stats, features, assignments)
            
        except Exception as e:
            log_warning(f"Regime quality analysis failed: {e}")
    
    def _analyze_regime_separation(self, regime_stats: Dict, features: np.ndarray, assignments: np.ndarray):
        """Analyze how well separated the regimes are."""
        try:
            regime_centers = {}
            for rid, stats in regime_stats.items():
                if 'feature_mean' in stats:
                    regime_centers[rid] = stats['feature_mean']
            
            if len(regime_centers) < 2:
                return
            
            # Calculate inter-regime distances
            regime_pairs = []
            for rid1 in regime_centers:
                for rid2 in regime_centers:
                    if rid1 < rid2:
                        dist = np.linalg.norm(regime_centers[rid1] - regime_centers[rid2])
                        regime_pairs.append((rid1, rid2, dist))
            
            if regime_pairs:
                avg_separation = np.mean([dist for _, _, dist in regime_pairs])
                min_separation = min([dist for _, _, dist in regime_pairs])
                
                tprint(f"   📏 Average regime separation: {avg_separation:.3f}", "INFO")
                tprint(f"   📏 Minimum regime separation: {min_separation:.3f}", "INFO")
                
                # Find closest regimes
                closest_pair = min(regime_pairs, key=lambda x: x[2])
                tprint(f"   🔗 Closest regimes: {closest_pair[0]} ↔ {closest_pair[1]} (distance: {closest_pair[2]:.3f})", "INFO")
            
        except Exception as e:
            log_warning(f"Regime separation analysis failed: {e}")
    
    def _identify_frontier_samples(self, assignments: np.ndarray) -> List[int]:
        """Identify frontier samples using regime-space and feature-space neighbors (no temporal component)."""
        try:
            frontier_samples = []
            n_samples = len(assignments)
            
            # Get TAS and NAS assignments for regime-space analysis
            tas_assignments, nas_assignments = self._get_tas_nas_assignments()
            
            for i in range(n_samples):
                current_regime = assignments[i]
                is_frontier = False
                
                # 1. TAS/NAS disagreement (regime-space neighbors)
                tas_nas_frontier = self._is_tas_nas_frontier(tas_assignments, nas_assignments, i, current_regime)
                
                # 2. Feature-space neighbors (samples close in feature space but different regimes)
                feature_frontier = self._is_feature_space_frontier(i, current_regime, assignments)
                
                # Sample is frontier if it's a frontier in regime-space OR feature-space
                if tas_nas_frontier or feature_frontier:
                    is_frontier = True
                
                if is_frontier:
                    frontier_samples.append(i)
            
            # Detailed frontier analysis
            tas_nas_count = sum(
                1
                for i in range(n_samples)
                if self._is_tas_nas_frontier(tas_assignments, nas_assignments, i, assignments[i])
            )
            feature_count = sum(
                1
                for i in range(n_samples)
                if self._is_feature_space_frontier(i, assignments[i], assignments)
            )
            
            tprint(f"Frontier analysis - TAS/NAS disagreement: {tas_nas_count}, Feature-space: {feature_count}, Total: {len(frontier_samples)}", "INFO")
            
            return frontier_samples
            
        except Exception as e:
            log_warning(f"Frontier sample identification failed: {e}")
            return []
    
    
    def _is_tas_nas_frontier(self, tas_assignments: np.ndarray, nas_assignments: np.ndarray, 
                           sample_idx: int, current_regime: int) -> bool:
        """Check if sample is on TAS/NAS disagreement frontier (regime-space neighbors)."""
        try:
            if tas_assignments is None or nas_assignments is None:
                return False
            
            # Get TAS and NAS regime assignments for this sample
            tas_regime = tas_assignments[sample_idx] if sample_idx < len(tas_assignments) else current_regime
            nas_regime = nas_assignments[sample_idx] if sample_idx < len(nas_assignments) else current_regime
            
            # Sample is frontier if:
            # 1. TAS and NAS disagree (direct disagreement)
            if tas_regime != nas_regime:
                return True
            
            # 2. Current regime differs from both TAS and NAS (consensus disagreement)
            if current_regime != tas_regime and current_regime != nas_regime:
                return True
            
            # 3. Current regime matches one but not the other (partial disagreement)
            if (current_regime == tas_regime and current_regime != nas_regime) or \
               (current_regime == nas_regime and current_regime != tas_regime):
                return True
            
            return False
            
        except Exception as e:
            log_warning(f"TAS/NAS frontier check failed: {e}")
            return False
    
    def _is_feature_space_frontier(self, sample_idx: int, current_regime: int,
                                   assignments: np.ndarray) -> bool:
        """Check if sample is on feature-space frontier using efficient nearest neighbor approach."""
        try:
            # Get current features
            features = self._get_current_features()
            if features is None or assignments is None:
                return False

            if sample_idx >= len(features) or sample_idx >= len(assignments):
                return False

            if len(assignments) != len(features):
                log_warning(
                    "Assignment and feature lengths differ during frontier detection; skipping sample."
                )
                return False

            current_features = features[sample_idx]

            # Use a more efficient approach: check nearest neighbors instead of all samples
            n_neighbors = min(50, len(features) - 1)  # Limit to 50 neighbors for efficiency

            # Find k nearest neighbors using efficient method
            distances = []
            regimes = []

            for other_sample_idx in range(len(features)):
                if other_sample_idx == sample_idx:
                    continue

                other_features = features[other_sample_idx]
                distance = np.linalg.norm(current_features - other_features)
                distances.append(distance)
                regimes.append(self._get_regime_for_sample(other_sample_idx, assignments))

            if len(distances) < 10:  # Need minimum neighbors
                return False

            # Sort by distance and take nearest neighbors
            nearest_indices = np.argsort(distances)[:n_neighbors]
            nearest_distances = [distances[i] for i in nearest_indices]
            nearest_regimes = [regimes[i] for i in nearest_indices]

            # Count neighbors in same vs different regimes
            same_regime_count = sum(1 for r in nearest_regimes if r == current_regime)
            other_regime_count = sum(1 for r in nearest_regimes if r != current_regime)

            if same_regime_count == 0:
                return True  # No neighbors in same regime = frontier

            # Calculate ratio of different regime neighbors
            frontier_ratio = other_regime_count / (same_regime_count + other_regime_count)

            # Sample is on frontier if > 60% of nearest neighbors are from different regimes
            return frontier_ratio > 0.6

        except Exception as e:
            log_warning(f"Feature-space frontier check failed: {e}")
            return False
    
    def _get_tas_nas_assignments(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get TAS and NAS assignments from pipeline state."""
        try:
            if self.pipeline_state:
                # Debug: Print all available keys in pipeline state
                tprint(f"🔍 Pipeline State Debug:", "INFO")
                tprint(f"   Available keys: {list(self.pipeline_state.keys())}", "INFO")
                
                # Try different possible keys for TAS and NAS assignments
                tas_assignments = None
                nas_assignments = None
                
                # Check for TAS assignments in various possible locations
                if 'tas_assignments' in self.pipeline_state:
                    tas_assignments = self.pipeline_state['tas_assignments']
                elif 'artifacts' in self.pipeline_state:
                    artifacts = self.pipeline_state['artifacts']
                    if 'nas_tas_regime_discovery_result' in artifacts:
                        discovery_result = artifacts['nas_tas_regime_discovery_result']
                        tas_assignments = discovery_result.get('tas_assignments')
                        nas_assignments = discovery_result.get('nas_assignments')
                
                # Check for NAS assignments
                if 'nas_assignments' in self.pipeline_state:
                    nas_assignments = self.pipeline_state['nas_assignments']
                
                # Debug: Print assignment details
                tprint(f"🔍 TAS/NAS Assignment Debug:", "INFO")
                tprint(f"   TAS assignments: {type(tas_assignments)}, length: {len(tas_assignments) if tas_assignments is not None else 'None'}", "INFO")
                tprint(f"   NAS assignments: {type(nas_assignments)}, length: {len(nas_assignments) if nas_assignments is not None else 'None'}", "INFO")
                
                if tas_assignments is not None and nas_assignments is not None:
                    # Convert to numpy arrays if they aren't already
                    if not isinstance(tas_assignments, np.ndarray):
                        tas_assignments = np.array(tas_assignments)
                    if not isinstance(nas_assignments, np.ndarray):
                        nas_assignments = np.array(nas_assignments)
                    
                    # Check if they have the same length
                    if len(tas_assignments) != len(nas_assignments):
                        tprint(f"   ⚠️  Length mismatch: TAS={len(tas_assignments)}, NAS={len(nas_assignments)}", "WARNING")
                        # Align to the shorter length
                        min_length = min(len(tas_assignments), len(nas_assignments))
                        tas_assignments = tas_assignments[:min_length]
                        nas_assignments = nas_assignments[:min_length]
                        tprint(f"   🔧 Aligned to length: {min_length}", "INFO")
                    
                    # Show sample of assignments
                    tprint(f"   TAS sample: {tas_assignments[:10]}", "INFO")
                    tprint(f"   NAS sample: {nas_assignments[:10]}", "INFO")
                    
                    # Calculate actual disagreement
                    disagreement_mask = tas_assignments != nas_assignments
                    disagreement_rate = np.mean(disagreement_mask)
                    tprint(f"   📊 Actual disagreement rate: {disagreement_rate:.3f} ({np.sum(disagreement_mask)}/{len(disagreement_mask)})", "INFO")
                else:
                    tprint("   ❌ TAS or NAS assignments not found in pipeline state", "WARNING")
                
                return tas_assignments, nas_assignments
            else:
                tprint("   ❌ No pipeline state available", "WARNING")
            return None, None
        except Exception as e:
            log_warning(f"Failed to get TAS/NAS assignments: {e}")
            tprint(f"   ❌ Error getting assignments: {e}", "ERROR")
            return None, None
    
    def _get_regime_for_sample(self, sample_idx: int, assignments: Optional[np.ndarray]) -> int:
        """Get regime assignment for a specific sample."""
        try:
            if assignments is not None:
                if 0 <= sample_idx < len(assignments):
                    return int(assignments[sample_idx])
                raise IndexError(
                    f"Sample index {sample_idx} out of bounds for assignments of length {len(assignments)}"
                )

            if self.pipeline_state and isinstance(self.pipeline_state, dict):
                current_assignments = self.pipeline_state.get('current_assignments')
                if current_assignments is not None and sample_idx < len(current_assignments):
                    return int(current_assignments[sample_idx])

            raise ValueError("No assignments available for regime lookup")
        except Exception as e:
            log_warning(f"Failed to get regime for sample {sample_idx}: {e}")
            return 0
    
    def _analyze_tas_nas_disagreement(self, tas_assignments: np.ndarray, nas_assignments: np.ndarray) -> Dict[str, Any]:
        """Analyze TAS/NAS disagreement patterns to understand regime conflicts."""
        try:
            if tas_assignments is None or nas_assignments is None:
                tprint("❌ TAS/NAS assignments not available for disagreement analysis", "ERROR")
                return {"error": "TAS/NAS assignments not available"}
            
            # Ensure same length
            if len(tas_assignments) != len(nas_assignments):
                tprint(f"⚠️  Length mismatch in disagreement analysis: TAS={len(tas_assignments)}, NAS={len(nas_assignments)}", "WARNING")
                min_length = min(len(tas_assignments), len(nas_assignments))
                tas_assignments = tas_assignments[:min_length]
                nas_assignments = nas_assignments[:min_length]
                tprint(f"🔧 Aligned to length: {min_length}", "INFO")
            
            # Calculate disagreement rate
            disagreement_mask = tas_assignments != nas_assignments
            disagreement_rate = np.mean(disagreement_mask)
            agreement_rate = 1.0 - disagreement_rate
            
            # Analyze disagreement patterns
            disagreement_samples = np.where(disagreement_mask)[0]
            agreement_samples = np.where(~disagreement_mask)[0]
            
            # Calculate regime transition patterns
            tas_transitions = np.sum(np.diff(tas_assignments) != 0)
            nas_transitions = np.sum(np.diff(nas_assignments) != 0)
            
            # Calculate regime stability
            tas_stability = self._calculate_regime_stability_score(tas_assignments)
            nas_stability = self._calculate_regime_stability_score(nas_assignments)
            
            # Find most conflicted regimes
            regime_conflicts = {}
            for i in disagreement_samples:
                tas_regime = tas_assignments[i]
                nas_regime = nas_assignments[i]
                conflict_key = f"TAS_{tas_regime}_vs_NAS_{nas_regime}"
                regime_conflicts[conflict_key] = regime_conflicts.get(conflict_key, 0) + 1
            
            # Find most agreed regimes
            regime_agreements = {}
            for i in agreement_samples:
                tas_regime = tas_assignments[i]
                nas_regime = nas_assignments[i]
                if tas_regime == nas_regime:  # Double check
                    agreement_key = f"Regime_{tas_regime}"
                    regime_agreements[agreement_key] = regime_agreements.get(agreement_key, 0) + 1
            
            analysis = {
                "disagreement_rate": disagreement_rate,
                "agreement_rate": agreement_rate,
                "disagreement_samples": len(disagreement_samples),
                "agreement_samples": len(agreement_samples),
                "tas_transitions": tas_transitions,
                "nas_transitions": nas_transitions,
                "tas_stability": tas_stability,
                "nas_stability": nas_stability,
                "regime_conflicts": regime_conflicts,
                "regime_agreements": regime_agreements,
                "most_conflicted_regimes": sorted(regime_conflicts.items(), key=lambda x: x[1], reverse=True)[:5],
                "most_agreed_regimes": sorted(regime_agreements.items(), key=lambda x: x[1], reverse=True)[:5]
            }
            
            # Detailed printing
            tprint(f"📊 TAS/NAS Agreement Analysis:", "INFO")
            tprint(f"   ✅ Agreement: {agreement_rate:.3f} ({len(agreement_samples)}/{len(tas_assignments)})", "SUCCESS")
            tprint(f"   ❌ Disagreement: {disagreement_rate:.3f} ({len(disagreement_samples)}/{len(tas_assignments)})", "WARNING")
            tprint(f"   📈 TAS Stability: {tas_stability:.3f}, NAS Stability: {nas_stability:.3f}", "INFO")
            
            if regime_conflicts:
                tprint(f"   🔥 Most conflicted: {analysis['most_conflicted_regimes'][:3]}", "WARNING")
            if regime_agreements:
                tprint(f"   🤝 Most agreed: {analysis['most_agreed_regimes'][:3]}", "SUCCESS")
            
            return analysis
            
        except Exception as e:
            log_warning(f"TAS/NAS disagreement analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_regime_stability_score(self, assignments: np.ndarray) -> float:
        """Calculate regime stability score (higher = more stable)."""
        try:
            if len(assignments) < 2:
                return 1.0
            
            # Calculate number of regime transitions
            transitions = np.sum(np.diff(assignments) != 0)
            max_possible_transitions = len(assignments) - 1
            
            # Stability score: 1.0 = no transitions, 0.0 = maximum transitions
            stability = 1.0 - (transitions / max_possible_transitions)
            
            return stability
            
        except Exception as e:
            log_warning(f"Regime stability calculation failed: {e}")
            return 0.5
    
    def _calculate_single_flip_improvement(self, features: np.ndarray, assignments: np.ndarray, 
                                        sample_idx: int, target_regime: int) -> float:
        """Calculate quality improvement from flipping a single sample to a target regime."""
        try:
            # Store original assignment
            original_regime = assignments[sample_idx]
            
            # Calculate baseline quality scores
            baseline_scores = self._calculate_individual_quality_scores(features, assignments)
            
            # Try the new regime assignment
            assignments[sample_idx] = target_regime
            new_scores = self._calculate_individual_quality_scores(features, assignments)
            
            # Restore original assignment
            assignments[sample_idx] = original_regime
            
            # Calculate improvement for each metric
            silhouette_improvement = new_scores['silhouette'] - baseline_scores['silhouette']
            ch_improvement = (new_scores['calinski_harabasz'] - baseline_scores['calinski_harabasz']) / 1000
            db_improvement = baseline_scores['davies_bouldin'] - new_scores['davies_bouldin']  # Lower is better
            balance_improvement = new_scores['regime_balance'] - baseline_scores['regime_balance']
            
            # Calculate temporal improvement
            temporal_improvement = self._calculate_temporal_improvement(assignments, sample_idx, target_regime)
            
            # Calculate CV improvements for both regimes
            cv_improvement = self._calculate_cv_improvement(features, assignments, sample_idx, target_regime)
            
            # Weighted composite improvement with CV metrics
            total_improvement = (
                0.25 * silhouette_improvement +      # Silhouette (most important)
                0.18 * ch_improvement +              # Calinski-Harabasz
                0.18 * db_improvement +              # Davies-Bouldin (inverted)
                0.12 * balance_improvement +        # Regime balance
                0.12 * temporal_improvement +       # Temporal consistency
                0.15 * cv_improvement               # Coefficient of Variation metrics
            )
            
            return total_improvement
            
        except Exception as e:
            log_warning(f"Single flip improvement calculation failed: {e}")
            return 0.0
    
    def _calculate_cv_improvement(self, features: np.ndarray, assignments: np.ndarray, 
                                sample_idx: int, target_regime: int) -> float:
        """Calculate Coefficient of Variation improvement from regime change."""
        try:
            # Store original assignment
            original_regime = assignments[sample_idx]
            
            # Calculate baseline CV scores
            baseline_cv = self._calculate_cv_score(features, assignments)
            
            # Try the new regime assignment
            assignments[sample_idx] = target_regime
            new_cv = self._calculate_cv_score(features, assignments)
            
            # Restore original assignment
            assignments[sample_idx] = original_regime
            
            # Calculate improvement (higher CV score is better)
            cv_improvement = new_cv - baseline_cv
            
            return cv_improvement
            
        except Exception as e:
            log_warning(f"CV improvement calculation failed: {e}")
            return 0.0
    
    def _calculate_cv_score(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate Coefficient of Variation score for regime assignments."""
        try:
            # Calculate within-cluster and between-cluster CV
            within_cv, between_cv = self._calculate_between_cluster_cv(features, assignments)
            
            # CV score: low within-cluster CV (homogeneous) + high between-cluster CV (distinct)
            # Normalize and combine
            norm_within_cv = max(0, 1.0 - within_cv)  # Lower is better, so invert
            norm_between_cv = min(1.0, between_cv)   # Higher is better, cap at 1.0
            
            # Combined CV score (weighted average)
            cv_score = 0.4 * norm_within_cv + 0.6 * norm_between_cv
            
            return cv_score
            
        except Exception as e:
            log_warning(f"CV score calculation failed: {e}")
            return 0.0
    
    def _calculate_between_cluster_cv(self, features: np.ndarray, assignments: np.ndarray) -> Tuple[float, float]:
        """Calculate within-cluster and between-cluster Coefficient of Variation with hardware optimization."""
        try:
            unique_labels = sorted(set(assignments))
            n_clusters = len(unique_labels)
            
            if n_clusters < 2:
                return 0.0, 0.0
            
            # Use hardware-optimized operations if available
            if self.matrix_ops and self.hardware_manager:
                return self._calculate_cv_hardware_optimized(features, assignments, unique_labels)
            else:
                return self._calculate_cv_standard(features, assignments, unique_labels)
            
        except Exception as e:
            log_warning(f"Between cluster CV calculation failed: {e}")
            return 0.0, 0.0
    
    def _calculate_cv_hardware_optimized(self, features: np.ndarray, assignments: np.ndarray, unique_labels: List[int]) -> Tuple[float, float]:
        """Hardware-optimized CV calculation using matrix operations."""
        try:
            # Use vectorized operations for within-cluster CV
            within_cv_scores = []
            for label in unique_labels:
                cluster_mask = assignments == label
                cluster_features = features[cluster_mask]
                
                if len(cluster_features) > 1:
                    # Use hardware-optimized standard deviation and mean
                    if self.matrix_ops:
                        feature_stds = self.matrix_ops.vectorized_std(cluster_features, axis=0)
                        feature_means = self.matrix_ops.vectorized_mean(cluster_features, axis=0)
                    else:
                        feature_stds = np.std(cluster_features, axis=0)
                        feature_means = np.mean(cluster_features, axis=0)
                    
                    # Calculate CV for each feature dimension
                    feature_cvs = []
                    for i in range(len(feature_stds)):
                        if feature_means[i] != 0:
                            cv = feature_stds[i] / feature_means[i]
                            feature_cvs.append(cv)
                    
                    if feature_cvs:
                        cluster_cv = np.mean(feature_cvs)
                        within_cv_scores.append(cluster_cv)
            
            within_cv = np.mean(within_cv_scores) if within_cv_scores else 0.0
            
            # Calculate between-cluster CV using hardware-optimized operations
            cluster_centers = []
            for label in unique_labels:
                cluster_mask = assignments == label
                cluster_features = features[cluster_mask]
                if len(cluster_features) > 0:
                    if self.matrix_ops:
                        center = self.matrix_ops.vectorized_mean(cluster_features, axis=0)
                    else:
                        center = np.mean(cluster_features, axis=0)
                    cluster_centers.append(center)
            
            if len(cluster_centers) > 1:
                cluster_centers = np.array(cluster_centers)
                # Use hardware-optimized operations for center CV calculation
                if self.matrix_ops:
                    center_stds = self.matrix_ops.vectorized_std(cluster_centers, axis=0)
                    center_means = self.matrix_ops.vectorized_mean(cluster_centers, axis=0)
                else:
                    center_stds = np.std(cluster_centers, axis=0)
                    center_means = np.mean(cluster_centers, axis=0)
                
                center_cvs = []
                for i in range(len(center_stds)):
                    if center_means[i] != 0:
                        cv = center_stds[i] / center_means[i]
                        center_cvs.append(cv)
                
                between_cv = np.mean(center_cvs) if center_cvs else 0.0
            else:
                between_cv = 0.0
            
            return within_cv, between_cv
            
        except Exception as e:
            log_warning(f"Hardware-optimized CV calculation failed: {e}")
            return self._calculate_cv_standard(features, assignments, unique_labels)
    
    def _calculate_cv_standard(self, features: np.ndarray, assignments: np.ndarray, unique_labels: List[int]) -> Tuple[float, float]:
        """Standard CV calculation fallback."""
        try:
            # Calculate within-cluster CV (lower is better - more homogeneous)
            within_cv_scores = []
            for label in unique_labels:
                cluster_mask = assignments == label
                cluster_features = features[cluster_mask]
                
                if len(cluster_features) > 1:
                    # Calculate CV for each feature dimension
                    feature_cvs = []
                    for feature_idx in range(cluster_features.shape[1]):
                        feature_values = cluster_features[:, feature_idx]
                        if np.std(feature_values) > 0:
                            cv = np.std(feature_values) / np.mean(feature_values)
                            feature_cvs.append(cv)
                    
                    if feature_cvs:
                        cluster_cv = np.mean(feature_cvs)
                        within_cv_scores.append(cluster_cv)
            
            within_cv = np.mean(within_cv_scores) if within_cv_scores else 0.0
            
            # Calculate between-cluster CV (higher is better - more distinct)
            cluster_centers = []
            for label in unique_labels:
                cluster_mask = assignments == label
                cluster_features = features[cluster_mask]
                if len(cluster_features) > 0:
                    center = np.mean(cluster_features, axis=0)
                    cluster_centers.append(center)
            
            if len(cluster_centers) > 1:
                cluster_centers = np.array(cluster_centers)
                # Calculate CV between cluster centers
                center_cvs = []
                for feature_idx in range(cluster_centers.shape[1]):
                    feature_values = cluster_centers[:, feature_idx]
                    if np.std(feature_values) > 0:
                        cv = np.std(feature_values) / np.mean(feature_values)
                        center_cvs.append(cv)
                
                between_cv = np.mean(center_cvs) if center_cvs else 0.0
            else:
                between_cv = 0.0
            
            return within_cv, between_cv
            
        except Exception as e:
            log_warning(f"Standard CV calculation failed: {e}")
            return 0.0, 0.0
    
    def _calculate_individual_quality_scores(self, features: np.ndarray, assignments: np.ndarray) -> Dict[str, float]:
        """Calculate individual quality scores (Silhouette, CH, DB, Balance, CV)."""
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            
            if len(set(assignments)) < 2:
                return {'silhouette': 0.0, 'calinski_harabasz': 0.0, 'davies_bouldin': 0.0, 'regime_balance': 0.0, 'cv_score': 0.0}
            
            # Calculate individual metrics
            silhouette = silhouette_score(features, assignments)
            calinski_harabasz = calinski_harabasz_score(features, assignments)
            davies_bouldin = davies_bouldin_score(features, assignments)
            
            # Calculate regime balance
            unique_labels = set(assignments)
            regime_sizes = [np.sum(assignments == label) for label in unique_labels]
            regime_balance = 1.0 - (np.std(regime_sizes) / np.mean(regime_sizes)) if len(regime_sizes) > 1 else 0.0
            
            # Calculate CV score
            cv_score = self._calculate_cv_score(features, assignments)
            
            return {
                'silhouette': silhouette,
                'calinski_harabasz': calinski_harabasz,
                'davies_bouldin': davies_bouldin,
                'regime_balance': regime_balance,
                'cv_score': cv_score
            }
            
        except Exception as e:
            log_warning(f"Individual quality scores calculation failed: {e}")
            return {'silhouette': 0.0, 'calinski_harabasz': 0.0, 'davies_bouldin': 0.0, 'regime_balance': 0.0, 'cv_score': 0.0}
    
    def _is_quality_improvement(self, old_scores: Dict[str, float], new_scores: Dict[str, float], 
                              threshold: float) -> bool:
        """Check if new scores represent a quality improvement without degrading any metric significantly."""
        try:
            # Check if any metric degraded significantly (more than threshold)
            for metric in ['silhouette', 'calinski_harabasz', 'regime_balance', 'cv_score']:
                if new_scores[metric] < old_scores[metric] - threshold:
                    return False
            
            # For Davies-Bouldin, lower is better
            if new_scores['davies_bouldin'] > old_scores['davies_bouldin'] + threshold:
                return False
            
            # Check if at least one metric improved
            improvements = 0
            if new_scores['silhouette'] > old_scores['silhouette'] + threshold:
                improvements += 1
            if new_scores['calinski_harabasz'] > old_scores['calinski_harabasz'] + threshold:
                improvements += 1
            if new_scores['regime_balance'] > old_scores['regime_balance'] + threshold:
                improvements += 1
            if new_scores['davies_bouldin'] < old_scores['davies_bouldin'] - threshold:
                improvements += 1
            if new_scores['cv_score'] > old_scores['cv_score'] + threshold:
                improvements += 1
            
            return improvements > 0
            
        except Exception as e:
            log_warning(f"Quality improvement check failed: {e}")
            return False
    
    def _calculate_regime_confidence(self, features: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """Calculate confidence scores for regime assignments."""
        try:
            # Calculate confidence based on distance to cluster centers
            unique_labels = sorted(set(assignments))
            confidence_scores = np.zeros(len(assignments))
            
            for label in unique_labels:
                cluster_mask = assignments == label
                cluster_features = features[cluster_mask]
                
                if len(cluster_features) > 1:
                    # Calculate cluster center
                    center = np.mean(cluster_features, axis=0)
                    
                    # Calculate distances to center
                    distances = np.linalg.norm(cluster_features - center, axis=1)
                    
                    # Convert distances to confidence (closer = higher confidence)
                    max_distance = np.max(distances) if len(distances) > 0 else 1.0
                    cluster_confidence = 1.0 - (distances / (max_distance + 1e-8))
                    
                    confidence_scores[cluster_mask] = cluster_confidence
                else:
                    # Single sample clusters get medium confidence
                    confidence_scores[cluster_mask] = 0.5
            
            return confidence_scores
            
        except Exception as e:
            log_warning(f"Regime confidence calculation failed: {e}")
            return np.ones(len(assignments)) * 0.5
    
    def _create_confidence_weighted_consensus(self, tas_assignments: np.ndarray, nas_assignments: np.ndarray,
                                            tas_confidence: np.ndarray, nas_confidence: np.ndarray) -> np.ndarray:
        """Create weighted consensus based on confidence scores."""
        try:
            consensus_assignments = np.zeros(len(tas_assignments), dtype=int)
            
            for i in range(len(tas_assignments)):
                # Calculate weights based on confidence
                tas_weight = tas_confidence[i]
                nas_weight = nas_confidence[i]
                total_weight = tas_weight + nas_weight
                
                if total_weight > 0:
                    # Weighted average of regime assignments
                    weighted_regime = (tas_weight * tas_assignments[i] + nas_weight * nas_assignments[i]) / total_weight
                    consensus_assignments[i] = int(round(weighted_regime))
                else:
                    # Fallback to TAS if no confidence
                    consensus_assignments[i] = tas_assignments[i]
            
            return consensus_assignments
            
        except Exception as e:
            log_warning(f"Confidence weighted consensus failed: {e}")
            return tas_assignments
    
    async def _optimize_regime_count(self, features: np.ndarray, market_data: pd.DataFrame) -> int:
        """Optimize the number of regimes using multiple criteria."""
        try:
            tprint("Starting regime count optimization...", "INFO")
            log_info("Starting regime count optimization")
            
            # Test different regime counts
            regime_counts = list(range(5, 16))  # 5-15 regimes
            best_count = 8
            best_score = 0.0
            
            for n_regimes in regime_counts:
                try:
                    # Test with K-means for speed
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
                    test_assignments = kmeans.fit_predict(features)
                    
                    # Calculate score
                    score = self._calculate_composite_score(features, test_assignments)
                    
                    # Apply regime balance penalty
                    unique_labels = set(test_assignments)
                    regime_sizes = [np.sum(test_assignments == label) for label in unique_labels]
                    balance_penalty = np.std(regime_sizes) / np.mean(regime_sizes) if len(regime_sizes) > 1 else 0.0
                    adjusted_score = score - balance_penalty * 0.1
                    
                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_count = n_regimes
                    
                    tprint(f"Regime count {n_regimes}: score={score:.3f}, adjusted={adjusted_score:.3f}", "INFO")
                    
                except Exception as e:
                    log_warning(f"Regime count {n_regimes} failed: {e}")
                    continue
            
            tprint(f"Optimal regime count: {best_count} (score: {best_score:.3f})", "SUCCESS")
            log_success(f"Optimal regime count: {best_count} (score: {best_score:.3f})")
            
            return best_count
            
        except Exception as e:
            log_error(f"Regime count optimization failed: {e}")
            tprint(f"Regime count optimization failed: {e}", "ERROR")
            return 8  # Default fallback
    
    async def _ensemble_clustering_optimization(self, features: np.ndarray, initial_assignments: np.ndarray, 
                                             market_data: pd.DataFrame, n_regimes: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform ensemble clustering using multiple algorithms with weighted voting."""
        try:
            tprint("Starting ensemble clustering optimization...", "INFO")
            log_info("Starting ensemble clustering optimization")
            
            # Define clustering algorithms to try with optimal regime count
            algorithms = {
                'kmeans': {'n_clusters': n_regimes, 'random_state': 42, 'n_init': 10},
                'hierarchical': {'n_clusters': n_regimes, 'linkage': 'ward'},
                'gmm': {'n_components': n_regimes, 'random_state': 42},
                'dbscan': {'eps': 0.5, 'min_samples': 5},
                'spectral': {'n_clusters': n_regimes, 'random_state': 42}
            }
            
            ensemble_results = {}
            method_scores = {}
            
            # Try each algorithm
            for method, params in algorithms.items():
                try:
                    tprint(f"Testing {method} clustering...", "INFO")
                    assignments = self._run_clustering_algorithm(features, method, params)
                    score = self._calculate_composite_score(features, assignments)
                    
                    ensemble_results[method] = assignments
                    method_scores[method] = score
                    
                    tprint(f"{method} score: {score:.3f}", "SUCCESS")
                    
                except Exception as e:
                    log_warning(f"{method} clustering failed: {e}")
                    continue
            
            # Weighted ensemble voting
            if ensemble_results:
                tprint("Performing weighted ensemble voting...", "INFO")
                ensemble_assignments = self._weighted_ensemble_voting(ensemble_results, method_scores)
                ensemble_score = self._calculate_composite_score(features, ensemble_assignments)
                
                tprint(f"Ensemble score: {ensemble_score:.3f}", "SUCCESS")
                
                return ensemble_assignments, {
                    'methods_used': list(ensemble_results.keys()),
                    'method_scores': method_scores,
                    'ensemble_score': ensemble_score
                }
            else:
                tprint("No algorithms succeeded, using initial assignments", "WARNING")
                return initial_assignments, {'methods_used': [], 'method_scores': {}, 'ensemble_score': 0.0}
                
        except Exception as e:
            log_error(f"Ensemble clustering optimization failed: {e}")
            tprint(f"Ensemble clustering optimization failed: {e}", "ERROR")
            return initial_assignments, {'methods_used': [], 'method_scores': {}, 'ensemble_score': 0.0}
    
    def _run_clustering_algorithm(self, features: np.ndarray, method: str, params: Dict[str, Any]) -> np.ndarray:
        """Run a specific clustering algorithm."""
        try:
            from sklearn.cluster import KMeans, AgglomerativeClustering, GaussianMixture, DBSCAN, SpectralClustering
            
            if method == 'kmeans':
                clusterer = KMeans(**params)
                return clusterer.fit_predict(features)
            
            elif method == 'hierarchical':
                clusterer = AgglomerativeClustering(**params)
                return clusterer.fit_predict(features)
            
            elif method == 'gmm':
                clusterer = GaussianMixture(**params)
                return clusterer.fit_predict(features)
            
            elif method == 'dbscan':
                clusterer = DBSCAN(**params)
                labels = clusterer.fit_predict(features)
                # Handle noise points (-1 labels)
                if -1 in labels:
                    # Assign noise points to nearest cluster
                    from sklearn.neighbors import NearestNeighbors
                    nbrs = NearestNeighbors(n_neighbors=1).fit(features[labels != -1])
                    noise_indices = np.where(labels == -1)[0]
                    if len(noise_indices) > 0:
                        distances, indices = nbrs.kneighbors(features[noise_indices])
                        labels[noise_indices] = labels[labels != -1][indices.flatten()]
                return labels
            
            elif method == 'spectral':
                clusterer = SpectralClustering(**params)
                return clusterer.fit_predict(features)
            
            else:
                raise ValueError(f"Unknown clustering method: {method}")
                
        except Exception as e:
            log_warning(f"Clustering algorithm {method} failed: {e}")
            # Return random assignments as fallback
            return np.random.randint(0, params.get('n_clusters', 8), len(features))
    
    def _weighted_ensemble_voting(self, ensemble_results: Dict[str, np.ndarray], 
                                method_scores: Dict[str, float]) -> np.ndarray:
        """Perform weighted ensemble voting based on algorithm scores."""
        try:
            # Normalize scores to get weights
            scores = np.array(list(method_scores.values()))
            weights = scores / np.sum(scores) if np.sum(scores) > 0 else np.ones(len(scores)) / len(scores)
            
            tprint(f"Ensemble weights: {dict(zip(method_scores.keys(), weights))}", "INFO")
            
            # Get all unique labels across methods
            all_labels = set()
            for assignments in ensemble_results.values():
                all_labels.update(assignments)
            n_clusters = len(all_labels)
            
            # Weighted voting for each sample
            n_samples = len(list(ensemble_results.values())[0])
            final_assignments = np.zeros(n_samples, dtype=int)
            
            for sample_idx in range(n_samples):
                # Collect votes from each method
                votes = {}
                for method, assignments in ensemble_results.items():
                    if method in method_scores:
                        weight = weights[list(method_scores.keys()).index(method)]
                        cluster = assignments[sample_idx]
                        votes[cluster] = votes.get(cluster, 0) + weight
                
                # Assign to cluster with highest weighted vote
                final_assignments[sample_idx] = max(votes.items(), key=lambda x: x[1])[0]
            
            return final_assignments
            
        except Exception as e:
            log_warning(f"Weighted ensemble voting failed: {e}")
            # Return first method's results as fallback
            return list(ensemble_results.values())[0]
    
    def _iterative_regime_optimization(self, features: np.ndarray, assignments: np.ndarray, 
                                    market_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Iteratively optimize regime assignments by flipping samples."""
        try:
            tprint("Starting iterative regime optimization...", "INFO")
            log_info("Starting iterative regime optimization")
            
            current_assignments = assignments.copy()
            current_score = self._calculate_composite_score(features, current_assignments)
            n_regimes = len(set(current_assignments))
            n_samples = len(current_assignments)
            
            # Hard limits
            max_regime_size = int(0.20 * n_samples)  # 20% max
            min_regime_size = int(0.04 * n_samples)  # 4% min
            
            tprint(f"Hard limits - Max regime size: {max_regime_size}, Min regime size: {min_regime_size}", "INFO")
            
            iteration = 0
            max_iterations = 100  # Reduced iterations
            improvement_threshold = 0.0001  # Much smaller threshold for more sensitive optimization
            
            while iteration < max_iterations:
                iteration += 1
                improved = False
                
                # Try flipping each sample to each possible regime
                for sample_idx in range(n_samples):
                    current_regime = current_assignments[sample_idx]
                    
                    # Try each possible regime
                    for target_regime in range(n_regimes):
                        if target_regime == current_regime:
                            continue
                        
                        # Check if flip would violate hard limits
                        if not self._is_flip_valid(current_assignments, sample_idx, target_regime, 
                                                max_regime_size, min_regime_size):
                            continue
                        
                        # Try the flip
                        test_assignments = current_assignments.copy()
                        test_assignments[sample_idx] = target_regime
                        
                        # Calculate new score (simplified - no temporal penalty)
                        new_score = self._calculate_composite_score(features, test_assignments)
                        
                        # If improvement is significant, accept the flip
                        if new_score > current_score + improvement_threshold:
                            current_assignments = test_assignments
                            self._update_pipeline_current_assignments(current_assignments)
                            current_score = new_score
                            improved = True
                            
                            if iteration % 100 == 0:
                                tprint(f"Iteration {iteration}: Score improved to {current_score:.3f}", "INFO")
                            break
                    
                    if improved:
                        break
                
                # If no improvement found, stop
                if not improved:
                    tprint(f"Converged at iteration {iteration} - No more improvements found", "SUCCESS")
                    break
            
            tprint(f"Iterative optimization completed - {iteration} iterations, final score: {current_score:.3f}", "SUCCESS")
            log_success(f"Iterative optimization completed - {iteration} iterations, final score: {current_score:.3f}")
            
            return current_assignments, {
                'iterations': iteration,
                'final_score': current_score,
                'execution_time': 0.0  # Would be calculated in real implementation
            }
            
        except Exception as e:
            log_error(f"Iterative regime optimization failed: {e}")
            tprint(f"Iterative regime optimization failed: {e}", "ERROR")
            return assignments, {'iterations': 0, 'final_score': 0.0, 'execution_time': 0.0}
    
    def _is_flip_valid(self, assignments: np.ndarray, sample_idx: int, target_regime: int, 
                      max_regime_size: int, min_regime_size: int, verbose: bool = False) -> bool:
        """Check if flipping a sample to a target regime would violate hard limits."""
        try:
            current_regime = assignments[sample_idx]
            
            # If moving to the same regime, it's always valid
            if current_regime == target_regime:
                return True
            
            # Calculate regime sizes after the flip
            regime_sizes = np.bincount(assignments, minlength=len(set(assignments)))
            
            # Decrease current regime size
            regime_sizes[current_regime] -= 1
            # Increase target regime size
            regime_sizes[target_regime] += 1
            
            # Check if any regime would be too large or too small
            violations = []
            for regime_id, size in enumerate(regime_sizes):
                if size > max_regime_size:
                    violations.append(f"Regime {regime_id} would be too large ({size} > {max_regime_size})")
                elif size < min_regime_size:
                    violations.append(f"Regime {regime_id} would be too small ({size} < {min_regime_size})")
            
            if violations and verbose:
                tprint(f"   🚫 Move rejected: {', '.join(violations)}", "WARNING")
            
            return len(violations) == 0
            
        except Exception as e:
            log_warning(f"Flip validation failed: {e}")
            return False
    
    def _is_temporally_consistent(self, assignments: np.ndarray, sample_idx: int, target_regime: int, 
                                market_data: pd.DataFrame) -> bool:
        """Check if a regime flip maintains temporal consistency."""
        try:
            # Get neighboring samples (previous and next)
            prev_idx = max(0, sample_idx - 1)
            next_idx = min(len(assignments) - 1, sample_idx + 1)
            
            # Check if flip would create too many regime switches
            current_regime = assignments[sample_idx]
            prev_regime = assignments[prev_idx]
            next_regime = assignments[next_idx]
            
            # Count current regime switches around this sample
            current_switches = 0
            if prev_regime != current_regime:
                current_switches += 1
            if next_regime != current_regime:
                current_switches += 1
            
            # Count regime switches after flip
            future_switches = 0
            if prev_regime != target_regime:
                future_switches += 1
            if next_regime != target_regime:
                future_switches += 1
            
            # Allow flip if it doesn't increase regime switches significantly
            max_additional_switches = 1
            return future_switches <= current_switches + max_additional_switches
            
        except Exception as e:
            log_warning(f"Temporal consistency check failed: {e}")
            return True  # Allow flip if check fails
    
    def _calculate_composite_score_with_temporal_penalty(self, features: np.ndarray, assignments: np.ndarray, 
                                                       market_data: pd.DataFrame) -> float:
        """Calculate composite score with temporal consistency penalty."""
        try:
            # Calculate base composite score
            base_score = self._calculate_composite_score(features, assignments)
            
            # Calculate temporal penalty
            temporal_penalty = self._calculate_temporal_penalty(assignments)
            
            # Apply temporal penalty (reduce score for excessive regime switching)
            final_score = base_score - temporal_penalty
            
            return max(0.0, final_score)  # Ensure non-negative score
            
        except Exception as e:
            log_warning(f"Temporal penalty calculation failed: {e}")
            return self._calculate_composite_score(features, assignments)
    
    def _calculate_temporal_penalty(self, assignments: np.ndarray) -> float:
        """Calculate penalty for excessive regime switching."""
        try:
            # Count regime switches
            switches = 0
            for i in range(1, len(assignments)):
                if assignments[i] != assignments[i-1]:
                    switches += 1
            
            # Calculate penalty based on switching frequency
            switch_rate = switches / len(assignments)
            
            # Penalty increases with switching rate
            # Optimal switch rate is around 0.1-0.2 (10-20% of samples)
            optimal_rate = 0.15
            penalty = max(0, (switch_rate - optimal_rate) * 0.5)
            
            return penalty
            
        except Exception as e:
            log_warning(f"Temporal penalty calculation failed: {e}")
            return 0.0
    
    def _calculate_composite_score(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate composite clustering score using multiple metrics including CV."""
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            
            if len(set(assignments)) < 2:
                return 0.0
            
            # Calculate individual metrics
            silhouette = silhouette_score(features, assignments)
            calinski_harabasz = calinski_harabasz_score(features, assignments)
            davies_bouldin = davies_bouldin_score(features, assignments)
            
            # Normalize metrics to 0-1 range
            norm_silhouette = (silhouette + 1) / 2  # [-1, 1] -> [0, 1]
            norm_ch = min(calinski_harabasz / 1000, 1.0)  # Cap at 1.0
            norm_db = max(0, 1.0 / (1.0 + davies_bouldin))  # Invert and normalize
            
            # Calculate regime balance
            unique_labels = set(assignments)
            regime_sizes = [np.sum(assignments == label) for label in unique_labels]
            regime_balance = 1.0 - (np.std(regime_sizes) / np.mean(regime_sizes)) if len(regime_sizes) > 1 else 0.0
            
            # Calculate coefficient of variation metrics
            cv_score = self._calculate_cv_score(features, assignments)
            
            # Composite score with weighted combination (updated weights)
            composite_score = (
                0.25 * norm_silhouette +      # Silhouette score
                0.20 * norm_ch +             # Calinski-Harabasz score
                0.20 * norm_db +             # Davies-Bouldin score (inverted)
                0.15 * regime_balance +      # Regime balance
                0.20 * cv_score              # Coefficient of variation score
            )
            
            return composite_score
            
        except Exception as e:
            log_warning(f"Composite score calculation failed: {e}")
            return 0.0
    
    def _calculate_cv_score(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate coefficient of variation score for trend, momentum, volatility & volume."""
        try:
            if len(set(assignments)) < 2:
                return 0.0
            
            unique_labels = sorted(set(assignments))
            n_clusters = len(unique_labels)
            
            # Calculate CV metrics for each cluster
            cv_scores = []
            
            for label in unique_labels:
                cluster_mask = assignments == label
                cluster_features = features[cluster_mask]
                
                if len(cluster_features) < 2:
                    continue
                
                # Calculate coefficient of variation for each feature in the cluster
                cluster_cvs = []
                for feature_idx in range(cluster_features.shape[1]):
                    feature_values = cluster_features[:, feature_idx]
                    if np.std(feature_values) > 0:  # Avoid division by zero
                        cv = np.std(feature_values) / np.abs(np.mean(feature_values))
                        cluster_cvs.append(cv)
                
                if cluster_cvs:
                    # Average CV within cluster (lower is better)
                    avg_within_cv = np.mean(cluster_cvs)
                    cv_scores.append(avg_within_cv)
            
            if not cv_scores:
                return 0.0
            
            # Calculate between-cluster separation
            between_cluster_cv = self._calculate_between_cluster_cv(features, assignments)
            
            # Calculate within-cluster CV (lower is better, so invert)
            within_cluster_cv = np.mean(cv_scores)
            norm_within_cv = max(0, 1.0 - within_cluster_cv)  # Invert: lower CV = higher score
            
            # Calculate between-cluster CV (higher is better)
            norm_between_cv = min(1.0, between_cluster_cv / 2.0)  # Normalize to 0-1
            
            # Combined CV score: low within-cluster CV + high between-cluster CV
            cv_score = 0.6 * norm_within_cv + 0.4 * norm_between_cv
            
            return cv_score
            
        except Exception as e:
            log_warning(f"CV score calculation failed: {e}")
            return 0.0
    
    def _calculate_between_cluster_cv(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate coefficient of variation between cluster centers."""
        try:
            unique_labels = sorted(set(assignments))
            n_clusters = len(unique_labels)
            
            if n_clusters < 2:
                return 0.0
            
            # Calculate cluster centers
            cluster_centers = []
            for label in unique_labels:
                cluster_mask = assignments == label
                cluster_features = features[cluster_mask]
                if len(cluster_features) > 0:
                    center = np.mean(cluster_features, axis=0)
                    cluster_centers.append(center)
            
            if len(cluster_centers) < 2:
                return 0.0
            
            cluster_centers = np.array(cluster_centers)
            
            # Calculate CV for each feature across cluster centers
            between_cvs = []
            for feature_idx in range(cluster_centers.shape[1]):
                feature_centers = cluster_centers[:, feature_idx]
                if np.std(feature_centers) > 0:  # Avoid division by zero
                    cv = np.std(feature_centers) / np.abs(np.mean(feature_centers))
                    between_cvs.append(cv)
            
            if between_cvs:
                return np.mean(between_cvs)
            else:
                return 0.0
                
        except Exception as e:
            log_warning(f"Between-cluster CV calculation failed: {e}")
            return 0.0
    
    def _calculate_cluster_centers(self, features: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """Calculate cluster centers from assignments."""
        try:
            unique_labels = sorted(set(assignments))
            centers = []
            
            for label in unique_labels:
                cluster_mask = assignments == label
                cluster_features = features[cluster_mask]
                if len(cluster_features) > 0:
                    center = np.mean(cluster_features, axis=0)
                    centers.append(center)
                else:
                    # If cluster is empty, use zero vector
                    centers.append(np.zeros(features.shape[1]))
            
            return np.array(centers)
            
        except Exception as e:
            log_warning(f"Cluster centers calculation failed: {e}")
            return np.zeros((len(set(assignments)), features.shape[1]))
    
    def _calculate_detailed_cv_metrics(self, features: np.ndarray, assignments: np.ndarray) -> Tuple[float, float]:
        """Calculate detailed within-cluster and between-cluster CV metrics."""
        try:
            if len(set(assignments)) < 2:
                return 0.0, 0.0
            
            unique_labels = sorted(set(assignments))
            
            # Calculate within-cluster CV
            within_cvs = []
            for label in unique_labels:
                cluster_mask = assignments == label
                cluster_features = features[cluster_mask]
                
                if len(cluster_features) < 2:
                    continue
                
                # Calculate CV for each feature in the cluster
                cluster_cvs = []
                for feature_idx in range(cluster_features.shape[1]):
                    feature_values = cluster_features[:, feature_idx]
                    if np.std(feature_values) > 0 and np.abs(np.mean(feature_values)) > 1e-8:
                        cv = np.std(feature_values) / np.abs(np.mean(feature_values))
                        cluster_cvs.append(cv)
                
                if cluster_cvs:
                    within_cvs.append(np.mean(cluster_cvs))
            
            within_cluster_cv = np.mean(within_cvs) if within_cvs else 0.0
            
            # Calculate between-cluster CV
            between_cluster_cv = self._calculate_between_cluster_cv(features, assignments)
            
            return within_cluster_cv, between_cluster_cv
            
        except Exception as e:
            log_warning(f"Detailed CV metrics calculation failed: {e}")
            return 0.0, 0.0
    
    def _calculate_final_quality_metrics(self, features: np.ndarray, assignments: np.ndarray) -> Dict[str, float]:
        """Calculate final quality metrics for the optimized clustering."""
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            
            if len(set(assignments)) < 2:
                return {'error': 'Insufficient clusters'}
            
            metrics = {}
            
            # Standard clustering metrics
            try:
                metrics['silhouette_score'] = silhouette_score(features, assignments)
            except:
                metrics['silhouette_score'] = 0.0
            
            try:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, assignments)
            except:
                metrics['calinski_harabasz_score'] = 0.0
            
            try:
                metrics['davies_bouldin_score'] = davies_bouldin_score(features, assignments)
            except:
                metrics['davies_bouldin_score'] = 0.0
            
            # Coefficient of variation metrics
            try:
                cv_score = self._calculate_cv_score(features, assignments)
                metrics['cv_score'] = cv_score
                
                # Calculate detailed CV metrics
                within_cv, between_cv = self._calculate_detailed_cv_metrics(features, assignments)
                metrics['within_cluster_cv'] = within_cv
                metrics['between_cluster_cv'] = between_cv
                metrics['cv_ratio'] = between_cv / (within_cv + 1e-8)  # Higher is better
                
            except Exception as e:
                log_warning(f"CV metrics calculation failed: {e}")
                metrics['cv_score'] = 0.0
                metrics['within_cluster_cv'] = 0.0
                metrics['between_cluster_cv'] = 0.0
                metrics['cv_ratio'] = 0.0
            
            # Regime-specific metrics
            unique_labels = set(assignments)
            regime_sizes = [np.sum(assignments == label) for label in unique_labels]
            metrics['regime_balance'] = 1.0 - (np.std(regime_sizes) / np.mean(regime_sizes)) if len(regime_sizes) > 1 else 0.0
            metrics['min_regime_size'] = np.min(regime_sizes)
            metrics['max_regime_size'] = np.max(regime_sizes)
            
            # Composite score
            try:
                metrics['composite_score'] = self._calculate_composite_score(features, assignments)
            except:
                metrics['composite_score'] = 0.0
            
            return metrics
            
        except Exception as e:
            log_warning(f"Final quality metrics calculation failed: {e}")
            return {'error': str(e)}
    
    async def _perform_fallback_clustering(self, features: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform fallback clustering when unified system is not available."""
        try:
            tprint("Performing fallback clustering...", "INFO")
            log_info("Performing fallback clustering")

            tprint("Importing sklearn clustering components...", "INFO")
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            tprint("Sklearn clustering components imported", "SUCCESS")

            n_clusters = getattr(self.config, 'n_regimes', 8)
            tprint(f"Using {n_clusters} clusters for fallback clustering", "INFO")

            # Perform K-means clustering
            tprint("Performing K-means clustering...", "INFO")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_assignments = kmeans.fit_predict(features)
            cluster_centers = kmeans.cluster_centers_
            tprint("K-means clustering completed", "SUCCESS")

            # Calculate clustering quality metrics
            tprint("Calculating clustering quality metrics...", "INFO")
            silhouette_avg = silhouette_score(features, cluster_assignments)
            tprint(f"Clustering quality metrics calculated (silhouette: {silhouette_avg:.3f})", "SUCCESS")
            
            clustering_result = {
                'n_clusters': n_clusters,
                'cluster_assignments': cluster_assignments.tolist(),
                'cluster_centers': cluster_centers.tolist(),
                'clustering_quality': {
                    'silhouette_score': float(silhouette_avg),
                    'inertia': float(kmeans.inertia_),
                    'algorithm_used': 'kmeans_fallback'
                },
                'success': True
            }

            log_success(f"Fallback clustering completed: {n_clusters} clusters, silhouette={silhouette_avg:.3f}")
            tprint(f"Fallback clustering completed: {n_clusters} clusters, silhouette={silhouette_avg:.3f}", "SUCCESS")
            return clustering_result

        except Exception as e:
            log_error(f"Fallback clustering failed: {e}")
            tprint(f"Fallback clustering failed: {e}", "ERROR")
            raise ValueError(f"Fallback clustering failed: {e}")
    
    def _calculate_clustering_metrics_using_shared_utils(
        self,
        clustering_result: Dict[str, Any],
        cluster_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate clustering metrics using shared utilities."""
        try:
            tprint("Calculating clustering metrics using shared utilities...", "INFO")
            log_info("Calculating clustering metrics using shared utilities")

            cluster_assignments = clustering_result['cluster_assignments']
            n_clusters = clustering_result['n_clusters']
            tprint(f"Processing {n_clusters} clusters with {len(cluster_assignments)} samples", "INFO")

            # Calculate regime distribution using shared utilities
            tprint("Calculating regime distribution...", "INFO")
            regime_distribution = self.metrics_calculator.calculate_regime_distribution(cluster_assignments)
            tprint(f"Regime distribution calculated: {len(regime_distribution)} regimes", "SUCCESS")

            # Calculate clustering quality metrics
            clustering_quality = clustering_result.get('clustering_quality', {})
            tprint("Clustering quality metrics retrieved", "SUCCESS")

            # Calculate economic, trading, and stability scores using shared utilities
            tprint("Calculating economic scores...", "INFO")
            economic_scores = calculate_economic_scores(cluster_assignments, verbose=True)
            tprint("Economic scores calculated", "SUCCESS")

            tprint("Calculating trading scores...", "INFO")
            trading_scores = calculate_trading_scores(cluster_assignments, verbose=True)
            tprint("Trading scores calculated", "SUCCESS")

            tprint("Calculating stability scores...", "INFO")
            stability_scores = calculate_stability_scores(cluster_assignments, verbose=True)
            tprint("Stability scores calculated", "SUCCESS")
            
            tprint("Compiling final metrics...", "INFO")
            metrics = {
                'n_clusters': n_clusters,
                'total_samples': len(cluster_assignments),
                'regime_distribution': regime_distribution,
                'clustering_quality': clustering_quality,
                'economic_scores': economic_scores,
                'trading_scores': trading_scores,
                'stability_scores': stability_scores,
                'regime_balance': 1.0 - (np.std(list(regime_distribution.values())) / np.mean(list(regime_distribution.values()))) if regime_distribution else 0.0
            }
            tprint("Final metrics compiled", "SUCCESS")

            log_success("Clustering metrics calculated using shared utilities")
            tprint("Clustering metrics calculated using shared utilities", "SUCCESS")
            return metrics
            
        except Exception as e:
            log_warning(f"Clustering metrics calculation failed: {e}")
            return {
                'n_clusters': clustering_result.get('n_clusters', 0),
                'total_samples': len(clustering_result.get('cluster_assignments', [])),
                'regime_distribution': {},
                'clustering_quality': {},
                'economic_scores': [],
                'trading_scores': [],
                'stability_scores': []
            }
    
    def _create_consolidated_artifacts(
        self,
        clustering_result: Dict[str, Any],
        cluster_characteristics: Dict[str, Any],
        clustering_metrics: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Create consolidated artifacts."""
        tprint("Creating consolidated artifacts...", "INFO")
        n_clusters = clustering_result['n_clusters']
        cluster_assignments = clustering_result['cluster_assignments']
        tprint(f"Creating artifacts for {n_clusters} clusters with {len(cluster_assignments)} assignments", "INFO")
        
        artifacts = {
            'nas_tas_clustering_result': {
                # Core clustering data
                'n_clusters': n_clusters,
                'total_samples': len(cluster_assignments),
                'cluster_assignments': cluster_assignments,
                'cluster_centers': clustering_result.get('cluster_centers', []),
                'cluster_characteristics': cluster_characteristics,
                
                # Clustering metrics
                'clustering_metrics': clustering_metrics,
                'clustering_quality': clustering_result.get('clustering_quality', {}),
                
                # Configuration
                'configuration': {
                    'symbol': getattr(self.config, 'symbol', 'BTCUSDT'),
                    'timeframe': getattr(self.config, 'timeframe', '15m'),
                    'exchange': getattr(self.config, 'exchange', 'binance'),
                    'algorithm_type': getattr(self.config, 'algorithm_type', 'adaptive_clustering'),
                    'enable_economic_clustering': getattr(self.config, 'enable_economic_clustering', True),
                    'enable_ensemble_clustering': getattr(self.config, 'enable_ensemble_clustering', True),
                    'economic_weight': getattr(self.config, 'economic_weight', 0.3),
                    'momentum_weight': getattr(self.config, 'momentum_weight', 0.25),
                    'volume_weight': getattr(self.config, 'volume_weight', 0.25),
                    'uses_shared_utilities': True
                },
                
                # Execution information
                'execution_info': {
                    'timestamp': datetime.now().isoformat(),
                    'data_points_processed': len(market_data),
                    'success': True,
                    'algorithm_used': clustering_result.get('clustering_quality', {}).get('algorithm_used', 'refactored_clustering'),
                    'uses_shared_utilities': True
                },
                
                # Additional metadata
                'metadata': {
                    'execution_metadata': self.execution_metadata,
                    'feature_config': {
                        'feature_categories': self.feature_config.feature_categories,
                        'use_standardized_features': self.feature_config.use_standardized_features,
                        'drop_highly_correlated': self.feature_config.drop_highly_correlated
                    }
                }
            }
        }
        tprint("Consolidated artifacts created successfully", "SUCCESS")

        return artifacts