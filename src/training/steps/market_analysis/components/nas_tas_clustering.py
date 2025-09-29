"""
Refactored NAS-TAS Clustering Component.

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
            # Step 1: Validate inputs and configuration using shared utilities
            log_info("Validating inputs and configuration using shared utilities")
            validation_errors = self.config_validator.validate_config(self.config)
            if validation_errors:
                log_error(f"Configuration validation failed: {validation_errors}")
                raise ValueError(f"Configuration validation failed: {validation_errors}")
            
            log_success("Configuration validation passed using shared utilities")
            
            # Step 2: Initialize execution metadata
            self.execution_metadata = {
                'start_time': datetime.now(),
                'symbol': getattr(self.config, 'symbol', 'BTCUSDT'),
                'timeframe': getattr(self.config, 'timeframe', '15m'),
                'exchange': getattr(self.config, 'exchange', 'binance'),
                'component': 'refactored_nas_tas_clustering',
                'uses_shared_utilities': True
            }
            
            # Step 3: Load and validate market data
            log_info("Loading and validating market data")
            market_data = await self._load_market_data(data)
            if market_data is None or market_data.empty:
                raise ValueError("No market data available for clustering")
            
            log_success(f"Market data loaded: {len(market_data)} rows")
            
            # Step 4: Prepare features using shared utilities
            log_info("Preparing features using shared utilities")
            features = prepare_market_features(market_data, self.feature_config, verbose=True)
            if features is None:
                raise ValueError("Failed to prepare features for clustering")
            
            log_success(f"Features prepared: {features.shape}")
            
            # Step 5: Create clustering configuration using shared utilities
            clustering_config = self._create_clustering_config_using_shared_utils()
            
            # Step 6: Initialize unified clustering
            log_info("Initializing unified clustering")
            self._initialize_unified_clustering(clustering_config)
            
            # Step 7: Perform clustering
            log_info("Performing clustering")
            clustering_result = await self._perform_clustering(features, market_data)
            
            # Step 8: Generate cluster characteristics using shared utilities
            log_info("Generating cluster characteristics using shared utilities")
            cluster_characteristics = generate_cluster_characteristics(
                market_data, clustering_result['cluster_assignments'], 
                clustering_result.get('cluster_centers'), verbose=True
            )
            
            # Step 9: Calculate metrics using shared utilities
            log_info("Calculating clustering metrics using shared utilities")
            clustering_metrics = self._calculate_clustering_metrics_using_shared_utils(
                clustering_result, cluster_characteristics
            )
            
            # Step 10: Create consolidated artifacts
            artifacts = self._create_consolidated_artifacts(
                clustering_result, cluster_characteristics, clustering_metrics, market_data
            )
            
            log_success(f'NAS-TAS Clustering completed: {clustering_result["n_clusters"]} clusters')
            
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
            
            import traceback
            error_traceback = traceback.format_exc()
            self.logger.error(f'❌ Error details: {error_traceback}')
            
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"NAS-TAS clustering failed: {str(e)}"
            )
    
    async def _load_market_data(self, data: Any) -> Optional[pd.DataFrame]:
        """Load and validate market data for clustering."""
        try:
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                log_warning("No market data provided, attempting to load from pipeline state")
                return None
            
            # If data is already a DataFrame, use it
            if isinstance(data, pd.DataFrame):
                log_info(f"Using provided DataFrame with {len(data)} rows")
                return data.copy()
            
            # If data is a dictionary with market data
            if isinstance(data, dict) and 'market_data' in data:
                market_data = data['market_data']
                if isinstance(market_data, pd.DataFrame):
                    log_info(f"Using market data from dictionary with {len(market_data)} rows")
                    return market_data.copy()
            
            log_warning("Unknown data type provided")
            return None
            
        except Exception as e:
            log_error(f"Market data loading failed: {e}")
            return None
    
    def _create_clustering_config_using_shared_utils(self) -> Dict[str, Any]:
        """Create clustering configuration using shared utilities."""
        try:
            log_info("Creating clustering configuration using shared utilities")
            
            # Use shared utilities to create configuration
            base_config = create_default_config(
                config_type="clustering",
                symbol=getattr(self.config, 'symbol', 'BTCUSDT'),
                timeframe=getattr(self.config, 'timeframe', '15m'),
                n_regimes=getattr(self.config, 'n_regimes', 8)
            )
            
            # Add clustering-specific parameters
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
            
            # Validate weights using shared utilities
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
            
            log_success("Clustering configuration created using shared utilities")
            return clustering_config
            
        except Exception as e:
            log_warning(f"Config creation failed: {e}, using defaults")
            return create_default_config("clustering")
    
    def _initialize_unified_clustering(self, clustering_config: Dict[str, Any]):
        """Initialize unified clustering system."""
        try:
            log_info("Initializing unified clustering system")
            
            # Import unified clustering components
            from src.training.steps.market_analysis.hybrid_nas_tas_regime.unified_clustering import (
                UnifiedClusteringSystem, UnifiedClusteringConfig
            )
            
            # Create unified clustering configuration
            unified_config = UnifiedClusteringConfig(
                n_clusters=clustering_config['n_regimes'],
                algorithm_type=clustering_config['algorithm_type'],
                enable_economic_clustering=clustering_config['enable_economic_clustering'],
                enable_ensemble_clustering=clustering_config['enable_ensemble_clustering'],
                economic_weight=clustering_config['economic_weight'],
                momentum_weight=clustering_config['momentum_weight'],
                volume_weight=clustering_config['volume_weight']
            )
            
            # Initialize unified clustering system
            self.unified_clustering = UnifiedClusteringSystem(unified_config)
            
            log_success("Unified clustering system initialized")
            
        except ImportError:
            log_warning("Unified clustering components not available, using fallback")
            self.unified_clustering = None
        except Exception as e:
            log_error(f"Unified clustering initialization failed: {e}")
            self.unified_clustering = None
    
    async def _perform_clustering(self, features: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering using unified system or fallback."""
        try:
            if self.unified_clustering is not None:
                log_info("Performing clustering using unified system")
                clustering_result = await self.unified_clustering.perform_clustering(features, market_data)
            else:
                log_info("Performing clustering using fallback method")
                clustering_result = await self._perform_fallback_clustering(features, market_data)
            
            log_success(f"Clustering completed: {clustering_result['n_clusters']} clusters")
            return clustering_result
            
        except Exception as e:
            log_error(f"Clustering failed: {e}")
            raise ValueError(f"Clustering failed: {e}")
    
    async def _perform_fallback_clustering(self, features: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform fallback clustering when unified system is not available."""
        try:
            log_info("Performing fallback clustering")
            
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            n_clusters = getattr(self.config, 'n_regimes', 8)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_assignments = kmeans.fit_predict(features)
            cluster_centers = kmeans.cluster_centers_
            
            # Calculate clustering quality metrics
            silhouette_avg = silhouette_score(features, cluster_assignments)
            
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
            return clustering_result
            
        except Exception as e:
            log_error(f"Fallback clustering failed: {e}")
            raise ValueError(f"Fallback clustering failed: {e}")
    
    def _calculate_clustering_metrics_using_shared_utils(
        self,
        clustering_result: Dict[str, Any],
        cluster_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate clustering metrics using shared utilities."""
        try:
            log_info("Calculating clustering metrics using shared utilities")
            
            cluster_assignments = clustering_result['cluster_assignments']
            n_clusters = clustering_result['n_clusters']
            
            # Calculate regime distribution using shared utilities
            regime_distribution = self.metrics_calculator.calculate_regime_distribution(cluster_assignments)
            
            # Calculate clustering quality metrics
            clustering_quality = clustering_result.get('clustering_quality', {})
            
            # Calculate economic, trading, and stability scores using shared utilities
            economic_scores = calculate_economic_scores(cluster_assignments, verbose=True)
            trading_scores = calculate_trading_scores(cluster_assignments, verbose=True)
            stability_scores = calculate_stability_scores(cluster_assignments, verbose=True)
            
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
            
            log_success("Clustering metrics calculated using shared utilities")
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
        n_clusters = clustering_result['n_clusters']
        cluster_assignments = clustering_result['cluster_assignments']
        
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
        
        return artifacts