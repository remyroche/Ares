"""
Refactored NAS-TAS Clustering Component.

This module provides a cleaner, more maintainable implementation of the
NAS-TAS clustering component with better separation of concerns.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from src.utils.tprint import (
    tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_performance, tprint_structured
)

from .base_component import BaseMarketAnalysisComponent, ComponentResult
from .clustering_config import NASTASClusteringConfig, ConfigurationManager
from .memory_manager import MemoryManager, memory_checkpoint
from .clustering_algorithms import (
    ClusteringAlgorithmFactory, BaseClusteringAlgorithm, ClusteringResult
)

from ..shared_utils import (
    prepare_market_features, FeatureConfig,
    get_logger, log_execution, log_performance, LoggingContext,
    calculate_consensus_metrics, calculate_disagreement_metrics,
    create_regime_characteristics, CharacteristicsGenerator
)

from ..regime_analysis.label_fusion_refactored import LabelFusionService


@dataclass
class ClusteringContext:
    """Context for clustering operations with memory management."""
    
    original_features: np.ndarray
    market_data: pd.DataFrame
    optimized_features: Optional[np.ndarray] = None
    optimal_k: Optional[int] = None
    clustering_result: Optional[ClusteringResult] = None
    fusion_result: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_manager: Optional[MemoryManager] = None
    
    def __enter__(self):
        """Context manager entry."""
        if self.memory_manager:
            self.memory_manager.force_cleanup()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self.memory_manager:
            self.memory_manager.force_cleanup()


class NASTASClusteringComponent(BaseMarketAnalysisComponent):
    """
    Refactored NAS-TAS Clustering Component.
    
    This component provides regime-aware clustering with improved:
    - Memory management
    - Error handling
    - Code organization
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[NASTASClusteringConfig] = None):
        """Initialize the refactored NAS-TAS clustering component."""
        with LoggingContext('NAS-TAS-Clustering-Refactored', 'Initialization', verbose=True):
            # Initialize base component
            super().__init__(config)
            
            # Initialize configuration manager
            self.config_manager = ConfigurationManager()
            if config is None:
                config = self.config_manager.create_config("nas_tas")
            self.config = config
            
            # Initialize logging
            self.logger = get_logger('NASTASClusteringRefactored')
            
            # Initialize memory manager
            self.memory_manager = MemoryManager(
                memory_limit_mb=getattr(config, 'memory_limit_mb', None),
                enable_m1_optimization=getattr(config, 'enable_m1_optimization', True)
            )
            
            # Initialize feature configuration
            self.feature_config = FeatureConfig(
                feature_categories=getattr(config, 'feature_categories', [
                    'regime_volatility', 
                    'regime_volume', 
                    'regime_structural_trend', 
                    'regime_statistical'
                ]),
                use_standardized_features=getattr(config, 'use_standardized_features', True),
                drop_highly_correlated=True
            )
            
            # Initialize clustering algorithm
            self.clustering_algorithm = ClusteringAlgorithmFactory.create_algorithm(
                algorithm_type=getattr(config, 'algorithm_type', 'adaptive_clustering'),
                config=config,
                memory_manager=self.memory_manager
            )
            
            # Initialize label fusion service
            self.label_fusion_service = LabelFusionService()
            
            # Initialize characteristics generator
            self.characteristics_generator = CharacteristicsGenerator(verbose=True)
            
            # Performance monitoring
            self.performance_metrics = {
                'start_time': None,
                'end_time': None,
                'memory_usage': [],
                'processing_times': {},
                'error_count': 0,
                'success_count': 0
            }
            
            tprint_success("🔍 Refactored NAS-TAS Clustering component initialized")
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """Execute NAS-TAS clustering with improved error handling and monitoring."""
        try:
            # Initialize performance monitoring
            self.performance_metrics['start_time'] = time.time()
            self.performance_metrics['error_count'] = 0
            self.performance_metrics['success_count'] = 0
            
            tprint_info("🚀 Starting refactored NAS-TAS clustering execution")
            
            # Step 1: Validate inputs
            self._validate_execution_inputs(data, pipeline_state)
            
            # Step 2: Load and prepare data
            market_data = await self._load_market_data(data)
            if market_data is None or market_data.empty:
                raise ValueError("No market data available for clustering")
            
            # Step 3: Prepare features
            features = self._prepare_features(market_data)
            
            # Step 4: Create clustering context
            with ClusteringContext(
                original_features=features,
                market_data=market_data,
                memory_manager=self.memory_manager
            ) as context:
                
                # Step 5: Perform clustering
                clustering_result = await self._perform_clustering(context)
                
                # Step 6: Generate characteristics
                characteristics = self._generate_characteristics(context, clustering_result)
                
                # Step 7: Calculate metrics
                metrics = self._calculate_metrics(clustering_result, characteristics)
                
                # Step 8: Build artifacts
                artifacts = self._build_artifacts(clustering_result, characteristics, metrics, market_data)
                
                # Update performance metrics
                self.performance_metrics['end_time'] = time.time()
                self.performance_metrics['success_count'] += 1
                
                tprint_success(f"NAS-TAS clustering completed: {clustering_result.n_clusters} clusters")
                
                return ComponentResult(
                    success=True,
                    artifacts=artifacts,
                    metadata={
                        'symbol': getattr(self.config, 'symbol', 'BTCUSDT'),
                        'timeframe': getattr(self.config, 'timeframe', '15m'),
                        'data_points_processed': len(market_data),
                        'n_clusters': clustering_result.n_clusters,
                        'algorithm_type': clustering_result.algorithm,
                        'execution_successful': True,
                        'refactored_component': True,
                        'performance_metrics': self.performance_metrics
                    }
                )
                
        except Exception as e:
            self.performance_metrics['error_count'] += 1
            tprint_error(f'NAS-TAS clustering failed: {e}')
            
            return ComponentResult(
                success=False,
                artifacts={},
                metadata={
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'execution_successful': False,
                    'refactored_component': True,
                    'performance_metrics': self.performance_metrics
                }
            )
    
    def _validate_execution_inputs(self, data: Any, pipeline_state: Dict[str, Any]) -> None:
        """Validate execution inputs."""
        try:
            if data is None:
                raise ValueError("Data cannot be None")
            
            if not isinstance(pipeline_state, dict):
                raise ValueError("Pipeline state must be a dictionary")
            
            tprint_info("Input validation passed")
            
        except Exception as e:
            tprint_error(f"Input validation failed: {e}")
            raise
    
    async def _load_market_data(self, data: Any) -> pd.DataFrame:
        """Load and validate market data."""
        try:
            if isinstance(data, pd.DataFrame):
                market_data = data.copy()
            elif isinstance(data, dict) and 'market_data' in data:
                market_data = data['market_data']
            else:
                raise ValueError("Invalid data format")
            
            # Validate market data
            if market_data.empty:
                raise ValueError("Market data is empty")
            
            # Optimize memory usage
            market_data = self.memory_manager.optimize_memory_usage(market_data)
            
            tprint_info(f"Market data loaded: {len(market_data)} rows")
            return market_data
            
        except Exception as e:
            tprint_error(f"Failed to load market data: {e}")
            raise
    
    def _prepare_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Prepare features for clustering."""
        try:
            with memory_checkpoint("feature_preparation", self.memory_manager):
                # Prepare features using shared utilities
                features = prepare_market_features(
                    market_data, 
                    self.feature_config,
                    verbose=True
                )
                
                # Validate features
                if features is None or len(features) == 0:
                    raise ValueError("No features generated")
                
                # Optimize memory usage
                features = self.memory_manager.optimize_memory_usage(features)
                
                tprint_info(f"Features prepared: {features.shape}")
                return features
                
        except Exception as e:
            tprint_error(f"Feature preparation failed: {e}")
            raise
    
    async def _perform_clustering(self, context: ClusteringContext) -> ClusteringResult:
        """Perform clustering using the configured algorithm."""
        try:
            with memory_checkpoint("clustering_execution", self.memory_manager):
                # Perform clustering
                clustering_result = self.clustering_algorithm.fit_predict(context.original_features)
                
                # Store result in context
                context.clustering_result = clustering_result
                
                tprint_success(f"Clustering completed: {clustering_result.n_clusters} clusters")
                return clustering_result
                
        except Exception as e:
            tprint_error(f"Clustering failed: {e}")
            raise
    
    def _generate_characteristics(self, context: ClusteringContext, clustering_result: ClusteringResult) -> Dict[str, Any]:
        """Generate cluster characteristics."""
        try:
            with memory_checkpoint("characteristics_generation", self.memory_manager):
                # Generate characteristics using shared utilities
                characteristics = create_regime_characteristics(
                    context.market_data,
                    clustering_result.labels,
                    verbose=True
                )
                
                tprint_success("Cluster characteristics generated")
                return characteristics
                
        except Exception as e:
            tprint_error(f"Characteristics generation failed: {e}")
            raise
    
    def _calculate_metrics(self, clustering_result: ClusteringResult, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate clustering metrics."""
        try:
            with memory_checkpoint("metrics_calculation", self.memory_manager):
                # Use clustering result metrics
                metrics = clustering_result.metrics.copy()
                
                # Add additional metrics
                metrics.update({
                    'execution_time': clustering_result.execution_time,
                    'algorithm': clustering_result.algorithm,
                    'n_clusters': clustering_result.n_clusters,
                    'n_samples': clustering_result.n_samples
                })
                
                tprint_success("Metrics calculated")
                return metrics
                
        except Exception as e:
            tprint_error(f"Metrics calculation failed: {e}")
            raise
    
    def _build_artifacts(
        self, 
        clustering_result: ClusteringResult, 
        characteristics: Dict[str, Any], 
        metrics: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Build artifacts for the clustering result."""
        try:
            artifacts = {
                'clustering_result': clustering_result.to_dict(),
                'characteristics': characteristics,
                'metrics': metrics,
                'market_data_info': {
                    'n_rows': len(market_data),
                    'n_columns': len(market_data.columns),
                    'columns': list(market_data.columns)
                },
                'config': self.config.to_dict(),
                'performance_metrics': self.performance_metrics
            }
            
            tprint_success("Artifacts built")
            return artifacts
            
        except Exception as e:
            tprint_error(f"Artifact building failed: {e}")
            raise
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    def reset_performance_metrics(self) -> None:
        """Reset performance metrics."""
        self.performance_metrics = {
            'start_time': None,
            'end_time': None,
            'memory_usage': [],
            'processing_times': {},
            'error_count': 0,
            'success_count': 0
        }
        tprint_info("Performance metrics reset")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            if self.memory_manager:
                self.memory_manager.force_cleanup()
        except Exception:
            pass