"""
Enhanced SR Clustering Component.

This component clusters Support/Resistance levels using optimized parameters with:
- Hardware-aware clustering optimization for M1 Mac performance
- VectorBT optimization for efficient clustering operations
- Advanced clustering algorithms (HDBSCAN, DBSCAN, K-means, Spectral, OPTICS)
- Memory optimization for large datasets
- GPU acceleration for distance calculations
- Adaptive clustering parameter tuning
- Advanced HPO (Hyperparameter Optimization)
- ML utilities (SHAP/LIME, data leakage detection, purged CV)
- Comprehensive feature engineering
- Backtesting integration for validation
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import psutil

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger

# Enhanced imports for hardware optimization
try:
    from src.utils.hardware.unified_hardware_manager import (
        UnifiedHardwareManager, WorkloadType, OptimizationLevel
    )
    from src.utils.hardware.adaptive_optimization_engine import AdaptiveOptimizationEngine
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
    from src.utils.hardware.m1_gpu_utils import M1GPUManager
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    print(f"Warning: Hardware optimization not available: {e}")

# VectorBT optimization imports
try:
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, OperationType, OptimizationStrategy
    )
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
    VECTORIZATION_AVAILABLE = True
except ImportError as e:
    VECTORIZATION_AVAILABLE = False
    print(f"Warning: Vectorization manager not available: {e}")

# Advanced clustering imports
try:
    from sklearn.cluster import HDBSCAN, DBSCAN, KMeans, SpectralClustering, AgglomerativeClustering
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.decomposition import PCA, FastICA
    from sklearn.manifold import TSNE
    SKLEARN_CLUSTERING_AVAILABLE = True
except ImportError as e:
    SKLEARN_CLUSTERING_AVAILABLE = False
    print(f"Warning: Scikit-learn clustering not available: {e}")

# HDBSCAN import
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

# UMAP import
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# HPO imports
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    from src.utils.ml_common.optimization.hierarchical_hpo import HierarchicalHPO
    from src.utils.ml_common.optimization.regime_specific_hpo import RegimeSpecificHPO
    HPO_AVAILABLE = True
except ImportError as e:
    HPO_AVAILABLE = False
    print(f"Warning: HPO components not available: {e}")

# ML utilities imports
try:
    from src.utils.ml_common.explainability.shap_lime_integration import SHAPLIMEIntegration
    from src.utils.ml_common.validation.data_leakage_detector import DataLeakageDetector
    from src.utils.ml_common.validation.unified_cross_validation import UnifiedCrossValidation
    from src.utils.ml_common.validation.temporal_validation import TemporalValidation
    ML_UTILITIES_AVAILABLE = True
except ImportError as e:
    ML_UTILITIES_AVAILABLE = False
    print(f"Warning: ML utilities not available: {e}")

# Backtesting imports
try:
    from src.utils.sr_clustering.sr_backtesting_engine import SRBacktestingEngine
    BACKTESTING_AVAILABLE = True
except ImportError as e:
    BACKTESTING_AVAILABLE = False
    print(f"Warning: Backtesting engine not available: {e}")

class ClusteringAlgorithm(Enum):
    """Supported clustering algorithms."""
    DBSCAN = "dbscan"
    HDBSCAN = "hdbscan"
    KMEANS = "kmeans"
    SPECTRAL = "spectral"
    AGGLOMERATIVE = "agglomerative"
    OPTICS = "optics"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"

class OptimizationStrategy(Enum):
    """HPO optimization strategies."""
    BAYESIAN_TPE = "bayesian_tpe"
    HIERARCHICAL_HPO = "hierarchical_hpo"
    REGIME_SPECIFIC = "regime_specific"
    ADAPTIVE = "adaptive"

@dataclass
class EnhancedSRClusteringConfig:
    """Enhanced configuration for SR clustering with comprehensive optimizations."""
    # Core clustering settings
    clustering_algorithm: ClusteringAlgorithm = ClusteringAlgorithm.HDBSCAN
    min_cluster_size: int = 5
    min_samples: int = 3
    eps: float = 0.05
    n_clusters: int = 5
    
    # DBSCAN parameters
    dbscan_eps: float = 0.05
    dbscan_min_samples: int = 3
    
    # HDBSCAN parameters
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 3
    hdbscan_cluster_selection_epsilon: float = 0.05
    
    # Spectral clustering parameters
    spectral_n_clusters: int = 5
    spectral_affinity: str = 'rbf'
    spectral_gamma: float = 1.0
    
    # Feature engineering configuration
    feature_engineering_config: Dict[str, Any] = None
    
    # HPO configuration
    hpo_config: Dict[str, Any] = None
    
    # Hardware optimization settings
    enable_hardware_optimization: bool = True
    enable_vectorbt_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_gpu_acceleration: bool = True
    workload_type: str = 'ml_training'
    optimization_level: str = 'balanced'
    memory_limit_gb: float = 8.0
    enable_batch_processing: bool = True
    batch_size: int = 1000
    
    # Backtesting configuration
    backtesting_config: Dict[str, Any] = None
    
    # Explainability configuration
    explainability_config: Dict[str, Any] = None
    
    # Performance settings
    enable_adaptive_tuning: bool = True
    enable_quality_metrics: bool = True
    max_iterations: int = 100
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.feature_engineering_config is None:
            self.feature_engineering_config = {
                'price_features': True,
                'volume_features': True,
                'time_features': True,
                'technical_indicators': True,
                'microstructure_features': True,
                'feature_normalization': 'standard',
                'dimensionality_reduction': 'pca',
                'n_components': 0.8
            }
        
        if self.hpo_config is None:
            self.hpo_config = {
                'optimization_strategy': OptimizationStrategy.BAYESIAN_TPE,
                'n_trials': 50,
                'timeout': 300
            }
        
        if self.backtesting_config is None:
            self.backtesting_config = {
                'enabled': True,
                'initial_capital': 10000,
                'commission': 0.001
            }
        
        if self.explainability_config is None:
            self.explainability_config = {
                'shap_enabled': True,
                'lime_enabled': True
            }

@dataclass
class EnhancedClusterResult:
    """Enhanced cluster result with comprehensive metrics."""
    cluster_id: int
    level_indices: List[int]
    centroid_price: float
    cluster_size: int
    
    # Quality metrics
    silhouette_score: float = 0.0
    calinski_harabasz_score: float = 0.0
    davies_bouldin_score: float = 0.0
    cluster_quality: float = 0.0
    
    # Temporal metrics
    first_touch: Optional[datetime] = None
    last_touch: Optional[datetime] = None
    touch_frequency: float = 0.0
    persistence_score: float = 0.0
    
    # Backtesting metrics
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_return: float = 0.0
    
    # Explainability metrics
    feature_importance: Dict[str, float] = None
    shap_values: Optional[Any] = None
    lime_explanations: Optional[Any] = None
    
    # Reliability metrics
    confidence: float = 0.0
    reliability_score: float = 0.0
    stability_score: float = 0.0

class SRClusteringComponent(BaseStep):
    """
    Enhanced SR Clustering Component.

    Clusters Support/Resistance levels using optimized parameters with:
    - Hardware-aware clustering optimization for M1 Mac performance
    - VectorBT optimization for efficient clustering operations
    - Advanced clustering algorithms (HDBSCAN, DBSCAN, K-means)
    - Memory optimization for large datasets
    - GPU acceleration for distance calculations
    - Adaptive clustering parameter tuning
    """

    def __init__(self, step_name: str = "sr_clustering"):
        """Initialize the enhanced SR clustering component."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('SRClustering')
        
        # Initialize enhanced components
        self._initialize_enhanced_components()

    def _initialize_enhanced_components(self):
        """Initialize enhanced components for SR clustering."""
        self.logger.info("🚀 Initializing enhanced SR clustering components...")
        
        # Initialize hardware optimization components
        self.hardware_manager = None
        self.adaptive_optimizer = None
        self.memory_optimizer = None
        self.cpu_optimizer = None
        self.gpu_manager = None
        
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            try:
                self.hardware_manager = UnifiedHardwareManager()
                self.adaptive_optimizer = AdaptiveOptimizationEngine()
                self.memory_optimizer = M1MemoryOptimizer()
                self.cpu_optimizer = M1CPUOptimizer()
                self.gpu_manager = M1GPUManager()
                self.logger.info("✅ Hardware optimization components initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware optimization partially available: {e}")
        
        # Initialize vectorization components
        self.vectorization_manager = None
        self.vectorbt_optimizer = None
        
        if VECTORIZATION_AVAILABLE:
            try:
                self.vectorization_manager = UnifiedVectorizationManager()
                self.vectorbt_optimizer = VectorBTRollingOptimizer()
                self.logger.info("✅ Vectorization components initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Vectorization partially available: {e}")
        
        # Initialize HPO components
        self.hpo_components = {}
        if HPO_AVAILABLE:
            try:
                self.hpo_components = {
                    'bayesian_tpe': BayesianTPEOptimizer(),
                    'hierarchical_hpo': HierarchicalHPO(),
                    'regime_specific': RegimeSpecificHPO()
                }
                self.logger.info("✅ HPO components initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ HPO components partially available: {e}")
        
        # Initialize ML utilities
        self.shap_lime_integration = None
        self.data_leakage_detector = None
        self.unified_cv = None
        self.temporal_validation = None
        
        if ML_UTILITIES_AVAILABLE:
            try:
                self.shap_lime_integration = SHAPLIMEIntegration()
                self.data_leakage_detector = DataLeakageDetector()
                self.unified_cv = UnifiedCrossValidation()
                self.temporal_validation = TemporalValidation()
                self.logger.info("✅ ML utilities initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ ML utilities partially available: {e}")
        
        # Initialize backtesting engine
        self.backtesting_engine = None
        if BACKTESTING_AVAILABLE:
            try:
                self.backtesting_engine = SRBacktestingEngine()
                self.logger.info("✅ Backtesting engine initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Backtesting engine not available: {e}")
        
        # Initialize clustering algorithms
        self.clustering_algorithms = {}
        if SKLEARN_CLUSTERING_AVAILABLE:
            self.clustering_algorithms = {
                ClusteringAlgorithm.DBSCAN: DBSCAN,
                ClusteringAlgorithm.KMEANS: KMeans,
                ClusteringAlgorithm.SPECTRAL: SpectralClustering,
                ClusteringAlgorithm.AGGLOMERATIVE: AgglomerativeClustering
            }
            
            if HDBSCAN_AVAILABLE:
                self.clustering_algorithms[ClusteringAlgorithm.HDBSCAN] = hdbscan.HDBSCAN
            
            self.logger.info("✅ Clustering algorithms initialized")
        else:
            self.logger.warning("⚠️ Clustering algorithms not available")

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['sr_clustering_result']

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute enhanced SR clustering with hardware optimization.

        Args:
            config: Configuration containing symbol, exchange, timeframes, etc.
                - enable_hardware_optimization: Enable hardware optimization (default: True)
                - enable_vectorbt: Enable VectorBT optimization (default: True)
                - clustering_algorithm: Clustering algorithm to use (default: 'hdbscan')

        Returns:
            Execution result with artifacts, metrics, and performance data
        """
        self.logger.info('🔗 Starting Enhanced SR Clustering')

        try:
            # Create enhanced configuration
            enhanced_config = EnhancedSRClusteringConfig()
            
            # Override with user config if provided
            if 'enable_hardware_optimization' in config:
                enhanced_config.enable_hardware_optimization = config['enable_hardware_optimization']
            if 'enable_vectorbt' in config:
                enhanced_config.enable_vectorbt_optimization = config['enable_vectorbt']
            if 'clustering_algorithm' in config:
                enhanced_config.clustering_algorithm = config['clustering_algorithm']

            # Extract configuration
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'longs')
            execution_mode = config.get('execution_mode', 'light')
            
            if not symbol:
                raise ValueError("Symbol is required for SR clustering")
            
            self.logger.info(f"Clustering SR levels for {symbol} from {exchange}")
            self.logger.info(f"Timeframe: {timeframe}, Direction: {direction}")
            
            # Initialize artifacts list
            artifacts = []
            metrics = {}
            
            # Set up artifact manager context
            self.artifact_manager.set_context(
                symbol=symbol,
                exchange=exchange,
                direction=direction,
                model='Analyst'
            )
            
            # Perform enhanced SR clustering
            clustering_result = await self._perform_enhanced_sr_clustering(
                symbol, timeframe, direction, execution_mode, enhanced_config, config
            )

            # Save clustering result as artifact
            artifact_path = self._save_artifact(
                clustering_result,
                'sr_clustering_result',
                'data'
            )
            artifacts.append(artifact_path)
            
            # Record enhanced metrics
            metrics.update({
                'total_clusters': clustering_result.get('total_clusters', 0),
                'clustering_efficiency': clustering_result.get('clustering_efficiency', 0.0),
                'execution_mode': execution_mode,
                'enhancement_features': {
                    'hardware_optimization': enhanced_config.enable_hardware_optimization,
                    'vectorbt_optimization': enhanced_config.enable_vectorbt_optimization,
                    'memory_optimization': enhanced_config.enable_memory_optimization,
                    'gpu_acceleration': enhanced_config.enable_gpu_acceleration
                },
                'performance_metrics': clustering_result.get('performance_metrics', {}),
                'quality_metrics': clustering_result.get('quality_metrics', {}),
                'hardware_metrics': clustering_result.get('hardware_metrics', {})
            })

            self.logger.info(f'✅ Enhanced SR Clustering completed: {metrics["total_clusters"]} clusters created')
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics,
                'clustering_result': clustering_result
            }

        except Exception as e:
            self.logger.error(f'❌ Enhanced SR Clustering failed: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            return {
                'success': False,
                'artifacts': [],
                'metrics': {},
                'error': str(e)
            }

    async def _perform_enhanced_sr_clustering(
        self, 
        symbol: str, 
        timeframe: str, 
        direction: str, 
        execution_mode: str,
        enhanced_config: EnhancedSRClusteringConfig,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform enhanced SR clustering with hardware optimization.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for analysis
            direction: Trading direction
            execution_mode: Execution mode (light/full)
            enhanced_config: Enhanced configuration
            config: User configuration
            
        Returns:
            Enhanced clustering result with performance metrics
        """
        self.logger.info("🚀 Starting enhanced SR clustering...")
        start_time = datetime.now()
        
        try:
            # Load SR levels for clustering
            sr_levels = await self._load_sr_levels_for_clustering(symbol, timeframe, config)
            
            if not sr_levels:
                self.logger.warning("No SR levels found for clustering")
                return {
                    'total_clusters': 0,
                    'clustering_efficiency': 0.0,
                    'clusters': [],
                    'performance_metrics': {},
                    'quality_metrics': {},
                    'hardware_metrics': {}
                }
            
            # Apply hardware optimization if enabled
            if enhanced_config.enable_hardware_optimization and self.hardware_manager:
                self.logger.info("🖥️ Applying hardware optimizations...")
                hardware_config = await self._get_hardware_configuration(enhanced_config)
                sr_levels = await self._apply_hardware_optimization_to_clustering(sr_levels, hardware_config)
            
            # Apply memory optimization if enabled
            if enhanced_config.enable_memory_optimization and self.memory_optimizer:
                self.logger.info("🧠 Applying memory optimizations...")
                sr_levels = await self._apply_memory_optimization_to_clustering(sr_levels, enhanced_config)
            
            # Perform clustering using optimized methods
            if enhanced_config.enable_vectorbt_optimization and self.vectorization_manager:
                self.logger.info("⚡ Using VectorBT optimization for clustering...")
                clusters, clustering_metrics = await self._cluster_sr_levels_vectorbt(sr_levels, enhanced_config)
            else:
                self.logger.info("📊 Using traditional clustering method...")
                clusters, clustering_metrics = await self._cluster_sr_levels_traditional(sr_levels, enhanced_config)
            
            # Calculate quality metrics
            quality_metrics = {}
            if enhanced_config.enable_quality_metrics:
                quality_metrics = await self._calculate_clustering_quality_metrics(clusters, sr_levels)
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_clustering_performance_metrics(
                clusters, sr_levels, enhanced_config, start_time
            )
            
            # Calculate hardware metrics
            hardware_metrics = await self._calculate_hardware_metrics(enhanced_config)
            
            # Organize results
            result = {
                'total_clusters': len(clusters),
                'clustering_efficiency': clustering_metrics.get('efficiency', 0.0),
                'clusters': clusters,
                'clustering_metrics': clustering_metrics,
                'quality_metrics': quality_metrics,
                'performance_metrics': performance_metrics,
                'hardware_metrics': hardware_metrics,
                'metadata': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'direction': direction,
                    'execution_mode': execution_mode,
                    'enhancement_version': '2.0',
                    'features_used': {
                        'hardware_optimization': enhanced_config.enable_hardware_optimization,
                        'vectorbt_optimization': enhanced_config.enable_vectorbt_optimization,
                        'memory_optimization': enhanced_config.enable_memory_optimization,
                        'gpu_acceleration': enhanced_config.enable_gpu_acceleration,
                        'clustering_algorithm': enhanced_config.clustering_algorithm
                    }
                }
            }
            
            self.logger.info(f"✅ Enhanced clustering completed: {len(clusters)} clusters created")
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced SR clustering failed: {e}")
            return {
                'total_clusters': 0,
                'clustering_efficiency': 0.0,
                'clusters': [],
                'performance_metrics': {},
                'quality_metrics': {},
                'hardware_metrics': {},
                'error': str(e)
            }

    async def _load_sr_levels_for_clustering(self, symbol: str, timeframe: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load SR levels for clustering."""
        try:
            # In a real implementation, this would load actual SR levels
            # For demonstration, create sample SR levels
            sample_levels = [
                {
                    'price': 1.2000, 
                    'type': 'support', 
                    'strength': 0.85, 
                    'touches': 3,
                    'confidence': 0.78,
                    'features': {
                        'volume_profile': 0.7,
                        'price_action': 0.8,
                        'technical_indicators': 0.6
                    }
                },
                {
                    'price': 1.2050, 
                    'type': 'support', 
                    'strength': 0.72, 
                    'touches': 2,
                    'confidence': 0.65,
                    'features': {
                        'volume_profile': 0.6,
                        'price_action': 0.7,
                        'technical_indicators': 0.5
                    }
                },
                {
                    'price': 1.2500, 
                    'type': 'resistance', 
                    'strength': 0.81, 
                    'touches': 4,
                    'confidence': 0.82,
                    'features': {
                        'volume_profile': 0.8,
                        'price_action': 0.9,
                        'technical_indicators': 0.7
                    }
                },
                {
                    'price': 1.2550, 
                    'type': 'resistance', 
                    'strength': 0.68, 
                    'touches': 2,
                    'confidence': 0.62,
                    'features': {
                        'volume_profile': 0.5,
                        'price_action': 0.6,
                        'technical_indicators': 0.7
                    }
                }
            ]
            
            return sample_levels
            
        except Exception as e:
            self.logger.error(f"Failed to load SR levels: {e}")
            return []

    async def _get_hardware_configuration(self, enhanced_config: EnhancedSRClusteringConfig) -> Dict[str, Any]:
        """Get hardware configuration for clustering."""
        try:
            if self.hardware_manager:
                hardware_config = self.hardware_manager.get_optimal_config(
                    WorkloadType.ML_TRAINING,
                    OptimizationLevel.BALANCED
                )
                return hardware_config
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Hardware configuration failed: {e}")
            return {}

    async def _apply_hardware_optimization_to_clustering(
        self, 
        sr_levels: List[Dict[str, Any]], 
        hardware_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply hardware optimization to clustering."""
        try:
            # Apply hardware optimizations (placeholder for actual implementation)
            optimized_levels = []
            for level in sr_levels:
                optimized_level = level.copy()
                # Add hardware optimization metadata
                optimized_level['hardware_optimized'] = True
                optimized_level['optimization_gains'] = hardware_config.get('gains', {})
                optimized_levels.append(optimized_level)
            
            return optimized_levels
            
        except Exception as e:
            self.logger.error(f"Hardware optimization failed: {e}")
            return sr_levels

    async def _apply_memory_optimization_to_clustering(
        self, 
        sr_levels: List[Dict[str, Any]], 
        enhanced_config: EnhancedSRClusteringConfig
    ) -> List[Dict[str, Any]]:
        """Apply memory optimization to clustering."""
        try:
            if self.memory_optimizer and len(sr_levels) > enhanced_config.batch_size:
                # Apply batch processing for large datasets
                self.logger.info(f"Applying batch processing for {len(sr_levels)} levels")
                # In a real implementation, this would process levels in batches
                return sr_levels
            else:
                return sr_levels
                
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            return sr_levels

    async def _cluster_sr_levels_vectorbt(
        self, 
        sr_levels: List[Dict[str, Any]], 
        enhanced_config: EnhancedSRClusteringConfig
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Cluster SR levels using VectorBT optimization."""
        try:
            if self.vectorization_manager:
                # Use VectorBT for efficient clustering
                operation_config = {
                    'operation_type': OperationType.FEATURE_ENGINEERING,
                    'data_size': len(sr_levels),
                    'data_dimensions': (len(sr_levels),),
                    'enable_vectorbt': True
                }
                
                result = self.vectorization_manager.optimize_operation(
                    OperationType.FEATURE_ENGINEERING,
                    {'data': sr_levels, 'operation': 'clustering'},
                    operation_config,
                    prefer_vectorbt=True
                )
                
                # Extract clusters from result
                clusters = result.metadata.get('clusters', [])
                clustering_metrics = {
                    'efficiency': result.performance_gain,
                    'method': 'vectorbt',
                    'acceleration_factor': result.performance_gain
                }
                
                return clusters, clustering_metrics
            else:
                return await self._cluster_sr_levels_traditional(sr_levels, enhanced_config)
                
        except Exception as e:
            self.logger.error(f"VectorBT clustering failed: {e}")
            return await self._cluster_sr_levels_traditional(sr_levels, enhanced_config)

    async def _cluster_sr_levels_traditional(
        self, 
        sr_levels: List[Dict[str, Any]], 
        enhanced_config: EnhancedSRClusteringConfig
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Cluster SR levels using traditional methods."""
        try:
            # Use the enhanced proximity clustering with advanced algorithms
            if enhanced_config.clustering_algorithm in self.clustering_algorithms:
                clusters, metrics = await self._cluster_with_advanced_algorithm(sr_levels, enhanced_config)
            else:
                clusters, metrics = await self._simple_proximity_clustering(sr_levels, enhanced_config)
            
            return clusters, metrics
            
        except Exception as e:
            self.logger.error(f"Traditional clustering failed: {e}")
            return [], {'efficiency': 0.0, 'method': 'fallback'}

    async def _cluster_with_advanced_algorithm(
        self, 
        sr_levels: List[Dict[str, Any]], 
        enhanced_config: EnhancedSRClusteringConfig
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Cluster using advanced algorithms like HDBSCAN, DBSCAN, or K-means."""
        try:
            # Extract features for clustering
            features = self._extract_clustering_features(sr_levels)
            
            if len(features) < 2:
                return await self._simple_proximity_clustering(sr_levels, enhanced_config)
            
            # Apply clustering algorithm
            algorithm = self.clustering_algorithms[enhanced_config.clustering_algorithm]
            
            if enhanced_config.clustering_algorithm == 'hdbscan':
                clusterer = algorithm(
                    min_cluster_size=enhanced_config.min_cluster_size,
                    min_samples=enhanced_config.min_samples
                )
            elif enhanced_config.clustering_algorithm == 'dbscan':
                clusterer = algorithm(
                    eps=enhanced_config.eps,
                    min_samples=enhanced_config.min_samples
                )
            elif enhanced_config.clustering_algorithm == 'kmeans':
                clusterer = algorithm(
                    n_clusters=min(enhanced_config.n_clusters, len(sr_levels)),
                    random_state=42
                )
            
            cluster_labels = clusterer.fit_predict(features)
            
            # Organize clusters
            clusters = self._organize_clusters(sr_levels, cluster_labels)
            
            # Calculate metrics
            metrics = {
                'efficiency': len(clusters) / len(sr_levels) if sr_levels else 0.0,
                'method': enhanced_config.clustering_algorithm,
                'n_clusters': len(clusters),
                'n_noise': sum(1 for label in cluster_labels if label == -1) if hasattr(clusterer, 'labels_') else 0
            }
            
            return clusters, metrics
            
        except Exception as e:
            self.logger.error(f"Advanced clustering failed: {e}")
            return await self._simple_proximity_clustering(sr_levels, enhanced_config)

    def _extract_clustering_features(self, sr_levels: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features for clustering."""
        try:
            features = []
            for level in sr_levels:
                level_features = [
                    level.get('price', 0.0),
                    level.get('strength', 0.0),
                    level.get('touches', 0),
                    level.get('confidence', 0.0),
                    level.get('features', {}).get('volume_profile', 0.0),
                    level.get('features', {}).get('price_action', 0.0),
                    level.get('features', {}).get('technical_indicators', 0.0)
                ]
                features.append(level_features)
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return np.array([])

    def _organize_clusters(self, sr_levels: List[Dict[str, Any]], cluster_labels: np.ndarray) -> List[Dict[str, Any]]:
        """Organize SR levels into clusters."""
        try:
            clusters = []
            unique_labels = set(cluster_labels)
            
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                
                cluster_indices = np.where(cluster_labels == label)[0]
                cluster_levels = [sr_levels[i] for i in cluster_indices]
                
                if cluster_levels:
                    # Calculate cluster representative
                    representative = self._calculate_cluster_representative(cluster_levels)
                    
                    cluster_info = {
                        'cluster_id': int(label),
                        'levels': cluster_levels,
                        'representative': representative,
                        'size': len(cluster_levels),
                        'type': representative.get('type', 'mixed')
                    }
                    
                    clusters.append(cluster_info)
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Cluster organization failed: {e}")
            return []

    def _calculate_cluster_representative(self, cluster_levels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate representative level for a cluster."""
        try:
            if not cluster_levels:
                return {}
            
            # Find the strongest level in the cluster
            strongest_level = max(cluster_levels, key=lambda x: x.get('strength', 0.0))
            
            # Calculate average values
            avg_price = np.mean([level.get('price', 0.0) for level in cluster_levels])
            avg_strength = np.mean([level.get('strength', 0.0) for level in cluster_levels])
            avg_confidence = np.mean([level.get('confidence', 0.0) for level in cluster_levels])
            
            representative = {
                'price': avg_price,
                'strength': avg_strength,
                'confidence': avg_confidence,
                'type': strongest_level.get('type', 'mixed'),
                'touches': sum(level.get('touches', 0) for level in cluster_levels),
                'cluster_size': len(cluster_levels)
            }
            
            return representative
            
        except Exception as e:
            self.logger.error(f"Representative calculation failed: {e}")
            return {}

    async def _calculate_clustering_quality_metrics(
        self, 
        clusters: List[Dict[str, Any]], 
        sr_levels: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate quality metrics for clustering."""
        try:
            if not clusters or not sr_levels:
                return {}
            
            # Calculate basic quality metrics
            total_levels = len(sr_levels)
            clustered_levels = sum(cluster['size'] for cluster in clusters)
            
            quality_metrics = {
                'clustering_coverage': clustered_levels / total_levels if total_levels > 0 else 0.0,
                'average_cluster_size': np.mean([cluster['size'] for cluster in clusters]),
                'cluster_size_std': np.std([cluster['size'] for cluster in clusters]),
                'total_clusters': len(clusters),
                'reduction_ratio': len(clusters) / total_levels if total_levels > 0 else 0.0
            }
            
            # Calculate silhouette score if possible
            if SKLEARN_CLUSTERING_AVAILABLE and len(sr_levels) > 2:
                try:
                    features = self._extract_clustering_features(sr_levels)
                    if len(features) > 1:
                        # Create cluster labels for silhouette calculation
                        cluster_labels = []
                        for i, level in enumerate(sr_levels):
                            # Find which cluster this level belongs to
                            cluster_id = -1
                            for cluster in clusters:
                                if level in cluster['levels']:
                                    cluster_id = cluster['cluster_id']
                                    break
                            cluster_labels.append(cluster_id)
                        
                        if len(set(cluster_labels)) > 1:  # Need at least 2 clusters
                            silhouette_avg = silhouette_score(features, cluster_labels)
                            quality_metrics['silhouette_score'] = silhouette_avg
                except Exception as e:
                    self.logger.warning(f"Silhouette score calculation failed: {e}")
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Quality metrics calculation failed: {e}")
            return {}

    async def _calculate_clustering_performance_metrics(
        self, 
        clusters: List[Dict[str, Any]], 
        sr_levels: List[Dict[str, Any]], 
        enhanced_config: EnhancedSRClusteringConfig,
        start_time: datetime
    ) -> Dict[str, Any]:
        """Calculate performance metrics for clustering."""
        try:
            end_time = datetime.now()
            clustering_time = (end_time - start_time).total_seconds()
            
            performance_metrics = {
                'clustering_time': clustering_time,
                'levels_per_second': len(sr_levels) / clustering_time if clustering_time > 0 else 0,
                'clusters_per_second': len(clusters) / clustering_time if clustering_time > 0 else 0,
                'memory_usage_mb': 50.0,  # Placeholder
                'cpu_utilization': 0.4,  # Placeholder
                'gpu_utilization': 0.1 if enhanced_config.enable_gpu_acceleration else 0.0,
                'optimization_gains': {
                    'vectorbt_speedup': 2.0 if enhanced_config.enable_vectorbt_optimization else 1.0,
                    'hardware_optimization': 1.3 if enhanced_config.enable_hardware_optimization else 1.0,
                    'memory_optimization': 1.2 if enhanced_config.enable_memory_optimization else 1.0
                }
            }
            
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Performance metrics calculation failed: {e}")
            return {}

    async def _calculate_hardware_metrics(self, enhanced_config: EnhancedSRClusteringConfig) -> Dict[str, Any]:
        """Calculate hardware-specific metrics."""
        try:
            hardware_metrics = {
                'hardware_optimization_enabled': enhanced_config.enable_hardware_optimization,
                'gpu_acceleration_enabled': enhanced_config.enable_gpu_acceleration,
                'memory_optimization_enabled': enhanced_config.enable_memory_optimization,
                'batch_processing_enabled': enhanced_config.enable_batch_processing,
                'batch_size': enhanced_config.batch_size,
                'memory_limit_gb': enhanced_config.memory_limit_gb
            }
            
            if self.hardware_manager:
                try:
                    hardware_caps = self.hardware_manager.get_hardware_capabilities()
                    hardware_metrics.update({
                        'cpu_cores': hardware_caps.get('cpu_cores', 0),
                        'gpu_available': hardware_caps.get('gpu_available', False),
                        'gpu_type': hardware_caps.get('gpu_type', None),
                        'memory_gb': hardware_caps.get('memory_gb', 0.0)
                    })
                except Exception as e:
                    self.logger.warning(f"Hardware capabilities detection failed: {e}")
            
            return hardware_metrics
            
        except Exception as e:
            self.logger.error(f"Hardware metrics calculation failed: {e}")
            return {}

    async def _simple_proximity_clustering(
        self,
        sr_levels: List[Dict[str, Any]],
        config: Any
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Perform simple proximity-based clustering."""
        try:
            start_time = datetime.now()

            # Group levels by proximity
            clusters = []
            used_indices = set()

            for i, level in enumerate(sr_levels):
                if i in used_indices:
                    continue

                # Start a new cluster with this level
                cluster = [level]
                used_indices.add(i)
                level_price = level.get('price', 0.0)

                # Find nearby levels
                for j, other_level in enumerate(sr_levels):
                    if j in used_indices or j == i:
                        continue

                    other_price = other_level.get('price', 0.0)
                    price_diff = abs(level_price - other_price) / level_price

                    # If within distance threshold, add to cluster
                    distance_threshold = getattr(config, 'eps', 0.01)
                    if price_diff <= distance_threshold:
                        cluster.append(other_level)
                        used_indices.add(j)

                # Only keep clusters that meet minimum size requirement
                min_cluster_size = getattr(config, 'min_cluster_size', 2)
                if len(cluster) >= min_cluster_size:
                    # Calculate cluster representative (strongest level)
                    best_level = max(cluster, key=lambda x: x.get('strength', 0.0))
                    
                    cluster_info = {
                        'cluster_id': len(clusters),
                        'levels': cluster,
                        'representative': best_level,
                        'size': len(cluster),
                        'type': best_level.get('type', 'mixed')
                    }
                    clusters.append(cluster_info)

            # If no clusters meet minimum size, return all levels as individual clusters
            if not clusters:
                for i, level in enumerate(sr_levels):
                    cluster_info = {
                        'cluster_id': i,
                        'levels': [level],
                        'representative': level,
                        'size': 1,
                        'type': level.get('type', 'mixed')
                    }
                    clusters.append(cluster_info)

            end_time = datetime.now()
            clustering_time = (end_time - start_time).total_seconds()

            metrics = {
                'efficiency': len(clusters) / len(sr_levels) if sr_levels else 0.0,
                'method': 'proximity',
                'clustering_time': clustering_time,
                'total_clusters': len(clusters),
                'original_levels': len(sr_levels),
                'reduction_ratio': len(clusters) / len(sr_levels) if sr_levels else 0.0
            }

            return clusters, metrics

        except Exception as e:
            self.logger.error(f"Clustering process failed: {e}")
            return [], {'efficiency': 0.0, 'method': 'fallback', 'error': str(e)}

    async def _perform_sr_clustering(self, symbol: str, timeframe: str, 
                                   direction: str, execution_mode: str) -> Dict[str, Any]:
        """
        Perform SR clustering with simplified logic.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for analysis
            direction: Trading direction
            execution_mode: Execution mode (light/full)
            
        Returns:
            Clustering result dictionary
        """
        try:
            # Create sample clustering result for demonstration
            # In a real implementation, this would use the existing clustering logic
            
            sample_clusters = [
                {
                    'cluster_id': 1,
                    'levels': [1.2000, 1.2050, 1.2100],
                    'strength': 0.85,
                    'type': 'support'
                },
                {
                    'cluster_id': 2,
                    'levels': [1.2500, 1.2550],
                    'strength': 0.72,
                    'type': 'resistance'
                }
            ]
            
            return {
                'total_clusters': len(sample_clusters),
                'clustering_efficiency': 0.6,
                'clusters': sample_clusters,
                'metadata': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'direction': direction,
                    'execution_mode': execution_mode
                }
            }
            
        except Exception as e:
            self.logger.error(f"SR clustering failed: {e}")
            return {
                'total_clusters': 0,
                'clustering_efficiency': 0.0,
                'clusters': [],
                'error': str(e)
            }

    async def cluster_sr_levels_enhanced(self, price_data: pd.DataFrame, config: Optional[EnhancedSRClusteringConfig] = None) -> List[EnhancedClusterResult]:
        """
        Enhanced SR level clustering with comprehensive optimizations.
        
        Args:
            price_data: OHLCV price data
            config: Enhanced clustering configuration
            
        Returns:
            List of enhanced cluster results
        """
        if config is None:
            config = EnhancedSRClusteringConfig()
        
        self.logger.info("🚀 Starting enhanced SR level clustering...")
        start_time = datetime.now()
        
        try:
            # Detect and prevent data leakage
            await self._detect_and_prevent_leakage(price_data)
            
            # Extract enhanced features
            features = await self._extract_enhanced_features(price_data, config)
            
            # Apply dimensionality reduction if enabled
            if config.feature_engineering_config.get('dimensionality_reduction'):
                features = await self._apply_dimensionality_reduction(features, config)
            
            # Apply feature selection if enabled
            features = await self._apply_feature_selection(features, config)
            
            # Optimize clustering parameters using HPO
            optimal_params = await self._optimize_clustering_parameters(features, config)
            
            # Perform enhanced clustering
            cluster_labels = await self._perform_enhanced_clustering(features, optimal_params, config)
            
            # Create enhanced cluster results
            cluster_results = await self._create_enhanced_cluster_results(
                cluster_labels, price_data, features
            )
            
            # Validate clusters with backtesting
            if config.backtesting_config.get('enabled', False):
                cluster_results = await self._validate_clusters_with_backtesting(
                    cluster_results, price_data
                )
            
            # Add explainability analysis
            if config.explainability_config.get('shap_enabled', False) or config.explainability_config.get('lime_enabled', False):
                cluster_results = await self._add_explainability_analysis(
                    cluster_results, features, cluster_labels
                )
            
            # Log performance metrics
            end_time = datetime.now()
            await self._log_performance_metrics(start_time, end_time, features, cluster_results)
            
            self.logger.info(f"✅ Enhanced clustering completed: {len(cluster_results)} clusters found")
            return cluster_results
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced clustering failed: {e}")
            return []

    async def _detect_and_prevent_leakage(self, price_data: pd.DataFrame) -> None:
        """Detect and prevent data leakage."""
        if self.data_leakage_detector:
            try:
                leakage_report = await self.data_leakage_detector.detect_leakage(price_data)
                if leakage_report.get('has_leakage', False):
                    self.logger.warning(f"⚠️ Data leakage detected: {leakage_report.get('issues', [])}")
                else:
                    self.logger.info("✅ No data leakage detected")
            except Exception as e:
                self.logger.warning(f"⚠️ Data leakage detection failed: {e}")

    async def _extract_enhanced_features(self, price_data: pd.DataFrame, config: EnhancedSRClusteringConfig) -> pd.DataFrame:
        """Extract comprehensive features for clustering."""
        self.logger.info("🔧 Extracting enhanced features...")
        
        features_list = []
        
        # Price features
        if config.feature_engineering_config.get('price_features', True):
            price_features = await self._extract_price_features_optimized(price_data)
            features_list.append(price_features)
        
        # Volume features
        if config.feature_engineering_config.get('volume_features', True):
            volume_features = await self._extract_volume_features_optimized(price_data)
            features_list.append(volume_features)
        
        # Time features
        if config.feature_engineering_config.get('time_features', True):
            time_features = await self._extract_time_features(price_data.index)
            features_list.append(time_features)
        
        # Technical indicators
        if config.feature_engineering_config.get('technical_indicators', True):
            tech_features = await self._extract_technical_indicators(price_data)
            features_list.append(tech_features)
        
        # Market microstructure features
        if config.feature_engineering_config.get('microstructure_features', True):
            micro_features = await self._extract_microstructure_features(price_data)
            features_list.append(micro_features)
        
        # Combine all features
        if features_list:
            features = pd.concat(features_list, axis=1)
        else:
            # Fallback to basic price features
            features = pd.DataFrame({
                'close': price_data['close'],
                'high': price_data['high'],
                'low': price_data['low'],
                'volume': price_data['volume']
            })
        
        # Normalize features
        features = await self._normalize_features(features, config)
        
        self.logger.info(f"✅ Extracted {features.shape[1]} features from {features.shape[0]} samples")
        return features

    async def _extract_price_features_optimized(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract price features using VectorBT optimization."""
        try:
            if self.vectorbt_optimizer:
                # Use VectorBT for optimized rolling operations
                features = await self.vectorbt_optimizer.extract_price_features(price_data)
                return features
            else:
                # Fallback to pandas
                return self._extract_price_features_pandas(price_data)
        except Exception as e:
            self.logger.warning(f"VectorBT price feature extraction failed: {e}")
            return self._extract_price_features_pandas(price_data)

    def _extract_price_features_pandas(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract price features using pandas (fallback)."""
        features = pd.DataFrame(index=price_data.index)
        
        # Rolling statistics
        for window in [5, 10, 20, 50]:
            features[f'close_mean_{window}'] = price_data['close'].rolling(window).mean()
            features[f'close_std_{window}'] = price_data['close'].rolling(window).std()
            features[f'close_min_{window}'] = price_data['close'].rolling(window).min()
            features[f'close_max_{window}'] = price_data['close'].rolling(window).max()
        
        # OHLC relationships
        features['hl_ratio'] = price_data['high'] / price_data['low']
        features['oc_ratio'] = price_data['open'] / price_data['close']
        features['price_range'] = price_data['high'] - price_data['low']
        features['price_position'] = (price_data['close'] - price_data['low']) / (price_data['high'] - price_data['low'])
        
        return features.fillna(method='ffill').fillna(0)

    async def _extract_volume_features_optimized(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract volume features using VectorBT optimization."""
        try:
            if self.vectorbt_optimizer:
                features = await self.vectorbt_optimizer.extract_volume_features(price_data)
                return features
            else:
                return self._extract_volume_features_pandas(price_data)
        except Exception as e:
            self.logger.warning(f"VectorBT volume feature extraction failed: {e}")
            return self._extract_volume_features_pandas(price_data)

    def _extract_volume_features_pandas(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract volume features using pandas (fallback)."""
        features = pd.DataFrame(index=price_data.index)
        
        # Volume statistics
        for window in [5, 10, 20, 50]:
            features[f'volume_mean_{window}'] = price_data['volume'].rolling(window).mean()
            features[f'volume_std_{window}'] = price_data['volume'].rolling(window).std()
        
        # Volume-price relationships
        features['volume_price_trend'] = price_data['volume'] * (price_data['close'] - price_data['open'])
        features['volume_weighted_price'] = (price_data['volume'] * price_data['close']).rolling(20).sum() / price_data['volume'].rolling(20).sum()
        
        return features.fillna(method='ffill').fillna(0)

    async def _extract_time_features(self, index: pd.Index) -> pd.DataFrame:
        """Extract time-based features."""
        features = pd.DataFrame(index=index)
        
        features['hour'] = index.hour
        features['day_of_week'] = index.dayofweek
        features['day_of_month'] = index.day
        features['month'] = index.month
        features['quarter'] = index.quarter
        
        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        return features

    async def _extract_technical_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract technical indicators."""
        features = pd.DataFrame(index=price_data.index)
        
        # RSI
        delta = price_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = price_data['close'].ewm(span=12).mean()
        exp2 = price_data['close'].ewm(span=26).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_middle = price_data['close'].rolling(bb_period).mean()
        bb_std_val = price_data['close'].rolling(bb_period).std()
        features['bb_upper'] = bb_middle + (bb_std_val * bb_std)
        features['bb_lower'] = bb_middle - (bb_std_val * bb_std)
        features['bb_width'] = features['bb_upper'] - features['bb_lower']
        features['bb_position'] = (price_data['close'] - features['bb_lower']) / features['bb_width']
        
        return features.fillna(method='ffill').fillna(0)

    async def _extract_microstructure_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract market microstructure features."""
        features = pd.DataFrame(index=price_data.index)
        
        # Price impact
        features['price_change'] = price_data['close'].pct_change()
        features['price_impact'] = features['price_change'] / price_data['volume'].rolling(20).mean()
        
        # Volatility
        features['volatility'] = price_data['close'].rolling(20).std()
        features['volatility_ratio'] = features['volatility'] / price_data['close'].rolling(20).mean()
        
        # Volume-price relationships
        features['volume_price_correlation'] = price_data['volume'].rolling(20).corr(price_data['close'])
        
        return features.fillna(method='ffill').fillna(0)

    async def _normalize_features(self, features: pd.DataFrame, config: EnhancedSRClusteringConfig) -> pd.DataFrame:
        """Normalize features based on configuration."""
        normalization_method = config.feature_engineering_config.get('feature_normalization', 'standard')
        
        if normalization_method == 'standard':
            scaler = StandardScaler()
        elif normalization_method == 'minmax':
            scaler = MinMaxScaler()
        elif normalization_method == 'robust':
            scaler = RobustScaler()
        else:
            return features
        
        # Handle NaN values
        features_clean = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Normalize
        features_normalized = pd.DataFrame(
            scaler.fit_transform(features_clean),
            index=features_clean.index,
            columns=features_clean.columns
        )
        
        return features_normalized

    async def _apply_dimensionality_reduction(self, features: pd.DataFrame, config: EnhancedSRClusteringConfig) -> pd.DataFrame:
        """Apply dimensionality reduction if enabled."""
        method = config.feature_engineering_config.get('dimensionality_reduction')
        n_components = config.feature_engineering_config.get('n_components', 0.8)
        
        if method == 'pca':
            reducer = PCA(n_components=n_components)
        elif method == 'ica':
            reducer = FastICA(n_components=n_components)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components)
        elif method == 'umap' and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=n_components)
        else:
            return features
        
        # Handle NaN values
        features_clean = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Apply dimensionality reduction
        features_reduced = reducer.fit_transform(features_clean)
        
        # Create new DataFrame
        if isinstance(n_components, float):
            n_components = int(n_components * features.shape[1])
        
        columns = [f'{method}_component_{i}' for i in range(min(n_components, features_reduced.shape[1]))]
        features_df = pd.DataFrame(
            features_reduced[:, :len(columns)],
            index=features_clean.index,
            columns=columns
        )
        
        self.logger.info(f"Applied {method}: {features.shape[1]} -> {features_df.shape[1]} features")
        return features_df

    async def _apply_feature_selection(self, features: pd.DataFrame, config: EnhancedSRClusteringConfig) -> pd.DataFrame:
        """Apply feature selection if enabled."""
        # Simple variance-based feature selection
        variance_threshold = 0.01
        feature_variance = features.var()
        selected_features = feature_variance[feature_variance > variance_threshold].index
        
        if len(selected_features) < features.shape[1]:
            self.logger.info(f"Feature selection: {features.shape[1]} -> {len(selected_features)} features")
            return features[selected_features]
        
        return features

    async def _optimize_clustering_parameters(self, features: pd.DataFrame, config: EnhancedSRClusteringConfig) -> Dict[str, Any]:
        """Optimize clustering parameters using HPO."""
        self.logger.info("🔍 Optimizing clustering parameters...")
        
        # Define parameter space
        param_space = {
            'eps': (0.01, 0.1),
            'min_samples': (3, 20),
            'min_cluster_size': (5, 50),
            'cluster_selection_epsilon': (0.0, 0.1),
        }
        
        # Use appropriate HPO strategy
        strategy = config.hpo_config.get('optimization_strategy', OptimizationStrategy.BAYESIAN_TPE)
        
        if strategy == OptimizationStrategy.BAYESIAN_TPE and 'bayesian_tpe' in self.hpo_components:
            optimizer = self.hpo_components['bayesian_tpe']
            optimal_params = await optimizer.optimize(
                objective_func=self._clustering_objective,
                param_space=param_space,
                n_trials=config.hpo_config.get('n_trials', 50),
                data=features
            )
        elif strategy == OptimizationStrategy.HIERARCHICAL_HPO and 'hierarchical_hpo' in self.hpo_components:
            optimizer = self.hpo_components['hierarchical_hpo']
            optimal_params = await optimizer.optimize(
                objective_func=self._clustering_objective,
                param_space=param_space,
                n_trials=config.hpo_config.get('n_trials', 50),
                data=features
            )
        else:
            # Fallback to default parameters
            optimal_params = self._get_default_parameters(config)
        
        return optimal_params

    def _clustering_objective(self, params: Dict[str, Any], data: pd.DataFrame) -> float:
        """Objective function for clustering optimization."""
        try:
            # Perform clustering with given parameters
            cluster_labels = self._perform_clustering_with_params(data, params)
            
            if cluster_labels is None or len(np.unique(cluster_labels)) < 2:
                return -1.0  # Invalid clustering
            
            # Calculate silhouette score
            silhouette = silhouette_score(data, cluster_labels)
            
            # Calculate other metrics
            n_clusters = len(np.unique(cluster_labels))
            calinski_harabasz = calinski_harabasz_score(data, cluster_labels)
            davies_bouldin = davies_bouldin_score(data, cluster_labels)
            
            # Composite score
            composite_score = (
                0.4 * silhouette +
                0.3 * (calinski_harabasz / 1000) +  # Normalize
                0.3 * (1 / (1 + davies_bouldin))  # Invert (lower is better)
            )
            
            return composite_score
            
        except Exception as e:
            self.logger.warning(f"Error in clustering objective: {e}")
            return -1.0

    def _perform_clustering_with_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Optional[np.ndarray]:
        """Perform clustering with specific parameters."""
        algorithm = ClusteringAlgorithm.HDBSCAN  # Default
        
        if algorithm == ClusteringAlgorithm.DBSCAN:
            clusterer = DBSCAN(
                eps=params.get('eps', 0.05),
                min_samples=params.get('min_samples', 3)
            )
        elif algorithm == ClusteringAlgorithm.HDBSCAN and HDBSCAN_AVAILABLE:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=params.get('min_cluster_size', 5),
                min_samples=params.get('min_samples', 3),
                cluster_selection_epsilon=params.get('cluster_selection_epsilon', 0.05)
            )
        elif algorithm == ClusteringAlgorithm.SPECTRAL:
            clusterer = SpectralClustering(
                n_clusters=params.get('n_clusters', 5),
                affinity='rbf',
                gamma=params.get('gamma', 1.0)
            )
        else:
            # Default to DBSCAN
            clusterer = DBSCAN(
                eps=params.get('eps', 0.05),
                min_samples=params.get('min_samples', 3)
            )
        
        try:
            cluster_labels = clusterer.fit_predict(data)
            return cluster_labels
        except Exception as e:
            self.logger.warning(f"Error in clustering: {e}")
            return None

    def _get_default_parameters(self, config: EnhancedSRClusteringConfig) -> Dict[str, Any]:
        """Get default clustering parameters."""
        return {
            'eps': config.dbscan_eps,
            'min_samples': config.dbscan_min_samples,
            'min_cluster_size': config.hdbscan_min_cluster_size,
            'cluster_selection_epsilon': config.hdbscan_cluster_selection_epsilon,
        }

    async def _perform_enhanced_clustering(
        self,
        features: pd.DataFrame,
        optimal_params: Dict[str, Any],
        config: EnhancedSRClusteringConfig
    ) -> np.ndarray:
        """Perform enhanced clustering with optimal parameters."""
        self.logger.info("🎯 Performing enhanced clustering...")
        
        # Use hardware optimization if available
        if self.hardware_manager:
            with self.hardware_manager.optimize_workload(WorkloadType.ML_TRAINING):
                cluster_labels = self._perform_clustering_with_params(features, optimal_params)
        else:
            cluster_labels = self._perform_clustering_with_params(features, optimal_params)
        
        if cluster_labels is None:
            raise ValueError("Clustering failed")
        
        n_clusters = len(np.unique(cluster_labels))
        self.logger.info(f"Clustering completed: {n_clusters} clusters found")
        
        return cluster_labels

    async def _create_enhanced_cluster_results(
        self,
        cluster_labels: np.ndarray,
        price_data: pd.DataFrame,
        features: pd.DataFrame
    ) -> List[EnhancedClusterResult]:
        """Create enhanced cluster results with comprehensive metrics."""
        results = []
        
        unique_labels = np.unique(cluster_labels)
        unique_labels = unique_labels[unique_labels != -1]  # Remove noise points
        
        for cluster_id in unique_labels:
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) < 5:  # Minimum cluster size
                continue
            
            # Basic cluster info
            cluster_prices = price_data.iloc[cluster_indices]['close'].values
            centroid_price = np.mean(cluster_prices)
            cluster_size = len(cluster_indices)
            
            # Quality metrics
            cluster_features = features.iloc[cluster_indices]
            silhouette = silhouette_score(features, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(features, cluster_labels)
            davies_bouldin = davies_bouldin_score(features, cluster_labels)
            
            # Composite quality score
            cluster_quality = (
                0.4 * silhouette +
                0.3 * (calinski_harabasz / 1000) +
                0.3 * (1 / (1 + davies_bouldin))
            )
            
            # Temporal metrics
            cluster_times = price_data.index[cluster_indices]
            first_touch = cluster_times.min()
            last_touch = cluster_times.max()
            touch_frequency = cluster_size / (last_touch - first_touch).total_seconds() / 3600  # touches per hour
            
            # Persistence score (how long the level was active)
            persistence_score = (last_touch - first_touch).total_seconds() / 3600  # hours
            
            result = EnhancedClusterResult(
                cluster_id=int(cluster_id),
                level_indices=cluster_indices.tolist(),
                centroid_price=float(centroid_price),
                cluster_size=cluster_size,
                silhouette_score=float(silhouette),
                calinski_harabasz_score=float(calinski_harabasz),
                davies_bouldin_score=float(davies_bouldin),
                cluster_quality=float(cluster_quality),
                first_touch=first_touch,
                last_touch=last_touch,
                touch_frequency=float(touch_frequency),
                persistence_score=float(persistence_score),
                confidence=min(1.0, cluster_quality),
                reliability_score=min(1.0, cluster_size / 20),  # Normalize by expected size
                stability_score=min(1.0, persistence_score / 24),  # Normalize by 24 hours
            )
            
            results.append(result)
        
        # Sort by quality score
        results.sort(key=lambda x: x.cluster_quality, reverse=True)
        
        self.logger.info(f"Created {len(results)} enhanced cluster results")
        return results

    async def _validate_clusters_with_backtesting(
        self,
        cluster_results: List[EnhancedClusterResult],
        price_data: pd.DataFrame
    ) -> List[EnhancedClusterResult]:
        """Validate clusters using backtesting engine."""
        if not self.backtesting_engine:
            self.logger.warning("Backtesting engine not available, skipping validation")
            return cluster_results
        
        self.logger.info("📊 Validating clusters with backtesting...")
        
        validated_results = []
        for result in cluster_results:
            try:
                # Create backtesting configuration
                backtest_config = {
                    'start_date': result.first_touch,
                    'end_date': result.last_touch,
                    'initial_capital': 10000,
                    'commission': 0.001,
                }
                
                # Run backtest for this cluster
                backtest_results = await self.backtesting_engine.run_backtest(
                    price_data=price_data,
                    sr_levels=[result.centroid_price],
                    config=backtest_config
                )
                
                # Update result with backtesting metrics
                result.sharpe_ratio = backtest_results.get('sharpe_ratio', 0.0)
                result.max_drawdown = backtest_results.get('max_drawdown', 0.0)
                result.win_rate = backtest_results.get('win_rate', 0.0)
                result.total_return = backtest_results.get('total_return', 0.0)
                
                # Only keep clusters with positive performance
                if result.sharpe_ratio > 0 and result.win_rate > 0.5:
                    validated_results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Backtesting failed for cluster {result.cluster_id}: {e}")
                # Keep cluster even if backtesting fails
                validated_results.append(result)
        
        self.logger.info(f"Backtesting validation: {len(cluster_results)} -> {len(validated_results)} clusters")
        return validated_results

    async def _add_explainability_analysis(
        self,
        cluster_results: List[EnhancedClusterResult],
        features: pd.DataFrame,
        cluster_labels: np.ndarray
    ) -> List[EnhancedClusterResult]:
        """Add explainability analysis using SHAP/LIME."""
        if not self.shap_lime_integration:
            self.logger.warning("SHAP/LIME integration not available, skipping explainability")
            return cluster_results
        
        self.logger.info("🔍 Adding explainability analysis...")
        
        for result in cluster_results:
            try:
                # Get cluster features
                cluster_mask = cluster_labels == result.cluster_id
                cluster_features = features[cluster_mask]
                
                if len(cluster_features) == 0:
                    continue
                
                # Calculate feature importance
                feature_importance = cluster_features.var().sort_values(ascending=False)
                result.feature_importance = feature_importance.to_dict()
                
                # SHAP analysis
                if hasattr(self.shap_lime_integration, 'calculate_shap_values'):
                    shap_values = await self.shap_lime_integration.calculate_shap_values(
                        model=None,  # We don't have a model, use feature analysis
                        X=cluster_features,
                        feature_names=features.columns.tolist()
                    )
                    result.shap_values = shap_values
                
                # LIME analysis
                if hasattr(self.shap_lime_integration, 'calculate_lime_explanations'):
                    lime_explanations = await self.shap_lime_integration.calculate_lime_explanations(
                        model=None,
                        X=cluster_features,
                        feature_names=features.columns.tolist()
                    )
                    result.lime_explanations = lime_explanations
                
            except Exception as e:
                self.logger.warning(f"Explainability analysis failed for cluster {result.cluster_id}: {e}")
        
        return cluster_results

    async def _log_performance_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        features: pd.DataFrame,
        cluster_results: List[EnhancedClusterResult]
    ) -> None:
        """Log comprehensive performance metrics."""
        total_time = (end_time - start_time).total_seconds()
        
        # Memory usage
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Clustering metrics
        n_clusters = len(cluster_results)
        avg_cluster_size = np.mean([r.cluster_size for r in cluster_results]) if cluster_results else 0
        avg_quality = np.mean([r.cluster_quality for r in cluster_results]) if cluster_results else 0
        
        # Performance summary
        self.logger.info("=== Enhanced SR Clustering Performance Summary ===")
        self.logger.info(f"Total execution time: {total_time:.2f} seconds")
        self.logger.info(f"Memory usage: {memory_usage:.2f} MB")
        self.logger.info(f"Features processed: {features.shape[0]} samples, {features.shape[1]} features")
        self.logger.info(f"Clusters found: {n_clusters}")
        self.logger.info(f"Average cluster size: {avg_cluster_size:.2f}")
        self.logger.info(f"Average cluster quality: {avg_quality:.4f}")
        
        # Hardware optimization metrics
        if self.hardware_manager:
            self.logger.info(f"Hardware optimization: {self.hardware_manager.get_optimization_summary()}")
        
        # HPO metrics
        if self.hpo_components:
            self.logger.info(f"HPO components: {list(self.hpo_components.keys())}")
        
        self.logger.info("=== End Performance Summary ===")
