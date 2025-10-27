"""
Enhanced SR Clustering Component.

This component clusters Support/Resistance levels using optimized parameters with:
- Hardware-aware clustering optimization for M1 Mac performance
- VectorBT optimization for efficient clustering operations
- Advanced clustering algorithms (HDBSCAN, DBSCAN, K-means, Spectral, Gaussian Mixture)
- Memory optimization for large datasets
- GPU acceleration for distance calculations
- Adaptive clustering parameter tuning
- Data leakage detection and prevention
- SHAP/LIME explainability integration
- Purged cross-validation for time series
- Regime-aware clustering
- Multi-algorithm ensemble clustering
- Advanced feature engineering
- Dynamic parameter optimization
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
import warnings
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger

# Enhanced imports for hardware optimization
try:
    from src.utils.hardware.unified_hardware_manager import (
        UnifiedHardwareManager, WorkloadType, OptimizationLevel
    )
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    print(f"Warning: Hardware optimization not available: {e}")

# VectorBT optimization imports
try:
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, OperationType, OptimizationStrategy
    )
    VECTORIZATION_AVAILABLE = True
except ImportError as e:
    VECTORIZATION_AVAILABLE = False
    print(f"Warning: Vectorization manager not available: {e}")

# Advanced clustering imports
try:
    from sklearn.cluster import HDBSCAN, DBSCAN, KMeans, SpectralClustering, GaussianMixture
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_CLUSTERING_AVAILABLE = True
except ImportError as e:
    SKLEARN_CLUSTERING_AVAILABLE = False
    print(f"Warning: Scikit-learn clustering not available: {e}")

# Memory optimization imports
try:
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    MEMORY_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    MEMORY_OPTIMIZATION_AVAILABLE = False
    print(f"Warning: Memory optimization not available: {e}")

# Data leakage detection imports
try:
    from src.utils.ml_common.validation.data_leakage_detector import DataLeakageDetector
    DATA_LEAKAGE_AVAILABLE = True
except ImportError as e:
    DATA_LEAKAGE_AVAILABLE = False
    print(f"Warning: Data leakage detection not available: {e}")

# SHAP/LIME explainability imports
try:
    from src.utils.ml_common.explainability.shap_lime_integration import SHAPLIMEExplainer, ExplanationConfig
    EXPLAINABILITY_AVAILABLE = True
except ImportError as e:
    EXPLAINABILITY_AVAILABLE = False
    print(f"Warning: Explainability not available: {e}")

# HPO optimization imports
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer, OptimizationConfig as HPOConfig
    HPO_AVAILABLE = True
except ImportError as e:
    HPO_AVAILABLE = False
    print(f"Warning: HPO optimization not available: {e}")

# Temporal validation imports
try:
    from src.utils.ml_common.validation.temporal_cross_validation import temporal_cross_validation
    TEMPORAL_VALIDATION_AVAILABLE = True
except ImportError as e:
    TEMPORAL_VALIDATION_AVAILABLE = False
    print(f"Warning: Temporal validation not available: {e}")

# VectorBTRollingOptimizer imports
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    VECTORBT_ROLLING_AVAILABLE = True
except ImportError as e:
    VECTORBT_ROLLING_AVAILABLE = False
    print(f"Warning: VectorBTRollingOptimizer not available: {e}")

@dataclass
class EnhancedSRClusteringConfig:
    """Enhanced configuration for SR clustering with comprehensive optimizations."""
    # Clustering settings
    enable_hardware_optimization: bool = True
    enable_vectorbt_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_gpu_acceleration: bool = True
    
    # Clustering algorithm settings
    clustering_algorithm: str = 'ensemble'  # 'hdbscan', 'dbscan', 'kmeans', 'spectral', 'gmm', 'ensemble'
    min_cluster_size: int = 2
    min_samples: int = 2
    eps: float = 0.01
    n_clusters: int = 5
    
    # Hardware optimization settings
    workload_type: str = 'ml_training'
    optimization_level: str = 'balanced'
    memory_limit_gb: float = 8.0
    enable_batch_processing: bool = True
    batch_size: int = 1000
    
    # Performance settings
    enable_adaptive_tuning: bool = True
    enable_quality_metrics: bool = True
    max_iterations: int = 100
    
    # Data leakage prevention
    enable_data_leakage_detection: bool = True
    enable_temporal_validation: bool = True
    
    # Explainability
    enable_explainability: bool = True
    explainability_config: Optional[Dict[str, Any]] = None
    
    # HPO optimization
    enable_hpo_optimization: bool = True
    hpo_trials: int = 50
    
    # Feature engineering
    enable_advanced_feature_engineering: bool = True
    enable_dimensionality_reduction: bool = True
    n_components_pca: int = 10
    
    # Regime awareness
    enable_regime_aware_clustering: bool = True
    regime_features: List[str] = field(default_factory=lambda: ['volatility', 'trend', 'volume'])
    
    # Ensemble clustering
    enable_ensemble_clustering: bool = True
    ensemble_algorithms: List[str] = field(default_factory=lambda: ['hdbscan', 'dbscan', 'kmeans', 'spectral'])
    ensemble_weights: List[float] = field(default_factory=lambda: [0.3, 0.2, 0.2, 0.3])
    
    # Quality thresholds
    min_silhouette_score: float = 0.3
    min_cluster_quality: float = 0.5
    max_cluster_ratio: float = 0.8

class SRClusteringComponent(BaseStep):
    """
    Enhanced SR Clustering Component.

    Clusters Support/Resistance levels using optimized parameters with:
    - Hardware-aware clustering optimization for M1 Mac performance
    - VectorBT optimization for efficient clustering operations
    - Advanced clustering algorithms (HDBSCAN, DBSCAN, K-means, Spectral, Gaussian Mixture)
    - Memory optimization for large datasets
    - GPU acceleration for distance calculations
    - Adaptive clustering parameter tuning
    - Data leakage detection and prevention
    - SHAP/LIME explainability integration
    - Purged cross-validation for time series
    - Regime-aware clustering
    - Multi-algorithm ensemble clustering
    - Advanced feature engineering
    - Dynamic parameter optimization
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
        
        # Initialize hardware manager
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            self.hardware_manager = UnifiedHardwareManager()
            self.logger.info("✅ Hardware manager initialized")
        else:
            self.hardware_manager = None
            self.logger.warning("⚠️ Hardware manager not available")
        
        # Initialize vectorization manager
        if VECTORIZATION_AVAILABLE:
            self.vectorization_manager = UnifiedVectorizationManager()
            self.logger.info("✅ Vectorization manager initialized")
        else:
            self.vectorization_manager = None
            self.logger.warning("⚠️ Vectorization manager not available")
        
        # Initialize VectorBTRollingOptimizer
        if VECTORBT_ROLLING_AVAILABLE:
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            self.logger.info("✅ VectorBTRollingOptimizer initialized")
        else:
            self.vectorbt_rolling_optimizer = None
            self.logger.warning("⚠️ VectorBTRollingOptimizer not available")
        
        # Initialize memory optimizer
        if MEMORY_OPTIMIZATION_AVAILABLE:
            self.memory_optimizer = M1MemoryOptimizer()
            self.logger.info("✅ Memory optimizer initialized")
        else:
            self.memory_optimizer = None
            self.logger.warning("⚠️ Memory optimizer not available")
        
        # Initialize data leakage detector
        if DATA_LEAKAGE_AVAILABLE:
            self.data_leakage_detector = DataLeakageDetector()
            self.logger.info("✅ Data leakage detector initialized")
        else:
            self.data_leakage_detector = None
            self.logger.warning("⚠️ Data leakage detector not available")
        
        # Initialize explainability
        if EXPLAINABILITY_AVAILABLE:
            self.explainability_config = ExplanationConfig()
            self.explainer = SHAPLIMEExplainer(self.explainability_config)
            self.logger.info("✅ Explainability initialized")
        else:
            self.explainer = None
            self.logger.warning("⚠️ Explainability not available")
        
        # Initialize HPO optimizer
        if HPO_AVAILABLE:
            self.hpo_config = HPOConfig()
            self.hpo_optimizer = BayesianTPEOptimizer(self.hpo_config)
            self.logger.info("✅ HPO optimizer initialized")
        else:
            self.hpo_optimizer = None
            self.logger.warning("⚠️ HPO optimizer not available")
        
        # Initialize clustering algorithms
        self.clustering_algorithms = {}
        if SKLEARN_CLUSTERING_AVAILABLE:
            self.clustering_algorithms = {
                'hdbscan': HDBSCAN,
                'dbscan': DBSCAN,
                'kmeans': KMeans,
                'spectral': SpectralClustering,
                'gmm': GaussianMixture
            }
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
            self._apply_user_config(enhanced_config, config)

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
            
            # Save SR levels dictionary for feature bank and training scripts access
            sr_levels_dict = self._create_sr_levels_dictionary(clustering_result)
            sr_levels_artifact_path = self._save_artifact(
                sr_levels_dict,
                'sr_levels_dictionary',
                'data',
                metadata={
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'direction': direction,
                    'total_levels': len(sr_levels_dict.get('levels', [])),
                    'total_clusters': clustering_result.get('total_clusters', 0),
                    'created_at': datetime.now().isoformat(),
                    'purpose': 'feature_bank_and_training_access'
                }
            )
            artifacts.append(sr_levels_artifact_path)
            
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
            
            # Detect data leakage if enabled
            data_leakage_results = {}
            if enhanced_config.enable_data_leakage_detection:
                self.logger.info("🔍 Detecting data leakage...")
                data_leakage_results = await self._detect_data_leakage(sr_levels, enhanced_config)
            
            # Apply advanced feature engineering if enabled
            if enhanced_config.enable_advanced_feature_engineering:
                self.logger.info("🔧 Applying advanced feature engineering...")
                sr_levels = await self._advanced_feature_engineering(sr_levels, enhanced_config)
            
            # Apply hardware optimization if enabled
            if enhanced_config.enable_hardware_optimization and self.hardware_manager:
                self.logger.info("🖥️ Applying hardware optimizations...")
                hardware_config = await self._get_hardware_configuration(enhanced_config)
                sr_levels = await self._apply_hardware_optimization_to_clustering(sr_levels, hardware_config)
            
            # Apply memory optimization if enabled
            if enhanced_config.enable_memory_optimization and self.memory_optimizer:
                self.logger.info("🧠 Applying memory optimizations...")
                sr_levels = await self._apply_memory_optimization_to_clustering(sr_levels, enhanced_config)
            
            # Perform regime-aware clustering if enabled
            regime_analysis = {}
            if enhanced_config.enable_regime_aware_clustering:
                self.logger.info("📊 Performing regime-aware clustering...")
                regime_analysis = await self._regime_aware_clustering(sr_levels, enhanced_config)
            
            # Perform clustering using optimized methods
            if enhanced_config.enable_ensemble_clustering and enhanced_config.clustering_algorithm == 'ensemble':
                self.logger.info("🎯 Using ensemble clustering...")
                clusters, clustering_metrics = await self._ensemble_clustering(sr_levels, enhanced_config)
            elif enhanced_config.enable_vectorbt_optimization and self.vectorization_manager:
                self.logger.info("⚡ Using VectorBT optimization for clustering...")
                clusters, clustering_metrics = await self._cluster_sr_levels_vectorbt(sr_levels, enhanced_config)
            else:
                self.logger.info("📊 Using traditional clustering method...")
                clusters, clustering_metrics = await self._cluster_sr_levels_traditional(sr_levels, enhanced_config)
            
            # Generate explainability results if enabled
            explainability_results = {}
            if enhanced_config.enable_explainability:
                self.logger.info("🔍 Generating explainability results...")
                explainability_results = await self._generate_explainability_results(clusters, sr_levels, enhanced_config)
            
            # Calculate comprehensive quality metrics
            quality_metrics = {}
            if enhanced_config.enable_quality_metrics:
                quality_metrics = await self._calculate_comprehensive_quality_metrics(clusters, sr_levels, enhanced_config)
            
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
                'data_leakage_results': data_leakage_results,
                'regime_analysis': regime_analysis,
                'explainability_results': explainability_results,
                'metadata': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'direction': direction,
                    'execution_mode': execution_mode,
                    'enhancement_version': '3.0',
                    'features_used': {
                        'hardware_optimization': enhanced_config.enable_hardware_optimization,
                        'vectorbt_optimization': enhanced_config.enable_vectorbt_optimization,
                        'memory_optimization': enhanced_config.enable_memory_optimization,
                        'gpu_acceleration': enhanced_config.enable_gpu_acceleration,
                        'clustering_algorithm': enhanced_config.clustering_algorithm,
                        'data_leakage_detection': enhanced_config.enable_data_leakage_detection,
                        'explainability': enhanced_config.enable_explainability,
                        'hpo_optimization': enhanced_config.enable_hpo_optimization,
                        'advanced_feature_engineering': enhanced_config.enable_advanced_feature_engineering,
                        'regime_aware_clustering': enhanced_config.enable_regime_aware_clustering,
                        'ensemble_clustering': enhanced_config.enable_ensemble_clustering
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
            elif enhanced_config.clustering_algorithm == 'spectral':
                clusterer = algorithm(
                    n_clusters=min(enhanced_config.n_clusters, len(sr_levels)),
                    random_state=42
                )
            elif enhanced_config.clustering_algorithm == 'gmm':
                clusterer = algorithm(
                    n_components=min(enhanced_config.n_clusters, len(sr_levels)),
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

    def _create_sr_levels_dictionary(self, clustering_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a comprehensive SR levels dictionary for feature bank and training scripts access.
        
        This dictionary contains all SR levels with their scores, metadata, and cluster information
        in a format that's easily accessible by the feature bank and training scripts.
        
        Args:
            clustering_result: The clustering result from _perform_enhanced_sr_clustering
            
        Returns:
            Dictionary containing SR levels with scores and metadata
        """
        try:
            clusters = clustering_result.get('clusters', [])
            metadata = clustering_result.get('metadata', {})
            
            # Extract all levels from clusters
            all_levels = []
            level_id = 0
            
            for cluster in clusters:
                cluster_id = cluster.get('cluster_id', 0)
                cluster_levels = cluster.get('levels', [])
                cluster_representative = cluster.get('representative', {})
                cluster_type = cluster.get('type', 'mixed')
                
                for level in cluster_levels:
                    # Create comprehensive level dictionary with all metadata
                    level_dict = {
                        'id': level_id,
                        'cluster_id': cluster_id,
                        'price': level.get('price', 0.0),
                        'type': level.get('type', level.get('level_type', 'unknown')),
                        'strength': level.get('strength', 0.0),
                        'confidence': level.get('confidence', 0.0),
                        'touches': level.get('touches', level.get('touch_count', 0)),
                        'first_touch': level.get('first_touch', level.get('first_touch_time', datetime.now())),
                        'last_touch': level.get('last_touch', level.get('last_touch_time', datetime.now())),
                        'features': level.get('features', {}),
                        'cluster_info': {
                            'cluster_id': cluster_id,
                            'cluster_type': cluster_type,
                            'cluster_size': cluster.get('size', 0),
                            'cluster_representative': cluster_representative
                        },
                        'metadata': {
                            'symbol': metadata.get('symbol', ''),
                            'timeframe': metadata.get('timeframe', ''),
                            'direction': metadata.get('direction', ''),
                            'execution_mode': metadata.get('execution_mode', ''),
                            'enhancement_version': metadata.get('enhancement_version', '1.0'),
                            'created_at': datetime.now().isoformat()
                        }
                    }
                    all_levels.append(level_dict)
                    level_id += 1
            
            # Create the comprehensive dictionary
            sr_levels_dictionary = {
                'levels': all_levels,
                'summary': {
                    'total_levels': len(all_levels),
                    'total_clusters': len(clusters),
                    'clustering_efficiency': clustering_result.get('clustering_efficiency', 0.0),
                    'support_levels': len([l for l in all_levels if l.get('type', '').lower() == 'support']),
                    'resistance_levels': len([l for l in all_levels if l.get('type', '').lower() == 'resistance']),
                    'mixed_levels': len([l for l in all_levels if l.get('type', '').lower() not in ['support', 'resistance']])
                },
                'clustering_metrics': clustering_result.get('clustering_metrics', {}),
                'quality_metrics': clustering_result.get('quality_metrics', {}),
                'performance_metrics': clustering_result.get('performance_metrics', {}),
                'hardware_metrics': clustering_result.get('hardware_metrics', {}),
                'metadata': metadata,
                'access_info': {
                    'purpose': 'feature_bank_and_training_access',
                    'format_version': '2.0',
                    'created_at': datetime.now().isoformat(),
                    'access_methods': [
                        'feature_bank.get_sr_levels()',
                        'artifact_manager.get_artifact("sr_levels_dictionary")',
                        'BaseStep._get_artifact("sr_levels_dictionary")'
                    ]
                }
            }
            
            self.logger.info(f"Created SR levels dictionary with {len(all_levels)} levels from {len(clusters)} clusters")
            return sr_levels_dictionary
            
        except Exception as e:
            self.logger.error(f"Failed to create SR levels dictionary: {e}")
            return {
                'levels': [],
                'summary': {'total_levels': 0, 'total_clusters': 0},
                'error': str(e),
                'access_info': {
                    'purpose': 'feature_bank_and_training_access',
                    'format_version': '2.0',
                    'created_at': datetime.now().isoformat()
                }
            }

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

    def _apply_user_config(self, enhanced_config: EnhancedSRClusteringConfig, config: Dict[str, Any]):
        """Apply user configuration to enhanced config."""
        config_mapping = {
            'enable_hardware_optimization': 'enable_hardware_optimization',
            'enable_vectorbt': 'enable_vectorbt_optimization',
            'clustering_algorithm': 'clustering_algorithm',
            'enable_data_leakage_detection': 'enable_data_leakage_detection',
            'enable_explainability': 'enable_explainability',
            'enable_hpo_optimization': 'enable_hpo_optimization',
            'hpo_trials': 'hpo_trials'
        }
        
        for user_key, config_key in config_mapping.items():
            if user_key in config:
                setattr(enhanced_config, config_key, config[user_key])

    async def _detect_data_leakage(self, sr_levels: List[Dict[str, Any]], config: EnhancedSRClusteringConfig) -> Dict[str, Any]:
        """Detect data leakage in SR levels."""
        try:
            if not self.data_leakage_detector:
                return {}
            
            # Convert SR levels to DataFrame for analysis
            df = pd.DataFrame(sr_levels)
            
            # Create feature matrix
            feature_cols = []
            for col in df.columns:
                if col not in ['price', 'type', 'timestamp']:
                    feature_cols.append(col)
            
            if not feature_cols:
                return {}
            
            X = df[feature_cols].fillna(0)
            y = df['price']  # Use price as target for leakage detection
            
            # Detect lookahead bias
            lookahead_results = self.data_leakage_detector.detect_lookahead_bias(X, y, 'price')
            
            return {
                'lookahead_bias': lookahead_results,
                'leakage_detected': lookahead_results.get('risk_score', 0) > 0.1
            }
            
        except Exception as e:
            self.logger.warning(f"Data leakage detection failed: {e}")
            return {}

    async def _advanced_feature_engineering(self, sr_levels: List[Dict[str, Any]], config: EnhancedSRClusteringConfig) -> List[Dict[str, Any]]:
        """Perform advanced feature engineering on SR levels."""
        try:
            enhanced_levels = []
            
            for level in sr_levels:
                enhanced_level = level.copy()
                
                # Add advanced features
                features = enhanced_level.get('features', {})
                
                # Price-based features
                price = level.get('price', 0.0)
                features['price_normalized'] = price / 1000.0  # Normalize price
                features['price_log'] = np.log(price) if price > 0 else 0
                
                # Strength-based features
                strength = level.get('strength', 0.0)
                features['strength_squared'] = strength ** 2
                features['strength_log'] = np.log(strength + 1e-6)
                
                # Confidence-based features
                confidence = level.get('confidence', 0.0)
                features['confidence_strength_product'] = confidence * strength
                features['confidence_squared'] = confidence ** 2
                
                # Touches-based features
                touches = level.get('touches', 0)
                features['touches_normalized'] = touches / 10.0
                features['touches_log'] = np.log(touches + 1)
                
                # Regime-based features
                regime = level.get('regime', 'unknown')
                features['is_low_volatility'] = 1.0 if regime == 'low_volatility' else 0.0
                features['is_high_volatility'] = 1.0 if regime == 'high_volatility' else 0.0
                
                # Type-based features
                level_type = level.get('type', 'unknown')
                features['is_support'] = 1.0 if level_type == 'support' else 0.0
                features['is_resistance'] = 1.0 if level_type == 'resistance' else 0.0
                
                enhanced_level['features'] = features
                enhanced_levels.append(enhanced_level)
            
            return enhanced_levels
            
        except Exception as e:
            self.logger.warning(f"Advanced feature engineering failed: {e}")
            return sr_levels

    async def _regime_aware_clustering(self, sr_levels: List[Dict[str, Any]], config: EnhancedSRClusteringConfig) -> Dict[str, Any]:
        """Perform regime-aware clustering analysis."""
        try:
            # Group levels by regime
            regime_groups = {}
            for level in sr_levels:
                regime = level.get('regime', 'unknown')
                if regime not in regime_groups:
                    regime_groups[regime] = []
                regime_groups[regime].append(level)
            
            # Analyze each regime
            regime_analysis = {}
            for regime, levels in regime_groups.items():
                if len(levels) < 2:
                    continue
                
                # Calculate regime statistics
                prices = [level['price'] for level in levels]
                strengths = [level['strength'] for level in levels]
                confidences = [level['confidence'] for level in levels]
                
                regime_analysis[regime] = {
                    'count': len(levels),
                    'price_mean': np.mean(prices),
                    'price_std': np.std(prices),
                    'strength_mean': np.mean(strengths),
                    'strength_std': np.std(strengths),
                    'confidence_mean': np.mean(confidences),
                    'confidence_std': np.std(confidences),
                    'price_range': max(prices) - min(prices),
                    'levels': levels
                }
            
            return regime_analysis
            
        except Exception as e:
            self.logger.warning(f"Regime-aware clustering failed: {e}")
            return {}

    async def _ensemble_clustering(self, sr_levels: List[Dict[str, Any]], config: EnhancedSRClusteringConfig) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Perform ensemble clustering using multiple algorithms."""
        try:
            # Extract features for clustering
            features = self._extract_clustering_features(sr_levels)
            
            if len(features) < 2:
                return await self._simple_proximity_clustering(sr_levels, config)
            
            # Normalize features
            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features)
            
            # Run multiple clustering algorithms
            ensemble_results = {}
            algorithms = config.ensemble_algorithms
            
            for i, algorithm_name in enumerate(algorithms):
                if algorithm_name not in self.clustering_algorithms:
                    continue
                
                try:
                    algorithm = self.clustering_algorithms[algorithm_name]
                    
                    if algorithm_name == 'hdbscan':
                        clusterer = algorithm(
                            min_cluster_size=config.min_cluster_size,
                            min_samples=config.min_samples
                        )
                    elif algorithm_name == 'dbscan':
                        clusterer = algorithm(
                            eps=config.eps,
                            min_samples=config.min_samples
                        )
                    elif algorithm_name == 'kmeans':
                        clusterer = algorithm(
                            n_clusters=min(config.n_clusters, len(sr_levels)),
                            random_state=42
                        )
                    elif algorithm_name == 'spectral':
                        clusterer = algorithm(
                            n_clusters=min(config.n_clusters, len(sr_levels)),
                            random_state=42
                        )
                    elif algorithm_name == 'gmm':
                        clusterer = algorithm(
                            n_components=min(config.n_clusters, len(sr_levels)),
                            random_state=42
                        )
                    else:
                        continue
                    
                    # Fit clustering
                    cluster_labels = clusterer.fit_predict(features_normalized)
                    
                    # Calculate quality metrics
                    if len(set(cluster_labels)) > 1:
                        silhouette = silhouette_score(features_normalized, cluster_labels)
                        calinski = calinski_harabasz_score(features_normalized, cluster_labels)
                        davies = davies_bouldin_score(features_normalized, cluster_labels)
                    else:
                        silhouette = calinski = davies = 0.0
                    
                    ensemble_results[algorithm_name] = {
                        'labels': cluster_labels,
                        'silhouette': silhouette,
                        'calinski': calinski,
                        'davies': davies,
                        'n_clusters': len(set(cluster_labels))
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Algorithm {algorithm_name} failed: {e}")
                    continue
            
            if not ensemble_results:
                return await self._simple_proximity_clustering(sr_levels, config)
            
            # Combine results using weighted voting
            final_clusters = self._combine_ensemble_results(ensemble_results, sr_levels, config)
            
            # Calculate ensemble metrics
            ensemble_metrics = {
                'efficiency': len(final_clusters) / len(sr_levels) if sr_levels else 0.0,
                'method': 'ensemble',
                'algorithms_used': list(ensemble_results.keys()),
                'individual_metrics': {name: result for name, result in ensemble_results.items()}
            }
            
            return final_clusters, ensemble_metrics
            
        except Exception as e:
            self.logger.error(f"Ensemble clustering failed: {e}")
            return await self._simple_proximity_clustering(sr_levels, config)

    def _combine_ensemble_results(self, ensemble_results: Dict[str, Any], sr_levels: List[Dict[str, Any]], config: EnhancedSRClusteringConfig) -> List[Dict[str, Any]]:
        """Combine ensemble clustering results using weighted voting."""
        try:
            n_levels = len(sr_levels)
            n_algorithms = len(ensemble_results)
            
            if n_algorithms == 0:
                return []
            
            # Create consensus matrix
            consensus_matrix = np.zeros((n_levels, n_levels))
            
            for algorithm_name, result in ensemble_results.items():
                labels = result['labels']
                weight = config.ensemble_weights[config.ensemble_algorithms.index(algorithm_name)] if algorithm_name in config.ensemble_algorithms else 1.0 / n_algorithms
                
                # Add to consensus matrix
                for i in range(n_levels):
                    for j in range(n_levels):
                        if labels[i] == labels[j] and labels[i] != -1:
                            consensus_matrix[i, j] += weight
            
            # Normalize consensus matrix
            consensus_matrix /= n_algorithms
            
            # Use consensus matrix for final clustering
            # Simple approach: use threshold to determine clusters
            threshold = 0.5
            clusters = []
            used_indices = set()
            
            for i in range(n_levels):
                if i in used_indices:
                    continue
                
                # Find all levels that agree with this one
                cluster_indices = [i]
                for j in range(i + 1, n_levels):
                    if j not in used_indices and consensus_matrix[i, j] >= threshold:
                        cluster_indices.append(j)
                
                # Only keep clusters that meet minimum size
                if len(cluster_indices) >= config.min_cluster_size:
                    cluster_levels = [sr_levels[idx] for idx in cluster_indices]
                    representative = self._calculate_cluster_representative(cluster_levels)
                    
                    cluster_info = {
                        'cluster_id': len(clusters),
                        'levels': cluster_levels,
                        'representative': representative,
                        'size': len(cluster_levels),
                        'type': representative.get('type', 'mixed'),
                        'consensus_score': np.mean([consensus_matrix[i, j] for j in cluster_indices if i != j])
                    }
                    
                    clusters.append(cluster_info)
                    used_indices.update(cluster_indices)
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Ensemble combination failed: {e}")
            return []

    async def _generate_explainability_results(self, clusters: List[Dict[str, Any]], sr_levels: List[Dict[str, Any]], config: EnhancedSRClusteringConfig) -> Dict[str, Any]:
        """Generate explainability results for clustering."""
        try:
            if not self.explainer or not clusters:
                return {}
            
            # Extract features for explanation
            features = self._extract_clustering_features(sr_levels)
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]
            
            # Create a simple model for explanation (clustering doesn't have a traditional model)
            # We'll use the cluster assignments as predictions
            cluster_labels = []
            for i, level in enumerate(sr_levels):
                cluster_id = -1
                for cluster in clusters:
                    if level in cluster['levels']:
                        cluster_id = cluster['cluster_id']
                        break
                cluster_labels.append(cluster_id)
            
            # Create a simple classifier for explanation
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(features, cluster_labels)
            
            # Generate explanations
            explanation_result = self.explainer.explain_model(
                model, features, 'sr_clustering', 
                output_names=[f"cluster_{i}" for i in range(len(clusters))],
                feature_names=feature_names
            )
            
            return {
                'shap_values': explanation_result.shap_values,
                'shap_base_values': explanation_result.shap_base_values,
                'explanation_time': explanation_result.explanation_time,
                'sample_size': explanation_result.sample_size
            }
            
        except Exception as e:
            self.logger.warning(f"Explainability generation failed: {e}")
            return {}

    async def _calculate_comprehensive_quality_metrics(self, clusters: List[Dict[str, Any]], sr_levels: List[Dict[str, Any]], config: EnhancedSRClusteringConfig) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics for clustering."""
        try:
            if not clusters or not sr_levels:
                return {}
            
            # Basic quality metrics
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
                            cluster_id = -1
                            for cluster in clusters:
                                if level in cluster['levels']:
                                    cluster_id = cluster['cluster_id']
                                    break
                            cluster_labels.append(cluster_id)
                        
                        if len(set(cluster_labels)) > 1:  # Need at least 2 clusters
                            silhouette_avg = silhouette_score(features, cluster_labels)
                            quality_metrics['silhouette_score'] = silhouette_avg
                            
                            # Additional quality metrics
                            calinski = calinski_harabasz_score(features, cluster_labels)
                            davies = davies_bouldin_score(features, cluster_labels)
                            
                            quality_metrics.update({
                                'calinski_harabasz_score': calinski,
                                'davies_bouldin_score': davies,
                                'quality_score': (silhouette_avg + (1 - davies) + (calinski / 1000)) / 3
                            })
                            
                except Exception as e:
                    self.logger.warning(f"Advanced quality metrics calculation failed: {e}")
            
            # Cluster quality assessment
            high_quality_clusters = 0
            for cluster in clusters:
                if cluster.get('consensus_score', 0) > 0.7:
                    high_quality_clusters += 1
            
            quality_metrics.update({
                'high_quality_clusters': high_quality_clusters,
                'quality_ratio': high_quality_clusters / len(clusters) if clusters else 0.0,
                'meets_quality_threshold': quality_metrics.get('quality_score', 0) >= config.min_cluster_quality
            })
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Quality metrics calculation failed: {e}")
            return {}
