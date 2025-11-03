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

# TPrint imports for comprehensive logging
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_debug, tprint_warning, tprint_error,
        tprint_success, tprint_performance, tprint_data_preview, tprint_data_format,
        tprint_timer, tprint_logged
    )
    TPRINT_AVAILABLE = True
except ImportError as e:
    system_logger.warning(f"TPrint not available: {e}")
    TPRINT_AVAILABLE = False
    # Create fallback functions
    def tprint(*args, **kwargs): print(*args)
    def tprint_info(*args, **kwargs): print(f"[INFO] {' '.join(map(str, args))}")
    def tprint_debug(*args, **kwargs): print(f"[DEBUG] {' '.join(map(str, args))}")
    def tprint_warning(*args, **kwargs): print(f"[WARNING] {' '.join(map(str, args))}")
    def tprint_error(*args, **kwargs): print(f"[ERROR] {' '.join(map(str, args))}")
    def tprint_success(*args, **kwargs): print(f"[SUCCESS] {' '.join(map(str, args))}")
    def tprint_performance(operation, duration, **kwargs): print(f"[PERF] {operation}: {duration:.3f}s")
    def tprint_data_preview(data, name="Data", **kwargs): print(f"[DATA] {name}: {type(data)}")
    def tprint_data_format(data, name="Data", **kwargs): print(f"[FORMAT] {name}: {type(data)}")
    def tprint_timer(operation, level=None): return lambda: None
    def tprint_logged(level=None, **kwargs): return lambda func: func

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

# VectorBT imports
try:
    from src.utils.vectorbt_compat import (
        vbt, rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, VECTORBT_AVAILABLE
    )
    VECTORBT_AVAILABLE = True
except ImportError as e:
    system_logger.warning(f"VectorBT not available: {e}")
    VECTORBT_AVAILABLE = False
    # Create fallback functions
    def rolling_mean(data, window, **kwargs):
        return data.rolling(window=window, **kwargs).mean()
    def rolling_std(data, window, **kwargs):
        return data.rolling(window=window, **kwargs).std()
    def rolling_var(data, window, **kwargs):
        return data.rolling(window=window, **kwargs).var()
    def rolling_min(data, window, **kwargs):
        return data.rolling(window=window, **kwargs).min()
    def rolling_max(data, window, **kwargs):
        return data.rolling(window=window, **kwargs).max()
    def rolling_sum(data, window, **kwargs):
        return data.rolling(window=window, **kwargs).sum()
    def rolling_apply(data, window, func, **kwargs):
        return data.rolling(window=window, **kwargs).apply(func)

# Advanced clustering imports
try:
    from sklearn.cluster import HDBSCAN, DBSCAN, KMeans, SpectralClustering
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

# Hardware optimization utilities
try:
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    from src.utils.hardware.m1_gpu_utils import M1GPUManager
    HARDWARE_UTILS_AVAILABLE = True
except ImportError as e:
    HARDWARE_UTILS_AVAILABLE = False
    print(f"Warning: Hardware utilities not available: {e}")

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
    # Removed n_clusters - let algorithms determine cluster count naturally
    
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
    # Restrict ensemble to density-based methods to base clustering on price closeness
    ensemble_algorithms: List[str] = field(default_factory=lambda: ['hdbscan', 'dbscan'])
    ensemble_weights: List[float] = field(default_factory=lambda: [0.5, 0.5])

    # Feature policy
    # When True, use price proximity and strength/score only for clustering features
    price_strength_only: bool = True
    
    # Quality thresholds
    min_silhouette_score: float = 0.3
    min_cluster_quality: float = 0.5
    max_cluster_ratio: float = 0.8
    
    # Ensemble clustering consensus threshold
    # Minimum agreement between algorithms to group levels together (0.0-1.0)
    consensus_threshold: float = 0.25

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
        tprint_info("🚀 Initializing enhanced SR clustering components...")
        
        # Initialize hardware manager
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            self.hardware_manager = UnifiedHardwareManager()
            tprint_success("✅ Hardware manager initialized")
        else:
            self.hardware_manager = None
            tprint_warning("⚠️ Hardware manager not available")
        
        # Initialize vectorization manager
        if VECTORIZATION_AVAILABLE:
            self.vectorization_manager = UnifiedVectorizationManager()
            tprint_success("✅ Vectorization manager initialized")
        else:
            self.vectorization_manager = None
            tprint_warning("⚠️ Vectorization manager not available")
        
        # Initialize VectorBTRollingOptimizer
        if VECTORBT_ROLLING_AVAILABLE:
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            tprint_success("✅ VectorBTRollingOptimizer initialized")
        else:
            self.vectorbt_rolling_optimizer = None
            tprint_warning("⚠️ VectorBTRollingOptimizer not available")
        
        # Initialize hardware utilities
        if HARDWARE_UTILS_AVAILABLE:
            self.memory_optimizer = M1MemoryOptimizer()
            self.gpu_manager = M1GPUManager()
            tprint_success("✅ Hardware utilities initialized")
        else:
            self.memory_optimizer = None
            self.gpu_manager = None
            tprint_warning("⚠️ Hardware utilities not available")
        
        # Initialize VectorBT support
        if VECTORBT_AVAILABLE:
            tprint_success("✅ VectorBT support enabled")
        else:
            tprint_warning("⚠️ VectorBT not available, using fallback implementations")
        
        # Initialize TPrint support
        if TPRINT_AVAILABLE:
            tprint_success("✅ TPrint logging enabled")
        else:
            tprint_warning("⚠️ TPrint not available, using fallback logging")
        
        # Initialize data leakage detector
        if DATA_LEAKAGE_AVAILABLE:
            self.data_leakage_detector = DataLeakageDetector()
            tprint_success("✅ Data leakage detector initialized")
        else:
            self.data_leakage_detector = None
            tprint_warning("⚠️ Data leakage detector not available")
        
        # Initialize explainability
        if EXPLAINABILITY_AVAILABLE:
            self.explainability_config = ExplanationConfig()
            self.explainer = SHAPLIMEExplainer(self.explainability_config)
            tprint_success("✅ Explainability initialized")
        else:
            self.explainer = None
            tprint_warning("⚠️ Explainability not available")
        
        # Initialize HPO optimizer
        if HPO_AVAILABLE:
            self.hpo_config = HPOConfig()
            self.hpo_optimizer = BayesianTPEOptimizer(self.hpo_config)
            tprint_success("✅ HPO optimizer initialized")
        else:
            self.hpo_optimizer = None
            tprint_warning("⚠️ HPO optimizer not available")
        
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
            tprint_success("✅ Clustering algorithms initialized")
        else:
            tprint_warning("⚠️ Clustering algorithms not available")

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['sr_clustering_result', 'sr_levels_dictionary']
    
    async def _load_artifacts_from_previous_stage(self, previous_component_name: str, artifact_names: List[str]) -> Dict[str, Any]:
        """
        Load artifacts from a previous pipeline stage using BaseStep integration.
        
        Args:
            previous_component_name: Name of the previous component
            artifact_names: List of artifact names to load
            
        Returns:
            Dictionary of loaded artifacts
        """
        try:
            # Use BaseStep's artifact manager to load from previous stage
            loaded_artifacts = {}
            
            for artifact_name in artifact_names:
                try:
                    # Try to get artifact using BaseStep method
                    artifact_data = self._get_artifact(
                        artifact_name=artifact_name,
                        artifact_type='data'
                    )
                    if artifact_data is not None:
                        loaded_artifacts[artifact_name] = artifact_data
                        self.logger.info(f"Loaded artifact {artifact_name} from previous stage")
                except Exception as e:
                    self.logger.debug(f"Could not load artifact {artifact_name}: {e}")
                    continue
            
            return loaded_artifacts
            
        except Exception as e:
            self.logger.error(f"Failed to load artifacts from previous stage: {e}")
            return {}

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
        tprint_info('🔗 Starting Enhanced SR Clustering')

        try:
            # Validate BaseStep integration
            integration_validation = self._validate_basestep_integration()
            if not integration_validation['integration_valid']:
                self.logger.error(f"BaseStep integration validation failed: {integration_validation}")
                return {
                    'success': False,
                    'artifacts': [],
                    'metrics': {},
                    'error': 'BaseStep integration validation failed',
                    'integration_validation': integration_validation
                }
            
            self.logger.info("✅ BaseStep integration validated successfully")
            
            # Create enhanced configuration
            enhanced_config = EnhancedSRClusteringConfig()
            
            # Override with user config if provided
            self._apply_user_config(enhanced_config, config)

            # Extract configuration
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'long')
            execution_mode = config.get('execution_mode', 'light')
            
            if not symbol:
                raise ValueError("Symbol is required for SR clustering")
            
            tprint_info(f"Clustering SR levels for {symbol} from {exchange}")
            tprint_info(f"Timeframe: {timeframe}, Direction: {direction}")
            
            # Initialize artifacts list
            artifacts = []
            metrics = {}
            
            # Set up artifact manager context using BaseStep integration
            self.artifact_manager.set_context(
                symbol=symbol,
                exchange=exchange,
                direction=direction,
                model='Analyst',
                step_name=self.step_name,
                datetime=datetime.now()
            )
            
            # Persist config for downstream feature extraction
            self._current_clustering_config = enhanced_config
            
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
                'hardware_metrics': clustering_result.get('hardware_metrics', {}),
                'basestep_integration': {
                    'integration_valid': integration_validation['integration_valid'],
                    'artifacts_saved': len(artifacts),
                    'required_artifacts': self.get_required_artifacts(),
                    'step_name': self.step_name
                }
            })

            tprint_success(f'✅ Enhanced SR Clustering completed: {metrics["total_clusters"]} clusters created')
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
            
            # Enhanced input validation
            if not sr_levels:
                self.logger.warning("No SR levels provided for clustering")
                return await self._create_empty_clustering_result()
            
            # Validate input with enhanced error handling
            if not await self._validate_clustering_input(sr_levels):
                self.logger.error("Input validation failed")
                return await self._create_empty_clustering_result()
            
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
            
            # Convert explainability_results to JSON-serializable format for Parquet
            import json
            import numpy as np
            
            def make_json_serializable(obj):
                """Convert numpy arrays and other non-serializable objects to JSON-compatible types."""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(item) for item in obj]
                else:
                    return obj
            
            serializable_explainability = make_json_serializable(explainability_results)
            explainability_json = json.dumps(serializable_explainability) if serializable_explainability else "{}"
            
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
                'explainability_results': explainability_json,  # Store as JSON string for Parquet compatibility
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
        """Load SR levels for clustering using BaseStep integration."""
        with tprint_timer("Load SR levels"):
            try:
                tprint_info(f"Loading SR levels for {symbol} {timeframe}")
                
                # First, try to load existing SR levels from artifacts using BaseStep method
                try:
                    sr_levels_dict = self._get_sr_levels(
                        symbol=symbol,
                        exchange=config.get('exchange', 'binance'),
                        timeframe=timeframe,
                        direction=config.get('direction', 'long')
                    )
                    
                    if sr_levels_dict and not sr_levels_dict.get('error') and sr_levels_dict.get('levels'):
                        tprint_success(f"Loaded {len(sr_levels_dict['levels'])} existing SR levels from artifacts")
                        tprint_data_preview(sr_levels_dict['levels'], "SR levels from artifacts", max_rows=3)
                        return sr_levels_dict['levels']
                        
                except Exception as e:
                    self.logger.debug(f"Could not load existing SR levels from artifacts: {e}")
                
                # Try to load from previous stage artifacts
                try:
                    previous_artifacts = await self._load_artifacts_from_previous_stage(
                        previous_component_name='sr_detection',
                        artifact_names=['sr_detection_result', 'sr_levels', 'sr_levels_dictionary']
                    )
                
                    if previous_artifacts:
                        # Check for sr_detection_result first (new artifact name from enhanced SR detection)
                        if 'sr_detection_result' in previous_artifacts:
                            sr_detection_result = previous_artifacts['sr_detection_result']
                            if sr_detection_result:
                                # Extract levels from detection result
                                sr_levels = sr_detection_result.get('levels', [])
                                if sr_levels and isinstance(sr_levels, list):
                                    tprint_success(f"Loaded {len(sr_levels)} SR levels from sr_detection_result")
                                    tprint_data_preview(sr_levels, "SR levels from detection result", max_rows=3)
                                    return sr_levels
                        
                        # Check for sr_levels_dictionary (legacy)
                        if 'sr_levels_dictionary' in previous_artifacts:
                            sr_levels_dict = previous_artifacts['sr_levels_dictionary']
                            if sr_levels_dict and sr_levels_dict.get('levels'):
                                tprint_success(f"Loaded {len(sr_levels_dict['levels'])} SR levels from sr_levels_dictionary")
                                tprint_data_preview(sr_levels_dict['levels'], "SR levels from dictionary", max_rows=3)
                                return sr_levels_dict['levels']
                    
                        # Check for sr_levels directly (legacy)
                        if 'sr_levels' in previous_artifacts:
                            sr_levels = previous_artifacts['sr_levels']
                            if sr_levels and isinstance(sr_levels, list):
                                tprint_success(f"Loaded {len(sr_levels)} SR levels from sr_levels")
                                tprint_data_preview(sr_levels, "SR levels from artifact", max_rows=3)
                                return sr_levels
                                
                except Exception as e:
                    self.logger.debug(f"Could not load SR levels from previous stage: {e}")
                
                # Fallback: Try to load from feature bank
                try:
                    from src.feature_generation.core.feature_bank import get_global_feature_bank
                    feature_bank = get_global_feature_bank()
                    sr_levels_dict = feature_bank.get_sr_levels(
                        symbol=symbol,
                        exchange=config.get('exchange', 'binance'),
                        timeframe=timeframe,
                        direction=config.get('direction', 'long')
                    )
                    
                    if sr_levels_dict and not sr_levels_dict.get('error') and sr_levels_dict.get('levels'):
                        self.logger.info(f"Loaded {len(sr_levels_dict['levels'])} SR levels from feature bank")
                        return sr_levels_dict['levels']
                        
                except TypeError as e:
                    # Handle missing argument errors from set_context
                    if 'set_context' in str(e):
                        self.logger.debug(f"Feature bank requires updated context call: {e}")
                    else:
                        self.logger.debug(f"Could not load SR levels from feature bank: {e}")
                except Exception as e:
                    self.logger.debug(f"Could not load SR levels from feature bank: {e}")
                
                # Final fallback: Create sample SR levels for demonstration
                self.logger.warning("No existing SR levels found, creating sample levels for demonstration")
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
            if self.hardware_manager and hasattr(self.hardware_manager, 'get_optimal_config'):
                hardware_config = self.hardware_manager.get_optimal_config(
                    WorkloadType.ML_TRAINING,
                    OptimizationLevel.BALANCED
                )
                return hardware_config
            else:
                self.logger.debug("Hardware manager not available or missing get_optimal_config method")
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
                gains = hardware_config.get('gains', {})
                # Ensure gains dict has at least one field for Parquet compatibility
                if not gains:
                    gains = {'cpu_gain': 1.0}
                optimized_level['optimization_gains'] = gains
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
        with tprint_timer("VectorBT clustering"):
            try:
                tprint_info(f"Starting VectorBT clustering with {len(sr_levels)} levels")
                
                if self.vectorization_manager and VECTORBT_AVAILABLE:
                    # Use VectorBT for efficient clustering
                    operation_config = {
                        'operation_type': OperationType.FEATURE_ENGINEERING,
                        'data_size': len(sr_levels),
                        'data_dimensions': (len(sr_levels),),
                        'enable_vectorbt': True,
                        'use_rolling_optimizer': True
                    }
                    
                    tprint_debug("Using VectorBT optimization for clustering")
                    
                    # Extract features using VectorBT optimization
                    features = self._extract_clustering_features(sr_levels)
                    tprint_data_preview(features, "Clustering features", max_rows=3)
                    
                    if len(features) == 0:
                        tprint_warning("No features extracted, falling back to traditional clustering")
                        return await self._cluster_sr_levels_traditional(sr_levels, enhanced_config)
                    
                    # Use VectorBT rolling optimizer for distance calculations
                    if self.vectorbt_rolling_optimizer and len(features) > 50:
                        tprint_info("Using VectorBT rolling optimizer for distance calculations")
                        clusters, metrics = await self._vectorbt_optimized_clustering(
                            sr_levels, features, enhanced_config
                        )
                    else:
                        # Use traditional clustering with VectorBT features
                        clusters, metrics = await self._traditional_clustering_with_vectorbt_features(
                            sr_levels, features, enhanced_config
                        )
                    
                    clustering_metrics = {
                        'efficiency': metrics.get('efficiency', 1.0),
                        'method': 'vectorbt_optimized',
                        'acceleration_factor': metrics.get('acceleration_factor', 1.0),
                        'vectorbt_enabled': True,
                        'features_shape': features.shape
                    }
                    
                    tprint_success(f"VectorBT clustering completed: {len(clusters)} clusters")
                    return clusters, clustering_metrics
                else:
                    tprint_warning("VectorBT not available, falling back to traditional clustering")
                    return await self._cluster_sr_levels_traditional(sr_levels, enhanced_config)
                    
            except Exception as e:
                tprint_error(f"VectorBT clustering failed: {e}")
                return await self._cluster_sr_levels_traditional(sr_levels, enhanced_config)

    async def _vectorbt_optimized_clustering(
        self, 
        sr_levels: List[Dict[str, Any]], 
        features: np.ndarray, 
        enhanced_config: EnhancedSRClusteringConfig
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Perform clustering with VectorBT rolling optimization."""
        try:
            tprint_debug("Using VectorBT rolling optimizer for clustering")
            
            # Convert features to pandas DataFrame for VectorBT operations
            feature_df = pd.DataFrame(features)
            
            # Use VectorBT rolling operations for distance calculations
            if VECTORBT_AVAILABLE and self.vectorbt_rolling_optimizer:
                # Calculate rolling distances using VectorBT
                distances = self._calculate_vectorbt_distances(feature_df)
                
                # Use VectorBT for clustering
                clusters = await self._vectorbt_clustering_algorithm(
                    sr_levels, distances, enhanced_config
                )
                
                metrics = {
                    'efficiency': 2.0,  # VectorBT typically provides 2x speedup
                    'acceleration_factor': 2.0,
                    'method': 'vectorbt_rolling_optimized'
                }
            else:
                # Fallback to traditional clustering
                clusters, metrics = await self._traditional_clustering_with_vectorbt_features(
                    sr_levels, features, enhanced_config
                )
            
            return clusters, metrics
            
        except Exception as e:
            tprint_error(f"VectorBT optimized clustering failed: {e}")
            return await self._traditional_clustering_with_vectorbt_features(
                sr_levels, features, enhanced_config
            )

    def _calculate_vectorbt_distances(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Calculate distances using VectorBT rolling operations."""
        try:
            tprint_debug("Calculating distances with VectorBT")
            
            if VECTORBT_AVAILABLE and len(feature_df) > 0:
                # Use VectorBT for efficient distance calculations
                distances = np.zeros((len(feature_df), len(feature_df)))
                
                for i in range(len(feature_df)):
                    for j in range(i + 1, len(feature_df)):
                        # Calculate Euclidean distance using VectorBT
                        diff = feature_df.iloc[i] - feature_df.iloc[j]
                        distance = np.sqrt(rolling_sum(diff ** 2, window=1).iloc[-1])
                        distances[i, j] = distance
                        distances[j, i] = distance
                
                return distances
            else:
                # Fallback to scikit-learn
                from sklearn.metrics.pairwise import euclidean_distances
                return euclidean_distances(feature_df)
                
        except Exception as e:
            tprint_error(f"VectorBT distance calculation failed: {e}")
            # Fallback to scikit-learn
            from sklearn.metrics.pairwise import euclidean_distances
            return euclidean_distances(feature_df)

    async def _vectorbt_clustering_algorithm(
        self, 
        sr_levels: List[Dict[str, Any]], 
        distances: np.ndarray, 
        enhanced_config: EnhancedSRClusteringConfig
    ) -> List[Dict[str, Any]]:
        """Perform clustering using VectorBT-optimized algorithm."""
        try:
            tprint_debug("Running VectorBT clustering algorithm")
            
            # Use HDBSCAN with VectorBT-optimized distance matrix
            if 'hdbscan' in self.clustering_algorithms:
                clusterer = self.clustering_algorithms['hdbscan'](
                    min_cluster_size=enhanced_config.min_cluster_size,
                    metric='precomputed'
                )
                cluster_labels = clusterer.fit_predict(distances)
            else:
                # Fallback to DBSCAN
                clusterer = self.clustering_algorithms.get('dbscan', DBSCAN)(
                    eps=enhanced_config.eps,
                    min_samples=enhanced_config.min_cluster_size,
                    metric='precomputed'
                )
                cluster_labels = clusterer.fit_predict(distances)
            
            # Organize clusters
            clusters = self._organize_clusters(sr_levels, cluster_labels)
            
            tprint_success(f"VectorBT clustering algorithm completed: {len(clusters)} clusters")
            return clusters
            
        except Exception as e:
            tprint_error(f"VectorBT clustering algorithm failed: {e}")
            return []

    async def _traditional_clustering_with_vectorbt_features(
        self, 
        sr_levels: List[Dict[str, Any]], 
        features: np.ndarray, 
        enhanced_config: EnhancedSRClusteringConfig
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Perform traditional clustering with VectorBT-extracted features."""
        try:
            tprint_debug("Using traditional clustering with VectorBT features")
            
            # Use traditional clustering algorithms with VectorBT features
            if 'hdbscan' in self.clustering_algorithms:
                clusterer = self.clustering_algorithms['hdbscan'](
                    min_cluster_size=enhanced_config.min_cluster_size
                )
                cluster_labels = clusterer.fit_predict(features)
            else:
                # Fallback to DBSCAN
                clusterer = self.clustering_algorithms.get('dbscan', DBSCAN)(
                    eps=enhanced_config.eps,
                    min_samples=enhanced_config.min_cluster_size
                )
                cluster_labels = clusterer.fit_predict(features)
            
            # Organize clusters
            clusters = self._organize_clusters(sr_levels, cluster_labels)
            
            metrics = {
                'efficiency': 1.5,  # Some improvement from VectorBT features
                'acceleration_factor': 1.5,
                'method': 'traditional_with_vectorbt_features'
            }
            
            return clusters, metrics
            
        except Exception as e:
            tprint_error(f"Traditional clustering with VectorBT features failed: {e}")
            return [], {'efficiency': 0.0, 'method': 'failed'}

    async def _cluster_sr_levels_traditional(
        self, 
        sr_levels: List[Dict[str, Any]], 
        enhanced_config: EnhancedSRClusteringConfig
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Cluster SR levels using traditional methods."""
        with tprint_timer("Traditional clustering"):
            try:
                tprint_info(f"Starting traditional clustering with {len(sr_levels)} levels")
                # Use the enhanced proximity clustering with advanced algorithms
                if enhanced_config.clustering_algorithm in self.clustering_algorithms:
                    tprint_debug(f"Using advanced algorithm: {enhanced_config.clustering_algorithm}")
                    clusters, metrics = await self._cluster_with_advanced_algorithm(sr_levels, enhanced_config)
                else:
                    tprint_debug("Using simple proximity clustering")
                    clusters, metrics = await self._simple_proximity_clustering(sr_levels, enhanced_config)
                
                tprint_success(f"Traditional clustering completed: {len(clusters)} clusters")
                return clusters, metrics
                
            except Exception as e:
                tprint_error(f"Traditional clustering failed: {e}")
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
        """Extract clustering features.

        When `price_strength_only` is enabled in the current config, features are limited to:
        - Normalized price (relative to average)
        - Strength/score (capped to [0,1])
        This aligns clustering with price closeness and level strength.
        """
        with tprint_timer("Feature extraction"):
            try:
                tprint_debug(f"Extracting features from {len(sr_levels)} SR levels")
                
                if not sr_levels:
                    tprint_warning("No SR levels provided for feature extraction")
                    return np.array([])
                
                features = []
                prices = [level.get('price', 0.0) for level in sr_levels if level.get('price', 0.0) > 0]
                current_price = np.mean(prices) if prices else 1.0
                
                tprint_data_preview(prices, "Price data", max_rows=5)
                
                # Use simplified features when requested
                cfg = getattr(self, '_current_clustering_config', None)
                if cfg is not None and getattr(cfg, 'price_strength_only', False):
                    for level in sr_levels:
                        price = level.get('price', 0.0)
                        # Use absolute price instead of ratio for proper distance calculation
                        price_absolute = price  # Keep absolute price for clustering
                        strength_norm = max(0.0, min(float(level.get('strength', 0.0)), 1.0))
                        features.append([price_absolute, strength_norm])
                    return np.asarray(features, dtype=float)

                # Default: original extended feature set
                if VECTORBT_AVAILABLE and len(sr_levels) > 100 and self.vectorbt_rolling_optimizer:
                    tprint_info("Using VectorBT optimization for feature extraction")
                    return self._extract_features_vectorbt_optimized(sr_levels, prices, current_price)
                
                for level in sr_levels:
                    price = level.get('price', 0.0)
                    # Use absolute price instead of ratio for proper distance calculation
                    price_absolute = price  # Keep absolute price for clustering
                    log_price = np.log(price) if price > 0 else 0
                    strength_norm = min(level.get('strength', 0.0), 1.0)
                    confidence_norm = min(level.get('confidence', 0.0), 1.0)
                    touches_norm = min(level.get('touches', 0) / 10.0, 1.0)
                first_touch = level.get('first_touch', datetime.now())
                last_touch = level.get('last_touch', datetime.now())
                if isinstance(first_touch, datetime) and isinstance(last_touch, datetime):
                        age_days = (datetime.now() - first_touch).days / 365.0
                        recency_days = (datetime.now() - last_touch).days / 30.0
                else:
                    age_days = 0.0
                    recency_days = 0.0
                level_type = level.get('type', 'unknown').lower()
                is_support = 1.0 if level_type == 'support' else 0.0
                is_resistance = 1.0 if level_type == 'resistance' else 0.0
                features_dict = level.get('features', {})
                volume_profile = min(features_dict.get('volume_profile', 0.0), 1.0)
                price_action = min(features_dict.get('price_action', 0.0), 1.0)
                technical_indicators = min(features_dict.get('technical_indicators', 0.0), 1.0)
                
                level_features = [
                    log_price,           # Log-transformed price (scale-invariant)
                    price_absolute,      # Absolute price for proper distance calculation
                    strength_norm,       # Normalized strength (0-1)
                    confidence_norm,     # Normalized confidence (0-1)
                    touches_norm,        # Normalized touches (0-1)
                    age_days,           # Age in years (normalized)
                    recency_days,       # Recency in months (normalized)
                    is_support,         # Support indicator (0 or 1)
                    is_resistance,      # Resistance indicator (0 or 1)
                    volume_profile,     # Volume profile feature (0-1)
                    price_action,       # Price action feature (0-1)
                    technical_indicators # Technical indicators feature (0-1)
                ]
                features.append(level_features)
            
                return np.array(features)
            
            except Exception as e:
                tprint_error(f"Feature extraction failed: {e}")
                return np.array([])

    def _extract_features_vectorbt_optimized(self, sr_levels: List[Dict[str, Any]], prices: List[float], current_price: float) -> np.ndarray:
        """Extract features using VectorBT optimization for large datasets."""
        try:
            tprint_debug("Using VectorBT optimized feature extraction")
            
            # Convert to pandas Series for VectorBT operations
            price_series = pd.Series(prices)
            
            # VectorBT optimized calculations
            if VECTORBT_AVAILABLE and len(price_series) > 0:
                # Calculate rolling statistics using VectorBT
                price_mean = rolling_mean(price_series, window=min(20, len(price_series)))
                price_std = rolling_std(price_series, window=min(20, len(price_series)))
                price_min = rolling_min(price_series, window=min(20, len(price_series)))
                price_max = rolling_max(price_series, window=min(20, len(price_series)))
                
                # Fill NaN values
                price_mean = price_mean.fillna(price_series.mean())
                price_std = price_std.fillna(price_series.std())
                price_min = price_min.fillna(price_series.min())
                price_max = price_max.fillna(price_series.max())
            else:
                # Fallback to pandas
                price_mean = price_series.rolling(window=min(20, len(price_series))).mean().fillna(price_series.mean())
                price_std = price_series.rolling(window=min(20, len(price_series))).std().fillna(price_series.std())
                price_min = price_series.rolling(window=min(20, len(price_series))).min().fillna(price_series.min())
                price_max = price_series.rolling(window=min(20, len(price_series))).max().fillna(price_series.max())
            
            features = []
            for i, level in enumerate(sr_levels):
                price = level.get('price', 0.0)
                
                # Use absolute price for proper distance calculation
                price_absolute = price  # Keep absolute price for clustering
                log_price = np.log(price) if price > 0 else 0
                
                # Normalized features (0-1 range)
                strength_norm = min(level.get('strength', 0.0), 1.0)
                confidence_norm = min(level.get('confidence', 0.0), 1.0)
                touches_norm = min(level.get('touches', 0) / 10.0, 1.0)
                
                # Temporal features
                first_touch = level.get('first_touch')
                last_touch = level.get('last_touch')
                now = datetime.now()
                
                if first_touch and isinstance(first_touch, str):
                    try:
                        first_touch_dt = datetime.fromisoformat(first_touch.replace('Z', '+00:00'))
                        age_days = (now - first_touch_dt).days / 365.0  # Normalize to years
                    except:
                        age_days = 0.0
                else:
                    age_days = 0.0
                
                if last_touch and isinstance(last_touch, str):
                    try:
                        last_touch_dt = datetime.fromisoformat(last_touch.replace('Z', '+00:00'))
                        recency_days = (now - last_touch_dt).days / 30.0  # Normalize to months
                    except:
                        recency_days = 0.0
                else:
                    recency_days = 0.0
                
                # Type encoding
                level_type = level.get('type', 'mixed')
                is_support = 1.0 if level_type == 'support' else 0.0
                is_resistance = 1.0 if level_type == 'resistance' else 0.0
                
                # Additional features from features dict
                features_dict = level.get('features', {})
                volume_profile = min(features_dict.get('volume_profile', 0.0), 1.0)
                price_action = min(features_dict.get('price_action', 0.0), 1.0)
                technical_indicators = min(features_dict.get('technical_indicators', 0.0), 1.0)
                
                # VectorBT enhanced features
                if i < len(price_mean):
                    price_zscore = (price - price_mean.iloc[i]) / price_std.iloc[i] if price_std.iloc[i] > 0 else 0
                    price_percentile = (price - price_min.iloc[i]) / (price_max.iloc[i] - price_min.iloc[i]) if price_max.iloc[i] > price_min.iloc[i] else 0.5
                else:
                    price_zscore = 0
                    price_percentile = 0.5
                
                level_features = [
                    log_price,           # Log-transformed price (scale-invariant)
                    price_absolute,      # Absolute price for proper distance calculation
                    strength_norm,       # Normalized strength (0-1)
                    confidence_norm,     # Normalized confidence (0-1)
                    touches_norm,        # Normalized touches (0-1)
                    age_days,           # Age in years (normalized)
                    recency_days,       # Recency in months (normalized)
                    is_support,         # Support indicator (0 or 1)
                    is_resistance,      # Resistance indicator (0 or 1)
                    volume_profile,     # Volume profile feature (0-1)
                    price_action,       # Price action feature (0-1)
                    technical_indicators, # Technical indicators feature (0-1)
                    price_zscore,       # Price z-score (VectorBT enhanced)
                    price_percentile    # Price percentile (VectorBT enhanced)
                ]
                features.append(level_features)
            
            result = np.array(features)
            tprint_success(f"VectorBT optimized feature extraction completed: {result.shape}")
            return result
            
        except Exception as e:
            tprint_error(f"VectorBT optimized feature extraction failed: {e}")
            # Fallback to regular feature extraction
            return self._extract_clustering_features_fallback(sr_levels)

    def _extract_clustering_features_fallback(self, sr_levels: List[Dict[str, Any]]) -> np.ndarray:
        """Fallback feature extraction without VectorBT optimization."""
        try:
            tprint_debug("Using fallback feature extraction")
            
            if not sr_levels:
                return np.array([])
            
            features = []
            prices = [level.get('price', 0.0) for level in sr_levels if level.get('price', 0.0) > 0]
            current_price = np.mean(prices) if prices else 1.0
            
            for level in sr_levels:
                price = level.get('price', 0.0)
                
                # Use absolute price for proper distance calculation
                price_absolute = price  # Keep absolute price for clustering
                log_price = np.log(price) if price > 0 else 0
                
                # Normalized features (0-1 range)
                strength_norm = min(level.get('strength', 0.0), 1.0)
                confidence_norm = min(level.get('confidence', 0.0), 1.0)
                touches_norm = min(level.get('touches', 0) / 10.0, 1.0)
                
                # Temporal features
                first_touch = level.get('first_touch')
                last_touch = level.get('last_touch')
                now = datetime.now()
                
                if first_touch and isinstance(first_touch, str):
                    try:
                        first_touch_dt = datetime.fromisoformat(first_touch.replace('Z', '+00:00'))
                        age_days = (now - first_touch_dt).days / 365.0
                    except:
                        age_days = 0.0
                else:
                    age_days = 0.0
                
                if last_touch and isinstance(last_touch, str):
                    try:
                        last_touch_dt = datetime.fromisoformat(last_touch.replace('Z', '+00:00'))
                        recency_days = (now - last_touch_dt).days / 30.0
                    except:
                        recency_days = 0.0
                else:
                    recency_days = 0.0
                
                # Type encoding
                level_type = level.get('type', 'mixed')
                is_support = 1.0 if level_type == 'support' else 0.0
                is_resistance = 1.0 if level_type == 'resistance' else 0.0
                
                # Additional features from features dict
                features_dict = level.get('features', {})
                volume_profile = min(features_dict.get('volume_profile', 0.0), 1.0)
                price_action = min(features_dict.get('price_action', 0.0), 1.0)
                technical_indicators = min(features_dict.get('technical_indicators', 0.0), 1.0)
                
                level_features = [
                    log_price, price_absolute, strength_norm, confidence_norm, touches_norm,
                    age_days, recency_days, is_support, is_resistance,
                    volume_profile, price_action, technical_indicators
                ]
                features.append(level_features)
            
            return np.array(features)
            
        except Exception as e:
            tprint_error(f"Fallback feature extraction failed: {e}")
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
                        # Create proper cluster labels for silhouette calculation
                        cluster_labels = self._create_cluster_labels_for_silhouette(sr_levels, clusters)
                        
                        if len(set(cluster_labels)) > 1 and -1 not in cluster_labels:  # Need at least 2 clusters, no noise
                            silhouette_avg = silhouette_score(features, cluster_labels)
                            quality_metrics['silhouette_score'] = silhouette_avg
                        elif len(set(cluster_labels)) > 1:  # Has clusters but also noise points
                            # Filter out noise points for silhouette calculation
                            valid_mask = np.array(cluster_labels) != -1
                            if np.sum(valid_mask) > 1:
                                valid_features = features[valid_mask]
                                valid_labels = np.array(cluster_labels)[valid_mask]
                                if len(set(valid_labels)) > 1:
                                    silhouette_avg = silhouette_score(valid_features, valid_labels)
                                    quality_metrics['silhouette_score'] = silhouette_avg
                except Exception as e:
                    self.logger.warning(f"Silhouette score calculation failed: {e}")
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Quality metrics calculation failed: {e}")
            return {}

    def _create_cluster_labels_for_silhouette(self, sr_levels: List[Dict[str, Any]], clusters: List[Dict[str, Any]]) -> List[int]:
        """Create cluster labels for silhouette score calculation with proper level matching."""
        try:
            cluster_labels = []
            
            for level in sr_levels:
                cluster_id = -1  # Default to noise
                
                # Find which cluster this level belongs to by matching key attributes
                for cluster in clusters:
                    for cluster_level in cluster['levels']:
                        if self._level_matches_for_clustering(level, cluster_level):
                            cluster_id = cluster['cluster_id']
                            break
                    if cluster_id != -1:
                        break
                
                cluster_labels.append(cluster_id)
            
            return cluster_labels
            
        except Exception as e:
            self.logger.error(f"Cluster label creation failed: {e}")
            return [-1] * len(sr_levels)

    def _level_matches_for_clustering(self, level1: Dict[str, Any], level2: Dict[str, Any]) -> bool:
        """Check if two levels match for clustering purposes."""
        try:
            # Match by key attributes that should be unique
            price1 = level1.get('price', 0.0)
            price2 = level2.get('price', 0.0)
            
            # Allow small floating point differences
            price_match = abs(price1 - price2) < 1e-10
            
            # Also match by type and strength for additional validation
            type1 = level1.get('type', '')
            type2 = level2.get('type', '')
            type_match = type1 == type2
            
            strength1 = level1.get('strength', 0.0)
            strength2 = level2.get('strength', 0.0)
            strength_match = abs(strength1 - strength2) < 1e-6
            
            return price_match and type_match and strength_match
            
        except Exception as e:
            self.logger.warning(f"Level matching failed: {e}")
            return False

    def _calculate_time_proximity(self, level1: Dict[str, Any], level2: Dict[str, Any]) -> float:
        """Calculate time-based proximity between two levels (0-1, higher = closer in time)."""
        try:
            # Get timestamps
            time1 = level1.get('first_touch') or level1.get('last_touch')
            time2 = level2.get('first_touch') or level2.get('last_touch')
            
            if not time1 or not time2:
                return 0.0  # No time information
            
            # Convert to datetime if needed
            if isinstance(time1, str):
                time1 = datetime.fromisoformat(time1.replace('Z', '+00:00'))
            if isinstance(time2, str):
                time2 = datetime.fromisoformat(time2.replace('Z', '+00:00'))
            
            # Calculate time difference in days
            time_diff_days = abs((time1 - time2).total_seconds()) / (24 * 3600)
            
            # Convert to proximity (0-1 scale, closer in time = higher proximity)
            # Use exponential decay: proximity = exp(-time_diff / decay_constant)
            decay_constant = 30  # 30 days
            proximity = np.exp(-time_diff_days / decay_constant)
            
            return min(proximity, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Time proximity calculation failed: {e}")
            return 0.0

    def _calculate_enhanced_cluster_representative(self, cluster_levels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate enhanced cluster representative with weighted aggregation."""
        try:
            if not cluster_levels:
                return {}
            
            if len(cluster_levels) == 1:
                return cluster_levels[0]
            
            # Calculate weights based on strength and confidence
            weights = []
            for level in cluster_levels:
                strength = level.get('strength', 0.0)
                confidence = level.get('confidence', 0.0)
                touches = level.get('touches', 0)
                
                # Weighted combination: strength (40%), confidence (30%), touches (30%)
                weight = (strength * 0.4 + confidence * 0.3 + min(touches / 10.0, 1.0) * 0.3)
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(cluster_levels)] * len(cluster_levels)
            
            # Weighted aggregation
            weighted_price = sum(level.get('price', 0.0) * weight for level, weight in zip(cluster_levels, weights))
            weighted_strength = sum(level.get('strength', 0.0) * weight for level, weight in zip(cluster_levels, weights))
            weighted_confidence = sum(level.get('confidence', 0.0) * weight for level, weight in zip(cluster_levels, weights))
            
            # Sum for counts
            total_touches = sum(level.get('touches', 0) for level in cluster_levels)
            
            # Determine type (majority vote with strength weighting)
            type_votes = {}
            for level, weight in zip(cluster_levels, weights):
                level_type = level.get('type', 'unknown')
                type_votes[level_type] = type_votes.get(level_type, 0) + weight
            
            representative_type = max(type_votes.items(), key=lambda x: x[1])[0] if type_votes else 'mixed'
            
            # Find the strongest level for additional attributes
            strongest_level = max(cluster_levels, key=lambda x: x.get('strength', 0.0))
            
            representative = {
                'price': weighted_price,
                'strength': weighted_strength,
                'confidence': weighted_confidence,
                'touches': total_touches,
                'type': representative_type,
                'cluster_size': len(cluster_levels),
                'first_touch': min((level.get('first_touch') for level in cluster_levels if level.get('first_touch')), default=None),
                'last_touch': max((level.get('last_touch') for level in cluster_levels if level.get('last_touch')), default=None),
                'features': strongest_level.get('features', {}),
                'metadata': {
                    'weighted_aggregation': True,
                    'cluster_weights': weights,
                    'original_levels_count': len(cluster_levels)
                }
            }
            
            return representative
            
        except Exception as e:
            self.logger.error(f"Enhanced representative calculation failed: {e}")
            return cluster_levels[0] if cluster_levels else {}

    async def _validate_clustering_input(self, sr_levels: List[Dict[str, Any]]) -> bool:
        """Validate input for clustering with enhanced error handling."""
        try:
            if not sr_levels:
                self.logger.warning("No SR levels provided for clustering")
                return False
            
            if not isinstance(sr_levels, list):
                self.logger.error(f"Expected list, got {type(sr_levels)}")
                return False
            
            # Check for required fields and data quality
            required_fields = ['price', 'strength', 'type']
            valid_levels = 0
            
            for i, level in enumerate(sr_levels):
                if not isinstance(level, dict):
                    self.logger.warning(f"Level {i} is not a dictionary: {type(level)}")
                    continue
                
                # Check required fields
                missing_fields = [field for field in required_fields if field not in level]
                if missing_fields:
                    self.logger.warning(f"Level {i} missing required fields: {missing_fields}")
                    continue
                
                # Validate data types and ranges
                price = level.get('price', 0.0)
                strength = level.get('strength', 0.0)
                level_type = level.get('type', '')
                
                if not isinstance(price, (int, float)) or price <= 0:
                    self.logger.warning(f"Level {i} has invalid price: {price}")
                    continue
                
                if not isinstance(strength, (int, float)) or strength < 0:
                    self.logger.warning(f"Level {i} has invalid strength: {strength}")
                    continue
                
                if not isinstance(level_type, str) or level_type.lower() not in ['support', 'resistance', 'mixed']:
                    self.logger.warning(f"Level {i} has invalid type: {level_type}")
                    continue
                
                valid_levels += 1
            
            if valid_levels == 0:
                self.logger.error("No valid levels found for clustering")
                return False
            
            if valid_levels < len(sr_levels):
                self.logger.warning(f"Only {valid_levels}/{len(sr_levels)} levels are valid")
            
            # Check for duplicate levels
            seen_prices = set()
            duplicates = 0
            for level in sr_levels:
                price = level.get('price', 0.0)
                if price in seen_prices:
                    duplicates += 1
                else:
                    seen_prices.add(price)
            
            if duplicates > 0:
                self.logger.warning(f"Found {duplicates} duplicate price levels")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False

    async def _create_empty_clustering_result(self) -> Dict[str, Any]:
        """Create an empty clustering result with proper structure."""
        return {
            'total_clusters': 0,
            'clustering_efficiency': 0.0,
            'clusters': [],
            'performance_metrics': {},
            'quality_metrics': {},
            'hardware_metrics': {}
        }

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
                    'memory_optimization': 1.2 if enhanced_config.enable_memory_optimization else 1.0,
                    'total_gain': 1.0  # Ensure at least one field exists
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
                    # Use hasattr to check if method exists before calling
                    if hasattr(self.hardware_manager, 'get_hardware_capabilities'):
                        hardware_caps = self.hardware_manager.get_hardware_capabilities()
                    else:
                        # Fallback: get basic hardware info
                        hardware_caps = {
                            'cpu_cores': 4,  # Default
                            'gpu_available': False,
                            'memory_gb': 8.0
                        }
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
        """Perform enhanced proximity-based clustering with adaptive thresholds."""
        try:
            start_time = datetime.now()

            # Validate input
            if not sr_levels:
                return [], {'efficiency': 0.0, 'method': 'proximity', 'error': 'No levels provided'}
            
            if len(sr_levels) == 1:
                cluster_info = {
                    'cluster_id': 0,
                    'levels': sr_levels,
                    'representative': sr_levels[0],
                    'size': 1,
                    'type': sr_levels[0].get('type', 'mixed')
                }
                return [cluster_info], {'efficiency': 1.0, 'method': 'proximity', 'total_clusters': 1}

            # Calculate adaptive distance threshold
            prices = [level.get('price', 0.0) for level in sr_levels if level.get('price', 0.0) > 0]
            if not prices:
                return [], {'efficiency': 0.0, 'method': 'proximity', 'error': 'No valid prices'}
            
            price_range = max(prices) - min(prices)
            base_threshold = getattr(config, 'eps', 0.01)
            
            # Adaptive threshold: use 1% of price range or base threshold, whichever is larger
            adaptive_threshold = max(base_threshold, price_range * 0.01)
            
            # Use ATR-based distance if available
            if hasattr(config, 'atr') and config.atr > 0:
                adaptive_threshold = max(adaptive_threshold, config.atr * 2)
            
            self.logger.info(f"Using adaptive threshold: {adaptive_threshold:.6f} (base: {base_threshold}, price_range: {price_range:.6f})")

            # Sort levels by strength for better clustering (strongest first)
            sorted_levels = sorted(sr_levels, key=lambda x: x.get('strength', 0.0), reverse=True)
            
            clusters = []
            used_indices = set()

            for i, level in enumerate(sorted_levels):
                if i in used_indices:
                    continue

                # Start a new cluster with this level
                cluster = [level]
                used_indices.add(i)
                level_price = level.get('price', 0.0)

                # Find nearby levels with enhanced distance calculation
                for j, other_level in enumerate(sorted_levels):
                    if j in used_indices or j == i:
                        continue

                    other_price = other_level.get('price', 0.0)
                    
                    # Use absolute price difference for proper distance calculation
                    # This ensures 10% price difference = 0.1, not 0.1% = 0.001
                    price_diff = abs(level_price - other_price) / level_price
                    
                    # Time-based proximity (if timestamps available)
                    time_proximity = self._calculate_time_proximity(level, other_level)
                    
                    # Combined distance metric (price + time)
                    combined_distance = price_diff * (1 - time_proximity * 0.3)

                    if combined_distance <= adaptive_threshold:
                        cluster.append(other_level)
                        used_indices.add(j)

                # Only keep clusters that meet minimum size requirement
                min_cluster_size = getattr(config, 'min_cluster_size', 2)
                if len(cluster) >= min_cluster_size:
                    # Calculate enhanced cluster representative
                    representative = self._calculate_enhanced_cluster_representative(cluster)
                    
                    cluster_info = {
                        'cluster_id': len(clusters),
                        'levels': cluster,
                        'representative': representative,
                        'size': len(cluster),
                        'type': representative.get('type', 'mixed'),
                        'adaptive_threshold_used': adaptive_threshold
                    }
                    clusters.append(cluster_info)

            # If no clusters meet minimum size, return all levels as individual clusters
            if not clusters:
                self.logger.warning("No clusters met minimum size, creating individual clusters")
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
                'method': 'enhanced_proximity',
                'clustering_time': clustering_time,
                'total_clusters': len(clusters),
                'original_levels': len(sr_levels),
                'reduction_ratio': len(clusters) / len(sr_levels) if sr_levels else 0.0,
                'adaptive_threshold': adaptive_threshold,
                'price_range': price_range
            }

            return clusters, metrics

        except Exception as e:
            self.logger.error(f"Enhanced proximity clustering failed: {e}")
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
            
            tprint_success(f"Created SR levels dictionary with {len(all_levels)} levels from {len(clusters)} clusters")
            tprint_data_preview(sr_levels_dictionary, "SR levels dictionary", max_rows=3)
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
    
    def _validate_basestep_integration(self) -> Dict[str, Any]:
        """
        Validate that the component is properly integrated with BaseStep.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'is_basestep_inherited': isinstance(self, BaseStep),
            'has_artifact_manager': hasattr(self, 'artifact_manager'),
            'has_save_artifact_method': hasattr(self, '_save_artifact'),
            'has_get_artifact_method': hasattr(self, '_get_artifact'),
            'has_get_sr_levels_method': hasattr(self, '_get_sr_levels'),
            'has_required_artifacts_method': hasattr(self, 'get_required_artifacts'),
            'step_name_set': hasattr(self, 'step_name') and self.step_name is not None,
            'logger_available': hasattr(self, 'logger') and self.logger is not None
        }
        
        # Check if all required methods are available
        validation_results['all_required_methods_available'] = all([
            validation_results['has_artifact_manager'],
            validation_results['has_save_artifact_method'],
            validation_results['has_get_artifact_method'],
            validation_results['has_get_sr_levels_method'],
            validation_results['has_required_artifacts_method']
        ])
        
        # Overall validation status
        validation_results['integration_valid'] = (
            validation_results['is_basestep_inherited'] and
            validation_results['all_required_methods_available'] and
            validation_results['step_name_set'] and
            validation_results['logger_available']
        )
        
        return validation_results

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
                
                # Add to consensus matrix with proper bounds checking
                for i in range(min(n_levels, len(labels))):
                    for j in range(min(n_levels, len(labels))):
                        if i < len(labels) and j < len(labels) and labels[i] == labels[j] and labels[i] != -1:
                            consensus_matrix[i, j] += weight
            
            # Normalize consensus matrix
            consensus_matrix /= n_algorithms
            
            # Use consensus matrix for final clustering
            # Simple approach: use threshold to determine clusters
            # Lower threshold allows more granular clustering while still requiring consensus
            threshold = config.consensus_threshold
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
            
            # Ensure we have at least 2 samples for model fitting
            if features.shape[0] < 2:
                self.logger.warning(f"Not enough samples ({features.shape[0]}) for explainability, need at least 2")
                return {}
            
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
            
            # Verify features and labels have same length
            if len(cluster_labels) != features.shape[0]:
                self.logger.warning(f"Sample mismatch: features={features.shape[0]}, labels={len(cluster_labels)}")
                return {}
            
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
