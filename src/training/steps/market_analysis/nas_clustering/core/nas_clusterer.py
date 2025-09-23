"""
NAS-driven clusterer for short-term trading regime detection.

This module provides the main NAS clustering functionality optimized for
short-term trading with micro-regime detection capabilities using actual
Neural Architecture Search for optimal clustering architectures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import talib

# Import actual NAS implementation
from src.utils.ml_common.optimization.neural_architecture_search import (
    NeuralArchitectureSearch, ArchitectureConfig, ArchitectureCandidate
)

# Import advanced optimization strategies
try:
    import optuna
    from optuna.samplers import TPESampler, NSGAIISampler
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import deap
    from deap import algorithms, base, creator, tools
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False

# Import matrix operations for optimized computations
from src.utils.matrix_operations import UnifiedMatrixOperations

# Import hardware optimization
from src.utils.hardware.unified_hardware_manager import (
    UnifiedHardwareManager, HardwareConfig, WorkloadType, OptimizationLevel
)
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager

from .nas_config import NASClusteringConfig, NASArchitectureType
from .nas_feature_extractor import NASFeatureExtractor, NASFeatureResult
from .micro_regime_detector import MicroRegimeDetector, MicroRegimeResult
from .nas_regime_optimizer import NASRegimeOptimizer, RegimeOptimizationResult

logger = logging.getLogger(__name__)


@dataclass
class NASClusteringResult:
    """Result of NAS clustering operation."""
    labels: np.ndarray
    cluster_centers: np.ndarray
    statistics: Dict[str, Any]
    quality_metrics: Dict[str, float]
    validation: Dict[str, Any]
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: Optional[str] = None
    
    # NAS-specific fields
    nas_architectures: Dict[str, Any] = None
    micro_regimes: Optional[MicroRegimeResult] = None
    regime_transitions: np.ndarray = None
    economic_significance_scores: np.ndarray = None
    trading_viability_scores: np.ndarray = None

    # True NAS fields
    best_nas_candidate: Optional[ArchitectureCandidate] = None
    nas_search_results: Dict[str, Any] = None
    regime_aware_architectures: Dict[str, ArchitectureCandidate] = None
    clustering_loss_history: List[float] = None


class NASClusterer:
    """NAS-driven clusterer for short-term trading regime detection."""
    
    def __init__(self, config: NASClusteringConfig):
        """Initialize NAS clusterer with matrix operations and hardware optimization.
        
        Args:
            config: NAS clustering configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize matrix operations for optimized computations
        self.matrix_ops = UnifiedMatrixOperations()
        self.logger.info("✅ Matrix operations initialized")
        
        # Initialize hardware optimization
        self.hardware_manager = None
        self.memory_optimizer = None
        self.cpu_optimizer = None
        self.gpu_manager = None
        
        if config.enable_hardware_acceleration:
            self._initialize_hardware_optimization()
        
        # Initialize components with hardware optimization
        self.feature_extractor = NASFeatureExtractor(config.get_feature_config())
        self.micro_regime_detector = MicroRegimeDetector(config.get_micro_regime_config())
        self.regime_optimizer = NASRegimeOptimizer({
            'min_regimes': 5,
            'max_regimes': 20,
            'optimization_methods': ['silhouette', 'calinski_harabasz', 'davies_bouldin'],
            'quality_threshold': 0.6,
            'stability_threshold': 0.7,
            'enable_data_analysis': True,
            'enable_volatility_analysis': True,
            'enable_trend_analysis': True,
            'enable_volume_analysis': True
        })
        
        # NAS architecture settings
        self.nas_architecture_type = config.nas_architecture_type
        self.n_regimes = config.n_regimes  # Will be optimized if data_driven=True
        self.min_regime_duration = config.min_regime_duration
        self.max_regime_duration = config.max_regime_duration
        
        # Data-driven regime count determination
        self.data_driven_regimes = config.get('data_driven_regimes', True)
        
        # Economic significance settings
        self.economic_significance_threshold = config.economic_significance_threshold
        self.trading_viability_threshold = config.trading_viability_threshold
        self.regime_transition_cost = config.regime_transition_cost
        
        # Initialize NAS architectures
        self.nas_architectures = self._initialize_nas_architectures()
        
        self.logger.info(f"✅ NAS Clusterer initialized for {config.timeframe} timeframe with {config.n_regimes} regimes")
        self.logger.info(f"🖥️ Hardware optimization: {self.hardware_manager is not None}")
        self.logger.info(f"🔢 Matrix operations: {self.matrix_ops is not None}")

        # Initialize NAS components with advanced strategies
        self.nas_config = ArchitectureConfig()
        self.nas_config.min_layers = 3
        self.nas_config.max_layers = 8
        self.nas_config.min_units = 64
        self.nas_config.max_units = 512
        self.nas_config.n_trials = 50  # Can be configured
        self.nas_config.timeout_seconds = 1800  # 30 minutes
        self.nas_config.objectives = ['clustering_quality', 'efficiency', 'regime_separation']

        # Advanced search strategies
        self.search_strategy = config.get('nas_search_strategy', 'standard')  # 'standard', 'evolutionary', 'multi_objective', 'bayesian', 'multi_modal'
        self.enable_evolutionary_search = config.get('enable_evolutionary_search', False)
        self.enable_bayesian_optimization = config.get('enable_bayesian_optimization', False)
        self.enable_multi_modal_nas = config.get('enable_multi_modal_nas', False)
        self.population_size = config.get('population_size', 50)
        self.mutation_rate = config.get('mutation_rate', 0.1)
        self.crossover_rate = config.get('crossover_rate', 0.7)

        # Multi-modal clustering settings
        self.clustering_methods = config.get('clustering_methods', ['kmeans', 'dbscan', 'agglomerative', 'neural'])
        self.fusion_strategy = config.get('fusion_strategy', 'ensemble')  # 'ensemble', 'stacked', 'adaptive'
        self.ensemble_weights = config.get('ensemble_weights', 'auto')  # 'auto' or list of weights

        self.nas_search = NeuralArchitectureSearch(self.nas_config)

        # Initialize regime-aware NAS
        self.regime_aware_nas = {}
        self.current_best_architecture = None
        self.clustering_loss_history = []

        # Initialize advanced optimizers
        self._initialize_advanced_optimizers()

        self.logger.info("🧠 NAS components initialized for clustering")
        self.logger.info(f"🔍 Search Strategy: {self.search_strategy}")
        self.logger.info(f"🧬 Evolutionary Search: {self.enable_evolutionary_search}")
        self.logger.info(f"📈 Bayesian Optimization: {self.enable_bayesian_optimization}")

    def _initialize_advanced_optimizers(self):
        """Initialize advanced optimization strategies."""
        try:
            if OPTUNA_AVAILABLE and self.enable_bayesian_optimization:
                # Bayesian optimization study
                self.optuna_study = optuna.create_study(
                    direction='maximize',
                    sampler=TPESampler(n_startup_trials=10, multivariate=True),
                    pruner=MedianPruner(n_startup_trials=5)
                )
                self.logger.info("✅ Bayesian optimization initialized with Optuna")

            if DEAP_AVAILABLE and self.enable_evolutionary_search:
                # Evolutionary algorithm setup
                self._setup_evolutionary_algorithm()
                self.logger.info("✅ Evolutionary algorithm initialized with DEAP")

        except Exception as e:
            self.logger.warning(f"⚠️ Advanced optimizer initialization failed: {e}")
            self.optuna_study = None

    def _setup_evolutionary_algorithm(self):
        """Setup evolutionary algorithm for architecture search."""
        try:
            # Create fitness and individual classes
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))  # Multi-objective
            creator.create("Individual", list, fitness=creator.FitnessMulti)

            # Create toolbox
            self.toolbox = base.Toolbox()

            # Register genetic operators
            self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
            self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
            self.toolbox.register("select", tools.selNSGA2)  # NSGA-II for multi-objective
            self.toolbox.register("evaluate", self._evaluate_architecture_fitness)

            # Create initial population
            self.population = self.toolbox.population(n=self.population_size)

            self.logger.info("✅ Evolutionary algorithm setup completed")

        except Exception as e:
            self.logger.error(f"❌ Evolutionary algorithm setup failed: {e}")
            self.toolbox = None
            self.population = None
    
    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization components."""
        try:
            # Initialize unified hardware manager
            hardware_config = HardwareConfig(
                cpu_optimization_level=OptimizationLevel.BALANCED,
                gpu_optimization_level=OptimizationLevel.BALANCED,
                memory_optimization_level=OptimizationLevel.BALANCED,
                memory_limit_gb=self.config.max_memory_usage * 8,  # Convert to GB
                enable_adaptive_optimization=True,
                learning_enabled=True,
                auto_tuning_enabled=True
            )
            self.hardware_manager = UnifiedHardwareManager(hardware_config)
            
            # Initialize M1-specific optimizers
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
            self.gpu_manager = get_m1_gpu_manager()
            
            self.logger.info("✅ Hardware optimization components initialized")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Hardware optimization initialization failed: {e}")
            self.hardware_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.gpu_manager = None
    
    def _optimize_data_array_with_matrix_ops(self, data_array: np.ndarray) -> np.ndarray:
        """Optimize data array using matrix operations."""
        try:
            # Use matrix operations for data preprocessing
            if self.matrix_ops:
                # Normalize data using matrix operations
                normalized_data = self.matrix_ops.matrix_normalize(data_array)
                self.logger.info("✅ Data array optimized with matrix operations")
                return normalized_data
            else:
                # Fallback to standard normalization
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                return scaler.fit_transform(data_array)
        except Exception as e:
            self.logger.warning(f"⚠️ Matrix operations optimization failed: {e}")
            return data_array
    
    def _perform_clustering_with_optimization(self, data_array: np.ndarray, 
                                            n_regimes: int, 
                                            optimize_parameters: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Perform clustering with hardware optimization."""
        try:
            # Start hardware optimization if available
            if self.hardware_manager:
                self.hardware_manager.start_optimization(
                    workload_type=WorkloadType.ML_TRAINING,
                    optimization_level=OptimizationLevel.BALANCED
                )
            
            # Use matrix operations for clustering
            if self.matrix_ops and optimize_parameters:
                # Use matrix operations for parameter optimization
                best_params = self._optimize_clustering_parameters_with_matrix_ops(data_array, n_regimes)
                clustering_model = KMeans(n_clusters=n_regimes, **best_params, random_state=42)
            else:
                clustering_model = KMeans(n_clusters=n_regimes, random_state=42)
            
            # Perform clustering
            labels = clustering_model.fit_predict(data_array)
            cluster_centers = clustering_model.cluster_centers_
            
            self.logger.info(f"✅ Clustering completed with {n_regimes} regimes")
            return labels, cluster_centers
            
        except Exception as e:
            self.logger.error(f"❌ Clustering with optimization failed: {e}")
            # Fallback to basic clustering
            clustering_model = KMeans(n_clusters=n_regimes, random_state=42)
            labels = clustering_model.fit_predict(data_array)
            cluster_centers = clustering_model.cluster_centers_
            return labels, cluster_centers
        
        finally:
            # Stop hardware optimization
            if self.hardware_manager:
                self.hardware_manager.stop_optimization()
    
    def _optimize_clustering_parameters_with_matrix_ops(self, data_array: np.ndarray, 
                                                      n_regimes: int) -> Dict[str, Any]:
        """Optimize clustering parameters using matrix operations."""
        try:
            # Use matrix operations to find optimal parameters
            best_params = {
                'n_init': 10,
                'max_iter': 300,
                'tol': 1e-4
            }
            
            # Test different parameter combinations using matrix operations
            param_combinations = [
                {'n_init': 5, 'max_iter': 100, 'tol': 1e-3},
                {'n_init': 10, 'max_iter': 300, 'tol': 1e-4},
                {'n_init': 20, 'max_iter': 500, 'tol': 1e-5}
            ]
            
            best_score = -np.inf
            for params in param_combinations:
                try:
                    model = KMeans(n_clusters=n_regimes, **params, random_state=42)
                    labels = model.fit_predict(data_array)
                    
                    # Calculate silhouette score using matrix operations
                    if self.matrix_ops:
                        score = self.matrix_ops.calculate_silhouette_score(data_array, labels)
                    else:
                        from sklearn.metrics import silhouette_score
                        score = silhouette_score(data_array, labels)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        
                except Exception:
                    continue
            
            self.logger.info(f"✅ Best clustering parameters found: {best_params}")
            return best_params
            
        except Exception as e:
            self.logger.warning(f"⚠️ Parameter optimization failed: {e}")
            return {'n_init': 10, 'max_iter': 300, 'tol': 1e-4}
    
    def _calculate_quality_metrics_with_matrix_ops(self, data_array: np.ndarray, 
                                                 labels: np.ndarray, 
                                                 cluster_centers: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics using matrix operations."""
        try:
            metrics = {}
            
            # Silhouette score using matrix operations
            if self.matrix_ops:
                metrics['silhouette_score'] = self.matrix_ops.calculate_silhouette_score(data_array, labels)
                metrics['calinski_harabasz_score'] = self.matrix_ops.calculate_calinski_harabasz_score(data_array, labels)
                metrics['davies_bouldin_score'] = self.matrix_ops.calculate_davies_bouldin_score(data_array, labels)
            else:
                # Fallback to sklearn
                from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
                metrics['silhouette_score'] = silhouette_score(data_array, labels)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(data_array, labels)
                metrics['davies_bouldin_score'] = davies_bouldin_score(data_array, labels)
            
            # Regime stability using matrix operations
            metrics['regime_stability'] = self._calculate_regime_stability_with_matrix_ops(data_array, labels)
            
            # Economic significance
            metrics['economic_significance'] = self._calculate_economic_significance_with_matrix_ops(data_array, labels)
            
            # Trading viability
            metrics['trading_viability'] = self._calculate_trading_viability_with_matrix_ops(data_array, labels)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Quality metrics calculation failed: {e}")
            return {'silhouette_score': 0.0, 'regime_stability': 0.0, 'economic_significance': 0.0, 'trading_viability': 0.0}
    
    def _calculate_regime_stability_with_matrix_ops(self, data_array: np.ndarray, labels: np.ndarray) -> float:
        """Calculate regime stability using matrix operations."""
        try:
            if self.matrix_ops:
                # Use matrix operations to calculate regime stability
                stability = self.matrix_ops.calculate_regime_stability(data_array, labels)
                return stability
            else:
                # Fallback calculation
                unique_labels, counts = np.unique(labels, return_counts=True)
                stability = 1.0 - (np.std(counts) / np.mean(counts))
                return max(0.0, min(1.0, stability))
        except Exception:
            return 0.0
    
    def _calculate_economic_significance_with_matrix_ops(self, data_array: np.ndarray, labels: np.ndarray) -> float:
        """Calculate economic significance using matrix operations."""
        try:
            if self.matrix_ops:
                # Use matrix operations to calculate economic significance
                significance = self.matrix_ops.calculate_economic_significance(data_array, labels)
                return significance
            else:
                # Fallback calculation based on regime separation
                unique_labels = np.unique(labels)
                if len(unique_labels) < 2:
                    return 0.0
                
                # Calculate separation between regimes
                regime_centers = []
                for label in unique_labels:
                    regime_data = data_array[labels == label]
                    regime_centers.append(np.mean(regime_data, axis=0))
                
                regime_centers = np.array(regime_centers)
                center_distances = []
                for i in range(len(regime_centers)):
                    for j in range(i+1, len(regime_centers)):
                        distance = np.linalg.norm(regime_centers[i] - regime_centers[j])
                        center_distances.append(distance)
                
                if center_distances:
                    significance = np.mean(center_distances) / (np.std(center_distances) + 1e-8)
                    return max(0.0, min(1.0, significance))
                else:
                    return 0.0
        except Exception:
            return 0.0
    
    def _calculate_trading_viability_with_matrix_ops(self, data_array: np.ndarray, labels: np.ndarray) -> float:
        """Calculate trading viability using matrix operations."""
        try:
            if self.matrix_ops:
                # Use matrix operations to calculate trading viability
                viability = self.matrix_ops.calculate_trading_viability(data_array, labels)
                return viability
            else:
                # Fallback calculation based on regime duration and consistency
                unique_labels, counts = np.unique(labels, return_counts=True)
                
                # Check minimum regime duration
                min_duration = self.min_regime_duration
                viable_regimes = np.sum(counts >= min_duration)
                viability = viable_regimes / len(unique_labels) if len(unique_labels) > 0 else 0.0
                
                return max(0.0, min(1.0, viability))
        except Exception:
            return 0.0
    
    def cluster(self, data: Union[pd.DataFrame, np.ndarray], 
                timestamps: Optional[np.ndarray] = None,
                optimize_parameters: bool = True,
                generate_report: bool = True) -> NASClusteringResult:
        """Perform NAS-driven clustering on market data.
        
        Args:
            data: Market data (DataFrame or numpy array)
            timestamps: Optional timestamps array
            optimize_parameters: Whether to optimize NAS parameters
            generate_report: Whether to generate clustering report
            
        Returns:
            NASClusteringResult with clustering results
        """
        start_time = time.time()
        
        try:
            self.logger.info("🚀 Starting NAS-driven clustering")
            
            # Prepare data
            if isinstance(data, pd.DataFrame):
                data_array = data.values
                if timestamps is None and 'timestamp' in data.columns:
                    timestamps = data['timestamp'].values
            else:
                data_array = data
                if timestamps is None:
                    timestamps = np.arange(len(data))
            
            # Extract NAS features
            feature_result = self.feature_extractor.extract_features(data_array, timestamps)
            
            # Optimize regime count if data-driven
            if self.data_driven_regimes:
                self.logger.info("🔍 Optimizing regime count based on data characteristics")
                regime_optimization = self.regime_optimizer.optimize_regime_count(
                    feature_result.features, data_array, timestamps, self.n_regimes
                )
                self.n_regimes = regime_optimization.optimal_n_regimes
                self.logger.info(f"📊 Optimal regime count determined: {self.n_regimes}")
            
            # Detect micro-regimes
            micro_regime_result = self.micro_regime_detector.detect_micro_regimes(
                data_array, timestamps, feature_result.features
            )
            
            # Perform NAS clustering
            clustering_result = self._perform_nas_clustering(
                feature_result, micro_regime_result, optimize_parameters
            )
            
            # Calculate economic significance and trading viability
            economic_scores = self._calculate_economic_significance_scores(
                data_array, clustering_result['labels']
            )
            trading_scores = self._calculate_trading_viability_scores(
                data_array, clustering_result['labels']
            )
            
            # Calculate regime transitions
            regime_transitions = self._calculate_regime_transitions(
                clustering_result['labels']
            )
            
            # Create NAS clustering result
            nas_result = self._create_nas_result(
                clustering_result, micro_regime_result, economic_scores,
                trading_scores, regime_transitions, feature_result,
                time.time() - start_time
            )
            
            # Generate report if requested
            if generate_report:
                self._generate_nas_report(nas_result, feature_result)
            
            self.logger.info(f"✅ NAS clustering completed in {nas_result.execution_time:.2f}s")
            return nas_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ NAS clustering failed: {e}")
            return NASClusteringResult(
                labels=np.array([]),
                cluster_centers=np.array([]),
                statistics={},
                quality_metrics={},
                validation={'valid': False, 'error': str(e)},
                metadata={'error': str(e), 'method': 'nas_clustering'},
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _initialize_nas_architectures(self) -> Dict[str, Any]:
        """Initialize NAS architectures for different regime types."""
        try:
            architectures = {}

            # Volatility-focused architecture
            architectures['volatility'] = {
                'type': 'volatility_focused',
                'layers': [64, 32, 16],
                'activation': 'relu',
                'dropout': 0.2,
                'optimizer': 'adam',
                'learning_rate': 0.001
            }

            # Trend-focused architecture
            architectures['trend'] = {
                'type': 'trend_focused',
                'layers': [128, 64, 32],
                'activation': 'tanh',
                'dropout': 0.3,
                'optimizer': 'adam',
                'learning_rate': 0.001
            }

            # Volume-focused architecture
            architectures['volume'] = {
                'type': 'volume_focused',
                'layers': [32, 16, 8],
                'activation': 'relu',
                'dropout': 0.1,
                'optimizer': 'adam',
                'learning_rate': 0.01
            }

            # Momentum-focused architecture
            architectures['momentum'] = {
                'type': 'momentum_focused',
                'layers': [96, 48, 24],
                'activation': 'swish',
                'dropout': 0.25,
                'optimizer': 'adam',
                'learning_rate': 0.001
            }

            # Hybrid architecture
            architectures['hybrid'] = {
                'type': 'hybrid',
                'layers': [256, 128, 64, 32],
                'activation': 'relu',
                'dropout': 0.2,
                'optimizer': 'adam',
                'learning_rate': 0.001
            }

            return architectures

        except Exception as e:
            self.logger.warning(f"⚠️ NAS architecture initialization failed: {e}")
            return {}

    def _perform_true_nas_clustering(self, features: np.ndarray,
                                    n_regimes: int,
                                    use_nas: bool = True) -> Dict[str, Any]:
        """Perform true NAS-based clustering using neural architecture search."""
        try:
            if not use_nas:
                # Fallback to traditional clustering
                return self._perform_traditional_clustering(features, n_regimes)

            self.logger.info("🧠 Performing true NAS-based clustering")

            # Prepare data for NAS
            features_scaled = self.scaler.fit_transform(features)
            n_samples = features_scaled.shape[0]

            # Use NAS to find optimal clustering architecture
            if self.current_best_architecture is None:
                self.logger.info("🔍 Searching for optimal clustering architecture")
                self._search_optimal_clustering_architecture(features_scaled, n_regimes)

            if self.current_best_architecture is None:
                self.logger.warning("⚠️ NAS search failed, falling back to traditional clustering")
                return self._perform_traditional_clustering(features, n_regimes)

            # Create clustering model from NAS architecture
            clustering_model = self._create_clustering_model_from_nas_architecture(
                self.current_best_architecture, features_scaled.shape[1]
            )

            # Perform clustering using the NAS-based model
            labels, cluster_centers, clustering_loss = self._cluster_with_nas_model(
                clustering_model, features_scaled, n_regimes
            )

            # Store clustering loss for monitoring
            self.clustering_loss_history.append(clustering_loss)

            self.logger.info(f"✅ NAS clustering completed with loss: {clustering_loss:.4f}")

            return {
                'labels': labels,
                'cluster_centers': cluster_centers,
                'clustering_loss': clustering_loss,
                'nas_architecture_used': self.current_best_architecture,
                'clustering_method': 'nas_based'
            }

        except Exception as e:
            self.logger.error(f"❌ NAS clustering failed: {e}")
            # Fallback to traditional clustering
            return self._perform_traditional_clustering(features, n_regimes)

    def _search_optimal_clustering_architecture(self, features: np.ndarray,
                                               n_regimes: int) -> None:
        """Search for optimal clustering architecture using advanced NAS strategies."""
        try:
            # Store current features and n_regimes for evaluation
            self.current_features = features
            self.current_n_regimes = n_regimes

            self.logger.info(f"🔍 Starting advanced NAS search: {self.search_strategy}")

            if self.search_strategy == 'evolutionary' and DEAP_AVAILABLE:
                self._perform_evolutionary_search(features, n_regimes)
            elif self.search_strategy == 'bayesian' and OPTUNA_AVAILABLE:
                self._perform_bayesian_search(features, n_regimes)
            elif self.search_strategy == 'multi_objective':
                self._perform_multi_objective_search(features, n_regimes)
            elif self.search_strategy == 'multi_modal' and self.enable_multi_modal_nas:
                self._perform_multi_modal_search(features, n_regimes)
            else:
                # Default standard search
                self._perform_standard_nas_search(features, n_regimes)

            self.logger.info(f"✅ Advanced NAS search completed. Best architecture: {self.current_best_architecture.overall_score:.4f}" if self.current_best_architecture else "✅ Advanced NAS search completed")

        except Exception as e:
            self.logger.error(f"❌ Advanced NAS search failed: {e}")
            # Fallback to standard search
            self._perform_standard_nas_search(features, n_regimes)

    def _perform_standard_nas_search(self, features: np.ndarray, n_regimes: int):
        """Perform standard NAS search."""
        try:
            clustering_targets = self._create_clustering_targets_for_nas(features, n_regimes)
            self.current_best_architecture = self.nas_search.search(
                X_train=features,
                y_train=clustering_targets,
                regime_labels=None
            )
        except Exception as e:
            self.logger.error(f"❌ Standard NAS search failed: {e}")
            self.current_best_architecture = None

    def _perform_evolutionary_search(self, features: np.ndarray, n_regimes: int):
        """Perform evolutionary architecture search."""
        try:
            if not DEAP_AVAILABLE:
                self.logger.warning("⚠️ DEAP not available, falling back to standard search")
                self._perform_standard_nas_search(features, n_regimes)
                return

            self.logger.info("🧬 Running evolutionary architecture search")

            # Generate initial population of architectures
            population = self._generate_architecture_population(features, n_regimes)

            # Run evolutionary algorithm
            n_generations = 20
            for gen in range(n_generations):
                # Evaluate fitness
                fitnesses = list(map(self._evaluate_architecture_fitness, population))
                for ind, fit in zip(population, fitnesses):
                    ind.fitness.values = fit

                # Select next generation
                population = self.toolbox.select(population, len(population))

                # Apply crossover and mutation
                offspring = algorithms.varAnd(population, self.toolbox, cxpb=self.crossover_rate, mutpb=self.mutation_rate)

                # Replace population
                population[:] = offspring

            # Find best individual
            best_individual = tools.selBest(population, 1)[0]
            self.current_best_architecture = self._individual_to_architecture(best_individual, features.shape[1])

            self.logger.info(f"🧬 Evolutionary search completed after {n_generations} generations")

        except Exception as e:
            self.logger.error(f"❌ Evolutionary search failed: {e}")
            self._perform_standard_nas_search(features, n_regimes)

    def _perform_bayesian_search(self, features: np.ndarray, n_regimes: int):
        """Perform Bayesian optimization for architecture search."""
        try:
            if not OPTUNA_AVAILABLE:
                self.logger.warning("⚠️ Optuna not available, falling back to standard search")
                self._perform_standard_nas_search(features, n_regimes)
                return

            self.logger.info("📈 Running Bayesian architecture optimization")

            def objective(trial):
                # Define architecture parameters
                n_layers = trial.suggest_int('n_layers', 2, 8)
                n_units = trial.suggest_int('n_units', 32, 512)
                dropout = trial.suggest_float('dropout', 0.0, 0.5)
                activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'swish'])

                # Create architecture
                architecture = ArchitectureCandidate(
                    layers=[{
                        'units': n_units,
                        'activation': activation,
                        'dropout': dropout
                    } for _ in range(n_layers)],
                    total_params=n_units * n_layers,
                    estimated_flops=n_units * n_layers * 1000
                )

                # Evaluate architecture
                try:
                    clustering_targets = self._create_clustering_targets_for_nas(features, n_regimes)
                    performance = self._evaluate_architecture_performance(architecture, features, clustering_targets)

                    # Multi-objective: clustering quality, efficiency, robustness
                    clustering_quality = performance.get('accuracy', 0.0)
                    efficiency = 1.0 / (1.0 + architecture.total_params / 100000)
                    robustness = performance.get('robustness_score', 0.5)

                    return clustering_quality, efficiency, robustness

                except Exception:
                    return 0.0, 0.0, 0.0

            # Run optimization
            n_trials = 50
            self.optuna_study.optimize(objective, n_trials=n_trials)

            # Get best architecture
            best_params = self.optuna_study.best_params
            self.current_best_architecture = self._create_architecture_from_params(best_params, features.shape[1])

            self.logger.info(f"📈 Bayesian optimization completed with best score: {self.optuna_study.best_value:.4f}")

        except Exception as e:
            self.logger.error(f"❌ Bayesian search failed: {e}")
            self._perform_standard_nas_search(features, n_regimes)

    def _perform_multi_objective_search(self, features: np.ndarray, n_regimes: int):
        """Perform multi-objective architecture search."""
        try:
            if not OPTUNA_AVAILABLE:
                self.logger.warning("⚠️ Optuna not available, falling back to standard search")
                self._perform_standard_nas_search(features, n_regimes)
                return

            self.logger.info("🎯 Running multi-objective architecture search")

            def multi_objective(trial):
                # Define architecture parameters
                n_layers = trial.suggest_int('n_layers', 2, 8)
                n_units = trial.suggest_int('n_units', 32, 512)
                dropout = trial.suggest_float('dropout', 0.0, 0.5)
                activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'swish'])

                # Create architecture
                architecture = ArchitectureCandidate(
                    layers=[{
                        'units': n_units,
                        'activation': activation,
                        'dropout': dropout
                    } for _ in range(n_layers)],
                    total_params=n_units * n_layers,
                    estimated_flops=n_units * n_layers * 1000
                )

                # Evaluate architecture
                clustering_targets = self._create_clustering_targets_for_nas(features, n_regimes)
                performance = self._evaluate_architecture_performance(architecture, features, clustering_targets)

                # Multi-objective optimization
                clustering_quality = performance.get('accuracy', 0.0)
                efficiency = 1.0 / (1.0 + architecture.total_params / 100000)
                robustness = performance.get('robustness_score', 0.5)

                return clustering_quality, efficiency, robustness

            # Multi-objective study
            study = optuna.create_study(
                directions=['maximize', 'maximize', 'maximize'],
                sampler=NSGAIISampler()
            )

            study.optimize(multi_objective, n_trials=50)

            # Select best architecture from Pareto front
            pareto_front = study.get_pareto_front_directions()
            best_idx = 0  # Can be enhanced to select based on preference

            if pareto_front:
                best_trial = study.trials[best_idx]
                self.current_best_architecture = self._create_architecture_from_trial(best_trial, features.shape[1])

            self.logger.info(f"🎯 Multi-objective search completed. Pareto front size: {len(pareto_front)}")

        except Exception as e:
            self.logger.error(f"❌ Multi-objective search failed: {e}")
            self._perform_standard_nas_search(features, n_regimes)

    def _perform_multi_modal_search(self, features: np.ndarray, n_regimes: int):
        """Perform multi-modal NAS combining different clustering approaches."""
        try:
            self.logger.info("🔄 Running multi-modal NAS search")

            # Get embeddings from multiple clustering methods
            clustering_results = {}
            embeddings_dict = {}

            for method in self.clustering_methods:
                try:
                    if method == 'neural':
                        # Use neural network embeddings
                        embeddings = self._get_neural_embeddings(features, n_regimes)
                        clustering_results[method] = self._cluster_with_embeddings(embeddings, n_regimes, method)
                    elif method == 'kmeans':
                        # Standard K-means
                        embeddings = features  # Use raw features
                        clustering_results[method] = self._cluster_with_method(embeddings, n_regimes, method)
                    elif method == 'dbscan':
                        embeddings = features
                        clustering_results[method] = self._cluster_with_method(embeddings, n_regimes, method)
                    elif method == 'agglomerative':
                        embeddings = features
                        clustering_results[method] = self._cluster_with_method(embeddings, n_regimes, method)

                    embeddings_dict[method] = embeddings
                    self.logger.info(f"✅ {method} clustering completed")

                except Exception as e:
                    self.logger.warning(f"⚠️ {method} clustering failed: {e}")
                    continue

            if not clustering_results:
                self.logger.error("❌ All clustering methods failed")
                self._perform_standard_nas_search(features, n_regimes)
                return

            # Fuse results using selected strategy
            if self.fusion_strategy == 'ensemble':
                fused_labels = self._ensemble_fusion(clustering_results, features.shape[0])
            elif self.fusion_strategy == 'stacked':
                fused_labels = self._stacked_fusion(clustering_results, embeddings_dict, features, n_regimes)
            elif self.fusion_strategy == 'adaptive':
                fused_labels = self._adaptive_fusion(clustering_results, features, n_regimes)
            else:
                # Default to ensemble
                fused_labels = self._ensemble_fusion(clustering_results, features.shape[0])

            # Create synthetic architecture representing multi-modal approach
            self.current_best_architecture = self._create_multi_modal_architecture(clustering_results)

            # Calculate quality metrics for fused result
            fused_metrics = self._calculate_fusion_quality_metrics(features, fused_labels, clustering_results)

            # Store multi-modal results
            self.multi_modal_results = {
                'individual_results': clustering_results,
                'fused_labels': fused_labels,
                'fusion_strategy': self.fusion_strategy,
                'quality_metrics': fused_metrics,
                'method_weights': self._calculate_method_weights(clustering_results)
            }

            self.logger.info(f"🔄 Multi-modal search completed with {self.fusion_strategy} fusion")

        except Exception as e:
            self.logger.error(f"❌ Multi-modal search failed: {e}")
            self._perform_standard_nas_search(features, n_regimes)

    def _get_neural_embeddings(self, features: np.ndarray, n_regimes: int) -> np.ndarray:
        """Get embeddings from neural network."""
        try:
            # Create a simple neural network for feature extraction
            input_dim = features.shape[1]

            class FeatureExtractor(nn.Module):
                def __init__(self, input_dim, hidden_dim=128, output_dim=64):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_dim // 2, output_dim)
                    )

                def forward(self, x):
                    return self.encoder(x)

            # Initialize model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = FeatureExtractor(input_dim).to(device)

            # Convert to tensors
            features_tensor = torch.FloatTensor(features).to(device)

            # Get embeddings
            model.eval()
            with torch.no_grad():
                embeddings = model(features_tensor).cpu().numpy()

            return embeddings

        except Exception as e:
            self.logger.warning(f"⚠️ Neural embedding extraction failed: {e}")
            return features  # Fallback to raw features

    def _cluster_with_embeddings(self, embeddings: np.ndarray, n_regimes: int, method: str) -> Dict[str, Any]:
        """Perform clustering using embeddings."""
        try:
            if method == 'neural':
                return self._perform_true_nas_clustering(embeddings, n_regimes, use_nas=False)
            else:
                return self._cluster_with_method(embeddings, n_regimes, method)
        except Exception as e:
            self.logger.error(f"❌ Clustering with embeddings failed: {e}")
            raise

    def _cluster_with_method(self, features: np.ndarray, n_regimes: int, method: str) -> Dict[str, Any]:
        """Perform clustering using specified method."""
        try:
            features_scaled = self.scaler.fit_transform(features)

            if method == 'kmeans':
                clusterer = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
                labels = clusterer.fit_predict(features_scaled)
                cluster_centers = clusterer.cluster_centers_
            elif method == 'dbscan':
                clusterer = DBSCAN(eps=0.5, min_samples=max(5, n_regimes))
                labels = clusterer.fit_predict(features_scaled)
                cluster_centers = self._calculate_cluster_centers(features_scaled, labels)
            elif method == 'agglomerative':
                clusterer = AgglomerativeClustering(n_clusters=n_regimes, linkage='ward')
                labels = clusterer.fit_predict(features_scaled)
                cluster_centers = self._calculate_cluster_centers(features_scaled, labels)
            else:
                raise ValueError(f"Unknown clustering method: {method}")

            return {
                'labels': labels,
                'cluster_centers': cluster_centers,
                'method': method,
                'success': True
            }

        except Exception as e:
            self.logger.error(f"❌ {method} clustering failed: {e}")
            return {'labels': None, 'cluster_centers': None, 'method': method, 'success': False}

    def _ensemble_fusion(self, clustering_results: Dict[str, Any], n_samples: int) -> np.ndarray:
        """Fuse clustering results using ensemble voting."""
        try:
            # Get valid results
            valid_results = {k: v for k, v in clustering_results.items() if v['success']}

            if not valid_results:
                raise ValueError("No valid clustering results to fuse")

            # Calculate weights
            if self.ensemble_weights == 'auto':
                weights = self._calculate_method_weights(valid_results)
            else:
                weights = self.ensemble_weights

            # Ensure weights match the number of methods
            method_names = list(valid_results.keys())
            if isinstance(weights, list):
                weights = {method_names[i]: weights[i] for i in range(len(method_names))}
            elif isinstance(weights, dict):
                weights = {k: weights.get(k, 1.0) for k in method_names}
            else:
                weights = {k: 1.0 for k in method_names}

            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: w / total_weight for k, w in weights.items()}

            # Create ensemble predictions
            ensemble_labels = np.zeros(n_samples)

            for method, result in valid_results.items():
                if result['labels'] is not None:
                    weight = weights[method]
                    labels = result['labels']

                    # Add weighted votes
                    for i in range(n_samples):
                        ensemble_labels[i] += weight * labels[i]

            # Round to nearest integer for final labels
            ensemble_labels = np.round(ensemble_labels).astype(int)

            # Ensure labels are non-negative
            ensemble_labels = np.maximum(ensemble_labels, 0)

            return ensemble_labels

        except Exception as e:
            self.logger.error(f"❌ Ensemble fusion failed: {e}")
            # Fallback to first valid method
            for result in valid_results.values():
                if result['success'] and result['labels'] is not None:
                    return result['labels']
            raise ValueError("No valid clustering results available")

    def _stacked_fusion(self, clustering_results: Dict[str, Any], embeddings_dict: Dict[str, np.ndarray],
                       features: np.ndarray, n_regimes: int) -> np.ndarray:
        """Fuse clustering results using stacked generalization."""
        try:
            # Create meta-features from individual clustering results
            meta_features = []

            for method, result in clustering_results.items():
                if result['success'] and result['labels'] is not None:
                    # Use clustering confidence or distance to centers as meta-feature
                    if method == 'kmeans' and result['cluster_centers'] is not None:
                        distances = self._calculate_distances_to_centers(features, result['labels'], result['cluster_centers'])
                        meta_features.append(distances.reshape(-1, 1))
                    else:
                        # Use one-hot encoding of cluster labels
                        labels_onehot = np.zeros((len(result['labels']), n_regimes))
                        for i, label in enumerate(result['labels']):
                            if 0 <= label < n_regimes:
                                labels_onehot[i, label] = 1
                        meta_features.append(labels_onehot)

            if not meta_features:
                raise ValueError("No meta-features available")

            # Combine meta-features
            meta_features_combined = np.concatenate(meta_features, axis=1)

            # Use simple classifier on meta-features
            from sklearn.ensemble import RandomForestClassifier
            meta_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            meta_classifier.fit(meta_features_combined, clustering_results[list(clustering_results.keys())[0]]['labels'])

            # Predict final labels
            predictions = meta_classifier.predict(meta_features_combined)

            return predictions

        except Exception as e:
            self.logger.error(f"❌ Stacked fusion failed: {e}")
            # Fallback to ensemble
            return self._ensemble_fusion(clustering_results, features.shape[0])

    def _adaptive_fusion(self, clustering_results: Dict[str, Any], features: np.ndarray, n_regimes: int) -> np.ndarray:
        """Adaptively fuse clustering results based on data characteristics."""
        try:
            # Calculate quality metrics for each method
            method_qualities = {}

            for method, result in clustering_results.items():
                if result['success'] and result['labels'] is not None:
                    try:
                        quality = self._calculate_clustering_quality(features, result['labels'], n_regimes)
                        method_qualities[method] = quality
                    except Exception:
                        method_qualities[method] = 0.0

            if not method_qualities:
                raise ValueError("No quality metrics available")

            # Weight methods by quality
            total_quality = sum(method_qualities.values())
            weights = {k: v / total_quality for k, v in method_qualities.items()}

            # Apply adaptive weighting
            n_samples = features.shape[0]
            adaptive_labels = np.zeros(n_samples)

            for method, result in clustering_results.items():
                if result['success'] and result['labels'] is not None:
                    weight = weights[method]
                    labels = result['labels']

                    for i in range(n_samples):
                        adaptive_labels[i] += weight * labels[i]

            # Round and ensure valid labels
            adaptive_labels = np.round(adaptive_labels).astype(int)
            adaptive_labels = np.maximum(adaptive_labels, 0)

            return adaptive_labels

        except Exception as e:
            self.logger.error(f"❌ Adaptive fusion failed: {e}")
            return self._ensemble_fusion(clustering_results, features.shape[0])

    def _calculate_method_weights(self, clustering_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate weights for different clustering methods."""
        try:
            # Base weights for different methods
            base_weights = {
                'kmeans': 1.0,
                'dbscan': 0.9,
                'agglomerative': 0.8,
                'neural': 0.95
            }

            weights = {}
            for method in clustering_results.keys():
                weights[method] = base_weights.get(method, 0.5)

            return weights

        except Exception:
            return {method: 1.0 for method in clustering_results.keys()}

    def _calculate_fusion_quality_metrics(self, features: np.ndarray, fused_labels: np.ndarray,
                                        clustering_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality metrics for fused clustering results."""
        try:
            # Standard metrics
            if len(np.unique(fused_labels)) < 2:
                return {'silhouette_score': 0.0, 'fusion_quality': 0.0}

            silhouette = silhouette_score(features, fused_labels)
            calinski_harabasz = calinski_harabasz_score(features, fused_labels)

            # Fusion-specific metrics
            method_agreement = self._calculate_method_agreement(clustering_results)
            consistency_score = self._calculate_consistency_score(clustering_results, fused_labels)

            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'method_agreement': method_agreement,
                'consistency_score': consistency_score,
                'fusion_quality': (silhouette + calinski_harabasz + method_agreement + consistency_score) / 4.0
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Fusion quality calculation failed: {e}")
            return {'silhouette_score': 0.0, 'fusion_quality': 0.0}

    def _calculate_method_agreement(self, clustering_results: Dict[str, Any]) -> float:
        """Calculate agreement between different clustering methods."""
        try:
            methods = list(clustering_results.keys())
            if len(methods) < 2:
                return 1.0

            # Calculate pairwise agreement
            agreements = []

            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    method1 = methods[i]
                    method2 = methods[j]

                    if (clustering_results[method1]['success'] and
                        clustering_results[method2]['success'] and
                        clustering_results[method1]['labels'] is not None and
                        clustering_results[method2]['labels'] is not None):

                        labels1 = clustering_results[method1]['labels']
                        labels2 = clustering_results[method2]['labels']

                        # Calculate adjusted rand index (simplified)
                        agreement = self._calculate_adjusted_rand_index(labels1, labels2)
                        agreements.append(agreement)

            return np.mean(agreements) if agreements else 0.0

        except Exception:
            return 0.0

    def _calculate_adjusted_rand_index(self, labels1: np.ndarray, labels2: np.ndarray) -> float:
        """Calculate adjusted Rand index between two labelings."""
        try:
            from sklearn.metrics import adjusted_rand_score
            return adjusted_rand_score(labels1, labels2)
        except Exception:
            return 0.0

    def _calculate_consistency_score(self, clustering_results: Dict[str, Any], fused_labels: np.ndarray) -> float:
        """Calculate consistency between fused labels and individual methods."""
        try:
            consistencies = []

            for method, result in clustering_results.items():
                if result['success'] and result['labels'] is not None:
                    consistency = self._calculate_adjusted_rand_index(result['labels'], fused_labels)
                    consistencies.append(consistency)

            return np.mean(consistencies) if consistencies else 0.0

        except Exception:
            return 0.0

    def _create_multi_modal_architecture(self, clustering_results: Dict[str, Any]) -> ArchitectureCandidate:
        """Create synthetic architecture representing multi-modal approach."""
        try:
            n_methods = len([r for r in clustering_results.values() if r['success']])
            n_regimes = len(np.unique([r['labels'] for r in clustering_results.values() if r['labels'] is not None]))

            # Create synthetic architecture description
            layers = [{
                'method': 'multi_modal_fusion',
                'n_methods': n_methods,
                'fusion_strategy': self.fusion_strategy,
                'units': n_methods * 10,
                'activation': 'ensemble',
                'dropout': 0.0
            }]

            return ArchitectureCandidate(
                layers=layers,
                total_params=n_methods * 100,  # Synthetic parameter count
                estimated_flops=n_methods * 1000
            )

        except Exception as e:
            self.logger.error(f"❌ Multi-modal architecture creation failed: {e}")
            return None

    def _create_clustering_targets_for_nas(self, features: np.ndarray,
                                         n_regimes: int) -> np.ndarray:
        """Create synthetic targets for NAS training from clustering data."""
        try:
            # Use KMeans to create initial clustering targets
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            initial_labels = kmeans.fit_predict(features)

            # Convert labels to one-hot encoding for classification training
            targets = np.zeros((len(initial_labels), n_regimes))
            targets[np.arange(len(initial_labels)), initial_labels] = 1

            return targets

        except Exception as e:
            self.logger.warning(f"⚠️ Target creation failed: {e}")
            # Fallback to random targets
            return np.random.randint(0, n_regimes, (features.shape[0], n_regimes))

    def _create_clustering_model_from_nas_architecture(self, architecture: ArchitectureCandidate,
                                                     input_dim: int) -> nn.Module:
        """Create a clustering model from NAS architecture."""
        try:
            class NASClusteringModel(nn.Module):
                def __init__(self, architecture: ArchitectureCandidate, input_dim: int):
                    super().__init__()
                    self.layers = nn.ModuleList()

                    # Add input layer
                    self.layers.append(nn.Linear(input_dim, architecture.layers[0]['units']))

                    # Add hidden layers
                    for i in range(len(architecture.layers) - 1):
                        current_layer = architecture.layers[i]
                        next_layer = architecture.layers[i + 1]

                        self.layers.append(
                            nn.Linear(current_layer['units'], next_layer['units'])
                        )

                        # Add activation
                        if current_layer['activation'] == 'relu':
                            self.layers.append(nn.ReLU())
                        elif current_layer['activation'] == 'tanh':
                            self.layers.append(nn.Tanh())
                        elif current_layer['activation'] == 'swish':
                            self.layers.append(nn.SiLU())

                        # Add dropout
                        if current_layer['dropout'] > 0:
                            self.layers.append(nn.Dropout(current_layer['dropout']))

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    for layer in self.layers:
                        x = layer(x)
                    return x

            model = NASClusteringModel(architecture, input_dim)
            model.total_params = sum(p.numel() for p in model.parameters())

            return model

        except Exception as e:
            self.logger.error(f"❌ Model creation failed: {e}")
            raise

    def _cluster_with_nas_model(self, model: nn.Module, features: np.ndarray,
                              n_regimes: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """Perform clustering using NAS-based model."""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            # Convert to tensors
            features_tensor = torch.FloatTensor(features).to(device)

            # Get embeddings from NAS model
            model.eval()
            with torch.no_grad():
                embeddings = model(features_tensor).cpu().numpy()

            # Perform clustering on learned embeddings
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)

            # Calculate cluster centers
            cluster_centers = kmeans.cluster_centers_

            # Calculate clustering loss (reconstruction + clustering quality)
            clustering_loss = self._calculate_nas_clustering_loss(embeddings, labels, cluster_centers)

            return labels, cluster_centers, clustering_loss

        except Exception as e:
            self.logger.error(f"❌ NAS clustering failed: {e}")
            raise

    def _calculate_nas_clustering_loss(self, embeddings: np.ndarray,
                                     labels: np.ndarray,
                                     cluster_centers: np.ndarray) -> float:
        """Calculate clustering loss for NAS model."""
        try:
            # Calculate reconstruction loss (distance to cluster centers)
            reconstruction_loss = 0.0
            for i, (embedding, label) in enumerate(zip(embeddings, labels)):
                center = cluster_centers[label]
                distance = np.linalg.norm(embedding - center)
                reconstruction_loss += distance ** 2

            reconstruction_loss /= len(embeddings)

            # Calculate separation loss (inter-cluster distances)
            separation_loss = 0.0
            n_clusters = len(cluster_centers)
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                    separation_loss += distance

            separation_loss /= (n_clusters * (n_clusters - 1) / 2) if n_clusters > 1 else 1.0

            # Combined loss
            total_loss = reconstruction_loss - 0.1 * separation_loss  # Encourage separation

            return float(total_loss)

        except Exception as e:
            self.logger.warning(f"⚠️ Loss calculation failed: {e}")
            return 0.0

    def _perform_traditional_clustering(self, features: np.ndarray,
                                      n_regimes: int) -> Dict[str, Any]:
        """Fallback to traditional clustering methods."""
        try:
            features_scaled = self.scaler.fit_transform(features)

            # Use KMeans as fallback
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            cluster_centers = kmeans.cluster_centers_

            return {
                'labels': labels,
                'cluster_centers': cluster_centers,
                'clustering_loss': 0.0,
                'nas_architecture_used': None,
                'clustering_method': 'traditional'
            }

        except Exception as e:
            self.logger.error(f"❌ Traditional clustering failed: {e}")
            raise

    def _generate_architecture_population(self, features: np.ndarray, n_regimes: int) -> list:
        """Generate initial population of architectures for evolutionary search."""
        try:
            population = []
            for _ in range(self.population_size):
                # Generate random architecture
                n_layers = np.random.randint(2, 8)
                layers = []

                for i in range(n_layers):
                    units = np.random.randint(32, 512)
                    activation = np.random.choice(['relu', 'tanh', 'swish'])
                    dropout = np.random.uniform(0.0, 0.5)

                    layers.append({
                        'units': units,
                        'activation': activation,
                        'dropout': dropout
                    })

                # Create architecture candidate
                architecture = ArchitectureCandidate(
                    layers=layers,
                    total_params=sum(layer['units'] for layer in layers),
                    estimated_flops=sum(layer['units'] for layer in layers) * 1000
                )

                population.append(architecture)

            return population

        except Exception as e:
            self.logger.error(f"❌ Population generation failed: {e}")
            return []

    def _evaluate_architecture_fitness(self, architecture: ArchitectureCandidate) -> tuple:
        """Evaluate fitness of architecture for evolutionary algorithm."""
        try:
            # Convert architecture to individual representation
            individual = self._architecture_to_individual(architecture)

            # Evaluate using clustering targets
            clustering_targets = self._create_clustering_targets_for_nas(self.current_features, self.current_n_regimes)
            performance = self._evaluate_architecture_performance(architecture, self.current_features, clustering_targets)

            # Multi-objective fitness
            clustering_quality = performance.get('accuracy', 0.0)
            efficiency = 1.0 / (1.0 + architecture.total_params / 100000)
            robustness = performance.get('robustness_score', 0.5)

            return clustering_quality, efficiency, robustness

        except Exception as e:
            self.logger.warning(f"⚠️ Fitness evaluation failed: {e}")
            return 0.0, 0.0, 0.0

    def _evaluate_architecture_performance(self, architecture: ArchitectureCandidate,
                                          features: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Evaluate architecture performance on clustering task."""
        try:
            # Create model from architecture
            model = self._create_clustering_model_from_nas_architecture(architecture, features.shape[1])

            # Simple evaluation (can be enhanced)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            # Convert to tensors
            features_tensor = torch.FloatTensor(features).to(device)
            targets_tensor = torch.FloatTensor(targets).to(device)

            # Evaluate model
            model.eval()
            with torch.no_grad():
                predictions = model(features_tensor)
                loss = F.mse_loss(predictions, targets_tensor).item()

            # Calculate metrics
            accuracy = 1.0 / (1.0 + loss)  # Convert loss to accuracy-like metric
            efficiency = 1.0 / (1.0 + architecture.total_params / 100000)
            robustness = 0.8  # Placeholder for robustness metric

            return {
                'accuracy': accuracy,
                'efficiency_score': efficiency,
                'robustness_score': robustness,
                'loss': loss
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Architecture evaluation failed: {e}")
            return {'accuracy': 0.0, 'efficiency_score': 0.0, 'robustness_score': 0.0}

    def _architecture_to_individual(self, architecture: ArchitectureCandidate) -> list:
        """Convert architecture to individual for evolutionary algorithm."""
        try:
            individual = []

            # Encode architecture parameters
            for layer in architecture.layers:
                individual.extend([
                    layer['units'],
                    0 if layer['activation'] == 'relu' else 1 if layer['activation'] == 'tanh' else 2,
                    layer['dropout']
                ])

            return individual

        except Exception:
            return []

    def _individual_to_architecture(self, individual: list, input_dim: int) -> ArchitectureCandidate:
        """Convert individual back to architecture."""
        try:
            layers = []
            n_layers = len(individual) // 3

            for i in range(n_layers):
                units = int(individual[i * 3])
                activation_idx = int(individual[i * 3 + 1])
                activation = ['relu', 'tanh', 'swish'][activation_idx]
                dropout = individual[i * 3 + 2]

                layers.append({
                    'units': units,
                    'activation': activation,
                    'dropout': dropout
                })

            return ArchitectureCandidate(
                layers=layers,
                total_params=sum(layer['units'] for layer in layers),
                estimated_flops=sum(layer['units'] for layer in layers) * 1000
            )

        except Exception as e:
            self.logger.error(f"❌ Individual conversion failed: {e}")
            return None

    def _create_architecture_from_params(self, params: dict, input_dim: int) -> ArchitectureCandidate:
        """Create architecture from optimization parameters."""
        try:
            n_layers = params['n_layers']
            n_units = params['n_units']
            dropout = params['dropout']
            activation = params['activation']

            layers = [{
                'units': n_units,
                'activation': activation,
                'dropout': dropout
            } for _ in range(n_layers)]

            return ArchitectureCandidate(
                layers=layers,
                total_params=n_units * n_layers,
                estimated_flops=n_units * n_layers * 1000
            )

        except Exception as e:
            self.logger.error(f"❌ Parameter conversion failed: {e}")
            return None

    def _create_architecture_from_trial(self, trial, input_dim: int) -> ArchitectureCandidate:
        """Create architecture from Optuna trial."""
        try:
            n_layers = trial.params['n_layers']
            n_units = trial.params['n_units']
            dropout = trial.params['dropout']
            activation = trial.params['activation']

            layers = [{
                'units': n_units,
                'activation': activation,
                'dropout': dropout
            } for _ in range(n_layers)]

            return ArchitectureCandidate(
                layers=layers,
                total_params=n_units * n_layers,
                estimated_flops=n_units * n_layers * 1000
            )

        except Exception as e:
            self.logger.error(f"❌ Trial conversion failed: {e}")
            return None

    def _calculate_distances_to_centers(self, features: np.ndarray, labels: np.ndarray, cluster_centers: np.ndarray) -> np.ndarray:
        """Calculate distances from points to their cluster centers."""
        try:
            distances = np.zeros(len(labels))

            for i, (point, label) in enumerate(zip(features, labels)):
                if 0 <= label < len(cluster_centers):
                    center = cluster_centers[label]
                    distance = np.linalg.norm(point - center)
                    distances[i] = distance

            return distances

        except Exception as e:
            self.logger.warning(f"⚠️ Distance calculation failed: {e}")
            return np.zeros(len(labels))

    def _calculate_clustering_quality(self, features: np.ndarray, labels: np.ndarray, n_regimes: int) -> float:
        """Calculate overall clustering quality score."""
        try:
            if len(np.unique(labels)) < 2:
                return 0.0

            silhouette = silhouette_score(features, labels)
            calinski_harabasz = calinski_harabasz_score(features, labels)

            # Combine metrics
            quality = (silhouette + calinski_harabasz) / 2.0
            return min(quality, 1.0)

        except Exception:
            return 0.0
    
    def _perform_nas_clustering(self, feature_result: NASFeatureResult,
                              micro_regime_result: MicroRegimeResult,
                              optimize_parameters: bool) -> Dict[str, Any]:
        """Perform true NAS-driven clustering using neural architecture search."""
        try:
            features = feature_result.features
            if features.size == 0:
                raise ValueError("No features available for clustering")

            self.logger.info("🧠 Using true NAS-based clustering")

            # Use the new NAS clustering implementation
            nas_clustering_result = self._perform_true_nas_clustering(
                features, self.n_regimes, use_nas=True
            )

            # Calculate quality metrics for the NAS clustering result
            quality_metrics = self._calculate_nas_quality_metrics(
                features, nas_clustering_result['labels']
            )

            # Calculate statistics
            statistics = self._calculate_clustering_statistics(
                nas_clustering_result['labels'],
                nas_clustering_result['cluster_centers']
            )

            # Validate clustering
            validation = self._validate_clustering(
                nas_clustering_result['labels'],
                quality_metrics
            )

            # Add NAS-specific information
            quality_metrics['nas_clustering_loss'] = nas_clustering_result['clustering_loss']
            quality_metrics['nas_architecture_score'] = (
                nas_clustering_result['nas_architecture_used'].overall_score
                if nas_clustering_result['nas_architecture_used'] else 0.0
            )

            return {
                'labels': nas_clustering_result['labels'],
                'cluster_centers': nas_clustering_result['cluster_centers'],
                'quality_metrics': quality_metrics,
                'statistics': statistics,
                'validation': validation,
                'clustering_method': nas_clustering_result['clustering_method'],
                'nas_architecture_used': nas_clustering_result['nas_architecture_used']
            }

        except Exception as e:
            self.logger.error(f"❌ NAS clustering failed: {e}")
            # Fallback to traditional clustering
            self.logger.warning("⚠️ Falling back to traditional clustering")
            try:
                features = feature_result.features
                return self._perform_traditional_clustering(features, self.n_regimes)
            except Exception as fallback_error:
                self.logger.error(f"❌ Fallback clustering also failed: {fallback_error}")
                raise e
    
    def _select_clustering_method(self, features: np.ndarray) -> str:
        """Select optimal clustering method based on NAS architecture."""
        try:
            # Use NAS architecture type to select method
            if self.nas_architecture_type == NASArchitectureType.VOLATILITY_FOCUSED:
                return 'kmeans'  # Good for volatility patterns
            elif self.nas_architecture_type == NASArchitectureType.TREND_FOCUSED:
                return 'agglomerative'  # Good for trend patterns
            elif self.nas_architecture_type == NASArchitectureType.VOLUME_FOCUSED:
                return 'dbscan'  # Good for volume patterns
            elif self.nas_architecture_type == NASArchitectureType.MOMENTUM_FOCUSED:
                return 'kmeans'  # Good for momentum patterns
            else:  # HYBRID
                return 'kmeans'  # Default to K-means for hybrid
                
        except Exception as e:
            self.logger.warning(f"⚠️ Clustering method selection failed: {e}")
            return 'kmeans'
    
    def _calculate_cluster_centers(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate cluster centers from labels."""
        try:
            unique_labels = np.unique(labels)
            cluster_centers = []
            
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                cluster_mask = labels == label
                cluster_center = np.mean(features[cluster_mask], axis=0)
                cluster_centers.append(cluster_center)
            
            return np.array(cluster_centers) if cluster_centers else np.array([])
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cluster center calculation failed: {e}")
            return np.array([])
    
    def _calculate_quality_metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        try:
            if len(np.unique(labels)) < 2:
                return {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0}

            # Silhouette score
            silhouette = silhouette_score(features, labels)

            # Calinski-Harabasz score
            calinski_harabasz = calinski_harabasz_score(features, labels)

            # Custom NAS metrics
            nas_score = self._calculate_nas_score(features, labels)

            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'nas_score': nas_score
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Quality metrics calculation failed: {e}")
            return {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0, 'nas_score': 0.0}

    def _calculate_nas_quality_metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate enhanced quality metrics for NAS-based clustering."""
        try:
            if len(np.unique(labels)) < 2:
                return {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0, 'nas_score': 0.0}

            # Standard metrics
            silhouette = silhouette_score(features, labels)
            calinski_harabasz = calinski_harabasz_score(features, labels)

            # Enhanced NAS-specific metrics
            nas_score = self._calculate_nas_score(features, labels)

            # Architecture efficiency metric (if architecture is available)
            architecture_efficiency = self._calculate_architecture_efficiency()

            # Clustering consistency metric
            clustering_consistency = self._calculate_clustering_consistency(labels)

            # Regime separation metric
            regime_separation = self._calculate_regime_separation_score(features, labels)

            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'nas_score': nas_score,
                'architecture_efficiency': architecture_efficiency,
                'clustering_consistency': clustering_consistency,
                'regime_separation': regime_separation,
                'overall_nas_quality': (silhouette + calinski_harabasz + nas_score + architecture_efficiency) / 4.0
            }

        except Exception as e:
            self.logger.warning(f"⚠️ NAS quality metrics calculation failed: {e}")
            return {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0, 'nas_score': 0.0}

    def _calculate_architecture_efficiency(self) -> float:
        """Calculate architecture efficiency based on current NAS architecture."""
        try:
            if self.current_best_architecture:
                # Efficiency based on parameter count and performance
                param_efficiency = 1.0 / (1.0 + self.current_best_architecture.total_params / 100000)
                score_efficiency = self.current_best_architecture.overall_score
                return (param_efficiency + score_efficiency) / 2.0
            else:
                return 0.5  # Default efficiency
        except Exception:
            return 0.5

    def _calculate_clustering_consistency(self, labels: np.ndarray) -> float:
        """Calculate clustering consistency across time."""
        try:
            # Calculate how consistent clustering is over time periods
            if len(labels) < 10:
                return 0.5

            # Split into chunks and measure consistency
            chunk_size = max(10, len(labels) // 5)
            chunks = []

            for i in range(0, len(labels) - chunk_size + 1, chunk_size):
                chunk_labels = labels[i:i + chunk_size]
                unique_in_chunk = len(np.unique(chunk_labels))
                chunks.append(unique_in_chunk)

            # Consistency is higher when chunks have similar regime distributions
            if len(chunks) > 1:
                avg_regimes = np.mean(chunks)
                std_regimes = np.std(chunks)
                consistency = 1.0 / (1.0 + std_regimes / avg_regimes)
                return min(consistency, 1.0)
            else:
                return 0.5

        except Exception:
            return 0.5

    def _calculate_regime_separation_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate how well regimes are separated in feature space."""
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0

            # Calculate inter-cluster vs intra-cluster distances
            inter_distances = []
            intra_distances = []

            for label in unique_labels:
                cluster_mask = labels == label
                cluster_features = features[cluster_mask]

                if len(cluster_features) < 2:
                    continue

                # Intra-cluster distance (mean pairwise distance within cluster)
                from scipy.spatial.distance import pdist
                intra_dist = np.mean(pdist(cluster_features))
                intra_distances.append(intra_dist)

                # Inter-cluster distance (distance to other clusters)
                other_mask = labels != label
                other_features = features[other_mask]

                if len(other_features) > 0:
                    # Distance to nearest other cluster center
                    cluster_center = np.mean(cluster_features, axis=0)
                    other_centers = []

                    for other_label in unique_labels:
                        if other_label != label:
                            other_cluster_mask = labels == other_label
                            other_cluster_features = features[other_cluster_mask]
                            if len(other_cluster_features) > 0:
                                other_center = np.mean(other_cluster_features, axis=0)
                                other_centers.append(other_center)

                    if other_centers:
                        other_centers = np.array(other_centers)
                        distances_to_others = np.linalg.norm(
                            other_centers - cluster_center, axis=1
                        )
                        min_inter_dist = np.min(distances_to_others)
                        inter_distances.append(min_inter_dist)

            if inter_distances and intra_distances:
                avg_intra = np.mean(intra_distances)
                avg_inter = np.mean(inter_distances)
                separation_score = avg_inter / (avg_intra + 1e-8)
                return min(separation_score, 1.0)
            else:
                return 0.0

        except Exception as e:
            self.logger.warning(f"⚠️ Regime separation calculation failed: {e}")
            return 0.0
    
    def _calculate_nas_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate custom NAS score for regime quality."""
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0
            
            # Calculate regime stability
            regime_stability = self._calculate_regime_stability(labels)
            
            # Calculate regime separation
            regime_separation = self._calculate_regime_separation(features, labels)
            
            # Calculate regime consistency
            regime_consistency = self._calculate_regime_consistency(features, labels)
            
            # Combined NAS score
            nas_score = (regime_stability + regime_separation + regime_consistency) / 3.0
            
            return nas_score
            
        except Exception as e:
            self.logger.warning(f"⚠️ NAS score calculation failed: {e}")
            return 0.0
    
    def _calculate_regime_stability(self, labels: np.ndarray) -> float:
        """Calculate regime stability score."""
        try:
            # Calculate regime persistence
            regime_changes = np.sum(np.diff(labels) != 0)
            total_periods = len(labels) - 1
            stability = 1.0 - (regime_changes / total_periods) if total_periods > 0 else 0.0
            
            return stability
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime stability calculation failed: {e}")
            return 0.0
    
    def _calculate_regime_separation(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate regime separation score."""
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0
            
            # Calculate inter-cluster distance
            inter_cluster_distances = []
            for i, label1 in enumerate(unique_labels):
                for j, label2 in enumerate(unique_labels):
                    if i < j:
                        cluster1_mask = labels == label1
                        cluster2_mask = labels == label2
                        
                        if np.any(cluster1_mask) and np.any(cluster2_mask):
                            center1 = np.mean(features[cluster1_mask], axis=0)
                            center2 = np.mean(features[cluster2_mask], axis=0)
                            distance = np.linalg.norm(center1 - center2)
                            inter_cluster_distances.append(distance)
            
            # Calculate intra-cluster distance
            intra_cluster_distances = []
            for label in unique_labels:
                cluster_mask = labels == label
                if np.any(cluster_mask):
                    cluster_features = features[cluster_mask]
                    center = np.mean(cluster_features, axis=0)
                    distances = np.linalg.norm(cluster_features - center, axis=1)
                    intra_cluster_distances.extend(distances)
            
            # Calculate separation ratio
            if inter_cluster_distances and intra_cluster_distances:
                avg_inter = np.mean(inter_cluster_distances)
                avg_intra = np.mean(intra_cluster_distances)
                separation = avg_inter / (avg_intra + 1e-8)
                return min(separation, 1.0)  # Cap at 1.0
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime separation calculation failed: {e}")
            return 0.0
    
    def _calculate_regime_consistency(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate regime consistency score."""
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0
            
            # Calculate consistency within each regime
            consistency_scores = []
            for label in unique_labels:
                cluster_mask = labels == label
                if np.any(cluster_mask):
                    cluster_features = features[cluster_mask]
                    # Calculate feature variance within cluster
                    feature_variance = np.var(cluster_features, axis=0)
                    # Lower variance = higher consistency
                    consistency = 1.0 / (1.0 + np.mean(feature_variance))
                    consistency_scores.append(consistency)
            
            return np.mean(consistency_scores) if consistency_scores else 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime consistency calculation failed: {e}")
            return 0.0
    
    def _calculate_clustering_statistics(self, labels: np.ndarray, 
                                       cluster_centers: np.ndarray) -> Dict[str, Any]:
        """Calculate clustering statistics."""
        try:
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            
            # Regime distribution
            regime_distribution = {}
            for label in unique_labels:
                count = np.sum(labels == label)
                regime_distribution[f'regime_{label}'] = count
            
            # Regime percentages
            regime_percentages = {}
            total_samples = len(labels)
            for label in unique_labels:
                count = np.sum(labels == label)
                percentage = (count / total_samples) * 100
                regime_percentages[f'regime_{label}'] = percentage
            
            return {
                'n_clusters': n_clusters,
                'regime_distribution': regime_distribution,
                'regime_percentages': regime_percentages,
                'total_samples': total_samples,
                'cluster_centers_shape': cluster_centers.shape if cluster_centers.size > 0 else (0, 0)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Clustering statistics calculation failed: {e}")
            return {}
    
    def _validate_clustering(self, labels: np.ndarray, 
                           quality_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Validate clustering results."""
        try:
            # Check minimum quality thresholds
            silhouette_threshold = 0.3
            nas_score_threshold = 0.4
            
            silhouette_valid = quality_metrics.get('silhouette_score', 0.0) >= silhouette_threshold
            nas_score_valid = quality_metrics.get('nas_score', 0.0) >= nas_score_threshold
            
            # Check regime count
            unique_labels = np.unique(labels)
            regime_count_valid = 5 <= len(unique_labels) <= 20
            
            # Overall validation
            is_valid = silhouette_valid and nas_score_valid and regime_count_valid
            
            return {
                'valid': is_valid,
                'silhouette_valid': silhouette_valid,
                'nas_score_valid': nas_score_valid,
                'regime_count_valid': regime_count_valid,
                'n_regimes': len(unique_labels),
                'silhouette_score': quality_metrics.get('silhouette_score', 0.0),
                'nas_score': quality_metrics.get('nas_score', 0.0)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Clustering validation failed: {e}")
            return {'valid': False, 'error': str(e)}
    
    def _calculate_economic_significance_scores(self, data: np.ndarray, 
                                              labels: np.ndarray) -> np.ndarray:
        """Calculate economic significance scores for each regime."""
        try:
            if data.shape[1] < 4:
                return np.zeros(len(labels))
            
            close_price = data[:, 3]
            volume = data[:, 4] if data.shape[1] > 4 else np.ones(len(close_price))
            
            economic_scores = np.zeros(len(labels))
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                regime_mask = labels == label
                if not np.any(regime_mask):
                    continue
                
                regime_close = close_price[regime_mask]
                regime_volume = volume[regime_mask]
                
                # Calculate economic significance based on price movement and volume
                price_change = abs((regime_close[-1] - regime_close[0]) / regime_close[0])
                volume_ratio = np.mean(regime_volume) / np.mean(volume)
                volatility = np.std(regime_close) / np.mean(regime_close)
                
                # Economic significance score
                economic_score = (price_change * volume_ratio * volatility) / 3.0
                economic_scores[regime_mask] = min(economic_score, 1.0)
            
            return economic_scores
            
        except Exception as e:
            self.logger.warning(f"⚠️ Economic significance calculation failed: {e}")
            return np.zeros(len(labels))
    
    def _calculate_trading_viability_scores(self, data: np.ndarray, 
                                          labels: np.ndarray) -> np.ndarray:
        """Calculate trading viability scores for each regime."""
        try:
            if data.shape[1] < 4:
                return np.zeros(len(labels))
            
            close_price = data[:, 3]
            high_price = data[:, 1]
            low_price = data[:, 2]
            volume = data[:, 4] if data.shape[1] > 4 else np.ones(len(close_price))
            
            trading_scores = np.zeros(len(labels))
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                regime_mask = labels == label
                if not np.any(regime_mask):
                    continue
                
                regime_close = close_price[regime_mask]
                regime_high = high_price[regime_mask]
                regime_low = low_price[regime_mask]
                regime_volume = volume[regime_mask]
                
                # Calculate trading viability based on multiple factors
                price_range = (np.max(regime_high) - np.min(regime_low)) / np.mean(regime_close)
                volume_consistency = 1.0 / (1.0 + np.std(regime_volume) / np.mean(regime_volume))
                trend_consistency = self._calculate_trend_consistency(regime_close)
                
                # Trading viability score
                trading_score = (price_range * volume_consistency * trend_consistency) / 3.0
                trading_scores[regime_mask] = min(trading_score, 1.0)
            
            return trading_scores
            
        except Exception as e:
            self.logger.warning(f"⚠️ Trading viability calculation failed: {e}")
            return np.zeros(len(labels))
    
    def _calculate_trend_consistency(self, prices: np.ndarray) -> float:
        """Calculate trend consistency for a price series."""
        try:
            if len(prices) < 3:
                return 0.5
            
            # Calculate price changes
            price_changes = np.diff(prices)
            
            # Calculate trend consistency
            positive_changes = np.sum(price_changes > 0)
            negative_changes = np.sum(price_changes < 0)
            total_changes = len(price_changes)
            
            if total_changes == 0:
                return 0.5
            
            # Consistency is higher when there's a clear trend direction
            consistency = max(positive_changes, negative_changes) / total_changes
            return consistency
            
        except Exception as e:
            self.logger.warning(f"⚠️ Trend consistency calculation failed: {e}")
            return 0.5
    
    def _calculate_regime_transitions(self, labels: np.ndarray) -> np.ndarray:
        """Calculate regime transition probabilities."""
        try:
            unique_labels = np.unique(labels)
            n_regimes = len(unique_labels)
            
            if n_regimes < 2:
                return np.array([])
            
            # Create transition matrix
            transition_matrix = np.zeros((n_regimes, n_regimes))
            
            for i in range(len(labels) - 1):
                current_regime = labels[i]
                next_regime = labels[i + 1]
                
                if current_regime in unique_labels and next_regime in unique_labels:
                    current_idx = np.where(unique_labels == current_regime)[0][0]
                    next_idx = np.where(unique_labels == next_regime)[0][0]
                    transition_matrix[current_idx, next_idx] += 1
            
            # Normalize transition matrix
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = transition_matrix / (row_sums[:, np.newaxis] + 1e-8)
            
            return transition_matrix
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime transition calculation failed: {e}")
            return np.array([])
    
    def _create_nas_result(self, clustering_result: Dict[str, Any],
                          micro_regime_result: MicroRegimeResult,
                          economic_scores: np.ndarray,
                          trading_scores: np.ndarray,
                          regime_transitions: np.ndarray,
                          feature_result: NASFeatureResult,
                          execution_time: float) -> NASClusteringResult:
        """Create NAS clustering result."""
        try:
            from datetime import datetime
            
            # Create base result
            result = NASClusteringResult(
                labels=clustering_result['labels'],
                cluster_centers=clustering_result['cluster_centers'],
                statistics=clustering_result['statistics'],
                quality_metrics=clustering_result['quality_metrics'],
                validation=clustering_result['validation'],
                metadata={
                    'method': 'nas_clustering',
                    'timeframe': self.config.timeframe,
                    'n_regimes': self.n_regimes,
                    'nas_architecture_type': self.nas_architecture_type.value,
                    'micro_regime_detection': self.config.enable_micro_regime_detection,
                    'feature_count': len(feature_result.feature_names),
                    'execution_time': execution_time,
                    'timestamp': datetime.now().isoformat()
                },
                success=clustering_result['validation']['valid'],
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                
                # NAS-specific fields
                nas_architectures=self.nas_architectures,
                micro_regimes=micro_regime_result,
                regime_transitions=regime_transitions,
                economic_significance_scores=economic_scores,
                trading_viability_scores=trading_scores,

                # True NAS fields
                best_nas_candidate=self.current_best_architecture,
                nas_search_results={
                    'search_performed': self.current_best_architecture is not None,
                    'clustering_loss_history': self.clustering_loss_history,
                    'architecture_score': self.current_best_architecture.overall_score if self.current_best_architecture else 0.0,
                    'total_params': self.current_best_architecture.total_params if self.current_best_architecture else 0,
                    'search_time': getattr(self.nas_search, 'search_time', 0.0)
                },
                clustering_loss_history=self.clustering_loss_history
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ NAS result creation failed: {e}")
            return NASClusteringResult(
                labels=np.array([]),
                cluster_centers=np.array([]),
                statistics={},
                quality_metrics={},
                validation={'valid': False, 'error': str(e)},
                metadata={'error': str(e)},
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _generate_nas_report(self, result: NASClusteringResult, 
                           feature_result: NASFeatureResult) -> None:
        """Generate NAS clustering report."""
        try:
            self.logger.info("📊 NAS Clustering Report:")
            self.logger.info(f"   - Method: {result.metadata.get('method', 'nas_clustering')}")
            self.logger.info(f"   - Timeframe: {result.metadata.get('timeframe', '15m')}")
            self.logger.info(f"   - Regimes: {result.metadata.get('n_regimes', 0)}")
            self.logger.info(f"   - Architecture: {result.metadata.get('nas_architecture_type', 'hybrid')}")
            self.logger.info(f"   - Silhouette Score: {result.quality_metrics.get('silhouette_score', 0.0):.4f}")
            self.logger.info(f"   - NAS Score: {result.quality_metrics.get('nas_score', 0.0):.4f}")
            self.logger.info(f"   - Economic Significance: {np.mean(result.economic_significance_scores):.4f}")
            self.logger.info(f"   - Trading Viability: {np.mean(result.trading_viability_scores):.4f}")
            self.logger.info(f"   - Micro-regimes: {len(result.micro_regimes.micro_regime_types) if result.micro_regimes else 0}")
            self.logger.info(f"   - Execution Time: {result.execution_time:.2f}s")
            
        except Exception as e:
            self.logger.warning(f"⚠️ NAS report generation failed: {e}")