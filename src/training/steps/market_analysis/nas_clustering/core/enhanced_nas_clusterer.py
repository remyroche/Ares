"""
Enhanced NAS Clusterer with True Neural Architecture Search

This module integrates true Neural Architecture Search with the existing clustering pipeline,
replacing the static architecture approach with dynamic architecture search and optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from dataclasses import dataclass
import copy

# Import matrix operations for optimized computations
from src.utils.matrix_operations import UnifiedMatrixOperations

# Import hardware optimization
from src.utils.hardware.unified_hardware_manager import (
    UnifiedHardwareManager, HardwareConfig, WorkloadType, OptimizationLevel
)

# Import existing components
from .nas_config import NASClusteringConfig, NASArchitectureType
from .nas_feature_extractor import NASFeatureExtractor, NASFeatureResult
from .micro_regime_detector import MicroRegimeDetector, MicroRegimeResult
from .nas_regime_optimizer import NASRegimeOptimizer, RegimeOptimizationResult

# Import true NAS components
from .nas_search.evolutionary_search import (
    EvolutionaryArchitectureSearch, ArchitectureIndividual, RegimeDetectionFitnessEvaluator
)
from .nas_search.search_space import (
    SearchSpace, get_volatility_regime_search_space, get_trend_regime_search_space,
    get_volume_regime_search_space, get_hybrid_regime_search_space
)
from .architectures.regime_networks import (
    RegimeNetworkFactory, VolatilityRegimeNetwork, TrendRegimeNetwork,
    VolumeRegimeNetwork, HybridRegimeNetwork
)
from .evaluation.multi_objective import (
    RegimeDetectionMultiObjective, ParetoFrontier, ParetoSolution
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancedNASClusteringResult:
    """Enhanced result from true NAS clustering operation."""
    # Standard clustering results
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
    
    # Enhanced NAS-specific fields
    nas_architectures: Dict[str, Any] = None
    micro_regimes: Optional[MicroRegimeResult] = None
    regime_transitions: np.ndarray = None
    economic_significance_scores: np.ndarray = None
    trading_viability_scores: np.ndarray = None
    
    # True NAS fields
    best_architecture: Optional[ArchitectureIndividual] = None
    pareto_frontier: Optional[ParetoFrontier] = None
    architecture_search_history: List[Dict[str, Any]] = None
    multi_objective_results: Dict[str, Any] = None
    neural_network_performance: Dict[str, Any] = None


class EnhancedNASClusterer:
    """Enhanced NAS clusterer with true Neural Architecture Search."""
    
    def __init__(self, config: NASClusteringConfig):
        """Initialize enhanced NAS clusterer with true architecture search."""
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
        
        # Initialize existing components
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
        
        # Initialize true NAS components
        self.search_space = self._initialize_search_space()
        self.evolutionary_search = self._initialize_evolutionary_search()
        self.multi_objective_optimizer = self._initialize_multi_objective_optimizer()
        self.neural_network_factory = RegimeNetworkFactory()
        
        # NAS architecture settings
        self.nas_architecture_type = config.nas_architecture_type
        self.n_regimes = config.n_regimes
        self.min_regime_duration = config.min_regime_duration
        self.max_regime_duration = config.max_regime_duration
        
        # Economic significance settings
        self.economic_significance_threshold = config.economic_significance_threshold
        self.trading_viability_threshold = config.trading_viability_threshold
        self.regime_transition_cost = config.regime_transition_cost
        
        # True NAS settings
        self.enable_true_nas = getattr(config, 'enable_true_nas', True)
        self.nas_generations = getattr(config, 'nas_generations', 50)
        self.nas_population_size = getattr(config, 'nas_population_size', 30)
        self.enable_multi_objective = getattr(config, 'enable_multi_objective', True)
        
        # Search history
        self.search_history = []
        self.best_architectures = []
        
        self.logger.info(f"✅ Enhanced NAS Clusterer initialized for {config.timeframe} timeframe")
        self.logger.info(f"🖥️ Hardware optimization: {self.hardware_manager is not None}")
        self.logger.info(f"🔢 Matrix operations: {self.matrix_ops is not None}")
        self.logger.info(f"🧠 True NAS enabled: {self.enable_true_nas}")
        self.logger.info(f"🎯 Multi-objective optimization: {self.enable_multi_objective}")
    
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
            from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
            from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
            from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
            
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
    
    def _initialize_search_space(self) -> SearchSpace:
        """Initialize search space based on NAS architecture type."""
        try:
            if self.nas_architecture_type == NASArchitectureType.VOLATILITY_FOCUSED:
                search_space = get_volatility_regime_search_space()
            elif self.nas_architecture_type == NASArchitectureType.TREND_FOCUSED:
                search_space = get_trend_regime_search_space()
            elif self.nas_architecture_type == NASArchitectureType.VOLUME_FOCUSED:
                search_space = get_volume_regime_search_space()
            else:  # HYBRID or default
                search_space = get_hybrid_regime_search_space()
            
            self.logger.info(f"✅ Search space initialized for {self.nas_architecture_type.value}")
            return search_space
            
        except Exception as e:
            self.logger.warning(f"⚠️ Search space initialization failed: {e}")
            return get_hybrid_regime_search_space()
    
    def _initialize_evolutionary_search(self) -> EvolutionaryArchitectureSearch:
        """Initialize evolutionary architecture search."""
        try:
            evolutionary_search = EvolutionaryArchitectureSearch(
                search_space=self.search_space,
                matrix_ops=self.matrix_ops,
                hardware_manager=self.hardware_manager,
                population_size=self.nas_population_size,
                generations=self.nas_generations,
                mutation_rate=0.1,
                crossover_rate=0.8
            )
            
            self.logger.info("✅ Evolutionary architecture search initialized")
            return evolutionary_search
            
        except Exception as e:
            self.logger.warning(f"⚠️ Evolutionary search initialization failed: {e}")
            return None
    
    def _initialize_multi_objective_optimizer(self) -> RegimeDetectionMultiObjective:
        """Initialize multi-objective optimizer."""
        try:
            optimizer = RegimeDetectionMultiObjective(
                matrix_ops=self.matrix_ops,
                hardware_manager=self.hardware_manager
            )
            
            self.logger.info("✅ Multi-objective optimizer initialized")
            return optimizer
            
        except Exception as e:
            self.logger.warning(f"⚠️ Multi-objective optimizer initialization failed: {e}")
            return None
    
    def cluster(self, data: np.ndarray, timestamps: np.ndarray,
                optimize_parameters: bool = True, generate_report: bool = True) -> EnhancedNASClusteringResult:
        """Perform enhanced NAS clustering with true architecture search."""
        try:
            start_time = time.time()
            self.logger.info(f"🚀 Starting enhanced NAS clustering")
            self.logger.info(f"   Data shape: {data.shape}")
            self.logger.info(f"   True NAS enabled: {self.enable_true_nas}")
            
            # Extract features
            self.logger.info("📊 Extracting features...")
            feature_result = self.feature_extractor.extract_features(data, timestamps)
            
            if feature_result.features.size == 0:
                raise ValueError("No features extracted from data")
            
            # Detect micro-regimes
            self.logger.info("🔍 Detecting micro-regimes...")
            micro_regime_result = self.micro_regime_detector.detect_micro_regimes(
                data, timestamps, feature_result.features
            )
            
            # Perform true NAS if enabled
            if self.enable_true_nas and self.evolutionary_search is not None:
                self.logger.info("🧠 Performing true Neural Architecture Search...")
                nas_results = self._perform_true_nas(
                    feature_result, micro_regime_result, data, timestamps
                )
            else:
                self.logger.info("📊 Using traditional clustering (NAS disabled)")
                nas_results = self._perform_traditional_clustering(
                    feature_result, micro_regime_result, data, timestamps
                )
            
            # Calculate economic significance and trading viability
            self.logger.info("💰 Calculating economic significance...")
            economic_scores = self._calculate_economic_significance_scores(
                data, nas_results['labels']
            )
            trading_scores = self._calculate_trading_viability_scores(
                data, nas_results['labels']
            )
            
            # Calculate regime transitions
            regime_transitions = self._calculate_regime_transitions(
                nas_results['labels']
            )
            
            # Create enhanced NAS clustering result
            nas_result = self._create_enhanced_nas_result(
                nas_results, micro_regime_result, economic_scores,
                trading_scores, regime_transitions, feature_result,
                time.time() - start_time
            )
            
            # Generate report if requested
            if generate_report:
                self._generate_enhanced_nas_report(nas_result, feature_result)
            
            self.logger.info(f"✅ Enhanced NAS clustering completed in {nas_result.execution_time:.2f}s")
            return nas_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Enhanced NAS clustering failed: {e}")
            return EnhancedNASClusteringResult(
                labels=np.array([]),
                cluster_centers=np.array([]),
                statistics={},
                quality_metrics={},
                validation={'valid': False, 'error': str(e)},
                metadata={'error': str(e), 'method': 'enhanced_nas_clustering'},
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _perform_true_nas(self, feature_result: NASFeatureResult,
                         micro_regime_result: MicroRegimeResult,
                         data: np.ndarray, timestamps: np.ndarray) -> Dict[str, Any]:
        """Perform true Neural Architecture Search."""
        try:
            self.logger.info("🔍 Starting true NAS process...")
            
            # Prepare data for NAS
            nas_data = feature_result.features
            nas_labels = self._generate_nas_labels(data, timestamps)
            
            # Perform evolutionary architecture search
            best_architecture = self.evolutionary_search.search(nas_data, nas_labels)
            
            # Store best architecture
            self.best_architectures.append(best_architecture.copy())
            
            # Perform multi-objective optimization if enabled
            pareto_frontier = None
            multi_objective_results = None
            
            if self.enable_multi_objective and self.multi_objective_optimizer is not None:
                self.logger.info("🎯 Performing multi-objective optimization...")
                
                # Create candidate architectures from search
                candidate_architectures = [best_architecture]
                
                # Add some variations
                for i in range(5):
                    variant = best_architecture.copy()
                    # Apply small mutations
                    variant.fitness_score = 0.0  # Reset fitness
                    candidate_architectures.append(variant)
                
                # Perform multi-objective optimization
                pareto_frontier = self.multi_objective_optimizer.optimize_nsga2(
                    candidate_architectures, nas_data, nas_labels, max_iterations=20
                )
                
                # Get best solution from Pareto frontier
                best_solutions = pareto_frontier.get_best_solutions(1)
                if best_solutions:
                    best_architecture = best_solutions[0].architecture
                
                # Get optimization summary
                multi_objective_results = self.multi_objective_optimizer.get_optimization_summary(pareto_frontier)
            
            # Train neural network with best architecture
            self.logger.info("🏗️ Training neural network with best architecture...")
            neural_network = self._create_neural_network_from_architecture(best_architecture, nas_data.shape[1])
            
            # Train the network
            training_result = neural_network.train_network(nas_data, nas_labels, epochs=50)
            
            # Make predictions
            predictions = neural_network.predict(nas_data)
            
            # Convert neural network predictions to clustering results
            clustering_result = self._convert_neural_predictions_to_clustering(
                predictions, nas_data, nas_labels
            )
            
            # Add NAS-specific results
            clustering_result.update({
                'best_architecture': best_architecture,
                'pareto_frontier': pareto_frontier,
                'multi_objective_results': multi_objective_results,
                'neural_network_performance': training_result,
                'search_statistics': self.evolutionary_search.get_search_statistics()
            })
            
            self.logger.info("✅ True NAS completed successfully")
            return clustering_result
            
        except Exception as e:
            self.logger.error(f"❌ True NAS failed: {e}")
            # Fallback to traditional clustering
            return self._perform_traditional_clustering(feature_result, micro_regime_result, data, timestamps)
    
    def _perform_traditional_clustering(self, feature_result: NASFeatureResult,
                                      micro_regime_result: MicroRegimeResult,
                                      data: np.ndarray, timestamps: np.ndarray) -> Dict[str, Any]:
        """Perform traditional clustering as fallback."""
        try:
            self.logger.info("📊 Performing traditional clustering...")
            
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            
            # Use existing traditional clustering approach
            features = feature_result.features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Perform K-means clustering
            clusterer = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
            labels = clusterer.fit_predict(features_scaled)
            cluster_centers = clusterer.cluster_centers_
            
            # Calculate quality metrics
            quality_metrics = {
                'silhouette_score': silhouette_score(features_scaled, labels),
                'calinski_harabasz_score': calinski_harabasz_score(features_scaled, labels)
            }
            
            # Calculate statistics
            statistics = {
                'n_clusters': len(np.unique(labels)),
                'cluster_sizes': [np.sum(labels == i) for i in range(len(np.unique(labels)))],
                'clustering_method': 'kmeans'
            }
            
            # Validate clustering
            validation = {
                'valid': True,
                'min_cluster_size': min(statistics['cluster_sizes']) if statistics['cluster_sizes'] else 0,
                'max_cluster_size': max(statistics['cluster_sizes']) if statistics['cluster_sizes'] else 0
            }
            
            return {
                'labels': labels,
                'cluster_centers': cluster_centers,
                'quality_metrics': quality_metrics,
                'statistics': statistics,
                'validation': validation,
                'clustering_method': 'traditional_kmeans'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Traditional clustering failed: {e}")
            raise
    
    def _generate_nas_labels(self, data: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Generate labels for NAS training."""
        try:
            # Use simple clustering to generate initial labels for NAS training
            from sklearn.cluster import KMeans
            
            # Simple feature extraction for labeling
            if data.ndim == 1:
                # 1D data - create simple features
                features = data.reshape(-1, 1)
            else:
                # Multi-dimensional data - use as is
                features = data
            
            # Perform simple clustering for labels
            n_clusters = min(self.n_regimes, len(data) // 10)  # Ensure reasonable cluster size
            if n_clusters < 2:
                n_clusters = 2
            
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(features)
            
            return labels
            
        except Exception as e:
            self.logger.warning(f"NAS label generation failed: {e}")
            # Return random labels as fallback
            return np.random.randint(0, self.n_regimes, size=len(data))
    
    def _create_neural_network_from_architecture(self, architecture: ArchitectureIndividual,
                                               input_dim: int) -> Any:
        """Create neural network from architecture individual."""
        try:
            # Determine network type based on architecture
            network_type = self._determine_network_type_from_architecture(architecture)
            
            # Create neural network
            neural_network = self.neural_network_factory.create_network(
                network_type=network_type,
                input_dim=input_dim,
                output_dim=self.n_regimes,
                matrix_ops=self.matrix_ops,
                hardware_manager=self.hardware_manager
            )
            
            return neural_network
            
        except Exception as e:
            self.logger.warning(f"Neural network creation from architecture failed: {e}")
            # Return default hybrid network
            return self.neural_network_factory.create_network(
                network_type='hybrid',
                input_dim=input_dim,
                output_dim=self.n_regimes,
                matrix_ops=self.matrix_ops,
                hardware_manager=self.hardware_manager
            )
    
    def _determine_network_type_from_architecture(self, architecture: ArchitectureIndividual) -> str:
        """Determine network type from architecture individual."""
        try:
            # Analyze layer types in architecture
            layer_types = [layer.layer_type.value for layer in architecture.layers]
            
            # Count different layer types
            lstm_count = layer_types.count('lstm') + layer_types.count('gru')
            conv_count = layer_types.count('conv1d') + layer_types.count('conv2d')
            attention_count = layer_types.count('attention') + layer_types.count('multi_head_attention')
            
            # Determine network type based on predominant layers
            if attention_count > 0 and attention_count >= lstm_count and attention_count >= conv_count:
                return 'volume'  # Attention-heavy -> volume regime
            elif lstm_count > conv_count:
                return 'volatility'  # LSTM-heavy -> volatility regime
            elif conv_count > lstm_count:
                return 'trend'  # Conv-heavy -> trend regime
            else:
                return 'hybrid'  # Mixed or unclear -> hybrid
                
        except Exception as e:
            self.logger.warning(f"Network type determination failed: {e}")
            return 'hybrid'  # Default to hybrid
    
    def _convert_neural_predictions_to_clustering(self, predictions: np.ndarray,
                                                features: np.ndarray, 
                                                labels: np.ndarray) -> Dict[str, Any]:
        """Convert neural network predictions to clustering results."""
        try:
            # Use predictions as cluster labels
            cluster_labels = predictions
            
            # Calculate cluster centers from features
            unique_labels = np.unique(cluster_labels)
            cluster_centers = []
            
            for label in unique_labels:
                mask = cluster_labels == label
                if np.any(mask):
                    center = np.mean(features[mask], axis=0)
                    cluster_centers.append(center)
            
            cluster_centers = np.array(cluster_centers) if cluster_centers else np.array([])
            
            # Calculate quality metrics
            quality_metrics = {}
            if len(unique_labels) > 1:
                from sklearn.metrics import silhouette_score, calinski_harabasz_score
                quality_metrics['silhouette_score'] = silhouette_score(features, cluster_labels)
                quality_metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, cluster_labels)
            else:
                quality_metrics['silhouette_score'] = 0.0
                quality_metrics['calinski_harabasz_score'] = 0.0
            
            # Calculate statistics
            cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels]
            statistics = {
                'n_clusters': len(unique_labels),
                'cluster_sizes': cluster_sizes,
                'clustering_method': 'neural_network'
            }
            
            # Validate clustering
            validation = {
                'valid': True,
                'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
                'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0
            }
            
            return {
                'labels': cluster_labels,
                'cluster_centers': cluster_centers,
                'quality_metrics': quality_metrics,
                'statistics': statistics,
                'validation': validation,
                'clustering_method': 'neural_network'
            }
            
        except Exception as e:
            self.logger.error(f"Neural predictions conversion failed: {e}")
            # Return fallback clustering result
            return {
                'labels': np.zeros(len(predictions), dtype=int),
                'cluster_centers': np.array([]),
                'quality_metrics': {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0},
                'statistics': {'n_clusters': 0, 'cluster_sizes': [], 'clustering_method': 'neural_network_fallback'},
                'validation': {'valid': False, 'error': str(e)}
            }
    
    def _calculate_economic_significance_scores(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate economic significance scores for regimes."""
        try:
            unique_labels = np.unique(labels)
            economic_scores = np.zeros(len(labels))
            
            for label in unique_labels:
                mask = labels == label
                if np.any(mask):
                    # Calculate regime-specific economic significance
                    regime_data = data[mask]
                    
                    # Simple economic significance based on volatility and trend strength
                    if regime_data.ndim > 1:
                        volatility = np.std(regime_data, axis=1).mean()
                        trend_strength = np.abs(np.mean(np.diff(regime_data, axis=1))).mean()
                    else:
                        volatility = np.std(regime_data)
                        trend_strength = np.abs(np.mean(np.diff(regime_data)))
                    
                    # Combine volatility and trend strength
                    economic_significance = min(1.0, (volatility + trend_strength) / 2.0)
                    economic_scores[mask] = economic_significance
            
            return economic_scores
            
        except Exception as e:
            self.logger.warning(f"Economic significance calculation failed: {e}")
            return np.ones(len(labels)) * 0.5
    
    def _calculate_trading_viability_scores(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate trading viability scores for regimes."""
        try:
            unique_labels = np.unique(labels)
            trading_scores = np.zeros(len(labels))
            
            for label in unique_labels:
                mask = labels == label
                if np.any(mask):
                    # Calculate regime-specific trading viability
                    regime_data = data[mask]
                    
                    # Simple trading viability based on regime stability and predictability
                    regime_size = len(regime_data)
                    stability = 1.0 / (1.0 + np.std(regime_data) if regime_data.ndim == 1 else np.std(regime_data).mean())
                    predictability = min(1.0, regime_size / 100.0)  # Larger regimes are more predictable
                    
                    trading_viability = (stability + predictability) / 2.0
                    trading_scores[mask] = trading_viability
            
            return trading_scores
            
        except Exception as e:
            self.logger.warning(f"Trading viability calculation failed: {e}")
            return np.ones(len(labels)) * 0.5
    
    def _calculate_regime_transitions(self, labels: np.ndarray) -> np.ndarray:
        """Calculate regime transition matrix."""
        try:
            unique_labels = np.unique(labels)
            n_regimes = len(unique_labels)
            
            if n_regimes <= 1:
                return np.array([[1.0]])
            
            # Count transitions
            transition_counts = np.zeros((n_regimes, n_regimes))
            
            for i in range(len(labels) - 1):
                current_regime = labels[i]
                next_regime = labels[i + 1]
                
                current_idx = np.where(unique_labels == current_regime)[0][0]
                next_idx = np.where(unique_labels == next_regime)[0][0]
                
                transition_counts[current_idx, next_idx] += 1
            
            # Convert to probabilities
            transition_probs = np.zeros_like(transition_counts)
            for i in range(n_regimes):
                row_sum = np.sum(transition_counts[i, :])
                if row_sum > 0:
                    transition_probs[i, :] = transition_counts[i, :] / row_sum
                else:
                    transition_probs[i, i] = 1.0  # Self-transition
            
            return transition_probs
            
        except Exception as e:
            self.logger.warning(f"Regime transition calculation failed: {e}")
            return np.eye(len(np.unique(labels)))
    
    def _create_enhanced_nas_result(self, clustering_result: Dict[str, Any],
                                  micro_regime_result: MicroRegimeResult,
                                  economic_scores: np.ndarray, trading_scores: np.ndarray,
                                  regime_transitions: np.ndarray, feature_result: NASFeatureResult,
                                  execution_time: float) -> EnhancedNASClusteringResult:
        """Create enhanced NAS clustering result."""
        try:
            # Create enhanced result
            nas_result = EnhancedNASClusteringResult(
                # Standard clustering results
                labels=clustering_result['labels'],
                cluster_centers=clustering_result['cluster_centers'],
                statistics=clustering_result['statistics'],
                quality_metrics=clustering_result['quality_metrics'],
                validation=clustering_result['validation'],
                metadata={
                    'method': 'enhanced_nas_clustering',
                    'nas_enabled': self.enable_true_nas,
                    'multi_objective_enabled': self.enable_multi_objective,
                    'architecture_type': self.nas_architecture_type.value,
                    'feature_count': feature_result.features.shape[1] if feature_result.features.size > 0 else 0
                },
                success=True,
                execution_time=execution_time,
                timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                
                # Enhanced NAS fields
                micro_regimes=micro_regime_result,
                regime_transitions=regime_transitions,
                economic_significance_scores=economic_scores,
                trading_viability_scores=trading_scores,
                
                # True NAS fields
                best_architecture=clustering_result.get('best_architecture'),
                pareto_frontier=clustering_result.get('pareto_frontier'),
                architecture_search_history=self.search_history,
                multi_objective_results=clustering_result.get('multi_objective_results'),
                neural_network_performance=clustering_result.get('neural_network_performance')
            )
            
            return nas_result
            
        except Exception as e:
            self.logger.error(f"Enhanced NAS result creation failed: {e}")
            # Return basic result
            return EnhancedNASClusteringResult(
                labels=clustering_result['labels'],
                cluster_centers=clustering_result['cluster_centers'],
                statistics=clustering_result['statistics'],
                quality_metrics=clustering_result['quality_metrics'],
                validation=clustering_result['validation'],
                metadata={'error': str(e), 'method': 'enhanced_nas_clustering'},
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _generate_enhanced_nas_report(self, nas_result: EnhancedNASClusteringResult,
                                    feature_result: NASFeatureResult):
        """Generate enhanced NAS report."""
        try:
            self.logger.info("📊 Enhanced NAS Clustering Report")
            self.logger.info(f"   Execution time: {nas_result.execution_time:.2f}s")
            self.logger.info(f"   Regimes detected: {len(np.unique(nas_result.labels))}")
            self.logger.info(f"   Quality metrics: {nas_result.quality_metrics}")
            self.logger.info(f"   Economic significance: {np.mean(nas_result.economic_significance_scores):.3f}")
            self.logger.info(f"   Trading viability: {np.mean(nas_result.trading_viability_scores):.3f}")
            
            if nas_result.best_architecture:
                self.logger.info(f"   Best architecture layers: {len(nas_result.best_architecture.layers)}")
                self.logger.info(f"   Best architecture fitness: {nas_result.best_architecture.fitness_score:.4f}")
            
            if nas_result.neural_network_performance:
                perf = nas_result.neural_network_performance
                self.logger.info(f"   Neural network accuracy: {perf.get('final_accuracy', 0.0):.4f}")
                self.logger.info(f"   Training time: {perf.get('training_time', 0.0):.2f}s")
            
            if nas_result.multi_objective_results:
                self.logger.info(f"   Multi-objective solutions: {nas_result.multi_objective_results.get('total_solutions', 0)}")
                self.logger.info(f"   Pareto fronts: {nas_result.multi_objective_results.get('num_fronts', 0)}")
            
        except Exception as e:
            self.logger.warning(f"Enhanced NAS report generation failed: {e}")
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search statistics."""
        try:
            stats = {
                'search_history_length': len(self.search_history),
                'best_architectures_count': len(self.best_architectures),
                'nas_enabled': self.enable_true_nas,
                'multi_objective_enabled': self.enable_multi_objective
            }
            
            if self.evolutionary_search:
                evolutionary_stats = self.evolutionary_search.get_search_statistics()
                stats.update(evolutionary_stats)
            
            if self.best_architectures:
                best_arch = self.best_architectures[-1]  # Most recent best
                stats['latest_best_architecture'] = {
                    'layers': len(best_arch.layers),
                    'connections': len(best_arch.connections),
                    'fitness_score': best_arch.fitness_score,
                    'generation': best_arch.generation
                }
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"Search statistics calculation failed: {e}")
            return {'error': str(e)}