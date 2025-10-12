"""
Enhanced Data-Driven Interaction Feature Generator with VectorBT Integration

This module provides a comprehensive data-driven approach to generating
interaction features with full VectorBT optimization integration.

Key Features:
- Data-driven interaction type selection
- Automatic parameter optimization
- VectorBTRollingOptimizer integration for optimized rolling operations
- UnifiedVectorizationManager for comprehensive vectorization
- VectorBTBatchProcessor for large-scale processing
- Advanced performance monitoring and statistics
- Memory-efficient processing with intelligent chunking
- GPU acceleration support
- Parallel processing capabilities
- Intelligent caching and optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
import warnings
from itertools import combinations, product
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import gc

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_quantile, rolling_skew, rolling_kurt
    )
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    rolling_quantile = None
    rolling_skew = None
    rolling_kurt = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Import our VectorBT utilities
try:
    from .vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    from .unified_vectorization_manager import UnifiedVectorizationManager, VectorizationConfig, get_unified_vectorization_manager
    from ..core.vectorbt_batch_processor import VectorBTBatchProcessor, BatchProcessingConfig, BatchProcessor
    VECTORBT_UTILS_AVAILABLE = True
except ImportError:
    VECTORBT_UTILS_AVAILABLE = False
    VectorBTRollingOptimizer = None
    UnifiedVectorizationManager = None
    VectorBTBatchProcessor = None
    logger.warning("VectorBT utilities not available. Some optimizations will be disabled.")

logger = logging.getLogger(__name__)


@dataclass
class InteractionType:
    """Represents a type of interaction feature with enhanced metadata."""
    name: str
    function: Callable
    description: str
    complexity: int  # 1-5 scale
    vectorbt_optimized: bool = True
    parameters: Optional[Dict[str, Any]] = None
    batch_processable: bool = True
    memory_efficient: bool = True
    gpu_accelerated: bool = False


@dataclass
class InteractionResult:
    """Result of interaction feature generation with enhanced metadata."""
    feature_name: str
    feature_series: pd.Series
    parent_features: List[str]
    interaction_type: str
    utility_score: float
    metadata: Dict[str, Any]
    processing_time: float = 0.0
    memory_usage: float = 0.0
    optimization_method: str = "pandas"


@dataclass
class EnhancedInteractionConfig:
    """Enhanced configuration for interaction generation."""
    # Basic settings
    max_interactions: int = 100
    utility_threshold: float = 0.1
    correlation_threshold: float = 0.95
    
    # VectorBT settings
    enable_vectorbt: bool = True
    enable_gpu: bool = False
    enable_parallel: bool = True
    
    # Memory management
    memory_efficient: bool = True
    max_memory_gb: float = 8.0
    chunk_size: int = 1000
    
    # Batch processing
    enable_batch_processing: bool = True
    batch_size: int = 10000
    max_workers: int = None
    
    # Performance monitoring
    enable_monitoring: bool = True
    enable_profiling: bool = False
    enable_caching: bool = True
    cache_size: int = 1000
    
    # Processing optimization
    enable_rolling_optimization: bool = True
    rolling_optimization_threshold: int = 1000
    enable_scaling_optimization: bool = True
    
    def __post_init__(self):
        if not VECTORBT_AVAILABLE:
            self.enable_vectorbt = False
            logger.warning("VectorBT not available, disabling optimizations")
        
        if self.max_workers is None:
            try:
                import multiprocessing as mp
                self.max_workers = min(mp.cpu_count(), 8)
            except:
                self.max_workers = 4


class DataDrivenInteractionGenerator:
    """
    Enhanced data-driven interaction generator with full VectorBT integration.
    
    This class provides comprehensive interaction feature generation with:
    - VectorBTRollingOptimizer for optimized rolling operations
    - UnifiedVectorizationManager for comprehensive vectorization
    - VectorBTBatchProcessor for large-scale processing
    - Advanced performance monitoring and statistics
    - Memory-efficient processing with intelligent chunking
    - GPU acceleration support
    - Parallel processing capabilities
    """
    
    def __init__(self, 
                 max_interactions: int = 100,
                 utility_threshold: float = 0.1,
                 correlation_threshold: float = 0.95,
                 enable_vectorbt: bool = True,
                 config: Optional[EnhancedInteractionConfig] = None):
        """
        Initialize the enhanced data-driven interaction generator.
        
        Args:
            max_interactions: Maximum number of interactions to generate
            utility_threshold: Minimum utility score for feature selection
            correlation_threshold: Maximum correlation for feature filtering
            enable_vectorbt: Whether to use VectorBT optimization
            config: Enhanced configuration (optional)
        """
        # Use provided config or create from parameters
        if config is not None:
            self.config = config
        else:
            self.config = EnhancedInteractionConfig(
                max_interactions=max_interactions,
                utility_threshold=utility_threshold,
                correlation_threshold=correlation_threshold,
                enable_vectorbt=enable_vectorbt
            )
        
        # Initialize VectorBT utilities
        self._initialize_vectorbt_utilities()
        
        # Initialize interaction types
        self.interaction_types = self._initialize_interaction_types()
        
        # Performance tracking
        self.performance_stats = {
            'total_interactions_generated': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'batch_operations': 0,
            'cached_operations': 0,
            'memory_optimizations': 0,
            'total_processing_time': 0.0,
            'average_utility_score': 0.0,
            'memory_savings': 0.0
        }
        
        # Cache for computed results
        self._result_cache = {}
        self._cache_enabled = self.config.enable_caching
        
        logger.info(f"✅ Enhanced data-driven interaction generator initialized")
        logger.info(f"📊 Max interactions: {self.config.max_interactions}")
        logger.info(f"📊 Utility threshold: {self.config.utility_threshold}")
        logger.info(f"📊 VectorBT enabled: {self.config.enable_vectorbt}")
        logger.info(f"📊 GPU enabled: {self.config.enable_gpu}")
        logger.info(f"📊 Batch processing: {self.config.enable_batch_processing}")
    
    def _initialize_vectorbt_utilities(self):
        """Initialize VectorBT utilities."""
        if not VECTORBT_UTILS_AVAILABLE:
            logger.warning("VectorBT utilities not available, using basic implementation")
            self.rolling_optimizer = None
            self.vectorization_manager = None
            self.batch_processor = None
            return
        
        # Initialize VectorBT rolling optimizer
        if self.config.enable_vectorbt:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=self.config.enable_gpu,
                enable_parallel=self.config.enable_parallel,
                memory_efficient=self.config.memory_efficient,
                chunk_size=self.config.chunk_size
            )
        else:
            self.rolling_optimizer = None
        
        # Initialize unified vectorization manager
        if self.config.enable_vectorbt:
            vectorization_config = VectorizationConfig(
                enable_vectorbt=self.config.enable_vectorbt,
                enable_gpu=self.config.enable_gpu,
                enable_parallel=self.config.enable_parallel,
                memory_efficient=self.config.memory_efficient,
                max_memory_gb=self.config.max_memory_gb,
                chunk_size=self.config.chunk_size,
                enable_monitoring=self.config.enable_monitoring,
                enable_profiling=self.config.enable_profiling,
                batch_size=self.config.batch_size,
                enable_batch_processing=self.config.enable_batch_processing,
                rolling_optimization_threshold=self.config.rolling_optimization_threshold,
                enable_rolling_optimization=self.config.enable_rolling_optimization
            )
            self.vectorization_manager = get_unified_vectorization_manager(vectorization_config)
        else:
            self.vectorization_manager = None
        
        # Initialize batch processor
        if self.config.enable_batch_processing and VECTORBT_UTILS_AVAILABLE:
            batch_config = BatchProcessingConfig(
                batch_size=self.config.batch_size,
                enable_gpu=self.config.enable_gpu,
                enable_parallel=self.config.enable_parallel,
                max_memory_gb=self.config.max_memory_gb,
                chunk_size=self.config.chunk_size,
                enable_memory_optimization=self.config.memory_efficient,
                enable_progress_tracking=self.config.enable_monitoring,
                max_workers=self.config.max_workers
            )
            self.batch_processor = VectorBTBatchProcessor(batch_config)
        else:
            self.batch_processor = None
    
    def _initialize_interaction_types(self) -> Dict[str, InteractionType]:
        """Initialize available interaction types."""
        interaction_types = {}
        
        # Basic arithmetic interactions
        interaction_types['product'] = InteractionType(
            name='product',
            function=self._product_interaction,
            description='Multiplication of two features',
            complexity=1,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True
        )
        
        interaction_types['ratio'] = InteractionType(
            name='ratio',
            function=self._ratio_interaction,
            description='Division of two features',
            complexity=1,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True
        )
        
        interaction_types['difference'] = InteractionType(
            name='difference',
            function=self._difference_interaction,
            description='Subtraction of two features',
            complexity=1,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True
        )
        
        interaction_types['sum'] = InteractionType(
            name='sum',
            function=self._sum_interaction,
            description='Addition of two features',
            complexity=1,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True
        )
        
        # Advanced interactions
        interaction_types['correlation'] = InteractionType(
            name='correlation',
            function=self._correlation_interaction,
            description='Rolling correlation between features',
            complexity=3,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True,
            parameters={'window': 20}
        )
        
        interaction_types['covariance'] = InteractionType(
            name='covariance',
            function=self._covariance_interaction,
            description='Rolling covariance between features',
            complexity=3,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True,
            parameters={'window': 20}
        )
        
        interaction_types['zscore_product'] = InteractionType(
            name='zscore_product',
            function=self._zscore_product_interaction,
            description='Product of z-scored features',
            complexity=2,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True
        )
        
        interaction_types['rank_correlation'] = InteractionType(
            name='rank_correlation',
            function=self._rank_correlation_interaction,
            description='Rank correlation between features',
            complexity=3,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True,
            parameters={'window': 20}
        )
        
        # Polynomial interactions
        interaction_types['quadratic'] = InteractionType(
            name='quadratic',
            function=self._quadratic_interaction,
            description='Quadratic transformation of feature',
            complexity=2,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True
        )
        
        
        # Statistical interactions
        interaction_types['skewness'] = InteractionType(
            name='skewness',
            function=self._skewness_interaction,
            description='Rolling skewness of feature',
            complexity=3,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True,
            parameters={'window': 20}
        )
        
        interaction_types['kurtosis'] = InteractionType(
            name='kurtosis',
            function=self._kurtosis_interaction,
            description='Rolling kurtosis of feature',
            complexity=3,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True,
            parameters={'window': 20}
        )
        
        # Momentum interactions
        interaction_types['momentum_divergence'] = InteractionType(
            name='momentum_divergence',
            function=self._momentum_divergence_interaction,
            description='Momentum divergence between features',
            complexity=2,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True
        )
        
        interaction_types['momentum_convergence'] = InteractionType(
            name='momentum_convergence',
            function=self._momentum_convergence_interaction,
            description='Momentum convergence between features',
            complexity=2,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True
        )
        
        # Advanced VectorBT-optimized interactions
        interaction_types['rolling_quantile'] = InteractionType(
            name='rolling_quantile',
            function=self._rolling_quantile_interaction,
            description='Rolling quantile of feature',
            complexity=3,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True,
            parameters={'window': 20, 'q': 0.5}
        )
        
        interaction_types['rolling_rank'] = InteractionType(
            name='rolling_rank',
            function=self._rolling_rank_interaction,
            description='Rolling rank of feature',
            complexity=2,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True,
            parameters={'window': 20}
        )
        
        # Scaled/Normalized interactions
        interaction_types['scaled_sum'] = InteractionType(
            name='scaled_sum',
            function=self._scaled_sum_interaction,
            description='Sum of scaled/normalized features',
            complexity=2,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True
        )
        
        interaction_types['scaled_difference'] = InteractionType(
            name='scaled_difference',
            function=self._scaled_difference_interaction,
            description='Difference of scaled/normalized features',
            complexity=2,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True
        )
        
        interaction_types['scaled_product'] = InteractionType(
            name='scaled_product',
            function=self._scaled_product_interaction,
            description='Product of scaled/normalized features',
            complexity=2,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True
        )
        
        interaction_types['scaled_ratio'] = InteractionType(
            name='scaled_ratio',
            function=self._scaled_ratio_interaction,
            description='Ratio of scaled/normalized features',
            complexity=2,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True
        )
        
        interaction_types['log_scaled_product'] = InteractionType(
            name='log_scaled_product',
            function=self._log_scaled_product_interaction,
            description='Log of scaled product features',
            complexity=3,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True
        )
        
        interaction_types['log_scaled_sum'] = InteractionType(
            name='log_scaled_sum',
            function=self._log_scaled_sum_interaction,
            description='Log of scaled sum features',
            complexity=3,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True
        )
        
        interaction_types['minmax_scaled_product'] = InteractionType(
            name='minmax_scaled_product',
            function=self._minmax_scaled_product_interaction,
            description='Product of min-max scaled features',
            complexity=2,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True
        )
        
        interaction_types['robust_scaled_difference'] = InteractionType(
            name='robust_scaled_difference',
            function=self._robust_scaled_difference_interaction,
            description='Difference of robust scaled features',
            complexity=2,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True
        )
        
        # Log multiplication interactions
        interaction_types['log_product'] = InteractionType(
            name='log_product',
            function=self._log_product_interaction,
            description='Log of product of features',
            complexity=3,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True
        )
        
        interaction_types['log_scaled_product'] = InteractionType(
            name='log_scaled_product',
            function=self._log_scaled_product_interaction,
            description='Log of scaled product features',
            complexity=3,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True
        )
        
        interaction_types['log_ratio'] = InteractionType(
            name='log_ratio',
            function=self._log_ratio_interaction,
            description='Log of ratio of features',
            complexity=3,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True
        )
        
        interaction_types['log_abs_product'] = InteractionType(
            name='log_abs_product',
            function=self._log_abs_product_interaction,
            description='Log of absolute product of features',
            complexity=3,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True
        )
        
        interaction_types['log_sqrt_product'] = InteractionType(
            name='log_sqrt_product',
            function=self._log_sqrt_product_interaction,
            description='Log of square root of product of features',
            complexity=3,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True
        )
        
        return interaction_types
    
    def generate_interactions(self, 
                            features: pd.DataFrame,
                            targets: Optional[pd.Series] = None) -> List[InteractionResult]:
        """
        Generate interaction features using enhanced data-driven approach.
        
        Args:
            features: Input features DataFrame
            targets: Target variable (optional)
            
        Returns:
            List of generated interaction results
        """
        logger.info(f"🚀 Starting enhanced data-driven interaction generation")
        logger.info(f"📊 Input features: {features.shape}")
        
        start_time = time.time()
        
        # Optimize input data
        if self.vectorization_manager:
            features = self.vectorization_manager.optimize_dataframe(features)
        
        # Analyze data characteristics
        data_characteristics = self._analyze_data_characteristics(features)
        
        # Select optimal interaction types
        selected_types = self._select_interaction_types(data_characteristics)
        
        # Generate feature combinations
        feature_combinations = self._generate_feature_combinations(features.columns.tolist())
        
        # Process interactions
        if (self.config.enable_batch_processing and 
            self.batch_processor and 
            len(feature_combinations) > self.config.batch_size):
            interactions = self._generate_interactions_batch(
                features, feature_combinations, selected_types, targets
            )
        else:
            interactions = self._generate_interactions_sequential(
                features, feature_combinations, selected_types, targets
            )
        
        # Filter and rank interactions
        filtered_interactions = self._filter_interactions(interactions, targets)
        ranked_interactions = self._rank_interactions(filtered_interactions, targets)
        
        # Select top interactions
        selected_interactions = ranked_interactions[:self.config.max_interactions]
        
        # Update performance stats
        execution_time = time.time() - start_time
        self.performance_stats['total_processing_time'] += execution_time
        self.performance_stats['total_interactions_generated'] += len(selected_interactions)
        
        if selected_interactions:
            self.performance_stats['average_utility_score'] = np.mean([
                i.utility_score for i in selected_interactions
            ])
        
        logger.info(f"✅ Generated {len(selected_interactions)} interactions in {execution_time:.2f}s")
        logger.info(f"📊 Average utility score: {self.performance_stats['average_utility_score']:.3f}")
        
        return selected_interactions
    
    def _generate_interactions_batch(self, 
                                   features: pd.DataFrame,
                                   feature_combinations: List[Tuple[str, ...]],
                                   selected_types: List[str],
                                   targets: Optional[pd.Series]) -> List[InteractionResult]:
        """Generate interactions using batch processing."""
        logger.info(f"🔄 Using batch processing for {len(feature_combinations)} combinations")
        
        # Create batch processors for each interaction type
        batch_processors = []
        for interaction_type_name in selected_types:
            interaction_type = self.interaction_types[interaction_type_name]
            if interaction_type.batch_processable:
                processor = InteractionBatchProcessor(
                    interaction_type, features, targets, self.vectorization_manager
                )
                batch_processors.append(processor)
        
        # Process in batches
        results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(feature_combinations), batch_size):
            batch_combinations = feature_combinations[i:i + batch_size]
            
            # Process batch
            batch_results = self.batch_processor.process_features_batch(
                features, batch_processors, **{'combinations': batch_combinations}
            )
            
            if not batch_results.empty:
                # Convert batch results to InteractionResult objects
                for _, row in batch_results.iterrows():
                    if 'feature_name' in row and 'utility_score' in row:
                        result = InteractionResult(
                            feature_name=row['feature_name'],
                            feature_series=row['feature_series'],
                            parent_features=row['parent_features'],
                            interaction_type=row['interaction_type'],
                            utility_score=row['utility_score'],
                            metadata=row.get('metadata', {}),
                            processing_time=row.get('processing_time', 0.0),
                            memory_usage=row.get('memory_usage', 0.0),
                            optimization_method=row.get('optimization_method', 'batch')
                        )
                        results.append(result)
            
            # Memory cleanup
            if i % (batch_size * 5) == 0:
                gc.collect()
        
        return results
    
    def _generate_interactions_sequential(self, 
                                        features: pd.DataFrame,
                                        feature_combinations: List[Tuple[str, ...]],
                                        selected_types: List[str],
                                        targets: Optional[pd.Series]) -> List[InteractionResult]:
        """Generate interactions using sequential processing."""
        logger.info(f"🔄 Using sequential processing for {len(feature_combinations)} combinations")
        
        interactions = []
        
        for interaction_type_name in selected_types:
            interaction_type = self.interaction_types[interaction_type_name]
            
            for combo in feature_combinations:
                try:
                    result = self._generate_single_interaction(
                        features, combo, interaction_type, targets
                    )
                    if result:
                        interactions.append(result)
                except Exception as e:
                    logger.warning(f"⚠️ Interaction generation failed: {e}")
                    continue
        
        return interactions
    
    def _analyze_data_characteristics(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data characteristics to inform interaction selection."""
        characteristics = {}
        
        # Basic statistics
        characteristics['n_features'] = len(features.columns)
        characteristics['n_samples'] = len(features)
        characteristics['data_types'] = features.dtypes.value_counts().to_dict()
        
        # Correlation analysis using VectorBT optimization
        if (self.config.enable_rolling_optimization and 
            self.vectorization_manager and 
            len(features) > self.config.rolling_optimization_threshold):
            corr_matrix = self.vectorization_manager.rolling_operation(
                features, 'corr', window=min(20, len(features))
            )
        else:
            corr_matrix = features.corr()
        
        characteristics['avg_correlation'] = corr_matrix.abs().mean().mean()
        characteristics['max_correlation'] = corr_matrix.abs().max().max()
        
        # Variance analysis
        characteristics['feature_variance'] = features.var().to_dict()
        characteristics['avg_variance'] = features.var().mean()
        
        # Skewness and kurtosis
        characteristics['feature_skewness'] = features.skew().to_dict()
        characteristics['feature_kurtosis'] = features.kurtosis().to_dict()
        
        # Missing values
        characteristics['missing_values'] = features.isnull().sum().to_dict()
        
        return characteristics
    
    def _select_interaction_types(self, characteristics: Dict[str, Any]) -> List[str]:
        """Select optimal interaction types based on data characteristics."""
        selected_types = []
        
        # Always include basic arithmetic interactions
        selected_types.extend(['product', 'ratio', 'difference', 'sum'])
        
        # Add correlation-based interactions if features are not highly correlated
        if characteristics['avg_correlation'] < 0.7:
            selected_types.extend(['correlation', 'covariance', 'rank_correlation'])
        
        # Add statistical interactions if data has sufficient variance
        if characteristics['avg_variance'] > 0.01:
            selected_types.extend(['skewness', 'kurtosis'])
        
        # Add polynomial interactions for non-normal distributions
        avg_skewness = np.mean([abs(s) for s in characteristics['feature_skewness'].values()])
        if avg_skewness > 0.5:
            selected_types.extend(['quadratic'])
        
        # Add scaled interactions for better normalization
        selected_types.extend(['scaled_sum', 'scaled_difference', 'scaled_product', 'scaled_ratio'])
        
        # Add advanced scaled interactions for complex patterns
        if characteristics['avg_variance'] > 0.01:
            selected_types.extend(['log_scaled_product', 'log_scaled_sum', 'minmax_scaled_product', 'robust_scaled_difference'])
        
        # Add log multiplication interactions for non-linear patterns
        if characteristics['avg_variance'] > 0.01:
            selected_types.extend(['log_product', 'log_ratio', 'log_abs_product', 'log_sqrt_product'])
        
        # Add momentum interactions for time series data
        if characteristics['n_samples'] > 50:
            selected_types.extend(['momentum_divergence', 'momentum_convergence'])
        
        # Add z-score interactions for normalization
        selected_types.append('zscore_product')
        
        return list(set(selected_types))  # Remove duplicates
    
    def _generate_feature_combinations(self, feature_names: List[str]) -> List[Tuple[str, ...]]:
        """Generate feature combinations for interactions."""
        combinations_list = []
        
        # Single feature interactions (polynomial, statistical)
        for feature in feature_names:
            combinations_list.append((feature,))
        
        # Two feature interactions
        for combo in combinations(feature_names, 2):
            combinations_list.append(combo)
        
        # Three feature interactions (limited)
        if len(feature_names) <= 10:  # Only for small feature sets
            for combo in combinations(feature_names, 3):
                combinations_list.append(combo)
        
        return combinations_list
    
    def _generate_single_interaction(self, 
                                   features: pd.DataFrame,
                                   feature_combo: Tuple[str, ...],
                                   interaction_type: InteractionType,
                                   targets: Optional[pd.Series]) -> Optional[InteractionResult]:
        """Generate a single interaction feature with enhanced optimization and early termination."""
        start_time = time.time()
        
        try:
            # Check cache first
            if self._cache_enabled:
                cache_key = self._generate_cache_key(feature_combo, interaction_type.name)
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    self.performance_stats['cached_operations'] += 1
                    return cached_result
            
            # Early termination: Check if we already have enough high-quality interactions
            if hasattr(self, '_current_interaction_count') and hasattr(self, '_high_quality_threshold'):
                if (self._current_interaction_count >= self.config.max_interactions and 
                    self._high_quality_interactions >= self._high_quality_threshold):
                    return None
            
            # Extract feature data with early validation
            feature_data = []
            for feat in feature_combo:
                if feat not in features.columns:
                    return None
                series = features[feat]
                
                # Early termination: Skip if feature has insufficient data
                if series.isna().sum() > len(series) * 0.5:  # More than 50% missing
                    return None
                
                # Early termination: Skip if feature has no variance
                if series.nunique() <= 1:
                    return None
                
                feature_data.append(series)
            
            # Early termination: Check feature correlation for multi-feature interactions
            if len(feature_combo) > 1:
                corr_matrix = pd.DataFrame(feature_data).T.corr()
                max_corr = corr_matrix.abs().max().max()
                if max_corr > self.config.correlation_threshold:
                    return None  # Features too correlated
            
            # Generate interaction with optimized calculation
            result_series = self._generate_interaction_optimized(
                feature_data, interaction_type, feature_combo
            )
            
            if result_series is None or result_series.empty:
                return None
            
            # Early termination: Quick utility check before expensive calculations
            quick_utility = self._calculate_quick_utility_score(result_series, targets)
            if quick_utility < self.config.utility_threshold * 0.8:  # 20% buffer for early termination
                return None
            
            # Calculate full utility score
            utility_score = self._calculate_utility_score(result_series, targets)
            
            if utility_score < self.config.utility_threshold:
                return None
            
            # Early termination: Check if result is too similar to existing interactions
            if self._is_duplicate_interaction(result_series, utility_score):
                return None
            
            # Create feature name
            feature_name = f"{interaction_type.name}_{'_'.join(feature_combo)}"
            
            # Calculate processing metrics
            processing_time = time.time() - start_time
            memory_usage = result_series.memory_usage(deep=True) / (1024 * 1024)  # MB
            
            result = InteractionResult(
                feature_name=feature_name,
                feature_series=result_series,
                parent_features=list(feature_combo),
                interaction_type=interaction_type.name,
                utility_score=utility_score,
                metadata={
                    'complexity': interaction_type.complexity,
                    'vectorbt_optimized': interaction_type.vectorbt_optimized,
                    'batch_processable': interaction_type.batch_processable,
                    'memory_efficient': interaction_type.memory_efficient,
                    'parameters': interaction_type.parameters,
                    'quick_utility': quick_utility,
                    'early_termination_checks': True
                },
                processing_time=processing_time,
                memory_usage=memory_usage,
                optimization_method='vectorbt' if interaction_type.vectorbt_optimized else 'pandas'
            )
            
            # Update counters for early termination
            if not hasattr(self, '_current_interaction_count'):
                self._current_interaction_count = 0
                self._high_quality_interactions = 0
                self._high_quality_threshold = self.config.max_interactions * 0.7
            
            self._current_interaction_count += 1
            if utility_score > self.config.utility_threshold * 1.5:  # High quality threshold
                self._high_quality_interactions += 1
            
            # Cache result
            if self._cache_enabled:
                self._put_in_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.debug(f"⚠️ Single interaction generation failed: {e}")
            return None
    
    def _generate_interaction_optimized(self, 
                                       feature_data: List[pd.Series],
                                       interaction_type: InteractionType,
                                       feature_combo: Tuple[str, ...]) -> Optional[pd.Series]:
        """Generate interaction with optimized calculations and early termination."""
        try:
            # Early termination: Check for obvious failures
            if not feature_data or any(series is None for series in feature_data):
                return None
            
            # Use VectorBT optimization when available
            if (interaction_type.vectorbt_optimized and 
                self.vectorization_manager and 
                len(feature_data) <= 2):  # Most interactions are 1-2 features
                
                return self._generate_vectorbt_optimized_interaction(
                    feature_data, interaction_type, feature_combo
                )
            
            # Fallback to standard generation
            if len(feature_combo) == 1:
                return interaction_type.function(feature_data[0])
            else:
                return interaction_type.function(*feature_data)
                
        except Exception as e:
            logger.debug(f"⚠️ Optimized interaction generation failed: {e}")
            return None
    
    def _generate_vectorbt_optimized_interaction(self, 
                                               feature_data: List[pd.Series],
                                               interaction_type: InteractionType,
                                               feature_combo: Tuple[str, ...]) -> Optional[pd.Series]:
        """Generate interaction using VectorBT optimization."""
        try:
            if len(feature_data) == 1:
                # Single feature interaction
                feat = feature_data[0]
                
                if interaction_type.name == 'quadratic':
                    return feat ** 2
                elif interaction_type.name == 'skewness':
                    return self.vectorization_manager.rolling_operation(feat, 'skew', 20)
                elif interaction_type.name == 'kurtosis':
                    return self.vectorization_manager.rolling_operation(feat, 'kurt', 20)
                elif interaction_type.name == 'rolling_quantile':
                    q = interaction_type.parameters.get('q', 0.5) if interaction_type.parameters else 0.5
                    return self.vectorization_manager.rolling_operation(feat, 'quantile', 20, q=q)
                elif interaction_type.name == 'rolling_rank':
                    return self.vectorization_manager.rolling_operation(feat, 'rank', 20)
                else:
                    return interaction_type.function(feat)
            
            elif len(feature_data) == 2:
                # Two feature interaction
                feat1, feat2 = feature_data
                
                if interaction_type.name in ['scaled_sum', 'scaled_difference', 'scaled_product', 'scaled_ratio']:
                    # Use optimized scaling
                    scaled1 = self.vectorization_manager.scale_data(feat1, method='zscore')
                    scaled2 = self.vectorization_manager.scale_data(feat2, method='zscore')
                    
                    if interaction_type.name == 'scaled_sum':
                        return scaled1 + scaled2
                    elif interaction_type.name == 'scaled_difference':
                        return scaled1 - scaled2
                    elif interaction_type.name == 'scaled_product':
                        return scaled1 * scaled2
                    elif interaction_type.name == 'scaled_ratio':
                        return scaled1 / (scaled2 + 1e-08)
                
                elif interaction_type.name in ['correlation', 'covariance']:
                    # Use optimized rolling operations
                    window = interaction_type.parameters.get('window', 20) if interaction_type.parameters else 20
                    if interaction_type.name == 'correlation':
                        return self.vectorization_manager.rolling_operation(feat1, 'corr', window, other=feat2)
                    else:
                        return self.vectorization_manager.rolling_operation(feat1, 'cov', window, other=feat2)
                
                elif interaction_type.name in ['log_product', 'log_ratio', 'log_abs_product', 'log_sqrt_product']:
                    # Use optimized log operations
                    if interaction_type.name == 'log_product':
                        product = feat1 * feat2
                        return np.log(np.abs(product) + 1e-08)
                    elif interaction_type.name == 'log_ratio':
                        ratio = feat1 / (feat2 + 1e-08)
                        return np.log(np.abs(ratio) + 1e-08)
                    elif interaction_type.name == 'log_abs_product':
                        abs_product = np.abs(feat1 * feat2)
                        return np.log(abs_product + 1e-08)
                    elif interaction_type.name == 'log_sqrt_product':
                        product = feat1 * feat2
                        sqrt_product = np.sqrt(np.abs(product))
                        return np.log(sqrt_product + 1e-08)
                
                else:
                    return interaction_type.function(feat1, feat2)
            
            else:
                # Multi-feature interaction (fallback)
                return interaction_type.function(*feature_data)
                
        except Exception as e:
            logger.debug(f"⚠️ VectorBT optimized interaction failed: {e}")
            return None
    
    def _calculate_quick_utility_score(self, series: pd.Series, targets: Optional[pd.Series]) -> float:
        """Calculate a quick utility score for early termination."""
        try:
            # Quick variance check
            variance = series.var()
            if pd.isna(variance) or variance < self.config.utility_threshold * 0.1:
                return 0.0
            
            # Quick correlation check (sample-based for speed)
            if targets is not None and len(series) > 100:
                # Sample every 10th point for speed
                sample_indices = series.dropna().index[::10]
                if len(sample_indices) > 10:
                    sample_series = series.loc[sample_indices]
                    sample_targets = targets.loc[sample_indices]
                    corr = sample_series.corr(sample_targets)
                    if not pd.isna(corr):
                        return abs(corr)
            
            # Fallback to variance-based score
            return min(1.0, variance)
            
        except:
            return 0.0
    
    def _is_duplicate_interaction(self, series: pd.Series, utility_score: float) -> bool:
        """Check if interaction is too similar to existing ones."""
        if not hasattr(self, '_interaction_signatures'):
            self._interaction_signatures = {}
        
        # Create a simple signature based on statistics
        signature = (
            float(series.mean()),
            float(series.std()),
            float(series.skew()),
            len(series.dropna())
        )
        
        # Check against existing signatures
        for existing_sig, existing_score in self._interaction_signatures.items():
            if (abs(signature[0] - existing_sig[0]) < 1e-6 and  # Mean
                abs(signature[1] - existing_sig[1]) < 1e-6 and  # Std
                abs(signature[2] - existing_sig[2]) < 1e-6 and  # Skew
                signature[3] == existing_sig[3]):  # Length
                # If similar signature and similar score, consider duplicate
                if abs(utility_score - existing_score) < 0.01:
                    return True
        
        # Store this signature
        self._interaction_signatures[signature] = utility_score
        return False
    
    def _calculate_utility_score(self, series: pd.Series, targets: Optional[pd.Series]) -> float:
        """Calculate utility score for a feature."""
        try:
            if targets is None:
                # Use variance as utility score
                return float(series.var())
            
            # Calculate correlation with targets
            correlation = series.corr(targets)
            if pd.isna(correlation):
                return 0.0
            
            # Use absolute correlation as utility score
            return abs(correlation)
            
        except Exception as e:
            logger.debug(f"⚠️ Utility score calculation failed: {e}")
            return 0.0
    
    def _filter_interactions(self, 
                           interactions: List[InteractionResult],
                           targets: Optional[pd.Series]) -> List[InteractionResult]:
        """Filter interactions based on quality criteria."""
        filtered = []
        
        for interaction in interactions:
            # Check utility threshold
            if interaction.utility_score < self.utility_threshold:
                continue
            
            # Check for valid values
            if interaction.feature_series.isna().all():
                continue
            
            # Check for infinite values
            if np.isinf(interaction.feature_series).any():
                continue
            
            # Check for constant values
            if interaction.feature_series.nunique() <= 1:
                continue
            
            filtered.append(interaction)
        
        return filtered
    
    def _rank_interactions(self, 
                         interactions: List[InteractionResult],
                         targets: Optional[pd.Series]) -> List[InteractionResult]:
        """Rank interactions by utility score."""
        return sorted(interactions, key=lambda x: x.utility_score, reverse=True)
    
    # Interaction type implementations
    def _product_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Product interaction."""
        return feat1 * feat2
    
    def _ratio_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Ratio interaction."""
        return feat1 / (feat2 + 1e-08)
    
    def _difference_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Difference interaction."""
        return feat1 - feat2
    
    def _sum_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Sum interaction."""
        return feat1 + feat2
    
    def _correlation_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Correlation interaction using VectorBT rolling optimizer."""
        window = 20
        if self.vectorization_manager:
            return self.vectorization_manager.rolling_operation(
                feat1, 'corr', window, other=feat2
            )
        elif self.enable_vectorbt and VECTORBT_AVAILABLE:
            return rolling_corr(feat1, feat2, window=window)
        else:
            return feat1.rolling(window=window).corr(feat2)
    
    def _covariance_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Covariance interaction using VectorBT rolling optimizer."""
        window = 20
        if self.vectorization_manager:
            return self.vectorization_manager.rolling_operation(
                feat1, 'cov', window, other=feat2
            )
        elif self.enable_vectorbt and VECTORBT_AVAILABLE:
            return rolling_cov(feat1, feat2, window=window)
        else:
            return feat1.rolling(window=window).cov(feat2)
    
    def _zscore_product_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Z-score product interaction using VectorBT scaling."""
        if self.vectorization_manager:
            z1 = self.vectorization_manager.scale_data(feat1, method='zscore')
            z2 = self.vectorization_manager.scale_data(feat2, method='zscore')
        elif self.enable_vectorbt and VECTORBT_AVAILABLE:
            z1 = zscore(feat1)
            z2 = zscore(feat2)
        else:
            z1 = (feat1 - feat1.mean()) / feat1.std()
            z2 = (feat2 - feat2.mean()) / feat2.std()
        
        return z1 * z2
    
    def _rank_correlation_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Rank correlation interaction using VectorBT operations."""
        window = 20
        if self.vectorization_manager:
            rank1 = self.vectorization_manager.scale_data(feat1, method='rank')
            rank2 = self.vectorization_manager.scale_data(feat2, method='rank')
            return self.vectorization_manager.rolling_operation(
                rank1, 'corr', window, other=rank2
            )
        elif self.enable_vectorbt and VECTORBT_AVAILABLE:
            rank1 = rank(feat1)
            rank2 = rank(feat2)
            return rolling_corr(rank1, rank2, window=window)
        else:
            rank1 = feat1.rank()
            rank2 = feat2.rank()
            return rank1.rolling(window=window).corr(rank2)
    
    def _quadratic_interaction(self, feat: pd.Series) -> pd.Series:
        """Quadratic interaction."""
        return feat ** 2
    
    
    def _skewness_interaction(self, feat: pd.Series) -> pd.Series:
        """Skewness interaction using VectorBT rolling optimizer."""
        window = 20
        if self.vectorization_manager:
            return self.vectorization_manager.rolling_operation(feat, 'skew', window)
        elif self.enable_vectorbt and VECTORBT_AVAILABLE:
            return rolling_skew(feat, window=window)
        else:
            return feat.rolling(window=window).skew()
    
    def _kurtosis_interaction(self, feat: pd.Series) -> pd.Series:
        """Kurtosis interaction using VectorBT rolling optimizer."""
        window = 20
        if self.vectorization_manager:
            return self.vectorization_manager.rolling_operation(feat, 'kurt', window)
        elif self.enable_vectorbt and VECTORBT_AVAILABLE:
            return rolling_kurt(feat, window=window)
        else:
            return feat.rolling(window=window).kurt()
    
    def _momentum_divergence_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Momentum divergence interaction."""
        momentum1 = feat1.pct_change()
        momentum2 = feat2.pct_change()
        return momentum1 - momentum2
    
    def _momentum_convergence_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Momentum convergence interaction."""
        momentum1 = feat1.pct_change()
        momentum2 = feat2.pct_change()
        return momentum1 * momentum2
    
    # Scaled/Normalized interaction implementations
    def _scaled_sum_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Sum of scaled/normalized features using VectorBT scaling."""
        if self.vectorization_manager:
            scaled1 = self.vectorization_manager.scale_data(feat1, method='zscore')
            scaled2 = self.vectorization_manager.scale_data(feat2, method='zscore')
        elif self.config.enable_vectorbt and VECTORBT_AVAILABLE:
            scaled1 = zscore(feat1)
            scaled2 = zscore(feat2)
        else:
            scaled1 = (feat1 - feat1.mean()) / feat1.std()
            scaled2 = (feat2 - feat2.mean()) / feat2.std()
        
        return scaled1 + scaled2
    
    def _scaled_difference_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Difference of scaled/normalized features using VectorBT scaling."""
        if self.vectorization_manager:
            scaled1 = self.vectorization_manager.scale_data(feat1, method='zscore')
            scaled2 = self.vectorization_manager.scale_data(feat2, method='zscore')
        elif self.config.enable_vectorbt and VECTORBT_AVAILABLE:
            scaled1 = zscore(feat1)
            scaled2 = zscore(feat2)
        else:
            scaled1 = (feat1 - feat1.mean()) / feat1.std()
            scaled2 = (feat2 - feat2.mean()) / feat2.std()
        
        return scaled1 - scaled2
    
    def _scaled_product_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Product of scaled/normalized features using VectorBT scaling."""
        if self.vectorization_manager:
            scaled1 = self.vectorization_manager.scale_data(feat1, method='zscore')
            scaled2 = self.vectorization_manager.scale_data(feat2, method='zscore')
        elif self.config.enable_vectorbt and VECTORBT_AVAILABLE:
            scaled1 = zscore(feat1)
            scaled2 = zscore(feat2)
        else:
            scaled1 = (feat1 - feat1.mean()) / feat1.std()
            scaled2 = (feat2 - feat2.mean()) / feat2.std()
        
        return scaled1 * scaled2
    
    def _scaled_ratio_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Ratio of scaled/normalized features using VectorBT scaling."""
        if self.vectorization_manager:
            scaled1 = self.vectorization_manager.scale_data(feat1, method='zscore')
            scaled2 = self.vectorization_manager.scale_data(feat2, method='zscore')
        elif self.config.enable_vectorbt and VECTORBT_AVAILABLE:
            scaled1 = zscore(feat1)
            scaled2 = zscore(feat2)
        else:
            scaled1 = (feat1 - feat1.mean()) / feat1.std()
            scaled2 = (feat2 - feat2.mean()) / feat2.std()
        
        return scaled1 / (scaled2 + 1e-08)
    
    def _log_scaled_product_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Log of scaled product features using VectorBT scaling."""
        if self.vectorization_manager:
            scaled1 = self.vectorization_manager.scale_data(feat1, method='zscore')
            scaled2 = self.vectorization_manager.scale_data(feat2, method='zscore')
        elif self.config.enable_vectorbt and VECTORBT_AVAILABLE:
            scaled1 = zscore(feat1)
            scaled2 = zscore(feat2)
        else:
            scaled1 = (feat1 - feat1.mean()) / feat1.std()
            scaled2 = (feat2 - feat2.mean()) / feat2.std()
        
        product = scaled1 * scaled2
        # Add small constant to avoid log(0)
        return np.log(np.abs(product) + 1e-08)
    
    def _log_scaled_sum_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Log of scaled sum features using VectorBT scaling."""
        if self.vectorization_manager:
            scaled1 = self.vectorization_manager.scale_data(feat1, method='zscore')
            scaled2 = self.vectorization_manager.scale_data(feat2, method='zscore')
        elif self.config.enable_vectorbt and VECTORBT_AVAILABLE:
            scaled1 = zscore(feat1)
            scaled2 = zscore(feat2)
        else:
            scaled1 = (feat1 - feat1.mean()) / feat1.std()
            scaled2 = (feat2 - feat2.mean()) / feat2.std()
        
        sum_val = scaled1 + scaled2
        # Add small constant to avoid log(0)
        return np.log(np.abs(sum_val) + 1e-08)
    
    def _minmax_scaled_product_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Product of min-max scaled features using VectorBT scaling."""
        if self.vectorization_manager:
            scaled1 = self.vectorization_manager.scale_data(feat1, method='minmax')
            scaled2 = self.vectorization_manager.scale_data(feat2, method='minmax')
        elif self.config.enable_vectorbt and VECTORBT_AVAILABLE:
            # Min-max scaling: (x - min) / (max - min)
            scaled1 = (feat1 - feat1.min()) / (feat1.max() - feat1.min())
            scaled2 = (feat2 - feat2.min()) / (feat2.max() - feat2.min())
        else:
            scaled1 = (feat1 - feat1.min()) / (feat1.max() - feat1.min())
            scaled2 = (feat2 - feat2.min()) / (feat2.max() - feat2.min())
        
        return scaled1 * scaled2
    
    def _robust_scaled_difference_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Difference of robust scaled features using VectorBT scaling."""
        if self.vectorization_manager:
            scaled1 = self.vectorization_manager.scale_data(feat1, method='robust')
            scaled2 = self.vectorization_manager.scale_data(feat2, method='robust')
        elif self.config.enable_vectorbt and VECTORBT_AVAILABLE:
            # Robust scaling: (x - median) / IQR
            q1_1, q3_1 = feat1.quantile([0.25, 0.75])
            q1_2, q3_2 = feat2.quantile([0.25, 0.75])
            iqr_1 = q3_1 - q1_1
            iqr_2 = q3_2 - q1_2
            scaled1 = (feat1 - feat1.median()) / (iqr_1 + 1e-08)
            scaled2 = (feat2 - feat2.median()) / (iqr_2 + 1e-08)
        else:
            q1_1, q3_1 = feat1.quantile([0.25, 0.75])
            q1_2, q3_2 = feat2.quantile([0.25, 0.75])
            iqr_1 = q3_1 - q1_1
            iqr_2 = q3_2 - q1_2
            scaled1 = (feat1 - feat1.median()) / (iqr_1 + 1e-08)
            scaled2 = (feat2 - feat2.median()) / (iqr_2 + 1e-08)
        
        return scaled1 - scaled2
    
    # Log multiplication interaction implementations
    def _log_product_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Log of product of features."""
        product = feat1 * feat2
        # Add small constant to avoid log(0) and handle negative values
        return np.log(np.abs(product) + 1e-08)
    
    def _log_ratio_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Log of ratio of features."""
        ratio = feat1 / (feat2 + 1e-08)
        # Add small constant to avoid log(0) and handle negative values
        return np.log(np.abs(ratio) + 1e-08)
    
    def _log_abs_product_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Log of absolute product of features."""
        abs_product = np.abs(feat1 * feat2)
        # Add small constant to avoid log(0)
        return np.log(abs_product + 1e-08)
    
    def _log_sqrt_product_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Log of square root of product of features."""
        product = feat1 * feat2
        # Ensure non-negative for square root
        sqrt_product = np.sqrt(np.abs(product))
        # Add small constant to avoid log(0)
        return np.log(sqrt_product + 1e-08)
    
    # Cache management methods
    def _generate_cache_key(self, feature_combo: Tuple[str, ...], interaction_type: str) -> str:
        """Generate cache key for operation."""
        import hashlib
        combo_str = '_'.join(sorted(feature_combo))
        return hashlib.md5(f"{interaction_type}_{combo_str}".encode()).hexdigest()[:16]
    
    def _get_from_cache(self, cache_key: str) -> Optional[InteractionResult]:
        """Get result from cache."""
        if not self._cache_enabled:
            return None
        return self._result_cache.get(cache_key)
    
    def _put_in_cache(self, cache_key: str, result: InteractionResult):
        """Put result in cache."""
        if not self._cache_enabled:
            return
        
        # Limit cache size
        if len(self._result_cache) >= self.config.cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self._result_cache))
            del self._result_cache[oldest_key]
        
        self._result_cache[cache_key] = result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add VectorBT utilities stats
        if self.rolling_optimizer:
            rolling_stats = self.rolling_optimizer.get_performance_stats()
            stats.update(rolling_stats)
        
        if self.vectorization_manager:
            vectorization_stats = self.vectorization_manager.get_performance_stats()
            stats.update(vectorization_stats)
        
        stats.update({
            'cache_hit_rate': (stats['cached_operations'] / max(stats['total_interactions_generated'], 1)) * 100,
            'average_processing_time_per_interaction': (
                stats['total_processing_time'] / max(stats['total_interactions_generated'], 1)
            )
        })
        
        return stats
    
    def reset_stats(self):
        """Reset all performance statistics."""
        self.performance_stats = {
            'total_interactions_generated': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'batch_operations': 0,
            'cached_operations': 0,
            'memory_optimizations': 0,
            'total_processing_time': 0.0,
            'average_utility_score': 0.0,
            'memory_savings': 0.0
        }
        
        if self.rolling_optimizer:
            self.rolling_optimizer.reset_stats()
        if self.vectorization_manager:
            self.vectorization_manager.reset_stats()
        self._result_cache.clear()


class InteractionBatchProcessor(BatchProcessor):
    """Batch processor for interaction generation."""
    
    def __init__(self, interaction_type: InteractionType, features: pd.DataFrame, 
                 targets: Optional[pd.Series], vectorization_manager: Optional[UnifiedVectorizationManager]):
        self.interaction_type = interaction_type
        self.features = features
        self.targets = targets
        self.vectorization_manager = vectorization_manager
    
    def process_batch(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Process a batch of feature combinations."""
        combinations = kwargs.get('combinations', [])
        results = []
        
        for combo in combinations:
            try:
                # Extract feature data
                feature_data = [self.features[feat] for feat in combo]
                
                # Generate interaction
                if len(combo) == 1:
                    result_series = self.interaction_type.function(feature_data[0])
                else:
                    result_series = self.interaction_type.function(*feature_data)
                
                if result_series is not None and not result_series.empty:
                    # Calculate utility score
                    utility_score = self._calculate_utility_score(result_series, self.targets)
                    
                    if utility_score >= 0.1:  # Basic threshold
                        feature_name = f"{self.interaction_type.name}_{'_'.join(combo)}"
                        
                        results.append({
                            'feature_name': feature_name,
                            'feature_series': result_series,
                            'parent_features': list(combo),
                            'interaction_type': self.interaction_type.name,
                            'utility_score': utility_score,
                            'metadata': {
                                'complexity': self.interaction_type.complexity,
                                'vectorbt_optimized': self.interaction_type.vectorbt_optimized
                            },
                            'processing_time': 0.0,
                            'memory_usage': result_series.memory_usage(deep=True) / (1024 * 1024),
                            'optimization_method': 'batch'
                        })
            except Exception as e:
                logger.warning(f"Batch processing failed for {combo}: {e}")
                continue
        
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    def _calculate_utility_score(self, series: pd.Series, targets: Optional[pd.Series]) -> float:
        """Calculate utility score for a feature."""
        try:
            if targets is None:
                return float(series.var())
            
            correlation = series.corr(targets)
            if pd.isna(correlation):
                return 0.0
            
            return abs(correlation)
        except:
            return 0.0
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for processing."""
        return list(self.features.columns)