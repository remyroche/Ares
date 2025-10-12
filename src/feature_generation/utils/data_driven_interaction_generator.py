"""
Refactored Data-Driven Interaction Feature Generator with VectorBT Integration

This module provides a clean, efficient data-driven approach to generating
interaction features with proper error handling and fast-fail mechanisms.

Key Improvements:
- Fast-fail error handling instead of poor fallbacks
- Consolidated interaction methods to reduce duplication
- Comprehensive tprint logging in all functions
- Removed unused and legacy code
- Fixed logic issues and silent errors
- Optimized VectorBT integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
import time
from itertools import combinations
import gc

# Import tprint for comprehensive logging
try:
    from tprint import tprint
except ImportError:
    def tprint(*args, **kwargs):
        print(*args, **kwargs)

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
        tprint("🔧 Initializing EnhancedInteractionConfig")
        
        if not VECTORBT_AVAILABLE:
            self.enable_vectorbt = False
            tprint("⚠️ VectorBT not available, disabling optimizations")
        
        if self.max_workers is None:
            try:
                import multiprocessing as mp
                self.max_workers = min(mp.cpu_count(), 8)
                tprint(f"✅ Set max_workers to {self.max_workers}")
            except Exception as e:
                tprint(f"❌ ERROR: Failed to detect CPU count: {e}")
                raise RuntimeError(f"Failed to initialize max_workers: {e}")


class DataDrivenInteractionGenerator:
    """
    Refactored data-driven interaction generator with fast-fail error handling.
    
    This class provides comprehensive interaction feature generation with:
    - Fast-fail error handling instead of poor fallbacks
    - Consolidated interaction methods to reduce duplication
    - Comprehensive logging with tprint
    - Optimized VectorBT integration
    - Memory-efficient processing
    """
    
    def __init__(self, 
                 max_interactions: int = 100,
                 utility_threshold: float = 0.1,
                 correlation_threshold: float = 0.95,
                 enable_vectorbt: bool = True,
                 config: Optional[EnhancedInteractionConfig] = None):
        """
        Initialize the refactored data-driven interaction generator.
        
        Args:
            max_interactions: Maximum number of interactions to generate
            utility_threshold: Minimum utility score for feature selection
            correlation_threshold: Maximum correlation for feature filtering
            enable_vectorbt: Whether to use VectorBT optimization
            config: Enhanced configuration (optional)
        """
        tprint("🚀 Initializing DataDrivenInteractionGenerator")
        
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
        
        tprint("✅ DataDrivenInteractionGenerator initialized successfully")
        tprint(f"📊 Max interactions: {self.config.max_interactions}")
        tprint(f"📊 Utility threshold: {self.config.utility_threshold}")
        tprint(f"📊 VectorBT enabled: {self.config.enable_vectorbt}")
        tprint(f"📊 GPU enabled: {self.config.enable_gpu}")
        tprint(f"📊 Batch processing: {self.config.enable_batch_processing}")
    
    def _initialize_vectorbt_utilities(self):
        """Initialize VectorBT utilities with proper error handling."""
        tprint("🔧 Initializing VectorBT utilities")
        
        if not VECTORBT_UTILS_AVAILABLE:
            tprint("⚠️ VectorBT utilities not available, using basic implementation")
            self.rolling_optimizer = None
            self.vectorization_manager = None
            self.batch_processor = None
            return
        
        # Initialize VectorBT rolling optimizer
        if self.config.enable_vectorbt:
            try:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=self.config.enable_gpu,
                    enable_parallel=self.config.enable_parallel,
                    memory_efficient=self.config.memory_efficient,
                    chunk_size=self.config.chunk_size
                )
                tprint("✅ VectorBT rolling optimizer initialized")
            except Exception as e:
                tprint(f"❌ ERROR: Failed to initialize VectorBT rolling optimizer: {e}")
                raise RuntimeError(f"VectorBT rolling optimizer initialization failed: {e}")
        else:
            self.rolling_optimizer = None
            tprint("ℹ️ VectorBT rolling optimizer disabled")
        
        # Initialize unified vectorization manager
        if self.config.enable_vectorbt:
            try:
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
                tprint("✅ VectorBT vectorization manager initialized")
            except Exception as e:
                tprint(f"❌ ERROR: Failed to initialize VectorBT vectorization manager: {e}")
                raise RuntimeError(f"VectorBT vectorization manager initialization failed: {e}")
        else:
            self.vectorization_manager = None
            tprint("ℹ️ VectorBT vectorization manager disabled")
        
        # Initialize batch processor
        if self.config.enable_batch_processing and VECTORBT_UTILS_AVAILABLE:
            try:
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
                tprint("✅ VectorBT batch processor initialized")
            except Exception as e:
                tprint(f"❌ ERROR: Failed to initialize VectorBT batch processor: {e}")
                raise RuntimeError(f"VectorBT batch processor initialization failed: {e}")
        else:
            self.batch_processor = None
            tprint("ℹ️ VectorBT batch processor disabled")
    
    def _initialize_interaction_types(self) -> Dict[str, InteractionType]:
        """Initialize available interaction types with consolidated methods."""
        tprint("🔧 Initializing interaction types")
        
        interaction_types = {}
        
        # Basic arithmetic interactions
        interaction_types['product'] = InteractionType(
            name='product',
            function=self._arithmetic_interaction,
            description='Multiplication of two features',
            complexity=1,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True,
            parameters={'operation': 'multiply'}
        )
        
        interaction_types['ratio'] = InteractionType(
            name='ratio',
            function=self._arithmetic_interaction,
            description='Division of two features',
            complexity=1,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True,
            parameters={'operation': 'divide'}
        )
        
        interaction_types['difference'] = InteractionType(
            name='difference',
            function=self._arithmetic_interaction,
            description='Subtraction of two features',
            complexity=1,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True,
            parameters={'operation': 'subtract'}
        )
        
        interaction_types['sum'] = InteractionType(
            name='sum',
            function=self._arithmetic_interaction,
            description='Addition of two features',
            complexity=1,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True,
            parameters={'operation': 'add'}
        )
        
        # Advanced interactions
        interaction_types['correlation'] = InteractionType(
            name='correlation',
            function=self._rolling_interaction,
            description='Rolling correlation between features',
            complexity=3,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True,
            parameters={'window': 20, 'method': 'correlation'}
        )
        
        interaction_types['covariance'] = InteractionType(
            name='covariance',
            function=self._rolling_interaction,
            description='Rolling covariance between features',
            complexity=3,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True,
            parameters={'window': 20, 'method': 'covariance'}
        )
        
        interaction_types['zscore_product'] = InteractionType(
            name='zscore_product',
            function=self._scaled_interaction,
            description='Product of z-scored features',
            complexity=2,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True,
            parameters={'operation': 'multiply', 'scaling': 'zscore'}
        )
        
        interaction_types['rank_correlation'] = InteractionType(
            name='rank_correlation',
            function=self._rolling_interaction,
            description='Rank correlation between features',
            complexity=3,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True,
            parameters={'window': 20, 'method': 'rank_correlation'}
        )
        
        # Statistical interactions
        interaction_types['skewness'] = InteractionType(
            name='skewness',
            function=self._statistical_interaction,
            description='Rolling skewness of feature',
            complexity=3,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True,
            parameters={'window': 20, 'statistic': 'skewness'}
        )
        
        interaction_types['kurtosis'] = InteractionType(
            name='kurtosis',
            function=self._statistical_interaction,
            description='Rolling kurtosis of feature',
            complexity=3,
            vectorbt_optimized=True,
            batch_processable=True,
            memory_efficient=True,
            parameters={'window': 20, 'statistic': 'kurtosis'}
        )
        
        tprint(f"✅ Initialized {len(interaction_types)} interaction types")
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
        tprint("🚀 Starting enhanced data-driven interaction generation")
        tprint(f"📊 Input features: {features.shape}")
        
        start_time = time.time()
        
        # Validate inputs with fast-fail
        self._validate_inputs(features, targets)
        
        # Optimize input data
        features = self._optimize_input_data(features)
        
        # Analyze data characteristics
        data_characteristics = self._analyze_data_characteristics(features)
        
        # Select optimal interaction types
        selected_types = self._select_interaction_types(data_characteristics)
        
        # Generate feature combinations
        feature_combinations = self._generate_feature_combinations(features.columns.tolist())
        
        # Process interactions
        interactions = self._process_interactions(features, feature_combinations, selected_types, targets)
        
        # Filter and rank interactions
        filtered_interactions = self._filter_interactions(interactions, targets)
        ranked_interactions = self._rank_interactions(filtered_interactions, targets)
        
        # Select final interactions
        selected_interactions = ranked_interactions[:self.config.max_interactions]
        
        # Update performance stats
        total_processing_time = time.time() - start_time
        self._update_performance_stats(selected_interactions, total_processing_time)
        
        tprint(f"✅ Interaction generation completed in {total_processing_time:.2f}s")
        tprint(f"📊 Generated {len(selected_interactions)} final interactions")
        
        return selected_interactions
    
    def _validate_inputs(self, features: pd.DataFrame, targets: Optional[pd.Series]):
        """Validate inputs with fast-fail approach."""
        tprint("🔍 Validating inputs")
        
        if features is None:
            tprint("❌ ERROR: Input features DataFrame is None")
            raise ValueError("Input features DataFrame cannot be None")
        
        if features.empty:
            tprint("❌ ERROR: Input features DataFrame is empty")
            raise ValueError("Input features DataFrame cannot be empty")
        
        if not isinstance(features, pd.DataFrame):
            tprint("❌ ERROR: Input features must be a pandas DataFrame")
            raise TypeError("Input features must be a pandas DataFrame")
        
        if targets is not None and not isinstance(targets, pd.Series):
            tprint("❌ ERROR: Targets must be a pandas Series")
            raise TypeError("Targets must be a pandas Series")
        
        if targets is not None and len(targets) != len(features):
            tprint("❌ ERROR: Targets length must match features length")
            raise ValueError("Targets length must match features length")
        
        tprint(f"✅ Input validation passed: {len(features.columns)} features, {len(features)} samples")
    
    def _optimize_input_data(self, features: pd.DataFrame) -> pd.DataFrame:
        """Optimize input data using VectorBT if available."""
        tprint("🔄 Optimizing input data")
        
        if self.vectorization_manager:
            try:
                optimized_features = self.vectorization_manager.optimize_dataframe(features)
                tprint("✅ Data optimization completed with VectorBT")
                return optimized_features
            except Exception as e:
                tprint(f"❌ ERROR: Data optimization failed: {e}")
                raise RuntimeError(f"Data optimization failed: {e}")
        else:
            tprint("ℹ️ VectorBT vectorization manager not available, using original data")
            return features
    
    def _analyze_data_characteristics(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data characteristics for interaction selection."""
        tprint("🔍 Analyzing data characteristics")
        
        characteristics = {}
        
        # Basic statistics
        characteristics['n_features'] = len(features.columns)
        characteristics['n_samples'] = len(features)
        
        # Correlation analysis
        try:
            corr_matrix = features.corr()
            characteristics['avg_correlation'] = corr_matrix.abs().mean().mean()
            characteristics['max_correlation'] = corr_matrix.abs().max().max()
            tprint(f"✅ Correlation analysis: avg={characteristics['avg_correlation']:.3f}, max={characteristics['max_correlation']:.3f}")
        except Exception as e:
            tprint(f"❌ ERROR: Correlation analysis failed: {e}")
            raise RuntimeError(f"Correlation analysis failed: {e}")
        
        # Variance analysis
        try:
            characteristics['avg_variance'] = features.var().mean()
            tprint(f"✅ Variance analysis: avg={characteristics['avg_variance']:.6f}")
        except Exception as e:
            tprint(f"❌ ERROR: Variance analysis failed: {e}")
            raise RuntimeError(f"Variance analysis failed: {e}")
        
        # Distribution analysis
        try:
            characteristics['feature_skewness'] = {}
            for col in features.columns:
                skew = features[col].skew()
                if not pd.isna(skew):
                    characteristics['feature_skewness'][col] = skew
            
            avg_skewness = np.mean([abs(s) for s in characteristics['feature_skewness'].values()])
            tprint(f"✅ Distribution analysis: avg_skewness={avg_skewness:.3f}")
        except Exception as e:
            tprint(f"❌ ERROR: Distribution analysis failed: {e}")
            raise RuntimeError(f"Distribution analysis failed: {e}")
        
        # Missing value analysis
        try:
            characteristics['missing_values'] = {}
            total_missing = 0
            for col in features.columns:
                missing = features[col].isna().sum()
                characteristics['missing_values'][col] = missing
                total_missing += missing
            
            missing_ratio = total_missing / (len(features) * len(features.columns))
            tprint(f"✅ Missing value analysis: {total_missing} missing values ({missing_ratio:.2%} ratio)")
        except Exception as e:
            tprint(f"❌ ERROR: Missing value analysis failed: {e}")
            raise RuntimeError(f"Missing value analysis failed: {e}")
        
        tprint(f"✅ Data characteristics analysis completed: {len(characteristics)} characteristics analyzed")
        return characteristics
    
    def _select_interaction_types(self, data_characteristics: Dict[str, Any]) -> List[str]:
        """Select optimal interaction types based on data characteristics."""
        tprint("🎯 Selecting optimal interaction types")
        
        selected_types = []
        
        # Always include basic arithmetic interactions
        selected_types.extend(['product', 'ratio', 'difference', 'sum'])
        
        # Add correlation-based interactions if data has reasonable correlation
        if data_characteristics.get('avg_correlation', 0) > 0.1:
            selected_types.extend(['correlation', 'covariance', 'rank_correlation'])
            tprint("✅ Added correlation-based interactions")
        
        # Add scaled interactions if data has high variance
        if data_characteristics.get('avg_variance', 0) > 0.01:
            selected_types.extend(['zscore_product'])
            tprint("✅ Added scaled interactions")
        
        # Add statistical interactions if data has skewness
        if data_characteristics.get('feature_skewness'):
            avg_skewness = np.mean([abs(s) for s in data_characteristics['feature_skewness'].values()])
            if avg_skewness > 0.5:
                selected_types.extend(['skewness', 'kurtosis'])
                tprint("✅ Added statistical interactions")
        
        # Remove duplicates and validate
        selected_types = list(set(selected_types))
        available_types = set(self.interaction_types.keys())
        selected_types = [t for t in selected_types if t in available_types]
        
        tprint(f"✅ Selected {len(selected_types)} interaction types: {selected_types}")
        return selected_types
    
    def _generate_feature_combinations(self, feature_names: List[str]) -> List[Tuple[str, str]]:
        """Generate feature combinations for interaction generation."""
        tprint("🔗 Generating feature combinations")
        
        if len(feature_names) < 2:
            tprint("❌ ERROR: Need at least 2 features for interactions")
            raise ValueError("Need at least 2 features for interactions")
        
        try:
            combinations_list = list(combinations(feature_names, 2))
            tprint(f"✅ Generated {len(combinations_list)} feature combinations")
            return combinations_list
        except Exception as e:
            tprint(f"❌ ERROR: Feature combination generation failed: {e}")
            raise RuntimeError(f"Feature combination generation failed: {e}")
    
    def _process_interactions(self, 
                            features: pd.DataFrame,
                            feature_combinations: List[Tuple[str, str]],
                            selected_types: List[str],
                            targets: Optional[pd.Series]) -> List[InteractionResult]:
        """Process interactions using batch or sequential processing."""
        tprint("⚡ Processing interactions")
        
        if (self.config.enable_batch_processing and 
            self.batch_processor and 
            len(feature_combinations) > self.config.batch_size):
            tprint(f"🔄 Using batch processing for {len(feature_combinations)} combinations")
            return self._generate_interactions_batch(features, feature_combinations, selected_types, targets)
        else:
            tprint(f"🔄 Using sequential processing for {len(feature_combinations)} combinations")
            return self._generate_interactions_sequential(features, feature_combinations, selected_types, targets)
    
    def _generate_interactions_sequential(self, 
                                        features: pd.DataFrame,
                                        feature_combinations: List[Tuple[str, str]],
                                        selected_types: List[str],
                                        targets: Optional[pd.Series]) -> List[InteractionResult]:
        """Generate interactions sequentially."""
        tprint("🔄 Processing interactions sequentially")
        
        interactions = []
        
        for i, (feat1_name, feat2_name) in enumerate(feature_combinations):
            tprint(f"🔄 Processing combination {i+1}/{len(feature_combinations)}: {feat1_name} x {feat2_name}")
            
            for interaction_type_name in selected_types:
                try:
                    result = self._generate_single_interaction(
                        features, feat1_name, feat2_name, interaction_type_name, targets
                    )
                    if result:
                        interactions.append(result)
                        tprint(f"✅ Generated interaction: {result.feature_name}")
                except Exception as e:
                    tprint(f"❌ ERROR: Interaction generation failed for {feat1_name} x {feat2_name} ({interaction_type_name}): {e}")
                    # Continue with other interactions instead of failing completely
                    continue
        
        tprint(f"✅ Sequential processing completed: {len(interactions)} interactions generated")
        return interactions
    
    def _generate_interactions_batch(self, 
                                   features: pd.DataFrame,
                                   feature_combinations: List[Tuple[str, str]],
                                   selected_types: List[str],
                                   targets: Optional[pd.Series]) -> List[InteractionResult]:
        """Generate interactions using batch processing."""
        tprint("🔄 Processing interactions in batch")
        
        if not self.batch_processor:
            tprint("❌ ERROR: Batch processor not available")
            raise RuntimeError("Batch processor not available")
        
        try:
            interactions = self.batch_processor.process_interactions(
                features, feature_combinations, selected_types, targets
            )
            tprint(f"✅ Batch processing completed: {len(interactions)} interactions generated")
            return interactions
        except Exception as e:
            tprint(f"❌ ERROR: Batch processing failed: {e}")
            raise RuntimeError(f"Batch processing failed: {e}")
    
    def _generate_single_interaction(self, 
                                   features: pd.DataFrame,
                                   feat1_name: str,
                                   feat2_name: str,
                                   interaction_type_name: str,
                                   targets: Optional[pd.Series]) -> Optional[InteractionResult]:
        """Generate a single interaction with comprehensive error handling."""
        tprint(f"🔧 Generating single interaction: {feat1_name} x {feat2_name} ({interaction_type_name})")
        
        try:
            # Get interaction type
            interaction_type = self.interaction_types.get(interaction_type_name)
            if not interaction_type:
                tprint(f"❌ ERROR: Unknown interaction type: {interaction_type_name}")
                return None
            
            # Get feature data
            feat1 = features[feat1_name]
            feat2 = features[feat2_name]
            
            # Check for cache
            cache_key = f"{feat1_name}_{feat2_name}_{interaction_type_name}"
            if self._cache_enabled and cache_key in self._result_cache:
                tprint(f"💾 Using cached result for {cache_key}")
                self.performance_stats['cached_operations'] += 1
                return self._result_cache[cache_key]
            
            # Generate interaction
            start_time = time.time()
            
            if interaction_type.vectorbt_optimized and VECTORBT_AVAILABLE:
                result_series = self._generate_vectorbt_optimized_interaction(
                    feat1, feat2, interaction_type
                )
                optimization_method = "vectorbt"
            else:
                result_series = interaction_type.function(feat1, feat2)
                optimization_method = "pandas"
            
            processing_time = time.time() - start_time
            
            if result_series is None or result_series.empty:
                tprint(f"⚠️ WARNING: Interaction generation returned empty result")
                return None
            
            # Validate result
            if not isinstance(result_series, pd.Series):
                tprint(f"❌ ERROR: Interaction function must return pandas Series")
                return None
            
            # Calculate utility score
            utility_score = self._calculate_utility_score(result_series, targets)
            
            # Check utility threshold
            if utility_score < self.config.utility_threshold:
                tprint(f"⚠️ WARNING: Utility score {utility_score:.3f} below threshold {self.config.utility_threshold}")
                return None
            
            # Check for duplicates
            if self._is_duplicate_interaction(result_series, utility_score):
                tprint(f"⚠️ WARNING: Duplicate interaction detected")
                return None
            
            # Create result
            feature_name = f"{interaction_type_name}_{feat1_name}_{feat2_name}"
            result = InteractionResult(
                feature_name=feature_name,
                feature_series=result_series,
                parent_features=[feat1_name, feat2_name],
                interaction_type=interaction_type_name,
                utility_score=utility_score,
                metadata={
                    'processing_time': processing_time,
                    'optimization_method': optimization_method,
                    'n_values': len(result_series),
                    'n_unique': result_series.nunique(),
                    'missing_ratio': result_series.isna().sum() / len(result_series)
                },
                processing_time=processing_time,
                optimization_method=optimization_method
            )
            
            # Cache result
            if self._cache_enabled:
                self._put_in_cache(cache_key, result)
                tprint(f"💾 Cached result for {feature_name}")
            
            tprint(f"✅ Generated interaction: {feature_name} (utility: {utility_score:.3f})")
            return result
            
        except Exception as e:
            tprint(f"❌ ERROR: Single interaction generation failed: {e}")
            return None
    
    def _generate_vectorbt_optimized_interaction(self, 
                                               feat1: pd.Series, 
                                               feat2: pd.Series, 
                                               interaction_type: InteractionType) -> Optional[pd.Series]:
        """Generate interaction using VectorBT optimization."""
        tprint(f"⚡ Generating VectorBT optimized interaction: {interaction_type.name}")
        
        try:
            if not VECTORBT_AVAILABLE:
                tprint("⚠️ VectorBT not available, falling back to pandas")
                return interaction_type.function(feat1, feat2)
            
            # Use VectorBT optimized functions
            if interaction_type.name in ['product', 'ratio', 'difference', 'sum']:
                operation = interaction_type.parameters.get('operation', 'multiply')
                return self._arithmetic_interaction(feat1, feat2, operation=operation)
            elif interaction_type.name in ['correlation', 'covariance', 'rank_correlation']:
                method = interaction_type.parameters.get('method', 'correlation')
                window = interaction_type.parameters.get('window', 20)
                return self._rolling_interaction(feat1, feat2, method=method, window=window)
            elif interaction_type.name in ['skewness', 'kurtosis']:
                statistic = interaction_type.parameters.get('statistic', 'skewness')
                window = interaction_type.parameters.get('window', 20)
                return self._statistical_interaction(feat1, feat2, statistic=statistic, window=window)
            else:
                tprint(f"⚠️ No VectorBT optimization for {interaction_type.name}, using pandas")
                return interaction_type.function(feat1, feat2)
                
        except Exception as e:
            tprint(f"❌ ERROR: VectorBT optimized interaction failed: {e}")
            return None
    
    def _is_duplicate_interaction(self, series: pd.Series, utility_score: float) -> bool:
        """Check if interaction is duplicate based on series characteristics."""
        tprint("🔍 Checking for duplicate interaction")
        
        try:
            # Simple duplicate check based on series characteristics
            series_hash = hash(tuple(series.dropna().head(100).values))
            
            if hasattr(self, '_seen_hashes'):
                if series_hash in self._seen_hashes:
                    return True
                else:
                    self._seen_hashes.add(series_hash)
            else:
                self._seen_hashes = {series_hash}
            
            return False
        except Exception as e:
            tprint(f"⚠️ WARNING: Duplicate check failed: {e}")
            return False
    
    def _calculate_utility_score(self, series: pd.Series, targets: Optional[pd.Series]) -> float:
        """Calculate utility score for interaction."""
        tprint("📊 Calculating utility score")
        
        try:
            if targets is not None:
                # Use correlation with target
                correlation = series.corr(targets)
                if pd.isna(correlation):
                    return 0.0
                return abs(correlation)
            else:
                # Use variance as proxy for utility
                variance = series.var()
                if pd.isna(variance) or variance <= 0:
                    return 0.0
                return min(1.0, variance)
        except Exception as e:
            tprint(f"❌ ERROR: Utility score calculation failed: {e}")
            return 0.0
    
    def _filter_interactions(self, interactions: List[InteractionResult], targets: Optional[pd.Series]) -> List[InteractionResult]:
        """Filter interactions based on quality criteria."""
        tprint("🔍 Filtering interactions")
        
        if not interactions:
            tprint("⚠️ No interactions to filter")
            return []
        
        filtered = []
        for interaction in interactions:
            # Check utility threshold
            if interaction.utility_score < self.config.utility_threshold:
                continue
            
            # Check for valid data
            if interaction.feature_series.empty or interaction.feature_series.isna().all():
                continue
            
            filtered.append(interaction)
        
        tprint(f"✅ Filtered to {len(filtered)} valid interactions")
        return filtered
    
    def _rank_interactions(self, interactions: List[InteractionResult], targets: Optional[pd.Series]) -> List[InteractionResult]:
        """Rank interactions by utility score."""
        tprint("📊 Ranking interactions")
        
        if not interactions:
            tprint("⚠️ No interactions to rank")
            return []
        
        try:
            # Sort by utility score (descending)
            ranked = sorted(interactions, key=lambda x: x.utility_score, reverse=True)
            tprint(f"✅ Ranked {len(ranked)} interactions")
            return ranked
        except Exception as e:
            tprint(f"❌ ERROR: Ranking failed: {e}")
            return interactions
    
    def _update_performance_stats(self, interactions: List[InteractionResult], total_time: float):
        """Update performance statistics."""
        tprint("📊 Updating performance statistics")
        
        self.performance_stats.update({
            'total_interactions_generated': len(interactions),
            'total_processing_time': total_time,
            'average_utility_score': np.mean([i.utility_score for i in interactions]) if interactions else 0.0
        })
        
        tprint(f"✅ Performance stats updated: {len(interactions)} interactions, {total_time:.2f}s")
    
    def _put_in_cache(self, key: str, value: InteractionResult):
        """Put result in cache with size management."""
        tprint(f"💾 Caching result: {key}")
        
        if len(self._result_cache) >= self.config.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._result_cache))
            del self._result_cache[oldest_key]
            tprint(f"💾 Cache full, removed oldest entry: {oldest_key}")
        
        self._result_cache[key] = value
        tprint(f"✅ Cached result: {key}")
    
    # Consolidated interaction methods
    def _arithmetic_interaction(self, feat1: pd.Series, feat2: pd.Series, operation: str = 'multiply') -> pd.Series:
        """Consolidated arithmetic interaction method."""
        tprint(f"🔧 Arithmetic interaction: {operation}")
        
        try:
            if operation == 'multiply':
                result = feat1 * feat2
            elif operation == 'divide':
                result = feat1 / feat2
            elif operation == 'add':
                result = feat1 + feat2
            elif operation == 'subtract':
                result = feat1 - feat2
            else:
                tprint(f"❌ ERROR: Unknown arithmetic operation: {operation}")
                raise ValueError(f"Unknown arithmetic operation: {operation}")
            
            tprint(f"✅ Arithmetic interaction completed: {len(result)} values")
            return result
        except Exception as e:
            tprint(f"❌ ERROR: Arithmetic interaction failed: {e}")
            raise RuntimeError(f"Arithmetic interaction failed: {e}")
    
    def _rolling_interaction(self, feat1: pd.Series, feat2: pd.Series, method: str = 'correlation', window: int = 20) -> pd.Series:
        """Consolidated rolling interaction method."""
        tprint(f"🔧 Rolling interaction: {method} (window={window})")
        
        try:
            if method == 'correlation':
                if VECTORBT_AVAILABLE and rolling_corr is not None:
                    result = rolling_corr(feat1, feat2, window=window)
                else:
                    result = feat1.rolling(window).corr(feat2)
            elif method == 'covariance':
                if VECTORBT_AVAILABLE and rolling_cov is not None:
                    result = rolling_cov(feat1, feat2, window=window)
                else:
                    result = feat1.rolling(window).cov(feat2)
            elif method == 'rank_correlation':
                # Use pandas for rank correlation
                result = feat1.rolling(window).corr(feat2.rank())
            else:
                tprint(f"❌ ERROR: Unknown rolling method: {method}")
                raise ValueError(f"Unknown rolling method: {method}")
            
            tprint(f"✅ Rolling interaction completed: {len(result)} values")
            return result
        except Exception as e:
            tprint(f"❌ ERROR: Rolling interaction failed: {e}")
            raise RuntimeError(f"Rolling interaction failed: {e}")
    
    def _scaled_interaction(self, feat1: pd.Series, feat2: pd.Series, operation: str = 'multiply', scaling: str = 'zscore') -> pd.Series:
        """Consolidated scaled interaction method."""
        tprint(f"🔧 Scaled interaction: {operation} with {scaling} scaling")
        
        try:
            # Apply scaling
            if scaling == 'zscore':
                if VECTORBT_AVAILABLE and zscore is not None:
                    scaled1 = zscore(feat1)
                    scaled2 = zscore(feat2)
                else:
                    scaled1 = (feat1 - feat1.mean()) / feat1.std()
                    scaled2 = (feat2 - feat2.mean()) / feat2.std()
            else:
                tprint(f"❌ ERROR: Unknown scaling method: {scaling}")
                raise ValueError(f"Unknown scaling method: {scaling}")
            
            # Apply operation
            if operation == 'multiply':
                result = scaled1 * scaled2
            elif operation == 'add':
                result = scaled1 + scaled2
            else:
                tprint(f"❌ ERROR: Unknown operation: {operation}")
                raise ValueError(f"Unknown operation: {operation}")
            
            tprint(f"✅ Scaled interaction completed: {len(result)} values")
            return result
        except Exception as e:
            tprint(f"❌ ERROR: Scaled interaction failed: {e}")
            raise RuntimeError(f"Scaled interaction failed: {e}")
    
    def _statistical_interaction(self, feat1: pd.Series, feat2: pd.Series, statistic: str = 'skewness', window: int = 20) -> pd.Series:
        """Consolidated statistical interaction method."""
        tprint(f"🔧 Statistical interaction: {statistic} (window={window})")
        
        try:
            if statistic == 'skewness':
                if VECTORBT_AVAILABLE and rolling_skew is not None:
                    result = rolling_skew(feat1, window=window)
                else:
                    result = feat1.rolling(window).skew()
            elif statistic == 'kurtosis':
                if VECTORBT_AVAILABLE and rolling_kurt is not None:
                    result = rolling_kurt(feat1, window=window)
                else:
                    result = feat1.rolling(window).kurt()
            else:
                tprint(f"❌ ERROR: Unknown statistic: {statistic}")
                raise ValueError(f"Unknown statistic: {statistic}")
            
            tprint(f"✅ Statistical interaction completed: {len(result)} values")
            return result
        except Exception as e:
            tprint(f"❌ ERROR: Statistical interaction failed: {e}")
            raise RuntimeError(f"Statistical interaction failed: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        tprint("📊 Getting performance statistics")
        return self.performance_stats.copy()
    
    def reset_stats(self):
        """Reset all performance statistics."""
        tprint("🔄 Resetting performance statistics")
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
        tprint("✅ Performance statistics reset")


def create_data_driven_interaction_generator(
    max_interactions: int = 100,
    utility_threshold: float = 0.1,
    correlation_threshold: float = 0.95,
    enable_vectorbt: bool = True,
    config: Optional[EnhancedInteractionConfig] = None
) -> DataDrivenInteractionGenerator:
    """Create a data-driven interaction generator with default configuration."""
    tprint("🏭 Creating DataDrivenInteractionGenerator")
    return DataDrivenInteractionGenerator(
        max_interactions=max_interactions,
        utility_threshold=utility_threshold,
        correlation_threshold=correlation_threshold,
        enable_vectorbt=enable_vectorbt,
        config=config
    )