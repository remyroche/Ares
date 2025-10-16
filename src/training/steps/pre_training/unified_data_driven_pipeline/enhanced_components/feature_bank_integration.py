"""
Feature Bank Integration for UnifiedDataDrivenPipeline

This module integrates the sophisticated feature generation system
from FeatureLookbackOptimizationComponent, including:
- Feature Bank system with 200+ features
- PID-based feature generation
- Multi-horizon profit labeling integration
- Advanced caching and serialization
- Memory-efficient operations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import time
import logging
from pathlib import Path
import gc

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

# Import feature bank system
try:
    from src.feature_generation.core.feature_bank import FeatureBank
    FEATURE_BANK_AVAILABLE = True
except ImportError:
    FEATURE_BANK_AVAILABLE = False
    FeatureBank = None

# Import enhanced feature generation utilities
try:
    from src.feature_generation.utils import (
        EnhancedFeatureEngineering, FeatureGenerationOptimizer,
        FeatureOptimizationConfig, CrossTimeframeAnalysisPipeline,
        FractionalDifferentiationPipeline, EnhancedMatrixOperations,
        validate_feature_quality, validate_features_dataframe,
        feature_validation_decorator
    )
    ENHANCED_FEATURE_GENERATION_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURE_GENERATION_AVAILABLE = False

# Import features_common utilities
try:
    from src.features_common import (
        OptimizationMixin, PerformanceMixin, VectorBTMixin,
        ValidationMixin, CachingMixin, MonitoringMixin,
        ScalerFactory, create_optimized_scaler, create_batch_scaler,
        UnifiedVectorBTManager, get_unified_vectorbt_manager,
        VectorBTOptimizationEngine, get_optimization_engine,
        GPUAccelerator, get_gpu_accelerator,
        VectorBTPerformanceMonitor, get_performance_monitor,
        validate_input_data, safe_execute, get_logger, log_operation
    )
    FEATURES_COMMON_AVAILABLE = True
except ImportError:
    FEATURES_COMMON_AVAILABLE = False

# Import multi-horizon profit labeler
try:
    from src.training.steps.pre_training.multi_horizon_profit_labeler import MultiHorizonConfig, MultiHorizonProfitLabeler
    MULTI_HORIZON_AVAILABLE = True
except ImportError:
    MULTI_HORIZON_AVAILABLE = False
    MultiHorizonConfig = None
    MultiHorizonProfitLabeler = None

# Import caching and serialization
try:
    from src.feature_generation.core.feature_cache import FeatureCacheService
    from src.utils.serialization_utils import UniversalSerializer, JSONSerializer, PickleSerializer
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False
    FeatureCacheService = None
    UniversalSerializer = None
    JSONSerializer = None
    PickleSerializer = None

# Import matrix operations
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_batch_matrix_processor,
        optimize_dataframe
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class FeatureBankConfig:
    """Configuration for feature bank integration."""
    enable_feature_bank: bool = True
    enable_caching: bool = True
    enable_multi_horizon: bool = True
    enable_memory_optimization: bool = True
    min_variance: float = 1e-8
    max_correlation_threshold: float = 0.95
    cache_force_refresh: bool = False
    memory_efficient: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4

@dataclass
class FeatureGenerationResult:
    """Result of feature generation."""
    feature_names: List[str]
    feature_data: pd.DataFrame
    generation_time: float
    n_features_generated: int
    cache_hit: bool
    memory_usage_mb: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class FeatureBankIntegration:
    """
    Feature Bank Integration for UnifiedDataDrivenPipeline.

    This class integrates the sophisticated feature generation system
    from FeatureLookbackOptimizationComponent, providing:
    - Feature Bank system with 200+ features
    - PID-based feature generation
    - Multi-horizon profit labeling integration
    - Advanced caching and serialization
    - Memory-efficient operations
    """

    def __init__(self, config: Optional[FeatureBankConfig] = None):
        """Initialize the feature bank integration."""
        self.config = config or FeatureBankConfig()

        # Initialize components
        self._initialize_components()
        self._initialize_enhanced_utilities()

        # Performance tracking
        self.performance_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_execution_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'features_generated': 0,
            'memory_usage_mb': 0.0
        }

        tprint_success("✅ Feature Bank Integration initialized")

    def _initialize_components(self):
        """Initialize all feature bank components."""
        tprint_debug("Initializing feature bank integration components")

        # Initialize feature bank
        if FEATURE_BANK_AVAILABLE:
            from src.feature_generation.core.feature_bank import get_global_feature_bank
            self.feature_bank = get_global_feature_bank()
            tprint_success("✅ Feature Bank initialized")
        else:
            self.feature_bank = None
            tprint_warning("⚠️ Feature Bank not available, using fallback implementations")

        # Initialize multi-horizon profit labeler
        if MULTI_HORIZON_AVAILABLE:
            self.multi_horizon_labeler = MultiHorizonProfitLabeler()
            tprint_success("✅ Multi-horizon profit labeler initialized")
        else:
            self.multi_horizon_labeler = None
            tprint_warning("⚠️ Multi-horizon profit labeler not available")

        # Initialize caching
        if CACHING_AVAILABLE:
            self.feature_cache = FeatureCacheService()
            self.universal_serializer = UniversalSerializer()
            self.json_serializer = JSONSerializer()
            self.pickle_serializer = PickleSerializer()
            tprint_success("✅ Caching initialized")
        else:
            self.feature_cache = None
            self.universal_serializer = None
            self.json_serializer = None
            self.pickle_serializer = None
            tprint_warning("⚠️ Caching not available")

        # Initialize matrix operations
        if MATRIX_OPS_AVAILABLE:
            self.matrix_ops = get_unified_matrix_operations()
            self.vectorized_core = get_vectorized_processing_core()
            self.batch_processor = get_batch_matrix_processor()
            tprint_success("✅ Matrix operations initialized")
        else:
            self.matrix_ops = None
            self.vectorized_core = None
            self.batch_processor = None
            tprint_warning("⚠️ Matrix operations not available")

    def _initialize_enhanced_utilities(self):
        """Initialize enhanced utilities from feature_generation and features_common."""
        tprint_debug("Initializing enhanced utilities")

        # Initialize enhanced feature generation utilities
        if ENHANCED_FEATURE_GENERATION_AVAILABLE:
            try:
                self.enhanced_feature_engineering = EnhancedFeatureEngineering({})
                self.feature_optimizer = FeatureGenerationOptimizer()
                self.cross_timeframe_pipeline = CrossTimeframeAnalysisPipeline()
                self.fractional_diff_pipeline = FractionalDifferentiationPipeline()
                self.enhanced_matrix_ops = EnhancedMatrixOperations()
                tprint_success("✅ Enhanced feature generation utilities initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Enhanced feature generation utilities initialization failed: {e}")
                self.enhanced_feature_engineering = None
                self.feature_optimizer = None
                self.cross_timeframe_pipeline = None
                self.fractional_diff_pipeline = None
                self.enhanced_matrix_ops = None
        else:
            self.enhanced_feature_engineering = None
            self.feature_optimizer = None
            self.cross_timeframe_pipeline = None
            self.fractional_diff_pipeline = None
            self.enhanced_matrix_ops = None

        # Initialize features common utilities
        if FEATURES_COMMON_AVAILABLE:
            try:
                self.unified_vectorbt_manager = get_unified_vectorbt_manager()
                self.vectorbt_optimization_engine = get_optimization_engine()
                self.gpu_accelerator = get_gpu_accelerator()
                self.vectorbt_performance_monitor = get_performance_monitor()
                self.scaler_factory = ScalerFactory()
                self.optimized_scaler = create_optimized_scaler()
                self.batch_scaler = create_batch_scaler()
                tprint_success("✅ Features common utilities initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Features common utilities initialization failed: {e}")
                self.unified_vectorbt_manager = None
                self.vectorbt_optimization_engine = None
                self.gpu_accelerator = None
                self.vectorbt_performance_monitor = None
                self.scaler_factory = None
                self.optimized_scaler = None
                self.batch_scaler = None
        else:
            self.unified_vectorbt_manager = None
            self.vectorbt_optimization_engine = None
            self.gpu_accelerator = None
            self.vectorbt_performance_monitor = None
            self.scaler_factory = None
            self.optimized_scaler = None
            self.batch_scaler = None

    def generate_features_for_optimization(
        self,
        data: pd.DataFrame,
        pipeline_state: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False
    ) -> FeatureGenerationResult:
        """
        Generate features using the feature bank system with caching support.

        This is the core feature generation method from FeatureLookbackOptimizationComponent.
        """
        tprint_info("🏦 Starting feature generation with Feature Bank system")
        tprint_debug(f"📊 Data shape: {data.shape}, Force refresh: {force_refresh}")

        start_time = time.time()

        try:
            # Check cache first
            cache_key = self._resolve_cache_key(data, pipeline_state)
            cached_features = None

            if self.config.enable_caching and cache_key and not force_refresh:
                cached_features = self._load_cached_features(cache_key)
                if cached_features is not None:
                    tprint_success(f"✅ Using cached features: {len(cached_features.columns)} features")
                    return FeatureGenerationResult(
                        feature_names=cached_features.columns.tolist(),
                        feature_data=cached_features,
                        generation_time=time.time() - start_time,
                        n_features_generated=len(cached_features.columns),
                        cache_hit=True,
                        memory_usage_mb=0.0,
                        success=True,
                        metadata={'cache_key': cache_key}
                    )

            # Generate features using feature bank
            tprint_debug("🔧 Generating features using Feature Bank system")
            feature_data = self._generate_features_with_bank(data, pipeline_state)

            if feature_data is None or len(feature_data) == 0:
                tprint_error("❌ Feature generation failed")
                return FeatureGenerationResult(
                    feature_names=[],
                    feature_data=pd.DataFrame(),
                    generation_time=time.time() - start_time,
                    n_features_generated=0,
                    cache_hit=False,
                    memory_usage_mb=0.0,
                    success=False,
                    error_message="Feature generation failed"
                )

            # Apply feature selection and filtering
            tprint_debug("🔍 Applying feature selection and filtering")
            filtered_features = self._apply_feature_filtering(feature_data, data)

            # Cache the results
            if self.config.enable_caching and cache_key:
                self._cache_features(cache_key, filtered_features)

            generation_time = time.time() - start_time
            memory_usage = self._calculate_memory_usage(filtered_features)

            # Update performance stats
            self.performance_stats.update({
                'total_generations': 1,
                'successful_generations': 1,
                'total_execution_time': generation_time,
                'features_generated': len(filtered_features.columns),
                'memory_usage_mb': memory_usage,
                'cache_misses': 1
            })

            tprint_success(f"✅ Generated {len(filtered_features.columns)} features in {generation_time:.3f}s")

            return FeatureGenerationResult(
                feature_names=filtered_features.columns.tolist(),
                feature_data=filtered_features,
                generation_time=generation_time,
                n_features_generated=len(filtered_features.columns),
                cache_hit=False,
                memory_usage_mb=memory_usage,
                success=True,
                metadata={
                    'cache_key': cache_key,
                    'original_features': len(feature_data.columns),
                    'filtered_features': len(filtered_features.columns)
                }
            )

        except Exception as e:
            tprint_error(f"❌ Feature generation failed: {e}")
            return FeatureGenerationResult(
                feature_names=[],
                feature_data=pd.DataFrame(),
                generation_time=time.time() - start_time,
                n_features_generated=0,
                cache_hit=False,
                memory_usage_mb=0.0,
                success=False,
                error_message=str(e)
            )

    def _generate_features_with_bank(
        self,
        data: pd.DataFrame,
        pipeline_state: Optional[Dict[str, Any]]
    ) -> Optional[pd.DataFrame]:
        """Generate features using enhanced utilities and Feature Bank system."""
        try:
            # First, try enhanced feature engineering if available
            if ENHANCED_FEATURE_GENERATION_AVAILABLE and self.enhanced_feature_engineering:
                tprint_debug("🔧 Using enhanced feature engineering")
                try:
                    enhanced_features = self.enhanced_feature_engineering.generate_features(data)
                    if enhanced_features is not None and not enhanced_features.empty:
                        tprint_success(f"✅ Generated {len(enhanced_features.columns)} features using enhanced feature engineering")

                        # Apply feature validation if available
                        if ENHANCED_FEATURE_GENERATION_AVAILABLE:
                            try:
                                validated_features = validate_features_dataframe(enhanced_features)
                                if validated_features is not None:
                                    enhanced_features = validated_features
                                    tprint_success("✅ Features validated successfully")
                            except Exception as e:
                                tprint_warning(f"⚠️ Feature validation failed: {e}")

                        return enhanced_features
                except Exception as e:
                    tprint_warning(f"⚠️ Enhanced feature engineering failed: {e}")

            # Try cross-timeframe analysis if available
            if ENHANCED_FEATURE_GENERATION_AVAILABLE and self.cross_timeframe_pipeline:
                tprint_debug("🔧 Using cross-timeframe analysis")
                try:
                    cross_timeframe_features = self.cross_timeframe_pipeline.generate_features(data)
                    if cross_timeframe_features is not None and not cross_timeframe_features.empty:
                        tprint_success(f"✅ Generated {len(cross_timeframe_features.columns)} features using cross-timeframe analysis")
                        return cross_timeframe_features
                except Exception as e:
                    tprint_warning(f"⚠️ Cross-timeframe analysis failed: {e}")

            # Fallback to Feature Bank system
            if not FEATURE_BANK_AVAILABLE or not self.feature_bank:
                tprint_error("❌ Feature Bank not available - this is required for feature generation")
                return None

            # Standardize column names to lowercase for consistent feature generation
            data_for_features = data.copy()
            data_for_features.columns = data_for_features.columns.str.lower()
            tprint_debug(f"📊 Standardized column names: {list(data_for_features.columns)[:10]}...")

            # Generate features using Feature Bank
            tprint_debug("🏦 Generating features with Feature Bank system")
            feature_data = self.feature_bank.generate_features(data_for_features)

            if feature_data is None or len(feature_data) == 0:
                tprint_error("❌ Feature Bank returned empty features - this is not expected")
                return None

            # Align with original data index
            feature_data = feature_data.reindex(data.index)

            # Apply enhanced matrix operations if available
            if ENHANCED_FEATURE_GENERATION_AVAILABLE and self.enhanced_matrix_ops:
                try:
                    feature_data = self.enhanced_matrix_ops.optimize_dataframe(feature_data)
                    tprint_success("✅ Applied enhanced matrix operations")
                except Exception as e:
                    tprint_warning(f"⚠️ Enhanced matrix operations failed: {e}")
            elif self.config.enable_memory_optimization and MATRIX_OPS_AVAILABLE:
                feature_data = optimize_dataframe(feature_data)

            # Apply feature validation if available
            if ENHANCED_FEATURE_GENERATION_AVAILABLE:
                try:
                    validated_features = validate_features_dataframe(feature_data)
                    if validated_features is not None:
                        feature_data = validated_features
                        tprint_success("✅ Features validated successfully")
                except Exception as e:
                    tprint_warning(f"⚠️ Feature validation failed: {e}")

            tprint_success(f"✅ Generated {len(feature_data.columns)} features using Feature Bank")
            return feature_data

        except Exception as e:
            tprint_error(f"❌ Feature generation failed: {e}")
            return None

    def _apply_feature_filtering(
        self,
        feature_data: pd.DataFrame,
        original_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply feature filtering and selection."""
        try:
            tprint_debug("🔍 Applying feature filtering")

            # Remove features with too much missing data
            missing_threshold = 0.5
            valid_features = feature_data.columns[
                feature_data.isnull().mean() < missing_threshold
            ]
            filtered_data = feature_data[valid_features]

            # Remove features with low variance
            if self.config.min_variance > 0:
                variance_threshold = self.config.min_variance
                high_variance_features = filtered_data.columns[
                    filtered_data.var() > variance_threshold
                ]
                filtered_data = filtered_data[high_variance_features]

            # Remove highly correlated features
            if self.config.max_correlation_threshold < 1.0:
                filtered_data = self._remove_highly_correlated_features(
                    filtered_data, self.config.max_correlation_threshold
                )

            # No artificial limit on number of features - let the Feature Bank generate all available features

            tprint_success(f"✅ Filtered to {len(filtered_data.columns)} features")
            return filtered_data

        except Exception as e:
            tprint_error(f"❌ Feature filtering failed: {e}")
            return feature_data

    def _remove_highly_correlated_features(
        self,
        data: pd.DataFrame,
        threshold: float
    ) -> pd.DataFrame:
        """Remove highly correlated features."""
        try:
            if len(data.columns) <= 1:
                return data

            # Calculate correlation matrix
            corr_matrix = data.corr().abs()

            # Find pairs of highly correlated features
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            # Find features to drop
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

            # Drop highly correlated features
            filtered_data = data.drop(columns=to_drop)

            tprint_debug(f"🔍 Removed {len(to_drop)} highly correlated features")
            return filtered_data

        except Exception as e:
            tprint_error(f"❌ Correlation filtering failed: {e}")
            return data

    def _resolve_cache_key(
        self,
        data: pd.DataFrame,
        pipeline_state: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Resolve cache key for feature data."""
        try:
            if not self.config.enable_caching:
                return None

            # Create cache key based on data characteristics
            data_hash = hash(str(data.shape) + str(data.columns.tolist()))
            state_hash = hash(str(pipeline_state)) if pipeline_state else 0

            cache_key = f"feature_bank_{data_hash}_{state_hash}"
            return cache_key

        except Exception as e:
            tprint_debug(f"Cache key resolution failed: {e}")
            return None

    def _load_cached_features(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load cached features."""
        try:
            if not CACHING_AVAILABLE or not self.feature_cache:
                return None

            cached_features = self.feature_cache.load(cache_key)
            if cached_features is not None and not cached_features.empty:
                self.performance_stats['cache_hits'] += 1
                return cached_features

            self.performance_stats['cache_misses'] += 1
            return None

        except Exception as e:
            tprint_debug(f"Cache load failed: {e}")
            return None

    def _cache_features(self, cache_key: str, features: pd.DataFrame):
        """Cache features."""
        try:
            if not CACHING_AVAILABLE or not self.feature_cache:
                return

            self.feature_cache.save(cache_key, features)
            tprint_debug(f"💾 Cached features with key: {cache_key}")

        except Exception as e:
            tprint_debug(f"Cache save failed: {e}")

    def _calculate_memory_usage(self, data: pd.DataFrame) -> float:
        """Calculate memory usage of DataFrame in MB."""
        try:
            memory_usage = data.memory_usage(deep=True).sum() / 1024 / 1024
            return float(memory_usage)
        except:
            return 0.0

    def generate_multi_horizon_targets(
        self,
        data: pd.DataFrame,
        config: Optional[MultiHorizonConfig] = None
    ) -> Optional[pd.DataFrame]:
        """Generate multi-horizon profit targets."""
        try:
            if not MULTI_HORIZON_AVAILABLE or not self.multi_horizon_labeler:
                tprint_warning("⚠️ Multi-horizon profit labeler not available")
                return None

            if config is None:
                config = MultiHorizonConfig()

            tprint_debug("🎯 Generating multi-horizon profit targets")
            targets = self.multi_horizon_labeler.generate_targets(data, config)

            if targets is not None:
                tprint_success(f"✅ Generated {len(targets.columns)} multi-horizon targets")

            return targets

        except Exception as e:
            tprint_error(f"❌ Multi-horizon target generation failed: {e}")
            return None

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()

    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_execution_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'features_generated': 0,
            'memory_usage_mb': 0.0
        }

def create_feature_bank_integration(config: Optional[FeatureBankConfig] = None) -> FeatureBankIntegration:
    """Create a feature bank integration with default configuration."""
    return FeatureBankIntegration(config)
