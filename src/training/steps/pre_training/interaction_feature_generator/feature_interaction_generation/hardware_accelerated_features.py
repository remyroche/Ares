"""
Hardware-Accelerated Feature Generation with Matrix Operations

This module provides highly optimized feature generation using hardware acceleration
and matrix operations for maximum performance in the lookback optimization system.
"""

import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Import matrix operations and hardware optimizations
try:
    from src.utils.matrix_operations.unified_operations import get_unified_matrix_operations
    from src.utils.matrix_operations.batch_operations import batch_matrix_multiply, batch_correlation_analysis
    from src.utils.matrix_operations.hardware_integration import HardwareOptimizedMatrixProcessor, HardwareConfig
    from src.utils.matrix_operations.vectorized_core import get_vectorized_processing_core
    from src.utils.matrix_operations.enhanced_operations import get_enhanced_matrix_operations
    from src.utils.hardware.m1_optimizations import M1MemoryOptimizer, M1CPUOptimizer
    from src.utils.hardware.memory_optimization import memory_efficient, optimize_dataframe_dtypes, chunk_dataframe
    from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
    from src.utils.hardware.advanced_memory_optimizer import AdvancedMemoryOptimizer
    from src.utils.hardware.advanced_cpu_optimizer import AdvancedCPUOptimizer
    MATRIX_OPS_AVAILABLE = True
    HARDWARE_AVAILABLE = True
except ImportError as e:
    MATRIX_OPS_AVAILABLE = False
    HARDWARE_AVAILABLE = False
    logging.warning(f"Matrix operations or hardware optimizations not available: {e}")

# Import base configuration
from .config import LookbackOptimizationConfig, FamilyType

# Import utilities
try:
    from src.utils.tprint import tprint, tprint_info, tprint_error, tprint_warning, tprint_success, tprint_performance
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class HardwareAcceleratedFeatureResult:
    """Result of hardware-accelerated feature generation."""
    family: FamilyType
    feature_name: str
    feature_values: np.ndarray
    lookback_spec: Any
    generation_time: float
    memory_usage_mb: float
    quality_score: float = 0.0
    
    # Hardware optimization metrics
    matrix_ops_used: int = 0
    hardware_accelerated_ops: int = 0
    vectorized_ops: int = 0
    memory_efficient_ops: int = 0
    gpu_ops: int = 0
    cpu_optimized_ops: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'family': self.family.value,
            'feature_name': self.feature_name,
            'feature_values': self.feature_values.tolist(),
            'lookback_spec': lookback_spec.to_dict() if hasattr(lookback_spec, 'to_dict') else str(lookback_spec),
            'generation_time': self.generation_time,
            'memory_usage_mb': self.memory_usage_mb,
            'quality_score': self.quality_score,
            'matrix_ops_used': self.matrix_ops_used,
            'hardware_accelerated_ops': self.hardware_accelerated_ops,
            'vectorized_ops': self.vectorized_ops,
            'memory_efficient_ops': self.memory_efficient_ops,
            'gpu_ops': self.gpu_ops,
            'cpu_optimized_ops': self.cpu_optimized_ops
        }


class HardwareAcceleratedFeatureBuilder:
    """Base class for hardware-accelerated feature builders."""
    
    def __init__(self, config: LookbackOptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize hardware optimizations
        self._initialize_hardware_optimizations()
        
        # Initialize matrix operations
        self._initialize_matrix_operations()
        
        # Performance tracking
        self.performance_metrics = {
            'matrix_ops_used': 0,
            'hardware_accelerated_ops': 0,
            'vectorized_ops': 0,
            'memory_efficient_ops': 0,
            'gpu_ops': 0,
            'cpu_optimized_ops': 0
        }
    
    def _initialize_hardware_optimizations(self):
        """Initialize hardware optimization components."""
        if not HARDWARE_AVAILABLE:
            tprint_warning("Hardware optimizations not available, using CPU-only mode")
            return
        
        try:
            # Initialize unified hardware manager
            self.hardware_manager = UnifiedHardwareManager()
            
            # Initialize advanced optimizers
            self.advanced_memory_optimizer = AdvancedMemoryOptimizer()
            self.advanced_cpu_optimizer = AdvancedCPUOptimizer()
            
            # Initialize M1 optimizations
            self.m1_memory_optimizer = M1MemoryOptimizer(memory_limit_gb=self.config.memory_limit_gb)
            self.m1_cpu_optimizer = M1CPUOptimizer()
            
            # Initialize hardware-optimized matrix processor
            hardware_config = HardwareConfig(
                max_memory_gb=self.config.memory_limit_gb,
                enable_gpu=self.config.enable_parallel,
                max_cpu_cores=self.config.n_workers,
                auto_optimize_dtypes=True,
                auto_chunk_large_data=True
            )
            
            self.hardware_processor = HardwareOptimizedMatrixProcessor(hardware_config)
            
            tprint_success("✅ Advanced hardware optimizations initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to initialize hardware optimizations: {e}")
            self.hardware_manager = None
            self.advanced_memory_optimizer = None
            self.advanced_cpu_optimizer = None
            self.m1_memory_optimizer = None
            self.m1_cpu_optimizer = None
            self.hardware_processor = None
    
    def _initialize_matrix_operations(self):
        """Initialize matrix operations for vectorized computations."""
        if not MATRIX_OPS_AVAILABLE:
            tprint_warning("Matrix operations not available, using basic numpy operations")
            self.matrix_ops = None
            self.vectorized_core = None
            self.enhanced_ops = None
            return
        
        try:
            # Initialize unified matrix operations
            self.matrix_ops = get_unified_matrix_operations(
                enable_gpu=self.config.enable_parallel,
                enable_memory_optimization=True,
                enable_parallel=self.config.enable_parallel
            )
            
            # Initialize vectorized processing core
            self.vectorized_core = get_vectorized_processing_core()
            
            # Initialize enhanced matrix operations
            self.enhanced_ops = get_enhanced_matrix_operations()
            
            tprint_success("✅ Advanced matrix operations initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to initialize matrix operations: {e}")
            self.matrix_ops = None
            self.vectorized_core = None
            self.enhanced_ops = None
    
    def build_feature(self, data: pd.DataFrame, lookback_spec: Any, 
                     feature_name: str) -> HardwareAcceleratedFeatureResult:
        """Build feature using hardware acceleration and matrix operations."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            # Use hardware-optimized data processing
            if self.hardware_processor:
                data = self.hardware_processor.optimize_dataframe_dtypes(data)
                self.performance_metrics['hardware_accelerated_ops'] += 1
            
            # Use memory optimization
            if self.advanced_memory_optimizer:
                data = self.advanced_memory_optimizer.optimize_dataframe(data)
                self.performance_metrics['memory_efficient_ops'] += 1
            
            # Generate feature based on decision type
            if lookback_spec.decision_type.value == 'discrete':
                feature_values = self._build_discrete_feature_hardware_accelerated(data, lookback_spec)
            elif lookback_spec.decision_type.value == 'blend':
                feature_values = self._build_blend_feature_hardware_accelerated(data, lookback_spec)
            else:
                feature_values = self._build_default_feature_hardware_accelerated(data, lookback_spec)
            
            generation_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            quality_score = self._calculate_quality_score_hardware_accelerated(feature_values)
            
            return HardwareAcceleratedFeatureResult(
                family=self._get_family_type(),
                feature_name=feature_name,
                feature_values=feature_values,
                lookback_spec=lookback_spec,
                generation_time=generation_time,
                memory_usage_mb=memory_usage,
                quality_score=quality_score,
                matrix_ops_used=self.performance_metrics['matrix_ops_used'],
                hardware_accelerated_ops=self.performance_metrics['hardware_accelerated_ops'],
                vectorized_ops=self.performance_metrics['vectorized_ops'],
                memory_efficient_ops=self.performance_metrics['memory_efficient_ops'],
                gpu_ops=self.performance_metrics['gpu_ops'],
                cpu_optimized_ops=self.performance_metrics['cpu_optimized_ops']
            )
            
        except Exception as e:
            generation_time = time.time() - start_time
            self.logger.error(f"Hardware-accelerated feature generation failed: {e}")
            return HardwareAcceleratedFeatureResult(
                family=self._get_family_type(),
                feature_name=feature_name,
                feature_values=np.zeros(len(data)),
                lookback_spec=lookback_spec,
                generation_time=generation_time,
                memory_usage_mb=0.0,
                quality_score=0.0
            )
    
    def _build_discrete_feature_hardware_accelerated(self, data: pd.DataFrame, lookback_spec: Any) -> np.ndarray:
        """Build discrete feature with hardware acceleration."""
        if lookback_spec.primary_lookback is None:
            return np.zeros(len(data))
        
        lookback = int(round(lookback_spec.primary_lookback))
        return self._compute_feature_hardware_accelerated(data, lookback)
    
    def _build_blend_feature_hardware_accelerated(self, data: pd.DataFrame, lookback_spec: Any) -> np.ndarray:
        """Build blended feature with hardware acceleration."""
        if (lookback_spec.primary_lookback is None or 
            lookback_spec.secondary_lookback is None or
            lookback_spec.blend_weights is None):
            return np.zeros(len(data))
        
        lookback1 = int(round(lookback_spec.primary_lookback))
        lookback2 = int(round(lookback_spec.secondary_lookback))
        w1, w2 = lookback_spec.blend_weights
        
        # Compute features using hardware acceleration
        feature1 = self._compute_feature_hardware_accelerated(data, lookback1)
        feature2 = self._compute_feature_hardware_accelerated(data, lookback2)
        
        # Blend using vectorized operations
        if self.matrix_ops:
            blended_feature = self.matrix_ops.vectorized_blend([feature1, feature2], [w1, w2])
            self.performance_metrics['matrix_ops_used'] += 1
            self.performance_metrics['vectorized_ops'] += 1
        else:
            blended_feature = w1 * feature1 + w2 * feature2
        
        return blended_feature
    
    def _build_default_feature_hardware_accelerated(self, data: pd.DataFrame, lookback_spec: Any) -> np.ndarray:
        """Build default feature with hardware acceleration."""
        if lookback_spec.primary_lookback is None:
            return np.zeros(len(data))
        
        lookback = int(round(lookback_spec.primary_lookback))
        return self._compute_feature_hardware_accelerated(data, lookback)
    
    def _compute_feature_hardware_accelerated(self, data: pd.DataFrame, lookback: int) -> np.ndarray:
        """Compute the actual feature values using hardware acceleration."""
        # Default implementation - subclasses should override this
        if 'close' not in data.columns:
            return np.zeros(len(data))
        
        # Basic momentum calculation as fallback
        close_prices = data['close'].values
        if len(close_prices) < lookback:
            return np.zeros(len(close_prices))
        
        # Simple momentum calculation
        momentum = np.zeros_like(close_prices)
        for i in range(lookback, len(close_prices)):
            momentum[i] = (close_prices[i] - close_prices[i - lookback]) / close_prices[i - lookback]
        
        return momentum
    
    def _get_family_type(self) -> FamilyType:
        """Get the family type."""
        # Default implementation - subclasses should override this
        return FamilyType.MOMENTUM
    
    def _calculate_quality_score_hardware_accelerated(self, feature_values: np.ndarray) -> float:
        """Calculate quality score using hardware acceleration."""
        try:
            # Remove NaN and infinite values
            clean_values = feature_values[np.isfinite(feature_values)]
            
            if len(clean_values) < 10:
                return 0.0
            
            # Use hardware acceleration for quality metrics if available
            if self.enhanced_ops:
                quality_score = self.enhanced_ops.compute_feature_quality_score(clean_values)
                self.performance_metrics['hardware_accelerated_ops'] += 1
                return quality_score
            else:
                # Fallback to basic quality calculation
                return self._calculate_quality_score_basic(clean_values)
            
        except Exception:
            return 0.0
    
    def _calculate_quality_score_basic(self, clean_values: np.ndarray) -> float:
        """Basic quality score calculation as fallback."""
        try:
            # Calculate various quality metrics
            variance = np.var(clean_values)
            skewness = abs(stats.skew(clean_values))
            kurtosis = abs(stats.kurtosis(clean_values))
            
            # Normalize metrics to [0, 1] range
            variance_score = min(1.0, variance / 0.01)
            skewness_score = max(0.0, 1.0 - skewness / 3.0)
            kurtosis_score = max(0.0, 1.0 - kurtosis / 10.0)
            
            # Weighted combination
            quality_score = (0.5 * variance_score + 
                           0.3 * skewness_score + 
                           0.2 * kurtosis_score)
            
            return float(quality_score)
            
        except Exception:
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
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
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0


class HardwareAcceleratedMomentumBuilder(HardwareAcceleratedFeatureBuilder):
    """Hardware-accelerated momentum feature builder."""
    
    def _get_family_type(self) -> FamilyType:
        return FamilyType.MOMENTUM
    
    def _compute_feature_hardware_accelerated(self, data: pd.DataFrame, lookback: int) -> np.ndarray:
        """Compute momentum feature using hardware acceleration."""
        if 'close' not in data.columns:
            return np.zeros(len(data))
        
        # Use hardware-accelerated momentum calculation
        if self.matrix_ops:
            try:
                # Vectorized momentum calculation
                close_prices = data['close'].values
                momentum = self.matrix_ops.compute_momentum(close_prices, lookback)
                self.performance_metrics['matrix_ops_used'] += 1
                self.performance_metrics['vectorized_ops'] += 1
                return momentum
            except Exception as e:
                logger.warning(f"Matrix momentum calculation failed: {e}, using fallback")
        
        # Fallback to basic calculation
        returns = data['close'].pct_change(lookback)
        return returns.fillna(0).values


class HardwareAcceleratedVolatilityBuilder(HardwareAcceleratedFeatureBuilder):
    """Hardware-accelerated volatility feature builder."""
    
    def _get_family_type(self) -> FamilyType:
        return FamilyType.VOLATILITY
    
    def _compute_feature_hardware_accelerated(self, data: pd.DataFrame, lookback: int) -> np.ndarray:
        """Compute EW volatility feature using hardware acceleration."""
        if 'close' not in data.columns:
            return np.zeros(len(data))
        
        # Use hardware-accelerated volatility calculation
        if self.matrix_ops:
            try:
                # Vectorized EW volatility calculation
                close_prices = data['close'].values
                alpha = 2 / (lookback + 1)
                ew_vol = self.matrix_ops.compute_ew_volatility(close_prices, alpha)
                self.performance_metrics['matrix_ops_used'] += 1
                self.performance_metrics['vectorized_ops'] += 1
                return ew_vol
            except Exception as e:
                logger.warning(f"Matrix volatility calculation failed: {e}, using fallback")
        
        # Fallback to basic calculation
        returns = data['close'].pct_change()
        alpha = 2 / (lookback + 1)
        ew_var = returns.ewm(alpha=alpha).var()
        ew_vol = np.sqrt(ew_var)
        return ew_vol.fillna(0).values


class HardwareAcceleratedRSIBuilder(HardwareAcceleratedFeatureBuilder):
    """Hardware-accelerated RSI feature builder."""
    
    def _get_family_type(self) -> FamilyType:
        return FamilyType.RSI
    
    def _compute_feature_hardware_accelerated(self, data: pd.DataFrame, lookback: int) -> np.ndarray:
        """Compute RSI feature using hardware acceleration."""
        if 'close' not in data.columns:
            return np.zeros(len(data))
        
        # Use hardware-accelerated RSI calculation
        if self.matrix_ops:
            try:
                # Vectorized RSI calculation
                close_prices = data['close'].values
                rsi = self.matrix_ops.compute_rsi(close_prices, lookback)
                self.performance_metrics['matrix_ops_used'] += 1
                self.performance_metrics['vectorized_ops'] += 1
                return rsi
            except Exception as e:
                logger.warning(f"Matrix RSI calculation failed: {e}, using fallback")
        
        # Fallback to basic calculation
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = self._vectorbt_rolling_operation(gain, "mean", lookback)
        avg_loss = self._vectorbt_rolling_operation(loss, "mean", lookback)
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50).values


class HardwareAcceleratedFeatureFactory:
    """Factory for creating hardware-accelerated feature builders."""
    
    @staticmethod
    def create_builder(family: FamilyType, config: LookbackOptimizationConfig) -> HardwareAcceleratedFeatureBuilder:
        """Create appropriate hardware-accelerated builder for family type."""
        builders = {
            FamilyType.MOMENTUM: HardwareAcceleratedMomentumBuilder,
            FamilyType.VOLATILITY: HardwareAcceleratedVolatilityBuilder,
            FamilyType.RSI: HardwareAcceleratedRSIBuilder,
            # Add other families as needed
        }
        
        builder_class = builders.get(family)
        if builder_class is None:
            raise ValueError(f"No hardware-accelerated builder available for family: {family}")
        
        return builder_class(config)


class HardwareAcceleratedMultiFamilyGenerator:
    """Generate features for multiple families using hardware acceleration."""
    
    def __init__(self, config: LookbackOptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize hardware optimizations
        self._initialize_hardware_optimizations()
        
        # Performance tracking
        self.total_metrics = {
            'matrix_ops_used': 0,
            'hardware_accelerated_ops': 0,
            'vectorized_ops': 0,
            'memory_efficient_ops': 0,
            'gpu_ops': 0,
            'cpu_optimized_ops': 0
        }
    
    def _initialize_hardware_optimizations(self):
        """Initialize hardware optimization components."""
        if not HARDWARE_AVAILABLE:
            tprint_warning("Hardware optimizations not available, using CPU-only mode")
            return
        
        try:
            # Initialize unified hardware manager
            self.hardware_manager = UnifiedHardwareManager()
            
            # Initialize advanced optimizers
            self.advanced_memory_optimizer = AdvancedMemoryOptimizer()
            self.advanced_cpu_optimizer = AdvancedCPUOptimizer()
            
            tprint_success("✅ Hardware-accelerated multi-family generator initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to initialize hardware optimizations: {e}")
            self.hardware_manager = None
            self.advanced_memory_optimizer = None
            self.advanced_cpu_optimizer = None
    
    def generate_features(self, data: pd.DataFrame, 
                         decisions: Dict[FamilyType, Any],
                         feature_names: Optional[Dict[FamilyType, str]] = None) -> Dict[FamilyType, HardwareAcceleratedFeatureResult]:
        """Generate features for all families using hardware acceleration."""
        results = {}
        
        if feature_names is None:
            feature_names = {family: f"{family.value}_feature" for family in FamilyType}
        
        # Use memory optimization for large datasets
        if self.advanced_memory_optimizer and len(data) > 10000:
            data = self.advanced_memory_optimizer.optimize_dataframe(data)
            self.total_metrics['memory_efficient_ops'] += 1
        
        for family, decision in decisions.items():
            try:
                tprint_performance(f"Generating {family.value} feature with hardware acceleration...")
                
                # Create hardware-accelerated builder for this family
                builder = HardwareAcceleratedFeatureFactory.create_builder(family, self.config)
                
                # Generate feature
                feature_result = builder.build_feature(
                    data, decision.lookback_spec, feature_names[family]
                )
                
                results[family] = feature_result
                
                # Update total metrics
                self._update_total_metrics(feature_result)
                
                tprint_performance(f"Generated {family.value} feature in {feature_result.generation_time:.3f}s")
                tprint_performance(f"Quality score: {feature_result.quality_score:.3f}")
                
            except Exception as e:
                self.logger.error(f"Failed to generate {family.value} feature: {e}")
                continue
        
        return results
    
    def _update_total_metrics(self, feature_result: HardwareAcceleratedFeatureResult):
        """Update total performance metrics."""
        self.total_metrics['matrix_ops_used'] += feature_result.matrix_ops_used
        self.total_metrics['hardware_accelerated_ops'] += feature_result.hardware_accelerated_ops
        self.total_metrics['vectorized_ops'] += feature_result.vectorized_ops
        self.total_metrics['memory_efficient_ops'] += feature_result.memory_efficient_ops
        self.total_metrics['gpu_ops'] += feature_result.gpu_ops
        self.total_metrics['cpu_optimized_ops'] += feature_result.cpu_optimized_ops
    
    def create_feature_matrix(self, feature_results: Dict[FamilyType, HardwareAcceleratedFeatureResult]) -> Tuple[np.ndarray, List[str]]:
        """Create feature matrix from hardware-accelerated results."""
        features = []
        feature_names = []
        
        for family, result in feature_results.items():
            if result.feature_values is not None and len(result.feature_values) > 0:
                features.append(result.feature_values)
                feature_names.append(result.feature_name)
        
        if features:
            feature_matrix = np.column_stack(features)
        else:
            feature_matrix = np.array([]).reshape(0, 0)
        
        return feature_matrix, feature_names
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            'total_metrics': self.total_metrics,
            'hardware_available': HARDWARE_AVAILABLE,
            'matrix_ops_available': MATRIX_OPS_AVAILABLE,
            'optimization_efficiency': self._calculate_optimization_efficiency()
        }
    
    def _calculate_optimization_efficiency(self) -> float:
        """Calculate overall optimization efficiency."""
        total_ops = sum(self.total_metrics.values())
        if total_ops == 0:
            return 0.0
        
        optimized_ops = (self.total_metrics['matrix_ops_used'] + 
                        self.total_metrics['hardware_accelerated_ops'] + 
                        self.total_metrics['vectorized_ops'])
        
        return optimized_ops / total_ops if total_ops > 0 else 0.0