"""
Optimized Lookback Component with Matrix Operations and Hardware Acceleration

This component integrates the data-driven lookback optimization system into the
Ares pipeline, replacing the PID-based approach with a more efficient and
rigorous Bayesian optimization system.
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import warnings

import numpy as np
import pandas as pd

# Import pipeline components
from ...components.base_component import BasePreTrainingComponent, ComponentConfig, ComponentResult
from ...components.component_factory import register_component

# Import the optimization system
from .orchestrator import LookbackOptimizationOrchestrator
from .config import create_production_config, FamilyType

# Import matrix operations and hardware optimizations
try:
    from src.utils.matrix_operations.unified_operations import get_unified_matrix_operations
    from src.utils.matrix_operations.batch_operations import batch_matrix_multiply, batch_correlation_analysis
    from src.utils.matrix_operations.hardware_integration import HardwareOptimizedMatrixProcessor, HardwareConfig
    from src.utils.hardware.m1_optimizations import M1MemoryOptimizer, M1CPUOptimizer
    from src.utils.hardware.memory_optimization import memory_efficient, optimize_dataframe_dtypes, chunk_dataframe
    from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
    MATRIX_OPS_AVAILABLE = True
    HARDWARE_AVAILABLE = True
except ImportError as e:
    MATRIX_OPS_AVAILABLE = False
    HARDWARE_AVAILABLE = False
    logging.warning(f"Matrix operations or hardware optimizations not available: {e}")

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


@register_component('optimized_lookback_generation')
class OptimizedLookbackComponent(BasePreTrainingComponent):
    """
    Optimized Lookback Component with Matrix Operations and Hardware Acceleration.
    
    This component replaces the PID-based feature generation with a more efficient
    data-driven lookback optimization system that uses:
    - Matrix operations for vectorized computations
    - Hardware acceleration for CPU/GPU optimization
    - Memory-efficient processing for large datasets
    - Parallel processing for multiple symbols
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the optimized lookback component."""
        super().__init__(config)
        
        # Initialize optimization system
        self.optimization_config = create_production_config()
        self.orchestrator = LookbackOptimizationOrchestrator(self.optimization_config)
        
        # Initialize hardware optimizations
        self._initialize_hardware_optimizations()
        
        # Initialize matrix operations
        self._initialize_matrix_operations()
        
        # Performance tracking
        self.performance_metrics = {
            'matrix_ops_used': 0,
            'hardware_accelerated_ops': 0,
            'memory_efficient_ops': 0,
            'parallel_ops': 0,
            'total_execution_time': 0.0,
            'memory_usage_mb': 0.0
        }
        
        tprint_info("🚀 Initialized OptimizedLookbackComponent with hardware acceleration")
    
    def _initialize_hardware_optimizations(self):
        """Initialize hardware optimization components."""
        if not HARDWARE_AVAILABLE:
            tprint_warning("Hardware optimizations not available, using CPU-only mode")
            return
        
        try:
            # Initialize unified hardware manager
            self.hardware_manager = UnifiedHardwareManager()
            
            # Initialize M1 optimizations
            self.m1_memory_optimizer = M1MemoryOptimizer(memory_limit_gb=8.0)
            self.m1_cpu_optimizer = M1CPUOptimizer()
            
            # Initialize hardware-optimized matrix processor
            hardware_config = HardwareConfig(
                max_memory_gb=self.optimization_config.memory_limit_gb,
                enable_gpu=self.optimization_config.enable_parallel,
                max_cpu_cores=self.optimization_config.n_workers,
                auto_optimize_dtypes=True,
                auto_chunk_large_data=True
            )
            
            self.hardware_processor = HardwareOptimizedMatrixProcessor(hardware_config)
            
            tprint_success("✅ Hardware optimizations initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to initialize hardware optimizations: {e}")
            self.hardware_manager = None
            self.m1_memory_optimizer = None
            self.m1_cpu_optimizer = None
            self.hardware_processor = None
    
    def _initialize_matrix_operations(self):
        """Initialize matrix operations for vectorized computations."""
        if not MATRIX_OPS_AVAILABLE:
            tprint_warning("Matrix operations not available, using basic numpy operations")
            self.matrix_ops = None
            return
        
        try:
            # Initialize unified matrix operations
            self.matrix_ops = get_unified_matrix_operations(
                enable_gpu=self.optimization_config.enable_parallel,
                enable_memory_optimization=True,
                enable_parallel=self.optimization_config.enable_parallel
            )
            
            tprint_success("✅ Matrix operations initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to initialize matrix operations: {e}")
            self.matrix_ops = None
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return [
            'optimized_lookback_results',
            'feature_interaction_matrix',
            'feature_names',
            'optimization_metrics',
            'hardware_utilization_report'
        ]
    
    async def execute(self, data: Optional[Any], pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute the optimized lookback optimization and feature generation.
        
        Args:
            data: Input data (not used, data comes from pipeline_state)
            pipeline_state: Pipeline state containing market data and labels
            
        Returns:
            ComponentResult with optimization results and generated features
        """
        start_time = time.time()
        
        try:
            tprint_info("🚀 Starting optimized lookback optimization and feature generation...")
            
            # Extract data from pipeline state
            market_data, targets, feature_names = self._extract_data_from_pipeline_state(pipeline_state)
            
            if market_data is None or targets is None:
                raise ValueError("No market data or targets found in pipeline state")
            
            # Optimize lookbacks using the three-stage system
            tprint_info("📊 Stage 1-3: Running lookback optimization...")
            optimization_result = await self._run_optimized_lookback_optimization(
                market_data, targets, feature_names
            )
            
            if not optimization_result.success:
                raise RuntimeError(f"Lookback optimization failed: {optimization_result.error_message}")
            
            # Generate optimized features using matrix operations
            tprint_info("⚙️ Generating optimized features with matrix operations...")
            feature_matrix, optimized_feature_names, feature_metrics = await self._generate_optimized_features(
                market_data, optimization_result
            )
            
            # Generate interaction features using optimized parents
            tprint_info("🔗 Generating interaction features...")
            interaction_matrix, interaction_names = await self._generate_interaction_features(
                feature_matrix, optimized_feature_names, targets, optimization_result
            )
            
            # Combine all features
            final_feature_matrix = np.column_stack([feature_matrix, interaction_matrix])
            final_feature_names = optimized_feature_names + interaction_names
            
            # Generate performance report
            performance_report = self._generate_performance_report(optimization_result, feature_metrics)
            
            # Update performance metrics
            execution_time = time.time() - start_time
            self.performance_metrics['total_execution_time'] = execution_time
            self.performance_metrics['memory_usage_mb'] = self._get_memory_usage()
            
            # Create artifacts
            artifacts = {
                'optimized_lookback_results': optimization_result.to_dict(),
                'feature_interaction_matrix': final_feature_matrix,
                'feature_names': final_feature_names,
                'optimization_metrics': self.performance_metrics,
                'hardware_utilization_report': performance_report,
                'feature_generation_metadata': {
                    'total_features': len(final_feature_names),
                    'base_features': len(optimized_feature_names),
                    'interaction_features': len(interaction_names),
                    'execution_time': execution_time,
                    'matrix_ops_used': self.performance_metrics['matrix_ops_used'],
                    'hardware_accelerated_ops': self.performance_metrics['hardware_accelerated_ops']
                }
            }
            
            tprint_success(f"✅ Optimized lookback optimization completed in {execution_time:.3f}s")
            tprint_success(f"📊 Generated {len(final_feature_names)} total features")
            tprint_success(f"⚡ Matrix operations used: {self.performance_metrics['matrix_ops_used']}")
            tprint_success(f"🚀 Hardware accelerated ops: {self.performance_metrics['hardware_accelerated_ops']}")
            
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'component_type': 'optimized_lookback_component',
                    'execution_time': execution_time,
                    'features_generated': len(final_feature_names),
                    'optimization_success': True
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = f"Optimized lookback component failed: {str(e)}"
            
            tprint_error(f"❌ {error_message}")
            tprint_error(f"❌ Error details: {traceback.format_exc()}")
            
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=error_message,
                metadata={
                    'component_type': 'optimized_lookback_component',
                    'execution_time': execution_time,
                    'optimization_success': False
                }
            )
    
    def _extract_data_from_pipeline_state(self, pipeline_state: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[Dict], Dict]:
        """Extract market data and targets from pipeline state."""
        try:
            # Get symbol and timeframe
            symbol = pipeline_state.get('symbol', 'ETHUSDT')
            timeframe = pipeline_state.get('timeframe', '15m')
            
            # Look for market data in various possible locations
            market_data = None
            targets = None
            
            # Check for direct market data
            if 'market_data' in pipeline_state:
                market_data = {symbol: pipeline_state['market_data']}
            elif 'historical_data' in pipeline_state:
                market_data = {symbol: pipeline_state['historical_data']}
            elif 'data' in pipeline_state:
                market_data = {symbol: pipeline_state['data']}
            
            # Check for labels/targets
            if 'multi_horizon_labeling_result' in pipeline_state:
                labeling_result = pipeline_state['multi_horizon_labeling_result']
                if 'labeled_data' in labeling_result:
                    labeled_data = labeling_result['labeled_data']
                    if isinstance(labeled_data, pd.DataFrame):
                        # Extract targets from labeled data
                        target_columns = [col for col in labeled_data.columns if 'target' in col.lower() or 'label' in col.lower()]
                        if target_columns:
                            targets = {symbol: labeled_data[target_columns[0]].values}
                        else:
                            # Use the last column as target
                            targets = {symbol: labeled_data.iloc[:, -1].values}
            
            # Create default feature names
            feature_names = {family: f"{family.value}_feature" for family in FamilyType}
            
            return market_data, targets, feature_names
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract data from pipeline state: {e}")
            return None, None, {}
    
    async def _run_optimized_lookback_optimization(self, market_data: Dict, targets: Dict, feature_names: Dict) -> Any:
        """Run the three-stage lookback optimization with hardware acceleration."""
        try:
            # Use hardware-optimized data processing if available
            if self.hardware_processor:
                optimized_market_data = {}
                for symbol, data in market_data.items():
                    optimized_data = self.hardware_processor.optimize_dataframe_dtypes(data)
                    optimized_market_data[symbol] = optimized_data
                    self.performance_metrics['hardware_accelerated_ops'] += 1
                
                market_data = optimized_market_data
            
            # Run optimization
            result = self.orchestrator.optimize_lookbacks(market_data, targets, feature_names)
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Lookback optimization failed: {e}")
            raise
    
    async def _generate_optimized_features(self, market_data: Dict, optimization_result: Any) -> Tuple[np.ndarray, List[str], Dict]:
        """Generate optimized features using matrix operations."""
        try:
            all_features = []
            all_feature_names = []
            feature_metrics = {}
            
            for symbol, symbol_data in market_data.items():
                if symbol not in optimization_result.decisions:
                    continue
                
                symbol_decisions = optimization_result.decisions[symbol]
                
                # Generate features for this symbol using matrix operations
                symbol_features, symbol_names, symbol_metrics = await self._generate_symbol_features(
                    symbol_data, symbol_decisions, symbol
                )
                
                if symbol_features is not None and len(symbol_features) > 0:
                    all_features.append(symbol_features)
                    all_feature_names.extend(symbol_names)
                    feature_metrics[symbol] = symbol_metrics
                    self.performance_metrics['matrix_ops_used'] += 1
            
            if all_features:
                # Combine features from all symbols
                feature_matrix = np.vstack(all_features)
            else:
                feature_matrix = np.array([]).reshape(0, 0)
            
            return feature_matrix, all_feature_names, feature_metrics
            
        except Exception as e:
            tprint_error(f"❌ Feature generation failed: {e}")
            return np.array([]).reshape(0, 0), [], {}
    
    async def _generate_symbol_features(self, symbol_data: pd.DataFrame, symbol_decisions: Dict, symbol: str) -> Tuple[Optional[np.ndarray], List[str], Dict]:
        """Generate features for a single symbol using optimized lookbacks."""
        try:
            features = []
            feature_names = []
            metrics = {'generation_time': 0.0, 'memory_usage': 0.0}
            
            start_time = time.time()
            
            for family, decision in symbol_decisions.items():
                if decision.lookback_spec.decision_type.value == 'inactive':
                    continue
                
                # Generate feature using optimized lookback
                feature_values = await self._generate_family_feature(
                    symbol_data, family, decision.lookback_spec
                )
                
                if feature_values is not None and len(feature_values) > 0:
                    features.append(feature_values)
                    feature_names.append(f"{symbol}_{family.value}_feature")
            
            if features:
                feature_matrix = np.column_stack(features)
            else:
                feature_matrix = None
            
            metrics['generation_time'] = time.time() - start_time
            metrics['memory_usage'] = self._get_memory_usage()
            
            return feature_matrix, feature_names, metrics
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate features for {symbol}: {e}")
            return None, [], {}
    
    async def _generate_family_feature(self, data: pd.DataFrame, family: FamilyType, lookback_spec: Any) -> Optional[np.ndarray]:
        """Generate feature for a specific family using optimized lookback."""
        try:
            if lookback_spec.effective_lookback is None:
                return None
            
            lookback = int(round(lookback_spec.effective_lookback))
            
            # Use matrix operations for efficient computation
            if self.matrix_ops and family == FamilyType.MOMENTUM:
                # Vectorized momentum calculation
                if 'close' in data.columns:
                    returns = data['close'].pct_change(lookback).values
                    return returns
            elif self.matrix_ops and family == FamilyType.VOLATILITY:
                # Vectorized volatility calculation
                if 'close' in data.columns:
                    returns = data['close'].pct_change().values
                    alpha = 2 / (lookback + 1)
                    # Use matrix operations for EW calculation
                    ew_var = self._compute_ew_variance_vectorized(returns, alpha)
                    return np.sqrt(ew_var)
            elif self.matrix_ops and family == FamilyType.RSI:
                # Vectorized RSI calculation
                if 'close' in data.columns:
                    returns = data['close'].pct_change().values
                    rsi = self._compute_rsi_vectorized(returns, lookback)
                    return rsi
            
            # Fallback to basic calculation
            return self._compute_basic_feature(data, family, lookback)
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate {family.value} feature: {e}")
            return None
    
    def _compute_ew_variance_vectorized(self, returns: np.ndarray, alpha: float) -> np.ndarray:
        """Compute exponentially weighted variance using vectorized operations."""
        if self.matrix_ops:
            # Use matrix operations for efficient EW calculation
            return self.matrix_ops.compute_ew_variance(returns, alpha)
        else:
            # Fallback to pandas
            series = pd.Series(returns)
            return series.ewm(alpha=alpha).var().values
    
    def _compute_rsi_vectorized(self, returns: np.ndarray, period: int) -> np.ndarray:
        """Compute RSI using vectorized operations."""
        if self.matrix_ops:
            # Use matrix operations for efficient RSI calculation
            return self.matrix_ops.compute_rsi(returns, period)
        else:
            # Fallback to pandas
            series = pd.Series(returns)
            gain = series.where(series > 0, 0)
            loss = -series.where(series < 0, 0)
            avg_gain = self._vectorbt_rolling_operation(gain, "mean", period)
            avg_loss = self._vectorbt_rolling_operation(loss, "mean", period)
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50).values
    
    def _compute_basic_feature(self, data: pd.DataFrame, family: FamilyType, lookback: int) -> Optional[np.ndarray]:
        """Compute basic feature as fallback."""
        try:
            if family == FamilyType.MOMENTUM and 'close' in data.columns:
                return data['close'].pct_change(lookback).fillna(0).values
            elif family == FamilyType.VOLATILITY and 'close' in data.columns:
                returns = data['close'].pct_change()
                alpha = 2 / (lookback + 1)
                ew_var = returns.ewm(alpha=alpha).var()
                return np.sqrt(ew_var.fillna(0)).values
            else:
                return np.zeros(len(data))
        except Exception:
            return np.zeros(len(data))
    
    async def _generate_interaction_features(self, feature_matrix: np.ndarray, feature_names: List[str], 
                                           targets: Dict, optimization_result: Any) -> Tuple[np.ndarray, List[str]]:
        """Generate interaction features using optimized parent features."""
        try:
            if len(feature_matrix) == 0 or len(feature_names) == 0:
                return np.array([]).reshape(0, 0), []
            
            # Use matrix operations for efficient interaction generation
            if self.matrix_ops:
                # Generate pairwise interactions using matrix operations
                interactions = self._generate_pairwise_interactions_matrix_ops(feature_matrix)
                interaction_names = [f"interaction_{i}" for i in range(interactions.shape[1])]
                
                self.performance_metrics['matrix_ops_used'] += 1
                self.performance_metrics['hardware_accelerated_ops'] += 1
                
                return interactions, interaction_names
            else:
                # Fallback to basic interaction generation
                interactions = self._generate_pairwise_interactions_basic(feature_matrix)
                interaction_names = [f"interaction_{i}" for i in range(interactions.shape[1])]
                
                return interactions, interaction_names
                
        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate interaction features: {e}")
            return np.array([]).reshape(0, 0), []
    
    def _generate_pairwise_interactions_matrix_ops(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Generate pairwise interactions using matrix operations."""
        if self.matrix_ops:
            # Use batch matrix operations for efficiency
            n_features = min(10, feature_matrix.shape[1])  # Limit for performance
            interactions = []
            
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    # Vectorized multiplication
                    interaction = feature_matrix[:, i] * feature_matrix[:, j]
                    interactions.append(interaction)
            
            if interactions:
                return np.column_stack(interactions)
            else:
                return np.array([]).reshape(feature_matrix.shape[0], 0)
        else:
            return self._generate_pairwise_interactions_basic(feature_matrix)
    
    def _generate_pairwise_interactions_basic(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Generate pairwise interactions using basic operations."""
        n_features = min(10, feature_matrix.shape[1])
        interactions = []
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                interaction = feature_matrix[:, i] * feature_matrix[:, j]
                interactions.append(interaction)
        
        if interactions:
            return np.column_stack(interactions)
        else:
            return np.array([]).reshape(feature_matrix.shape[0], 0)
    
    def _generate_performance_report(self, optimization_result: Any, feature_metrics: Dict) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            'optimization_performance': {
                'execution_time': optimization_result.execution_time,
                'success': optimization_result.success,
                'symbols_processed': len(optimization_result.ic_surface_results)
            },
            'feature_generation_performance': feature_metrics,
            'hardware_utilization': {
                'matrix_ops_used': self.performance_metrics['matrix_ops_used'],
                'hardware_accelerated_ops': self.performance_metrics['hardware_accelerated_ops'],
                'memory_efficient_ops': self.performance_metrics['memory_efficient_ops'],
                'parallel_ops': self.performance_metrics['parallel_ops']
            },
            'memory_usage': {
                'peak_memory_mb': self.performance_metrics['memory_usage_mb'],
                'memory_optimization_enabled': self.m1_memory_optimizer is not None
            }
        }
    
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
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            'component_metrics': self.performance_metrics,
            'hardware_available': HARDWARE_AVAILABLE,
            'matrix_ops_available': MATRIX_OPS_AVAILABLE,
            'optimization_config': self.optimization_config.to_dict()
        }