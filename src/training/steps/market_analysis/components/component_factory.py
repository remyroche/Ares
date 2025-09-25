"""
Component Factory for Market Analysis Pipeline Components.

This factory manages the creation and registration of all pipeline components.
Enhanced with comprehensive error handling, logging, and utility integration.
"""

import numpy as np
import pandas as pd
import glob
import pickle
import logging
import time
import asyncio
from typing import Dict, Type, Any, Optional, List, Union, Callable
from pathlib import Path
from contextlib import contextmanager

# Enhanced imports with comprehensive utility integration
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer,
    tprint_structured, tprint_with_level, LogLevel
)
from src.utils.common_operations import (
    safe_json_load, safe_json_dump, safe_file_exists, ensure_directory,
    validate_dataframe, validate_dataframe_columns, safe_dataframe_operation,
    create_data_quality_report, get_dataframe_info, optimize_dataframe_dtypes,
    safe_merge_dataframes, safe_drop_columns, safe_rename_columns,
    integrate_with_m1_optimizers, cleanup_m1_optimizers, get_m1_gpu_manager,
    get_m1_memory_optimizer, get_m1_cpu_optimizer, memory_checkpoint, gpu_context
)
from src.utils.common_utilities import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite, validate_positive,
    validate_range, safe_correlation, safe_covariance, safe_mean, safe_std,
    safe_percentile, validate_correlation_matrix, safe_matrix_inverse, math_safe,
    MathValidation, MathValidationError
)
from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)
from src.utils.nas_tas.bayesian_tpe_optimizer import (
    BayesianTPEOptimizer, BayesianTPEConfig, OptimizationResult,
    optimize_with_bayesian_tpe, create_search_space_from_bounds
)
from src.utils.hardware.m1_gpu_utils import (
    get_m1_gpu_manager, is_m1_available, is_mps_available, optimize_dataframe_for_m1,
    create_m1_optimized_array, m1_backtesting_simulate, m1_monte_carlo_simulate
)
from src.utils.hardware.m1_memory_optimizer import (
    get_m1_memory_optimizer, optimize_dataframe_memory, optimize_memory,
    get_memory_usage, start_m1_memory_monitoring, stop_m1_memory_monitoring
)
from src.utils.hardware.m1_cpu_optimizer import (
    get_m1_cpu_optimizer, optimize_function_for_m1, parallel_map_m1,
    create_m1_optimized_thread_pool, run_cpu_intensive_task,
    parallel_backtesting_worker, parallel_monte_carlo_simulation
)

# Component imports
from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
from .sr_parameter_optimization import SRParameterOptimizationComponent
from .sr_detection import SRDetectionComponent
from .sr_clustering import SRClusteringComponent
from .nas_regime_discovery import NASRegimeDiscoveryComponent
from .tas_regime_discovery import TASRegimeDiscoveryComponent
from .hybrid_nas_tas_regime_discovery import HybridNASTASRegimeDiscoveryComponent
from .nas_tas_clustering import NASTASClusteringComponent
from .feature_lookback_optimization import FeatureLookbackOptimizationComponent
from .cross_timeframe_analysis import CrossTimeframeAnalysisComponent
from .final_feature_selection import FinalFeatureSelectionComponent

# Import the actual PID-based component for direct use
try:
    from ..pid_based_feature_generation.pid_based_feature_generation_component import PIDBasedFeatureGenerationComponent
    PID_COMPONENT_AVAILABLE = True
    tprint_info("✅ PID-based feature generation component available")
except ImportError as e:
    PID_COMPONENT_AVAILABLE = False
    tprint_warning(f"⚠️ PID-based feature generation component not available: {e}")

# Enhanced logging setup
logger = logging.getLogger(__name__)

# Initialize utility managers
_math_validator = MathValidation()
_serializer = UniversalSerializer()
_m1_gpu_manager = None
_m1_memory_optimizer = None
_m1_cpu_optimizer = None

def _initialize_managers():
    """Initialize utility managers with error handling."""
    global _m1_gpu_manager, _m1_memory_optimizer, _m1_cpu_optimizer
    
    try:
        _m1_gpu_manager = get_m1_gpu_manager()
        tprint_info("✅ M1 GPU manager initialized")
    except Exception as e:
        tprint_warning(f"⚠️ M1 GPU manager initialization failed: {e}")
        _m1_gpu_manager = None
    
    try:
        _m1_memory_optimizer = get_m1_memory_optimizer()
        tprint_info("✅ M1 memory optimizer initialized")
    except Exception as e:
        tprint_warning(f"⚠️ M1 memory optimizer initialization failed: {e}")
        _m1_memory_optimizer = None
    
    try:
        _m1_cpu_optimizer = get_m1_cpu_optimizer()
        tprint_info("✅ M1 CPU optimizer initialized")
    except Exception as e:
        tprint_warning(f"⚠️ M1 CPU optimizer initialization failed: {e}")
        _m1_cpu_optimizer = None

# Initialize managers on import
_initialize_managers()


class MultiHorizonComponentWrapper(BaseMarketAnalysisComponent):
    """Enhanced wrapper for Multi-Horizon Profit Labeler with comprehensive error handling and utility integration."""
    
    def __init__(self, adapter_class, config: Optional[ComponentConfig] = None):
        """Initialize the wrapper with enhanced error handling."""
        try:
            super().__init__(config)
            self.adapter_class = adapter_class
            self.adapter_instance = None
            self._execution_count = 0
            self._last_execution_time = 0.0
            
            # Validate adapter class
            if not callable(adapter_class):
                raise ValueError(f"Adapter class must be callable, got {type(adapter_class)}")
            
            tprint_success(f"✅ MultiHorizonComponentWrapper initialized with {adapter_class.__name__}")
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize MultiHorizonComponentWrapper: {e}")
            raise
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['multi_horizon_labeling_result']
    
    def _validate_input_data(self, data, pipeline_state: Dict[str, Any]) -> bool:
        """Validate input data and pipeline state."""
        try:
            # Validate data
            if data is None:
                tprint_error("❌ Input data is None")
                return False
            
            if not hasattr(data, '__len__'):
                tprint_error("❌ Input data must have length attribute")
                return False
            
            if len(data) == 0:
                tprint_error("❌ Input data is empty")
                return False
            
            # Validate pipeline state
            if not isinstance(pipeline_state, dict):
                tprint_error("❌ Pipeline state must be a dictionary")
                return False
            
            # Check for required keys
            required_keys = ['symbol', 'exchange', 'timeframe']
            missing_keys = [key for key in required_keys if key not in pipeline_state]
            if missing_keys:
                tprint_warning(f"⚠️ Missing pipeline state keys: {missing_keys}")
            
            tprint_debug(f"✅ Input validation passed - data length: {len(data)}")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Input validation failed: {e}")
            return False
    
    def _optimize_data_for_execution(self, data, execution_mode: str) -> Any:
        """Optimize data for execution with M1 hardware optimizations."""
        try:
            original_size = len(data)
            
            # Apply execution mode filtering
            if execution_mode.lower() == 'light' and original_size > 20000:
                data = data.tail(14400).copy()  # 10 days for 1m data
                tprint_info(f"🔥 LIGHT FILTERING: {original_size:,} → {len(data):,} rows")
            elif execution_mode.lower() == 'blank' and original_size > 300000:
                data = data.tail(259200).copy()  # 180 days for 1m data  
                tprint_info(f"🔥 BLANK FILTERING: {original_size:,} → {len(data):,} rows")
            
            # Apply M1 optimizations if available
            if _m1_memory_optimizer and hasattr(data, 'memory_usage'):
                try:
                    data = _m1_memory_optimizer.optimize_dataframe_memory(data)
                    tprint_debug("✅ Applied M1 memory optimization to data")
                except Exception as e:
                    tprint_warning(f"⚠️ M1 memory optimization failed: {e}")
            
            # Apply GPU optimizations if available
            if _m1_gpu_manager and is_m1_available():
                try:
                    data = optimize_dataframe_for_m1(data)
                    tprint_debug("✅ Applied M1 GPU optimization to data")
                except Exception as e:
                    tprint_warning(f"⚠️ M1 GPU optimization failed: {e}")
            
            return data
            
        except Exception as e:
            tprint_error(f"❌ Data optimization failed: {e}")
            return data
    
    def _create_adapter_instance(self) -> Any:
        """Create adapter instance with error handling."""
        try:
            if self.adapter_instance is None:
                tprint_debug("🔧 Creating adapter instance")
                self.adapter_instance = self.adapter_class()
                
                # Validate adapter instance
                if not hasattr(self.adapter_instance, 'execute_multi_horizon_labeling_step'):
                    raise AttributeError("Adapter instance must have 'execute_multi_horizon_labeling_step' method")
                
                tprint_success("✅ Adapter instance created successfully")
            
            return self.adapter_instance
            
        except Exception as e:
            tprint_error(f"❌ Failed to create adapter instance: {e}")
            raise
    
    def _extract_labeling_config(self) -> Dict[str, Any]:
        """Extract labeling configuration with validation."""
        try:
            labeling_config = {}
            
            if self.config and hasattr(self.config, 'custom_params'):
                custom_params = self.config.custom_params
                if isinstance(custom_params, dict):
                    labeling_config = custom_params.get('multi_horizon_labeling', {})
                    tprint_debug(f"✅ Extracted labeling config: {len(labeling_config)} parameters")
                else:
                    tprint_warning("⚠️ Custom params is not a dictionary")
            else:
                tprint_debug("ℹ️ No custom params found, using default config")
            
            return labeling_config
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract labeling config: {e}")
            return {}
    
    def _determine_execution_mode(self, pipeline_state: Dict[str, Any]) -> str:
        """Determine execution mode with fallback logic."""
        try:
            execution_mode = 'full'  # Default
            
            # Try multiple sources for execution mode
            if pipeline_state.get('execution_mode'):
                execution_mode = pipeline_state.get('execution_mode')
                tprint_debug(f"✅ Execution mode from pipeline state: {execution_mode}")
            elif self.config and hasattr(self.config, 'mode'):
                execution_mode = self.config.mode.value if hasattr(self.config.mode, 'value') else str(self.config.mode)
                tprint_debug(f"✅ Execution mode from config: {execution_mode}")
            elif pipeline_state.get('mode'):
                execution_mode = pipeline_state.get('mode')
                tprint_debug(f"✅ Execution mode from pipeline mode: {execution_mode}")
            else:
                tprint_debug("ℹ️ Using default execution mode: full")
            
            # Validate execution mode
            valid_modes = ['full', 'light', 'blank', 'test']
            if execution_mode.lower() not in valid_modes:
                tprint_warning(f"⚠️ Invalid execution mode '{execution_mode}', using 'full'")
                execution_mode = 'full'
            
            return execution_mode.lower()
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to determine execution mode: {e}")
            return 'full'
    
    async def execute(self, data, pipeline_state: Dict[str, Any]) -> 'ComponentResult':
        """Execute multi-horizon labeling with comprehensive error handling and optimization."""
        start_time = time.time()
        self._execution_count += 1
        
        tprint_info(f"🚀 Starting Multi-Horizon Component execution #{self._execution_count}")
        
        try:
            # Input validation
            if not self._validate_input_data(data, pipeline_state):
                return ComponentResult(
                    success=False,
                    artifacts={},
                    metadata={'execution_count': self._execution_count},
                    error_message="Input validation failed"
                )
            
            # Determine execution mode
            execution_mode = self._determine_execution_mode(pipeline_state)
            tprint_info(f"📊 Execution mode: {execution_mode}")
            
            # Optimize data for execution
            optimized_data = self._optimize_data_for_execution(data, execution_mode)
            
            # Create adapter instance
            adapter_instance = self._create_adapter_instance()
            
            # Extract configuration
            labeling_config = self._extract_labeling_config()
            
            # Prepare execution parameters
            execution_params = {
                'data': optimized_data,
                'regime_labels': pipeline_state.get('regime_labels'),
                'config': labeling_config,
                'symbol': pipeline_state.get('symbol', 'UNKNOWN'),
                'exchange': pipeline_state.get('exchange', 'UNKNOWN'),
                'timeframe': pipeline_state.get('timeframe', 'UNKNOWN'),
                'mode': execution_mode,
                'features': pipeline_state.get('pid_based_features')
            }
            
            tprint_info("🎯 Executing multi-horizon labeling step")
            
            # Execute with memory checkpoint if available
            if _m1_memory_optimizer:
                with _m1_memory_optimizer.memory_checkpoint("multi_horizon_labeling"):
                    result = adapter_instance.execute_multi_horizon_labeling_step(**execution_params)
            else:
                result = adapter_instance.execute_multi_horizon_labeling_step(**execution_params)
            
            # Validate result
            if result is None:
                tprint_error("❌ Multi-horizon labeling returned None result")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    metadata={'execution_count': self._execution_count, 'execution_time': time.time() - start_time},
                    error_message="Multi-horizon labeling returned None result"
                )
            
            # Process result
            if result.get('status') == 'completed':
                execution_time = time.time() - start_time
                self._last_execution_time = execution_time
                
                tprint_success(f"✅ Multi-horizon labeling completed in {execution_time:.2f}s")
                
                return ComponentResult(
                    success=True,
                    artifacts=result.get('artifacts', {}),
                    metadata={
                        'execution_count': self._execution_count,
                        'execution_time': execution_time,
                        'execution_mode': execution_mode,
                        'data_size': len(optimized_data)
                    },
                    error_message=None
                )
            else:
                error_msg = result.get('error', 'Unknown error in multi-horizon labeling')
                tprint_error(f"❌ Multi-horizon labeling failed: {error_msg}")
                
                return ComponentResult(
                    success=False,
                    artifacts=result.get('artifacts', {}),
                    metadata={
                        'execution_count': self._execution_count,
                        'execution_time': time.time() - start_time,
                        'execution_mode': execution_mode
                    },
                    error_message=error_msg
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Multi-horizon labeling component failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            
            return ComponentResult(
                success=False,
                artifacts={},
                metadata={
                    'execution_count': self._execution_count,
                    'execution_time': execution_time,
                    'error_type': type(e).__name__
                },
                error_message=error_msg
            )

class HMMModelsTrainingComponentWrapper(BaseMarketAnalysisComponent):
    """Enhanced wrapper for HMM Models Training with comprehensive error handling and utility integration."""
    
    def __init__(self, training_class, config: Optional[ComponentConfig] = None):
        """Initialize the wrapper with enhanced error handling."""
        try:
            super().__init__(config)
            self.training_class = training_class
            self.training_instance = None
            self._execution_count = 0
            self._last_execution_time = 0.0
            
            # Validate training class
            if not callable(training_class):
                raise ValueError(f"Training class must be callable, got {type(training_class)}")
            
            tprint_success(f"✅ HMMModelsTrainingComponentWrapper initialized with {training_class.__name__}")
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize HMMModelsTrainingComponentWrapper: {e}")
            raise
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['hmm_models_training_result']
    
    def _create_training_instance(self) -> Any:
        """Create training instance with error handling and configuration."""
        try:
            if self.training_instance is None:
                tprint_debug("🔧 Creating HMM training instance")
                self.training_instance = self.training_class()
                
                # Enforce 15m timeframe for HMM models at runtime
                if hasattr(self.training_instance, 'config'):
                    setattr(self.training_instance.config, 'timeframe', '15m')
                    current_timeframe = getattr(self.training_instance.config, 'timeframe', None)
                    if current_timeframe != '15m':
                        tprint_warning("⚠️ HMM Models: Non-15m timeframe supplied; overriding to 15m for consistency")
                
                tprint_success("✅ HMM training instance created successfully")
            
            return self.training_instance
            
        except Exception as e:
            tprint_error(f"❌ Failed to create HMM training instance: {e}")
            raise
    
    def _validate_training_data(self, X, y, cluster_assignments) -> bool:
        """Validate training data with comprehensive checks."""
        try:
            # Check X and y alignment
            if X is not None and y is not None and len(X) != len(y):
                tprint_error(f"❌ X and y length mismatch: X={len(X)}, y={len(y)}")
                return False
            
            # Validate cluster assignments
            if cluster_assignments is not None:
                if len(cluster_assignments) == 0:
                    tprint_error("❌ Cluster assignments is empty")
                    return False
                
                # Check for valid cluster values
                unique_clusters = len(set(cluster_assignments))
                if unique_clusters < 2:
                    tprint_warning(f"⚠️ Only {unique_clusters} unique clusters found")
                
                tprint_debug(f"✅ Cluster assignments validated: {len(cluster_assignments)} samples, {unique_clusters} clusters")
            
            # Validate features
            if X is not None:
                if X.shape[0] == 0:
                    tprint_error("❌ Features array is empty")
                    return False
                
                # Check for NaN values
                nan_count = np.isnan(X).sum()
                if nan_count > 0:
                    tprint_warning(f"⚠️ Features contain {nan_count} NaN values")
                
                tprint_debug(f"✅ Features validated: shape {X.shape}")
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Training data validation failed: {e}")
            return False
    
    def _load_cluster_assignments_from_file(self) -> Optional[np.ndarray]:
        """Load cluster assignments from HMM training input file with error handling."""
        try:
            # Find the latest HMM training input file
            hmm_input_pattern = "optimal_clusters/binance/ETHUSDT/15m/market_analysis_hmm_training_input_ETHUSDT_BINANCE_15m_*.pkl"
            hmm_input_files = glob.glob(hmm_input_pattern)

            if not hmm_input_files:
                tprint_warning(f"⚠️ No HMM training input files found matching pattern: {hmm_input_pattern}")
                return None

            # Get the most recent file
            latest_file = max(hmm_input_files, key=lambda x: x.split('_')[-1].replace('.pkl', ''))
            tprint_info(f"🔍 Loading cluster assignments from: {latest_file}")

            # Load data using safe serialization
            hmm_input_data = _serializer.load(latest_file)
            if hmm_input_data is None:
                tprint_error(f"❌ Failed to load data from {latest_file}")
                return None

            if 'cluster_assignments' in hmm_input_data:
                cluster_assignments = hmm_input_data['cluster_assignments']
                tprint_success(f"✅ Loaded {len(cluster_assignments)} cluster assignments")
                tprint_debug(f"📊 Cluster assignments shape: {cluster_assignments.shape}, Unique clusters: {len(set(cluster_assignments))}")
                return cluster_assignments
            else:
                tprint_error("❌ No cluster_assignments found in HMM training input file")
                return None

        except Exception as e:
            tprint_error(f"❌ Error loading cluster assignments from file: {e}")
            return None
    
    def _extract_features_from_dataframe(self, dataframe: pd.DataFrame) -> tuple:
        """Extract features and targets from DataFrame with comprehensive feature engineering."""
        try:
            if not validate_dataframe(dataframe):
                tprint_error("❌ Invalid DataFrame provided")
                return None, None, None
            
            if 'close' not in dataframe.columns:
                tprint_error("❌ DataFrame must contain 'close' column")
                return None, None, None
            
            tprint_info("🔧 Extracting features from DataFrame")
            
            # Create lagged features to avoid data leakage
            raw_returns = dataframe['close'].pct_change().fillna(0)

            # Features: lagged returns (past information only)
            returns_lag1 = raw_returns.shift(1).fillna(0)
            returns_lag2 = raw_returns.shift(2).fillna(0)
            returns_lag5 = raw_returns.shift(5).fillna(0)

            # Volatility features (also lagged)
            volatility = raw_returns.rolling(20).std().fillna(0).shift(1).fillna(0)

            # Volume features
            min_periods_30d = min(len(dataframe), 96)
            volume_30d_avg = dataframe['volume'].rolling(window=2880, min_periods=min_periods_30d).mean()
            volume_ratio_30d = (dataframe['volume'] / volume_30d_avg.replace(0, dataframe['volume'].mean())).fillna(1) if 'volume' in dataframe.columns else pd.Series([1] * len(dataframe), index=dataframe.index)

            # Technical indicators
            sma_20 = dataframe['close'].rolling(20).mean().shift(1).fillna(dataframe['close'].iloc[0])
            sma_50 = dataframe['close'].rolling(50).mean().shift(1).fillna(dataframe['close'].iloc[0])
            price_position = (dataframe['close'] - sma_20) / sma_20.shift(1).fillna(1)

            # EMA indicators
            ema_12 = dataframe['close'].ewm(span=12).mean().shift(1).fillna(dataframe['close'].iloc[0])
            ema_26 = dataframe['close'].ewm(span=26).mean().shift(1).fillna(dataframe['close'].iloc[0])

            # RSI-like indicator
            price_changes = raw_returns
            gains = np.where(price_changes > 0, price_changes, 0)
            losses = np.where(price_changes < 0, -price_changes, 0)
            avg_gain = pd.Series(gains).rolling(14).mean().fillna(0).shift(1).fillna(0)
            avg_loss = pd.Series(losses).rolling(14).mean().fillna(0).shift(1).fillna(0)
            rs = avg_gain / avg_loss.replace(0, 1e-8)
            rsi = 100 - (100 / (1 + rs))

            # Bollinger Bands
            bb_middle = sma_20
            bb_std = raw_returns.rolling(20).std().shift(1).fillna(0)
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            bb_position = (dataframe['close'] - bb_middle) / (bb_upper - bb_lower).replace(0, 1)

            # VWAP
            typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
            vwap = (typical_price * dataframe['volume']).rolling(20).sum() / dataframe['volume'].rolling(20).sum()
            vwap_position = (dataframe['close'] - vwap.shift(1).fillna(dataframe['close'].iloc[0])) / dataframe['close'].shift(1).fillna(dataframe['close'].iloc[0])

            # Momentum indicators
            momentum_5 = dataframe['close'] / dataframe['close'].shift(5).fillna(1) - 1
            momentum_10 = dataframe['close'] / dataframe['close'].shift(10).fillna(1) - 1

            # Volatility ratios
            vol_short = raw_returns.rolling(5).std().shift(1).fillna(0)
            vol_long = raw_returns.rolling(20).std().shift(1).fillna(0)
            vol_ratio = vol_short / vol_long.replace(0, 1)

            # Combine features
            X = np.column_stack([
                returns_lag1.values, returns_lag2.values, returns_lag5.values,
                volatility.values, volume_ratio_30d.values, sma_20.values, sma_50.values,
                price_position.values, ema_12.values, ema_26.values, rsi.values,
                bb_position.values, vwap_position.values, momentum_5.values,
                momentum_10.values, vol_ratio.values
            ])
            
            feature_names = [
                'returns_lag1', 'returns_lag2', 'returns_lag5', 'volatility', 'volume_ratio_30d',
                'sma_20', 'sma_50', 'price_position', 'ema_12', 'ema_26', 'rsi',
                'bb_position', 'vwap_position', 'momentum_5', 'momentum_10', 'vol_ratio'
            ]

            # Create targets from future returns
            future_returns = raw_returns.shift(-1).fillna(0)
            y_continuous = future_returns.values
            y = np.zeros_like(y_continuous, dtype=int)
            y[y_continuous < -0.02] = 0  # Strong Down
            y[(y_continuous >= -0.02) & (y_continuous < -0.005)] = 1  # Down
            y[(y_continuous >= -0.005) & (y_continuous <= 0.005)] = 2  # Sideways
            y[(y_continuous > 0.005) & (y_continuous <= 0.02)] = 3  # Up
            y[y_continuous > 0.02] = 4  # Strong Up
            
            # Remove first row where returns is NaN
            X = X[1:]
            y = y[1:]
            
            tprint_success(f"✅ Extracted features: {X.shape}, targets: {y.shape}")
            return X, y, feature_names
            
        except Exception as e:
            tprint_error(f"❌ Feature extraction failed: {e}")
            return None, None, None
    
    async def execute(self, data, pipeline_state: Dict[str, Any]) -> 'ComponentResult':
        """Execute HMM models training with comprehensive error handling and optimization."""
        start_time = time.time()
        self._execution_count += 1
        
        tprint_info(f"🚀 Starting HMM Models Training execution #{self._execution_count}")
        
        try:
            # Create training instance
            training_instance = self._create_training_instance()
            
            # Extract required data from pipeline state
            X = pipeline_state.get('features')
            y = pipeline_state.get('targets')
            cluster_assignments = pipeline_state.get('cluster_assignments')
            feature_names = pipeline_state.get('feature_names')
            market_data = pipeline_state.get('market_data') or data
            
            # Try to get cluster assignments from hmm_clusters
            if cluster_assignments is None:
                hmm_clusters = pipeline_state.get('hmm_clusters', {})
                cluster_assignments = hmm_clusters.get('cluster_assignments')
                if cluster_assignments is not None:
                    tprint_success(f"✅ Found cluster_assignments in hmm_clusters: {len(cluster_assignments)} samples")
            
            # Load cluster assignments from file if still missing
            if cluster_assignments is None:
                cluster_assignments = self._load_cluster_assignments_from_file()
                if cluster_assignments is None:
                    tprint_warning("⚠️ No cluster assignments available, will extract from DataFrame")
            
            # Extract features from DataFrame if missing
            if X is None or y is None:
                dataframe = pipeline_state.get('dataframe')
                if dataframe is not None:
                    X, y, feature_names = self._extract_features_from_dataframe(dataframe)
                    if X is None:
                        tprint_error("❌ Failed to extract features from DataFrame")
                        return ComponentResult(
                            success=False,
                            artifacts={},
                            metadata={'execution_count': self._execution_count},
                            error_message="Failed to extract features from DataFrame"
                        )
            
            # Validate training data
            if not self._validate_training_data(X, y, cluster_assignments):
                return ComponentResult(
                    success=False,
                    artifacts={},
                    metadata={'execution_count': self._execution_count},
                    error_message="Training data validation failed"
                )
            
            # Adjust cluster assignments length if necessary
            if cluster_assignments is not None and X is not None and len(cluster_assignments) > len(X):
                cluster_assignments = cluster_assignments[:len(X)]
                tprint_info(f"✅ Adjusted cluster assignments length to match features: {len(cluster_assignments)}")
            
            # Use HMM state recognition as the training objective
            if cluster_assignments is not None:
                y = cluster_assignments
                tprint_info("✅ Using cluster assignments as training targets")
            
            # Execute training with M1 optimizations
            tprint_info("🎯 Executing HMM models training")
            
            if _m1_memory_optimizer:
                with _m1_memory_optimizer.memory_checkpoint("hmm_training"):
                    results = training_instance.execute(X, y, cluster_assignments, feature_names, market_data=market_data)
            else:
                results = training_instance.execute(X, y, cluster_assignments, feature_names, market_data=market_data)
            
            # Create comprehensive artifact
            execution_time = time.time() - start_time
            self._last_execution_time = execution_time
            
            artifact = {
                'hmm_models_training_result': {
                    'hmm_models': results.get('model_results', {}),
                    'hmm_training_metrics': results.get('comprehensive_report', {}),
                    'metadata': results.get('metadata', {}),
                    'training_time': results.get('training_time', execution_time),
                    'success': 'error' not in results,
                    'execution_count': self._execution_count,
                    'data_shape': X.shape if X is not None else None,
                    'feature_names': feature_names
                }
            }
            
            tprint_success(f"✅ HMM models training completed in {execution_time:.2f}s")
            
            return ComponentResult(
                success=True,
                artifacts=artifact,
                metadata={
                    'component_type': 'hmm_models_training',
                    'execution_time': execution_time,
                    'execution_count': self._execution_count
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"HMM models training failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            
            return ComponentResult(
                success=False,
                artifacts={},
                metadata={
                    'component_type': 'hmm_models_training',
                    'execution_time': execution_time,
                    'execution_count': self._execution_count,
                    'error_type': type(e).__name__
                },
                error_message=error_msg
            )


class HMMEnsembleTrainingComponentWrapper(BaseMarketAnalysisComponent):
    """Wrapper for HMM Ensemble Training Component to work as a component."""
    
    def __init__(self, training_class, config: Optional[ComponentConfig] = None):
        super().__init__(config)
        self.training_class = training_class
        self.training_instance = None
    
    def _convert_to_numpy_array(self, data):
        """Convert list data to numpy array if needed."""
        if data is not None:
            if isinstance(data, list):
                return np.array(data)
        return data
    
    def get_required_artifacts(self) -> list[str]:
        """Get list of required artifacts this component must produce."""
        return ['hmm_ensemble_training_result']
    
    async def execute(self, data, pipeline_state: Dict[str, Any]) -> 'ComponentResult':
        """Execute HMM ensemble training as a component."""
        try:
            # Create training instance if not exists
            if self.training_instance is None:
                self.training_instance = self.training_class()
                try:
                    # Enforce 15m timeframe for HMM ensemble at runtime
                    if hasattr(self.training_instance, 'config'):
                        setattr(self.training_instance.config, 'timeframe', '15m')
                        if getattr(self.training_instance.config, 'timeframe', None) != '15m':
                            print("⚠️ HMM Ensemble: Non-15m timeframe supplied; overriding to 15m for consistency")
                except Exception:
                    pass
            
            # Extract required data from pipeline state
            X = pipeline_state.get('features')
            y = pipeline_state.get('targets')
            cluster_assignments = pipeline_state.get('cluster_assignments')
            feature_names = pipeline_state.get('feature_names')
            hmm_states = pipeline_state.get('hmm_states')
            base_hmm_models = pipeline_state.get('hmm_models', {}).get('hmm_models', {})
            hmm_training_metrics = pipeline_state.get('hmm_models', {}).get('hmm_training_metrics', {})
            
            # If cluster_assignments is missing, try to get from hmm_clusters
            if cluster_assignments is None:
                hmm_clusters = pipeline_state.get('hmm_clusters', {})
                cluster_assignments = hmm_clusters.get('cluster_assignments')
                if cluster_assignments is not None:
                    print(f"✅ Found cluster_assignments in hmm_clusters: {len(cluster_assignments)} samples")
            
            # Load cluster assignments directly from HMM training input file
            if cluster_assignments is None:
                try:

                    # Find the latest HMM training input file
                    hmm_input_pattern = "optimal_clusters/binance/ETHUSDT/15m/market_analysis_hmm_training_input_ETHUSDT_BINANCE_15m_*.pkl"
                    hmm_input_files = glob.glob(hmm_input_pattern)

                    if hmm_input_files:
                        # Get the most recent file
                        latest_file = max(hmm_input_files, key=lambda x: x.split('_')[-1].replace('.pkl', ''))
                        print(f"🔍 Loading cluster assignments from latest HMM training input file: {latest_file}")

                        with open(latest_file, 'rb') as f:
                            hmm_input_data = pickle.load(f)

                        if 'cluster_assignments' in hmm_input_data:
                            cluster_assignments = hmm_input_data['cluster_assignments']
                            print(f"✅ Loaded {len(cluster_assignments)} cluster assignments from HMM training input file")
                            print(f"📊 Cluster assignments shape: {cluster_assignments.shape}, Unique clusters: {len(set(cluster_assignments))}")
                        else:
                            print(f"❌ No cluster_assignments found in HMM training input file")
                            raise ValueError("No cluster_assignments found in HMM training input file")
                    else:
                        print(f"❌ No HMM training input files found matching pattern: {hmm_input_pattern}")
                        raise ValueError("No HMM training input files found")

                except Exception as e:
                    print(f"❌ Error loading cluster assignments from HMM training input file: {e}")
                    raise ValueError(f"Failed to load cluster assignments: {e}")
            
            # If we don't have features/targets, try to extract from dataframe
            if X is None or y is None:
                dataframe = pipeline_state.get('dataframe')
                if dataframe is not None:
                    
                    # Create basic features and targets from OHLCV data
                    if 'close' in dataframe.columns:
                        # Create lagged features to avoid data leakage
                        # Shift returns by 1 period to ensure features are from past, target is from future
                        raw_returns = dataframe['close'].pct_change().fillna(0)

                        # Features: lagged returns (past information only)
                        returns_lag1 = raw_returns.shift(1).fillna(0)  # 1-period lagged returns
                        returns_lag2 = raw_returns.shift(2).fillna(0)  # 2-period lagged returns
                        returns_lag5 = raw_returns.shift(5).fillna(0)  # 5-period lagged returns

                        # Volatility features (also lagged)
                        volatility = raw_returns.rolling(20).std().fillna(0).shift(1).fillna(0)

                        # Volume features
                        min_periods_30d = min(len(dataframe), 96)  # At least 1 day of data
                        volume_30d_avg = dataframe['volume'].rolling(window=2880, min_periods=min_periods_30d).mean()
                        volume_ratio_30d = (dataframe['volume'] / volume_30d_avg.replace(0, dataframe['volume'].mean())).fillna(1) if 'volume' in dataframe.columns else pd.Series([1] * len(dataframe), index=dataframe.index)

                        # Additional technical features - more diverse indicators
                        sma_20 = dataframe['close'].rolling(20).mean().shift(1).fillna(dataframe['close'].iloc[0])
                        sma_50 = dataframe['close'].rolling(50).mean().shift(1).fillna(dataframe['close'].iloc[0])
                        price_position = (dataframe['close'] - sma_20) / sma_20.shift(1).fillna(1)

                        # More technical indicators
                        ema_12 = dataframe['close'].ewm(span=12).mean().shift(1).fillna(dataframe['close'].iloc[0])
                        ema_26 = dataframe['close'].ewm(span=26).mean().shift(1).fillna(dataframe['close'].iloc[0])

                        # RSI-like indicator
                        price_changes = raw_returns
                        gains = np.where(price_changes > 0, price_changes, 0)
                        losses = np.where(price_changes < 0, -price_changes, 0)
                        avg_gain = pd.Series(gains).rolling(14).mean().fillna(0).shift(1).fillna(0)
                        avg_loss = pd.Series(losses).rolling(14).mean().fillna(0).shift(1).fillna(0)
                        rs = avg_gain / avg_loss.replace(0, 1e-8)
                        rsi = 100 - (100 / (1 + rs))

                        # Bollinger Bands position
                        bb_middle = sma_20
                        bb_std = raw_returns.rolling(20).std().shift(1).fillna(0)
                        bb_upper = bb_middle + (bb_std * 2)
                        bb_lower = bb_middle - (bb_std * 2)
                        bb_position = (dataframe['close'] - bb_middle) / (bb_upper - bb_lower).replace(0, 1)

                        # Volume-weighted average price (VWAP) components
                        typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
                        vwap = (typical_price * dataframe['volume']).rolling(20).sum() / dataframe['volume'].rolling(20).sum()
                        vwap_position = (dataframe['close'] - vwap.shift(1).fillna(dataframe['close'].iloc[0])) / dataframe['close'].shift(1).fillna(dataframe['close'].iloc[0])

                        # Price momentum indicators
                        momentum_5 = dataframe['close'] / dataframe['close'].shift(5).fillna(1) - 1
                        momentum_10 = dataframe['close'] / dataframe['close'].shift(10).fillna(1) - 1

                        # Volatility ratios
                        vol_short = raw_returns.rolling(5).std().shift(1).fillna(0)
                        vol_long = raw_returns.rolling(20).std().shift(1).fillna(0)
                        vol_ratio = vol_short / vol_long.replace(0, 1)

                        X = np.column_stack([
                            returns_lag1.values,    # Lagged returns (past info)
                            returns_lag2.values,    # Lagged returns (past info)
                            returns_lag5.values,    # Lagged returns (past info)
                            volatility.values,      # Historical volatility
                            volume_ratio_30d.values, # Volume ratio
                            sma_20.values,          # Moving average
                            sma_50.values,          # Moving average
                            price_position.values,  # Price position
                            ema_12.values,          # Exponential moving average
                            ema_26.values,          # Exponential moving average
                            rsi.values,             # RSI indicator
                            bb_position.values,     # Bollinger Bands position
                            vwap_position.values,   # VWAP position
                            momentum_5.values,      # Short-term momentum
                            momentum_10.values,     # Medium-term momentum
                            vol_ratio.values        # Volatility ratio
                        ])
                        feature_names = [
                            'returns_lag1', 'returns_lag2', 'returns_lag5',
                            'volatility', 'volume_ratio_30d', 'sma_20', 'sma_50', 'price_position',
                            'ema_12', 'ema_26', 'rsi', 'bb_position', 'vwap_position',
                            'momentum_5', 'momentum_10', 'vol_ratio'
                        ]

                        # Create targets from future returns (not current returns) to avoid data leakage
                        future_returns = raw_returns.shift(-1).fillna(0)  # Next period returns
                        
                        # Convert continuous future returns to discrete classes for predictive modeling
                        # Class 0: Strong Down (< -2%), Class 1: Down (-2% to -0.5%),
                        # Class 2: Sideways (-0.5% to 0.5%), Class 3: Up (0.5% to 2%), Class 4: Strong Up (> 2%)
                        y_continuous = future_returns.values
                        y = np.zeros_like(y_continuous, dtype=int)
                        y[y_continuous < -0.02] = 0  # Strong Down
                        y[(y_continuous >= -0.02) & (y_continuous < -0.005)] = 1  # Down
                        y[(y_continuous >= -0.005) & (y_continuous <= 0.005)] = 2  # Sideways
                        y[(y_continuous > 0.005) & (y_continuous <= 0.02)] = 3  # Up
                        y[y_continuous > 0.02] = 4  # Strong Up
                        
                        # Remove first row where returns is NaN (due to pct_change)
                        X = X[1:]
                        y = y[1:]
                        
                        # Adjust cluster_assignments length if necessary
                        if cluster_assignments is not None and len(cluster_assignments) > len(X):
                            cluster_assignments = cluster_assignments[:len(X)]
            
            if X is None or y is None or cluster_assignments is None:
                missing_items = []
                if X is None: missing_items.append("features")
                if y is None: missing_items.append("targets")
                if cluster_assignments is None: missing_items.append("cluster_assignments")
                raise ValueError(f"Missing required data: {', '.join(missing_items)}")
            
            # Ensure all data is in proper numpy format before training
            cluster_assignments = self._convert_to_numpy_array(cluster_assignments)
            
            # Use HMM state recognition as the training objective
            y = cluster_assignments

            # Execute training
            results = self.training_instance.execute(
                X, y, cluster_assignments, feature_names, hmm_states, 
                base_hmm_models, hmm_training_metrics
            )
            
            # Create comprehensive artifact
            artifact = {
                'hmm_ensemble_training_result': {
                    'hmm_ensemble': results.get('models', {}),
                    'hmm_ensemble_metrics': results.get('comprehensive_report', {}),
                    'ensemble_metrics': results.get('ensemble_metrics', {}),
                    'performance_summary': results.get('performance_summary', {}),
                    'metadata': results.get('metadata', {}),
                    'training_time': results.get('training_time', 0),
                    'success': 'error' not in results
                }
            }
            
            return ComponentResult(
                success=True,
                artifacts=artifact,
                metadata={'component_type': 'hmm_ensemble_training', 'execution_time': results.get('training_time', 0)}
            )
            
        except Exception as e:
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e),
                metadata={'component_type': 'hmm_ensemble_training'}
            )


class ComponentFactory:
    """
    Enhanced factory for creating market analysis pipeline components.
    
    Provides centralized component creation and management with comprehensive
    error handling, logging, and utility integration.
    """
    
    _components: Dict[str, Type[BaseMarketAnalysisComponent]] = {
        'sr_parameter_optimization': SRParameterOptimizationComponent,
        'sr_detection': SRDetectionComponent,
        'sr_clustering': SRClusteringComponent,
        'nas_regime_discovery': NASRegimeDiscoveryComponent,
        'tas_regime_discovery': TASRegimeDiscoveryComponent,
        'hybrid_nas_tas_regime_discovery': HybridNASTASRegimeDiscoveryComponent,
        'nas_tas_regime_discovery': HybridNASTASRegimeDiscoveryComponent,
        'nas_tas_clustering': NASTASClusteringComponent,
        'nas_clustering': NASTASClusteringComponent,  # Backward-compatible alias
        'feature_lookback_optimization': FeatureLookbackOptimizationComponent,
        'cross_timeframe_analysis': CrossTimeframeAnalysisComponent,
        'pid_based_feature_generation': PIDBasedFeatureGenerationComponent if PID_COMPONENT_AVAILABLE else CrossTimeframeAnalysisComponent,
        'final_feature_selection': FinalFeatureSelectionComponent
    }
    
    _lazy_components: List[str] = [
        'regime_data_splitting', 'multi_horizon_profit_labeler'
    ]
    
    _deprecated_components: List[str] = [
        'hmm_models_training', 'hmm_ensemble_training'
    ]
    
    @classmethod
    def _validate_component_name(cls, component_name: str) -> bool:
        """Validate component name with comprehensive checks."""
        try:
            if not isinstance(component_name, str):
                tprint_error(f"❌ Component name must be a string, got {type(component_name)}")
                return False
            
            if not component_name.strip():
                tprint_error("❌ Component name cannot be empty")
                return False
            
            # Check for deprecated components
            if component_name in cls._deprecated_components:
                tprint_warning(f"⚠️ Component '{component_name}' is deprecated")
                return False
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Component name validation failed: {e}")
            return False
    
    @classmethod
    def _validate_config(cls, config: Optional[ComponentConfig]) -> bool:
        """Validate component configuration."""
        try:
            if config is not None and not isinstance(config, ComponentConfig):
                tprint_warning(f"⚠️ Config must be ComponentConfig instance, got {type(config)}")
                return False
            
            return True
            
        except Exception as e:
            tprint_warning(f"⚠️ Config validation failed: {e}")
            return False
    
    @classmethod
    def _handle_lazy_component(cls, component_name: str, config: Optional[ComponentConfig]) -> BaseMarketAnalysisComponent:
        """Handle lazy-loaded components with error handling."""
        try:
            if component_name == 'regime_data_splitting':
                tprint_info("🔧 Loading RegimeDataSplittingComponent")
                from .regime_data_splitting import RegimeDataSplittingComponent
                component = RegimeDataSplittingComponent(config)
                tprint_success("✅ Created RegimeDataSplittingComponent")
                return component
            
            elif component_name == 'multi_horizon_profit_labeler':
                tprint_info("🔧 Loading MultiHorizonSubPipelineAdapter")
                from ..multi_horizon_sub_pipeline_adapter import MultiHorizonSubPipelineAdapter
                component = MultiHorizonComponentWrapper(MultiHorizonSubPipelineAdapter, config)
                tprint_success("✅ Created MultiHorizonComponentWrapper")
                return component
            
            else:
                raise ValueError(f"Unknown lazy component: {component_name}")
                
        except ImportError as e:
            tprint_error(f"❌ Failed to import {component_name}: {e}")
            raise ValueError(f"Failed to import {component_name}: {e}")
        except Exception as e:
            tprint_error(f"❌ Failed to create {component_name}: {e}")
            raise
    
    @classmethod
    def _handle_deprecated_component(cls, component_name: str) -> None:
        """Handle deprecated components with informative messages."""
        if component_name == 'hmm_models_training':
            tprint_warning("⚠️ HMM models training is deprecated and no longer available")
            raise ValueError("HMM models training is deprecated and no longer available")
        
        elif component_name == 'hmm_ensemble_training':
            tprint_warning("⚠️ HMM ensemble training is deprecated and no longer available")
            raise ValueError("HMM ensemble training is deprecated and no longer available")
    
    @classmethod
    def _create_component_with_optimization(cls, component_class: Type[BaseMarketAnalysisComponent], 
                                           config: Optional[ComponentConfig]) -> BaseMarketAnalysisComponent:
        """Create component with M1 optimizations and error handling."""
        try:
            # Create component with memory checkpoint if available
            if _m1_memory_optimizer:
                with _m1_memory_optimizer.memory_checkpoint(f"component_creation_{component_class.__name__}"):
                    component = component_class(config)
            else:
                component = component_class(config)
            
            # Apply M1 optimizations if available
            if _m1_cpu_optimizer and hasattr(component, 'execute'):
                try:
                    component.execute = _m1_cpu_optimizer.optimize_function_for_m1(component.execute)
                    tprint_debug(f"✅ Applied M1 CPU optimization to {component_class.__name__}")
                except Exception as e:
                    tprint_warning(f"⚠️ M1 CPU optimization failed for {component_class.__name__}: {e}")
            
            return component
            
        except Exception as e:
            tprint_error(f"❌ Failed to create component {component_class.__name__}: {e}")
            raise
    
    @classmethod
    def create_component(
        cls, 
        component_name: str, 
        config: Optional[ComponentConfig] = None
    ) -> BaseMarketAnalysisComponent:
        """
        Create a component instance with comprehensive error handling and optimization.
        
        Args:
            component_name: Name of the component to create
            config: Component configuration
            
        Returns:
            Component instance
            
        Raises:
            ValueError: If component name is not registered or invalid
            ImportError: If required dependencies are missing
        """
        start_time = time.time()
        
        try:
            tprint_info(f"🏭 Creating component: {component_name}")
            
            # Validate inputs
            if not cls._validate_component_name(component_name):
                raise ValueError(f"Invalid component name: {component_name}")
            
            if not cls._validate_config(config):
                tprint_warning("⚠️ Using default config due to validation failure")
                config = None
            
            # Handle deprecated components
            if component_name in cls._deprecated_components:
                cls._handle_deprecated_component(component_name)
            
            # Handle lazy-loaded components
            if component_name in cls._lazy_components:
                return cls._handle_lazy_component(component_name, config)
            
            # Handle registered components
            if component_name in cls._components:
                component_class = cls._components[component_name]
                tprint_debug(f"🔧 Creating {component_name} from registered components")
                
                component = cls._create_component_with_optimization(component_class, config)
                
                execution_time = time.time() - start_time
                tprint_success(f"✅ Successfully created {component_name} in {execution_time:.3f}s")
                return component
            
            # Component not found
            available_components = list(cls._components.keys()) + cls._lazy_components
            tprint_error(f"❌ Unknown component: {component_name}")
            tprint_info(f"📊 Available components: {available_components}")
            
            raise ValueError(
                f"Unknown component: {component_name}. "
                f"Available components: {available_components}"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ Component creation failed after {execution_time:.3f}s: {e}")
            raise
    
    @classmethod
    def register_component(
        cls, 
        name: str, 
        component_class: Type[BaseMarketAnalysisComponent]
    ) -> None:
        """
        Register a new component with validation.
        
        Args:
            name: Component name
            component_class: Component class
            
        Raises:
            ValueError: If component class is invalid
        """
        try:
            # Validate component class
            if not issubclass(component_class, BaseMarketAnalysisComponent):
                raise ValueError(
                    f"Component class must inherit from BaseMarketAnalysisComponent, "
                    f"got {component_class.__name__}"
                )
            
            # Validate name
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Component name must be a non-empty string")
            
            # Register component
            cls._components[name] = component_class
            tprint_success(f"✅ Registered component: {name} ({component_class.__name__})")
            
        except Exception as e:
            tprint_error(f"❌ Failed to register component {name}: {e}")
            raise
    
    @classmethod
    def unregister_component(cls, name: str) -> bool:
        """
        Unregister a component.
        
        Args:
            name: Component name to unregister
            
        Returns:
            True if component was unregistered, False if not found
        """
        try:
            if name in cls._components:
                del cls._components[name]
                tprint_success(f"✅ Unregistered component: {name}")
                return True
            else:
                tprint_warning(f"⚠️ Component not found for unregistration: {name}")
                return False
                
        except Exception as e:
            tprint_error(f"❌ Failed to unregister component {name}: {e}")
            return False
    
    @classmethod
    def get_available_components(cls) -> List[str]:
        """
        Get list of available component names.
        
        Returns:
            List of component names
        """
        try:
            all_components = list(cls._components.keys()) + cls._lazy_components
            tprint_debug(f"📊 Available components: {len(all_components)}")
            return all_components
            
        except Exception as e:
            tprint_error(f"❌ Failed to get available components: {e}")
            return []
    
    @classmethod
    def is_component_available(cls, component_name: str) -> bool:
        """
        Check if a component is available.
        
        Args:
            component_name: Name of the component
            
        Returns:
            True if component is available
        """
        try:
            is_available = (component_name in cls._components or 
                          component_name in cls._lazy_components)
            
            if is_available:
                tprint_debug(f"✅ Component available: {component_name}")
            else:
                tprint_debug(f"❌ Component not available: {component_name}")
            
            return is_available
            
        except Exception as e:
            tprint_error(f"❌ Failed to check component availability: {e}")
            return False
    
    @classmethod
    def get_component_info(cls, component_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Dictionary with component information
        """
        try:
            info = {
                'name': component_name,
                'available': cls.is_component_available(component_name),
                'registered': component_name in cls._components,
                'lazy_loaded': component_name in cls._lazy_components,
                'deprecated': component_name in cls._deprecated_components
            }
            
            if component_name in cls._components:
                component_class = cls._components[component_name]
                info.update({
                    'class_name': component_class.__name__,
                    'module': component_class.__module__,
                    'docstring': component_class.__doc__
                })
            
            return info
            
        except Exception as e:
            tprint_error(f"❌ Failed to get component info for {component_name}: {e}")
            return {'name': component_name, 'error': str(e)}
    
    @classmethod
    def validate_all_components(cls) -> Dict[str, Any]:
        """
        Validate all registered components.
        
        Returns:
            Dictionary with validation results
        """
        try:
            validation_results = {
                'total_components': len(cls._components),
                'valid_components': 0,
                'invalid_components': 0,
                'errors': []
            }
            
            for name, component_class in cls._components.items():
                try:
                    # Test component creation
                    test_component = component_class()
                    validation_results['valid_components'] += 1
                    tprint_debug(f"✅ Component valid: {name}")
                    
                except Exception as e:
                    validation_results['invalid_components'] += 1
                    validation_results['errors'].append(f"{name}: {e}")
                    tprint_warning(f"⚠️ Component invalid: {name} - {e}")
            
            tprint_info(f"📊 Component validation: {validation_results['valid_components']} valid, "
                       f"{validation_results['invalid_components']} invalid")
            
            return validation_results
            
        except Exception as e:
            tprint_error(f"❌ Component validation failed: {e}")
            return {'error': str(e)}


# Enhanced utility functions for component factory
def create_optimized_component(component_name: str, config: Optional[ComponentConfig] = None) -> BaseMarketAnalysisComponent:
    """
    Create an optimized component with M1 hardware optimizations.
    
    Args:
        component_name: Name of the component to create
        config: Component configuration
        
    Returns:
        Optimized component instance
    """
    try:
        tprint_info(f"🚀 Creating optimized component: {component_name}")
        
        # Create component with factory
        component = ComponentFactory.create_component(component_name, config)
        
        # Apply additional optimizations
        if _m1_memory_optimizer and hasattr(component, 'execute'):
            try:
                # Wrap execute method with memory optimization
                original_execute = component.execute
                
                async def optimized_execute(data, pipeline_state):
                    with _m1_memory_optimizer.memory_checkpoint(f"component_execution_{component_name}"):
                        return await original_execute(data, pipeline_state)
                
                component.execute = optimized_execute
                tprint_debug(f"✅ Applied memory optimization to {component_name}")
                
            except Exception as e:
                tprint_warning(f"⚠️ Memory optimization failed for {component_name}: {e}")
        
        # Apply GPU optimizations if available
        if _m1_gpu_manager and is_m1_available() and hasattr(component, 'execute'):
            try:
                # Wrap execute method with GPU optimization
                original_execute = component.execute
                
                async def gpu_optimized_execute(data, pipeline_state):
                    with gpu_context(f"component_execution_{component_name}"):
                        return await original_execute(data, pipeline_state)
                
                component.execute = gpu_optimized_execute
                tprint_debug(f"✅ Applied GPU optimization to {component_name}")
                
            except Exception as e:
                tprint_warning(f"⚠️ GPU optimization failed for {component_name}: {e}")
        
        tprint_success(f"✅ Optimized component created: {component_name}")
        return component
        
    except Exception as e:
        tprint_error(f"❌ Failed to create optimized component {component_name}: {e}")
        raise


def validate_component_pipeline(components: List[str]) -> Dict[str, Any]:
    """
    Validate a component pipeline for compatibility and dependencies.
    
    Args:
        components: List of component names in pipeline order
        
    Returns:
        Dictionary with validation results
    """
    try:
        tprint_info(f"🔍 Validating component pipeline: {components}")
        
        validation_results = {
            'total_components': len(components),
            'valid_components': 0,
            'invalid_components': 0,
            'missing_components': [],
            'deprecated_components': [],
            'warnings': [],
            'errors': []
        }
        
        for i, component_name in enumerate(components):
            try:
                # Check if component is available
                if not ComponentFactory.is_component_available(component_name):
                    validation_results['missing_components'].append(component_name)
                    validation_results['invalid_components'] += 1
                    continue
                
                # Check if component is deprecated
                if component_name in ComponentFactory._deprecated_components:
                    validation_results['deprecated_components'].append(component_name)
                    validation_results['warnings'].append(f"Component {component_name} is deprecated")
                
                # Try to create component
                try:
                    test_component = ComponentFactory.create_component(component_name)
                    validation_results['valid_components'] += 1
                    tprint_debug(f"✅ Component {i+1}/{len(components)}: {component_name}")
                    
                except Exception as e:
                    validation_results['invalid_components'] += 1
                    validation_results['errors'].append(f"{component_name}: {e}")
                    tprint_warning(f"⚠️ Component {i+1}/{len(components)}: {component_name} - {e}")
                
            except Exception as e:
                validation_results['invalid_components'] += 1
                validation_results['errors'].append(f"{component_name}: {e}")
                tprint_error(f"❌ Component {i+1}/{len(components)}: {component_name} - {e}")
        
        # Summary
        tprint_info(f"📊 Pipeline validation: {validation_results['valid_components']} valid, "
                   f"{validation_results['invalid_components']} invalid")
        
        if validation_results['missing_components']:
            tprint_warning(f"⚠️ Missing components: {validation_results['missing_components']}")
        
        if validation_results['deprecated_components']:
            tprint_warning(f"⚠️ Deprecated components: {validation_results['deprecated_components']}")
        
        return validation_results
        
    except Exception as e:
        tprint_error(f"❌ Pipeline validation failed: {e}")
        return {'error': str(e)}


def create_component_pipeline(components: List[str], configs: Optional[List[ComponentConfig]] = None) -> List[BaseMarketAnalysisComponent]:
    """
    Create a complete component pipeline with optimizations.
    
    Args:
        components: List of component names in pipeline order
        configs: Optional list of configurations for each component
        
    Returns:
        List of optimized component instances
    """
    try:
        tprint_info(f"🏗️ Creating component pipeline: {components}")
        
        # Validate pipeline first
        validation_results = validate_component_pipeline(components)
        if validation_results.get('invalid_components', 0) > 0:
            tprint_error("❌ Pipeline validation failed, cannot create components")
            raise ValueError("Pipeline validation failed")
        
        # Create components
        pipeline_components = []
        configs = configs or [None] * len(components)
        
        for i, (component_name, config) in enumerate(zip(components, configs)):
            try:
                tprint_progress(i + 1, len(components), f"Creating {component_name}")
                component = create_optimized_component(component_name, config)
                pipeline_components.append(component)
                
            except Exception as e:
                tprint_error(f"❌ Failed to create component {i+1}/{len(components)}: {component_name} - {e}")
                raise
        
        tprint_success(f"✅ Created component pipeline with {len(pipeline_components)} components")
        return pipeline_components
        
    except Exception as e:
        tprint_error(f"❌ Failed to create component pipeline: {e}")
        raise


def get_component_statistics() -> Dict[str, Any]:
    """
    Get comprehensive statistics about available components.
    
    Returns:
        Dictionary with component statistics
    """
    try:
        tprint_info("📊 Gathering component statistics")
        
        stats = {
            'total_registered': len(ComponentFactory._components),
            'total_lazy': len(ComponentFactory._lazy_components),
            'total_deprecated': len(ComponentFactory._deprecated_components),
            'total_available': len(ComponentFactory.get_available_components()),
            'component_categories': {
                'registered': list(ComponentFactory._components.keys()),
                'lazy_loaded': ComponentFactory._lazy_components,
                'deprecated': ComponentFactory._deprecated_components
            },
            'm1_optimizations': {
                'gpu_available': is_m1_available() and is_mps_available(),
                'memory_optimizer_available': _m1_memory_optimizer is not None,
                'cpu_optimizer_available': _m1_cpu_optimizer is not None
            },
            'utility_integration': {
                'math_validator_available': _math_validator is not None,
                'serializer_available': _serializer is not None,
                'bayesian_tpe_available': True  # We imported it
            }
        }
        
        tprint_success(f"✅ Component statistics: {stats['total_available']} available components")
        return stats
        
    except Exception as e:
        tprint_error(f"❌ Failed to get component statistics: {e}")
        return {'error': str(e)}


# Enhanced initialization function
def initialize_component_factory() -> Dict[str, Any]:
    """
    Initialize the component factory with all optimizations and utilities.
    
    Returns:
        Dictionary with initialization results
    """
    try:
        tprint_info("🚀 Initializing enhanced component factory")
        
        initialization_results = {
            'factory_initialized': True,
            'utility_managers': {
                'm1_gpu_manager': _m1_gpu_manager is not None,
                'm1_memory_optimizer': _m1_memory_optimizer is not None,
                'm1_cpu_optimizer': _m1_cpu_optimizer is not None,
                'math_validator': _math_validator is not None,
                'serializer': _serializer is not None
            },
            'component_counts': {
                'registered': len(ComponentFactory._components),
                'lazy_loaded': len(ComponentFactory._lazy_components),
                'deprecated': len(ComponentFactory._deprecated_components)
            },
            'optimizations_available': {
                'm1_hardware': is_m1_available(),
                'mps_gpu': is_mps_available(),
                'bayesian_tpe': True,
                'memory_optimization': _m1_memory_optimizer is not None
            }
        }
        
        # Start memory monitoring if available
        if _m1_memory_optimizer:
            try:
                start_m1_memory_monitoring()
                tprint_success("✅ M1 memory monitoring started")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to start memory monitoring: {e}")
        
        tprint_success("✅ Enhanced component factory initialized successfully")
        return initialization_results
        
    except Exception as e:
        tprint_error(f"❌ Component factory initialization failed: {e}")
        return {'error': str(e), 'factory_initialized': False}


# Auto-initialize on import
_initialization_results = initialize_component_factory()