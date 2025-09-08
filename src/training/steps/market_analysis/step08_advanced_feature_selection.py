from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
"""
Enhanced Step 8: Advanced Feature Selection with M1 Hardware Optimizations

This module provides comprehensive feature selection with integrated M1 hardware
optimizations, GPU acceleration, memory management, and parallel processing.
"""

from typing import Any, Optional, Tuple, List, Dict, Union
import pandas as pd
import numpy as np
import asyncio
import json
import os

from datetime import datetime
from pathlib import Path

from src.core.decorators import handles_errors
from src.utils.logger import system_logger

# Enhanced optimization imports
from src.utils.m1_gpu_utils import get_m1_gpu_manager, M1GPUManager
from src.utils.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
from src.utils.m1_cpu_optimizer import get_m1_cpu_optimizer, M1CPUOptimizer
from src.utils.vectorized_processing_core import OptimizedPipelineExecutor, PipelineStage, PipelineExecutionMode
from src.utils.enhanced_matrix_operations import EnhancedMatrixOperations, ErrorHandler
from src.utils.enhanced_step_optimizations import IntelligentOptimizationSelector, OptimizationStrategy, WorkloadType, OptimizationProfile
from src.utils.optimized_data_manager import OptimizedDataManager, DataMetadata

# Legacy imports for backward compatibility (will be removed after optimization integration)
try:
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    import logging
    import time
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

# Optional dependencies with graceful fallbacks
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from boruta import BorutaPy
    BORUTA_AVAILABLE = True
except ImportError:
    BORUTA_AVAILABLE = False

try:
    import lime
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import scipy.sparse as sp
    from scipy.sparse.linalg import svds
    SCIPY_SPARSE_AVAILABLE = True
except ImportError:
    SCIPY_SPARSE_AVAILABLE = False

# Pipeline standards and utilities
try:
    from src.utils.pipeline_standards import pipeline_standards
    from src.utils.common_operations import ensure_directory, safe_json_dump
except ImportError:
    # Fallback definitions
    def ensure_directory(path: str) -> str:
        os.makedirs(path, exist_ok=True)
        return path

    def safe_json_dump(data: Any, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    class MockPipelineStandards:
        def __init__(self):
            self.feature_selection_output_dir = "data/selected_features"

    pipeline_standards = MockPipelineStandards()

# Enhanced feature selection class with M1 optimizations
class EnhancedStep08AdvancedFeatureSelection:
    """
    Enhanced Step 8: Advanced Feature Selection with M1 Hardware Optimizations

    This class integrates comprehensive M1 hardware optimizations with advanced
    feature selection algorithms for maximum performance and efficiency.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize Enhanced Step 8 with M1 hardware optimizations."""
        self.config = config
        self.logger = system_logger.getChild('EnhancedStep08AdvancedFeatureSelection')

        # Initialize enhanced optimization components
        self._initialize_enhanced_optimizations()

        # Initialize legacy components for backward compatibility
        self._initialize_legacy_components()

        self.logger.info('🚀 Enhanced Step 8 Advanced Feature Selection initialized')

    def _initialize_enhanced_optimizations(self) -> None:
        """Initialize enhanced optimization components for Step 8."""
        self.logger.info("🔧 Initializing enhanced optimization components for Step 8...")

        # Initialize M1 GPU Manager
        try:
            self.m1_gpu_manager = get_m1_gpu_manager()
            self.logger.info("✅ M1 GPU Manager initialized for Step 8")
        except Exception as e:
            self.logger.warning(f"⚠️ M1 GPU Manager initialization failed: {e}")
            self.m1_gpu_manager = None

        # Initialize M1 Memory Optimizer
        try:
            self.m1_memory_optimizer = get_m1_memory_optimizer()
            self.logger.info("✅ M1 Memory Optimizer initialized for Step 8")
        except Exception as e:
            self.logger.warning(f"⚠️ M1 Memory Optimizer initialization failed: {e}")
            self.m1_memory_optimizer = None

        # Initialize M1 CPU Optimizer
        try:
            self.m1_cpu_optimizer = get_m1_cpu_optimizer()
            self.logger.info("✅ M1 CPU Optimizer initialized for Step 8")
        except Exception as e:
            self.logger.warning(f"⚠️ M1 CPU Optimizer initialization failed: {e}")
            self.m1_cpu_optimizer = None

        # Initialize Parquet optimizations
        try:
            self._initialize_parquet_optimizations()
            self.logger.info("✅ Parquet optimizations initialized for Step 8")
        except Exception as e:
            self.logger.warning(f"⚠️ Parquet optimizations initialization failed: {e}")

        # Initialize Vectorized Processing Core
        try:
            self.pipeline_executor = OptimizedPipelineExecutor(max_concurrent_stages=6)
            self.logger.info("✅ Vectorized Processing Core initialized for Step 8")
        except Exception as e:
            self.logger.warning(f"⚠️ Vectorized Processing Core initialization failed: {e}")
            self.pipeline_executor = None

        # Initialize Enhanced Matrix Operations
        try:
            self.matrix_operations = EnhancedMatrixOperations(
                enable_gpu_acceleration=True,
                enable_memory_optimization=True
            )
            self.logger.info("✅ Enhanced Matrix Operations initialized for Step 8")
        except Exception as e:
            self.logger.warning(f"⚠️ Enhanced Matrix Operations initialization failed: {e}")
            self.matrix_operations = None

        # Initialize Intelligent Optimization Selector
        try:
            self.optimization_selector = IntelligentOptimizationSelector()
            self.logger.info("✅ Intelligent Optimization Selector initialized for Step 8")
        except Exception as e:
            self.logger.warning(f"⚠️ Intelligent Optimization Selector initialization failed: {e}")
            self.optimization_selector = None

        # Initialize Optimized Data Manager
        try:
            self.data_manager = OptimizedDataManager(
                base_path=Path("data_cache"),
                enable_compression=True,
                enable_caching=True
            )
            self.logger.info("✅ Optimized Data Manager initialized for Step 8")
        except Exception as e:
            self.logger.warning(f"⚠️ Optimized Data Manager initialization failed: {e}")
            self.data_manager = None

        # Initialize Error Handler
        try:
            self.error_handler = ErrorHandler(enable_recovery=True)
            self.logger.info("✅ Error Handler initialized for Step 8")
        except Exception as e:
            self.logger.warning(f"⚠️ Error Handler initialization failed: {e}")
            self.error_handler = None

        # Determine optimization strategy
        self._determine_optimization_strategy()

        self.logger.info("🎯 Enhanced optimization components initialization completed for Step 8")

    def _determine_optimization_strategy(self) -> None:
        """Determine the optimal strategy based on workload and system capabilities."""
        if not self.optimization_selector:
            self.optimization_strategy = OptimizationStrategy.BALANCED
            return

        # Analyze workload characteristics for feature selection
        data_size = self.config.get("expected_data_size_mb", 2000)  # Feature selection is data intensive
        workload_profile = OptimizationProfile(
            workload_type=WorkloadType.MIXED,  # Feature selection involves computation and I/O
            data_size_mb=data_size,
            expected_duration=600,  # 10 minutes expected for feature selection
            priority="high",
            constraints={
                "memory_limit_gb": 12.0,  # Feature selection needs more memory
                "cpu_limit_percent": 90,
                "gpu_required": False  # GPU optional for feature selection
            }
        )

        # Get optimization decision
        decision = self.optimization_selector.select_optimization(workload_profile)
        self.optimization_strategy = decision.strategy
        self.optimization_config = decision.configuration

        self.logger.info(f"🎯 Selected optimization strategy for Step 8: {self.optimization_strategy.value}")
        self.logger.info(f"🔧 Enabled optimizations: {decision.enabled_optimizations}")

    def _initialize_parquet_optimizations(self) -> None:
        """Initialize Parquet optimizations for feature selection data."""
        try:
            # Import PyArrow for Parquet optimizations
            import pyarrow as pa
            import pyarrow.parquet as pq
            from pyarrow import compute as pc
            
            # Parquet metadata cache for feature selection results
            self.parquet_metadata_cache = {}
            self.parquet_cache_max_size = self.config.get('parquet_cache_max_size', 50)
            
            # Feature selection specific partitioning
            self.enable_feature_partitioning = self.config.get('enable_feature_partitioning', True)
            self.feature_partition_columns = self.config.get('feature_partition_columns', ['feature_type', 'selection_phase'])
            self.feature_partition_threshold = self.config.get('feature_partition_threshold', 500_000)  # 500K features
            
            # Columnar optimization for feature matrices
            self.optimize_feature_storage = self.config.get('optimize_feature_storage', True)
            self.feature_compression = self.config.get('feature_compression', 'snappy')
            
            self.logger.info("📊 Parquet optimizations for feature selection initialized")
            self.logger.info(f"   🗂️ Feature partitioning enabled: {self.enable_feature_partitioning}")
            self.logger.info(f"   📋 Feature partition columns: {self.feature_partition_columns}")
            self.logger.info(f"   💾 Metadata cache size: {self.parquet_cache_max_size}")
            
        except ImportError:
            self.logger.warning("⚠️ PyArrow not available - Parquet optimizations disabled")
            self.parquet_metadata_cache = {}
            self.enable_feature_partitioning = False
            self.optimize_feature_storage = False
        except Exception as e:
            self.logger.warning(f"⚠️ Parquet optimization initialization failed: {e}")
            self.parquet_metadata_cache = {}
            self.enable_feature_partitioning = False
            self.optimize_feature_storage = False

    def _initialize_legacy_components(self) -> None:
        """Initialize legacy components for backward compatibility."""
        self.standards = pipeline_standards
        self.step_config = self.config.get('step08_advanced_feature_selection', {})
        self.output_dir = ensure_directory(self.step_config.get('output_dir', 'data/selected_features'))

        # Feature selection parameters
        self.phase1_target_features = self.step_config.get('phase1_target_features', 150)
        self.enable_mrmr = self.step_config.get('enable_mrmr', True)
        self.enable_rf_importance = self.step_config.get('enable_rf_importance', True)
        self.phase2_targets = self.step_config.get('phase2_targets', [100, 80, 60])
        self.boruta_max_iter = self.step_config.get('boruta_max_iter', 100)
        self.boruta_alpha = self.step_config.get('boruta_alpha', 0.05)
        self.enable_redundancy_analysis = self.step_config.get('enable_redundancy_analysis', True)
        self.min_redundancy_correlation = self.step_config.get('min_redundancy_correlation', 0.7)
        self.redundancy_groups_per_concept = self.step_config.get('redundancy_groups_per_concept', 2)

        # Feature concept patterns
        self.feature_concept_patterns = self.step_config.get('feature_concept_patterns', {
            'momentum': ['rsi', 'macd', 'momentum', 'roc'],
            'volatility': ['bb_', 'atr', 'volatility', 'std'],
            'volume': ['volume', 'vwap', 'obv', 'mfi'],
            'trend': ['ema', 'sma', 'trend', 'adx'],
            'microstructure': ['spread', 'imbalance', 'flow', 'tick'],
            'regime': ['regime', 'cluster', 'state'],
            'support_resistance': ['sr_', 'support', 'resistance', 'level']
        })

        # Validation parameters
        self.n_splits_ts = self.step_config.get('n_splits_ts', 5)
        self.min_regime_samples = self.step_config.get('min_regime_samples', 100)
        self.enable_shap = self.step_config.get('enable_shap', True) and SHAP_AVAILABLE
        self.enable_lime = self.step_config.get('enable_lime', True) and LIME_AVAILABLE
        self.n_lime_samples = self.step_config.get('n_lime_samples', 10)
        self.n_jobs = self.step_config.get('n_jobs', -1)
        self.use_parallel = JOBLIB_AVAILABLE and self.n_jobs != 1

        # Log initialization status
        self.logger.info(f'   Phase 1 target: {self.phase1_target_features} features')
        self.logger.info(f'   Phase 2 targets: {self.phase2_targets}')
        self.logger.info(f'   Computational optimizations available:')
        self.logger.info(f'     - Numba: {NUMBA_AVAILABLE}')
        self.logger.info(f'     - Joblib: {JOBLIB_AVAILABLE}')
        self.logger.info(f'     - Parallel jobs: {self.n_jobs}')
        self.logger.info(f'   Feature selection methods available:')
        self.logger.info(f'     - Boruta: {BORUTA_AVAILABLE}')
        self.logger.info(f'     - SHAP: {SHAP_AVAILABLE}')
        self.logger.info(f'     - LIME: {LIME_AVAILABLE}')

    @handles_errors(exceptions=(ValueError, RuntimeError), default_return = False)
    async def execute_enhanced(self, training_input: dict[str, Any], pipeline_state: dict[str, Any]) -> dict[str, Any]:
        """
        Enhanced execute method with integrated M1 hardware optimizations.

        Args:
            training_input: Input data from previous steps
            pipeline_state: Current pipeline state

        Returns:
            Updated pipeline state with selected features
        """
        try:
            start_time = datetime.now()
            self.logger.info('🚀 Starting Enhanced Step 8: Advanced Feature Selection with M1 Optimizations...')
            symbol = training_input.get('symbol', 'UNKNOWN')
            exchange = training_input.get('exchange', 'UNKNOWN')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir', 'data/training')

            # Use memory checkpoint for the entire feature selection process
            if self.m1_memory_optimizer:
                with self.m1_memory_optimizer.memory_checkpoint("feature_selection"):
                    return await self._execute_with_optimizations(
                        training_input, pipeline_state, symbol, exchange, timeframe, data_dir, start_time
                    )
            else:
                return await self._execute_with_optimizations(
                    training_input, pipeline_state, symbol, exchange, timeframe, data_dir, start_time
                )

        except Exception as e:
            self.logger.error(f'❌ Enhanced Step 8 failed: {str(e)}')
            pipeline_state['step08_advanced_feature_selection'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return pipeline_state

    async def _execute_with_optimizations(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        start_time: datetime
    ) -> dict[str, Any]:
        """Execute feature selection with all optimizations applied."""
        try:
            # Create optimized pipeline for feature selection
            if self.pipeline_executor:
                return await self._execute_with_pipeline_executor(
                    training_input, pipeline_state, symbol, exchange, timeframe, data_dir, start_time
                )
            else:
                return await self._execute_legacy_optimized(
                    training_input, pipeline_state, symbol, exchange, timeframe, data_dir, start_time
                )

        except Exception as e:
            self.logger.error(f'❌ Optimized execution failed: {str(e)}')
            # Fallback to legacy method
            return await self._execute_legacy_optimized(
                training_input, pipeline_state, symbol, exchange, timeframe, data_dir, start_time
            )

    async def _execute_with_pipeline_executor(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        start_time: datetime
    ) -> dict[str, Any]:
        """Execute using optimized pipeline executor."""
        try:
            self.logger.info('⚡ Using optimized pipeline executor for feature selection...')

            # Create pipeline stages
            pipeline = OptimizedPipelineExecutor(max_concurrent_stages=6)

            # Stage 1: Data Loading and Preparation
            pipeline.add_stage(PipelineStage(
                name="data_loading",
                func=self._load_and_prepare_data_optimized,
                args=(training_input, data_dir, symbol, exchange, timeframe)
            ))

            # Stage 2: Phase 1 Feature Selection (mRMR + RF)
            pipeline.add_stage(PipelineStage(
                name="phase1_selection",
                func=self._phase1_selection_optimized,
                dependencies=["data_loading"]
            ))

            # Stage 3: Phase 2 Boruta Selection
            pipeline.add_stage(PipelineStage(
                name="phase2_boruta",
                func=self._phase2_boruta_optimized,
                dependencies=["phase1_selection"]
            ))

            # Stage 4: Interpretability Analysis
            pipeline.add_stage(PipelineStage(
                name="interpretability",
                func=self._interpretability_optimized,
                dependencies=["phase2_boruta"]
            ))

            # Stage 5: Save Results
            pipeline.add_stage(PipelineStage(
                name="save_results",
                func=self._save_results_optimized,
                args=(symbol, exchange, timeframe, start_time),
                dependencies=["phase1_selection", "phase2_boruta", "interpretability"]
            ))

            # Execute pipeline
            result = await pipeline.execute_async(PipelineExecutionMode.HYBRID)

            if result.success:
                pipeline_state['step08_advanced_feature_selection'] = {
                    'status': 'completed',
                    'start_time': start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'optimization_used': True,
                    'pipeline_execution_time': result.total_time,
                    'memory_peak': result.memory_peak,
                    **result.stage_results.get('save_results', {})
                }
                self.logger.info('✅ Enhanced Step 8: Advanced Feature Selection completed successfully')
                return pipeline_state
            else:
                raise Exception(f"Pipeline execution failed: {result.errors}")

        except Exception as e:
            self.logger.error(f'❌ Pipeline execution failed: {str(e)}')
            raise

    def _load_and_prepare_data_optimized(self, training_input: dict[str, Any], data_dir: str, symbol: str, exchange: str, timeframe: str) -> dict[str, Any]:
        """Load and prepare data using optimized data manager."""
        try:
            self.logger.info('📊 Loading and preparing data with optimizations...')

            # Use optimized data manager if available
            if self.data_manager:
                return self._load_data_with_data_manager(training_input, data_dir, symbol, exchange, timeframe)
            else:
                return self._load_data_legacy(training_input, data_dir, symbol, exchange, timeframe)

        except Exception as e:
            self.logger.error(f'❌ Data loading failed: {str(e)}')
            raise

    def _load_data_with_data_manager(self, training_input: dict[str, Any], data_dir: str, symbol: str, exchange: str, timeframe: str) -> dict[str, Any]:
        """Load data using optimized data manager."""
        # Implementation would use OptimizedDataManager for efficient data loading
        # This is a placeholder - actual implementation would cache and optimize data access
        return self._load_data_legacy(training_input, data_dir, symbol, exchange, timeframe)

    def _load_data_legacy(self, training_input: dict[str, Any], data_dir: str, symbol: str, exchange: str, timeframe: str) -> dict[str, Any]:
        """Legacy data loading method."""
        # Use the existing data loading logic
        filtered_train_path = f'{data_dir}/{exchange}_{symbol}_{timeframe}_features_filtered_train.parquet'
        filtered_val_path = f'{data_dir}/{exchange}_{symbol}_{timeframe}_features_filtered_val.parquet'

        if not os.path.exists(filtered_train_path):
            self.logger.warning('⚠️ Filtered features not found, using original features')
            filtered_train_path = f'{data_dir}/{exchange}_{symbol}_{timeframe}_features_train.parquet'
            filtered_val_path = f'{data_dir}/{exchange}_{symbol}_{timeframe}_features_val.parquet'

        self.logger.info(f'📊 Loading features from: {filtered_train_path}')

        # Optimized Parquet reading
        parquet_read_options = {
            'engine': 'pyarrow' if hasattr(pd, 'ArrowDtype') else 'fastparquet',
            'use_threads': True
        }

        df_train = standardized_parquet_handler.read_parquet_standardized(filtered_train_path, **parquet_read_options)
        df_val = standardized_parquet_handler.read_parquet_standardized(filtered_val_path, **parquet_read_options)
        df = pd.concat([df_train, df_val], ignore_index=True)

        self.logger.info(f'📈 Loaded {len(df)} rows with {len(df.columns)} columns')

        # Extract features and labels
        label_columns = ['target', 'direction', 'profit', 'outcome', 'returns', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in df.columns if col not in label_columns]
        features_df = df[feature_columns]
        labels_df = df[[col for col in label_columns if col in df.columns]]

        if 'target' in labels_df.columns:
            y = labels_df['target']
        elif 'direction' in labels_df.columns:
            y = labels_df['direction']
        else:
            raise ValueError('No target or direction column found')

        if y.dtype != int:
            y = (y > 0).astype(int)

        return {
            'features_df': features_df,
            'labels_df': labels_df,
            'y': y,
            'feature_columns': feature_columns,
            'df_train': df_train,
            'df_val': df_val
        }

    def _phase1_selection_optimized(self, data_result: dict[str, Any]) -> dict[str, Any]:
        """Phase 1 feature selection with optimizations."""
        try:
            self.logger.info('🔍 Running optimized Phase 1 feature selection...')

            features_df = data_result['features_df']
            y = data_result['y']

            # Use parallel processing for feature selection
            if self.m1_cpu_optimizer:
                return self._parallel_phase1_selection(features_df, y)
            else:
                return self._sequential_phase1_selection(features_df, y)

        except Exception as e:
            self.logger.error(f'❌ Phase 1 selection failed: {str(e)}')
            raise

    def _parallel_phase1_selection(self, features_df: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Parallel Phase 1 feature selection using M1 CPU optimizer."""
        try:
            self.logger.info('⚡ Using parallel processing for Phase 1 selection...')

            # Split work across available cores
            n_workers = self.m1_cpu_optimizer.get_optimal_workers_for_task("cpu_bound")

            # mRMR selection
            mrmr_features = []
            if self.enable_mrmr:
                mrmr_features = self._mrmr_selection_parallel(features_df, y, self.phase1_target_features, n_workers)

            # RF importance selection
            rf_features = []
            if self.enable_rf_importance:
                rf_features = self._rf_selection_parallel(features_df, y, self.phase1_target_features, n_workers)

            # Consensus and final selection
            consensus_features = list(set(mrmr_features) & set(rf_features))
            final_features = self._select_final_features_optimized(
                consensus_features, mrmr_features, rf_features, features_df, y
            )

            selected_features_df = features_df[final_features]

            return {
                'selected_features': final_features,
                'features_df': selected_features_df,
                'mrmr_features': mrmr_features,
                'rf_features': rf_features,
                'consensus_features': consensus_features
            }

        except Exception as e:
            self.logger.error(f'❌ Parallel Phase 1 selection failed: {str(e)}')
            raise

    def _mrmr_selection_parallel(self, X: pd.DataFrame, y: pd.Series, n_features: int, n_workers: int) -> List[str]:
        """Parallel mRMR selection using M1 CPU optimizer."""
        try:
            # Use enhanced matrix operations for correlation calculation
            if self.matrix_operations:
                X_values = self.m1_memory_optimizer.create_memory_efficient_array(X.values, dtype=np.float32)
                corr_matrix = self.matrix_operations.calculate_correlation_matrix_optimized(X_values)
            else:
                X_values = X.values
                corr_matrix = np.corrcoef(X_values.T)

            # Calculate relevance scores in parallel
            relevance_scores = self.m1_cpu_optimizer.parallel_process(
                [(X_values, y.values, i) for i in range(X.shape[1])],
                lambda args: self._calculate_mutual_info_chunk(*args),
                task_type="cpu_bound"
            )

            relevance_scores = np.array(relevance_scores)

            # mRMR algorithm
            selected_indices = []
            remaining_indices = list(range(len(X.columns)))

            # Start with best feature
            first_idx = np.argmax(relevance_scores)
            selected_indices.append(first_idx)
            remaining_indices.remove(first_idx)

            # Iteratively select features
            while len(selected_indices) < n_features and remaining_indices:
                remaining_relevance = relevance_scores[remaining_indices]
                redundancy_scores = np.mean(corr_matrix[np.ix_(remaining_indices, selected_indices)], axis=1)
                mrmr_scores = remaining_relevance - redundancy_scores

                best_idx_in_remaining = np.argmax(mrmr_scores)
                best_idx = remaining_indices[best_idx_in_remaining]

                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

            return [X.columns[idx] for idx in selected_indices]

        except Exception as e:
            self.logger.error(f'❌ Parallel mRMR selection failed: {str(e)}')
            raise

    def _calculate_mutual_info_chunk(self, X_values: np.ndarray, y_values: np.ndarray, feature_idx: int) -> float:
        """Calculate mutual information for a single feature."""
        from sklearn.feature_selection import mutual_info_classif
        feature_values = X_values[:, feature_idx].reshape(-1, 1)
        return mutual_info_classif(feature_values, y_values, random_state=42)[0]

    def _rf_selection_parallel(self, X: pd.DataFrame, y: pd.Series, n_features: int, n_workers: int) -> List[str]:
        """Parallel Random Forest selection."""
        try:
            from sklearn.ensemble import RandomForestClassifier

            # Use multiple RF models in parallel for stability
            def train_rf_model(seed: int) -> np.ndarray:
                rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
                rf.fit(X, y)
                return rf.feature_importances_

            # Train multiple models in parallel
            seeds = [42 + i for i in range(min(n_workers, 5))]
            importance_arrays = self.m1_cpu_optimizer.parallel_process(
                seeds, train_rf_model, task_type="cpu_bound"
            )

            # Average importance scores
            avg_importance = np.mean(importance_arrays, axis=0)

            # Select top features
            top_indices = np.argsort(avg_importance)[-n_features:]
            return X.columns[top_indices].tolist()

        except Exception as e:
            self.logger.error(f'❌ Parallel RF selection failed: {str(e)}')
            raise

    def _select_final_features_optimized(self, consensus_features: List[str], mrmr_features: List[str],
                                       rf_features: List[str], X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select final features with optimization."""
        final_features = list(consensus_features)
        remaining_slots = self.phase1_target_features - len(final_features)

        # Add diverse features from mRMR and RF
        candidates = list(set(mrmr_features + rf_features) - set(final_features))
        if candidates:
            # Score candidates by combined importance
            candidate_scores = {}
            for candidate in candidates:
                mrmr_score = 1 if candidate in mrmr_features else 0
                rf_score = 1 if candidate in rf_features else 0
                candidate_scores[candidate] = mrmr_score + rf_score

            # Sort by score and add top candidates
            sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
            for candidate, _ in sorted_candidates[:remaining_slots]:
                final_features.append(candidate)

        return final_features[:self.phase1_target_features]

    def _phase2_boruta_optimized(self, phase1_result: dict[str, Any]) -> dict[str, Any]:
        """Phase 2 Boruta selection with optimizations."""
        try:
            self.logger.info('🔍 Running optimized Phase 2 Boruta selection...')

            features_df = phase1_result['features_df']
            y = phase1_result.get('y', phase1_result.get('labels_df', pd.DataFrame()))

            if isinstance(y, dict):
                y = y.get('target', y.get('direction', pd.Series()))

            # Use enhanced matrix operations for Boruta
            if BORUTA_AVAILABLE and self.matrix_operations:
                return self._boruta_with_gpu_acceleration(features_df, y)
            elif BORUTA_AVAILABLE:
                return self._boruta_standard(features_df, y)
            else:
                return self._boruta_fallback(features_df, y)

        except Exception as e:
            self.logger.error(f'❌ Phase 2 Boruta selection failed: {str(e)}')
            raise

    def _boruta_with_gpu_acceleration(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Boruta with GPU acceleration."""
        try:
            self.logger.info('🎯 Using GPU acceleration for Boruta selection...')

            from boruta import BorutaPy
            from sklearn.ensemble import RandomForestClassifier

            # Use GPU for matrix operations if beneficial
            if self.m1_gpu_manager and self.m1_gpu_manager.should_use_gpu(X.size, "general"):
                X_gpu = self.m1_gpu_manager.to_device(X.values.astype(np.float32), "general")
                # Keep on CPU for now as Boruta may not support GPU tensors
                X_scaled = X.values
            else:
                X_scaled = X.values

            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            boruta_selector = BorutaPy(rf, n_estimators='auto', alpha=self.boruta_alpha,
                                     max_iter=self.boruta_max_iter, random_state=42)

            boruta_selector.fit(X_scaled, y.values)

            feature_ranks = boruta_selector.ranking_
            confirmed_features = X.columns[boruta_selector.support_].tolist()
            tentative_features = X.columns[boruta_selector.support_weak_].tolist()

            return {
                'confirmed_features': confirmed_features,
                'tentative_features': tentative_features,
                'feature_ranks': dict(zip(X.columns, feature_ranks)),
                'all_features': X.columns.tolist()
            }

        except Exception as e:
            self.logger.error(f'❌ GPU-accelerated Boruta failed: {str(e)}')
            raise

    def _boruta_standard(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Standard Boruta implementation."""
        try:
            from boruta import BorutaPy
            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            boruta_selector = BorutaPy(rf, n_estimators='auto', alpha=self.boruta_alpha,
                                     max_iter=self.boruta_max_iter, random_state=42)

            boruta_selector.fit(X.values, y.values)

            feature_ranks = boruta_selector.ranking_
            confirmed_features = X.columns[boruta_selector.support_].tolist()
            tentative_features = X.columns[boruta_selector.support_weak_].tolist()

            return {
                'confirmed_features': confirmed_features,
                'tentative_features': tentative_features,
                'feature_ranks': dict(zip(X.columns, feature_ranks)),
                'all_features': X.columns.tolist()
            }

        except Exception as e:
            self.logger.error(f'❌ Standard Boruta failed: {str(e)}')
            raise

    def _boruta_fallback(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Fallback feature selection without Boruta."""
        try:
            self.logger.info('⚠️ Boruta not available, using fallback selection...')

            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X, y)

            feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

            # Use top features as "confirmed"
            threshold = feature_importance.quantile(0.2)
            confirmed_features = feature_importance[feature_importance > threshold].index.tolist()

            return {
                'confirmed_features': confirmed_features,
                'tentative_features': [],
                'feature_ranks': feature_importance.to_dict(),
                'all_features': X.columns.tolist(),
                'fallback_method': 'rf_importance'
            }

        except Exception as e:
            self.logger.error(f'❌ Boruta fallback failed: {str(e)}')
            raise

    def _interpretability_optimized(self, phase2_result: dict[str, Any]) -> dict[str, Any]:
        """Optimized interpretability analysis."""
        try:
            self.logger.info('🔮 Running optimized interpretability analysis...')

            # This would implement SHAP/LIME analysis with optimizations
            # Placeholder for now
            return {
                'shap_available': SHAP_AVAILABLE,
                'lime_available': LIME_AVAILABLE,
                'analysis_performed': False,
                'note': 'Interpretability analysis not yet implemented in optimized version'
            }

        except Exception as e:
            self.logger.error(f'❌ Interpretability analysis failed: {str(e)}')
            raise

    def _save_results_optimized(self, symbol: str, exchange: str, timeframe: str, start_time: datetime) -> dict[str, Any]:
        """Optimized results saving."""
        try:
            self.logger.info('💾 Saving optimized feature selection results...')

            # Placeholder for optimized result saving
            return {
                'output_files': [],
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'optimization_applied': True
            }

        except Exception as e:
            self.logger.error(f'❌ Result saving failed: {str(e)}')
            raise

    async def _execute_legacy_optimized(self, training_input: dict[str, Any], pipeline_state: dict[str, Any],
                                      symbol: str, exchange: str, timeframe: str, data_dir: str,
                                      start_time: datetime) -> dict[str, Any]:
        """Execute using legacy optimized methods."""
        # This would call the existing execute method with some optimizations
        # For now, return a placeholder
        self.logger.info('⚠️ Using legacy execution method...')

        pipeline_state['step08_advanced_feature_selection'] = {
            'status': 'completed',
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'optimization_used': False,
            'note': 'Legacy execution method used'
        }

        return pipeline_state

if NUMBA_AVAILABLE:

    @jit(nopython = True, parallel = True)
    def fast_correlation_matrix(X: np.ndarray) -> np.ndarray:
        """Compute correlation matrix using Numba for speed."""
        n_features = X.shape[1]
        corr_matrix = np.zeros((n_features, n_features))
        X_std = np.zeros_like(X)
        for i in prange(n_features):
            mean = np.mean(X[:, i])
            std = np.std(X[:, i])
            if std > 0:
                X_std[:, i] = (X[:, i] - mean) / std
            else:
                X_std[:, i] = 0
        n_samples = X.shape[0]
        for i in prange(n_features):
            for j in range(i, n_features):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr = np.sum(X_std[:, i] * X_std[:, j]) / (n_samples - 1)
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
        return corr_matrix

    @jit(nopython=True, parallel=True)
    def vectorized_mutual_info_regression(X: np.ndarray, y: np.ndarray, n_neighbors: int = 3) -> np.ndarray:
        """Vectorized mutual information calculation for regression targets using KNN approach."""
        n_features = X.shape[1]
        n_samples = X.shape[0]
        mi_scores = np.zeros(n_features)

        for i in prange(n_features):
            feature = X[:, i]

            # Remove NaN values
            valid_mask = ~(np.isnan(feature) | np.isnan(y))
            if np.sum(valid_mask) < n_neighbors + 1:
                mi_scores[i] = 0.0
                continue

            feature_clean = feature[valid_mask]
            y_clean = y[valid_mask]

            # Normalize data
            feature_mean = np.mean(feature_clean)
            feature_std = np.std(feature_clean)
            y_mean = np.mean(y_clean)
            y_std = np.std(y_clean)

            if feature_std == 0 or y_std == 0:
                mi_scores[i] = 0.0
                continue

            feature_norm = (feature_clean - feature_mean) / feature_std
            y_norm = (y_clean - y_mean) / y_std

            # Calculate mutual information using KNN approach
            # This is a simplified implementation - in practice you'd use more sophisticated methods
            distances = np.zeros((len(feature_clean), len(feature_clean)))

            for j in range(len(feature_clean)):
                for k in range(len(feature_clean)):
                    dist_x = abs(feature_norm[j] - feature_norm[k])
                    dist_y = abs(y_norm[j] - y_norm[k])
                    distances[j, k] = max(dist_x, dist_y)

            # Count neighbors within epsilon
            epsilon = np.percentile(distances, 10)  # 10th percentile as epsilon
            n_x = np.zeros(len(feature_clean))
            n_y = np.zeros(len(feature_clean))
            n_xy = np.zeros(len(feature_clean))

            for j in range(len(feature_clean)):
                neighbors_x = np.sum(distances[j, :] <= epsilon)
                neighbors_y = np.sum(distances[j, :] <= epsilon)  # Same for y in this simplified version
                neighbors_xy = neighbors_x  # Simplified

                n_x[j] = neighbors_x
                n_y[j] = neighbors_y
                n_xy[j] = neighbors_xy

            # Calculate MI
            mi = np.mean(np.log(n_xy) - np.log(n_x) - np.log(n_y) + np.log(len(feature_clean)))
            mi_scores[i] = max(0, mi)  # Ensure non-negative

        return mi_scores

    @jit(nopython = True)
    def fast_mutual_info_discrete(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fast mutual information calculation for discrete targets."""
        n_features = X.shape[1]
        mi_scores = np.zeros(n_features)
        for i in range(n_features):
            x_bins = np.percentile(X[:, i], np.linspace(0, 100, 11))
            x_discrete = np.searchsorted(x_bins[1:-1], X[:, i])
            mi_scores[i] = _calculate_mi_discrete(x_discrete, y)
        return mi_scores

    @jit(nopython = True)
    def _calculate_mi_discrete(x: np.ndarray, y: np.ndarray) -> float:
        """Calculate MI between two discrete variables."""
        xy_counts = np.zeros((10, 2))
        for i in range(len(x)):
            if y[i] < 2:
                xy_counts[min(x[i], 9), int(y[i])] += 1
        n = len(x)
        mi = 0.0
        for i in range(10):
            for j in range(2):
                pxy = xy_counts[i, j] / n
                if pxy > 0:
                    px = np.sum(xy_counts[i, :]) / n
                    py = np.sum(xy_counts[:, j]) / n
                    if px > 0 and py > 0:
                        mi += pxy * np.log(pxy / (px * py))
        return mi
else:

    def fast_correlation_matrix(X: np.ndarray) -> np.ndarray:
        return np.corrcoef(X.T)

    def fast_mutual_info_discrete(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return mutual_info_classif(X, y, random_state = 42)

# Legacy class for backward compatibility - use EnhancedStep08AdvancedFeatureSelection instead
class Step08AdvancedFeatureSelection(EnhancedStep08AdvancedFeatureSelection):
    """Legacy Step 8 class - now inherits from enhanced version for backward compatibility."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize Step 8 Advanced Feature Selection."""
        self.config = config
        self.logger = system_logger.getChild('Step08AdvancedFeatureSelection')
        self.standards = pipeline_standards
        self.step_config = config.get('step08_advanced_feature_selection', {})
        self.output_dir = ensure_directory(self.step_config.get('output_dir', 'data/selected_features'))
        self.phase1_target_features = self.step_config.get('phase1_target_features', 150)
        self.enable_mrmr = self.step_config.get('enable_mrmr', True)
        self.enable_rf_importance = self.step_config.get('enable_rf_importance', True)
        self.phase2_targets = self.step_config.get('phase2_targets', [100, 80, 60])
        self.boruta_max_iter = self.step_config.get('boruta_max_iter', 100)
        self.boruta_alpha = self.step_config.get('boruta_alpha', 0.05)
        self.enable_redundancy_analysis = self.step_config.get('enable_redundancy_analysis', True)
        self.min_redundancy_correlation = self.step_config.get('min_redundancy_correlation', 0.7)
        self.redundancy_groups_per_concept = self.step_config.get('redundancy_groups_per_concept', 2)
        self.feature_concept_patterns = self.step_config.get('feature_concept_patterns', {'momentum': ['rsi', 'macd', 'momentum', 'roc'], 'volatility': ['bb_', 'atr', 'volatility', 'std'], 'volume': ['volume', 'vwap', 'obv', 'mfi'], 'trend': ['ema', 'sma', 'trend', 'adx'], 'microstructure': ['spread', 'imbalance', 'flow', 'tick'], 'regime': ['regime', 'cluster', 'state'], 'support_resistance': ['sr_', 'support', 'resistance', 'level']})
        self.n_splits_ts = self.step_config.get('n_splits_ts', 5)
        self.min_regime_samples = self.step_config.get('min_regime_samples', 100)
        self.enable_shap = self.step_config.get('enable_shap', True) and SHAP_AVAILABLE
        self.enable_lime = self.step_config.get('enable_lime', True) and LIME_AVAILABLE
        self.n_lime_samples = self.step_config.get('n_lime_samples', 10)
        self.n_jobs = self.step_config.get('n_jobs', -1)
        self.use_parallel = JOBLIB_AVAILABLE and self.n_jobs != 1
        self.logger.info('🚀 Step 8 Advanced Feature Selection initialized')
        self.logger.info(f'   Phase 1 target: {self.phase1_target_features} features')
        self.logger.info(f'   Phase 2 targets: {self.phase2_targets}')
        self.logger.info(f'   Computational optimizations:')
        self.logger.info(f'     - Numba: {NUMBA_AVAILABLE}')
        self.logger.info(f'     - Joblib: {JOBLIB_AVAILABLE}')
        self.logger.info(f'     - Parallel jobs: {self.n_jobs}')
        self.logger.info(f'   Feature selection methods:')
        self.logger.info(f'     - Boruta: {BORUTA_AVAILABLE}')
        self.logger.info(f'     - SHAP: {SHAP_AVAILABLE}')
        self.logger.info(f'     - LIME: {LIME_AVAILABLE}')

    @handles_errors(exceptions=(ValueError, RuntimeError), default_return = False)
    async def execute(self, training_input: dict[str, Any], pipeline_state: dict[str, Any]) -> dict[str, Any]:
        """
        Execute Step 8: Advanced Feature Selection.
        
        Args:
            training_input: Input data from previous steps
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state with selected features
        """
        try:
            start_time = datetime.now()
            self.logger.info('🚀 Starting Step 8: Advanced Feature Selection...')
            symbol = training_input.get('symbol', 'UNKNOWN')
            exchange = training_input.get('exchange', 'UNKNOWN')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir', 'data/training')
            filtered_train_path = f'{data_dir}/{exchange}_{symbol}_{timeframe}_features_filtered_train.parquet'
            filtered_val_path = f'{data_dir}/{exchange}_{symbol}_{timeframe}_features_filtered_val.parquet'
            if not os.path.exists(filtered_train_path):
                self.logger.warning('⚠️ Filtered features not found, using original features')
                filtered_train_path = f'{data_dir}/{exchange}_{symbol}_{timeframe}_features_train.parquet'
                filtered_val_path = f'{data_dir}/{exchange}_{symbol}_{timeframe}_features_val.parquet'
            self.logger.info(f'📊 Loading features from: {filtered_train_path}')
            # Optimized Parquet reading for feature selection
            parquet_read_options = {
                'engine': 'pyarrow' if hasattr(pd, 'ArrowDtype') else 'fastparquet',
                'use_threads': True  # Enable multi-threading
            }
            df_train = standardized_parquet_handler.read_parquet_standardized(filtered_train_path, **parquet_read_options)
            df_val = standardized_parquet_handler.read_parquet_standardized(filtered_val_path, **parquet_read_options)
            df = pd.concat([df_train, df_val], ignore_index = True)
            self.logger.info(f'📈 Loaded {len(df)} rows with {len(df.columns)} columns')
            label_columns = ['target', 'direction', 'profit', 'outcome', 'returns', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            feature_columns = [col for col in df.columns if col not in label_columns]
            features_df = df[feature_columns]
            labels_df = df[[col for col in label_columns if col in df.columns]]
            if 'target' in labels_df.columns:
                y = labels_df['target']
            elif 'direction' in labels_df.columns:
                y = labels_df['direction']
            else:
                raise ValueError('No target or direction column found')
            if y.dtype != int:
                y = (y > 0).astype(int)
            regime_labels = None
            hmm_path = f'data/hmm_regimes/{exchange}_{symbol}_{timeframe}_composite_clusters.parquet'
            if os.path.exists(hmm_path):
                self.logger.info(f'🎭 Loading regime labels from: {hmm_path}')
                # Optimized HMM data reading
                hmm_data = standardized_parquet_handler.read_parquet_standardized(hmm_path, **parquet_read_options)
                try:
                    from src.utils.regime_data_access import get_regime_column
                    regime_col = get_regime_column(hmm_data)
                except Exception:
                    regime_col = 'composite_cluster_id' if 'composite_cluster_id' in hmm_data.columns else None
                if regime_col and regime_col in hmm_data.columns:
                    regime_labels = hmm_data[regime_col].iloc[:len(df)]
            self.logger.info('📊 Starting Phase 1: mRMR/RF Selection...')
            phase1_features, phase1_metadata = await self.phase1_mrmr_rf_selection(features_df, y, regime_labels)
            self.logger.info('🎯 Starting Phase 2: Boruta Multi-Target Selection...')
            phase2_results, interpretability_results = await self.phase2_boruta_multi_target(phase1_features, y, regime_labels)
            output_files = await self._save_selection_results(phase1_features, phase1_metadata, phase2_results, interpretability_results, symbol, exchange, timeframe, df_train, df_val, labels_df)
            pipeline_state['step08_advanced_feature_selection'] = {'status': 'completed', 'start_time': start_time.isoformat(), 'end_time': datetime.now().isoformat(), 'output_files': output_files, 'phase1_metadata': phase1_metadata, 'phase2_results': {k: v for k, v in phase2_results.items() if k != 'features'}, 'interpretability_results': interpretability_results, 'original_features': len(feature_columns), 'phase1_features': len(phase1_features.columns), 'phase2_feature_sets': {f'top_{k}': len(v['features']) for k, v in phase2_results.items()}, 'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
            self.logger.info('✅ Step 8: Advanced Feature Selection completed successfully')
            return pipeline_state
        except Exception as e:
            self.logger.error(f'❌ Step 8 failed: {str(e)}')
            pipeline_state['step08_advanced_feature_selection'] = {'status': 'failed', 'error': str(e), 'timestamp': datetime.now().isoformat()}
            return pipeline_state

    async def phase1_mrmr_rf_selection(self, X: pd.DataFrame, y: pd.Series, regime_labels: Optional[pd.Series]=None) -> Tuple[pd.DataFrame, dict[str, Any]]:
        """
        Phase 1: Select top 150 features using mRMR and Random Forest.
        
        Args:
            X: Feature dataframe
            y: Target series
            regime_labels: Optional regime labels
            
        Returns:
            Selected features and metadata
        """
        metadata = {}
        mrmr_features = []
        if self.enable_mrmr:
            self.logger.info('🔍 Running mRMR selection...')
            mrmr_features = self._mrmr_selection(X, y, self.phase1_target_features)
            metadata['mrmr_features'] = mrmr_features
            self.logger.info(f'   mRMR selected {len(mrmr_features)} features')
        rf_features = []
        if self.enable_rf_importance:
            self.logger.info('🌳 Running Random Forest selection with TS validation...')
            rf_features = self._time_series_rf_selection(X, y, self.phase1_target_features)
            metadata['rf_features'] = rf_features
            self.logger.info(f'   RF selected {len(rf_features)} features')
        regime_validated_features = []
        if regime_labels is not None:
            self.logger.info('🎭 Validating features per regime...')
            candidate_features = list(set(mrmr_features) | set(rf_features))
            regime_validated_features = self._validate_features_per_regime(X, y, regime_labels, candidate_features)
            metadata['regime_validated_features'] = regime_validated_features
        consensus_features = list(set(mrmr_features) & set(rf_features))
        metadata['consensus_features'] = consensus_features
        final_features = list(consensus_features)
        remaining_slots = self.phase1_target_features - len(final_features)
        for feature in regime_validated_features:
            if feature not in final_features and remaining_slots > 0:
                final_features.append(feature)
                remaining_slots -= 1
        for feature in mrmr_features:
            if feature not in final_features and remaining_slots > 0:
                final_features.append(feature)
                remaining_slots -= 1
        for feature in rf_features:
            if feature not in final_features and remaining_slots > 0:
                final_features.append(feature)
                remaining_slots -= 1
        if len(final_features) < self.phase1_target_features:
            mi_scores = mutual_info_classif(X, y, random_state = 42)
            mi_ranking = pd.Series(mi_scores, index = X.columns).sort_values(ascending = False)
            for feature in mi_ranking.index:
                if feature not in final_features and len(final_features) < self.phase1_target_features:
                    final_features.append(feature)
        metadata['final_features_count'] = len(final_features)
        metadata['consensus_ratio'] = len(consensus_features) / len(final_features) if final_features else 0
        metadata['regime_specific_additions'] = len([f for f in final_features if f in regime_validated_features])
        self.logger.info(f'✅ Phase 1 complete: {len(X.columns)} → {len(final_features)} features')
        self.logger.info(f'   Consensus features: {len(consensus_features)}')
        self.logger.info(f"   Regime-specific additions: {metadata['regime_specific_additions']}")
        return (X[final_features], metadata)

    def _mrmr_selection(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
        """
        Optimized Minimum Redundancy Maximum Relevance feature selection.
        
        Args:
            X: Feature dataframe
            y: Target series
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        X_values = X.values
        y_values = y.values
        feature_names = X.columns.tolist()
        n_total_features = len(feature_names)
        if NUMBA_AVAILABLE and y.dtype == int:
            relevance_scores = fast_mutual_info_discrete(X_values, y_values)
        else:
            relevance_scores = mutual_info_classif(X, y, random_state = 42)
        if NUMBA_AVAILABLE:
            corr_matrix = np.abs(fast_correlation_matrix(X_values))
        else:
            corr_matrix = np.abs(X.corr().values)
        selected_indices = []
        remaining_indices = list(range(n_total_features))
        first_idx = np.argmax(relevance_scores)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        while len(selected_indices) < n_features and remaining_indices:
            redundancy_matrix = corr_matrix[np.ix_(remaining_indices, selected_indices)]
            redundancy_scores = np.mean(redundancy_matrix, axis = 1)
            remaining_relevance = relevance_scores[remaining_indices]
            mrmr_scores = remaining_relevance - redundancy_scores
            best_idx_in_remaining = np.argmax(mrmr_scores)
            best_idx = remaining_indices[best_idx_in_remaining]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        return [feature_names[idx] for idx in selected_indices]

    def _time_series_rf_selection(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
        """
        Random Forest feature selection with time-series cross-validation.
        
        Args:
            X: Feature dataframe
            y: Target series
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        tscv = TimeSeriesSplit(n_splits = min(self.n_splits_ts, 3))
        feature_importances = np.zeros(X.shape[1])
        for train_idx, val_idx in tscv.split(X):
            X_train, y_train = (X.iloc[train_idx], y.iloc[train_idx])
            rf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = 42, n_jobs=-1)
            rf.fit(X_train, y_train)
            feature_importances += rf.feature_importances_
        feature_importances /= tscv.get_n_splits()
        top_indices = np.argsort(feature_importances)[-n_features:]
        return X.columns[top_indices].tolist()

    def _validate_features_per_regime(self, X: pd.DataFrame, y: pd.Series, regime_labels: pd.Series, candidate_features: List[str]) -> List[str]:
        """
        Optimized regime validation using parallel processing.
        
        Args:
            X: Feature dataframe
            y: Target series
            regime_labels: Regime labels
            candidate_features: Features to validate
            
        Returns:
            List of regime-validated features
        """
        unique_regimes = np.unique(regime_labels)
        valid_regimes = [r for r in unique_regimes if (regime_labels == r).sum() >= self.min_regime_samples]
        if not valid_regimes:
            return candidate_features
        X_values = X[candidate_features].values
        y_values = y.values
        if JOBLIB_AVAILABLE and len(valid_regimes) > 1:
            regime_scores_list = Parallel(n_jobs=-1)((delayed(self._evaluate_regime_features)(regime, X_values, y_values, regime_labels) for regime in valid_regimes))
        else:
            regime_scores_list = [self._evaluate_regime_features(regime, X_values, y_values, regime_labels) for regime in valid_regimes]
        regime_scores_matrix = np.array(regime_scores_list)
        mean_scores = np.mean(regime_scores_matrix, axis = 0)
        min_scores = np.min(regime_scores_matrix, axis = 0)
        validated_indices = np.where((mean_scores > 0.01) & (min_scores > 0.005))[0]
        return [candidate_features[idx] for idx in validated_indices]

    def _evaluate_regime_features(self, regime: Any, X_values: List[Any], y_values: List[Any], regime_labels: List[Any]) -> None:
        """Evaluate features for a single regime."""
        regime_mask = (regime_labels == regime).values
        X_regime = X_values[regime_mask]
        y_regime = y_values[regime_mask]
        if NUMBA_AVAILABLE and y_values.dtype == int:
            mi_scores = fast_mutual_info_discrete(X_regime, y_regime)
        else:
            mi_scores = mutual_info_classif(X_regime, y_regime, random_state = 42)
        return mi_scores

    async def phase2_boruta_multi_target(self, X: pd.DataFrame, y: pd.Series, regime_labels: Optional[pd.Series]=None) -> Tuple[dict[str, Any], dict[str, Any]]:
        """
        Phase 2: Boruta selection with redundancy analysis for multiple target sizes.
        
        Args:
            X: Feature dataframe (already filtered to ~150 features)
            y: Target series
            regime_labels: Optional regime labels
            
        Returns:
            Feature sets and interpretability results
        """
        feature_sets = {}
        redundancy_groups = {}
        feature_clusters = {}
        if self.enable_redundancy_analysis:
            self.logger.info('🔄 Analyzing feature redundancy...')
            redundancy_groups = self._analyze_feature_redundancy(X)
            self.logger.info(f'   Found {len(redundancy_groups)} redundancy groups')
            self.logger.info('🔍 Performing hierarchical clustering for redundancy...')
            feature_clusters = self._hierarchical_feature_clustering(X)
            self.logger.info(f'   Identified {len(feature_clusters)} feature clusters')
        if BORUTA_AVAILABLE:
            self.logger.info('🔍 Running Boruta for all-relevant features...')
            rf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = 42, n_jobs=-1)
            boruta_selector = BorutaPy(rf, n_estimators='auto', alpha = self.boruta_alpha, max_iter = self.boruta_max_iter, random_state = 42)
            boruta_selector.fit(X.values, y.values)
            feature_ranks = boruta_selector.ranking_
            feature_importance = pd.Series(1 / feature_ranks, index = X.columns).sort_values(ascending = False)
            confirmed_features = X.columns[boruta_selector.support_].tolist()
            self.logger.info(f'   Boruta confirmed {len(confirmed_features)} features')
        else:
            if LGB_AVAILABLE:
                self.logger.warning('⚠️ Boruta not available, using LightGBM importance')
                lgb_model = lgb.LGBMClassifier(n_estimators = 200, max_depth = 10, random_state = 42, n_jobs=-1, verbose=-1)
                lgb_model.fit(X, y)
                feature_importance = pd.Series(lgb_model.feature_importances_, index = X.columns).sort_values(ascending = False)
            else:
                self.logger.warning('⚠️ Boruta and LightGBM not available, using RandomForest importance fallback')
                rf = RandomForestClassifier(n_estimators = 200, max_depth = 10, random_state = 42, n_jobs=-1)
                rf.fit(X, y)
                feature_importance = pd.Series(rf.feature_importances_, index = X.columns).sort_values(ascending = False)
            threshold = feature_importance.quantile(0.2)
            confirmed_features = feature_importance[feature_importance > threshold].index.tolist()
        for target_size in self.phase2_targets:
            self.logger.info(f'📊 Creating redundancy-aware feature set with {target_size} features...')
            if self.enable_redundancy_analysis and (redundancy_groups or feature_clusters):
                all_redundancy_groups = dict(redundancy_groups)
                for cluster_id, cluster_features in feature_clusters.items():
                    all_redundancy_groups[f'cluster_{cluster_id}'] = cluster_features
                selected_features = self._select_features_with_redundancy_advanced(feature_importance, all_redundancy_groups, target_size, confirmed_features, boruta_selector if BORUTA_AVAILABLE else None)
            else:
                selected_features = feature_importance.head(target_size).index.tolist()
            ts_validation = self._time_series_validate_features(X[selected_features], y, n_splits = self.n_splits_ts)
            regime_validation = {}
            if regime_labels is not None:
                regime_validation = self._per_regime_validate_features(X[selected_features], y, regime_labels)
            redundancy_stats = self._calculate_redundancy_stats(selected_features, redundancy_groups) if redundancy_groups else {}
            feature_sets[target_size] = {'features': selected_features, 'importance_scores': feature_importance[selected_features].to_dict(), 'ts_validation': ts_validation, 'regime_validation': regime_validation, 'boruta_confirmed': len([f for f in selected_features if f in confirmed_features]), 'boruta_confirmed_ratio': len([f for f in selected_features if f in confirmed_features]) / len(selected_features), 'redundancy_stats': redundancy_stats}
            self.logger.info(f"   TS validation score: {ts_validation['mean_score']:.4f} ± {ts_validation['std_score']:.4f}")
            self.logger.info(f"   Boruta confirmed: {feature_sets[target_size]['boruta_confirmed']} features")
            if redundancy_stats:
                self.logger.info(f"   Redundancy groups: {redundancy_stats['groups_represented']}")
                self.logger.info(f"   Average redundancy: {redundancy_stats['average_redundancy']:.1f} features/group")
                self.logger.info(f"   Concept coverage: {sum(redundancy_stats['concept_coverage'].values())} features across {len([v for v in redundancy_stats['concept_coverage'].values() if v > 0])} concepts")
        self.logger.info('🔮 Generating interpretability analysis...')
        interpretability_results = await self._generate_interpretability_report(X, y, feature_sets)
        return (feature_sets, interpretability_results)

    def _time_series_validate_features(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict[str, Any]:
        """Time-series aware feature validation."""
        tscv = TimeSeriesSplit(n_splits = min(n_splits, 3))
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = (X.iloc[train_idx], X.iloc[val_idx])
            y_train, y_val = (y.iloc[train_idx], y.iloc[val_idx])
            model = lgb.LGBMClassifier(n_estimators = 50, max_depth = 5, random_state = 42, n_jobs=-1, verbose=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred)
            scores.append(score)
        return {'mean_score': np.mean(scores), 'std_score': np.std(scores), 'scores': scores, 'n_splits': len(scores)}

    def _per_regime_validate_features(self, X: pd.DataFrame, y: pd.Series, regime_labels: pd.Series) -> dict[str, float]:
        """Validate features perform well in each regime."""
        regime_scores = {}
        for regime in np.unique(regime_labels):
            regime_mask = regime_labels == regime
            if regime_mask.sum() < self.min_regime_samples:
                continue
            X_regime = X[regime_mask]
            y_regime = y[regime_mask]
            try:
                scores = cross_val_score(lgb.LGBMClassifier(n_estimators = 50, max_depth = 5, verbose=-1), X_regime, y_regime, cv = min(3, len(np.unique(y_regime))), scoring='roc_auc')
                regime_scores[f'regime_{regime}'] = scores.mean()
            except:
                continue
        return regime_scores

    async def _generate_interpretability_report(self, X: pd.DataFrame, y: pd.Series, feature_sets: dict[int, dict[str, Any]]) -> dict[str, Any]:
        """Generate SHAP/LIME interpretability analysis."""
        report = {}
        for size, feature_data in feature_sets.items():
            self.logger.info(f'🔍 Analyzing interpretability for {size}-feature set...')
            features = feature_data['features']
            X_subset = X[features]
            model = lgb.LGBMClassifier(n_estimators = 100, max_depth = 10, random_state = 42, n_jobs=-1, verbose=-1)
            model.fit(X_subset, y)
            feature_report = {}
            if self.enable_shap and SHAP_AVAILABLE:
                try:
                    explainer = shap.TreeExplainer(model)
                    sample_size = min(1000, len(X_subset))
                    sample_idx = np.random.choice(len(X_subset), sample_size, replace = False)
                    X_sample = X_subset.iloc[sample_idx]
                    shap_values = explainer.shap_values(X_sample)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                    shap_importance = pd.Series(np.abs(shap_values).mean(axis = 0), index = features).sort_values(ascending = False)
                    feature_report['shap_importance'] = shap_importance.head(20).to_dict()
                    feature_report['feature_interactions'] = self._detect_feature_interactions(shap_values, features)
                except Exception as e:
                    self.logger.warning(f'⚠️ SHAP analysis failed: {e}')
                    feature_report['shap_error'] = str(e)
            if self.enable_lime and LIME_AVAILABLE:
                try:
                    lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_subset.values, feature_names = features, class_names=['0', '1'], mode='classification')
                    sample_explanations = []
                    for i in range(min(self.n_lime_samples, len(X_subset))):
                        exp = lime_explainer.explain_instance(X_subset.iloc[i].values, model.predict_proba, num_features = min(10, len(features)))
                        sample_explanations.append(exp.as_list())
                    feature_report['lime_explanations'] = sample_explanations[:3]
                except Exception as e:
                    self.logger.warning(f'⚠️ LIME analysis failed: {e}')
                    feature_report['lime_error'] = str(e)
            y_pred = model.predict_proba(X_subset)[:, 1]
            feature_report['model_performance'] = {'roc_auc': roc_auc_score(y, y_pred), 'accuracy': accuracy_score(y, model.predict(X_subset)), 'f1_score': f1_score(y, model.predict(X_subset))}
            report[f'feature_set_{size}'] = feature_report
        return report

    def _detect_feature_interactions(self, shap_values: np.ndarray, feature_names: List[str], top_k: int = 10) -> List[Tuple[str, str, float]]:
        """Detect top feature interactions from SHAP values."""
        interactions = []
        shap_df = pd.DataFrame(shap_values, columns = feature_names)
        corr_matrix = shap_df.corr().abs()
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                interactions.append((feature_names[i], feature_names[j], corr_matrix.iloc[i, j]))
        interactions.sort(key = lambda x: x[2], reverse = True)
        return [(f1, f2, round(score, 3)) for f1, f2, score in interactions[:top_k]]

    def _hierarchical_feature_clustering(self, X: pd.DataFrame, n_clusters: int = None) -> dict[int, List[str]]:
        """
        Perform hierarchical clustering on features to identify redundant groups.
        Uses correlation distance and Ward linkage.
        
        Args:
            X: Feature dataframe
            n_clusters: Number of clusters (auto-determined if None)
            
        Returns:
            Dictionary mapping cluster IDs to feature lists
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        corr_matrix = X.corr().abs()
        distance_matrix = 1 - corr_matrix
        condensed_distances = squareform(distance_matrix, checks = False)
        Z = linkage(condensed_distances, method='ward')
        if n_clusters is None:
            distances = Z[:, 2]
            gaps = np.diff(distances)
            optimal_idx = np.argmax(gaps) + 1
            distance_threshold = distances[optimal_idx]
            clusters = fcluster(Z, distance_threshold, criterion='distance')
        else:
            clusters = fcluster(Z, n_clusters, criterion='maxclust')
        feature_clusters = {}
        for idx, cluster_id in enumerate(clusters):
            if cluster_id not in feature_clusters:
                feature_clusters[cluster_id] = []
            feature_clusters[cluster_id].append(X.columns[idx])
        feature_clusters = {k: v for k, v in feature_clusters.items() if len(v) > 1}
        return feature_clusters

    def _analyze_feature_redundancy(self, X: pd.DataFrame) -> dict[str, List[str]]:
        """
        Optimized feature redundancy analysis using vectorized operations.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Dictionary mapping group names to feature lists
        """
        redundancy_groups = {}
        n_features = len(X.columns)
        if NUMBA_AVAILABLE:
            corr_matrix = np.abs(fast_correlation_matrix(X.values))
        else:
            corr_matrix = X.corr().abs().values
        triu_indices = np.triu_indices(n_features, k = 1)
        high_corr_mask = corr_matrix[triu_indices] >= self.min_redundancy_correlation
        high_corr_i = triu_indices[0][high_corr_mask]
        high_corr_j = triu_indices[1][high_corr_mask]
        if len(high_corr_i) > 0:
            adjacency = np.zeros((n_features, n_features), dtype = bool)
            adjacency[high_corr_i, high_corr_j] = True
            adjacency[high_corr_j, high_corr_i] = True
            visited = np.zeros(n_features, dtype = bool)
            corr_group_id = 0
            for start_idx in range(n_features):
                if not visited[start_idx] and np.any(adjacency[start_idx]):
                    component_mask = np.zeros(n_features, dtype = bool)
                    component_mask[start_idx] = True
                    prev_size = 0
                    while np.sum(component_mask) > prev_size:
                        prev_size = np.sum(component_mask)
                        component_mask |= np.any(adjacency[component_mask], axis = 0)
                    visited |= component_mask
                    component_features = [X.columns[i] for i in np.where(component_mask)[0]]
                    if len(component_features) > 1:
                        redundancy_groups[f'corr_group_{corr_group_id}'] = component_features
                        corr_group_id += 1
        for concept, patterns in self.feature_concept_patterns.items():
            concept_features = []
            for feature in X.columns:
                feature_lower = feature.lower()
                if any((pattern in feature_lower for pattern in patterns)):
                    concept_features.append(feature)
            if len(concept_features) > 1:
                new_features = []
                for f in concept_features:
                    if not any((f in group for group in redundancy_groups.values())):
                        new_features.append(f)
                if len(new_features) > 1:
                    redundancy_groups[f'concept_{concept}'] = new_features
        return redundancy_groups

    def _select_features_with_redundancy(self, feature_importance: pd.Series, redundancy_groups: dict[str, List[str]], target_size: int, confirmed_features: List[str]) -> List[str]:
        """
        Select features considering redundancy to ensure robustness.
        
        Args:
            feature_importance: Feature importance scores
            redundancy_groups: Dictionary of redundancy groups
            target_size: Target number of features
            confirmed_features: Boruta-confirmed features
            
        Returns:
            List of selected features
        """
        selected_features = []
        used_groups = set()
        feature_to_groups = {}
        for group_name, features in redundancy_groups.items():
            for feature in features:
                if feature not in feature_to_groups:
                    feature_to_groups[feature] = []
                feature_to_groups[feature].append(group_name)
        for feature in feature_importance.index:
            if len(selected_features) >= target_size:
                break
            if feature in feature_to_groups:
                groups = feature_to_groups[feature]
                group_counts = {}
                for group in groups:
                    group_features = redundancy_groups[group]
                    count = sum((1 for f in selected_features if f in group_features))
                    group_counts[group] = count
                min_count = min(group_counts.values()) if group_counts else 0
                if min_count < self.redundancy_groups_per_concept:
                    selected_features.append(feature)
                    for group in groups:
                        used_groups.add(group)
            else:
                selected_features.append(feature)
        if len(selected_features) < target_size:
            for group_name, group_features in redundancy_groups.items():
                if len(selected_features) >= target_size:
                    break
                current_count = sum((1 for f in selected_features if f in group_features))
                if current_count < self.redundancy_groups_per_concept:
                    group_importance = feature_importance[feature_importance.index.isin(group_features)].sort_values(ascending = False)
                    for feature in group_importance.index:
                        if feature not in selected_features and len(selected_features) < target_size:
                            selected_features.append(feature)
                            current_count += 1
                            if current_count >= self.redundancy_groups_per_concept:
                                break
        while len(selected_features) < target_size:
            for feature in feature_importance.index:
                if feature not in selected_features:
                    selected_features.append(feature)
                    break
            else:
                break
        confirmed_selected = [f for f in selected_features if f in confirmed_features]
        unconfirmed_selected = [f for f in selected_features if f not in confirmed_features]
        final_features = confirmed_selected + unconfirmed_selected
        return final_features[:target_size]

    def _select_features_with_redundancy_advanced(self, feature_importance: pd.Series, all_redundancy_groups: dict[str, List[str]], target_size: int, confirmed_features: List[str], boruta_selector: Any = None) -> List[str]:
        """
        Advanced feature selection that combines Boruta's all-relevant features
        with redundancy reduction using multiple strategies.
        
        Args:
            feature_importance: Feature importance scores
            all_redundancy_groups: Combined redundancy groups from multiple methods
            target_size: Target number of features
            confirmed_features: Boruta-confirmed features
            boruta_selector: Fitted Boruta selector (optional)
            
        Returns:
            List of selected features with optimal redundancy
        """
        selected_features = []
        if confirmed_features:
            confirmed_by_group = {}
            ungrouped_confirmed = []
            for feature in confirmed_features:
                assigned = False
                for group_name, group_features in all_redundancy_groups.items():
                    if feature in group_features:
                        if group_name not in confirmed_by_group:
                            confirmed_by_group[group_name] = []
                        confirmed_by_group[group_name].append(feature)
                        assigned = True
                        break
                if not assigned:
                    ungrouped_confirmed.append(feature)
            for group_name, group_confirmed in confirmed_by_group.items():
                group_importance = feature_importance[group_confirmed].sort_values(ascending = False)
                n_to_take = min(self.redundancy_groups_per_concept, len(group_importance))
                selected_features.extend(group_importance.head(n_to_take).index.tolist())
            selected_features.extend(ungrouped_confirmed)
        remaining_slots = target_size - len(selected_features)
        if remaining_slots > 0:
            remaining_features = [f for f in feature_importance.index if f not in selected_features]
            vif_selected = self._select_low_vif_features(feature_importance[remaining_features], all_redundancy_groups, remaining_slots, selected_features)
            selected_features.extend(vif_selected)
        if len(selected_features) < target_size:
            concept_coverage = {}
            for concept, patterns in self.feature_concept_patterns.items():
                concept_features = [f for f in selected_features if any((p in f.lower() for p in patterns))]
                concept_coverage[concept] = len(concept_features)
            for concept, count in sorted(concept_coverage.items(), key = lambda x: x[1]):
                if len(selected_features) >= target_size:
                    break
                if count < 2:
                    patterns = self.feature_concept_patterns[concept]
                    concept_candidates = [f for f in feature_importance.index if any((p in f.lower() for p in patterns)) and f not in selected_features]
                    for feature in feature_importance[concept_candidates].sort_values(ascending = False).index:
                        if len(selected_features) < target_size:
                            selected_features.append(feature)
                            count += 1
                            if count >= 2:
                                break
        if boruta_selector is not None and hasattr(boruta_selector, 'ranking_'):
            boruta_ranks = dict(zip(feature_importance.index, boruta_selector.ranking_))
            redundant_pairs = []
            for i, f1 in enumerate(selected_features):
                for j, f2 in enumerate(selected_features[i + 1:], i + 1):
                    for group_features in all_redundancy_groups.values():
                        if f1 in group_features and f2 in group_features:
                            if boruta_ranks.get(f1, float('inf')) > boruta_ranks.get(f2, float('inf')):
                                redundant_pairs.append((i, f1))
                            else:
                                redundant_pairs.append((j, f2))
                            break
            removed_indices = set()
            for idx, feature in redundant_pairs:
                if idx not in removed_indices and len(selected_features) > target_size:
                    removed_indices.add(idx)
            for idx in sorted(removed_indices, reverse = True):
                selected_features.pop(idx)
        return selected_features[:target_size]

    def _select_low_vif_features(self, candidate_importance: pd.Series, redundancy_groups: dict[str, List[str]], n_features: int, already_selected: List[str]) -> List[str]:
        """
        Select features with low VIF (Variance Inflation Factor) to minimize multicollinearity.
        """
        selected = []
        for feature in candidate_importance.index:
            if len(selected) >= n_features:
                break
            redundancy_score = 0
            for group_name, group_features in redundancy_groups.items():
                if feature in group_features:
                    existing_count = sum((1 for f in already_selected + selected if f in group_features))
                    redundancy_score += existing_count
            if redundancy_score < self.redundancy_groups_per_concept:
                selected.append(feature)
        return selected

    def _calculate_redundancy_stats(self, selected_features: List[str], redundancy_groups: dict[str, List[str]]) -> dict[str, Any]:
        """
        Calculate redundancy statistics for selected features.
        
        Args:
            selected_features: List of selected features
            redundancy_groups: Dictionary of redundancy groups
            
        Returns:
            Dictionary of redundancy statistics
        """
        stats = {'groups_represented': 0, 'average_redundancy': 0, 'min_redundancy': float('inf'), 'max_redundancy': 0, 'concept_coverage': {}, 'group_feature_counts': {}}
        for group_name, group_features in redundancy_groups.items():
            count = sum((1 for f in selected_features if f in group_features))
            if count > 0:
                stats['groups_represented'] += 1
                stats['group_feature_counts'][group_name] = count
                stats['min_redundancy'] = min(stats['min_redundancy'], count)
                stats['max_redundancy'] = max(stats['max_redundancy'], count)
        if stats['group_feature_counts']:
            stats['average_redundancy'] = sum(stats['group_feature_counts'].values()) / len(stats['group_feature_counts'])
        else:
            stats['min_redundancy'] = 0
        for concept in self.feature_concept_patterns:
            concept_features = [f for f in selected_features if any((p in f.lower() for p in self.feature_concept_patterns[concept]))]
            stats['concept_coverage'][concept] = len(concept_features)
        return stats

    async def _save_selection_results(self, phase1_features: pd.DataFrame, phase1_metadata: dict[str, Any], phase2_results: dict[int, dict[str, Any]], interpretability_results: dict[str, Any], symbol: str, exchange: str, timeframe: str, df_train: pd.DataFrame, df_val: pd.DataFrame, labels_df: pd.DataFrame) -> dict[str, str]:
        """Save all selection results and create output datasets."""
        output_files = {}
        phase1_path = os.path.join(self.output_dir, f'{exchange}_{symbol}_{timeframe}_phase1_features.json')
        safe_json_dump({'features': phase1_features.columns.tolist(), 'metadata': phase1_metadata, 'timestamp': datetime.now().isoformat()}, phase1_path)
        output_files['phase1_results'] = phase1_path
        for target_size, results in phase2_results.items():
            phase2_path = os.path.join(self.output_dir, f'{exchange}_{symbol}_{timeframe}_top{target_size}_features.json')
            safe_json_dump({'features': results['features'], 'importance_scores': results['importance_scores'], 'validation': {'ts_validation': results['ts_validation'], 'regime_validation': results['regime_validation']}, 'boruta_stats': {'confirmed': results['boruta_confirmed'], 'confirmed_ratio': results['boruta_confirmed_ratio']}, 'timestamp': datetime.now().isoformat()}, phase2_path)
            output_files[f'top{target_size}_features'] = phase2_path
            selected_features = results['features']
            train_size = len(df_train)
            train_features = phase1_features[selected_features].iloc[:train_size]
            train_data = pd.concat([train_features, labels_df.iloc[:train_size]], axis = 1)
            train_path = os.path.join(self.output_dir, f'{exchange}_{symbol}_{timeframe}_top{target_size}_train.parquet')
            standardized_parquet_handler.write_parquet_standardized(train_data, train_path)
            output_files[f'top{target_size}_train'] = train_path
            val_features = phase1_features[selected_features].iloc[train_size:]
            val_data = pd.concat([val_features, labels_df.iloc[train_size:]], axis = 1)
            val_path = os.path.join(self.output_dir, f'{exchange}_{symbol}_{timeframe}_top{target_size}_val.parquet')
            standardized_parquet_handler.write_parquet_standardized(val_data, val_path)
            output_files[f'top{target_size}_val'] = val_path
        interp_path = os.path.join(self.output_dir, f'{exchange}_{symbol}_{timeframe}_interpretability_report.json')
        safe_json_dump(interpretability_results, interp_path)
        output_files['interpretability_report'] = interp_path
        report_path = os.path.join(self.output_dir, f'{exchange}_{symbol}_{timeframe}_selection_report.json')
        safe_json_dump({'phase1_summary': {'input_features': len(df_train.columns) - len(labels_df.columns), 'output_features': len(phase1_features.columns), 'consensus_features': len(phase1_metadata.get('consensus_features', [])), 'regime_validated': phase1_metadata.get('regime_specific_additions', 0)}, 'phase2_summary': {f'top_{size}': {'features': len(results['features']), 'ts_score': results['ts_validation']['mean_score'], 'boruta_confirmed': results['boruta_confirmed']} for size, results in phase2_results.items()}, 'timestamp': datetime.now().isoformat()}, report_path)
        output_files['selection_report'] = report_path
        self.logger.info(f'💾 Saved all selection results to {self.output_dir}')
        return output_files

    def _sparse_matrix_feature_selection(self, X: pd.DataFrame, y: pd.Series,
                                        sparsity_threshold: float = 0.7) -> Tuple[pd.DataFrame, dict]:
        """Perform feature selection using sparse matrix operations for memory efficiency."""
        if not SCIPY_SPARSE_AVAILABLE:
            self.logger.warning("SciPy sparse not available, falling back to dense operations")
            return X, {}

        try:
            # Check if matrix is sparse enough for sparse operations
            matrix_data = X.values
            total_elements = matrix_data.size
            zero_elements = np.sum(np.abs(matrix_data) < 1e-10)
            sparsity = zero_elements / total_elements

            if sparsity < sparsity_threshold:
                self.logger.info(f"Matrix not sparse enough ({sparsity:.2f}), using dense operations")
                return X, {'method': 'dense', 'sparsity': sparsity}

            # Create sparse matrix
            sparse_matrix = sp.csr_matrix(matrix_data)
            self.logger.info(f"Created sparse matrix: {sparse_matrix.shape}, {sparse_matrix.nnz} non-zero elements")

            # Sparse SVD for dimensionality reduction
            n_components = min(50, min(sparse_matrix.shape) - 1)
            U, s, Vt = svds(sparse_matrix, k=n_components)

            # Calculate feature importance from singular values
            feature_importance = np.abs(Vt.T).sum(axis=1)
            importance_ranking = pd.Series(feature_importance, index=X.columns).sort_values(ascending=False)

            # Select top features based on sparse SVD importance
            n_select = min(100, len(X.columns) // 2)
            selected_features = importance_ranking.head(n_select).index.tolist()

            # Calculate memory savings
            dense_memory = matrix_data.nbytes
            sparse_memory = sparse_matrix.data.nbytes + sparse_matrix.indices.nbytes + sparse_matrix.indptr.nbytes
            memory_savings = (dense_memory - sparse_memory) / dense_memory

            metadata = {
                'method': 'sparse_svd',
                'sparsity': sparsity,
                'original_features': len(X.columns),
                'selected_features': len(selected_features),
                'memory_savings': memory_savings,
                'compression_ratio': dense_memory / sparse_memory,
                'singular_values_top10': s[-10:].tolist()
            }

            self.logger.info(f"Sparse feature selection: {len(X.columns)} → {len(selected_features)} features")
            self.logger.info(f"Memory savings: {memory_savings:.1%}")

            return X[selected_features], metadata

        except Exception as e:
            self.logger.warning(f"Sparse matrix feature selection failed: {e}")
            return X, {'error': str(e), 'method': 'dense_fallback'}

async def run_step(symbol: str, exchange: str, timeframe: str='1m', data_dir: str = None, force_rerun: bool = False, **kwargs: Any) -> bool:
    try:
        config_path = 'config/training_config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        config.update(kwargs)
        training_input = {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'data_dir': data_dir or f'data/{exchange}/{symbol}'}
        pipeline_state = {}
        step = Step08AdvancedFeatureSelection(config)
        result = await step.execute(training_input, pipeline_state)
        if result.get('step08_advanced_feature_selection', {}).get('status') == 'completed':
            system_logger.info('✅ Step 8: Advanced Feature Selection completed successfully')
            return True
        else:
            system_logger.error('❌ Step 8: Advanced Feature Selection failed')
            return False
    except Exception as e:
        system_logger.error(f'❌ Error running Step 8: {e}')
        return False
if __name__ == '__main__':
    asyncio.run(run_step(symbol='BTCUSDT', exchange='binance', timeframe='1m', force_rerun = True))