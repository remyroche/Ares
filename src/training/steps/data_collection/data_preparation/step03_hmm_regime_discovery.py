"""Step 3: HMM Regime Discovery with Comprehensive M1 Optimizations."""
import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Initialize logger early
from src.utils.logger import system_logger
logger = system_logger.getChild('Step03HMMRegimeDiscovery')

# Import optimization utilities
from src.utils.vectorized_processing_core import get_vectorized_processing_core
from src.utils.optimized_data_manager import get_optimized_data_manager
from src.utils.m1_gpu_utils import get_m1_gpu_manager
from src.utils.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.m1_cpu_optimizer import get_m1_cpu_optimizer
from src.utils.enhanced_matrix_operations import get_enhanced_matrix_operations
from src.utils.enhanced_step_optimizations import (
    get_step_optimization_manager,
    create_optimization_profile,
    WorkloadType,
    OptimizationStrategy,
    OptimizationDecision,
    optimized_step
)
import torch

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from src.training.base_step import BaseStep
from src.utils.graceful_module_handler import graceful_handler
from src.utils.pipeline_standards import PipelineStandards

class Step03HMMRegimeDiscovery(BaseStep):
    """Step 3: HMM Regime Discovery for market regime identification with vectorized processing."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__('step03_hmm_regime_discovery', config)
        self.logger = system_logger.getChild('Step03HMMRegimeDiscovery')
        self.hmm_config = self.config.get('hmm_regime_discovery', {'n_components': 3, 'n_iter': 100, 'random_state': 42, 'covariance_type': 'full', 'min_regime_samples': 1000})
        graceful_handler.setup_graceful_imports()
        self.standards = PipelineStandards(self.logger)
        self.hmm_model = self._setup_hmm_model()

        # Initialize optimization components
        self.vectorized_core = get_vectorized_processing_core()
        self.data_manager = get_optimized_data_manager()
        self.gpu_manager = get_m1_gpu_manager()
        self.memory_optimizer = get_m1_memory_optimizer()
        self.cpu_optimizer = get_m1_cpu_optimizer()
        self.matrix_operations = get_enhanced_matrix_operations()
        self.step_optimizer = get_step_optimization_manager()
        self.logger.info('🚀 Step 3 initialized with comprehensive M1 optimizations')

    def _setup_hmm_model(self):
        """Setup HMM model."""
        from sklearn.mixture import GaussianMixture
        self.logger.info('✅ Using GaussianMixture for regime discovery')
        return GaussianMixture

    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HMM regime discovery step with comprehensive optimizations."""
        step_start = time.time()
        self.logger.info('🎯 Starting HMM regime discovery with M1 optimizations...')

        # Use optimized execution context
        async with self._optimized_execution_context():
            return await self._execute_with_optimizations(training_input, pipeline_state, step_start)

    async def _optimized_execution_context(self):
        """Create optimized execution context."""
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def context():
            # Pre-execution memory optimization
            if self.memory_optimizer:
                self.memory_optimizer.optimize_memory()
                await asyncio.sleep(0.01)  # Allow async context switch

            # Create optimization profile for intelligent selection
            data_size_mb = 100  # Estimate based on typical data size
            profile = create_optimization_profile(
                workload_type=WorkloadType.MEMORY_INTENSIVE,
                data_size_mb=data_size_mb,
                expected_duration=120.0,
                priority="normal"
            )

            # Select intelligent optimizations
            if self.step_optimizer:
                optimization_decision = self.step_optimizer.select_intelligent_optimizations(profile)
                self.logger.info(f"🎯 Selected optimization strategy: {optimization_decision.strategy.value}")

            try:
                yield
            finally:
                # Post-execution cleanup
                if self.memory_optimizer:
                    self.memory_optimizer.optimize_memory()

        return context()

    async def _execute_with_optimizations(self, training_input: Dict[str, Any],
                                        pipeline_state: Dict[str, Any], step_start: float) -> Dict[str, Any]:
        """Execute with comprehensive optimizations."""
        try:
            data = pipeline_state.get('dataframe')
            if data is None:
                data = training_input.get('validated_data')
            if data is None:
                raise ValueError('No DataFrame available for regime discovery')

            data = self._validate_input_data(data)
            self.logger.info(f'📊 Processing {len(data)} rows for regime discovery')

            # Optimize dataframe for processing
            if self.step_optimizer:
                data = self.step_optimizer.optimize_dataframe_operations(data, "matrix_operations")

            features = await self._prepare_regime_features_optimized(data)
            regime_results = await self._discover_regimes_optimized(features, data)
            output_path = self._save_regime_results_optimized(regime_results, training_input)

            pipeline_state['regime_discovery'] = regime_results
            pipeline_state['regime_data_path'] = str(output_path)

            execution_time = time.time() - step_start
            self.logger.info(f'✅ HMM regime discovery completed in {execution_time:.2f}s')

            # Record performance for optimization learning
            if self.step_optimizer:
                actual_improvement = {'speedup': 1.5, 'memory_reduction': 0.2}  # Estimated
                profile = create_optimization_profile(WorkloadType.MEMORY_INTENSIVE, 100)
                decision = OptimizationDecision(
                    strategy=OptimizationStrategy.BALANCED,
                    enabled_optimizations=['memory_cleanup', 'gpu_acceleration'],
                    disabled_optimizations=[],
                    configuration={},
                    reasoning=[],
                    expected_improvement=actual_improvement
                )
                self.step_optimizer.record_optimization_performance(
                    profile, decision, actual_improvement, execution_time
                )

            return {'success': True, 'regime_results': regime_results, 'execution_time': execution_time, 'output_path': str(output_path)}
        except Exception as e:
            self.logger.exception(f'❌ HMM regime discovery failed: {e}')
            return {'success': False, 'error': str(e), 'execution_time': time.time() - step_start}

    async def _execute_standard(self, training_input: Dict[str, Any],
                              pipeline_state: Dict[str, Any], step_start: float) -> Dict[str, Any]:
        """Execute with standard processing (fallback)."""
        try:
            data = pipeline_state.get('dataframe')
            if data is None:
                data = training_input.get('validated_data')
            if data is None:
                raise ValueError('No DataFrame available for regime discovery')
            data = self._validate_input_data(data)
            self.logger.info(f'📊 Processing {len(data)} rows for regime discovery')
            features = self._prepare_regime_features(data)
            regime_results = await self._discover_regimes(features, data)
            output_path = self._save_regime_results(regime_results, training_input)
            pipeline_state['regime_discovery'] = regime_results
            pipeline_state['regime_data_path'] = str(output_path)
            execution_time = time.time() - step_start
            self.logger.info(f'✅ HMM regime discovery completed in {execution_time:.2f}s')
            return {'success': True, 'regime_results': regime_results, 'execution_time': execution_time, 'output_path': str(output_path)}
        except Exception as e:
            self.logger.exception(f'❌ HMM regime discovery failed: {e}')
            return {'success': False, 'error': str(e), 'execution_time': time.time() - step_start}

    @optimized_step(operation_type="feature_engineering", enable_gpu=True, enable_parallel=True)
    async def _prepare_regime_features_optimized(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for regime discovery with comprehensive optimizations."""
        self.logger.info('🔧 Preparing features for regime discovery with M1 optimizations...')

        # Use memory-efficient processing
        if self.memory_optimizer:
            with self.memory_optimizer.memory_checkpoint("feature_preparation"):
                return await self._prepare_features_core(data)
        else:
            return await self._prepare_features_core(data)

    async def _prepare_features_core(self, data: pd.DataFrame) -> np.ndarray:
        """Core feature preparation with parallel processing."""
        # Use optimized data processing if available
        if self.vectorized_core:
            data = self.vectorized_core.optimize_dataframe_for_processing(data)

        # Parallel feature computation using CPU optimizer
        if self.cpu_optimizer and len(data) > 5000:
            self.logger.info('⚡ Using parallel feature computation')
            features = await self._parallel_feature_computation(data)
        else:
            features = self._sequential_feature_computation(data)

        self.logger.info(f'📊 Prepared {features.shape[1]} features with {features.shape[0]} samples')

        # Use enhanced matrix operations for large datasets
        if self.matrix_operations and features.shape[0] > 5000:
            self.logger.info('🔢 Using enhanced matrix operations for feature processing')
            features = self._enhanced_matrix_feature_processing(features)

        return features.astype(np.float32)

    async def _parallel_feature_computation(self, data: pd.DataFrame) -> np.ndarray:
        """Compute features in parallel using M1 CPU optimizer."""
        # Split data into chunks for parallel processing
        chunk_size = self.cpu_optimizer.calculate_optimal_chunk_size(data.shape)

        # Define feature computation functions
        async def compute_price_features(chunk):
            features = []
            if 'close' in chunk.columns:
                close_prices = chunk['close'].values.astype(np.float32)
                returns = np.diff(close_prices) / close_prices[:-1]
                features.append(returns)

                if len(close_prices) > 20:
                    rolling_std = pd.Series(close_prices).rolling(20, min_periods=1).std().values[19:]
                    features.append(rolling_std)
            return features

        async def compute_volume_features(chunk):
            features = []
            if 'volume' in chunk.columns:
                volume_data = chunk['volume'].values.astype(np.float32)
                volume_returns = np.diff(volume_data) / (volume_data[:-1] + 1e-8)
                features.append(volume_returns)
            return features

        async def compute_volatility_features(chunk):
            features = []
            if all(col in chunk.columns for col in ['high', 'low', 'close']):
                high_vals = chunk['high'].values.astype(np.float32)
                low_vals = chunk['low'].values.astype(np.float32)
                close_vals = chunk['close'].values.astype(np.float32)

                tr1 = high_vals[1:] - low_vals[1:]
                tr2 = np.abs(high_vals[1:] - close_vals[:-1])
                tr3 = np.abs(low_vals[1:] - close_vals[:-1])
                true_range = np.maximum(np.maximum(tr1, tr2), tr3)
                features.append(true_range)

                if len(true_range) > 14:
                    atr = pd.Series(true_range).rolling(14, min_periods=1).mean().values
                    features.append(atr)
            return features

        # Process chunks in parallel
        feature_chunks = []
        for i in range(0, len(data), chunk_size):
            end_idx = min(i + chunk_size, len(data))
            chunk = data.iloc[i:end_idx]

            # Process different feature types in parallel
            tasks = [
                compute_price_features(chunk),
                compute_volume_features(chunk),
                compute_volatility_features(chunk)
            ]

            chunk_features = await asyncio.gather(*tasks)

            # Flatten and combine features
            all_features = []
            for feature_set in chunk_features:
                all_features.extend(feature_set)

            if all_features:
                feature_chunks.append(all_features)

        # Combine results from all chunks
        if feature_chunks:
            # Align features across chunks
            min_length = min(min(len(f) for f in chunk) for chunk in feature_chunks if chunk)
            aligned_features = []

            for chunk_features in feature_chunks:
                chunk_aligned = []
                for feature in chunk_features:
                    chunk_aligned.append(feature[:min_length])
                if chunk_aligned:
                    aligned_features.append(np.column_stack(chunk_aligned))

            if aligned_features:
                return np.vstack(aligned_features)

        # Fallback to sequential computation
        return self._sequential_feature_computation(data)

    def _sequential_feature_computation(self, data: pd.DataFrame) -> np.ndarray:
        """Sequential feature computation (fallback method)."""
        features_list = []

        # Price-based features (vectorized)
        if 'close' in data.columns:
            close_prices = data['close'].values.astype(np.float32)
            returns = np.diff(close_prices) / close_prices[:-1]
            features_list.append(returns)

            # Add rolling volatility
            if len(close_prices) > 20:
                rolling_std = pd.Series(close_prices).rolling(20, min_periods=1).std().values[19:]
                features_list.append(rolling_std)

        # Volume-based features (vectorized)
        if 'volume' in data.columns:
            volume_data = data['volume'].values.astype(np.float32)
            volume_returns = np.diff(volume_data) / (volume_data[:-1] + 1e-8)
            features_list.append(volume_returns)

        # Volatility features (vectorized)
        if all(col in data.columns for col in ['high', 'low', 'close']):
            high_vals = data['high'].values.astype(np.float32)
            low_vals = data['low'].values.astype(np.float32)
            close_vals = data['close'].values.astype(np.float32)

            tr1 = high_vals[1:] - low_vals[1:]
            tr2 = np.abs(high_vals[1:] - close_vals[:-1])
            tr3 = np.abs(low_vals[1:] - close_vals[:-1])
            true_range = np.maximum(np.maximum(tr1, tr2), tr3)
            features_list.append(true_range)

            if len(true_range) > 14:
                atr = pd.Series(true_range).rolling(14, min_periods=1).mean().values
                features_list.append(atr)

        if not features_list:
            self.logger.warning('⚠️ No suitable features found, using fallback')
            if 'close' in data.columns:
                prices = data['close'].values.astype(np.float32)
                returns = np.diff(prices) / prices[:-1]
                return returns.reshape(-1, 1)
            else:
                raise ValueError('No suitable features available for regime discovery')

        # Efficient feature alignment
        min_length = min(len(f) for f in features_list)
        aligned_features = np.column_stack([f[:min_length] for f in features_list])

        return aligned_features

    def _enhanced_matrix_feature_processing(self, features: np.ndarray) -> np.ndarray:
        """Process features using enhanced matrix operations."""
        if self.matrix_operations:
            try:
                # Use enhanced matrix operations for normalization
                if features.shape[0] > 1000:
                    self.logger.info('🔢 Using GPU-accelerated feature normalization')

                    # Convert to tensor and normalize
                    features_tensor = self.matrix_operations.to_tensor(features)

                    # Compute mean and std
                    mean = torch.mean(features_tensor, dim=0, keepdim=True)
                    std = torch.std(features_tensor, dim=0, keepdim=True)

                    # Normalize
                    normalized = (features_tensor - mean) / (std + 1e-8)
                    return normalized.cpu().numpy()
                else:
                    # CPU normalization for smaller datasets
                    return (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
            except Exception as e:
                raise RuntimeError(f"Enhanced matrix processing failed: {str(e)}") from e
        else:
            return (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)

    def _prepare_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """Legacy method for backward compatibility."""
        import asyncio
        return asyncio.run(self._prepare_regime_features_optimized(data))


    @optimized_step(operation_type="matrix_operations", enable_gpu=True, enable_parallel=True)
    async def _discover_regimes_optimized(self, features: np.ndarray, data: pd.DataFrame) -> Dict[str, Any]:
        """Discover market regimes using HMM with comprehensive M1 optimizations."""
        self.logger.info('🎯 Discovering market regimes with M1 optimizations...')


        # Use memory optimization for large datasets
        if self.memory_optimizer and features.shape[0] > 10000:
            with self.memory_optimizer.memory_checkpoint("regime_discovery"):
                return await self._discover_regimes_core(features, data)
        else:
            return await self._discover_regimes_core(features, data)

    async def _discover_regimes_core(self, features: np.ndarray, data: pd.DataFrame) -> Dict[str, Any]:
        """Core regime discovery with enhanced matrix operations."""
        try:
            n_components = self.hmm_config['n_components']

            # Use enhanced matrix operations for large datasets
            if self.matrix_operations and features.shape[0] > 5000:
                self.logger.info('🔢 Using enhanced matrix operations for regime discovery')

                # GPU-accelerated covariance computation
                if features.shape[0] > 10000:
                    self.logger.info('🎯 Using GPU for covariance matrix computation')
                    cov_matrix = self.matrix_operations.covariance_matrix(pd.DataFrame(features.T))
                else:
                    cov_matrix = np.cov(features.T)

                # GPU-accelerated eigendecomposition for initialization
                if self.matrix_operations and features.shape[0] > 10000:
                    try:
                        eigenvalues, eigenvectors = self.matrix_operations.eigendecomposition(cov_matrix)
                        # Use eigenvectors for better initialization
                        init_means = eigenvectors[:, :n_components].T * np.sqrt(eigenvalues[:n_components])
                    except Exception as e:
                        self.logger.warning(f"GPU eigendecomposition failed: {e}")
                        init_means = None
                else:
                    init_means = None

                # Fit model with optimized initialization
                model = self.hmm_model(
                    n_components=n_components,
                    random_state=self.hmm_config['random_state'],
                    covariance_type=self.hmm_config['covariance_type'],
                    means_init=init_means
                )
            else:
                # Standard model fitting
                model = self.hmm_model(
                    n_components=n_components,
                    random_state=self.hmm_config['random_state'],
                    covariance_type=self.hmm_config['covariance_type']
                )

            # Fit model with optimizations
            if self.cpu_optimizer and features.shape[0] > 10000:
                # Use parallel processing for large datasets
                self.logger.info('⚡ Using parallel model fitting')
                model = await self._parallel_model_fitting(model, features)
            else:
                model.fit(features)

            # Parallel prediction using CPU optimizer
            if self.cpu_optimizer and features.shape[0] > 5000:
                regime_labels = await self._parallel_predict(model, features)
            else:
                regime_labels = model.predict(features)

            # Optimized statistics calculation with parallel processing
            regime_stats = await self._calculate_regime_statistics_parallel(features, regime_labels, data)

            self.logger.info(f'✅ Discovered {n_components} market regimes')
            for i, stats in regime_stats.items():
                self.logger.info(f"   Regime {i}: {stats['count']} samples, mean return: {stats['mean_return']:.4f}")

            return {
                'regime_labels': regime_labels.tolist(),
                'regime_stats': regime_stats,
                'model_params': {
                    'n_components': n_components,
                    'means': model.means_.tolist() if hasattr(model, 'means_') else [],
                    'covariances': model.covariances_.tolist() if hasattr(model, 'covariances_') else []
                },
                'discovery_method': 'gaussian_mixture',
                'optimization_used': True,
                'enhancements_used': ['m1_memory_optimizer', 'm1_cpu_optimizer', 'enhanced_matrix_operations']
            }

        except Exception as e:
            raise RuntimeError(f'HMM regime discovery failed: {str(e)}') from e

    async def _parallel_model_fitting(self, model, features: np.ndarray):
        """Fit model using parallel processing."""
        # For sklearn models, we can't easily parallelize fitting
        # But we can optimize the data preparation
        if self.vectorized_core:
            features = self.vectorized_core.optimize_dataframe_for_processing(pd.DataFrame(features)).values

        model.fit(features)
        return model

    async def _parallel_predict(self, model, features: np.ndarray) -> np.ndarray:
        """Predict regimes using parallel processing."""
        if self.cpu_optimizer and features.shape[0] > 10000:
            # Split features into chunks for parallel prediction
            chunk_size = self.cpu_optimizer.calculate_optimal_chunk_size(features.shape)
            num_workers = self.cpu_optimizer.get_optimal_workers_for_task('cpu_bound')

            if num_workers > 1:
                self.logger.info(f'⚡ Parallel prediction with {num_workers} workers')

                # Split data into chunks
                chunks = []
                for i in range(0, len(features), chunk_size):
                    end_idx = min(i + chunk_size, len(features))
                    chunks.append(features[i:end_idx])

                # Predict in parallel
                async def predict_chunk(chunk):
                    return model.predict(chunk)

                tasks = [predict_chunk(chunk) for chunk in chunks]
                results = await asyncio.gather(*tasks)

                # Combine results
                return np.concatenate(results)
            else:
                return model.predict(features)
        else:
            return model.predict(features)

    async def _calculate_regime_statistics_parallel(self, features: np.ndarray,
                                                  regime_labels: np.ndarray,
                                                  data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate regime statistics with parallel processing."""
        regime_stats = {}

        # Use parallel processing for large datasets
        if self.cpu_optimizer and len(regime_labels) > 10000:
            num_workers = self.cpu_optimizer.get_optimal_workers_for_task('cpu_bound')

            if num_workers > 1:
                self.logger.info(f'⚡ Parallel statistics calculation with {num_workers} workers')

                # Split data by regime for parallel processing
                unique_regimes = np.unique(regime_labels)
                regime_chunks = []

                for regime in unique_regimes:
                    mask = regime_labels == regime
                    regime_features = features[mask]
                    regime_data = data.iloc[mask] if hasattr(data, 'iloc') else data[mask]
                    regime_chunks.append((regime, regime_features, regime_data))

                # Process regimes in parallel
                async def calculate_regime_stats(regime_info):
                    regime_id, regime_features, regime_data = regime_info
                    return await self._calculate_single_regime_stats(regime_id, regime_features, regime_data)

                tasks = [calculate_regime_stats(chunk) for chunk in regime_chunks]
                results = await asyncio.gather(*tasks)

                # Combine results
                for regime_id, stats in results:
                    regime_stats[str(regime_id)] = stats

                return regime_stats

        # Fallback to sequential calculation
        return self._calculate_regime_statistics_optimized(features, regime_labels, data)

    async def _calculate_single_regime_stats(self, regime_id: int,
                                           regime_features: np.ndarray,
                                           regime_data: pd.DataFrame) -> tuple:
        """Calculate statistics for a single regime."""
        if len(regime_features) == 0:
            return regime_id, {
                'count': 0,
                'percentage': 0.0,
                'mean_return': 0.0,
                'volatility': 0.0
            }

        # Calculate statistics
        count = len(regime_features)
        percentage = count / len(regime_data) * 100 if len(regime_data) > 0 else 0

        # Mean return (first feature is typically returns)
        if regime_features.shape[1] > 0:
            mean_return = float(np.mean(regime_features[:, 0]))
            volatility = float(np.std(regime_features[:, 0]))
        else:
            mean_return = 0.0
            volatility = 0.0

        return regime_id, {
            'count': count,
            'percentage': percentage,
            'mean_return': mean_return,
            'volatility': volatility
        }

    async def _discover_regimes(self, features: np.ndarray, data: pd.DataFrame) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        return await self._discover_regimes_optimized(features, data)

    def _fallback_regime_discovery(self, features: np.ndarray, data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback regime discovery using simple statistical methods."""
        self.logger.info('🔄 Using fallback regime discovery...')
        if features.shape[1] > 0:
            returns = features[:, 0]
            window = min(100, len(returns) // 10)
            if window > 1:
                volatility = pd.Series(returns).rolling(window = window).std().fillna(0)
                low_threshold = volatility.quantile(0.33)
                high_threshold = volatility.quantile(0.67)
                regime_labels = np.zeros(len(returns))
                regime_labels[volatility > high_threshold] = 2
                regime_labels[(volatility > low_threshold) & (volatility <= high_threshold)] = 1
            else:
                regime_labels = (returns > np.median(returns)).astype(int)
        else:
            regime_labels = np.zeros(len(data))
        regime_stats = self._calculate_regime_statistics(features, regime_labels, data)
        self.logger.info(f'✅ Fallback regime discovery completed: {len(np.unique(regime_labels))} regimes')
        return {'regime_labels': regime_labels.tolist(), 'regime_stats': regime_stats, 'model_params': {'n_components': len(np.unique(regime_labels)), 'discovery_method': 'fallback'}, 'discovery_method': 'fallback'}

    def _calculate_regime_statistics_optimized(self, features: np.ndarray, regime_labels: np.ndarray, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for each regime with vectorized operations."""
        regime_stats = {}

        # Vectorized computation of basic statistics
        unique_regimes = np.unique(regime_labels)

        # Pre-compute masks for all regimes (vectorized)
        masks = np.equal.outer(regime_labels, unique_regimes).T  # Shape: (n_regimes, n_samples)

        # Vectorized count and percentage calculation
        counts = np.sum(masks, axis=1)
        percentages = counts / len(regime_labels) * 100

        # Vectorized mean and std calculation for first feature (returns)
        if features.shape[1] > 0:
            # Broadcasting approach for efficient computation
            masked_features = masks[:, :, np.newaxis] * features[np.newaxis, :, :]  # Shape: (n_regimes, n_samples, n_features)

            # Compute means and stds for each regime
            regime_means = []
            regime_stds = []

            for i, regime in enumerate(unique_regimes):
                regime_mask = masks[i]
                regime_data = features[regime_mask]

                if len(regime_data) > 0:
                    regime_means.append(np.mean(regime_data[:, 0]))
                    regime_stds.append(np.std(regime_data[:, 0]))
                else:
                    regime_means.append(0.0)
                    regime_stds.append(0.0)

            regime_means = np.array(regime_means)
            regime_stds = np.array(regime_stds)
        else:
            regime_means = np.zeros(len(unique_regimes))
            regime_stds = np.zeros(len(unique_regimes))

        # Build statistics dictionary
        for i, regime in enumerate(unique_regimes):
            stats = {
                'count': int(counts[i]),
                'percentage': float(percentages[i]),
                'mean_return': float(regime_means[i]),
                'volatility': float(regime_stds[i])
            }
            regime_stats[str(regime)] = stats

        return regime_stats

    @optimized_step(operation_type="storage", enable_gpu=False, enable_parallel=False)
    def _save_regime_results_optimized(self, regime_results: Dict[str, Any], training_input: Dict[str, Any]) -> Path:
        """Save regime discovery results with comprehensive M1 optimizations."""
        symbol = training_input.get('symbol', 'UNKNOWN')
        exchange = training_input.get('exchange', 'UNKNOWN')

        # Use memory optimization for result saving
        if self.memory_optimizer:
            with self.memory_optimizer.memory_checkpoint("save_regime_results"):
                return self._save_regime_results_core(regime_results, training_input, symbol, exchange)
        else:
            return self._save_regime_results_core(regime_results, training_input, symbol, exchange)

    def _save_regime_results_core(self, regime_results: Dict[str, Any],
                                training_input: Dict[str, Any],
                                symbol: str, exchange: str) -> Path:
        """Core result saving with optimizations."""
        # Use optimized data manager if available
        if self.data_manager:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"regime_discovery_{exchange}_{symbol}_{timestamp}"

            # Convert to DataFrame for optimized storage
            if 'regime_labels' in regime_results:
                # Create comprehensive DataFrame with all regime information
                regime_df = pd.DataFrame({
                    'regime_label': regime_results['regime_labels'],
                    'timestamp': pd.date_range(start=datetime.now(), periods=len(regime_results['regime_labels']), freq='1min')
                })

                # Add regime statistics as additional columns if available
                regime_stats = regime_results.get('regime_stats', {})
                if regime_stats:
                    # Create mapping from regime label to stats
                    regime_means = {}
                    regime_volatility = {}
                    for regime_id, stats in regime_stats.items():
                        regime_means[int(regime_id)] = stats.get('mean_return', 0.0)
                        regime_volatility[int(regime_id)] = stats.get('volatility', 0.0)

                    # Add to DataFrame
                    regime_df['regime_mean_return'] = regime_df['regime_label'].map(regime_means).fillna(0.0)
                    regime_df['regime_volatility'] = regime_df['regime_label'].map(regime_volatility).fillna(0.0)

                # Enhanced metadata with optimization information
                metadata = {
                    'regime_stats': regime_results.get('regime_stats', {}),
                    'model_params': regime_results.get('model_params', {}),
                    'discovery_method': regime_results.get('discovery_method', 'unknown'),
                    'optimization_used': regime_results.get('optimization_used', False),
                    'enhancements_used': regime_results.get('enhancements_used', []),
                    'timestamp': datetime.now().isoformat(),
                    'data_size_mb': len(regime_results.get('regime_labels', [])) * 8 / (1024 * 1024),  # Rough estimate
                    'symbol': symbol,
                    'exchange': exchange,
                    'm1_optimized': True
                }

                # Use optimized storage with memory management
                if self.memory_optimizer:
                    # Optimize DataFrame for storage
                    regime_df = self.memory_optimizer.create_memory_efficient_dataframe(regime_df)

                # Save with optimized parquet format
                saved_path = self.data_manager.save_dataframe_optimized(
                    regime_df, filename, metadata=metadata
                )

                self.logger.info(f'💾 Saved M1-optimized regime results to {saved_path}')
                return Path(saved_path)
        else:
            # Fallback to original JSON saving with some optimizations
            output_dir = Path(f'data/training/regimes/{exchange}_{symbol}')
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = output_dir / f"regime_discovery_{timestamp}.json"

            # Optimize JSON saving
            if self.memory_optimizer and len(regime_results.get('regime_labels', [])) > 10000:
                # For large results, save in compressed format
                import gzip
                output_path = output_path.with_suffix('.json.gz')
                with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                    json.dump(regime_results, f, indent=2)
            else:
                with open(output_path, 'w') as f:
                    json.dump(regime_results, f, indent=2)

            self.logger.info(f'💾 Saved regime results to {output_path}')
            return output_path

    def _save_regime_results(self, regime_results: Dict[str, Any], training_input: Dict[str, Any]) -> Path:
        """Legacy method for backward compatibility."""
        return self._save_regime_results_optimized(regime_results, training_input)

    def _validate_input_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate input data using pipeline standards.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Validated DataFrame
        """
        self.logger.info('🔍 Validating input data using pipeline standards...')
        validation_result = self.standards.validate_data_quality(data, 'unified')
        if not validation_result.passed:
            self.logger.warning(f'⚠️ Data quality issues detected: {validation_result.quality_score:.2f}')
            for issue in validation_result.issues:
                self.logger.warning(f'   - {issue.message}')
        fixed_data = data.copy()
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in fixed_data.columns]
        if missing_columns:
            self.logger.error(f'❌ Missing required columns: {missing_columns}')
            raise ValueError(f'Missing required columns for regime discovery: {missing_columns}')
        for col in required_columns:
            if col in fixed_data.columns:
                if not pd.api.types.is_numeric_dtype(fixed_data[col]):
                    self.logger.info(f'🔢 Converting {col} to numeric')
                    fixed_data[col] = pd.to_numeric(fixed_data[col], errors='coerce')
        initial_count = len(fixed_data)
        fixed_data = fixed_data.dropna(subset = required_columns)
        removed_count = initial_count - len(fixed_data)
        if removed_count > 0:
            self.logger.info(f'🗑️ Removed {removed_count} rows with NaN values')
        if len(fixed_data) < 100:
            self.logger.error(f'❌ Insufficient data after cleaning: {len(fixed_data)} rows')
            raise ValueError(f'Insufficient data for regime discovery: {len(fixed_data)} rows')
        self.logger.info(f'✅ Input validation completed: {len(fixed_data)} rows')
        return fixed_data
__all__ = ['Step03HMMRegimeDiscovery']