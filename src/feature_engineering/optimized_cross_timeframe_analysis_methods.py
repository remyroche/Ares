"""
Optimized Cross Timeframe Analysis Methods

This module contains the optimized implementation methods for cross timeframe analysis.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

from src.utils.logger import system_logger
from src.utils.math_validation import (
    validate_finite, validate_positive, validate_range,
    safe_divide, safe_log, safe_sqrt, safe_power,
    MathValidationError
)

logger = system_logger.getChild('OptimizedCrossTimeframeMethods')

class OptimizedCrossTimeframeMethods:
    """Optimized methods for cross timeframe analysis."""
    
    def __init__(self, parent_analyzer):
        """Initialize with reference to parent analyzer."""
        self.analyzer = parent_analyzer
        self.config = parent_analyzer.config
        self.logger = logger.getChild('OptimizedMethods')
        
        # Get optimizers from parent
        self.memory_optimizer = parent_analyzer.memory_optimizer
        self.cpu_optimizer = parent_analyzer.cpu_optimizer
        self.gpu_manager = parent_analyzer.gpu_manager
        self.feature_selector = parent_analyzer.feature_selector
        self.data_validator = parent_analyzer.data_validator
        self.data_cleaner = parent_analyzer.data_cleaner
        self.data_transformer = parent_analyzer.data_transformer
    
    async def _load_and_validate_data(
        self,
        data_dir: str,
        symbol: str,
        exchange: str,
        timeframes: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """Load and validate data with optimizations."""
        self.logger.info(f"📊 Loading and validating data for {len(timeframes)} timeframes")
        
        timeframe_data = {}
        
        # Use optimized thread pool if available
        if self.cpu_optimizer:
            executor = self.cpu_optimizer.create_optimized_thread_pool(max_workers=self.config.max_workers)
        else:
            executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        try:
            # Load data in parallel
            load_tasks = []
            for timeframe in timeframes:
                task = self._load_single_timeframe_data(data_dir, symbol, exchange, timeframe)
                load_tasks.append(task)
            
            # Execute in parallel
            results = await asyncio.gather(*load_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"❌ Failed to load data for {timeframes[i]}: {result}")
                    continue
                
                if result is not None:
                    timeframe_data[timeframes[i]] = result
            
            # Validate data quality
            if self.data_validator:
                validated_data = {}
                for timeframe, data in timeframe_data.items():
                    try:
                        # Validate data quality
                        validation_result = self.data_validator.validate_dataframe(data)
                        if validation_result.is_valid:
                            validated_data[timeframe] = data
                        else:
                            self.logger.warning(f"⚠️ Data quality issues for {timeframe}: {validation_result.issues}")
                            # Clean data if possible
                            if self.data_cleaner:
                                cleaned_data = self.data_cleaner.clean_dataframe(data)
                                validated_data[timeframe] = cleaned_data
                            else:
                                validated_data[timeframe] = data
                    except Exception as e:
                        self.logger.warning(f"⚠️ Data validation failed for {timeframe}: {e}")
                        validated_data[timeframe] = data
                
                timeframe_data = validated_data
            
            self.logger.info(f"📊 Successfully loaded and validated data for {len(timeframe_data)} timeframes")
            return timeframe_data
            
        finally:
            executor.shutdown(wait=False)
    
    async def _load_single_timeframe_data(
        self,
        data_dir: str,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Load data for a single timeframe with optimizations."""
        try:
            # Construct file path
            from pathlib import Path
            file_path = Path(data_dir) / f"aggtrades_{exchange}_{symbol}_consolidated.parquet"
            
            if not file_path.exists():
                self.logger.warning(f"⚠️ Data file not found for {timeframe}: {file_path}")
                return None
            
            # Load data using standardized handler
            data = standardized_parquet_handler.read_parquet_standardized(file_path)
            
            # Basic validation
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                self.logger.warning(f"⚠️ Missing columns for {timeframe}: {missing_columns}")
                return None
            
            # Sort by timestamp if available
            if 'timestamp' in data.columns:
                data = data.sort_values('timestamp').reset_index(drop=True)
            
            # Resample to target timeframe if needed
            if timeframe != self.config.base_timeframe:
                data = await self._resample_data_optimized(data, timeframe)
            
            # Memory optimization
            if self.memory_optimizer:
                # Convert to memory-efficient data types
                data = self._optimize_dataframe_memory(data)
            
            self.logger.info(f"📊 Loaded {len(data)} data points for {timeframe}")
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load data for {timeframe}: {e}")
            return None
    
    async def _resample_data_optimized(self, data: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """Resample data to target timeframe with optimizations."""
        try:
            # Convert timeframe to pandas frequency
            timeframe_map = {
                '1m': '1T',
                '5m': '5T',
                '15m': '15T',
                '30m': '30T',
                '1h': '1H',
                '4h': '4H',
                '1d': '1D'
            }
            
            if target_timeframe not in timeframe_map:
                self.logger.warning(f"⚠️ Unknown timeframe: {target_timeframe}")
                return data
            
            frequency = timeframe_map[target_timeframe]
            
            # Ensure timestamp column exists
            if 'timestamp' not in data.columns:
                # Create timestamp index
                data = data.copy()
                data['timestamp'] = pd.date_range(start='2023-01-01', periods=len(data), freq='1T')
            
            # Set timestamp as index
            data_indexed = data.set_index('timestamp')
            
            # Resample OHLCV data with optimizations
            if self.memory_optimizer and len(data) > 100000:
                # Use chunked resampling for large datasets
                resampled = await self._chunked_resample(data_indexed, frequency)
            else:
                # Direct resampling
                resampled = data_indexed.resample(frequency).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            
            # Reset index
            resampled = resampled.reset_index()
            
            return resampled
            
        except Exception as e:
            self.logger.error(f"❌ Data resampling failed: {e}")
            return data
    
    async def _chunked_resample(self, data: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """Perform chunked resampling for large datasets."""
        try:
            chunk_size = 50000  # Process in chunks of 50k rows
            
            chunks = []
            for i in range(0, len(data), chunk_size):
                chunk = data.iloc[i:i+chunk_size]
                resampled_chunk = chunk.resample(frequency).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                chunks.append(resampled_chunk)
                
                # Memory cleanup
                if self.memory_optimizer:
                    self.memory_optimizer.optimize_memory()
            
            # Combine chunks
            if chunks:
                result = pd.concat(chunks, ignore_index=False)
                # Final resampling to handle chunk boundaries
                result = result.resample(frequency).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                return result
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"❌ Chunked resampling failed: {e}")
            return data.resample(frequency).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
    
    def _optimize_dataframe_memory(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        try:
            optimized_data = data.copy()
            
            # Convert float64 to float32 where possible
            for col in optimized_data.select_dtypes(include=['float64']).columns:
                if optimized_data[col].min() >= np.finfo(np.float32).min and \
                   optimized_data[col].max() <= np.finfo(np.float32).max:
                    optimized_data[col] = optimized_data[col].astype(np.float32)
            
            # Convert int64 to int32 where possible
            for col in optimized_data.select_dtypes(include=['int64']).columns:
                if optimized_data[col].min() >= np.iinfo(np.int32).min and \
                   optimized_data[col].max() <= np.iinfo(np.int32).max:
                    optimized_data[col] = optimized_data[col].astype(np.int32)
            
            # Convert object columns to category if beneficial
            for col in optimized_data.select_dtypes(include=['object']).columns:
                if optimized_data[col].nunique() / len(optimized_data) < 0.5:
                    optimized_data[col] = optimized_data[col].astype('category')
            
            return optimized_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Memory optimization failed: {e}")
            return data
    
    async def _align_timeframes_optimized(
        self,
        timeframe_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Align timeframes with optimizations."""
        self.logger.info("🔄 Aligning timeframes with optimizations")
        
        try:
            aligned_data = {}
            
            # Use base timeframe as reference
            base_timeframe = self.config.base_timeframe
            if base_timeframe not in timeframe_data:
                base_timeframe = list(timeframe_data.keys())[0]
            
            base_data = timeframe_data[base_timeframe]
            
            # Create common time index
            if 'timestamp' in base_data.columns:
                common_index = base_data['timestamp']
            else:
                # Create synthetic time index
                common_index = pd.date_range(start='2023-01-01', periods=len(base_data), freq='1T')
            
            # Align each timeframe to common index with parallel processing
            if self.cpu_optimizer:
                executor = self.cpu_optimizer.create_optimized_thread_pool(max_workers=self.config.max_workers)
            else:
                executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
            
            try:
                # Process timeframes in parallel
                alignment_tasks = []
                for timeframe, data in timeframe_data.items():
                    if timeframe == base_timeframe:
                        aligned_data[timeframe] = data.copy()
                        continue
                    
                    task = self._align_single_timeframe(data, common_index, timeframe)
                    alignment_tasks.append((timeframe, task))
                
                # Execute alignment tasks
                for timeframe, task in alignment_tasks:
                    try:
                        aligned_df = await task
                        aligned_data[timeframe] = aligned_df
                    except Exception as e:
                        self.logger.error(f"❌ Failed to align {timeframe}: {e}")
                        aligned_data[timeframe] = timeframe_data[timeframe]
                
            finally:
                executor.shutdown(wait=False)
            
            self.logger.info("✅ Timeframes aligned with optimizations")
            return aligned_data
            
        except Exception as e:
            self.logger.error(f"❌ Timeframe alignment failed: {e}")
            return timeframe_data
    
    async def _align_single_timeframe(
        self,
        data: pd.DataFrame,
        common_index: pd.Index,
        timeframe: str
    ) -> pd.DataFrame:
        """Align a single timeframe to common index."""
        try:
            # Forward fill to align with common index
            aligned_df = pd.DataFrame(index=common_index)
            
            # Interpolate or forward fill data
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in data.columns:
                    if 'timestamp' in data.columns:
                        series = pd.Series(data[col].values, index=data['timestamp'])
                    else:
                        series = pd.Series(data[col].values)
                    
                    # Forward fill to align with common index
                    aligned_series = series.reindex(common_index, method='ffill')
                    aligned_df[col] = aligned_series
            
            return aligned_df.dropna()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to align {timeframe}: {e}")
            return data
    
    async def _engineer_features_optimized(
        self,
        aligned_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Engineer cross timeframe features with optimizations."""
        self.logger.info("🔧 Engineering cross timeframe features with optimizations")
        
        try:
            features = pd.DataFrame()
            timeframes = list(aligned_data.keys())
            
            # Base timeframe features
            base_timeframe = timeframes[0]
            base_data = aligned_data[base_timeframe]
            
            # Create base features with GPU acceleration if available
            if self.gpu_manager:
                base_features = await self._create_base_features_gpu_accelerated(base_data)
            else:
                base_features = await self._create_base_features_cpu(base_data)
            
            features = pd.concat([features, base_features], axis=1)
            
            # Cross timeframe interaction features with parallel processing
            interaction_features = await self._create_interaction_features_parallel(aligned_data)
            features = pd.concat([features, interaction_features], axis=1)
            
            # Multi-timeframe aggregation features
            aggregation_features = await self._create_aggregation_features_parallel(aligned_data)
            features = pd.concat([features, aggregation_features], axis=1)
            
            # High leverage specific features
            specialized_features = await self._create_specialized_features_parallel(aligned_data)
            features = pd.concat([features, specialized_features], axis=1)
            
            # Remove rows with NaN values
            features = features.dropna()
            
            # Memory optimization
            if self.memory_optimizer:
                features = self._optimize_dataframe_memory(features)
            
            self.logger.info(f"🔧 Engineered {len(features.columns)} cross timeframe features with optimizations")
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Cross timeframe feature engineering failed: {e}")
            raise
    
    async def _create_base_features_gpu_accelerated(self, base_data: pd.DataFrame) -> pd.DataFrame:
        """Create base features with GPU acceleration."""
        try:
            features = pd.DataFrame()
            
            # Convert to numpy for GPU processing
            close_values = base_data['close'].values.astype(np.float32)
            volume_values = base_data['volume'].values.astype(np.float32)
            
            # GPU-accelerated calculations
            if self.gpu_manager:
                # Use GPU for calculations
                returns_gpu = self.gpu_manager.optimize_tensor_operations(
                    np.diff(close_values, prepend=close_values[0])
                )
                volatility_gpu = self.gpu_manager.optimize_tensor_operations(
                    pd.Series(returns_gpu).rolling(20).std().values
                )
            else:
                # Fallback to CPU
                returns_gpu = np.diff(close_values, prepend=close_values[0])
                volatility_gpu = pd.Series(returns_gpu).rolling(20).std().values
            
            # Create features DataFrame
            features['base_close'] = base_data['close']
            features['base_volume'] = base_data['volume']
            features['base_returns'] = pd.Series(returns_gpu, index=base_data.index)
            features['base_volatility'] = pd.Series(volatility_gpu, index=base_data.index)
            
            return features
            
        except Exception as e:
            self.logger.warning(f"⚠️ GPU-accelerated base features failed: {e}")
            return await self._create_base_features_cpu(base_data)
    
    async def _create_base_features_cpu(self, base_data: pd.DataFrame) -> pd.DataFrame:
        """Create base features with CPU processing."""
        features = pd.DataFrame()
        
        features['base_close'] = base_data['close']
        features['base_volume'] = base_data['volume']
        features['base_returns'] = base_data['close'].pct_change()
        features['base_volatility'] = features['base_returns'].rolling(20).std()
        
        return features
    
    async def _create_interaction_features_parallel(
        self,
        aligned_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Create interaction features with parallel processing."""
        try:
            timeframes = list(aligned_data.keys())
            features = pd.DataFrame()
            
            # Use thread pool for parallel processing
            if self.cpu_optimizer:
                executor = self.cpu_optimizer.create_optimized_thread_pool(max_workers=self.config.max_workers)
            else:
                executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
            
            try:
                # Create tasks for each interaction type
                tasks = []
                
                if 'correlation' in self.config.interaction_features:
                    tasks.append(self._create_correlation_features(aligned_data))
                
                if 'momentum' in self.config.interaction_features:
                    tasks.append(self._create_momentum_features(aligned_data))
                
                if 'volatility' in self.config.interaction_features:
                    tasks.append(self._create_volatility_features(aligned_data))
                
                if 'volume' in self.config.interaction_features:
                    tasks.append(self._create_volume_features(aligned_data))
                
                # Execute tasks in parallel
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Combine results
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.warning(f"⚠️ Interaction feature creation failed: {result}")
                        continue
                    
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        features = pd.concat([features, result], axis=1)
                
            finally:
                executor.shutdown(wait=False)
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Parallel interaction feature creation failed: {e}")
            return pd.DataFrame()
    
    async def _create_correlation_features(
        self,
        aligned_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Create correlation features between timeframes."""
        try:
            features = pd.DataFrame()
            timeframes = list(aligned_data.keys())
            
            for i, tf1 in enumerate(timeframes):
                for j, tf2 in enumerate(timeframes[i+1:], i+1):
                    data1 = aligned_data[tf1]
                    data2 = aligned_data[tf2]
                    
                    # Correlation features
                    corr_5 = data1['close'].rolling(5).corr(data2['close'])
                    corr_20 = data1['close'].rolling(20).corr(data2['close'])
                    
                    features[f'corr_{tf1}_{tf2}_5'] = corr_5
                    features[f'corr_{tf1}_{tf2}_20'] = corr_20
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Correlation features creation failed: {e}")
            return pd.DataFrame()
    
    async def _create_momentum_features(
        self,
        aligned_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Create momentum features between timeframes."""
        try:
            features = pd.DataFrame()
            timeframes = list(aligned_data.keys())
            
            for i, tf1 in enumerate(timeframes):
                for j, tf2 in enumerate(timeframes[i+1:], i+1):
                    data1 = aligned_data[tf1]
                    data2 = aligned_data[tf2]
                    
                    # Momentum features
                    mom1 = data1['close'].pct_change(5)
                    mom2 = data2['close'].pct_change(5)
                    features[f'mom_diff_{tf1}_{tf2}'] = mom1 - mom2
                    features[f'mom_ratio_{tf1}_{tf2}'] = mom1 / (mom2 + 1e-10)
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Momentum features creation failed: {e}")
            return pd.DataFrame()
    
    async def _create_volatility_features(
        self,
        aligned_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Create volatility features between timeframes."""
        try:
            features = pd.DataFrame()
            timeframes = list(aligned_data.keys())
            
            for i, tf1 in enumerate(timeframes):
                for j, tf2 in enumerate(timeframes[i+1:], i+1):
                    data1 = aligned_data[tf1]
                    data2 = aligned_data[tf2]
                    
                    # Volatility features
                    vol1 = data1['close'].pct_change().rolling(20).std()
                    vol2 = data2['close'].pct_change().rolling(20).std()
                    features[f'vol_ratio_{tf1}_{tf2}'] = vol1 / (vol2 + 1e-10)
                    features[f'vol_diff_{tf1}_{tf2}'] = vol1 - vol2
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Volatility features creation failed: {e}")
            return pd.DataFrame()
    
    async def _create_volume_features(
        self,
        aligned_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Create volume features between timeframes."""
        try:
            features = pd.DataFrame()
            timeframes = list(aligned_data.keys())
            
            for i, tf1 in enumerate(timeframes):
                for j, tf2 in enumerate(timeframes[i+1:], i+1):
                    data1 = aligned_data[tf1]
                    data2 = aligned_data[tf2]
                    
                    # Volume features
                    vol_ratio = data1['volume'] / (data2['volume'] + 1e-10)
                    features[f'volume_ratio_{tf1}_{tf2}'] = vol_ratio
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Volume features creation failed: {e}")
            return pd.DataFrame()
    
    async def _create_aggregation_features_parallel(
        self,
        aligned_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Create aggregation features with parallel processing."""
        try:
            features = pd.DataFrame()
            timeframes = list(aligned_data.keys())
            
            # Use thread pool for parallel processing
            if self.cpu_optimizer:
                executor = self.cpu_optimizer.create_optimized_thread_pool(max_workers=self.config.max_workers)
            else:
                executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
            
            try:
                # Create tasks for each timeframe
                tasks = []
                for timeframe in timeframes:
                    task = self._create_timeframe_aggregation_features(aligned_data[timeframe], timeframe)
                    tasks.append(task)
                
                # Execute tasks in parallel
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Combine results
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.warning(f"⚠️ Aggregation feature creation failed: {result}")
                        continue
                    
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        features = pd.concat([features, result], axis=1)
                
            finally:
                executor.shutdown(wait=False)
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Parallel aggregation feature creation failed: {e}")
            return pd.DataFrame()
    
    async def _create_timeframe_aggregation_features(
        self,
        data: pd.DataFrame,
        timeframe: str
    ) -> pd.DataFrame:
        """Create aggregation features for a single timeframe."""
        try:
            features = pd.DataFrame()
            
            # Price position across timeframes
            for period in self.config.lookback_periods:
                high_period = data['high'].rolling(period).max()
                low_period = data['low'].rolling(period).min()
                price_position = (data['close'] - low_period) / (high_period - low_period + 1e-10)
                features[f'price_pos_{timeframe}_{period}'] = price_position
            
            # Volume profile
            volume_ma = data['volume'].rolling(20).mean()
            volume_ratio = data['volume'] / (volume_ma + 1e-10)
            features[f'volume_profile_{timeframe}'] = volume_ratio
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Timeframe aggregation features creation failed for {timeframe}: {e}")
            return pd.DataFrame()
    
    async def _create_specialized_features_parallel(
        self,
        aligned_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Create specialized features with parallel processing."""
        try:
            features = pd.DataFrame()
            
            # Use thread pool for parallel processing
            if self.cpu_optimizer:
                executor = self.cpu_optimizer.create_optimized_thread_pool(max_workers=self.config.max_workers)
            else:
                executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
            
            try:
                # Create tasks for each specialized feature type
                tasks = []
                
                if 'microstructure' in self.config.interaction_features:
                    tasks.append(self._create_microstructure_features(aligned_data))
                
                if 'order_flow' in self.config.interaction_features:
                    tasks.append(self._create_order_flow_features(aligned_data))
                
                if 'momentum_divergence' in self.config.interaction_features:
                    tasks.append(self._create_momentum_divergence_features(aligned_data))
                
                if 'volatility_spillover' in self.config.interaction_features:
                    tasks.append(self._create_volatility_spillover_features(aligned_data))
                
                # Execute tasks in parallel
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Combine results
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.warning(f"⚠️ Specialized feature creation failed: {result}")
                        continue
                    
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        features = pd.concat([features, result], axis=1)
                
            finally:
                executor.shutdown(wait=False)
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Parallel specialized feature creation failed: {e}")
            return pd.DataFrame()
    
    async def _create_microstructure_features(
        self,
        aligned_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Create microstructure features for high leverage trading."""
        try:
            features = pd.DataFrame()
            timeframes = list(aligned_data.keys())
            
            for timeframe in timeframes:
                data = aligned_data[timeframe]
                
                # Bid-ask spread proxy (using high-low as proxy)
                features[f'spread_proxy_{timeframe}'] = (data['high'] - data['low']) / data['close']
                
                # Price impact proxy (volume vs price movement)
                price_change = data['close'].pct_change().abs()
                volume_normalized = data['volume'] / data['volume'].rolling(20).mean()
                features[f'price_impact_{timeframe}'] = price_change / (volume_normalized + 1e-10)
                
                # Tick-by-tick volatility (using high-low range)
                features[f'tick_volatility_{timeframe}'] = (data['high'] - data['low']) / data['close']
                
                # Order flow imbalance proxy (close position within bar)
                features[f'order_flow_imbalance_{timeframe}'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-10)
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Microstructure features creation failed: {e}")
            return pd.DataFrame()
    
    async def _create_order_flow_features(
        self,
        aligned_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Create order flow features for high leverage trading."""
        try:
            features = pd.DataFrame()
            timeframes = list(aligned_data.keys())
            
            for timeframe in timeframes:
                data = aligned_data[timeframe]
                
                # Volume-weighted average price (VWAP) deviation
                vwap = (data['high'] + data['low'] + data['close']) / 3
                vwap_volume = (vwap * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
                features[f'vwap_deviation_{timeframe}'] = (data['close'] - vwap_volume) / vwap_volume
                
                # Volume momentum
                volume_momentum = data['volume'].pct_change(5)
                features[f'volume_momentum_{timeframe}'] = volume_momentum
                
                # Price-volume relationship
                price_momentum = data['close'].pct_change(5)
                features[f'price_volume_correlation_{timeframe}'] = price_momentum.rolling(10).corr(volume_momentum)
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Order flow features creation failed: {e}")
            return pd.DataFrame()
    
    async def _create_momentum_divergence_features(
        self,
        aligned_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Create momentum divergence features between timeframes."""
        try:
            features = pd.DataFrame()
            timeframes = list(aligned_data.keys())
            
            # Calculate momentum for each timeframe
            momentum_data = {}
            for timeframe in timeframes:
                data = aligned_data[timeframe]
                momentum_data[timeframe] = {
                    'momentum_5': data['close'].pct_change(5),
                    'momentum_10': data['close'].pct_change(10),
                    'momentum_20': data['close'].pct_change(20)
                }
            
            # Calculate divergences between timeframes
            for i, tf1 in enumerate(timeframes):
                for j, tf2 in enumerate(timeframes[i+1:], i+1):
                    for period in [5, 10, 20]:
                        mom1 = momentum_data[tf1][f'momentum_{period}']
                        mom2 = momentum_data[tf2][f'momentum_{period}']
                        
                        # Momentum divergence
                        features[f'momentum_divergence_{tf1}_{tf2}_{period}'] = mom1 - mom2
                        
                        # Momentum ratio
                        features[f'momentum_ratio_{tf1}_{tf2}_{period}'] = mom1 / (mom2 + 1e-10)
                        
                        # Momentum correlation
                        features[f'momentum_correlation_{tf1}_{tf2}_{period}'] = mom1.rolling(20).corr(mom2)
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Momentum divergence features creation failed: {e}")
            return pd.DataFrame()
    
    async def _create_volatility_spillover_features(
        self,
        aligned_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Create volatility spillover features between timeframes."""
        try:
            features = pd.DataFrame()
            timeframes = list(aligned_data.keys())
            
            # Calculate volatility for each timeframe
            volatility_data = {}
            for timeframe in timeframes:
                data = aligned_data[timeframe]
                returns = data['close'].pct_change()
                volatility_data[timeframe] = {
                    'volatility_5': returns.rolling(5).std(),
                    'volatility_10': returns.rolling(10).std(),
                    'volatility_20': returns.rolling(20).std()
                }
            
            # Calculate volatility spillovers
            for i, tf1 in enumerate(timeframes):
                for j, tf2 in enumerate(timeframes[i+1:], i+1):
                    for period in [5, 10, 20]:
                        vol1 = volatility_data[tf1][f'volatility_{period}']
                        vol2 = volatility_data[tf2][f'volatility_{period}']
                        
                        # Volatility spillover (lagged correlation)
                        vol1_lagged = vol1.shift(1)
                        features[f'volatility_spillover_{tf1}_{tf2}_{period}'] = vol1_lagged.rolling(20).corr(vol2)
                        
                        # Volatility ratio
                        features[f'volatility_ratio_{tf1}_{tf2}_{period}'] = vol1 / (vol2 + 1e-10)
                        
                        # Volatility difference
                        features[f'volatility_diff_{tf1}_{tf2}_{period}'] = vol1 - vol2
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Volatility spillover features creation failed: {e}")
            return pd.DataFrame()