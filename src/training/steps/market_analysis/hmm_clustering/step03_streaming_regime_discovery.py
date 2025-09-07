from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

from typing import Dict
import pandas as pd
from typing import Any
import numpy as np

'Streaming Regime Discovery with Memory Optimization.\n\nThis module implements chunked processing and memory-efficient regime discovery\nto handle large datasets without loading everything into memory.\n'
import gc
from collections import deque
import psutil
import warnings
import datetime
import logging

warnings.filterwarnings('ignore')

class StreamingRegimeDiscovery:
    """Memory-efficient streaming regime discovery."""
    @log_important_calls

    def __init__(self, config: Dict[str, Any]=None) -> None:
        self.config = config or {}
        self.chunk_size = self.config.get('chunk_size', 10000)
        self.max_memory_usage = self.config.get('max_memory_usage', 0.8)
        self.overlap_size = self.config.get('overlap_size', 1000)
        self.buffer_size = self.config.get('buffer_size', 5000)
        self.regime_buffer = deque(maxlen = self.buffer_size)
        self.feature_buffer = deque(maxlen = self.buffer_size)
        self.regime_model = None
        self.is_model_trained = False
        self.update_frequency = self.config.get('update_frequency', 10000)
        self.min_samples_for_training = self.config.get('min_samples_for_training', 5000)
        self.memory_usage_history = []
        self.processing_times = []
    @log_all_calls

    def _check_memory_usage(self) -> bool:
        """Check if memory usage is within limits."""
        memory_percent = psutil.virtual_memory().percent / 100
        return memory_percent < self.max_memory_usage
    @log_all_calls

    def _force_garbage_collection(self) -> None:
        """Force garbage collection to free memory."""
        gc.collect()
    @log_all_calls

    def _get_optimal_chunk_size(self, data_size: int) -> int:
        """Dynamically determine optimal chunk size based on available memory."""
        available_memory = psutil.virtual_memory().available
        estimated_memory_per_sample = 1024
        optimal_chunk_size = int(available_memory * 0.5 / estimated_memory_per_sample)
        optimal_chunk_size = max(1000, min(optimal_chunk_size, self.chunk_size))
        return optimal_chunk_size

    def process_data_stream(self, data_iterator: Iterator[pd.DataFrame]) -> Generator[Dict[str, Any], None, None]:
        """Process data stream in chunks with memory optimization."""
        chunk_count = 0
        total_samples_processed = 0
        for chunk in data_iterator:
            chunk_count += 1
            if not self._check_memory_usage():
                self._force_garbage_collection()
                if not self._check_memory_usage():
                    raise MemoryError('Memory usage too high, cannot process more data')
            start_time = pd.Timestamp.now()
            chunk_result = self._process_chunk(chunk, chunk_count)
            processing_time = (pd.Timestamp.now() - start_time).total_seconds()
            self._update_buffers(chunk_result)
            if self._should_update_model(total_samples_processed):
                self._update_regime_model()
            total_samples_processed += len(chunk)
            self.processing_times.append(processing_time)
            yield {'chunk_id': chunk_count, 'regimes': chunk_result['regimes'], 'features': chunk_result['features'], 'processing_time': processing_time, 'memory_usage': psutil.virtual_memory().percent, 'total_samples': total_samples_processed}
            del chunk_result
            self._force_garbage_collection()
    @log_all_calls

    def _process_chunk(self, chunk: pd.DataFrame, chunk_id: int) -> Dict[str, Any]:
        """Process a single chunk of data."""
        features = self._extract_chunk_features(chunk)
        if self.is_model_trained and self.regime_model is not None:
            regimes = self._predict_regimes_chunk(features)
        else:
            regimes = self._heuristic_regime_assignment(chunk)
        return {'regimes': regimes, 'features': features, 'chunk_id': chunk_id, 'timestamp': chunk.index[-1] if hasattr(chunk.index, '__len__') else None}
    @log_all_calls

    def _extract_chunk_features(self, chunk: pd.DataFrame) -> np.ndarray:
        """Extract features for a chunk of data."""
        features = []
        if 'close' in chunk.columns:
            returns = chunk['close'].pct_change().dropna()
            features.extend([returns.mean(), returns.std(), returns.skew(), returns.kurtosis()])
        if 'volume' in chunk.columns:
            volume = chunk['volume']
            features.extend([volume.mean(), volume.std(), (volume > volume.rolling(20).mean()).sum() / len(volume)])
        if 'high' in chunk.columns and 'low' in chunk.columns:
            volatility = (chunk['high'] - chunk['low']) / chunk['close']
            features.extend([volatility.mean(), volatility.std()])
        return np.array(features)
    @log_all_calls

    def _predict_regimes_chunk(self, features: np.ndarray) -> np.ndarray:
        """Predict regimes for a chunk using the trained model."""
        if self.regime_model is None:
            return np.zeros(len(features))
        try:
            if features.ndim == 1:
                features = features.reshape(1, -1)
            regimes = self.regime_model.predict(features)
            return regimes
        except Exception as e:
            print(f'Error predicting regimes: {e}')
            return np.zeros(len(features))
    @log_all_calls

    def _heuristic_regime_assignment(self, chunk: pd.DataFrame) -> np.ndarray:
        """Simple heuristic regime assignment for initial chunks."""
        if 'close' not in chunk.columns:
            return np.zeros(len(chunk))
        returns = chunk['close'].pct_change().dropna()
        if len(returns) == 0:
            return np.zeros(len(chunk))
        volatility = returns.rolling(20).std()
        regime_threshold = volatility.quantile(0.5)
        regimes = np.where(volatility > regime_threshold, 1, 0)
        if len(regimes) < len(chunk):
            regimes = np.pad(regimes, (0, len(chunk) - len(regimes)), 'constant')
        return regimes
    @log_all_calls

    def _update_buffers(self, chunk_result: Dict[str, Any]) -> None:
        """Update internal buffers with chunk results."""
        self.regime_buffer.extend(chunk_result['regimes'])
        self.feature_buffer.extend(chunk_result['features'])
    @log_all_calls

    def _should_update_model(self, total_samples: int) -> bool:
        """Determine if the regime model should be updated."""
        return total_samples >= self.min_samples_for_training and total_samples % self.update_frequency == 0
    @log_all_calls

    def _update_regime_model(self) -> None:
        """Update the regime model using buffered data."""
        if len(self.feature_buffer) < self.min_samples_for_training:
            return
        try:
            features_array = np.array(list(self.feature_buffer))
            regimes_array = np.array(list(self.regime_buffer))
            from sklearn.cluster import KMeans
            n_regimes = len(np.unique(regimes_array))
            n_regimes = max(2, min(n_regimes, 5))
            self.regime_model = KMeans(n_clusters = n_regimes, random_state = 42)
            self.regime_model.fit(features_array)
            self.is_model_trained = True
            print(f'Updated regime model with {len(features_array)} samples, {n_regimes} regimes')
        except Exception as e:
            print(f'Error updating regime model: {e}')

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        memory = psutil.virtual_memory()
        return {'total_memory_gb': memory.total / 1024 ** 3, 'available_memory_gb': memory.available / 1024 ** 3, 'used_memory_percent': memory.percent, 'chunk_size': self.chunk_size, 'buffer_size': len(self.regime_buffer), 'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0}

    def create_data_iterator(self, data: pd.DataFrame, chunk_size: Optional[int]=None) -> Iterator[pd.DataFrame]:
        """Create an iterator for chunked data processing."""
        if chunk_size is None:
            chunk_size = self._get_optimal_chunk_size(len(data))
        for i in range(0, len(data), chunk_size):
            end_idx = min(i + chunk_size, len(data))
            chunk = data.iloc[i:end_idx].copy()
            if i > 0 and self.overlap_size > 0:
                overlap_start = max(0, i - self.overlap_size)
                overlap_data = data.iloc[overlap_start:i]
                chunk = pd.concat([overlap_data, chunk], ignore_index = False)
            yield chunk

class MemoryOptimizedRegimeDiscovery:
    """Main interface for memory-optimized regime discovery."""
    @log_important_calls

    def __init__(self, config: Dict[str, Any]=None) -> None:
        self.config = config or {}
        self.streaming_processor = StreamingRegimeDiscovery(config)

    async def discover_regimes_streaming(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Discover regimes using streaming processing."""
        print('🚀 Starting memory-optimized regime discovery...')
        data_iterator = self.streaming_processor.create_data_iterator(data)
        results = []
        regime_sequence = []
        for chunk_result in self.streaming_processor.process_data_stream(data_iterator):
            results.append(chunk_result)
            regime_sequence.extend(chunk_result['regimes'])
            if chunk_result['chunk_id'] % 10 == 0:
                print(f"Processed chunk {chunk_result['chunk_id']}, Memory: {chunk_result['memory_usage']:.1f}%, Time: {chunk_result['processing_time']:.2f}s")
        final_regimes = np.array(regime_sequence[:len(data)])
        return {'regimes': final_regimes, 'chunk_results': results, 'memory_stats': self.streaming_processor.get_memory_stats(), 'total_chunks': len(results), 'avg_processing_time': np.mean([r['processing_time'] for r in results])}
from typing import Dict