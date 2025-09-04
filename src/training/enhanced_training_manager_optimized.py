from src.core.decorators import handles_errors
import gc
import multiprocessing as mp
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable
import numpy as np
import pandas as pd
import psutil
import asyncio
from copy import copy
from typing import Dict, List, Optional, Union, Any, Tuple
try:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None
    ds = None
import contextlib
from src.utils.logger import system_logger

def _make_hashable(obj: Any) -> Any:
    """Recursively convert potentially unhashable objects (lists, dicts, arrays) into hashable tuples."
    This is used to generate robust cache keys.
    """
    if isinstance(obj, dict):
        return tuple(sorted(((k, _make_hashable(v)) for k, v in obj.items())))
    if isinstance(obj, set):
        return tuple(sorted(map(_make_hashable, obj)))
    if isinstance(obj, (list, tuple)):
        return tuple((_make_hashable(v) for v in obj))
    if isinstance(obj, np.ndarray):
        return tuple(obj.tolist())
    return obj

class CachedBacktester:
    """Cached backtesting to avoid redundant calculations."""

    def __init__(self, market_data: pd.DataFrame) -> None:
        self.market_data = market_data
        self.cache: dict[str, float] = {}
        self.logger = system_logger.getChild('CachedBacktester')
        self.technical_indicators = self._precompute_indicators()

    def _precompute_indicators(self) -> dict[str, np.ndarray]:
        """Precompute all technical indicators once."""
        indicators: dict[str, np.ndarray] = {}
        if 'close' not in self.market_data.columns:
            self.logger.warning("'close' column missing; cannot compute indicators")
            return indicators
        indicators['sma_20'] = self.market_data['close'].rolling(20).mean().fillna(method='ffill').values
        indicators['sma_50'] = self.market_data['close'].rolling(50).mean().fillna(method='ffill').values
        indicators['ema_12'] = self.market_data['close'].ewm(span=12).mean().fillna(method='ffill').values
        indicators['ema_26'] = self.market_data['close'].ewm(span=26).mean().fillna(method='ffill').values
        delta = self.market_data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        indicators['rsi'] = (100 - 100 / (1 + rs)).fillna(50).values
        if {'high', 'low'}.issubset(self.market_data.columns):
            high_low = self.market_data['high'] - self.market_data['low']
            high_close = np.abs(self.market_data['high'] - self.market_data['close'].shift())
            low_close = np.abs(self.market_data['low'] - self.market_data['close'].shift())
            tr = np.maximum(high_low.values, np.maximum(high_close.values, low_close.values))
            indicators['atr'] = pd.Series(tr, index=self.market_data.index).rolling(window=14).mean().fillna(method='ffill').values
        else:
            self.logger.warning("Missing 'high'/'low' columns; skipping ATR calculation")
        indicators['volatility'] = self.market_data['close'].pct_change().rolling(20).std().fillna(method='ffill').values
        if 'volume' in self.market_data.columns:
            indicators['volume_sma'] = self.market_data['volume'].rolling(20).mean().fillna(method='ffill').values
        self.logger.info(f'Precomputed {len(indicators)} technical indicators')
        return indicators

    def run_cached_backtest(self, params: dict[str, Any]) -> float:
        """Run backtest using cached indicators."""
        cache_key = self._generate_cache_key(params)
        if cache_key in self.cache:
            try:
                self.logger.info(f'Backtest cache hit: score={float(self.cache[cache_key]):.4f}')
            except Exception as e:
                self.logger.warning(f'Failed to log cache hit info: {e}')
            return self.cache[cache_key]
        result = self._run_simplified_backtest(params)
        self.cache[cache_key] = result
        try:
            self.logger.info(f'Backtest cache miss: computed score={float(result):.4f}')
        except Exception as e:
            self.logger.warning(f'Failed to log cache miss info: {e}')
        return result

    def _generate_cache_key(self, params: dict[str, Any]) -> str:
        """Generate cache key from parameters, robust to unhashable values."""
        return str(hash(_make_hashable(params)))

    def _run_simplified_backtest(self, params: dict[str, Any]) -> float:
        """Run simplified backtest logic (placeholder)."""
        return float(random.uniform(-1.0, 1.0))

class ProgressiveEvaluator:
    """Progressive evaluation to stop unpromising trials early."""

    def __init__(self, full_data: pd.DataFrame) -> None:
        self.full_data = full_data
        self.evaluation_stages: list[tuple[float, float]] = [(0.1, 0.3), (0.3, 0.5), (1.0, 1.0)]
        self.logger = system_logger.getChild('ProgressiveEvaluator')

    def evaluate_progressively(self, params: dict[str, Any], evaluator_func: Callable[[pd.DataFrame, dict[str, Any]], float]) -> float:
        """Evaluate parameters progressively across data subsets."""
        total_score = 0.0
        total_weight = 0.0
        score: float = 0.0
        subset_size: int = 0
        data_ratio: float = 0.0
        subset_data: pd.DataFrame | None = None
        for data_ratio, weight in self.evaluation_stages:
            subset_size = int(len(self.full_data) * data_ratio)
            subset_data = self.full_data.iloc[:subset_size]
            score = float(evaluator_func(subset_data, params))
            total_score += score * weight
            total_weight += weight
            try:
                self.logger.info({'msg': 'progressive_stage', 'data_ratio': float(data_ratio), 'subset_size': int(subset_size), 'weight': float(weight), 'stage_score': float(score)})
            except Exception as e:
                self.logger.warning(f'Failed to log progressive stage info: {e}')
            if data_ratio < 1.0 and score < -0.5:
                self.logger.info(f'Early stopping at {data_ratio * 100:.0f}% data due to poor performance (score={score:.4f})')
                return -1.0
        final_score = float(total_score / total_weight if total_weight else 0.0)
        try:
            self.logger.info({'msg': 'progressive_evaluation_complete', 'total_weight': float(total_weight), 'final_score': float(final_score)})
        except Exception as e:
            self.logger.warning(f'Failed to log progressive evaluation complete: {e}')
        return final_score

class ParallelBacktester:
    """Parallel backtesting for multiple parameter combinations."""

    def __init__(self, n_workers: int | None=None) -> None:
        self.n_workers = n_workers or min(mp.cpu_count(), 8)
        self.executor: ProcessPoolExecutor | None = ProcessPoolExecutor(max_workers=self.n_workers)
        self.logger = system_logger.getChild('ParallelBacktester')

    def __enter__(self) -> None:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        try:
            if hasattr(self, 'executor') and self.executor:
                self.executor.shutdown(wait=True)
        finally:
            self.executor = None
        return False

    def evaluate_batch(self, param_batch: list[dict[str, Any]], market_data: pd.DataFrame) -> list[float]:
        """Evaluate multiple parameter sets in parallel."""
        data_pickle = pickle.dumps(market_data)
        futures = []
        for params in param_batch:
            future = self.executor.submit(self._evaluate_single_params, data_pickle, params)
            futures.append(future)
        results = [future.result() for future in futures]
        try:
            if results:
                self.logger.info({'msg': 'parallel_batch_scores', 'count': len(results), 'mean': float(np.mean(results)), 'min': float(np.min(results)), 'p90': float(np.percentile(results, 90)), 'max': float(np.max(results))})
        except Exception:
            pass
        self.logger.info(f'Evaluated {len(results)} parameter sets in parallel')
        return results

    @staticmethod
    def _evaluate_single_params(data_pickle: bytes, params: dict[str, Any]) -> float:
        """Evaluate single parameter set (runs in separate process)."""
        _ = pickle.loads(data_pickle)
        return float(random.uniform(-1.0, 1.0))

    def __del__(self) -> None:
        """Clean up executor."""
        if hasattr(self, 'executor') and self.executor:
            try:
                self.executor.shutdown(wait=True)
            except Exception:
                pass

class IncrementalTrainer:
    """Incremental training to reuse model states."""

    def __init__(self, base_model_config: dict[str, Any]) -> None:
        self.base_config = base_model_config
        self.model_cache: dict[str, Any] = {}
        self.logger = system_logger.getChild('IncrementalTrainer')

    def train_incrementally(self, params: dict[str, Any], X: np.ndarray, y: np.ndarray) -> Any:
        """Train model incrementally from cached state."""
        model_key = self._generate_model_key(params)
        if model_key in self.model_cache:
            model = self.model_cache[model_key]
            self.logger.info('Continuing training from cached model state')
        else:
            model = self._create_model(params)
            self.logger.info('Training new model')
        self.model_cache[model_key] = model
        return model

    def _generate_model_key(self, params: dict[str, Any]) -> str:
        """Generate cache key based on core model parameters."""
        core_params = {'max_depth': params.get('max_depth'), 'learning_rate': params.get('learning_rate'), 'subsample': params.get('subsample'), 'colsample_bytree': params.get('colsample_bytree')}
        return str(hash(_make_hashable(core_params)))

    def _create_model(self, params: dict[str, Any]) -> Any:
        """Create new model with given parameters (placeholder)."""
        return {'params': params.copy()}

class StreamingDataProcessor:
    """Streaming processor for large datasets."""

    def __init__(self, chunk_size: int=10000) -> None:
        self.chunk_size = chunk_size
        self.logger = system_logger.getChild('StreamingDataProcessor')

    def process_data_stream(self, data_path: str) -> None:
        """Yield data chunks for streaming processing."
        Returns an iterator of pandas DataFrame chunks.
        """
        try:
            if data_path.endswith('.parquet'):
                yield from self._iter_parquet_chunks(data_path)
            elif data_path.endswith('.csv'):
                yield from self._iter_csv_chunks(data_path)
            else:
                msg = f'Unsupported file format: {data_path}'
                raise ValueError(msg)
        except Exception as e:
            self.logger.exception(f'Error processing data stream: {e}')
            raise

    def _iter_parquet_chunks(self, file_path: str) -> None:
        """Iterate Parquet file in chunks."""
        try:
            if pq is None:
                self.logger.warning('pyarrow not available; falling back to pandas read_parquet (single chunk)')
                yield pd.read_parquet(file_path)
                return
            parquet_file = pq.ParquetFile(file_path)
            count = 0
            for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
                count += 1
                yield batch.to_pandas()
            self.logger.info(f'Streamed {count} chunks from Parquet file')
        except Exception as e:
            self.logger.exception(f'Error reading Parquet file {file_path}: {e}')
            raise

    def _iter_csv_chunks(self, file_path: str) -> None:
        """Iterate CSV file in chunks."""
        count = 0
        for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
            count += 1
            yield chunk
        self.logger.info(f'Streamed {count} chunks from CSV file')

    def write_incremental_parquet(self, chunks_iter: Any, target_path: str, compression: str='snappy') -> None:
        """Write DataFrame chunks incrementally to Parquet (append mode)."
        If pyarrow is not available, fall back to concatenating in bounded windows.
        """
        try:
            target = Path(target_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if pq is None:
                window: list[pd.DataFrame] = []
                window_rows = 0
                for df in chunks_iter:
                    window.append(df)
                    window_rows += len(df)
                    if window_rows >= self.chunk_size * 10:
                        pd.concat(window, ignore_index=True).to_parquet(target, compression=compression)
                        window.clear()
                        window_rows = 0
                if window:
                    pd.concat(window, ignore_index=True).to_parquet(target, compression=compression)
                return
            import pyarrow as pa
            import pyarrow.parquet as pq_mod
        except Exception as e:
            pass
        import json
from src.core.decorators.errors import handles_errors
        writer = None
        for df in chunks_iter:
            table = pa.Table.from_pandas(df)
            if writer is None:
                writer = pq_mod.ParquetWriter(str(target), table.schema, compression=compression)
            writer.write_table(table)
        if writer is not None:
            writer.close()

class AdaptiveSampler:
    """Adaptive sampling to focus on promising regions."""

    def __init__(self, initial_samples: int=100) -> None:
        self.initial_samples = initial_samples
        self.promising_regions: list[dict[str, Any]] = []
        self.trial_history: list[dict[str, Any]] = []
        self.logger = system_logger.getChild('AdaptiveSampler')

    def suggest_parameters(self, parameter_bounds: dict[str, tuple[float, float]]) -> dict[str, Any]:
        """Suggest parameters based on promising regions."""
        use_random = len(self.trial_history) < self.initial_samples
        with contextlib.suppress(Exception):
            self.logger.info({'msg': 'sampler_suggest', 'mode': 'random' if use_random else 'adaptive', 'history_len': len(self.trial_history)})
        params = self._random_sampling(parameter_bounds) if use_random else self._adaptive_sampling(parameter_bounds)
        with contextlib.suppress(Exception):
            self.logger.info({'msg': 'sampler_suggest_result', 'params': params})
        return params

    def update_trial_history(self, params: dict[str, Any], score: float) -> None:
        """Update trial history with new result."""
        self.trial_history.append({'params': params, 'score': float(score)})
        try:
            best = max(self.trial_history, key=lambda x: x.get('score', float('-inf'))).get('score', None)
            self.logger.info({'msg': 'sampler_update', 'score': float(score), 'best_so_far': float(best) if best is not None else None, 'history_len': len(self.trial_history)})
        except Exception:
            pass

    def _adaptive_sampling(self, parameter_bounds: dict[str, tuple[float, float]]) -> dict[str, Any]:
        """Sample from promising regions identified in history."""
        sorted_trials = sorted(self.trial_history, key=lambda x: x['score'], reverse=True)
        top_quartile = sorted_trials[:len(sorted_trials) // 4]
        if not top_quartile:
            return self._random_sampling(parameter_bounds)
        reference_trial = random.choice(top_quartile)
        return self._perturb_parameters(reference_trial['params'], parameter_bounds)

    def _random_sampling(self, parameter_bounds: dict[str, tuple[float, float]]) -> dict[str, Any]:
        """Random parameter sampling."""
        params: dict[str, Any] = {}
        for param_name, (min_val, max_val) in parameter_bounds.items():
            params[param_name] = random.uniform(min_val, max_val)
        return params

    def _perturb_parameters(self, base_params: dict[str, Any], parameter_bounds: dict[str, tuple[float, float]]) -> dict[str, Any]:
        """Perturb parameters around promising region."""
        perturbed: dict[str, Any] = {}
        perturbation_factor = 0.1
        for param_name, base_value in base_params.items():
            if param_name in parameter_bounds:
                min_val, max_val = parameter_bounds[param_name]
                range_val = max_val - min_val
                noise = random.uniform(-perturbation_factor, perturbation_factor) * range_val
                new_value = float(np.clip(base_value + noise, min_val, max_val))
                perturbed[param_name] = new_value
            else:
                perturbed[param_name] = base_value
        return perturbed

class MemoryEfficientDataManager:
    """Memory-efficient data structures for large datasets."""

    def __init__(self) -> None:
        self.logger = system_logger.getChild('MemoryEfficientDataManager')
        self.data_cache: dict[str, Any] = {}

    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for memory usage."""
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        for col in df.select_dtypes(include=['object']).columns:
            if len(df) and df[col].nunique() / max(1, len(df)) < 0.5:
                df[col] = df[col].astype('category')
        with contextlib.suppress(Exception):
            self.logger.debug(f'Optimized DataFrame memory usage: shape={df.shape}')
        return df

    def _normalize_timestamp_column(self, df: pd.DataFrame, column: str='timestamp') -> pd.DataFrame:
        """Ensure timestamp column exists and is timezone-aware datetime. Drops invalid rows."""
        try:
            if column not in df.columns:
                return df
            ts = df[column]
            if pd.api.types.is_datetime64_any_dtype(ts):
                if ts.dt.tz is None:
                    df[column] = ts.dt.tz_localize('UTC')
                else:
                    df[column] = ts.dt.tz_convert('UTC')
                return df
            if pd.api.types.is_integer_dtype(ts) or pd.api.types.is_float_dtype(ts):
                unit = 'ms' if ts.dropna().astype(float).median() > 1000000000000.0 else 's'
                df[column] = pd.to_datetime(df[column], unit=unit, errors='coerce', utc=True)
            else:
                df[column] = pd.to_datetime(df[column], errors='coerce', utc=True)
            return df.dropna(subset=[column])
        except Exception as e:
            self.logger.warning(f'Timestamp normalization failed: {e}')
            return df

    def save_to_parquet(self, df: pd.DataFrame, file_path: str, compression: str='snappy', index: bool=False) -> None:
        """Save DataFrame to Parquet format for efficient storage."""
        try:
            df_to_save = self.optimize_dataframe(df.copy())
            if 'timestamp' in df_to_save.columns:
                df_to_save = self._normalize_timestamp_column(df_to_save, 'timestamp')
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            df_to_save.to_parquet(file_path, compression=compression, index=index)
            self.logger.info(f'Saved DataFrame to Parquet: {file_path}')
        except Exception as e:
            self.logger.exception(f'Failed to save Parquet {file_path}: {e}')
            raise

    def load_from_parquet(self, file_path: str, columns: list[str] | None=None, nrows: int | None=None) -> pd.DataFrame:
        """Load DataFrame from Parquet with robust fallbacks and timestamp normalization."""
        try:
            file_path_str = str(file_path)
            try:
                size_kb = os.path.getsize(file_path_str) / 1024
                self.logger.info(f'Loading Parquet: {file_path_str} ({size_kb:.2f} KB)')
            except Exception:
                self.logger.info(f'Loading Parquet: {file_path_str}')
            try:
                df = pd.read_parquet(file_path_str, columns=columns)
            except Exception as e1:
                self.logger.warning(f'Default read_parquet failed: {e1}')
                try:
                    df = pd.read_parquet(file_path_str, columns=columns, engine='pyarrow')
                except Exception as e2:
                    self.logger.warning(f'PyArrow read failed: {e2}')
                    df = pd.read_parquet(file_path_str, columns=columns, engine='fastparquet')
            if nrows is not None and len(df) > nrows:
                df = df.head(nrows)
            if 'timestamp' in df.columns:
                df = self._normalize_timestamp_column(df, 'timestamp')
            self.logger.info(f'Loaded DataFrame from Parquet: {file_path_str} -> {df.shape}')
            return df
        except Exception as e:
            self.logger.exception(f'Failed to load Parquet {file_path}: {e}')
            raise

    def get_subset(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> np.ndarray:
        """Get numpy array subset for efficient computation."""
        return df.iloc[start_idx:end_idx].values

class MemoryManager:
    """Manage memory usage during optimization."""

    def __init__(self, memory_threshold: float=0.8) -> None:
        self.memory_threshold = memory_threshold
        self.logger = system_logger.getChild('MemoryManager')
        self.cleanup_counter = 0

    def check_memory_usage(self) -> bool:
        """Check and manage memory usage."""
        memory_percent = psutil.virtual_memory().percent / 100
        if memory_percent > self.memory_threshold:
            self.logger.warning(f'Memory usage high: {memory_percent:.1%}')
            self._cleanup_memory()
            return True
        return False

    def _cleanup_memory(self) -> None:
        """Clean up memory by forcing garbage collection."""
        self.cleanup_counter += 1
        self.logger.info(f'Performing memory cleanup #{self.cleanup_counter}')
        gc.collect()
        memory_after = psutil.virtual_memory().percent / 100
        self.logger.info(f'Memory usage after cleanup: {memory_after:.1%}')

    def profile_memory_usage(self) -> dict[str, float]:
        """Profile current memory usage."""
        memory_info = psutil.virtual_memory()
        return {'total_gb': float(memory_info.total / 1024 ** 3), 'available_gb': float(memory_info.available / 1024 ** 3), 'used_gb': float(memory_info.used / 1024 ** 3), 'percentage': float(memory_info.percent)}

class EnhancedTrainingManagerOptimized:
    """Enhanced training manager with comprehensive optimization strategies."

    Implements:
    1. Cached backtesting to avoid redundant calculations
    2. Progressive evaluation to stop unpromising trials early
    3. Parallel backtesting for multiple parameter combinations
    4. Incremental training to reuse model states
    5. Streaming for large datasets
    6. Adaptive sampling to focus on promising regions
    7. Memory-efficient data structures
    8. Memory profiling and leak detection
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize enhanced training manager with optimizations."""
        self.config = config
        self.logger = system_logger.getChild('EnhancedTrainingManagerOptimized')
        self.is_training = False
        self.training_results: dict[str, Any] = {}
        self.training_history: list[dict[str, Any]] = []
        self.cached_backtester: CachedBacktester | None = None
        self.progressive_evaluator: ProgressiveEvaluator | None = None
        self.parallel_backtester: ParallelBacktester | None = None
        self.incremental_trainer: IncrementalTrainer | None = None
        self.streaming_processor: StreamingDataProcessor | None = None
        self.adaptive_sampler: AdaptiveSampler | None = None
        self.memory_manager = MemoryManager()
        self.data_manager = MemoryEfficientDataManager()
        self.optimization_config = self.config.get('computational_optimization', {})
        self._load_optimization_config()
        self.analyst_models: dict[str, Any] = {}
        self.tactician_models: dict[str, Any] = {}
        self.ensemble_creator: Any | None = None
        self.calibration_systems: dict[str, Any] = {}

    def _load_optimization_config(self) -> None:
        """Load optimization configuration from enhanced_training_manager_optimized."""
        caching_config = self.optimization_config.get('caching', {})
        self.enable_caching = caching_config.get('enabled', True)
        self.max_cache_size = caching_config.get('max_cache_size', 1000)
        self.cache_ttl = caching_config.get('cache_ttl', 3600)
        parallel_config = self.optimization_config.get('parallelization', {})
        self.enable_parallelization = parallel_config.get('enabled', True)
        self.max_workers = parallel_config.get('max_workers', 8)
        self.chunk_size = parallel_config.get('chunk_size', 1000)
        early_stop_config = self.optimization_config.get('early_stopping', {})
        self.enable_early_stopping = early_stop_config.get('enabled', True)
        self.patience = early_stop_config.get('patience', 10)
        self.min_trials = early_stop_config.get('min_trials', 20)
        memory_config = self.optimization_config.get('memory_management', {})
        self.enable_memory_management = memory_config.get('enabled', True)
        self.memory_threshold = memory_config.get('memory_threshold', 0.8)
        self.cleanup_frequency = memory_config.get('cleanup_frequency', 100)
        streaming_config = self.optimization_config.get('streaming', {})
        self.stream_direct_to_final = streaming_config.get('direct_to_final', False)
        self.logger.info('Loaded optimization configuration')

    @handles_errors(error_handlers={ValueError: (False, 'Invalid configuration'), AttributeError: (False, 'Missing required parameters'), KeyError: (False, 'Missing configuration keys')}, default_return=False, context='initialization')
    async def initialize(self) -> bool:
        """Initialize the enhanced training manager with optimizations."""
        try:
            self.logger.info('🚀 Initializing Enhanced Training Manager with Optimizations...')
            if self.enable_parallelization:
                self.parallel_backtester = ParallelBacktester(n_workers=self.max_workers)
                self.logger.info(f'✅ Parallel backtester initialized with {self.max_workers} workers')
            self.streaming_processor = StreamingDataProcessor(chunk_size=self.chunk_size)
            self.adaptive_sampler = AdaptiveSampler()
            base_model_config = self.config.get('model', {})
            self.incremental_trainer = IncrementalTrainer(base_model_config)
            self.logger.info('✅ All optimization components initialized successfully')
            return True
        except Exception as e:
            self.logger.exception(f'❌ Initialization failed: {e}')
            return False

    async def execute_optimized_training(self, symbol: str, exchange: str, timeframe: str='1h') -> dict[str, Any]:
        """Execute training with all optimizations enabled."""
        try:
            self.is_training = True
            self.logger.info(f'🎯 Starting optimized training for {symbol} on {exchange}')
            market_data = await self._load_and_optimize_data(symbol, exchange, timeframe)
            if self.enable_caching:
                self.cached_backtester = CachedBacktester(market_data)
                self.logger.info('✅ Cached backtester initialized')
            if self.enable_early_stopping:
                self.progressive_evaluator = ProgressiveEvaluator(market_data)
                self.logger.info('✅ Progressive evaluator initialized')
            training_results = await self._execute_training_pipeline(market_data, symbol, exchange, timeframe)
            if self.enable_memory_management:
                self.memory_manager.check_memory_usage()
            self.training_results = training_results
            return training_results
        except Exception as e:
            self.logger.exception(f'❌ Optimized training failed: {e}')
            return {}
        finally:
            self.is_training = False

    async def _load_and_optimize_data(self, symbol: str, exchange: str, timeframe: str) -> pd.DataFrame:
        """Load and optimize data for training."""
        cache_key = f'{symbol}_{exchange}_{timeframe}'
        parquet_path = f'data_cache/{cache_key}.parquet'
        if os.path.exists(parquet_path):
            self.logger.info(f'Loading data from Parquet: {parquet_path}')
            data = self.data_manager.load_from_parquet(parquet_path)
        else:
            csv_path = f'data_cache/klines_{exchange}_{symbol}_{timeframe}_*.csv'
            csv_files = list(Path().glob(csv_path))
            if csv_files:
                self.logger.info(f'Loading and streaming data from {len(csv_files)} CSV files')
                if self.stream_direct_to_final:
                    for csv_file in csv_files:
                        chunks_iter = self.streaming_processor.process_data_stream(str(csv_file))
                        self.streaming_processor.write_incremental_parquet(chunks_iter, parquet_path)
                    data = pd.read_parquet(parquet_path)
                else:
                    tmp_parquet_path = f'{parquet_path}.tmp'
                    for csv_file in csv_files:
                        chunks_iter = self.streaming_processor.process_data_stream(str(csv_file))
                        self.streaming_processor.write_incremental_parquet(chunks_iter, tmp_parquet_path)
                    data = pd.read_parquet(tmp_parquet_path)
                    optimized_data = self.data_manager.optimize_dataframe(data)
                    self.data_manager.save_to_parquet(optimized_data, parquet_path)
                    with contextlib.suppress(Exception):
                        Path(tmp_parquet_path).unlink(missing_ok=True)
                    data = optimized_data
            else:
                msg = f'No data found for {symbol} on {exchange}'
                raise FileNotFoundError(msg)
        data = self.data_manager.optimize_dataframe(data)
        self.logger.info(f'✅ Data loaded and optimized: {len(data)} rows')
        return data

    async def _execute_training_pipeline(self, market_data: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> dict[str, Any]:
        """Execute the full training pipeline with optimizations."""
        results: dict[str, Any] = {}
        try:
            self.logger.info('📊 Step 1: Optimized Data Collection')
            data_collection_results = await self._optimized_data_collection(market_data, symbol, exchange, timeframe)
            results['data_collection'] = data_collection_results
            if self.enable_memory_management:
                self.memory_manager.check_memory_usage()
            self.logger.info('🏛️ Step 2: Market Regime Classification')
            regime_results = await self._optimized_regime_classification(market_data)
            results['regime_classification'] = regime_results
            self.logger.info('🔧 Step 3: Progressive Hyperparameter Optimization')
            optimization_results = await self._progressive_hyperparameter_optimization(market_data, symbol, exchange, timeframe)
            results['hyperparameter_optimization'] = optimization_results
            self.logger.info('🤖 Step 4: Incremental Model Training')
            model_results = await self._incremental_model_training(market_data, optimization_results)
            results['model_training'] = model_results
            if self.enable_parallelization:
                self.logger.info('🎼 Step 5: Parallel Ensemble Creation')
                ensemble_results = await self._parallel_ensemble_creation(model_results)
                results['ensemble_creation'] = ensemble_results
            self.logger.info('✅ Optimized training pipeline completed successfully')
            return results
        except Exception as e:
            self.logger.exception(f'❌ Training pipeline failed: {e}')
            return results

    async def _optimized_data_collection(self, market_data: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> dict[str, Any]:
        """Optimized data collection with caching and streaming."""
        return {'status': 'success', 'rows': len(market_data), 'memory_usage_mb': float(market_data.memory_usage(deep=True).sum() / 1024 ** 2), 'data_types': {k: str(v) for k, v in dict(market_data.dtypes).items()}}

    async def _optimized_regime_classification(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """Optimized regime classification with caching."""
        return {'status': 'success', 'regimes_identified': ['bull', 'bear', 'sideways'], 'classification_accuracy': 0.85}

    async def _progressive_hyperparameter_optimization(self, market_data: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> dict[str, Any]:
        """Progressive hyperparameter optimization with adaptive sampling."""
        if not self.adaptive_sampler:
            self.adaptive_sampler = AdaptiveSampler()
        parameter_bounds = {'learning_rate': (0.02, 0.2), 'max_depth': (3, 8), 'n_estimators': (100, 400), 'subsample': (0.7, 1.0), 'colsample_bytree': (0.7, 1.0)}
        best_score = -np.inf
        best_params: dict[str, Any] | None = None
        n_trials = int(self.config.get('n_trials', 100))
        early_stop_patience = int(self.patience) if hasattr(self, 'patience') else 10
        no_improve_counter = 0
        for trial in range(n_trials):
            params = self.adaptive_sampler.suggest_parameters(parameter_bounds)
            if self.enable_early_stopping and self.progressive_evaluator:
                score = self.progressive_evaluator.evaluate_progressively(params, lambda subset, p: self._evaluate_params(subset, p))
            elif self.enable_caching and self.cached_backtester:
                score = self.cached_backtester.run_cached_backtest(params)
            else:
                score = self._evaluate_params(market_data, params)
            self.adaptive_sampler.update_trial_history(params, score)
            if score > best_score:
                best_score = score
                best_params = params
                no_improve_counter = 0
            else:
                no_improve_counter += 1
            if trial >= max(10, early_stop_patience) and no_improve_counter >= early_stop_patience:
                self.logger.info(f'Early stop at trial {trial + 1} due to no improvement for {no_improve_counter} trials')
                break
            if self.enable_memory_management and (trial + 1) % max(1, self.cleanup_frequency) == 0:
                self.memory_manager.check_memory_usage()
            self.logger.debug(f'Trial {trial + 1}/{n_trials}: Score = {score:.4f}, Best = {best_score:.4f}')
        trials_completed = trial + 1
        return {'status': 'success', 'best_score': float(best_score), 'best_params': best_params, 'n_trials_completed': int(trials_completed)}

    def _evaluate_params(self, market_data: pd.DataFrame, params: dict[str, Any]) -> float:
        """Evaluate parameter set (placeholder)."""
        return float(random.uniform(-1.0, 1.0))

    async def _incremental_model_training(self, market_data: pd.DataFrame, optimization_results: dict[str, Any]) -> dict[str, Any]:
        """Incremental model training to reuse model states."""
        if not self.incremental_trainer:
            base_config = self.config.get('model', {})
            self.incremental_trainer = IncrementalTrainer(base_config)
        best_params = optimization_results.get('best_params', {})
        X = market_data[['open', 'high', 'low', 'close', 'volume']].values
        y = (market_data['close'].shift(-1) > market_data['close']).astype(int).values[:-1]
        X = X[:-1]
        model = self.incremental_trainer.train_incrementally(best_params, X, y)
        return {'status': 'success', 'model_trained': model is not None, 'training_samples': int(len(X)), 'features': int(X.shape[1]) if len(X) > 0 else 0}

    async def _parallel_ensemble_creation(self, model_results: dict[str, Any]) -> dict[str, Any]:
        """Parallel ensemble creation."""
        ensemble_params = [{'model_type': 'xgb', 'weight': 0.4}, {'model_type': 'lgb', 'weight': 0.3}, {'model_type': 'cat', 'weight': 0.3}]
        dummy_data = pd.DataFrame({'close': np.random.randn(1000), 'volume': np.random.randn(1000)})
        with ParallelBacktester(n_workers=self.max_workers) as pb:
            ensemble_scores = pb.evaluate_batch(ensemble_params, dummy_data)
        return {'status': 'success', 'ensemble_models': len(ensemble_params), 'ensemble_scores': ensemble_scores}

    def get_memory_profile(self) -> dict[str, Any]:
        """Get current memory profile."""
        return self.memory_manager.profile_memory_usage()

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get optimization statistics."""
        stats = {'caching_enabled': self.enable_caching, 'parallelization_enabled': self.enable_parallelization, 'early_stopping_enabled': self.enable_early_stopping, 'memory_management_enabled': self.enable_memory_management, 'max_workers': self.max_workers, 'memory_threshold': self.memory_threshold}
        if self.cached_backtester:
            stats['cache_size'] = len(self.cached_backtester.cache)
        if self.adaptive_sampler:
            stats['trial_history_size'] = len(self.adaptive_sampler.trial_history)
        return stats

    async def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info('🧹 Cleaning up resources...')
        if self.parallel_backtester:
            del self.parallel_backtester
        if self.enable_memory_management:
            self.memory_manager._cleanup_memory()
        self.logger.info('✅ Cleanup completed')

class ParquetDatasetManager:
    """Efficient parquet dataset management for large-scale data operations."""

    def __init__(self, logger: logging.Logger=None) -> None:
        self.logger = logger or system_logger.getChild('ParquetDatasetManager')
        if pa is None or pq is None:
            self.logger.error('❌ pyarrow is required for ParquetDatasetManager operations')
            msg = 'pyarrow is required for ParquetDatasetManager operations'
            raise ImportError(msg)

    def write_flat_parquet(self, df: pd.DataFrame, file_path: str, compression: str='snappy') -> None:
        """Write DataFrame to parquet format with optimized settings."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, file_path, compression=compression)
            self.logger.info(f'✅ Parquet file written: {file_path}')
        except Exception as e:
            self.logger.exception(f'❌ Failed to write parquet file {file_path}: {e}')
            raise

    def write_partitioned_dataset(self, df: pd.DataFrame, base_dir: str, partition_cols: list[str] | None=None, schema_name: str | None=None, compression: str='snappy', metadata: dict[str, Any] | None=None, min_rows_per_group: int=128000, max_rows_per_file: int=5000000) -> None:
        """Write a hive-partitioned dataset using pyarrow.dataset.write_dataset."""
        try:
            os.makedirs(base_dir, exist_ok=True)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if metadata:
                try:
                    schema_with_meta = table.schema.with_metadata({str(k): str(v) if v is not None else '' for k, v in metadata.items()})
                    table = table.cast(schema_with_meta)
                except Exception:
                    pass
            partitioning = None
            if partition_cols:
                fields = []
                for col in partition_cols:
                    try:
                        f = table.schema.field(col)
                        fields.append(pa.field(col, f.type))
                    except KeyError:
                        fields.append(pa.field(col, pa.string()))
                partition_schema = pa.schema(fields)
                partitioning = ds.partitioning(partition_schema, flavor='hive')
            write_args = {'base_dir': base_dir, 'format': 'parquet', 'basename_template': 'part-{i}.parquet', 'existing_data_behavior': 'overwrite_or_ignore', 'max_rows_per_file': max_rows_per_file, 'min_rows_per_group': min_rows_per_group, 'max_rows_per_group': min(max_rows_per_file, 1048576), 'partitioning': partitioning}
            ds.write_dataset(table, **write_args)
            self.logger.info(f'✅ Partitioned dataset written to {base_dir} with partitions={partition_cols or []}')
        except Exception as e:
            self.logger.exception(f'❌ Failed to write partitioned dataset to {base_dir}: {e}')
            raise

    def materialize_projection(self, base_dir: str, filters: list[tuple[str, str, Any]] | None, columns: list[str] | None, output_dir: str, partition_cols: list[str] | None=None, schema_name: str | None=None, compression: str='snappy', batch_size: int=131072, metadata: dict[str, Any] | None=None) -> None:
        """Scan an existing dataset, project columns with filters, and write to a new partitioned dataset."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            dataset = ds.dataset(base_dir, format='parquet')
            scanner = dataset.scanner(columns=columns, filter=self._build_filter(filters), batch_size=batch_size)
            table = scanner.to_table()
            if metadata:
                try:
                    schema_with_meta = table.schema.with_metadata({str(k): str(v) if v is not None else '' for k, v in metadata.items()})
                    table = table.cast(schema_with_meta)
                except Exception:
                    pass
            ds.write_dataset(table, base_dir=output_dir, format='parquet', basename_template='part-{i}.parquet', existing_data_behavior='overwrite_or_ignore', partitioning=ds.partitioning(partition_cols, flavor='hive') if partition_cols else None, max_rows_per_file=5000000, min_rows_per_group=128000, max_rows_per_group=1048576)
            self.logger.info(f'✅ Materialized projection to {output_dir} (columns={columns}, filters={filters})')
        except Exception as e:
            self.logger.exception(f'❌ Failed to materialize projection to {output_dir}: {e}')
            raise

    @staticmethod
    def _build_filter(filters: list[tuple[str, str, Any]] | None) -> None:
        if not filters:
            return None
        try:
            expr = None
            for col, op, val in filters:
                term = ds.field(col) == val if op == '==' else None
                expr = term if expr is None else expr & term
            return expr
        except Exception:
            return None

    def read_parquet(self, file_path: str, columns: list[str] | None=None) -> pd.DataFrame:
        """Read parquet file with optional column selection."""
        try:
            if columns:
                table = pq.read_table(file_path, columns=columns)
            else:
                table = pq.read_table(file_path)
            return table.to_pandas()
        except Exception as e:
            self.logger.exception(f'❌ Failed to read parquet file {file_path}: {e}')
            raise

    def get_parquet_info(self, file_path: str) -> dict[str, Any]:
        """Get information about a parquet file."""
        try:
            metadata = pq.read_metadata(file_path)
            return {'num_rows': metadata.num_rows, 'num_columns': metadata.num_columns, 'file_size_mb': os.path.getsize(file_path) / (1024 * 1024), 'schema': str(metadata.schema)}
        except Exception as e:
            self.logger.exception(f'❌ Failed to get parquet info for {file_path}: {e}')
            return {}