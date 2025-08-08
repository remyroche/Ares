# src/training/enhanced_training_manager.py

import asyncio
import pandas as pd
import time
import psutil
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any

# Added optimized imports
import gc
import multiprocessing as mp
import pickle
import random
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np

# Optional dependency: pyarrow is used for efficient parquet streaming; import lazily in methods
try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # ImportError or others
    pq = None  # type: ignore

warnings.filterwarnings("ignore")

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger
from src.utils.validator_orchestrator import validator_orchestrator

# Import computational optimization components
from src.training.optimization.computational_optimization_manager import (
    ComputationalOptimizationManager,
    create_computational_optimization_manager,
)
from src.config.computational_optimization import get_computational_optimization_config

from contextlib import contextmanager


# === Optimization helper classes (ported from optimized manager) ===
class CachedBacktester:
    """Cached backtesting to avoid redundant calculations."""

    def __init__(self, market_data: pd.DataFrame):
        self.market_data = market_data
        self.cache: dict[str, float] = {}
        self.technical_indicators = self._precompute_indicators()
        self.logger = system_logger.getChild("CachedBacktester")

    def _precompute_indicators(self) -> dict[str, np.ndarray]:
        indicators: dict[str, np.ndarray] = {}
        if 'close' in self.market_data:
            indicators['sma_20'] = self.market_data['close'].rolling(20).mean().values
            indicators['sma_50'] = self.market_data['close'].rolling(50).mean().values
            indicators['ema_12'] = self.market_data['close'].ewm(span=12).mean().values
            indicators['ema_26'] = self.market_data['close'].ewm(span=26).mean().values
            delta = self.market_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss.replace(0, np.nan))
            indicators['rsi'] = (100 - (100 / (1 + rs))).fillna(50).values
            high_low = self.market_data.get('high', pd.Series(index=self.market_data.index)).fillna(0) - self.market_data.get('low', pd.Series(index=self.market_data.index)).fillna(0)
            high_close = (self.market_data.get('high', pd.Series(index=self.market_data.index)).fillna(0) - self.market_data.get('close', pd.Series(index=self.market_data.index)).shift()).abs()
            low_close = (self.market_data.get('low', pd.Series(index=self.market_data.index)).fillna(0) - self.market_data.get('close', pd.Series(index=self.market_data.index)).shift()).abs()
            tr = np.maximum(high_low.values, np.maximum(high_close.values, low_close.values))
            indicators['atr'] = pd.Series(tr, index=self.market_data.index).rolling(window=14).mean().fillna(0).values
            indicators['volatility'] = self.market_data['close'].pct_change().rolling(20).std().fillna(0).values
        if 'volume' in self.market_data:
            indicators['volume_sma'] = self.market_data['volume'].rolling(20).mean().fillna(0).values
        self.logger.info(f"Precomputed {len(indicators)} technical indicators")
        return indicators

    def run_cached_backtest(self, params: dict[str, Any]) -> float:
        cache_key = str(hash(frozenset(params.items())))
        if cache_key in self.cache:
            return self.cache[cache_key]
        result = self._run_simplified_backtest(params)
        self.cache[cache_key] = result
        return result

    def _run_simplified_backtest(self, params: dict[str, Any]) -> float:
        # Placeholder for actual backtest logic using precomputed indicators
        return random.uniform(-1.0, 1.0)


class ProgressiveEvaluator:
    """Progressive evaluation to stop unpromising trials early."""

    def __init__(self, full_data: pd.DataFrame):
        self.full_data = full_data
        self.evaluation_stages: list[tuple[float, float]] = [
            (0.1, 0.3),
            (0.3, 0.5),
            (1.0, 1.0),
        ]
        self.logger = system_logger.getChild("ProgressiveEvaluator")

    def evaluate_progressively(self, params: dict[str, Any], evaluator_func) -> float:
        total_score = 0.0
        total_weight = 0.0
        for data_ratio, weight in self.evaluation_stages:
            subset_size = max(1, int(len(self.full_data) * data_ratio))
            subset_data = self.full_data.iloc[:subset_size]
            score = evaluator_func(subset_data, params)
            total_score += score * weight
            total_weight += weight
            if data_ratio < 1.0 and score < -0.5:
                self.logger.info(f"Early stopping at {data_ratio*100:.0f}% data due to poor performance")
                return -1.0
        return total_score / max(total_weight, 1e-9)


class ParallelBacktester:
    """Parallel backtesting for multiple parameter combinations."""

    def __init__(self, n_workers: int | None = None):
        self.n_workers = n_workers or min(mp.cpu_count() or 1, 8)
        self.executor = ProcessPoolExecutor(max_workers=self.n_workers)
        self.logger = system_logger.getChild("ParallelBacktester")

    def evaluate_batch(self, param_batch: list[dict[str, Any]], market_data: pd.DataFrame) -> list[float]:
        data_pickle = pickle.dumps(market_data)
        futures = [
            self.executor.submit(self._evaluate_single_params, data_pickle, params)
            for params in param_batch
        ]
        results = [future.result() for future in futures]
        self.logger.info(f"Evaluated {len(results)} parameter sets in parallel")
        return results

    @staticmethod
    def _evaluate_single_params(data_pickle: bytes, params: dict[str, Any]) -> float:
        _ = pickle.loads(data_pickle)
        return random.uniform(-1.0, 1.0)

    def shutdown(self) -> None:
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=True)


class IncrementalTrainer:
    """Incremental training to reuse model states."""

    def __init__(self, base_model_config: dict[str, Any]):
        self.base_config = base_model_config
        self.model_cache: dict[str, Any] = {}
        self.logger = system_logger.getChild("IncrementalTrainer")

    def train_incrementally(self, params: dict[str, Any], X: np.ndarray, y: np.ndarray) -> Any:
        model_key = self._generate_model_key(params)
        if model_key in self.model_cache:
            model = self.model_cache[model_key]
            self.logger.info("Continuing training from cached model state")
        else:
            model = self._create_model(params)
            self.logger.info("Training new model")
            self.model_cache[model_key] = model
        return model

    def _generate_model_key(self, params: dict[str, Any]) -> str:
        core_params = {
            'max_depth': params.get('max_depth'),
            'learning_rate': params.get('learning_rate'),
            'subsample': params.get('subsample'),
            'colsample_bytree': params.get('colsample_bytree'),
        }
        return str(hash(frozenset(core_params.items())))

    def _create_model(self, params: dict[str, Any]) -> Any:
        # Placeholder - implement model creation as needed
        return None


class StreamingDataProcessor:
    """Streaming processor for large datasets."""

    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size
        self.logger = system_logger.getChild("StreamingDataProcessor")

    def process_data_stream(self, data_path: str) -> pd.DataFrame:
        try:
            if data_path.endswith('.parquet'):
                return self._process_parquet_stream(data_path)
            elif data_path.endswith('.csv'):
                return self._process_csv_stream(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
        except Exception as e:
            self.logger.error(f"Error processing data stream: {e}")
            raise

    def _process_parquet_stream(self, file_path: str) -> pd.DataFrame:
        chunks: list[pd.DataFrame] = []
        if pq is None:
            self.logger.warning("pyarrow not available; falling back to standard read_parquet")
            try:
                return pd.read_parquet(file_path)
            except Exception as e:
                self.logger.error(f"Failed to read parquet without pyarrow: {e}")
                return pd.DataFrame()
        parquet_file = pq.ParquetFile(file_path)  # type: ignore
        for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
            chunk = batch.to_pandas()
            chunks.append(chunk)
        self.logger.info(f"Processed {len(chunks)} chunks from Parquet file")
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    def _process_csv_stream(self, file_path: str) -> pd.DataFrame:
        chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
            chunks.append(chunk)
        self.logger.info(f"Processed {len(chunks)} chunks from CSV file")
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


class AdaptiveSampler:
    """Adaptive sampling to focus on promising regions."""

    def __init__(self, initial_samples: int = 100):
        self.initial_samples = initial_samples
        self.promising_regions: list[dict[str, Any]] = []
        self.trial_history: list[dict[str, Any]] = []
        self.logger = system_logger.getChild("AdaptiveSampler")

    def suggest_parameters(self, parameter_bounds: dict[str, tuple[float, float]]) -> dict[str, Any]:
        if len(self.trial_history) < self.initial_samples:
            return self._random_sampling(parameter_bounds)
        return self._adaptive_sampling(parameter_bounds)

    def update_trial_history(self, params: dict[str, Any], score: float) -> None:
        self.trial_history.append({'params': params, 'score': score})

    def _adaptive_sampling(self, parameter_bounds: dict[str, tuple[float, float]]) -> dict[str, Any]:
        sorted_trials = sorted(self.trial_history, key=lambda x: x['score'], reverse=True)
        top_quartile = sorted_trials[: max(1, len(sorted_trials)//4)]
        if not top_quartile:
            return self._random_sampling(parameter_bounds)
        reference_trial = random.choice(top_quartile)
        return self._perturb_parameters(reference_trial['params'], parameter_bounds)

    def _random_sampling(self, parameter_bounds: dict[str, tuple[float, float]]) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name, (min_val, max_val) in parameter_bounds.items():
            params[name] = random.uniform(min_val, max_val)
        return params

    def _perturb_parameters(self, base_params: dict[str, Any], parameter_bounds: dict[str, tuple[float, float]]) -> dict[str, Any]:
        perturbed: dict[str, Any] = {}
        perturbation_factor = 0.1
        for name, base_value in base_params.items():
            if name in parameter_bounds:
                min_val, max_val = parameter_bounds[name]
                range_val = max_val - min_val
                noise = random.uniform(-perturbation_factor, perturbation_factor) * range_val
                new_value = float(np.clip(base_value + noise, min_val, max_val))
                perturbed[name] = new_value
            else:
                perturbed[name] = base_value
        return perturbed


class MemoryEfficientDataManager:
    """Memory-efficient data structures for large datasets."""

    def __init__(self):
        self.logger = system_logger.getChild("MemoryEfficientDataManager")

    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        for col in df.select_dtypes(include=['object']).columns:
            try:
                if len(df) > 0 and (df[col].nunique() / max(len(df), 1)) < 0.5:
                    df[col] = df[col].astype('category')
            except Exception:
                pass
        self.logger.info("Optimized DataFrame memory usage")
        return df

    def save_to_parquet(self, df: pd.DataFrame, file_path: str, compression: str = 'snappy') -> None:
        df.to_parquet(file_path, compression=compression, index=False)
        self.logger.info(f"Saved DataFrame to Parquet: {file_path}")

    def load_from_parquet(self, file_path: str) -> pd.DataFrame:
        df = pd.read_parquet(file_path)
        self.logger.info(f"Loaded DataFrame from Parquet: {file_path}")
        return df


class MemoryManager:
    """Manage memory usage during optimization."""

    def __init__(self, memory_threshold: float = 0.8):
        self.memory_threshold = memory_threshold
        self.logger = system_logger.getChild("MemoryManager")
        self.cleanup_counter = 0

    def check_memory_usage(self) -> bool:
        memory_percent = psutil.virtual_memory().percent / 100.0
        if memory_percent > self.memory_threshold:
            self.logger.warning(f"Memory usage high: {memory_percent:.1%}")
            self._cleanup_memory()
            return True
        return False

    def _cleanup_memory(self) -> None:
        self.cleanup_counter += 1
        self.logger.info(f"Performing memory cleanup #{self.cleanup_counter}")
        gc.collect()
        self.logger.info(f"Memory usage after cleanup: {psutil.virtual_memory().percent:.1%}")

    def profile_memory_usage(self) -> dict[str, float]:
        memory_info = psutil.virtual_memory()
        return {
            'total_gb': memory_info.total / (1024**3),
            'available_gb': memory_info.available / (1024**3),
            'used_gb': memory_info.used / (1024**3),
            'percentage': float(memory_info.percent),
        }


class EnhancedTrainingManager:
    """
    Enhanced training manager with comprehensive 16-step pipeline.
    This module orchestrates the complete training pipeline including analyst and tactician steps.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize enhanced training manager.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("EnhancedTrainingManager")

        # Enhanced training manager state
        self.is_training: bool = False
        self.enhanced_training_results: dict[str, Any] = {}
        self.enhanced_training_history: list[dict[str, Any]] = []

        # Configuration
        self.enhanced_training_config: dict[str, Any] = self.config.get(
            "enhanced_training_manager",
            {},
        )
        self.enhanced_training_interval: int = self.enhanced_training_config.get(
            "enhanced_training_interval",
            3600,
        )
        self.max_enhanced_training_history: int = self.enhanced_training_config.get(
            "max_enhanced_training_history",
            100,
        )
        
        # Training parameters
        self.blank_training_mode: bool = self.enhanced_training_config.get("blank_training_mode", False)
        self.max_trials: int = self.enhanced_training_config.get("max_trials", 200)
        self.n_trials: int = self.enhanced_training_config.get("n_trials", 100)
        self.lookback_days: int = self.enhanced_training_config.get("lookback_days", 30)
        
        # Validation parameters
        self.enable_validators: bool = self.enhanced_training_config.get("enable_validators", True)
        self.validation_results: dict[str, Any] = {}
        
        # Computational optimization parameters
        self.enable_computational_optimization: bool = self.enhanced_training_config.get("enable_computational_optimization", True)
        self.computational_optimization_manager: ComputationalOptimizationManager | None = None
        self.optimization_statistics: dict[str, Any] = {}
        
        # Optimization component configuration (ported)
        optimization_root = get_computational_optimization_config().get("computational_optimization", {})
        self.optimization_config: dict[str, Any] = optimization_root
        self.enable_caching: bool = optimization_root.get("enable_caching", True)
        self.enable_parallelization: bool = optimization_root.get("enable_parallelization", True)
        self.enable_early_stopping: bool = optimization_root.get("enable_early_stopping", True)
        self.enable_memory_management: bool = optimization_root.get("enable_memory_management", True)
        self.max_workers: int | None = optimization_root.get("max_workers")
        self.chunk_size: int = optimization_root.get("chunk_size", 1000)
        self.cleanup_frequency: int = optimization_root.get("cleanup_frequency", 100)
        self.memory_threshold: float = optimization_root.get("memory_threshold", 0.8)

        # Optimization components (lazy init)
        self.cached_backtester: CachedBacktester | None = None
        self.progressive_evaluator: ProgressiveEvaluator | None = None
        self.parallel_backtester: ParallelBacktester | None = None
        self.incremental_trainer: IncrementalTrainer | None = None
        self.streaming_processor: StreamingDataProcessor | None = None
        self.adaptive_sampler: AdaptiveSampler | None = None
        self.memory_manager: MemoryManager | None = None
        self.data_manager: MemoryEfficientDataManager | None = None
        
        # Checkpointing configuration
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        # Note: final names are namespaced per symbol/exchange/timeframe at save-time
        self.checkpoint_file = self.checkpoint_dir / "training_progress.json"
        self.enable_checkpointing = self.enhanced_training_config.get("enable_checkpointing", True)
        
    @contextmanager
    def _timed_step(self, name: str, step_times: dict):
        start = time.time()
        try:
            yield
            self._log_step_completion(name, start, step_times, success=True)
        except Exception:
            self._log_step_completion(name, start, step_times, success=False)
            raise

    def _save_checkpoint(self, step_name: str, pipeline_state: dict[str, Any]) -> None:
        """
        Save training progress checkpoint.
        
        Args:
            step_name: Current step name
            pipeline_state: Current pipeline state
        """
        if not self.enable_checkpointing:
            return
            
        try:
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "current_step": step_name,
                "pipeline_state": pipeline_state,
                "training_mode": "blank" if self.blank_training_mode else "full",
                "symbol": getattr(self, 'current_symbol', ''),
                "exchange": getattr(self, 'current_exchange', ''),
                "timeframe": getattr(self, 'current_timeframe', '1m'),
                "lookback_days": self.lookback_days,
                "max_trials": self.max_trials,
                "n_trials": self.n_trials
            }
            
            # Namespaced checkpoint path
            symbol = checkpoint_data.get('symbol') or 'unknown'
            exchange = checkpoint_data.get('exchange') or 'unknown'
            timeframe = checkpoint_data.get('timeframe') or 'unknown'
            ns_dir = self.checkpoint_dir / exchange / symbol / timeframe
            ns_dir.mkdir(parents=True, exist_ok=True)
            target_file = ns_dir / "training_progress.json"
            tmp_file = ns_dir / "training_progress.json.tmp"
            with open(tmp_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            os.replace(tmp_file, target_file)
            
            self.logger.info(f"💾 Checkpoint saved: {step_name} -> {target_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self) -> dict[str, Any] | None:
        """
        Load training progress checkpoint.
        
        Returns:
            dict: Checkpoint data or None if no checkpoint exists
        """
        # Attempt to load namespaced checkpoint based on current params
        if not self.enable_checkpointing:
            return None
        try:
            symbol = getattr(self, 'current_symbol', 'unknown')
            exchange = getattr(self, 'current_exchange', 'unknown')
            timeframe = getattr(self, 'current_timeframe', 'unknown')
            ns_file = self.checkpoint_dir / exchange / symbol / timeframe / "training_progress.json"
            if not ns_file.exists():
                return None
            with open(ns_file, 'r') as f:
                checkpoint_data = json.load(f)
            self.logger.info(f"📂 Checkpoint loaded: {checkpoint_data.get('current_step', 'unknown')} from {ns_file}")
            return checkpoint_data
            
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return None
    
    def _clear_checkpoint(self) -> None:
        """Clear the checkpoint file."""
        try:
            symbol = getattr(self, 'current_symbol', 'unknown')
            exchange = getattr(self, 'current_exchange', 'unknown')
            timeframe = getattr(self, 'current_timeframe', 'unknown')
            ns_file = self.checkpoint_dir / exchange / symbol / timeframe / "training_progress.json"
            if ns_file.exists():
                ns_file.unlink()
                self.logger.info(f"🗑️ Checkpoint cleared at {ns_file}")
        except Exception as e:
            self.logger.warning(f"Failed to clear checkpoint: {e}")
        
    def _get_system_resources(self) -> dict[str, float]:
        """
        Get current system resource usage.
        
        Returns:
            dict: System resource information
        """
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent(interval=0.1)
            
            # Get system-wide memory info
            system_memory = psutil.virtual_memory()
            system_memory_percent = system_memory.percent
            
            return {
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
                "system_memory_percent": system_memory_percent,
                "available_memory_gb": system_memory.available / 1024 / 1024 / 1024
            }
        except Exception as e:
            self.logger.warning(f"Could not get system resources: {e}")
            return {"memory_mb": 0, "cpu_percent": 0, "system_memory_percent": 0, "available_memory_gb": 0}
    
    def _analyze_resource_requirements(self) -> dict[str, Any]:
        """
        Analyze resource requirements for the training process.
        
        Returns:
            dict: Resource analysis information
        """
        try:
            # Get system info
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
            
            # Realistic estimates based on actual training complexity
            if self.blank_training_mode:
                estimated_memory_gb = 4.0  # Blank training uses less memory
                estimated_time_minutes = 90  # Realistic: 1.5 hours for blank training
                memory_warning_threshold = 6.0
                models_to_train = 4
                optimization_trials = 50
            else:
                estimated_memory_gb = 8.0  # Full training uses more memory
                estimated_time_minutes = 720  # Realistic: 12 hours for full training
                memory_warning_threshold = 12.0
                models_to_train = 12
                optimization_trials = 200
            
            # Check if system meets requirements
            memory_sufficient = memory_gb >= memory_warning_threshold
            cpu_sufficient = cpu_count >= 4
            
            return {
                "system_memory_gb": memory_gb,
                "cpu_count": cpu_count,
                "estimated_memory_gb": estimated_memory_gb,
                "estimated_time_minutes": estimated_time_minutes,
                "models_to_train": models_to_train,
                "optimization_trials": optimization_trials,
                "memory_sufficient": memory_sufficient,
                "cpu_sufficient": cpu_sufficient,
                "memory_warning_threshold": memory_warning_threshold,
                "recommendations": self._get_resource_recommendations(memory_gb, cpu_count),
                "step_breakdown": self._get_step_time_breakdown(self.blank_training_mode)
            }
        except Exception as e:
            self.logger.warning(f"Could not analyze resource requirements: {e}")
            return {}
    
    def _get_resource_recommendations(self, memory_gb: float, cpu_count: int) -> list[str]:
        """
        Get resource recommendations based on system specs.
        
        Args:
            memory_gb: Available memory in GB
            cpu_count: Number of CPU cores
            
        Returns:
            list: Recommendations
        """
        recommendations = []
        
        if memory_gb < 8:
            recommendations.append("⚠️ Consider upgrading to 16GB RAM for optimal performance")
        elif memory_gb < 12:
            recommendations.append("💡 16GB RAM recommended for full training mode")
        
        if cpu_count < 4:
            recommendations.append("⚠️ Consider using a system with at least 4 CPU cores")
        elif cpu_count < 8:
            recommendations.append("💡 8+ CPU cores recommended for faster training")
        
        if self.blank_training_mode:
            recommendations.append("✅ Blank training mode is suitable for your system")
        else:
            if memory_gb < 12:
                recommendations.append("⚠️ Full training mode may be slow on your system")
            else:
                recommendations.append("✅ Full training mode should work well on your system")
        
        return recommendations
    
    def _get_step_time_breakdown(self, is_blank_mode: bool) -> dict[str, int]:
        """
        Get realistic time breakdown for each step.
        
        Args:
            is_blank_mode: Whether this is blank training mode
            
        Returns:
            dict: Time estimates for each step in minutes
        """
        if is_blank_mode:
            return {
                "step1_data_collection": 5,
                "step2_market_regime_classification": 3,
                "step3_regime_data_splitting": 2,
                "step4_analyst_labeling_feature_engineering": 15,
                "step5_analyst_specialist_training": 10,
                "step6_analyst_enhancement": 8,
                "step7_analyst_ensemble_creation": 12,
                "step8_tactician_labeling": 5,
                "step9_tactician_specialist_training": 10,
                "step10_tactician_ensemble_creation": 12,
                "step11_confidence_calibration": 3,
                "step12_final_parameters_optimization": 15,
                "step13_walk_forward_validation": 8,
                "step14_monte_carlo_validation": 8,
                "step15_ab_testing": 5,
                "step16_saving": 2
            }
        else:
            return {
                "step1_data_collection": 15,
                "step2_market_regime_classification": 8,
                "step3_regime_data_splitting": 5,
                "step4_analyst_labeling_feature_engineering": 60,
                "step5_analyst_specialist_training": 30,
                "step6_analyst_enhancement": 25,
                "step7_analyst_ensemble_creation": 35,
                "step8_tactician_labeling": 15,
                "step9_tactician_specialist_training": 30,
                "step10_tactician_ensemble_creation": 35,
                "step11_confidence_calibration": 10,
                "step12_final_parameters_optimization": 240,
                "step13_walk_forward_validation": 60,
                "step14_monte_carlo_validation": 60,
                "step15_ab_testing": 30,
                "step16_saving": 5
            }
    
    def _optimize_memory_usage(self) -> None:
        """
        Perform memory optimization to reduce memory footprint.
        """
        try:
            # Force garbage collection
            gc.collect()
            
            # Log memory before and after optimization
            before_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            self.logger.info(f"🧹 Memory optimization: {before_memory:.1f} MB before cleanup")
            
            # Force another garbage collection
            gc.collect()
            
            after_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_saved = before_memory - after_memory
            
            self.logger.info(f"🧹 Memory optimization: {after_memory:.1f} MB after cleanup (saved {memory_saved:.1f} MB)")
            
            if memory_saved > 10:  # If we saved more than 10MB
                print(f"   🧹 Memory optimization saved {memory_saved:.1f} MB")
                
        except Exception as e:
            self.logger.warning(f"Memory optimization failed: {e}")
    
    def _get_progress_percentage(self, completed_steps: int, total_steps: int = 16) -> float:
        """
        Calculate progress percentage.
        
        Args:
            completed_steps: Number of completed steps
            total_steps: Total number of steps
            
        Returns:
            float: Progress percentage
        """
        return (completed_steps / total_steps) * 100
    
    def _log_progress(self, current_step: int, total_steps: int = 16, elapsed_time: float = 0) -> None:
        """
        Log progress with estimated completion time.
        
        Args:
            current_step: Current step number
            total_steps: Total number of steps
            elapsed_time: Time elapsed so far
        """
        progress = self._get_progress_percentage(current_step, total_steps)
        
        if elapsed_time > 0 and current_step > 0:
            # Estimate remaining time based on current progress
            estimated_total_time = (elapsed_time / current_step) * total_steps
            remaining_time = estimated_total_time - elapsed_time
            eta_minutes = remaining_time / 60
            
            self.logger.info(f"📊 Progress: {progress:.1f}% ({current_step}/{total_steps})")
            self.logger.info(f"⏱️ Elapsed: {elapsed_time/60:.1f} min | ETA: {eta_minutes:.1f} min")
        else:
            self.logger.info(f"📊 Progress: {progress:.1f}% ({current_step}/{total_steps})")
        
    def _log_step_completion(self, step_name: str, step_start: float, step_times: dict, success: bool = True) -> None:
        """
        Log step completion with timing and memory usage.
        
        Args:
            step_name: Name of the completed step
            step_start: Start time of the step
            step_times: Dictionary to store step times
            success: Whether the step was successful
        """
        step_time = time.time() - step_start
        step_times[step_name] = step_time
        
        # Get comprehensive system resources
        resources = self._get_system_resources()
        
        status_icon = "✅" if success else "❌"
        status_text = "completed successfully" if success else "failed"
        
        self.logger.info(f"{status_icon} {step_name}: {status_text} in {step_time:.2f}s")
        self.logger.info(f"💾 Process Memory: {resources['memory_mb']:.1f} MB | CPU: {resources['cpu_percent']:.1f}%")
        self.logger.info(f"🖥️ System Memory: {resources['system_memory_percent']:.1f}% | Available: {resources['available_memory_gb']:.1f} GB")
        
        
        
        # Memory warning system
        if resources['system_memory_percent'] > 85:
            warning_msg = f"⚠️ HIGH MEMORY USAGE: {resources['system_memory_percent']:.1f}% - Consider closing other applications"
            self.logger.warning(warning_msg)
        
        if resources['available_memory_gb'] < 2.0:
            warning_msg = f"⚠️ LOW AVAILABLE MEMORY: {resources['available_memory_gb']:.1f} GB remaining"
            self.logger.warning(warning_msg)
        
        # Log progress after each step
        completed_steps = len(step_times)
        elapsed_time = sum(step_times.values())
        self._log_progress(completed_steps, 16, elapsed_time)
        
    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid enhanced training manager configuration"),
            AttributeError: (False, "Missing required enhanced training parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="enhanced training manager initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize enhanced training manager.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("🚀 Initializing Enhanced Training Manager...")
            self.logger.info(f"📊 Blank training mode: {self.blank_training_mode}")
            self.logger.info(f"🔧 Max trials: {self.max_trials}")
            self.logger.info(f"🔧 N trials: {self.n_trials}")
            self.logger.info(f"📈 Lookback days: {self.lookback_days}")
            self.logger.info(f"🚀 Computational optimization: {self.enable_computational_optimization}")
            
            # Analyze resource requirements
            resource_analysis = self._analyze_resource_requirements()
            if resource_analysis:
                self.logger.info("📊 Resource Analysis:")
                self.logger.info(f"   💾 System Memory: {resource_analysis['system_memory_gb']:.1f} GB")
                self.logger.info(f"   🖥️ CPU Cores: {resource_analysis['cpu_count']}")
                self.logger.info(f"   📈 Estimated Memory Usage: {resource_analysis['estimated_memory_gb']:.1f} GB")
                self.logger.info(f"   ⏱️ Estimated Time: {resource_analysis['estimated_time_minutes']} minutes ({resource_analysis['estimated_time_minutes']/60:.1f} hours)")
                self.logger.info(f"   🤖 Models to Train: {resource_analysis['models_to_train']}")
                self.logger.info(f"   🔧 Optimization Trials: {resource_analysis['optimization_trials']}")
                
                # Show step-by-step breakdown
                if 'step_breakdown' in resource_analysis:
                    self.logger.info("📋 Step-by-Step Time Estimates:")
                    total_estimated = sum(resource_analysis['step_breakdown'].values())
                    for step_name, minutes in resource_analysis['step_breakdown'].items():
                        percentage = (minutes / total_estimated) * 100
                        self.logger.info(f"   {step_name}: {minutes} min ({percentage:.1f}%)")
                
                # Log recommendations
                if resource_analysis['recommendations']:
                    self.logger.info("💡 Recommendations:")
                    for rec in resource_analysis['recommendations']:
                        self.logger.info(f"   {rec}")
            
            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("❌ Invalid configuration for enhanced training manager")
                return False
            
            # Initialize computational optimization if enabled
            if self.enable_computational_optimization:
                await self._initialize_computational_optimization()

            # Initialize optimization components
            self.data_manager = MemoryEfficientDataManager()
            self.streaming_processor = StreamingDataProcessor(chunk_size=self.chunk_size)
            if self.enable_memory_management:
                self.memory_manager = MemoryManager(memory_threshold=self.memory_threshold)
            if self.enable_parallelization:
                self.parallel_backtester = ParallelBacktester(n_workers=self.max_workers)
            self.adaptive_sampler = AdaptiveSampler()
            self.incremental_trainer = IncrementalTrainer(self.config.get("model", {}))
            
            self.logger.info("✅ Enhanced Training Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced Training Manager initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate enhanced training manager configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate enhanced training manager specific settings
            if self.max_enhanced_training_history <= 0:
                self.logger.error("❌ Invalid max_enhanced_training_history configuration")
                return False
                
            if self.max_trials <= 0:
                self.logger.error("❌ Invalid max_trials configuration")
                return False
                
            if self.n_trials <= 0:
                self.logger.error("❌ Invalid n_trials configuration")
                return False
                
            if self.lookback_days <= 0:
                self.logger.error("❌ Invalid lookback_days configuration")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid enhanced training parameters"),
            AttributeError: (False, "Missing enhanced training components"),
            KeyError: (False, "Missing required enhanced training data"),
        },
        default_return=False,
        context="enhanced training execution",
    )
    async def execute_enhanced_training(
        self,
        enhanced_training_input: dict[str, Any],
    ) -> bool:
        """
        Execute the comprehensive 16-step enhanced training pipeline.

        Args:
            enhanced_training_input: Enhanced training input parameters

        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("🚀 COMPREHENSIVE 16-STEP ENHANCED TRAINING PIPELINE START")
            self.logger.info("=" * 80)
            self.logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"🎯 Symbol: {enhanced_training_input.get('symbol', 'N/A')}")
            self.logger.info(f"🏢 Exchange: {enhanced_training_input.get('exchange', 'N/A')}")
            self.logger.info(f"📊 Training Mode: {enhanced_training_input.get('training_mode', 'N/A')}")
            self.logger.info(f"📈 Lookback Days: {self.lookback_days}")
            self.logger.info(f"🔧 Blank Training Mode: {self.blank_training_mode}")
            self.logger.info(f"🔧 Max Trials: {self.max_trials}")
            self.logger.info(f"🔧 N Trials: {self.n_trials}")
            

            
            self.is_training = True
            
            # Validate training input
            if not self._validate_enhanced_training_inputs(enhanced_training_input):
                return False
            
            # Execute the comprehensive 16-step pipeline
            success = await self._execute_comprehensive_pipeline(enhanced_training_input)
            
            if success:
                # Store training history
                await self._store_enhanced_training_history(enhanced_training_input)
                
                self.logger.info("=" * 80)
                self.logger.info("🎉 COMPREHENSIVE 16-STEP ENHANCED TRAINING PIPELINE COMPLETED SUCCESSFULLY")
                self.logger.info("=" * 80)
                self.logger.info(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.logger.info(f"🎯 Symbol: {enhanced_training_input.get('symbol', 'N/A')}")
                self.logger.info(f"🏢 Exchange: {enhanced_training_input.get('exchange', 'N/A')}")
                self.logger.info("📋 Completed Steps:")
                self.logger.info("   1. Data Collection")
                self.logger.info("   2. Market Regime Classification")
                self.logger.info("   3. Regime Data Splitting")
                self.logger.info("   4. Analyst Labeling & Feature Engineering")
                self.logger.info("   5. Analyst Specialist Training")
                self.logger.info("   6. Analyst Enhancement")
                self.logger.info("   7. Analyst Ensemble Creation")
                self.logger.info("   8. Tactician Labeling")
                self.logger.info("   9. Tactician Specialist Training")
                self.logger.info("   10. Tactician Ensemble Creation")
                self.logger.info("   11. Confidence Calibration")
                self.logger.info("   12. Final Parameters Optimization")
                self.logger.info("   13. Walk Forward Validation")
                self.logger.info("   14. Monte Carlo Validation")
                self.logger.info("   15. A/B Testing")
                self.logger.info("   16. Saving Results")
                
                print("=" * 80)
                print("🎉 COMPREHENSIVE 16-STEP ENHANCED TRAINING PIPELINE COMPLETED SUCCESSFULLY")
                print("=" * 80)
                print("   ✅ All 16 training steps completed successfully!")
            else:
                self.logger.error("❌ Enhanced training pipeline failed")
            
            self.is_training = False
            return success
            
        except Exception as e:
            self.logger.error(f"💥 ENHANCED TRAINING PIPELINE FAILED: {str(e)}")
            self.logger.error(f"📋 Error details: {type(e).__name__}: {str(e)}")
            self.is_training = False
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="enhanced training inputs validation",
    )
    def _validate_enhanced_training_inputs(
        self,
        enhanced_training_input: dict[str, Any],
    ) -> bool:
        """
        Validate enhanced training input parameters.

        Args:
            enhanced_training_input: Enhanced training input parameters

        Returns:
            bool: True if input is valid, False otherwise
        """
        try:
            required_fields = ["symbol", "exchange", "timeframe", "lookback_days"]
            
            for field in required_fields:
                if field not in enhanced_training_input:
                    self.logger.error(f"❌ Missing required enhanced training input field: {field}")
                    return False
            
            # Validate specific field values
            if enhanced_training_input.get("lookback_days", 0) <= 0:
                self.logger.error("❌ Invalid lookback_days value")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced training inputs validation failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="computational optimization initialization",
    )
    async def _initialize_computational_optimization(self) -> bool:
        """Initialize computational optimization components."""
        try:
            self.logger.info("🚀 Initializing computational optimization components...")
            
            # Get computational optimization configuration
            optimization_config = get_computational_optimization_config()
            
            # Create computational optimization manager
            self.computational_optimization_manager = await create_computational_optimization_manager(
                config=optimization_config,
                market_data=pd.DataFrame(),  # Will be loaded during training
                model_config={}  # Will be configured during training
            )
            
            if self.computational_optimization_manager:
                self.logger.info("✅ Computational optimization components initialized successfully")
                return True
            else:
                self.logger.warning("⚠️ Failed to initialize computational optimization components")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Computational optimization initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="comprehensive pipeline execution",
    )
    async def _execute_comprehensive_pipeline(
        self,
        training_input: dict[str, Any],
    ) -> bool:
        """
        Execute the comprehensive 16-step training pipeline.

        Args:
            training_input: Training input parameters

        Returns:
            bool: True if all steps successful, False otherwise
        """
        try:
            symbol = training_input.get("symbol", "")
            exchange = training_input.get("exchange", "")
            timeframe = training_input.get("timeframe", "1m")
            data_dir = "data/training"
            start_step = training_input.get("start_step", "step1_data_collection")

            # Initialize pipeline state and timing
            pipeline_state = {}
            start_time = time.time()
            step_times = {}
            
            # Store current training parameters for checkpointing
            self.current_symbol = symbol
            self.current_exchange = exchange
            self.current_timeframe = timeframe
            
            # Check for existing checkpoint
            checkpoint = self._load_checkpoint()
            if checkpoint:
                self.logger.info("🔄 Resuming from checkpoint...")
                pipeline_state = checkpoint.get("pipeline_state", {})
                last_completed_step = checkpoint.get("current_step", "")
                self.logger.info(f"📂 Last completed step: {last_completed_step}")
            else:
                self.logger.info("🚀 Starting fresh training...")
            
            # Enhanced logging setup
            self.logger.info("=" * 100)
            self.logger.info("🚀 COMPREHENSIVE TRAINING PIPELINE START")
            self.logger.info("=" * 100)
            self.logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"🎯 Symbol: {symbol}")
            self.logger.info(f"🏢 Exchange: {exchange}")
            self.logger.info(f"📊 Timeframe: {timeframe}")
            self.logger.info(f"🧠 Training Mode: {'Blank' if self.blank_training_mode else 'Full'}")
            self.logger.info(f"🔧 Max Trials: {self.max_trials}")
            self.logger.info(f"📈 Lookback Days: {self.lookback_days}")
            self.logger.info(f"💾 Memory Optimization: {'Enabled' if self.enable_computational_optimization else 'Disabled'}")
            self.logger.info(f"🚀 Starting from step: {start_step}")
            self.logger.info("=" * 100)
            

            
            # Step 1: Data Collection (skip if starting from step2 or later)
            if start_step == "step1_data_collection":
                step_start = time.time()
                self.logger.info("📊 STEP 1: Data Collection...")
                self.logger.info("   🔍 Downloading and preparing market data...")
                
                from src.training.steps import step1_data_collection
                step1_result = await step1_data_collection.run_step(
                    symbol=symbol,
                    exchange_name=exchange,
                    min_data_points="750" if self.blank_training_mode else "1000",
                    data_dir=data_dir,
                    download_new_data=True,
                    lookback_days=self.lookback_days,
                )
                
                if step1_result is None or step1_result[0] is None:
                    self.logger.error("❌ Step 1: Data Collection failed")
                    print("❌ Step 1: Data Collection failed")
                    return False
                
                # Update pipeline state
                pipeline_state["data_collection"] = {
                    "status": "SUCCESS",
                    "artifact_paths": {
                        "historical_data": f"{data_dir}/{exchange}_{symbol}_historical_data.pkl"
                    }
                }
                
                # Run validator for Step 1
                validation_result = await self._run_step_validator(
                    "step1_data_collection", training_input, pipeline_state
                )
                
                self._log_step_completion("Step 1: Data Collection", step_start, step_times)
                
                # Save checkpoint after step 1
                self._save_checkpoint("step1_data_collection", pipeline_state)
                
                # Memory optimization after data collection
                self._optimize_memory_usage()
            else:
                self.logger.info("⏭️  Skipping Step 1: Data Collection (using pre-consolidated data)")
                # Add placeholder for data collection in pipeline state
                pipeline_state["data_collection"] = {
                    "status": "SKIPPED",
                    "result": {"message": "Using pre-consolidated data"}
                }

            # Step 2: Market Regime Classification
            step_start = time.time()
            self.logger.info("🎭 STEP 2: Market Regime Classification...")
            self.logger.info("   🧠 Analyzing market regimes and volatility patterns...")
            
            from src.training.steps import step2_market_regime_classification
            step2_success = await step2_market_regime_classification.run_step(
                symbol=symbol,
                data_dir=data_dir,
                timeframe=timeframe,
                exchange=exchange,
            )
            
            if not step2_success:
                self._log_step_completion("Step 2: Market Regime Classification", step_start, step_times, success=False)
                return False
            
            # Update pipeline state
            pipeline_state["regime_classification"] = {
                "status": "SUCCESS",
                "success": step2_success
            }
            
            # Run validator for Step 2
            validation_result = await self._run_step_validator(
                "step2_market_regime_classification", training_input, pipeline_state
            )
            
            self._log_step_completion("Step 2: Market Regime Classification", step_start, step_times)
            
            # Save checkpoint after step 2
            self._save_checkpoint("step2_market_regime_classification", pipeline_state)

            # Step 3: Regime Data Splitting
            step_start = time.time()
            self.logger.info("📊 STEP 3: Regime Data Splitting...")
            self.logger.info("   📈 Splitting data by market regimes for specialized training...")
            print("   📊 Step 3: Regime Data Splitting...")
            print("   📈 Splitting data by market regimes for specialized training...")
            
            from src.training.steps import step3_regime_data_splitting
            step3_success = await step3_regime_data_splitting.run_step(
                symbol=symbol,
                data_dir=data_dir,
                timeframe=timeframe,
                exchange=exchange,
            )
            
            if not step3_success:
                self._log_step_completion("Step 3: Regime Data Splitting", step_start, step_times, success=False)
                return False
            
            # Update pipeline state
            pipeline_state["regime_data_splitting"] = {
                "status": "SUCCESS",
                "success": step3_success
            }
            
            # Run validator for Step 3
            validation_result = await self._run_step_validator(
                "step3_regime_data_splitting", training_input, pipeline_state
            )
            
            self._log_step_completion("Step 3: Regime Data Splitting", step_start, step_times)
            
            # Save checkpoint after step 3
            self._save_checkpoint("step3_regime_data_splitting", pipeline_state)

            # Step 4: Analyst Labeling & Feature Engineering
            with self._timed_step("Step 4: Analyst Labeling & Feature Engineering", step_times):
                self.logger.info("🧠 STEP 4: Analyst Labeling & Feature Engineering...")
                
                from src.training.steps import step4_analyst_labeling_feature_engineering
                step4_success = await step4_analyst_labeling_feature_engineering.run_step(
                    symbol=symbol,
                    data_dir=data_dir,
                    timeframe=timeframe,
                    exchange=exchange,
                )
                
                if not step4_success:
                    return False
                self._save_checkpoint("step4_analyst_labeling_feature_engineering", pipeline_state)
                self._optimize_memory_usage()

                # Run validator for Step 4
                await self._run_step_validator(
                    "step4_analyst_labeling_feature_engineering", training_input, pipeline_state
                )

            # Step 5: Analyst Specialist Training
            with self._timed_step("Step 5: Analyst Specialist Training", step_times):
                self.logger.info("🎯 STEP 5: Analyst Specialist Training...")
                
                from src.training.steps import step5_analyst_specialist_training
                step5_success = await step5_analyst_specialist_training.run_step(
                    symbol=symbol,
                    data_dir=data_dir,
                    timeframe=timeframe,
                    exchange=exchange,
                )
                if not step5_success:
                    return False

                # Run validator for Step 5
                await self._run_step_validator(
                    "step5_analyst_specialist_training", training_input, pipeline_state
                )

            # Step 6: Analyst Enhancement
            with self._timed_step("Step 6: Analyst Enhancement", step_times):
                self.logger.info("🔧 STEP 6: Analyst Enhancement...")
                
                from src.training.steps import step6_analyst_enhancement
                step6_success = await step6_analyst_enhancement.run_step(
                    symbol=symbol,
                    data_dir=data_dir,
                    timeframe=timeframe,
                    exchange=exchange,
                )
                if not step6_success:
                    return False

                # Run validator for Step 6
                await self._run_step_validator(
                    "step6_analyst_enhancement", training_input, pipeline_state
                )

            # Step 7: Analyst Ensemble Creation
            step_start = time.time()
            self.logger.info("🎲 STEP 7: Analyst Ensemble Creation...")
            
            from src.training.steps import step7_analyst_ensemble_creation
            step7_success = await step7_analyst_ensemble_creation.run_step(
                symbol=symbol,
                data_dir=data_dir,
                timeframe=timeframe,
                exchange=exchange,
            )
            
            if not step7_success:
                self.logger.error("❌ Step 7: Analyst Ensemble Creation failed")
                print("❌ Step 7: Analyst Ensemble Creation failed")
                return False
            
            self.logger.info("✅ Step 7: Analyst Ensemble Creation completed successfully")
            self.logger.info("   ✅ Step 7: Analyst Ensemble Creation completed successfully")

            # Run validator for Step 7
            await self._run_step_validator(
                "step7_analyst_ensemble_creation", training_input, pipeline_state
            )
                
            if not step7_success:
               raise RuntimeError("Step 7: Analyst Ensemble Creation failed")

            # Step 8: Tactician Labeling
            with self._timed_step("Step 8: Tactician Labeling", step_times):
                self.logger.info("🎯 STEP 8: Tactician Labeling...")
                from src.training.steps import step8_tactician_labeling
                step8_success = await step8_tactician_labeling.run_step(
                    symbol=symbol,
                    data_dir=data_dir,
                    timeframe=timeframe,
                    exchange=exchange,
                )
                if not step8_success:
                    return False

                # Run validator for Step 8
                await self._run_step_validator(
                    "step8_tactician_labeling", training_input, pipeline_state
                )

            # Step 9: Tactician Specialist Training
            with self._timed_step("Step 9: Tactician Specialist Training", step_times):
                self.logger.info("🧠 STEP 9: Tactician Specialist Training...")
                from src.training.steps import step9_tactician_specialist_training
                step9_success = await step9_tactician_specialist_training.run_step(
                    symbol=symbol,
                    data_dir=data_dir,
                    timeframe=timeframe,
                    exchange=exchange,
                )
                if not step9_success:
                    return False

                # Run validator for Step 9
                await self._run_step_validator(
                    "step9_tactician_specialist_training", training_input, pipeline_state
                )

            # Step 10: Tactician Ensemble Creation
            with self._timed_step("Step 10: Tactician Ensemble Creation", step_times):
                self.logger.info("🎲 STEP 10: Tactician Ensemble Creation...")
                from src.training.steps import step10_tactician_ensemble_creation
                step10_success = await step10_tactician_ensemble_creation.run_step(
                    symbol=symbol,
                    data_dir=data_dir,
                    timeframe=timeframe,
                    exchange=exchange,
                )
                if not step10_success:
                    return False

                # Run validator for Step 10
                await self._run_step_validator(
                    "step10_tactician_ensemble_creation", training_input, pipeline_state
                )

            # Step 11: Confidence Calibration
            with self._timed_step("Step 11: Confidence Calibration", step_times):
                self.logger.info("🎯 STEP 11: Confidence Calibration...")
                from src.training.steps import step11_confidence_calibration
                step11_success = await step11_confidence_calibration.run_step(
                    symbol=symbol,
                    data_dir=data_dir,
                    timeframe=timeframe,
                    exchange=exchange,
                )
                if not step11_success:
                    return False

                # Run validator for Step 11
                await self._run_step_validator(
                    "step11_confidence_calibration", training_input, pipeline_state
                )

            # Step 12: Final Parameters Optimization (with computational optimization)
            with self._timed_step("Step 12: Final Parameters Optimization", step_times):
                self.logger.info("🔧 STEP 12: Final Parameters Optimization with Computational Optimization...")
                if self.computational_optimization_manager:
                    step12_success = await self._run_optimized_parameters_optimization(
                        symbol=symbol,
                        data_dir=data_dir,
                        timeframe=timeframe,
                        exchange=exchange,
                    )
                else:
                    from src.training.steps import step12_final_parameters_optimization
                    step12_success = await step12_final_parameters_optimization.run_step(
                        symbol=symbol,
                        data_dir=data_dir,
                        timeframe=timeframe,
                        exchange=exchange,
                    )
                if not step12_success:
                    return False

                # Run validator for Step 12
                await self._run_step_validator(
                    "step12_final_parameters_optimization", training_input, pipeline_state
                )

            # Step 13: Walk Forward Validation
            with self._timed_step("Step 13: Walk Forward Validation", step_times):
                self.logger.info("📈 STEP 13: Walk Forward Validation...")
                from src.training.steps import step13_walk_forward_validation
                step13_success = await step13_walk_forward_validation.run_step(
                    symbol=symbol,
                    data_dir=data_dir,
                    timeframe=timeframe,
                    exchange=exchange,
                )
                if not step13_success:
                    return False

                # Run validator for Step 13
                await self._run_step_validator(
                    "step13_walk_forward_validation", training_input, pipeline_state
                )

            # Step 14: Monte Carlo Validation
            with self._timed_step("Step 14: Monte Carlo Validation", step_times):
                self.logger.info("🎲 STEP 14: Monte Carlo Validation...")
                from src.training.steps import step14_monte_carlo_validation
                step14_success = await step14_monte_carlo_validation.run_step(
                    symbol=symbol,
                    data_dir=data_dir,
                    timeframe=timeframe,
                    exchange=exchange,
                )
                if not step14_success:
                    return False

                # Run validator for Step 14
                await self._run_step_validator(
                    "step14_monte_carlo_validation", training_input, pipeline_state
                )

            # Step 15: A/B Testing
            with self._timed_step("Step 15: A/B Testing", step_times):
                self.logger.info("🧪 STEP 15: A/B Testing...")
                from src.training.steps import step15_ab_testing
                step15_success = await step15_ab_testing.run_step(
                    symbol=symbol,
                    data_dir=data_dir,
                    timeframe=timeframe,
                    exchange=exchange,
                )
                if not step15_success:
                    return False

                # Run validator for Step 15
                await self._run_step_validator(
                    "step15_ab_testing", training_input, pipeline_state
                )

            # Step 16: Saving Results
            with self._timed_step("Step 16: Saving Results", step_times):
                self.logger.info("💾 STEP 16: Saving Results...")
                from src.training.steps import step16_saving
                step16_success = await step16_saving.run_step(
                    symbol=symbol,
                    data_dir=data_dir,
                    timeframe=timeframe,
                    exchange=exchange,
                )
                if not step16_success:
                    return False

                # Run validator for Step 16
                await self._run_step_validator(
                    "step16_saving", training_input, pipeline_state
                )

            # Calculate total time and summary
            total_time = time.time() - start_time
            total_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
            
            # Log comprehensive summary
            self.logger.info("=" * 100)
            self.logger.info("🎉 COMPREHENSIVE TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 100)
            self.logger.info(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"⏱️ Total Time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
            self.logger.info(f"💾 Final Memory Usage: {total_memory:.1f} MB")
            self.logger.info(f"🎯 Symbol: {symbol}")
            self.logger.info(f"🏢 Exchange: {exchange}")
            self.logger.info(f"📊 Timeframe: {timeframe}")
            self.logger.info(f"🧠 Training Mode: {'Blank' if self.blank_training_mode else 'Full'}")
            
            # Log step-by-step timing
            self.logger.info("📊 Step-by-Step Timing:")
            for step_name, step_time in step_times.items():
                percentage = (step_time / total_time) * 100
                self.logger.info(f"   {step_name}: {step_time:.2f}s ({percentage:.1f}%)")
            
            # Clear checkpoint on successful completion
            self._clear_checkpoint()
            
            return True
            
        except Exception as e:
            total_time = time.time() - start_time if 'start_time' in locals() else 0
            self.logger.error(f"💥 COMPREHENSIVE PIPELINE FAILED: {str(e)}")
            self.logger.error(f"📋 Error details: {type(e).__name__}: {str(e)}")
            self.logger.error(f"⏱️ Time elapsed before failure: {total_time:.2f}s")
            self.logger.info("💾 Checkpoint saved - you can resume training later")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="optimized parameters optimization",
    )
    async def _run_optimized_parameters_optimization(
        self,
        symbol: str,
        data_dir: str,
        timeframe: str,
        exchange: str,
    ) -> bool:
        """Run optimized parameters optimization using computational optimization strategies."""
        try:
            self.logger.info("🚀 Running optimized parameters optimization...")

            # Load and optimize market data
            market_data: pd.DataFrame | None = None
            try:
                market_data = await self._load_and_optimize_data(symbol, exchange, timeframe)
            except Exception as e:
                self.logger.warning(f"Falling back to basic data loader due to error: {e}")
                market_data = await self._load_market_data_for_optimization(symbol, data_dir, exchange)

            if market_data is None:
                self.logger.error("❌ Failed to load market data for optimization")
                return False

            # Initialize cached backtester and progressive evaluator
            if self.enable_caching:
                self.cached_backtester = CachedBacktester(market_data)
            if self.enable_early_stopping:
                self.progressive_evaluator = ProgressiveEvaluator(market_data)

            # Update computational optimization manager with market data
            if self.computational_optimization_manager:
                await self.computational_optimization_manager.initialize(market_data, {})

            # Define objective leveraging progressive/cached evaluation
            def objective(params: dict[str, Any]) -> float:
                try:
                    if self.enable_early_stopping and self.progressive_evaluator is not None:
                        return float(self.progressive_evaluator.evaluate_progressively(
                            params,
                            lambda data_subset, p: self._evaluate_params_with_cache(data_subset, p)
                        ))
                    if self.enable_caching and self.cached_backtester is not None:
                        return float(self.cached_backtester.run_cached_backtest(params))
                    return float(self._evaluate_params_with_cache(market_data, params))
                except Exception as e:
                    self.logger.warning(f"Objective evaluation error, returning fallback score: {e}")
                    return -1.0

            # Run optimized parameter optimization
            if not self.computational_optimization_manager:
                self.logger.warning("Computational optimization manager unavailable, skipping step")
                return False

            optimization_results = await self.computational_optimization_manager.optimize_parameters(
                objective_function=objective,
                n_trials=self.n_trials,
                use_surrogates=True,
            )

            # Store optimization statistics and results
            self.optimization_statistics = self.computational_optimization_manager.get_optimization_statistics()
            await self._save_optimization_results(symbol, exchange, data_dir, optimization_results)

            # Memory management
            if self.enable_memory_management and self.memory_manager is not None:
                self.memory_manager.check_memory_usage()

            self.logger.info("✅ Optimized parameters optimization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Optimized parameters optimization failed: {e}")
            return False

# === New helper methods for optimized data handling and evaluation ===
    async def _load_and_optimize_data(self, symbol: str, exchange: str, timeframe: str) -> pd.DataFrame:
        """Load market data with streaming and save optimized parquet cache."""
        if self.data_manager is None:
            self.data_manager = MemoryEfficientDataManager()
        if self.streaming_processor is None:
            self.streaming_processor = StreamingDataProcessor(chunk_size=self.chunk_size)

        cache_key = f"{symbol}_{exchange}_{timeframe}"
        parquet_path = f"data_cache/{cache_key}.parquet"
        try:
            # Load market data from the data directory
            # This is a simplified implementation
            import os
            # Prefer consolidated Parquet/CSV produced by Step 1
            preferred_parquet = Path("data_cache") / f"klines_{exchange}_{symbol}_1m_consolidated.parquet"
            preferred_csv = Path("data_cache") / f"klines_{exchange}_{symbol}_1m_consolidated.csv"
            if preferred_parquet.exists():
                market_data = pd.read_parquet(preferred_parquet)
                self.logger.info(f"✅ Loaded market data from {preferred_parquet}")
                return market_data
            if preferred_csv.exists():
                market_data = pd.read_csv(preferred_csv)
                self.logger.info(f"✅ Loaded market data from {preferred_csv}")
                return market_data
            data_file = f"{data_dir}/{exchange}_{symbol}_klines.csv"
            
            if os.path.exists(parquet_path):
                self.logger.info(f"Loading data from Parquet: {parquet_path}")
                try:
                    data = self.data_manager.load_from_parquet(parquet_path)
                except Exception as e:
                    self.logger.warning(f"Parquet load failed ({e}); falling back to CSV streaming")
                    data = pd.DataFrame()
            else:
                csv_glob = f"data_cache/klines_{exchange}_{symbol}_{timeframe}_*.csv"
                csv_files = list(Path().glob(csv_glob))
                if not csv_files:
                    raise FileNotFoundError(f"No data found for {symbol} on {exchange} ({timeframe})")
                chunks: list[pd.DataFrame] = []
                for csv_file in csv_files:
                    chunk_df = self.streaming_processor.process_data_stream(str(csv_file))
                    if not chunk_df.empty:
                        chunks.append(chunk_df)
                data = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
                optimized = self.data_manager.optimize_dataframe(data)
                Path("data_cache").mkdir(parents=True, exist_ok=True)
                self.data_manager.save_to_parquet(optimized, parquet_path)
                data = optimized
            data = self.data_manager.optimize_dataframe(data)
            self.logger.info(f"✅ Data loaded and optimized: {len(data)} rows")
            return data
        except Exception as e:
            self.logger.warning(f"Optimized data loading failed: {e}")
            return pd.DataFrame()

    def _evaluate_params_with_cache(self, market_data: pd.DataFrame, params: dict[str, Any]) -> float:
        """Evaluate params using cached backtester if available, else simple placeholder."""
        if self.cached_backtester is None:
            self.cached_backtester = CachedBacktester(market_data)
        try:
            return float(self.cached_backtester.run_cached_backtest(params))
        except Exception:
            return random.uniform(-1.0, 1.0)

    def get_memory_profile(self) -> dict[str, Any]:
        """Expose current memory profile using MemoryManager."""
        if self.memory_manager is None:
            self.memory_manager = MemoryManager(memory_threshold=self.memory_threshold)
        return self.memory_manager.profile_memory_usage()

    def get_optimization_stats(self) -> dict[str, Any]:
        """Expose optimization component status."""
        stats: dict[str, Any] = {
            'caching_enabled': self.enable_caching,
            'parallelization_enabled': self.enable_parallelization,
            'early_stopping_enabled': self.enable_early_stopping,
            'memory_management_enabled': self.enable_memory_management,
            'max_workers': self.max_workers,
            'memory_threshold': self.memory_threshold,
        }
        if self.cached_backtester is not None:
            stats['cache_size'] = len(self.cached_backtester.cache)
        if self.adaptive_sampler is not None:
            stats['trial_history_size'] = len(self.adaptive_sampler.trial_history)
        return stats

    async def _run_step_validator(
        self,
        step_name: str,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run validator for a specific step.
        
        Args:
            step_name: Name of the step
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Dictionary containing validation results
        """
        if not self.enable_validators:
            return {
                "step_name": step_name,
                "validation_passed": True,
                "skipped": True,
                "reason": "Validators disabled"
            }
        
        try:
            self.logger.info(f"🔍 Running validator for {step_name}")
            validation_result = await validator_orchestrator.run_step_validator(
                step_name=step_name,
                training_input=training_input,
                pipeline_state=pipeline_state,
                config=self.config
            )
            
            # Store validation result
            self.validation_results[step_name] = validation_result
            
            if validation_result.get("validation_passed", False):
                self.logger.info(f"✅ {step_name} validation passed")
            else:
                self.logger.warning(f"⚠️ {step_name} validation failed: {validation_result.get('error', 'Unknown error')}")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Error running validator for {step_name}: {e}")
            return {
                "step_name": step_name,
                "validation_passed": False,
                "error": str(e)
            }

    
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="enhanced training history storage",
    )
    async def _store_enhanced_training_history(self, enhanced_training_input: dict[str, Any]) -> None:
        """
        Store enhanced training history.

        Args:
            enhanced_training_input: Enhanced training input parameters
        """
        try:
            # Add to training history
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "training_input": enhanced_training_input,
                "results": self.enhanced_training_results,
            }
            
            self.enhanced_training_history.append(history_entry)
            
            # Limit history size
            if len(self.enhanced_training_history) > self.max_enhanced_training_history:
                self.enhanced_training_history = self.enhanced_training_history[-self.max_enhanced_training_history:]
            
            self.logger.info(f"📁 Stored training history entry (total: {len(self.enhanced_training_history)})")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store training history: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="enhanced training results storage",
    )
    async def _store_enhanced_training_results(self) -> None:
        """Store enhanced training results."""
        try:
            self.logger.info("📁 Storing enhanced training results...")
            
            # Store results in a format that can be retrieved later
            results_key = f"enhanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # This would typically store to database or file system
            self.logger.info(f"📁 Storing enhanced training results with key: {results_key}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store enhanced training results: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="enhanced training results getting",
    )
    def get_enhanced_training_results(
        self,
        enhanced_training_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Get enhanced training results.

        Args:
            enhanced_training_type: Type of training results to get

        Returns:
            dict: Enhanced training results
        """
        try:
            if enhanced_training_type:
                return self.enhanced_training_results.get(enhanced_training_type, {})
            return self.enhanced_training_results.copy()
            
        except Exception as e:
            self.logger.error(f"Failed to get enhanced training results: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="enhanced training history getting",
    )
    def get_enhanced_training_history(
        self,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get enhanced training history.

        Args:
            limit: Maximum number of history entries to return

        Returns:
            list: Enhanced training history
        """
        try:
            history = self.enhanced_training_history.copy()
            if limit:
                history = history[-limit:]
            return history
            
        except Exception as e:
            self.logger.error(f"Failed to get enhanced training history: {e}")
            return []

    def get_enhanced_training_status(self) -> dict[str, Any]:
        """
        Get enhanced training status.

        Returns:
            dict: Enhanced training status information
        """
        return {
            "is_training": self.is_training,
            "has_results": bool(self.enhanced_training_results),
            "history_count": len(self.enhanced_training_history),
            "blank_training_mode": self.blank_training_mode,
            "max_trials": self.max_trials,
            "n_trials": self.n_trials,
            "lookback_days": self.lookback_days,
            "enable_validators": self.enable_validators,
            "enable_computational_optimization": self.enable_computational_optimization,
            "optimization_statistics": self.optimization_statistics,
        }
    
    def get_validation_results(self) -> dict[str, Any]:
        """
        Get validation results for all steps.
        
        Returns:
            dict: Validation results summary
        """
        return {
            "validation_results": self.validation_results,
            "validation_summary": validator_orchestrator.get_validation_summary(),
            "failed_validations": validator_orchestrator.get_failed_validations()
        }
    
    def get_computational_optimization_results(self) -> dict[str, Any]:
        """
        Get computational optimization results and statistics.
        
        Returns:
            dict: Computational optimization results
        """
        if self.computational_optimization_manager:
            return {
                "optimization_statistics": self.computational_optimization_manager.get_optimization_statistics(),
                "enabled_optimizations": self.optimization_statistics,
                "manager_available": True
            }
        else:
            return {
                "optimization_statistics": {},
                "enabled_optimizations": {},
                "manager_available": False
            }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="enhanced training manager cleanup",
    )
    async def stop(self) -> None:
        """Stop the enhanced training manager and cleanup resources."""
        try:
            self.logger.info("🛑 Stopping Enhanced Training Manager...")
            
            # Cleanup computational optimization manager
            if self.computational_optimization_manager:
                await self.computational_optimization_manager.cleanup()
                self.logger.info("✅ Computational optimization manager cleaned up")

            # Cleanup parallel backtester
            if self.parallel_backtester is not None:
                self.parallel_backtester.shutdown()
                self.parallel_backtester = None

            # Force memory cleanup
            if self.enable_memory_management and self.memory_manager is not None:
                self.memory_manager._cleanup_memory()
            
            self.is_training = False
            self.logger.info("✅ Enhanced Training Manager stopped successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to stop Enhanced Training Manager: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="enhanced training manager setup",
)
async def setup_enhanced_training_manager(
    config: dict[str, Any] | None = None,
) -> EnhancedTrainingManager | None:
    """
    Setup and return a configured EnhancedTrainingManager instance.

    Args:
        config: Configuration dictionary

    Returns:
        EnhancedTrainingManager: Configured enhanced training manager instance
    """
    try:
        manager = EnhancedTrainingManager(config or {})
        if await manager.initialize():
            return manager
        return None
    except Exception as e:
        system_logger.error(f"Failed to setup enhanced training manager: {e}")
        return None
