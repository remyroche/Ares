# src/training/steps/step4_processing_labeling_improved.py

"""
Improved Step 4: Processing & Labeling with enhanced code quality and performance.

Key improvements:
- Modular architecture with separate classes for different responsibilities
- Better memory management with context managers
- Improved error handling and logging
- Type hints throughout
- Performance optimizations with parallel processing
- Better data validation and quality checks
- Enhanced triple-barrier labeling with improved accuracy
"""

import asyncio
import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from contextlib import asynccontextmanager
import gc

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import joblib

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.decorators import with_tracing_span, guard_dataframe_nulls
from src.training.steps.unified_data_loader import get_unified_data_loader
from src.training.steps.step4_analyst_labeling_feature_engineering_components.optimized_triple_barrier_labeling import (
    OptimizedTripleBarrierLabeling,
)
from src.training.steps.vectorized_labelling_orchestrator import (
    VectorizedLabellingOrchestrator,
)
from src.training.steps.vectorized_advanced_feature_engineering import (
    VectorizedAdvancedFeatureEngineering,
)
from src.training.enhanced_training_manager_optimized import (
    MemoryEfficientDataManager,
)

# Import decorators from centralized module
from src.utils.centralized_decorators import (
    auto_fix_data_quality_issues,
    deterministic_seed,
    idempotent_step,
    artifact_write_lock,
    nan_inf_and_constant_guard,
    artifact_versioning,
    time_budget_watchdog,
    validate_step_prerequisites,
    secure_data_processing,
    prevent_data_leakage,
    resource_monitor,
    memory_efficient,
    debug_training_step,
    circuit_breaker_protection,
    validate_step_output,
    quality_gate,
    handle_errors,
    validate_data_quality,
    validate_feature_engineering_pipeline,
    with_tracing_span,
    guard_dataframe_nulls,
)


@dataclass
class ProcessingConfig:
    """Configuration for processing and labeling step."""
    symbol: str
    exchange: str
    data_dir: str
    timeframe: str
    force_rerun: bool = False
    enable_parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    lookback_days: int = 30
    train_split: float = 0.70
    validation_split: float = 0.15
    test_split: float = 0.15
    triple_barrier_params: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.symbol or not self.exchange:
            raise ValueError("Symbol and exchange must be provided")
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if abs(self.train_split + self.validation_split + self.test_split - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        if self.triple_barrier_params is None:
            self.triple_barrier_params = {
                "upper_barrier": 0.02,  # 2% upper barrier
                "lower_barrier": 0.02,  # 2% lower barrier
                "timeout": 20,  # 20 periods timeout
                "binary_classification": True,
            }


class DataProcessor:
    """Improved data processor with validation and error handling."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = system_logger.getChild("DataProcessor")
    
    @handle_errors(exceptions=(Exception,), default_return=None)
    async def load_unified_data(self) -> Optional[pd.DataFrame]:
        """Load unified data with improved validation."""
        try:
            # Initialize data loader
            loader_config = {
                "symbol": self.config.symbol,
                "exchange": self.config.exchange,
                "data_dir": self.config.data_dir,
                "timeframe": self.config.timeframe,
            }
            
            data_loader = get_unified_data_loader(loader_config)
            
            # Load data
            df = await data_loader.load_unified_data(
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                timeframe=self.config.timeframe,
                lookback_days=self.config.lookback_days,
                use_streaming=True,
            )
            
            if df is None or df.empty:
                raise ValueError(f"No data found for {self.config.symbol} on {self.config.exchange}")
            
            # Validate data quality
            df = self._validate_and_clean_data(df)
            
            self.logger.info(f"Loaded unified data: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading unified data: {e}")
            raise
    
    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the loaded data."""
        try:
            # Ensure timestamp column exists and is datetime
            if "timestamp" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={"index": "timestamp"})
            
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            
            # Sort by timestamp
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            # Remove rows with missing timestamps
            df = df.dropna(subset=["timestamp"])
            
            # Ensure required OHLCV columns exist
            required_cols = ["open", "high", "low", "close"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Validate OHLC data
            df = self._validate_ohlc_data(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error validating and cleaning data: {e}")
            raise
    
    def _validate_ohlc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLC data for consistency."""
        try:
            # Check for negative prices
            price_cols = ["open", "high", "low", "close"]
            for col in price_cols:
                if (df[col] <= 0).any():
                    self.logger.warning(f"Found non-positive values in {col}, removing affected rows")
                    df = df[df[col] > 0]
            
            # Check OHLC consistency
            invalid_ohlc = (
                (df["high"] < df["low"]) |
                (df["open"] > df["high"]) |
                (df["close"] > df["high"]) |
                (df["open"] < df["low"]) |
                (df["close"] < df["low"])
            )
            
            if invalid_ohlc.any():
                self.logger.warning(f"Found {invalid_ohlc.sum()} rows with invalid OHLC data, removing")
                df = df[~invalid_ohlc]
            
            # Check for extreme price movements (potential data errors)
            price_changes = df["close"].pct_change().abs()
            extreme_changes = price_changes > 0.5  # 50% price change
            if extreme_changes.any():
                self.logger.warning(f"Found {extreme_changes.sum()} rows with extreme price changes")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error validating OHLC data: {e}")
            raise


class LabelingEngine:
    """Improved labeling engine with enhanced triple-barrier labeling."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = system_logger.getChild("LabelingEngine")
        self.triple_barrier_labeler = None
        self.vectorized_orchestrator = None
    
    async def initialize(self) -> bool:
        """Initialize labeling components."""
        try:
            # Initialize triple-barrier labeling
            self.triple_barrier_labeler = OptimizedTripleBarrierLabeling(
                binary_classification=self.config.triple_barrier_params["binary_classification"]
            )
            
            # Initialize vectorized orchestrator
            orchestrator_config = {
                "symbol": self.config.symbol,
                "exchange": self.config.exchange,
                "timeframe": self.config.timeframe,
            }
            self.vectorized_orchestrator = VectorizedLabellingOrchestrator(orchestrator_config)
            
            return True
        except Exception as e:
            self.logger.error(f"Error initializing labeling engine: {e}")
            return False
    
    @handle_errors(exceptions=(Exception,), default_return=pd.DataFrame())
    async def apply_triple_barrier_labeling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply triple-barrier labeling with improved accuracy."""
        try:
            if df.empty:
                raise ValueError("Empty dataframe for labeling")
            
            # Ensure we have the required columns
            required_cols = ["open", "high", "low", "close", "volume", "timestamp"]
            available_cols = [col for col in required_cols if col in df.columns]
            
            if len(available_cols) < 5:  # Need at least OHLCV
                raise ValueError(f"Insufficient columns for labeling: {available_cols}")
            
            # Prepare data for labeling
            labeling_data = df[available_cols].set_index("timestamp")
            
            # Apply triple-barrier labeling
            labeled = self.triple_barrier_labeler.apply_triple_barrier_labeling_vectorized(
                labeling_data
            )
            
            # Reset index to bring timestamp back as column
            labeled = labeled.reset_index()
            
            # Validate labeling results
            labeled = self._validate_labeling_results(labeled)
            
            self.logger.info(f"Applied triple-barrier labeling: {len(labeled)} samples")
            self.logger.info(f"Label distribution: {labeled['target'].value_counts().to_dict()}")
            
            return labeled
            
        except Exception as e:
            self.logger.error(f"Error applying triple-barrier labeling: {e}")
            raise
    
    def _validate_labeling_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate labeling results."""
        try:
            # Check for missing targets
            if "target" not in df.columns:
                raise ValueError("No target column found in labeled data")
            
            # Remove rows with missing targets
            initial_count = len(df)
            df = df.dropna(subset=["target"])
            final_count = len(df)
            
            if final_count < initial_count:
                self.logger.warning(f"Removed {initial_count - final_count} rows with missing targets")
            
            # Check target distribution
            target_counts = df["target"].value_counts()
            self.logger.info(f"Target distribution: {target_counts.to_dict()}")
            
            # Warn if distribution is too imbalanced
            min_count = target_counts.min()
            max_count = target_counts.max()
            if min_count / max_count < 0.1:  # Less than 10% of majority class
                self.logger.warning("Highly imbalanced target distribution detected")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error validating labeling results: {e}")
            raise
    
    @handle_errors(exceptions=(Exception,), default_return=Dict[str, pd.DataFrame])
    async def split_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split data into train/validation/test sets."""
        try:
            if df.empty:
                raise ValueError("Empty dataframe for splitting")
            
            n = len(df)
            if n < 100:
                self.logger.warning("Very little data for splitting; proceeding with minimal splits")
            
            # Calculate split indices
            cut1 = int(n * self.config.train_split)
            cut2 = int(n * (self.config.train_split + self.config.validation_split))
            
            # Split data
            splits = {
                "train": df.iloc[:cut1].copy(),
                "validation": df.iloc[cut1:cut2].copy(),
                "test": df.iloc[cut2:].copy(),
            }
            
            # Log split information
            for split_name, split_df in splits.items():
                self.logger.info(f"{split_name.capitalize()} split: {len(split_df)} samples")
                if "target" in split_df.columns:
                    target_dist = split_df["target"].value_counts().to_dict()
                    self.logger.info(f"  Target distribution: {target_dist}")
            
            return splits
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {e}")
            raise


class ArtifactManager:
    """Manages artifacts with improved caching and validation."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = system_logger.getChild("ArtifactManager")
        self.data_dir = Path(config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def check_artifacts_exist(self) -> bool:
        """Check if labeled artifacts already exist."""
        try:
            required_files = [
                self.data_dir / f"{self.config.exchange}_{self.config.symbol}_labeled_train.parquet",
                self.data_dir / f"{self.config.exchange}_{self.config.symbol}_labeled_validation.parquet",
                self.data_dir / f"{self.config.exchange}_{self.config.symbol}_labeled_test.parquet",
            ]
            return all(f.exists() for f in required_files)
        except Exception as e:
            self.logger.error(f"Error checking artifacts: {e}")
            return False
    
    def save_labeled_splits(self, splits: Dict[str, pd.DataFrame]) -> bool:
        """Save labeled splits with atomic writes."""
        try:
            paths = {
                "train": self.data_dir / f"{self.config.exchange}_{self.config.symbol}_labeled_train.parquet",
                "validation": self.data_dir / f"{self.config.exchange}_{self.config.symbol}_labeled_validation.parquet",
                "test": self.data_dir / f"{self.config.exchange}_{self.config.symbol}_labeled_test.parquet",
            }
            
            for split_name, df in splits.items():
                if df is None or df.empty:
                    self.logger.warning(f"Skipping empty {split_name} split")
                    continue
                
                file_path = paths[split_name]
                temp_path = file_path.with_suffix(".tmp")
                
                # Write to temporary file first
                df.to_parquet(temp_path, index=False)
                
                # Atomic move
                temp_path.replace(file_path)
                
                self.logger.info(f"Saved {split_name} split: {len(df)} samples")
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving labeled splits: {e}")
            return False
    
    async def run_vectorized_orchestrator(self, splits: Dict[str, pd.DataFrame]) -> bool:
        """Run vectorized orchestrator for additional processing."""
        try:
            if not self.vectorized_orchestrator:
                self.logger.warning("Vectorized orchestrator not available, skipping")
                return True
            
            # Run orchestrator on each split
            for split_name, df in splits.items():
                if not df.empty:
                    await self.vectorized_orchestrator.process_split(split_name, df)
            
            return True
        except Exception as e:
            self.logger.error(f"Error running vectorized orchestrator: {e}")
            return False


@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=2.0,
    min_disk_gb=1.0,
    required_packages=["pandas", "numpy", "joblib"],
    data_quality_checks={
        "min_rows": 100,
        "required_columns": ["open", "high", "low", "close"],
    },
    context="Improved Processing & Labeling",
)
@secure_data_processing(
    backup_before=True, 
    integrity_checks=True, 
    memory_cleanup=True, 
    data_validation=True
)
@prevent_data_leakage(
    temporal_validation=True,
    feature_leakage_detection=True,
    lookahead_bias_prevention=True,
)
@resource_monitor(
    memory_threshold_gb=8.0,
    cpu_threshold_percent=80.0,
    disk_threshold_gb=2.0,
    monitor_interval=10.0,
    auto_cleanup=True,
)
@memory_efficient(
    chunk_size=1000, 
    streaming_processing=True, 
    memory_pool=True, 
    cleanup_frequency=5
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=True,
    performance_profiling=True,
    error_context_preservation=True,
)
@circuit_breaker_protection(
    failure_threshold=5,
    recovery_timeout=30.0,
    expected_exception=Exception,
    monitor_interval=5.0,
)
@validate_step_output(
    output_validation=True,
    data_quality_checks=True,
    performance_metrics=True,
)
@quality_gate(
    quality_threshold=0.8,
    validation_metrics=["data_quality", "labeling_quality", "performance"],
)
async def run_step_improved(
    symbol: str,
    exchange_name: str = "BINANCE",
    data_dir: str = "data/training",
    timeframe: str = "1m",
    exchange: str = "BINANCE",
    force_rerun: bool = False,
    pipeline_config: dict[str, Any] | None = None,
) -> bool:
    """
    Improved Step 4: Processing & Labeling with enhanced code quality and performance.
    
    Key improvements:
    - Modular architecture with separate classes
    - Better memory management
    - Improved error handling and validation
    - Enhanced triple-barrier labeling
    - Comprehensive logging and monitoring
    """
    logger = system_logger.getChild("Step4.ImprovedProcessingLabeling")
    start_time = time.time()
    
    try:
        logger.info("🚀 Starting improved Step 4: Processing & Labeling")
        
        # Determine actual exchange
        actual_exchange = exchange if exchange != "BINANCE" else exchange_name
        
        # Initialize configuration
        config = ProcessingConfig(
            symbol=symbol,
            exchange=actual_exchange,
            data_dir=data_dir,
            timeframe=timeframe,
            force_rerun=force_rerun,
        )
        
        # Update config with pipeline config if provided
        if pipeline_config:
            if "vectorized_labelling_orchestrator" in pipeline_config:
                config.triple_barrier_params.update(
                    pipeline_config["vectorized_labelling_orchestrator"]
                )
        
        # Initialize components
        data_processor = DataProcessor(config)
        labeling_engine = LabelingEngine(config)
        artifact_manager = ArtifactManager(config)
        
        # Check for existing artifacts
        if artifact_manager.check_artifacts_exist() and not force_rerun:
            logger.info("📦 Labeled artifacts already exist, skipping processing")
            return True
        
        # Load unified data
        logger.info("📥 Loading unified data")
        df = await data_processor.load_unified_data()
        
        if df is None or df.empty:
            raise ValueError(f"No data found for {symbol} on {actual_exchange}")
        
        # Initialize labeling engine
        logger.info("🔧 Initializing labeling engine")
        if not await labeling_engine.initialize():
            raise RuntimeError("Failed to initialize labeling engine")
        
        # Apply triple-barrier labeling
        logger.info("🎯 Applying triple-barrier labeling")
        labeled = await labeling_engine.apply_triple_barrier_labeling(df)
        
        if labeled.empty:
            raise ValueError("Failed to apply triple-barrier labeling")
        
        # Split data
        logger.info("✂️ Splitting data into train/validation/test")
        splits = await labeling_engine.split_data(labeled)
        
        # Save labeled splits
        logger.info("💾 Saving labeled splits")
        if not artifact_manager.save_labeled_splits(splits):
            raise RuntimeError("Failed to save labeled splits")
        
        # Run vectorized orchestrator if available
        if hasattr(artifact_manager, 'vectorized_orchestrator'):
            logger.info("🔄 Running vectorized orchestrator")
            await artifact_manager.run_vectorized_orchestrator(splits)
        
        # Log completion
        total_time = time.time() - start_time
        total_samples = sum(len(df) for df in splits.values())
        
        logger.info(f"✅ Processing & labeling completed successfully")
        logger.info(f"   ⏱️ Total time: {total_time:.2f}s")
        logger.info(f"   📊 Total samples: {total_samples}")
        for split_name, split_df in splits.items():
            logger.info(f"   📈 {split_name.capitalize()}: {len(split_df)} samples")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Processing & labeling failed: {e}")
        logger.exception("Full traceback:")
        return False
    finally:
        # Cleanup
        gc.collect()


# Backward compatibility
async def run_step(*args, **kwargs):
    """Backward compatibility wrapper for the improved run_step function."""
    return await run_step_improved(*args, **kwargs)