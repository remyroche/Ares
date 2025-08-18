# src/training/steps/step2_feature_engineering_improved.py

"""
Improved Step 2: Feature Engineering with enhanced code quality and performance.

Key improvements:
- Modular architecture with separate classes for different responsibilities
- Better memory management with context managers
- Improved error handling and logging
- Type hints throughout
- Performance optimizations with parallel processing
- Better data validation and quality checks
"""

import asyncio
import os
import json
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from contextlib import asynccontextmanager
import gc

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import joblib

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.decorators import with_tracing_span, guard_dataframe_nulls
from src.training.steps.raw_data_quality_checker import auto_fix_data_quality_issues
from src.training.optimized_feature_selection_manager import OptimizedFeatureSelectionManager
from src.tactician.sr_breakout_predictor import setup_sr_breakout_predictor
from src.training.steps.vectorized_advanced_feature_engineering import VectorizedAdvancedFeatureEngineering
from src.training.enhanced_training_manager_optimized import MemoryEfficientDataManager

# Import training pipeline decorators
from src.utils.training_pipeline_decorators import (
    validate_step_prerequisites,
    secure_data_processing,
    prevent_data_leakage,
    resource_monitor,
    memory_efficient,
    debug_training_step,
    circuit_breaker_protection,
    validate_step_output,
    quality_gate,
)


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering step."""
    symbol: str
    exchange: str
    data_dir: str
    timeframe: str
    force_rerun: bool = False
    enable_parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    enable_feature_caching: bool = True
    cache_dir: str = "data/feature_cache"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.symbol or not self.exchange:
            raise ValueError("Symbol and exchange must be provided")
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if self.memory_limit_gb < 1.0:
            raise ValueError("memory_limit_gb must be at least 1.0")


class FeatureArtifactManager:
    """Manages feature artifacts with improved caching and validation."""
    
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.logger = system_logger.getChild("FeatureArtifactManager")
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_artifact_hash(self) -> str:
        """Generate deterministic hash for feature artifacts."""
        try:
            data_dir = Path(self.config.data_dir)
            labeled_paths = [
                data_dir / f"{self.config.exchange}_{self.config.symbol}_labeled_train.parquet",
                data_dir / f"{self.config.exchange}_{self.config.symbol}_labeled_validation.parquet",
                data_dir / f"{self.config.exchange}_{self.config.symbol}_labeled_test.parquet",
            ]
            
            hash_input = f"{self.config.symbol}_{self.config.exchange}_{self.config.timeframe}"
            for path in labeled_paths:
                if path.exists():
                    stat = path.stat()
                    hash_input += f"_{stat.st_mtime}_{stat.st_size}"
            
            return hashlib.md5(hash_input.encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"Error generating artifact hash: {e}")
            return hashlib.md5(f"{self.config.symbol}_{self.config.exchange}_{self.config.timeframe}".encode()).hexdigest()
    
    def check_artifacts_exist(self) -> bool:
        """Check if feature artifacts already exist."""
        try:
            data_dir = Path(self.config.data_dir)
            required_files = [
                data_dir / f"{self.config.exchange}_{self.config.symbol}_features_train.parquet",
                data_dir / f"{self.config.exchange}_{self.config.symbol}_features_validation.parquet",
                data_dir / f"{self.config.exchange}_{self.config.symbol}_features_test.parquet",
            ]
            return all(f.exists() for f in required_files)
        except Exception as e:
            self.logger.error(f"Error checking artifacts: {e}")
            return False
    
    def load_artifacts(self) -> Dict[str, pd.DataFrame]:
        """Load existing feature artifacts with validation."""
        try:
            data_dir = Path(self.config.data_dir)
            features = {}
            
            for split in ["train", "validation", "test"]:
                file_path = data_dir / f"{self.config.exchange}_{self.config.symbol}_features_{split}.parquet"
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    if not df.empty:
                        features[split] = df
                        self.logger.info(f"Loaded {split} features: {len(df)} rows, {len(df.columns)} features")
                    else:
                        self.logger.warning(f"Empty {split} features file")
                else:
                    self.logger.warning(f"Missing {split} features file")
            
            return features
        except Exception as e:
            self.logger.error(f"Error loading artifacts: {e}")
            return {}
    
    def save_artifacts(self, features: Dict[str, pd.DataFrame]) -> bool:
        """Save feature artifacts with atomic writes and validation."""
        try:
            data_dir = Path(self.config.data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
            
            for split, df in features.items():
                if df is None or df.empty:
                    self.logger.warning(f"Skipping empty {split} features")
                    continue
                
                file_path = data_dir / f"{self.config.exchange}_{self.config.symbol}_features_{split}.parquet"
                temp_path = file_path.with_suffix(".tmp")
                
                # Write to temporary file first
                df.to_parquet(temp_path, index=True)
                
                # Atomic move
                temp_path.replace(file_path)
                
                self.logger.info(f"Saved {split} features: {len(df)} rows, {len(df.columns)} features")
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving artifacts: {e}")
            return False


class DataLoader:
    """Improved data loading with validation and error handling."""
    
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.logger = system_logger.getChild("DataLoader")
    
    @handle_errors(exceptions=(Exception,), default_return={})
    async def load_labeled_data(self) -> Dict[str, pd.DataFrame]:
        """Load labeled data with improved validation."""
        try:
            data_dir = Path(self.config.data_dir)
            paths = {
                "train": data_dir / f"{self.config.exchange}_{self.config.symbol}_labeled_train.parquet",
                "validation": data_dir / f"{self.config.exchange}_{self.config.symbol}_labeled_validation.parquet",
                "test": data_dir / f"{self.config.exchange}_{self.config.symbol}_labeled_test.parquet",
            }
            
            labeled = {}
            for split_name, path in paths.items():
                if not path.exists():
                    raise FileNotFoundError(f"Missing labeled data file: {path}")
                
                df = pd.read_parquet(path)
                if df.empty:
                    raise ValueError(f"Empty labeled data for {split_name}")
                
                # Ensure timestamp column exists and is properly formatted
                df = self._normalize_timestamp_column(df)
                labeled[split_name] = df
                
                self.logger.info(f"Loaded {split_name}: {len(df)} rows")
            
            return labeled
        except Exception as e:
            self.logger.error(f"Error loading labeled data: {e}")
            raise
    
    def _normalize_timestamp_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize timestamp column for consistency."""
        try:
            # Handle case where timestamp is the index
            if "timestamp" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={"index": "timestamp"})
            
            # Ensure timestamp column exists
            if "timestamp" not in df.columns:
                raise ValueError("No timestamp column found in data")
            
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
            df = df.set_index("timestamp")
            
            return df
        except Exception as e:
            self.logger.error(f"Error normalizing timestamp: {e}")
            raise
    
    @staticmethod
    def extract_ohlcv_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract OHLCV data with validation."""
        try:
            price_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            if len(price_cols) < 4:  # Need at least OHLC
                raise ValueError("Missing required OHLC columns")
            
            price = df[price_cols].copy()
            volume = price[["volume"]].copy() if "volume" in price.columns else pd.DataFrame({"volume": 1.0}, index=price.index)
            
            return price, volume
        except Exception as e:
            raise ValueError(f"Error extracting OHLCV data: {e}")


class FeatureEngineer:
    """Improved feature engineering with parallel processing and memory management."""
    
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.logger = system_logger.getChild("FeatureEngineer")
        self.feature_engineering = None
        self.sr_predictor = None
    
    @asynccontextmanager
    async def _memory_context(self):
        """Context manager for memory cleanup."""
        try:
            yield
        finally:
            gc.collect()
    
    async def initialize(self) -> bool:
        """Initialize feature engineering components."""
        try:
            # Initialize vectorized feature engineering
            feature_config = {
                "vectorized_advanced_features": {
                    "enable_difference_acceleration_features": True,
                    "enable_volatility_modeling": True,
                    "enable_correlation_analysis": True,
                    "enable_momentum_analysis": True,
                    "enable_liquidity_analysis": True,
                    "enable_candlestick_patterns": True,
                    "enable_sr_distance": True,
                    "enable_wavelet_transforms": True,
                    "enable_multi_timeframe": True,
                    "enable_meta_labeling": False,
                    "enable_explicit_meta_labels": False,
                },
                "symbol": self.config.symbol,
                "exchange": self.config.exchange,
            }
            
            self.feature_engineering = VectorizedAdvancedFeatureEngineering(feature_config)
            await self.feature_engineering.initialize()
            
            # Initialize SR breakout predictor
            self.sr_predictor = await setup_sr_breakout_predictor(feature_config)
            
            return True
        except Exception as e:
            self.logger.error(f"Error initializing feature engineer: {e}")
            return False
    
    async def engineer_features_parallel(
        self, 
        labeled_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Engineer features with parallel processing."""
        try:
            features = {}
            
            if self.config.enable_parallel_processing:
                # Use ThreadPoolExecutor for I/O-bound operations
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    futures = {}
                    
                    for split_name, df in labeled_data.items():
                        future = executor.submit(
                            self._engineer_features_single_split, 
                            split_name, 
                            df
                        )
                        futures[future] = split_name
                    
                    # Collect results
                    for future in futures:
                        split_name = futures[future]
                        try:
                            features[split_name] = future.result()
                            self.logger.info(f"Completed feature engineering for {split_name}")
                        except Exception as e:
                            self.logger.error(f"Error in {split_name} feature engineering: {e}")
                            raise
            else:
                # Sequential processing
                for split_name, df in labeled_data.items():
                    features[split_name] = await self._engineer_features_single_split(split_name, df)
            
            return features
        except Exception as e:
            self.logger.error(f"Error in parallel feature engineering: {e}")
            raise
    
    async def _engineer_features_single_split(
        self, 
        split_name: str, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Engineer features for a single data split."""
        try:
            async with self._memory_context():
                # Extract OHLCV data
                price_df, volume_df = DataLoader.extract_ohlcv_data(df)
                
                # Generate SR levels
                sr_levels = await self._generate_sr_levels(price_df, split_name)
                
                # Engineer basic features
                basic_features = await self.feature_engineering.engineer_features(
                    price_df, volume_df, sr_levels=sr_levels
                )
                
                # Generate comprehensive SR features
                sr_features = await self._generate_comprehensive_sr_features(price_df, sr_levels)
                
                # Merge features
                all_features = self._merge_features(basic_features, sr_features)
                
                # Add target column if present
                if "target" in df.columns:
                    all_features["target"] = df["target"]
                
                return all_features
        except Exception as e:
            self.logger.error(f"Error engineering features for {split_name}: {e}")
            raise
    
    async def _generate_sr_levels(
        self, 
        price_df: pd.DataFrame, 
        split_name: str
    ) -> Dict[str, Any]:
        """Generate support/resistance levels."""
        try:
            # Simple SR level generation - can be enhanced
            lows = price_df["low"].astype(float)
            highs = price_df["high"].astype(float)
            window = min(len(lows), 2000)
            
            if window <= 0:
                return {"support_levels": [], "resistance_levels": []}
            
            lt = lows.tail(window).dropna()
            ht = highs.tail(window).dropna()
            
            if lt.empty or ht.empty:
                return {"support_levels": [], "resistance_levels": []}
            
            support_prices = np.percentile(lt.values, [5, 15, 30]).tolist()
            resistance_prices = np.percentile(ht.values, [70, 85, 95]).tolist()
            
            def _make_levels(values: List[float], strength: float = 0.2) -> List[Dict[str, float]]:
                levels = []
                seen = set()
                for value in values:
                    rounded = round(float(value), 8)
                    if rounded not in seen:
                        seen.add(rounded)
                        levels.append({"price": rounded, "strength": strength})
                return levels
            
            return {
                "support_levels": _make_levels(support_prices, 0.2),
                "resistance_levels": _make_levels(resistance_prices, 0.2),
            }
        except Exception as e:
            self.logger.error(f"Error generating SR levels: {e}")
            return {"support_levels": [], "resistance_levels": []}
    
    async def _generate_comprehensive_sr_features(
        self, 
        price_df: pd.DataFrame, 
        sr_levels: Dict[str, Any]
    ) -> Dict[str, pd.Series]:
        """Generate comprehensive SR features."""
        try:
            if not self.sr_predictor:
                self.logger.warning("SR predictor not available, skipping comprehensive features")
                return {}
            
            features = self.sr_predictor.calculate_comprehensive_sr_features(price_df, sr_levels)
            
            if features:
                self.logger.info(f"Generated {len(features)} comprehensive SR features")
            
            return features
        except Exception as e:
            self.logger.error(f"Error generating comprehensive SR features: {e}")
            return {}
    
    def _merge_features(
        self, 
        basic_features: pd.DataFrame, 
        sr_features: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """Merge basic and SR features."""
        try:
            if not sr_features:
                return basic_features
            
            # Convert SR features to DataFrame
            sr_df = pd.DataFrame(sr_features)
            
            # Merge with basic features
            merged = pd.concat([basic_features, sr_df], axis=1)
            
            # Remove duplicate columns
            merged = merged.loc[:, ~merged.columns.duplicated()]
            
            return merged
        except Exception as e:
            self.logger.error(f"Error merging features: {e}")
            return basic_features


@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=2.0,
    min_disk_gb=1.0,
    required_packages=["pandas", "numpy", "joblib"],
    data_quality_checks={
        "min_rows": 100,
        "required_columns": ["open", "high", "low", "close"],
    },
    context="Improved Feature Engineering",
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
    validation_metrics=["feature_count", "data_quality", "performance"],
)
async def run_step_improved(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    timeframe: str = "1m",
    force_rerun: bool = False,
    **kwargs: Any,
) -> bool:
    """
    Improved Step 2: Feature Engineering with enhanced code quality and performance.
    
    Key improvements:
    - Modular architecture with separate classes
    - Better memory management
    - Parallel processing support
    - Improved error handling and validation
    - Comprehensive logging and monitoring
    """
    logger = system_logger.getChild("Step2.ImprovedFeatureEngineering")
    start_time = time.time()
    
    try:
        logger.info("🚀 Starting improved Step 2: Feature Engineering")
        
        # Initialize configuration
        config = FeatureEngineeringConfig(
            symbol=symbol,
            exchange=exchange,
            data_dir=data_dir,
            timeframe=timeframe,
            force_rerun=force_rerun,
            **kwargs
        )
        
        # Initialize components
        artifact_manager = FeatureArtifactManager(config)
        data_loader = DataLoader(config)
        feature_engineer = FeatureEngineer(config)
        
        # Check for existing artifacts
        if artifact_manager.check_artifacts_exist() and not force_rerun:
            logger.info("📦 Loading existing feature artifacts")
            features = artifact_manager.load_artifacts()
            if features:
                logger.info("✅ Feature engineering completed (using cached artifacts)")
                return True
        
        # Load labeled data
        logger.info("📥 Loading labeled data")
        labeled_data = await data_loader.load_labeled_data()
        
        # Initialize feature engineering
        logger.info("🔧 Initializing feature engineering components")
        if not await feature_engineer.initialize():
            raise RuntimeError("Failed to initialize feature engineering components")
        
        # Engineer features
        logger.info("🎯 Engineering features")
        features = await feature_engineer.engineer_features_parallel(labeled_data)
        
        # Save artifacts
        logger.info("💾 Saving feature artifacts")
        if not artifact_manager.save_artifacts(features):
            raise RuntimeError("Failed to save feature artifacts")
        
        # Log completion
        total_time = time.time() - start_time
        total_features = sum(len(df.columns) for df in features.values())
        total_rows = sum(len(df) for df in features.values())
        
        logger.info(f"✅ Feature engineering completed successfully")
        logger.info(f"   ⏱️ Total time: {total_time:.2f}s")
        logger.info(f"   📊 Total features: {total_features}")
        logger.info(f"   📈 Total rows: {total_rows}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature engineering failed: {e}")
        logger.exception("Full traceback:")
        return False
    finally:
        # Cleanup
        gc.collect()


# Backward compatibility
async def run_step(*args, **kwargs):
    """Backward compatibility wrapper for the improved run_step function."""
    return await run_step_improved(*args, **kwargs)