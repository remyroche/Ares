"""Memory Manager Module.

Handles memory optimization and spilling strategies for large artifacts.
"""

import gc
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, List

# Optional dependencies
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
    NUMPY_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    NUMPY_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .logger import system_logger


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    max_memory_mb: float = 2000.0
    spill_threshold_mb: float = 150.0
    cleanup_interval_seconds: float = 300.0
    enable_gc_collection: bool = True
    enable_spilling: bool = True
    enable_optimization: bool = True


@dataclass
class MemoryProfile:
    """Memory usage profile for artifacts."""
    artifact_id: str
    memory_usage_mb: float
    spilled: bool = False
    compression_ratio: float = 1.0
    column_count: int = 0
    row_count: int = 0
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)


class MemoryManager:
    """Handles memory optimization and spilling strategies."""
    
    def __init__(self, config: MemoryConfig, spill_dir: Path):
        """Initialize memory manager.
        
        Args:
            config: Memory configuration
            spill_dir: Directory for spilled artifacts
        """
        self.config = config
        self.spill_dir = spill_dir
        self.logger = system_logger.getChild("MemoryManager")
        
        # Memory profiles
        self._memory_profiles: Dict[str, MemoryProfile] = {}
        self._total_memory_mb = 0.0
        
        # Spill directory
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance metrics
        self._spill_operations = 0
        self._optimization_savings_mb = 0.0
        
        # Background cleanup
        self._last_cleanup = time.time()
    
    def profile_memory_usage(self, artifact_id: str, data: Any) -> MemoryProfile:
        """Profile memory usage of an artifact.
        
        Args:
            artifact_id: Unique identifier for the artifact
            data: Data to profile
            
        Returns:
            Memory profile for the artifact
        """
        memory_usage_mb = 0
        column_count = 0
        row_count = 0
        
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            memory_usage_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
            column_count = len(data.columns)
            row_count = len(data)
        elif NUMPY_AVAILABLE and isinstance(data, np.ndarray):
            memory_usage_mb = data.nbytes / (1024 * 1024)
            row_count = data.shape[0] if len(data.shape) > 0 else 0
            column_count = data.shape[1] if len(data.shape) > 1 else 1
        else:
            # Estimate for other types
            try:
                import sys
                memory_usage_mb = sys.getsizeof(data) / (1024 * 1024)
            except:
                memory_usage_mb = 0
        
        profile = MemoryProfile(
            artifact_id=artifact_id,
            memory_usage_mb=memory_usage_mb,
            column_count=column_count,
            row_count=row_count
        )
        
        self._memory_profiles[artifact_id] = profile
        self._total_memory_mb += memory_usage_mb
        
        self.logger.debug(f"Profiled {artifact_id}: {memory_usage_mb:.2f}MB")
        return profile
    
    def should_spill(self, profile: MemoryProfile) -> bool:
        """Determine if an artifact should be spilled to disk.
        
        Args:
            profile: Memory profile of the artifact
            
        Returns:
            True if artifact should be spilled
        """
        if not self.config.enable_spilling:
            return False
        
        return profile.memory_usage_mb > self.config.spill_threshold_mb
    
    def spill_artifact(self, artifact_id: str, data: Any, profile: MemoryProfile) -> bool:
        """Spill artifact to disk with optimization.
        
        Args:
            artifact_id: Unique identifier for the artifact
            data: Data to spill
            profile: Memory profile of the artifact
            
        Returns:
            True if successful, False otherwise
        """
        try:
            spill_path = self.spill_dir / f"{artifact_id}.spilled"
            
            if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
                # Save as optimized parquet
                parquet_path = spill_path.with_suffix('.parquet')
                data.to_parquet(parquet_path, compression='snappy', engine='pyarrow')
                profile.spilled = True
                profile.compression_ratio = 0.3  # Estimate for parquet
            else:
                # Fallback to compressed pickle
                compressed_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                with open(spill_path, 'wb') as f:
                    f.write(compressed_data)
                profile.spilled = True
                profile.compression_ratio = 0.7  # Estimate for pickle
            
            # Update profile
            self._memory_profiles[artifact_id] = profile
            self._spill_operations += 1
            
            self.logger.info(f"Spilled {artifact_id} ({profile.memory_usage_mb:.2f}MB)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to spill artifact {artifact_id}: {e}")
            return False
    
    def load_spilled_artifact(self, artifact_id: str) -> Optional[Any]:
        """Load a spilled artifact from disk.
        
        Args:
            artifact_id: Unique identifier for the artifact
            
        Returns:
            Loaded data or None if failed
        """
        try:
            spill_path = self.spill_dir / f"{artifact_id}.spilled"
            parquet_path = self.spill_dir / f"{artifact_id}.parquet"
            
            # Check for parquet file first
            if parquet_path.exists() and PANDAS_AVAILABLE:
                data = pd.read_parquet(parquet_path)
                # Update access info
                if artifact_id in self._memory_profiles:
                    profile = self._memory_profiles[artifact_id]
                    profile.last_accessed = datetime.now()
                    profile.access_count += 1
                return data
            
            # Fallback to spilled file
            if spill_path.exists():
                with open(spill_path, 'rb') as f:
                    data = pickle.load(f)
                # Update access info
                if artifact_id in self._memory_profiles:
                    profile = self._memory_profiles[artifact_id]
                    profile.last_accessed = datetime.now()
                    profile.access_count += 1
                return data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load spilled artifact {artifact_id}: {e}")
            return None
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for memory efficiency.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame
        """
        if not PANDAS_AVAILABLE or not isinstance(df, pd.DataFrame) or not self.config.enable_optimization:
            return df
        
        optimized_df = df.copy()
        original_memory = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Optimize numeric columns
        for col in optimized_df.select_dtypes(include=[np.number]).columns:
            col_data = optimized_df[col]
            
            # Compute min/max once per column
            col_min = col_data.min()
            col_max = col_data.max()
            has_nans = col_data.isna().any()
            
            # Handle integer columns with NaNs using nullable dtypes
            if col_data.dtype == np.int64:
                if has_nans:
                    # Use pandas nullable dtypes for integers with NaNs
                    if col_max < np.iinfo(np.int32).max and col_min > np.iinfo(np.int32).min:
                        optimized_df[col] = col_data.astype('Int32')
                    elif col_max < np.iinfo(np.int16).max and col_min > np.iinfo(np.int16).min:
                        optimized_df[col] = col_data.astype('Int16')
                    elif col_max < np.iinfo(np.int8).max and col_min > np.iinfo(np.int8).min:
                        optimized_df[col] = col_data.astype('Int8')
                else:
                    # No NaNs, can use regular downcast
                    if col_max < np.iinfo(np.int32).max and col_min > np.iinfo(np.int32).min:
                        optimized_df[col] = col_data.astype(np.int32)
                    elif col_max < np.iinfo(np.int16).max and col_min > np.iinfo(np.int16).min:
                        optimized_df[col] = col_data.astype(np.int16)
                    elif col_max < np.iinfo(np.int8).max and col_min > np.iinfo(np.int8).min:
                        optimized_df[col] = col_data.astype(np.int8)
            
            # Handle float columns with safe downcast
            elif col_data.dtype == np.float64:
                if col_max < np.finfo(np.float32).max and col_min > np.finfo(np.float32).min:
                    optimized_df[col] = col_data.astype(np.float32)
        
        # Optimize object columns to category if beneficial
        for col in optimized_df.select_dtypes(include=['object']).columns:
            if len(optimized_df) > 100000:  # Large DataFrame
                # Sample for estimation
                sample_size = min(10000, len(optimized_df))
                sample = optimized_df[col].sample(n=sample_size, random_state=42)
                uniqueness_ratio = sample.nunique() / len(sample)
                if uniqueness_ratio < 0.5 and sample.nunique() < 10000:
                    optimized_df[col] = optimized_df[col].astype('category')
            else:
                uniqueness_ratio = optimized_df[col].nunique() / len(optimized_df)
                if uniqueness_ratio < 0.5 and optimized_df[col].nunique() < 10000:
                    optimized_df[col] = optimized_df[col].astype('category')
        
        # Use convert_dtypes for efficient extension dtypes
        try:
            optimized_df = optimized_df.convert_dtypes()
        except Exception:
            pass  # Fallback if convert_dtypes fails
        
        optimized_memory = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)
        savings = original_memory - optimized_memory
        
        if savings > 0.1:  # Only log if significant savings
            self._optimization_savings_mb += savings
            self.logger.debug(f"DataFrame optimization: {savings:.1f}MB saved ({savings/original_memory*100:.1f}% reduction)")
        
        return optimized_df
    
    def cleanup_expired_profiles(self) -> int:
        """Clean up expired memory profiles.
        
        Returns:
            Number of profiles cleaned up
        """
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=24)  # 24 hour TTL
        
        expired_keys = []
        for key, profile in self._memory_profiles.items():
            if profile.last_accessed < cutoff_time and profile.access_count < 2:
                expired_keys.append(key)
        
        for key in expired_keys:
            profile = self._memory_profiles.pop(key)
            self._total_memory_mb -= profile.memory_usage_mb
        
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired memory profiles")
        
        return len(expired_keys)
    
    def periodic_cleanup(self) -> None:
        """Perform periodic memory cleanup."""
        current_time = time.time()
        if current_time - self._last_cleanup > self.config.cleanup_interval_seconds:
            # Clean up expired profiles
            self.cleanup_expired_profiles()
            
            # Force garbage collection if enabled
            if self.config.enable_gc_collection:
                collected = gc.collect()
                if collected > 0:
                    self.logger.debug(f"Garbage collection freed {collected} objects")
            
            self._last_cleanup = current_time
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        spilled_count = sum(1 for profile in self._memory_profiles.values() if profile.spilled)
        
        # Calculate average compression ratio
        if spilled_count > 0 and NUMPY_AVAILABLE:
            avg_compression_ratio = np.mean([p.compression_ratio for p in self._memory_profiles.values() if p.spilled])
        elif spilled_count > 0:
            compression_ratios = [p.compression_ratio for p in self._memory_profiles.values() if p.spilled]
            avg_compression_ratio = sum(compression_ratios) / len(compression_ratios)
        else:
            avg_compression_ratio = 1.0
        
        # Get system memory info if available
        system_memory = {}
        if PSUTIL_AVAILABLE:
            memory_info = psutil.virtual_memory()
            system_memory = {
                "system_memory_percent": memory_info.percent,
                "system_memory_available_mb": memory_info.available / (1024 * 1024),
                "system_memory_total_mb": memory_info.total / (1024 * 1024)
            }
        
        return {
            "total_artifacts": len(self._memory_profiles),
            "total_memory_mb": self._total_memory_mb,
            "spilled_artifacts": spilled_count,
            "in_memory_artifacts": len(self._memory_profiles) - spilled_count,
            "average_compression_ratio": avg_compression_ratio,
            "spill_operations": self._spill_operations,
            "optimization_savings_mb": self._optimization_savings_mb,
            "config": {
                "max_memory_mb": self.config.max_memory_mb,
                "spill_threshold_mb": self.config.spill_threshold_mb,
                "enable_spilling": self.config.enable_spilling,
                "enable_optimization": self.config.enable_optimization
            },
            **system_memory
        }