"""Compression Manager Module.

Handles compression and decompression of artifacts.
"""

import pickle
from enum import Enum
from typing import Any, Tuple, Optional
from dataclasses import dataclass

# Optional compression libraries
try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

try:
    import gzip
    GZIP_AVAILABLE = True
except ImportError:
    GZIP_AVAILABLE = False

from .logger import system_logger


class CompressionType(Enum):
    """Supported compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    AUTO = "auto"


@dataclass
class CompressionConfig:
    """Configuration for compression."""
    enabled: bool = True
    algorithm: CompressionType = CompressionType.AUTO
    min_size_mb: float = 1.0
    compression_level: int = 6


class CompressionManager:
    """Handles compression and decompression of artifacts."""
    
    def __init__(self, config: CompressionConfig):
        """Initialize compression manager.
        
        Args:
            config: Compression configuration
        """
        self.config = config
        self.logger = system_logger.getChild("CompressionManager")
    
    def should_compress(self, data_size_bytes: int) -> bool:
        """Determine if data should be compressed.
        
        Args:
            data_size_bytes: Size of data in bytes
            
        Returns:
            True if data should be compressed
        """
        if not self.config.enabled:
            return False
        
        min_size_bytes = int(self.config.min_size_mb * 1024 * 1024)
        return data_size_bytes >= min_size_bytes
    
    def choose_compression(self, data_size_bytes: int) -> CompressionType:
        """Choose the best compression algorithm for the data.
        
        Args:
            data_size_bytes: Size of data in bytes
            
        Returns:
            Best compression type for the data
        """
        if not self.config.enabled:
            return CompressionType.NONE
        
        if self.config.algorithm != CompressionType.AUTO:
            return self.config.algorithm
        
        # Auto-select based on data size and availability
        if data_size_bytes > 100 * 1024 * 1024:  # > 100MB
            return CompressionType.LZ4 if LZ4_AVAILABLE else CompressionType.GZIP
        else:
            return CompressionType.GZIP if GZIP_AVAILABLE else CompressionType.NONE
    
    def compress_data(self, data: Any, compression_type: CompressionType) -> Tuple[bytes, float]:
        """Compress data using the specified algorithm.
        
        Args:
            data: Data to compress
            compression_type: Compression algorithm to use
            
        Returns:
            Tuple of (compressed_data, compression_ratio)
        """
        try:
            # Serialize data first
            serialized_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            original_size = len(serialized_data)
            
            if compression_type == CompressionType.GZIP and GZIP_AVAILABLE:
                compressed_data = gzip.compress(serialized_data, compresslevel=self.config.compression_level)
            elif compression_type == CompressionType.LZ4 and LZ4_AVAILABLE:
                compressed_data = lz4.frame.compress(serialized_data, compression_level=1)
            else:
                compressed_data = serialized_data
            
            compression_ratio = len(compressed_data) / original_size if original_size > 0 else 1.0
            return compressed_data, compression_ratio
            
        except Exception as e:
            self.logger.warning(f"Compression failed, falling back to no compression: {e}")
            serialized_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            return serialized_data, 1.0
    
    def decompress_data(self, compressed_data: bytes, compression_type: CompressionType) -> Any:
        """Decompress data using the specified algorithm.
        
        Args:
            compressed_data: Compressed data
            compression_type: Compression algorithm used
            
        Returns:
            Decompressed data
        """
        try:
            if compression_type == CompressionType.GZIP and GZIP_AVAILABLE:
                decompressed_data = gzip.decompress(compressed_data)
            elif compression_type == CompressionType.LZ4 and LZ4_AVAILABLE:
                decompressed_data = lz4.frame.decompress(compressed_data)
            else:
                decompressed_data = compressed_data
            
            return pickle.loads(decompressed_data)
            
        except Exception as e:
            self.logger.error(f"Decompression failed: {e}")
            raise
    
    def get_compression_stats(self) -> dict:
        """Get compression statistics.
        
        Returns:
            Dictionary with compression statistics
        """
        return {
            "lz4_available": LZ4_AVAILABLE,
            "gzip_available": GZIP_AVAILABLE,
            "config": {
                "enabled": self.config.enabled,
                "algorithm": self.config.algorithm.value,
                "min_size_mb": self.config.min_size_mb,
                "compression_level": self.config.compression_level
            }
        }