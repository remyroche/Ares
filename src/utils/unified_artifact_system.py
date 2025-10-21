"""
Unified Artifact Management System

This module provides a unified interface that integrates:
- KlinesParquetManager for specialized klines data handling
- serialization_utils for generic data serialization
- artifact_manager for comprehensive artifact lifecycle management
- BaseStep integration for step-based workflows

The system provides a single, consistent interface for all artifact operations
while leveraging the strengths of each component.
"""

from __future__ import annotations

import os
import gc
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

# Import the three core components
from .kline_parquet import KlinesParquetManager, StorageConfig, KlinesMetadata
from .serialization_utils import UniversalSerializer, safe_serialize, safe_deserialize
from .artifact_manager import ArtifactManager, ArtifactMetadata, OperationMetrics
from .logger import system_logger
from .tprint import tprint, tprint_success, tprint_info, tprint_warning, tprint_error

# Import BaseStep for integration
from src.training.base_step import BaseStep


@dataclass
class UnifiedConfig:
    """Configuration for the unified artifact system."""
    # Base configuration
    base_dir: str = "unified_artifacts"
    enable_klines_optimization: bool = True
    enable_compression: bool = True
    enable_caching: bool = True
    enable_memory_optimization: bool = True
    
    # Klines-specific configuration
    klines_config: Optional[StorageConfig] = None
    
    # Artifact manager configuration
    artifact_config: Optional[Dict[str, Any]] = None
    
    # Serialization preferences
    default_serialization_format: str = "auto"  # auto, json, pickle, parquet
    klines_serialization_format: str = "parquet"
    
    # Metadata consistency
    enforce_metadata_consistency: bool = True
    metadata_version: str = "1.0"


@dataclass
class UnifiedMetadata:
    """Unified metadata that works across all components."""
    # Core identification
    artifact_id: str
    artifact_type: str  # klines, data, model, metadata, etc.
    step_name: Optional[str] = None
    
    # Data characteristics
    symbol: Optional[str] = None
    exchange: Optional[str] = None
    interval: Optional[str] = None
    direction: Optional[str] = None
    model: Optional[str] = None
    
    # Storage information
    storage_location: str = "memory"
    file_path: Optional[str] = None
    file_size_bytes: int = 0
    compression_ratio: float = 1.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    
    # Data quality
    data_quality_score: float = 1.0
    validation_status: str = "unknown"
    
    # Relationships
    parent_artifacts: List[str] = field(default_factory=list)
    child_artifacts: List[str] = field(default_factory=list)
    
    # Additional metadata
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    version: str = "1.0"
    
    # Component-specific metadata
    klines_metadata: Optional[KlinesMetadata] = None
    artifact_metadata: Optional[ArtifactMetadata] = None


class UnifiedArtifactSystem:
    """
    Unified artifact management system that integrates all three components.
    
    Provides a single interface for:
    - Klines data storage and retrieval with optimization
    - Generic data serialization
    - Comprehensive artifact lifecycle management
    - Step-based workflow integration
    """
    
    def __init__(self, config: Optional[UnifiedConfig] = None):
        """Initialize the unified artifact system."""
        self.config = config or UnifiedConfig()
        self.logger = system_logger.getChild("UnifiedArtifactSystem")
        
        # Initialize base directory
        self.base_dir = Path(self.config.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._init_klines_manager()
        self._init_serializer()
        self._init_artifact_manager()
        
        # Unified metadata registry
        self._metadata_registry: Dict[str, UnifiedMetadata] = {}
        
        # Performance tracking
        self._performance_metrics = {
            'total_operations': 0,
            'klines_operations': 0,
            'generic_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'compression_savings_mb': 0.0
        }
        
        tprint_success("✅ Unified Artifact System initialized")
    
    def _init_klines_manager(self):
        """Initialize the klines manager."""
        klines_config = self.config.klines_config or StorageConfig()
        klines_config.base_dir = str(self.base_dir / "klines")
        self._klines_manager = KlinesParquetManager(klines_config)
    
    def _init_serializer(self):
        """Initialize the universal serializer."""
        self._serializer = UniversalSerializer()
    
    def _init_artifact_manager(self):
        """Initialize the artifact manager."""
        artifact_config = self.config.artifact_config or {
            "paths": {"data_dir": str(self.base_dir / "artifacts")},
            "enable_compression": self.config.enable_compression,
            "enable_caching": self.config.enable_caching,
            "enable_memory_optimization": self.config.enable_memory_optimization,
            "enable_thread_safety": True
        }
        self._artifact_manager = ArtifactManager(artifact_config)
    
    def set_context(self, step_name: str, symbol: Optional[str] = None,
                   exchange: Optional[str] = None, interval: Optional[str] = None,
                   direction: str = "long", model: str = "Analyst") -> None:
        """Set context for all components."""
        tprint_info(f"📁 SETTING UNIFIED CONTEXT: {step_name} | {symbol} | {exchange} | {interval} | {direction} | {model}")
        
        # Set context for artifact manager
        self._artifact_manager.set_context(
            step_name=step_name,
            symbol=symbol,
            exchange=exchange,
            direction=direction,
            model=model
        )
    
    def store_klines(self, df: pd.DataFrame, symbol: str, exchange: str, interval: str,
                    batch_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store klines data using the specialized klines manager."""
        try:
            tprint_info(f"📊 STORING KLINES: {symbol} {exchange} {interval}")
            
            # Store using klines manager
            success = self._klines_manager.store_klines(
                df=df, symbol=symbol, exchange=exchange, interval=interval,
                batch_id=batch_id, metadata=metadata
            )
            
            if not success:
                raise Exception("Failed to store klines data")
            
            # Create unified metadata
            unified_metadata = self._create_unified_metadata(
                artifact_id=f"klines_{symbol}_{exchange}_{interval}_{batch_id or 'latest'}",
                artifact_type="klines",
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                file_size_bytes=df.memory_usage(deep=True).sum(),
                klines_metadata=self._klines_manager._metadata_cache.get(
                    f"{symbol}_{exchange}_{interval}_{batch_id or 'latest'}"
                )
            )
            
            # Register metadata
            self._metadata_registry[unified_metadata.artifact_id] = unified_metadata
            
            # Update performance metrics
            self._performance_metrics['klines_operations'] += 1
            self._performance_metrics['total_operations'] += 1
            
            tprint_success(f"✅ KLINES STORED: {symbol} {exchange} {interval}")
            return unified_metadata.artifact_id
            
        except Exception as e:
            tprint_error(f"❌ FAILED TO STORE KLINES: {str(e)}")
            raise
    
    def load_klines(self, symbol: str, exchange: str, interval: str,
                   start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                   batch_id: Optional[str] = None) -> pd.DataFrame:
        """Load klines data using the specialized klines manager."""
        try:
            tprint_info(f"📊 LOADING KLINES: {symbol} {exchange} {interval}")
            
            # Load using klines manager
            df = self._klines_manager.load_klines(
                symbol=symbol, exchange=exchange, interval=interval,
                start_time=start_time, end_time=end_time, batch_id=batch_id
            )
            
            # Update metadata if found
            artifact_id = f"klines_{symbol}_{exchange}_{interval}_{batch_id or 'latest'}"
            if artifact_id in self._metadata_registry:
                self._metadata_registry[artifact_id].last_accessed = datetime.utcnow()
            
            tprint_success(f"✅ KLINES LOADED: {symbol} {exchange} {interval} ({len(df)} records)")
            return df
            
        except Exception as e:
            tprint_error(f"❌ FAILED TO LOAD KLINES: {str(e)}")
            return pd.DataFrame()
    
    def store_artifact(self, data: Any, artifact_name: str, artifact_type: str = "data",
                      compression: str = "auto", metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store generic artifact using the artifact manager."""
        try:
            tprint_info(f"💾 STORING ARTIFACT: {artifact_name}")
            
            # Store using artifact manager
            file_path = self._artifact_manager.save(
                data=data, artifact_name=artifact_name, artifact_type=artifact_type,
                compression=compression, metadata=metadata
            )
            
            # Create unified metadata
            unified_metadata = self._create_unified_metadata(
                artifact_id=artifact_name,
                artifact_type=artifact_type,
                file_path=file_path,
                file_size_bytes=self._estimate_data_size(data)
            )
            
            # Register metadata
            self._metadata_registry[artifact_name] = unified_metadata
            
            # Update performance metrics
            self._performance_metrics['generic_operations'] += 1
            self._performance_metrics['total_operations'] += 1
            
            tprint_success(f"✅ ARTIFACT STORED: {artifact_name}")
            return artifact_name
            
        except Exception as e:
            tprint_error(f"❌ FAILED TO STORE ARTIFACT: {str(e)}")
            raise
    
    def load_artifact(self, artifact_name: str, artifact_type: str = "data") -> Any:
        """Load generic artifact using the artifact manager."""
        try:
            tprint_info(f"🔍 LOADING ARTIFACT: {artifact_name}")
            
            # Load using artifact manager
            data = self._artifact_manager.get_artifact(
                artifact_name=artifact_name, artifact_type=artifact_type
            )
            
            # Update metadata if found
            if artifact_name in self._metadata_registry:
                self._metadata_registry[artifact_name].last_accessed = datetime.utcnow()
            
            if data is not None:
                tprint_success(f"✅ ARTIFACT LOADED: {artifact_name}")
            else:
                tprint_warning(f"⚠️ ARTIFACT NOT FOUND: {artifact_name}")
            
            return data
            
        except Exception as e:
            tprint_error(f"❌ FAILED TO LOAD ARTIFACT: {str(e)}")
            return None
    
    def store_unified(self, data: Any, artifact_name: str, artifact_type: str = "data",
                     symbol: Optional[str] = None, exchange: Optional[str] = None,
                     interval: Optional[str] = None, **kwargs) -> str:
        """Store data with automatic type detection and optimal storage method."""
        try:
            # Determine if this is klines data
            is_klines = (
                isinstance(data, pd.DataFrame) and
                'timestamp' in data.columns and
                'open' in data.columns and
                'high' in data.columns and
                'low' in data.columns and
                'close' in data.columns and
                'volume' in data.columns and
                symbol is not None and
                exchange is not None and
                interval is not None
            )
            
            if is_klines and self.config.enable_klines_optimization:
                # Use specialized klines storage
                return self.store_klines(
                    df=data, symbol=symbol, exchange=exchange, interval=interval,
                    batch_id=kwargs.get('batch_id'), metadata=kwargs.get('metadata')
                )
            else:
                # Use generic artifact storage
                return self.store_artifact(
                    data=data, artifact_name=artifact_name, artifact_type=artifact_type,
                    compression=kwargs.get('compression', 'auto'), metadata=kwargs.get('metadata')
                )
                
        except Exception as e:
            tprint_error(f"❌ FAILED TO STORE UNIFIED: {str(e)}")
            raise
    
    def load_unified(self, artifact_name: str, artifact_type: str = "data",
                    symbol: Optional[str] = None, exchange: Optional[str] = None,
                    interval: Optional[str] = None, **kwargs) -> Any:
        """Load data with automatic type detection and optimal loading method."""
        try:
            # Check if this is a klines artifact
            if (symbol and exchange and interval and 
                artifact_name.startswith('klines_') and 
                self.config.enable_klines_optimization):
                # Use specialized klines loading
                return self.load_klines(
                    symbol=symbol, exchange=exchange, interval=interval,
                    start_time=kwargs.get('start_time'),
                    end_time=kwargs.get('end_time'),
                    batch_id=kwargs.get('batch_id')
                )
            else:
                # Use generic artifact loading
                return self.load_artifact(
                    artifact_name=artifact_name, artifact_type=artifact_type
                )
                
        except Exception as e:
            tprint_error(f"❌ FAILED TO LOAD UNIFIED: {str(e)}")
            return None
    
    def delete_artifact(self, artifact_name: str, artifact_type: str = "data") -> bool:
        """Delete artifact from all systems."""
        try:
            tprint_info(f"🗑️ DELETING ARTIFACT: {artifact_name}")
            
            # Try artifact manager first
            success = self._artifact_manager.delete_artifact(
                artifact_name=artifact_name, artifact_type=artifact_type
            )
            
            # Remove from metadata registry
            if artifact_name in self._metadata_registry:
                del self._metadata_registry[artifact_name]
            
            if success:
                tprint_success(f"✅ ARTIFACT DELETED: {artifact_name}")
            else:
                tprint_warning(f"⚠️ ARTIFACT NOT FOUND FOR DELETION: {artifact_name}")
            
            return success
            
        except Exception as e:
            tprint_error(f"❌ FAILED TO DELETE ARTIFACT: {str(e)}")
            return False
    
    def list_artifacts(self, pattern: str = "*", artifact_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all artifacts with unified metadata."""
        try:
            artifacts = []
            
            # Get artifacts from artifact manager
            artifact_paths = self._artifact_manager.list_artifacts(pattern)
            for path in artifact_paths:
                artifact_info = {
                    'name': path.stem,
                    'type': 'generic',
                    'path': str(path),
                    'size_bytes': path.stat().st_size if path.exists() else 0,
                    'created_at': datetime.fromtimestamp(path.stat().st_ctime) if path.exists() else None
                }
                artifacts.append(artifact_info)
            
            # Get klines data info
            klines_data = self._klines_manager.list_available_data()
            for kline_info in klines_data:
                artifact_info = {
                    'name': f"klines_{kline_info['symbol']}_{kline_info['exchange']}_{kline_info['interval']}",
                    'type': 'klines',
                    'symbol': kline_info['symbol'],
                    'exchange': kline_info['exchange'],
                    'interval': kline_info['interval'],
                    'size_bytes': int(kline_info['file_size_mb'] * 1024 * 1024),
                    'created_at': kline_info['created_at']
                }
                artifacts.append(artifact_info)
            
            # Filter by artifact type if specified
            if artifact_type:
                artifacts = [a for a in artifacts if a['type'] == artifact_type]
            
            return artifacts
            
        except Exception as e:
            tprint_error(f"❌ FAILED TO LIST ARTIFACTS: {str(e)}")
            return []
    
    def get_metadata(self, artifact_name: str) -> Optional[UnifiedMetadata]:
        """Get unified metadata for an artifact."""
        return self._metadata_registry.get(artifact_name)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        base_metrics = self._performance_metrics.copy()
        
        # Add component-specific metrics
        if hasattr(self._artifact_manager, 'get_stats'):
            base_metrics['artifact_manager'] = self._artifact_manager.get_stats()
        
        if hasattr(self._klines_manager, 'get_storage_stats'):
            base_metrics['klines_manager'] = self._klines_manager.get_storage_stats()
        
        return base_metrics
    
    def cleanup(self) -> None:
        """Perform cleanup across all components."""
        tprint_info("🧹 PERFORMING UNIFIED CLEANUP")
        
        # Cleanup artifact manager
        if hasattr(self._artifact_manager, 'cleanup'):
            self._artifact_manager.cleanup()
        
        # Clear metadata registry
        self._metadata_registry.clear()
        
        # Force garbage collection
        gc.collect()
        
        tprint_success("✅ UNIFIED CLEANUP COMPLETED")
    
    def _create_unified_metadata(self, artifact_id: str, artifact_type: str,
                               symbol: Optional[str] = None, exchange: Optional[str] = None,
                               interval: Optional[str] = None, file_size_bytes: int = 0,
                               klines_metadata: Optional[KlinesMetadata] = None) -> UnifiedMetadata:
        """Create unified metadata from component-specific metadata."""
        return UnifiedMetadata(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            file_size_bytes=file_size_bytes,
            klines_metadata=klines_metadata
        )
    
    def _estimate_data_size(self, data: Any) -> int:
        """Estimate the size of data in bytes."""
        try:
            if hasattr(data, 'memory_usage'):
                return data.memory_usage(deep=True).sum()
            elif hasattr(data, 'nbytes'):
                return data.nbytes
            else:
                import sys
                return sys.getsizeof(data)
        except:
            return 0


class EnhancedBaseStep(BaseStep):
    """
    Enhanced BaseStep that integrates with the unified artifact system.
    
    This class extends the base BaseStep to provide seamless artifact management
    for step-based workflows.
    """
    
    def __init__(self, config: Dict[str, Any], artifact_system: Optional[UnifiedArtifactSystem] = None):
        """Initialize the enhanced base step."""
        super().__init__(config)
        
        # Initialize artifact system
        self.artifact_system = artifact_system or UnifiedArtifactSystem()
        
        # Set context from config
        self.step_name = config.get('step_name', self.__class__.__name__)
        self.symbol = config.get('symbol')
        self.exchange = config.get('exchange')
        self.interval = config.get('interval')
        self.direction = config.get('direction', 'long')
        self.model = config.get('model', 'Analyst')
        
        # Set context
        self.artifact_system.set_context(
            step_name=self.step_name,
            symbol=self.symbol,
            exchange=self.exchange,
            interval=self.interval,
            direction=self.direction,
            model=self.model
        )
    
    def store_data(self, data: Any, name: str, data_type: str = "data", **kwargs) -> str:
        """Store data using the unified artifact system."""
        return self.artifact_system.store_unified(
            data=data, artifact_name=name, artifact_type=data_type,
            symbol=self.symbol, exchange=self.exchange, interval=self.interval,
            **kwargs
        )
    
    def load_data(self, name: str, data_type: str = "data", **kwargs) -> Any:
        """Load data using the unified artifact system."""
        return self.artifact_system.load_unified(
            artifact_name=name, artifact_type=data_type,
            symbol=self.symbol, exchange=self.exchange, interval=self.interval,
            **kwargs
        )
    
    def delete_data(self, name: str, data_type: str = "data") -> bool:
        """Delete data using the unified artifact system."""
        return self.artifact_system.delete_artifact(name, data_type)
    
    def list_data(self, pattern: str = "*", data_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List data using the unified artifact system."""
        return self.artifact_system.list_artifacts(pattern, data_type)
    
    def get_metadata(self, name: str) -> Optional[UnifiedMetadata]:
        """Get metadata for stored data."""
        return self.artifact_system.get_metadata(name)
    
    def get_step_metrics(self) -> Dict[str, Any]:
        """Get step-specific metrics."""
        return {
            'step_name': self.step_name,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'interval': self.interval,
            'direction': self.direction,
            'model': self.model,
            'artifact_count': len(self.artifact_system._metadata_registry),
            'performance_metrics': self.artifact_system.get_performance_metrics()
        }


# Convenience functions
def create_unified_system(config: Optional[UnifiedConfig] = None) -> UnifiedArtifactSystem:
    """Create a new unified artifact system."""
    return UnifiedArtifactSystem(config)


def get_unified_system() -> UnifiedArtifactSystem:
    """Get a singleton unified artifact system."""
    if not hasattr(get_unified_system, '_instance'):
        get_unified_system._instance = UnifiedArtifactSystem()
    return get_unified_system._instance


def create_enhanced_step(step_class: type, config: Dict[str, Any], 
                        artifact_system: Optional[UnifiedArtifactSystem] = None) -> EnhancedBaseStep:
    """Create an enhanced step instance."""
    return step_class(config, artifact_system)