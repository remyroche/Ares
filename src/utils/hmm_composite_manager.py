#!/usr/bin/env python3
"""
HMM Composite Cluster Manager

Centralized manager for HMM composite cluster files that can be used by:
- step03_hmm_regime_discovery (to create files)
- VectorizedAdvancedFeatureEngineering (to check if files exist)
- CompositeHMMRegimeSystem (to load files)

This ensures consistent behavior and prevents infinite loops.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.training.steps.step03_hmm_regime_discovery import run_step as run_step3
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger

# Module-level sets to avoid duplicate logs across multiple instances
# This prevents log spam when different components instantiate the manager separately
_GLOBAL_LOGGED_LOADS: set[str] = set()
_GLOBAL_LOGGED_EVENTS: set[str] = set()


class HMMCompositeManager:
    """Centralized manager for HMM composite cluster files."""

    def __init__(self) -> None:
        """Initialize the HMM composite manager."""
        self.logger = system_logger.getChild("HMMCompositeManager")
        self._cache: Dict[str, Dict[str, Any]] = {}  # Simple cache to avoid repeated file checks/loads
        
        # Use shared global sets so multiple instances do not re-log the same events
        self._logged_loads = _GLOBAL_LOGGED_LOADS
        self._logged_events = _GLOBAL_LOGGED_EVENTS

        # Enhanced features
        self._file_metadata_cache: Dict[str, Dict[str, Any]] = {}  # Cache for file metadata
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # Cleanup cache every hour

    def _get_file_paths(
        self, exchange: str, symbol: str, timeframe: str, data_dir: str
    ) -> Dict[str, str]:
        """Get file paths for HMM composite cluster files."""
        base_name = f"{exchange}_{symbol}_hmm_composite_clusters_{timeframe}"
        return {
            "composite_clusters": os.path.join(data_dir, f"{base_name}.parquet"),
            "block_states": os.path.join(
                data_dir, f"{exchange}_{symbol}_hmm_block_states_{timeframe}.parquet",
            ),
            "intensity": os.path.join(
                data_dir, f"{exchange}_{symbol}_hmm_composite_intensity_{timeframe}.parquet",
            ),
            "meta": os.path.join(
                data_dir, f"{exchange}_{symbol}_hmm_composite_meta_{timeframe}.json",
            ),
            "basic_meta": os.path.join(
                data_dir, f"{exchange}_{symbol}_hmm_basic_meta_{timeframe}.json",
            ),
        }

    def _check_files_exist(
        self, file_paths: Dict[str, str]
    ) -> Tuple[bool, List[str]]:
        """Check if all required files exist."""
        missing_files: List[str] = []
        all_exist = True

        for file_type, file_path in file_paths.items():
            if not os.path.exists(file_path):
                all_exist = False
                missing_files.append(f"{file_type} ({file_path})")

        return all_exist, missing_files

    def _cleanup_cache_if_needed(self) -> None:
        """Clean up old cache entries if needed."""
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            # Remove old cache entries (older than 1 hour)
            cutoff_time = current_time - 3600
            
            # Clean up main cache
            old_keys = [
                key for key, data in self._cache.items()
                if data.get("timestamp", 0) < cutoff_time
            ]
            for key in old_keys:
                del self._cache[key]

            # Clean up metadata cache
            old_meta_keys = [
                key for key, data in self._file_metadata_cache.items()
                if data.get("timestamp", 0) < cutoff_time
            ]
            for key in old_meta_keys:
                del self._file_metadata_cache[key]

            self._last_cleanup = current_time
            self.logger.debug("Cache cleanup completed")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="HMM block states loading",
    )
    def load_block_states(
        self, exchange: str, symbol: str, timeframe: str, data_dir: str
    ) -> Optional[pd.DataFrame]:
        """Load HMM block states from file."""
        file_paths = self._get_file_paths(exchange, symbol, timeframe, data_dir)
        block_states_path = file_paths["block_states"]

        # Check cache first
        cache_key = f"{data_dir}|{exchange}|{symbol}|{timeframe}|block_states"
        
        # Cleanup cache if needed
        self._cleanup_cache_if_needed()
        
        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            if time.time() - cached_data.get("timestamp", 0) < 3600:  # 1 hour cache
                self.logger.debug(f"Returning cached block states for {exchange}:{symbol}:{timeframe}")
                return cached_data["data"]

        # Load from file
        if os.path.exists(block_states_path):
            try:
                df = pd.read_parquet(block_states_path)
                
                # Cache the result
                self._cache[cache_key] = {
                    "data": df,
                    "timestamp": time.time(),
                    "file_path": block_states_path
                }
                
                # Log the load event (only once per session)
                log_key = f"block_states_loaded_{exchange}_{symbol}_{timeframe}"
                if log_key not in self._logged_loads:
                    self.logger.info(f"✅ Loaded HMM block states: {exchange}:{symbol}:{timeframe}")
                    self._logged_loads.add(log_key)
                
                return df
            except Exception as e:
                self.logger.error(f"Error loading block states from {block_states_path}: {e}")
                return None
        else:
            self.logger.warning(f"Block states file not found: {block_states_path}")
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="HMM composite cluster loading",
    )
    def load_composite_clusters(
        self, exchange: str, symbol: str, timeframe: str, data_dir: str
    ) -> Optional[pd.DataFrame]:
        """Load HMM composite clusters from file."""
        file_paths = self._get_file_paths(exchange, symbol, timeframe, data_dir)
        composite_path = file_paths["composite_clusters"]

        # Check cache first
        cache_key = f"{data_dir}|{exchange}|{symbol}|{timeframe}|composite"
        
        # Cleanup cache if needed
        self._cleanup_cache_if_needed()
        
        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            if time.time() - cached_data.get("timestamp", 0) < 3600:  # 1 hour cache
                self.logger.debug(f"Returning cached composite clusters for {exchange}:{symbol}:{timeframe}")
                return cached_data["data"]

        # Load from file
        if os.path.exists(composite_path):
            try:
                df = pd.read_parquet(composite_path)
                
                # Cache the result
                self._cache[cache_key] = {
                    "data": df,
                    "timestamp": time.time(),
                    "file_path": composite_path
                }
                
                # Log the load event (only once per session)
                log_key = f"composite_clusters_loaded_{exchange}_{symbol}_{timeframe}"
                if log_key not in self._logged_loads:
                    self.logger.info(f"✅ Loaded HMM composite clusters: {exchange}:{symbol}:{timeframe}")
                    self._logged_loads.add(log_key)
                
                return df
            except Exception as e:
                self.logger.error(f"Error loading composite clusters from {composite_path}: {e}")
                return None
        else:
            self.logger.warning(f"Composite clusters file not found: {composite_path}")
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="HMM meta loading",
    )
    def load_meta(
        self, exchange: str, symbol: str, timeframe: str, data_dir: str
    ) -> Optional[Dict[str, Any]]:
        """Load HMM composite meta information from file."""
        file_paths = self._get_file_paths(exchange, symbol, timeframe, data_dir)
        meta_path = file_paths["meta"]

        # Check cache first
        cache_key = f"{data_dir}|{exchange}|{symbol}|{timeframe}|meta"
        
        # Cleanup cache if needed
        self._cleanup_cache_if_needed()
        
        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            if time.time() - cached_data.get("timestamp", 0) < 3600:  # 1 hour cache
                self.logger.debug(f"Returning cached meta for {exchange}:{symbol}:{timeframe}")
                return cached_data["data"]

        # Load from file
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta_data = json.load(f)
                
                # Cache the result
                self._cache[cache_key] = {
                    "data": meta_data,
                    "timestamp": time.time(),
                    "file_path": meta_path
                }
                
                # Log the load event (only once per session)
                log_key = f"meta_loaded_{exchange}_{symbol}_{timeframe}"
                if log_key not in self._logged_loads:
                    self.logger.info(f"✅ Loaded HMM meta: {exchange}:{symbol}:{timeframe}")
                    self._logged_loads.add(log_key)
                
                return meta_data
            except Exception as e:
                self.logger.error(f"Error loading meta from {meta_path}: {e}")
                return None
        else:
            self.logger.warning(f"Meta file not found: {meta_path}")
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="HMM intensity loading",
    )
    def load_intensity(
        self, exchange: str, symbol: str, timeframe: str, data_dir: str
    ) -> Optional[pd.DataFrame]:
        """Load HMM composite intensity from file."""
        file_paths = self._get_file_paths(exchange, symbol, timeframe, data_dir)
        intensity_path = file_paths["intensity"]

        # Check cache first
        cache_key = f"{data_dir}|{exchange}|{symbol}|{timeframe}|intensity"
        
        # Cleanup cache if needed
        self._cleanup_cache_if_needed()
        
        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            if time.time() - cached_data.get("timestamp", 0) < 3600:  # 1 hour cache
                self.logger.debug(f"Returning cached intensity for {exchange}:{symbol}:{timeframe}")
                return cached_data["data"]

        # Load from file
        if os.path.exists(intensity_path):
            try:
                df = pd.read_parquet(intensity_path)
                
                # Cache the result
                self._cache[cache_key] = {
                    "data": df,
                    "timestamp": time.time(),
                    "file_path": intensity_path
                }
                
                # Log the load event (only once per session)
                log_key = f"intensity_loaded_{exchange}_{symbol}_{timeframe}"
                if log_key not in self._logged_loads:
                    self.logger.info(f"✅ Loaded HMM intensity: {exchange}:{symbol}:{timeframe}")
                    self._logged_loads.add(log_key)
                
                return df
            except Exception as e:
                self.logger.error(f"Error loading intensity from {intensity_path}: {e}")
                return None
        else:
            self.logger.warning(f"Intensity file not found: {intensity_path}")
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="HMM basic meta loading",
    )
    def load_basic_meta(
        self, exchange: str, symbol: str, timeframe: str, data_dir: str
    ) -> Optional[Dict[str, Any]]:
        """Load HMM basic meta information from file."""
        file_paths = self._get_file_paths(exchange, symbol, timeframe, data_dir)
        basic_meta_path = file_paths["basic_meta"]

        # Check cache first
        cache_key = f"{data_dir}|{exchange}|{symbol}|{timeframe}|basic_meta"
        
        # Cleanup cache if needed
        self._cleanup_cache_if_needed()
        
        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            if time.time() - cached_data.get("timestamp", 0) < 3600:  # 1 hour cache
                self.logger.debug(f"Returning cached basic meta for {exchange}:{symbol}:{timeframe}")
                return cached_data["data"]

        # Load from file
        if os.path.exists(basic_meta_path):
            try:
                with open(basic_meta_path, 'r') as f:
                    basic_meta_data = json.load(f)
                
                # Cache the result
                self._cache[cache_key] = {
                    "data": basic_meta_data,
                    "timestamp": time.time(),
                    "file_path": basic_meta_path
                }
                
                # Log the load event (only once per session)
                log_key = f"basic_meta_loaded_{exchange}_{symbol}_{timeframe}"
                if log_key not in self._logged_loads:
                    self.logger.info(f"✅ Loaded HMM basic meta: {exchange}:{symbol}:{timeframe}")
                    self._logged_loads.add(log_key)
                
                return basic_meta_data
            except Exception as e:
                self.logger.error(f"Error loading basic meta from {basic_meta_path}: {e}")
                return None
        else:
            self.logger.warning(f"Basic meta file not found: {basic_meta_path}")
            return None

    def check_files_exist(
        self, exchange: str, symbol: str, timeframe: str, data_dir: str
    ) -> Tuple[bool, List[str]]:
        """Check if all HMM composite cluster files exist."""
        file_paths = self._get_file_paths(exchange, symbol, timeframe, data_dir)
        return self._check_files_exist(file_paths)

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="HMM composite cluster creation",
    )
    async def create_composite_clusters(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        data_dir: str,
        force_recreate: bool = False,
    ) -> bool:
        """Create HMM composite cluster files by running step 3."""
        file_paths = self._get_file_paths(exchange, symbol, timeframe, data_dir)
        
        # Check if files already exist
        all_exist, missing_files = self._check_files_exist(file_paths)
        
        if all_exist and not force_recreate:
            self.logger.info(f"HMM composite cluster files already exist for {exchange}:{symbol}:{timeframe}")
            return True

        # Log the creation event (only once per session)
        log_key = f"composite_clusters_created_{exchange}_{symbol}_{timeframe}"
        if log_key not in self._logged_events:
            self.logger.info(f"🚀 Creating HMM composite cluster files for {exchange}:{symbol}:{timeframe}")
            self._logged_events.add(log_key)

        try:
            # Run step 3 to create the files
            success = await run_step3(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                data_dir=data_dir,
                force_recreate=force_recreate
            )

            if success:
                # Verify files were created
                all_exist, missing_files = self._check_files_exist(file_paths)
                if all_exist:
                    self.logger.info(f"✅ Successfully created HMM composite cluster files for {exchange}:{symbol}:{timeframe}")
                    
                    # Clear cache for this symbol to force reload
                    self._clear_cache_for_symbol(exchange, symbol, timeframe, data_dir)
                    
                    return True
                else:
                    self.logger.error(f"❌ Files not created properly. Missing: {missing_files}")
                    return False
            else:
                self.logger.error(f"❌ Failed to create HMM composite cluster files for {exchange}:{symbol}:{timeframe}")
                return False

        except Exception as e:
            self.logger.exception(f"❌ Error creating HMM composite cluster files: {e}")
            return False

    def _clear_cache_for_symbol(
        self, exchange: str, symbol: str, timeframe: str, data_dir: str
    ) -> None:
        """Clear cache entries for a specific symbol."""
        cache_keys_to_remove = [
            f"{data_dir}|{exchange}|{symbol}|{timeframe}|block_states",
            f"{data_dir}|{exchange}|{symbol}|{timeframe}|composite",
            f"{data_dir}|{exchange}|{symbol}|{timeframe}|meta",
            f"{data_dir}|{exchange}|{symbol}|{timeframe}|intensity",
            f"{data_dir}|{exchange}|{symbol}|{timeframe}|basic_meta",
        ]
        
        for key in cache_keys_to_remove:
            if key in self._cache:
                del self._cache[key]
        
        self.logger.debug(f"Cleared cache for {exchange}:{symbol}:{timeframe}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="HMM composite cluster management",
    )
    async def get_or_create_composite_clusters(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        data_dir: str,
        force_recreate: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Get existing composite clusters or create them if they don't exist."""
        # Try to load existing files first
        df = self.load_composite_clusters(exchange, symbol, timeframe, data_dir)
        
        if df is not None and not force_recreate:
            return df

        # Files don't exist or force recreate requested, create them
        success = await self.create_composite_clusters(
            exchange, symbol, timeframe, data_dir, force_recreate
        )
        
        if success:
            # Try to load the newly created files
            return self.load_composite_clusters(exchange, symbol, timeframe, data_dir)
        else:
            return None

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache state."""
        return {
            "cache_size": len(self._cache),
            "metadata_cache_size": len(self._file_metadata_cache),
            "last_cleanup": self._last_cleanup,
            "cleanup_interval": self._cleanup_interval,
            "logged_loads_count": len(self._logged_loads),
            "logged_events_count": len(self._logged_events),
        }

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._file_metadata_cache.clear()
        self.logger.info("Cache cleared")

    def get_file_info(
        self, exchange: str, symbol: str, timeframe: str, data_dir: str
    ) -> Dict[str, Any]:
        """Get information about HMM composite cluster files."""
        file_paths = self._get_file_paths(exchange, symbol, timeframe, data_dir)
        file_info = {}
        
        for file_type, file_path in file_paths.items():
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                file_info[file_type] = {
                    "exists": True,
                    "size_bytes": stat.st_size,
                    "modified_time": stat.st_mtime,
                    "path": file_path
                }
            else:
                file_info[file_type] = {
                    "exists": False,
                    "size_bytes": 0,
                    "modified_time": None,
                    "path": file_path
                }
        
        return file_info
