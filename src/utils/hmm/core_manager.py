#!/usr/bin/env python3
"""
HMM Core Manager

This module contains the core HMM management functionality, extracted from
the monolithic hmm_composite_manager.py file. It handles basic HMM operations
and cluster file management.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..logger import system_logger

class HMMCoreManager:
    """Core manager for HMM operations and file management."""

    def __init__(self):
        """Initialize the core HMM manager."""
        self.logger = system_logger.getChild('HMMCoreManager')
        self.composite_cluster_files = {}
        self.performance_metrics = {}

        self.logger.info("HMM Core Manager initialized")

    def get_composite_cluster_file_path(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        file_type: str = 'parquet'
    ) -> str:
        """
        Get standardized path for composite cluster files.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            file_type: File extension (parquet, json, pkl)

        Returns:
            Standardized file path
        """
        filename = f"{exchange}_{symbol}_{timeframe}_composite_clusters.{file_type}"
        return os.path.join(data_dir, filename)

    def file_exists(self, filepath: str) -> bool:
        """Check if a file exists."""
        return os.path.exists(filepath) and os.path.getsize(filepath) > 0

    def load_composite_clusters(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> Optional[pd.DataFrame]:
        """
        Load composite cluster data if available.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory

        Returns:
            DataFrame with cluster data or None if not found
        """
        try:
            # Try parquet first
            parquet_path = self.get_composite_cluster_file_path(
                symbol, exchange, timeframe, data_dir, 'parquet'
            )

            if self.file_exists(parquet_path):
                self.logger.info(f"Loading composite clusters from {parquet_path}")
                return pd.read_parquet(parquet_path)

            # Try CSV as fallback
            csv_path = self.get_composite_cluster_file_path(
                symbol, exchange, timeframe, data_dir, 'csv'
            )

            if self.file_exists(csv_path):
                self.logger.info(f"Loading composite clusters from {csv_path}")
                return pd.read_csv(csv_path)

            self.logger.warning(f"No composite cluster files found for {symbol}_{exchange}_{timeframe}")
            return None

        except Exception as e:
            self.logger.error(f"Error loading composite clusters: {e}")
            return None

    def save_composite_clusters(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save composite cluster data.

        Args:
            data: DataFrame to save
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            metadata: Optional metadata to save alongside

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(data_dir, exist_ok=True)

            # Save as parquet (primary format)
            parquet_path = self.get_composite_cluster_file_path(
                symbol, exchange, timeframe, data_dir, 'parquet'
            )
            data.to_parquet(parquet_path, index=False)

            # Save metadata if provided
            if metadata:
                metadata_path = self.get_composite_cluster_file_path(
                    symbol, exchange, timeframe, data_dir, 'json'
                )
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)

            self.logger.info(f"Saved composite clusters to {parquet_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving composite clusters: {e}")
            return False

    def get_file_info(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> Dict[str, Any]:
        """
        Get information about composite cluster files.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory

        Returns:
            Dictionary with file information
        """
        info = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'data_dir': data_dir,
            'files_found': [],
            'total_size': 0,
            'last_modified': None
        }

        # Check different file types
        for file_type in ['parquet', 'csv', 'json', 'pkl']:
            filepath = self.get_composite_cluster_file_path(
                symbol, exchange, timeframe, data_dir, file_type
            )

            if self.file_exists(filepath):
                stat = os.stat(filepath)
                info['files_found'].append({
                    'type': file_type,
                    'path': filepath,
                    'size': stat.st_size,
                    'modified': stat.st_mtime
                })
                info['total_size'] += stat.st_size

                if info['last_modified'] is None or stat.st_mtime > info['last_modified']:
                    info['last_modified'] = stat.st_mtime

        return info

    def cleanup_old_files(
        self,
        data_dir: str,
        max_age_days: int = 30,
        dry_run: bool = True
    ) -> List[str]:
        """
        Clean up old composite cluster files.

        Args:
            data_dir: Directory to clean
            max_age_days: Maximum age in days
            dry_run: If True, only list files that would be deleted

        Returns:
            List of files that were (or would be) deleted
        """
        deleted_files = []

        try:
            if not os.path.exists(data_dir):
                return deleted_files

            cutoff_time = time.time() - (max_age_days * 24 * 3600)

            for filename in os.listdir(data_dir):
                if 'composite_clusters' in filename:
                    filepath = os.path.join(data_dir, filename)

                    if os.path.isfile(filepath):
                        stat = os.stat(filepath)

                        if stat.st_mtime < cutoff_time:
                            if not dry_run:
                                os.remove(filepath)
                                self.logger.info(f"Deleted old file: {filepath}")
                            else:
                                self.logger.info(f"Would delete old file: {filepath}")

                            deleted_files.append(filepath)

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

        return deleted_files

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the manager."""
        return self.performance_metrics.copy()

    def reset_performance_metrics(self) -> None:
        """Reset performance metrics."""
        self.performance_metrics.clear()
        self.logger.info("Performance metrics reset")
