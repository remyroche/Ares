"""Unified artifact and path management for reads/writes.

Provides a single place to resolve data, reports, cache, optimization, and tmp
paths based on configuration. Ensures directories exist before use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .logger import system_logger
from .common_operations import ensure_directory
from .version_manager import get_version_manager


@dataclass
class ArtifactManager:
	config: dict

	def __post_init__(self) -> None:
		self.logger = system_logger.getChild("ArtifactManager")
		paths = self.config.get("paths", {}) if isinstance(self.config, dict) else {}
		self._data_dir = Path(paths.get("data_dir", "data"))
		self._reports_dir = Path(paths.get("reports_dir", "reports"))
		self._cache_dir = Path(paths.get("cache_dir", "data_cache"))
		self._optimization_dir = Path(paths.get("optimization_dir", self._data_dir / "optimization"))
		self._tmp_dir = Path(paths.get("tmp_dir", "tmp"))

		# Ensure base directories exist
		for d in (self._data_dir, self._reports_dir, self._cache_dir, self._optimization_dir, self._tmp_dir):
			ensure_directory(str(d))
		
		# Initialize version manager
		self.version_manager = get_version_manager()

	def get_data_dir(self, *subdirs: str) -> Path:
		return self._ensure(self._data_dir, *subdirs)

	def get_reports_dir(self, *subdirs: str) -> Path:
		return self._ensure(self._reports_dir, *subdirs)

	def get_cache_dir(self, *subdirs: str) -> Path:
		return self._ensure(self._cache_dir, *subdirs)

	def get_optimization_dir(self, *subdirs: str) -> Path:
		return self._ensure(self._optimization_dir, *subdirs)

	def get_tmp_dir(self, *subdirs: str) -> Path:
		return self._ensure(self._tmp_dir, *subdirs)

	def get_tmp_path(self, filename: str) -> Path:
		return self.get_tmp_dir() / filename

	def _ensure(self, base: Path, *subdirs: str) -> Path:
		path = base
		for s in subdirs:
			path = path / s
		ensure_directory(str(path))
		return path
	
	def get_versioned_filename(self, base_name: str, extension: str = ".pkl") -> str:
		"""Generate a versioned filename with timestamp.
		
		Args:
			base_name: Base name for the file
			extension: File extension
			
		Returns:
			Versioned filename
		"""
		version = self.version_manager.get_ares_version()
		timestamp = self.version_manager.generate_timestamp()
		return f"{base_name}_{version}_{timestamp}{extension}"
	
	def get_ares_version(self) -> str:
		"""Get the current Ares version.
		
		Returns:
			Current Ares version
		"""
		return self.version_manager.get_ares_version()

