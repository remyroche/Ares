"""
Version Manager for Ares Pipeline

This module provides version management functionality for the Ares pipeline,
including configuration-driven version handling and timestamp generation.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .logger import system_logger


class VersionManager:
    """Manages version information for the Ares pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the version manager.
        
        Args:
            config_path: Path to version configuration file
        """
        self.logger = system_logger.getChild("VersionManager")
        self.config_path = config_path or "config/version_config.json"
        self._version_config = None
        self._load_version_config()
    
    def _load_version_config(self) -> None:
        """Load version configuration from file."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    self._version_config = json.load(f)
                self.logger.info(f"✅ Loaded version config from {self.config_path}")
            else:
                # Create default config
                self._version_config = {
                    "ares_version": "v1",
                    "version_history": [],
                    "created_at": datetime.now().isoformat(),
                    "description": "Ares pipeline version configuration"
                }
                self._save_version_config()
                self.logger.info(f"📝 Created default version config at {self.config_path}")
        except Exception as e:
            self.logger.error(f"❌ Failed to load version config: {e}")
            self._version_config = {"ares_version": "v1"}
    
    def _save_version_config(self) -> None:
        """Save version configuration to file."""
        try:
            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(self._version_config, f, indent=2)
            
            self.logger.debug(f"💾 Saved version config to {self.config_path}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save version config: {e}")
    
    def get_ares_version(self) -> str:
        """Get the current Ares version.
        
        Returns:
            Current Ares version string
        """
        return self._version_config.get("ares_version", "v1")
    
    def set_ares_version(self, version: str) -> None:
        """Set the Ares version.
        
        Args:
            version: New version string
        """
        old_version = self.get_ares_version()
        self._version_config["ares_version"] = version
        
        # Add to version history
        if "version_history" not in self._version_config:
            self._version_config["version_history"] = []
        
        self._version_config["version_history"].append({
            "version": version,
            "previous_version": old_version,
            "changed_at": datetime.now().isoformat()
        })
        
        self._save_version_config()
        self.logger.info(f"🔄 Updated Ares version: {old_version} -> {version}")
    
    def get_version_info(self) -> Dict[str, Any]:
        """Get comprehensive version information.
        
        Returns:
            Dictionary with version information
        """
        return {
            "current_version": self.get_ares_version(),
            "version_history": self._version_config.get("version_history", []),
            "config_created_at": self._version_config.get("created_at"),
            "last_updated": datetime.now().isoformat()
        }
    
    def generate_timestamp(self) -> str:
        """Generate a timestamp string for artifact naming.
        
        Returns:
            Timestamp string in YYYYMMDD_HHMMSS format
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def get_artifact_config(self) -> Dict[str, Any]:
        """Get configuration for artifact management.
        
        Returns:
            Dictionary with artifact configuration
        """
        return {
            "ares_version": self.get_ares_version(),
            "timestamp": self.generate_timestamp(),
            "version_info": self.get_version_info()
        }


# Global instance
_version_manager: Optional[VersionManager] = None


def get_version_manager(config_path: Optional[str] = None) -> VersionManager:
    """Get the global version manager instance.
    
    Args:
        config_path: Path to version configuration file
        
    Returns:
        VersionManager instance
    """
    global _version_manager
    if _version_manager is None:
        _version_manager = VersionManager(config_path)
    return _version_manager


def get_ares_version() -> str:
    """Get the current Ares version.
    
    Returns:
        Current Ares version string
    """
    return get_version_manager().get_ares_version()


def set_ares_version(version: str) -> None:
    """Set the Ares version.
    
    Args:
        version: New version string
    """
    get_version_manager().set_ares_version(version)


def get_artifact_config() -> Dict[str, Any]:
    """Get configuration for artifact management.
    
    Returns:
        Dictionary with artifact configuration
    """
    return get_version_manager().get_artifact_config()