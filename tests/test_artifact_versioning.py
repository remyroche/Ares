"""
Tests for Artifact Versioning and Pickup System

This module contains comprehensive tests for the enhanced artifact management
system with version and timestamp support.
"""

import pytest
import tempfile
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils.enhanced_artifact_manager import (
    EnhancedArtifactManager, ArtifactMetadata, get_artifact_manager, initialize_artifact_manager
)
from src.utils.version_manager import VersionManager, get_version_manager, set_ares_version
from src.utils.artifact_pickup_utils import ArtifactPickupUtils, get_artifact_pickup_utils


class TestVersionManager:
    """Test cases for VersionManager."""
    
    def test_version_manager_initialization(self):
        """Test version manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "version_config.json"
            vm = VersionManager(str(config_path))
            
            assert vm.get_ares_version() == "v1"
            assert config_path.exists()
    
    def test_version_management(self):
        """Test version setting and getting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "version_config.json"
            vm = VersionManager(str(config_path))
            
            # Test setting version
            vm.set_ares_version("v2")
            assert vm.get_ares_version() == "v2"
            
            # Test version history
            version_info = vm.get_version_info()
            assert len(version_info["version_history"]) == 1
            assert version_info["version_history"][0]["version"] == "v2"
    
    def test_timestamp_generation(self):
        """Test timestamp generation."""
        vm = VersionManager()
        timestamp = vm.generate_timestamp()
        
        # Should be in YYYYMMDD_HHMMSS format
        assert len(timestamp) == 15
        assert timestamp.count("_") == 1
        assert timestamp.replace("_", "").isdigit()


class TestEnhancedArtifactManager:
    """Test cases for EnhancedArtifactManager."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "ares_version": "v1",
            "data_dir": self.temp_dir,
            "artifacts_dir": self.temp_dir,
            "models_dir": self.temp_dir,
            "cache_dir": self.temp_dir,
            "output_dir": self.temp_dir
        }
        self.manager = EnhancedArtifactManager(self.config)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_filename_generation(self):
        """Test versioned filename generation."""
        filename = self.manager.generate_timestamped_filename("test_model", ".pkl")
        
        # Should contain base name, version, and timestamp
        assert "test_model" in filename
        assert "v1" in filename
        assert filename.endswith(".pkl")
        assert filename.count("_") == 2  # base_version_timestamp
    
    def test_artifact_saving_and_loading(self):
        """Test saving and loading artifacts."""
        # Test data
        test_data = {"key": "value", "number": 42}
        
        # Save artifact
        file_path = self.manager.save_artifact(
            test_data, "test_artifact", ".json", "artifacts"
        )
        
        assert Path(file_path).exists()
        assert "test_artifact_v1_" in Path(file_path).name
        
        # Load artifact
        loaded_data, metadata = self.manager.load_most_recent_artifact(
            "test_artifact", "artifacts", extension=".json"
        )
        
        assert loaded_data == test_data
        assert metadata is not None
        assert metadata.base_name == "test_artifact"
        assert metadata.version == "v1"
    
    def test_artifact_discovery(self):
        """Test finding artifacts."""
        # Create multiple artifacts
        for i in range(3):
            test_data = {"iteration": i}
            self.manager.save_artifact(
                test_data, "test_discovery", ".json", "artifacts"
            )
        
        # Find all artifacts
        artifacts = self.manager.find_artifacts("test_discovery", "artifacts")
        assert len(artifacts) == 3
        
        # Should be sorted by timestamp (most recent first)
        timestamps = [a.timestamp for a in artifacts]
        assert timestamps == sorted(timestamps, reverse=True)
    
    def test_most_recent_artifact(self):
        """Test getting most recent artifact."""
        # Create artifacts with different timestamps
        base_time = datetime.now()
        
        for i in range(3):
            with patch('src.utils.enhanced_artifact_manager.datetime') as mock_datetime:
                mock_datetime.now.return_value = base_time + timedelta(minutes=i)
                test_data = {"iteration": i}
                self.manager.save_artifact(
                    test_data, "test_recent", ".json", "artifacts"
                )
        
        # Get most recent
        metadata = self.manager.get_most_recent_artifact("test_recent", "artifacts")
        assert metadata is not None
        assert metadata.base_name == "test_recent"
    
    def test_cleanup_old_artifacts(self):
        """Test cleanup of old artifacts."""
        # Create multiple artifacts
        for i in range(5):
            test_data = {"iteration": i}
            self.manager.save_artifact(
                test_data, "test_cleanup", ".json", "artifacts"
            )
        
        # Cleanup (keep 2 most recent)
        deleted_files = self.manager.cleanup_old_artifacts(
            "test_cleanup", "artifacts", keep_count=2
        )
        
        assert len(deleted_files) == 3  # Should delete 3 old files
        
        # Check remaining artifacts
        remaining = self.manager.find_artifacts("test_cleanup", "artifacts")
        assert len(remaining) == 2


class TestArtifactPickupUtils:
    """Test cases for ArtifactPickupUtils."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "ares_version": "v1",
            "artifacts_dir": self.temp_dir
        }
        self.manager = EnhancedArtifactManager(self.config)
        self.pickup_utils = ArtifactPickupUtils()
        self.pickup_utils.artifact_manager = self.manager
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_find_most_recent_artifact(self):
        """Test finding most recent artifact."""
        # Create multiple artifacts
        for i in range(3):
            test_data = {"iteration": i}
            self.manager.save_artifact(
                test_data, "test_pickup", ".json", "artifacts"
            )
        
        # Find most recent
        recent_path = self.pickup_utils.find_most_recent_artifact("test_pickup", "artifacts")
        assert recent_path is not None
        assert Path(recent_path).exists()
    
    def test_load_most_recent_artifact(self):
        """Test loading most recent artifact."""
        # Create artifact
        test_data = {"test": "data"}
        self.manager.save_artifact(
            test_data, "test_load", ".json", "artifacts"
        )
        
        # Load most recent
        loaded_data, metadata = self.pickup_utils.load_most_recent_artifact(
            "test_load", "artifacts"
        )
        
        assert loaded_data == test_data
        assert metadata is not None
    
    def test_find_artifacts_by_pattern(self):
        """Test finding artifacts by pattern."""
        # Create artifacts with different names
        for name in ["test1", "test2", "other"]:
            test_data = {"name": name}
            self.manager.save_artifact(
                test_data, name, ".json", "artifacts"
            )
        
        # Find by pattern
        test_files = self.pickup_utils.find_artifacts_by_pattern("test*", "artifacts")
        assert len(test_files) == 2
        
        # Should be sorted by time
        assert test_files[0] != test_files[1]
    
    def test_get_artifact_info(self):
        """Test getting artifact information."""
        # Create artifact
        test_data = {"info": "test"}
        file_path = self.manager.save_artifact(
            test_data, "test_info", ".json", "artifacts"
        )
        
        # Get info
        info = self.pickup_utils.get_artifact_info(file_path)
        
        assert info is not None
        assert info["base_name"] == "test_info"
        assert info["version"] == "v1"
        assert info["is_versioned"] is True
    
    def test_list_available_artifacts(self):
        """Test listing available artifacts."""
        # Create multiple artifacts
        for name in ["artifact1", "artifact2"]:
            for i in range(2):
                test_data = {"name": name, "iteration": i}
                self.manager.save_artifact(
                    test_data, name, ".json", "artifacts"
                )
        
        # List artifacts
        artifacts = self.pickup_utils.list_available_artifacts("artifacts")
        
        assert len(artifacts) == 2
        assert "artifact1" in artifacts
        assert "artifact2" in artifacts
        assert len(artifacts["artifact1"]) == 2
        assert len(artifacts["artifact2"]) == 2


class TestIntegration:
    """Integration tests for the complete system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "ares_version": "v1",
            "artifacts_dir": self.temp_dir,
            "models_dir": self.temp_dir
        }
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_pipeline_artifact_flow(self):
        """Test complete pipeline artifact flow."""
        # Initialize system
        manager = initialize_artifact_manager(self.config)
        pickup_utils = get_artifact_pickup_utils()
        
        # Stage 1: Create data artifact
        data = pd.DataFrame({"price": [100, 101, 102]})
        data_path = manager.save_artifact(data, "market_data", ".parquet", "artifacts")
        
        # Stage 2: Create features artifact
        features = pd.DataFrame({"sma": [100.5, 101.5]})
        features_path = manager.save_artifact(features, "features", ".parquet", "artifacts")
        
        # Stage 3: Create model artifact
        model_data = {"model_type": "linear", "params": {"alpha": 0.1}}
        model_path = manager.save_artifact(model_data, "trained_model", ".pkl", "artifacts")
        
        # Verify all artifacts exist
        assert Path(data_path).exists()
        assert Path(features_path).exists()
        assert Path(model_path).exists()
        
        # Test pickup in next pipeline stage
        recent_data, data_meta = pickup_utils.load_most_recent_artifact("market_data", "artifacts")
        recent_features, features_meta = pickup_utils.load_most_recent_artifact("features", "artifacts")
        recent_model, model_meta = pickup_utils.load_most_recent_artifact("trained_model", "artifacts")
        
        assert recent_data is not None
        assert recent_features is not None
        assert recent_model is not None
        
        assert data_meta.base_name == "market_data"
        assert features_meta.base_name == "features"
        assert model_meta.base_name == "trained_model"
    
    def test_version_upgrade_flow(self):
        """Test artifact handling across version upgrades."""
        # Start with v1
        manager = initialize_artifact_manager(self.config)
        
        # Create v1 artifacts
        v1_data = {"version": "v1", "data": [1, 2, 3]}
        v1_path = manager.save_artifact(v1_data, "test_data", ".json", "artifacts")
        
        # Upgrade to v2
        set_ares_version("v2")
        
        # Create v2 artifacts
        v2_data = {"version": "v2", "data": [4, 5, 6]}
        v2_path = manager.save_artifact(v2_data, "test_data", ".json", "artifacts")
        
        # Test finding artifacts by version
        pickup_utils = get_artifact_pickup_utils()
        
        # Find most recent (should be v2)
        recent_data, recent_meta = pickup_utils.load_most_recent_artifact("test_data", "artifacts")
        assert recent_data["version"] == "v2"
        assert recent_meta.version == "v2"
        
        # Find v1 specifically
        v1_artifacts = manager.find_artifacts("test_data", "artifacts", version="v1")
        assert len(v1_artifacts) == 1
        assert v1_artifacts[0].version == "v1"
        
        # Find v2 specifically
        v2_artifacts = manager.find_artifacts("test_data", "artifacts", version="v2")
        assert len(v2_artifacts) == 1
        assert v2_artifacts[0].version == "v2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])