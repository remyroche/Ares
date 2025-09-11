#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Basic functionality test for artifact versioning system.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add workspace to path
sys.path.insert(0, '/workspace')

def test_version_manager():
    """Test version manager functionality."""
    tprint("🧪 Testing Version Manager...")
    
    try:
        from src.utils.version_manager import VersionManager
        
        # Create temporary config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"ares_version": "v1"}')
            config_path = f.name
        
        try:
            vm = VersionManager(config_path)
            assert vm.get_ares_version() == "v1"
            
            vm.set_ares_version("v2")
            assert vm.get_ares_version() == "v2"
            
            timestamp = vm.generate_timestamp()
            assert len(timestamp) == 15
            assert timestamp.count("_") == 1
            
            tprint("✅ Version Manager tests passed")
            return True
        finally:
            os.unlink(config_path)
            
    except Exception as e:
        tprint(f"❌ Version Manager test failed: {e}")
        return False

def test_artifact_manager():
    """Test artifact manager functionality."""
    tprint("🧪 Testing Artifact Manager...")
    
    try:
        from src.utils.enhanced_artifact_manager import EnhancedArtifactManager
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "ares_version": "v1",
                "artifacts_dir": temp_dir,
                "data_dir": temp_dir,
                "models_dir": temp_dir,
                "cache_dir": temp_dir,
                "output_dir": temp_dir
            }
            
            am = EnhancedArtifactManager(config)
            
            # Test filename generation
            filename = am.generate_timestamped_filename("test_model", ".pkl")
            assert "test_model" in filename
            assert "v1" in filename
            assert filename.endswith(".pkl")
            assert filename.count("_") == 2
            
            # Test saving JSON artifact
            test_data = {"key": "value", "number": 42}
            file_path = am.save_artifact(test_data, "test_artifact", ".json", "artifacts")
            assert Path(file_path).exists()
            
            # Test loading artifact
            loaded_data, metadata = am.load_most_recent_artifact("test_artifact", "artifacts", extension=".json")
            assert loaded_data == test_data
            assert metadata is not None
            assert metadata.base_name == "test_artifact"
            assert metadata.version == "v1"
            
            tprint("✅ Artifact Manager tests passed")
            return True
            
    except Exception as e:
        tprint(f"❌ Artifact Manager test failed: {e}")
        return False

def test_artifact_pickup():
    """Test artifact pickup functionality."""
    tprint("🧪 Testing Artifact Pickup...")
    
    try:
        from src.utils.enhanced_artifact_manager import EnhancedArtifactManager
        from src.utils.artifact_pickup_utils import ArtifactPickupUtils
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "ares_version": "v1",
                "artifacts_dir": temp_dir
            }
            
            am = EnhancedArtifactManager(config)
            pickup_utils = ArtifactPickupUtils()
            pickup_utils.artifact_manager = am
            
            # Create multiple artifacts
            for i in range(3):
                test_data = {"iteration": i}
                am.save_artifact(test_data, "test_pickup", ".json", "artifacts")
            
            # Test finding most recent
            recent_path = pickup_utils.find_most_recent_artifact("test_pickup", "artifacts")
            assert recent_path is not None
            assert Path(recent_path).exists()
            
            # Test loading most recent
            loaded_data, metadata = pickup_utils.load_most_recent_artifact("test_pickup", "artifacts")
            assert loaded_data is not None
            assert metadata is not None
            
            tprint("✅ Artifact Pickup tests passed")
            return True
            
    except Exception as e:
        tprint(f"❌ Artifact Pickup test failed: {e}")
        return False

def main():
    """Run all tests."""
    tprint("🚀 Basic Functionality Test for Artifact Versioning System")
    tprint("=" * 60)
    
    tests = [
        test_version_manager,
        test_artifact_manager,
        test_artifact_pickup
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        tprint()
    
    tprint("📊 Test Results:")
    tprint(f"✅ Passed: {passed}/{total}")
    tprint(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        tprint("🎉 All tests passed! The artifact versioning system is working correctly.")
        return 0
    else:
        tprint("⚠️ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())