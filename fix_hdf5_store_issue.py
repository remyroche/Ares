#!/usr/bin/env python3
"""
Fix for the HDF5 versioned artifacts store issue where multiple store instances
show 0 versions despite being recently modified.

The issue is that each context (symbol/exchange/timeframe/direction/model) creates a separate store,
and when checking a specific store, it only shows versions for that context.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def fix_artifact_router():
    """Fix the artifact router to properly handle multiple contexts."""
    file_path = Path("src/utils/artifact_router.py")
    
    # Read the current file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add a method to list all versions across all stores
    new_method = '''
    def list_all_versions(self) -> List[str]:
        """
        List all versions across all stores.
        
        Returns:
            List of all version names from all stores
        """
        from src.utils.tprint import tprint
        
        tprint("🐛 DEBUG: list_all_versions() called", "INFO")
        all_versions = []
        
        for store_key, store in self._versioned_stores.items():
            versions = store.list_versions()
            tprint(f"🐛 DEBUG: Store {store_key} has {len(versions)} versions: {versions}", "INFO")
            all_versions.extend(versions)
        
        tprint(f"🐛 DEBUG: Total versions across all stores: {len(all_versions)}", "INFO")
        return all_versions
'''
    
    # Add the method to the ArtifactRouter class
    class_end = content.rfind('class ArtifactRouter:')
    if class_end == -1:
        print("ERROR: Could not find ArtifactRouter class")
        return
    
    # Find the end of the class (next class or end of file)
    next_class = content.find('class ', class_end + 1)
    if next_class == -1:
        insert_pos = len(content)
    else:
        insert_pos = next_class
    
    # Insert the new method before the end of the class
    updated_content = content[:insert_pos] + new_method + content[insert_pos:]
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Added list_all_versions() method to ArtifactRouter class in {file_path}")

def fix_base_step_adapter():
    """Fix the base step adapter to properly handle multiple contexts."""
    file_path = Path("src/utils/versioned_artifacts/base_step_adapter.py")
    
    # Read the current file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add a method to list all versions across all stores
    new_method = '''
    def list_all_versions(self) -> List[str]:
        """
        List all versions across all stores.
        
        Returns:
            List of all version names from all stores
        """
        from src.utils.tprint import tprint
        
        tprint("🐛 DEBUG: VersionedArtifactAdapter.list_all_versions() called", "INFO")
        all_versions = []
        
        # Get all store directories
        store_dirs = [d for d in self.store.store_path.parent.iterdir() if d.is_dir()]
        tprint(f"🐛 DEBUG: Found {len(store_dirs)} store directories: {store_dirs}", "INFO")
        
        for store_dir in store_dirs:
            store_path = self.store.store_path.parent / store_dir
            if store_path.exists():
                # Create a temporary store to list versions
                temp_store = VersionedArtifactStore(
                    store_path=store_path,
                    auto_version=True,
                    enable_row_versioning=True
                )
                versions = temp_store.list_versions()
                tprint(f"🐛 DEBUG: Store {store_dir} has {len(versions)} versions: {versions}", "INFO")
                all_versions.extend(versions)
        
        tprint(f"🐛 DEBUG: Total versions across all stores: {len(all_versions)}", "INFO")
        return all_versions
'''
    
    # Add the method to the VersionedArtifactAdapter class
    class_end = content.rfind('class VersionedArtifactAdapter:')
    if class_end == -1:
        print("ERROR: Could not find VersionedArtifactAdapter class")
        return
    
    # Find the end of the class (next class or end of file)
    next_class = content.find('class ', class_end + 1)
    if next_class == -1:
        insert_pos = len(content)
    else:
        insert_pos = next_class
    
    # Insert the new method before the end of the class
    updated_content = content[:insert_pos] + new_method + content[insert_pos:]
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Added list_all_versions() method to VersionedArtifactAdapter class in {file_path}")

if __name__ == "__main__":
    print("Fixing HDF5 versioned artifacts store issue...")
    fix_artifact_router()
    fix_base_step_adapter()
    print("Fixes applied successfully!")