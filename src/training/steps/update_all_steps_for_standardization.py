#!/usr/bin/env python3
import pandas as pd

"""
Update All Steps for Parquet Standardization

This script updates all training steps to use the standardized Parquet handler
for consistent file paths, column names, and data formats.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any
import collections
import sys


class StepUpdater:
    """Updates training steps to use standardized Parquet handling."""
    
    def __init__(self, steps_dir: str = "/workspace/src/training/steps"):
        self.steps_dir = Path(steps_dir)
        self.updated_files = []
        self.errors = []
        
    def find_step_files(self) -> List[Path]:
        """Find all Python files in the steps directory."""
        step_files = []
        
        # Find all Python files recursively
        for py_file in self.steps_dir.rglob("*.py"):
            # Skip __init__.py and test files
            if py_file.name == "__init__.py" or "test" in py_file.name.lower():
                continue
            step_files.append(py_file)
        
        return step_files
    
    def update_imports(self, file_path: Path) -> bool:
        """Add standardized Parquet handler import to a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if already imported
            if "standardized_parquet_handler" in content:
                return True
            
            # Find a good place to add the import
            lines = content.split('\n')
            import_section_end = 0
            
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    import_section_end = i + 1
                elif line.strip() and not line.strip().startswith('#'):
                    break
            
            # Add the import
            import_line = "from src.training.steps.standardized_parquet_handler import standardized_parquet_handler"
            lines.insert(import_section_end, import_line)
            
            # Write back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            return True
            
        except Exception as e:
            self.errors.append(f"Error updating imports in {file_path}: {e}")
            return False
    
    def update_parquet_operations(self, file_path: Path) -> bool:
        """Update Parquet read/write operations to use standardized handler."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Replace pd.read_parquet with standardized handler
            content = re.sub(
                r'pd\.read_parquet\(([^)]+)\)',
                r'standardized_parquet_handler.read_parquet_standardized(\1)',
                content
            )
            
            # Replace .to_parquet with standardized handler
            content = re.sub(
                r'(\w+)\.to_parquet\(([^)]+)\)',
                r'standardized_parquet_handler.write_parquet_standardized(\1, \2)',
                content
            )
            
            # Replace pipeline_standards.build_path with standardized handler
            content = re.sub(
                r'pipeline_standards\.build_path\(([^)]+)\)',
                r'standardized_parquet_handler.get_standardized_path(\1)',
                content
            )
            
            # Replace pipeline_standards.generate_file_name with standardized handler
            content = re.sub(
                r'pipeline_standards\.generate_file_name\(([^)]+)\)',
                r'standardized_parquet_handler.get_standardized_filename(\1)',
                content
            )
            
            # Only write if changes were made
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
            
            return True
            
        except Exception as e:
            self.errors.append(f"Error updating Parquet operations in {file_path}: {e}")
            return False
    
    def update_file(self, file_path: Path) -> bool:
        """Update a single file for standardization."""
        try:
            # Skip if it's the standardized handler itself
            if file_path.name == "standardized_parquet_handler.py":
                return True
            
            # Update imports
            imports_updated = self.update_imports(file_path)
            
            # Update Parquet operations
            operations_updated = self.update_parquet_operations(file_path)
            
            if imports_updated and operations_updated:
                self.updated_files.append(str(file_path))
                return True
            else:
                return False
                
        except Exception as e:
            self.errors.append(f"Error updating {file_path}: {e}")
            return False
    
    def update_all_steps(self) -> Dict[str, Any]:
        """Update all training steps for standardization."""
        print("🔍 Finding training step files...")
        step_files = self.find_step_files()
        print(f"📁 Found {len(step_files)} step files")
        
        print("🔄 Updating files for standardization...")
        updated_count = 0
        
        for file_path in step_files:
            print(f"   Processing: {file_path.relative_to(self.steps_dir)}")
            if self.update_file(file_path):
                updated_count += 1
        
        return {
            'total_files': len(step_files),
            'updated_files': updated_count,
            'updated_file_list': self.updated_files,
            'errors': self.errors,
            'success': len(self.errors) == 0
        }


def main():
    """Main function to update all steps."""
    print("🚀 Starting Parquet Standardization Update")
    print("=" * 50)
    
    updater = StepUpdater()
    results = updater.update_all_steps()
    
    print("\n📊 Update Results:")
    print(f"   Total files processed: {results['total_files']}")
    print(f"   Files updated: {results['updated_files']}")
    print(f"   Errors: {len(results['errors'])}")
    
    if results['errors']:
        print("\n❌ Errors encountered:")
        for error in results['errors']:
            print(f"   - {error}")
    
    if results['updated_files'] > 0:
        print(f"\n✅ Successfully updated {results['updated_files']} files")
        print("\n📋 Updated files:")
        for file_path in results['updated_file_list']:
            print(f"   - {file_path}")
    
    print("\n🎉 Parquet standardization update completed!")
    
    return results['success']


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)