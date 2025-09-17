#!/usr/bin/env python3
"""
Migration Script: Move Optimization System from feature_generation to feature_engineering

This script consolidates the optimization systems to eliminate redundancy.
"""

import os
import shutil
import re
from pathlib import Path
from typing import List, Tuple

def find_python_files(directory: str) -> List[Path]:
    """Find all Python files in directory."""
    path = Path(directory)
    return list(path.rglob("*.py"))

def update_imports_in_file(file_path: Path, import_mappings: List[Tuple[str, str]]) -> bool:
    """Update imports in a single file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        for old_import, new_import in import_mappings:
            content = re.sub(old_import, new_import, content)
        
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"✅ Updated imports in {file_path}")
            return True
        return False
    
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def migrate_optimization_system():
    """Main migration function."""
    print("🚀 Starting Optimization System Migration")
    print("=" * 50)
    
    # Define paths
    source_dir = Path("src/feature_generation/optimization")
    target_dir = Path("src/feature_engineering/optimization")
    
    # Step 1: Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created target directory: {target_dir}")
    
    # Step 2: Check if source optimization exists and is not empty
    if not source_dir.exists():
        print(f"⚠️ Source directory {source_dir} doesn't exist")
        return
    
    source_files = list(source_dir.glob("*.py"))
    if not source_files:
        print(f"⚠️ No Python files found in {source_dir}")
        return
    
    print(f"📊 Found {len(source_files)} files to migrate")
    
    # Step 3: Move files to feature_engineering/optimization
    for file_path in source_files:
        target_file = target_dir / file_path.name
        
        if target_file.exists():
            print(f"⚠️ Target file exists, backing up: {target_file}")
            backup_file = target_file.with_suffix(f"{target_file.suffix}.backup")
            shutil.copy2(target_file, backup_file)
        
        shutil.copy2(file_path, target_file)
        print(f"📦 Moved {file_path.name} to {target_dir}")
    
    # Step 4: Update imports across the codebase
    print("\n🔄 Updating imports across codebase...")
    
    import_mappings = [
        (r'from src\.feature_generation\.optimization', 'from src.feature_engineering.optimization'),
        (r'from \.\.optimization', 'from ...feature_engineering.optimization'),
        (r'import src\.feature_generation\.optimization', 'import src.feature_engineering.optimization'),
    ]
    
    # Find all Python files in src/
    python_files = find_python_files("src/")
    updated_files = 0
    
    for file_path in python_files:
        if update_imports_in_file(file_path, import_mappings):
            updated_files += 1
    
    print(f"✅ Updated imports in {updated_files} files")
    
    # Step 5: Update __init__.py files
    print("\n📝 Updating __init__.py files...")
    
    # Update feature_generation/__init__.py to remove optimization exports
    fg_init = Path("src/feature_generation/__init__.py")
    if fg_init.exists():
        with open(fg_init, 'r') as f:
            content = f.read()
        
        # Comment out optimization imports and exports
        optimization_sections = [
            "# Lookback optimization system",
            "from .optimization import",
            "OPTIMIZATION_AVAILABLE",
            "# Optimization",
            "__all__.extend([",
            "\"LookbackOptimizer\"",
            "\"FeatureOptimizationConfig\"",
            "\"FeatureOptimizationResult\"",
            "\"optimize_feature_lookbacks\"",
            "\"get_optimization_config\""
        ]
        
        lines = content.split('\n')
        updated_lines = []
        in_optimization_section = False
        
        for line in lines:
            if any(section in line for section in optimization_sections):
                if not line.strip().startswith('#'):
                    line = f"# MIGRATED TO feature_engineering: {line}"
                in_optimization_section = True
            elif in_optimization_section and line.strip() == "":
                in_optimization_section = False
            elif in_optimization_section and not line.strip().startswith('#'):
                line = f"# {line}"
            
            updated_lines.append(line)
        
        with open(fg_init, 'w') as f:
            f.write('\n'.join(updated_lines))
        
        print(f"✅ Updated {fg_init}")
    
    # Update feature_engineering/__init__.py to include optimization exports
    fe_init = Path("src/feature_engineering/__init__.py")
    if fe_init.exists():
        with open(fe_init, 'r') as f:
            content = f.read()
        
        # Add optimization imports if not already present
        if "optimization" not in content.lower():
            optimization_imports = '''
# Feature optimization system (migrated from feature_generation)
from .optimization.lookback_optimizer import (
    LookbackOptimizer,
    FeatureOptimizationConfig,
    FeatureOptimizationResult,
    OptimizationMethod,
    optimize_feature_lookbacks,
    get_optimization_config
)
'''
            
            # Find the imports section and add optimization imports
            lines = content.split('\n')
            insert_index = -1
            
            for i, line in enumerate(lines):
                if line.startswith('from .') and 'import' in line:
                    insert_index = i + 1
            
            if insert_index > 0:
                lines.insert(insert_index, optimization_imports)
                
                # Also update __all__ list
                all_index = -1
                for i, line in enumerate(lines):
                    if line.strip().startswith('__all__'):
                        all_index = i
                        break
                
                if all_index > 0:
                    optimization_exports = '''
    # Feature optimization system
    'LookbackOptimizer',
    'FeatureOptimizationConfig', 
    'FeatureOptimizationResult',
    'OptimizationMethod',
    'optimize_feature_lookbacks',
    'get_optimization_config',
'''
                    # Find the end of __all__ and insert before closing
                    for i in range(all_index, len(lines)):
                        if ']' in lines[i]:
                            lines.insert(i, optimization_exports)
                            break
                
                with open(fe_init, 'w') as f:
                    f.write('\n'.join(lines))
                
                print(f"✅ Updated {fe_init}")
    
    # Step 6: Remove the old optimization directory
    if source_dir.exists() and list(source_dir.glob("*.py")):
        backup_dir = Path("src/feature_generation/optimization_backup")
        shutil.move(source_dir, backup_dir)
        print(f"📦 Backed up original optimization to {backup_dir}")
    
    print("\n🎉 Migration completed successfully!")
    print("📋 Next steps:")
    print("  1. Run tests to verify functionality")
    print("  2. Check for any remaining import issues")
    print("  3. Remove backup files once verified")
    print("  4. Update documentation")

if __name__ == "__main__":
    migrate_optimization_system()