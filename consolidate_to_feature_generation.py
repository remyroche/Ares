#!/usr/bin/env python3
"""
Consolidation Script: Move feature_engineering to feature_generation/utils/

This script moves all contents from src/feature_engineering/ to src/feature_generation/utils/
and updates all necessary imports across the codebase.
"""

import os
import shutil
import re
from pathlib import Path
from typing import List, Tuple, Dict

def create_directory_structure():
    """Create the target directory structure."""
    print("📁 Creating target directory structure...")
    
    target_dir = Path("src/feature_generation/utils")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created {target_dir}")
    return target_dir

def move_feature_engineering_contents():
    """Move all contents from feature_engineering to feature_generation/utils."""
    print("\n📦 Moving feature_engineering contents...")
    
    source_dir = Path("src/feature_engineering")
    target_dir = Path("src/feature_generation/utils")
    
    if not source_dir.exists():
        print(f"❌ Source directory {source_dir} doesn't exist")
        return False
    
    # Get all files and directories in feature_engineering
    items = list(source_dir.iterdir())
    moved_items = []
    
    for item in items:
        target_item = target_dir / item.name
        
        try:
            if item.is_file():
                # Copy file
                shutil.copy2(item, target_item)
                print(f"📄 Moved {item.name}")
            elif item.is_dir():
                # Copy directory recursively
                if target_item.exists():
                    shutil.rmtree(target_item)
                shutil.copytree(item, target_item)
                print(f"📁 Moved {item.name}/")
            
            moved_items.append(item.name)
            
        except Exception as e:
            print(f"❌ Error moving {item}: {e}")
    
    print(f"✅ Moved {len(moved_items)} items to {target_dir}")
    return True

def find_python_files(directory: str) -> List[Path]:
    """Find all Python files in directory."""
    path = Path(directory)
    return list(path.rglob("*.py"))

def update_imports_in_file(file_path: Path, import_mappings: List[Tuple[str, str]]) -> bool:
    """Update imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        for old_import, new_import in import_mappings:
            content = re.sub(old_import, new_import, content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Updated imports in {file_path.relative_to(Path.cwd())}")
            return True
        return False
    
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def update_all_imports():
    """Update all imports across the codebase."""
    print("\n🔄 Updating imports across codebase...")
    
    # Define import mapping patterns
    import_mappings = [
        # Direct imports
        (r'from src\.feature_engineering', 'from src.feature_generation.utils'),
        (r'import src\.feature_engineering', 'import src.feature_generation.utils'),
        
        # Relative imports within the moved code
        (r'from \.\.\.feature_engineering', 'from ..utils'),
        (r'from \.\.feature_engineering', 'from .utils'),
        
        # Specific common patterns
        (r'src\.feature_engineering\.', 'src.feature_generation.utils.'),
        (r'feature_engineering\.', 'feature_generation.utils.'),
    ]
    
    # Find all Python files in src/
    python_files = find_python_files("src/")
    updated_files = 0
    
    for file_path in python_files:
        if update_imports_in_file(file_path, import_mappings):
            updated_files += 1
    
    print(f"✅ Updated imports in {updated_files} files")
    return updated_files

def create_utils_init():
    """Create __init__.py for the utils package."""
    print("\n📝 Creating utils package __init__.py...")
    
    utils_init = Path("src/feature_generation/utils/__init__.py")
    
    init_content = '''"""
Feature Generation Utils Package

This package contains advanced feature engineering utilities, optimization systems,
and analysis tools. Previously located in src/feature_engineering/.

Main Components:
- Optimization system (unified_optimizer.py, optimization/)
- Advanced feature engineering (step06_* files)
- Cross-timeframe analysis (cross_timeframe_*)
- Matrix operations and GPU acceleration
- Triple barrier labeling and regime analysis
- Utility containers and dependency injection
"""

# Import main utility classes for easy access
try:
    from .step06_utility_container import (
        Step06UtilityContainer,
        UtilityConfig,
        get_utility_container,
        utility_container_context,
        inject_utilities
    )
except ImportError:
    pass

try:
    from .step06_enhanced_feature_engineering import (
        EnhancedFeatureEngineering
    )
except ImportError:
    pass

try:
    from .optimization import (
        FeatureGenerationOptimizer,
        FeatureOptimizationConfig,
        FeatureOptimizationResult,
        OptimizationMethod,
        get_feature_optimizer,
        optimize_feature_lookback,
        get_optimization_config,
        LookbackOptimizer  # Backward compatibility
    )
except ImportError:
    pass

# Advanced utilities
try:
    from .cross_timeframe_analysis_pipeline import CrossTimeframeAnalysisPipeline
    from .fractional_differentiation_pipeline import FractionalDifferentiationPipeline
    from .enhanced_matrix_operations import EnhancedMatrixOperations
except ImportError:
    pass

# Feature validation
try:
    from .math_validation import (
        validate_feature_quality,
        validate_features_dataframe,
        feature_validation_decorator
    )
except ImportError:
    pass

__version__ = "2.0.0"
__description__ = "Feature Generation Utils - Advanced feature engineering and optimization utilities"
'''
    
    with open(utils_init, 'w', encoding='utf-8') as f:
        f.write(init_content)
    
    print(f"✅ Created {utils_init}")

def update_feature_generation_init():
    """Update feature_generation/__init__.py to include utils."""
    print("\n📝 Updating feature_generation/__init__.py...")
    
    fg_init = Path("src/feature_generation/__init__.py")
    if not fg_init.exists():
        print(f"❌ {fg_init} not found")
        return
    
    with open(fg_init, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add utils import section
    utils_import_section = '''
# Advanced utilities (moved from feature_engineering)
try:
    from .utils import (
        # Optimization system
        FeatureGenerationOptimizer,
        FeatureOptimizationConfig,
        FeatureOptimizationResult,
        OptimizationMethod,
        get_feature_optimizer,
        optimize_feature_lookback,
        get_optimization_config,
        LookbackOptimizer,
        
        # Advanced feature engineering
        EnhancedFeatureEngineering,
        Step06UtilityContainer,
        UtilityConfig,
        
        # Analysis pipelines
        CrossTimeframeAnalysisPipeline,
        FractionalDifferentiationPipeline,
        EnhancedMatrixOperations,
        
        # Validation
        validate_feature_quality,
        validate_features_dataframe
    )
    UTILS_AVAILABLE = True
except ImportError as e:
    UTILS_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Advanced utils not available: {e}")
'''
    
    # Find where to insert the utils import
    lines = content.split('\n')
    insert_index = -1
    
    # Look for the end of existing imports
    for i, line in enumerate(lines):
        if 'CONVENIENCE_AVAILABLE' in line:
            insert_index = i + 5  # After the convenience import block
            break
    
    if insert_index > 0:
        lines.insert(insert_index, utils_import_section)
        
        # Update __all__ list to include utils exports
        utils_exports = '''
# Advanced utils
if UTILS_AVAILABLE:
    __all__.extend([
        # Optimization system
        "FeatureGenerationOptimizer",
        "FeatureOptimizationConfig", 
        "FeatureOptimizationResult",
        "OptimizationMethod",
        "get_feature_optimizer",
        "optimize_feature_lookback",
        "get_optimization_config",
        "LookbackOptimizer",
        
        # Advanced utilities
        "EnhancedFeatureEngineering",
        "Step06UtilityContainer",
        "UtilityConfig",
        "CrossTimeframeAnalysisPipeline",
        "FractionalDifferentiationPipeline",
        "EnhancedMatrixOperations",
        "validate_feature_quality",
        "validate_features_dataframe"
    ])
'''
        
        # Find the end of __all__ exports and add utils
        for i in range(len(lines)):
            if 'if CONVENIENCE_AVAILABLE:' in lines[i] and '__all__.extend' in lines[i+1]:
                # Find the end of this block
                for j in range(i+1, len(lines)):
                    if lines[j].strip() == '])':
                        lines.insert(j+2, utils_exports)
                        break
                break
        
        with open(fg_init, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ Updated {fg_init}")

def update_compatibility_layers():
    """Update compatibility layers to reference the new location."""
    print("\n🔧 Updating compatibility layers...")
    
    # Update HMM compatibility
    hmm_compat = Path("src/feature_generation/compatibility/hmm_compatibility.py")
    if hmm_compat.exists():
        with open(hmm_compat, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update imports to use utils
        updated_content = content.replace(
            'from ...feature_engineering.feature_generators import FeatureGenerators',
            'from ..utils.feature_generators import FeatureGenerators'
        ).replace(
            'from ...feature_engineering.optimization import get_feature_optimizer',
            'from ..utils.optimization import get_feature_optimizer'
        )
        
        with open(hmm_compat, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"✅ Updated {hmm_compat}")
    
    # Update standalone HMM compatibility
    standalone_hmm = Path("src/hmm_feature_compatibility.py")
    if standalone_hmm.exists():
        with open(standalone_hmm, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update any references to feature_engineering
        updated_content = content.replace(
            'feature_engineering',
            'feature_generation.utils'
        )
        
        if content != updated_content:
            with open(standalone_hmm, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            print(f"✅ Updated {standalone_hmm}")

def backup_original_structure():
    """Create backup of original feature_engineering directory."""
    print("\n📦 Creating backup of original structure...")
    
    source_dir = Path("src/feature_engineering")
    backup_dir = Path("src/feature_engineering_backup")
    
    if source_dir.exists() and not backup_dir.exists():
        shutil.copytree(source_dir, backup_dir)
        print(f"✅ Backed up original structure to {backup_dir}")
        return True
    return False

def remove_original_feature_engineering():
    """Remove the original feature_engineering directory."""
    print("\n🗑️ Removing original feature_engineering directory...")
    
    source_dir = Path("src/feature_engineering")
    if source_dir.exists():
        shutil.rmtree(source_dir)
        print(f"✅ Removed {source_dir}")
        return True
    return False

def main():
    """Main consolidation function."""
    print("🚀 Starting Feature Engineering → Feature Generation Consolidation")
    print("=" * 70)
    
    try:
        # Step 1: Create backup
        backup_original_structure()
        
        # Step 2: Create target structure
        create_directory_structure()
        
        # Step 3: Move contents
        if not move_feature_engineering_contents():
            print("❌ Failed to move contents")
            return False
        
        # Step 4: Create utils package
        create_utils_init()
        
        # Step 5: Update main feature_generation init
        update_feature_generation_init()
        
        # Step 6: Update imports across codebase
        updated_files = update_all_imports()
        
        # Step 7: Update compatibility layers
        update_compatibility_layers()
        
        # Step 8: Remove original directory
        remove_original_feature_engineering()
        
        print("\n🎉 Consolidation completed successfully!")
        print("📋 Summary:")
        print(f"  ✅ Moved feature_engineering → feature_generation/utils/")
        print(f"  ✅ Updated imports in {updated_files} files")
        print(f"  ✅ Created utils package structure")
        print(f"  ✅ Updated compatibility layers")
        print(f"  ✅ Backed up original to feature_engineering_backup/")
        
        print("\n📁 New Structure:")
        print("  src/feature_generation/")
        print("    ├── core/                    # Core framework")
        print("    ├── categories/              # Category generators")
        print("    ├── utils/                   # Advanced utilities (from feature_engineering)")
        print("    │   ├── optimization/        # Optimization system")
        print("    │   ├── step06_*/            # Legacy utilities")
        print("    │   ├── cross_timeframe_*/   # Analysis pipelines")
        print("    │   └── *.py                 # Various utilities")
        print("    ├── compatibility/           # HMM compatibility")
        print("    └── convenience/             # Convenience functions")
        
        print("\n📋 Next Steps:")
        print("  1. Test the new structure")
        print("  2. Verify all imports work correctly")
        print("  3. Run any existing tests")
        print("  4. Remove backup after verification")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during consolidation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)