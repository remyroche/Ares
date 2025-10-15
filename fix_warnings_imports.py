#!/usr/bin/env python3
"""
Script to add missing warnings imports to files that need them.
"""

import os
import sys

def add_warnings_import(file_path):
    """Add warnings import to a file if it's missing."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if warnings is already imported
        if 'import warnings' in content or 'from warnings import' in content:
            print(f"✅ {file_path} already has warnings import")
            return True
        
        # Find the first import statement
        lines = content.split('\n')
        import_line_index = -1
        
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_line_index = i
                break
        
        if import_line_index == -1:
            # No imports found, add at the beginning
            lines.insert(0, 'import warnings')
        else:
            # Add warnings import before the first import
            lines.insert(import_line_index, 'import warnings')
        
        # Write back to file
        with open(file_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ Added warnings import to {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def main():
    """Main function to fix warnings imports."""
    files_to_fix = [
        "src/training/steps/pre_training/unified_data_driven_pipeline/steps/feature_generation_data_validation_step.py",
        "src/training/steps/pre_training/unified_data_driven_pipeline/steps/feature_generation_feature_generation_step.py",
        "src/training/steps/pre_training/unified_data_driven_pipeline/steps/feature_generation_feature_selection_step.py",
        "src/training/steps/pre_training/unified_data_driven_pipeline/steps/feature_generation_final_validation_step.py",
        "src/training/steps/pre_training/unified_data_driven_pipeline/steps/feature_generation_interaction_generation_step.py",
        "src/training/steps/pre_training/unified_data_driven_pipeline/steps/feature_generation_labeling_integration_step.py",
        "src/training/steps/pre_training/unified_data_driven_pipeline/steps/feature_generation_lookback_optimization_step.py",
        "src/training/steps/pre_training/unified_data_driven_pipeline/steps/feature_generation_period_optimization_step.py",
        "src/training/steps/pre_training/unified_data_driven_pipeline/steps/feature_generation_period_lookback_optimization_step.py",
        "src/training/steps/pre_training/unified_data_driven_pipeline/steps/feature_generation_vectorization_step.py",
        "src/training/steps/pre_training/interaction_feature_generator/feature_interaction_generation/interactive_feature_generation_component.py",
        "src/training/steps/pre_training/components/final_feature_selection.py"
    ]
    
    success_count = 0
    total_count = len(files_to_fix)
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            if add_warnings_import(file_path):
                success_count += 1
        else:
            print(f"⚠️ File not found: {file_path}")
    
    print(f"\n📊 Results: {success_count}/{total_count} files processed successfully")
    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
