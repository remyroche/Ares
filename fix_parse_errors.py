#!/usr/bin/env python3
import os
import subprocess
import sys

# Files with parse errors from the analysis
parse_error_files = [
    "src/feature_generation/utils/step06_enhanced_feature_engineering_step.py",
    "src/feature_generation/utils/unified_optimization_system.py",
    "src/feature_generation/utils/contrastive_learning_guide.py",
    "src/feature_generation/utils/feature_generators_compatibility.py",
    "src/feature_generation/utils/feature_generation_optimization.py",
    "src/feature_generation/utils/sr_feature_extractor.py",
    "src/feature_generation/utils/optimized_feature_orchestrator.py",
    "src/feature_generation/utils/cross_timeframe_interaction_features.py",
    "src/feature_generation/utils/temporal_feature_integration.py",
    "src/feature_generation/utils/vectorbt_memory_optimizer.py"
]

def check_syntax(file_path):
    """Check if a file has syntax errors."""
    try:
        result = subprocess.run([sys.executable, '-m', 'py_compile', file_path],
                              capture_output=True, text=True, timeout=30)
        return result.returncode == 0, result.stderr
    except Exception as e:
        return False, str(e)

def main():
    print(f"🔧 Processing {len(parse_error_files)} files with parse errors")

    fixed_count = 0
    for file_path in parse_error_files:
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue

        is_valid, error = check_syntax(file_path)
        if is_valid:
            print(f"✅ {file_path} - Already valid")
            continue

        print(f"❌ {file_path} - Syntax error: {error.strip()}")
        # For now, we'll need manual fixes for complex syntax errors
        # The first file we fixed manually was step06_enhanced_feature_engineering_step.py

    print(f"\n📊 Summary: {fixed_count}/{len(parse_error_files)} files fixed")

if __name__ == "__main__":
    main()
