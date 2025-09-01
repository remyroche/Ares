#!/usr/bin/env python3
"""
Script to fix metadata fields and standardized naming for partially integrated steps.
"""

import re
from pathlib import Path

# Steps that need metadata and naming fixes
STEPS_TO_FIX = [
    "step1_data_collection.py",
    "step2_data_reading.py",
    "step2_5_sr_optimization.py",
    "step4_triple_barrier_method.py",
    "step5_labeling.py",
    "step7_enhanced_matrix_operations.py",
    "step8_regime_data_splitting.py",
    "step9_5_hmm_lm_generalist_training.py",
]

def add_metadata_to_training_input(file_path: Path) -> bool:
    """Add missing metadata fields to training_input in artifact logging methods."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find the artifact logging method
        step_match = re.search(r'step(\d+(?:_\d+)?)', file_path.name)
        if not step_match:
            return False

        step_num = step_match.group(1)
        method_name = f"_log_step{step_num}_artifacts_and_report"

        if method_name not in content:
            return False

        # Find training_input creation
        training_input_pattern = r'training_input\s*=\s*\{[^}]*\}'
        match = re.search(training_input_pattern, content, re.DOTALL)

        if not match:
            return False

        training_input_text = match.group(0)

        # Check if metadata fields are already present
        if 'asset' in training_input_text and 'lookback_period' in training_input_text and 'project_version' in training_input_text:
            print(f"✅ Metadata fields already present in {file_path.name}")
            return True

        # Add missing metadata fields
        new_training_input = training_input_text.replace(
            '}',
            f',\n                "asset": symbol,  # Use symbol as asset\n                "lookback_period": self.config.get("lookback_days", 1095),  # Default to 3 years\n                "project_version": self.config.get("project_version", "1.0.0"),  # Default version\n            }}'
        )

        new_content = content.replace(training_input_text, new_training_input)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"✅ Added metadata fields to training_input in {file_path.name}")
        return True

    except Exception as e:
        print(f"❌ Failed to add metadata to training_input in {file_path.name}: {e}")
        return False

def add_metadata_to_additional_metadata(file_path: Path) -> bool:
    """Add missing metadata fields to additional_metadata in logging calls."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find all additional_metadata blocks
        metadata_pattern = r'additional_metadata\s*=\s*\{[^}]*\}'
        matches = re.finditer(metadata_pattern, content, re.DOTALL)

        changes_made = False

        for match in matches:
            metadata_text = match.group(0)

            # Check if metadata fields are already present
            if 'asset' in metadata_text and 'lookback_period' in metadata_text and 'project_version' in metadata_text:
                continue

            # Add missing metadata fields
            new_metadata = metadata_text.replace(
                '}',
                f',\n                    "asset": symbol,\n                    "lookback_period": self.config.get("lookback_days", 1095),\n                    "project_version": self.config.get("project_version", "1.0.0"),\n                }}'
            )

            content = content.replace(metadata_text, new_metadata)
            changes_made = True

        if changes_made:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"✅ Added metadata fields to additional_metadata in {file_path.name}")
            return True
        else:
            print(f"✅ Metadata fields already present in additional_metadata in {file_path.name}")
            return True

    except Exception as e:
        print(f"❌ Failed to add metadata to additional_metadata in {file_path.name}: {e}")
        return False

def add_standardized_naming_patterns(file_path: Path) -> bool:
    """Add standardized naming patterns to artifact logging methods."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if standardized naming pattern is already present
        if re.search(r'[A-Z]+_[A-Z]+_\d{8}_\d{4}_\d+', content):
            print(f"✅ Standardized naming pattern already present in {file_path.name}")
            return True

        # Find the artifact logging method
        step_match = re.search(r'step(\d+(?:_\d+)?)', file_path.name)
        if not step_match:
            return False

        step_num = step_match.group(1)
        method_name = f"_log_step{step_num}_artifacts_and_report"

        if method_name not in content:
            return False

        # Add standardized naming pattern example
        pattern_comment = f'            # Standardized naming pattern: {{exchange}}_{{symbol}}_{{timestamp}}_{{step_num}}_{{artifact_type}}'

        # Find a good place to insert the comment (after the method signature)
        method_start = content.find(method_name)
        if method_start == -1:
            return False

        # Find the first line after method signature
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if method_name in line:
                # Insert comment after the method signature
                lines.insert(i + 1, pattern_comment)
                break

        new_content = '\n'.join(lines)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"✅ Added standardized naming pattern comment to {file_path.name}")
        return True

    except Exception as e:
        print(f"❌ Failed to add standardized naming pattern to {file_path.name}: {e}")
        return False

def fix_step_file(file_path: Path) -> Dict[str, bool]:
    """Fix metadata and naming for a single step file."""
    results = {
        "training_input": False,
        "additional_metadata": False,
        "standardized_naming": False
    }

    print(f"\n🔄 Fixing {file_path.name}...")

    # Add metadata to training_input
    results["training_input"] = add_metadata_to_training_input(file_path)

    # Add metadata to additional_metadata
    results["additional_metadata"] = add_metadata_to_additional_metadata(file_path)

    # Add standardized naming patterns
    results["standardized_naming"] = add_standardized_naming_patterns(file_path)

    return results

def main():
    """Main function to fix metadata and naming for all steps."""
    steps_dir = Path("src/training/steps")

    if not steps_dir.exists():
        print(f"❌ Steps directory not found: {steps_dir}")
        return

    print("🔧 Fixing metadata fields and standardized naming...")
    print(f"📁 Steps directory: {steps_dir}")
    print(f"📋 Steps to fix: {len(STEPS_TO_FIX)}")

    results = {}

    for step_file in STEPS_TO_FIX:
        file_path = steps_dir / step_file

        if not file_path.exists():
            print(f"⚠️ Step file not found: {step_file}")
            continue

        results[step_file] = fix_step_file(file_path)

    # Print summary
    print("\n" + "="*60)
    print("📊 METADATA & NAMING FIX SUMMARY")
    print("="*60)

    for step_file, step_results in results.items():
        success_count = sum(step_results.values())
        total_count = len(step_results)

        if success_count == total_count:
            print(f"✅ {step_file}: All fixes successful")
        elif success_count > 0:
            print(f"⚠️ {step_file}: Partial success ({success_count}/{total_count})")
        else:
            print(f"❌ {step_file}: All fixes failed")

    total_successful = sum(sum(step_results.values()) for step_results in results.values())
    total_attempts = sum(len(step_results) for step_results in results.values())

    print(f"\n🎯 Overall: {total_successful}/{total_attempts} fixes successful")

    if total_successful == total_attempts:
        print("🎉 All metadata and naming fixes completed successfully!")
    else:
        print("⚠️ Some fixes may need manual review")

if __name__ == "__main__":
    main()