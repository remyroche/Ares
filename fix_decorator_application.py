#!/usr/bin/env python3
"""
Script to fix decorator application for steps that have the decorator in the file
but not applied to the execute method.
"""

import re
from pathlib import Path
from typing import List, Tuple

# Steps that need decorator application fixes
STEPS_TO_FIX = [
    "step2_5_sr_optimization.py",
    "step4_triple_barrier_method.py",
    "step5_labeling.py",
    "step7_enhanced_matrix_operations.py",
    "step9_5_hmm_lm_generalist_training.py",
]

def find_execute_method(file_path: Path) -> Tuple[bool, str, int]:
    """Find the execute method in a step file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')

        # Look for different execute method patterns
        patterns = [
            r'async def execute\s*\([^)]*\)\s*->[^:]*:',
            r'def execute\s*\([^)]*\)\s*->[^:]*:',
            r'async def execute_[a-zA-Z_]*\s*\([^)]*\)\s*->[^:]*:',
            r'def execute_[a-zA-Z_]*\s*\([^)]*\)\s*->[^:]*:',
            r'async def execute_triple_barrier_method\s*\([^)]*\)\s*->[^:]*:',
            r'async def execute_labeling\s*\([^)]*\)\s*->[^:]*:',
        ]

        for i, line in enumerate(lines):
            for pattern in patterns:
                if re.search(pattern, line):
                    return True, line.strip(), i

        return False, "", -1

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return False, "", -1

def check_decorator_applied(file_path: Path) -> bool:
    """Check if decorator is applied to execute method."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find execute method
        found, method_line, line_num = find_execute_method(file_path)

        if not found:
            return False

        # Check if decorator is present before execute method
        lines = content.split('\n')
        for i in range(line_num - 1, max(0, line_num - 10), -1):
            if lines[i].strip().startswith('@with_enhanced_mlflow_logging'):
                return True
            elif lines[i].strip() and not lines[i].strip().startswith('@'):
                break

        return False

    except Exception as e:
        print(f"Error checking decorator in {file_path}: {e}")
        return False

def apply_decorator_to_execute(file_path: Path) -> bool:
    """Apply decorator to execute method if not already applied."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if decorator is already applied
        if check_decorator_applied(file_path):
            print(f"✅ Decorator already applied to execute method in {file_path.name}")
            return True

        # Find execute method
        found, method_line, line_num = find_execute_method(file_path)

        if not found:
            print(f"⚠️ Could not find execute method in {file_path.name}")
            return False

        # Extract step number from filename
        step_match = re.search(r'step(\d+(?:_\d+)?)', file_path.name)
        if not step_match:
            print(f"⚠️ Could not extract step number from {file_path.name}")
            return False

        step_num = step_match.group(1)

        # Add decorator before execute method
        lines = content.split('\n')
        decorator = f'    @with_enhanced_mlflow_logging("step{step_num}")'

        # Find the right position (before other decorators)
        insert_pos = line_num
        for i in range(line_num - 1, -1, -1):
            if lines[i].strip().startswith('@'):
                insert_pos = i
            elif lines[i].strip() and not lines[i].strip().startswith('#'):
                break

        lines.insert(insert_pos, decorator)
        new_content = '\n'.join(lines)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"✅ Applied decorator to execute method in {file_path.name}")
        return True

    except Exception as e:
        print(f"❌ Failed to apply decorator to {file_path.name}: {e}")
        return False

def main():
    """Main function to fix decorator application."""
    steps_dir = Path("src/training/steps")

    if not steps_dir.exists():
        print(f"❌ Steps directory not found: {steps_dir}")
        return

    print("🔧 Fixing decorator application for execute methods...")
    print(f"📁 Steps directory: {steps_dir}")
    print(f"📋 Steps to fix: {len(STEPS_TO_FIX)}")

    results = {}

    for step_file in STEPS_TO_FIX:
        file_path = steps_dir / step_file

        if not file_path.exists():
            print(f"⚠️ Step file not found: {step_file}")
            continue

        print(f"\n🔄 Processing {step_file}...")

        # Check current status
        decorator_present = "@with_enhanced_mlflow_logging" in file_path.read_text()
        decorator_applied = check_decorator_applied(file_path)

        print(f"   - Decorator in file: {'✅' if decorator_present else '❌'}")
        print(f"   - Decorator applied to execute: {'✅' if decorator_applied else '❌'}")

        if decorator_present and not decorator_applied:
            success = apply_decorator_to_execute(file_path)
            results[step_file] = success
        elif decorator_applied:
            print(f"   - ✅ Already correctly applied")
            results[step_file] = True
        else:
            print(f"   - ⚠️ No decorator found in file")
            results[step_file] = False

    # Print summary
    print("\n" + "="*60)
    print("📊 DECORATOR FIX SUMMARY")
    print("="*60)

    successful_fixes = sum(results.values())
    total_steps = len(results)

    for step_file, success in results.items():
        status = "✅ Fixed" if success else "❌ Failed"
        print(f"{status} {step_file}")

    print(f"\n🎯 Overall: {successful_fixes}/{total_steps} decorators fixed")

    if successful_fixes == total_steps:
        print("🎉 All decorators successfully applied to execute methods!")
    else:
        print("⚠️ Some steps may need manual review")

if __name__ == "__main__":
    main()